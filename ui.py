"""Terminal-style web UI for Hisaab-Kitaab.  Run:  python ui.py   then open http://localhost:8765

Standard library only. Serves ui/index.html and three JSON endpoints:
  POST /api/ask     run a question through the model alone (A) and the harness (K)
  GET  /api/status  model server up/down + ping, CPU and RAM use of this machine
  GET  /api/bench   the 20-task results table written by harness.py
"""
import ctypes
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import demo
import harness as h

PORT = 8765
HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()  # the local model serves one request at a time


# ------------------------------------------------------------ machine metrics (Windows)
class _MemStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                ("ptotal", ctypes.c_ulonglong), ("pavail", ctypes.c_ulonglong),
                ("vtotal", ctypes.c_ulonglong), ("vavail", ctypes.c_ulonglong),
                ("vext", ctypes.c_ulonglong)]


_last_cpu = None


def _filetime(ft):
    return ft.dwHighDateTime << 32 | ft.dwLowDateTime


def machine():
    """CPU % since the last call, RAM used GB / total GB. Returns None values off Windows."""
    global _last_cpu
    try:
        from ctypes import wintypes
        idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
        now = (_filetime(idle), _filetime(kern), _filetime(user))
        cpu = None
        if _last_cpu:
            d = [a - b for a, b in zip(now, _last_cpu)]
            if d[1] + d[2]:  # kernel time includes idle time
                cpu = round(100 * (1 - d[0] / (d[1] + d[2])), 1)
        _last_cpu = now
        m = _MemStatus()
        m.length = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return cpu, round((m.total - m.avail) / 2**30, 1), round(m.total / 2**30, 1)
    except Exception:
        return None, None, None


def model_ping():
    t0 = time.time()
    try:
        urllib.request.urlopen(h.URL.replace("/chat/completions", "/models"), timeout=2).read()
        return True, round((time.time() - t0) * 1000)
    except Exception:
        return False, None


# --------------------------------------------------------------------------- one query
def run_mode(fn, text, names):
    n0, t0 = len(h.STATS), time.time()
    try:
        call = fn(text, names)
    except Exception as e:  # server down / timed out
        return dict(error=str(e)[:120], ok=False, secs=round(time.time() - t0, 2), calls=0)
    secs = time.time() - t0
    stats = h.STATS[n0:]
    tool, result = h.run_call(call)
    ok = call is not None and not str(result).startswith("error")
    return dict(
        tool=tool, args=call[1] if call else None, result=str(result), ok=ok,
        say=demo.SAY[tool].format(r=result) if ok else None,
        secs=round(secs, 2), calls=len(stats),
        tok_in=sum(s["prompt"] for s in stats), tok_out=sum(s["out"] for s in stats),
        tps=next((round(s["tps"], 1) for s in stats[::-1] if s["tps"]), None))


def ask(text, baseline):
    names = list(h.TOOLS)
    norm = h.normalize(text)
    with LOCK:
        a = run_mode(h.mode_a, text, names) if baseline else None
        k = run_mode(h.mode_k, text, names)
    return dict(q=text, clean=norm, shortlist=h.shortlist(norm), a=a, k=k)


# ------------------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "ui", "index.html"), "rb") as f:
                self.send(f.read(), "text/html")
        elif self.path == "/api/status":
            up, ping = model_ping()
            cpu, used, total = machine()
            self.send(dict(up=up, ping=ping, model=h.MODEL, cpu=cpu, ram_used=used, ram_total=total))
        elif self.path == "/api/bench":
            path = os.path.join(HERE, h.BENCH_FILE)
            self.send(json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/api/ask":
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        text = str(body.get("q", "")).strip()[:300]
        if not text:
            return self.send_error(400)
        self.send(ask(text, bool(body.get("baseline", True))))


if __name__ == "__main__":
    machine()  # prime the CPU counter
    print("Hisaab-Kitaab UI on http://localhost:%d  (model server expected at %s)" % (PORT, h.URL))
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
