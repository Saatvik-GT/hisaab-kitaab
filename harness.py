"""Hisaab-Kitaab harness: everyday maths for a 1B local model.

Measures how reliable Gemma 3 1B (CPU only, via llama.cpp) is at tool calling
under different harness strategies (modes). Python does the exact maths; the
model only picks a tool and extracts the numbers.

Modes:
  A  plain JSON prompt, no help (baseline)
  B  same prompt, output constrained by a generated GBNF grammar
  N  B plus a number normaliser (5 lakh -> 500000, Hindi/Punjabi digits)
  K  N plus a keyword shortlist that narrows 7 tools to 1-2 before the grammar
  C  natural-language YES/NO tool selection (kept as a negative result)

Run:  python harness.py          (all modes)   python harness.py K   (letters pick modes)
"""
import ast
import json
import operator
import os
import re
import sys
import time
import urllib.error
import urllib.request

# 127.0.0.1, not "localhost": on Windows Python tries IPv6 first and waits ~2 s before falling back
URL = "http://127.0.0.1:9931/v1/chat/completions"
MODEL = "ggml-org/gemma-3-1b-it-qat-GGUF:Q4_0"


# ---------------------------------------------------------------- LLM client
STATS = []  # one dict per model call (tokens, seconds, tokens/s); ui.py reads this


def llm(messages, grammar=None, max_tokens=120, cache=True):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "cache_prompt": cache,  # reuse the KV cache for the identical prefix
    }
    if grammar:
        body["grammar"] = grammar
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        raise RuntimeError("server said: " + e.read().decode()) from None
    usage, timings = resp.get("usage", {}), resp.get("timings", {})
    STATS.append(dict(prompt=usage.get("prompt_tokens", 0), out=usage.get("completion_tokens", 0),
                      secs=time.time() - t0, tps=timings.get("predicted_per_second")))
    return resp["choices"][0]["message"]["content"]


# --------------------------------------------------------------------- tools
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _ev(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _ev(node.left), _ev(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError("bad expression")


def calculator(expression):
    return str(_ev(ast.parse(expression, mode="eval").body))


def emi(principal, rate, years):
    """Monthly loan instalment. rate is the yearly percent."""
    p, r, n = float(principal), float(rate) / 1200, float(years) * 12
    if r == 0:
        return str(round(p / n, 2))
    return str(round(p * r * (1 + r) ** n / ((1 + r) ** n - 1), 2))


def gst(amount, rate, mode):
    """mode 'add' puts GST on top, 'remove' takes it out of a GST-inclusive price."""
    a, r = float(amount), float(rate) / 100
    return str(round(a * (1 + r) if mode == "add" else a / (1 + r), 2))


def percent(value, percent):
    return str(round(float(value) * float(percent) / 100, 2))


def discount(price, percent):
    """Price you pay after a percent discount."""
    return str(round(float(price) * (1 - float(percent) / 100), 2))


def simple_interest(principal, rate, years):
    return str(round(float(principal) * float(rate) * float(years) / 100, 2))


_CONV = {("km", "miles"): 0.621371, ("miles", "km"): 1.609344,
         ("kg", "lbs"): 2.204623, ("lbs", "kg"): 0.453592}


def convert(value, from_unit, to_unit):
    v = float(value)
    if (from_unit, to_unit) in _CONV:
        return str(round(v * _CONV[(from_unit, to_unit)], 3))
    if (from_unit, to_unit) == ("c", "f"):
        return str(round(v * 9 / 5 + 32, 2))
    if (from_unit, to_unit) == ("f", "c"):
        return str(round((v - 32) * 5 / 9, 2))
    raise ValueError("unsupported conversion")


def _lit(s):
    """GBNF string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _enum(*values):
    return "(" + " | ".join(_lit(v) for v in values) + ")"


UNIT_ENUM = _enum("km", "miles", "kg", "lbs", "c", "f")
# a plain number: no leading dot, no "1.2.3". 1 lakh = 100000, so the model must write 500000.
NUM = '[0-9]{1,9} ("." [0-9]{1,3})?'

# name -> function, one-line description, params {name: GBNF fragment}
TOOLS = {
    "calculator": dict(fn=calculator, desc="do arithmetic with + - * / ( )",
                       params={"expression": "[0-9+*/(). -]{1,40}"}),
    "convert": dict(fn=convert, desc="convert a number between units (km, miles, kg, lbs, c, f)",
                    params={"value": NUM, "from_unit": UNIT_ENUM, "to_unit": UNIT_ENUM}),
    "emi": dict(fn=emi, desc="monthly loan EMI; principal in rupees, rate is yearly percent",
                params={"principal": NUM, "rate": NUM, "years": NUM}),
    "gst": dict(fn=gst, desc="add GST to a price or remove it from a GST-inclusive price (mode add/remove)",
                params={"amount": NUM, "rate": NUM, "mode": _enum("add", "remove")}),
    "percent": dict(fn=percent, desc="X percent of a value",
                    params={"value": NUM, "percent": NUM}),
    "discount": dict(fn=discount, desc="final price after a percent discount",
                     params={"price": NUM, "percent": NUM}),
    "simple_interest": dict(fn=simple_interest, desc="simple interest amount; rate is yearly percent",
                            params={"principal": NUM, "rate": NUM, "years": NUM}),
}


# ------------------------------------------------------------ grammar + prompt
def build_grammar(tool_names):
    """One grammar alternative per tool, so tool name and args are always valid."""
    rules, names = [], []
    for name in tool_names:
        parts = [_lit('{"tool": "%s", "args": {' % name)]
        for i, (pname, frag) in enumerate(TOOLS[name]["params"].items()):
            if i:
                parts.append(_lit(", "))
            parts.append(_lit('"%s": "' % pname))
            parts.append(frag)
            parts.append(_lit('"'))
        parts.append(_lit("}}"))
        rule = "t-" + name.replace("_", "-")
        rules.append("%s ::= %s" % (rule, " ".join(parts)))
        names.append(rule)
    return "root ::= " + " | ".join(names) + "\n" + "\n".join(rules)


def system_prompt(tool_names):
    lines = ["You pick one tool for the user's request. Reply with JSON only, like",
             '{"tool": "<name>", "args": {...}}', "", "Tools:"]
    for name in tool_names:
        t = TOOLS[name]
        lines.append("- %s(%s): %s" % (name, ", ".join(t["params"]), t["desc"]))
    return "\n".join(lines)


# ----------------------------------------------------------------------- modes
def parse_call(text):
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        name, args = data["tool"], data["args"]
        if name in TOOLS and set(args) == set(TOOLS[name]["params"]):
            return name, args
    except (ValueError, KeyError, TypeError):
        pass
    return None


def mode_a(task, names):
    msgs = [{"role": "system", "content": system_prompt(names)},
            {"role": "user", "content": task}]
    return parse_call(llm(msgs, None, 80))


def mode_b(task, names):
    msgs = [{"role": "system", "content": system_prompt(names)},
            {"role": "user", "content": task}]
    return parse_call(llm(msgs, build_grammar(names), 80))


# ---- number normaliser: code does the exact part, the model never multiplies by 100000
_SCALE = {"lakh": 10**5, "lac": 10**5, "लाख": 10**5, "ਲੱਖ": 10**5,
          "crore": 10**7, "करोड़": 10**7, "करोड": 10**7, "ਕਰੋੜ": 10**7}
_LAKH = re.compile(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|लाख|करोड़|करोड|ਲੱਖ|ਕਰੋੜ)", re.I)
# Devanagari and Gurmukhi digits -> ASCII
_DIGITS = str.maketrans("०१२३४५६७८९੦੧੨੩੪੫੬੭੮੯", "01234567890123456789")


def normalize(text):
    text = text.translate(_DIGITS)

    def repl(m):
        word = m.group(2).lower().rstrip("s")
        return str(round(float(m.group(1)) * _SCALE[word]))
    return _LAKH.sub(repl, text)


def mode_bn(task, names):
    """B plus the normaliser, to show what the normaliser alone contributes."""
    return mode_b(normalize(task), names)


# ---- mode C: natural-language tool selection (YES/NO per tool), then extract args
# One plain question per tool. Checked most specific first; first YES wins.
ASK = {
    "gst": "Does the request ask to add GST/tax to a price, or remove GST/tax from a price?",
    "discount": "Does the request ask for the price after a discount or percent off?",
    "emi": "Does the request ask for the monthly EMI instalment of a loan?",
    "simple_interest": "Does the request ask for simple interest (byaaj) on money over some years?",
    "convert": "Does the request ask to convert between units like km, miles, kg, pounds, Celsius, Fahrenheit?",
    "percent": "Does the request ask for X percent of a number?",
    "calculator": "Does the request need plain arithmetic, like multiplying or adding numbers?",
}


def says_yes(task, name):
    msgs = [{"role": "system", "content": "Answer YES or NO only."},
            {"role": "user", "content": 'Request: "%s"\n%s' % (task, ASK[name])}]
    # no grammar here: we only read the first word of a 3-token answer
    # cache off: these tiny prompts hung the server twice with cache reuse (suspected llama.cpp bug)
    return llm(msgs, None, 3, cache=False).strip().upper().startswith("YES")


def mode_c(task, names):
    task = normalize(task)
    for name in ASK:
        if name in names and says_yes(task, name):
            # second pass: only this one tool is in the prompt and in the grammar
            msgs = [{"role": "system", "content": system_prompt([name])},
                    {"role": "user", "content": task}]
            return parse_call(llm(msgs, build_grammar([name]), 80))
    return None


# ---- mode K: keyword shortlist. Plain code spots cue words (English/Hinglish/Hindi/Punjabi),
# narrows 7 tools to 1-2, and the grammar only allows those. The model still extracts the numbers.
CUES = {
    "gst": r"gst|tax|जीएसटी|टैक्स|ਜੀਐਸਟੀ|ਟੈਕਸ",
    "emi": r"emi|instal?lment|kist|ईएमआई|किस्त|ਈਐਮਆਈ|ਕਿਸ਼ਤ",
    "simple_interest": r"interest|byaaj|byaj|ब्याज|सूद|ਵਿਆਜ",
    "discount": r"\boff\b|discount|chhoot|छूट|छुट|ਛੋਟ",
    "convert": r"\b(km|kms|miles?|kg|kgs|pounds?|lbs?|celsius|fahrenheit)\b"
               r"|किलोमीटर|मील|पाउंड|सेल्सियस|फारेनहाइट|ਕਿਲੋਮੀਟਰ|ਮੀਲ|ਪਾਊਂਡ|ਸੈਲਸੀਅਸ|ਫਾਰਨਹੀਟ",
}
PERCENT_CUE = r"%|percent|प्रतिशत|परसेंट|ਪ੍ਰਤੀਸ਼ਤ|ਪਰਸੈਂਟ"


def shortlist(text):
    text = text.lower()
    found = [t for t, cue in CUES.items() if re.search(cue, text)]
    if "emi" in found and "simple_interest" in found:
        found.remove("simple_interest")  # "interest" appears in every loan question
    if found:
        return found
    return ["percent"] if re.search(PERCENT_CUE, text) else ["calculator"]


def mode_k(task, names):
    task = normalize(task)
    short = [t for t in shortlist(task) if t in names]
    msgs = [{"role": "system", "content": system_prompt(short)},
            {"role": "user", "content": task}]
    return parse_call(llm(msgs, build_grammar(short), 80))


MODES = {"A plain JSON": mode_a, "B grammar": mode_b,
         "N grammar+norm": mode_bn, "K shortlist": mode_k, "C yes/no": mode_c}

# --------------------------------------------------------------------- tasks
# (request, expected tool, text the tool result must contain or None)
# English, Hinglish, Hindi (Devanagari) and Punjabi (Gurmukhi). 1 lakh = 100000.
TASKS = [
    ("What is 17 times 23?", "calculator", "391"),
    ("Compute (12+8)/4", "calculator", "5.0"),
    ("3 kilo aam 120 rupaye kilo ke hain, total kitna hua?", "calculator", "360"),
    ("Convert 10 km to miles", "convert", "6.214"),
    ("How many km is 26 miles?", "convert", "41.843"),
    ("25 kg kitne pounds hote hain?", "convert", "55.116"),
    ("98.6 Fahrenheit ko Celsius mein batao", "convert", "37.0"),
    ("EMI on a 5 lakh loan at 8% for 3 years", "emi", "15668.18"),
    ("5 lakh ka loan 8% pe 3 saal ka EMI kitna hoga?", "emi", "15668.18"),
    ("ਮੈਨੂੰ 2 ਲੱਖ ਦੇ ਲੋਨ ਤੇ 10% ਵਿਆਜ ਨਾਲ 2 ਸਾਲ ਦੀ EMI ਦੱਸੋ", "emi", "9228.99"),
    ("Add 18% GST to 1000 rupees", "gst", "1180.0"),
    ("2360 mein se 18% GST hata do", "gst", "2000.0"),
    ("ਮੇਰੇ ਬਿੱਲ 5000 ਵਿੱਚ 12% GST ਜੋੜੋ", "gst", "5600.0"),
    ("What is 15% of 2400?", "percent", "360.0"),
    ("2400 का 15 प्रतिशत कितना होता है?", "percent", "360.0"),
    ("A shirt costs 1500 with 20% off. What do I pay?", "discount", "1200.0"),
    ("1500 की शर्ट पर 20 परसेंट छूट है, कितने देने होंगे?", "discount", "1200.0"),
    ("ਦੁਕਾਨਦਾਰ 800 ਦੀ ਚੀਜ਼ ਤੇ 25% ਛੋਟ ਦੇ ਰਿਹਾ ਹੈ, ਕੀਮਤ ਕੀ ਹੋਵੇਗੀ?", "discount", "600.0"),
    ("Simple interest on 10000 at 5% for 2 years", "simple_interest", "1000.0"),
    ("Mujhe 2 lakh ka 6% se 1 saal ka byaaj batao", "simple_interest", "12000.0"),
]


def run_call(call):
    if call is None:
        return None, "parse_fail"
    name, args = call
    try:
        return name, TOOLS[name]["fn"](**args)
    except Exception as e:  # tool errors count as failures
        return name, "error: %s" % e


# per-task answers are saved here; delete the file after changing a mode
CACHE_FILE = "results_cache.json"
BENCH_FILE = "bench.json"  # summary table, committed; the UI draws its benchmark panel from it


def evaluate():
    sys.stdout.reconfigure(encoding="utf-8")  # Hindi/Punjabi text on a Windows console
    names = list(TOOLS)
    bench = json.load(open(BENCH_FILE, encoding="utf-8")) if os.path.exists(BENCH_FILE) else {}
    cache = json.load(open(CACHE_FILE, encoding="utf-8")) if os.path.exists(CACHE_FILE) else {}
    print("%-14s %8s %8s %8s %8s" % ("mode", "parsed", "tool_ok", "result_ok", "sec/task"))
    for mode_name, fn in MODES.items():
        if len(sys.argv) > 1 and mode_name[0] not in sys.argv[1].upper():
            continue  # e.g. `python harness.py C` runs only mode C
        parsed = tool_ok = result_ok = 0
        secs = 0.0
        for text, want_tool, want_text in TASKS:
            key = mode_name + "|" + text
            if key not in cache:  # finished tasks survive a server hang: just re-run
                t0 = time.time()
                call = fn(text, names)
                cache[key] = [call, time.time() - t0]
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
            call, took = cache[key]
            call = tuple(call) if call else None
            secs += took
            tool, result = run_call(call)
            print("  ", mode_name[0], "| ok" if tool == want_tool else "| XX", text, "->", call, "->", str(result)[:40])
            parsed += call is not None
            tool_ok += tool == want_tool
            good = tool == want_tool and not str(result).startswith("error")
            if want_text is not None:
                good = good and want_text in str(result)
            result_ok += good
        n = len(TASKS)
        print("%-14s %5d/%d %5d/%d %6d/%d %8.1f" % (
            mode_name, parsed, n, tool_ok, n, result_ok, n, secs / n))
        bench[mode_name] = dict(n=n, parsed=parsed, tool_ok=tool_ok, result_ok=result_ok,
                                sec_per_task=round(secs / n, 2))
        with open(BENCH_FILE, "w", encoding="utf-8") as f:
            json.dump(bench, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    evaluate()