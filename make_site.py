"""Build the static replay site in docs/ from REAL recorded runs. Needs the model server running.

    python make_site.py

Runs every task (development + unseen) through the model alone and the harness, samples CPU/RAM while
doing so, and writes docs/replay.json plus a copy of ui/index.html. Nothing is invented: if a question
failed on the day, the replay shows the failure.
"""
import json
import os
import shutil
import threading
import time

import harness as h
import ui

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs")


def main():
    os.makedirs(OUT, exist_ok=True)
    status, stop = [], threading.Event()

    def sample():
        while not stop.is_set():
            cpu, used, total = ui.machine()
            up, ping = ui.model_ping()
            if cpu is not None:
                status.append(dict(cpu=cpu, ram_used=used, ram_total=total, ping=ping, model=h.MODEL))
            time.sleep(1)

    threading.Thread(target=sample, daemon=True).start()
    answers = {}
    for text, _, _ in h.TASKS + h.TASKS_HOLDOUT:
        answers[text.strip().lower()] = ui.ask(text, True)
        print("recorded:", text[:60])
    stop.set()

    bench = json.load(open(os.path.join(HERE, h.BENCH_FILE), encoding="utf-8"))
    with open(os.path.join(OUT, "replay.json"), "w", encoding="utf-8") as f:
        json.dump(dict(bench=bench, answers=answers, status=status), f, ensure_ascii=False)
    shutil.copy(os.path.join(HERE, "ui", "index.html"), os.path.join(OUT, "index.html"))
    print("wrote docs/ with %d answers and %d system samples" % (len(answers), len(status)))


if __name__ == "__main__":
    main()
