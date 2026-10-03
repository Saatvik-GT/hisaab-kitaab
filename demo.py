"""Demo: type an everyday maths question in English, Hinglish, Hindi or Punjabi.

Shows the 1B model alone (mode A), then through the harness (mode K), then the answer.
    python demo.py "5 lakh ka loan 8% pe 3 saal ka EMI kitna hoga?"
    python demo.py            # interactive, blank line to quit
"""
import sys

import harness as h

# how to phrase each tool's exact result for a human
SAY = {
    "calculator": "Answer: {r}",
    "convert": "Converted: {r}",
    "emi": "EMI: Rs {r} per month",
    "gst": "Price with/without GST: Rs {r}",
    "percent": "Result: {r}",
    "discount": "You pay: Rs {r}",
    "simple_interest": "Simple interest: Rs {r}",
}


def show(label, call):
    tool, result = h.run_call(call)
    print("  %-22s tool=%s  args=%s" % (label, tool, call[1] if call else None))
    ok = call is not None and not str(result).startswith("error")
    print("  %-22s %s" % ("", ("-> " + SAY[tool].format(r=result)) if ok else "-> FAILED (" + str(result) + ")"))


def ask(text):
    names = list(h.TOOLS)
    print("\nQ:", text)
    print("  after number cleanup:", h.normalize(text))
    print("  harness shortlist:   ", h.shortlist(h.normalize(text)))
    try:
        show("1B alone (mode A)", h.mode_a(text, names))
        show("with harness (mode K)", h.mode_k(text, names))
    except Exception as e:  # e.g. the model server is not running
        print("  model server problem:", e)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        while True:
            q = input("\nask> ").strip()
            if not q:
                break
            ask(q)
