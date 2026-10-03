# Hisaab-Kitaab

An offline, open-weight helper for everyday money maths in **English, Hinglish, Hindi and Punjabi**.
Ask "5 lakh ka loan 8% pe 3 saal ka EMI kitna hoga?" and get an exact answer, from a 1B model
running on a laptop CPU with no GPU and no internet.

Built at Hacktoberfest Hack Day Chandigarh (MLH), 3 Oct 2026, solo, for *Best Open-Source AI Project*.

## The problem

A 1B model (Gemma 3 1B, QAT Q4_0) is private, free and offline, but it is unreliable. It picks the wrong
tool, writes `principal: "5 lakh"` where a number is needed, and cannot do arithmetic. People who would
benefit most from an offline helper (low connectivity, regional languages) get the least reliable model.

## The idea

The model does **not** do the maths. It does two small jobs: pick a tool and extract the numbers.
Python does the exact computation. The **harness** is the code around the model that makes those two
jobs reliable. That harness is the project.

```
"2 lakh ka 6% se 1 saal ka byaaj batao"
   -> number cleanup:   "200000 ka 6% se 1 saal ka byaaj batao"
   -> keyword shortlist: [simple_interest]
   -> grammar-constrained call: {"tool": "simple_interest", "args": {"principal": "200000", ...}}
   -> Python:           Rs 12000.0
```

Tools: `calculator`, `convert` (km/miles, kg/lbs, C/F), `emi`, `gst` (add/remove), `percent`,
`discount`, `simple_interest`.

## Demo

```
python demo.py "5 lakh ka loan 8% pe 3 saal ka EMI kitna hoga?"
```
```
  1B alone (mode A)      args={'principal': '5 lakh', ...}  -> FAILED (could not convert '5 lakh')
  with harness (mode K)  args={'principal': '500000', ...}  -> EMI: Rs 15668.18 per month
```
Run `python demo.py` with no argument for an interactive prompt.

### Terminal UI

```
python ui.py        # then open http://localhost:8765
```

A terminal-style page (standard library only, no build step). Type a question, or click an example in
English, Hinglish, Hindi or Punjabi. Each query runs twice, the 1B model alone and through the harness,
and both results print side by side. Side panels show live, real metrics:

- **system:** CPU % and RAM of the machine (a CPU-only model makes this the interesting number)
- **model server:** up/down, ping, generation speed in tokens/s, tokens in/out, model calls
- **this session:** valid tool calls, model alone vs. harness, and latency per query
- **benchmark:** the 20-task results table from `bench.json`

Commands: `help`, `bench`, `baseline off|on` (skip the unaided run to halve latency), `clear`.
"Valid tool call" in the session panel means the call parsed and the tool ran without error. It cannot
know whether an arbitrary question got the *right* answer; that is what the 20-task benchmark measures.

## Harness modes

| Mode | What it adds |
|---|---|
| A | Plain JSON prompt, no help. Baseline. |
| B | Output constrained by a GBNF grammar generated from the tool definitions: always valid JSON, valid tool name, well-formed numbers. |
| N | B plus a **number normaliser**: `5 lakh` -> `500000`, `crore`, Hindi/Punjabi words and digits. The model never multiplies by 100000. |
| K | N plus a **keyword shortlist**: cue words in all four languages narrow 7 tools to 1-2 before the grammar. |
| C | Natural-language tool selection (YES/NO per tool). **Negative result**, see below. |

## Results

20 hand-written tasks (9 in English, the rest Hinglish, Hindi and Punjabi), Gemma 3 1B on an
i7-1185G7 CPU, temperature 0. "Right answer" means the right tool *and* the exact expected result.

| Mode | Parsed | Right tool | Right answer | sec/task |
|---|---|---|---|---|
| A plain prompt | 11/20 | 7/20 | **4/20** | 1.0 |
| B grammar | 20/20 | 13/20 | **9/20** | 1.0 |
| N grammar + normaliser | 20/20 | 13/20 | **11/20** | 1.2 |
| K + keyword shortlist | 20/20 | 20/20 | **17/20** | 0.9 |

Each task is one short model call (about 40 output tokens at roughly 45 tokens/s). Latency is end to end
including the HTTP request. Early runs showed ~3 s/task because Python resolved `localhost` via IPv6 first
and waited ~2 s on every call on Windows; the code now uses `127.0.0.1` and the benchmark was re-run
(accuracy unchanged).

What each step teaches:
- **Grammar** fixes the *shape* (every output parses) but not the *values*.
- **Normaliser** fixes the lakh/crore argument errors (+2).
- **Shortlist** fixes tool choice. With 7 tools the 1B model confuses anything containing a `%`
  (e.g. a shirt discount became `convert`, km to miles).
- Remaining failures in K are argument reading: `(12+8)/4` loses the `/4`, a word problem produces a
  nonsense expression, and one GST-removal question swaps amount and rate.

## Limitations (read these)

- **The shortlist is tuned on the same 20 tasks.** I wrote the cue words after seeing them, so K's 20/20
  tool choice does not show it generalises. A held-out set of unseen phrasings has not been run yet.
  In mode K the model effectively no longer chooses the tool (the shortlist usually has exactly one);
  the model's job there is extracting numbers.
- **Mode C is a negative result.** It asks the model YES/NO for each tool. The 1B model says YES to
  almost everything (an EMI question got YES for `discount`) and it is slow (up to 7 calls per task).
  It did not finish a full run: the local server hung during it, so only 12 of 20 tasks were observed,
  and those were worse than B. It is kept in the code for transparency.
- **Not perfectly deterministic.** One B answer changed between runs at temperature 0.
- 20 tasks is a small set. Treat the table as indicative, not statistically strong.
- Hindi and Punjabi phrasings were written by one person and cover a limited range of everyday speech.

## Run it

Requires Python 3.11+ (standard library only) and a local OpenAI-compatible llama.cpp server.

1. Serve Gemma 3 1B: `ggml-org/gemma-3-1b-it-qat-GGUF:Q4_0`, e.g. `llama serve --jinja --port 9931`
   (this project used the llama.app tray app, which wraps llama.cpp). Edit `URL` and `MODEL` at the top
   of `harness.py` if yours differ.
2. `python demo.py` for the demo, or `python harness.py` for the full results table
   (`python harness.py K` runs only mode K). Per-task answers are cached in `results_cache.json` so an
   interrupted run resumes; delete it after changing a mode.

Native llama.cpp tool calling is not used: llama.cpp rejects a custom grammar together with `tools`,
so this project runs its own tool loop.

## Papers that motivated this

Ideas drawn from these papers; the implementation is original. Please check the originals for what they
actually claim.

- DCCD, arXiv [2603.03305](https://arxiv.org/abs/2603.03305)
- Natural Language Tools, arXiv [2510.14453](https://arxiv.org/abs/2510.14453): the idea behind mode C
- The Deterministic Horizon, arXiv [2606.00376](https://arxiv.org/abs/2606.00376): the idea of handing
  exact, deterministic work to code instead of the model

## What was written from scratch

All code in this repo was written during the event window: the tools, the GBNF grammar builder, the
prompts, the number normaliser, the keyword shortlist, modes A/B/N/K/C, the evaluation harness with
its resumable cache, the 20 test tasks, and the demo. The only third-party pieces are the Gemma 3 1B
weights and llama.cpp (the inference server).

## AI-assistant disclosure

I used Claude (Claude Code) as a coding assistant during the event for design discussion, writing and
debugging code, and drafting this README. I chose the project direction, ran every experiment on my
machine, and decided what to keep or cut.

## License

MIT, see [LICENSE](LICENSE). Model weights are Google's Gemma 3, under the Gemma terms of use.
