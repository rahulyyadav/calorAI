# Clean-clone checklist

Every claim in this repository has to survive a reviewer who has no API key, no Meta app, and no
memory of how it was built. This is the run that checks that, recorded with what actually happened
rather than what should happen.

The test: clone the repository into an empty directory, delete every credential, and follow only the
commands the README documents. Anything that fails, hangs, or needs an undocumented step is a defect.

## What was run, and what happened

Environment: Apple silicon (arm64), Darwin, Python 3.13.3, no `CALORAI_*_API_KEY` set, no network
calls made on purpose.

```bash
git clone https://github.com/rahulyyadav/calorAI.git && cd calorAI
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
cp .env.example .env
```

Installs clean. `calorai --help` and `calorai-whatsapp --help` both work with an empty `.env`, which
is the point: `--help` must not require a key.

The README's `Try:` block, piped into the interactive CLI in one process:

| Input | Reply |
|---|---|
| `had 2 parathas and chai for breakfast` | `Logged 2 paratha, 1 milk chai — about 640 kcal and 15g protein.` |
| `remember this as my usual breakfast` | `Got it — I will remember your usual breakfast is 2 paratha, 1 milk chai.` |
| `i'm vegetarian btw` | `Got it — I will remember you are vegetarian.` |
| `aim for 120g protein a day` | `Got it — I will remember daily 120 g protein target.` |
| `how am I doing today?` | `Today: 640 kcal, 15g protein, 94g carbs, and 22g fat across 1 meal. Against your targets: 15 of 120 g protein.` |
| `what did I eat today?` | `Today you logged: breakfast at 08:00: 2 paratha, 1 milk chai.` |
| `actually that was 3 parathas` | `Updated your breakfast at 08:00 to 3 paratha, 1 milk chai — now about 900 kcal and 21g protein.` |
| `no eggs` | `Noted — I have not logged a meal for that.` |
| `my usual` | `Logged 2 paratha, 1 milk chai — about 640 kcal and 15g protein.` |

Totals after the correction are 1540 kcal across 2 meals — the revised breakfast counts once, not
twice, and `no eggs` became a fact rather than a phantom meal.

Restarting in a **new process** (`calorai "how am I doing today?"`) returns the same totals: memory
and meals live in SQLite, not in the process that wrote them.

## Commands a reviewer is told to run

```bash
ruff check .                                    # All checks passed!
mypy src                                        # Success: no issues found in 22 source files
pytest --cov=calorai_agent --cov-report=term-missing   # 425 passed, 95% coverage
python scripts/run_evals.py                     # 28/28 conversation scenarios passed
python scripts/run_photo_evals.py               # 16/16 photo scenarios passed
python scripts/benchmark_latency.py --samples 40  # writes benchmarks/latency.json
```

All six ran from the fresh clone with no key and no network. The benchmark reproduces the shape the
README describes — the read paths land around 4.2-4.4 ms and the logging paths above them — though
its exact milliseconds move with whatever else the machine is doing, which is why the artifact
records its own load average instead of pretending the number is absolute.

## The two things a keyless reviewer cannot do, and what happens instead

**A photo with no vision key** says so rather than inventing a plate:

```bash
calorai --image ~/Downloads/plate.jpg "this is lunch, and half of it is my brother's"
```

> I cannot read a photo until a vision model is configured — set CALORAI_VISION_MODEL_API_KEY, or any
> text model key, and I will use it. Describe the plate instead and I will log it.

Every photo behaviour is still reviewable keyless, because `scripts/run_photo_evals.py` drives the
same graph against a scripted vision answer.

**Starting the webhook** needs four real Meta credentials, so a keyless reviewer is told to stop
here — and the code stops for them: `calorai-whatsapp` exits `2` with an explanation when the
credentials are missing, and exits `2` again when the credentials are present but
`CALORAI_WHATSAPP_ALLOWED_USERS` is empty, because a server that accepts every phone number is not a
demo, it is an open endpoint. Both refusals are asserted in
`tests/test_whatsapp_webhook.py:1023`; this checklist does not claim to have opened a public tunnel,
because doing that from a review machine is exactly what the allow-list is there to prevent.

## What this run proves about the logs

Running one turn at `--logs INFO` produces a single line:

```text
2026-09-28T02:11:15.270+00:00 info trace=37aa8d40afb9 turn status=ok channel=cli redelivered=False route=log_meal photo=False duration_ms=6.7
```

The turn logged two parathas and a chai, and the log line contains neither. Ids, intent, flags and a
duration — that is the whole payload. This is health data, and the privacy rule is observable here
rather than only asserted in a docstring.

## Known rough edges a reviewer will hit

- `data/calorai.sqlite3` accumulates across runs. `rm -rf data/` is the reset, and the CLI says
  nothing about it because a fresh database is the normal case.
- With `CALORAI_PLANNER=deterministic`, a food the reference table cannot price is left out of the
  reply quietly — `had 2 idli and sambar for breakfast` logs the idli and says nothing about the
  sambar. The model-backed planner discloses the gap instead. This is documented in `docs/EVALS.md`
  as a boundary of the rules, not a bug to be surprised by.
- Latency numbers measured on a laptop with a browser open are 1-2 ms slower than the same code on a
  quiet machine. Compare the `load_average_1m` field before comparing two runs.
