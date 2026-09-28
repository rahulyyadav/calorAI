# CalorAI WhatsApp Meal-Logging Agent

An interview-focused implementation of a conversational nutrition agent that accepts natural-language meal descriptions and food photos, remembers useful user facts, supports corrections without double-counting, and reports accurate daily totals.

This repository deliberately separates the **agent core** from its delivery channels:

- WhatsApp Cloud API is the primary real-world interface, using Meta's free test number during development.
- A CLI remains available for clean-clone evaluation, deterministic testing, and demos when a public webhook is unavailable.

## Current status

Phases 0 to 6 are complete. The repository contains a locally runnable LangGraph vertical slice with SQLite persistence, deterministic nutrition reference data, typed meal tools, timezone-correct daily totals, and a CLI. Corrections are written as immutable meal revisions inside one transaction, ambiguous requests ask one focused question instead of guessing, refusals and impossible portions never reach a persisted row, and retried inbound messages are answered from an exactly-once event ledger. A text-model planner implements the same interface as the deterministic one and falls back to it on any unusable output, so the slice still runs without API keys. Memory survives a restart as typed records with confidence and provenance: a stated diet shapes later log replies, a protein or calorie target turns totals into progress, and `my usual` replays a saved routine as a new meal. A changed fact supersedes the one it replaces instead of contradicting it, and retrieval is bounded per kind — one diet, one target per nutrient, a few routines — so no saved habit can evict the facts every reply depends on. A photo takes its own path: bytes are checked against their signature before anything is billed, a dedicated vision model reports only the foods and portions it can separate, and the application prices every line from the reference table, so no model's calorie guess reaches the database. The caption beside a photo modifies that plate instead of becoming a second meal — `half of this` halves it, `2 cups of rice` outranks the model's portion, `my usual` cannot also replay a saved routine — and a read too shaky to price asks one focused question while a shaky-but-close one logs a disclosed estimate. Confidence and the model that read the photo are stored with the meal. WhatsApp is now the second
transport for the same graph: a signed Cloud API webhook verifies the delivery, normalizes text and
photo messages into the envelope the CLI already used, acknowledges inside the request thread and
answers behind it, and replies through the Graph API. An unlisted number never reaches the agent,
and Meta's retries are the same inbound event rather than a second meal. The claims above are now
measured rather than asserted: 26 conversation scenarios (the eleven from the brief plus fifteen
adversarial) and 16 photo scenarios grade tool choice, meal state, totals, clarifications, memory and
one-meal fusion with no API key and no network; a benchmark reports p50/p95 for the cold and warm
text, read and image paths at n=40, with the model requests each turn billed and the machine load
recorded, in a machine-readable file that names whether a provider was in the loop;
resilience tests cover provider timeouts, refusals and garbage, duplicate concurrent deliveries, and
unreadable photos; and every turn emits a trace-id'd structured event carrying ids, intents, models
and durations — never meal text or photo bytes.

Start with:

- [Delivery phases](docs/PHASES.md)
- [System design](docs/SYSTEM_DESIGN.md)
- [Requirements traceability](docs/REQUIREMENTS.md)
- [Eval harness and what it proves](docs/EVALS.md)
- [Original task brief](docs/brief/AI%20Engineer%20Test%20Task.pdf)

## Guiding product decisions

1. **Correctness before cleverness.** A correction updates an existing meal revision; it never creates nutrition that is counted twice.
2. **Ask only when uncertainty changes the outcome materially.** Otherwise, log a reasonable estimate and state it briefly.
3. **Memory is selective structured state, not a transcript.** Preferences, targets, and named routines are persisted; casual dialogue is not injected into every prompt.
4. **One inbound event produces at most one meal.** A photo and its caption are fused before logging.
5. **Fast path first.** Deterministic reads and totals bypass an unnecessary open-ended agent loop.
6. **WhatsApp is an adapter.** Domain logic is testable without Meta credentials or a public webhook.

## Proposed stack

- Python 3.12
- A stdlib `ThreadingHTTPServer` for the webhook, in front of a framework-free `WebhookApplication`
  (the phase added no dependency; the same object can be mounted on FastAPI unchanged)
- LangGraph for explicit conversational state and tool routing
- Pydantic for model/tool contracts
- SQLite for the interview build, behind repository interfaces that can move to Postgres
- Separate configurable text and vision models
- WhatsApp Cloud API test number plus local CLI
- Pytest for correctness tests and a small scenario eval harness

Exact provider/model selections will be recorded after a short latency-and-quality spike rather than hard-coded prematurely.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
calorai
```

Try:

```text
had 2 parathas and chai for breakfast
remember this as my usual breakfast
i'm vegetarian btw
aim for 120g protein a day
how am I doing today?
what did I eat today?
actually that was 3 parathas
no eggs
my usual
```

Each `calorai` invocation is a fresh process, so `my usual` and the stated target keep working after
you close and reopen the CLI.

### With a photo

Set `CALORAI_VISION_MODEL` and its API key in `.env`, then attach a plate:

```bash
calorai --image ~/Downloads/plate.jpg "this is lunch, and half of it is my brother's"
```

The photo goes to the vision model and never to the text model. The model names the foods it can
separate and their visible portions; the application prices each line from the reference table, so
no model's calorie guess reaches the database. Words beside the photo change that one meal — they
add a food, restate a portion, or share the plate — and never log a second one.

Without a vision key the CLI says so instead of inventing a plate:

```text
I cannot read a photo until a vision model is configured — set CALORAI_VISION_MODEL_API_KEY, or any
text model key, and I will use it. Describe the plate instead and I will log it.
```

The same path runs with a scripted vision answer, so a reviewer can see every photo case with no
key and no network:

```bash
python scripts/run_photo_evals.py
```

### On WhatsApp

The WhatsApp transport is an adapter around the same graph the CLI drives, so it needs a Meta app
and nothing else: no framework, no queue, no extra dependency.

1. On developers.facebook.com, create an app, add the **WhatsApp** product, and use the free test
   number. It gives you a `Phone number ID` and a short-lived access token.
2. Take the **App secret** from *App settings → Basic*, and choose your own long random
   **verify token**.
3. Copy `.env.example` to `.env` and fill in `CALORAI_WHATSAPP_VERIFY_TOKEN`,
   `CALORAI_WHATSAPP_APP_SECRET`, `CALORAI_WHATSAPP_ACCESS_TOKEN` and
   `CALORAI_WHATSAPP_PHONE_NUMBER_ID`. **These are secrets: `.env` is gitignored, and a real one
   must never be committed or pasted into a screenshot.**
4. List the numbers that may talk to the agent in `CALORAI_WHATSAPP_ALLOWED_USERS` (comma-separated
   `wa_id`s, your own included). An empty list refuses everyone, so the process will not start.
5. Serve it and expose it over HTTPS, which Meta requires:

```bash
calorai-whatsapp --port 8080
cloudflared tunnel --url http://localhost:8080   # or: ngrok http 8080
```

6. In the WhatsApp product's **Configuration** screen, set the callback URL to
   `https://<your-tunnel>/webhook` and the verify token to the same string you put in `.env`, then
   subscribe to the **messages** webhook field.
7. Send `I ate 2 dosa and a coffee` — or a photo of the plate with `half of this` — from a listed
   number.

What the transport does and does not decide:

- A delivery is only accepted when `X-Hub-Signature-256` matches an HMAC of the exact raw body
  under your app secret. Anything else is refused before it is parsed, and a body declaring more
  than a megabyte is refused before any of it is read.
- The handshake and the acknowledgement are fast: Meta is answered with a `200` first, and the
  meal is worked on on a worker thread behind it, so a slow model never turns into a retry storm.
- The read receipt carries Meta's typing indicator on the same request — the Cloud API has no
  separate typing call — and a message that arrives without a `wamid` is answered without being
  marked read, because there is no message id to receipt.
- Meta's retries carry the same message id, so a redelivered message is the same inbound event and
  logs at most one meal.
- A photo's bytes come from the Graph API and pass the same signature and size checks as a CLI
  photo, then go only to the vision model.
- A voice note, sticker, PDF or location is answered honestly as something the agent cannot read
  yet, rather than silently dropped.
- Everything above is testable without any of these credentials: `pytest tests/test_whatsapp.py
  tests/test_whatsapp_webhook.py` runs the whole path against a stand-in Graph API.

Run the verification suite:

```bash
ruff check .
mypy src
pytest --cov=calorai_agent --cov-report=term-missing
python scripts/run_evals.py
python scripts/run_photo_evals.py
```

## Evals

Two scenario sets, both runnable with no API key and no network:

```bash
python scripts/run_evals.py          # 11 supplied conversations + 15 adversarial ones
python scripts/run_evals.py --only correction
python scripts/run_photo_evals.py    # 16 photographed plates
```

Every scenario is graded against the database and the trace, not against a transcript: the tool the
turn chose, the meals it left behind, the day's totals, whether it asked instead of guessing,
whether a stored memory actually reached the reply, and whether a photo and its caption became one
meal. [docs/EVALS.md](docs/EVALS.md) defines each of those and states what these evals do not prove.

## Observability

Every turn gets one trace id, and the turns, model calls, media reads and deliveries each log a
duration:

```bash
CALORAI_LOG_FORMAT=json calorai-whatsapp --port 8080     # what a log aggregator reads
calorai --logs INFO "had 2 parathas and chai"            # one turn, traced on a terminal
```

```text
2026-09-28T00:18:29.710+00:00 info    trace=fad042bbe1c3 turn status=ok channel=cli redelivered=False route=log_meal photo=False duration_ms=7.21
Logged 2 paratha, 1 milk chai — about 640 kcal and 15g protein.
```

The line names no food, no quantity and no caption: this is health data, so logs carry ids,
intents, models and durations only. `CALORAI_LOG_LEVEL` defaults to `WARNING` so the CLI stays
quiet between replies, and the webhook raises it to `INFO` in its own entrypoint. Query strings are
scrubbed from HTTP access logs, because Meta's `hub.challenge` and token parameters are not
something to archive.

Optional LangSmith tracing is off unless `CALORAI_TRACING=true` **and** `LANGCHAIN_API_KEY` are
both present; asking for tracing without a key logs why it declined rather than buffering spans to
upload later.

## Latency

`benchmarks/latency.json` is written by a reproducible command and carries its own environment
(python, platform, CPU count, commit, the machine's 1-minute load average, sample size, how many
model requests each turn billed, and which model backing produced it):

```bash
python scripts/benchmark_latency.py --samples 40
```

Measured on an Apple silicon laptop with no API key, so these are **application overhead** numbers —
no provider round trip is included, and the result file says so in `model_backing`:

| Path | p50 | p95 | Model calls per turn | Samples |
|---|---|---|---|---|
| text, cold database | 4.69 ms | 6.22 ms | 0.0 | 40 |
| text, warm database | 5.01 ms | 5.49 ms | 0.0 | 40 |
| totals/list read, cold | 4.22 ms | 4.85 ms | 0.0 | 40 |
| totals/list read, warm | 4.18 ms | 4.82 ms | 0.0 | 40 |
| image, cold database | 4.76 ms | 5.14 ms | 0.0 | 40 |
| image, warm database | 5.14 ms | 5.75 ms | 0.0 | 40 |

"Cold" is the first turn on a database file the process created and had never opened. Opening that
file is part of the cold path but not part of the turn, so the artifact reports it on its own
(`setup_p50_ms`, 10.2–10.5 ms here) rather than hiding it inside a percentile or letting a reader
assume it was included. The file also names the commit the timings came from, or `-dirty` if the
working tree was not that commit when they were taken.

The load average is the reason any of this is readable. It is sampled at both ends of the run and
reported as the worse of the two: three runs at load 1.7–2.3 on 8 CPUs agreed within 0.25 ms at p50 on
every path, while a run taken while the machine sat at load 11.6 was up to 1 ms slower at p50 and
produced a 10.6 ms p95 for a path that measures 4.8 ms quietly. Same code, same 40 samples, different
machine.

What bought that, and what is still honest to complain about:

- **The model call that never happens.** "how am I doing today?" is read by the rules and answered
  from the database, and the `model_calls_per_sample` column is how that is checked rather than
  asserted: a read turn bills zero. The read rows are also the cheapest in the table, about 0.5 ms
  under the cold text row on the same run — but with no key configured the alternative was never
  going to be billed either, so these rows show what the read path *costs*, not what it saves. Set a
  key and re-run for the saving, which is the whole provider round trip.
- **Concurrency where the work is independent.** Photos in one delivery download in parallel, and a
  media id is fetched once per delivery no matter how many messages carry it. Answers stay strictly
  per-message and in order: parallelising the *replies* would let a later correction overtake the
  message it corrects.
- **The acknowledged request thread.** Meta is answered with a `200` before any of the above work
  starts, so provider latency never becomes a retry storm.
- **Warm is not reliably faster than cold here, and at 5 ms that is noise, not a finding.** SQLite
  page-cache and allocator state dominate a measurement this small: the two read rows are a wash
  (4.18 vs 4.22 ms) while `image_warm` sits 0.38 ms over `image_cold`, same machine, same 40 samples.
  The gap worth watching is the one that grows with the data, which is context gathering over a
  longer day.
- **One stall in forty.** `text_cold` has a 13.01 ms max behind a 6.22 ms p95 on the run above. At
  this scale that is the OS, not the application, and it is why the headline is a percentile rather
  than a mean.
- **What is not in these numbers.** With keys set, the dominant term becomes the `model_request`
  span, which the benchmark reports separately per path. Real user-visible latency is provider time
  plus the table above; run the command with a key to get the number that means something to a
  reviewer, and pass `--image` a real plate photo.

## Repository shape

```text
docs/                 brief, plan, architecture, decision record
src/calorai_agent/    application package (added phase by phase)
tests/                deterministic unit/integration tests
evals/                conversation scenarios and graders
scripts/              setup, benchmark, and demo helpers
```

## Interview strategy

The strongest demo path will show:

1. a natural-language meal logged through WhatsApp;
2. a correction that changes totals rather than double-counting;
3. `same as yesterday` or `my usual` using persistent selective memory;
4. one photo plus caption becoming exactly one meal;
5. surfaced vision uncertainty;
6. measured p50/p95 latency for text and image paths.

## Original constraints

The supplied brief allows a CLI and does not require WhatsApp integration. We are intentionally adding WhatsApp because it demonstrates the actual product surface, but it is isolated so it cannot endanger the required local evaluator experience. The task's stated 6-8 hour budget will be tracked honestly once implementation begins.
