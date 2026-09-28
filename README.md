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
measured rather than asserted: 28 conversation scenarios (the eleven from the brief plus seventeen
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
- [Clean-clone checklist](docs/CLEAN_CLONE.md)
- [Original task brief](docs/brief/AI%20Engineer%20Test%20Task.pdf)

## Guiding product decisions

1. **Correctness before cleverness.** A correction updates an existing meal revision; it never creates nutrition that is counted twice.
2. **Ask only when uncertainty changes the outcome materially.** Otherwise, log a reasonable estimate and state it briefly.
3. **Memory is selective structured state, not a transcript.** Preferences, targets, and named routines are persisted; casual dialogue is not injected into every prompt.
4. **One inbound event produces at most one meal.** A photo and its caption are fused before logging.
5. **Fast path first.** Deterministic reads and totals bypass an unnecessary open-ended agent loop.
6. **WhatsApp is an adapter.** Domain logic is testable without Meta credentials or a public webhook.

## Stack

- Python 3.11 or newer (`requires-python = ">=3.11"`; everything here was run on 3.13.3)
- A stdlib `ThreadingHTTPServer` for the webhook, in front of a framework-free `WebhookApplication`
  with no dependency on the server — the same object can be mounted on FastAPI unchanged
- LangGraph for explicit conversational state and tool routing
- Pydantic for model/tool contracts
- SQLite for the interview build, behind repository interfaces that can move to Postgres
- Separate configurable text and vision models
- WhatsApp Cloud API test number plus local CLI
- Pytest for correctness tests and a small scenario eval harness

## Models, and why these

Both model slots default to `gpt-4o-mini` against any OpenAI-compatible endpoint
(`CALORAI_TEXT_MODEL`, `CALORAI_VISION_MODEL`, each with its own base URL and key). The reason is
the shape of the turn rather than a benchmark between vendors: a meal log needs a tool name, a few
food names and a quantity — a classification with a strict JSON envelope, on the one path where
latency and cost are per-message and user-visible. No model-versus-model comparison was run here,
and the numbers below are honest about that: every eval and every latency sample in this repository
was produced with the deterministic planner and a scripted vision answer, because that is the only
configuration a reviewer can reproduce without a key.

The choice is hedged on purpose, because a single-provider pick is the part of this design most
likely to be wrong next year:

- The text model is optional. `RuleBasedPlanner` is the default when no key is set and the fallback
  when a model's output is unusable, so a provider failing or being swapped cannot lose a meal.
- The model never prices food. It names foods and portions; the reference table converts them into
  kcal and macros, so a hallucinated calorie count cannot reach the database. Switching providers
  changes interpretation quality, not arithmetic correctness.
- Vision is a separate slot because photo bytes must not enter the text planner's context. The two
  are configured, timed and tested apart: `scripts/run_photo_evals.py` exercises the vision and
  fusion path alone, with no text planner in the loop.
- Swapping is a config change, not a code change: both slots speak the OpenAI-compatible chat
  completions shape (`chat/completions`, with an `image_url` part for vision), so any endpoint that
  accepts it works. `docs/SYSTEM_DESIGN.md` treats a provider change as a swap under "scale".

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

## How memory works

Three typed kinds are stored, and nothing else (`MemoryKind` in `src/calorai_agent/domain.py`):

- **`dietary_constraint`** — `i'm vegetarian btw`, `no eggs`. One active fact, the newest one, so a
  changed diet replaces the old statement instead of contradicting it in every later prompt.
- **`nutrition_target`** — `aim for 120g protein a day`. Keyed per nutrient, so a protein target and
  a calorie target can both be active and neither can evict the other.
- **`named_routine`** — `remember this as my usual breakfast`. Written when the user names it, or
  when the same meal recurs enough to be a habit the user then refers to.

What is deliberately **not** memory: the transcript, one-off meals merely mentioned, and anything
inferred about the user without an explicit statement. That is the point of the type — health data
that was never volunteered should not end up in a prompt.

A fact is written by exactly two nodes, both through the same `remember` tool call: `save_memory`
when the turn is only a statement, and `log_meal` when a photo caption carries a durable fact beside
the plate (the meal lands first, the fact rides on the same inbound event). Nothing else in the graph
can save a fact as a side effect of phrasing. Each record carries confidence and provenance
(`InterpretationOrigin`: rule-based, text model, vision fusion, user-confirmed), which is what makes
`docs/EVALS.md` able to assert *which* kind of memory a turn used rather than that the reply looked
reasonable.

Retrieval is bounded twice over: `MEMORY_KIND_LIMITS` in `src/calorai_agent/memory.py` caps one diet,
one target per nutrient, and six routines per turn, and `MEMORY_CONTEXT_LIMIT` caps the whole context
at eight lines. The per-kind bound is the part that matters — a global cap would let a user with many
saved routines quietly evict the diet and the targets, which are usually the oldest facts and the ones
that shape every later reply. Retrieval filters by kind and validity first, then takes the newest of
what survives, so a long history cannot inflate the prompt or the latency of a turn.

`my usual` is a read *and* a write: it replays the saved routine as a **new** meal through
`repeat_meal`, because eating your usual breakfast twice in a week is two meals, not one row
updated twice. If the routine is ambiguous (two saved routines could be "usual"), the turn asks.

## Tools, and why they are split this way

The graph ends in exactly one of these nodes per inbound event, and each node is one narrow database
operation rather than a general-purpose API:

| Node | What it may do | Why it is separate |
|---|---|---|
| `log_meal` | commit one validated meal for one inbound event | the unit of "did the agent understand me" |
| `repeat_meal` | expand a saved routine into a new meal | replay is not the same as remembering |
| `revise_meal` | append a superseding revision to a resolved meal | corrections must not double-count |
| `delete_meal` | mark a resolved meal inactive, keeping history | refusal and removal are different claims |
| `get_totals` | sum active revisions for one local calendar day | arithmetic the model is never trusted with |
| `list_meals` | return bounded meals for a range | reads must not need a model at all |
| `save_memory` | persist one typed fact with provenance | facts are volunteered, never inferred from a reply |
| `respond` | answer conversationally, or ask one question | a turn that mutates nothing is still a turn |

Below them, `MealTools` (`src/calorai_agent/tools.py`) is a thin typed surface over the repository:
`log_meal`, `revise_meal`, `delete_meal`, `get_meals_in_range`, `get_daily_totals`, `remember`,
`list_memories`, plus the exactly-once ledger calls `record_inbound`, `complete_inbound`,
`outcome_for_event` and `memory_for_event`.

The split is deliberate in three ways:

1. **No generic SQL tool, no generic "update" tool.** Narrow operations are what make the evals
   legible: `docs/EVALS.md` grades *which tool the turn chose*, so a tool that does several things
   would make the grade meaningless as well as making the write path unauditable.
2. **Reads are tools too.** `get_totals` and `list_meals` are reachable without a model decision,
   which is why the benchmark's read rows bill zero model calls.
3. **Every mutating tool validates before it writes.** Portions, plausibility limits and nutrition
   pricing happen in the tool, so a model that invents `calories: 9000` cannot put 9000 in the
   database — the row holds what the reference table prices, and the reply never repeats the
   invention.

One inbound event produces at most one meal, enforced by the ledger rather than by prompt prose:
`record_inbound` is keyed on the message id, and a redelivered event replays the stored reply
instead of running the graph again.

## Evals

Two scenario sets, both runnable with no API key and no network:

```bash
python scripts/run_evals.py          # 11 supplied conversations + 17 adversarial ones
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

It writes `benchmarks/latency.json` in place; `--out` points it somewhere else when a comparison run
should not overwrite the committed artifact, and `--image` gives it a real plate photo instead of the
1×1 stand-in.

Measured on an Apple silicon laptop with no API key, so these are **application overhead** numbers —
no provider round trip is included, and the result file says so in `model_backing`:

| Path | p50 | p95 | Model calls per turn | Samples |
|---|---|---|---|---|
| text, cold database | 5.24 ms | 7.41 ms | 0.0 | 40 |
| text, warm database | 5.04 ms | 6.40 ms | 0.0 | 40 |
| totals/list read, cold | 4.33 ms | 5.20 ms | 0.0 | 40 |
| totals/list read, warm | 4.28 ms | 7.94 ms | 0.0 | 40 |
| image, cold database | 4.91 ms | 6.27 ms | 0.0 | 40 |
| image, warm database | 5.50 ms | 6.79 ms | 0.0 | 40 |

"Cold" is the first turn on a database file the process created and had never opened. Opening that
file is part of the cold path but not part of the turn, so the artifact reports it on its own
(`setup_p50_ms`, 10.7–11.2 ms here) rather than hiding it inside a percentile or letting a reader
assume it was included. The file also names the commit the timings came from, or `-dirty` if the
working tree was not that commit when they were taken.

The load average is why any of this is reproducible rather than anecdotal. It is sampled at both ends
of the run and reported as the worse of the two, because the same code on the same laptop measures
differently when the machine is busy: the committed artifact was written at load 2.23, and re-running
the command on a loaded machine is the fastest way to see a p95 move by 5 ms without the code
changing. Compare `load_average_1m` before comparing two runs, and treat a table whose environment
section says nothing about load as a number with no provenance.

What bought that, and what is still honest to complain about:

- **The model call that never happens.** "how am I doing today?" is read by the rules and answered
  from the database, and the `model_calls_per_sample` column is how that is checked rather than
  asserted: a read turn bills zero. The read rows are also the cheapest in the table, about 0.9 ms
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
  page-cache and allocator state dominate a measurement this small: `text_warm` lands 0.2 ms under
  `text_cold` while `image_warm` sits 0.59 ms *over* `image_cold`, and `read_warm` has the best p50
  in the table (4.28 ms) and the worst p95 (7.94 ms). Same machine, same code, same 40 samples. The
  gap worth watching is the one that grows with the data, which is context gathering over a longer
  day.
- **A stall in every forty.** `read_cold` has an 11.02 ms max behind a 5.20 ms p95 on the run above,
  and `text_cold` a 9.02 ms max. At this scale that is the OS, not the application, and it is why the
  headline is a percentile rather than a mean.
- **What is not in these numbers.** With keys set, the dominant term becomes the `model_request`
  span, which the benchmark reports separately per path. Real user-visible latency is provider time
  plus the table above; run the command with a key to get the number that means something to a
  reviewer, and pass `--image` a real plate photo.

## Assumptions and trade-offs

Each of these is a choice that could be argued with, stated where a reviewer can argue with it:

- **The application prices every line; the model only names food.** That buys the invariant no
  hallucinated calorie count can reach the database, and costs coverage: 32 reference foods
  (`src/calorai_agent/nutrition.py`), reached by name through an alias table (`chapati` and `phulka`
  are `roti`), so a dish with no row is named back to the user and left out of the total rather than
  invented. It also means the agent cannot answer "how many calories are in an apple" — it says so
  instead of guessing.
- **Ambiguity is a policy, not a feeling.** `policy.py` holds the thresholds the brief's examples
  imply: at or above `high_confidence` (0.8) log plainly, between it and `material_confidence`
  (0.55) log and say it is an estimate, below that ask one question; above
  `MAX_PLAUSIBLE_QUANTITY` (40 of anything) refuse out loud. Those numbers were tuned against
  `evals/conversation_scenarios.json`, which is the only honest justification available for them.
- **SQLite, one writer, no queue.** Correct for one user's interview build and for exactly-once
  delivery: each mutation is one `BEGIN IMMEDIATE` transaction, and the two invariants the
  conversation depends on are database constraints rather than check-then-write races —
  `UNIQUE(channel, external_id)` on the inbound ledger, and a unique index allowing one active meal
  per source event. A redelivery inserts nothing and finds the reply already there. Postgres plus a
  durable queue is the scale answer in `docs/SYSTEM_DESIGN.md`, and the repository interface is the
  seam that would make it a swap.
- **A redelivered message is the same event, keyed on its id.** That assumes Meta keeps `wamid`
  stable across retries, which it documents. An id that changed between retries would log twice —
  and is the reason the ledger is keyed at all rather than deduplicating on message text.
- **Photo bytes never outlive the process.** The media cache is in-memory under a byte ceiling and
  nothing about the photo is persisted beyond what the model said about it, so a restart re-downloads.
- **The deterministic planner is the default, not the model.** Without a key the app understands the
  food vocabulary the brief's conversations use, and anything outside it asks rather than guesses.
  A key widens interpretation; it never changes what is already true about the arithmetic.

## Limitations

- Numbers in `benchmarks/latency.json` are application overhead with no provider round trip, taken
  on a keyless machine; the saving the fast paths buy is real but is not measured here.
- This repository has never run against Meta's servers. The webhook is verified against the real
  HTTP shell, the real signature scheme and real Cloud API payload shapes in
  `tests/test_whatsapp_webhook.py`, and the transport refuses to start without a verify token, an
  app secret and an allow-list — but a live test-number round trip needs a public HTTPS tunnel and
  credentials, and no fake success is claimed for that.
- One photo is one meal. Two plates in one frame are read as one meal, and the reply says what it
  counted so the user can correct it.
- Meal timing comes from the words in the message (`for breakfast`, `late-night`), not from a
  clock, so a message typed at 09:00 about last night's dinner lands on the day the user named.
- The reference table is Indian-home-cooking weighted, because that is what the brief's conversations
  eat. Expanding it is data entry, not architecture.

## Next steps

In the order I would take them, each one something this build makes possible rather than requires:

1. Put a key in and re-run `scripts/benchmark_latency.py` and both eval scripts: the model paths are
   implemented and unit-tested with a fake client, and the honest gap in this submission is that no
   real provider has been timed end to end.
2. Expose the same `WebhookApplication` on FastAPI behind a tunnel and complete one live
   text-and-photo round trip with a Meta test number.
3. Move the reference table out of code into data with a source cited per row, widen it beyond the
   32 foods it holds, and price a dish from its parts when its name has no row of its own.
4. Add a durable queue and Postgres behind the existing repository interface for anything past one
   process, keeping the same inbound-ledger idempotency.
5. Give the user a deletion endpoint that actually forgets (row removal plus retention), since the
   current delete marks a meal inactive and keeps the history.

## Repository shape

```text
docs/                 brief, plan, architecture, decision record
src/calorai_agent/    application package (added phase by phase)
tests/                deterministic unit/integration tests
evals/                conversation scenarios and graders
scripts/              setup, benchmark, and demo helpers
```

## Walkthrough

Five to ten minutes, in this order, each step a command that was actually run from a fresh clone
(`docs/CLEAN_CLONE.md` holds the transcript and the exact replies).

1. **A meal from words** — `calorai`, then `had 2 parathas and chai for breakfast`.
   `Logged 2 paratha, 1 milk chai — about 640 kcal and 15g protein.` The numbers come from the
   reference table, not from a model: with no key configured, the deterministic planner is what
   read that sentence.
2. **A correction, not a second meal** — `actually that was 3 parathas`.
   `Updated your breakfast at 08:00 to 3 paratha, 1 milk chai — now about 900 kcal and 21g protein.`
   Then `what did I eat today?` still lists one breakfast. The first row is not edited and not
   deleted: the revision supersedes it, and both rows stay in `meal_revisions` for the audit trail.
3. **Memory that outlives the process** — `i'm vegetarian btw`, `aim for 120g protein a day`,
   `remember this as my usual breakfast`. Quit, start a new `calorai`, and
   `how am I doing today?` answers against the protein target and `my usual` logs the routine.
   Those are three typed records with provenance, not a transcript in a prompt.
4. **A plate** — `calorai --image ~/Downloads/plate.jpg "this is lunch, and half of it is my
   brother's"`. Without a vision key this is the honest refusal printed verbatim in the Quick start,
   which is the behaviour to show; the caption-modifies-the-photo cases run keyless through
   `python scripts/run_photo_evals.py`, because that harness drives the same graph with a scripted
   vision answer. One photo plus its caption is one meal, priced by the application.
5. **The architecture in one screen** — the graph in `src/calorai_agent/graph.py`: gather context,
   read media, plan, resolve a reference, then exactly one terminating node. Point at the two
   constraints a reviewer will ask about: one inbound event can produce at most one meal, and a
   reply is chosen by the tool that ran, so a question the agent asks never becomes a meal row.
6. **Latency, with its provenance** — `python scripts/benchmark_latency.py --samples 40`, then open
   `benchmarks/latency.json`. The conversation to have is about the environment block, not the
   milliseconds: it names the commit that produced it (or `-dirty`), the machine's load at both ends
   of the run, and the zero model calls per turn — because these are overhead-only numbers and the
   file says so.
7. **Limitations, volunteered before they are asked** — the Assumptions and trade-offs and
   Limitations sections below: 32 reference foods, thresholds tuned on the eval set rather than
   derived, no provider ever timed end to end, no live Meta round trip.

For a conversation rather than a demo, the three questions this build has real answers to are scale
(the `BEGIN IMMEDIATE` ledger and where Postgres plus a queue would go), privacy (why logs carry ids
and never the plate, and why the sender digest is keyed on the app secret), and reliability (what an
acknowledged `200` costs, and how a delivery lost behind it is still traceable).

## Original constraints

The supplied brief allows a CLI and does not require WhatsApp integration. WhatsApp is added
deliberately, because it is the actual product surface — and isolated so it cannot endanger the local
evaluator experience the brief does require: the whole transport runs against a stand-in Graph API
with no credentials, and nothing in the domain layer — `graph.py`, `planning.py`, `tools.py`,
`repository.py` — imports the transport or an HTTP type.

## Time log

The brief budgets 6-8 hours, and the honest version of that number is `git log`, not an estimate:

```bash
git log --date=iso --pretty='%ad %h %s' | tail -20
```

A short history, on two dates. 2026-09-26, 21:25 to 21:47: the delivery plan, the architecture doc,
and the phase-one vertical slice. 2026-09-28 from 01:36: phases two through six, each one followed by
its own review-and-fix commit, and this document's review round at the end of the list. That is a
6 h 51 m wall-clock window for the
implementation phases, so the work fits the stated budget on the face of the history — with two
caveats worth more than the number: the window includes thinking time that produced no commit, and it
starts at the first commit rather than at the first read of the brief.

## AI tool usage

This was built with an AI coding agent, and the split is worth stating plainly because it is the part
of "who wrote this" a reviewer cannot read off the code:

- The agent wrote essentially all of `src/`, `tests/`, `evals/` and `scripts/`, and ran every gate
  quoted above: pytest with coverage, strict mypy, ruff check and format, both eval harnesses, and
  the latency benchmark.
- The author chose the product surface, the invariants worth defending (the model never prices food;
  logs never carry the plate), and the two policies that shaped every phase: commit and push at each
  phase boundary, and do not open the next phase until the current one's independent review findings
  are fixed.
- The review passes were run by separate agent invocations with no memory of writing the code, and
  they found real defects: a yes/no question the planner was logging as a meal it invented, and an
  unverifiable latency anecdote in this README. Both are gone; the first is now a clause-level rule
  with a regression test and two eval scenarios.
- Every claim in this file is therefore tied to something a reviewer can run — a command, a test
  file, or a committed artifact. Agent-written prose is not evidence, and treating it as such is the
  failure mode this section exists to guard against.
