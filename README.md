# CalorAI WhatsApp Meal-Logging Agent

An interview-focused implementation of a conversational nutrition agent that accepts natural-language meal descriptions and food photos, remembers useful user facts, supports corrections without double-counting, and reports accurate daily totals.

This repository deliberately separates the **agent core** from its delivery channels:

- WhatsApp Cloud API is the primary real-world interface, using Meta's free test number during development.
- A CLI remains available for clean-clone evaluation, deterministic testing, and demos when a public webhook is unavailable.

## Current status

Phases 0 to 4 are complete. The repository contains a locally runnable LangGraph vertical slice with SQLite persistence, deterministic nutrition reference data, typed meal tools, timezone-correct daily totals, and a CLI. Corrections are written as immutable meal revisions inside one transaction, ambiguous requests ask one focused question instead of guessing, refusals and impossible portions never reach a persisted row, and retried inbound messages are answered from an exactly-once event ledger. A text-model planner implements the same interface as the deterministic one and falls back to it on any unusable output, so the slice still runs without API keys. Memory survives a restart as typed records with confidence and provenance: a stated diet shapes later log replies, a protein or calorie target turns totals into progress, and `my usual` replays a saved routine as a new meal. A changed fact supersedes the one it replaces instead of contradicting it, and retrieval is bounded per kind — one diet, one target per nutrient, a few routines — so no saved habit can evict the facts every reply depends on. A photo takes its own path: bytes are checked against their signature before anything is billed, a dedicated vision model reports only the foods and portions it can separate, and the application prices every line from the reference table, so no model's calorie guess reaches the database. The caption beside a photo modifies that plate instead of becoming a second meal — `half of this` halves it, `2 cups of rice` outranks the model's portion, `my usual` cannot also replay a saved routine — and a read too shaky to price asks one focused question while a shaky-but-close one logs a disclosed estimate. Confidence and the model that read the photo are stored with the meal. WhatsApp is now the second
transport for the same graph: a signed Cloud API webhook verifies the delivery, normalizes text and
photo messages into the envelope the CLI already used, acknowledges inside the request thread and
answers behind it, and replies through the Graph API. An unlisted number never reaches the agent,
and Meta's retries are the same inbound event rather than a second meal.

Start with:

- [Delivery phases](docs/PHASES.md)
- [System design](docs/SYSTEM_DESIGN.md)
- [Requirements traceability](docs/REQUIREMENTS.md)
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
  under your app secret. Anything else is refused before it is parsed.
- The handshake and the acknowledgement are fast: Meta is answered with a `200` first, and the
  meal is worked on behind it, so a slow model never turns into a retry storm.
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
```

Phase 1 deliberately uses a small deterministic food table. This keeps the local slice fast, testable, and free of API keys while the provider-backed structured planner is introduced later.

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
