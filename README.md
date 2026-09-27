# CalorAI WhatsApp Meal-Logging Agent

An interview-focused implementation of a conversational nutrition agent that accepts natural-language meal descriptions and food photos, remembers useful user facts, supports corrections without double-counting, and reports accurate daily totals.

This repository deliberately separates the **agent core** from its delivery channels:

- WhatsApp Cloud API is the primary real-world interface, using Meta's free test number during development.
- A CLI remains available for clean-clone evaluation, deterministic testing, and demos when a public webhook is unavailable.

## Current status

Phases 0 to 2 are complete. The repository contains a locally runnable LangGraph vertical slice with SQLite persistence, deterministic nutrition reference data, typed meal tools, timezone-correct daily totals, and a CLI. Corrections are written as immutable meal revisions inside one transaction, ambiguous requests ask one focused question instead of guessing, refusals and impossible portions never reach a persisted row, and retried inbound messages are answered from an exactly-once event ledger. A text-model planner implements the same interface as the deterministic one and falls back to it on any unusable output, so the slice still runs without API keys. Persistent memory, vision, and WhatsApp arrive in subsequent phases and are not claimed as implemented yet.

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
- FastAPI for webhook and health endpoints
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
how am I doing today?
what did I eat today?
actually that was 3 parathas
no eggs
```

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
