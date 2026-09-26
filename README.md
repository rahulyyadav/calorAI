# CalorAI WhatsApp Meal-Logging Agent

An interview-focused implementation of a conversational nutrition agent that accepts natural-language meal descriptions and food photos, remembers useful user facts, supports corrections without double-counting, and reports accurate daily totals.

This repository deliberately separates the **agent core** from its delivery channels:

- WhatsApp Cloud API is the primary real-world interface, using Meta's free test number during development.
- A CLI remains available for clean-clone evaluation, deterministic testing, and demos when a public webhook is unavailable.

## Current status

Phase 0 is complete: the source brief is preserved, requirements are mapped, the system boundary is defined, and the work is split into implementation phases. No claim is made yet that the agent is implemented.

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

