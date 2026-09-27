# Delivery phases

The plan is ordered to maximize evaluator value early. Each phase ends with a demonstrable checkpoint and a clean commit; bonus work cannot displace a missing core feature.

## Phase 0 - Scope, design, and evidence plan (complete)

**Outcome:** an agreed build boundary before code.

- Preserve the original PDF in `docs/brief/`.
- Convert every must-have and submission requirement into a traceable checklist.
- Define transport-independent architecture with WhatsApp and CLI adapters.
- Decide what evidence each feature needs: tests, evals, latency measurements, and demo moments.
- Record risks and explicit non-goals.

**Exit check:** another engineer can explain what will be built, why, and in what order.

## Phase 1 - Thin vertical slice (complete)

**Outcome:** a user can log a text meal and retrieve today's totals through the CLI.

- Bootstrap Python package, configuration, linting, tests, and `.env.example`.
- Create SQLite schema and migrations.
- Define core entities: user, conversation, inbound event, meal, meal item, meal revision, memory.
- Implement structured nutrition estimation and deterministic total aggregation.
- Add `log_meal`, `get_meals`, and `get_daily_totals` operations.
- Add a minimal LangGraph route from message to tools to response.

**Evidence:** clean-clone setup test; persistence test; totals test.

Implemented evidence: strict type checking passes, lint/format checks pass, nine tests pass with 82% package coverage, and a packaged CLI smoke test logs and retrieves a persisted meal. The Phase 1 planner intentionally uses deterministic reference foods; provider-backed interpretation remains a later concern.

## Phase 2 - Corrections and conversational judgment (complete)

**Outcome:** the difficult text cases work correctly and conversationally.

- Resolve references such as `that`, `same as yesterday`, and explicit meal/time hints.
- Implement update/delete as meal revisions inside a transaction.
- Establish ambiguity policy with confidence bands and materiality thresholds.
- Handle `skipped lunch` without inventing a meal; respond naturally to vague grazing.
- Add idempotency for retried inbound events.

**Evidence:** `2 rotis` corrected to `3` changes the original meal and totals exactly once; test-set text scenarios pass.

Implemented evidence: 107 tests pass at 95% package coverage with ruff and strict mypy clean,
and the packaged CLI is exercised end to end in the suite. Reference resolution
returns RESOLVED/AMBIGUOUS/NOT_FOUND and a materially-identical candidate set resolves to the
most recent meal instead of asking. Update and delete write immutable `meal_revisions` rows
inside one `BEGIN IMMEDIATE` transaction, and the mutation is stamped on the inbound event
(`003_mutation_outcomes.sql`) so a redelivered message answers from the database instead of
mutating twice. Encoded judgment: refusals (`no eggs`, `I didn't have rotis`) and impossible
portions (`0`, `-2`, `3 / 0`, `200`) never reach a persisted row — the first is acknowledged
and the second drops into the policy's ask band. Corrections are held to the same confidence
bar as fresh logs. A pointer with no day of its own still resolves shortly after midnight,
while an explicitly dated pointer does not leak across days.

## Phase 3 - Selective persistent memory

**Outcome:** cross-session memory affects behavior without bloating prompts.

- Persist stable dietary constraints, nutrition targets, and user-approved named routines.
- Extract memory candidates as typed records with confidence and provenance.
- Upsert/replace conflicting facts instead of accumulating contradictions.
- Retrieve only relevant memory by type and intent; cap prompt payload.
- Make `my usual` and user targets work after a fresh process start.

**Evidence:** restart process, then demonstrate vegetarian preference, protein target, and usual breakfast retrieval.

## Phase 4 - Separate vision path and multimodal fusion

**Outcome:** a plate photo can be logged safely with an optional caption.

- Download/validate WhatsApp media or accept a CLI image path.
- Call a dedicated vision model with a strict food-observation schema.
- Fuse vision observations and caption into one normalized meal proposal.
- Surface ambiguity when food identity or portion materially affects nutrition.
- Store model confidence/provenance; never create a second meal for the caption.

**Evidence:** image-only and image-plus-caption evals; one inbound event maps to one meal; uncertain image asks one focused question.

## Phase 5 - WhatsApp Cloud API integration

**Outcome:** the agent works on Meta's test number while retaining the local CLI.

- Configure Meta developer app, WhatsApp product, test number, and recipient allow-list.
- Implement webhook verification and signed inbound-event handling.
- Normalize text, image, caption, contact, and message IDs into the transport-neutral envelope.
- Return quick acknowledgement/typing behavior where supported, process safely, then send reply.
- Deduplicate webhook retries by WhatsApp message ID.
- Use a secure public HTTPS tunnel for the demo; never commit tokens.

**Evidence:** end-to-end test-number demo for text, correction, totals, and photo-plus-caption.

**Meta setup note:** a Meta developer app and business portfolio/WhatsApp Business Account are the important Cloud API resources. A Facebook Page may be useful for the broader business presence, but the architecture must not couple meal logging to a Page object. We will verify the exact dashboard flow against the account UI during this phase because Meta changes onboarding screens frequently.

## Phase 6 - Evals, latency, resilience, and observability

**Outcome:** claims are backed by measurements.

- Encode the supplied conversation set plus adversarial cases.
- Define correctness: tool choice, meal state, totals, clarification behavior, memory use, and single-meal multimodal fusion.
- Benchmark warm and cold text/image paths; report sample size, environment, p50, and p95.
- Parallelize independent image/media work; cache safe nutrition lookups; bypass the LLM for deterministic totals.
- Add structured logs, trace IDs, model timings, and optional LangSmith tracing.
- Test provider errors, malformed webhooks, duplicate delivery, and timeouts.

**Evidence:** reproducible benchmark command and machine-readable result file.

## Phase 7 - Submission polish and interview rehearsal

**Outcome:** a reviewer can run, understand, and evaluate the system quickly.

- Finalize README sections required by the brief: setup, models, memory, tools, latency, trade-offs, time log, next steps, and AI-tool use.
- Add architecture and sequence diagrams based on the implemented system.
- Run from a fresh clone with only documented commands.
- Record a 5-10 minute walkthrough: image case, correction case, memory, architecture, latency, and limitations.
- Prepare system-design discussion for scale, privacy, reliability, and provider migration.
- Audit repository for secrets and make the final commit history readable.

**Evidence:** clean-clone checklist, video outline, and completed submission checklist.

## Time-box discipline

For the official 6-8 hour exercise, phases 1-4 and measurable latency are the core. WhatsApp is an intentional enhancement, but the CLI and required behaviors remain shippable if external Meta setup consumes too much time. LangSmith, streaming, vector search, and elaborate deployment are strictly optional.
