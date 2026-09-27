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

Implemented evidence: 116 tests pass at 95% package coverage with ruff and strict mypy clean,
and the packaged CLI is exercised end to end in the suite. Reference resolution
returns RESOLVED/AMBIGUOUS/NOT_FOUND and a materially-identical candidate set resolves to the
most recent meal instead of asking. Update and delete write immutable `meal_revisions` rows
inside one `BEGIN IMMEDIATE` transaction, and the mutation is stamped on the inbound event
(`003_mutation_outcomes.sql`) so a redelivered message answers from the database instead of
mutating twice. Encoded judgment: refusals (`no eggs`, `I didn't have rotis`) and impossible
portions (`0`, `-2`, `3 / 0`, `200`) never reach a persisted row — the first is acknowledged
and the second drops into the policy's ask band. Corrections are held to the same confidence
bar as fresh logs. A correction does not have to announce itself: `that was 3 rotis` updates
the roti meal already on the record instead of logging a second one, and it only does so when
there is such a meal. A correction that names a food the logged meals do not contain replaces
that meal, while additive wording (`also`, `plus`) merges. A pointer with no day of its own
still resolves shortly after midnight but reaches back no further than last night, so it can
never silently rewrite days-old history, while an explicitly dated pointer does not leak
across days at all.

## Phase 3 - Selective persistent memory (complete)

**Outcome:** cross-session memory affects behavior without bloating prompts.

- Persist stable dietary constraints, nutrition targets, and user-approved named routines.
- Extract memory candidates as typed records with confidence and provenance.
- Upsert/replace conflicting facts instead of accumulating contradictions.
- Retrieve only relevant memory by type and intent; cap prompt payload.
- Make `my usual` and user targets work after a fresh process start.

**Evidence:** restart process, then demonstrate vegetarian preference, protein target, and usual breakfast retrieval.

Implemented evidence: 161 tests pass at 95% package coverage with ruff and strict mypy clean, and
the exit evidence is demonstrated in two separate CLI processes: `2 idlis and coffee for breakfast`
then `remember this as my usual breakfast`, `i'm vegetarian btw`, `aim for 120g protein a day` in
the first, and `my usual breakfast`, `1 chicken and 2 idlis`, `how am I doing today?` in the second,
which logs the saved routine from the database, notes that chicken is not vegetarian, and reports
`44 of 120 g protein` against the stored target. Memory is three typed records, not a free-text
scratch pad: `DietaryConstraint`, `NutritionTarget` (keyed per nutrient, bounds-checked so a
`900000 g protein` target never lands), and `NamedRoutine` (refused without foods). Each carries the
confidence it was stated at and the `source_event_id` that stated it, so a redelivered or
crash-windowed memory message is answered from the database instead of writing twice. A new fact
replaces the active fact in its slot in the same transaction (`memory_key` gives one `diet` slot, so
a changed mind supersedes rather than contradicts; `active_memories` reads only the live rows, newest
first). Retrieval is bounded per kind — one diet, one target per nutrient, six routines — because a
global cap would let a habit of saved routines evict the diet and the targets that shape every reply,
and the payload reaching a planner prompt is grouped by kind and capped at 8 lines. A message can
state a fact and a meal at once: `i'm vegetarian, had 2 idlis` keeps the diet and names the 2 idlis
it did not log instead of dropping them. Memory changes behavior, not
just context: a diet names the logged foods it rules out and says so once per reply, a target turns
totals into progress, and `my usual` becomes a `LOG_MEAL` draft marked user-confirmed, with extra
foods in the same message (`my usual and 1 egg`) folded into the remembered lines rather than
replacing them. An ambiguous
routine is never guessed — an unlabeled save asks which meal to remember, `my usual lunch` with only
a breakfast saved says which slot is missing, and several saved routines with no meal type asks
instead of inventing a number.

## Phase 4 - Separate vision path and multimodal fusion (complete)

**Outcome:** a plate photo can be logged safely with an optional caption.

- Download/validate WhatsApp media or accept a CLI image path.
- Call a dedicated vision model with a strict food-observation schema.
- Fuse vision observations and caption into one normalized meal proposal.
- Surface ambiguity when food identity or portion materially affects nutrition.
- Store model confidence/provenance; never create a second meal for the caption.

**Evidence:** image-only and image-plus-caption evals; one inbound event maps to one meal; uncertain image asks one focused question.

Implemented evidence: 206 tests pass at 95% package coverage with ruff and strict mypy clean, and
`python scripts/run_photo_evals.py` runs 14 image-only and image-plus-caption scenarios through the
real graph with a scripted vision answer — no API key, no network. A photo owns a graph node
(`read_media`) and its own client: bytes are judged by their jpeg/png/webp signature, capped at
8 MB, and only ever reach the vision model, never the text planner. The model returns one strict
JSON object of `{name, quantity, confidence, alternative}` per food with `extra="forbid"`, so an
answer that carries its own calorie estimate is refused outright instead of logged. Every line is
then priced from the reference table — the 960 kcal on a photographed biryani is the table's number
— while the observation's confidence and the reading model are stored with the meal as
`origin = vision_fusion`. Fusion applies the caption as *modifiers* on the photographed plate, with
recent meals and memory withheld from it: a stated portion outranks the model's guess line by line,
`half of this` scales every line while `half a dosa` stays one food's portion, and `my usual` cannot
also replay a saved routine. The share applies only to what the photo shows — `half of this, plus a
banana` is a half plate and a whole banana — and a caption asking about *another* meal (a correction,
a delete, a totals question) neither acts on that meal nor lends its date to the photo: the plate
logs today and the request is told it belongs in a message of its own. One inbound event therefore
maps to exactly one meal, verified by a redelivered photo that logs one meal and shows the model one
photo, and by two photos sent in the same second becoming two meals. Uncertainty is decided by
arithmetic rather than prose: a read between 0.55 and 0.8 confidence logs a disclosed estimate, a
weaker one asks exactly one question — an either/or when its second guess sits 150 kcal or 12 g
protein away, how much of a food it was when the portion (or the total two alias lines add up to) is
one no plate could hold, the plain naming question otherwise — and a caption that already named the
food ends the question before it is asked. A durable fact stated beside a plate that cannot be
counted is still saved before the question is asked, and a food the caption mentioned without a
place on the plate is named as not logged rather than dropped quietly. Every
failure boundary answers honestly instead of guessing: no vision key configured, a file that is not
a photo, an oversized attachment, a provider error, an empty plate, and a food the table cannot
price (which is named, never invented). Two known boundaries: only a local CLI path is readable
until Phase 5 supplies a WhatsApp media source, and a redelivered photo turn replays its logged
numbers without the photo's reply notes, because those notes are conversation text and not
persisted state.

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
