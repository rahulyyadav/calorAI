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

Implemented evidence: 211 tests pass at 95% package coverage with ruff and strict mypy clean, and
`python scripts/run_photo_evals.py` runs 16 image-only and image-plus-caption scenarios through the
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
logs today and the request is told it belongs in a message of its own. That guard reads the caption's
words and not only the intent the planner settled on, because a durable fact answers first —
`i'm vegetarian, delete yesterday's lunch` still keeps the diet *and* surfaces the delete. A day named
in a caption moves the plate only when the caption also says what was eaten, so a photo compared with
"yesterday's lunch" stays today. The plausibility ceiling is checked against the meal rather than one
line of it — in the photographed reading, in the model planner and in the deterministic one — so
`20 rotis and 25 rotis` asks how much instead of logging 45. One inbound event therefore maps to
exactly one meal, verified by a redelivered photo that logs one meal and shows the model one photo,
and by two photos sent in the same second becoming two meals. Uncertainty is decided by
arithmetic rather than prose: a read between 0.55 and 0.8 confidence logs a disclosed estimate, a
weaker one asks exactly one question — an either/or when its second guess sits 150 kcal or 12 g
protein away, how much of a food it was when the portion (or the total two alias lines add up to) is
one no plate could hold, the plain naming question otherwise — and a caption that already named the
food ends the question before it is asked. A durable fact stated beside a plate that cannot be
counted is still saved before the question is asked, and a food the caption mentioned without a
place on the plate is named as not logged rather than dropped quietly. Every failure boundary
answers honestly instead of guessing: no vision key configured, a file that is not
a photo, an oversized attachment, a provider error, an empty plate, and a food the table cannot
price (which is named, never invented). Two known boundaries: a photo arrives either as a local CLI
path or as a WhatsApp media id, and nothing else is read, and a redelivered photo turn replays its
logged numbers without the photo's reply notes, because those notes are conversation text and not
persisted state.

## Phase 5 - WhatsApp Cloud API integration (complete)

**Outcome:** the agent works on Meta's test number while retaining the local CLI.

- Configure Meta developer app, WhatsApp product, test number, and recipient allow-list.
- Implement webhook verification and signed inbound-event handling.
- Normalize text, image, caption, contact, and message IDs into the transport-neutral envelope.
- Return quick acknowledgement/typing behavior where supported, process safely, then send reply.
- Deduplicate webhook retries by WhatsApp message ID.
- Use a secure public HTTPS tunnel for the demo; never commit tokens.

**Evidence:** end-to-end test-number demo for text, correction, totals, and photo-plus-caption.

Implemented evidence: 338 tests pass at 95% package coverage with ruff and strict mypy clean.
`whatsapp.py` decides nothing about meals; it decides only whether an event may be trusted. An
`X-Hub-Signature-256` is an HMAC-SHA256 of the *exact raw bytes* under the app secret, compared in
constant time before the body is parsed, so re-serializing the JSON cannot make a forged delivery
look valid, and the subscription handshake echoes `hub.challenge` only for our own verify token.
`normalize_webhook` reads Meta's `entry → changes → value → messages` into the same
`InboundMessage` envelope the CLI builds — words from `text.body`, a photo into a `MediaRef` with
`source = whatsapp_media` carrying its own message id, a caption as the message text — and it
authorizes every item by the number that actually sent it, falling back to the contact block's
`wa_id` when only that names the sender, so a stranger riding in a listed conversation is refused
and an empty allow-list refuses everyone. A delivery receipt for a message *we* sent is skipped
rather than answered; a voice note, video, file, sticker, map pin, saved contact or button press is
answered with one honest sentence naming what it sounded like, and because declines share the
inbound ledger a redelivered voice note declines once. Meta's `wamid` is the deduplication key, and
a message that arrives without one is fingerprinted from its own content, so a retry is still one
event and at most one meal. `WebhookApplication` acknowledges first and works behind that ack
through an injectable executor whose default is the production thread pool — a test opts into the
inline one, a deployment never does — and one test proves the `200` is written while the turn is
still blocked inside the planner. The message is marked seen with Meta's typing indicator carried
on that same read request, because the Cloud API has no standalone typing call. Every item of a
delivery is guarded on its own, so a locked database or a lost reply costs one message and not the
batch: the user hears `I could not finish that message`, and the ledger row that turn leaves open
is closed with that same honest reply rather than promising "still working on it" to every retry
forever. A message this adapter fingerprinted because Meta sent no id is answered but never marked
read, since there is no such message to receipt. `GraphClient` keeps the token in a header only,
because URLs are copied into access logs, reports a failure by its class rather than echoing a
provider message that can carry one, and follows a media pointer only when it names an `https` url,
because the bearer token travels on that second request; `WhatsAppMediaSource` holds downloaded
bytes to the same jpeg/png/webp signature test and 8 MB cap as a CLI photo. The HTTP shell is a
stdlib `ThreadingHTTPServer` on a configurable path — 404 off path, 401 and 403 on the way in, a
declared body over the cap refused with 413 and a non-numeric `Content-Length` with 400 before any
of it is read, and access lines at debug level cut at the first `?` because the handshake carries
the verify token in the query string, so no body or token reaches a log — and it added no
dependency. `main()` refuses to start on half a configuration: any one of the four
WhatsApp secrets missing, or an empty allow-list, exits 2 with the instructions to fix it. The
secret itself stays out of the repository: `.env.example` lists the Phase 5 keys as empty values
and `.env` remains gitignored. Known boundary: the live test-number walkthrough (text, correction,
totals, photo-plus-caption through a public HTTPS tunnel) has not been run, since it needs the
credentials the reviewer supplies; every claim above is exercised against a stand-in Graph API
instead, with no key and no network.

**Meta setup note:** a Meta developer app and business portfolio/WhatsApp Business Account are the important Cloud API resources. A Facebook Page may be useful for the broader business presence, but the architecture must not couple meal logging to a Page object, and it does not: nothing in the code reads a Page id. Meta changes the onboarding screens frequently, so the numbered steps in the README are written against the values the process needs (a phone number id, a token, an app secret, a verify token of our own choosing) rather than as a transcript of a dashboard that may have moved since.

## Phase 6 - Evals, latency, resilience, and observability (complete)

**Outcome:** claims are backed by measurements.

- Encode the supplied conversation set plus adversarial cases.
- Define correctness: tool choice, meal state, totals, clarification behavior, memory use, and single-meal multimodal fusion.
- Benchmark warm and cold text/image paths; report sample size, environment, p50, and p95.
- Parallelize independent image/media work; cache safe nutrition lookups; bypass the LLM for deterministic totals.
- Add structured logs, trace IDs, model timings, and optional LangSmith tracing.
- Test provider errors, malformed webhooks, duplicate delivery, and timeouts.

**Evidence:** reproducible benchmark command and machine-readable result file.

Implemented evidence: 425 tests pass at 95% package coverage with strict mypy clean on 22 source
files, ruff check and format clean, 16/16 photo scenarios and 28/28 conversation scenarios green —
all with no API key and no network, so a reviewer can reproduce every claim on a clean clone.
`observability.py` gives each turn a trace id from a contextvar and emits nine named events: `turn`
(channel, route, photo, redelivered, duration_ms), `model_request` (kind, model, failed,
response_chars), `media_fetch`, `message_answered`, `media_prefetched`, `media_prefetch_failed`,
`planner_fast_path`, `delivery_failed`, and `tracing_disabled`. Meal text and photo bytes are
deliberately absent from every line: this is health data, and a log that quotes the plate is a leak
that outlives the database. Sender ids appear only as a digest keyed on the app secret, so a number
cannot be recovered from a log by guessing the ones it might have been. Logs render as text for the
CLI or JSON for a collector, and LangSmith is asked for
only when `CALORAI_TRACING` and a key are *both* present — enabling tracing without a key reports
"untraced" instead of buffering spans to send somewhere later.
`evals/conversation_scenarios.json` carries all eleven conversations from the brief verbatim plus
seventeen adversarial cases, and `scripts/run_evals.py` grades each on the five dimensions the brief
asks about: tool choice, meal state, totals, clarification behavior, memory use, and single-meal
multimodal fusion. Grading reads routes from the application's own trace, meals and totals from the
repository, memory kinds from active records, and fusion from the vision client's call count — so a
route alone never passes an eval, and the harness needed no test-only hook inside the graph. Each
scenario can span day offsets and timezones, rebuild the agent mid-run to prove a restart kept its
memory, redeliver one event to prove exactly-once, and assert the number of vision calls, which is
how "one photo plus its caption is one meal" is measured rather than claimed. `docs/EVALS.md`
records the semantics and, honestly, what these evals do not prove.
`scripts/benchmark_latency.py` measures cold and warm text, read, and image paths at 40 samples each
and writes `benchmarks/latency.json` with sample size, environment (including the machine's 1-minute
load average, sampled at both ends of the run and reported as the worse, because a 5 ms number timed
during a contact-indexing burst is a different number), p50 and p95, how many model requests each path
billed, and a `model_backing` field naming that both models were stand-ins — a 5 ms figure without
that label is misreadable as a provider result. The commit field says `-dirty` when the measured tree
was not the commit it names.
Measured on Apple Silicon/Darwin with 8 CPUs at load 2.23, from the commit that carries this
sentence: text cold 5.24/7.41 ms, text warm 5.04/6.40 ms, read cold 4.33/5.20 ms, read warm
4.28/7.94 ms, image cold 4.91/6.27 ms, image warm 5.50/6.79 ms p50/p95, each with 0.0 model calls per
turn, and the cold paths reporting database setup separately at 10.7-11.2 ms p50 rather than hidden
inside a percentile. The load field exists because
the same command on the same laptop measures a slower p95 when the machine is busy — compare
`load_average_1m` before comparing two runs, which is the honest way to read a 5 ms number. Warm
running no faster than cold is the honest shape of a 5 ms measurement: `text_warm` beat `text_cold`
by 0.2 ms on this run while `image_warm` lost to `image_cold` by 0.59 ms, and `read_warm` holds both
the best p50 and the worst p95 in the table. SQLite page cache and allocator state dominate at this
scale.
What keeps the numbers in milliseconds at all is the fast paths — `RuleBasedPlanner.deterministic_read`
answers totals and "what did I eat" without a model call, which the artifact now proves by billing
0.0 model requests per read turn instead of asserting it; the nutrition walk is a bounded
`functools.lru_cache`, bounded because the names arriving in it are model output; and photo bytes are
fetched in parallel per delivery through a byte-bounded `MediaCache` keyed on the media id.
`tests/test_resilience.py` covers the failure side:
a timeout, refused connection, 500, 429, malformed JSON or prose instead of JSON from the provider
each still logs the meal through the deterministic fallback, a model that invents `calories: 9000`
cannot put 9000 in the database (the row holds the 520 the reference table prices, and the reply
never repeats the invention), the trace names the surviving exception class, a dead or chatty vision
model gets one honest sentence and zero meals, non-image bytes never reach a provider, and two
threads delivering the same `wamid` produce one meal and one inbound-ledger row.
Building this found a real bug: "had a banana for a snack" was asking the user how many bananas they
ate, because the article in "for a snack" was counted as an unclaimed portion. `_ARTICLES` in
`planning.py` now requires a food name after an article before it counts as a quantity, digits still
guard against absurd amounts, and a regression test holds it. Known boundary: the deterministic
planner cannot name a food the table cannot price, so "a protein bar and a smoothie" logs the
smoothie and stays silent about the bar, while the model-backed path discloses it — recorded in
`docs/EVALS.md` and in that scenario's own claim rather than hidden.

Independent review of this phase found two ways the application lost a meal it had already been
given, and both are now pinned by tests and evals rather than by prose. A question that also stated a
portion — "what did I eat today? i also had 2 parathas for breakfast" — was read as only the
question, so the parathas were never logged and the reply said nothing was logged; the list read now
refuses any message that states an amount, and it refuses an amount too large to believe as well, so
the impossible portion surfaces as a question instead of vanishing. Reviewing that rule also exposed
its neighbour: "did I eat biryani today?" was answering by logging a biryani, inventing the very meal
the user was asking about. Behind the webhook, a photo download that failed for a reason nobody
modelled (a `httpx.InvalidURL` from a base URL without a scheme, not a `MediaError`) escaped the
warm-up, ended the worker, and left an already-acknowledged delivery unanswered with no Meta retry
ever coming — the warm-up now swallows anything, the delivery logs how many photos it actually got,
and a discarded worker future reports itself instead of dying silently. The privacy rule outlived
the two rounds it had already survived: a rejected vision answer was quoting the model's description
of the plate into the log, and refusal lines were filing a blocked stranger's phone number at the
default level, so both now report a class name or a stable digest — enough to correlate one sender,
not enough to become a contact list. A media cache that added one photo's bytes twice was evicting
the other dinners to pay for it. The benchmark could not support the claims made of it: the committed
artifact predated every file it was cited as measuring, timed only `log_meal` turns, and called the
path "cold" while excluding the cold cost, so it was rebuilt from the code it describes with the read
path added, model calls counted per turn, database setup reported separately, and load average
recorded —
then regenerated again, because the first regeneration was itself timed during a load-11.6 burst and
the artifact said nothing about it. Load is now sampled at both ends of the run and reported as the
worse of the two, and the commit field gains a `-dirty` suffix when the measured tree is not the
commit it names, so an artifact cannot claim to describe a commit it did not run from.
Trace ids now cross the parallel-fetch pool, `span()` cannot be crashed by a field the measured work
wrote into the dict it was handed, and text log values are quoted so a reason with spaces in it is
still one field.

Re-verifying the same phase found the mirror of its first bug rather than a new one. The guardrail
that stopped a list question from swallowing a stated meal had made a yes/no question carrying an
amount ("did i eat 2 parathas today?") log two parathas — inventing the meal the user was asking
about, which is the same failure in the opposite direction. No whole-message test can express both,
so the planner splits on the boundaries the user actually typed (`_CLAUSE_BOUNDARY`; a comma or
period counts only when text follows it, so "1.5 cups" stays one number) and asks a question of the
clauses that are not asking. A stated meal no longer needs a number either: "what did i eat today? i
also had eggs for breakfast" logs the eggs, because reading it as only the question dropped them in
silence. Four regression tests and two new eval scenarios pin the two phrasings the fix is for.

The same round closed two smaller findings. The digest that replaced a logged phone number was an
unsalted truncation of SHA-256 over a ten-digit id space, which is a lookup table rather than a mask:
anyone holding the log can confirm a number they already suspect by trying it. It is now an HMAC
keyed on the app secret, and the refusal path and the failed-reply path use the same key so one
sender still correlates across lines. And a delivery that died behind the `200` could only be
reported by the future's done callback, which runs on another thread with no trace of its own and can
name nothing but the exception class — the delivery now logs its own trace id and the number of
messages left unanswered before it re-raises.

## Phase 7 - Submission polish and interview rehearsal

**Outcome:** a reviewer can run, understand, and evaluate the system quickly.

- [x] Finalize README sections required by the brief: setup, models, memory, tools, latency, trade-offs, time log, next steps, and AI-tool use.
- [ ] Add architecture and sequence diagrams based on the implemented system.
- [x] Run from a fresh clone with only documented commands — `docs/CLEAN_CLONE.md`.
- [x] Write the 5-10 minute walkthrough: image case, correction case, memory, architecture, latency, and limitations — README `## Walkthrough`. The video recording of it is the author's step and has not happened.
- [ ] Prepare system-design discussion for scale, privacy, reliability, and provider migration.
- [x] Audit repository for secrets and make the final commit history readable: `git log -p` over all commits and a working-tree scan found nothing — `.env` is untracked and ignored, `.env.example` carries keys with empty values only, and the single match for an assignment-looking string was a docstring showing a user how to set one.

**Evidence:** clean-clone checklist, video outline, and completed submission checklist.

## Time-box discipline

For the official 6-8 hour exercise, phases 1-4 and measurable latency are the core. WhatsApp is an intentional enhancement, but the CLI and required behaviors remain shippable if external Meta setup consumes too much time. LangSmith, streaming, vector search, and elaborate deployment are strictly optional.
