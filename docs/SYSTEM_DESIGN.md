# System design v0

## Boundary

```text
WhatsApp webhook ----\
                      -> transport normalizer -> conversation graph -> response adapter
Local CLI -----------/                              |      |      |
                                                     |      |      +-> vision model
                                                     |      +--------> text model
                                                     +---------------> application tools
                                                                         |
                                                              repositories / SQLite
```

The transport layer knows WhatsApp IDs and media retrieval. The application layer knows users, meals, nutrition, memory, and conversation decisions. Neither model writes directly to the database; typed tools validate and apply mutations.

## Proposed request flow

1. Adapter converts text and/or image into one `InboundMessage` with an idempotency key.
2. A deterministic router handles simple totals/retrieval requests directly when safe.
3. Images go to the dedicated vision model; its structured observations are fused with the caption.
4. The graph retrieves only intent-relevant memories and recent meal references.
5. The text model emits a typed decision: answer, clarify, or call a bounded application tool.
6. A database transaction creates/revises meal state and recalculates totals from active revisions.
7. The adapter sends one conversational reply and records latency spans.

## Tool boundaries

- `propose_meal`: normalize candidate items and nutrition estimates without persistence.
- `log_meal`: commit one validated meal proposal for one inbound event.
- `revise_meal`: replace the active revision of a resolved prior meal.
- `delete_meal`: mark a resolved meal inactive while keeping audit history.
- `list_meals`: retrieve bounded meals by user/date for references and summaries.
- `get_daily_totals`: deterministic aggregation; no model arithmetic.
- `upsert_memory`: persist a typed, worthwhile user fact with provenance.
- `get_relevant_memories`: bounded retrieval by type and current intent.

The model does not receive a generic SQL tool. Narrow operations make correctness and evals legible.

## Data correctness model

- `inbound_events.external_id` is unique, preventing WhatsApp retry duplication.
- A logical meal owns multiple immutable revisions; exactly one revision is active.
- Corrections create a new revision and supersede the previous one atomically.
- Daily totals sum active meal-item revisions for the user's local calendar day.
- Original text, structured interpretation, confidence, and model/provider provenance are retained for debugging.

This makes correction history auditable while ensuring totals never include both old and new values.

## Memory policy

Store:

- stable dietary constraints (`vegetarian`, allergies);
- explicit goals (`140g protein`, calorie target);
- user-approved or repeatedly confirmed named routines (`my usual`);
- lightweight locale/timezone preferences needed for correct dates.

Do not store:

- every message or full transcript as memory;
- one-off meals merely because they were mentioned;
- inferred sensitive facts without clear user evidence;
- low-confidence facts that could meaningfully alter nutrition advice.

Retrieval is filtered first by typed intent and validity, then limited to a tiny ranked set. Explicit new statements supersede conflicting active facts.

## Ambiguity policy

- **Low impact / high confidence:** log and mention the estimate briefly.
- **Medium uncertainty:** log a marked estimate when the likely range does not materially change the user's decision.
- **High impact or conflicting evidence:** ask one focused question before committing.
- **Vision uncertainty:** describe what is uncertain; never silently present a confident food identity.

The decision threshold will be encoded and eval-tested, not left solely to prompt prose.

## Latency strategy

- Avoid an agent loop for totals and straightforward retrieval.
- Use structured single-pass extraction for common meal logs.
- Fetch media and independent context concurrently when possible.
- Keep memory retrieval bounded and indexed.
- Cache stable nutrition reference data, not user-specific mutations.
- Instrument adapter, media download, memory lookup, model, tool, database, and send-reply spans.

Benchmarks will distinguish text vs image, warm vs cold, and model time vs total user-visible time.

## Production discussion beyond the test

For scale, replace in-process work with a durable queue, use Postgres, partition by user, encrypt sensitive fields, apply retention/deletion controls, rotate Meta tokens, add provider fallbacks and circuit breakers, and keep idempotency across webhook receipt, tool mutation, and reply delivery. Those are interview discussion points, not prerequisites for a strong 6-8 hour submission.

## Risks

- Meta onboarding/tunnel setup can consume the time budget; CLI remains the evaluator-safe path.
- Model-derived nutrition is approximate; communicate estimates and optimize for conversational/system correctness.
- Relative references need careful candidate resolution; never edit a meal when multiple candidates remain material.
- Provider latency can dominate; measure honestly and preserve trace evidence.

