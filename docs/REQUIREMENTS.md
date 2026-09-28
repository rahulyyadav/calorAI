# Requirements traceability

## Core requirements

| Brief requirement | How it is met | Primary evidence |
|---|---|---|
| Conversational agent with tool calling | LangGraph plus narrow typed tools | Scenario evals and traces |
| Persistent database | SQLite repositories and migrations | Restart integration test |
| Correct daily totals | Deterministic aggregation over active meal revisions | Correction/delete tests |
| Separate image model | Dedicated vision client feeding caption fusion | Provider-spy test and image eval |
| Persistent memory | Typed selective memories with bounded retrieval | Cross-session eval |
| Multi-turn ambiguity | Confidence/materiality policy and focused clarification | Ambiguity eval set |
| p50/p95 text and image latency | Instrumented benchmark harness | Versioned benchmark output |

## Supplied scenarios

Every conversation in the brief appears verbatim in `evals/conversation_scenarios.json` as the eleven
`supplied:` scenarios — the table below splits the photo case into "a photo" and "a photo with a
caption", which the harness also grades as two. `python scripts/run_evals.py` and
`python scripts/run_photo_evals.py` print a `PASS` per scenario.

| Scenario | Expected behavior |
|---|---|
| `had 2 parathas and chai for breakfast` | Log one breakfast with reasonable estimated items |
| `leftover biryani, maybe two thirds of the box` | Interpret fraction, expose estimate without excessive questioning |
| `skipped lunch but grazed all afternoon` | Do not invent lunch; ask one useful question about grazing if needed |
| `same as yesterday` | Retrieve yesterday's appropriate meal and copy deliberately |
| `actually that was 3 rotis not 2` | Revise the referenced meal; totals include only 3 rotis |
| protein/calorie questions | Return deterministic current-day totals |
| photo | Dedicated vision path; clarify material uncertainty |
| photo + `half ... brother's` | Fuse caption and vision into exactly one half-portion meal |
| `my usual` | Resolve a persisted named routine or ask once to establish it |
| `i'm vegetarian btw` | Persist a dietary preference and use it in later interpretation |

## Submission requirements to preserve

Status as of the last phase: done, or named as the thing that is not.

- [x] Clean clone setup and descriptive incremental commits — `docs/CLEAN_CLONE.md` is that run,
  recorded with what happened rather than what should happen.
- [x] README sections for overview, setup, text/vision model choices, memory, tool design, p50/p95
  latency, trade-offs, time breakdown, next steps, and AI-tool usage.
- [x] A written 5-10 minute walkthrough covering one image case and one correction case, plus
  architecture, memory, latency and limitations — the `## Walkthrough` section of the README, with
  the replies it prints quoted from the clean-clone run.
- [ ] The video itself. It has not been recorded: this environment can start the CLI but cannot
  capture a screen, so recording it is the author's step, and the walkthrough section is its script.
- [x] Honest disclosure of incomplete work — the README's Limitations, and the two keyless gaps in
  `docs/CLEAN_CLONE.md` (no vision key, no public tunnel).
- [ ] A public LangSmith trace link — optional, and deliberately absent: tracing is implemented
  behind `CALORAI_TRACING` and unit-tested, but no project was created and no trace published. The
  equivalent evidence that does exist is the per-turn `trace=` field on every log line.

## Intentional extension

The brief explicitly says WhatsApp integration is not required. This project adds WhatsApp Cloud API with Meta's test number as a thin adapter because it improves product realism. CLI support remains mandatory internally so external configuration cannot prevent local evaluation.

## Non-goals for the first complete submission

- Production medical or dietary advice.
- A polished frontend.
- Kubernetes or elaborate cloud infrastructure.
- A comprehensive nutrition database.
- Vector search unless typed memory retrieval proves insufficient.

