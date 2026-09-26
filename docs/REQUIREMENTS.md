# Requirements traceability

## Core requirements

| Brief requirement | Planned implementation | Primary evidence |
|---|---|---|
| Conversational agent with tool calling | LangGraph plus narrow typed tools | Scenario evals and traces |
| Persistent database | SQLite repositories and migrations | Restart integration test |
| Correct daily totals | Deterministic aggregation over active meal revisions | Correction/delete tests |
| Separate image model | Dedicated vision client feeding caption fusion | Provider-spy test and image eval |
| Persistent memory | Typed selective memories with bounded retrieval | Cross-session eval |
| Multi-turn ambiguity | Confidence/materiality policy and focused clarification | Ambiguity eval set |
| p50/p95 text and image latency | Instrumented benchmark harness | Versioned benchmark output |

## Supplied scenarios

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

- Clean clone setup and descriptive incremental commits.
- README sections for overview, setup, text/vision model choices, memory, tool design, p50/p95 latency, trade-offs, time breakdown, next steps, and AI-tool usage.
- 5-10 minute video demonstrating at least one image case and one correction case, plus architecture, memory, latency, challenges, and any bonuses.
- Honest disclosure of incomplete work.
- Optional public LangSmith trace link.

## Intentional extension

The brief explicitly says WhatsApp integration is not required. This project adds WhatsApp Cloud API with Meta's test number as a thin adapter because it improves product realism. CLI support remains mandatory internally so external configuration cannot prevent local evaluation.

## Non-goals for the first complete submission

- Production medical or dietary advice.
- A polished frontend.
- Kubernetes or elaborate cloud infrastructure.
- A comprehensive nutrition database.
- Vector search unless typed memory retrieval proves insufficient.

