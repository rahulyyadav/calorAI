# Evals

The claim this file defends is narrow on purpose: the agent's *judgement* is measured, not admired.
Every scenario is graded against the database and the trace, so a passing eval says something a
reviewer can check, not something the model once happened to say.

## What "correct" means

Six things are graded, one per promise the brief makes. Each has a field in the scenario file and a
concrete object it is read from.

| Dimension | Scenario field | Read from | What a failure looks like |
|---|---|---|---|
| Tool choice | `routes` | the `turn` span the graph logs for itself | a totals question answers by logging a meal |
| Meal state | `meals`, `meal_items`, `meal_quantities` | `MealRepository.list_for_day` | a correction leaves two meals where one belongs |
| Totals | `totals` (`kcal`, `protein_g`, `carbs_g`, `fat_g`) | `MealRepository.totals_for_day` | a deletion does not move the day's number |
| Clarification | `asks`, plus meal state | the reply text and the day's rows | a question that also logged, or a guess that did not |
| Memory | `memories`, plus a reply fragment | `MealRepository.active_memories` | a saved diet that never reaches the next reply |
| Multimodal fusion | `meals`, `vision_calls` | the day's rows and the vision client's call count | a photo and its caption becoming two meals |

Reply text is asserted by fragment (`turn_contains`), never in full: the phrasing is allowed to
improve, the numbers are not.

`routes` names the graph node that answered the turn, so it is read as *where the turn went*, not as
*what happened*. A message can enter `log_meal` and still be refused by the ambiguity policy — the
meal-state checks are what decide whether anything was actually written. That is why no scenario in
this suite is graded on route alone.

## How the runner works

`scripts/run_evals.py` replays a scenario turn by turn against a real SQLite database in a temporary
directory, then grades the six dimensions above.

- Days are keyed by offset from the scenario's `start`, resolved in the scenario's timezone, so a
  23:30 meal in `Asia/Kolkata` is graded on the user's date rather than the server's.
- A turn marked `"new_session": true` rebuilds the whole application — database, repository, planner,
  agent — from the same file. That is the cross-session half of the memory requirement.
- A turn marked `"redeliver": true` delivers the identical event twice. One meal and a replayed
  second turn is the pass condition; two meals is a data-integrity failure.
- `vision_calls` counts how often the photo was actually shown to the model, so a caption cannot
  quietly trigger a second read of the same plate.

## The two sets

| Set | Runner | Cases | Covers |
|---|---|---|---|
| `evals/conversation_scenarios.json` | `scripts/run_evals.py` | 11 supplied + 17 adversarial | the brief's conversation set verbatim, then the ways it breaks |
| `evals/photo_scenarios.json` | `scripts/run_photo_evals.py` | 16 | vision reads, caption fusion, unreadable plates, absurd portions |

The adversarial cases exist because the supplied set is the easy half: a correction with nothing to
correct, "same as yesterday" on an empty yesterday, "my usual" with no routine saved, totals on an
empty day, the same delivery twice, a plate the vision model cannot read, an amount no plate could
hold, a diet stated in the same breath as a meal, and a nutrition question that is not a totals
request.

## No key, no network, no luck

Both runners use the deterministic planner and a vision answer scripted from the scenario file. A
reviewer sees exactly which plate and which message the agent was given, and a failure points at the
application's judgement rather than at a provider's mood or a bill. The contract of the model-backed
planner is graded separately, with every model payload scripted, in `tests/test_model_planner.py`.

Reproduce with:

```bash
.venv/bin/python scripts/run_evals.py
.venv/bin/python scripts/run_photo_evals.py
```

`run_evals.py --only <name>` runs a subset by name. Both scripts exit non-zero on any failure, so
they work as CI steps.

## What these evals do not prove

- **They grade the fallback path.** With the reference-table planner the parsing is deterministic, so
  a scenario measures policy and state handling. Pointed at a real model the same files become the
  scorecard, but the numbers will move and the phrasing may differ.
- **An unpriceable food is dropped quietly on that path.** "had a protein bar and a smoothie" logs
  the smoothie and says nothing about the bar: the deterministic planner only recognises names the
  reference table can price, and it would rather under-count than invent. The model path does
  disclose it ("I have no reference data for protein bar"), which is what
  `adversarial: a food with no reference row is dropped, never invented` pins down.
- **Memory provenance is not always in the prose.** A replayed routine stamps the note on the stored
  meal rather than repeating it back, so routine scenarios assert the logged rows and the memory
  kind, and only quote the reply where the reply is the feature.
- **Nutrition accuracy is out of scope.** The reference table is a pricing device, not a dietitian;
  the evals assert that the same input always produces the same number, which is the property a user
  can actually rely on.
