"""Phase 3: memory that survives a restart, changes behavior, and never contradicts itself."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest

from calorai_agent.domain import (
    AgentIntent,
    DietaryConstraint,
    InboundMessage,
    MealItemDraft,
    MealType,
    MemoryKind,
    MemoryRecord,
    NamedRoutine,
    NutrientMetric,
    Nutrition,
    NutritionTarget,
    ParsedMessage,
)
from calorai_agent.graph import MealAgent
from calorai_agent.llm_planner import ModelPlanner
from calorai_agent.memory import (
    MEMORY_CONTEXT_LIMIT,
    MEMORY_KIND_LIMITS,
    avoided_foods,
    conflicting_foods,
    current_diet,
    diets,
)
from calorai_agent.planning import PlannerRequest, RuleBasedPlanner
from calorai_agent.repository import MealRepository
from calorai_agent.tools import ListMemoriesInput

NOW = datetime(2026, 9, 26, 8, tzinfo=UTC)

_IDLI = MealItemDraft(
    name="idli",
    quantity=Decimal("2"),
    unit="piece",
    nutrition=Nutrition(
        calories=Decimal("78"), protein_g=Decimal("4"), carbs_g=Decimal("16"), fat_g=Decimal("0.6")
    ),
)
_CHICKEN = MealItemDraft(
    name="chicken",
    quantity=Decimal("1"),
    unit="serving",
    nutrition=Nutrition(
        calories=Decimal("250"),
        protein_g=Decimal("31"),
        carbs_g=Decimal("0"),
        fat_g=Decimal("13"),
    ),
)
_PANEER = MealItemDraft(
    name="paneer",
    quantity=Decimal("1"),
    unit="serving",
    nutrition=Nutrition(
        calories=Decimal("260"),
        protein_g=Decimal("14"),
        carbs_g=Decimal("6"),
        fat_g=Decimal("20"),
    ),
)


def _parse(text: str, **context: Any) -> Any:
    return RuleBasedPlanner().parse(
        PlannerRequest(text=text, occurred_at=NOW, timezone="UTC", **context)
    )


def _send(agent: MealAgent, clock, text: str, hour: int, minute: int = 0, day: int = 26) -> str:
    return agent.invoke("user-1", text, now=clock(hour, minute, day))


def _memory(content: Any) -> MemoryRecord:
    return MemoryRecord(
        id="m-1",
        user_id="user-1",
        content=content,
        source_event_id=None,
        created_at=NOW,
    )


def _rows(repository: MealRepository, memory_type: str) -> list[str]:
    with repository.database.connect() as connection:
        found = connection.execute(
            "SELECT value_json FROM memories WHERE memory_type = ? ORDER BY created_at",
            (memory_type,),
        ).fetchall()
    return [row["value_json"] for row in found]


class RecordingClient:
    """Minimal text-model client that answers with one fixed payload."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete(self, *, system: str, user: str) -> str:
        self.system_prompts.append(system)
        self.user_prompts.append(user)
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


def _model_planner(payload: Any) -> tuple[ModelPlanner, RecordingClient]:
    client = RecordingClient(payload)
    return ModelPlanner(client, model="test-model"), client


# --- extraction -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "diet"),
    [
        ("I'm vegetarian btw", "vegetarian"),
        ("im vegan", "vegan"),
        ("I am a vegetarian", "vegetarian"),
        ("I'm pure veg", "vegetarian"),
        ("my diet is vegetarian", "vegetarian"),
        ("I'm not vegetarian anymore", "non-vegetarian"),
        ("I'm non-veg", "non-vegetarian"),
    ],
)
def test_a_diet_statement_becomes_a_typed_memory(spoken: str, diet: str) -> None:
    parsed = _parse(spoken)

    assert parsed.intent is AgentIntent.SAVE_MEMORY
    assert isinstance(parsed.memory, DietaryConstraint)
    assert parsed.memory.diet == diet


def test_an_ordinary_meal_is_not_mistaken_for_a_preference() -> None:
    parsed = _parse("had 2 rotis and chai for breakfast")

    assert parsed.intent is AgentIntent.LOG_MEAL


@pytest.mark.parametrize(
    ("spoken", "metric", "value"),
    [
        ("aim for 120g protein a day", NutrientMetric.PROTEIN_G, Decimal("120")),
        ("my target is 1800 calories", NutrientMetric.CALORIES, Decimal("1800")),
        ("protein target of 150 g", NutrientMetric.PROTEIN_G, Decimal("150")),
        ("I'm trying to stay under 200g of carbs", NutrientMetric.CARBS_G, Decimal("200")),
        ("my fat goal is 60", NutrientMetric.FAT_G, Decimal("60")),
    ],
)
def test_a_stated_target_becomes_a_typed_memory(
    spoken: str, metric: NutrientMetric, value: Decimal
) -> None:
    parsed = _parse(spoken)

    assert parsed.intent is AgentIntent.SAVE_MEMORY
    assert parsed.memory == NutritionTarget(metric=metric, value=value)


def test_a_portion_in_a_meal_is_not_a_target() -> None:
    parsed = _parse("had 30g of protein from dal")

    assert parsed.intent is not AgentIntent.SAVE_MEMORY


def test_an_absurd_target_never_becomes_a_memory() -> None:
    parsed = _parse("my target is 900000 g protein")

    assert parsed.intent is not AgentIntent.SAVE_MEMORY


def test_a_routine_save_points_at_a_meal_without_inventing_one() -> None:
    parsed = _parse("remember this as my usual breakfast")

    assert parsed.intent is AgentIntent.SAVE_MEMORY
    assert parsed.memory == NamedRoutine(slot="breakfast")
    assert parsed.reference is not None
    assert parsed.reference.meal_type is MealType.BREAKFAST


def test_a_routine_needs_a_pointer_before_it_can_be_saved() -> None:
    with pytest.raises(ValueError, match="meal reference"):
        ParsedMessage(intent=AgentIntent.SAVE_MEMORY, memory=NamedRoutine(slot="breakfast"))


# --- retrieval --------------------------------------------------------------------


def test_a_changed_diet_supersedes_instead_of_contradicting(repository: MealRepository) -> None:
    repository.remember("user-1", DietaryConstraint(diet="vegetarian"))
    repository.remember("user-1", DietaryConstraint(diet="non-vegetarian"))

    active = repository.active_memories("user-1")

    assert current_diet(active) == "non-vegetarian"
    assert len(diets(active)) == 1
    assert len(_rows(repository, "dietary_constraint")) == 2


def test_independent_facts_stay_active_together(repository: MealRepository) -> None:
    repository.remember("user-1", DietaryConstraint(diet="vegetarian"))
    repository.remember("user-1", NutritionTarget(metric=NutrientMetric.PROTEIN_G, value=120))
    repository.remember("user-1", NutritionTarget(metric=NutrientMetric.CALORIES, value=1800))

    assert len(repository.active_memories("user-1")) == 3


def test_each_kind_is_bounded_on_its_own(repository: MealRepository) -> None:
    routine_cap = MEMORY_KIND_LIMITS[MemoryKind.NAMED_ROUTINE]
    saved_slots = routine_cap + 4
    for index in range(saved_slots):
        repository.remember("user-1", NamedRoutine(slot=f"slot-{index}", items=(_IDLI,)))
    repository.remember("user-1", DietaryConstraint(diet="vegetarian"))

    retrieved = repository.active_memories("user-1")
    loaded_routines = [
        record.content for record in retrieved if isinstance(record.content, NamedRoutine)
    ]

    assert [routine.slot for routine in loaded_routines] == [
        f"slot-{index}" for index in range(saved_slots - 1, saved_slots - 1 - routine_cap, -1)
    ]
    assert diets(retrieved) == (DietaryConstraint(diet="vegetarian"),)
    assert repository.active_memories("user-1", kinds=(MemoryKind.NUTRITION_TARGET,)) == []


def test_a_busy_routine_habit_costs_the_user_neither_diet_nor_targets(
    repository: MealRepository, make_agent, clock
) -> None:
    """The bound is per kind, so one chatty kind cannot evict the facts every reply needs."""
    first = make_agent()
    _send(first, clock, "i'm vegetarian btw", 6)
    _send(first, clock, "aim for 120g protein a day", 6)
    _send(first, clock, "2 idlis for breakfast", 7, 15)
    for index in range(MEMORY_KIND_LIMITS[MemoryKind.NAMED_ROUTINE] + 5):
        repository.remember("user-1", NamedRoutine(slot=f"slot-{index}", items=(_IDLI,)))

    reopened = make_agent()
    logged = _send(reopened, clock, "1 chicken for lunch", 12)
    totals = _send(reopened, clock, "how am I doing today?", 13)

    usual = _send(reopened, clock, "my usual", 14)

    assert "Heads up — chicken is not vegetarian." in logged
    assert "Against your targets: 35 of 120 g protein." in totals
    assert "Which one is your usual today" in usual and "slot-10" in usual


def test_a_routine_without_foods_is_refused(repository: MealRepository) -> None:
    with pytest.raises(ValueError, match="at least one food"):
        repository.remember("user-1", NamedRoutine(slot="breakfast"))


def test_memories_are_attributed_to_the_message_that_stated_them(
    repository: MealRepository,
) -> None:
    event = repository.record_inbound("user-1", "wa-1", "whatsapp", "i'm vegetarian")

    record = repository.remember("user-1", DietaryConstraint(diet="vegetarian"), event.id)

    assert record.source_event_id == event.id
    assert repository.memory_for_event(event.id) is not None
    assert repository.memory_for_event("no-such-event") is None


# --- what memory is allowed to change ---------------------------------------------


def test_a_diet_names_the_foods_it_rules_out() -> None:
    memories = [_memory(DietaryConstraint(diet="vegetarian"))]

    assert "chicken" in avoided_foods(diets(memories))
    assert conflicting_foods(memories, [_CHICKEN, _IDLI]) == ("chicken",)


def test_a_non_vegetarian_diet_rules_out_nothing() -> None:
    memories = [_memory(DietaryConstraint(diet="non-vegetarian"))]

    assert conflicting_foods(memories, [_CHICKEN]) == ()


def test_a_vegan_diet_rules_out_dairy_that_a_vegetarian_keeps() -> None:
    vegan = [_memory(DietaryConstraint(diet="vegan"))]
    vegetarian = [_memory(DietaryConstraint(diet="vegetarian"))]

    assert conflicting_foods(vegan, [_PANEER, _IDLI]) == ("paneer",)
    assert conflicting_foods(vegetarian, [_PANEER]) == ()


def test_a_saved_vegan_diet_warns_about_a_dairy_meal_after_a_restart(make_agent, clock) -> None:
    _send(make_agent(), clock, "im vegan", 7)

    response = _send(make_agent(), clock, "1 paneer for lunch", 13)

    assert "Heads up — paneer is not vegan." in response


def test_a_saved_preference_survives_a_restart_and_shapes_the_next_reply(make_agent, clock) -> None:
    _send(make_agent(), clock, "i'm vegetarian btw", 7)

    response = _send(make_agent(), clock, "1 chicken and 2 idlis", 9)

    assert response.startswith("Logged 1 chicken, 2 idli — about 328 kcal")
    assert "Heads up — chicken is not vegetarian." in response


def test_a_target_survives_a_restart_and_shows_up_in_totals(make_agent, clock) -> None:
    _send(make_agent(), clock, "aim for 120g protein a day", 7)

    reopened = make_agent()
    _send(reopened, clock, "2 rotis", 10)

    assert _send(reopened, clock, "how am I doing today?", 11).endswith(
        "Against your targets: 8 of 120 g protein."
    )


def test_my_usual_is_logged_after_a_fresh_process_start(make_agent, clock) -> None:
    agent = make_agent()
    _send(agent, clock, "2 idlis and coffee for breakfast", 8, 15)
    saved = _send(agent, clock, "remember this as my usual breakfast", 9)

    reopened = make_agent()
    logged = _send(reopened, clock, "my usual breakfast", 12)

    assert saved == "Got it — I will remember your usual breakfast is 2 idli, 1 coffee."
    assert logged == "Logged 2 idli, 1 coffee — about 83 kcal and 4g protein."


def test_an_unsaved_usual_asks_once_and_offers_the_recipe(agent: MealAgent, clock) -> None:
    response = _send(agent, clock, "my usual", 8)

    assert "What is your usual?" in response
    assert "remember this as my usual" in response


def test_a_routine_for_another_meal_says_which_one_is_missing(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 idlis for breakfast", 8, 15)
    _send(agent, clock, "remember this as my usual breakfast", 9)

    response = _send(agent, clock, "my usual lunch", 13)

    assert "I don't have your usual lunch saved yet." in response


def test_a_routine_copy_is_a_new_meal_not_a_correction(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    _send(agent, clock, "2 idlis for breakfast", 8, 15)
    _send(agent, clock, "remember this as my usual breakfast", 9)

    _send(agent, clock, "my usual breakfast", 20)

    logged = repository.list_for_day("user-1", date(2026, 9, 26))
    assert len(logged) == 2
    assert [meal.revision_number for meal in logged] == [1, 1]
    assert {meal.nutrition.calories for meal in logged} == {Decimal("78.00")}


def test_a_routine_without_a_meal_to_remember_says_so(agent: MealAgent, clock) -> None:
    response = _send(agent, clock, "remember this as my usual breakfast", 8)

    assert response == "I couldn't find a recent meal that matches."
    assert not agent.tools.list_memories(ListMemoriesInput(user_id="user-1"))


def test_an_ambiguous_routine_asks_which_meal_to_remember(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 idlis for breakfast", 8, 15)
    _send(agent, clock, "1 chicken for lunch", 13, 15)

    response = _send(agent, clock, "remember this as my usual", 14)

    assert "Which today meal do you mean to remember" in response
    assert "breakfast at 08:00" in response and "lunch at 13:30" in response


def test_two_usuals_leave_the_choice_to_the_user(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 idlis for breakfast", 8, 15)
    _send(agent, clock, "remember this as my usual breakfast", 9)
    _send(agent, clock, "1 chicken for lunch", 13, 15)
    _send(agent, clock, "remember this as my usual lunch", 14)

    response = _send(agent, clock, "my usual", 18)

    assert "Which one is your usual today — lunch, breakfast?" in response


def test_a_redelivered_memory_message_is_answered_from_the_database(
    agent: MealAgent, repository: MealRepository
) -> None:
    message = InboundMessage(user_id="user-1", text="i'm vegetarian", external_id="wa-9")

    first = agent.handle(message)
    second = agent.handle(message)

    assert first == second == "Got it — I will remember you are vegetarian."
    assert len(repository.active_memories("user-1")) == 1


def test_a_memory_written_before_a_crash_still_answers_its_message(
    repository: MealRepository, make_agent
) -> None:
    event = repository.record_inbound("user-1", "wa-10", "cli", "i'm vegetarian")
    repository.remember("user-1", DietaryConstraint(diet="vegetarian"), event.id)

    response = make_agent().handle(
        InboundMessage(user_id="user-1", text="i'm vegetarian", external_id="wa-10")
    )

    assert response == "Got it — I will remember you are vegetarian."


# --- model-backed extraction ------------------------------------------------------


def test_the_model_prompt_treats_a_stated_fact_as_memory_not_a_meal() -> None:
    planner, client = _model_planner({"intent": "acknowledge", "statement": "noted"})

    planner.parse(PlannerRequest(text="ok", occurred_at=NOW))

    assert "durable fact" in client.system_prompts[0]
    assert '"save_memory", not a meal to log' in client.system_prompts[0]


def test_the_model_can_ask_for_a_memory_to_be_saved() -> None:
    planner, _ = _model_planner(
        {"intent": "save_memory", "memory": {"diet": "Vegan", "confidence": 0.9}}
    )

    parsed = planner.parse(PlannerRequest(text="i'm vegan from now on", occurred_at=NOW))

    assert parsed.intent is AgentIntent.SAVE_MEMORY
    assert parsed.memory == DietaryConstraint(diet="vegan", confidence=0.9)


def test_a_model_routine_without_a_pointer_falls_back_to_the_rules() -> None:
    planner, _ = _model_planner({"intent": "save_memory", "memory": {"slot": "breakfast"}})

    parsed = planner.parse(PlannerRequest(text="remember this as my usual", occurred_at=NOW))

    assert parsed.intent is AgentIntent.SAVE_MEMORY
    assert parsed.memory == NamedRoutine(slot="default")
    assert parsed.reference is not None


def test_a_model_target_becomes_a_typed_memory() -> None:
    planner, _ = _model_planner(
        {
            "intent": "save_memory",
            "memory": {"metric": "protein_g", "value": 140},
        }
    )

    parsed = planner.parse(PlannerRequest(text="i want 140g of protein", occurred_at=NOW))

    assert parsed.memory == NutritionTarget(metric=NutrientMetric.PROTEIN_G, value=140)


def test_memories_reach_the_model_prompt_in_a_bounded_form() -> None:
    planner, client = _model_planner({"intent": "get_totals"})
    memories = [_memory(DietaryConstraint(diet="vegetarian"))] * MEMORY_CONTEXT_LIMIT

    planner.parse(PlannerRequest(text="how am I doing", occurred_at=NOW, memories=tuple(memories)))

    prompt = client.user_prompts[-1]
    assert "you are vegetarian" in prompt
    assert prompt.count("- you are vegetarian") <= MEMORY_CONTEXT_LIMIT
