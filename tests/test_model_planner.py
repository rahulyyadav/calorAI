import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from calorai_agent.domain import AgentIntent, InterpretationOrigin, MealType
from calorai_agent.llm_planner import ModelPlanner
from calorai_agent.planning import PlannerRequest
from calorai_agent.policy import UNUSABLE_QUANTITY_CONFIDENCE
from calorai_agent.providers import ModelProviderError

NOW = datetime(2026, 9, 26, 8, tzinfo=UTC)


class RecordingClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete(self, *, system: str, user: str) -> str:
        self.system_prompts.append(system)
        self.user_prompts.append(user)
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


def _request(text: str = "had two eggs") -> PlannerRequest:
    return PlannerRequest(text=text, occurred_at=NOW, timezone="UTC")


def _planner(payload: Any) -> tuple[ModelPlanner, RecordingClient]:
    client = RecordingClient(payload)
    return ModelPlanner(client, model="test-model"), client


def test_model_items_get_reference_nutrition_not_model_math() -> None:
    planner, _ = _planner(
        {
            "intent": "log_meal",
            "items": [{"name": "paratha", "quantity": 2, "confidence": 0.95}],
            "meal_type": "breakfast",
        }
    )

    parsed = planner.parse(_request("had 2 parathas for breakfast"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert parsed.draft.items[0].nutrition.calories == Decimal("520.00")
    assert parsed.draft.origin is InterpretationOrigin.TEXT_MODEL
    assert parsed.draft.model == "test-model"
    assert parsed.draft.meal_type is MealType.BREAKFAST
    assert parsed.draft.occurred_at.hour == 8


def test_prompt_offers_the_reference_food_list_and_recent_meals() -> None:
    planner, client = _planner({"intent": "get_totals"})

    planner.parse(
        PlannerRequest(
            text="how am I doing",
            occurred_at=NOW,
            timezone="UTC",
            recent_meals=(),
        )
    )

    assert "paratha" in client.system_prompts[0]
    assert "Current time: 2026-09-26T08:00:00+00:00" in client.user_prompts[0]


def test_unknown_food_names_are_dropped_and_the_rules_planner_answers() -> None:
    planner, _ = _planner(
        {
            "intent": "log_meal",
            "items": [{"name": "dragonfruit", "quantity": 1, "confidence": 0.9}],
        }
    )

    parsed = planner.parse(_request("had a dragonfruit"))

    assert parsed.intent is AgentIntent.UNKNOWN
    assert parsed.explanation is not None


def test_a_food_without_reference_data_is_reported_alongside_the_known_ones() -> None:
    planner, _ = _planner(
        {
            "intent": "log_meal",
            "items": [
                {"name": "roti", "quantity": 2, "confidence": 0.95},
                {"name": "dragonfruit", "quantity": 1, "confidence": 0.9},
            ],
        }
    )

    parsed = planner.parse(_request("had 2 rotis and a dragonfruit"))

    assert parsed.draft is not None
    assert [item.name for item in parsed.draft.items] == ["roti"]
    assert parsed.unrecognized == ("dragonfruit",)


def test_provider_failure_falls_back_to_the_deterministic_planner() -> None:
    planner, _ = _planner(ModelProviderError("boom"))

    parsed = planner.parse(_request("had 2 parathas and chai for breakfast"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert parsed.draft.origin is InterpretationOrigin.RULE_BASED


def test_malformed_model_output_falls_back_instead_of_raising() -> None:
    planner, _ = _planner("{not json at all")

    parsed = planner.parse(_request("how many calories today?"))

    assert parsed.intent is AgentIntent.GET_TOTALS


def test_schema_violation_falls_back() -> None:
    planner, _ = _planner({"intent": "log_meal", "items": [{"name": "roti", "quantity": -2}]})

    parsed = planner.parse(_request("2 rotis"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert parsed.draft.items[0].quantity == Decimal("2")


def test_revision_decision_carries_items_and_a_reference() -> None:
    planner, _ = _planner(
        {
            "intent": "revise_meal",
            "items": [{"name": "roti", "quantity": 3, "confidence": 0.9}],
            "reference": {"day_offset": 0, "meal_type": None, "food_hint": "roti"},
        }
    )

    parsed = planner.parse(_request("actually 3 rotis"))

    assert parsed.intent is AgentIntent.REVISE_MEAL
    assert parsed.reference is not None
    assert parsed.reference.food_hint == "roti"
    assert parsed.items[0].quantity == Decimal("3")


def test_skipped_meal_becomes_an_acknowledgement() -> None:
    planner, _ = _planner({"intent": "acknowledge", "statement": "Noted — no lunch logged."})

    parsed = planner.parse(_request("skipped lunch"))

    assert parsed.intent is AgentIntent.ACKNOWLEDGE
    assert parsed.statement == "Noted — no lunch logged."


def test_grazing_becomes_one_question() -> None:
    planner, _ = _planner({"intent": "clarify", "question": "What did you graze on, and how much?"})

    parsed = planner.parse(_request("grazed all afternoon"))

    assert parsed.intent is AgentIntent.CLARIFY
    assert parsed.question is not None


def test_repeat_decision_defaults_to_today_when_the_model_omits_a_reference() -> None:
    planner, _ = _planner({"intent": "repeat_meal", "day_offset": -1})

    parsed = planner.parse(_request("same as yesterday"))

    assert parsed.intent is AgentIntent.REPEAT_MEAL
    assert parsed.reference is not None
    assert parsed.reference.day_offset == -1


@pytest.mark.parametrize("quantity", [0, -1, "abc"])
def test_invalid_item_quantities_reject_to_the_deterministic_planner(quantity: Any) -> None:
    planner, _ = _planner({"intent": "log_meal", "items": [{"name": "roti", "quantity": quantity}]})

    parsed = planner.parse(_request("2 rotis"))

    assert parsed.draft is not None
    assert parsed.draft.items[0].quantity == Decimal("2")


def test_an_explicit_day_in_the_reference_is_preserved() -> None:
    planner, _ = _planner(
        {
            "intent": "revise_meal",
            "items": [{"name": "roti", "quantity": 3, "confidence": 0.9}],
            "reference": {"day_offset": -1, "day_explicit": True, "food_hint": "roti"},
        }
    )

    parsed = planner.parse(_request("yesterday's rotis were actually 3"))

    assert parsed.reference is not None
    assert parsed.reference.day_offset == -1
    assert parsed.reference.day_explicit is True


def test_a_bare_reference_is_not_marked_day_explicit() -> None:
    planner, _ = _planner(
        {
            "intent": "revise_meal",
            "items": [{"name": "roti", "quantity": 3, "confidence": 0.9}],
            "reference": {"food_hint": "roti"},
        }
    )

    parsed = planner.parse(_request("actually 3 rotis"))

    assert parsed.reference is not None
    assert parsed.reference.day_explicit is False


@pytest.mark.parametrize("quantity", [400, 41])
def test_an_implausible_model_quantity_downgrades_instead_of_logging(
    quantity: int,
) -> None:
    planner, _ = _planner(
        {"intent": "log_meal", "items": [{"name": "roti", "quantity": quantity, "confidence": 0.9}]}
    )

    parsed = planner.parse(_request(f"{quantity} rotis"))

    assert parsed.draft is not None
    assert parsed.draft.items[0].quantity == Decimal(str(quantity))
    assert parsed.draft.items[0].confidence == UNUSABLE_QUANTITY_CONFIDENCE


def test_a_model_revision_can_ask_to_replace_the_whole_item_list() -> None:
    planner, _ = _planner(
        {
            "intent": "revise_meal",
            "items": [{"name": "dosa", "quantity": 2, "confidence": 0.9}],
            "reference": {"day_offset": 0},
            "replace_items": True,
        }
    )

    parsed = planner.parse(_request("it was actually 2 dosas"))

    assert parsed.intent is AgentIntent.REVISE_MEAL
    assert parsed.replace_items is True


def test_a_model_revision_keeps_untouched_items_by_default() -> None:
    planner, _ = _planner(
        {
            "intent": "revise_meal",
            "items": [{"name": "dosa", "quantity": 2, "confidence": 0.9}],
            "reference": {"day_offset": 0},
        }
    )

    parsed = planner.parse(_request("actually 2 dosas"))

    assert parsed.replace_items is False


def test_the_agent_honours_a_model_replace_items_revision(make_agent, clock, repository) -> None:
    from calorai_agent.graph import MealAgent
    from calorai_agent.tools import MealTools

    make_agent().invoke("user-1", "2 rotis and 1 egg", now=clock(8))
    planner, _ = _planner(
        {
            "intent": "revise_meal",
            "items": [{"name": "dosa", "quantity": 2, "confidence": 0.9}],
            "reference": {"day_offset": 0},
            "replace_items": True,
        }
    )

    response = MealAgent(planner, MealTools(repository)).invoke(
        "user-1", "it was actually 2 dosas", now=clock(9)
    )

    assert response == "Updated your meal at 08:00 to 2 dosa — now about 336 kcal and 8g protein."
