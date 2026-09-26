from datetime import UTC, datetime
from decimal import Decimal

from calorai_agent.domain import AgentIntent, MealType
from calorai_agent.nutrition import RuleBasedPlanner


def test_parses_multiple_foods_and_scales_nutrition() -> None:
    parsed = RuleBasedPlanner().parse(
        "had 2 parathas and chai for breakfast",
        datetime(2026, 9, 26, 8, tzinfo=UTC),
    )

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert parsed.draft.meal_type is MealType.BREAKFAST
    assert [item.name for item in parsed.draft.items] == ["paratha", "milk chai"]
    assert parsed.draft.items[0].quantity == Decimal("2")
    assert sum(item.nutrition.calories for item in parsed.draft.items) == Decimal("640.00")


def test_routes_totals_without_attempting_food_extraction() -> None:
    parsed = RuleBasedPlanner().parse(
        "how much protein have I had today?",
        datetime(2026, 9, 26, 8, tzinfo=UTC),
    )

    assert parsed.intent is AgentIntent.GET_TOTALS
    assert parsed.draft is None


def test_unknown_food_requests_actionable_input() -> None:
    parsed = RuleBasedPlanner().parse(
        "I ate something nice",
        datetime(2026, 9, 26, 8, tzinfo=UTC),
    )

    assert parsed.intent is AgentIntent.UNKNOWN
    assert "2 parathas" in (parsed.explanation or "")
