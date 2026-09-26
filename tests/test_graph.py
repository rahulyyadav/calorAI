from datetime import UTC, datetime

from calorai_agent.graph import MealAgent
from calorai_agent.nutrition import RuleBasedPlanner
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools


def _agent(repository: MealRepository) -> MealAgent:
    return MealAgent(RuleBasedPlanner(), MealTools(repository))


def test_logs_then_returns_current_totals(repository: MealRepository) -> None:
    agent = _agent(repository)
    now = datetime(2026, 9, 26, 8, tzinfo=UTC)

    logged = agent.invoke(
        "user-1",
        "had 2 parathas and chai for breakfast",
        now=now,
    )
    totals = agent.invoke("user-1", "how am I doing today?", now=now)

    assert logged == "Logged 2 paratha, 1 milk chai — about 640 kcal and 15g protein."
    assert totals == ("Today: 640 kcal, 15g protein, 94g carbs, and 22g fat across 1 meal.")


def test_lists_today_meals(repository: MealRepository) -> None:
    agent = _agent(repository)
    now = datetime(2026, 9, 26, 8, tzinfo=UTC)
    agent.invoke("user-1", "2 rotis", now=now)

    response = agent.invoke("user-1", "what did I eat today?", now=now)

    assert response == "Today you logged: 2 roti."


def test_empty_totals_are_friendly(repository: MealRepository) -> None:
    response = _agent(repository).invoke(
        "user-1",
        "calories today?",
        now=datetime(2026, 9, 26, 8, tzinfo=UTC),
    )

    assert response == "Nothing logged today yet."
