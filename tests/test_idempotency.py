from datetime import UTC, date, datetime
from decimal import Decimal

from calorai_agent.domain import InboundMessage, MealDraft, MealItemDraft, MealType, Nutrition
from calorai_agent.graph import MealAgent
from calorai_agent.repository import MealRepository

LOGGED = "Logged 2 roti — about 240 kcal and 8g protein."


def _inbound(text: str, external_id: str) -> InboundMessage:
    return InboundMessage(
        user_id="user-1", text=text, external_id=external_id, channel="cli", timezone="UTC"
    )


def _draft() -> MealDraft:
    return MealDraft(
        meal_type=MealType.LUNCH,
        occurred_at=datetime(2026, 9, 26, 13, 30, tzinfo=UTC),
        source_text="2 rotis for lunch",
        items=(
            MealItemDraft(
                name="roti",
                quantity=Decimal("2"),
                unit="piece",
                nutrition=Nutrition(
                    calories=Decimal("240"),
                    protein_g=Decimal("8"),
                    carbs_g=Decimal("48"),
                    fat_g=Decimal("2"),
                ),
            ),
        ),
    )


def test_redelivered_message_logs_exactly_one_meal(agent: MealAgent) -> None:
    first = agent.handle(_inbound("2 rotis for lunch", "wa:abc"))
    replay = agent.handle(_inbound("2 rotis for lunch", "wa:abc"))

    assert first == LOGGED
    assert replay == LOGGED


def test_repeated_message_ids_are_distinct_messages(agent: MealAgent) -> None:
    agent.handle(_inbound("2 rotis for lunch", "wa:one"))
    agent.handle(_inbound("2 rotis for lunch", "wa:two"))

    assert agent.handle(_inbound("how many calories today?", "wa:totals")) == (
        "Today: 480 kcal, 16g protein, 96g carbs, and 4g fat across 2 meals."
    )


def test_completed_event_replay_never_reaches_the_tools(
    agent: MealAgent, repository: MealRepository
) -> None:
    event = repository.record_inbound("user-1", "wa:done", "cli", "2 rotis for lunch")
    repository.complete_inbound(event.id, "cached answer")

    assert agent.handle(_inbound("2 rotis for lunch", "wa:done")) == "cached answer"
    assert repository.totals_for_day("user-1", date(2026, 9, 26)).meal_count == 0


def test_replay_after_a_crash_between_mutation_and_reply(
    agent: MealAgent, repository: MealRepository
) -> None:
    event = repository.record_inbound("user-1", "wa:half", "cli", "2 rotis for lunch")
    repository.create("user-1", _draft(), source_event_id=event.id)

    assert agent.handle(_inbound("2 rotis for lunch", "wa:half")) == LOGGED
    assert repository.totals_for_day("user-1", date(2026, 9, 26)).meal_count == 1


def test_unfinished_event_without_a_mismatched_mutation_says_so(
    agent: MealAgent, repository: MealRepository
) -> None:
    repository.record_inbound("user-1", "wa:stuck", "cli", "2 rotis for lunch")

    assert agent.handle(_inbound("2 rotis for lunch", "wa:stuck")) == (
        "I already received that message and am still working on it."
    )


def test_replay_after_a_crash_between_revision_and_reply(
    agent: MealAgent, repository: MealRepository
) -> None:
    logged = repository.record_inbound("user-1", "wa:lunch", "cli", "2 rotis for lunch")
    meal = repository.create("user-1", _draft(), source_event_id=logged.id)
    repository.complete_inbound(logged.id, LOGGED)
    event = repository.record_inbound("user-1", "wa:fix", "cli", "actually it was 3 rotis")
    repository.revise(
        meal_id=meal.id,
        replacement_items=_rotis(3),
        source_text="actually it was 3 rotis",
        source_event_id=event.id,
    )
    before = _revision_rows(repository, meal.id)

    assert agent.handle(_inbound("actually it was 3 rotis", "wa:fix")) == (
        "Updated your lunch at 13:30 to 3 roti — now about 360 kcal and 12g protein."
    )
    assert repository.totals_for_day("user-1", date(2026, 9, 26)).meal_count == 1
    assert _revision_rows(repository, meal.id) == before


def test_replay_after_a_crash_between_deletion_and_reply(
    agent: MealAgent, repository: MealRepository
) -> None:
    logged = repository.record_inbound("user-1", "wa:lunch", "cli", "2 rotis for lunch")
    meal = repository.create("user-1", _draft(), source_event_id=logged.id)
    repository.complete_inbound(logged.id, LOGGED)
    event = repository.record_inbound("user-1", "wa:drop", "cli", "delete that")
    repository.soft_delete(meal.id, event.id)

    assert agent.handle(_inbound("delete that", "wa:drop")) == "Removed 2 roti (240 kcal)."
    assert repository.totals_for_day("user-1", date(2026, 9, 26)).meal_count == 0


def _revision_rows(repository: MealRepository, meal_id: str) -> list[int]:
    with repository.database.connect() as connection:
        return [
            row["revision_number"]
            for row in connection.execute(
                "SELECT revision_number FROM meal_revisions WHERE meal_id = ?", (meal_id,)
            ).fetchall()
        ]


def _rotis(count: int) -> tuple[MealItemDraft, ...]:
    return (
        MealItemDraft(
            name="roti",
            quantity=Decimal(str(count)),
            unit="piece",
            nutrition=Nutrition(
                calories=Decimal(str(120 * count)),
                protein_g=Decimal(str(4 * count)),
                carbs_g=Decimal(str(24 * count)),
                fat_g=Decimal(str(1 * count)),
            ),
        ),
    )
