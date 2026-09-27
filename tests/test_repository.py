from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from calorai_agent.db import Database
from calorai_agent.domain import MealDraft, MealItemDraft, MealType, Nutrition
from calorai_agent.repository import MealRepository


def _draft(at: datetime) -> MealDraft:
    return MealDraft(
        meal_type=MealType.BREAKFAST,
        occurred_at=at,
        source_text="2 rotis",
        items=(
            MealItemDraft(
                name="roti",
                quantity=Decimal("2"),
                unit="piece",
                nutrition=Nutrition(calories=240, protein_g=8, carbs_g=48, fat_g=2),
                confidence=0.95,
            ),
        ),
    )


def test_meal_persists_and_totals_are_deterministic(repository: MealRepository) -> None:
    at = datetime(2026, 9, 26, 7, tzinfo=UTC)
    created = repository.create("user-1", _draft(at))

    meals = repository.list_for_day("user-1", date(2026, 9, 26))
    totals = repository.totals_for_day("user-1", date(2026, 9, 26))

    assert [meal.id for meal in meals] == [created.id]
    assert meals[0].nutrition.calories == Decimal("240.00")
    assert totals.meal_count == 1
    assert totals.nutrition == Nutrition(calories=240, protein_g=8, carbs_g=48, fat_g=2)


def test_totals_respect_user_local_day(repository: MealRepository) -> None:
    repository.create(
        "user-1",
        _draft(datetime(2026, 9, 25, 20, 30, tzinfo=UTC)),
    )

    kathmandu_day = repository.totals_for_day("user-1", date(2026, 9, 26), "Asia/Kathmandu")
    utc_day = repository.totals_for_day("user-1", date(2026, 9, 26), "UTC")

    assert kathmandu_day.meal_count == 1
    assert utc_day.meal_count == 0


def test_database_can_be_reopened_without_losing_meals(repository: MealRepository) -> None:
    at = datetime(2026, 9, 26, 7, tzinfo=UTC)
    repository.create("user-1", _draft(at))

    reopened = MealRepository(repository.database)

    assert len(reopened.list_for_day("user-1", date(2026, 9, 26))) == 1


def test_initialize_is_repeatable_and_keeps_data(tmp_path: Path) -> None:
    database = Database(tmp_path / "fresh.sqlite3")
    database.initialize()
    repository = MealRepository(database)
    repository.ensure_user("user-1", "UTC")
    repository.create("user-1", _draft(datetime(2026, 9, 26, 7, tzinfo=UTC)))

    database.initialize()

    assert MealRepository(database).totals_for_day("user-1", date(2026, 9, 26)).meal_count == 1


def test_migrations_apply_in_unique_numeric_order(tmp_path: Path) -> None:
    versions = [version for version, _ in Database(tmp_path / "t.sqlite3").migrations]

    assert versions == sorted(versions)
    assert len(versions) == len(set(versions))
    assert versions[0] == 1


def test_a_spring_forward_day_is_23_hours_not_24(repository: MealRepository) -> None:
    """On 2026-03-08 America/New_York skips an hour, so a naive +24h window leaks a meal."""
    eighth = datetime(2026, 3, 8, 7, 30, tzinfo=UTC)  # 03:30 EDT on the 8th
    repository.create("user-1", _draft(eighth))
    repository.create("user-1", _draft(datetime(2026, 3, 9, 4, 0, tzinfo=UTC)))  # 00:00 on the 9th

    on_the_eighth = repository.list_for_day("user-1", date(2026, 3, 8), "America/New_York")

    assert [meal.occurred_at for meal in on_the_eighth] == [eighth]
