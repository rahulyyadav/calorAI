from datetime import UTC, date, datetime
from decimal import Decimal

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
