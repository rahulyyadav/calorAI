import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from calorai_agent.domain import (
    InterpretationOrigin,
    MealDraft,
    MealItemDraft,
    MealType,
    Nutrition,
)
from calorai_agent.repository import MealRepository


def _item(name: str, quantity: Decimal, calories: str) -> MealItemDraft:
    return MealItemDraft(
        name=name,
        quantity=quantity,
        unit="piece",
        nutrition=Nutrition(
            calories=Decimal(calories),
            protein_g=Decimal("4"),
            carbs_g=Decimal("24"),
            fat_g=Decimal("1"),
        ),
        confidence=0.95,
    )


def _draft(items: tuple[MealItemDraft, ...], text: str = "2 rotis") -> MealDraft:
    return MealDraft(
        meal_type=MealType.LUNCH,
        occurred_at=datetime(2026, 9, 26, 13, 30, tzinfo=UTC),
        source_text=text,
        items=items,
    )


def _rotis(count: int) -> tuple[MealItemDraft, ...]:
    return (_item("roti", Decimal(count), str(120 * count)),)


def test_revision_supersedes_the_previous_one(
    repository: MealRepository,
) -> None:
    created = repository.create("user-1", _draft(_rotis(2)))

    revised = repository.revise(
        created.id,
        _rotis(3),
        "actually 3 rotis",
        origin=InterpretationOrigin.RULE_BASED,
    )

    assert revised is not None
    assert revised.revision_number == 2
    assert revised.items[0].quantity == Decimal("3")
    assert repository.get(created.id) is not None
    assert repository.totals_for_day("user-1", date(2026, 9, 26)).nutrition.calories == Decimal(
        "360.00"
    )


def test_history_is_auditable(repository: MealRepository) -> None:
    created = repository.create("user-1", _draft(_rotis(2)))
    repository.revise(created.id, _rotis(3), "actually 3 rotis")
    repository.revise(created.id, _rotis(4), "make that 4", replace_items=True)

    with repository.database.connect() as connection:
        rows = connection.execute(
            "SELECT revision_number, source_text, superseded_at FROM meal_revisions "
            "WHERE meal_id = ? ORDER BY revision_number",
            (created.id,),
        ).fetchall()

    assert [row["revision_number"] for row in rows] == [1, 2, 3]
    assert [row["superseded_at"] is None for row in rows] == [False, False, True]
    assert [row["source_text"] for row in rows] == ["2 rotis", "actually 3 rotis", "make that 4"]


def test_partial_revision_keeps_other_items(repository: MealRepository) -> None:
    created = repository.create(
        "user-1",
        _draft((_item("roti", Decimal("2"), "240"), _item("milk chai", Decimal("1"), "120"))),
    )

    revised = repository.revise(created.id, _rotis(3), "actually 3 rotis")

    assert revised is not None
    assert [item.name for item in revised.items] == ["roti", "milk chai"]
    assert revised.nutrition.calories == Decimal("480.00")


def test_full_revision_replaces_every_item(repository: MealRepository) -> None:
    created = repository.create(
        "user-1",
        _draft((_item("roti", Decimal("2"), "240"), _item("milk chai", Decimal("1"), "120"))),
    )

    revised = repository.revise(created.id, _rotis(1), "just one roti", replace_items=True)

    assert revised is not None
    assert [item.name for item in revised.items] == ["roti"]
    assert revised.nutrition.calories == Decimal("120.00")


def test_deleting_a_meal_keeps_the_row_but_drops_it_from_totals(
    repository: MealRepository,
) -> None:
    created = repository.create("user-1", _draft(_rotis(2)))

    assert repository.soft_delete(created.id) is True
    assert repository.get(created.id) is None
    assert repository.list_for_day("user-1", date(2026, 9, 26)) == []
    assert repository.totals_for_day("user-1", date(2026, 9, 26)).meal_count == 0

    with repository.database.connect() as connection:
        row = connection.execute(
            "SELECT deleted_at FROM meals WHERE id = ?", (created.id,)
        ).fetchone()
    assert row["deleted_at"] is not None


def test_deleting_twice_is_not_double_counted(repository: MealRepository) -> None:
    created = repository.create("user-1", _draft(_rotis(2)))
    repository.soft_delete(created.id)

    assert repository.soft_delete(created.id) is False


def test_revising_a_deleted_meal_is_refused(repository: MealRepository) -> None:
    created = repository.create("user-1", _draft(_rotis(2)))
    repository.soft_delete(created.id)

    assert repository.revise(created.id, _rotis(3), "actually 3") is None


def test_one_meal_per_inbound_event_is_enforced_by_the_schema(
    repository: MealRepository,
) -> None:
    draft = _draft(_rotis(2))
    event = repository.record_inbound("user-1", "wa:msg-1", "cli", "2 rotis")
    repository.create("user-1", draft, source_event_id=event.id)

    duplicate = repository.record_inbound("user-1", "wa:msg-1", "cli", "2 rotis")
    assert duplicate.created is False
    assert duplicate.id == event.id

    with pytest.raises(sqlite3.IntegrityError):
        repository.create("user-1", draft, source_event_id=event.id)
