from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from calorai_agent.db import Database
from calorai_agent.domain import (
    DailyTotals,
    MealDraft,
    MealItemDraft,
    MealRecord,
    MealType,
    Nutrition,
)


def _now() -> datetime:
    return datetime.now(UTC)


class MealRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def ensure_user(self, user_id: str, timezone_name: str = "UTC") -> None:
        ZoneInfo(timezone_name)
        with self.database.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO users(id, timezone, created_at) VALUES (?, ?, ?)",
                (user_id, timezone_name, _now().isoformat()),
            )

    def create(
        self,
        user_id: str,
        draft: MealDraft,
        source_event_id: str | None = None,
    ) -> MealRecord:
        meal_id = str(uuid4())
        revision_id = str(uuid4())
        created_at = _now()
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO meals(id, user_id, source_event_id, meal_type, occurred_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    meal_id,
                    user_id,
                    source_event_id,
                    draft.meal_type.value,
                    draft.occurred_at.astimezone(UTC).isoformat(),
                    created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO meal_revisions(
                    id, meal_id, revision_number, source_text, notes, created_at
                )
                VALUES (?, ?, 1, ?, ?, ?)
                """,
                (revision_id, meal_id, draft.source_text, draft.notes, created_at.isoformat()),
            )
            self._insert_items(connection, revision_id, draft.items)
        return MealRecord(
            id=meal_id,
            user_id=user_id,
            meal_type=draft.meal_type,
            occurred_at=draft.occurred_at,
            source_text=draft.source_text,
            created_at=created_at,
            items=draft.items,
        )

    def list_for_day(self, user_id: str, day: date, timezone_name: str = "UTC") -> list[MealRecord]:
        start, end = self._utc_day_bounds(day, timezone_name)
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.user_id, m.meal_type, m.occurred_at, m.created_at,
                       r.id AS revision_id, r.source_text
                FROM meals m
                JOIN meal_revisions r ON r.meal_id = m.id AND r.superseded_at IS NULL
                WHERE m.user_id = ? AND m.deleted_at IS NULL
                  AND m.occurred_at >= ? AND m.occurred_at < ?
                ORDER BY m.occurred_at, m.created_at
                """,
                (user_id, start.isoformat(), end.isoformat()),
            ).fetchall()
            return [self._record_from_row(connection, row) for row in rows]

    def totals_for_day(self, user_id: str, day: date, timezone_name: str = "UTC") -> DailyTotals:
        meals = self.list_for_day(user_id, day, timezone_name)
        total = Nutrition.zero()
        for meal in meals:
            total += meal.nutrition
        return DailyTotals(user_id=user_id, day=day, meal_count=len(meals), nutrition=total)

    @staticmethod
    def _utc_day_bounds(day: date, timezone_name: str) -> tuple[datetime, datetime]:
        zone = ZoneInfo(timezone_name)
        local_start = datetime.combine(day, time.min, tzinfo=zone)
        return (
            local_start.astimezone(UTC),
            (local_start + timedelta(days=1)).astimezone(UTC),
        )

    @staticmethod
    def _insert_items(
        connection: sqlite3.Connection, revision_id: str, items: tuple[MealItemDraft, ...]
    ) -> None:
        connection.executemany(
            """
            INSERT INTO meal_items(
                id, revision_id, name, quantity, unit, calories, protein_g,
                carbs_g, fat_g, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(uuid4()),
                    revision_id,
                    item.name,
                    str(item.quantity),
                    item.unit,
                    str(item.nutrition.calories),
                    str(item.nutrition.protein_g),
                    str(item.nutrition.carbs_g),
                    str(item.nutrition.fat_g),
                    item.confidence,
                )
                for item in items
            ],
        )

    @staticmethod
    def _record_from_row(connection: sqlite3.Connection, row: sqlite3.Row) -> MealRecord:
        item_rows = connection.execute(
            """
            SELECT name, quantity, unit, calories, protein_g, carbs_g, fat_g, confidence
            FROM meal_items WHERE revision_id = ? ORDER BY rowid
            """,
            (row["revision_id"],),
        ).fetchall()
        items = tuple(
            MealItemDraft(
                name=item["name"],
                quantity=Decimal(item["quantity"]),
                unit=item["unit"],
                nutrition=Nutrition(
                    calories=Decimal(item["calories"]),
                    protein_g=Decimal(item["protein_g"]),
                    carbs_g=Decimal(item["carbs_g"]),
                    fat_g=Decimal(item["fat_g"]),
                ),
                confidence=item["confidence"],
            )
            for item in item_rows
        )
        return MealRecord(
            id=row["id"],
            user_id=row["user_id"],
            meal_type=MealType(row["meal_type"]),
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            source_text=row["source_text"],
            created_at=datetime.fromisoformat(row["created_at"]),
            items=items,
        )
