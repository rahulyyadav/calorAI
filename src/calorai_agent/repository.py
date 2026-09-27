from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import TypeAdapter

from calorai_agent.db import Database
from calorai_agent.domain import (
    DailyTotals,
    InboundEvent,
    InterpretationOrigin,
    MealDraft,
    MealItemDraft,
    MealOutcome,
    MealRecord,
    MealType,
    MemoryContent,
    MemoryKind,
    MemoryRecord,
    MutationKind,
    NamedRoutine,
    Nutrition,
    memory_key,
)
from calorai_agent.memory import MEMORY_KIND_LIMITS

_MEMORY_ADAPTER: TypeAdapter[MemoryContent] = TypeAdapter(MemoryContent)


def _now() -> datetime:
    return datetime.now(UTC)


class MealRepository:
    """SQLite persistence for meals, memories, immutable revisions, and event idempotency."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def ensure_user(self, user_id: str, timezone_name: str = "UTC") -> None:
        ZoneInfo(timezone_name)
        with self.database.write_transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO users(id, timezone, created_at) VALUES (?, ?, ?)",
                (user_id, timezone_name, _now().isoformat()),
            )
            connection.execute(
                "UPDATE users SET timezone = ? WHERE id = ?", (timezone_name, user_id)
            )

    def create(
        self,
        user_id: str,
        draft: MealDraft,
        source_event_id: str | None = None,
    ) -> MealRecord:
        meal_id = str(uuid4())
        created_at = _now()
        with self.database.write_transaction() as connection:
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
            self._insert_revision(
                connection,
                meal_id,
                revision_number=1,
                source_text=draft.source_text,
                notes=draft.notes,
                items=draft.items,
                origin=draft.origin,
                model=draft.model,
                superseded_at=None,
            )
            self._stamp(connection, source_event_id, MutationKind.LOGGED, meal_id)
        return MealRecord(
            id=meal_id,
            user_id=user_id,
            meal_type=draft.meal_type,
            occurred_at=draft.occurred_at,
            source_text=draft.source_text,
            created_at=created_at,
            items=draft.items,
            revision_number=1,
        )

    def revise(
        self,
        meal_id: str,
        replacement_items: tuple[MealItemDraft, ...],
        source_text: str,
        *,
        replace_items: bool = False,
        notes: str | None = None,
        origin: InterpretationOrigin = InterpretationOrigin.RULE_BASED,
        model: str | None = None,
        source_event_id: str | None = None,
    ) -> MealRecord | None:
        """Supersede the active revision and write the corrected one atomically.

        Totals read only the active revision, so a correction changes the numbers
        exactly once instead of double-counting.
        """
        superseded_at = _now()
        with self.database.write_transaction() as connection:
            active = connection.execute(
                """
                SELECT id, revision_number, notes
                FROM meal_revisions
                WHERE meal_id = ? AND superseded_at IS NULL
                """,
                (meal_id,),
            ).fetchone()
            if active is None:
                return None
            existing = self._load_items(connection, active["id"])
            items = replacement_items if replace_items else merge_items(existing, replacement_items)
            connection.execute(
                "UPDATE meal_revisions SET superseded_at = ? WHERE id = ?",
                (superseded_at.isoformat(), active["id"]),
            )
            self._insert_revision(
                connection,
                meal_id,
                revision_number=active["revision_number"] + 1,
                source_text=source_text,
                notes=notes if notes is not None else active["notes"],
                items=items,
                origin=origin,
                model=model,
                superseded_at=None,
            )
            self._stamp(connection, source_event_id, MutationKind.REVISED, meal_id)
        return self.get(meal_id)

    def soft_delete(self, meal_id: str, source_event_id: str | None = None) -> bool:
        deleted_at = _now()
        with self.database.write_transaction() as connection:
            cursor = connection.execute(
                "UPDATE meals SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
                (deleted_at.isoformat(), meal_id),
            )
            connection.execute(
                "UPDATE meal_revisions SET superseded_at = ? "
                "WHERE meal_id = ? AND superseded_at IS NULL",
                (deleted_at.isoformat(), meal_id),
            )
            self._stamp(connection, source_event_id, MutationKind.DELETED, meal_id)
            return cursor.rowcount > 0

    @staticmethod
    def _stamp(
        connection: sqlite3.Connection,
        source_event_id: str | None,
        kind: MutationKind,
        meal_id: str,
    ) -> None:
        """Record what this inbound message changed, in the mutation's own transaction."""
        if source_event_id is None:
            return
        connection.execute(
            "UPDATE inbound_events SET result_kind = ?, result_meal_id = ? WHERE id = ?",
            (kind.value, meal_id, source_event_id),
        )

    def outcome_for_event(self, event_id: str) -> MealOutcome | None:
        """The mutation an inbound message already committed, if it committed one."""
        with self.database.connect() as connection:
            row = connection.execute(
                "SELECT result_kind, result_meal_id FROM inbound_events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None or not row["result_kind"] or not row["result_meal_id"]:
                return None
            meal = self._load_record(connection, row["result_meal_id"], include_deleted=True)
            if meal is None:  # pragma: no cover - an outcome always names a real meal
                return None
            return MealOutcome(kind=MutationKind(row["result_kind"]), meal=meal)

    def get(self, meal_id: str) -> MealRecord | None:
        with self.database.connect() as connection:
            return self._load_record(connection, meal_id)

    @classmethod
    def _load_record(
        cls, connection: sqlite3.Connection, meal_id: str, *, include_deleted: bool = False
    ) -> MealRecord | None:
        # A deleted meal keeps no active revision, so replay reads its last one.
        revision = (
            "r.superseded_at IS NULL"
            if not include_deleted
            else "r.revision_number = ("
            "SELECT MAX(revision_number) FROM meal_revisions WHERE meal_id = m.id)"
        )
        deleted = "" if include_deleted else "AND m.deleted_at IS NULL"
        row = connection.execute(
            f"""
            SELECT m.id, m.user_id, m.meal_type, m.occurred_at, m.created_at,
                   r.id AS revision_id, r.revision_number, r.source_text
            FROM meals m
            JOIN meal_revisions r ON r.meal_id = m.id AND {revision}
            WHERE m.id = ? {deleted}
            """,
            (meal_id,),
        ).fetchone()
        if row is None:
            return None
        return cls._record_from_row(connection, row)

    def list_for_day(self, user_id: str, day: date, timezone_name: str = "UTC") -> list[MealRecord]:
        start, end = self._local_day_bounds(day, timezone_name)
        return self._list_between(user_id, start, end)

    def list_for_range(
        self, user_id: str, start_day: date, end_day: date, timezone_name: str = "UTC"
    ) -> list[MealRecord]:
        start, _ = self._local_day_bounds(start_day, timezone_name)
        _, end = self._local_day_bounds(end_day, timezone_name)
        return self._list_between(user_id, start, end)

    def totals_for_day(self, user_id: str, day: date, timezone_name: str = "UTC") -> DailyTotals:
        meals = self.list_for_day(user_id, day, timezone_name)
        total = Nutrition.zero()
        for meal in meals:
            total += meal.nutrition
        return DailyTotals(user_id=user_id, day=day, meal_count=len(meals), nutrition=total)

    def record_inbound(
        self, user_id: str, external_id: str, kind: str, raw_text: str
    ) -> InboundEvent:
        """Register an inbound message, returning the original row on redelivery."""
        event_id = str(uuid4())
        with self.database.write_transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO inbound_events(
                    id, user_id, external_id, kind, raw_text, received_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, user_id, external_id, kind, raw_text, _now().isoformat()),
            )
            row = connection.execute(
                """
                SELECT id, external_id, response_text, completed_at
                FROM inbound_events WHERE external_id = ?
                """,
                (external_id,),
            ).fetchone()
        return InboundEvent(
            id=row["id"],
            external_id=row["external_id"],
            response_text=row["response_text"],
            completed=row["completed_at"] is not None,
            created=row["id"] == event_id,
        )

    def complete_inbound(self, event_id: str, response_text: str) -> None:
        with self.database.write_transaction() as connection:
            connection.execute(
                "UPDATE inbound_events SET response_text = ?, completed_at = ? WHERE id = ?",
                (response_text, _now().isoformat(), event_id),
            )

    def remember(
        self,
        user_id: str,
        content: MemoryContent,
        source_event_id: str | None = None,
    ) -> MemoryRecord:
        """Store a stated fact, retiring the active fact it replaces in the same transaction.

        Contradictions never accumulate: one slot holds one active memory, and the
        superseded row stays behind as the provenance of a changed mind.
        """
        if isinstance(content, NamedRoutine) and not content.items:
            raise ValueError("a named routine needs at least one food")
        created_at = _now()
        with self.database.write_transaction() as connection:
            connection.execute(
                """
                UPDATE memories SET superseded_at = ?
                WHERE user_id = ? AND memory_type = ? AND key = ? AND superseded_at IS NULL
                """,
                (
                    created_at.isoformat(),
                    user_id,
                    content.kind.value,
                    memory_key(content),
                ),
            )
            memory_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO memories(
                    id, user_id, memory_type, key, value_json, confidence,
                    source_event_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    user_id,
                    content.kind.value,
                    memory_key(content),
                    content.model_dump_json(),
                    content.confidence,
                    source_event_id,
                    created_at.isoformat(),
                ),
            )
        return MemoryRecord(
            id=memory_id,
            user_id=user_id,
            content=content,
            source_event_id=source_event_id,
            created_at=created_at,
        )

    def memory_for_event(self, event_id: str) -> MemoryRecord | None:
        """The memory an inbound message committed, so a redelivery can be answered."""
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, value_json, source_event_id, created_at
                FROM memories WHERE source_event_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (event_id,),
            ).fetchone()
        return None if row is None else self._memory_from_row(row)

    def active_memories(
        self,
        user_id: str,
        kinds: Sequence[MemoryKind] = (),
    ) -> list[MemoryRecord]:
        """The facts still standing for a user, bounded per kind and newest first within it.

        One shared cap would be spent by whichever kind the user saves most often, quietly
        evicting the diet or the targets that every later reply depends on. Each kind gets its
        own budget instead, so asking for routines can never cost the user their vegetarian
        rule.
        """
        records: list[MemoryRecord] = []
        with self.database.connect() as connection:
            for kind in tuple(kinds) or tuple(MEMORY_KIND_LIMITS):
                rows = connection.execute(
                    """
                    SELECT id, user_id, value_json, source_event_id, created_at
                    FROM memories
                    WHERE user_id = ? AND superseded_at IS NULL AND memory_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, kind.value, MEMORY_KIND_LIMITS[kind]),
                ).fetchall()
                records.extend(self._memory_from_row(row) for row in rows)
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            content=_MEMORY_ADAPTER.validate_json(row["value_json"]),
            source_event_id=row["source_event_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def _list_between(self, user_id: str, start: datetime, end: datetime) -> list[MealRecord]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT m.id, m.user_id, m.meal_type, m.occurred_at, m.created_at,
                       r.id AS revision_id, r.revision_number, r.source_text
                FROM meals m
                JOIN meal_revisions r ON r.meal_id = m.id AND r.superseded_at IS NULL
                WHERE m.user_id = ? AND m.deleted_at IS NULL
                  AND m.occurred_at >= ? AND m.occurred_at < ?
                ORDER BY m.occurred_at, m.created_at
                """,
                (user_id, start.isoformat(), end.isoformat()),
            ).fetchall()
            return [self._record_from_row(connection, row) for row in rows]

    @staticmethod
    def _local_day_bounds(day: date, timezone_name: str) -> tuple[datetime, datetime]:
        """Local-calendar-day bounds, DST-safe (midnight-to-midnight, not +24h)."""
        zone = ZoneInfo(timezone_name)
        start = datetime.combine(day, time.min, tzinfo=zone)
        end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=zone)
        return start.astimezone(UTC), end.astimezone(UTC)

    @staticmethod
    def _insert_revision(
        connection: sqlite3.Connection,
        meal_id: str,
        revision_number: int,
        source_text: str,
        notes: str | None,
        items: tuple[MealItemDraft, ...],
        origin: InterpretationOrigin,
        model: str | None,
        superseded_at: datetime | None,
    ) -> None:
        revision_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO meal_revisions(
                id, meal_id, revision_number, source_text, notes, created_at,
                superseded_at, origin, model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                meal_id,
                revision_number,
                source_text,
                notes,
                _now().isoformat(),
                superseded_at.isoformat() if superseded_at else None,
                origin.value,
                model,
            ),
        )
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
    def _load_items(connection: sqlite3.Connection, revision_id: str) -> tuple[MealItemDraft, ...]:
        rows = connection.execute(
            """
            SELECT name, quantity, unit, calories, protein_g, carbs_g, fat_g, confidence
            FROM meal_items WHERE revision_id = ? ORDER BY rowid
            """,
            (revision_id,),
        ).fetchall()
        return tuple(
            MealItemDraft(
                name=row["name"],
                quantity=Decimal(row["quantity"]),
                unit=row["unit"],
                nutrition=Nutrition(
                    calories=Decimal(row["calories"]),
                    protein_g=Decimal(row["protein_g"]),
                    carbs_g=Decimal(row["carbs_g"]),
                    fat_g=Decimal(row["fat_g"]),
                ),
                confidence=row["confidence"],
            )
            for row in rows
        )

    @classmethod
    def _record_from_row(cls, connection: sqlite3.Connection, row: sqlite3.Row) -> MealRecord:
        return MealRecord(
            id=row["id"],
            user_id=row["user_id"],
            meal_type=MealType(row["meal_type"]),
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            source_text=row["source_text"],
            created_at=datetime.fromisoformat(row["created_at"]),
            items=cls._load_items(connection, row["revision_id"]),
            revision_number=row["revision_number"],
        )


def merge_items(
    existing: tuple[MealItemDraft, ...], replacements: tuple[MealItemDraft, ...]
) -> tuple[MealItemDraft, ...]:
    """Apply a partial correction while keeping untouched items from the meal.

    "actually 3 rotis not 2" replaces only the roti line; the chai stays logged,
    and the original item order is preserved so replies read consistently.
    """
    by_name = {item.name.lower(): item for item in replacements}
    merged: list[MealItemDraft] = []
    used: set[str] = set()
    for item in existing:
        replacement = by_name.get(item.name.lower())
        if replacement is None:
            merged.append(item)
        else:
            merged.append(replacement)
            used.add(item.name.lower())
    merged.extend(item for item in replacements if item.name.lower() not in used)
    return tuple(merged)
