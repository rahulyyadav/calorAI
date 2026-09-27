from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from calorai_agent.domain import (
    DailyTotals,
    InboundEvent,
    InterpretationOrigin,
    MealDraft,
    MealItemDraft,
    MealOutcome,
    MealRecord,
    MemoryContent,
    MemoryKind,
    MemoryRecord,
)
from calorai_agent.memory import MEMORY_CONTEXT_LIMIT
from calorai_agent.repository import MealRepository


class LogMealInput(BaseModel):
    user_id: str = Field(min_length=1)
    meal: MealDraft
    source_event_id: str | None = None


class ReviseMealInput(BaseModel):
    user_id: str = Field(min_length=1)
    meal_id: str = Field(min_length=1)
    items: tuple[MealItemDraft, ...] = Field(min_length=1)
    source_text: str = Field(min_length=1)
    replace_items: bool = False
    origin: InterpretationOrigin = InterpretationOrigin.RULE_BASED
    model: str | None = None
    source_event_id: str | None = None


class DeleteMealInput(BaseModel):
    user_id: str = Field(min_length=1)
    meal_id: str = Field(min_length=1)
    source_event_id: str | None = None


class GetMealsInRangeInput(BaseModel):
    user_id: str = Field(min_length=1)
    start_day: date
    end_day: date
    timezone: str = "UTC"


class GetDailyTotalsInput(BaseModel):
    user_id: str = Field(min_length=1)
    day: date
    timezone: str = "UTC"


class RecordInboundInput(BaseModel):
    user_id: str = Field(min_length=1)
    external_id: str = Field(min_length=1)
    text: str
    channel: str = Field(min_length=1)


class RememberInput(BaseModel):
    user_id: str = Field(min_length=1)
    memory: MemoryContent
    source_event_id: str | None = None


class ListMemoriesInput(BaseModel):
    user_id: str = Field(min_length=1)
    kinds: tuple[MemoryKind, ...] = ()
    limit: int = Field(default=MEMORY_CONTEXT_LIMIT, ge=1, le=50)


class MealTools:
    """Small application-level tool surface; models never receive database access."""

    def __init__(self, repository: MealRepository) -> None:
        self.repository = repository

    def log_meal(self, command: LogMealInput) -> MealRecord:
        return self.repository.create(command.user_id, command.meal, command.source_event_id)

    def revise_meal(self, command: ReviseMealInput) -> MealRecord | None:
        return self.repository.revise(
            meal_id=command.meal_id,
            replacement_items=command.items,
            source_text=command.source_text,
            replace_items=command.replace_items,
            origin=command.origin,
            model=command.model,
            source_event_id=command.source_event_id,
        )

    def delete_meal(self, command: DeleteMealInput) -> bool:
        return self.repository.soft_delete(command.meal_id, command.source_event_id)

    def get_meals_in_range(self, query: GetMealsInRangeInput) -> list[MealRecord]:
        if query.start_day > query.end_day:
            raise ValueError("start_day must not be after end_day")
        return self.repository.list_for_range(
            query.user_id, query.start_day, query.end_day, query.timezone
        )

    def get_daily_totals(self, query: GetDailyTotalsInput) -> DailyTotals:
        return self.repository.totals_for_day(query.user_id, query.day, query.timezone)

    def remember(self, command: RememberInput) -> MemoryRecord:
        return self.repository.remember(command.user_id, command.memory, command.source_event_id)

    def list_memories(self, query: ListMemoriesInput) -> list[MemoryRecord]:
        return self.repository.active_memories(query.user_id, query.kinds, query.limit)

    def record_inbound(self, command: RecordInboundInput) -> InboundEvent:
        return self.repository.record_inbound(
            command.user_id, command.external_id, command.channel, command.text
        )

    def complete_inbound(self, event_id: str, response_text: str) -> None:
        self.repository.complete_inbound(event_id, response_text)

    def outcome_for_event(self, event_id: str) -> MealOutcome | None:
        return self.repository.outcome_for_event(event_id)

    def memory_for_event(self, event_id: str) -> MemoryRecord | None:
        return self.repository.memory_for_event(event_id)
