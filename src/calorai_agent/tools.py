from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from calorai_agent.domain import DailyTotals, MealDraft, MealRecord
from calorai_agent.repository import MealRepository


class LogMealInput(BaseModel):
    user_id: str = Field(min_length=1)
    meal: MealDraft
    source_event_id: str | None = None


class GetMealsInput(BaseModel):
    user_id: str = Field(min_length=1)
    day: date
    timezone: str = "UTC"


class GetDailyTotalsInput(BaseModel):
    user_id: str = Field(min_length=1)
    day: date
    timezone: str = "UTC"


class MealTools:
    """Small application-level tool surface; models never receive database access."""

    def __init__(self, repository: MealRepository) -> None:
        self.repository = repository

    def log_meal(self, command: LogMealInput) -> MealRecord:
        return self.repository.create(command.user_id, command.meal, command.source_event_id)

    def get_meals(self, query: GetMealsInput) -> list[MealRecord]:
        return self.repository.list_for_day(query.user_id, query.day, query.timezone)

    def get_daily_totals(self, query: GetDailyTotalsInput) -> DailyTotals:
        return self.repository.totals_for_day(query.user_id, query.day, query.timezone)
