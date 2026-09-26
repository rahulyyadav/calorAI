from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MealType(StrEnum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    UNSPECIFIED = "unspecified"


class Nutrition(BaseModel):
    model_config = ConfigDict(frozen=True)

    calories: Decimal = Field(ge=0)
    protein_g: Decimal = Field(ge=0)
    carbs_g: Decimal = Field(ge=0)
    fat_g: Decimal = Field(ge=0)

    @field_validator("calories", "protein_g", "carbs_g", "fat_g")
    @classmethod
    def normalize_precision(cls, value: Decimal) -> Decimal:
        return value.quantize(Decimal("0.01"))

    def scaled(self, quantity: Decimal) -> Nutrition:
        return Nutrition(
            calories=self.calories * quantity,
            protein_g=self.protein_g * quantity,
            carbs_g=self.carbs_g * quantity,
            fat_g=self.fat_g * quantity,
        )

    def __add__(self, other: Nutrition) -> Nutrition:
        return Nutrition(
            calories=self.calories + other.calories,
            protein_g=self.protein_g + other.protein_g,
            carbs_g=self.carbs_g + other.carbs_g,
            fat_g=self.fat_g + other.fat_g,
        )

    @classmethod
    def zero(cls) -> Nutrition:
        zero = Decimal("0")
        return cls(calories=zero, protein_g=zero, carbs_g=zero, fat_g=zero)


class MealItemDraft(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    quantity: Decimal = Field(default=Decimal("1"), gt=0)
    unit: str = Field(default="serving", min_length=1, max_length=40)
    nutrition: Nutrition
    confidence: float = Field(default=1.0, ge=0, le=1)


class MealDraft(BaseModel):
    meal_type: MealType = MealType.UNSPECIFIED
    occurred_at: datetime
    source_text: str = Field(min_length=1)
    items: tuple[MealItemDraft, ...] = Field(min_length=1)
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class MealRecord:
    id: str
    user_id: str
    meal_type: MealType
    occurred_at: datetime
    source_text: str
    created_at: datetime
    items: tuple[MealItemDraft, ...]

    @property
    def nutrition(self) -> Nutrition:
        total = Nutrition.zero()
        for item in self.items:
            total += item.nutrition
        return total


@dataclass(frozen=True, slots=True)
class DailyTotals:
    user_id: str
    day: date
    meal_count: int
    nutrition: Nutrition


class AgentIntent(StrEnum):
    LOG_MEAL = "log_meal"
    GET_TOTALS = "get_totals"
    LIST_MEALS = "list_meals"
    UNKNOWN = "unknown"


class ParsedMessage(BaseModel):
    intent: AgentIntent
    draft: MealDraft | None = None
    explanation: str | None = None
