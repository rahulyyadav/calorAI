from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class InterpretationOrigin(StrEnum):
    RULE_BASED = "rule_based"
    TEXT_MODEL = "text_model"
    VISION_FUSION = "vision_fusion"
    USER_CONFIRMED = "user_confirmed"


class MealDraft(BaseModel):
    meal_type: MealType = MealType.UNSPECIFIED
    occurred_at: datetime
    source_text: str = Field(min_length=1)
    items: tuple[MealItemDraft, ...] = Field(min_length=1)
    notes: str | None = None
    origin: InterpretationOrigin = InterpretationOrigin.RULE_BASED
    model: str | None = None


@dataclass(frozen=True, slots=True)
class MealRecord:
    id: str
    user_id: str
    meal_type: MealType
    occurred_at: datetime
    source_text: str
    created_at: datetime
    items: tuple[MealItemDraft, ...]
    revision_number: int = 1

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
    REPEAT_MEAL = "repeat_meal"
    REVISE_MEAL = "revise_meal"
    DELETE_MEAL = "delete_meal"
    GET_TOTALS = "get_totals"
    LIST_MEALS = "list_meals"
    ACKNOWLEDGE = "acknowledge"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"


class MealReference(BaseModel):
    """A message's pointer at an already-stored meal."""

    model_config = ConfigDict(frozen=True)

    day_offset: int = Field(default=0, ge=-365, le=365)
    day_explicit: bool = False
    meal_type: MealType | None = None
    food_hint: str | None = None


class ParsedMessage(BaseModel):
    """Typed planner output: one intent plus the payload that intent requires."""

    model_config = ConfigDict(frozen=True)

    intent: AgentIntent
    draft: MealDraft | None = None
    items: tuple[MealItemDraft, ...] = ()
    reference: MealReference | None = None
    target_meal_type: MealType | None = None
    question: str | None = None
    statement: str | None = None
    explanation: str | None = None
    replace_items: bool = False

    @model_validator(mode="after")
    def validate_payload(self) -> ParsedMessage:
        if self.intent is AgentIntent.LOG_MEAL and self.draft is None:
            raise ValueError("log_meal requires a meal draft")
        if self.intent is AgentIntent.REPEAT_MEAL and self.reference is None:
            raise ValueError("repeat_meal requires a meal reference")
        if self.intent in (AgentIntent.REVISE_MEAL, AgentIntent.DELETE_MEAL):
            if self.reference is None:
                raise ValueError(f"{self.intent.value} requires a meal reference")
            if self.intent is AgentIntent.REVISE_MEAL and not self.items:
                raise ValueError("revise_meal requires replacement items")
        return self


@dataclass(frozen=True, slots=True)
class InboundMessage:
    """Transport-neutral envelope shared by the CLI and WhatsApp adapters."""

    user_id: str
    text: str
    external_id: str
    channel: str = "cli"
    timezone: str = "UTC"
    received_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class InboundEvent:
    """A persisted inbound message plus its exactly-once delivery bookkeeping."""

    id: str
    external_id: str
    response_text: str | None
    completed: bool
    created: bool


class MutationKind(StrEnum):
    LOGGED = "logged"
    REVISED = "revised"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class MealOutcome:
    """What an inbound message actually changed.

    Recorded in the same transaction as the mutation, so a message redelivered after a
    crash between the write and the reply can still be answered from the database.
    """

    kind: MutationKind
    meal: MealRecord
