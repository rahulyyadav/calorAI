from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

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


def combine_portions(
    base: tuple[MealItemDraft, ...], additions: tuple[MealItemDraft, ...]
) -> tuple[MealItemDraft, ...]:
    """Fold extra portions into the lines they belong to, keeping the original order.

    "my usual breakfast, plus a banana" is one meal: the routine's own foods with the banana
    added to it. Nutrition is summed rather than recomputed, because both lines are priced from
    the same reference table, so 2 idli plus 1 idli is exactly 3 idli.
    """
    combined = list(base)
    for addition in additions:
        existing = next(
            (index for index, item in enumerate(combined) if item.name == addition.name), None
        )
        if existing is None:
            combined.append(addition)
            continue
        line = combined[existing]
        combined[existing] = line.model_copy(
            update={
                "quantity": line.quantity + addition.quantity,
                "nutrition": line.nutrition + addition.nutrition,
                "confidence": min(line.confidence, addition.confidence),
            }
        )
    return tuple(combined)


class InterpretationOrigin(StrEnum):
    RULE_BASED = "rule_based"
    TEXT_MODEL = "text_model"
    VISION_FUSION = "vision_fusion"
    USER_CONFIRMED = "user_confirmed"


class MediaRef(BaseModel):
    """Where a photo came from, phrased so no adapter detail reaches the graph.

    A CLI hands over a path; WhatsApp hands over a media id that Phase 5 downloads. Either
    way the graph only ever sees this reference and an external id that deduplicates it.
    """

    model_config = ConfigDict(frozen=True)

    external_id: str = Field(min_length=1, max_length=200)
    locator: str = Field(min_length=1, max_length=2000)
    source: Literal["local_path", "whatsapp_media"] = "local_path"


class FoodObservation(BaseModel):
    """One line of a vision model's read of a plate, with the model's own doubt attached.

    Deliberately carries no nutrition: the application prices every line from the reference
    table, so a model cannot invent a calorie number.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=200)
    quantity: Decimal = Field(default=Decimal("1"), gt=0)
    confidence: float = Field(default=1.0, ge=0, le=1)
    alternative: str | None = Field(default=None, max_length=200)


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
    SAVE_MEMORY = "save_memory"
    ACKNOWLEDGE = "acknowledge"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"


class MemoryKind(StrEnum):
    DIETARY_CONSTRAINT = "dietary_constraint"
    NUTRITION_TARGET = "nutrition_target"
    NAMED_ROUTINE = "named_routine"


class NutrientMetric(StrEnum):
    CALORIES = "calories"
    PROTEIN_G = "protein_g"
    CARBS_G = "carbs_g"
    FAT_G = "fat_g"


class MemoryFact(BaseModel):
    """One durable fact, kept with the certainty the user stated it at."""

    model_config = ConfigDict(frozen=True)

    confidence: float = Field(default=1.0, ge=0, le=1)


class DietaryConstraint(MemoryFact):
    kind: Literal[MemoryKind.DIETARY_CONSTRAINT] = MemoryKind.DIETARY_CONSTRAINT
    diet: str = Field(min_length=1, max_length=60)


class NutritionTarget(MemoryFact):
    kind: Literal[MemoryKind.NUTRITION_TARGET] = MemoryKind.NUTRITION_TARGET
    metric: NutrientMetric
    value: Decimal = Field(gt=0, le=20000)


class NamedRoutine(MemoryFact):
    kind: Literal[MemoryKind.NAMED_ROUTINE] = MemoryKind.NAMED_ROUTINE
    slot: str = Field(min_length=1, max_length=40)
    items: tuple[MealItemDraft, ...] = ()


MemoryContent = Annotated[
    DietaryConstraint | NutritionTarget | NamedRoutine,
    Field(discriminator="kind"),
]


DIET_KEY = "diet"


def memory_key(content: MemoryContent) -> str:
    """The slot a new memory replaces, so a changed fact supersedes instead of contradicting.

    A diet occupies one slot however it is worded, so becoming non-vegetarian retires the
    vegetarian record. Targets key on the nutrient and routines on their name, letting
    several of each stay active at once.
    """
    if isinstance(content, DietaryConstraint):
        return DIET_KEY
    if isinstance(content, NutritionTarget):
        return content.metric.value
    return content.slot


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A persisted memory: the fact, and the inbound message that earned it."""

    id: str
    user_id: str
    content: MemoryContent
    source_event_id: str | None
    created_at: datetime


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
    unrecognized: tuple[str, ...] = ()
    memory: MemoryContent | None = None
    unlogged: tuple[str, ...] = ()
    reply_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_payload(self) -> ParsedMessage:
        if self.intent is AgentIntent.LOG_MEAL and self.draft is None:
            raise ValueError("log_meal requires a meal draft")
        if self.intent is AgentIntent.LOG_MEAL and isinstance(self.memory, NamedRoutine):
            # A routine remembers a meal that already exists. On the turn that creates the
            # meal there is nothing to point at yet, so it stays its own message.
            raise ValueError("a routine can only be remembered as its own turn")
        if self.intent is AgentIntent.REPEAT_MEAL and self.reference is None:
            raise ValueError("repeat_meal requires a meal reference")
        if self.intent in (AgentIntent.REVISE_MEAL, AgentIntent.DELETE_MEAL):
            if self.reference is None:
                raise ValueError(f"{self.intent.value} requires a meal reference")
            if self.intent is AgentIntent.REVISE_MEAL and not self.items:
                raise ValueError("revise_meal requires replacement items")
        if self.intent is AgentIntent.SAVE_MEMORY:
            if self.memory is None:
                raise ValueError("save_memory requires a memory")
            if isinstance(self.memory, NamedRoutine) and self.reference is None:
                raise ValueError("a named routine requires a meal reference")
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
    media: MediaRef | None = None


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
