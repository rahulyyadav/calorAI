"""What the agent remembers, and how a memory is allowed to change behavior.

Memories are few, short, and typed: a diet, a daily target, a named routine. They are read
as a bounded context so a long history can never inflate the prompt or the latency of a
turn, and each consumer asks for the one kind its intent actually needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from calorai_agent.domain import (
    DietaryConstraint,
    MealItemDraft,
    MealType,
    MemoryContent,
    MemoryRecord,
    NamedRoutine,
    NutrientMetric,
    Nutrition,
    NutritionTarget,
)

# A cap on what one turn carries, not on what the database keeps.
MEMORY_CONTEXT_LIMIT = 8

DEFAULT_ROUTINE_SLOT = "default"

_ANIMAL_FOODS = frozenset({"chicken", "fish", "egg"})
_DAIRY_FOODS = frozenset({"milk", "milk chai", "curd", "yogurt", "paneer"})

# Canonical food names each diet keeps out of a meal.
DIET_AVOIDS: dict[str, frozenset[str]] = {
    "non-vegetarian": frozenset(),
    "eggetarian": _ANIMAL_FOODS - {"egg"},
    "pescatarian": _ANIMAL_FOODS - {"fish"} | {"egg"},
    "vegetarian": _ANIMAL_FOODS,
    "vegan": _ANIMAL_FOODS | _DAIRY_FOODS,
}

# Spoken diet words folded onto a canonical diet, so "veg" and "vegetarian" are one fact.
DIET_ALIASES: dict[str, str] = {
    "vegetarian": "vegetarian",
    "veg": "vegetarian",
    "veggie": "vegetarian",
    "pure veg": "vegetarian",
    "veg only": "vegetarian",
    "vegan": "vegan",
    "pescatarian": "pescatarian",
    "eggetarian": "eggetarian",
    "eggietarian": "eggetarian",
    "non-vegetarian": "non-vegetarian",
    "non vegetarian": "non-vegetarian",
    "non-veg": "non-vegetarian",
    "not vegetarian": "non-vegetarian",
    "not vegan": "non-vegetarian",
    "stopped being vegetarian": "non-vegetarian",
}

METRIC_WORDS: dict[str, NutrientMetric] = {
    "calorie": NutrientMetric.CALORIES,
    "calories": NutrientMetric.CALORIES,
    "calorie-intake": NutrientMetric.CALORIES,
    "kcal": NutrientMetric.CALORIES,
    "kcals": NutrientMetric.CALORIES,
    "cals": NutrientMetric.CALORIES,
    "protein": NutrientMetric.PROTEIN_G,
    "carbs": NutrientMetric.CARBS_G,
    "carb": NutrientMetric.CARBS_G,
    "carbohydrate": NutrientMetric.CARBS_G,
    "carbohydrates": NutrientMetric.CARBS_G,
    "fat": NutrientMetric.FAT_G,
}

METRIC_LABELS: dict[NutrientMetric, str] = {
    NutrientMetric.CALORIES: "calories",
    NutrientMetric.PROTEIN_G: "protein",
    NutrientMetric.CARBS_G: "carbs",
    NutrientMetric.FAT_G: "fat",
}


def metric_value(nutrition: Nutrition, metric: NutrientMetric) -> Decimal:
    values: dict[NutrientMetric, Decimal] = {
        NutrientMetric.CALORIES: nutrition.calories,
        NutrientMetric.PROTEIN_G: nutrition.protein_g,
        NutrientMetric.CARBS_G: nutrition.carbs_g,
        NutrientMetric.FAT_G: nutrition.fat_g,
    }
    return values[metric]


def target_phrase(target: NutritionTarget) -> str:
    """A target stated the way a person states it: "120 g protein", "1800 calories"."""
    if target.metric is NutrientMetric.CALORIES:
        return f"{target.value:g} calories"
    return f"{target.value:g} g {METRIC_LABELS[target.metric]}"


def normalized_diet(spoken: str) -> str:
    """Fold a spoken diet word onto the canonical diet it names."""
    return DIET_ALIASES.get(" ".join(spoken.lower().split()), spoken.lower())


def avoided_foods(constraints: Sequence[DietaryConstraint]) -> frozenset[str]:
    """Foods the user has ruled out. Diets arrive newest first, so the first one wins."""
    if not constraints:
        return frozenset()
    return DIET_AVOIDS.get(constraints[0].diet, frozenset())


def current_diet(memories: Sequence[MemoryRecord]) -> str | None:
    """The diet the user last claimed for themselves, if they ever did."""
    stated = diets(memories)
    return stated[0].diet if stated else None


def conflicting_foods(
    memories: Sequence[MemoryRecord], items: Sequence[MealItemDraft]
) -> tuple[str, ...]:
    """Foods being logged that the user's own diet rules out."""
    avoided = avoided_foods(diets(memories))
    if not avoided:
        return ()
    return tuple(item.name for item in items if item.name in avoided)


def diets(memories: Sequence[MemoryRecord]) -> tuple[DietaryConstraint, ...]:
    return tuple(
        record.content for record in memories if isinstance(record.content, DietaryConstraint)
    )


def targets(memories: Sequence[MemoryRecord]) -> tuple[NutritionTarget, ...]:
    return tuple(
        record.content for record in memories if isinstance(record.content, NutritionTarget)
    )


def routines(memories: Sequence[MemoryRecord]) -> tuple[NamedRoutine, ...]:
    return tuple(record.content for record in memories if isinstance(record.content, NamedRoutine))


def routine_for(
    memories: Sequence[MemoryRecord], meal_type: MealType | None
) -> NamedRoutine | None:
    """The routine a message asks for, or None when it could name more than one.

    An unqualified "my usual" is only safe to honour when the user has saved one routine;
    otherwise guessing between their usual breakfast and usual dinner is a wrong number
    waiting to happen, so the caller asks instead.
    """
    saved = routines(memories)
    if not saved:
        return None
    if meal_type is not None and meal_type is not MealType.UNSPECIFIED:
        wanted = meal_type.value
        return next((routine for routine in saved if routine.slot == wanted), None)
    if len(saved) == 1:
        return saved[0]
    defaults = [routine for routine in saved if routine.slot == DEFAULT_ROUTINE_SLOT]
    return defaults[0] if len(defaults) == 1 else None


def routine_meal_type(routine: NamedRoutine) -> MealType:
    try:
        return MealType(routine.slot)
    except ValueError:
        return MealType.UNSPECIFIED


def describe(content: MemoryContent) -> str:
    """A memory in the user's words, used both to confirm a save and to prompt a model."""
    if isinstance(content, DietaryConstraint):
        return f"you are {content.diet}"
    if isinstance(content, NutritionTarget):
        return f"daily {target_phrase(content)} target"
    items = ", ".join(f"{item.quantity:g} {item.name}" for item in content.items)
    slot = "meal" if content.slot == DEFAULT_ROUTINE_SLOT else content.slot
    return f"your usual {slot} is {items}" if items else f"your usual {slot}"


def context_lines(memories: Sequence[MemoryRecord]) -> list[str]:
    """Memories rendered for a model prompt, newest first and capped."""
    return [f"- {describe(record.content)}" for record in memories][:MEMORY_CONTEXT_LIMIT]
