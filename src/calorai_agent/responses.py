from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from zoneinfo import ZoneInfo

from calorai_agent.domain import (
    DailyTotals,
    MealItemDraft,
    MealRecord,
    MealType,
    MemoryContent,
    Nutrition,
    NutritionTarget,
)
from calorai_agent.memory import describe, metric_value, target_phrase
from calorai_agent.policy import LogDecision


def quantity(value: Decimal) -> str:
    """A portion rounded no further than two decimals: 0.75 of a shared plate must not
    come back as 0.8, or the reply states a portion the user never ate."""
    if value == value.to_integral():
        return str(int(value))
    return str(value.quantize(Decimal("0.01")).normalize())


def item_line(item: MealItemDraft) -> str:
    counted = quantity(item.quantity)
    if item.quantity == item.quantity.to_integral():
        return f"{counted} {item.name}"
    return f"{counted} {item.unit} of {item.name}"


def item_summary(meal: MealRecord) -> str:
    return ", ".join(item_line(item) for item in meal.items)


def logged(meal: MealRecord, decision: LogDecision, notes: Sequence[str] = ()) -> str:
    nutrition = meal.nutrition
    if decision is LogDecision.LOG_AS_ESTIMATE:
        return (
            f"Logged about {item_summary(meal)} — roughly {nutrition.calories:.0f} kcal and "
            f"{nutrition.protein_g:.0f}g protein. That is an estimate; correct me if the "
            f"portion was different.{_notes(notes)}"
        )
    return (
        f"Logged {item_summary(meal)} — about {nutrition.calories:.0f} kcal and "
        f"{nutrition.protein_g:.0f}g protein.{_notes(notes)}"
    )


def revised(meal: MealRecord, notes: Sequence[str] = (), *, timezone_name: str) -> str:
    nutrition = meal.nutrition
    return (
        f"Updated your {_meal_label(meal, timezone_name)} to {item_summary(meal)} — now about "
        f"{nutrition.calories:.0f} kcal and {nutrition.protein_g:.0f}g "
        f"protein.{_notes(notes)}"
    )


def _notes(notes: Sequence[str]) -> str:
    return "".join(f" {note}" for note in notes if note)


def omission(unrecognized: Sequence[str]) -> str:
    """Name the foods left out, so a partial meal is never presented as complete."""
    if not unrecognized:
        return ""
    named = ", ".join(unrecognized[:3])
    counted = "it is" if len(unrecognized) == 1 else "they are"
    return f"I have no reference data for {named}, so {counted} not counted."


def diet_conflict(diet: str, foods: Sequence[str]) -> str:
    if not foods:
        return ""
    named = ", ".join(foods[:3])
    counted = "is" if len(foods) == 1 else "are"
    return f"Heads up — {named} {counted} not {diet}. Tell me if that has changed."


def remembered(content: MemoryContent, unlogged: Sequence[str] = ()) -> str:
    """Confirm the fact, and name the foods the same message stated but this turn did not log."""
    kept = f"Got it — I will remember {describe(content)}."
    if not unlogged:
        return kept
    named = ", ".join(unlogged[:3])
    return f"{kept} I did not log {named} with it — send it again when you want it counted."


def nothing_to_remember(label: str) -> str:
    return (
        f"I couldn't find a logged meal {label.lower()} to remember. "
        "Log it first, then ask me to keep it as your usual."
    )


def deleted(meal: MealRecord) -> str:
    return f"Removed {item_summary(meal)} ({meal.nutrition.calories:.0f} kcal)."


def repeated(
    source: MealRecord,
    copy: MealRecord,
    day_label: str,
    notes: Sequence[str] = (),
    *,
    timezone_name: str,
) -> str:
    nutrition = copy.nutrition
    return (
        f"Logged the same as {day_label} — {item_summary(copy)}, about "
        f"{nutrition.calories:.0f} kcal and {nutrition.protein_g:.0f}g protein. "
        f"(Original: {_meal_label(source, timezone_name)}.){_notes(notes)}"
    )


def totals(day: DailyTotals, label: str, targets: Sequence[NutritionTarget] = ()) -> str:
    nutrition = day.nutrition
    progress = target_progress(nutrition, targets)
    if day.meal_count == 0:
        return f"Nothing logged {label.lower()} yet.{progress}"
    return (
        f"{label}: {nutrition.calories:.0f} kcal, {nutrition.protein_g:.0f}g protein, "
        f"{nutrition.carbs_g:.0f}g carbs, and {nutrition.fat_g:.0f}g fat "
        f"across {day.meal_count} {_plural(day.meal_count, 'meal')}.{progress}"
    )


def target_progress(nutrition: Nutrition, targets: Sequence[NutritionTarget]) -> str:
    """Where the day stands against the targets the user set, when they set any."""
    if not targets:
        return ""
    standing = ", ".join(
        f"{metric_value(nutrition, target.metric):.0f} of {target_phrase(target)}"
        for target in targets
    )
    return f" Against your targets: {standing}."


def meal_list(meals: Sequence[MealRecord], label: str, *, timezone_name: str) -> str:
    if not meals:
        return f"Nothing logged {label.lower()} yet."
    described = "; ".join(
        f"{_meal_label(meal, timezone_name)}: {item_summary(meal)}" for meal in meals
    )
    return f"{label} you logged: {described}."


def ambiguous(
    candidates: Sequence[MealRecord], label: str, verb: str, *, timezone_name: str
) -> str:
    described = " or ".join(_meal_label(meal, timezone_name) for meal in candidates[:3])
    return f"Which {label.lower()} meal do you mean to {verb} — {described}?"


def not_found(label: str) -> str:
    return f"I couldn't find a logged meal {label.lower()} to use."


def no_reference_match(candidates: Sequence[MealRecord], *, timezone_name: str) -> str:
    """Reply to a pointer that only made sense against the recent window, not today."""
    if not candidates:
        return "I couldn't find a recent meal that matches."
    described = " or ".join(_meal_label(meal, timezone_name) for meal in candidates[:3])
    return f"Which of your recent meals do you mean — {described}?"


def vision_unavailable() -> str:
    """A photo needs a vision model. Without one the turn is refused, never guessed around."""
    return (
        "I cannot read a photo until a vision model is configured — set "
        "CALORAI_VISION_MODEL_API_KEY, or any text model key, and I will use it. Describe the "
        "plate instead and I will log it."
    )


def media_rejected(reason: str) -> str:
    return f"I could not use that attachment: {reason} Describe the meal and I will log it."


def photo_unreadable() -> str:
    return (
        "I could not get a reliable read of that photo. Tell me what was on the plate and "
        "roughly how much, and I will log that instead."
    )


def _meal_label(meal: MealRecord, timezone_name: str) -> str:
    """A meal named the way the user would name it: its slot and its local clock time."""
    type_label = (
        meal.meal_type.value.replace("_", " ")
        if meal.meal_type is not MealType.UNSPECIFIED
        else "meal"
    )
    local = meal.occurred_at.astimezone(ZoneInfo(timezone_name))
    return f"{type_label} at {local:%H:%M}"


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else noun + "s"


def day_label(day: date, today: date) -> str:
    offset = (day - today).days
    return {0: "Today", 1: "Tomorrow", -1: "Yesterday"}.get(offset, f"{day:%d %b}")
