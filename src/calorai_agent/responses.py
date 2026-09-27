from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from calorai_agent.domain import DailyTotals, MealItemDraft, MealRecord, MealType
from calorai_agent.policy import LogDecision


def quantity(value: Decimal) -> str:
    if value == value.to_integral():
        return str(int(value))
    return str(value.quantize(Decimal("0.1")).normalize())


def item_line(item: MealItemDraft) -> str:
    counted = quantity(item.quantity)
    if item.quantity == item.quantity.to_integral():
        return f"{counted} {item.name}"
    return f"{counted} {item.unit} of {item.name}"


def item_summary(meal: MealRecord) -> str:
    return ", ".join(item_line(item) for item in meal.items)


def logged(meal: MealRecord, decision: LogDecision) -> str:
    nutrition = meal.nutrition
    if decision is LogDecision.LOG_AS_ESTIMATE:
        return (
            f"Logged about {item_summary(meal)} — roughly {nutrition.calories:.0f} kcal and "
            f"{nutrition.protein_g:.0f}g protein. That is an estimate; correct me if the "
            "portion was different."
        )
    return (
        f"Logged {item_summary(meal)} — about {nutrition.calories:.0f} kcal and "
        f"{nutrition.protein_g:.0f}g protein."
    )


def revised(meal: MealRecord) -> str:
    nutrition = meal.nutrition
    return (
        f"Updated your {_meal_label(meal)} to {item_summary(meal)} — now about "
        f"{nutrition.calories:.0f} kcal and {nutrition.protein_g:.0f}g protein."
    )


def deleted(meal: MealRecord) -> str:
    return f"Removed {item_summary(meal)} ({meal.nutrition.calories:.0f} kcal)."


def repeated(source: MealRecord, copy: MealRecord, day_label: str) -> str:
    nutrition = copy.nutrition
    return (
        f"Logged the same as {day_label} — {item_summary(copy)}, about "
        f"{nutrition.calories:.0f} kcal and {nutrition.protein_g:.0f}g protein. "
        f"(Original: {_meal_label(source)}.)"
    )


def totals(day: DailyTotals, label: str) -> str:
    nutrition = day.nutrition
    if day.meal_count == 0:
        return f"Nothing logged {label.lower()} yet."
    return (
        f"{label}: {nutrition.calories:.0f} kcal, {nutrition.protein_g:.0f}g protein, "
        f"{nutrition.carbs_g:.0f}g carbs, and {nutrition.fat_g:.0f}g fat "
        f"across {day.meal_count} {_plural(day.meal_count, 'meal')}."
    )


def meal_list(meals: Sequence[MealRecord], label: str) -> str:
    if not meals:
        return f"Nothing logged {label.lower()} yet."
    described = "; ".join(f"{_meal_label(meal)}: {item_summary(meal)}" for meal in meals)
    return f"{label} you logged: {described}."


def ambiguous(candidates: Sequence[MealRecord], label: str, action: str = "correct") -> str:
    described = " or ".join(_meal_label(meal) for meal in candidates[:3])
    return f"Which {label.lower()} meal do you mean for this {action} — {described}?"


def not_found(label: str) -> str:
    return f"I couldn't find a logged meal {label.lower()} to use."


def no_reference_match(candidates: Sequence[MealRecord]) -> str:
    """Reply to a pointer that only made sense against the recent window, not today."""
    if not candidates:
        return "I couldn't find a recent meal that matches."
    described = " or ".join(_meal_label(meal) for meal in candidates[:3])
    return f"Which of your recent meals do you mean — {described}?"


def _meal_label(meal: MealRecord) -> str:
    type_label = (
        meal.meal_type.value.replace("_", " ")
        if meal.meal_type is not MealType.UNSPECIFIED
        else "meal"
    )
    return f"{type_label} at {meal.occurred_at:%H:%M}"


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else noun + "s"


def day_label(day: date, today: date) -> str:
    offset = (day - today).days
    return {0: "Today", 1: "Tomorrow", -1: "Yesterday"}.get(offset, f"{day:%d %b}")
