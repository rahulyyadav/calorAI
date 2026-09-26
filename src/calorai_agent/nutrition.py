from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from calorai_agent.domain import (
    AgentIntent,
    MealDraft,
    MealItemDraft,
    MealType,
    Nutrition,
    ParsedMessage,
)


@dataclass(frozen=True, slots=True)
class FoodReference:
    canonical_name: str
    unit: str
    nutrition: Nutrition


def _nutrition(calories: str, protein: str, carbs: str, fat: str) -> Nutrition:
    return Nutrition(
        calories=Decimal(calories),
        protein_g=Decimal(protein),
        carbs_g=Decimal(carbs),
        fat_g=Decimal(fat),
    )


class MessagePlanner(Protocol):
    def parse(self, text: str, occurred_at: datetime) -> ParsedMessage: ...


FOODS: dict[str, FoodReference] = {
    "paratha": FoodReference(
        "paratha",
        "piece",
        _nutrition("260", "6", "38", "9"),
    ),
    "roti": FoodReference(
        "roti",
        "piece",
        _nutrition("120", "4", "24", "1"),
    ),
    "chai": FoodReference(
        "milk chai",
        "cup",
        _nutrition("120", "3", "18", "4"),
    ),
    "biryani": FoodReference(
        "biryani",
        "serving",
        _nutrition("600", "22", "82", "20"),
    ),
    "rice": FoodReference(
        "cooked rice",
        "cup",
        _nutrition("205", "4", "45", "0.4"),
    ),
    "egg": FoodReference(
        "egg",
        "piece",
        _nutrition("78", "6", "0.6", "5"),
    ),
    "banana": FoodReference(
        "banana",
        "piece",
        _nutrition("105", "1.3", "27", "0.4"),
    ),
}

ALIASES = {
    "parathas": "paratha",
    "rotis": "roti",
    "chapati": "roti",
    "chapatis": "roti",
    "eggs": "egg",
    "bananas": "banana",
}

NUMBER_WORDS = {
    "a": Decimal("1"),
    "an": Decimal("1"),
    "one": Decimal("1"),
    "two": Decimal("2"),
    "three": Decimal("3"),
    "four": Decimal("4"),
    "half": Decimal("0.5"),
}


class RuleBasedPlanner:
    """Deterministic Phase 1 planner; replaceable by a structured LLM planner later."""

    _quantity_pattern = re.compile(
        r"(?:(?P<number>\d+(?:\.\d+)?)|(?P<word>a|an|one|two|three|four|half))\s+"
        r"(?P<food>parathas?|rotis?|chapatis?|eggs?|bananas?|cups?\s+of\s+rice)",
        re.IGNORECASE,
    )

    def parse(self, text: str, occurred_at: datetime) -> ParsedMessage:
        normalized = " ".join(text.lower().strip().split())
        if self._is_totals_question(normalized):
            return ParsedMessage(intent=AgentIntent.GET_TOTALS)
        if self._is_meal_list_question(normalized):
            return ParsedMessage(intent=AgentIntent.LIST_MEALS)

        items = self._extract_items(normalized)
        if not items:
            return ParsedMessage(
                intent=AgentIntent.UNKNOWN,
                explanation=(
                    "I couldn't identify a supported food yet. Try something like "
                    "'had 2 parathas and chai for breakfast'."
                ),
            )

        return ParsedMessage(
            intent=AgentIntent.LOG_MEAL,
            draft=MealDraft(
                meal_type=self._meal_type(normalized),
                occurred_at=occurred_at,
                source_text=text,
                items=tuple(items),
                notes="Nutrition values are Phase 1 reference estimates.",
            ),
        )

    @staticmethod
    def _is_totals_question(text: str) -> bool:
        return any(
            phrase in text
            for phrase in (
                "how am i doing",
                "total today",
                "totals today",
                "calories today",
                "protein today",
                "protein have i had today",
            )
        )

    @staticmethod
    def _is_meal_list_question(text: str) -> bool:
        return any(phrase in text for phrase in ("what did i eat", "show meals", "meals today"))

    def _extract_items(self, text: str) -> list[MealItemDraft]:
        items: list[MealItemDraft] = []
        consumed_foods: set[str] = set()
        for match in self._quantity_pattern.finditer(text):
            raw_food = match.group("food").lower()
            key = "rice" if "rice" in raw_food else ALIASES.get(raw_food, raw_food)
            quantity = (
                Decimal(match.group("number"))
                if match.group("number")
                else NUMBER_WORDS[match.group("word").lower()]
            )
            items.append(self._item(key, quantity, confidence=0.95))
            consumed_foods.add(key)

        for key in FOODS:
            aliases = {key, *(alias for alias, target in ALIASES.items() if target == key)}
            found = any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases)
            if key not in consumed_foods and found:
                items.append(self._item(key, Decimal("1"), confidence=0.8))
        return items

    @staticmethod
    def _item(key: str, quantity: Decimal, confidence: float) -> MealItemDraft:
        reference = FOODS[key]
        return MealItemDraft(
            name=reference.canonical_name,
            quantity=quantity,
            unit=reference.unit,
            nutrition=reference.nutrition.scaled(quantity),
            confidence=confidence,
        )

    @staticmethod
    def _meal_type(text: str) -> MealType:
        for meal_type in (MealType.BREAKFAST, MealType.LUNCH, MealType.DINNER, MealType.SNACK):
            if meal_type.value in text:
                return meal_type
        return MealType.UNSPECIFIED
