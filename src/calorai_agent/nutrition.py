from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from functools import cache

from calorai_agent.domain import Nutrition


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


FOODS: dict[str, FoodReference] = {
    "paratha": FoodReference("paratha", "piece", _nutrition("260", "6", "38", "9")),
    "roti": FoodReference("roti", "piece", _nutrition("120", "4", "24", "1")),
    "chai": FoodReference("milk chai", "cup", _nutrition("120", "3", "18", "4")),
    "coffee": FoodReference("coffee", "cup", _nutrition("5", "0.3", "0", "0")),
    "biryani": FoodReference("biryani", "serving", _nutrition("600", "22", "82", "20")),
    "rice": FoodReference("cooked rice", "cup", _nutrition("205", "4", "45", "0.4")),
    "egg": FoodReference("egg", "piece", _nutrition("78", "6", "0.6", "5")),
    "banana": FoodReference("banana", "piece", _nutrition("105", "1.3", "27", "0.4")),
    "apple": FoodReference("apple", "piece", _nutrition("95", "0.5", "25", "0.3")),
    "idli": FoodReference("idli", "piece", _nutrition("39", "2", "8", "0.3")),
    "dosa": FoodReference("dosa", "piece", _nutrition("168", "4", "24", "6")),
    "poha": FoodReference("poha", "bowl", _nutrition("250", "5", "46", "5")),
    "dal": FoodReference("dal", "bowl", _nutrition("180", "10", "24", "4")),
    "paneer": FoodReference("paneer", "serving", _nutrition("260", "14", "6", "20")),
    "chicken": FoodReference("chicken", "serving", _nutrition("250", "31", "0", "13")),
    "fish": FoodReference("fish", "serving", _nutrition("200", "26", "0", "9")),
    "curd": FoodReference("curd", "bowl", _nutrition("120", "8", "10", "5")),
    "yogurt": FoodReference("yogurt", "cup", _nutrition("120", "8", "10", "5")),
    "milk": FoodReference("milk", "glass", _nutrition("150", "8", "12", "8")),
    "oats": FoodReference("oats", "bowl", _nutrition("160", "6", "27", "3")),
    "bread": FoodReference("bread", "slice", _nutrition("80", "3", "14", "1")),
    "sandwich": FoodReference("sandwich", "piece", _nutrition("300", "12", "30", "14")),
    "pizza": FoodReference("pizza", "slice", _nutrition("285", "12", "36", "10")),
    "burger": FoodReference("burger", "piece", _nutrition("350", "15", "30", "18")),
    "salad": FoodReference("salad", "bowl", _nutrition("80", "3", "12", "3")),
    "soup": FoodReference("soup", "bowl", _nutrition("120", "5", "15", "4")),
    "nuts": FoodReference("mixed nuts", "handful", _nutrition("170", "5", "7", "15")),
    "almond": FoodReference("almonds", "handful", _nutrition("160", "6", "6", "14")),
    "chocolate": FoodReference("chocolate", "bar", _nutrition("230", "3", "26", "13")),
    "biscuits": FoodReference("biscuits", "piece", _nutrition("50", "1", "8", "2")),
    "namkeen": FoodReference("namkeen", "handful", _nutrition("160", "3", "20", "8")),
    "smoothie": FoodReference("smoothie", "glass", _nutrition("220", "8", "40", "3")),
}

ALIASES: dict[str, str] = {
    "parathas": "paratha",
    "rotis": "roti",
    "chapati": "roti",
    "chapatis": "roti",
    "phulka": "roti",
    "phulkas": "roti",
    "eggs": "egg",
    "bananas": "banana",
    "apples": "apple",
    "idlis": "idli",
    "dosas": "dosa",
    "dals": "dal",
    "pizzas": "pizza",
    "burgers": "burger",
    "sandwiches": "sandwich",
    "salads": "salad",
    "soups": "soup",
    "nuts": "nuts",
    "almonds": "almond",
    "biscuit": "biscuits",
    "yoghurt": "yogurt",
    "curd-yogurt": "curd",
    "tea": "chai",
    "coffees": "coffee",
    "omelette": "egg",
    "omelet": "egg",
}


@cache
def _resolve(cleaned: str) -> FoodReference | None:
    """The table walk for an already-normalized name. Cached because it is pure and hot.

    Every food in every message, vision answer and alias scan lands here, and the answer cannot
    change while the process runs — a stale cache is not a thing this function can produce.
    """
    if cleaned in FOODS:
        return FOODS[cleaned]
    aliased = ALIASES.get(cleaned)
    if aliased:
        return FOODS[aliased]
    for key, candidate in FOODS.items():
        if cleaned in (candidate.canonical_name.lower(), key):
            return candidate
    singular = cleaned[:-1] if cleaned.endswith("s") and not cleaned.endswith("ss") else cleaned
    return FOODS.get(singular) or FOODS.get(ALIASES.get(singular, ""))


def lookup(name: str) -> FoodReference | None:
    """Resolve a free-text or pluralized food name to its reference row."""
    return _resolve(" ".join(name.lower().strip().split()))
