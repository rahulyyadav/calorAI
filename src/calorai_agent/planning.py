from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Protocol
from zoneinfo import ZoneInfo

from calorai_agent.domain import (
    AgentIntent,
    MealDraft,
    MealItemDraft,
    MealRecord,
    MealReference,
    MealType,
    ParsedMessage,
)
from calorai_agent.nutrition import ALIASES, FOODS
from calorai_agent.policy import MAX_PLAUSIBLE_QUANTITY, UNUSABLE_QUANTITY_CONFIDENCE

MEAL_TIMES: dict[MealType, time] = {
    MealType.BREAKFAST: time(8, 0),
    MealType.LUNCH: time(13, 30),
    MealType.DINNER: time(20, 30),
    MealType.SNACK: time(16, 30),
}

NUMBER_WORDS: dict[str, Decimal] = {
    "a": Decimal(1),
    "an": Decimal(1),
    "one": Decimal(1),
    "two": Decimal(2),
    "three": Decimal(3),
    "four": Decimal(4),
    "five": Decimal(5),
    "six": Decimal(6),
    "seven": Decimal(7),
    "eight": Decimal(8),
    "nine": Decimal(9),
    "ten": Decimal(10),
}

DENOMINATOR_WORDS: dict[str, Decimal] = {
    "half": Decimal(2),
    "halves": Decimal(2),
    "third": Decimal(3),
    "thirds": Decimal(3),
    "quarter": Decimal(4),
    "quarters": Decimal(4),
    "fourth": Decimal(4),
    "fourths": Decimal(4),
    "fifth": Decimal(5),
    "fifths": Decimal(5),
}

# Words that may sit between a quantity and the food it quantifies.
FILLER_WORDS = frozenset(
    {
        "of",
        "the",
        "a",
        "an",
        "some",
        "leftover",
        "leftovers",
        "piece",
        "pieces",
        "slice",
        "slices",
        "bowl",
        "bowls",
        "cup",
        "cups",
        "plate",
        "plates",
        "serving",
        "servings",
        "box",
        "packet",
        "container",
        "remainder",
        "rest",
        "and",
        "with",
        "had",
        "ate",
        "just",
        "only",
        "maybe",
        "about",
        "around",
        "roughly",
    }
)

ESTIMATE_WORDS = (
    "maybe",
    "about",
    "around",
    "roughly",
    "approximately",
    "i think",
    "not sure",
    "leftover",
    "probably",
    "guess",
    "share",
    "shared",
)

VAGUE_EATING_WORDS = (
    "grazed",
    "grazing",
    "picked at",
    "nibbled",
    "nibbling",
    "munched",
    "munching",
    "bits and pieces",
    "small bites",
    "nothing much",
)

SKIP_WORDS = (
    "skipped",
    "skip",
    "didn't eat",
    "did not eat",
    "didn't have",
    "did not have",
    "don't have",
    "do not have",
    "haven't eaten",
    "fasted",
    "fasting",
    "no breakfast",
    "no lunch",
    "no dinner",
)

DELETE_WORDS = ("delete", "remove", "undo", "forget", "cancel", "scrap", "discard")

# A revision has to say it is one. Bare contrast ("dosa for lunch not dinner") is not a
# correction cue, or every plain log with a "not" in it would rewrite history.
REVISION_WORDS = (
    "actually",
    "i meant",
    "correction",
    "make it",
    "change it",
    "instead",
    "oops",
    "typo",
    "wasn't",
    "weren't",
    "wrong",
    "update",
)

REPEAT_WORDS = ("same as", "same for", "same thing", "what i had", "repeat", "usual", "again")

TOTALS_PHRASES = (
    "how am i doing",
    "totals",
    "total",
    "how many calories",
    "how much protein",
    "how much did i eat",
    "calories today",
    "protein today",
    "macros",
)

# Amount words deliberately left unresolved, so the agent asks instead of inventing a number.
UNSUPPORTED_AMOUNT_WORDS = (
    "eleven",
    "twelve",
    "dozen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
)

# "no eggs" / "not any rice" reports eating nothing, not a portion of something.
DENIAL_WORDS = ("no", "zero", "none", "not any")

# The same refusal in verb form: "I didn't have rotis" rejects the food it names.
_NEGATED_EATING = re.compile(
    r"\b(?:don'?t|didn'?t|did\s+not|do\s+not|haven'?t|have\s+not|had\s+not|hasn'?t"
    r"|\bnot\b|never)"
    r"(?:\s+(?:really\s+)?(?:have|had|eat|eaten|any|of\s+any))*\s*$",
    re.IGNORECASE,
)

# A message names the day outright ("yesterday", "today") instead of pointing at the meal
# the user last talked about.
DAY_WORDS = ("today", "tonight", "yesterday", "day before", "last night")

_MEAL_WORD = r"(?:breakfast|lunch|dinner|supper|tea break)"

# "how many calories are in a pizza" asks about a food, not about the day's log.
_FOOD_QUESTION = re.compile(r"\bhow (?:many|much)\b[^?]*\b(?:is|are)\s+in\b", re.IGNORECASE)

_SKIP_STATEMENT = "Noted — I have not logged a meal for that."

# Confidence tiers for how a portion was stated. The policy's ask band is what an
# unusable amount drops into, so a nonsensical explicit amount becomes a question.
EXPLICIT_CONFIDENCE = 0.95
BARE_CONFIDENCE = 0.8
HEDGED_EXPLICIT_CONFIDENCE = 0.75
HEDGED_BARE_CONFIDENCE = 0.6

_LEADING_QTY = re.compile(
    r"\b(?P<number>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|a|an|half)"
    r"(?:\s*/\s*(?P<slash>\d+(?:\.\d+)?))?"
    r"(?:\s+(?P<denominator>halves|thirds?|quarters?|fourths?|fifths?))?\s*$",
    re.IGNORECASE,
)

# Fractions that trail the food name: "biryani, maybe two thirds of the box".
_TRAILING_QTY = re.compile(
    r"^[,\s]*(?:maybe|about|around|roughly|like|probably|i think)?[,\s]*"
    r"(?P<number>\d+(?:\.\d+)?|one|two|three|four|five|half)"
    r"(?:\s*/\s*(?P<slash>\d+(?:\.\d+)?))?"
    r"(?:\s+(?P<denominator>halves|thirds?|quarters?|fourths?|fifths?))?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PlannerRequest:
    """Everything a planner needs to interpret one inbound message."""

    text: str
    occurred_at: datetime
    timezone: str = "UTC"
    recent_meals: Sequence[MealRecord] = ()


@dataclass(frozen=True, slots=True)
class _Mention:
    """The portion one food mention states.

    `quantity` is always positive so no unusable parse can reach a persisted row: a
    quantity that cannot be trusted keeps a placeholder amount and drops into the
    policy's ask band instead.
    """

    quantity: Decimal
    explicit: bool
    usable: bool = True
    denied: bool = False


_UNUSABLE_MENTION = _Mention(Decimal(1), explicit=True, usable=False)
_DENIED_MENTION = _Mention(Decimal(1), explicit=True, usable=False, denied=True)

# Every amount the message spells out, used to spot a number no food claimed.
_STATED_NUMBER = re.compile(
    r"\b(?:\d+(?:[./]\d+)?|" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE
)


class MessagePlanner(Protocol):
    def parse(self, request: PlannerRequest) -> ParsedMessage: ...


class RuleBasedPlanner:
    """Deterministic interpreter for the supported food vocabulary.

    Deliberately replaceable: a model planner implements the same `MessagePlanner`
    protocol and returns the same typed `ParsedMessage`.
    """

    _food_pattern = re.compile(
        r"\b("
        + "|".join(sorted((re.escape(n) for n in (*FOODS, *ALIASES)), key=len, reverse=True))
        + r")\b",
        re.IGNORECASE,
    )

    def parse(self, request: PlannerRequest) -> ParsedMessage:
        text = _normalize_amounts(" ".join(request.text.lower().strip().split()))
        if _is_food_nutrition_question(text):
            return ParsedMessage(
                intent=AgentIntent.UNKNOWN,
                explanation=(
                    "I can total the meals you have logged, but I do not look up "
                    "nutrition for an arbitrary food yet. Try 'how many calories have "
                    "i eaten today'."
                ),
            )
        if self._is_totals_question(text):
            return ParsedMessage(intent=AgentIntent.GET_TOTALS, reference=_reference(text))
        if self._is_list_question(text):
            return ParsedMessage(intent=AgentIntent.LIST_MEALS, reference=_reference(text))
        if _contains(text, DELETE_WORDS):
            return ParsedMessage(
                intent=AgentIntent.DELETE_MEAL,
                reference=_reference(text, self._first_food(text)),
            )

        mentions = self._mentions(text)
        stated = {key: mention for key, mention in mentions.items() if not mention.denied}
        if mentions and not stated:
            # Every food in the message was refused ("no eggs"): nothing to log.
            return ParsedMessage(intent=AgentIntent.ACKNOWLEDGE, statement=_SKIP_STATEMENT)
        items = _items_from(stated, _contains(text, ESTIMATE_WORDS), text)
        if items and _contains(text, REVISION_WORDS):
            return ParsedMessage(
                intent=AgentIntent.REVISE_MEAL,
                items=tuple(items),
                reference=_reference(text, items[0].name),
            )
        if items:
            return ParsedMessage(
                intent=AgentIntent.LOG_MEAL, draft=self._draft(request, text, items)
            )
        if _contains(text, REPEAT_WORDS):
            return self._repeat(text)
        return self._without_items(text)

    @staticmethod
    def _is_totals_question(text: str) -> bool:
        return _contains(text, TOTALS_PHRASES) and not RuleBasedPlanner._food_pattern.search(text)

    @staticmethod
    def _is_list_question(text: str) -> bool:
        return any(
            phrase in text
            for phrase in ("what did i eat", "show meals", "meals today", "what have i eaten")
        )

    @staticmethod
    def _repeat(text: str) -> ParsedMessage:
        if "usual" in text:
            return ParsedMessage(
                intent=AgentIntent.CLARIFY,
                question=(
                    "What is your usual? Log it once and say 'remember this as my usual' "
                    "and I will reuse it later."
                ),
            )
        # "same as yesterday for dinner" names the *target* slot for the copy, so it
        # must not filter the meal being copied.
        source_text = re.sub(rf"\bfor {_MEAL_WORD}\b", " ", text)
        return ParsedMessage(
            intent=AgentIntent.REPEAT_MEAL,
            reference=_reference(source_text),
            target_meal_type=_type_hint(text),
        )

    @staticmethod
    def _without_items(text: str) -> ParsedMessage:
        if _contains(text, VAGUE_EATING_WORDS):
            return ParsedMessage(
                intent=AgentIntent.CLARIFY,
                question=(
                    "I have not logged anything yet. Roughly what did you graze on, "
                    "and how much of it?"
                ),
            )
        if _contains(text, SKIP_WORDS):
            return ParsedMessage(
                intent=AgentIntent.ACKNOWLEDGE,
                statement="Noted — I have not logged a meal for that.",
            )
        return ParsedMessage(
            intent=AgentIntent.UNKNOWN,
            explanation=(
                "I couldn't identify a supported food yet. Try something like "
                "'had 2 parathas and chai for breakfast'."
            ),
        )

    def _mentions(self, text: str) -> dict[str, _Mention]:
        """Every food mention in the message, aggregated by canonical food."""
        mentioned: dict[str, _Mention] = {}
        for match in self._food_pattern.finditer(text):
            key = ALIASES.get(match.group(1), match.group(1))
            mentioned[key] = _merged(mentioned.get(key), _quantity_around(text, match))
        return mentioned

    def _draft(self, request: PlannerRequest, text: str, items: list[MealItemDraft]) -> MealDraft:
        meal_type = _meal_type(text)
        return MealDraft(
            meal_type=meal_type,
            occurred_at=local_time(request, _day_offset(text), meal_type),
            source_text=request.text,
            items=tuple(items),
            notes="Nutrition values are reference estimates.",
        )

    @staticmethod
    def _first_food(text: str) -> str | None:
        match = RuleBasedPlanner._food_pattern.search(text)
        if match is None:
            return None
        return FOODS[ALIASES.get(match.group(1), match.group(1))].canonical_name


def _contains(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


_AND_A_HALF = re.compile(r"\b(one|a|an|\d+(?:\.\d+)?)\s+and\s+a\s+half\b", re.IGNORECASE)


def _normalize_amounts(text: str) -> str:
    """Fold "two and a half rotis" into "2.5 rotis" so one amount governs the mention."""
    return _AND_A_HALF.sub(lambda m: f"{_number_value(m.group(1)) or m.group(1)}.5", text)


def _is_food_nutrition_question(text: str) -> bool:
    """True for "how many calories are in a pizza": a question about food, not a log."""
    return _contains(text, TOTALS_PHRASES) and _FOOD_QUESTION.search(text) is not None


def _meal_type(text: str) -> MealType:
    return _type_hint(text) or MealType.UNSPECIFIED


def _type_hint(text: str) -> MealType | None:
    if any(word in text for word in ("breakfast", "morning")):
        return MealType.BREAKFAST
    if any(word in text for word in ("lunch", "noon", "midday")):
        return MealType.LUNCH
    if any(word in text for word in ("dinner", "evening", "night", "supper")):
        return MealType.DINNER
    if any(word in text for word in ("snack", "tea break")):
        return MealType.SNACK
    return None


def _day_offset(text: str) -> int:
    if "day before yesterday" in text:
        return -2
    if any(word in text for word in ("yesterday", "last night", "day before")):
        return -1
    return 0


def _reference(text: str, food_hint: str | None = None) -> MealReference:
    return MealReference(
        day_offset=_day_offset(text),
        day_explicit=_contains(text, DAY_WORDS),
        meal_type=_type_hint(text),
        food_hint=food_hint,
    )


def local_time(request: PlannerRequest, day_offset: int, meal_type: MealType) -> datetime:
    """Resolve a message's wall-clock time in the user's timezone."""
    zone = ZoneInfo(request.timezone)
    local = request.occurred_at.astimezone(zone) + timedelta(days=day_offset)
    if meal_type is MealType.UNSPECIFIED:
        return local
    hint = MEAL_TIMES[meal_type]
    return local.replace(hour=hint.hour, minute=hint.minute)


def _quantity_around(text: str, match: re.Match[str]) -> _Mention:
    """Find the quantity governing a food mention, before it or trailing after it."""
    before = _quantity_before(text[: match.start()])
    if before is not None:
        return before
    after = _quantity_after(text[match.end() :])
    if after is not None:
        return after
    return _Mention(Decimal(1), explicit=False)


def _quantity_before(prefix: str) -> _Mention | None:
    """The amount stated just before a food, if one is adjacent to it."""
    tokens = [token for token in re.split(r"[\s,]+", prefix.strip()) if token]
    while tokens and tokens[-1] in FILLER_WORDS:
        tokens.pop()
    if not tokens:
        return None
    if tokens[-1] in DENIAL_WORDS or " ".join(tokens[-2:]) in DENIAL_WORDS:
        return _DENIED_MENTION
    if _NEGATED_EATING.search(" ".join(tokens[-4:])) is not None:
        return _DENIED_MENTION
    if tokens[-1] in UNSUPPORTED_AMOUNT_WORDS or tokens[-1].startswith(("-", "+")):
        return _UNUSABLE_MENTION
    return _parse_quantity(" ".join(tokens[-3:]))


def _quantity_after(suffix: str) -> _Mention | None:
    head = " ".join(re.split(r"[\s,]+", suffix.strip())[:4])
    match = _TRAILING_QTY.match(head)
    if match is None:
        return None
    # A trailing number only counts as this food's quantity when it is a fraction
    # of a container ("two thirds of the box"), not a new item's count.
    if match.group("denominator") is None and match.group("slash") is None:
        return None
    return _from_match(match)


def _parse_quantity(tail: str) -> _Mention | None:
    if not tail:
        return None
    match = _LEADING_QTY.search(tail)
    if match is None:
        return None
    return _from_match(match)


def _from_match(match: re.Match[str]) -> _Mention:
    number_raw = match.group("number")
    if number_raw is None:
        return _unusable(Decimal(1))

    normalized = number_raw.lower()
    denominator: Decimal | None = None
    if match.group("denominator"):
        denominator = DENOMINATOR_WORDS[match.group("denominator").lower()]
    elif match.group("slash"):
        denominator = Decimal(match.group("slash"))

    number = NUMBER_WORDS.get(normalized)
    if number is None and normalized in DENOMINATOR_WORDS:
        number = Decimal(1)
        denominator = denominator or DENOMINATOR_WORDS[normalized]
    if number is None:
        number = Decimal(normalized)

    if denominator is None:
        return _checked(number)
    if denominator <= 0:
        return _unusable(number)
    return _checked(number / denominator)


def _checked(quantity: Decimal) -> _Mention:
    """Keep an implausible stated amount out of the totals by asking about it."""
    if quantity <= 0 or quantity > MAX_PLAUSIBLE_QUANTITY:
        return _unusable(quantity)
    return _Mention(quantity, explicit=True)


def _unusable(stated: Decimal) -> _Mention:
    placeholder = stated if 0 < stated <= MAX_PLAUSIBLE_QUANTITY else Decimal(1)
    return _Mention(placeholder, explicit=True, usable=False)


def _merged(previous: _Mention | None, mention: _Mention) -> _Mention:
    if previous is None:
        return mention
    explicit = previous.explicit or mention.explicit
    return _Mention(
        quantity=previous.quantity + mention.quantity if explicit else Decimal(1),
        explicit=previous.explicit and mention.explicit,
        usable=previous.usable and mention.usable,
        denied=previous.denied and mention.denied,
    )


def _items_from(mentions: dict[str, _Mention], hedged: bool, text: str) -> list[MealItemDraft]:
    """Turn parsed mentions into reference-backed rows.

    A spelled-out amount that no food could claim means the portions were paired up wrong,
    so anything the message did not state outright drops into the policy's ask band.
    """
    orphan_amount = _has_unclaimed_amount(text, mentions)
    items: list[MealItemDraft] = []
    for key, mention in mentions.items():
        confidence = _confidence(mention, hedged)
        if orphan_amount and not mention.explicit:
            confidence = min(confidence, UNUSABLE_QUANTITY_CONFIDENCE)
        food = FOODS[key]
        items.append(
            MealItemDraft(
                name=food.canonical_name,
                quantity=mention.quantity,
                unit=food.unit,
                nutrition=food.nutrition.scaled(mention.quantity),
                confidence=confidence,
            )
        )
    return items


def _has_unclaimed_amount(text: str, mentions: dict[str, _Mention]) -> bool:
    claimed = [mention.quantity for mention in mentions.values() if mention.usable]
    for match in _STATED_NUMBER.finditer(text):
        value = _number_value(match.group(0))
        if value is None:
            continue
        if value in claimed:
            claimed.remove(value)
        else:
            return True
    return False


def _number_value(raw: str) -> Decimal | None:
    word = NUMBER_WORDS.get(raw.lower())
    if word is not None:
        return word
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def _confidence(mention: _Mention, hedged: bool) -> float:
    """Map how a portion was stated onto the ambiguity policy's confidence bands."""
    if not mention.usable:
        return UNUSABLE_QUANTITY_CONFIDENCE
    if mention.explicit:
        return HEDGED_EXPLICIT_CONFIDENCE if hedged else EXPLICIT_CONFIDENCE
    return HEDGED_BARE_CONFIDENCE if hedged else BARE_CONFIDENCE
