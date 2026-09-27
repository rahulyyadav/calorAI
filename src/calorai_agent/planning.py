from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Protocol
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from calorai_agent.domain import (
    AgentIntent,
    DietaryConstraint,
    InterpretationOrigin,
    MealDraft,
    MealItemDraft,
    MealRecord,
    MealReference,
    MealType,
    MediaRef,
    MemoryRecord,
    NamedRoutine,
    NutritionTarget,
    ParsedMessage,
    combine_portions,
)
from calorai_agent.memory import (
    DEFAULT_ROUTINE_SLOT,
    DIET_ALIASES,
    METRIC_WORDS,
    normalized_diet,
    routine_for,
    routine_meal_type,
    routines,
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

# "it was only rice and dal" restates the whole meal, so the correction replaces it;
# a bare restated quantity merges into the foods that meal already listed.
WHOLE_MEAL_RESTATE_WORDS = ("only", "just")

# A correction word that adds to the meal rather than swapping it out.
ADDITIVE_WORDS = ("also", "plus", "extra", "with", "too", "another", "on the side")

REPEAT_WORDS = ("same as", "same for", "same thing", "what i had", "repeat", "usual", "again")

# Cues that point at a meal already on the record, rather than at the foods being said now.
# "chicken again" logs chicken; "the same as yesterday, plus a banana" copies yesterday and
# adds the banana, because the copy is what the message is built on.
ANAPHORIC_REPEAT_WORDS = ("same as", "same for", "same thing", "what i had")

# A memory statement is its own turn: the agent records the fact and answers that, rather
# than logging food and updating a preference in the same breath.
_DIET_PHRASES = "|".join(
    re.escape(phrase) for phrase in sorted(DIET_ALIASES, key=len, reverse=True)
)
_DIET_STATEMENT = re.compile(
    r"\b(?:i(?:'?m|\s+am)|my\s+diet\s+is)\s+(?:an?\s+)?(?P<diet>" + _DIET_PHRASES + r")\b",
    re.IGNORECASE,
)

_TARGET_CUES = (
    "target",
    "goal",
    "aim",
    "trying",
    "try to",
    "shoot for",
    "plan to",
    "limit",
    "max",
    "i want",
    "i need",
)

_METRIC_NAMES = "|".join(re.escape(word) for word in sorted(METRIC_WORDS, key=len, reverse=True))

# Both orders people state a target in: "120g protein" and "protein of 120g".
# A leading minus or dot disqualifies the amount: a negative or fractional target is a typo
# or a portion, not a durable fact worth keeping.
_METRIC_THEN_AMOUNT = re.compile(
    r"\b(?P<metric>" + _METRIC_NAMES + r")\s*(?:target|goal|aim)?\s*(?:of|is|to|:)?\s*"
    r"(?<![-.\d])(?P<amount>\d+(?:\.\d+)?)\s*(?:g|grams?|gm|kcal)?\b",
    re.IGNORECASE,
)
_AMOUNT_THEN_METRIC = re.compile(
    r"(?<![-.\d])(?P<amount>\d+(?:\.\d+)?)\s*(?:g|grams?|gm)?\s*(?:of\s+)?"
    r"(?P<metric>" + _METRIC_NAMES + r")\b",
    re.IGNORECASE,
)

_ROUTINE_SAVE_CUE = re.compile(
    r"\b(?:remember|save|note|set|make|call)\b|\b(?:this|that)\s+(?:is|was)\b",
    re.IGNORECASE,
)
_USUAL_PHRASE = re.compile(
    r"\b(?:my|the)\s+(?:usual|regular|normal|daily|default|go-?to)\b",
    re.IGNORECASE,
)

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
    "target",
    "goal",
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
    memories: Sequence[MemoryRecord] = ()
    media: MediaRef | None = None


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


# Every word the reference table can price, longest first so "milk chai" beats "milk".
_FOOD_PATTERN = re.compile(
    r"\b("
    + "|".join(sorted((re.escape(n) for n in (*FOODS, *ALIASES)), key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def mentions_in(text: str) -> dict[str, _Mention]:
    """Every food mention in a message, aggregated by canonical food."""
    mentioned: dict[str, _Mention] = {}
    for match in _FOOD_PATTERN.finditer(text):
        key = ALIASES.get(match.group(1), match.group(1))
        mentioned[key] = _merged(mentioned.get(key), _quantity_around(text, match))
    return mentioned


def meal_type_hint(text: str) -> MealType | None:
    """The meal a message places itself at, whatever casing it arrived in."""
    return _type_hint(" ".join(text.lower().strip().split()))


def day_offset_hint(text: str) -> int:
    """How many days back a message points: 0 today, -1 yesterday."""
    return _day_offset(" ".join(text.lower().strip().split()))


class RuleBasedPlanner:
    """Deterministic interpreter for the supported food vocabulary.

    Deliberately replaceable: a model planner implements the same `MessagePlanner`
    protocol and returns the same typed `ParsedMessage`.
    """

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
        remembered = self._stated_memory(request, text)
        if remembered is not None:
            return remembered
        if self._is_totals_question(text):
            return ParsedMessage(intent=AgentIntent.GET_TOTALS, reference=_reference(text))
        if self._is_list_question(text):
            return ParsedMessage(intent=AgentIntent.LIST_MEALS, reference=_reference(text))
        if _contains(text, DELETE_WORDS):
            return ParsedMessage(
                intent=AgentIntent.DELETE_MEAL,
                reference=_reference(text, self._first_food(text)),
            )

        mentions = mentions_in(text)
        stated = {key: mention for key, mention in mentions.items() if not mention.denied}
        if mentions and not stated:
            # Every food in the message was refused ("no eggs"): nothing to log.
            return ParsedMessage(intent=AgentIntent.ACKNOWLEDGE, statement=_SKIP_STATEMENT)
        items = _items_from(stated, _contains(text, ESTIMATE_WORDS), text)
        if items and (
            _contains(text, REVISION_WORDS)
            or _restates_a_logged_food(text, items, request.recent_meals)
        ):
            return ParsedMessage(
                intent=AgentIntent.REVISE_MEAL,
                items=tuple(items),
                reference=_revision_reference(text, items, request.recent_meals),
                replace_items=_replaces_the_meal(text, items, request.recent_meals),
            )
        remembered = routine_log(request, text, tuple(items))
        if remembered is not None:
            return remembered
        if items and _contains(text, ANAPHORIC_REPEAT_WORDS):
            # "same as yesterday, plus a banana" copies a meal and adds to it: both halves of
            # the message count, so neither the routine nor the banana goes missing.
            return self._repeat(request, tuple(items), text)
        if items:
            return ParsedMessage(
                intent=AgentIntent.LOG_MEAL, draft=self._draft(request, text, items)
            )
        if _contains(text, REPEAT_WORDS):
            return self._repeat(request, (), text)
        return self._without_items(text)

    @staticmethod
    def _is_totals_question(text: str) -> bool:
        return _contains(text, TOTALS_PHRASES) and _FOOD_PATTERN.search(text) is None

    @staticmethod
    def _is_list_question(text: str) -> bool:
        return any(
            phrase in text
            for phrase in ("what did i eat", "show meals", "meals today", "what have i eaten")
        )

    def _stated_memory(self, request: PlannerRequest, text: str) -> ParsedMessage | None:
        """A message whose whole point is a durable fact the agent should keep.

        A message can state a fact and name a meal in one breath ("i'm vegetarian, had 2 idlis").
        The fact is what this turn keeps, so the foods left out are named back instead of dropped:
        a meal that went uncounted has to be visible in the reply.
        """
        alongside = self._stated_foods(text)
        diet = _DIET_STATEMENT.search(text)
        if diet is not None:
            return ParsedMessage(
                intent=AgentIntent.SAVE_MEMORY,
                memory=DietaryConstraint(diet=normalized_diet(diet.group("diet"))),
                unlogged=alongside,
            )
        target = _nutrition_target(text)
        if target is not None:
            return ParsedMessage(intent=AgentIntent.SAVE_MEMORY, memory=target, unlogged=alongside)
        if _ROUTINE_SAVE_CUE.search(text) and _USUAL_PHRASE.search(text):
            slot = _type_hint(text)
            return ParsedMessage(
                intent=AgentIntent.SAVE_MEMORY,
                memory=NamedRoutine(slot=slot.value if slot else DEFAULT_ROUTINE_SLOT),
                reference=_reference(text),
            )
        return None

    def _stated_foods(self, text: str) -> tuple[str, ...]:
        """The foods a message names, phrased the way the user phrased them."""
        return tuple(
            f"{mention.quantity:g} {name}"
            for name, mention in mentions_in(text).items()
            if not mention.denied
        )

    @staticmethod
    def _repeat(
        request: PlannerRequest, extras: tuple[MealItemDraft, ...], text: str
    ) -> ParsedMessage:
        # "same as yesterday for dinner" names the *target* slot for the copy, so it
        # must not filter the meal being copied.
        source_text = re.sub(rf"\bfor {_MEAL_WORD}\b", " ", text)
        return ParsedMessage(
            intent=AgentIntent.REPEAT_MEAL,
            reference=_reference(source_text),
            target_meal_type=_type_hint(text),
            items=extras,
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
        match = _FOOD_PATTERN.search(text)
        if match is None:
            return None
        return FOODS[ALIASES.get(match.group(1), match.group(1))].canonical_name


def _contains(text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text for phrase in phrases)


_AND_A_HALF = re.compile(
    r"\b("
    + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    + r"|\d+(?:\.\d+)?)\s+and\s+a\s+half\b",
    re.IGNORECASE,
)

# A correction does not have to say so: "that was 3 rotis" opens by restating something
# already on the record. Only treated as a revision when there is in fact such a meal.
_ANAPHORIC_RESTATE = re.compile(
    r"^(?:that|it|this|those|these)\s+(?:was|were|is|are)\b",
    re.IGNORECASE,
)


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


def _restates_a_logged_food(
    text: str, items: Sequence[MealItemDraft], recent_meals: Sequence[MealRecord]
) -> bool:
    """True for "that was 3 rotis": an anaphoric subject naming an already-logged food."""
    if _ANAPHORIC_RESTATE.match(text) is None or not recent_meals:
        return False
    logged = {item.name for meal in recent_meals for item in meal.items}
    return any(item.name in logged for item in items)


def _revision_reference(
    text: str, items: Sequence[MealItemDraft], recent_meals: Sequence[MealRecord]
) -> MealReference:
    reference = _reference(text, items[0].name)
    if (
        reference.food_hint
        and recent_meals
        and not any(_meal_lists(meal, reference.food_hint) for meal in recent_meals)
    ):
        # The correction renames the food, so the new food cannot identify the meal to
        # change. Drop the hint and let the day/meal-type pointer resolve, or ask.
        return reference.model_copy(update={"food_hint": None})
    return reference


def _meal_lists(meal: MealRecord, food_name: str) -> bool:
    hint = food_name.lower()
    return any(hint in item.name.lower() for item in meal.items)


def _nutrition_target(text: str) -> NutritionTarget | None:
    """A stated daily target, in either "120g protein" or "protein of 120g" order.

    Only an explicit aim counts: a plain "30g protein" inside a meal description is a
    portion being logged, not a goal being set.
    """
    if not _contains(text, _TARGET_CUES):
        return None
    for pattern in (_AMOUNT_THEN_METRIC, _METRIC_THEN_AMOUNT):
        match = pattern.search(text)
        if match is None:
            continue
        try:
            return NutritionTarget(
                metric=METRIC_WORDS[match.group("metric").lower()],
                value=Decimal(match.group("amount")),
            )
        except ValidationError:
            return None
    return None


def routine_log(
    request: PlannerRequest, text: str, extras: tuple[MealItemDraft, ...] = ()
) -> ParsedMessage | None:
    """`my usual`, as the routine the user actually saved.

    None when the message is not about a routine at all, so the caller keeps its own reading of
    it. A stated extra folds into the routine instead of replacing it, because "my usual, plus a
    banana" is one meal and silently logging only the banana would understate the day.
    """
    if _USUAL_PHRASE.search(text) is None:
        return None
    meal_type = _type_hint(text)
    routine = routine_for(request.memories, meal_type)
    if routine is None:
        return ParsedMessage(
            intent=AgentIntent.CLARIFY,
            question=_no_routine_question(request.memories, meal_type),
        )
    return ParsedMessage(
        intent=AgentIntent.LOG_MEAL, draft=_routine_draft(request, routine, extras)
    )


def _routine_draft(
    request: PlannerRequest, routine: NamedRoutine, extras: tuple[MealItemDraft, ...] = ()
) -> MealDraft:
    meal_type = routine_meal_type(routine)
    items = combine_portions(routine.items, extras) if extras else routine.items
    notes = f"Copied from your saved routine: {routine.slot}."
    if extras:
        added = ", ".join(f"{item.quantity:g} {item.name}" for item in extras)
        notes = f"{notes} You added {added}."
    return MealDraft(
        meal_type=meal_type,
        occurred_at=local_time(request, 0, meal_type),
        source_text=request.text,
        items=items,
        notes=notes,
        origin=InterpretationOrigin.USER_CONFIRMED,
    )


def _no_routine_question(memories: Sequence[MemoryRecord], meal_type: MealType | None) -> str:
    """Ask once to establish a routine, instead of inventing what 'usual' means."""
    saved = routines(memories)
    slot = (
        meal_type.value if meal_type is not None and meal_type is not MealType.UNSPECIFIED else None
    )
    if not saved:
        named = f" {slot}" if slot else ""
        return (
            f"What is your usual{named}? Log it once and say "
            "'remember this as my usual' and I will reuse it later."
        )
    if slot is None:
        saved_slots = ", ".join(routine.slot for routine in saved)
        return f"Which one is your usual today — {saved_slots}?"
    return (
        f"I don't have your usual {slot} saved yet. Log a {slot} and say "
        f"'remember this as my usual {slot}' and I will reuse it later."
    )


def _replaces_the_meal(
    text: str, items: Sequence[MealItemDraft], recent_meals: Sequence[MealRecord]
) -> bool:
    """True when the correction restates the whole meal instead of amending one food.

    "only/just" says it outright, and so does naming a food none of the logged meals
    contain: the portion being corrected has to be the one that is being replaced.
    """
    if _contains(text, ADDITIVE_WORDS):
        return False
    if _contains(text, WHOLE_MEAL_RESTATE_WORDS):
        return True
    logged = {item.name for meal in recent_meals for item in meal.items}
    return bool(logged) and all(item.name not in logged for item in items)


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
