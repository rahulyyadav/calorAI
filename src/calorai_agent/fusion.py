"""Phase 4: one photo and the words around it become exactly one meal.

The vision model only describes the plate. The caption is read by the same deterministic planner
that handles a typed message, then applied as *modifiers* to the photographed meal — never as a
second meal of its own. Nutrition is priced here from the reference table, so no model number
reaches the database.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal

from calorai_agent import responses
from calorai_agent.domain import (
    AgentIntent,
    DietaryConstraint,
    FoodObservation,
    InterpretationOrigin,
    MealDraft,
    MealItemDraft,
    MealType,
    MemoryContent,
    NamedRoutine,
    NutritionTarget,
    ParsedMessage,
)
from calorai_agent.memory import describe
from calorai_agent.nutrition import lookup
from calorai_agent.planning import (
    PlannerRequest,
    RuleBasedPlanner,
    day_offset_hint,
    local_time,
    meal_ask_hint,
    meal_type_hint,
)
from calorai_agent.policy import AmbiguityPolicy
from calorai_agent.vision import VisionReading

# "half of this", "a quarter of the plate", "the whole thing": a share of the photographed plate,
# which no per-food amount in the caption could express. The object is required so that "half a
# dosa" stays a portion of one food instead of scaling the entire photo a second time.
_PLATE_SHARE = re.compile(
    r"\b(?P<fraction>half|three ?quarters?|a quarter|one quarter|the whole|the entire)"
    r"\s+(?:of\s+)?(?P<object>this|that|it|the plate|the portion|the dish|the food|the box)\b",
    re.IGNORECASE,
)
_SHARE_VALUES: dict[str, Decimal] = {
    "half": Decimal("0.5"),
    "three quarters": Decimal("0.75"),
    "three quarter": Decimal("0.75"),
    "a quarter": Decimal("0.25"),
    "one quarter": Decimal("0.25"),
    "the whole": Decimal(1),
    "the entire": Decimal(1),
}
_SHARE_WORDS = {
    Decimal("0.25"): "a quarter",
    Decimal("0.5"): "half",
    Decimal("0.75"): "three quarters",
}
_USUAL_REQUEST = re.compile(r"\bmy usual\b")

# What a caption may ask for that a photograph cannot answer. The plate still gets logged, but
# the request needs a message of its own — silently dropping it would leave the user believing
# their meal had been corrected or deleted.
_NON_PLATE_ASKS: dict[AgentIntent, str] = {
    AgentIntent.REVISE_MEAL: "correction",
    AgentIntent.DELETE_MEAL: "request to remove a meal",
    AgentIntent.REPEAT_MEAL: "request to log a saved meal again",
    AgentIntent.GET_TOTALS: "totals question",
    AgentIntent.LIST_MEALS: "question about what you have already eaten",
}


@dataclass(frozen=True, slots=True)
class CaptionReading:
    """What the words beside a photo say about the photographed plate."""

    scale: Decimal = Decimal(1)
    meal_type: MealType = MealType.UNSPECIFIED
    day_offset: int = 0
    stated: tuple[MealItemDraft, ...] = ()
    memory: MemoryContent | None = None
    routine_asked: bool = False
    unrecognized: tuple[str, ...] = ()
    unlogged: tuple[str, ...] = ()
    other_ask: str | None = None


def read_caption(request: PlannerRequest) -> CaptionReading:
    """Interpret a caption as meal context only: no history is consulted, nothing is resolved.

    The caption always goes through the deterministic planner, even when a text model is
    configured for typed messages — a model free to invent a meal from five words is exactly the
    second-meal bug this phase exists to prevent.

    Recent meals and memories are withheld on purpose. A caption may rename, resize, or add to the
    plate in the photo, but it may not point at yesterday's lunch or replay a saved routine —
    either would quietly produce a second meal for one message.
    """
    stripped = " ".join(request.text.split())
    if not stripped:
        return CaptionReading()
    parsed = RuleBasedPlanner().parse(replace(request, text=stripped, recent_meals=(), memories=()))
    memory = parsed.memory
    if not isinstance(memory, (DietaryConstraint, NutritionTarget)):
        memory = None
    routine_asked = (
        isinstance(parsed.memory, NamedRoutine)
        or _USUAL_REQUEST.search(stripped.lower()) is not None
    )
    other_ask = _ask_about_other_meal(parsed, stripped)
    if other_ask is not None:
        # These words are about a different meal, so none of their portions, slots or dates may be
        # read onto this plate — a photo sent at noon must not land on yesterday because the
        # caption wanted yesterday's lunch deleted. Only a durable fact stays with the user.
        return CaptionReading(
            memory=memory,
            routine_asked=routine_asked,
            unrecognized=parsed.unrecognized,
            unlogged=parsed.unlogged,
            other_ask=other_ask,
        )
    stated = parsed.items or (parsed.draft.items if parsed.draft is not None else ())
    return CaptionReading(
        scale=plate_share(stripped),
        meal_type=meal_type_hint(stripped) or parsed.target_meal_type or MealType.UNSPECIFIED,
        # A day named in a caption moves the plate only when the caption says what was eaten that
        # day. "this looks better than yesterday's lunch" names another meal, not this plate, and
        # a photo taken now is today's food whatever it is being compared with.
        day_offset=day_offset_hint(stripped) if stated else 0,
        stated=stated,
        memory=memory,
        routine_asked=routine_asked,
        unrecognized=parsed.unrecognized,
        unlogged=parsed.unlogged,
    )


def _ask_about_other_meal(parsed: ParsedMessage, text: str) -> str | None:
    """Name the request a caption makes about a meal other than the photographed plate.

    The planner answers a durable fact before it looks for a request, so "i'm vegetarian, delete
    yesterday's lunch" arrives as a memory and the delete would vanish unheard. The words are
    scanned as well as the intent the planner settled on.
    """
    for intent in (parsed.intent, meal_ask_hint(text)):
        if intent is not None and intent in _NON_PLATE_ASKS:
            return _NON_PLATE_ASKS[intent]
    return None


def plate_share(text: str) -> Decimal:
    """The fraction of the photographed plate the user claims, when the caption says so."""
    for match in _PLATE_SHARE.finditer(text):
        value = _SHARE_VALUES.get(match.group("fraction").lower())
        if value is not None:
            return value
    return Decimal(1)


def fuse(request: PlannerRequest, reading: VisionReading, policy: AmbiguityPolicy) -> ParsedMessage:
    """Fold photo and caption into one meal proposal, or one focused question."""
    caption = read_caption(request)
    photo = tuple(
        item
        for item in (_priced(observation) for observation in reading.observations)
        if item is not None
    )
    # The share applies to what the photo shows, never to an amount the caption stated outright:
    # "half of this, plus a banana" is a half plate and a whole banana.
    if caption.scale != 1:
        photo = tuple(_scale(item, caption.scale) for item in photo)
    items = _overlay(photo, caption.stated)

    question = _uncertain_question(items, reading, caption, policy)
    if question is None and not items:
        question = _nothing_to_count(reading)
    if question is not None:
        # Nothing can be logged this turn. A fact the caption stated is nothing the photo has to
        # confirm first, so it is kept anyway — dropping it would make the user say it twice.
        if caption.memory is not None:
            return _keep_fact(caption.memory, question, caption.unlogged)
        return ParsedMessage(intent=AgentIntent.CLARIFY, question=question)

    unrecognized = _unrecognized(reading, caption, items)
    return ParsedMessage(
        intent=AgentIntent.LOG_MEAL,
        draft=MealDraft(
            meal_type=caption.meal_type,
            occurred_at=local_time(request, caption.day_offset, caption.meal_type),
            source_text=_source_text(request),
            items=items,
            notes=_audit_note(request, reading, caption),
            origin=InterpretationOrigin.VISION_FUSION,
            model=reading.model,
        ),
        memory=caption.memory,
        unrecognized=unrecognized,
        unlogged=caption.unlogged,
        reply_notes=_reply_notes(caption, photo, reading, plain=not _has_caption(request)),
    )


def _keep_fact(memory: MemoryContent, question: str, unlogged: tuple[str, ...]) -> ParsedMessage:
    """Keep what the caption stated even when the photographed plate cannot become a meal.

    The turn cannot log anything, so it saves the fact and then asks the photo's own question —
    the user's dinner stays uncounted only until they answer, and the fact survives the photo.
    """
    return ParsedMessage(
        intent=AgentIntent.SAVE_MEMORY,
        memory=memory,
        question=question,
        unlogged=unlogged,
    )


def _priced(observation: FoodObservation) -> MealItemDraft | None:
    """One photographed food, priced from the reference table at the portion the model saw."""
    reference = lookup(observation.name)
    if reference is None:
        return None
    return MealItemDraft(
        name=reference.canonical_name,
        quantity=observation.quantity,
        unit=reference.unit,
        nutrition=reference.nutrition.scaled(observation.quantity),
        confidence=observation.confidence,
    )


def _overlay(
    photo: tuple[MealItemDraft, ...], stated: tuple[MealItemDraft, ...]
) -> tuple[MealItemDraft, ...]:
    """The user's own words about the plate outrank the model's read of it, line by line."""
    merged = list(photo)
    for item in stated:
        existing = next(
            (index for index, line in enumerate(merged) if line.name == item.name), None
        )
        if existing is None:
            merged.append(item)
            continue
        merged[existing] = item
    return tuple(merged)


def _scale(item: MealItemDraft, fraction: Decimal) -> MealItemDraft:
    return item.model_copy(
        update={
            "quantity": item.quantity * fraction,
            "nutrition": item.nutrition.scaled(fraction),
        }
    )


def _uncertain_question(
    items: tuple[MealItemDraft, ...],
    reading: VisionReading,
    caption: CaptionReading,
    policy: AmbiguityPolicy,
) -> str | None:
    """Ask once when the photo is shaky in a way that could move the number.

    A caption that named the food already replaced the uncertain line, so this only fires on what
    the user left for us to judge. How shaky decides whether we ask at all: a read above the
    material-confidence floor logs as a disclosed estimate, while a weaker one cannot become a
    number and must be asked about. What we ask follows the doubt: an unreadable portion asks how
    much, a second guess far enough away to move the totals asks which of two dishes it is, and
    anything else asks the plain naming question.
    """
    stated_names = {item.name for item in caption.stated}
    for observation in reading.observations:
        item = next((line for line in items if line.name == observation.name), None)
        if item is None or item.name in stated_names:
            continue
        if item.confidence >= policy.material_confidence:
            continue
        if observation.unusable_portion:
            return (
                f"How much of the {item.name} was on the plate? I could not read a portion I "
                "would trust, and I would rather ask than invent one."
            )
        alternative = _other_candidate(observation, item, caption.scale)
        if alternative is not None and policy.materially_differ(
            item.nutrition, alternative.nutrition
        ):
            return (
                f"Is that {item.name} or {alternative.name}? I cannot separate them in the "
                "photo and the two are far enough apart that I would rather ask."
            )
        return (
            f"What is the {item.name} in this photo? I am not confident enough about "
            "it to log a number."
        )
    return None


def _other_candidate(
    observation: FoodObservation, item: MealItemDraft, scale: Decimal
) -> MealItemDraft | None:
    """The model's second guess at the same visible portion, for a materiality comparison.

    Priced with the share the caption claimed: comparing a half-plate line against a whole
    alternative would ask a different question depending on the order the foods were listed in.
    """
    if observation.alternative is None:
        return None
    priced = _priced(
        FoodObservation(
            name=observation.alternative,
            quantity=observation.quantity * scale,
            confidence=observation.confidence,
        )
    )
    if priced is None or priced.name == item.name:
        return None
    return priced


def _nothing_to_count(reading: VisionReading) -> str:
    """Say what the photo was read as when none of it can become a number.

    A food the model named but the reference table cannot price is still information the user
    gave us; hiding it behind "I could not see a meal" would make them send the photo again.
    """
    omission = responses.omission(reading.unrecognized)
    if omission:
        return f"I could not put a number on that photo. {omission} What else was on the plate?"
    if reading.unclear:
        return f"I could not make a meal out of that photo ({reading.unclear}). What did you eat?"
    return "I could not see a meal in that photo. Tell me what was on the plate and how much."


def _unrecognized(
    reading: VisionReading, caption: CaptionReading, items: tuple[MealItemDraft, ...]
) -> tuple[str, ...]:
    """Names from either half of the message that the reference table cannot price."""
    counted = {item.name for item in items}
    return tuple(
        dict.fromkeys(
            [name for name in (*reading.unrecognized, *caption.unrecognized) if name not in counted]
        )
    )


def _audit_note(request: PlannerRequest, reading: VisionReading, caption: CaptionReading) -> str:
    media_id = request.media.external_id if request.media is not None else "unknown"
    note = (
        f"Read from photo {media_id} by {reading.model}. Nutrition values are reference estimates."
    )
    if caption.scale != 1:
        note = f"{note} The caption claimed {caption.scale:g} of the plate."
    if caption.memory is not None:
        note = f"{note} The same message also stated {describe(caption.memory)}."
    return note


def _reply_notes(
    caption: CaptionReading,
    photo: tuple[MealItemDraft, ...],
    reading: VisionReading,
    *,
    plain: bool,
) -> tuple[str, ...]:
    notes: list[str] = []
    if caption.scale != 1:
        shared = _SHARE_WORDS.get(caption.scale, f"{caption.scale:g}")
        notes.append(f"I counted {shared} of the plate, as you said.")
    if caption.routine_asked:
        notes.append("Say 'remember this as my usual' and I will keep this plate.")
    if caption.other_ask is not None:
        notes.append(
            f"I logged only the plate in your photo — your {caption.other_ask} belongs in a "
            "message of its own."
        )
    if reading.unclear and photo:
        notes.append("I only counted what I could separate in the photo.")
    if plain:
        notes.append("I read that off your photo, so correct me if I misjudged a portion.")
    return tuple(notes)


def _has_caption(request: PlannerRequest) -> bool:
    return bool(" ".join(request.text.split()))


def _source_text(request: PlannerRequest) -> str:
    if _has_caption(request):
        return request.text
    media_id = request.media.external_id if request.media is not None else "photo"
    return f"[photo {media_id}]"
