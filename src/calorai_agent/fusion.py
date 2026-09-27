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
    return CaptionReading(
        scale=plate_share(stripped),
        meal_type=meal_type_hint(stripped) or parsed.target_meal_type or MealType.UNSPECIFIED,
        day_offset=day_offset_hint(stripped),
        stated=parsed.items or (parsed.draft.items if parsed.draft is not None else ()),
        memory=memory,
        routine_asked=isinstance(parsed.memory, NamedRoutine)
        or _USUAL_REQUEST.search(stripped.lower()) is not None,
        unrecognized=parsed.unrecognized,
    )


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
    items = _overlay(photo, caption.stated)
    if caption.scale != 1:
        items = tuple(_scale(item, caption.scale) for item in items)

    unrecognized = _unrecognized(reading, caption, items)
    question = _uncertain_question(items, reading, caption, policy)
    if question is not None:
        return ParsedMessage(intent=AgentIntent.CLARIFY, question=question)
    if not items:
        return ParsedMessage(
            intent=AgentIntent.CLARIFY,
            question=_nothing_to_count(reading),
            unrecognized=unrecognized,
        )

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
        reply_notes=_reply_notes(caption, photo, reading, plain=not _has_caption(request)),
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
    """Ask once when the photo's identity is shaky in a way that could move the number.

    A caption that named the food already replaced the uncertain line, so this only fires on what
    the user left for us to judge. How shaky decides whether we ask at all: a read above the
    material-confidence floor logs as a disclosed estimate, while a weaker one cannot become a
    number and must be asked about. What we ask depends on the model's second guess — two dishes
    far apart in nutrition earn an either/or, anything else earns the plain naming question.
    """
    stated_names = {item.name for item in caption.stated}
    for observation in reading.observations:
        item = next((line for line in items if line.name == observation.name), None)
        if item is None or item.name in stated_names:
            continue
        if item.confidence >= policy.material_confidence:
            continue
        alternative = _other_candidate(observation, item)
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


def _other_candidate(observation: FoodObservation, item: MealItemDraft) -> MealItemDraft | None:
    """The model's second guess, priced at the same portion, for a materiality comparison."""
    if observation.alternative is None:
        return None
    priced = _priced(
        FoodObservation(
            name=observation.alternative,
            quantity=observation.quantity,
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
