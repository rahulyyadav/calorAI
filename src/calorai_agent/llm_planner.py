from __future__ import annotations

import logging
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError

from calorai_agent.domain import (
    AgentIntent,
    DietaryConstraint,
    InterpretationOrigin,
    MealDraft,
    MealItemDraft,
    MealRecord,
    MealReference,
    MealType,
    MemoryContent,
    NamedRoutine,
    NutrientMetric,
    NutritionTarget,
    ParsedMessage,
)
from calorai_agent.memory import context_lines, normalized_diet
from calorai_agent.nutrition import FOODS, lookup
from calorai_agent.planning import PlannerRequest, RuleBasedPlanner, local_time
from calorai_agent.policy import MAX_PLAUSIBLE_QUANTITY, UNUSABLE_QUANTITY_CONFIDENCE
from calorai_agent.providers import ModelProviderError, TextModelClient, parse_json_object

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You interpret meal-logging messages for a nutrition assistant.

Return ONE JSON object with these fields:
- intent: "log_meal" | "revise_meal" | "delete_meal" | "repeat_meal" | "get_totals"
  | "list_meals" | "save_memory" | "acknowledge" | "clarify" | "unknown"
- items: array of {{"name", "quantity", "confidence"}} for the foods being logged or corrected.
  Use ONLY these food names: {foods}. Quantity is a decimal number of units.
- meal_type: "breakfast" | "lunch" | "dinner" | "snack" | "unspecified"
- day_offset: integer, 0 = today, -1 = yesterday
- reference: {{"day_offset", "day_explicit", "meal_type", "food_hint"}} identifying an existing
  meal for revise/delete/repeat. day_explicit is true only when the message names the day
  ("today", "yesterday"); set food_hint to one of the known food names when the message names
  a food, otherwise null.
- replace_items: boolean, true only when the message restates the entire corrected meal
  rather than fixing one line of it.
- memory: only for intent "save_memory", one of {{"diet": "vegetarian"}},
  {{"metric": "calories|protein_g|carbs_g|fat_g", "value": 1800}}, or
  {{"slot": "breakfast|lunch|dinner|snack|default"}} to remember a logged meal as a routine.
  A routine must also carry a reference to the meal it remembers.
- question: string when intent is "clarify"; otherwise null.
- statement: string when intent is "acknowledge" (for example, the user skipped a meal).
  Otherwise null.

Rules:
- You never estimate calories, protein, carbs, or fat. The application computes those.
- Never invent a meal the user did not describe. "skipped lunch" is an acknowledge.
- Vague descriptions such as "grazed all afternoon" need exactly one clarifying question.
- A durable fact the user states about themselves (a diet, a daily target, a routine) is
  "save_memory", not a meal to log.
- Confidence is your certainty about the food identity and quantity, from 0 to 1.
- A message is one meal at most.
- If the message mentions a food outside that list, name it anyway; the application
  decides whether it can count it.
"""


class ModelFoodItem(BaseModel):
    name: str
    quantity: Decimal = Decimal("1")
    confidence: float = Field(default=1.0, ge=0, le=1)


class ModelReference(BaseModel):
    day_offset: int = 0
    day_explicit: bool = False
    meal_type: MealType | None = None
    food_hint: str | None = None


class ModelMemory(BaseModel):
    diet: str | None = None
    metric: NutrientMetric | None = None
    value: Decimal | None = None
    slot: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class ModelDecision(BaseModel):
    intent: AgentIntent
    items: tuple[ModelFoodItem, ...] = ()
    meal_type: MealType = MealType.UNSPECIFIED
    day_offset: int = 0
    reference: ModelReference | None = None
    replace_items: bool = False
    memory: ModelMemory | None = None
    question: str | None = None
    statement: str | None = None


class ModelPlanner:
    """Model-backed interpreter that keeps nutrition math deterministic.

    The model identifies foods, quantities, and intent. Reference nutrition data
    and totals always come from the application, and any unusable model output
    falls back to the deterministic planner instead of writing bad rows.
    """

    def __init__(self, client: TextModelClient, model: str) -> None:
        self.client = client
        self.model = model
        self._fallback = RuleBasedPlanner()

    def parse(self, request: PlannerRequest) -> ParsedMessage:
        try:
            raw = self.client.complete(
                system=SYSTEM_PROMPT.format(foods=", ".join(sorted(FOODS))),
                user=_render_user_prompt(request),
            )
            decision = ModelDecision.model_validate(parse_json_object(raw))
        except (ModelProviderError, ValidationError, ValueError) as error:
            logger.warning(
                "text model failed to produce a decision (%s); interpreting with rules",
                type(error).__name__,
            )
            return self._fallback.parse(request)
        try:
            return self._to_parsed_message(decision, request)
        except ValueError as error:
            logger.warning(
                "text model decision was unusable (%s); interpreting with rules",
                type(error).__name__,
            )
            return self._fallback.parse(request)

    def _to_parsed_message(self, decision: ModelDecision, request: PlannerRequest) -> ParsedMessage:
        items, unrecognized = _build_items(decision.items)
        reference = (
            MealReference(
                day_offset=decision.reference.day_offset,
                day_explicit=decision.reference.day_explicit,
                meal_type=decision.reference.meal_type,
                food_hint=decision.reference.food_hint,
            )
            if decision.reference
            else None
        )

        if decision.intent is AgentIntent.LOG_MEAL:
            if not items:
                raise ValueError("log_meal decision had no known foods")
            return ParsedMessage(
                intent=AgentIntent.LOG_MEAL,
                draft=self._draft(request, items, decision),
                unrecognized=unrecognized,
            )
        if decision.intent is AgentIntent.REVISE_MEAL:
            if not items or reference is None:
                raise ValueError("revise_meal decision needs foods and a reference")
            return ParsedMessage(
                intent=AgentIntent.REVISE_MEAL,
                items=tuple(items),
                reference=reference,
                replace_items=decision.replace_items,
                unrecognized=unrecognized,
            )
        if decision.intent is AgentIntent.DELETE_MEAL:
            if reference is None:
                raise ValueError("delete_meal decision needs a reference")
            return ParsedMessage(intent=AgentIntent.DELETE_MEAL, reference=reference)
        if decision.intent is AgentIntent.REPEAT_MEAL:
            return ParsedMessage(
                intent=AgentIntent.REPEAT_MEAL,
                reference=reference or MealReference(day_offset=decision.day_offset),
                target_meal_type=decision.meal_type
                if decision.meal_type is not MealType.UNSPECIFIED
                else None,
            )
        if decision.intent in (AgentIntent.GET_TOTALS, AgentIntent.LIST_MEALS):
            return ParsedMessage(
                intent=decision.intent,
                reference=MealReference(day_offset=decision.day_offset),
            )
        if decision.intent is AgentIntent.SAVE_MEMORY:
            return ParsedMessage(
                intent=AgentIntent.SAVE_MEMORY,
                memory=_build_memory(decision.memory),
                reference=reference,
            )
        if decision.intent is AgentIntent.CLARIFY:
            return ParsedMessage(
                intent=AgentIntent.CLARIFY,
                question=decision.question or "Could you tell me a bit more?",
            )
        if decision.intent is AgentIntent.ACKNOWLEDGE:
            return ParsedMessage(
                intent=AgentIntent.ACKNOWLEDGE,
                statement=decision.statement or "Noted — nothing logged.",
            )
        return ParsedMessage(
            intent=AgentIntent.UNKNOWN,
            explanation=decision.question or _UNKNOWN_EXPLANATION,
        )

    def _draft(
        self, request: PlannerRequest, items: list[MealItemDraft], decision: ModelDecision
    ) -> MealDraft:
        return MealDraft(
            meal_type=decision.meal_type,
            occurred_at=local_time(request, decision.day_offset, decision.meal_type),
            source_text=request.text,
            items=tuple(items),
            notes="Nutrition values come from reference data for the identified foods.",
            origin=InterpretationOrigin.TEXT_MODEL,
            model=self.model,
        )


_UNKNOWN_EXPLANATION = (
    "I couldn't identify a supported food yet. Try something like "
    "'had 2 parathas and chai for breakfast'."
)


def _build_items(
    model_items: tuple[ModelFoodItem, ...],
) -> tuple[list[MealItemDraft], tuple[str, ...]]:
    """Translate model-identified foods into reference-backed nutrition rows.

    Foods with no reference entry cannot be priced in nutrition, so they are not logged --
    but they are reported back rather than dropped, so a partial meal is never presented
    as a complete one.
    """
    merged: dict[str, MealItemDraft] = {}
    unknown: list[str] = []
    for entry in model_items:
        reference = lookup(entry.name)
        if reference is None:
            if entry.name not in unknown:
                unknown.append(entry.name)
            continue
        if entry.quantity <= 0:
            continue
        confidence = (
            UNUSABLE_QUANTITY_CONFIDENCE
            if entry.quantity > MAX_PLAUSIBLE_QUANTITY
            else entry.confidence
        )
        existing = merged.get(reference.canonical_name)
        if existing is None:
            merged[reference.canonical_name] = MealItemDraft(
                name=reference.canonical_name,
                quantity=entry.quantity,
                unit=reference.unit,
                nutrition=reference.nutrition.scaled(entry.quantity),
                confidence=confidence,
            )
        else:
            total = existing.quantity + entry.quantity
            merged[reference.canonical_name] = existing.model_copy(
                update={
                    "quantity": total,
                    "nutrition": reference.nutrition.scaled(total),
                    "confidence": min(existing.confidence, confidence),
                }
            )
    return list(merged.values()), tuple(unknown)


def _build_memory(memory: ModelMemory | None) -> MemoryContent:
    if memory is None:
        raise ValueError("save_memory decision named no fact")
    if memory.diet:
        return DietaryConstraint(diet=normalized_diet(memory.diet), confidence=memory.confidence)
    if memory.metric is not None and memory.value is not None:
        return NutritionTarget(
            metric=memory.metric, value=memory.value, confidence=memory.confidence
        )
    if memory.slot:
        return NamedRoutine(slot=memory.slot, confidence=memory.confidence)
    raise ValueError("save_memory decision named no fact")


def _render_user_prompt(request: PlannerRequest) -> str:
    lines = [f"Current time: {request.occurred_at.isoformat()}", f"Timezone: {request.timezone}"]
    if request.recent_meals:
        lines.append("Recent meals:")
        lines.extend(f"- {_describe(meal)}" for meal in request.recent_meals[-8:])
    if request.memories:
        lines.append("What the user asked us to remember:")
        lines.extend(context_lines(request.memories))
    lines.append(f"Message: {request.text}")
    return "\n".join(lines)


def _describe(meal: MealRecord) -> str:
    described = ", ".join(f"{item.quantity:g} {item.name}" for item in meal.items)
    return f"{meal.occurred_at.isoformat()} {meal.meal_type.value}: {described}"
