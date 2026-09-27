from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import NotRequired, TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph

from calorai_agent import responses
from calorai_agent.domain import (
    AgentIntent,
    InboundMessage,
    MealDraft,
    MealItemDraft,
    MealRecord,
    MealType,
    MemoryKind,
    MemoryRecord,
    MutationKind,
    NamedRoutine,
    ParsedMessage,
    combine_portions,
)
from calorai_agent.memory import (
    DEFAULT_ROUTINE_SLOT,
    conflicting_foods,
    current_diet,
    targets,
)
from calorai_agent.planning import MEAL_TIMES, MessagePlanner, PlannerRequest
from calorai_agent.policy import AmbiguityPolicy, LogDecision
from calorai_agent.resolution import MealReferenceResolver, Resolution, ResolutionStatus
from calorai_agent.tools import (
    DeleteMealInput,
    GetDailyTotalsInput,
    GetMealsInRangeInput,
    ListMemoriesInput,
    LogMealInput,
    MealTools,
    RecordInboundInput,
    RememberInput,
    ReviseMealInput,
)

REFERENCE_LOOKBACK_DAYS = 3

_REFERENCE_MUTATIONS = frozenset(
    {
        AgentIntent.REPEAT_MEAL,
        AgentIntent.REVISE_MEAL,
        AgentIntent.DELETE_MEAL,
        AgentIntent.SAVE_MEMORY,
    }
)

_NO_TOOL_CALLS = frozenset({AgentIntent.ACKNOWLEDGE, AgentIntent.CLARIFY, AgentIntent.UNKNOWN})

# What an unresolved pointer was meant to do, phrased the way the reply asks about it.
_REFERENCE_VERBS = {
    AgentIntent.DELETE_MEAL: "delete",
    AgentIntent.REVISE_MEAL: "correct",
    AgentIntent.SAVE_MEMORY: "remember",
}

# The kinds a turn can act on. Memories of other kinds stay in the database, unread.
_CONTEXT_KINDS = (
    MemoryKind.DIETARY_CONSTRAINT,
    MemoryKind.NUTRITION_TARGET,
    MemoryKind.NAMED_ROUTINE,
)


class AgentState(TypedDict):
    user_id: str
    timezone: str
    message: str
    now: datetime
    event_id: NotRequired[str | None]
    today: NotRequired[date]
    recent_meals: NotRequired[list[MealRecord]]
    memories: NotRequired[list[MemoryRecord]]
    parsed: NotRequired[ParsedMessage]
    resolution: NotRequired[Resolution]
    reference_from_window: NotRequired[bool]
    response: NotRequired[str]


class MealAgent:
    """context -> plan -> resolve -> one bounded tool, or one conversational reply."""

    def __init__(
        self,
        planner: MessagePlanner,
        tools: MealTools,
        policy: AmbiguityPolicy | None = None,
        resolver: MealReferenceResolver | None = None,
    ) -> None:
        self.planner = planner
        self.tools = tools
        self.policy = policy or AmbiguityPolicy()
        self.resolver = resolver or MealReferenceResolver(self.policy)

        builder = StateGraph(AgentState)
        builder.add_node("gather_context", self._gather_context)
        builder.add_node("plan", self._plan)
        builder.add_node("resolve_reference", self._resolve_reference)
        builder.add_node("log_meal", self._log_meal)
        builder.add_node("repeat_meal", self._repeat_meal)
        builder.add_node("revise_meal", self._revise_meal)
        builder.add_node("delete_meal", self._delete_meal)
        builder.add_node("get_totals", self._get_totals)
        builder.add_node("list_meals", self._list_meals)
        builder.add_node("save_memory", self._save_memory)
        builder.add_node("respond", self._respond)

        builder.add_edge(START, "gather_context")
        builder.add_edge("gather_context", "plan")
        builder.add_edge("plan", "resolve_reference")
        builder.add_conditional_edges(
            "resolve_reference",
            self._route,
            {
                AgentIntent.LOG_MEAL.value: "log_meal",
                AgentIntent.REPEAT_MEAL.value: "repeat_meal",
                AgentIntent.REVISE_MEAL.value: "revise_meal",
                AgentIntent.DELETE_MEAL.value: "delete_meal",
                AgentIntent.GET_TOTALS.value: "get_totals",
                AgentIntent.LIST_MEALS.value: "list_meals",
                AgentIntent.SAVE_MEMORY.value: "save_memory",
                "respond": "respond",
            },
        )
        for node in (
            "log_meal",
            "repeat_meal",
            "revise_meal",
            "delete_meal",
            "get_totals",
            "list_meals",
            "save_memory",
            "respond",
        ):
            builder.add_edge(node, END)

        self._graph = builder.compile()

    def invoke(
        self,
        user_id: str,
        message: str,
        *,
        timezone: str = "UTC",
        now: datetime | None = None,
    ) -> str:
        zone = ZoneInfo(timezone)
        current = now or datetime.now(zone)
        if current.tzinfo is None:
            current = current.replace(tzinfo=zone)
        return self.handle(
            InboundMessage(
                user_id=user_id,
                text=message,
                external_id=cli_event_id(user_id, message, current),
                channel="cli",
                timezone=timezone,
                received_at=current,
            )
        )

    def handle(self, inbound: InboundMessage) -> str:
        """Process one inbound message exactly once, even when it is redelivered."""
        event = self.tools.record_inbound(
            RecordInboundInput(
                user_id=inbound.user_id,
                external_id=inbound.external_id,
                text=inbound.text,
                channel=inbound.channel,
            )
        )
        if not event.created:
            return self._replay(event.id, event.response_text, inbound.timezone)

        response = self._run(inbound, event.id)
        self.tools.complete_inbound(event.id, response)
        return response

    def _replay(self, event_id: str, cached: str | None, timezone_name: str) -> str:
        if cached is not None:
            return cached
        remembered = self.tools.memory_for_event(event_id)
        if remembered is not None:
            return responses.remembered(remembered.content)
        outcome = self.tools.outcome_for_event(event_id)
        if outcome is None:
            return "I already received that message and am still working on it."
        meal = outcome.meal
        match outcome.kind:
            case MutationKind.LOGGED:
                return responses.logged(meal, self.policy.assess_items(meal.items))
            case MutationKind.REVISED:
                return responses.revised(meal, timezone_name=timezone_name)
            case MutationKind.DELETED:
                return responses.deleted(meal)
        raise AssertionError(f"unhandled outcome kind {outcome.kind!r}")

    def _run(self, inbound: InboundMessage, event_id: str) -> str:
        now = inbound.received_at or datetime.now(ZoneInfo(inbound.timezone))
        result = self._graph.invoke(
            {
                "user_id": inbound.user_id,
                "timezone": inbound.timezone,
                "message": inbound.text,
                "now": now,
                "event_id": event_id,
            }
        )
        response = result.get("response")
        if not isinstance(response, str):
            raise RuntimeError("agent graph completed without a response")
        return response

    def _gather_context(self, state: AgentState) -> dict[str, object]:
        today = _local_day(state["now"], state["timezone"])
        meals = self.tools.get_meals_in_range(
            GetMealsInRangeInput(
                user_id=state["user_id"],
                start_day=today - timedelta(days=REFERENCE_LOOKBACK_DAYS),
                end_day=today,
                timezone=state["timezone"],
            )
        )
        memories = self.tools.list_memories(
            ListMemoriesInput(user_id=state["user_id"], kinds=_CONTEXT_KINDS)
        )
        return {"today": today, "recent_meals": meals, "memories": memories}

    def _plan(self, state: AgentState) -> dict[str, ParsedMessage]:
        parsed = self.planner.parse(
            PlannerRequest(
                text=state["message"],
                occurred_at=state["now"],
                timezone=state["timezone"],
                recent_meals=tuple(state["recent_meals"]),
                memories=tuple(state["memories"]),
            )
        )
        return {"parsed": parsed}

    def _resolve_reference(self, state: AgentState) -> dict[str, Resolution | bool]:
        parsed = state["parsed"]
        if parsed.intent not in _REFERENCE_MUTATIONS or parsed.reference is None:
            return {}
        reference = parsed.reference
        target_day = state["today"] + timedelta(days=reference.day_offset)
        on_day = [
            meal
            for meal in state["recent_meals"]
            if _local_day(meal.occurred_at, state["timezone"]) == target_day
        ]
        if on_day or reference.day_explicit:
            return {"resolution": self.resolver.resolve(on_day, reference)}
        # A pointer with no day of its own ("actually 3 eggs") usually means the meal the
        # user last logged, which is still on the previous calendar day shortly after
        # midnight. Search that far back, and no further: silently rewriting a meal from
        # days ago is worse than saying there is nothing to correct.
        boundary = state["today"] - timedelta(days=1)
        last_logged = [
            meal
            for meal in state["recent_meals"]
            if _local_day(meal.occurred_at, state["timezone"]) >= boundary
        ]
        return {
            "resolution": self.resolver.resolve(last_logged, reference),
            "reference_from_window": True,
        }

    def _route(self, state: AgentState) -> str:
        parsed = state["parsed"]
        if parsed.intent in _NO_TOOL_CALLS:
            return "respond"
        if parsed.intent in _REFERENCE_MUTATIONS and parsed.reference is not None:
            resolution = state.get("resolution")
            if resolution is None or resolution.status is not ResolutionStatus.RESOLVED:
                return "respond"
        return parsed.intent.value

    def _log_meal(self, state: AgentState) -> dict[str, str]:
        parsed = state["parsed"]
        draft = parsed.draft
        if draft is None:  # pragma: no cover - enforced by ParsedMessage validation
            raise ValueError("log_meal intent requires a meal draft")
        decision = self.policy.assess_items(draft.items)
        if decision is LogDecision.ASK:
            return {"response": _clarify(draft.items, self.policy)}
        meal = self.tools.log_meal(
            LogMealInput(user_id=state["user_id"], meal=draft, source_event_id=state["event_id"])
        )
        return {
            "response": responses.logged(
                meal, decision, _meal_notes(parsed.unrecognized, state["memories"], meal.items)
            )
        }

    def _repeat_meal(self, state: AgentState) -> dict[str, str]:
        source = self._resolved_meal(state)
        parsed = state["parsed"]
        today = state["today"]
        target_type = parsed.target_meal_type or source.meal_type
        notes = f"Copied from meal {source.id} revision {source.revision_number}."
        items = source.items
        if parsed.items:
            # "the same as yesterday, plus a banana" is one meal: the copy with the banana
            # folded in, so neither half of the message goes uncounted.
            items = combine_portions(source.items, parsed.items)
            added = ", ".join(f"{item.quantity:g} {item.name}" for item in parsed.items)
            notes = f"{notes} You added {added}."
        copy = MealDraft(
            meal_type=target_type,
            occurred_at=_at_local_time(today, target_type, source, state["timezone"]),
            source_text=state["message"],
            items=items,
            notes=notes,
        )
        logged = self.tools.log_meal(
            LogMealInput(user_id=state["user_id"], meal=copy, source_event_id=state["event_id"])
        )
        label = responses.day_label(
            _local_day(source.occurred_at, state["timezone"]), today
        ).lower()
        return {
            "response": responses.repeated(
                source,
                logged,
                label,
                _meal_notes(parsed.unrecognized, state["memories"], logged.items),
                timezone_name=state["timezone"],
            )
        }

    def _revise_meal(self, state: AgentState) -> dict[str, str]:
        parsed = state["parsed"]
        if self.policy.assess_items(parsed.items) is LogDecision.ASK:
            # A correction is a rewrite of history: it needs the same confidence
            # bar as a fresh log, or a misparse silently changes a real number.
            return {"response": _clarify(parsed.items, self.policy)}
        revised = self.tools.revise_meal(
            ReviseMealInput(
                user_id=state["user_id"],
                meal_id=self._resolved_meal(state).id,
                items=parsed.items,
                source_text=state["message"],
                replace_items=parsed.replace_items,
                source_event_id=state.get("event_id"),
            )
        )
        if revised is None:
            return {"response": responses.not_found("that")}
        notes = _meal_notes(parsed.unrecognized, state["memories"], revised.items)
        return {"response": responses.revised(revised, notes, timezone_name=state["timezone"])}

    def _save_memory(self, state: AgentState) -> dict[str, str]:
        parsed = state["parsed"]
        content = parsed.memory
        if content is None:  # pragma: no cover - enforced by ParsedMessage validation
            raise ValueError("save_memory intent requires a memory")
        if isinstance(content, NamedRoutine):
            content = _routine_from(content, self._resolved_meal(state))
        saved = self.tools.remember(
            RememberInput(
                user_id=state["user_id"],
                memory=content,
                source_event_id=state.get("event_id"),
            )
        )
        return {"response": responses.remembered(saved.content, parsed.unlogged)}

    def _delete_meal(self, state: AgentState) -> dict[str, str]:
        meal = self._resolved_meal(state)
        self.tools.delete_meal(
            DeleteMealInput(
                user_id=state["user_id"],
                meal_id=meal.id,
                source_event_id=state.get("event_id"),
            )
        )
        return {"response": responses.deleted(meal)}

    def _get_totals(self, state: AgentState) -> dict[str, str]:
        day = _target_day(state)
        totals = self.tools.get_daily_totals(
            GetDailyTotalsInput(user_id=state["user_id"], day=day, timezone=state["timezone"])
        )
        return {
            "response": responses.totals(
                totals,
                responses.day_label(day, state["today"]),
                targets(state["memories"]),
            )
        }

    def _list_meals(self, state: AgentState) -> dict[str, str]:
        day = _target_day(state)
        meals = self.tools.get_meals_in_range(
            GetMealsInRangeInput(
                user_id=state["user_id"],
                start_day=day,
                end_day=day,
                timezone=state["timezone"],
            )
        )
        return {
            "response": responses.meal_list(
                meals, responses.day_label(day, state["today"]), timezone_name=state["timezone"]
            )
        }

    def _respond(self, state: AgentState) -> dict[str, str]:
        parsed = state["parsed"]
        resolution = state.get("resolution")
        if resolution is not None and parsed.intent in _REFERENCE_MUTATIONS:
            if state.get("reference_from_window"):
                return {
                    "response": responses.no_reference_match(
                        resolution.candidates, timezone_name=state["timezone"]
                    )
                }
            label = responses.day_label(_target_day(state), state["today"])
            if resolution.status is ResolutionStatus.AMBIGUOUS and resolution.candidates:
                verb = _REFERENCE_VERBS[parsed.intent]
                return {
                    "response": responses.ambiguous(
                        resolution.candidates, label, verb, timezone_name=state["timezone"]
                    )
                }
            if parsed.intent is AgentIntent.SAVE_MEMORY:
                return {"response": responses.nothing_to_remember(label)}
            return {"response": responses.not_found(label)}
        return {
            "response": parsed.question
            or parsed.statement
            or parsed.explanation
            or "I need a little more detail."
        }

    def _resolved_meal(self, state: AgentState) -> MealRecord:
        resolution = state.get("resolution")
        if resolution is None or resolution.meal is None:  # pragma: no cover - routed earlier
            raise ValueError("no meal resolved for this request")
        return resolution.meal


def _local_day(value: datetime, timezone_name: str) -> date:
    return value.astimezone(ZoneInfo(timezone_name)).date()


def _target_day(state: AgentState) -> date:
    reference = state["parsed"].reference
    return state["today"] + timedelta(days=reference.day_offset if reference else 0)


def _at_local_time(
    day: date, meal_type: MealType, source: MealRecord, timezone_name: str
) -> datetime:
    zone = ZoneInfo(timezone_name)
    clock = MEAL_TIMES.get(meal_type)
    if clock is None:
        local = source.occurred_at.astimezone(zone)
        return datetime.combine(day, local.time(), tzinfo=zone)
    return datetime.combine(day, clock, tzinfo=zone)


def cli_event_id(user_id: str, message: str, at: datetime) -> str:
    """Stable per-message key so an identical CLI resend cannot log a second meal."""
    stamp = at.isoformat(timespec="seconds")
    digest = hashlib.sha256(f"{user_id}|{message}|{stamp}".encode()).hexdigest()
    return f"cli:{digest[:24]}"


def _clarify(items: Sequence[MealItemDraft], policy: AmbiguityPolicy) -> str:
    unsure = [item for item in items if item.confidence < policy.material_confidence]
    names = ", ".join(item.name for item in unsure or items)
    return (
        f"I am not confident enough about the {names} portion to log a number. "
        "Roughly how much did you have?"
    )


def _meal_notes(
    unrecognized: Sequence[str], memories: Sequence[MemoryRecord], items: Sequence[MealItemDraft]
) -> list[str]:
    """What a written meal should be answered with beyond its own numbers."""
    notes = [responses.omission(unrecognized)]
    conflicts = conflicting_foods(memories, items)
    diet = current_diet(memories)
    if conflicts and diet is not None:
        notes.append(responses.diet_conflict(diet, conflicts))
    return [note for note in notes if note]


def _routine_from(routine: NamedRoutine, meal: MealRecord) -> NamedRoutine:
    slot = routine.slot
    if slot == DEFAULT_ROUTINE_SLOT and meal.meal_type is not MealType.UNSPECIFIED:
        slot = meal.meal_type.value
    return NamedRoutine(slot=slot, items=meal.items, confidence=routine.confidence)
