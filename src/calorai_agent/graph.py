from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import NotRequired, TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph

from calorai_agent.domain import AgentIntent, MealRecord, ParsedMessage
from calorai_agent.nutrition import MessagePlanner
from calorai_agent.tools import GetDailyTotalsInput, GetMealsInput, LogMealInput, MealTools


class AgentState(TypedDict):
    user_id: str
    timezone: str
    message: str
    now: datetime
    parsed: NotRequired[ParsedMessage]
    response: NotRequired[str]


class MealAgent:
    def __init__(self, planner: MessagePlanner, tools: MealTools) -> None:
        self.planner = planner
        self.tools = tools
        builder = StateGraph(AgentState)
        builder.add_node("plan", self._plan)
        builder.add_node("log_meal", self._log_meal)
        builder.add_node("get_totals", self._get_totals)
        builder.add_node("list_meals", self._list_meals)
        builder.add_node("fallback", self._fallback)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            self._route,
            {
                AgentIntent.LOG_MEAL.value: "log_meal",
                AgentIntent.GET_TOTALS.value: "get_totals",
                AgentIntent.LIST_MEALS.value: "list_meals",
                AgentIntent.UNKNOWN.value: "fallback",
            },
        )
        for node in ("log_meal", "get_totals", "list_meals", "fallback"):
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
        result = self._graph.invoke(
            {"user_id": user_id, "timezone": timezone, "message": message, "now": current}
        )
        response = result.get("response")
        if not isinstance(response, str):
            raise RuntimeError("agent graph completed without a response")
        return response

    def _plan(self, state: AgentState) -> dict[str, ParsedMessage]:
        return {"parsed": self.planner.parse(state["message"], state["now"])}

    @staticmethod
    def _route(state: AgentState) -> str:
        return state["parsed"].intent.value

    def _log_meal(self, state: AgentState) -> dict[str, str]:
        parsed = state["parsed"]
        if parsed.draft is None:
            raise ValueError("log_meal intent requires a meal draft")
        meal = self.tools.log_meal(LogMealInput(user_id=state["user_id"], meal=parsed.draft))
        item_summary = ", ".join(
            f"{self._quantity(item.quantity)} {item.name}" for item in meal.items
        )
        nutrition = meal.nutrition
        return {
            "response": (
                f"Logged {item_summary} — about {nutrition.calories:.0f} kcal and "
                f"{nutrition.protein_g:.0f}g protein."
            )
        }

    def _get_totals(self, state: AgentState) -> dict[str, str]:
        totals = self.tools.get_daily_totals(
            GetDailyTotalsInput(
                user_id=state["user_id"],
                day=self._local_day(state),
                timezone=state["timezone"],
            )
        )
        n = totals.nutrition
        if totals.meal_count == 0:
            return {"response": "Nothing logged today yet."}
        return {
            "response": (
                f"Today: {n.calories:.0f} kcal, {n.protein_g:.0f}g protein, "
                f"{n.carbs_g:.0f}g carbs, and {n.fat_g:.0f}g fat "
                f"across {totals.meal_count} {self._plural(totals.meal_count, 'meal')}."
            )
        }

    def _list_meals(self, state: AgentState) -> dict[str, str]:
        meals = self.tools.get_meals(
            GetMealsInput(
                user_id=state["user_id"],
                day=self._local_day(state),
                timezone=state["timezone"],
            )
        )
        if not meals:
            return {"response": "Nothing logged today yet."}
        summary = "; ".join(self._describe(meal) for meal in meals)
        return {"response": f"Today you logged: {summary}."}

    @staticmethod
    def _fallback(state: AgentState) -> dict[str, str]:
        return {"response": state["parsed"].explanation or "I need a little more detail."}

    @staticmethod
    def _local_day(state: AgentState) -> date:
        return state["now"].astimezone(ZoneInfo(state["timezone"])).date()

    @staticmethod
    def _quantity(value: Decimal) -> str:
        return str(int(value)) if value == value.to_integral() else str(value.normalize())

    @classmethod
    def _describe(cls, meal: MealRecord) -> str:
        return ", ".join(f"{cls._quantity(item.quantity)} {item.name}" for item in meal.items)

    @staticmethod
    def _plural(count: int, noun: str) -> str:
        return noun if count == 1 else noun + "s"
