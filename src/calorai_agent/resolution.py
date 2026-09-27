from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from calorai_agent.domain import MealRecord, MealReference
from calorai_agent.policy import AmbiguityPolicy


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class Resolution:
    status: ResolutionStatus
    meal: MealRecord | None = None
    candidates: tuple[MealRecord, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.status is ResolutionStatus.RESOLVED and self.meal is not None


class MealReferenceResolver:
    """Turn a message pointer into at most one meal, or refuse to guess."""

    def __init__(self, policy: AmbiguityPolicy) -> None:
        self.policy = policy

    def resolve(self, candidates: Sequence[MealRecord], reference: MealReference) -> Resolution:
        pool = sorted(candidates, key=lambda meal: meal.occurred_at, reverse=True)

        if reference.meal_type is not None:
            pool = [meal for meal in pool if meal.meal_type is reference.meal_type]
        if not pool:
            return Resolution(ResolutionStatus.NOT_FOUND)

        if reference.food_hint:
            hinted = [meal for meal in pool if self._mentions(meal, reference.food_hint)]
            if not hinted:
                return Resolution(ResolutionStatus.NOT_FOUND)
            pool = hinted

        if len(pool) == 1:
            return Resolution(ResolutionStatus.RESOLVED, meal=pool[0])

        first, *rest = pool
        if not any(self.policy.materially_differ(first.nutrition, m.nutrition) for m in rest):
            # Several matches, but they are interchangeable for the user's purpose.
            return Resolution(ResolutionStatus.RESOLVED, meal=first, candidates=tuple(pool))
        return Resolution(ResolutionStatus.AMBIGUOUS, candidates=tuple(pool))

    @staticmethod
    def _mentions(meal: MealRecord, food_hint: str) -> bool:
        hint = food_hint.lower()
        return any(hint in item.name.lower() for item in meal.items)
