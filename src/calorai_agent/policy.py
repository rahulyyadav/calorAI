from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from calorai_agent.domain import MealItemDraft, Nutrition


class LogDecision(StrEnum):
    LOG = "log"
    LOG_AS_ESTIMATE = "log_as_estimate"
    ASK = "ask"


# No single meal holds this much of one food, so an amount past it is a misparse or a
# typo and goes to the ask band instead of the totals.
MAX_PLAUSIBLE_QUANTITY = Decimal("40")

# Confidence for an amount that was stated but cannot be trusted.
UNUSABLE_QUANTITY_CONFIDENCE = 0.2


@dataclass(frozen=True, slots=True)
class AmbiguityPolicy:
    """Encoded confidence bands and materiality thresholds.

    Uncertainty earns a question only when it can change the number the user is
    about to act on; otherwise the agent logs a disclosed estimate.
    """

    high_confidence: float = 0.8
    material_confidence: float = 0.55
    material_calories: Decimal = Decimal("150")
    material_protein_g: Decimal = Decimal("12")

    def assess_items(self, items: Iterable[MealItemDraft]) -> LogDecision:
        confidences = [item.confidence for item in items]
        if not confidences:
            return LogDecision.ASK
        lowest = min(confidences)
        if lowest >= self.high_confidence:
            return LogDecision.LOG
        if lowest >= self.material_confidence:
            return LogDecision.LOG_AS_ESTIMATE
        return LogDecision.ASK

    def materially_differ(self, left: Nutrition, right: Nutrition) -> bool:
        return (
            abs(left.calories - right.calories) >= self.material_calories
            or abs(left.protein_g - right.protein_g) >= self.material_protein_g
        )
