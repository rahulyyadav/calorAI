from datetime import UTC, datetime
from decimal import Decimal

from calorai_agent.domain import MealItemDraft, MealRecord, MealReference, MealType, Nutrition
from calorai_agent.policy import AmbiguityPolicy, LogDecision
from calorai_agent.resolution import MealReferenceResolver, ResolutionStatus


def _item(name: str, calories: str, quantity: str = "1") -> MealItemDraft:
    return MealItemDraft(
        name=name,
        quantity=Decimal(quantity),
        unit="serving",
        nutrition=Nutrition(
            calories=Decimal(calories),
            protein_g=Decimal(calories) / Decimal("20"),
            carbs_g=Decimal("20"),
            fat_g=Decimal("5"),
        ),
    )


def _meal(
    meal_id: str,
    hour: int,
    items: tuple[MealItemDraft, ...],
    meal_type: MealType = MealType.UNSPECIFIED,
    day: int = 26,
) -> MealRecord:
    return MealRecord(
        id=meal_id,
        user_id="user-1",
        meal_type=meal_type,
        occurred_at=datetime(2026, 9, day, hour, 0, tzinfo=UTC),
        source_text=items[0].name,
        created_at=datetime(2026, 9, day, hour, 0, tzinfo=UTC),
        items=items,
    )


def _resolver() -> MealReferenceResolver:
    return MealReferenceResolver(AmbiguityPolicy())


def test_single_candidate_resolves() -> None:
    lunch = _meal("m1", 13, (_item("roti", "240"),), MealType.LUNCH)

    resolution = _resolver().resolve([lunch], MealReference())

    assert resolution.status is ResolutionStatus.RESOLVED
    assert resolution.meal is lunch


def test_empty_candidates_report_not_found() -> None:
    resolution = _resolver().resolve([], MealReference(meal_type=MealType.LUNCH))

    assert resolution.status is ResolutionStatus.NOT_FOUND
    assert resolution.meal is None


def test_meal_type_filter_is_strict_rather_than_guessing() -> None:
    lunch = _meal("m1", 13, (_item("roti", "240"),), MealType.LUNCH)

    resolution = _resolver().resolve([lunch], MealReference(meal_type=MealType.DINNER))

    assert resolution.status is ResolutionStatus.NOT_FOUND


def test_food_hint_selects_the_matching_meal() -> None:
    lunch = _meal("m1", 13, (_item("roti", "240"),), MealType.LUNCH)
    dinner = _meal("m2", 20, (_item("biryani", "600"),), MealType.DINNER)

    resolution = _resolver().resolve([lunch, dinner], MealReference(food_hint="biryani"))

    assert resolution.meal is dinner


def test_materially_different_candidates_stay_ambiguous() -> None:
    lunch = _meal("m1", 13, (_item("roti", "240"),), MealType.LUNCH)
    dinner = _meal("m2", 20, (_item("biryani", "600"),), MealType.DINNER)

    resolution = _resolver().resolve([lunch, dinner], MealReference())

    assert resolution.status is ResolutionStatus.AMBIGUOUS
    assert resolution.meal is None
    assert [meal.id for meal in resolution.candidates] == ["m2", "m1"]


def test_interchangeable_candidates_resolve_to_the_most_recent() -> None:
    earlier = _meal("m1", 13, (_item("roti", "240"),), MealType.LUNCH)
    later = _meal("m2", 20, (_item("roti", "240"),), MealType.DINNER)

    resolution = _resolver().resolve([earlier, later], MealReference())

    assert resolution.meal is later


def test_policy_bands_map_confidence_to_action() -> None:
    policy = AmbiguityPolicy()

    assert policy.assess_items([_item("roti", "240")]) is LogDecision.LOG
    assert policy.assess_items([_item("roti", "240", "0.5")]) is LogDecision.LOG
    low = _item("mystery", "300").model_copy(update={"confidence": 0.7})
    assert policy.assess_items([_item("roti", "240"), low]) is LogDecision.LOG_AS_ESTIMATE
    guessing = _item("mystery", "300").model_copy(update={"confidence": 0.2})
    assert policy.assess_items([guessing]) is LogDecision.ASK
    assert policy.assess_items([]) is LogDecision.ASK


def test_materiality_thresholds_are_explicit() -> None:
    policy = AmbiguityPolicy()
    small = Nutrition(
        calories=Decimal("400"), protein_g=Decimal("20"), carbs_g=Decimal("40"), fat_g=Decimal("5")
    )
    nearby = small.model_copy(update={"calories": Decimal("540")})
    bigger = small.model_copy(update={"protein_g": Decimal("35")})

    assert policy.materially_differ(small, nearby) is False
    assert policy.materially_differ(small, bigger) is True
