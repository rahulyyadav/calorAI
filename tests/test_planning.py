from datetime import UTC, datetime
from decimal import Decimal

import pytest

from calorai_agent.domain import AgentIntent, MealItemDraft, MealRecord, MealType, Nutrition
from calorai_agent.planning import EXPLICIT_CONFIDENCE, PlannerRequest, RuleBasedPlanner
from calorai_agent.policy import UNUSABLE_QUANTITY_CONFIDENCE

NOW = datetime(2026, 9, 26, 8, tzinfo=UTC)


def _request(text: str) -> PlannerRequest:
    return PlannerRequest(text=text, occurred_at=NOW, timezone="UTC")


def test_parses_multiple_foods_and_scales_nutrition() -> None:
    parsed = RuleBasedPlanner().parse(_request("had 2 parathas and chai for breakfast"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert parsed.draft.meal_type is MealType.BREAKFAST
    assert [item.name for item in parsed.draft.items] == ["paratha", "milk chai"]
    assert parsed.draft.items[0].quantity == Decimal("2")
    assert sum(item.nutrition.calories for item in parsed.draft.items) == Decimal("640.00")


def test_routes_totals_without_attempting_food_extraction() -> None:
    parsed = RuleBasedPlanner().parse(_request("how much protein have I had today?"))

    assert parsed.intent is AgentIntent.GET_TOTALS
    assert parsed.draft is None


def test_unknown_food_requests_actionable_input() -> None:
    parsed = RuleBasedPlanner().parse(_request("I ate something nice"))

    assert parsed.intent is AgentIntent.UNKNOWN
    assert "2 parathas" in (parsed.explanation or "")


def test_fraction_after_food_name_is_applied_to_that_food() -> None:
    parsed = RuleBasedPlanner().parse(_request("leftover biryani, maybe two thirds of the box"))

    assert parsed.draft is not None
    item = parsed.draft.items[0]
    assert item.name == "biryani"
    assert abs(item.quantity - Decimal("0.667")) < Decimal("0.01")
    assert item.confidence < 0.8  # hedged wording must lower, not raise, confidence


def test_skipped_meal_is_acknowledged_without_logging() -> None:
    parsed = RuleBasedPlanner().parse(_request("skipped lunch"))

    assert parsed.intent is AgentIntent.ACKNOWLEDGE
    assert parsed.draft is None


def test_vague_grazing_asks_exactly_one_question() -> None:
    parsed = RuleBasedPlanner().parse(_request("skipped lunch but grazed all afternoon"))

    assert parsed.intent is AgentIntent.CLARIFY
    assert parsed.question is not None
    assert parsed.question.count("?") == 1


def test_correction_words_produce_a_revision_with_a_reference() -> None:
    parsed = RuleBasedPlanner().parse(_request("actually that was 3 rotis not 2"))

    assert parsed.intent is AgentIntent.REVISE_MEAL
    assert parsed.reference is not None
    assert parsed.reference.food_hint == "roti"
    assert parsed.items[0].quantity == Decimal("3")


def test_repeat_of_yesterday_needs_no_items() -> None:
    parsed = RuleBasedPlanner().parse(_request("same as yesterday"))

    assert parsed.intent is AgentIntent.REPEAT_MEAL
    assert parsed.reference is not None
    assert parsed.reference.day_offset == -1


def test_named_food_with_retry_word_logs_rather_than_repeats() -> None:
    parsed = RuleBasedPlanner().parse(_request("I had biryani again"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert parsed.draft.items[0].name == "biryani"


def test_delete_takes_precedence_over_logging_the_named_food() -> None:
    parsed = RuleBasedPlanner().parse(_request("delete the pizza I just logged"))

    assert parsed.intent is AgentIntent.DELETE_MEAL
    assert parsed.reference is not None
    assert parsed.reference.food_hint == "pizza"


def test_repeat_names_the_target_meal_without_a_food_hint() -> None:
    parsed = RuleBasedPlanner().parse(_request("same as yesterday for dinner"))

    assert parsed.intent is AgentIntent.REPEAT_MEAL
    assert parsed.reference is not None
    assert parsed.reference.day_offset == -1
    assert parsed.target_meal_type is MealType.DINNER
    # The target slot must not leak into the pointer, or "dinner" resolves as a food hint.
    assert parsed.reference.food_hint is None


def test_denied_food_is_acknowledged_instead_of_logged() -> None:
    planner = RuleBasedPlanner()

    for text in (
        "no eggs",
        "not any rice",
        "i had zero eggs",
        "not rotis",
        "i didnt have eggs",
        "i did not have any rice",
        "i never ate dosa",
        "i have not had eggs",
    ):
        parsed = planner.parse(_request(text))
        assert parsed.intent is AgentIntent.ACKNOWLEDGE, text
        assert parsed.draft is None, text


def test_a_denial_does_not_swallow_the_food_it_negates() -> None:
    parsed = RuleBasedPlanner().parse(_request("i didnt have eggs but had 2 rotis"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert [(item.name, str(item.quantity)) for item in parsed.draft.items] == [("roti", "2")]


def test_bare_contrast_words_do_not_make_a_log_into_a_revision() -> None:
    parsed = RuleBasedPlanner().parse(_request("dosa for lunch not dinner"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert parsed.draft.meal_type is MealType.LUNCH


def test_nutrition_question_about_an_unlogged_food_is_not_a_meal() -> None:
    parsed = RuleBasedPlanner().parse(_request("how many calories are in a pizza"))

    assert parsed.intent is AgentIntent.UNKNOWN
    assert parsed.draft is None
    assert parsed.explanation is not None and "pizza" not in parsed.explanation


def test_spelled_out_half_is_not_truncated() -> None:
    planner = RuleBasedPlanner()

    for text, expected in (
        ("one and a half rotis", Decimal("1.5")),
        ("two and a half rotis", Decimal("2.5")),
        ("three and a half eggs", Decimal("3.5")),
        ("2 and a half parathas", Decimal("2.5")),
    ):
        parsed = planner.parse(_request(text))
        assert parsed.draft is not None, text
        assert parsed.draft.items[0].quantity == expected, text
        assert parsed.draft.items[0].confidence == EXPLICIT_CONFIDENCE, text


def test_a_bare_restatement_corrects_instead_of_logging_a_second_meal() -> None:
    logged = _meal("u", (("roti", Decimal("2")),))
    parsed = RuleBasedPlanner().parse(
        PlannerRequest(
            text="that was 3 rotis", occurred_at=NOW, timezone="UTC", recent_meals=(logged,)
        )
    )

    assert parsed.intent is AgentIntent.REVISE_MEAL
    assert parsed.reference is not None
    assert parsed.reference.food_hint == "roti"
    assert parsed.replace_items is False


def test_a_restatement_with_nothing_to_correct_is_still_a_new_meal() -> None:
    parsed = RuleBasedPlanner().parse(_request("it was 2 rotis for lunch"))

    assert parsed.intent is AgentIntent.LOG_MEAL


def test_a_correction_that_names_a_new_food_replaces_the_meal() -> None:
    logged = _meal("u", (("roti", Decimal("3")),))
    parsed = RuleBasedPlanner().parse(
        PlannerRequest(
            text="actually it was 2 dosas", occurred_at=NOW, timezone="UTC", recent_meals=(logged,)
        )
    )

    assert parsed.intent is AgentIntent.REVISE_MEAL
    assert parsed.replace_items is True
    # The new food cannot identify the meal to change, so it must not filter candidates.
    assert parsed.reference is not None and parsed.reference.food_hint is None


def test_an_additive_correction_still_merges_into_the_meal() -> None:
    logged = _meal("u", (("roti", Decimal("3")),))
    parsed = RuleBasedPlanner().parse(
        PlannerRequest(
            text="actually there was also a dosa",
            occurred_at=NOW,
            timezone="UTC",
            recent_meals=(logged,),
        )
    )

    assert parsed.intent is AgentIntent.REVISE_MEAL
    assert parsed.replace_items is False


def test_only_and_just_state_the_whole_meal() -> None:
    logged = _meal("u", (("roti", Decimal("3")),))
    parsed = RuleBasedPlanner().parse(
        PlannerRequest(
            text="actually it was only 2 dosas",
            occurred_at=NOW,
            timezone="UTC",
            recent_meals=(logged,),
        )
    )

    assert parsed.replace_items is True


def _meal(user_id: str, foods: tuple[tuple[str, Decimal], ...]) -> MealRecord:
    items = tuple(
        MealItemDraft(
            name=name,
            quantity=quantity,
            unit="piece",
            nutrition=Nutrition(calories=120, protein_g=4, carbs_g=24, fat_g=1),
        )
        for name, quantity in foods
    )
    return MealRecord(
        id=f"meal-{user_id}",
        user_id=user_id,
        meal_type=MealType.LUNCH,
        occurred_at=NOW,
        source_text="logged meal",
        created_at=NOW,
        items=items,
    )


def test_repeated_food_mention_counts_the_portion_once() -> None:
    parsed = RuleBasedPlanner().parse(_request("chicken tikka not butter chicken"))

    assert parsed.draft is not None
    assert [item.name for item in parsed.draft.items] == ["chicken"]
    assert parsed.draft.items[0].quantity == Decimal("1")


@pytest.mark.parametrize(
    "text",
    [
        "i had 0 rotis",
        "0.0 eggs",
        "i had 3 / 0 rotis",
        "i had -2 rotis",
        "i had 200 rotis",
        "i had twenty rotis",
        "i had 20 rotis and 25 rotis",
        "35 rice and 10 rice",
    ],
)
def test_impossible_quantities_ask_instead_of_fabricating_a_portion(text: str) -> None:
    parsed = RuleBasedPlanner().parse(_request(text))

    assert parsed.draft is not None, text
    item = parsed.draft.items[0]
    assert item.confidence == UNUSABLE_QUANTITY_CONFIDENCE, text
    assert item.quantity > 0, text


def test_amount_that_matches_no_food_lowers_confidence() -> None:
    parsed = RuleBasedPlanner().parse(_request("5 things with roti"))

    assert parsed.draft is not None
    assert parsed.draft.items[0].name == "roti"
    assert parsed.draft.items[0].confidence == UNUSABLE_QUANTITY_CONFIDENCE


def test_an_article_naming_an_occasion_is_not_read_as_a_second_portion() -> None:
    parsed = RuleBasedPlanner().parse(_request("had a banana for a snack"))

    assert parsed.draft is not None
    assert parsed.draft.items[0].name == "banana"
    assert parsed.draft.items[0].quantity == Decimal("1")
    assert parsed.draft.items[0].confidence == 0.8


def test_a_question_that_also_states_a_portion_logs_the_portion_it_stated() -> None:
    """A fast read must never swallow the meal that came with the question.

    "what did i eat today? i also had 2 parathas" asks for the list and states a plate in one
    breath. Reading it as only the question answered half the message and dropped two parathas
    without saying they were missing, which is the one failure this product cannot afford.
    """
    planner = RuleBasedPlanner()
    parsed = planner.parse(_request("what did I eat today? i also had 2 parathas for breakfast"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert [(item.name, item.quantity) for item in parsed.draft.items] == [
        ("paratha", Decimal("2"))
    ]
    # The same message on the fast path: it is not a pure read, so no planner route takes it.
    assert planner.deterministic_read(_request("what did I eat today? i also had 2 parathas")) is (
        None
    )


def test_an_amount_too_large_to_believe_still_leaves_the_read_path() -> None:
    """An unbelievable amount is still a claim, and a claim is refused out loud, not dropped."""
    parsed = RuleBasedPlanner().parse(_request("what did I eat today? i ate 50 rotis"))

    assert parsed.intent is AgentIntent.LOG_MEAL
    assert parsed.draft is not None
    assert parsed.draft.items[0].confidence < 0.55


def test_a_question_about_the_record_lists_it_instead_of_logging_what_it_asked_about() -> None:
    """ "did i eat biryani today?" is about the record, and answering it by logging a biryani would
    invent the very meal the user was asking about."""
    parsed = RuleBasedPlanner().parse(_request("did I eat biryani today?"))

    assert parsed.intent is AgentIntent.LIST_MEALS
    assert parsed.draft is None


def test_yesterday_meal_is_stamped_on_yesterday() -> None:
    parsed = RuleBasedPlanner().parse(_request("had 2 eggs for dinner yesterday"))

    assert parsed.draft is not None
    assert parsed.draft.occurred_at.astimezone(UTC).date() == NOW.date().replace(day=25)
    assert parsed.draft.meal_type is MealType.DINNER
