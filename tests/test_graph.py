from datetime import date
from decimal import Decimal

from calorai_agent.domain import AgentIntent, MealDraft, MealItemDraft, Nutrition, ParsedMessage
from calorai_agent.graph import MealAgent
from calorai_agent.planning import PlannerRequest
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools


class _ScriptedPlanner:
    """Answers every message with one fixed decision, to test the reply wiring."""

    def __init__(self, parsed: ParsedMessage) -> None:
        self._parsed = parsed

    def parse(self, request: PlannerRequest) -> ParsedMessage:
        return self._parsed


def _send(agent: MealAgent, clock, text: str, hour: int, minute: int = 0, day: int = 26) -> str:
    return agent.invoke("user-1", text, now=clock(hour, minute, day))


def test_logs_then_reports_current_totals(agent: MealAgent, clock) -> None:
    logged = _send(agent, clock, "had 2 parathas and chai for breakfast", 8)
    totals = _send(agent, clock, "how am I doing today?", 9)

    assert logged == "Logged 2 paratha, 1 milk chai — about 640 kcal and 15g protein."
    assert totals == "Today: 640 kcal, 15g protein, 94g carbs, and 22g fat across 1 meal."


def test_lists_today_meals_with_types(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 rotis for lunch", 13)

    assert (
        _send(agent, clock, "what did I eat today?", 14)
        == "Today you logged: lunch at 13:30: 2 roti."
    )


def test_empty_totals_are_friendly(agent: MealAgent, clock) -> None:
    assert _send(agent, clock, "calories today?", 8) == "Nothing logged today yet."


def test_totals_can_ask_about_yesterday(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 rotis", 8, day=25)

    assert _send(agent, clock, "how many calories today?", 9) == "Nothing logged today yet."
    assert _send(agent, clock, "how many calories yesterday?", 9).startswith("Yesterday: 240 kcal")


def test_correction_changes_the_original_meal_not_the_meal_count(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    _send(agent, clock, "2 rotis", 8)
    before = repository.totals_for_day("user-1", date(2026, 9, 26))

    response = _send(agent, clock, "actually that was 3 rotis not 2", 9, 30)
    after = repository.totals_for_day("user-1", date(2026, 9, 26))

    assert before.nutrition.calories == Decimal("240.00")
    assert response == "Updated your meal at 08:00 to 3 roti — now about 360 kcal and 12g protein."
    assert after.meal_count == 1
    assert after.nutrition.calories == Decimal("360.00")


def test_correction_keeps_untouched_items_of_the_same_meal(agent: MealAgent, clock) -> None:
    _send(agent, clock, "had 2 parathas and chai for breakfast", 8)

    response = _send(agent, clock, "actually it was 3 parathas", 9, 30)

    assert response == (
        "Updated your breakfast at 08:00 to 3 paratha, 1 milk chai — now about 900 kcal "
        "and 21g protein."
    )


def test_delete_removes_the_meal_from_totals(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 rotis", 8)

    assert _send(agent, clock, "delete that", 9, 30) == "Removed 2 roti (240 kcal)."
    assert _send(agent, clock, "calories today?", 10) == "Nothing logged today yet."


def test_same_as_yesterday_copies_yesterdays_meal(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 idlis and coffee for breakfast", 8, day=25)

    response = _send(agent, clock, "same as yesterday", 9)

    assert response.startswith("Logged the same as yesterday — 2 idli, 1 coffee")
    assert _send(agent, clock, "calories today?", 10).startswith("Today: 83 kcal")


def test_repeat_wording_with_a_named_food_logs_a_new_meal(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 rotis for lunch", 13)
    _send(agent, clock, "I had biryani again for dinner", 20)

    assert _send(agent, clock, "what are my totals today?", 21) == (
        "Today: 840 kcal, 30g protein, 130g carbs, and 22g fat across 2 meals."
    )


def test_ambiguous_correction_asks_one_question_and_changes_nothing(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    _send(agent, clock, "2 rotis and chicken for lunch", 13)
    _send(agent, clock, "3 rotis for dinner", 20, 15)
    before = repository.totals_for_day("user-1", date(2026, 9, 26))

    response = _send(agent, clock, "actually it was 4 rotis not 3", 21)
    after = repository.totals_for_day("user-1", date(2026, 9, 26))

    assert response.count("?") == 1
    assert "lunch" in response and "dinner" in response
    assert after.meal_count == before.meal_count == 2
    assert after.nutrition.calories == before.nutrition.calories


def test_materially_identical_candidates_resolve_without_asking(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    _send(agent, clock, "2 rotis for lunch", 13)
    _send(agent, clock, "2 rotis for dinner", 20)

    response = _send(agent, clock, "actually it was 3 rotis not 2", 21)

    assert response == (
        "Updated your dinner at 20:30 to 3 roti — now about 360 kcal and 12g protein."
    )
    assert repository.totals_for_day("user-1", date(2026, 9, 26)).nutrition.calories == Decimal(
        "600.00"
    )


def test_skipped_meal_invents_nothing(agent: MealAgent, clock, repository: MealRepository) -> None:
    assert _send(agent, clock, "skipped lunch", 13) == (
        "Noted — I have not logged a meal for that."
    )
    assert repository.list_for_day("user-1", date(2026, 9, 26)) == []


def test_vague_grazing_asks_instead_of_logging(agent: MealAgent, clock) -> None:
    response = _send(agent, clock, "skipped lunch but grazed all afternoon", 16)

    assert response.startswith("I have not logged anything yet.")
    assert response.count("?") == 1


def test_hedged_portion_is_logged_as_a_disclosed_estimate(agent: MealAgent, clock) -> None:
    response = _send(agent, clock, "leftover biryani, maybe two thirds of the box", 20)

    assert response.startswith("Logged about 0.7 serving of biryani — roughly 400 kcal")
    assert "estimate" in response


def test_unsupported_request_stays_useful(agent: MealAgent, clock) -> None:
    assert _send(agent, clock, "what should I eat tomorrow?", 9) == (
        "I couldn't identify a supported food yet. Try something like "
        "'had 2 parathas and chai for breakfast'."
    )


def test_references_survive_a_process_restart(make_agent, clock) -> None:
    make_agent().invoke("user-1", "2 rotis", now=clock(8))

    reopened = make_agent()
    response = reopened.invoke("user-1", "actually that was 3 rotis not 2", now=clock(9, 30))

    assert response == "Updated your meal at 08:00 to 3 roti — now about 360 kcal and 12g protein."


def test_a_correction_shortly_after_midnight_still_finds_tonights_last_meal(
    agent: MealAgent, clock
) -> None:
    _send(agent, clock, "3 idlis", 23, 40)

    response = _send(agent, clock, "actually that was 4 idlis", 0, 20, day=27)

    assert response == ("Updated your meal at 23:40 to 4 idli — now about 156 kcal and 8g protein.")


def test_a_pointer_with_an_explicit_day_never_falls_back_to_the_window(
    agent: MealAgent, clock
) -> None:
    _send(agent, clock, "2 rotis", 23, 50)

    response = _send(agent, clock, "actually yesterday it was 3 rotis", 0, 20, day=28)

    assert not response.startswith("Updated")
    assert "recent meal" in response or "couldn't find" in response


def test_an_impossible_portion_asks_and_logs_nothing(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    response = _send(agent, clock, "i had 0 rotis", 8)

    assert response.count("?") == 1
    assert "not confident" in response
    assert repository.list_for_day("user-1", date(2026, 9, 26)) == []


def test_denied_food_logs_nothing(agent: MealAgent, clock, repository: MealRepository) -> None:
    assert _send(agent, clock, "no eggs", 8) == "Noted — I have not logged a meal for that."
    assert repository.list_for_day("user-1", date(2026, 9, 26)) == []


def test_a_food_nutrition_question_logs_nothing(agent: MealAgent, clock) -> None:
    response = _send(agent, clock, "how many calories are in a pizza", 8)

    assert response.startswith("I can total the meals you have logged")


def test_an_ambiguous_deletion_asks_which_meal_to_remove(agent: MealAgent, clock) -> None:
    _send(agent, clock, "2 rotis and chicken for lunch", 13)
    _send(agent, clock, "3 rotis for dinner", 20, 15)

    response = _send(agent, clock, "delete that", 21)

    assert "deletion" in response
    assert "lunch" in response and "dinner" in response
    assert _send(agent, clock, "calories today?", 22).endswith("across 2 meals.")


def test_a_bare_restatement_corrects_rather_than_logging_twice(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    _send(agent, clock, "2 rotis", 8)

    response = _send(agent, clock, "that was 3 rotis", 9)
    totals = repository.totals_for_day("user-1", date(2026, 9, 26))

    assert response.startswith("Updated")
    assert totals.meal_count == 1
    assert totals.nutrition.calories == Decimal("360.00")


def test_a_correction_to_a_different_food_replaces_the_meal(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    _send(agent, clock, "3 rotis for lunch", 13, 30)

    response = _send(agent, clock, "actually it was 2 dosas", 14)

    assert response == (
        "Updated your lunch at 13:30 to 2 dosa — now about 336 kcal and 8g protein."
    )
    assert repository.totals_for_day("user-1", date(2026, 9, 26)).nutrition.calories == Decimal(
        "336.00"
    )


def test_an_additive_correction_keeps_the_foods_already_logged(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    _send(agent, clock, "3 rotis for lunch", 13, 30)

    _send(agent, clock, "actually there was also a dosa", 14)

    assert repository.totals_for_day("user-1", date(2026, 9, 26)).nutrition.calories == Decimal(
        "528.00"
    )


def test_a_day_less_pointer_does_not_rewrite_a_days_old_meal(
    agent: MealAgent, clock, repository: MealRepository
) -> None:
    _send(agent, clock, "2 eggs", 8, day=23)

    response = _send(agent, clock, "actually that was 5 eggs", 0, 20, day=26)

    assert not response.startswith("Updated")
    assert repository.totals_for_day("user-1", date(2026, 9, 23)).nutrition.calories == Decimal(
        "156.00"
    )


def test_clarification_names_only_the_uncertain_item(agent: MealAgent, clock) -> None:
    response = _send(agent, clock, "2 rotis and 5 things with eggs", 8)

    assert "roti" not in response
    assert "egg" in response
    assert response.count("?") == 1


def test_a_food_without_reference_data_is_named_instead_of_silently_missing(
    repository: MealRepository, clock
) -> None:
    parsed = ParsedMessage(
        intent=AgentIntent.LOG_MEAL,
        draft=MealDraft(
            occurred_at=clock(8),
            source_text="had 2 rotis and a dragonfruit",
            items=(
                MealItemDraft(
                    name="roti",
                    quantity=Decimal("2"),
                    unit="piece",
                    nutrition=Nutrition(
                        calories=Decimal(120),
                        protein_g=Decimal(4),
                        carbs_g=Decimal(20),
                        fat_g=Decimal(3),
                    ),
                ),
            ),
        ),
        unrecognized=("dragonfruit",),
    )
    agent = MealAgent(_ScriptedPlanner(parsed), MealTools(repository))

    response = agent.invoke("user-1", "had 2 rotis and a dragonfruit", now=clock(8))

    assert "Logged 2 roti" in response
    assert "dragonfruit" in response
