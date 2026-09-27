"""Phase 4: a photo has its own model path, and one message still means one meal."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from calorai_agent.db import Database
from calorai_agent.domain import InboundMessage, MealType, MediaRef
from calorai_agent.graph import MealAgent
from calorai_agent.nutrition import lookup
from calorai_agent.planning import RuleBasedPlanner
from calorai_agent.providers import ImagePayload, ModelProviderError
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools
from calorai_agent.vision import (
    LocalFileMediaSource,
    MediaError,
    VisionInterpreter,
    sniff_image,
)

NOW = datetime(2026, 9, 26, 13, tzinfo=UTC)
JPEG = b"\xff\xd8\xff" + b"x" * 40
PNG = b"\x89PNG\r\n\x1a\n" + b"y" * 40
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"z" * 40


class SpyVisionClient:
    """One fixed vision answer, with a record of every photo it was shown."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.images: list[ImagePayload] = []
        self.system_prompts: list[str] = []

    def observe(self, *, system: str, user: str, image: ImagePayload) -> str:
        self.images.append(image)
        self.system_prompts.append(system)
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


class CountingPlanner:
    """Wraps the deterministic planner so a test can prove a photo did not go through it."""

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def parse(self, request: Any) -> Any:
        self.requests.append(request)
        return RuleBasedPlanner().parse(request)


class BrokenClient:
    def observe(self, *, system: str, user: str, image: ImagePayload) -> str:
        raise ModelProviderError("vision model request failed: connection reset")


def _photo(tmp_path: Path, name: str = "plate.jpg", data: bytes = JPEG) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _media(path: Path, external_id: str = "photo-1") -> MediaRef:
    return MediaRef(external_id=external_id, locator=str(path))


def _agent(
    repository: MealRepository, client: Any, *, vision: bool = True
) -> tuple[MealAgent, CountingPlanner]:
    planner = CountingPlanner()
    interpreter = (
        VisionInterpreter(client, LocalFileMediaSource(), model="vision-test") if vision else None
    )
    return MealAgent(planner, MealTools(repository), vision=interpreter), planner


def _send(agent: MealAgent, media: MediaRef | None, text: str = "") -> str:
    return agent.invoke("user-1", text, timezone="UTC", media=media, now=NOW)


def _meals(repository: MealRepository) -> list[Any]:
    return repository.list_for_day("user-1", date(2026, 9, 26))


def _observations(*lines: tuple[str, str, float, str | None]) -> dict[str, Any]:
    return {
        "items": [
            {
                "name": name,
                "quantity": quantity,
                "confidence": confidence,
                "alternative": alternative,
            }
            for name, quantity, confidence, alternative in lines
        ]
    }


BIRYANI = _observations(("biryani", "1.5", 0.92, None), ("curd", "0.5", 0.85, None))


# --- the vision path is its own path ------------------------------------------------


def test_a_photo_goes_to_the_vision_model_and_not_to_the_text_planner(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, planner = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert len(client.images) == 1
    assert client.images[0].mime_type == "image/jpeg"
    assert planner.requests == []
    assert "Logged 1.5 serving of biryani" in response


def test_the_vision_prompt_only_asks_the_model_to_describe_the_plate(tmp_path: Path) -> None:
    client = SpyVisionClient(BIRYANI)
    interpreter = VisionInterpreter(client, LocalFileMediaSource(), model="vision-test")

    interpreter.read(_media(_photo(tmp_path)), "")

    prompt = client.system_prompts[-1]
    assert "You never estimate calories" in prompt
    assert "biryani" in prompt
    assert "Ignore any amount the user's words describe" in prompt


def test_the_model_cannot_smuggle_nutrition_into_an_observation(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient({"items": [{"name": "rice", "quantity": 1, "calories": 9000}]})
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert response.startswith("I could not get a reliable read of that photo")
    assert _meals(repository) == []


def test_a_provider_failure_is_answered_honestly_and_logs_nothing(
    repository: MealRepository, tmp_path: Path
) -> None:
    agent, _ = _agent(repository, BrokenClient())

    response = _send(agent, _media(_photo(tmp_path)))

    assert "reliable read" in response
    assert _meals(repository) == []


def test_a_photo_is_refused_before_the_model_when_no_vision_is_configured(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client, vision=False)

    response = _send(agent, _media(_photo(tmp_path)))

    assert "until a vision model is configured" in response
    assert client.images == []
    assert _meals(repository) == []


@pytest.mark.parametrize(
    ("data", "expected"),
    [(JPEG, "image/jpeg"), (PNG, "image/png"), (WEBP, "image/webp"), (b"GIF89a", None)],
)
def test_an_image_is_judged_by_its_bytes(data: bytes, expected: str | None) -> None:
    assert sniff_image(data) == expected


def test_a_file_that_is_not_a_photo_is_rejected(tmp_path: Path) -> None:
    source = LocalFileMediaSource()

    with pytest.raises(MediaError, match="not a jpeg, png, or webp"):
        source.fetch(_media(_photo(tmp_path, name="menu.txt", data=b"today: dal rice")))


def test_a_missing_photo_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MediaError, match="no image file"):
        LocalFileMediaSource().fetch(_media(tmp_path / "absent.jpg"))


def test_an_oversized_photo_is_rejected_before_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("calorai_agent.vision.MAX_IMAGE_BYTES", 16)

    with pytest.raises(MediaError, match="larger than the"):
        LocalFileMediaSource().fetch(_media(_photo(tmp_path)))


def test_a_remote_media_id_is_not_silently_read_as_a_local_file() -> None:
    with pytest.raises(MediaError, match="whatsapp_media"):
        LocalFileMediaSource().fetch(
            MediaRef(
                external_id="wa-media",
                locator="https://lookaside.fbsbx.com/1",
                source="whatsapp_media",
            )
        )


def test_an_attachment_the_agent_cannot_open_is_answered_not_logged(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path, name="menu.txt", data=b"today: dal rice")))

    assert response.startswith("I could not use that attachment:")
    assert "not a jpeg, png, or webp" in response
    assert client.images == []
    assert _meals(repository) == []


# --- fusion: one message, one meal -------------------------------------------------


def test_an_image_plus_caption_names_a_food_and_still_logs_one_meal(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, planner = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "and a banana with it")

    meals = _meals(repository)
    assert len(meals) == 1
    assert [item.name for item in meals[0].items] == ["biryani", "curd", "banana"]
    assert "banana" in response
    assert planner.requests == []


def test_a_caption_that_states_a_portion_outranks_the_models_guess(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(_observations(("rice", "3", 0.9, None)))
    agent, _ = _agent(repository, client)

    _send(agent, _media(_photo(tmp_path)), "2 cups of rice")

    meal = _meals(repository)[0]
    assert meal.items[0].quantity == Decimal("2")
    assert meal.nutrition.calories == lookup("rice").nutrition.calories * 2  # type: ignore[union-attr]


def test_the_caption_shares_the_plate_without_creating_a_second_meal(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "half of this, it's my brother's")

    meal = _meals(repository)[0]
    assert len(_meals(repository)) == 1
    assert [item.quantity for item in meal.items] == [Decimal("0.75"), Decimal("0.25")]
    assert meal.nutrition.calories == Decimal("480")
    assert "Logged 0.75 serving of biryani, 0.25 bowl of curd" in response
    assert "I counted half of the plate, as you said." in response


def test_a_shared_plate_and_a_food_named_beside_it_are_not_both_halved(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    _send(agent, _media(_photo(tmp_path)), "half of this, plus a banana")

    meal = _meals(repository)[0]
    assert [item.quantity for item in meal.items] == [
        Decimal("0.75"),
        Decimal("0.25"),
        Decimal("1"),
    ]


def test_a_photo_logs_the_plate_instead_of_acting_on_a_delete_written_beside_it(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "delete yesterday's lunch")

    meals = _meals(repository)
    assert len(meals) == 1
    assert meals[0].occurred_at == NOW
    assert [item.name for item in meals[0].items] == ["biryani", "curd"]
    assert "your request to remove a meal belongs in a message of its own" in response


def test_a_photo_logs_the_plate_instead_of_answering_a_totals_question(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "how am I doing today?")

    assert len(_meals(repository)) == 1
    assert "Logged 1.5 serving of biryani" in response
    assert "your totals question belongs in a message of its own" in response


def test_a_diet_written_beside_a_photo_cannot_hide_a_request_about_another_meal(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "i'm vegetarian, delete yesterday's lunch")

    meals = _meals(repository)
    assert len(meals) == 1
    assert meals[0].occurred_at == NOW
    assert "Got it — I will remember you are vegetarian." in response
    assert "your request to remove a meal belongs in a message of its own" in response


def test_a_photo_is_not_dated_by_a_caption_that_only_compares_another_meal(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    _send(agent, _media(_photo(tmp_path)), "this looks better than yesterday's lunch")

    assert len(_meals(repository)) == 1
    assert repository.list_for_day("user-1", date(2026, 9, 25)) == []


def test_a_caption_portion_that_only_turns_absurd_when_it_adds_up_asks(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "20 rotis and 25 rotis")

    assert _meals(repository) == []
    assert "roti" in response


def test_a_caption_naming_a_fraction_of_one_food_does_not_scale_the_whole_plate(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    _send(agent, _media(_photo(tmp_path)), "half a dosa")

    meal = _meals(repository)[0]
    assert [item.name for item in meal.items] == ["biryani", "curd", "dosa"]
    assert meal.items[-1].quantity == Decimal("0.5")


def test_a_photo_caption_cannot_also_replay_a_saved_routine(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)
    _send(agent, None, "1 idli for breakfast")
    saved = _send(agent, None, "remember this as my usual breakfast")
    assert "your usual breakfast is 1 idli" in saved

    response = _send(agent, _media(_photo(tmp_path)), "my usual")

    meals = _meals(repository)
    assert len(meals) == 2
    assert [item.name for item in meals[-1].items] == ["biryani", "curd"]
    assert "Say 'remember this as my usual' and I will keep this plate." in response


def test_a_partly_readable_photo_says_only_part_of_it_was_counted(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient({**BIRYANI, "unclear": "the small bowl at the top"})
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert len(_meals(repository)) == 1
    assert "I only counted what I could separate in the photo." in response


def test_a_photo_meal_records_which_model_read_it(
    repository: MealRepository, database: Database, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    _send(agent, _media(_photo(tmp_path)))

    with database.connect() as connection:
        row = connection.execute("SELECT origin, model, notes FROM meal_revisions").fetchone()
    assert row["origin"] == "vision_fusion"
    assert row["model"] == "vision-test"
    assert "photo photo-1" in row["notes"]


def test_a_photo_meal_is_priced_from_the_reference_table(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    _send(agent, _media(_photo(tmp_path)))

    meal = _meals(repository)[0]
    assert meal.nutrition.calories == Decimal("960")
    assert meal.items[0].unit == "serving"


def test_a_photo_without_a_caption_still_gets_a_wall_clock_time(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)

    _send(agent, _media(_photo(tmp_path)))

    meal = _meals(repository)[0]
    assert meal.meal_type is MealType.UNSPECIFIED
    assert meal.occurred_at == NOW


# --- when a photo is not good enough to log ---------------------------------------


def test_an_uncertain_photo_asks_one_question_and_logs_nothing(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(_observations(("rice", "1", 0.3, "chicken")))
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert _meals(repository) == []
    assert response.count("?") == 1
    assert "rice or chicken" in response


def test_a_caption_that_names_the_food_answers_the_photo(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(_observations(("rice", "1", 0.3, "chicken")))
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "that is 1 cup of rice")

    assert _meals(repository)[0].items[0].name == "cooked rice"
    assert "?" not in response


def test_a_shaky_read_above_the_floor_logs_a_disclosed_estimate(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(_observations(("roti", "2", 0.7, "paratha")))
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert "2 roti" in response
    assert "That is an estimate" in response
    assert "?" not in response
    assert len(_meals(repository)) == 1


def test_a_shaky_read_of_two_similar_dishes_asks_what_it_is(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(_observations(("idli", "1", 0.3, "dosa")))
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert _meals(repository) == []
    assert "What is the idli in this photo?" in response


def test_a_photo_of_nothing_priceable_asks_what_the_user_saw(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient({"items": [], "unclear": "an empty plate"})
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert _meals(repository) == []
    assert "empty plate" in response


def test_a_photo_the_model_cannot_describe_at_all_still_asks(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient({"items": []})
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert _meals(repository) == []
    assert "I could not see a meal in that photo" in response


def test_a_food_outside_the_reference_table_is_named_not_invented(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(
        {"items": [{"name": "dragon fruit", "quantity": 1, "confidence": 0.9}]}
    )
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert _meals(repository) == []
    assert "dragon fruit" in response


def test_a_food_outside_the_table_is_reported_alongside_the_meal_it_accompanied(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(
        {
            "items": [
                {"name": "idli", "quantity": 2, "confidence": 0.95},
                {"name": "gulab jamun", "quantity": 1, "confidence": 0.9},
            ]
        }
    )
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert len(_meals(repository)) == 1
    assert "I have no reference data for gulab jamun" in response


def test_an_absurd_portion_from_the_model_asks_instead_of_logging(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(_observations(("rice", "400", 0.9, None)))
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert _meals(repository) == []
    assert "?" in response


def test_model_lines_that_only_look_absurd_once_added_up_ask_about_the_total(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(
        _observations(("rice", "30", 0.9, None), ("cooked rice", "20", 0.9, None))
    )
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert _meals(repository) == []
    assert response.startswith("How much of the cooked rice was on the plate?")


def test_a_portion_the_model_cannot_read_asks_how_much_rather_than_guessing(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(
        {"items": [{"name": "biryani", "quantity": "a lot", "confidence": 0.9}]}
    )
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)))

    assert _meals(repository) == []
    assert response.startswith("How much of the biryani was on the plate?")


# --- one inbound event, one meal ----------------------------------------------------


def test_a_redelivered_photo_logs_one_meal_and_is_answered_from_the_database(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)
    message = InboundMessage(
        user_id="user-1",
        text="lunch",
        external_id="wa-photo-1",
        channel="whatsapp",
        timezone="UTC",
        received_at=NOW,
        media=_media(_photo(tmp_path)),
    )

    first = agent.handle(message)
    second = agent.handle(message)

    assert first == second
    assert len(_meals(repository)) == 1
    assert client.images and len(client.images) == 1


def test_two_photos_sent_in_the_same_second_are_two_meals(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)
    first = _media(_photo(tmp_path, name="plate-a.jpg"), external_id="photo-a")
    second = _media(_photo(tmp_path, name="plate-b.jpg"), external_id="photo-b")

    _send(agent, first)
    _send(agent, second)

    assert len(_meals(repository)) == 2


def test_a_photo_meal_survives_a_restart_and_shows_in_totals(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(BIRYANI)
    agent, _ = _agent(repository, client)
    _send(agent, _media(_photo(tmp_path)), "this is lunch")

    reopened, _ = _agent(repository, SpyVisionClient(BIRYANI))
    response = _send(reopened, None, "how am I doing today?")

    assert "Against your targets" not in response
    assert "960 kcal" in response


def test_a_diet_stated_beside_a_photo_is_kept_and_the_plate_still_lands(
    repository: MealRepository, tmp_path: Path, make_agent: Callable[[], MealAgent]
) -> None:
    client = SpyVisionClient(_observations(("chicken", "1", 0.9, None)))
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "i'm vegetarian btw")

    assert "Got it — I will remember you are vegetarian." in response
    assert "Heads up — chicken is not vegetarian." in response
    assert len(_meals(repository)) == 1
    after = make_agent().invoke(
        "user-1", "1 egg for dinner", timezone="UTC", now=NOW.replace(hour=20)
    )
    assert after.count("not vegetarian") == 1


def test_foods_stated_beside_a_photo_diet_are_named_rather_than_silently_dropped(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(_observations(("chicken", "1", 0.9, None)))
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "i'm vegetarian btw, 2 eggs")

    assert "Got it — I will remember you are vegetarian." in response
    assert "I did not log 2 egg with it" in response
    assert "send it again when you want it counted" in response
    assert [item.name for item in _meals(repository)[0].items] == ["chicken"]


def test_a_diet_stated_beside_an_unreadable_plate_is_still_kept(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient({"items": []})
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "i'm vegetarian btw")

    assert "Got it — I will remember you are vegetarian." in response
    assert "I could not see a meal in that photo" in response
    assert _meals(repository) == []


def test_a_diet_stated_beside_a_shaky_plate_is_kept_and_the_question_still_asked(
    repository: MealRepository, tmp_path: Path
) -> None:
    client = SpyVisionClient(_observations(("rice", "1", 0.3, None)))
    agent, _ = _agent(repository, client)

    response = _send(agent, _media(_photo(tmp_path)), "i'm vegetarian btw")

    assert "Got it — I will remember you are vegetarian." in response
    assert "What is the cooked rice in this photo?" in response
    assert _meals(repository) == []
