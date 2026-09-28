"""Phase 6: the fast paths, and the invariants they are not allowed to break.

A fast path is only worth taking if it answers the same way the slow one did. Each test here names
the work skipped — a model round trip, a table walk, a second download — and re-asserts that the
answer did not move.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from calorai_agent.app import EitherMediaSource
from calorai_agent.domain import AgentIntent, MediaRef
from calorai_agent.llm_planner import ModelPlanner
from calorai_agent.nutrition import ALIASES, FOODS, lookup
from calorai_agent.planning import PlannerRequest, RuleBasedPlanner
from calorai_agent.providers import ImagePayload, ModelProviderError
from calorai_agent.vision import MediaError
from calorai_agent.whatsapp import MediaCache

NOW = datetime(2026, 9, 26, 8, tzinfo=UTC)
PLAN = RuleBasedPlanner()


class RecordingClient:
    """The text model, counted. An empty list of prompts is the whole point of several tests."""

    def __init__(self, payload: Any = None) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return json.dumps(self.payload or {"intent": "acknowledge", "statement": "Noted."})


def _request(text: str) -> PlannerRequest:
    return PlannerRequest(text=text, occurred_at=NOW, timezone="UTC")


def _model_planner(payload: Any = None) -> tuple[ModelPlanner, RecordingClient]:
    client = RecordingClient(payload)
    return ModelPlanner(client, model="test-model"), client


# --- a totals question is not a language problem -------------------------------------


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("how am I doing today?", AgentIntent.GET_TOTALS),
        ("how many calories have I eaten", AgentIntent.GET_TOTALS),
        ("what did I eat today?", AgentIntent.LIST_MEALS),
        ("show meals", AgentIntent.LIST_MEALS),
    ],
)
def test_a_pure_read_is_recognised_without_a_model(text: str, intent: AgentIntent) -> None:
    assert PLAN.deterministic_read(_request(text)) == PLAN.parse(_request(text))
    assert PLAN.parse(_request(text)).intent is intent


@pytest.mark.parametrize(
    "text",
    [
        "had 2 parathas and chai for breakfast",
        "how many calories are in biryani",
        "i'm vegetarian btw",
        "aim for 120g protein a day",
        "actually that was 3 rotis",
        "my usual",
    ],
)
def test_anything_that_is_not_a_pure_read_is_left_to_the_planner(text: str) -> None:
    """A fast path that guesses is worse than a slow path: these all still need interpreting."""
    assert PLAN.deterministic_read(_request(text)) is None


def test_a_model_planner_answers_a_totals_question_without_calling_a_model() -> None:
    planner, client = _model_planner()

    parsed = planner.parse(_request("how am I doing today?"))

    assert parsed.intent is AgentIntent.GET_TOTALS
    assert client.calls == 0


def test_a_model_planner_still_asks_when_the_message_needs_reading() -> None:
    planner, client = _model_planner({"intent": "acknowledge", "statement": "Noted."})

    planner.parse(_request("leftover biryani, maybe two thirds of the box"))

    assert client.calls == 1


def test_totals_survive_a_provider_that_is_completely_down() -> None:
    """The fast path is also the only path that works with no network: a number, not a guess."""
    planner, client = _model_planner(ModelProviderError("connection refused"))

    parsed = planner.parse(_request("what did I eat today?"))

    assert parsed.intent is AgentIntent.LIST_MEALS
    assert client.calls == 0


# --- a reference table that never changes underneath a process ------------------------


def test_a_cached_lookup_returns_the_same_row_as_a_cold_one() -> None:
    assert lookup("parathas") is lookup("paratha")
    assert lookup("  Chapatis ") is lookup("roti")


def test_every_alias_still_resolves_to_the_row_it_always_did() -> None:
    for alias, target in ALIASES.items():
        assert lookup(alias) is FOODS[target]


def test_an_unpriceable_food_is_still_unpriceable_the_second_time() -> None:
    assert lookup("dragonfruit") is None
    assert lookup("dragonfruit") is None


# --- a photo downloaded once ----------------------------------------------------------


def _payload(byte: int) -> ImagePayload:
    return ImagePayload(data=bytes([byte]) * 10, mime_type="image/png")


def test_a_warmed_photo_is_read_from_the_cache_instead_of_the_network() -> None:
    calls: list[str] = []
    cache = MediaCache()

    def fetch() -> ImagePayload:
        calls.append("media-1")
        return _payload(1)

    assert cache.get_or_fetch("media-1", fetch) is cache.get_or_fetch("media-1", fetch)
    assert calls == ["media-1"]


def test_a_cold_cache_holds_only_the_newest_photos_it_can_afford() -> None:
    cache = MediaCache(max_bytes=25)

    for index in range(4):
        cache.get_or_fetch(f"media-{index}", lambda index=index: _payload(index))

    assert "media-0" not in cache
    assert "media-1" not in cache
    assert len(cache) == 2
    assert "media-3" in cache


def test_a_photo_that_will_not_warm_reports_instead_of_raising() -> None:
    cache = MediaCache()

    def broken() -> ImagePayload:
        raise MediaError("it could not be downloaded from WhatsApp")

    assert cache.warm("media-1", broken) is False
    assert len(cache) == 0
    assert cache.warm("media-1", lambda: _payload(2)) is True


# --- the pipeline that chooses where a photo comes from -------------------------------


class RecordingSource:
    def __init__(self) -> None:
        self.warmed: list[str] = []

    def fetch(self, media: MediaRef) -> ImagePayload:
        return _payload(3)

    def prefetch(self, media: MediaRef) -> None:
        self.warmed.append(media.locator)


class SilentSource:
    """A media source with no prefetch at all: a local file is already as warm as it gets."""

    def fetch(self, media: MediaRef) -> ImagePayload:
        return _payload(4)


def _media(source: str, locator: str) -> MediaRef:
    return MediaRef(external_id=f"event-{locator}", locator=locator, source=source)


def test_a_whatsapp_photo_is_warmed_through_the_source_that_owns_it() -> None:
    recorder = RecordingSource()
    pipeline = EitherMediaSource(local=SilentSource(), whatsapp=recorder)

    pipeline.prefetch(_media("whatsapp_media", "media-9"))

    assert recorder.warmed == ["media-9"]


def test_a_local_photo_is_never_warmed_and_nothing_is_said_about_it() -> None:
    recorder = RecordingSource()
    pipeline = EitherMediaSource(local=SilentSource(), whatsapp=recorder)

    pipeline.prefetch(_media("local_path", "/tmp/plate.jpg"))

    assert recorder.warmed == []


def test_a_source_that_cannot_warm_is_simply_not_asked_to() -> None:
    pipeline = EitherMediaSource(local=SilentSource(), whatsapp=SilentSource())

    pipeline.prefetch(_media("whatsapp_media", "media-9"))
