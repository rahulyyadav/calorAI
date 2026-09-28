"""Phase 6e: the turn when the model, the network, or Meta misbehaves.

These run against a real HTTP client whose transport is scripted to fail, so every fallback is
exercised through the error types a provider actually produces (timeouts, refused connections,
429s, truncated JSON) rather than through a hand-written exception. Malformed webhook bodies are
held by `tests/test_whatsapp_webhook.py`; this file owns what happens underneath them: a dead
provider, an unreadable photo, and the same delivery arriving twice at the same moment.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from calorai_agent.domain import InboundMessage, MediaRef
from calorai_agent.graph import MealAgent
from calorai_agent.llm_planner import ModelPlanner
from calorai_agent.observability import FIELDS_ATTR
from calorai_agent.providers import OpenAICompatibleClient, OpenAICompatibleVisionClient
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools
from calorai_agent.vision import LocalFileMediaSource, VisionInterpreter

DAY = datetime(2026, 9, 26, 8, tzinfo=UTC)
MEAL_MESSAGE = "had 2 parathas and chai for breakfast"
PNG = b"\x89PNG\r\n\x1a\n" + b"a plate" * 8
Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key", model="test-model", transport=httpx.MockTransport(handler)
    )


def _text_agent(repository: MealRepository, handler: Handler) -> MealAgent:
    client = _client(handler)
    return MealAgent(ModelPlanner(client, model="test-model"), MealTools(repository))


def _vision_agent(repository: MealRepository, handler: Handler) -> MealAgent:
    """The plate goes to the vision client; the caption would go to the text one."""
    return MealAgent(
        ModelPlanner(_client(handler), model="test-model"),
        MealTools(repository),
        vision=VisionInterpreter(
            OpenAICompatibleVisionClient(
                api_key="test-key", model="test-model", transport=httpx.MockTransport(handler)
            ),
            LocalFileMediaSource(),
            model="test-model",
        ),
    )


def _photo(tmp_path: Path, name: str = "plate.png", body: bytes = PNG) -> MediaRef:
    path = tmp_path / name
    path.write_bytes(body)
    return MediaRef(external_id=f"wamid.{name}", locator=str(path))


def _meals(repository: MealRepository) -> list[Any]:
    return repository.list_for_day("user-1", DAY.date(), "UTC")


def _event_count(repository: MealRepository) -> int:
    with repository.database.connect() as connection:
        row = connection.execute("SELECT COUNT(*) AS n FROM inbound_events").fetchone()
    return int(row["n"])


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.TimeoutException("the provider took all day", request=request)


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no route to the model host", request=request)


def _status(code: int) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"error": {"message": "no"}})

    return handler


def _content(value: Any) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": value}}]})

    return handler


def _envelope(body: Any) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    return handler


PROVIDER_FAILURES: dict[str, Handler] = {
    "a request that times out": _timeout,
    "a refused connection": _refused,
    "an http 500": _status(500),
    "an http 429": _status(429),
    "prose instead of json": _content("I think you meant two parathas?"),
    "an envelope with no choices": _envelope({"choices": []}),
}

# A model that answers with a number of its own: the reference table still prices the meal.
INVENTED_NUTRITION = _content(
    json.dumps(
        {
            "intent": "log_meal",
            "items": [{"name": "paratha", "quantity": 2, "confidence": 0.9, "calories": 9000}],
        }
    )
)


@pytest.mark.parametrize("handler", list(PROVIDER_FAILURES.values()), ids=list(PROVIDER_FAILURES))
def test_a_provider_outage_costs_the_answer_nothing(
    repository: MealRepository, handler: Handler
) -> None:
    """The rules read the same words, so a down model degrades the parse, never the data."""
    agent = _text_agent(repository, handler)

    reply = agent.invoke("user-1", MEAL_MESSAGE, now=DAY)

    assert reply.startswith("Logged 2 paratha")
    meals = _meals(repository)
    assert len(meals) == 1
    assert meals[0].nutrition.calories == Decimal("640")


def test_a_model_that_invents_a_number_cannot_put_it_in_the_database(
    repository: MealRepository,
) -> None:
    """The model may name foods; only the reference table prices them."""
    agent = _text_agent(repository, INVENTED_NUTRITION)

    reply = agent.invoke("user-1", MEAL_MESSAGE, now=DAY)

    meals = _meals(repository)
    assert len(meals) == 1
    assert meals[0].nutrition.calories == Decimal("520")
    assert "9000" not in reply


def test_the_trace_names_the_provider_failure_it_survived(repository: MealRepository) -> None:
    seen: list[dict[str, Any]] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            fields = getattr(record, FIELDS_ATTR, None)
            if fields is not None:
                seen.append({"event": record.getMessage(), **fields})

    logger = logging.getLogger("calorai_agent")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers, logger.level, logger.propagate = [Collect()], logging.DEBUG, False
    try:
        _text_agent(repository, _timeout).invoke("user-1", MEAL_MESSAGE, now=DAY)
    finally:
        logger.handlers, logger.level, logger.propagate = saved

    failed = [row for row in seen if row["event"] == "model_request"]
    assert [row["status"] for row in failed] == ["error"]
    assert failed[0]["failed"] == "TimeoutException"


def test_a_down_provider_never_touches_the_totals_question(repository: MealRepository) -> None:
    """The fast path is what keeps "how am I doing?" working with no network at all."""
    calls = 0

    def counting(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.TimeoutException("would have been slow", request=request)

    agent = _text_agent(repository, counting)
    agent.invoke("user-1", MEAL_MESSAGE, now=DAY)
    calls = 0

    reply = agent.invoke("user-1", "how am I doing on calories?", now=DAY)

    assert calls == 0
    assert "640 kcal" in reply


def test_a_dead_vision_model_logs_no_meal_and_says_so(
    repository: MealRepository, tmp_path: Path
) -> None:
    agent = _vision_agent(repository, _timeout)

    reply = agent.invoke("user-1", "", now=DAY, media=_photo(tmp_path))

    assert "could not get a reliable read" in reply
    assert _meals(repository) == []


def test_a_vision_model_answering_in_prose_is_refused_not_guessed(
    repository: MealRepository, tmp_path: Path
) -> None:
    agent = _vision_agent(repository, _content("looks like a nice plate!"))

    reply = agent.invoke("user-1", "", now=DAY, media=_photo(tmp_path))

    assert _meals(repository) == []
    assert "could not get a reliable read" in reply


def test_a_rejected_vision_answer_does_not_put_the_plate_in_the_log(
    repository: MealRepository, tmp_path: Path, caplog
) -> None:
    """A validation error quotes the value it rejected, and that value is someone's dinner.

    The model here answers in the wrong field type, so pydantic would print the dish description
    verbatim in its error. The failure is worth logging — how often a read comes back malformed is
    the reason to change a prompt — the words in it are not.
    """
    plate = "a large portion of butter chicken with two naan, ghee visible"
    agent = _vision_agent(
        repository,
        _content(json.dumps({"items": [{"name": plate, "quantity": 1, "confidence": plate}]})),
    )

    with caplog.at_level(logging.WARNING, logger="calorai_agent"):
        reply = agent.invoke("user-1", "", now=DAY, media=_photo(tmp_path))

    assert "could not get a reliable read" in reply
    assert _meals(repository) == []
    assert plate not in caplog.text
    assert "ValidationError" in caplog.text


def test_a_corrupt_photo_costs_no_provider_request(
    repository: MealRepository, tmp_path: Path
) -> None:
    """The bytes are checked before they are billed: a non-image never reaches the model."""
    calls = 0

    def counting(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    agent = _vision_agent(repository, counting)
    junk = _photo(tmp_path, "invoice.png", b"%PDF-1.4 this is not a plate")

    reply = agent.invoke("user-1", "", now=DAY, media=junk)

    assert calls == 0
    assert "could not use that attachment" in reply
    assert _meals(repository) == []


def test_the_same_delivery_racing_itself_logs_one_meal(repository: MealRepository) -> None:
    """Two workers, one message id: the ledger decides, not the clock."""

    def deliver(replies: list[str], index: int) -> None:
        agent = _text_agent(repository, _timeout)
        replies[index] = agent.handle(
            InboundMessage(
                user_id="user-1",
                text=MEAL_MESSAGE,
                external_id="wamid.SAME",
                channel="whatsapp",
                timezone="UTC",
                received_at=DAY,
            )
        )

    replies = ["", ""]
    threads = [threading.Thread(target=deliver, args=(replies, i)) for i in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads)
    assert len(_meals(repository)) == 1
    assert _event_count(repository) == 1
    # Both threads answered, and each answer is one a user can act on. The loser of the race is
    # allowed to say the message is already in hand — that is the truthful report of a turn the
    # other thread is still running — but "understood" or silence would not be.
    assert all(reply for reply in replies), replies
    assert all(
        reply.startswith(("Logged", "I already received that message")) for reply in replies
    ), replies
