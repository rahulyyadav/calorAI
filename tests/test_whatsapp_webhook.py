"""Phase 5: a signed delivery is acknowledged first and answered second, with one meal per message.

These run the real graph over a real SQLite file and a real webhook body — only Meta's servers are
stood in for, so the same call the app makes in production is the call asserted here.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

import httpx
import pytest

from calorai_agent.app import build_graph_client, create_whatsapp_app
from calorai_agent.config import Settings
from calorai_agent.domain import MediaRef
from calorai_agent.graph import MealAgent
from calorai_agent.observability import FIELDS_ATTR, current_trace, trace_scope
from calorai_agent.planning import RuleBasedPlanner
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools
from calorai_agent.vision import VisionInterpreter
from calorai_agent.whatsapp import (
    GraphClient,
    MediaCache,
    WhatsAppError,
    WhatsAppMediaSource,
    normalize_webhook,
    sender_for_log,
)
from calorai_agent.whatsapp_server import (
    INTERNAL_FAILURE_REPLY,
    InlineExecutor,
    WebhookApplication,
    WorkerExecutor,
    serve,
)

WA_USER = "919812345678"
STRANGER = "919999999999"
SECRET = "app-secret"
VERIFY = "verify-token"
TS = 1_790_000_000
ZONE = "Asia/Kolkata"
DAY = datetime.fromtimestamp(TS, UTC).date()
PNG = b"\x89PNG\r\n\x1a\n" + b"y" * 40
BIRYANI = {
    "items": [
        {"name": "biryani", "quantity": "1.5", "confidence": 0.92, "alternative": None},
        {"name": "curd", "quantity": "0.5", "confidence": 0.85, "alternative": None},
    ]
}
ALLOW_ENV = (
    "CALORAI_WHATSAPP_VERIFY_TOKEN",
    "CALORAI_WHATSAPP_APP_SECRET",
    "CALORAI_WHATSAPP_ACCESS_TOKEN",
    "CALORAI_WHATSAPP_PHONE_NUMBER_ID",
    "CALORAI_WHATSAPP_ALLOWED_USERS",
    "CALORAI_TEXT_MODEL_API_KEY",
    "OPENAI_API_KEY",
    "CALORAI_VISION_MODEL_API_KEY",
    "CALORAI_DB_PATH",
    "CALORAI_PLANNER",
)


class FakeGraph(GraphClient):
    """The four Graph calls this transport makes, recorded instead of sent."""

    def __init__(
        self,
        *,
        photo: bytes | None = PNG,
        downloads_fail: bool = False,
        crash: Exception | None = None,
        sends_fail: bool = False,
        receipts_fail: bool = False,
        delay: float = 0.0,
    ) -> None:
        super().__init__(
            access_token="token", phone_number_id="1000", base_url="https://graph.test"
        )
        self.photo = photo
        self.downloads_fail = downloads_fail
        # A failure the media reader does not model, raised from underneath it.
        self.crash = crash
        self.sends_fail = sends_fail
        self.receipts_fail = receipts_fail
        self.delay = delay
        self.replies: list[tuple[str, str]] = []
        self.statuses: list[str] = []
        self.receipt_ids: list[str] = []
        self.download_calls: list[str] = []

    def send_text(self, to: str, body: str) -> None:
        if self.sends_fail:
            raise WhatsAppError("graph request failed: ConnectError")
        self.replies.append((to, body))

    def mark_seen(self, message_id: str, *, typing: bool = False) -> None:
        if self.receipts_fail:
            raise WhatsAppError("graph request failed: ConnectError")
        self.receipt_ids.append(message_id)
        self.statuses.append("read+typing" if typing else "read")

    def download(self, media_id: str) -> bytes:
        self.download_calls.append(media_id)
        if self.crash is not None:
            raise self.crash
        if self.delay:
            time.sleep(self.delay)
        if self.downloads_fail:
            raise WhatsAppError(f"media {media_id} came back with no download url")
        assert self.photo is not None
        return self.photo


class SlowGraph(FakeGraph):
    """A Graph API that takes a while to hand a photo over, which is the normal one."""

    def __init__(self, delay: float = 0.2) -> None:
        super().__init__(delay=delay)


class SpyVision:
    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.images: list[bytes] = []

    def observe(self, *, system: str, user: str, image: Any) -> str:
        self.images.append(image.data)
        return json.dumps(self.answer)


class BoomPlanner:
    def parse(self, request: Any) -> Any:
        raise RuntimeError("the model endpoint went away")


class QueuedExecutor:
    """Defers work the way a thread pool does, so a test can see what the request thread did."""

    def __init__(self) -> None:
        self.pending: list[Callable[[], None]] = []

    def submit(self, work: Callable[[], None]) -> None:
        self.pending.append(work)


def _body(*items: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "100",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {"messaging_product": "whatsapp", "messages": list(items)},
                        }
                    ],
                }
            ],
        }
    ).encode()


def _signed(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _text(body: str, *, msg_id: str = "wamid.TXT", sender: str = WA_USER) -> dict[str, Any]:
    return {
        "from": sender,
        "id": msg_id,
        "timestamp": str(TS),
        "type": "text",
        "text": {"body": body},
    }


def _photo(
    *,
    msg_id: str = "wamid.IMG",
    caption: str = "",
    sender: str = WA_USER,
    media_id: str = "media-1",
) -> dict[str, Any]:
    image: dict[str, Any] = {"id": media_id}
    if caption:
        image["caption"] = caption
    return {"from": sender, "id": msg_id, "timestamp": str(TS), "type": "image", "image": image}


def _receipt(*, msg_id: str = "wamid.OURS") -> dict[str, Any]:
    return {"status": "delivered", "id": msg_id, "timestamp": str(TS)}


def _app(
    repository: MealRepository,
    graph: FakeGraph | None = None,
    *,
    planner: Any = None,
    vision_answer: Any = BIRYANI,
    allowed: tuple[str, ...] = (WA_USER,),
    executor: Any = None,
    prefetcher: Any = None,
) -> tuple[WebhookApplication, FakeGraph]:
    client = graph if graph is not None else FakeGraph()
    # One media source, shared by the delivery that warms photos and the pipeline that reads them.
    media = WhatsAppMediaSource(client, cache=MediaCache())
    vision = VisionInterpreter(SpyVision(vision_answer), media, model="vision-test")
    agent = MealAgent(
        planner if planner is not None else RuleBasedPlanner(),
        MealTools(repository),
        vision=vision,
    )
    application = WebhookApplication(
        agent,
        client,
        verify_token=VERIFY,
        app_secret=SECRET,
        repository=repository,
        timezone=ZONE,
        allowed_users=frozenset(allowed),
        # Tests want the answer inside the call they just made; the production default is a worker,
        # and `test_the_production_answerer_runs_behind_the_acknowledgement` holds that open.
        executor=executor if executor is not None else InlineExecutor(),
        media=prefetcher if prefetcher is not None else media,
    )
    return application, client


def _deliver(application: WebhookApplication, *items: dict[str, Any]) -> tuple[int, str]:
    body = _body(*items)
    return application.post(_signed(body), body)


def _events(repository: MealRepository, user_id: str = WA_USER) -> list[Any]:
    database = repository.database
    with database.connect() as connection:
        return list(
            connection.execute(
                "SELECT external_id, response_text FROM inbound_events WHERE user_id = ?"
                " ORDER BY external_id",
                (user_id,),
            ).fetchall()
        )


# --- the handshake and the signature -------------------------------------------------


def test_the_handshake_echoes_metas_challenge_for_our_verify_token(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)
    query = parse_qs(f"hub.mode=subscribe&hub.verify_token={VERIFY}&hub.challenge=1158601234")

    assert application.get(query) == (200, "1158601234")
    assert graph.replies == []


def test_a_handshake_with_the_wrong_verify_token_is_refused(
    repository: MealRepository,
) -> None:
    application, _ = _app(repository)
    query = parse_qs("hub.mode=subscribe&hub.verify_token=guessed&hub.challenge=1158601234")

    status, _ = application.get(query)
    assert status == 403


@pytest.mark.parametrize(
    ("signature", "phrase"),
    [
        (None, "invalid signature"),
        ("", "invalid signature"),
        ("sha256=" + "0" * 64, "invalid signature"),
        ("sha1=abc", "invalid signature"),
    ],
)
def test_an_event_that_cannot_be_proven_as_metas_is_refused_and_answers_nothing(
    repository: MealRepository, signature: str | None, phrase: str
) -> None:
    application, graph = _app(repository)
    body = _body(_text("I ate 6 biryani"))

    status, text = application.post(signature, body)

    assert text == phrase
    assert graph.replies == []
    assert repository.list_for_day(WA_USER, DAY) == []


def test_a_signed_body_that_is_not_json_is_acknowledged_so_it_stops_retrying(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)
    body = b"this is not the webhook you are looking for"

    status, text = application.post(_signed(body), body)

    assert (status, text) == (200, "ok")
    assert graph.replies == []


# --- answering a text message --------------------------------------------------------


def test_a_signed_text_message_is_logged_and_answered(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)

    status, _ = _deliver(application, _text("I ate 2 dosa and 1 coffee"))

    assert status == 200
    assert len(graph.replies) == 1
    to, reply = graph.replies[0]
    assert to == WA_USER
    assert "Logged" in reply
    assert "dosa" in reply


def test_the_day_the_user_sent_it_decides_the_day_the_meal_lands_on(
    repository: MealRepository,
) -> None:
    # A webhook can be delivered seconds or hours after the send; the meal belongs to the plate's
    # own moment, not to whenever Meta got around to retrying.
    application, _ = _app(repository)
    _deliver(application, _text("1 dosa and 1 coffee"))

    logged = repository.list_for_day(WA_USER, DAY, ZONE)
    assert [meal.occurred_at.astimezone(ZoneInfo(ZONE)).date() for meal in logged] == [DAY]
    assert repository.list_for_day(WA_USER, DAY - timedelta(days=1), ZONE) == []


def test_a_read_receipt_and_typing_indicator_are_sent_before_the_reply(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)
    _deliver(application, _text("1 idli"))

    assert graph.statuses == ["read+typing"]
    assert graph.receipt_ids == ["wamid.TXT"]
    assert len(graph.replies) == 1


def test_a_failed_read_receipt_does_not_cost_the_user_their_answer(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository, FakeGraph(receipts_fail=True))
    _deliver(application, _text("1 idli"))

    assert len(graph.replies) == 1
    assert "Logged" in graph.replies[0][1]


def test_a_reply_that_cannot_be_sent_does_not_stop_the_meal_being_logged(
    repository: MealRepository,
    caplog,
) -> None:
    application, _ = _app(repository, FakeGraph(sends_fail=True))

    with caplog.at_level(logging.WARNING, logger="calorai_agent"):
        status, _ = _deliver(application, _text("1 idli"))

    assert status == 200
    assert len(repository.list_for_day(WA_USER, DAY)) == 1
    # The one line that names the failed reply names them by digest, the way a refusal does.
    assert WA_USER not in caplog.text
    assert sender_for_log(WA_USER, SECRET) in caplog.text


# --- one delivery, one meal ----------------------------------------------------------


def test_a_redelivered_message_id_is_answered_from_the_ledger_not_a_second_meal(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)
    body = _body(_text("I ate 2 dosa"))
    signature = _signed(body)

    application.post(signature, body)
    application.post(signature, body)

    assert len(repository.list_for_day(WA_USER, DAY)) == 1
    assert len(graph.replies) == 2
    assert graph.replies[0][1] == graph.replies[1][1]


def test_two_photos_sent_in_the_same_second_are_two_meals(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)

    _deliver(application, _photo(msg_id="wamid.A"))
    _deliver(application, _photo(msg_id="wamid.B"))

    meals = repository.list_for_day(WA_USER, DAY)
    assert len(meals) == 2
    assert len(graph.replies) == 2


def test_work_is_handed_to_the_worker_so_the_request_thread_only_acknowledges(
    repository: MealRepository,
) -> None:
    executor = QueuedExecutor()
    application, graph = _app(repository, executor=executor)

    status, _ = _deliver(application, _text("1 idli"))

    assert status == 200
    assert graph.replies == []
    assert executor.pending

    executor.pending[0]()
    assert "Logged" in graph.replies[0][1]


def test_a_delivery_with_only_a_receipt_asks_nothing_of_the_worker(
    repository: MealRepository,
) -> None:
    executor = QueuedExecutor()
    application, graph = _app(repository, executor=executor)

    status, _ = _deliver(application, _receipt())

    assert (status, graph.replies, executor.pending) == (200, [], [])
    assert _events(repository) == []
    assert graph.statuses == []


# --- photos through the transport ----------------------------------------------------


def test_a_photo_is_downloaded_by_its_media_id_and_the_plate_is_logged(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)

    status, _ = _deliver(application, _photo(caption="my lunch"))

    assert status == 200
    assert graph.download_calls == ["media-1"]
    assert graph.replies[0][1].startswith("Logged 1.5 serving of biryani")
    meal = repository.list_for_day(WA_USER, DAY)[0]
    assert [item.name for item in meal.items] == ["biryani", "curd"]


def test_a_photo_that_cannot_be_downloaded_is_answered_honestly(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository, FakeGraph(downloads_fail=True))

    _deliver(application, _photo())

    assert repository.list_for_day(WA_USER, DAY) == []
    assert len(graph.replies) == 1
    assert "could not be downloaded" in graph.replies[0][1].lower()


def test_a_photo_that_is_not_an_image_names_the_file(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository, FakeGraph(photo=b"%PDF-1.4" + b"x" * 40))

    _deliver(application, _photo())

    assert repository.list_for_day(WA_USER, DAY) == []
    assert "not a jpeg, png, or webp" in graph.replies[0][1]


def test_a_caption_beside_the_photo_is_read_as_a_description_of_the_plate(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)

    _deliver(application, _photo(caption="half of this"))

    meal = repository.list_for_day(WA_USER, DAY)[0]
    assert str(meal.items[0].quantity) == "0.75"
    assert "half" in graph.replies[0][1].lower() or "0.75" in graph.replies[0][1]


def test_two_photos_in_one_delivery_are_downloaded_while_the_first_is_still_reading(
    repository: MealRepository,
) -> None:
    """Photos are fetched side by side, because waiting for them in turn is lost latency.

    Each plate is then read on its own user's thread in order, so a correction can never arrive
    before the meal it corrects. What overlaps is the part with no conversation in it: the bytes.
    """
    application, graph = _app(repository, SlowGraph(0.2))

    started = time.perf_counter()
    _deliver(
        application,
        _photo(msg_id="wamid.A", media_id="media-1", caption="my lunch"),
        _photo(msg_id="wamid.B", media_id="media-2", caption="my dinner"),
    )
    elapsed = time.perf_counter() - started

    assert sorted(graph.download_calls) == ["media-1", "media-2"]
    assert len(repository.list_for_day(WA_USER, DAY)) == 2
    # Two round trips back to back would have cost at least 0.4 seconds.
    assert elapsed < 0.35


def test_a_photo_one_message_downloaded_is_not_paid_for_twice(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)

    _deliver(application, _photo(msg_id="wamid.A"), _photo(msg_id="wamid.B"))

    assert graph.download_calls == ["media-1"]
    assert len(repository.list_for_day(WA_USER, DAY)) == 2


def test_a_photo_that_will_not_warm_still_tells_the_user_the_truth(
    repository: MealRepository,
) -> None:
    """A failed prefetch is not a failure: the turn fetches again and answers truthfully."""
    application, graph = _app(repository, FakeGraph(downloads_fail=True))

    _deliver(application, _photo())

    assert graph.download_calls == ["media-1", "media-1"]
    assert "could not be downloaded" in graph.replies[0][1].lower()


def test_a_text_delivery_warms_nothing_at_all(repository: MealRepository) -> None:
    application, graph = _app(repository)

    _deliver(application, _text("had two dosas"))

    assert graph.download_calls == []


def test_a_download_that_fails_for_a_reason_nobody_modelled_still_answers_the_delivery(
    repository: MealRepository,
) -> None:
    """An unforeseen crash in a warm-up cannot be allowed to delete a delivery.

    `MediaError` is the failure the media reader raises deliberately; a base URL without a scheme
    raises `httpx.InvalidURL` from underneath it. Meta has already been promised a 200 by then, so
    work that died at that point would leave both messages unanswered with no retry ever coming —
    and the text message had nothing to do with the photo.
    """
    application, graph = _app(repository, FakeGraph(crash=httpx.InvalidURL("no scheme")))

    _deliver(application, _text("had two dosas"), _photo(msg_id="wamid.P"))

    bodies = [reply for _, reply in graph.replies]
    assert len(bodies) == 2
    assert any("dosa" in body.lower() for body in bodies)
    assert INTERNAL_FAILURE_REPLY in bodies
    assert len(repository.list_for_day(WA_USER, DAY)) == 1


class TracedPrefetcher:
    """A warm-up that reports which trace and thread it ran on, and overlaps with its sibling."""

    def __init__(self) -> None:
        self.traces: list[str] = []
        self.threads: list[int] = []

    def prefetch(self, media: MediaRef) -> bool:
        self.traces.append(current_trace())
        self.threads.append(threading.get_ident())
        time.sleep(0.01)
        return True


def test_photos_warmed_off_the_delivery_thread_keep_its_trace(repository: MealRepository) -> None:
    """A worker thread inherits no contextvar, so the fetch it logs would be uncorrelated.

    The parallel warm-up exists to overlap two downloads; if that is where a photo read fails, the
    line has to belong to the delivery that asked for it or the trace stops being a trace.
    """
    application, _graph = _app(repository)
    prefetcher = TracedPrefetcher()
    application.media = prefetcher
    payload = json.loads(_body(_photo(msg_id="wamid.A"), _photo(msg_id="wamid.B")))
    batch, _refused = normalize_webhook(payload, timezone=ZONE, allowed_users=frozenset({WA_USER}))

    with trace_scope("delivery-trace"):
        application.prefetch(batch)

    assert prefetcher.traces == ["delivery-trace", "delivery-trace"]
    assert len(set(prefetcher.threads)) == 2


def test_the_delivery_reports_how_many_photos_the_warm_up_actually_got(
    repository: MealRepository,
) -> None:
    """One line, however many plates: asking for three photos and getting one is a fact."""
    events: list[dict[str, Any]] = []
    handler = _FieldHandler(events)
    loggers = logging.getLogger("calorai_agent")
    was = loggers.level
    loggers.setLevel(logging.INFO)
    loggers.addHandler(handler)
    try:
        application, _graph = _app(repository)
        _deliver(application, _photo(msg_id="wamid.A"), _photo(msg_id="wamid.B"))
        _deliver(application, _photo(msg_id="wamid.C", media_id="media-3"))
    finally:
        loggers.removeHandler(handler)
        loggers.setLevel(was)

    assert [row for row in events if row["event"] == "media_prefetched"] == [
        {"event": "media_prefetched", "photos": 2, "warmed": 2},
        {"event": "media_prefetched", "photos": 1, "warmed": 1},
    ]


class _FieldHandler(logging.Handler):
    """Collects the structured fields a log line carried, the way an aggregator would read them."""

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        fields = getattr(record, FIELDS_ATTR, None)
        if isinstance(fields, dict):
            self.sink.append({"event": record.getMessage(), **fields})


class _CrashingPrefetcher:
    """A media source that raises something the shipped reader never lets escape.

    `MediaCache.warm` turns every failure into a logged warning, so no photo can end a delivery
    through it. This stands in for the one thing that still can: a `MediaPrefetcher` handed to the
    delivery is a public seam, and whatever it throws, it throws outside the per-message safety net.
    """

    def prefetch(self, media: MediaRef) -> bool:
        raise httpx.InvalidURL("no scheme")


def test_a_delivery_lost_after_the_acknowledgement_names_its_trace_and_nothing_else(
    repository: MealRepository,
) -> None:
    """The future's callback can only report a class name; the delivery line reports the trace.

    A worker that dies behind the 200 leaves the user waiting with no retry coming, so the one line
    that survives has to be enough to find the turn — and not enough to read the plate.
    """
    events: list[dict[str, Any]] = []

    class _TracedHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            fields = getattr(record, FIELDS_ATTR, None)
            if isinstance(fields, dict):
                # Read on the emitting thread, which is the thread the delivery's scope is open on.
                events.append({"event": record.getMessage(), "trace": current_trace(), **fields})

    handler = _TracedHandler()
    loggers = logging.getLogger("calorai_agent")
    was = loggers.level
    loggers.setLevel(logging.ERROR)
    loggers.addHandler(handler)
    try:
        crashing, _graph = _app(repository, prefetcher=_CrashingPrefetcher())
        with pytest.raises(httpx.InvalidURL):
            _deliver(
                crashing,
                _photo(msg_id="wamid.IMG", caption="two dosas with coconut chutney"),
            )
    finally:
        loggers.removeHandler(handler)
        loggers.setLevel(was)

    failed = [row for row in events if row["event"] == "delivery_failed"]
    assert len(failed) == 1
    assert failed[0]["error"] == "InvalidURL"
    assert failed[0]["messages"] == 1
    # A trace id a reviewer can grep for, and no meal text anywhere in the line.
    assert failed[0]["trace"]
    assert "dosa" not in repr(events).lower()


# --- what we cannot eat, and who may send it -----------------------------------------


def test_a_voice_note_is_declined_in_words_the_user_can_act_on(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)
    note = {
        "from": WA_USER,
        "id": "wamid.V",
        "timestamp": str(TS),
        "type": "audio",
        "audio": {"id": "media-2", "voice": True},
    }

    _deliver(application, note)

    assert "voice note" in graph.replies[0][1]
    assert repository.list_for_day(WA_USER, DAY) == []


def test_a_retried_voice_note_declines_once(repository: MealRepository) -> None:
    application, graph = _app(repository)
    note = {
        "from": WA_USER,
        "id": "wamid.V",
        "timestamp": str(TS),
        "type": "audio",
        "audio": {"id": "media-2"},
    }
    body = _body(note)
    signature = _signed(body)

    application.post(signature, body)
    application.post(signature, body)

    assert len(graph.replies) == 1
    assert len(_events(repository)) == 1


def test_a_decline_is_ledgered_against_the_user_it_was_sent_to(
    repository: MealRepository,
) -> None:
    application, _ = _app(repository)
    note = {"from": WA_USER, "id": "wamid.V", "type": "sticker", "sticker": {"id": "media-3"}}

    _deliver(application, note)

    assert [event[0] for event in _events(repository)] == ["wamid.V"]
    assert _events(repository, STRANGER) == []


def test_an_unlisted_number_gets_no_reply_and_no_meal(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)

    status, _ = _deliver(application, _text("I ate 10 biryani", sender=STRANGER))

    assert status == 200
    assert graph.replies == []
    assert repository.list_for_day(STRANGER, DAY) == []
    assert _events(repository, STRANGER) == []


def test_an_unlisted_sender_is_refused_by_digest_under_our_own_secret(
    repository: MealRepository,
    caplog,
) -> None:
    """The whole request thread, checked for the thing a refusal is most likely to leak.

    The warning is written by the server, which holds the app secret, so the digest it files has to
    be the keyed one: an unsalted truncation of a number anyone can guess is a lookup table, and a
    reply that later fails has to name the same digest or the two lines are uncorrelatable.
    """
    unkeyed = hashlib.sha256(STRANGER.encode()).hexdigest()[:10]
    with caplog.at_level(logging.WARNING, logger="calorai_agent"):
        _deliver(_app(repository)[0], _text("I ate 10 biryani", sender=STRANGER))

    assert STRANGER not in caplog.text
    assert f"wa#{unkeyed}" not in caplog.text
    assert sender_for_log(STRANGER, SECRET) in caplog.text


def test_an_empty_allow_list_answers_nothing_at_all(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository, allowed=())

    status, _ = _deliver(application, _text("1 idli"))

    assert (status, graph.replies) == (200, [])


# --- when the agent itself fails -----------------------------------------------------


def test_a_message_the_agent_cannot_finish_is_answered_instead_of_gone_silent(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository, planner=BoomPlanner())

    status, _ = _deliver(application, _text("1 idli"))

    assert status == 200
    assert graph.replies == [(WA_USER, INTERNAL_FAILURE_REPLY)]


def test_one_users_failure_does_not_stop_the_rest_of_the_delivery(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository, planner=BoomPlanner())

    _deliver(application, _text("1 idli", msg_id="wamid.1"), _text("2 idli", msg_id="wamid.2"))

    assert len(graph.replies) == 2


# --- the HTTP shell ------------------------------------------------------------------


def _settings(tmp_path: Path, **extra: Any) -> Settings:
    return Settings(
        database_path=tmp_path / "webhook.sqlite3",
        default_user_id="local-demo-user",
        default_timezone="UTC",
        planner="deterministic",
        **extra,
    )


def _request(
    port: int,
    method: str,
    target: str,
    *,
    body: bytes | None = None,
    signature: str | None = None,
    content_length: str | None = None,
) -> tuple[int, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        headers = {"Content-Type": "application/json"}
        if signature is not None:
            headers["X-Hub-Signature-256"] = signature
        if content_length is not None:
            # A stranger chooses this number, so it is the one thing the shell must not trust.
            headers["Content-Length"] = content_length
        connection.request(method, target, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read().decode()
    finally:
        connection.close()


def _deliver_and_stop(server: Any, work: Callable[[int], None]) -> None:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        work(server.server_address[1])
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_the_http_shell_serves_the_handshake_then_the_events_on_its_configured_path(
    tmp_path: Path,
) -> None:
    graph = FakeGraph()
    settings = replace(
        _settings(
            tmp_path,
            whatsapp_verify_token=VERIFY,
            whatsapp_app_secret=SECRET,
            whatsapp_access_token="token",
            whatsapp_phone_number_id="1000",
            whatsapp_allowed_users=("local-demo-user",),
        ),
        webhook_host="127.0.0.1",
        webhook_port=0,
    )
    application = create_whatsapp_app(settings)
    application.client = graph
    application.executor = InlineExecutor()
    body = _body(_text("I ate 3 idli and 1 coffee", sender="local-demo-user"))
    seen: list[tuple[int, str]] = []

    def visit(port: int) -> None:
        seen.append(
            _request(
                port,
                "GET",
                f"/webhook?hub.mode=subscribe&hub.verify_token={VERIFY}&hub.challenge=1158601234",
            )
        )
        seen.append(
            _request(
                port,
                "GET",
                "/webhook?hub.mode=subscribe&hub.verify_token=nope&hub.challenge=1158601234",
            )
        )
        seen.append(_request(port, "GET", "/health"))
        seen.append(_request(port, "POST", "/webhook", body=body))
        seen.append(_request(port, "POST", "/webhook", body=body, signature="sha256=" + "0" * 64))
        seen.append(_request(port, "POST", "/health", body=body, signature=_signed(body)))
        seen.append(_request(port, "POST", "/webhook", body=body, signature=_signed(body)))

    _deliver_and_stop(serve(settings, application), visit)

    assert [status for status, _ in seen] == [200, 403, 404, 401, 401, 404, 200]
    assert seen[0][1] == "1158601234"
    assert seen[-1][1] == "ok"
    assert "Logged" in graph.replies[0][1]
    assert len(application.repository.list_for_day("local-demo-user", DAY)) == 1


def _live_app(tmp_path: Path) -> tuple[Settings, WebhookApplication]:
    settings = replace(
        _settings(
            tmp_path,
            whatsapp_verify_token=VERIFY,
            whatsapp_app_secret=SECRET,
            whatsapp_access_token="token",
            whatsapp_phone_number_id="1000",
            whatsapp_allowed_users=("local-demo-user",),
        ),
        webhook_host="127.0.0.1",
        webhook_port=0,
    )
    return settings, create_whatsapp_app(settings)


class GatedPlanner:
    """Parses only when the test lets it, so a turn can be caught still running."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def parse(self, request: Any) -> Any:
        self.started.set()
        assert self.release.wait(5), "the turn never reached the planner"
        return RuleBasedPlanner().parse(request)


class OnceLocked:
    """Fails the first user lookup the way a locked database would, then behaves normally."""

    def __init__(self, inner: MealRepository) -> None:
        self.inner = inner
        self.locked = False

    def ensure_user(self, user_id: str, timezone_name: str = "UTC") -> None:
        if not self.locked:
            self.locked = True
            raise sqlite3.OperationalError("database is locked")
        self.inner.ensure_user(user_id, timezone_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("the condition never became true")
        time.sleep(0.01)


def test_the_webhook_is_acknowledged_while_the_turn_is_still_running(
    tmp_path: Path,
) -> None:
    # Meta times out and retries anything slow, so a model that takes a minute must never turn into
    # the same message worked on by two connection threads at once.
    graph = FakeGraph()
    settings, application = _live_app(tmp_path)
    application.client = graph
    planner = GatedPlanner()
    application.agent.planner = planner
    assert isinstance(application.executor, WorkerExecutor)
    body = _body(_text("I ate 3 idli and 1 coffee", sender="local-demo-user"))
    ack: list[tuple[int, str]] = []

    def visit(port: int) -> None:
        ack.append(_request(port, "POST", "/webhook", body=body, signature=_signed(body)))
        assert planner.started.wait(5), "the worker never took the turn"
        assert graph.replies == []
        planner.release.set()
        _wait_for(lambda: bool(graph.replies))

    _deliver_and_stop(serve(settings, application), visit)

    assert ack[0] == (200, "ok")
    assert "Logged" in graph.replies[0][1]


def test_the_verify_token_never_reaches_an_access_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings, application = _live_app(tmp_path)
    application.client = FakeGraph()
    application.executor = InlineExecutor()

    def visit(port: int) -> None:
        _request(
            port,
            "GET",
            f"/webhook?hub.mode=subscribe&hub.verify_token={VERIFY}&hub.challenge=7",
        )

    with caplog.at_level(logging.DEBUG, logger="calorai_agent.whatsapp_server"):
        _deliver_and_stop(serve(settings, application), visit)

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert VERIFY not in logged
    assert "/webhook" in logged


def test_an_absurd_declared_body_is_refused_before_any_of_it_is_read(
    tmp_path: Path,
) -> None:
    settings, application = _live_app(tmp_path)
    application.client = FakeGraph()
    application.executor = InlineExecutor()
    body = _body(_text("1 idli", sender="local-demo-user"))
    seen: list[tuple[int, str]] = []

    def visit(port: int) -> None:
        seen.append(
            _request(
                port,
                "POST",
                "/webhook",
                body=body,
                signature=_signed(body),
                content_length=str(10_000_000_000),
            )
        )
        seen.append(
            _request(
                port, "POST", "/webhook", body=body, signature=_signed(body), content_length="junk"
            )
        )

    _deliver_and_stop(serve(settings, application), visit)

    assert [status for status, _ in seen] == [413, 400]


def test_one_locked_database_turn_does_not_cost_the_rest_of_the_delivery(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)
    application.repository = OnceLocked(application.repository)  # type: ignore[assignment]

    status, _ = _deliver(
        application,
        _text("1 idli", msg_id="wamid.1"),
        _text("2 idli", msg_id="wamid.2"),
        {"from": WA_USER, "id": "wamid.V", "type": "audio", "audio": {"id": "media-2"}},
    )

    assert status == 200
    assert [reply for _, reply in graph.replies][0] == INTERNAL_FAILURE_REPLY
    assert "Logged" in graph.replies[1][1]
    assert "voice note" in graph.replies[2][1]
    assert len(repository.list_for_day(WA_USER, DAY)) == 1


def test_a_retry_of_a_failed_turn_says_the_same_true_thing(
    repository: MealRepository,
) -> None:
    # Leaving the ledger row open makes every later retry promise work this process has stopped
    # doing, which is worse than repeating the honest failure.
    application, graph = _app(repository, planner=BoomPlanner())
    body = _body(_text("1 idli"))
    signature = _signed(body)

    application.post(signature, body)
    application.post(signature, body)

    assert graph.replies == [(WA_USER, INTERNAL_FAILURE_REPLY)] * 2
    assert all("still working" not in reply for _, reply in graph.replies)


def test_a_message_with_no_wamid_is_answered_but_never_asked_to_be_marked_read(
    repository: MealRepository,
) -> None:
    application, graph = _app(repository)
    item = {"from": WA_USER, "type": "text", "text": {"body": "1 idli"}, "timestamp": str(TS)}

    _deliver(application, item)

    assert graph.receipt_ids == []
    assert "Logged" in graph.replies[0][1]


def test_the_server_refuses_to_start_on_an_incomplete_whatsapp_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ALLOW_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.argv", ["calorai-whatsapp"])
    monkeypatch.setenv("CALORAI_DB_PATH", str(tmp_path / "unused.sqlite3"))

    from calorai_agent.whatsapp_server import main

    assert main() == 2
    for name, value in {
        "CALORAI_WHATSAPP_VERIFY_TOKEN": VERIFY,
        "CALORAI_WHATSAPP_APP_SECRET": SECRET,
        "CALORAI_WHATSAPP_ACCESS_TOKEN": "token",
        "CALORAI_WHATSAPP_PHONE_NUMBER_ID": "1000",
    }.items():
        monkeypatch.setenv(name, value)
    assert main() == 2


def test_the_allow_list_env_reads_a_comma_separated_list_of_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ALLOW_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CALORAI_WHATSAPP_ALLOWED_USERS", f" {WA_USER} , {STRANGER} ,{WA_USER} ")

    settings = Settings.from_env()

    assert settings.whatsapp_allowed_users == (WA_USER, STRANGER)
    assert settings.use_whatsapp is False


# --- configuration -----------------------------------------------------------------


def test_the_graph_sender_exists_only_once_a_token_and_a_number_are_both_set() -> None:
    assert build_graph_client(_settings(Path("/unused"))) is None
    assert build_graph_client(_settings(Path("/unused"), whatsapp_access_token="token")) is None
    client = build_graph_client(
        _settings(
            Path("/unused"),
            whatsapp_access_token="token",
            whatsapp_phone_number_id="1000",
        )
    )
    assert isinstance(client, GraphClient)
