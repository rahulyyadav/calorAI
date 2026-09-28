"""Phase 5: Meta's wire format becomes the same envelope the CLI already speaks.

Nothing here needs a WhatsApp number: requests are answered by `httpx.MockTransport`, so the
Graph API is exercised exactly as it will be in production — same URLs, same bodies, same
failures — without a token or the network.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from calorai_agent.app import EitherMediaSource
from calorai_agent.domain import InboundMessage, MediaRef
from calorai_agent.vision import MediaError
from calorai_agent.whatsapp import (
    GraphClient,
    WebhookBatch,
    WhatsAppError,
    WhatsAppMediaSource,
    is_fingerprinted_event,
    normalize_webhook,
    parse_webhook_body,
    sender_for_log,
    signature_matches,
    webhook_challenge,
)

SENDER = "919812345678"
STRANGER = "919999999999"
SECRET = "app-secret"
TS = 1_790_000_000
PNG = b"\x89PNG\r\n\x1a\n" + b"y" * 40
JPEG = b"\xff\xd8\xff" + b"x" * 40
BASE = "https://graph.test/v23.0"
ALLOWED = frozenset({SENDER})
MEDIA_URL = "https://lookaside.test/photo"


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _payload(*items: dict[str, Any], field: str = "messages", **extra: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"messaging_product": "whatsapp", "messages": list(items), **extra}
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "100", "changes": [{"field": field, "value": value}]}],
    }


def _text(body: str, *, msg_id: str = "wamid.TXT", sender: str = SENDER, **item: Any) -> dict:
    return {"from": sender, "id": msg_id, "type": "text", "text": {"body": body}, **item}


def _image(
    media_id: str | None = "media-1",
    *,
    caption: str | None = None,
    msg_id: str = "wamid.IMG",
    sender: str = SENDER,
    **item: Any,
) -> dict:
    image: dict[str, Any] = {}
    if media_id is not None:
        image["id"] = media_id
    if caption is not None:
        image["caption"] = caption
    return {"from": sender, "id": msg_id, "type": "image", "image": image, **item}


def _normalize(
    *items: dict[str, Any],
    allowed: frozenset[str] = ALLOWED,
    log_key: str = "",
    **extra: Any,
) -> tuple[WebhookBatch, tuple[str, ...]]:
    payload = _payload(*items, **extra)
    return normalize_webhook(
        payload, timezone="Asia/Kolkata", allowed_users=allowed, log_key=log_key
    )


# --- proving an event came from Meta -------------------------------------------------


def test_a_body_signed_with_our_app_secret_verifies() -> None:
    body = b'{"object":"whatsapp_business_account"}'
    assert signature_matches(app_secret=SECRET, body=body, signature=_sign(body))


def test_a_signature_does_not_survive_one_changed_byte_of_the_body() -> None:
    body = b'{"text":"1 dosa"}'
    signature = _sign(body)
    tampered = body.replace(b"dosa", b"idli")
    assert signature_matches(app_secret=SECRET, body=tampered, signature=signature) is False


def test_a_secret_that_is_not_ours_does_not_verify() -> None:
    body = b'{"text":"1 dosa"}'
    other = _sign(body, "someone-else")
    assert signature_matches(app_secret=SECRET, body=body, signature=other) is False


@pytest.mark.parametrize(
    "signature",
    [None, "", "sha256=", "md5=abc", "a" * 64, "sha256=" + "z" * 64],
)
def test_a_signature_without_a_usable_sha256_prefix_is_refused(signature: str | None) -> None:
    assert signature_matches(app_secret=SECRET, body=b"{}", signature=signature) is False


def test_a_signature_covers_the_wire_bytes_not_reparsed_json() -> None:
    # Re-serializing changes spacing, so verifying the re-spaced bytes would prove nothing about
    # what Meta actually sent.
    body = b'{  "a":  1 }'
    assert signature_matches(app_secret=SECRET, body=body, signature=_sign(body))
    assert signature_matches(app_secret=SECRET, body=body, signature=_sign(b'{"a":1}')) is False


def test_an_uppercase_hex_signature_still_verifies() -> None:
    body = b"{}"
    signature = "sha256=" + _sign(body)[len("sha256=") :].upper()
    assert signature_matches(app_secret=SECRET, body=body, signature=signature)


# --- the subscription handshake ------------------------------------------------------


def test_the_handshake_echoes_the_challenge_for_our_verify_token() -> None:
    echoed = webhook_challenge(
        mode="subscribe", token="s-3cret", challenge="1234567890", verify_token="s-3cret"
    )
    assert echoed == "1234567890"


@pytest.mark.parametrize(
    ("mode", "token", "challenge"),
    [
        ("subscribe", "someone-elses", "123"),
        ("subscribe", None, "123"),
        ("subscribe", "", "123"),
        ("unsubscribe", "s-3cret", "123"),
        (None, "s-3cret", "123"),
        ("subscribe", "s-3cret", None),
    ],
)
def test_a_handshake_that_is_not_ours_is_refused(
    mode: str | None, token: str | None, challenge: str | None
) -> None:
    refused = webhook_challenge(mode=mode, token=token, challenge=challenge, verify_token="s-3cret")
    assert refused is None


def test_parse_webhook_body_keeps_objects_and_refuses_everything_else() -> None:
    assert parse_webhook_body(b'{"a": 1}') == {"a": 1}
    assert parse_webhook_body(b"not json") is None
    assert parse_webhook_body(b"") is None
    assert parse_webhook_body(b"[1, 2]") is None
    assert parse_webhook_body(b'"a string"') is None


# --- normalizing Meta's envelope -----------------------------------------------------


def test_a_text_message_becomes_the_same_envelope_the_cli_builds() -> None:
    batch, refused = _normalize(_text("I had 2 idli", timestamp=TS))

    assert refused == ()
    assert len(batch.messages) == 1
    message = batch.messages[0]
    assert message.user_id == SENDER
    assert message.text == "I had 2 idli"
    assert message.external_id == "wamid.TXT"
    assert message.channel == "whatsapp"
    assert message.timezone == "Asia/Kolkata"
    assert message.received_at == datetime.fromtimestamp(TS, UTC)
    assert message.media is None


def test_a_photo_becomes_a_media_reference_to_its_own_message_id() -> None:
    batch, _ = _normalize(_image("media-9", caption="my lunch", timestamp=TS))

    message = batch.messages[0]
    assert message.text == "my lunch"
    assert message.media == MediaRef(
        external_id="wamid.IMG", locator="media-9", source="whatsapp_media"
    )


def test_a_photo_without_a_caption_is_still_a_message_with_a_plate() -> None:
    message = _normalize(_image())[0].messages[0]
    assert message.text == ""
    assert message.media is not None


def test_the_day_the_user_sent_it_survives_the_transport() -> None:
    # Sent at 23:58 and read after midnight: the meal belongs to the day it was eaten.
    sent = int(datetime(2026, 9, 26, 23, 58, tzinfo=UTC).timestamp())
    message = _normalize(_text("1 dinner", timestamp=sent))[0].messages[0]
    assert message.received_at == datetime(2026, 9, 26, 23, 58, tzinfo=UTC)


@pytest.mark.parametrize("stamp", ["not-a-number", 99_999_999_999_999, 0, -62_135_596_800])
def test_an_unreadable_timestamp_costs_the_timestamp_not_the_message(stamp: Any) -> None:
    message = _normalize(_text("1 idli", timestamp=stamp))[0].messages[0]
    assert message.received_at is None


def test_a_stamp_from_a_century_ahead_is_not_treated_as_a_send_time() -> None:
    # A meal dated to year 9999 answers nothing: it leaves every totals question and every
    # recent-meal window, so the clock the graph falls back to is the honest one.
    far_future = int(datetime(9999, 12, 31, tzinfo=UTC).timestamp())
    message = _normalize(_text("1 idli", timestamp=far_future))[0].messages[0]
    assert message.received_at is None


def test_a_stamp_a_few_minutes_ahead_of_our_own_clock_is_still_a_send_time() -> None:
    slightly_ahead = int(datetime.now(UTC).timestamp()) + 300
    message = _normalize(_text("1 idli", timestamp=slightly_ahead))[0].messages[0]
    assert message.received_at is not None


def test_a_message_retries_carry_metas_id_so_the_ledger_can_recognise_them() -> None:
    first, _ = _normalize(_text("1 idli", msg_id="wamid.SAME"))
    second, _ = _normalize(_text("1 idli", msg_id="wamid.SAME"))
    assert first.messages[0].external_id == second.messages[0].external_id == "wamid.SAME"


def test_a_message_that_arrives_without_an_id_is_fingerprinted_from_its_own_content() -> None:
    item = {"from": SENDER, "type": "text", "text": {"body": "1 idli"}}
    batch, _ = _normalize(item)
    repeated, _ = _normalize(item)

    event_id = batch.messages[0].external_id
    assert event_id.startswith("wamid-sha:")
    assert repeated.messages[0].external_id == event_id


def test_different_words_without_an_id_are_not_the_same_fingerprint() -> None:
    first = {"from": SENDER, "type": "text", "text": {"body": "1 idli"}}
    second = {"from": SENDER, "type": "text", "text": {"body": "2 idli"}}
    assert (
        _normalize(first)[0].messages[0].external_id
        != _normalize(second)[0].messages[0].external_id
    )


def test_a_fingerprinted_event_is_known_as_one_so_no_receipt_is_asked_for() -> None:
    # An id this adapter invented names no WhatsApp message, so marking it read would be a request
    # against something Meta never issued.
    batch, _ = _normalize({"from": SENDER, "type": "text", "text": {"body": "1 idli"}})

    assert is_fingerprinted_event(batch.messages[0].external_id)
    assert not is_fingerprinted_event("wamid.TXT")


def test_two_messages_in_one_delivery_are_two_messages() -> None:
    batch, _ = _normalize(_text("1 idli", msg_id="wamid.1"), _text("2 coffee", msg_id="wamid.2"))
    assert [m.external_id for m in batch.messages] == ["wamid.1", "wamid.2"]


def test_two_changes_in_one_delivery_are_read_together() -> None:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "100",
                "changes": [
                    {"field": "messages", "value": {"messages": [_text("1 idli", msg_id="a")]}},
                    {"field": "messages", "value": {"messages": [_text("2 idli", msg_id="b")]}},
                ],
            }
        ],
    }
    batch, _ = normalize_webhook(payload, timezone="UTC", allowed_users=ALLOWED)
    assert [m.external_id for m in batch.messages] == ["a", "b"]


def test_a_change_about_something_other_than_messages_is_ignored() -> None:
    batch, refused = _normalize(_text("1 idli"), field="contact")
    assert batch.empty
    assert refused == ()


def test_a_delivery_with_no_sender_id_carries_no_message_for_anyone() -> None:
    payload = _payload({"id": "wamid.X", "type": "text", "text": {"body": "1 idli"}})
    batch, refused = normalize_webhook(payload, timezone="UTC", allowed_users=ALLOWED)
    assert batch.empty
    assert refused == ()


def test_a_change_value_that_is_not_an_object_is_ignored() -> None:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [{"id": "100", "changes": [{"field": "messages", "value": "broken"}]}],
    }
    assert normalize_webhook(payload, timezone="UTC", allowed_users=ALLOWED)[0].empty


def test_an_entry_without_changes_is_ignored() -> None:
    payload = {"object": "whatsapp_business_account", "entry": [{"id": "100"}]}
    assert normalize_webhook(payload, timezone="UTC", allowed_users=ALLOWED)[0].empty


def test_a_message_body_that_is_not_an_object_is_ignored_not_crashed_on() -> None:
    payload = _payload({"from": SENDER, "id": "wamid.X", "type": "text", "text": None})
    batch, _ = normalize_webhook(payload, timezone="UTC", allowed_users=ALLOWED)
    assert batch.messages == ()


# --- the allow-list ------------------------------------------------------------------


def test_a_sender_we_did_not_list_never_reaches_the_agent() -> None:
    batch, refused = _normalize(_text("1 biryani", sender=STRANGER))

    assert batch.empty
    assert refused == (STRANGER,)


def test_a_refused_sender_is_logged_without_filing_their_number(caplog) -> None:
    """A `wa_id` is a phone number, and refusals log at the default level.

    The blocked stranger's number is the one thing a log should not accumulate: it identifies them,
    it is of no use to the meal, and it would sit in a file long after the conversation ended. A
    stable digest keeps one sender correlatable across lines, and the full id is in the app's own
    webhook payload for whoever decides whether to allow it.
    """
    with caplog.at_level(logging.WARNING, logger="calorai_agent"):
        _normalize(_text("1 biryani", sender=STRANGER))

    assert STRANGER not in caplog.text
    assert sender_for_log(STRANGER) in caplog.text
    assert sender_for_log(STRANGER) == sender_for_log(STRANGER)
    assert sender_for_log(STRANGER) != sender_for_log(SENDER)


def test_a_refused_sender_digest_is_keyed_so_the_number_stays_unfindable(caplog) -> None:
    """An unsalted truncation of a phone number is one dictionary away from the number itself.

    The id space is small and well known, so anyone holding the log can digest the numbers they
    already have until one matches a line. Keyed on the app secret, the same check needs a secret
    nobody should have, and a sender still correlates with themselves inside one deployment.
    """
    unkeyed = hashlib.sha256(STRANGER.encode()).hexdigest()[:10]
    with caplog.at_level(logging.WARNING, logger="calorai_agent"):
        _normalize(_text("1 biryani", sender=STRANGER), log_key=SECRET)

    assert f"wa#{unkeyed}" not in caplog.text
    assert sender_for_log(STRANGER, SECRET) in caplog.text
    assert sender_for_log(STRANGER, SECRET) != sender_for_log(STRANGER, "a different secret")
    assert sender_for_log(STRANGER, SECRET) == sender_for_log(STRANGER, SECRET)


def test_an_empty_allow_list_refuses_everyone() -> None:
    batch, refused = _normalize(_text("1 biryani"), allowed=frozenset())
    assert batch.empty
    assert refused == (SENDER,)


def test_a_listed_sender_gets_through_beside_an_unlisted_one() -> None:
    batch, refused = _normalize(
        _text("1 biryani", msg_id="wamid.1", sender=STRANGER),
        _text("1 idli", msg_id="wamid.2"),
    )
    assert [m.text for m in batch.messages] == ["1 idli"]
    assert refused == (STRANGER,)


# --- receipts and things we cannot eat -----------------------------------------------


def test_a_delivery_receipt_for_a_message_we_sent_is_skipped_not_answered() -> None:
    batch, refused = _normalize({"status": "delivered", "id": "wamid.OUR-OWN", "timestamp": TS})
    assert batch.empty
    assert refused == ()


def test_a_null_status_on_the_users_own_message_does_not_swallow_it() -> None:
    batch, _ = _normalize(_text("1 idli", status=None))
    assert len(batch.messages) == 1


@pytest.mark.parametrize(
    ("item", "phrase"),
    [
        ({"type": "audio", "audio": {"id": "m"}}, "voice note"),
        ({"type": "video", "video": {"id": "m"}}, "video"),
        ({"type": "document", "document": {"id": "m"}}, "file"),
        ({"type": "sticker", "sticker": {"id": "m"}}, "sticker"),
        ({"type": "location", "location": {"latitude": 1}}, "map pin"),
        ({"type": "button", "button": {"text": "hi"}}, "button press"),
        ({"type": "interactive", "interactive": {"type": "list_reply"}}, "list reply"),
        ({"type": "reaction", "reaction": {"emoji": "🔥"}}, "reaction"),
        ({"type": "system", "system": {"type": "updated"}}, "conversation itself"),
        ({"type": "unknown_thing"}, "attachment I do not know"),
    ],
)
def test_an_attachment_we_cannot_eat_is_answered_honestly(item: dict, phrase: str) -> None:
    batch, refused = _normalize({"from": SENDER, "id": "wamid.X", **item})

    assert batch.messages == ()
    assert len(batch.declines) == 1
    user_id, external_id, reply = batch.declines[0]
    assert (user_id, external_id) == (SENDER, "wamid.X")
    assert phrase in reply
    assert "I can only work with" in reply
    assert refused == ()


def test_a_saved_contact_declines_even_though_it_names_the_sender() -> None:
    batch, refused = _normalize(
        {"from": SENDER, "id": "wamid.C", "type": "contact", "contact": {"name": {"fn": "R"}}},
        contacts=[{"wa_id": SENDER, "profile": {"name": "Rahul"}}],
    )
    assert batch.messages == ()
    assert "saved contact" in batch.declines[0][2]
    assert refused == ()


def test_a_sender_only_a_contact_block_names_still_counts_as_that_sender() -> None:
    payload = _payload(
        {"id": "wamid.C", "type": "text", "text": {"body": "1 idli"}},
        contacts=[{"wa_id": SENDER}],
    )
    batch, _ = normalize_webhook(payload, timezone="UTC", allowed_users=ALLOWED)
    assert len(batch.messages) == 1


def test_a_contact_from_an_unlisted_number_is_refused_before_it_is_answered() -> None:
    payload = _payload(
        {"id": "wamid.C", "type": "text", "text": {"body": "1 idli"}},
        contacts=[{"wa_id": STRANGER}],
    )
    batch, refused = normalize_webhook(payload, timezone="UTC", allowed_users=ALLOWED)
    assert batch.empty
    assert refused == (STRANGER,)


def test_a_contact_share_from_a_listed_number_is_not_reported_as_a_refusal() -> None:
    # A block that names a conversation but carries no message is nothing to answer, and calling a
    # permitted number "refused" in the log would send a reviewer hunting a bug that is not there.
    payload = _payload(contacts=[{"wa_id": SENDER, "profile": {"name": "Rahul"}}])
    batch, refused = normalize_webhook(payload, timezone="UTC", allowed_users=ALLOWED)

    assert batch.empty
    assert refused == ()


def test_a_text_message_with_no_words_in_it_declines() -> None:
    batch, _ = _normalize(_text("   "))
    assert batch.messages == ()
    assert "no words in it" in batch.declines[0][2]


def test_a_photo_whose_pointer_is_missing_declines_instead_of_logging_nothing() -> None:
    item = _image()
    del item["image"]["id"]
    batch, _ = _normalize(item)
    assert batch.messages == ()
    assert "no media attached" in batch.declines[0][2]


def test_an_empty_delivery_asks_for_nothing_at_all() -> None:
    batch, refused = _normalize()
    assert batch.empty
    assert refused == ()


# --- the Graph API client ------------------------------------------------------------


class FakeGraph:
    """A stand-in Meta server that records every request the transport makes."""

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.handler = handler
        self.client = GraphClient(
            access_token="bearer-token",
            phone_number_id="1000",
            base_url=BASE,
            transport=httpx.MockTransport(self._handle),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.handler is not None:
            return self.handler(request)
        if request.method == "GET":
            if str(request.url).endswith("/media-1"):
                return httpx.Response(200, json={"url": MEDIA_URL, "mime_type": "image/png"})
            return httpx.Response(200, content=PNG)
        return httpx.Response(200, json={"messaging_product": "whatsapp"})

    @property
    def sent_bodies(self) -> list[dict]:
        return [
            json.loads(request.content) for request in self.requests if request.method == "POST"
        ]

    def text_bodies(self) -> list[str]:
        return [body["text"]["body"] for body in self.sent_bodies if body.get("type") == "text"]

    def statuses(self) -> list[str]:
        return [body["status"] for body in self.sent_bodies if "status" in body]


def test_a_reply_goes_to_the_messages_endpoint_for_our_number() -> None:
    graph = FakeGraph()
    graph.client.send_text(SENDER, "Logged 2 idli")

    request = graph.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE}/1000/messages"
    assert json.loads(request.content) == {
        "messaging_product": "whatsapp",
        "to": SENDER,
        "type": "text",
        "text": {"body": "Logged 2 idli", "preview_url": False},
    }


def test_the_token_travels_as_a_header_and_never_in_a_url() -> None:
    # URLs are copied into access logs and support tickets, so a token in one leaks.
    graph = FakeGraph()
    graph.client.send_text(SENDER, "hi")
    graph.client.download("media-1")

    assert len(graph.requests) == 3
    assert all(
        request.headers.get("Authorization") == "Bearer bearer-token" for request in graph.requests
    )
    assert all("bearer-token" not in str(request.url) for request in graph.requests)


def test_a_read_receipt_carries_the_typing_indicator_the_way_meta_documents_it() -> None:
    # Meta has no standalone typing request: the indicator is one field of a read status, and a
    # body it does not recognise is a typing bubble the user never sees.
    graph = FakeGraph()
    graph.client.mark_seen("wamid.1")
    graph.client.mark_seen("wamid.2", typing=True)

    assert graph.sent_bodies == [
        {"messaging_product": "whatsapp", "status": "read", "message_id": "wamid.1"},
        {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": "wamid.2",
            "typing_indicator": {"type": "text"},
        },
    ]


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_a_graph_refusal_becomes_a_whatsapp_error_not_a_crash(status: int) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "rate limited"}})

    graph = FakeGraph(refuse)
    with pytest.raises(WhatsAppError, match="graph request failed"):
        graph.client.send_text(SENDER, "hi")


def test_a_transport_failure_names_itself_without_repeating_the_token() -> None:
    class Exploding(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

    client = GraphClient(
        access_token="secret-token",
        phone_number_id="1000",
        base_url=BASE,
        transport=Exploding(),
    )
    with pytest.raises(WhatsAppError) as error:
        client.send_text(SENDER, "hi")

    assert "ConnectError" in str(error.value)
    assert "secret-token" not in str(error.value)


def test_a_photo_is_fetched_by_pointer_then_by_the_url_that_pointer_gave() -> None:
    graph = FakeGraph()
    assert graph.client.download("media-1") == PNG
    assert [str(request.url) for request in graph.requests] == [f"{BASE}/media-1", MEDIA_URL]


@pytest.mark.parametrize(
    "body",
    [b"not json", json.dumps({"id": "media-1"}).encode(), json.dumps({"url": ""}).encode()],
)
def test_a_pointer_with_no_download_url_says_so(body: bytes) -> None:
    def pointer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    graph = FakeGraph(pointer)
    with pytest.raises(WhatsAppError, match="no download url"):
        graph.client.download("media-1")


@pytest.mark.parametrize("link", ["http://lookaside.test/photo", "", "https:/x"])
def test_a_pointer_that_does_not_name_an_https_url_is_not_fetched(link: str) -> None:
    def pointer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"url": link})

    graph = FakeGraph(pointer)
    with pytest.raises(WhatsAppError, match="no download url"):
        graph.client.download("media-1")

    assert len(graph.requests) == 1


# --- media ids to validated photo bytes ----------------------------------------------


def _graph_answering(data: bytes) -> GraphClient:
    def handle(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/media-1"):
            return httpx.Response(200, json={"url": MEDIA_URL})
        return httpx.Response(200, content=data)

    return GraphClient(
        access_token="t",
        phone_number_id="1",
        base_url=BASE,
        transport=httpx.MockTransport(handle),
    )


def _fetch(data: bytes, *, source: str = "whatsapp_media") -> Any:
    return WhatsAppMediaSource(_graph_answering(data)).fetch(
        MediaRef(external_id="e", locator="media-1", source=source)  # type: ignore[arg-type]
    )


def test_a_media_id_becomes_photo_bytes_and_a_mime_type() -> None:
    payload = _fetch(PNG)
    assert payload.data == PNG
    assert payload.mime_type == "image/png"


@pytest.mark.parametrize("data", [JPEG, PNG])
def test_both_photo_kinds_a_phone_sends_are_read_by_signature(data: bytes) -> None:
    assert _fetch(data).data == data


def test_a_local_path_is_not_something_the_whatsapp_server_can_fetch() -> None:
    with pytest.raises(MediaError, match="local_path"):
        _fetch(PNG, source="local_path")


def test_a_photo_that_failed_to_download_says_why() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "gone"}})

    client = GraphClient(
        access_token="t",
        phone_number_id="1",
        base_url=BASE,
        transport=httpx.MockTransport(refuse),
    )
    with pytest.raises(MediaError, match="could not be downloaded"):
        WhatsAppMediaSource(client).fetch(
            MediaRef(external_id="e", locator="media-1", source="whatsapp_media")
        )


def test_an_attachment_bigger_than_the_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("calorai_agent.whatsapp.MAX_IMAGE_BYTES", 32)
    with pytest.raises(MediaError, match="larger than the"):
        _fetch(PNG + b"z" * 64)


@pytest.mark.parametrize("bytes_", [b"GIF89a" + b"x" * 40, b"%PDF-1.4" + b"x" * 40, b""])
def test_a_file_that_is_not_a_photo_is_named_as_one(bytes_: bytes) -> None:
    with pytest.raises(MediaError, match="not a jpeg, png, or webp"):
        _fetch(bytes_)


# --- one pipeline, two transports ----------------------------------------------------


class LocalSpy:
    def fetch(self, media: MediaRef) -> str:
        return f"local:{media.locator}"


def test_a_cli_path_is_still_read_when_the_whatsapp_sender_exists() -> None:
    graph = FakeGraph()
    source: EitherMediaSource = EitherMediaSource(LocalSpy(), WhatsAppMediaSource(graph.client))

    assert (
        source.fetch(MediaRef(external_id="e", locator="/tmp/plate.jpg")) == "local:/tmp/plate.jpg"
    )
    assert graph.requests == []


def test_a_photo_from_whatsapp_without_a_graph_sender_is_not_guessed_at() -> None:
    source = EitherMediaSource(LocalSpy())
    with pytest.raises(MediaError, match="access token"):
        source.fetch(MediaRef(external_id="e", locator="media-1", source="whatsapp_media"))


def test_a_whatsapp_media_id_is_fetched_through_the_graph_sender() -> None:
    graph = FakeGraph()
    source = EitherMediaSource(LocalSpy(), WhatsAppMediaSource(graph.client))
    payload = source.fetch(MediaRef(external_id="e", locator="media-1", source="whatsapp_media"))
    assert payload.mime_type == "image/png"


def test_an_inbound_message_keeps_its_media_when_the_graph_hands_it_over() -> None:
    batch, _ = _normalize(_image(timestamp=TS))
    message: InboundMessage = batch.messages[0]
    assert message.media is not None
    assert message.media.source == "whatsapp_media"
