"""Phase 5: the WhatsApp Cloud API transport — signed webhooks in, Graph API replies out.

Nothing here decides what a meal is. This module proves an event came from Meta, turns Meta's
JSON into the transport-neutral `InboundMessage` envelope the CLI already uses, and delivers the
agent's reply. The graph cannot tell the two transports apart.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from calorai_agent.domain import InboundMessage, MediaRef
from calorai_agent.providers import ImagePayload
from calorai_agent.vision import MAX_IMAGE_BYTES, MediaError, sniff_image

logger = logging.getLogger(__name__)

SIGNATURE_PREFIX = "sha256="
CHANNEL = "whatsapp"

# The message types this agent can act on: words about a plate, or a photo of one.
_SUPPORTED_TYPES = ("text", "image")

# What Meta calls each type it can deliver that carries no meal. Answering honestly beats
# pretending a voice note was never sent.
_UNREADABLE_AS = {
    "audio": "a voice note, and I cannot listen to those yet",
    "video": "a video, and I cannot watch those yet",
    "document": "a file, and I cannot read those yet",
    "sticker": "a sticker, and I cannot tell a plate from one",
    "location": "a map pin, and I cannot tell a plate from one",
    "contact": "a saved contact, and there is no meal in that",
    "button": "a button press, and there is no meal in that",
    "interactive": "a button or list reply, and there is no meal in that",
    "reaction": "an emoji reaction",
    "system": "a change to the conversation itself",
    "text": "a message with no words in it",
    "image": "a photo with no media attached to it",
}

_UNSUPPORTED_REPLY = (
    "I can only work with a text description of a meal or a photo of the plate — "
    "that sounded like {what}."
)


class WhatsAppError(RuntimeError):
    """Meta rejected the request, or answered with something we cannot act on."""


def signature_matches(*, app_secret: str, body: bytes, signature: str | None) -> bool:
    """Did Meta sign this exact body with our app secret?

    The compare is constant time and the bytes are the raw wire body, not re-serialized JSON: the
    signature covers the wire format, so parsing first and hashing second would verify nothing.
    """
    if signature is None or not signature.startswith(SIGNATURE_PREFIX):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature[len(SIGNATURE_PREFIX) :].strip().lower())


def webhook_challenge(
    *, mode: str | None, token: str | None, challenge: str | None, verify_token: str
) -> str | None:
    """The subscription handshake: echo the challenge only when the verify token is ours.

    Anything else returns None and the caller answers 403, so Meta shows the subscription as
    failed rather than pointing a live number at an endpoint that will never answer.
    """
    if mode != "subscribe" or challenge is None:
        return None
    if not hmac.compare_digest(token or "", verify_token):
        return None
    return challenge


def parse_webhook_body(raw: bytes) -> Mapping[str, Any] | None:
    """The JSON object Meta sent, or None when it sent something that is not one."""
    try:
        loaded = json.loads(raw)
    except ValueError:
        return None
    return loaded if isinstance(loaded, Mapping) else None


@dataclass(frozen=True, slots=True)
class WebhookBatch:
    """What one webhook delivery asks of the agent, and what it asks of the transport."""

    messages: tuple[InboundMessage, ...] = ()
    declines: tuple[tuple[str, str, str], ...] = ()  # (user id, external id, reply text)

    @property
    def empty(self) -> bool:
        return not self.messages and not self.declines


def normalize_webhook(
    payload: Mapping[str, Any], *, timezone: str, allowed_users: frozenset[str]
) -> tuple[WebhookBatch, tuple[str, ...]]:
    """Read Meta's envelope into inbound messages, and name the senders it refused.

    An item with a `status` and no `type` is a delivery receipt for a message *we* sent, so it is
    skipped rather than answered. The allow-list is the only thing keeping a stranger's lunch off a
    shared test number, so it is checked against the number that actually sent each item — a
    conversation block can name its sender only through a contact card, and an unlisted sender is
    refused and logged, never guessed at. An empty list refuses everyone, which is what a
    half-configured deployment should do.
    """
    messages: list[InboundMessage] = []
    declines: list[tuple[str, str, str]] = []
    refused: list[str] = []
    for entry in _list(payload.get("entry")):
        for change in _list(entry.get("changes")):
            if change.get("field") != "messages":
                continue
            value = change.get("value")
            if not isinstance(value, Mapping):
                continue
            block_sender = _sender(value)
            if block_sender is None:
                continue
            items = _list(value.get("messages"))
            if not items:
                _refuse(refused, block_sender)
                continue
            for item in items:
                if item.get("status"):
                    continue
                sender = _wa_id(item.get("from")) or block_sender
                if sender not in allowed_users:
                    _refuse(refused, sender)
                    continue
                converted = _as_inbound(item, sender=sender, timezone=timezone)
                if converted is None:
                    declines.append(_decline(item, sender=sender))
                else:
                    messages.append(converted)
    return WebhookBatch(tuple(messages), tuple(declines)), tuple(refused)


def _refuse(refused: list[str], sender: str) -> None:
    if sender not in refused:
        logger.warning("refused webhook from unlisted sender %s", sender)
        refused.append(sender)


def _wa_id(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _list(value: Any) -> list[Mapping[str, Any]]:
    """The mappings in a JSON array, ignoring anything Meta did not document."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _sender(value: Mapping[str, Any]) -> str | None:
    for item in _list(value.get("messages")):
        sent_by = _wa_id(item.get("from"))
        if sent_by is not None:
            return sent_by
    for contact in _list(value.get("contacts")):
        wa_id = _wa_id(contact.get("wa_id"))
        if wa_id is not None:
            return wa_id
    return None


def _event_id(item: Mapping[str, Any]) -> str:
    """Meta's message id, which is what makes its retries the same message twice.

    A message with no id cannot be deduplicated that way, so its own content is fingerprinted
    instead: a retry of it still arrives as the same event rather than a second meal.
    """
    stated = item.get("id")
    if isinstance(stated, str) and stated:
        return stated
    digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]
    return f"wamid-sha:{digest}"


def _sent_at(item: Mapping[str, Any]) -> datetime | None:
    """When the user actually sent it, so a meal logged after midnight lands on the right day."""
    stamp = item.get("timestamp")
    if stamp is None:
        return None
    try:
        seconds = int(str(stamp))
    except ValueError:
        return None
    if seconds <= 0:
        # No message was ever sent before the epoch, and a stamp of zero would date a meal to 1970.
        return None
    try:
        return datetime.fromtimestamp(seconds, UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _as_inbound(item: Mapping[str, Any], *, sender: str, timezone: str) -> InboundMessage | None:
    """One supported message as the same envelope the CLI builds, or None when it is unreadable."""
    kind = item.get("type")
    if kind not in _SUPPORTED_TYPES or not isinstance(item.get(kind), Mapping):
        return None
    body = item[kind]
    event_id = _event_id(item)
    sent_at = _sent_at(item)
    if kind == "text":
        text = str(body.get("body") or "").strip()
        if not text:
            return None
        return InboundMessage(
            user_id=sender,
            text=text,
            external_id=event_id,
            channel=CHANNEL,
            timezone=timezone,
            received_at=sent_at,
        )
    media_id = body.get("id")
    if not isinstance(media_id, str) or not media_id:
        return None
    return InboundMessage(
        user_id=sender,
        text=str(body.get("caption") or "").strip(),
        external_id=event_id,
        channel=CHANNEL,
        timezone=timezone,
        received_at=sent_at,
        media=MediaRef(external_id=event_id, locator=media_id, source="whatsapp_media"),
    )


def _decline(item: Mapping[str, Any], *, sender: str) -> tuple[str, str, str]:
    what = _UNREADABLE_AS.get(str(item.get("type")), "an attachment I do not know how to read yet")
    return sender, _event_id(item), _UNSUPPORTED_REPLY.format(what=what)


class GraphClient:
    """The handful of Graph calls this transport needs, with an injectable httpx transport."""

    def __init__(
        self,
        *,
        access_token: str,
        phone_number_id: str,
        base_url: str,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def request(self, method: str, *, url: str, json_body: Any = None) -> httpx.Response:
        """One Graph request. The token travels as a header only: URLs end up in access logs."""
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                transport=self.transport,
                headers={"Authorization": f"Bearer {self.access_token}"},
            ) as client:
                response = client.request(method, url, json=json_body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            # The provider message is httpx's own and can carry the url; the class is enough.
            raise WhatsAppError(f"graph request failed: {type(error).__name__}") from error
        return response

    def send_text(self, to: str, body: str) -> None:
        self.request(
            "POST",
            url=f"{self.base_url}/{self.phone_number_id}/messages",
            json_body={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": body, "preview_url": False},
            },
        )

    def mark_seen(self, message_id: str) -> None:
        self.request(
            "POST",
            url=f"{self.base_url}/{self.phone_number_id}/messages",
            json_body={
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
            },
        )

    def typing_on(self, message_id: str) -> None:
        self.request(
            "POST",
            url=f"{self.base_url}/{self.phone_number_id}/messages",
            json_body={
                "messaging_product": "whatsapp",
                "status": "typing",
                "message_id": message_id,
                "typing": {"typing": True},
            },
        )

    def download(self, media_id: str) -> bytes:
        """Fetch a photo by its media id: the pointer first, then the bytes it points at."""
        pointer = self.request("GET", url=f"{self.base_url}/{media_id}")
        try:
            link = pointer.json()["url"]
        except (ValueError, KeyError, TypeError) as error:
            raise WhatsAppError(f"media {media_id} came back with no download url") from error
        if not isinstance(link, str) or not link:
            raise WhatsAppError(f"media {media_id} came back with no download url")
        return self.request("GET", url=link).content


class WhatsAppMediaSource:
    """Turns a WhatsApp media id into validated photo bytes, with the same gates as a local file."""

    def __init__(self, client: GraphClient) -> None:
        self.client = client

    def fetch(self, media: MediaRef) -> ImagePayload:
        if media.source != "whatsapp_media":
            raise MediaError(f"I cannot read a {media.source} attachment from the WhatsApp server.")
        try:
            data = self.client.download(media.locator)
        except WhatsAppError as error:
            raise MediaError("it could not be downloaded from WhatsApp") from error
        if len(data) > MAX_IMAGE_BYTES:
            megabytes = MAX_IMAGE_BYTES // (1024 * 1024)
            raise MediaError(f"that image is larger than the {megabytes} MB I can look at.")
        mime = sniff_image(data)
        if mime is None:
            raise MediaError("that file is not a jpeg, png, or webp image.")
        return ImagePayload(data=data, mime_type=mime)
