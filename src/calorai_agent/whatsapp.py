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
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from calorai_agent.domain import InboundMessage, MediaRef
from calorai_agent.observability import log_event, span
from calorai_agent.providers import ImagePayload
from calorai_agent.vision import MAX_IMAGE_BYTES, MediaError, sniff_image

logger = logging.getLogger(__name__)

SIGNATURE_PREFIX = "sha256="
CHANNEL = "whatsapp"
SYNTHETIC_EVENT_PREFIX = "wamid-sha:"
# How many downloaded photos the process holds, and how many it will fetch for one delivery.
# Both are ceilings on the same thing: a photo is up to 8 MB of someone's dinner held in RAM.
MAX_CACHED_MEDIA_BYTES = 32 * 1024 * 1024
MAX_MEDIA_PER_DELIVERY = 6
# One day of slack for clock skew, and no more: a stamp further ahead is not a send time, and a
# meal dated into another century quietly leaves every totals question and recent-meal window.
MAX_SENT_AHEAD_SECONDS = 86_400

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
                # A block that names a conversation but carries no message (a shared contact card,
                # say) still tells us who is on the other end. Only an unlisted one is refused.
                if block_sender not in allowed_users:
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
    return f"{SYNTHETIC_EVENT_PREFIX}{digest}"


def is_fingerprinted_event(external_id: str) -> bool:
    """True for the id this adapter invented when Meta sent none.

    Such a message has no WhatsApp message id at all, so there is nothing to mark as read: the
    receipt would be sent against an id Meta never issued.
    """
    return external_id.startswith(SYNTHETIC_EVENT_PREFIX)


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
        sent = datetime.fromtimestamp(seconds, UTC)
    except (OSError, OverflowError, ValueError):
        return None
    if sent > datetime.now(UTC) + timedelta(seconds=MAX_SENT_AHEAD_SECONDS):
        # A meal dated centuries ahead vanishes from every totals question and every recent-meal
        # window, which is worse than the clock we fall back to when the stamp is missing.
        return None
    return sent


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

    def mark_seen(self, message_id: str, *, typing: bool = False) -> None:
        """The read receipt, optionally carrying the typing indicator that rides on it.

        Meta has no standalone typing request: the indicator is one field of a read status, so a
        single call shows the user both that their message was seen and that an answer is coming.
        """
        body: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        if typing:
            body["typing_indicator"] = {"type": "text"}
        self.request("POST", url=f"{self.base_url}/{self.phone_number_id}/messages", json_body=body)

    def download(self, media_id: str) -> bytes:
        """Fetch a photo by its media id: the pointer first, then the bytes it points at."""
        pointer = self.request("GET", url=f"{self.base_url}/{media_id}")
        try:
            link = pointer.json()["url"]
        except (ValueError, KeyError, TypeError) as error:
            raise WhatsAppError(f"media {media_id} came back with no download url") from error
        # The bearer token goes on this second request, so the pointer must name an https host and
        # not something else a configured base url could steer it to.
        if not isinstance(link, str) or not link.startswith("https://"):
            raise WhatsAppError(f"media {media_id} came back with no download url")
        return self.request("GET", url=link).content


class MediaCache:
    """Photos this process has already paid to download, kept under a hard ceiling.

    A WhatsApp media id names one immutable object, so a cached payload cannot go stale; the only
    thing it can do is grow, which is what the byte budget is for. The lock covers the dictionary
    and never a download, so warming three photos at once really does download three at once.
    Photos are the user's dinner, so nothing here outlives the process and nothing here is logged.
    """

    def __init__(self, *, max_bytes: int = MAX_CACHED_MEDIA_BYTES) -> None:
        self._entries: OrderedDict[str, ImagePayload] = OrderedDict()
        self._bytes = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def get_or_fetch(self, key: str, fetch: Callable[[], ImagePayload]) -> ImagePayload:
        """The photo under `key`, downloaded through `fetch` only if nobody has it yet."""
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                return cached
        payload = fetch()
        with self._lock:
            # Two threads can fetch the same photo while the cache is cold, and whoever finishes
            # second overwrites the first. A duplicate download is the cost; a torn payload is not.
            self._entries[key] = payload
            self._bytes += len(payload.data)
            self._trim()
        return payload

    def warm(self, key: str, fetch: Callable[[], ImagePayload]) -> bool:
        """Fetch into the cache for later, reporting whether the bytes landed in it."""
        try:
            self.get_or_fetch(key, fetch)
        except MediaError as error:
            # The turn that needs this photo will fetch it itself and hear the truth from it.
            log_event(
                logger,
                "media_prefetch_failed",
                level=logging.WARNING,
                media_id=key,
                reason=str(error),
            )
            return False
        return True

    def _trim(self) -> None:
        while self._entries and self._bytes > self._max_bytes:
            _, oldest = self._entries.popitem(last=False)
            self._bytes -= len(oldest.data)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries


class WhatsAppMediaSource:
    """Turns a WhatsApp media id into validated photo bytes, with the same gates as a local file."""

    def __init__(self, client: GraphClient, *, cache: MediaCache | None = None) -> None:
        self.client = client
        self.cache = cache

    def fetch(self, media: MediaRef) -> ImagePayload:
        if self.cache is None:
            return self._read(media)
        return self.cache.get_or_fetch(media.locator, lambda: self._read(media))

    def prefetch(self, media: MediaRef) -> None:
        """Download a photo this delivery is going to need, before the turn asks for it.

        A warm-up cannot fail a user: a photo that will not download is fetched again by the turn
        that needs it, which then gives the same honest answer it would have given anyway.
        """
        if self.cache is not None and media.source == "whatsapp_media":
            self.cache.warm(media.locator, lambda: self._read(media))

    def _read(self, media: MediaRef) -> ImagePayload:
        with span(logger, "media_fetch", kind="whatsapp_media", media_id=media.locator) as report:
            if media.source != "whatsapp_media":
                raise MediaError(f"I cannot read a {media.source} attachment from WhatsApp.")
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
            report["bytes"] = len(data)
            report["mime_type"] = mime
            return ImagePayload(data=data, mime_type=mime)
