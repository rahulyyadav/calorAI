"""Phase 5: the HTTPS-facing half of the WhatsApp transport.

Meta expects a fast 200 and retries anything slower, so a verified delivery is acknowledged and
then answered on a worker thread. Everything the request thread does is bounded: hash one body,
parse one JSON object, hand it over.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from calorai_agent.config import Settings
from calorai_agent.domain import InboundMessage
from calorai_agent.graph import MealAgent
from calorai_agent.repository import MealRepository
from calorai_agent.tools import RecordInboundInput
from calorai_agent.whatsapp import (
    CHANNEL,
    GraphClient,
    WebhookBatch,
    WhatsAppError,
    normalize_webhook,
    parse_webhook_body,
    signature_matches,
    webhook_challenge,
)

logger = logging.getLogger(__name__)

# What the user is told when the agent itself fails. Answering that the message was not finished
# is the only honest option: a reply we could not send is not a meal that was logged.
INTERNAL_FAILURE_REPLY = (
    "I could not finish that message. Ask for your totals and tell me if anything is missing."
)

Body = tuple[int, str]


class InlineExecutor:
    """Runs submitted work immediately. Tests use it so a webhook can be asserted in one call."""

    def submit(self, work: Callable[[], None]) -> None:
        work()


class WebhookApplication:
    """Verify, acknowledge, answer — with no knowledge of sockets."""

    def __init__(
        self,
        agent: MealAgent,
        client: GraphClient,
        *,
        verify_token: str,
        app_secret: str,
        repository: MealRepository,
        timezone: str = "UTC",
        allowed_users: frozenset[str] = frozenset(),
        executor: Any = None,
    ) -> None:
        self.agent = agent
        self.client = client
        self.verify_token = verify_token
        self.app_secret = app_secret
        self.repository = repository
        self.timezone = timezone
        self.allowed_users = allowed_users
        self.executor = executor if executor is not None else InlineExecutor()

    def get(self, query: Mapping[str, list[str]]) -> Body:
        """Meta's subscription handshake: one exact challenge echo, or a refusal."""
        challenge = webhook_challenge(
            mode=_one(query, "hub.mode"),
            token=_one(query, "hub.verify_token"),
            challenge=_one(query, "hub.challenge"),
            verify_token=self.verify_token,
        )
        if challenge is None:
            return 403, "verification failed"
        return 200, challenge

    def post(self, signature: str | None, body: bytes) -> Body:
        """Accept one delivery. Nothing runs on this thread but a hash and a parse."""
        if not signature_matches(app_secret=self.app_secret, body=body, signature=signature):
            logger.warning("rejected webhook with an unusable X-Hub-Signature-256")
            return 401, "invalid signature"
        payload = parse_webhook_body(body)
        if payload is None:
            # Acknowledged on purpose: an unsigned or unparseable body that returns 4xx is
            # retried forever, and a retry of junk is still junk.
            logger.warning("acknowledged webhook body that was not a JSON object")
            return 200, "ok"
        batch, refused = normalize_webhook(
            payload, timezone=self.timezone, allowed_users=self.allowed_users
        )
        for sender in refused:
            logger.warning("ignored %s message(s) from unlisted sender %s", CHANNEL, sender)
        if not batch.empty:
            self.executor.submit(lambda: self.answer(batch))
        return 200, "ok"

    def answer(self, batch: WebhookBatch) -> None:
        """Do the work behind the acknowledgement: every meal, then every honest decline."""
        for message in batch.messages:
            self._answer_message(message)
        for user_id, external_id, reply in batch.declines:
            self._answer_decline(user_id, external_id, reply)

    def _answer_message(self, message: InboundMessage) -> None:
        self.repository.ensure_user(message.user_id, self.timezone)
        self._seen(message)
        try:
            reply = self.agent.handle(message)
        except Exception:  # noqa: BLE001 - one user's failure must not stop the worker
            logger.exception("agent failed for message %s", message.external_id)
            self._send(message.user_id, INTERNAL_FAILURE_REPLY)
            return
        self._send(message.user_id, reply)

    def _answer_decline(self, user_id: str, external_id: str, reply: str) -> None:
        """Declines share the inbound ledger, so a redelivered voice note declines once.

        The user row comes first: a voice note from a number that has never messaged before would
        otherwise violate the ledger's foreign key before it could be answered.
        """
        self.repository.ensure_user(user_id, self.timezone)
        event = self.agent.tools.record_inbound(
            RecordInboundInput(user_id=user_id, external_id=external_id, text="", channel=CHANNEL)
        )
        if not event.created:
            return
        self._send(user_id, reply)
        self.agent.tools.complete_inbound(event.id, reply)

    def _seen(self, message: InboundMessage) -> None:
        """Best-effort read receipt and typing indicator; a failure here is not the user's."""
        try:
            self.client.mark_seen(message.external_id)
            self.client.typing_on(message.external_id)
        except WhatsAppError as error:
            logger.warning("could not acknowledge message %s: %s", message.external_id, error)

    def _send(self, to: str, body: str) -> None:
        try:
            self.client.send_text(to, body)
        except WhatsAppError as error:
            logger.error("reply to %s failed: %s", to, error)


def _one(query: Mapping[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    return values[0] if values else None


def build_handler(application: WebhookApplication, path: str) -> type[BaseHTTPRequestHandler]:
    """The thin HTTP shell around `WebhookApplication`: read, dispatch, write."""

    class WhatsAppHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - the name is fixed by BaseHTTPRequestHandler
            if urlparse(self.path).path != path:
                self._respond((404, "not found"))
                return
            self._respond(application.get(parse_qs(urlparse(self.path).query)))

        def do_POST(self) -> None:  # noqa: N802 - the name is fixed by BaseHTTPRequestHandler
            if urlparse(self.path).path != path:
                self._respond((404, "not found"))
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            self._respond(application.post(self.headers.get("X-Hub-Signature-256"), body))

        def _respond(self, result: Body) -> None:
            status, text = result
            payload = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, message: str, *args: Any) -> None:
            # Access lines go through the app logger, which never sees bodies or tokens.
            logger.debug("webhook %s - %s", self.address_string(), message % args)

    return WhatsAppHandler


def serve(settings: Settings, application: WebhookApplication) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(
        (settings.webhook_host, settings.webhook_port),
        build_handler(application, settings.webhook_path),
    )
    logger.info(
        "webhook listening on http://%s:%d%s",
        settings.webhook_host,
        settings.webhook_port,
        settings.webhook_path,
    )
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the WhatsApp Cloud API webhook")
    parser.add_argument("--host", help="Override CALORAI_WEBHOOK_HOST (default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Override CALORAI_WEBHOOK_PORT")
    parser.add_argument(
        "--path", help="Override CALORAI_WEBHOOK_PATH (must match Meta's callback URL)"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    if not settings.use_whatsapp:
        print(
            "WhatsApp needs CALORAI_WHATSAPP_VERIFY_TOKEN, CALORAI_WHATSAPP_APP_SECRET, "
            "CALORAI_WHATSAPP_ACCESS_TOKEN and CALORAI_WHATSAPP_PHONE_NUMBER_ID in the "
            "environment. Copy .env.example, fill it in, and never commit the result."
        )
        return 2
    if not settings.whatsapp_allowed_users:
        print(
            "CALORAI_WHATSAPP_ALLOWED_USERS is empty, so every sender would be refused. "
            "List the wa_ids you are testing with, separated by commas."
        )
        return 2

    from calorai_agent.app import create_whatsapp_app  # late import: --help stays cheap

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    application = create_whatsapp_app(settings)
    host = args.host or settings.webhook_host
    port = args.port or settings.webhook_port
    path = args.path or settings.webhook_path
    served = replace(settings, webhook_host=host, webhook_port=port, webhook_path=path)
    server = serve(served, application)
    print(
        f"CalorAI webhook on http://{host}:{port}{path}.\n"
        "Meta needs a public HTTPS url: tunnel this port (cloudflared or ngrok) and paste that "
        "url, path included, into the app's WhatsApp webhook callback."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
