"""Phase 5: the HTTPS-facing half of the WhatsApp transport.

Meta expects a fast 200 and retries anything slower, so a verified delivery is acknowledged and
then answered on a worker thread. Everything the request thread does is bounded: hash one body,
parse one JSON object, hand it over.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
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
    is_fingerprinted_event,
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

# Meta's own payloads are a few kilobytes. A declared body this much bigger is not a delivery, and
# reading it into RAM before verifying it is the cheapest denial of service available on an open
# HTTPS endpoint.
MAX_WEBHOOK_BODY_BYTES = 1_000_000


class InlineExecutor:
    """Runs submitted work immediately. Tests use it so a webhook can be asserted in one call."""

    def submit(self, work: Callable[[], None]) -> None:
        work()

    def shutdown(self) -> None:
        return None


class WorkerExecutor:
    """Answers on a worker thread, so the acknowledgement is written before the meal is worked on.

    Meta retries anything slower than its timeout, and a retry of a turn this process is still
    running would otherwise be a second connection thread doing the same work.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="calorai-reply")

    def submit(self, work: Callable[[], None]) -> None:
        self.pool.submit(work)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


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
        # The default is the production one on purpose: an inline default would silently turn a
        # slow model into a retry storm, and only a test should opt out of the worker.
        self.executor = executor if executor is not None else WorkerExecutor()

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
        """Do the work behind the acknowledgement: every meal, then every honest decline.

        Each item is answered on its own terms. The request thread has already promised Meta a 200,
        so a locked database or a lost reply is that one message's failure, not the delivery's.
        """
        for message in batch.messages:
            try:
                self._answer_message(message)
            except Exception:  # noqa: BLE001 - one message must not end the others
                logger.exception("could not answer message %s", message.external_id)
        for user_id, external_id, reply in batch.declines:
            try:
                self._answer_decline(user_id, external_id, reply)
            except Exception:  # noqa: BLE001
                logger.exception("could not decline message %s", external_id)

    def shutdown(self) -> None:
        """Let answers already in flight finish before the process leaves."""
        stop = getattr(self.executor, "shutdown", None)
        if callable(stop):
            stop()

    def _answer_message(self, message: InboundMessage) -> None:
        self._seen(message)
        try:
            self.repository.ensure_user(message.user_id, self.timezone)
            reply = self.agent.handle(message)
        except Exception:  # noqa: BLE001 - the user deserves to know it did not land
            logger.exception("agent failed for message %s", message.external_id)
            reply = INTERNAL_FAILURE_REPLY
            self._close_failed_turn(message, reply)
        self._send(message.user_id, reply)

    def _close_failed_turn(self, message: InboundMessage, reply: str) -> None:
        """Finish the ledger row a failed turn left open.

        An event that stays incomplete makes every later retry answer "still working on it", which
        is a promise this process has already stopped keeping. Closing it with the honest failure
        reply means a retry says the same true thing.
        """
        try:
            event = self.agent.tools.record_inbound(
                RecordInboundInput(
                    user_id=message.user_id,
                    external_id=message.external_id,
                    text=message.text,
                    channel=message.channel,
                )
            )
            self.agent.tools.complete_inbound(event.id, reply)
        except Exception:  # noqa: BLE001 - already the failure path, with nowhere left to report
            logger.exception("could not ledger the failure of message %s", message.external_id)

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
        """The read receipt and typing indicator, best-effort: a failure here is not the user's.

        Meta shows typing on the same read request, so one call says both that the message was seen
        and that an answer is coming.
        """
        if is_fingerprinted_event(message.external_id):
            # This adapter invented that id because Meta sent none, so there is no message to mark.
            return
        try:
            self.client.mark_seen(message.external_id, typing=True)
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
            declared = self._declared_length()
            if declared is None:
                self._close_after((400, "invalid Content-Length"))
                return
            if declared > MAX_WEBHOOK_BODY_BYTES:
                # Nothing here was signed, so nothing here is a delivery: an endpoint on a public
                # tunnel must not let a stranger choose how much of our memory gets read in.
                logger.warning("refused webhook body of %d bytes", declared)
                self._close_after((413, "body too large"))
                return
            body = self.rfile.read(declared) if declared else b""
            self._respond(application.post(self.headers.get("X-Hub-Signature-256"), body))

        def _declared_length(self) -> int | None:
            """The body size this connection claims, or None when the claim is not a size."""
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                return None
            return length if length >= 0 else None

        def _close_after(self, result: Body) -> None:
            # The body is deliberately left unread, so this connection cannot be reused.
            self.close_connection = True
            self._respond(result)

        def _respond(self, result: Body) -> None:
            status, text = result
            payload = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, message: str, *args: Any) -> None:
            # Access lines go through the app logger, which never sees bodies or tokens — and the
            # handshake carries its verify token in the query string, so a logged request line is
            # cut at the first "?". The path alone says what the request was for.
            logger.debug(
                "webhook %s - %s", self.address_string(), (message % args).split("?", 1)[0]
            )

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
        # A message already being answered is finished before the process leaves, so the user is
        # never left with a read receipt and no reply.
        application.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
