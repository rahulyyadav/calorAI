"""Phase 6: every turn carries a trace id and every log line is one machine-readable record.

Two rules shape this module. A duration is measured where the work happens, so a slow model
reads as a model number rather than a vague complaint about latency. And nothing a user *said*
is ever a log field: meal text is health data, so events carry ids, intents, models and timings
and leave the plate out.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

LOG_FORMAT_ENV = "CALORAI_LOG_FORMAT"
LOG_LEVEL_ENV = "CALORAI_LOG_LEVEL"

# LangSmith is read from the environment by the LangChain runtime, so opting in means setting
# the variables it looks for before the graph is first compiled — and nothing at all when they
# are absent, because tracing a meal-logging agent to a third party is never the default.
TRACING_ENV = "LANGCHAIN_TRACING_V2"
TRACING_KEY_ENV = "LANGCHAIN_API_KEY"
TRACING_PROJECT_ENV = "LANGCHAIN_PROJECT"

FIELDS_ATTR = "calorai_fields"

_root = logging.getLogger("calorai_agent")
_trace_id: ContextVar[str] = ContextVar("calorai_trace_id", default="")


def new_trace_id() -> str:
    return uuid4().hex[:12]


def current_trace() -> str:
    """The id this call is running under, or "" outside any scope."""
    return _trace_id.get()


@contextmanager
def trace_scope(trace_id: str = "") -> Iterator[str]:
    """Correlate every line logged from here on, inheriting the enclosing id when given none.

    A turn opened by the transport already has an id the reviewer can match against a delivery,
    so the agent's own scope must not mint a second one for the same message.
    """
    value = trace_id or current_trace() or new_trace_id()
    token = _trace_id.set(value)
    try:
        yield value
    finally:
        _trace_id.reset(token)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Log one named event with its fields, whatever format the deployment asked for."""
    logged = {key: value for key, value in fields.items() if value is not None}
    logger.log(level, event, extra={FIELDS_ATTR: logged})


class _TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = current_trace()
        return True


def _fields_of(record: logging.LogRecord) -> dict[str, Any]:
    fields = getattr(record, FIELDS_ATTR, None)
    if isinstance(fields, Mapping):
        return {"event": record.getMessage(), **fields}
    return {"event": record.getMessage()}


def _stamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    """One JSON object per line: parseable by `jq`, and identical in content to the text view."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _stamp(record),
            "level": record.levelname.lower(),
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", ""),
            **_fields_of(record),
        }
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """The human reading of the same record, for a terminal during a live walkthrough."""

    def format(self, record: logging.LogRecord) -> str:
        fields = _fields_of(record)
        event = str(fields.pop("event"))
        trace = getattr(record, "trace_id", "") or "-"
        head = f"{_stamp(record)} {record.levelname.lower():7} trace={trace}"
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        line = f"{head} {event} {rendered}".rstrip()
        if record.exc_info:
            # Overriding format() skips the traceback the base class appends, and a failure log
            # without one is the least useful line in the file.
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(*, fmt: str = "text", level: str | int = "INFO") -> logging.Logger:
    """Put one handler on the package logger, replacing any earlier one.

    Called by each entrypoint rather than at import: a library that installs handlers owns the
    application's output, and the test runner is one such application.
    """
    _root.handlers.clear()
    handler = logging.StreamHandler()
    handler.addFilter(_TraceFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    _root.addHandler(handler)
    _root.setLevel(logging.getLevelName(level) if isinstance(level, str) else level)
    _root.propagate = False
    return _root


def log_level_from_env(default: str = "INFO") -> str:
    return os.getenv(LOG_LEVEL_ENV) or default


def log_format_from_env(default: str = "text") -> str:
    return (os.getenv(LOG_FORMAT_ENV) or default).strip().lower()


@contextmanager
def span(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time one unit of work and log it exactly once, however it ends.

    The yielded dict is writable, so the work being measured can add what it learned — a model
    name, a byte count — without the caller having to know where those values come from. A span
    that ends through an exception still reports, because an error with no duration is the one
    log line a reviewer always wishes for.
    """
    reported: dict[str, Any] = dict(fields)
    started = time.perf_counter()
    status = "error"
    try:
        yield reported
        status = "ok"
    finally:
        reported["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        log_event(
            logger,
            event,
            level=logging.WARNING if status == "error" else logging.INFO,
            status=status,
            **reported,
        )


def configure_tracing(*, enabled: bool, project: str = "calorai") -> bool:
    """Ask the LangChain runtime to trace, and say whether it is actually in a position to.

    Enabling without a key is not a half-configured upload: the graph runs untraced and says so,
    because silently buffering spans is how a demo ends up sending them somewhere later.
    """
    if enabled and not os.getenv(TRACING_KEY_ENV):
        log_event(
            _root,
            "tracing_disabled",
            level=logging.WARNING,
            reason="no api key",
            project=project,
        )
        return False
    os.environ[TRACING_ENV] = "true" if enabled else "false"
    if enabled:
        os.environ[TRACING_PROJECT_ENV] = project
    return enabled


def tracing_active() -> bool:
    return os.getenv(TRACING_ENV) == "true" and bool(os.getenv(TRACING_KEY_ENV))
