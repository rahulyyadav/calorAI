"""Phase 6: the trace, not the transcript.

These tests hold the two promises observability makes: one id stitches every line of a turn
together, and every line is a duration and an intent with none of the user's words in it.
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import pytest

from calorai_agent.observability import (
    LOG_FORMAT_ENV,
    LOG_LEVEL_ENV,
    TRACING_ENV,
    TRACING_KEY_ENV,
    TRACING_PROJECT_ENV,
    configure_logging,
    configure_tracing,
    current_trace,
    log_event,
    log_format_from_env,
    log_level_from_env,
    new_trace_id,
    span,
    trace_scope,
    tracing_active,
)

LOGGER = logging.getLogger("calorai_agent.test_observability")


@pytest.fixture
def restored() -> Iterator[logging.Logger]:
    """The package logger as the test process found it, handlers and all."""
    root = logging.getLogger("calorai_agent")
    saved = (root.handlers[:], root.level, root.propagate)
    yield root
    root.handlers, root.level, root.propagate = saved


def _installed(fmt: str) -> io.StringIO:
    """Configure the real handler, then point it at a buffer instead of a terminal.

    A handler built on `sys.stderr` under a test runner writes into whoever last replaced it,
    which is not the capture the assertions are reading. The buffer is the same object the
    production handler would have used, so nothing about the formatting is being faked.
    """
    configure_logging(fmt=fmt, level="DEBUG")
    out = io.StringIO()
    logging.getLogger("calorai_agent").handlers[0].setStream(out)
    return out


@pytest.fixture
def records(restored: logging.Logger) -> Iterator[Any]:
    """The JSON lines the package logger has written so far."""
    out = _installed("json")

    def read() -> list[dict[str, Any]]:
        return [json.loads(line) for line in out.getvalue().splitlines() if line]

    yield read


def test_a_trace_id_names_every_line_logged_inside_its_scope() -> None:
    with trace_scope("trace-1") as value:
        assert value == "trace-1"
        assert current_trace() == "trace-1"
    assert current_trace() == ""


def test_an_unnamed_scope_inherits_the_one_it_sits_in() -> None:
    """A turn inside a delivery keeps the delivery's id, so one grep finds the whole story."""
    with trace_scope("outer"), trace_scope() as inner:
        assert inner == "outer"


def test_a_scope_with_nothing_above_it_mints_its_own_id() -> None:
    with trace_scope() as value:
        assert len(value) == 12
    assert current_trace() == ""


def test_ids_are_short_enough_to_paste_and_wide_enough_to_be_unique() -> None:
    assert len({new_trace_id() for _ in range(200)}) == 200


def test_a_logged_event_carries_its_fields_and_its_trace(records: Any) -> None:
    with trace_scope("abc"):
        log_event(LOGGER, "model_request", model="gpt-test", duration_ms=12.5)

    record = records()[-1]
    assert record["event"] == "model_request"
    assert record["model"] == "gpt-test"
    assert record["trace_id"] == "abc"
    assert record["level"] == "info"


def test_an_absent_field_is_left_out_rather_than_logged_as_null(records: Any) -> None:
    log_event(LOGGER, "turn", route="get_totals", photo=None)
    assert "photo" not in records()[-1]


def test_a_warning_keeps_its_level(records: Any) -> None:
    log_event(LOGGER, "vision_failed", level=logging.WARNING, reason="unreadable")
    assert records()[-1]["level"] == "warning"


def test_a_span_reports_how_long_the_work_took(records: Any) -> None:
    with span(LOGGER, "media_fetch", kind="local_path") as report:
        report["bytes"] = 40

    record = records()[-1]
    assert record["event"] == "media_fetch"
    assert record["kind"] == "local_path"
    assert record["status"] == "ok"
    assert record["bytes"] == 40
    assert record["duration_ms"] >= 0


def test_a_span_that_ends_in_a_failure_still_reports(records: Any) -> None:
    with pytest.raises(ZeroDivisionError), span(LOGGER, "turn", channel="cli"):
        raise ZeroDivisionError

    record = records()[-1]
    assert record["status"] == "error"
    assert record["level"] == "warning"
    assert record["duration_ms"] >= 0


def test_the_text_view_shows_the_same_fields_as_json(restored: logging.Logger) -> None:
    out = _installed("text")
    with trace_scope("t1"):
        log_event(LOGGER, "planner_fast_path", kind="get_totals")

    written = out.getvalue()
    assert "trace=t1" in written
    assert "planner_fast_path kind=get_totals" in written


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_a_failure_log_keeps_its_traceback(restored: logging.Logger, fmt: str) -> None:
    out = _installed(fmt)
    try:
        raise ValueError("the plate was unreadable")
    except ValueError:
        LOGGER.exception("agent failed")

    assert "ValueError: the plate was unreadable" in out.getvalue()


def test_configuring_logging_twice_does_not_double_every_line(restored: logging.Logger) -> None:
    configure_logging(fmt="text")
    configure_logging(fmt="json")
    assert len(restored.handlers) == 1


def test_the_handler_only_listens_to_this_package(restored: logging.Logger) -> None:
    configure_logging(fmt="text")
    assert restored.propagate is False
    assert restored.name == "calorai_agent"


def test_the_level_env_wins_over_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    assert log_level_from_env("INFO") == "INFO"
    monkeypatch.setenv(LOG_LEVEL_ENV, "WARNING")
    assert log_level_from_env("INFO") == "WARNING"


def test_the_format_env_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_FORMAT_ENV, " JSON ")
    assert log_format_from_env("text") == "json"


def test_tracing_stays_off_when_nobody_supplied_a_key(
    monkeypatch: pytest.MonkeyPatch, records: Any
) -> None:
    monkeypatch.delenv(TRACING_KEY_ENV, raising=False)
    assert configure_tracing(enabled=True, project="calorai") is False
    assert tracing_active() is False
    # Saying so is the point: a demo that quietly stops tracing looks identical to one that started.
    assert records()[-1]["event"] == "tracing_disabled"


def test_tracing_turns_on_with_a_key_and_names_the_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TRACING_KEY_ENV, "ls-test-key")
    assert configure_tracing(enabled=True, project="meal-agent") is True
    assert tracing_active() is True
    assert os.environ[TRACING_ENV] == "true"
    assert os.environ[TRACING_PROJECT_ENV] == "meal-agent"


def test_turning_tracing_off_writes_an_explicit_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRACING_KEY_ENV, "ls-test-key")
    assert configure_tracing(enabled=False) is False
    assert os.environ[TRACING_ENV] == "false"
    assert tracing_active() is False


def test_a_turn_line_names_no_thing_a_user_ever_ate(records: Any) -> None:
    """A turn line is an id, an intent and a duration. Never a food, never a sentence."""
    with span(LOGGER, "turn", channel="whatsapp", route="log_meal") as report:
        report["photo"] = True
    assert set(records()[-1]) == {
        "ts",
        "level",
        "logger",
        "trace_id",
        "event",
        "channel",
        "route",
        "status",
        "duration_ms",
        "photo",
    }
