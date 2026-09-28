#!/usr/bin/env python3
"""Run the conversation eval set: the messages from the brief plus the ones that break agents.

Nothing here needs an API key or the network. The planner is the deterministic one and the vision
model is scripted from the scenario, so a failing eval points at the application's judgement
rather than at a provider's mood, and a reviewer can rerun it on a laptop in a second.

Correctness is graded on six things (docs/EVALS.md defines each): the tool a turn chose, the meal
state left behind, the day's totals, whether the agent asked instead of guessing, whether a stored
memory actually reached the reply, and whether a photo and its caption became one meal.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from calorai_agent.db import Database
from calorai_agent.domain import MediaRef
from calorai_agent.graph import MealAgent
from calorai_agent.observability import FIELDS_ATTR
from calorai_agent.planning import RuleBasedPlanner
from calorai_agent.providers import ImagePayload
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools
from calorai_agent.vision import LocalFileMediaSource, VisionInterpreter

USER = "eval-user"
# Enough bytes to pass the signature check; the scripted model never looks at them.
PHOTO = b"\xff\xd8\xff\xe0" + b"eval plate" * 8
MACROS = {"kcal": "calories", "protein_g": "protein_g", "carbs_g": "carbs_g", "fat_g": "fat_g"}


class ScriptedVisionClient:
    """Answers every photo with the scenario's JSON, and counts how many photos it was shown."""

    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.calls = 0

    def observe(self, *, system: str, user: str, image: ImagePayload) -> str:
        self.calls += 1
        return json.dumps(self.answer)


class TurnRecorder(logging.Handler):
    """The routes the agent logged for itself: an eval reads the trace, not a test hook."""

    def __init__(self) -> None:
        super().__init__()
        self.turns: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "turn":
            self.turns.append(dict(getattr(record, FIELDS_ATTR, {})))


@contextmanager
def recorded_turns() -> Iterator[TurnRecorder]:
    logger = logging.getLogger("calorai_agent")
    saved = (logger.handlers[:], logger.level, logger.propagate)
    recorder = TurnRecorder()
    logger.handlers, logger.level, logger.propagate = [recorder], logging.DEBUG, False
    try:
        yield recorder
    finally:
        logger.handlers, logger.level, logger.propagate = saved


def build(db_path: Path, vision: VisionInterpreter | None) -> tuple[MealAgent, MealRepository]:
    """A whole application, the way a fresh process would make one: same file, nothing reused."""
    database = Database(db_path)
    database.initialize()
    repository = MealRepository(database)
    repository.ensure_user(USER, "UTC")
    return MealAgent(RuleBasedPlanner(), MealTools(repository), vision=vision), repository


class Runner:
    """Feeds one scenario's turns to the agent and keeps the replies and the routes."""

    def __init__(self, scenario: dict[str, Any], workdir: Path) -> None:
        self.scenario = scenario
        self.timezone = scenario.get("timezone", "UTC")
        self.start = datetime.fromisoformat(scenario.get("start", "2026-09-26T13:00:00+00:00"))
        self.db_path = workdir / "eval.sqlite3"
        self.photo_dir = workdir
        answer = scenario.get("vision_answer")
        self.client = ScriptedVisionClient(answer) if answer else None
        self.vision = (
            VisionInterpreter(self.client, LocalFileMediaSource(), model="scripted-eval-model")
            if self.client is not None
            else None
        )
        self.agent, self.repository = build(self.db_path, self.vision)
        self.replies: list[str] = []
        self.days: list[date] = []

    def at(self, turn: dict[str, Any]) -> datetime:
        if "at" in turn:
            value = datetime.fromisoformat(turn["at"])
            return value if value.tzinfo else value.replace(tzinfo=ZoneInfo(self.timezone))
        return self.start + timedelta(days=int(turn.get("day", 0)))

    def media(self, turn: dict[str, Any], index: int) -> MediaRef | None:
        if "photo" not in turn:
            return None
        path = self.photo_dir / f"plate-{index}.jpg"
        path.write_bytes(PHOTO)
        return MediaRef(external_id=str(turn.get("photo")), locator=str(path))

    def run(self, recorder: TurnRecorder) -> None:
        for index, turn in enumerate(self.scenario["turns"], start=1):
            if turn.get("new_session"):
                # A new process would rebuild from the same file; so does this scenario.
                self.agent, self.repository = build(self.db_path, self.vision)
            moment = self.at(turn)
            media = self.media(turn, index)
            self.days.append(moment.astimezone(ZoneInfo(self.timezone)).date())
            reply = self._send(turn, moment, media)
            if turn.get("redeliver"):
                # The same delivery arriving twice is what a retried webhook does.
                self._send(turn, moment, media)
            self.replies.append(reply)

    def _send(self, turn: dict[str, Any], moment: datetime, media: MediaRef | None) -> str:
        return self.agent.invoke(
            USER, turn["text"], timezone=self.timezone, now=moment, media=media
        )


def check(
    expect: dict[str, Any], state: Runner, recorder: TurnRecorder, failures: list[str]
) -> None:
    def want(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    routes = [turn.get("route") for turn in recorder.turns]
    if "routes" in expect:
        want(routes == expect["routes"], f"expected routes {expect['routes']}, got {routes}")
    if "asks" in expect:
        asked = ["?" in reply for reply in state.replies]
        want(asked == expect["asks"], f"expected questions {expect['asks']}, got {asked}")
    for turn_index, text in enumerate(expect.get("turn_contains", []), start=1):
        if text is None:
            continue
        reply = state.replies[turn_index - 1]
        want(text in reply, f"expected turn {turn_index} to say {text!r}, it said {reply!r}")

    zone = ZoneInfo(state.timezone)

    def meals_on(offset: str) -> list[Any]:
        day = (state.start + timedelta(days=int(offset))).astimezone(zone).date()
        return state.repository.list_for_day(USER, day, state.timezone)

    for offset, wanted in expect.get("meals", {}).items():
        want(
            len(meals_on(offset)) == wanted,
            f"expected {wanted} meal(s) on day {offset}, found {len(meals_on(offset))}",
        )
    for offset, wanted in expect.get("meal_items", {}).items():
        got = [[item.name for item in meal.items] for meal in meals_on(offset)]
        want(got == wanted, f"expected day {offset} to hold {wanted}, it holds {got}")
    for offset, wanted in expect.get("meal_quantities", {}).items():
        got = [[str(item.quantity) for item in meal.items] for meal in meals_on(offset)]
        want(got == wanted, f"expected day {offset} portions {wanted}, got {got}")
    for offset, wanted in expect.get("totals", {}).items():
        day = (state.start + timedelta(days=int(offset))).astimezone(zone).date()
        totals = state.repository.totals_for_day(USER, day, state.timezone)
        for key, value in wanted.items():
            actual = getattr(totals.nutrition, MACROS[key])
            want(actual == Decimal(value), f"expected day {offset} {key} {value}, got {actual}")

    if "memories" in expect:
        kinds = {record.content.kind.value for record in state.repository.active_memories(USER)}
        want(
            set(expect["memories"]) <= kinds,
            f"expected memories {expect['memories']}, the store holds {sorted(kinds)}",
        )
    if "vision_calls" in expect and state.client is not None:
        want(
            state.client.calls == expect["vision_calls"],
            f"expected the photo shown {expect['vision_calls']} time(s), it was shown "
            f"{state.client.calls}",
        )


def run(scenario: dict[str, Any], workdir: Path) -> tuple[list[str], list[str]]:
    """Return the failures for one scenario (empty means it passed) and the replies to show."""
    state = Runner(scenario, workdir)
    with recorded_turns() as recorder:
        try:
            state.run(recorder)
        except Exception as error:  # a crash is a failed eval, not a stack trace at the reviewer
            return [f"the agent raised {type(error).__name__}: {error}"], state.replies
    failures: list[str] = []
    check(scenario["expect"], state, recorder, failures)
    return failures, state.replies


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the conversation eval scenarios")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals" / "conversation_scenarios.json",
        help="Scenario file (default: evals/conversation_scenarios.json)",
    )
    parser.add_argument("--only", default="", help="Run only scenarios whose name contains this")
    args = parser.parse_args()
    scenarios = json.loads(args.scenarios.read_text())["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if args.only in s["name"]]

    failed = 0
    for index, scenario in enumerate(scenarios, start=1):
        with tempfile.TemporaryDirectory(prefix="calorai-conversation-eval-") as workdir:
            failures, replies = run(scenario, Path(workdir))
        status = "PASS" if not failures else "FAIL"
        print(f"{status} {index:02d} {scenario['name']}")
        for number, reply in enumerate(replies, start=1):
            print(f"     {number}> {reply}")
        for failure in failures:
            print(f"     ! {failure}")
        failed += bool(failures)

    print(f"\n{len(scenarios) - failed}/{len(scenarios)} conversation scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
