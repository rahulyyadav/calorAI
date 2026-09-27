#!/usr/bin/env python3
"""Run the Phase 4 photo evals against the agent with a scripted vision answer.

Nothing here needs an API key or the network: the "model" is the JSON in
`evals/photo_scenarios.json`, so a reviewer sees exactly which plate the agent was shown and
exactly what it decided. The assertions are the phase's claims — one photo is one meal, the
caption modifies that meal instead of logging a second one, and a plate too shaky to price asks
one question rather than inventing a number.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from calorai_agent.db import Database
from calorai_agent.domain import InboundMessage, MediaRef
from calorai_agent.graph import MealAgent
from calorai_agent.planning import RuleBasedPlanner
from calorai_agent.providers import ImagePayload
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools
from calorai_agent.vision import LocalFileMediaSource, VisionInterpreter

NOW = datetime(2026, 9, 26, 13, tzinfo=UTC)
DAY = NOW.date()
USER = "eval-user"
# Enough bytes to pass the signature check; the scripted model never looks at them.
PHOTO = b"\xff\xd8\xff\xe0" + b"eval plate" * 8


class ScriptedVisionClient:
    """Answers every photo with the scenario's JSON, and counts how many photos it was shown."""

    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.calls = 0

    def observe(self, *, system: str, user: str, image: ImagePayload) -> str:
        self.calls += 1
        return json.dumps(self.answer)


def build_agent(
    answer: dict[str, Any], workdir: Path
) -> tuple[MealAgent, MealRepository, ScriptedVisionClient]:
    database = Database(workdir / "eval.sqlite3")
    database.initialize()
    repository = MealRepository(database)
    repository.ensure_user(USER, "UTC")
    client = ScriptedVisionClient(answer)
    interpreter = VisionInterpreter(client, LocalFileMediaSource(), model="scripted-eval-model")
    return (
        MealAgent(RuleBasedPlanner(), MealTools(repository), vision=interpreter),
        repository,
        client,
    )


def run(scenario: dict[str, Any], workdir: Path) -> tuple[list[str], str]:
    """Return the failures for one scenario (empty means it passed) and the reply to show."""
    photo = workdir / "plate.jpg"
    photo.write_bytes(PHOTO)
    media = MediaRef(external_id=scenario["name"], locator=str(photo))
    agent, repository, client = build_agent(scenario["vision_answer"], workdir)

    reply = ""
    for _ in range(int(scenario.get("sends", 1))):
        reply = agent.handle(
            InboundMessage(
                user_id=USER,
                text=scenario["caption"],
                external_id=f"eval:{scenario['name']}",
                channel="cli",
                timezone="UTC",
                received_at=NOW,
                media=media,
            )
        )

    expect: dict[str, Any] = scenario["expect"]
    meals = repository.list_for_day(USER, DAY)
    failures: list[str] = []

    def check(condition: bool, wanted: str) -> None:
        if not condition:
            failures.append(wanted)

    check(len(meals) == expect["meals"], f"expected {expect['meals']} meal(s), logged {len(meals)}")
    asked = "?" in reply
    check(asked is expect["asks"], f"expected a question: {expect['asks']}, reply was {reply!r}")
    check(
        client.calls == expect.get("vision_calls", 1),
        f"expected the photo shown to the model {expect.get('vision_calls', 1)} time(s), "
        f"it was shown {client.calls}",
    )
    if "items" in expect and meals:
        names = [item.name for item in meals[-1].items]
        check(names == expect["items"], f"expected items {expect['items']}, got {names}")
    if "kcal" in expect and meals:
        total = sum((meal.nutrition.calories for meal in meals), Decimal(0))
        check(total == Decimal(expect["kcal"]), f"expected {expect['kcal']} kcal, got {total}")
    for fragment in expect.get("reply_contains", []):
        check(fragment in reply, f"expected the reply to contain {fragment!r}")

    return failures, reply


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the photo eval scenarios")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evals" / "photo_scenarios.json",
        help="Scenario file (default: evals/photo_scenarios.json)",
    )
    args = parser.parse_args()
    scenarios = json.loads(args.scenarios.read_text())["scenarios"]

    failed = 0
    for index, scenario in enumerate(scenarios, start=1):
        with tempfile.TemporaryDirectory(prefix="calorai-eval-") as workdir:
            failures, reply = run(scenario, Path(workdir))
        status = "PASS" if not failures else "FAIL"
        print(f"{status} {index:02d} {scenario['name']}")
        print(f"     > {reply}")
        for failure in failures:
            print(f"     ! {failure}")
        failed += bool(failures)

    print(f"\n{len(scenarios) - failed}/{len(scenarios)} photo scenarios passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
