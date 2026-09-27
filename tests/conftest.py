from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from calorai_agent.db import Database
from calorai_agent.graph import MealAgent
from calorai_agent.planning import RuleBasedPlanner
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools

DAY = 26


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.sqlite3")
    database.initialize()
    return database


@pytest.fixture
def repository(database: Database) -> Iterator[MealRepository]:
    repository = MealRepository(database)
    repository.ensure_user("user-1", "UTC")
    yield repository


@pytest.fixture
def agent(repository: MealRepository) -> MealAgent:
    return build_agent(repository)


@pytest.fixture
def make_agent(repository: MealRepository) -> Callable[[], MealAgent]:
    """Build fresh agent instances over the same database, like a process restart."""

    return lambda: build_agent(repository)


@pytest.fixture
def clock() -> Callable[..., datetime]:
    """Fixed wall clock so conversation tests stay deterministic."""

    def at(hour: int, minute: int = 0, day: int = DAY) -> datetime:
        return datetime(2026, 9, day, hour, minute, tzinfo=UTC)

    return at


def build_agent(repository: MealRepository) -> MealAgent:
    return MealAgent(RuleBasedPlanner(), MealTools(repository))
