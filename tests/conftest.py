from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from calorai_agent.db import Database
from calorai_agent.repository import MealRepository


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
