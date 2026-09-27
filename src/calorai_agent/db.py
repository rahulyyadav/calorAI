from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Database:
    """SQLite access with explicit migration ordering and serialized write transactions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def migrations(self) -> list[tuple[int, Path]]:
        directory = Path(__file__).with_name("migrations")
        found: list[tuple[int, Path]] = []
        for file in sorted(directory.glob("*.sql")):
            prefix = file.name.split("_", 1)[0]
            if not prefix.isdigit():
                raise ValueError(f"migration {file.name} must start with a numeric version")
            found.append((int(prefix), file))
        return found

    def initialize(self) -> None:
        """Apply unapplied migrations in version order.

        Runs outside a transaction because some statements (journal_mode) cannot
        execute inside one.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            applied = {
                row["version"]
                for row in connection.execute("SELECT version FROM schema_migrations").fetchall()
            }
            for version, file in self.migrations:
                if version in applied:
                    continue
                connection.executescript(file.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (version,)
                )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Autocommit connection for reads and for statements that manage their own txn."""
        connection = self._open()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def write_transaction(self) -> Iterator[sqlite3.Connection]:
        """Take the write lock up front so read-modify-write steps cannot interleave."""
        connection = self._open()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            # Only roll back a live transaction: failing to open one would otherwise
            # replace the real error with "cannot rollback - no transaction is active".
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _open(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
