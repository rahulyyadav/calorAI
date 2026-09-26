from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    default_user_id: str
    default_timezone: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_path=Path(os.getenv("CALORAI_DB_PATH", "data/calorai.sqlite3")),
            default_user_id=os.getenv("CALORAI_USER_ID", "local-demo-user"),
            default_timezone=os.getenv("CALORAI_TIMEZONE", "UTC"),
        )
