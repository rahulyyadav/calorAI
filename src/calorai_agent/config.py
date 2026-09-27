from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PLANNER_DETERMINISTIC = "deterministic"
PLANNER_MODEL = "model"
PLANNER_AUTO = "auto"


def _env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _optional_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path
    default_user_id: str
    default_timezone: str
    planner: str = PLANNER_AUTO
    text_model: str = "gpt-4o-mini"
    text_model_api_key: str | None = None
    text_model_base_url: str = "https://api.openai.com/v1"
    vision_model: str = "gpt-4o-mini"
    vision_model_api_key: str | None = None
    vision_model_base_url: str = "https://api.openai.com/v1"
    request_timeout_seconds: float = 20.0

    @classmethod
    def from_env(cls) -> Settings:
        text_api_key = _optional_env("CALORAI_TEXT_MODEL_API_KEY", "OPENAI_API_KEY")
        return cls(
            database_path=Path(_env("CALORAI_DB_PATH", default="data/calorai.sqlite3")),
            default_user_id=_env("CALORAI_USER_ID", default="local-demo-user"),
            default_timezone=_env("CALORAI_TIMEZONE", default="UTC"),
            planner=_env("CALORAI_PLANNER", default=PLANNER_AUTO),
            text_model=_env("CALORAI_TEXT_MODEL", "OPENAI_MODEL", default="gpt-4o-mini"),
            text_model_api_key=text_api_key,
            text_model_base_url=_env(
                "CALORAI_TEXT_MODEL_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.openai.com/v1",
            ),
            vision_model=_env("CALORAI_VISION_MODEL", default="gpt-4o-mini"),
            # A photo is worthless without a vision endpoint, and one key for the same
            # provider is less to type. Left unset together, both paths stay deterministic.
            vision_model_api_key=_optional_env("CALORAI_VISION_MODEL_API_KEY") or text_api_key,
            vision_model_base_url=_env(
                "CALORAI_VISION_MODEL_BASE_URL",
                "CALORAI_TEXT_MODEL_BASE_URL",
                "OPENAI_BASE_URL",
                default="https://api.openai.com/v1",
            ),
        )

    @property
    def use_model_planner(self) -> bool:
        if self.planner == PLANNER_MODEL:
            return True
        if self.planner == PLANNER_DETERMINISTIC:
            return False
        return self.text_model_api_key is not None

    @property
    def use_vision(self) -> bool:
        return self.vision_model_api_key is not None
