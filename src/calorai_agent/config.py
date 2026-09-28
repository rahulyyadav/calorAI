from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PLANNER_DETERMINISTIC = "deterministic"
PLANNER_MODEL = "model"
PLANNER_AUTO = "auto"

GRAPH_API_BASE = "https://graph.facebook.com/v23.0"


def _env_list(*names: str) -> tuple[str, ...]:
    """A comma-separated allow-list, trimmed and deduplicated, keeping order."""
    for name in names:
        value = os.getenv(name)
        if value:
            return tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    return ()


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


def _int_env(name: str, *, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value else default
    except ValueError:
        return default


def _bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, *, default: float) -> float:
    value = os.getenv(name)
    try:
        return float(value) if value else default
    except ValueError:
        return default


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
    whatsapp_verify_token: str | None = None
    whatsapp_app_secret: str | None = None
    whatsapp_access_token: str | None = None
    whatsapp_phone_number_id: str | None = None
    whatsapp_allowed_users: tuple[str, ...] = ()
    graph_api_base: str = GRAPH_API_BASE
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 8080
    webhook_path: str = "/webhook"
    log_format: str = "text"
    # Quiet by default: a CLI that prints its own trace between every reply is a CLI that gets
    # `CALORAI_LOG_LEVEL=WARNING` in the README. The webhook raises this to INFO in its own main().
    log_level: str = "WARNING"
    tracing: bool = False
    tracing_project: str = "calorai"

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
            whatsapp_verify_token=_optional_env("CALORAI_WHATSAPP_VERIFY_TOKEN"),
            whatsapp_app_secret=_optional_env("CALORAI_WHATSAPP_APP_SECRET"),
            whatsapp_access_token=_optional_env("CALORAI_WHATSAPP_ACCESS_TOKEN"),
            whatsapp_phone_number_id=_optional_env("CALORAI_WHATSAPP_PHONE_NUMBER_ID"),
            whatsapp_allowed_users=_env_list("CALORAI_WHATSAPP_ALLOWED_USERS"),
            graph_api_base=_env("CALORAI_GRAPH_API_BASE", default=GRAPH_API_BASE),
            webhook_host=_env("CALORAI_WEBHOOK_HOST", default="127.0.0.1"),
            webhook_port=_int_env("CALORAI_WEBHOOK_PORT", default=8080),
            webhook_path=_env("CALORAI_WEBHOOK_PATH", default="/webhook"),
            request_timeout_seconds=_float_env("CALORAI_REQUEST_TIMEOUT_SECONDS", default=20.0),
            log_format=_env("CALORAI_LOG_FORMAT", default="text"),
            log_level=_env("CALORAI_LOG_LEVEL", default="WARNING"),
            tracing=_bool_env("CALORAI_TRACING"),
            tracing_project=_env("CALORAI_TRACING_PROJECT", default="calorai"),
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

    @property
    def use_whatsapp(self) -> bool:
        """Half a WhatsApp configuration is worse than none.

        A webhook with no app secret accepts events it cannot prove came from Meta, and one with
        no access token accepts them and then cannot answer, so the transport stays off until all
        four pieces are set.
        """
        return all(
            value is not None
            for value in (
                self.whatsapp_verify_token,
                self.whatsapp_app_secret,
                self.whatsapp_access_token,
                self.whatsapp_phone_number_id,
            )
        )
