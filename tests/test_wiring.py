import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from calorai_agent.app import build_planner, create_agent
from calorai_agent.cli import main as cli_main
from calorai_agent.config import Settings
from calorai_agent.graph import MealAgent
from calorai_agent.llm_planner import ModelPlanner
from calorai_agent.planning import RuleBasedPlanner
from calorai_agent.providers import ModelProviderError, OpenAICompatibleClient

RESPONSE: dict[str, Any] = {
    "choices": [{"message": {"content": json.dumps({"intent": "get_totals"})}}]
}


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base = {
        "database_path": tmp_path / "calorai.sqlite3",
        "default_user_id": "cli-user",
        "default_timezone": "UTC",
    }
    return Settings(**{**base, **overrides})


def test_client_posts_a_bearer_authenticated_json_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RESPONSE)

    client = OpenAICompatibleClient(
        api_key="secret-key",
        model="some-model",
        transport=httpx.MockTransport(handler),
    )

    assert json.loads(client.complete(system="s", user="u"))["intent"] == "get_totals"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["body"]["model"] == "some-model"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_client_converts_http_failures_into_provider_errors() -> None:
    client = OpenAICompatibleClient(
        api_key="k",
        model="m",
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom")),
    )

    with pytest.raises(ModelProviderError):
        client.complete(system="s", user="u")


def test_client_rejects_an_unexpected_envelope() -> None:
    client = OpenAICompatibleClient(
        api_key="k",
        model="m",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"oops": True})),
    )

    with pytest.raises(ModelProviderError):
        client.complete(system="s", user="u")


def test_no_api_key_keeps_the_deterministic_planner(tmp_path: Path) -> None:
    assert isinstance(build_planner(_settings(tmp_path)), RuleBasedPlanner)


def test_api_key_switches_on_the_model_planner(tmp_path: Path) -> None:
    settings = _settings(tmp_path, text_model_api_key="secret")

    planner = build_planner(settings)

    assert isinstance(planner, ModelPlanner)
    assert planner.model == settings.text_model


def test_requiring_the_model_planner_without_a_key_is_a_configuration_error(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, planner="model")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_planner(settings)


def test_create_agent_answers_the_phase_one_questions(tmp_path: Path) -> None:
    agent = create_agent(_settings(tmp_path))

    assert isinstance(agent, MealAgent)
    assert agent.invoke("cli-user", "2 rotis for breakfast").startswith("Logged 2 roti")
    assert agent.invoke("cli-user", "how many calories today?").startswith("Today: 240 kcal")


def test_environment_driven_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALORAI_DB_PATH", str(tmp_path / "from-env.sqlite3"))
    monkeypatch.setenv("CALORAI_USER_ID", "env-user")
    monkeypatch.setenv("CALORAI_TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("CALORAI_PLANNER", "deterministic")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    settings = Settings.from_env()

    assert settings.database_path == tmp_path / "from-env.sqlite3"
    assert settings.default_user_id == "env-user"
    assert settings.default_timezone == "Asia/Kolkata"
    assert settings.use_model_planner is False


def test_settings_prefer_the_dedicated_model_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CALORAI_TEXT_MODEL_API_KEY", "primary")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback")

    assert Settings.from_env().text_model_api_key == "primary"


def test_the_packaged_cli_logs_corrects_and_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented clean-clone entry point has to keep working phase after phase."""
    monkeypatch.setenv("CALORAI_DB_PATH", str(tmp_path / "cli.sqlite3"))
    monkeypatch.setenv("CALORAI_USER_ID", "cli-user")
    monkeypatch.setenv("CALORAI_PLANNER", "deterministic")
    monkeypatch.delenv("CALORAI_TEXT_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    for message in (
        "2 rotis for breakfast",
        "actually that was 3 rotis",
        "no eggs",
        "how many calories today?",
    ):
        monkeypatch.setattr("sys.argv", ["calorai", message])
        cli_main()

    printed = capsys.readouterr().out
    assert "Logged 2 roti" in printed
    assert "Updated" in printed and "3 roti" in printed
    assert "Noted" in printed
    assert "360 kcal" in printed
