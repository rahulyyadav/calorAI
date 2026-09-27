from __future__ import annotations

from calorai_agent.config import Settings
from calorai_agent.db import Database
from calorai_agent.graph import MealAgent
from calorai_agent.llm_planner import ModelPlanner
from calorai_agent.planning import MessagePlanner, RuleBasedPlanner
from calorai_agent.policy import AmbiguityPolicy
from calorai_agent.providers import OpenAICompatibleClient
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools


def build_planner(settings: Settings) -> MessagePlanner:
    """Model planner when configured, deterministic planner otherwise."""
    if not settings.use_model_planner:
        return RuleBasedPlanner()
    if settings.text_model_api_key is None:
        raise RuntimeError(
            "CALORAI_PLANNER=model needs CALORAI_TEXT_MODEL_API_KEY (or OPENAI_API_KEY)."
        )
    client = OpenAICompatibleClient(
        api_key=settings.text_model_api_key,
        model=settings.text_model,
        base_url=settings.text_model_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return ModelPlanner(client, model=settings.text_model)


def create_agent(
    settings: Settings,
    *,
    planner: MessagePlanner | None = None,
    policy: AmbiguityPolicy | None = None,
) -> MealAgent:
    database = Database(settings.database_path)
    database.initialize()
    repository = MealRepository(database)
    repository.ensure_user(settings.default_user_id, settings.default_timezone)
    return MealAgent(
        planner or build_planner(settings),
        MealTools(repository),
        policy=policy or AmbiguityPolicy(),
    )
