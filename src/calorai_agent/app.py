from __future__ import annotations

from calorai_agent.config import Settings
from calorai_agent.db import Database
from calorai_agent.graph import MealAgent
from calorai_agent.nutrition import RuleBasedPlanner
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools


def create_agent(settings: Settings) -> MealAgent:
    database = Database(settings.database_path)
    database.initialize()
    repository = MealRepository(database)
    repository.ensure_user(settings.default_user_id, settings.default_timezone)
    return MealAgent(RuleBasedPlanner(), MealTools(repository))
