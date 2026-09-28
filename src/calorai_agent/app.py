from __future__ import annotations

from calorai_agent.config import Settings
from calorai_agent.db import Database
from calorai_agent.domain import MediaRef
from calorai_agent.graph import MealAgent
from calorai_agent.llm_planner import ModelPlanner
from calorai_agent.observability import configure_logging, configure_tracing
from calorai_agent.planning import MessagePlanner, RuleBasedPlanner
from calorai_agent.policy import AmbiguityPolicy
from calorai_agent.providers import (
    ImagePayload,
    OpenAICompatibleClient,
    OpenAICompatibleVisionClient,
)
from calorai_agent.repository import MealRepository
from calorai_agent.tools import MealTools
from calorai_agent.vision import (
    LocalFileMediaSource,
    MediaError,
    MediaSource,
    VisionInterpreter,
)
from calorai_agent.whatsapp import GraphClient, MediaCache, WhatsAppMediaSource
from calorai_agent.whatsapp_server import WebhookApplication


def prepare_runtime(settings: Settings) -> Settings:
    """Install logging and optional tracing once, at the process entrypoint.

    Returns the settings so a caller can keep building from one object; a library that configured
    logging at import time would own the test runner's output along with everything else.
    """
    configure_logging(fmt=settings.log_format, level=settings.log_level)
    configure_tracing(enabled=settings.tracing, project=settings.tracing_project)
    return settings


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


class EitherMediaSource:
    """One vision pipeline that serves both transports: a CLI path or a WhatsApp media id.

    The graph only ever hands over a `MediaRef`, so choosing where its bytes come from belongs
    here rather than in the agent.
    """

    def __init__(self, local: MediaSource, whatsapp: MediaSource | None = None) -> None:
        self.local = local
        self.whatsapp = whatsapp

    def fetch(self, media: MediaRef) -> ImagePayload:
        if media.source == "local_path":
            return self.local.fetch(media)
        if self.whatsapp is None:
            raise MediaError("WhatsApp media needs the Graph access token to be configured.")
        return self.whatsapp.fetch(media)

    def prefetch(self, media: MediaRef) -> None:
        """Warm a photo the delivery will ask for. A local file is already as warm as it gets."""
        source = self.whatsapp if media.source != "local_path" else self.local
        prefetch = getattr(source, "prefetch", None)
        if callable(prefetch):
            prefetch(media)


def build_graph_client(settings: Settings) -> GraphClient | None:
    """The Meta sender is configured only once a token and a number id both exist."""
    if settings.whatsapp_access_token is None or settings.whatsapp_phone_number_id is None:
        return None
    return GraphClient(
        access_token=settings.whatsapp_access_token,
        phone_number_id=settings.whatsapp_phone_number_id,
        base_url=settings.graph_api_base,
        timeout_seconds=settings.request_timeout_seconds,
    )


def build_media_source(settings: Settings) -> EitherMediaSource:
    graph = build_graph_client(settings)
    return EitherMediaSource(
        LocalFileMediaSource(),
        WhatsAppMediaSource(graph, cache=MediaCache()) if graph is not None else None,
    )


def build_vision(settings: Settings, media: MediaSource | None = None) -> VisionInterpreter | None:
    """A photo needs its own model. Without one the CLI says so instead of guessing a plate."""
    if not settings.use_vision or settings.vision_model_api_key is None:
        return None
    client = OpenAICompatibleVisionClient(
        api_key=settings.vision_model_api_key,
        model=settings.vision_model,
        base_url=settings.vision_model_base_url,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return VisionInterpreter(
        client,
        media if media is not None else build_media_source(settings),
        model=settings.vision_model,
    )


def build_repository(settings: Settings) -> MealRepository:
    database = Database(settings.database_path)
    database.initialize()
    repository = MealRepository(database)
    repository.ensure_user(settings.default_user_id, settings.default_timezone)
    return repository


def create_agent(
    settings: Settings,
    *,
    planner: MessagePlanner | None = None,
    policy: AmbiguityPolicy | None = None,
    vision: VisionInterpreter | None = None,
) -> MealAgent:
    return MealAgent(
        planner or build_planner(settings),
        MealTools(build_repository(settings)),
        policy=policy or AmbiguityPolicy(),
        vision=vision if vision is not None else build_vision(settings),
    )


def create_whatsapp_app(settings: Settings) -> WebhookApplication:
    """Wire the transport: the same agent the CLI runs, reached through signed webhooks."""
    graph = build_graph_client(settings)
    if graph is None:
        raise RuntimeError(
            "WhatsApp needs CALORAI_WHATSAPP_ACCESS_TOKEN and CALORAI_WHATSAPP_PHONE_NUMBER_ID."
        )
    if settings.whatsapp_verify_token is None or settings.whatsapp_app_secret is None:
        raise RuntimeError("WhatsApp needs CALORAI_WHATSAPP_VERIFY_TOKEN and _APP_SECRET.")
    repository = build_repository(settings)
    # Photos are only worth warming when there is a model to read them. One media source serves
    # both halves: the delivery prefetches into it, the vision pipeline reads back out of it.
    media = build_media_source(settings) if settings.use_vision else None
    return WebhookApplication(
        MealAgent(
            build_planner(settings),
            MealTools(repository),
            vision=build_vision(settings, media),
        ),
        graph,
        verify_token=settings.whatsapp_verify_token,
        app_secret=settings.whatsapp_app_secret,
        repository=repository,
        timezone=settings.default_timezone,
        allowed_users=frozenset(settings.whatsapp_allowed_users),
        media=media,
    )
