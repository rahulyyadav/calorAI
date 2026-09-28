from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from calorai_agent.observability import span

logger = logging.getLogger(__name__)


class ModelProviderError(RuntimeError):
    """Raised when a provider call fails or returns something unusable."""


class TextModelClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


class VisionModelClient(Protocol):
    def observe(self, *, system: str, user: str, image: ImagePayload) -> str: ...


@dataclass(frozen=True, slots=True)
class ImagePayload:
    """One photo's bytes, already validated, plus how to hand them to a model."""

    data: bytes
    mime_type: str

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.data).decode()
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass(frozen=True, slots=True)
class _Endpoint:
    """Shared transport for every OpenAI-compatible chat endpoint, text or vision."""

    api_key: str
    model: str
    base_url: str
    timeout_seconds: float
    temperature: float
    transport: httpx.BaseTransport | None

    def chat_json(self, messages: list[dict[str, Any]], *, label: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        # The span carries no prompt: a photo turn's prompt holds what the user said about their
        # dinner, and a timing log has no reason to carry that to a log aggregator.
        with span(logger, "model_request", kind=label, model=self.model) as report:
            try:
                with httpx.Client(
                    base_url=f"{self.base_url.rstrip('/')}/",
                    timeout=self.timeout_seconds,
                    transport=self.transport,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                ) as client:
                    response = client.post("chat/completions", json=payload)
                response.raise_for_status()
                body: dict[str, Any] = response.json()
            except httpx.HTTPError as error:
                report["failed"] = type(error).__name__
                raise ModelProviderError(f"{label} model request failed: {error}") from error

            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise ModelProviderError(
                    f"{label} model returned an unexpected envelope"
                ) from error
            if not isinstance(content, str):
                raise ModelProviderError(f"{label} model returned non-text content")
            report["response_chars"] = len(content)
            return content


@dataclass(frozen=True, slots=True)
class OpenAICompatibleClient:
    """Minimal chat-completions client so the provider choice stays swappable."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 20.0
    temperature: float = 0.0
    transport: httpx.BaseTransport | None = None

    def complete(self, *, system: str, user: str) -> str:
        return _Endpoint(
            self.api_key,
            self.model,
            self.base_url,
            self.timeout_seconds,
            self.temperature,
            self.transport,
        ).chat_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            label="text",
        )


@dataclass(frozen=True, slots=True)
class OpenAICompatibleVisionClient:
    """Separate from the text client on purpose: a photo never goes to the text model."""

    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 20.0
    temperature: float = 0.0
    transport: httpx.BaseTransport | None = None

    def observe(self, *, system: str, user: str, image: ImagePayload) -> str:
        return _Endpoint(
            self.api_key,
            self.model,
            self.base_url,
            self.timeout_seconds,
            self.temperature,
            self.transport,
        ).chat_json(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": image.data_url}},
                    ],
                },
            ],
            label="vision",
        )


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ModelProviderError(f"model output was not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise ModelProviderError("model output was not a JSON object")
    return loaded
