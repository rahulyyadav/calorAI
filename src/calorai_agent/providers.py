from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class ModelProviderError(RuntimeError):
    """Raised when a provider call fails or returns something unusable."""


class TextModelClient(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...


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
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
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
            raise ModelProviderError(f"text model request failed: {error}") from error

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelProviderError("text model returned an unexpected envelope") from error
        if not isinstance(content, str):
            raise ModelProviderError("text model returned non-text content")
        return content


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ModelProviderError(f"model output was not valid JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise ModelProviderError("model output was not a JSON object")
    return loaded
