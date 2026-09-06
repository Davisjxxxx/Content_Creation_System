from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ProviderResponse:
    content: str
    provider: str
    model: str
    input_units: float = 0.0
    output_units: float = 0.0
    cost_usd: float = 0.0
    metadata: dict | None = None


class LLMProvider(Protocol):
    def complete(self, *, system: str, user: str, model: str | None = None) -> ProviderResponse: ...


class SearchProvider(Protocol):
    def search(self, *, query: str, limit: int = 10) -> list[dict]: ...


class VoiceProvider(Protocol):
    def synthesize(self, *, text: str, voice_id: str) -> ProviderResponse: ...


class ImageProvider(Protocol):
    def generate(self, *, prompt: str, aspect_ratio: str) -> ProviderResponse: ...


class VideoProvider(Protocol):
    def generate(self, *, prompt: str, seconds: float, aspect_ratio: str) -> ProviderResponse: ...


class PresenterProvider(Protocol):
    def render(self, *, script: str, presenter_id: str, aspect_ratio: str) -> ProviderResponse: ...


class Publisher(Protocol):
    def prepare(self, *, platform: str, asset_path: str, metadata: dict) -> dict: ...
    def publish(self, *, prepared: dict, approval_token: str) -> dict: ...
