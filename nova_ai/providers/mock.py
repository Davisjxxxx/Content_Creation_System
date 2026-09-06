from __future__ import annotations

from .base import ProviderResponse


class MockLLMProvider:
    def complete(self, *, system: str, user: str, model: str | None = None) -> ProviderResponse:
        return ProviderResponse(
            content='{"mock": true}',
            provider="mock",
            model=model or "mock-model",
            input_units=len(system) + len(user),
            output_units=14,
            cost_usd=0.0,
        )
