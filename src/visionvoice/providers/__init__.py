"""Swappable model-provider layer.

The same agent runs on any provider that implements :class:`ModelProvider`.
Select one with ``VV_PROVIDER`` (anthropic | ollama | mock).
"""

from __future__ import annotations

from visionvoice.config import Settings
from visionvoice.providers.base import ModelProvider


def build_provider(settings: Settings) -> ModelProvider:
    """Factory: instantiate the provider named by ``settings.provider``.

    Heavy/optional SDKs are imported lazily so selecting ``mock`` never requires
    the ``anthropic`` or ``ollama`` packages to be installed.
    """

    name = settings.provider
    if name == "mock":
        from visionvoice.providers.mock_provider import MockProvider

        return MockProvider(settings)
    if name == "anthropic":
        from visionvoice.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if name == "ollama":
        from visionvoice.providers.ollama_provider import OllamaProvider

        return OllamaProvider(settings)
    raise ValueError(f"Unknown provider: {name!r}")


__all__ = ["ModelProvider", "build_provider"]
