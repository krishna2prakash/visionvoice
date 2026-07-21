"""Abstract base class every model backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from visionvoice.config import Settings
from visionvoice.types import Message, ProviderResponse, ToolSpec


class ModelProvider(ABC):
    """A pluggable reasoning + vision backend.

    Implementations must provide:

    * :meth:`chat` — a single agent turn, optionally emitting tool calls.
    * :meth:`describe_image` — a vision-language caption for an image.

    Providers that cannot see images should set :attr:`supports_vision` to ``False``;
    the pipeline then falls back to a detection-derived description.
    """

    #: Human-readable backend name (used in ``visionvoice info``).
    name: str = "base"
    #: Whether :meth:`describe_image` uses a real VLM.
    supports_vision: bool = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ProviderResponse:
        """Run one turn of the conversation.

        Returns either final text or a set of tool calls to execute.
        """

    @abstractmethod
    def describe_image(
        self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg"
    ) -> str:
        """Return a natural-language description of an image."""

    def health(self) -> tuple[bool, str]:
        """Cheap readiness probe. Returns ``(ok, detail)``."""
        return True, "ok"
