"""Fully-offline, on-device reasoner — the Raspberry Pi 5 backend.

No API, no network, no LLM weights: a compact rule-based natural-language layer that
routes a question to the right perception tool and composes a spoken answer from the
result. This is what makes VisionVoice run **offline at near-zero cost** on edge hardware,
while the exact same agent loop and tools also run against Claude or Ollama when available.

It performs the same genuine two-step tool-use cycle as any other provider: pick a tool,
then turn the observation into a concise, natural reply.
"""

from __future__ import annotations

from visionvoice.config import Settings
from visionvoice.providers.base import ModelProvider
from visionvoice.types import Message, ProviderResponse, ToolCall, ToolSpec

# Keyword → tool routing. First match wins.
_ROUTES: list[tuple[tuple[str, ...], str]] = [
    (("read", "sign", "text", "label", "written", "say"), "read_text"),
    (("safe", "cross", "danger", "hazard", "careful", "clear"), "assess_safety"),
    (("how many", "count", "number of"), "list_objects"),
    (("front", "around", "see", "there", "describe", "scene", "what", "where"), "describe_scene"),
]


class OfflineProvider(ModelProvider):
    name = "offline"
    supports_vision = False  # on-device: describe from detections, no heavy VLM

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"offline-call-{self._counter}"

    def _route(self, text: str, tools: list[ToolSpec]) -> str:
        names = {t.name for t in tools}
        low = text.lower()
        for keywords, tool in _ROUTES:
            if tool in names and any(k in low for k in keywords):
                return tool
        return "describe_scene" if "describe_scene" in names else tools[0].name

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ProviderResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        if not tools:
            # Used for the (optional) translation step; offline we just echo the source.
            return ProviderResponse(text=last_user)

        tool_results = [m.content for m in messages if m.role == "tool" and m.content]
        if tool_results:
            return ProviderResponse(text=self._compose(tool_results))

        tool = self._route(last_user, tools)
        return ProviderResponse(tool_calls=[ToolCall(id=self._next_id(), name=tool, arguments={})])

    def _compose(self, tool_results: list[str]) -> str:
        """Turn one or more tool observations into a single concise spoken reply."""
        # Deduplicate while preserving order.
        seen: list[str] = []
        for result in tool_results:
            cleaned = result.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        if not seen:
            return "I could not perceive anything clearly right now."
        return " ".join(seen)

    def describe_image(
        self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg"
    ) -> str:
        return "A scene is visible in front of the camera."

    def health(self) -> tuple[bool, str]:
        return True, "offline on-device reasoner (no API, no network)"
