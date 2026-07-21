"""Deterministic, dependency-free provider.

This backend needs no API key, no network, no GPU and no model download. It exists so
the whole agent — including the multi-step tool-use loop — can run in CI and in the
offline ``visionvoice demo``. It performs a genuine two-step agent cycle: first it emits
a tool call chosen from the user's question, then (once a tool result is present) it
synthesizes a final spoken answer.
"""

from __future__ import annotations

from visionvoice.config import Settings
from visionvoice.providers.base import ModelProvider
from visionvoice.types import Message, ProviderResponse, ToolCall, ToolSpec

# Keyword → tool routing. First match wins.
_ROUTES: list[tuple[tuple[str, ...], str]] = [
    (("read", "sign", "text", "label", "written"), "read_text"),
    (("safe", "cross", "danger", "hazard", "careful"), "assess_safety"),
    (("how many", "count", "number of"), "list_objects"),
    (("front", "around", "see", "there", "describe", "scene", "what"), "describe_scene"),
]


class MockProvider(ModelProvider):
    name = "mock"
    supports_vision = False  # falls back to detection-derived captions

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"mock-call-{self._counter}"

    def _route(self, text: str, tools: list[ToolSpec]) -> str:
        names = {t.name for t in tools}
        low = text.lower()
        for keywords, tool in _ROUTES:
            if tool in names and any(k in low for k in keywords):
                return tool
        # Default to describing the scene if available, else the first tool.
        if "describe_scene" in names:
            return "describe_scene"
        return tools[0].name

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ProviderResponse:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        if not tools:
            return ProviderResponse(text=self._answer_from_text(last_user))

        # If a tool has already run this turn, synthesize the final answer.
        tool_results = [m.content for m in messages if m.role == "tool"]
        if tool_results:
            joined = " ".join(r for r in tool_results if r)
            return ProviderResponse(text=self._compose(last_user, joined))

        # Otherwise, decide which tool to call.
        tool = self._route(last_user, tools)
        return ProviderResponse(tool_calls=[ToolCall(id=self._next_id(), name=tool, arguments={})])

    def _compose(self, question: str, tool_output: str) -> str:
        if not tool_output.strip():
            return "I could not perceive anything clearly right now."
        return tool_output.strip()

    def _answer_from_text(self, text: str) -> str:
        if not text:
            return "I'm ready. Ask me what's in front of you."
        return f"(mock) I heard: {text}"

    def describe_image(
        self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg"
    ) -> str:
        # No real vision; the scene stage will use detections instead.
        return "A scene is visible in front of the camera."
