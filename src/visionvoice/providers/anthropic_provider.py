"""Anthropic (Claude) cloud backend.

Uses the Messages API for both agent reasoning (tool use) and vision-language scene
description (base64 image blocks). One :meth:`chat` call is a single agent turn — the
provider-agnostic agent loop in :mod:`visionvoice.agent.assistant` drives the tool cycle.

Thinking is disabled to minimise latency for the real-time pipeline; flip it to adaptive
if you want deeper reasoning at the cost of response time.
"""

from __future__ import annotations

import base64

from visionvoice.config import Settings
from visionvoice.providers.base import ModelProvider
from visionvoice.types import Message, ProviderResponse, ToolCall, ToolSpec


class AnthropicProvider(ModelProvider):
    name = "anthropic"
    supports_vision = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The 'anthropic' package is required for VV_PROVIDER=anthropic. "
                'Install it with:  pip install -e ".[anthropic]"'
            ) from exc

        if not settings.anthropic_api_key:
            raise RuntimeError(
                "VV_ANTHROPIC_API_KEY is not set. Add it to your .env or environment."
            )
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # -- agent turn ---------------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ProviderResponse:
        api_messages = self._to_api_messages(messages)
        kwargs: dict = {
            "model": self.settings.anthropic_model,
            "max_tokens": 1024,
            "thinking": {"type": "disabled"},
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
                )
        return ProviderResponse(text="".join(text_parts).strip(), tool_calls=tool_calls)

    # -- vision -------------------------------------------------------------
    def describe_image(
        self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg"
    ) -> str:
        data = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = self._client.messages.create(
            model=self.settings.anthropic_vision_model,
            max_tokens=512,
            thinking={"type": "disabled"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": data},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def health(self) -> tuple[bool, str]:
        if not self.settings.anthropic_api_key:
            return False, "VV_ANTHROPIC_API_KEY not set"
        return True, f"anthropic ready (model={self.settings.anthropic_model})"

    # -- helpers ------------------------------------------------------------
    def _to_api_messages(self, messages: list[Message]) -> list[dict]:
        """Convert internal messages to the Anthropic wire format.

        Consecutive tool results are merged into a single user message, as recommended.
        """

        api: list[dict] = []
        pending_tool_results: list[dict] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                api.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for msg in messages:
            if msg.role == "tool":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }
                )
                continue

            flush_tool_results()

            if msg.role == "assistant" and msg.tool_calls:
                content: list[dict] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for call in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                api.append({"role": "assistant", "content": content})
            elif msg.role in ("user", "assistant"):
                api.append({"role": msg.role, "content": msg.content})
            # 'system' messages are handled via the top-level system param.

        flush_tool_results()
        return api
