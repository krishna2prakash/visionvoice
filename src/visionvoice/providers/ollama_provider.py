"""Ollama local backend.

Runs entirely on the user's machine (no API key, no cloud) via a local Ollama server.
Reasoning + tool use go to a text model (e.g. ``llama3.1``); vision captions go to a
local vision model (e.g. ``llava``). Ideal for offline / Raspberry Pi / privacy setups.
"""

from __future__ import annotations

import base64

from visionvoice.config import Settings
from visionvoice.providers.base import ModelProvider
from visionvoice.types import Message, ProviderResponse, ToolCall, ToolSpec


class OllamaProvider(ModelProvider):
    name = "ollama"
    supports_vision = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The 'ollama' package is required for VV_PROVIDER=ollama. "
                'Install it with:  pip install -e ".[ollama]"  (and run the Ollama server)'
            ) from exc

        self._ollama = ollama
        self._client = ollama.Client(host=settings.ollama_host)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"ollama-call-{self._counter}"

    def chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ProviderResponse:
        api_messages: list[dict] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "tool":
                api_messages.append({"role": "tool", "content": msg.content})
            elif msg.role == "assistant" and msg.tool_calls:
                api_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {"function": {"name": c.name, "arguments": c.arguments}}
                            for c in msg.tool_calls
                        ],
                    }
                )
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        kwargs: dict = {"model": self.settings.ollama_text_model, "messages": api_messages}
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        response = self._client.chat(**kwargs)
        message = response["message"]
        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(id=self._next_id(), name=fn.get("name", ""), arguments=args))
        return ProviderResponse(text=(message.get("content") or "").strip(), tool_calls=tool_calls)

    def describe_image(
        self, image_bytes: bytes, prompt: str, media_type: str = "image/jpeg"
    ) -> str:
        # Ollama accepts base64-encoded image strings in the message.
        data = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = self._client.chat(
            model=self.settings.ollama_vision_model,
            messages=[{"role": "user", "content": prompt, "images": [data]}],
        )
        return (response["message"].get("content") or "").strip()

    def health(self) -> tuple[bool, str]:
        try:
            self._client.list()
        except Exception as exc:  # pragma: no cover - network dependent
            return False, f"cannot reach Ollama at {self.settings.ollama_host}: {exc}"
        return True, f"ollama ready (host={self.settings.ollama_host})"
