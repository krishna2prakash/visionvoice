"""Tests for the model-provider abstraction (mock backend)."""

from __future__ import annotations

from visionvoice.providers import build_provider
from visionvoice.providers.mock_provider import MockProvider
from visionvoice.types import Message, ToolSpec

TOOLS = [
    ToolSpec("describe_scene", "describe the scene"),
    ToolSpec("read_text", "read text"),
    ToolSpec("assess_safety", "check safety"),
    ToolSpec("list_objects", "list objects"),
]


def test_factory_builds_mock(settings):
    provider = build_provider(settings)
    assert isinstance(provider, MockProvider)
    assert provider.name == "mock"


def test_chat_without_tools_returns_text(settings):
    provider = build_provider(settings)
    resp = provider.chat([Message(role="user", content="hello there")])
    assert resp.text
    assert not resp.wants_tools


def test_chat_routes_to_relevant_tool(settings):
    provider = build_provider(settings)
    resp = provider.chat([Message(role="user", content="please read this sign")], tools=TOOLS)
    assert resp.wants_tools
    assert resp.tool_calls[0].name == "read_text"


def test_chat_finalizes_after_tool_result(settings):
    provider = build_provider(settings)
    messages = [
        Message(role="user", content="what's around me?"),
        Message(role="assistant", content="", tool_calls=[]),
        Message(role="tool", content="a chair to your center", tool_call_id="x"),
    ]
    resp = provider.chat(messages, tools=TOOLS)
    assert not resp.wants_tools
    assert "chair" in resp.text.lower()


def test_describe_image_returns_text(settings):
    provider = build_provider(settings)
    assert provider.describe_image(b"fake", "describe").strip()
