"""Tests for the agent loop and tools."""

from __future__ import annotations

from visionvoice.agent.assistant import Assistant
from visionvoice.agent.tools import PerceptionContext, execute_tool
from visionvoice.providers import build_provider


def test_agent_answers_and_terminates(settings, snapshot):
    provider = build_provider(settings)
    agent = Assistant(provider, max_iterations=4)
    ctx = PerceptionContext(provider=provider, snapshot=snapshot)
    reply = agent.ask("What's in front of me?", ctx)
    assert isinstance(reply, str) and reply.strip()


def test_list_objects_tool(settings, snapshot):
    provider = build_provider(settings)
    ctx = PerceptionContext(provider=provider, snapshot=snapshot)
    out = execute_tool("list_objects", {}, ctx)
    assert "person" in out and "car" in out


def test_assess_safety_flags_nearby_hazard(settings, snapshot):
    provider = build_provider(settings)
    ctx = PerceptionContext(provider=provider, snapshot=snapshot)
    out = execute_tool("assess_safety", {}, ctx)
    # A car is 'very close' in the fixture → should warn.
    assert "caution" in out.lower() or "hazard" in out.lower()


def test_read_text_without_frame(settings, snapshot):
    provider = build_provider(settings)
    ctx = PerceptionContext(provider=provider, snapshot=snapshot, frame=None)
    out = execute_tool("read_text", {}, ctx)
    assert "no camera frame" in out.lower()


def test_unknown_tool(settings, snapshot):
    provider = build_provider(settings)
    ctx = PerceptionContext(provider=provider, snapshot=snapshot)
    assert "unknown tool" in execute_tool("nope", {}, ctx).lower()
