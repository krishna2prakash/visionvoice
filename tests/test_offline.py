"""Tests for the fully-offline on-device backend (the Raspberry Pi 5 path)."""

from __future__ import annotations

from visionvoice.agent.assistant import Assistant
from visionvoice.agent.tools import PerceptionContext
from visionvoice.config import Settings
from visionvoice.pipeline import Pipeline
from visionvoice.providers import build_provider
from visionvoice.providers.offline_provider import OfflineProvider


def _offline_settings() -> Settings:
    return Settings(provider="offline", tts_engine="print", stt_engine="text", languages="en")


def test_factory_builds_offline():
    provider = build_provider(_offline_settings())
    assert isinstance(provider, OfflineProvider)
    assert provider.name == "offline"
    ok, _ = provider.health()
    assert ok


def test_offline_needs_no_network(snapshot):
    # The offline provider must answer without any API/network access.
    provider = build_provider(_offline_settings())
    agent = Assistant(provider)
    ctx = PerceptionContext(provider=provider, snapshot=snapshot)
    reply = agent.ask("Is it safe to move forward?", ctx)
    assert reply.strip()
    assert "hazard" in reply.lower() or "caution" in reply.lower() or "clear" in reply.lower()


def test_offline_pipeline_end_to_end(snapshot):
    pipe = Pipeline(_offline_settings())
    reply = pipe.answer("what's in front of me?", snapshot=snapshot, speak=False)
    assert reply.strip()
    pipe.close()
