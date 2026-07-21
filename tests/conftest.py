"""Shared test fixtures. Everything runs on the deterministic mock backend."""

from __future__ import annotations

import pytest

from visionvoice.config import Settings
from visionvoice.types import Detection, PerceptionSnapshot


@pytest.fixture
def settings() -> Settings:
    # Explicit mock config — no .env, no keys, no camera.
    return Settings(provider="mock", tts_engine="print", stt_engine="text", languages="en")


@pytest.fixture
def snapshot() -> PerceptionSnapshot:
    return PerceptionSnapshot(
        detections=[
            Detection("person", 0.95, (0.05, 0.3, 0.28, 0.95), position="left", distance="close"),
            Detection("car", 0.82, (0.7, 0.4, 0.99, 0.85), position="far-right", distance="very close"),
            Detection("chair", 0.80, (0.42, 0.55, 0.63, 0.98), position="center", distance="mid"),
        ],
        width=1280,
        height=720,
    )
