"""Tests for the pipeline, scene summarization, config, and detection helpers."""

from __future__ import annotations

from visionvoice.config import Settings
from visionvoice.detection import area_to_distance, bbox_to_position
from visionvoice.pipeline import LatencyMeter, Pipeline
from visionvoice.types import Detection
from visionvoice.vision.scene import summarize_detections


def test_pipeline_answers_from_snapshot(settings, snapshot):
    pipe = Pipeline(settings)
    reply = pipe.answer("what's in front of me?", snapshot=snapshot, speak=False)
    assert reply.strip()
    pipe.close()


def test_pipeline_describe_from_snapshot(settings, snapshot):
    pipe = Pipeline(settings)
    caption = pipe.describe(snapshot=snapshot, speak=False)
    assert "person" in caption or "front of you" in caption
    pipe.close()


def test_speak_records_latency(settings, snapshot):
    pipe = Pipeline(settings)
    pipe.answer("what's around me?", snapshot=snapshot, speak=True)
    assert "reason" in pipe.latency.averages_ms()
    pipe.close()


def test_summarize_empty():
    assert "don't see anything" in summarize_detections([])


def test_summarize_groups_labels():
    dets = [
        Detection("person", 0.9, (0.0, 0.0, 0.2, 0.9), position="left", distance="close"),
        Detection("person", 0.8, (0.3, 0.0, 0.5, 0.9), position="center", distance="mid"),
    ]
    out = summarize_detections(dets)
    assert "2 persons" in out


def test_detection_helpers():
    assert bbox_to_position(0.1) == "far-left"
    assert bbox_to_position(0.5) == "center"
    assert bbox_to_position(0.9) == "far-right"
    assert area_to_distance(0.5) == "very close"
    assert area_to_distance(0.001) == "far"


def test_config_language_list():
    s = Settings(languages="en, ta ,ml")
    assert s.language_list == ["en", "ta", "ml"]
    assert s.primary_language == "en"


def test_latency_meter():
    m = LatencyMeter()
    m.record("detect", 0.05)
    m.record("detect", 0.03)
    assert 30 < m.averages_ms()["detect"] < 60
