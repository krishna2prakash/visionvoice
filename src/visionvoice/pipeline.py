"""The multi-stage pipeline: capture → detect → reason → speak.

This orchestrates the whole assistant and is the single object the CLI and web server
talk to. Heavy resources (camera, YOLOv8 model) are created lazily so a snapshot supplied
from a still image or a test never spins up a camera.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from visionvoice.agent.assistant import Assistant
from visionvoice.agent.tools import PerceptionContext
from visionvoice.config import Settings, get_settings
from visionvoice.providers import ModelProvider, build_provider
from visionvoice.speech.tts import TTS, build_tts
from visionvoice.types import PerceptionSnapshot
from visionvoice.vision.scene import describe_scene


@dataclass
class LatencyMeter:
    """Rolling per-stage latency, so we can prove the 'real-time' claim."""

    window: int = 30
    _stages: dict[str, deque] = field(default_factory=dict)

    def record(self, stage: str, seconds: float) -> None:
        self._stages.setdefault(stage, deque(maxlen=self.window)).append(seconds * 1000.0)

    def averages_ms(self) -> dict[str, float]:
        return {s: sum(v) / len(v) for s, v in self._stages.items() if v}

    def summary(self) -> str:
        avgs = self.averages_ms()
        if not avgs:
            return "no timings yet"
        total = sum(avgs.values())
        parts = ", ".join(f"{s} {ms:.0f}ms" for s, ms in avgs.items())
        return f"{parts}  |  total {total:.0f}ms"


class Pipeline:
    """End-to-end VisionVoice assistant."""

    def __init__(self, settings: Settings | None = None, provider: ModelProvider | None = None):
        self.settings = settings or get_settings()
        self.provider = provider or build_provider(self.settings)
        self.assistant = Assistant(self.provider)
        self.tts: TTS = build_tts(self.settings)
        self.latency = LatencyMeter()
        self._detector = None  # lazy
        self._camera = None  # lazy

    # -- resources ----------------------------------------------------------
    @property
    def detector(self):
        if self._detector is None:
            from visionvoice.detection import Detector

            self._detector = Detector(self.settings.yolo_model, self.settings.yolo_conf)
        return self._detector

    @property
    def camera(self):
        if self._camera is None:
            from visionvoice.capture import ThreadedCamera

            self._camera = ThreadedCamera(self.settings.camera_index)
        return self._camera

    # -- perception ---------------------------------------------------------
    def perceive(self, frame) -> PerceptionSnapshot:
        """Run detection + encoding on a single frame and build a snapshot."""
        from visionvoice.capture import encode_jpeg

        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        self.latency.record("detect", time.perf_counter() - t0)

        height, width = frame.shape[:2]
        image_bytes = encode_jpeg(frame) if self.provider.supports_vision else None
        return PerceptionSnapshot(
            detections=detections,
            image_bytes=image_bytes,
            width=width,
            height=height,
        )

    def perceive_live(self):
        """Grab a frame from the camera and perceive it. Returns (snapshot, frame)."""
        t0 = time.perf_counter()
        frame = self.camera.read()
        self.latency.record("capture", time.perf_counter() - t0)
        if frame is None:
            raise RuntimeError("No frame available from the camera yet.")
        return self.perceive(frame), frame

    # -- high-level actions -------------------------------------------------
    def answer(
        self,
        query: str,
        snapshot: PerceptionSnapshot | None = None,
        frame: Any | None = None,
        speak: bool = True,
    ) -> str:
        """Answer a spoken/typed question about the current scene."""
        if snapshot is None:
            snapshot, frame = self.perceive_live()
        ctx = PerceptionContext(provider=self.provider, snapshot=snapshot, frame=frame)

        t0 = time.perf_counter()
        reply = self.assistant.ask(query, ctx)
        self.latency.record("reason", time.perf_counter() - t0)

        if speak:
            self.speak(reply)
        return reply

    def describe(
        self,
        snapshot: PerceptionSnapshot | None = None,
        frame: Any | None = None,
        speak: bool = True,
    ) -> str:
        """One-shot scene description (no question)."""
        if snapshot is None:
            snapshot, frame = self.perceive_live()
        caption = describe_scene(self.provider, snapshot)
        if speak:
            self.speak(caption)
        return caption

    # -- output -------------------------------------------------------------
    def speak(self, text: str) -> None:
        """Speak in every configured language (translating for non-primary ones)."""
        t0 = time.perf_counter()
        primary = self.settings.primary_language
        for lang in self.settings.language_list:
            spoken = text if lang == primary else self._translate(text, lang)
            self.tts.speak(spoken, lang)
        self.latency.record("speak", time.perf_counter() - t0)

    def _translate(self, text: str, lang: str) -> str:
        """Best-effort translation via the active provider (falls back to source text)."""
        from visionvoice.speech.tts import LANGUAGE_NAMES
        from visionvoice.types import Message

        target = LANGUAGE_NAMES.get(lang, lang)
        try:
            resp = self.provider.chat(
                [Message(role="user", content=f"Translate to {target}:\n{text}")],
                system=f"You are a translator. Output only the {target} translation, nothing else.",
            )
            return resp.text or text
        except Exception:
            return text

    def close(self) -> None:
        if self._camera is not None:
            self._camera.release()
        self.tts.close()
