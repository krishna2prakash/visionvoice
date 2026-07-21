"""Speech-to-text (voice commands in).

Three engines, chosen by ``VV_STT_ENGINE``:

* ``text``    — read a typed line (default; works everywhere, great for demos/tests).
* ``google``  — microphone → Google Web Speech via SpeechRecognition (online).
* ``whisper`` — microphone → local OpenAI Whisper (offline).

Every engine implements :meth:`STT.listen`, returning the recognized text (or "").
"""

from __future__ import annotations

import logging

from visionvoice.config import Settings

logger = logging.getLogger("visionvoice.stt")


class STT:
    """Base speech-to-text engine."""

    def listen(self, prompt: str = "You: ") -> str:
        raise NotImplementedError


class TextSTT(STT):
    """Keyboard input standing in for a microphone. The safe default."""

    def listen(self, prompt: str = "You: ") -> str:
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return ""


class GoogleSTT(STT):
    """Microphone capture + Google Web Speech recognition."""

    def __init__(self) -> None:
        import speech_recognition as sr

        self._sr = sr
        self._recognizer = sr.Recognizer()

    def listen(self, prompt: str = "You: ") -> str:  # pragma: no cover - needs a mic
        with self._sr.Microphone() as source:
            print(prompt + "(listening…)")
            audio = self._recognizer.listen(source, phrase_time_limit=6)
        try:
            return self._recognizer.recognize_google(audio)
        except Exception as exc:
            logger.warning("Google STT failed: %s", exc)
            return ""


class WhisperSTT(STT):
    """Microphone capture + local Whisper transcription (offline)."""

    def __init__(self, model_name: str = "base") -> None:  # pragma: no cover - heavy
        import speech_recognition as sr
        import whisper

        self._sr = sr
        self._recognizer = sr.Recognizer()
        self._model = whisper.load_model(model_name)

    def listen(self, prompt: str = "You: ") -> str:  # pragma: no cover - needs a mic
        import numpy as np

        with self._sr.Microphone(sample_rate=16000) as source:
            print(prompt + "(listening…)")
            audio = self._recognizer.listen(source, phrase_time_limit=6)
        data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
        result = self._model.transcribe(data, fp16=False)
        return str(result.get("text", "")).strip()


def build_stt(settings: Settings) -> STT:
    """Instantiate the STT engine named by ``settings.stt_engine`` (with fallback)."""
    engine = settings.stt_engine
    try:
        if engine == "google":
            return GoogleSTT()
        if engine == "whisper":
            return WhisperSTT()
    except Exception as exc:
        logger.warning("STT engine '%s' unavailable (%s); falling back to text.", engine, exc)
    return TextSTT()
