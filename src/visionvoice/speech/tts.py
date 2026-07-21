"""Multilingual text-to-speech.

Three engines, chosen by ``VV_TTS_ENGINE``:

* ``print``   — logs the spoken text (default; works headless, in CI, and in tests).
* ``pyttsx3`` — fully offline OS speech synthesis.
* ``gtts``    — Google TTS; supports Tamil (``ta``) and Malayalam (``ml``) and saves MP3s.

Every engine implements :meth:`TTS.speak`. Playback is best-effort — if audio can't be
played (headless box, missing codec), the text is still logged so nothing is lost.
"""

from __future__ import annotations

import logging
from pathlib import Path

from visionvoice.config import Settings

logger = logging.getLogger("visionvoice.tts")

# Human-readable names for the language codes we support.
LANGUAGE_NAMES = {"en": "English", "ta": "Tamil", "ml": "Malayalam"}


class TTS:
    """Base text-to-speech engine."""

    def speak(self, text: str, lang: str = "en") -> None:
        raise NotImplementedError

    def close(self) -> None:  # optional cleanup hook
        pass


class PrintTTS(TTS):
    """Logs spoken output. The safe default for headless / CI / tests."""

    def speak(self, text: str, lang: str = "en") -> None:
        name = LANGUAGE_NAMES.get(lang, lang)
        logger.info("[TTS:%s] %s", name, text)
        print(f"🔊 ({name}) {text}")


class Pyttsx3TTS(TTS):
    """Offline OS speech synthesis via pyttsx3 (no network)."""

    def __init__(self) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()

    def speak(self, text: str, lang: str = "en") -> None:
        print(f"🔊 ({LANGUAGE_NAMES.get(lang, lang)}) {text}")
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:  # pragma: no cover - platform dependent
            logger.warning("pyttsx3 playback failed: %s", exc)


class GttsTTS(TTS):
    """Google TTS — supports Tamil/Malayalam; writes MP3s to ``captures/``."""

    def __init__(self, out_dir: str = "captures") -> None:
        from gtts import gTTS  # noqa: F401  (validate the import early)

        self._out = Path(out_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def speak(self, text: str, lang: str = "en") -> None:
        from gtts import gTTS

        print(f"🔊 ({LANGUAGE_NAMES.get(lang, lang)}) {text}")
        self._counter += 1
        path = self._out / f"tts_{lang}_{self._counter:04d}.mp3"
        try:
            gTTS(text=text, lang=lang).save(str(path))
            _try_play(path)
        except Exception as exc:  # pragma: no cover - network/codec dependent
            logger.warning("gTTS synthesis/playback failed (%s); text was: %s", exc, text)


def _try_play(path: Path) -> None:
    """Best-effort audio playback across platforms."""
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Windows":  # pragma: no cover - platform dependent
            import os

            os.startfile(str(path))  # type: ignore[attr-defined]
        elif system == "Darwin":  # pragma: no cover
            subprocess.run(["afplay", str(path)], check=False)
        else:  # pragma: no cover
            subprocess.run(["mpg123", "-q", str(path)], check=False)
    except Exception as exc:  # pragma: no cover
        logger.debug("Could not auto-play %s: %s", path, exc)


def build_tts(settings: Settings) -> TTS:
    """Instantiate the TTS engine named by ``settings.tts_engine`` (with fallback)."""
    engine = settings.tts_engine
    try:
        if engine == "pyttsx3":
            return Pyttsx3TTS()
        if engine == "gtts":
            return GttsTTS()
    except Exception as exc:
        logger.warning("TTS engine '%s' unavailable (%s); falling back to print.", engine, exc)
    return PrintTTS()
