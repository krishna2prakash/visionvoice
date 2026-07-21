"""Central configuration, loaded from environment / .env (prefix ``VV_``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "ollama", "offline", "mock"]


class Settings(BaseSettings):
    """Resolved runtime settings.

    Every field can be set via an environment variable, e.g. ``VV_PROVIDER=anthropic``.
    Defaults are chosen so a fresh clone runs the offline demo and the test suite with
    no API keys, no GPU and no camera.
    """

    model_config = SettingsConfigDict(
        env_prefix="VV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Backend selection ---
    provider: Provider = "mock"

    # --- Anthropic (cloud) ---
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    anthropic_vision_model: str = "claude-sonnet-5"

    # --- Ollama (local) ---
    ollama_host: str = "http://localhost:11434"
    ollama_text_model: str = "llama3.1"
    ollama_vision_model: str = "llava"

    # --- Perception ---
    camera_index: int = 0
    yolo_model: str = "yolo26n.pt"
    yolo_conf: float = Field(default=0.35, ge=0.0, le=1.0)

    # --- Speech ---
    tts_engine: Literal["gtts", "pyttsx3", "print"] = "print"
    stt_engine: Literal["whisper", "google", "text"] = "text"
    languages: str = "en,ta,ml"

    @property
    def language_list(self) -> list[str]:
        return [c.strip() for c in self.languages.split(",") if c.strip()]

    @property
    def primary_language(self) -> str:
        langs = self.language_list
        return langs[0] if langs else "en"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (reads env / .env once)."""
    return Settings()
