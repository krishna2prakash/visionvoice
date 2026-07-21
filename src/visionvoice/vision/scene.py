"""Turn perception into a natural-language scene description.

If the active provider has a real vision-language model, we caption the actual image.
Otherwise (e.g. the mock/Ollama-text-only path) we synthesize a description from the
YOLOv8 detections — so the pipeline always produces something useful to speak.
"""

from __future__ import annotations

from visionvoice.providers.base import ModelProvider
from visionvoice.types import Detection, PerceptionSnapshot

_VLM_PROMPT = (
    "You are the eyes of a blind user. In 1-2 short sentences, describe the most "
    "important things in this scene and roughly where they are (left, ahead, right). "
    "Be concrete and calm. Mention hazards first."
)


def _count_phrase(label: str, n: int) -> str:
    return f"{n} {label}s" if n > 1 else f"a {label}"


def summarize_detections(detections: list[Detection], limit: int = 5) -> str:
    """Compose a plain-language summary from detections alone (no VLM needed)."""
    if not detections:
        return "I don't see anything notable in front of you right now."

    # Group by label, keep the nearest instance's position for phrasing.
    seen: dict[str, list[Detection]] = {}
    for det in detections:
        seen.setdefault(det.label, []).append(det)

    phrases: list[str] = []
    for label, group in list(seen.items())[:limit]:
        nearest = max(group, key=lambda d: d.area)
        phrases.append(
            f"{_count_phrase(label, len(group))} to your {nearest.position.replace('-', ' ')} "
            f"({nearest.distance})"
        )
    joined = "; ".join(phrases)
    return f"In front of you: {joined}."


def describe_scene(provider: ModelProvider, snapshot: PerceptionSnapshot) -> str:
    """Best available scene description for the current snapshot."""
    if provider.supports_vision and snapshot.has_image:
        try:
            caption = provider.describe_image(
                snapshot.image_bytes, _VLM_PROMPT, snapshot.image_media_type
            )
            if caption:
                return caption
        except Exception:  # pragma: no cover - network/model dependent
            pass  # fall through to the detection-based summary
    return summarize_detections(snapshot.detections)
