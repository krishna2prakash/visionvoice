"""Shared, dependency-free data types used across the pipeline.

Keeping these free of heavy imports (numpy/cv2/torch) means the agent, providers and
tests can be exercised without installing the vision/voice extras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Position = Literal["far-left", "left", "center", "right", "far-right"]
Distance = Literal["very close", "close", "mid", "far"]


@dataclass(frozen=True)
class Detection:
    """A single detected object with a coarse, human-friendly location."""

    label: str
    confidence: float
    # Normalized bounding box in [0, 1]: (x1, y1, x2, y2)
    bbox: tuple[float, float, float, float]
    position: Position = "center"
    distance: Distance = "mid"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    def describe(self) -> str:
        return f"{self.label} ({self.distance}, {self.position})"


@dataclass
class PerceptionSnapshot:
    """Everything the agent can perceive at one instant in time."""

    detections: list[Detection] = field(default_factory=list)
    # Optional raw image bytes (JPEG/PNG) for the vision-language model.
    image_bytes: bytes | None = None
    image_media_type: str = "image/jpeg"
    # Optional pre-computed caption (populated by the VLM stage on demand).
    caption: str | None = None
    width: int = 0
    height: int = 0

    @property
    def has_image(self) -> bool:
        return self.image_bytes is not None

    def object_labels(self) -> list[str]:
        return [d.label for d in self.detections]


# ----------------------------------------------------------------------------
# LLM / agent message + tool types (provider-agnostic)
# ----------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolSpec:
    """Declaration of a tool the agent may call."""

    name: str
    description: str
    # JSON-schema-style parameter description.
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class ToolCall:
    """A request by the model to call a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: Role
    content: str = ""
    # For assistant messages that requested tools:
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For tool-result messages: which call this answers.
    tool_call_id: str | None = None


@dataclass
class ProviderResponse:
    """A single turn returned by a model provider."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0
