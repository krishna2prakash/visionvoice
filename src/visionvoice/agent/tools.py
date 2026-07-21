"""The agent's tools — how it perceives on demand.

Each tool turns raw perception into a short text observation the LLM can reason over.
Exposing perception *as tools* (rather than dumping everything into the prompt) is what
makes this an agent: the model decides which senses to use for a given question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from visionvoice.ocr import read_text as ocr_read_text
from visionvoice.providers.base import ModelProvider
from visionvoice.types import PerceptionSnapshot, ToolSpec
from visionvoice.vision.scene import describe_scene

# Objects that warrant a safety warning when they're close and ahead.
_HAZARDS = {
    "car", "truck", "bus", "motorcycle", "bicycle", "train",
    "person", "dog", "stairs", "pothole", "traffic light", "stop sign",
}


@dataclass
class PerceptionContext:
    """Everything the tools need to answer a question about *this* moment."""

    provider: ModelProvider
    snapshot: PerceptionSnapshot
    # Optional raw frame (numpy) for OCR; may be None on offline/test paths.
    frame: Any | None = None


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="list_objects",
        description=(
            "List the objects currently detected in front of the user, with their "
            "direction (left/center/right) and rough distance. Use for 'what's around "
            "me', counting, or locating a specific object."
        ),
    ),
    ToolSpec(
        name="describe_scene",
        description=(
            "Get a natural-language description of the whole scene from the camera. "
            "Use for open questions like 'what do you see' or 'describe my surroundings'."
        ),
    ),
    ToolSpec(
        name="read_text",
        description=(
            "Read any text visible in front of the user (signs, labels, documents) via "
            "OCR. Use when asked to read something."
        ),
    ),
    ToolSpec(
        name="assess_safety",
        description=(
            "Assess whether it looks safe to move forward, based on nearby hazards such "
            "as people, vehicles, or obstacles. Use for 'is it safe' / 'can I cross'."
        ),
    ),
]


def execute_tool(name: str, arguments: dict, ctx: PerceptionContext) -> str:
    """Dispatch a tool call and return a short text observation."""
    if name == "list_objects":
        return _list_objects(ctx)
    if name == "describe_scene":
        return describe_scene(ctx.provider, ctx.snapshot)
    if name == "read_text":
        return _read_text(ctx)
    if name == "assess_safety":
        return _assess_safety(ctx)
    return f"Unknown tool: {name}"


def _list_objects(ctx: PerceptionContext) -> str:
    dets = ctx.snapshot.detections
    if not dets:
        return "No objects detected."
    lines = [
        f"- {d.label}: {d.distance}, to the {d.position.replace('-', ' ')} "
        f"(confidence {d.confidence:.0%})"
        for d in dets
    ]
    return "Detected objects (nearest first):\n" + "\n".join(lines)


def _read_text(ctx: PerceptionContext) -> str:
    if ctx.frame is None:
        return "No camera frame available to read text from."
    text = ocr_read_text(ctx.frame)
    return f'The visible text reads: "{text}"' if text else "No readable text detected."


def _assess_safety(ctx: PerceptionContext) -> str:
    hazards = [
        d
        for d in ctx.snapshot.detections
        if d.label in _HAZARDS and d.distance in ("very close", "close")
    ]
    if not hazards:
        return "No nearby hazards detected. It appears clear ahead, but stay alert."
    described = ", ".join(
        f"{d.label} ({d.distance}, {d.position.replace('-', ' ')})" for d in hazards
    )
    return f"Caution — nearby hazards: {described}. It may not be safe to move forward."
