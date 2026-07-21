"""YOLO26 object detection wrapper.

Converts raw model output into :class:`~visionvoice.types.Detection` objects with a
coarse, human-friendly location (left / center / right) and distance estimate derived
from bounding-box size. Ultralytics/torch are imported lazily so the rest of the package
installs and tests without them.
"""

from __future__ import annotations

from visionvoice.types import Detection, Distance, Position


def bbox_to_position(cx: float) -> Position:
    """Map a normalized horizontal center [0, 1] to a spoken direction."""
    if cx < 0.20:
        return "far-left"
    if cx < 0.40:
        return "left"
    if cx < 0.60:
        return "center"
    if cx < 0.80:
        return "right"
    return "far-right"


def area_to_distance(area: float) -> Distance:
    """Very rough monocular distance proxy from normalized bbox area."""
    if area > 0.35:
        return "very close"
    if area > 0.12:
        return "close"
    if area > 0.03:
        return "mid"
    return "far"


class Detector:
    """Thin wrapper around an Ultralytics YOLO26 model."""

    def __init__(self, model_path: str = "yolo26n.pt", conf: float = 0.35) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "ultralytics is required for object detection. "
                'Install it with:  pip install -e ".[vision]"'
            ) from exc
        self._model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame) -> list[Detection]:
        """Run detection on a BGR/RGB numpy frame and return normalized detections."""
        height, width = frame.shape[:2]
        results = self._model(frame, conf=self.conf, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                nb = (x1 / width, y1 / height, x2 / width, y2 / height)
                cx = (nb[0] + nb[2]) / 2
                area = max(0.0, nb[2] - nb[0]) * max(0.0, nb[3] - nb[1])
                label = names[int(box.cls[0])]
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        bbox=nb,
                        position=bbox_to_position(cx),
                        distance=area_to_distance(area),
                    )
                )
        # Nearest (largest) first — that's what the user most needs to hear.
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections
