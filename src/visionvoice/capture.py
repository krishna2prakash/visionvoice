"""Frame capture sources.

A small abstraction over "where frames come from" so the pipeline is identical whether
it reads a live webcam, a Raspberry Pi camera, or a single image file (handy for demos,
tests, and one-shot Q&A). OpenCV is imported lazily.
"""

from __future__ import annotations

import threading
from pathlib import Path


def encode_jpeg(frame) -> bytes:
    """Encode a numpy BGR frame to JPEG bytes (for the vision-language model)."""
    import cv2

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:  # pragma: no cover - defensive
        raise RuntimeError("Failed to JPEG-encode frame")
    return buf.tobytes()


class ImageFileSource:
    """A single still image, exposed through the same ``read()`` interface as a camera."""

    def __init__(self, path: str | Path) -> None:
        import cv2

        self.path = str(path)
        frame = cv2.imread(self.path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {self.path}")
        self._frame = frame

    def read(self):
        return self._frame

    def release(self) -> None:  # symmetry with camera sources
        self._frame = None


class ThreadedCamera:
    """Background-threaded webcam grabber.

    Grabbing frames on a worker thread keeps the newest frame always available and stops
    the detection stage from stalling on camera I/O — a standard low-latency pattern.
    """

    def __init__(self, index: int = 0) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self) -> object | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()
