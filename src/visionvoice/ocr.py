"""Optical character recognition (reading signs / labels).

Wraps pytesseract when available; degrades to an empty string otherwise so the rest of
the pipeline keeps working without the OCR extra or a Tesseract install.
"""

from __future__ import annotations


def read_text(frame) -> str:
    """Return any text detected in the frame, or an empty string.

    Requires the ``vision`` extra (pytesseract) plus a system Tesseract binary.
    """
    try:
        import cv2
        import pytesseract
    except ImportError:
        return ""

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
    except Exception:  # pragma: no cover - depends on system tesseract
        return ""
    return " ".join(text.split()).strip()
