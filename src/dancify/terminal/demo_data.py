"""Generated production fixtures for the headless demo."""

from __future__ import annotations

from math import cos, isfinite, sin
from typing import Any


def _validate(duration: float, rate: int) -> None:
    if not isfinite(duration) or duration < 0.5 or rate <= 0:
        raise ValueError("demo duration must be finite and at least 0.5 seconds; rate must be positive")


def routine_payload(duration: float = 2.0, fps: int = 30) -> dict[str, Any]:
    """Generate a valid ingestion payload matching the backend reference schema."""

    _validate(duration, fps)
    rows: list[dict[str, Any]] = []
    for index in range(int(duration * fps) + 1):
        timestamp = min(duration, index / fps)
        horizontal, vertical = sin(timestamp * 4), cos(timestamp * 4)
        wrist = {
            "x": 0.5,
            "y": 0.5,
            "vx": horizontal,
            "vy": -vertical,
            "ax": horizontal,
            "ay": -vertical,
        }
        rows.append({"timestamp": timestamp, "left_wrist": wrist, "right_wrist": wrist})
    return {
        "title": "Dancify terminal demo",
        "metadata": {
            "source_video": "generated://dancify-demo",
            "fps": float(fps),
            "duration_seconds": duration,
        },
        "motion_signal": rows,
    }


def calibration_payload() -> dict[str, Any]:
    """Generate a stable guided spatial/timing calibration payload."""

    return {
        "clockObservations": [
            {"clientSend": 0.0, "serverReceive": 0.01, "serverSend": 0.02, "clientReceive": 0.03},
            {"clientSend": 1.0, "serverReceive": 1.01, "serverSend": 1.02, "clientReceive": 1.03},
        ],
        "neutral": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        "upward": [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        "outward": [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
    }


def motion_features(duration: float, rate_hz: int = 50) -> list[dict[str, object]]:
    """Generate canonical two-wrist features aligned with ``routine_payload``."""

    _validate(duration, rate_hz)
    result: list[dict[str, object]] = []
    for index in range(int(duration * rate_hz)):
        timestamp = index / rate_hz
        horizontal, vertical = sin(timestamp * 4), cos(timestamp * 4)
        intensity = (horizontal * horizontal + vertical * vertical) ** 0.5
        for wrist in ("left", "right"):
            result.append(
                {
                    "timestamp": timestamp,
                    "wrist": wrist,
                    "verticalDirection": vertical / intensity,
                    "horizontalDirection": horizontal / intensity,
                    "horizontalConfidence": 1.0,
                    "linearIntensity": intensity,
                    "movementActive": True,
                    "sampleQuality": 1.0,
                }
            )
    return result
