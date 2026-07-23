"""Video-ingestion contract adapter and reference feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from math import ceil, hypot, isfinite
from typing import Any, cast

from dancify.domain import (
    MotionFeatures,
    ReferenceMotionSample,
    ReferenceWristSample,
    ScoringWindow,
    WristSide,
)


@dataclass(frozen=True, slots=True)
class RoutineImport:
    title: str
    source_video_url: str
    duration_seconds: float
    fps: float
    reference_motion: tuple[ReferenceMotionSample, ...]
    scoring_windows: tuple[ScoringWindow, ...]


def parse_routine(data: dict[str, Any]) -> RoutineImport:
    metadata = _mapping(data.get("metadata", {}), "metadata")
    raw_motion = data.get("motion_signal", data.get("referenceMotion"))
    if not isinstance(raw_motion, list) or not raw_motion:
        raise ValueError("motion_signal/referenceMotion must be a non-empty list")
    motion_items = cast(list[Any], raw_motion)  # type: ignore[redundant-cast]
    samples = tuple(_sample(_mapping(item, "motion sample")) for item in motion_items)
    if any(right.timestamp_seconds <= left.timestamp_seconds for left, right in pairwise(samples)):
        raise ValueError("reference timestamps must be strictly increasing")

    source = data.get("sourceVideoURL", metadata.get("source_video"))
    if not isinstance(source, str) or not source.strip():
        raise ValueError("sourceVideoURL or metadata.source_video is required")
    title_value = data.get("title", metadata.get("title", source.rsplit("/", 1)[-1]))
    if not isinstance(title_value, str) or not title_value.strip():
        raise ValueError("title must be a non-empty string")
    duration = _number(data.get("duration", metadata.get("duration_seconds", samples[-1].timestamp_seconds)))
    fps = _number(metadata.get("fps", data.get("fps", 30.0)))
    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    windows = generate_windows(duration)
    return RoutineImport(title_value.strip(), source.strip(), duration, fps, samples, windows)


def generate_windows(duration_seconds: float, size_seconds: float = 1.0) -> tuple[ScoringWindow, ...]:
    if duration_seconds <= 0 or size_seconds <= 0:
        raise ValueError("duration and window size must be positive")
    count = ceil(duration_seconds / size_seconds)
    return tuple(
        ScoringWindow(
            index=index,
            start_seconds=index * size_seconds,
            end_seconds=min((index + 1) * size_seconds, duration_seconds),
            scoreable=(min((index + 1) * size_seconds, duration_seconds) - index * size_seconds) >= 0.5,
        )
        for index in range(count)
    )


def serialize_reference(samples: tuple[ReferenceMotionSample, ...]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for sample in samples:
        result.append(
            {
                "timestamp": sample.timestamp_seconds,
                "left_wrist": None if sample.left_wrist is None else asdict(sample.left_wrist),
                "right_wrist": None if sample.right_wrist is None else asdict(sample.right_wrist),
            }
        )
    return result


def deserialize_reference(items: list[dict[str, Any]]) -> tuple[ReferenceMotionSample, ...]:
    return tuple(_sample(item) for item in items)


def reference_features(
    samples: tuple[ReferenceMotionSample, ...],
    start: float,
    end: float,
) -> tuple[MotionFeatures, ...]:
    features: list[MotionFeatures] = []
    for sample in samples:
        if not start <= sample.timestamp_seconds < end:
            continue
        for wrist, value in (
            (WristSide.LEFT, sample.left_wrist),
            (WristSide.RIGHT, sample.right_wrist),
        ):
            if value is None or value.ax is None or value.ay is None:
                continue
            intensity = hypot(value.ax, value.ay)
            if intensity <= 1e-12:
                horizontal = 0.0
                vertical = 0.0
                active = False
            else:
                horizontal = value.ax / intensity
                vertical = -value.ay / intensity
                active = True
            features.append(
                MotionFeatures(
                    synchronized_time=sample.timestamp_seconds,
                    wrist=wrist,
                    vertical_direction=vertical,
                    horizontal_direction=horizontal,
                    horizontal_confidence=1.0,
                    linear_intensity=intensity,
                    movement_active=active,
                )
            )
    return tuple(features)


def _sample(data: dict[str, Any]) -> ReferenceMotionSample:
    timestamp = _number(data.get("timestamp"))
    return ReferenceMotionSample(
        timestamp_seconds=timestamp,
        left_wrist=_wrist(data.get("left_wrist", data.get("leftWristVector"))),
        right_wrist=_wrist(data.get("right_wrist", data.get("rightWristVector"))),
    )


def _wrist(value: Any) -> ReferenceWristSample | None:
    if value is None:
        return None
    data = _mapping(value, "wrist")
    return ReferenceWristSample(*(_optional_number(data.get(key)) for key in ("x", "y", "vx", "vy", "ax", "ay")))


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("numeric value required")
    result = float(value)
    if not isfinite(result):
        raise ValueError("numeric values must be finite")
    return result


def _optional_number(value: Any) -> float | None:
    return None if value is None else _number(value)
