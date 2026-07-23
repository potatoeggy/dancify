"""Transport-independent domain models for Dancify."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite, sqrt
from typing import Any, cast


class WristSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class SessionState(StrEnum):
    CREATED = "created"
    CALIBRATING = "calibrating"
    READY = "ready"
    SCHEDULED = "scheduled"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.z)):
            raise ValueError("vector values must be finite")

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, amount: float) -> Vector3:
        return Vector3(self.x * amount, self.y * amount, self.z * amount)

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return sqrt(self.dot(self))

    def normalized(self) -> Vector3:
        magnitude = self.norm()
        return self.scale(1.0 / magnitude) if magnitude > 1e-12 else Vector3(0.0, 0.0, 0.0)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_value(cls, value: Any) -> Vector3:
        if not isinstance(value, list):
            raise ValueError("vector must be a three-element list")
        values = cast(list[Any], value)  # type: ignore[redundant-cast]
        if len(values) != 3:
            raise ValueError("vector must be a three-element list")
        if any(isinstance(item, bool) or not isinstance(item, int | float) for item in values):
            raise ValueError("vector elements must be numbers")
        return cls(*(float(item) for item in values))


@dataclass(frozen=True, slots=True)
class ReferenceWristSample:
    x: float | None
    y: float | None
    vx: float | None
    vy: float | None
    ax: float | None
    ay: float | None

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.vx, self.vy, self.ax, self.ay)
        if any(value is not None and not isfinite(value) for value in values):
            raise ValueError("reference wrist values must be finite or null")


@dataclass(frozen=True, slots=True)
class ReferenceMotionSample:
    timestamp_seconds: float
    left_wrist: ReferenceWristSample | None
    right_wrist: ReferenceWristSample | None

    def __post_init__(self) -> None:
        if not isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0:
            raise ValueError("reference timestamp must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ScoringWindow:
    index: int
    start_seconds: float
    end_seconds: float
    scoreable: bool = True


@dataclass(frozen=True, slots=True)
class RawImuSample:
    device_id: str
    device_timestamp_us: int
    packet_number: int
    acceleration_g: Vector3
    angular_velocity_dps: Vector3

    def __post_init__(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id is required")
        if self.device_timestamp_us < 0 or self.packet_number < 0:
            raise ValueError("timestamp and packet number must be non-negative")


@dataclass(frozen=True, slots=True)
class MotionFeatures:
    synchronized_time: float
    wrist: WristSide
    vertical_direction: float
    horizontal_direction: float | None
    horizontal_confidence: float
    linear_intensity: float
    movement_active: bool
    sample_quality: float = 1.0

    def __post_init__(self) -> None:
        finite_values = (
            self.synchronized_time,
            self.vertical_direction,
            self.horizontal_confidence,
            self.linear_intensity,
            self.sample_quality,
        )
        if not all(isfinite(value) for value in finite_values):
            raise ValueError("motion feature values must be finite")
        if self.horizontal_direction is not None and not isfinite(self.horizontal_direction):
            raise ValueError("horizontal direction must be finite or null")
        if self.synchronized_time < 0 or self.linear_intensity < 0:
            raise ValueError("time and intensity must be non-negative")
        if not 0 <= self.horizontal_confidence <= 1 or not 0 <= self.sample_quality <= 1:
            raise ValueError("confidence and quality must be between zero and one")


@dataclass(frozen=True, slots=True)
class MotionWindow:
    index: int
    start_seconds: float
    end_seconds: float
    samples: tuple[MotionFeatures, ...]
    valid: bool = True
    quality: float = 1.0


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    direction: float
    magnitude: float
    timing: float
    quality: float

    def to_dict(self) -> dict[str, float]:
        return {
            "direction": self.direction,
            "magnitude": self.magnitude,
            "timing": self.timing,
            "quality": self.quality,
        }


@dataclass(frozen=True, slots=True)
class ScoreResult:
    window_index: int
    window_start_seconds: float
    value: float
    cumulative_score: float
    valid: bool
    breakdown: ScoreBreakdown

    def to_dict(self) -> dict[str, object]:
        return {
            "windowIndex": self.window_index,
            "windowStartSeconds": self.window_start_seconds,
            "windowScore": self.value,
            "cumulativeScore": self.cumulative_score,
            "valid": self.valid,
            "accuracyBreakdown": self.breakdown.to_dict(),
        }


@dataclass(slots=True)
class GameSession:
    id: str
    routine_id: str
    player_id: str
    state: SessionState = SessionState.CREATED
    playback_start_time: float | None = None
    current_timestamp: float = 0.0
    current_window: int = 0
    cumulative_score: float = 0.0
    event_sequence: int = 0
    scored_windows: set[int] = field(default_factory=set[int])

    def snapshot(self) -> dict[str, object]:
        return {
            "id": self.id,
            "routine_id": self.routine_id,
            "player_id": self.player_id,
            "state": self.state.value,
            "playback_start_time": self.playback_start_time,
            "current_timestamp": self.current_timestamp,
            "current_window": self.current_window,
            "cumulative_score": self.cumulative_score,
            "event_sequence": self.event_sequence,
        }
