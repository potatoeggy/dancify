"""Typed DTOs at the untrusted backend boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from dancify.domain import SessionState
from dancify.terminal.errors import ProtocolError

JsonObject = dict[str, Any]


def object_value(value: Any, label: str = "response") -> JsonObject:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be a JSON object")
    return cast(JsonObject, value)


def _str(data: JsonObject, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{key} must be a non-empty string")
    return value


def _float(data: JsonObject, key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(f"{key} must be numeric")
    return float(value)


def _int(data: JsonObject, key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an integer")
    return value


def _objects(data: JsonObject, key: str, label: str) -> list[JsonObject]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ProtocolError(f"{key} must be a list")
    return [object_value(item, label) for item in cast(list[Any], value)]  # type: ignore[redundant-cast]


@dataclass(frozen=True, slots=True)
class Routine:
    id: str
    title: str
    source_video_url: str
    duration: float
    fps: float
    schema_version: int

    @classmethod
    def from_dict(cls, data: JsonObject) -> Routine:
        return cls(
            _str(data, "routineID"),
            _str(data, "title"),
            _str(data, "sourceVideoURL"),
            _float(data, "duration"),
            _float(data, "fps"),
            _int(data, "schemaVersion"),
        )


@dataclass(frozen=True, slots=True)
class RoutineWindow:
    index: int
    start_time: float
    end_time: float
    scoreable: bool

    @classmethod
    def from_dict(cls, data: JsonObject) -> RoutineWindow:
        scoreable = data.get("scoreable")
        if not isinstance(scoreable, bool):
            raise ProtocolError("scoreable must be boolean")
        return cls(_int(data, "index"), _float(data, "startTime"), _float(data, "endTime"), scoreable)


@dataclass(frozen=True, slots=True)
class Score:
    window_index: int
    window_start_seconds: float
    value: float
    cumulative_score: float
    valid: bool

    @classmethod
    def from_dict(cls, data: JsonObject) -> Score:
        valid = data.get("valid")
        if not isinstance(valid, bool):
            raise ProtocolError("valid must be boolean")
        return cls(
            _int(data, "windowIndex"),
            _float(data, "windowStartSeconds"),
            _float(data, "windowScore"),
            _float(data, "cumulativeScore"),
            valid,
        )


@dataclass(frozen=True, slots=True)
class WristMotionHealth:
    accepted: int
    dropped: int
    duplicates: int
    out_of_order: int
    invalid_timing: int
    quality: float

    @classmethod
    def from_dict(cls, data: JsonObject) -> WristMotionHealth:
        return cls(
            _int(data, "accepted"),
            _int(data, "dropped"),
            _int(data, "duplicates"),
            _int(data, "outOfOrder"),
            _int(data, "invalidTiming"),
            _float(data, "quality"),
        )


@dataclass(frozen=True, slots=True)
class MotionHealth:
    accepted: int
    dropped: int
    malformed: int
    quality: float
    wrists: dict[str, WristMotionHealth]

    @classmethod
    def from_dict(cls, data: JsonObject) -> MotionHealth:
        raw_wrists = object_value(data.get("wrists"), "motion health wrists")
        wrists = {
            name: WristMotionHealth.from_dict(object_value(value, f"{name} wrist health"))
            for name, value in raw_wrists.items()
        }
        if set(wrists) != {"left", "right"}:
            raise ProtocolError("motion health must contain left and right wrists")
        return cls(
            _int(data, "accepted"),
            _int(data, "dropped"),
            _int(data, "malformed"),
            _float(data, "quality"),
            wrists,
        )


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    routine_id: str
    player_id: str
    state: SessionState
    playback_start_time: float | None
    current_timestamp: float
    current_window: int
    cumulative_score: float
    event_sequence: int
    scores: tuple[Score, ...] = ()
    calibration_version: int = 0
    motion_health: MotionHealth | None = None
    active_wrists: tuple[str, ...] = ("left", "right")

    @classmethod
    def from_dict(cls, data: JsonObject) -> Session:
        raw_start = data.get("playback_start_time")
        if raw_start is not None and (isinstance(raw_start, bool) or not isinstance(raw_start, int | float)):
            raise ProtocolError("playback_start_time must be numeric or null")
        try:
            state = SessionState(_str(data, "state"))
        except ValueError as exc:
            raise ProtocolError(f"unknown session state: {data.get('state')}") from exc
        raw_scores = data.get("scores", [])
        if not isinstance(raw_scores, list):
            raise ProtocolError("scores must be a list")
        scores = tuple(Score.from_dict(object_value(item, "score")) for item in cast(list[Any], raw_scores))  # type: ignore[redundant-cast]
        version = data.get("calibrationVersion", 0)
        if isinstance(version, bool) or not isinstance(version, int):
            raise ProtocolError("calibrationVersion must be an integer")
        raw_health = data.get("motionHealth")
        health = None if raw_health is None else MotionHealth.from_dict(object_value(raw_health, "motion health"))
        raw_active = data.get("activeWrists", ["left", "right"])
        if not isinstance(raw_active, list):
            raise ProtocolError("activeWrists must be a list")
        active_values: list[str] = []
        for wrist in cast(list[Any], raw_active):  # type: ignore[redundant-cast]
            if not isinstance(wrist, str) or wrist not in {"left", "right"}:
                raise ProtocolError("activeWrists must contain only left and right")
            active_values.append(wrist)
        active = tuple(active_values)
        if len(set(active)) != len(active):
            raise ProtocolError("activeWrists must not contain duplicates")
        return cls(
            _str(data, "id"),
            _str(data, "routine_id"),
            _str(data, "player_id"),
            state,
            None if raw_start is None else float(raw_start),
            _float(data, "current_timestamp"),
            _int(data, "current_window"),
            _float(data, "cumulative_score"),
            _int(data, "event_sequence"),
            scores,
            version,
            health,
            active,
        )


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    timing_offset_seconds: float
    horizontal_confidence: float
    schema_version: int = 1
    wrist_confidence: dict[str, float] = field(default_factory=dict[str, float])

    @classmethod
    def from_dict(cls, data: JsonObject) -> CalibrationResult:
        version = data.get("schemaVersion", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version not in {1, 2}:
            raise ProtocolError("calibration schemaVersion must be 1 or 2")
        raw_wrists = data.get("wrists", {})
        if not isinstance(raw_wrists, dict):
            raise ProtocolError("calibration wrists must be an object")
        wrists = {
            name: _float(object_value(value, f"{name} calibration"), "horizontalConfidence")
            for name, value in cast(JsonObject, raw_wrists).items()
        }
        if version == 2 and set(wrists) not in ({"right"}, {"left", "right"}):
            raise ProtocolError("calibration v2 result must contain right or left and right wrists")
        return cls(_float(data, "timingOffsetSeconds"), _float(data, "horizontalConfidence"), version, wrists)


@dataclass(frozen=True, slots=True)
class ClockObservation:
    client_send: float
    server_receive: float
    server_send: float
    client_receive: float

    def to_client_time(self, server_time: float) -> float:
        """Map server monotonic time onto this client's monotonic clock."""

        client_midpoint = (self.client_send + self.client_receive) / 2.0
        server_midpoint = (self.server_receive + self.server_send) / 2.0
        return server_time + client_midpoint - server_midpoint

    def to_payload(self) -> dict[str, float]:
        return {
            "clientSend": self.client_send,
            "serverReceive": self.server_receive,
            "serverSend": self.server_send,
            "clientReceive": self.client_receive,
        }


@dataclass(frozen=True, slots=True)
class RawUploadError:
    index: int
    code: str
    message: str

    @classmethod
    def from_dict(cls, data: JsonObject) -> RawUploadError:
        return cls(_int(data, "index"), _str(data, "code"), _str(data, "message"))


@dataclass(frozen=True, slots=True)
class RawUploadResult:
    accepted: int
    dropped: int
    errors: tuple[RawUploadError, ...]
    motion_health: MotionHealth

    @classmethod
    def from_dict(cls, data: JsonObject, attempted: int | None = None) -> RawUploadResult:
        result = cls(
            _int(data, "accepted"),
            _int(data, "dropped"),
            tuple(RawUploadError.from_dict(item) for item in _objects(data, "errors", "raw upload error")),
            MotionHealth.from_dict(object_value(data.get("motionHealth"), "motion health")),
        )
        if attempted is not None and result.accepted + result.dropped != attempted:
            raise ProtocolError(f"raw upload accounted for {result.accepted + result.dropped} of {attempted} samples")
        if result.dropped != len(result.errors):
            raise ProtocolError("raw upload dropped count does not match errors")
        return result
