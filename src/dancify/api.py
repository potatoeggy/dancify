"""Versioned REST resources."""

from __future__ import annotations

from math import isfinite
from typing import Any, cast

from flask import Blueprint, Response, current_app, jsonify, request

from dancify.calibration import ClockObservation
from dancify.domain import MotionFeatures, RawMotionSample, Vector3, WristSide
from dancify.service import GameplaySessionService, RoutineService, require_mapping

api = Blueprint("api", __name__, url_prefix="/api/v1")


def _routines() -> RoutineService:
    service: RoutineService = current_app.extensions["routine_service"]
    return service


def _sessions() -> GameplaySessionService:
    service: GameplaySessionService = current_app.extensions["session_service"]
    return service


@api.get("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@api.post("/routines")
def create_routine() -> tuple[Response, int]:
    return jsonify(_routines().create(require_mapping(request.get_json()))), 201


@api.get("/routines/<routine_id>")
def get_routine(routine_id: str) -> Response:
    return jsonify(_routines().metadata(routine_id))


@api.get("/routines/<routine_id>/windows")
def get_windows(routine_id: str) -> Response:
    return jsonify({"windows": _routines().windows(routine_id)})


@api.post("/sessions")
def create_session() -> tuple[Response, int]:
    data = require_mapping(request.get_json())
    session = _sessions().create(
        _string(data, "routineID"), _string(data, "playerID"), str(data.get("scoringAlgorithm", "weighted_dtw"))
    )
    return jsonify(_sessions().snapshot(session.id)), 201


@api.get("/sessions/<session_id>")
def get_session(session_id: str) -> Response:
    return jsonify(_sessions().snapshot(session_id))


@api.post("/sessions/<session_id>/calibration")
def calibrate(session_id: str) -> Response:
    data = require_mapping(request.get_json())
    version_value = data.get("schemaVersion", 1)
    if isinstance(version_value, bool) or not isinstance(version_value, int) or version_value not in {1, 2}:
        raise ValueError("calibration schemaVersion must be 1 or 2")
    observations = _clock_observations(data)
    if version_value == 1:
        result = _sessions().calibrate(
            session_id,
            observations,
            _vectors(data, "neutral"),
            _vectors(data, "upward"),
            _vectors(data, "outward"),
        )
    else:
        wrists = require_mapping(data.get("wrists"))
        wrist_names = set(wrists)
        allowed_wrist_sets = ({WristSide.RIGHT.value}, {side.value for side in WristSide})
        if wrist_names not in allowed_wrist_sets:
            raise ValueError("schemaVersion 2 wrists must contain exactly right or left and right")
        gestures = {
            side: (
                _vectors(require_mapping(wrists[side.value]), "neutral"),
                _vectors(require_mapping(wrists[side.value]), "upward"),
                _vectors(require_mapping(wrists[side.value]), "outward"),
            )
            for side in WristSide
            if side.value in wrists
        }
        result = _sessions().calibrate(
            session_id,
            observations,
            [],
            [],
            [],
            calibration_version=2,
            wrist_gestures=gestures,
        )
    return jsonify(result)


@api.post("/sessions/<session_id>/start")
def start(session_id: str) -> Response:
    data = require_mapping(request.get_json(silent=True) or {})
    delay = data.get("delaySeconds", 1.0)
    if isinstance(delay, bool) or not isinstance(delay, int | float):
        raise ValueError("delaySeconds must be numeric")
    return jsonify(_sessions().start(session_id, float(delay)))


@api.post("/sessions/<session_id>/motion")
def ingest_motion(session_id: str) -> tuple[Response, int]:
    data = require_mapping(request.get_json())
    features = [_feature(require_mapping(item)) for item in _list(data, "features")]
    return jsonify({"accepted": _sessions().ingest_features(session_id, features)}), 202


@api.post("/sessions/<session_id>/motion/raw")
def ingest_raw_motion(session_id: str) -> tuple[Response, int]:
    data = require_mapping(request.get_json())
    raw_items = _list(data, "samples")
    if not raw_items:
        raise ValueError("samples must contain at least one raw motion sample")
    parsed: list[tuple[int, RawMotionSample]] = []
    errors: list[dict[str, object]] = []
    for index, item in enumerate(raw_items):
        try:
            parsed.append((index, _raw_motion_sample(require_mapping(item))))
        except ValueError as exc:
            errors.append({"index": index, "code": "invalid_sample", "message": str(exc)})
    return jsonify(_sessions().ingest_raw_motion(session_id, parsed, errors)), 202


@api.post("/sessions/<session_id>/progress")
def progress(session_id: str) -> Response:
    data = require_mapping(request.get_json())
    video_time = _float(data, "videoTime")
    server_value = data.get("serverTime")
    server_time = None if server_value is None else _numeric(server_value, "serverTime")
    return jsonify({"scores": [score.to_dict() for score in _sessions().progress(session_id, video_time, server_time)]})


def _clock_observations(data: dict[str, Any]) -> list[ClockObservation]:
    return [
        ClockObservation(
            *(
                _float(require_mapping(item), key)
                for key in ("clientSend", "serverReceive", "serverSend", "clientReceive")
            )
        )
        for item in _list(data, "clockObservations")
    ]


def _raw_motion_sample(data: dict[str, Any]) -> RawMotionSample:
    capture_us: int
    for key in ("captureTimestampUs", "captureTimestampMicros", "deviceTimestampUs"):
        if key in data:
            capture_us = _integer(data.get(key), key)
            break
    else:
        if "captureTime" not in data:
            raise ValueError("captureTimestampUs is required")
        capture_seconds = _numeric(data.get("captureTime"), "captureTime")
        capture_us = round(capture_seconds * 1_000_000)

    client_key = "clientTimestamp" if "clientTimestamp" in data else "clientTime"
    if client_key not in data:
        raise ValueError("clientTimestamp is required")
    acceleration_key = "accelerationG" if "accelerationG" in data else "acceleration"
    angular_key = "angularVelocityDps" if "angularVelocityDps" in data else "angularVelocity"
    if acceleration_key not in data:
        raise ValueError("accelerationG is required")
    if angular_key not in data:
        raise ValueError("angularVelocityDps is required")
    return RawMotionSample(
        wrist=WristSide(_string(data, "wrist")),
        capture_timestamp_us=capture_us,
        client_timestamp=_numeric(data.get(client_key), client_key),
        packet_number=_integer(data.get("packetNumber"), "packetNumber"),
        acceleration_g=Vector3.from_value(data.get(acceleration_key)),
        angular_velocity_dps=Vector3.from_value(data.get(angular_key)),
    )


@api.post("/sessions/<session_id>/abort")
def abort(session_id: str) -> Response:
    _sessions().abort(session_id)
    return jsonify(_sessions().snapshot(session_id))


def _feature(data: dict[str, Any]) -> MotionFeatures:
    horizontal = data.get("horizontalDirection")
    if horizontal is not None:
        horizontal = _numeric(horizontal, "horizontalDirection")
    movement_active = data.get("movementActive", True)
    if not isinstance(movement_active, bool):
        raise ValueError("movementActive must be boolean")
    return MotionFeatures(
        synchronized_time=_float(data, "timestamp"),
        wrist=WristSide(_string(data, "wrist")),
        vertical_direction=_float(data, "verticalDirection"),
        horizontal_direction=horizontal,
        horizontal_confidence=_float(data, "horizontalConfidence"),
        linear_intensity=_float(data, "linearIntensity"),
        movement_active=movement_active,
        sample_quality=_numeric(data.get("sampleQuality", 1.0), "sampleQuality"),
    )


def _vectors(data: dict[str, Any], key: str) -> list[Vector3]:
    return [Vector3.from_value(item) for item in _list(data, key)]


def _list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return cast(list[Any], value)  # type: ignore[redundant-cast]


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _float(data: dict[str, Any], key: str) -> float:
    return _numeric(data.get(key), key)


def _numeric(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _integer(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    result: int = value
    return result
