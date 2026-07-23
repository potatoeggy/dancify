"""Versioned REST resources."""

from __future__ import annotations

from typing import Any, cast

from flask import Blueprint, Response, current_app, jsonify, request

from dancify.calibration import ClockObservation
from dancify.domain import MotionFeatures, Vector3, WristSide
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
    return jsonify(session.snapshot()), 201


@api.get("/sessions/<session_id>")
def get_session(session_id: str) -> Response:
    return jsonify(_sessions().snapshot(session_id))


@api.post("/sessions/<session_id>/calibration")
def calibrate(session_id: str) -> Response:
    data = require_mapping(request.get_json())
    observations = [
        ClockObservation(
            *(
                _float(require_mapping(item), key)
                for key in ("clientSend", "serverReceive", "serverSend", "clientReceive")
            )
        )
        for item in _list(data, "clockObservations")
    ]
    result = _sessions().calibrate(
        session_id, observations, _vectors(data, "neutral"), _vectors(data, "upward"), _vectors(data, "outward")
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


@api.post("/sessions/<session_id>/progress")
def progress(session_id: str) -> Response:
    data = require_mapping(request.get_json())
    video_time = _float(data, "videoTime")
    server_value = data.get("serverTime")
    server_time = None if server_value is None else _numeric(server_value, "serverTime")
    return jsonify({"scores": [score.to_dict() for score in _sessions().progress(session_id, video_time, server_time)]})


@api.post("/sessions/<session_id>/abort")
def abort(session_id: str) -> Response:
    return jsonify(_sessions().abort(session_id).snapshot())


def _feature(data: dict[str, Any]) -> MotionFeatures:
    horizontal = data.get("horizontalDirection")
    if horizontal is not None:
        horizontal = _numeric(horizontal, "horizontalDirection")
    return MotionFeatures(
        synchronized_time=_float(data, "timestamp"),
        wrist=WristSide(_string(data, "wrist")),
        vertical_direction=_float(data, "verticalDirection"),
        horizontal_direction=horizontal,
        horizontal_confidence=_float(data, "horizontalConfidence"),
        linear_intensity=_float(data, "linearIntensity"),
        movement_active=bool(data.get("movementActive", True)),
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
    return float(value)
