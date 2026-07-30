"""Dancify Flask application factory."""

from __future__ import annotations

import os
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any, cast

from flask import Flask, Response, jsonify
from sqlalchemy.engine import make_url

from dancify.api import api
from dancify.debug_scoring import WindowScoringEvaluator
from dancify.debug_ui import scoring_debug
from dancify.environment import load_environment
from dancify.extensions import db, socketio
from dancify.scoring import ScorerRegistry, ScoringConfig, WeightedDtwScoringAlgorithm
from dancify.service import ConflictError, GameplaySessionService, NotFoundError, RoutineService
from dancify.socket_events import register_socket_events

_SCORING_FLOAT_OVERRIDES: dict[str, tuple[str, float, float]] = {
    "DANCIFY_SCORING_DIRECTION_WEIGHT": ("direction_weight", 0.0, 1.0),
    "DANCIFY_SCORING_MAGNITUDE_WEIGHT": ("magnitude_weight", 0.0, 1.0),
    "DANCIFY_SCORING_TIMING_WEIGHT": ("timing_weight", 0.0, 1.0),
    "DANCIFY_SCORING_TIMING_GRACE_SECONDS": ("timing_grace_seconds", 0.0, 1.0),
    "DANCIFY_SCORING_TIMING_FALLOFF_SECONDS": ("timing_falloff_seconds", 0.001, 2.0),
    "DANCIFY_SCORING_TIMING_PATH_COST_WEIGHT": ("timing_path_cost_weight", 0.0, 1.0),
    "DANCIFY_SCORING_MIN_COVERAGE": ("minimum_coverage", 0.0, 1.0),
    "DANCIFY_SCORING_FULL_COVERAGE": ("full_coverage", 0.0, 1.0),
    "DANCIFY_SCORING_COVERAGE_QUALITY_FLOOR": ("coverage_quality_floor", 0.0, 1.0),
    "DANCIFY_SCORING_SAMPLE_QUALITY_FLOOR": ("sample_quality_floor", 0.0, 1.0),
    "DANCIFY_SCORING_RESAMPLE_MAX_GAP_SECONDS": ("resample_max_gap_seconds", 0.001, 0.5),
}
_SCORING_INT_OVERRIDES: dict[str, tuple[str, int, int]] = {
    "DANCIFY_SCORING_SAKOE_CHIBA_RADIUS": ("sakoe_chiba_radius", 0, 100),
    "DANCIFY_SCORING_SAMPLE_RATE_HZ": ("sample_rate_hz", 1, 240),
}


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"true", "false"}:
        return value == "true"
    raise RuntimeError(f"{name} must be exactly true or false")


def _database_uri(uri: str) -> str:
    """Resolve relative SQLite files like Alembic does: from the current directory."""

    url = make_url(uri)
    database = url.database
    if url.drivername.startswith("sqlite") and database is not None and database not in {"", ":memory:"}:
        path = Path(database)
        if not path.is_absolute():
            return url.set(database=str(Path.cwd() / path)).render_as_string(hide_password=False)
    return uri


def _bounded_float(value: object, key: str, lower: float, upper: float) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{key} must be a finite number between {lower:g} and {upper:g}")
    try:
        parsed = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} must be a finite number between {lower:g} and {upper:g}") from exc
    if not isfinite(parsed) or not lower <= parsed <= upper:
        raise RuntimeError(f"{key} must be a finite number between {lower:g} and {upper:g}")
    return parsed


def _bounded_int(value: object, key: str, lower: int, upper: int) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{key} must be an integer between {lower} and {upper}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise RuntimeError(f"{key} must be an integer between {lower} and {upper}") from exc
    else:
        raise RuntimeError(f"{key} must be an integer between {lower} and {upper}")
    if not lower <= parsed <= upper:
        raise RuntimeError(f"{key} must be an integer between {lower} and {upper}")
    return parsed


def _resolve_scoring_config(values: dict[str, Any]) -> ScoringConfig:
    profile_key = "DANCIFY_SCORING_PROFILE"
    profile = values.get(profile_key, "generous")
    if not isinstance(profile, str) or profile not in {"generous", "balanced", "strict"}:
        raise RuntimeError(f"{profile_key} must be exactly generous, balanced, or strict")
    active = ScoringConfig.named(profile)
    overrides: dict[str, object] = {}
    source_keys: list[str] = []
    for key, (field_name, lower, upper) in _SCORING_FLOAT_OVERRIDES.items():
        if key in values and values[key] is not None:
            overrides[field_name] = _bounded_float(values[key], key, lower, upper)
            source_keys.append(key)
    for key, (field_name, lower, upper) in _SCORING_INT_OVERRIDES.items():
        if key in values and values[key] is not None:
            overrides[field_name] = _bounded_int(values[key], key, lower, upper)
            source_keys.append(key)
    try:
        return replace(active, **overrides)  # type: ignore[arg-type]
    except ValueError as exc:
        named = ", ".join(source_keys) if source_keys else profile_key
        raise RuntimeError(f"invalid scoring configuration ({named}): {exc}") from exc


def create_app(config: dict[str, Any] | None = None) -> Flask:
    load_environment()
    app = Flask(__name__)
    scoring_environment = {
        key: raw
        for key in {"DANCIFY_SCORING_PROFILE", *_SCORING_FLOAT_OVERRIDES, *_SCORING_INT_OVERRIDES}
        if (raw := os.getenv(key)) is not None
    }
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "development-only-change-me"),
        SQLALCHEMY_DATABASE_URI=os.getenv(
            "DATABASE_URL", "postgresql+psycopg://dancify:dancify@localhost:5432/dancify"
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        JSON_SORT_KEYS=False,
        MAX_RAW_MOTION_BATCH=_positive_int_env("DANCIFY_MAX_RAW_MOTION_BATCH", 1000),
        DANCIFY_ENVIRONMENT=os.getenv("DANCIFY_ENVIRONMENT", "production"),
        DANCIFY_ENABLE_DEBUG_UI=os.getenv("DANCIFY_ENABLE_DEBUG_UI", "false"),
        DANCIFY_SCORING_PROFILE="generous",
    )
    app.config.update(scoring_environment)  # pyright: ignore[reportUnknownMemberType]
    if config:
        app.config.update(config)  # pyright: ignore[reportUnknownMemberType]
    debug_enabled = _strict_bool(cast(object, app.config["DANCIFY_ENABLE_DEBUG_UI"]), "DANCIFY_ENABLE_DEBUG_UI")
    environment = cast(object, app.config["DANCIFY_ENVIRONMENT"])
    if not isinstance(environment, str) or environment not in {"development", "production"}:
        raise RuntimeError("DANCIFY_ENVIRONMENT must be exactly development or production")
    if debug_enabled and environment != "development":
        raise RuntimeError("DANCIFY_ENABLE_DEBUG_UI=true is allowed only in development")
    scoring_config = _resolve_scoring_config(cast(dict[str, Any], app.config))
    app.config["SQLALCHEMY_DATABASE_URI"] = _database_uri(cast(str, app.config["SQLALCHEMY_DATABASE_URI"]))
    db.init_app(app)
    socketio.init_app(app)

    routines = RoutineService()
    scorers = ScorerRegistry()
    weighted_dtw = WeightedDtwScoringAlgorithm.from_config(scoring_config)
    scorers.register(weighted_dtw)
    window_evaluator = WindowScoringEvaluator.from_config(scoring_config)

    def publish(session_id: str, event: str, payload: dict[str, object]) -> None:
        socketio.emit(  # pyright: ignore[reportUnknownMemberType]
            event, payload, to=session_id, namespace="/gameplay"
        )

    sessions = GameplaySessionService(
        routines,
        scorers,
        publish,
        max_raw_batch=cast(int, app.config["MAX_RAW_MOTION_BATCH"]),
        window_evaluator=window_evaluator,
    )
    app.extensions["routine_service"] = routines
    app.extensions["session_service"] = sessions
    app.extensions["scorer_registry"] = scorers
    app.extensions["weighted_dtw_scorer"] = weighted_dtw
    app.extensions["window_scoring_evaluator"] = window_evaluator
    app.extensions["scoring_config"] = scoring_config
    app.register_blueprint(api)
    if debug_enabled:
        app.register_blueprint(scoring_debug)
    register_socket_events(sessions)

    @app.get("/health")
    def root_health() -> Response:
        return jsonify({"status": "ok"})

    @app.errorhandler(NotFoundError)
    def not_found(error: NotFoundError) -> tuple[Response, int]:
        return jsonify({"error": {"code": "not_found", "message": str(error)}}), 404

    @app.errorhandler(ConflictError)
    def conflict(error: ConflictError) -> tuple[Response, int]:
        return jsonify({"error": {"code": "invalid_state", "message": str(error)}}), 409

    @app.errorhandler(ValueError)
    def bad_request(error: ValueError) -> tuple[Response, int]:
        return jsonify({"error": {"code": "invalid_request", "message": str(error)}}), 400

    return app
