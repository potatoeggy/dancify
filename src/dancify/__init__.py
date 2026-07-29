"""Dancify Flask application factory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from flask import Flask, Response, jsonify
from sqlalchemy.engine import make_url

from dancify.api import api
from dancify.environment import load_environment
from dancify.extensions import db, socketio
from dancify.scoring import ScorerRegistry, WeightedDtwScoringAlgorithm
from dancify.service import ConflictError, GameplaySessionService, NotFoundError, RoutineService
from dancify.socket_events import register_socket_events


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


def _database_uri(uri: str) -> str:
    """Resolve relative SQLite files like Alembic does: from the current directory."""

    url = make_url(uri)
    database = url.database
    if url.drivername.startswith("sqlite") and database is not None and database not in {"", ":memory:"}:
        path = Path(database)
        if not path.is_absolute():
            return url.set(database=str(Path.cwd() / path)).render_as_string(hide_password=False)
    return uri


def create_app(config: dict[str, Any] | None = None) -> Flask:
    load_environment()
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "development-only-change-me"),
        SQLALCHEMY_DATABASE_URI=os.getenv(
            "DATABASE_URL", "postgresql+psycopg://dancify:dancify@localhost:5432/dancify"
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        JSON_SORT_KEYS=False,
        MAX_RAW_MOTION_BATCH=_positive_int_env("DANCIFY_MAX_RAW_MOTION_BATCH", 1000),
    )
    if config:
        app.config.update(config)  # pyright: ignore[reportUnknownMemberType]
    app.config["SQLALCHEMY_DATABASE_URI"] = _database_uri(cast(str, app.config["SQLALCHEMY_DATABASE_URI"]))
    db.init_app(app)
    socketio.init_app(app)

    routines = RoutineService()
    scorers = ScorerRegistry()
    scorers.register(WeightedDtwScoringAlgorithm())

    def publish(session_id: str, event: str, payload: dict[str, object]) -> None:
        socketio.emit(  # pyright: ignore[reportUnknownMemberType]
            event, payload, to=session_id, namespace="/gameplay"
        )

    sessions = GameplaySessionService(
        routines,
        scorers,
        publish,
        max_raw_batch=cast(int, app.config["MAX_RAW_MOTION_BATCH"]),
    )
    app.extensions["routine_service"] = routines
    app.extensions["session_service"] = sessions
    app.register_blueprint(api)
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
