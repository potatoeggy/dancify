"""Dancify Flask application factory."""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, Response, jsonify

from dancify.api import api
from dancify.extensions import db, socketio
from dancify.scoring import ScorerRegistry, WeightedDtwScoringAlgorithm
from dancify.service import ConflictError, GameplaySessionService, NotFoundError, RoutineService
from dancify.socket_events import register_socket_events


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "development-only-change-me"),
        SQLALCHEMY_DATABASE_URI=os.getenv(
            "DATABASE_URL", "postgresql+psycopg://dancify:dancify@localhost:5432/dancify"
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)  # pyright: ignore[reportUnknownMemberType]
    db.init_app(app)
    socketio.init_app(app)

    routines = RoutineService()
    scorers = ScorerRegistry()
    scorers.register(WeightedDtwScoringAlgorithm())

    def publish(session_id: str, event: str, payload: dict[str, object]) -> None:
        socketio.emit(  # pyright: ignore[reportUnknownMemberType]
            event, payload, to=session_id, namespace="/gameplay"
        )

    sessions = GameplaySessionService(routines, scorers, publish)
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
