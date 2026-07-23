"""Socket.IO `/gameplay` event protocol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flask_socketio import emit, join_room  # type: ignore[import-untyped]

from dancify.extensions import socketio
from dancify.service import GameplaySessionService, require_mapping


def register_socket_events(service: GameplaySessionService) -> None:
    def guarded(
        function: Callable[[dict[str, Any]], dict[str, object]],
    ) -> Callable[[Any], dict[str, object]]:
        def wrapper(payload: Any = None) -> dict[str, object]:
            try:
                return function(require_mapping(payload or {}))
            except (ValueError, LookupError) as exc:
                error = {"code": "invalid_request", "message": str(exc)}
                emit("session.error", error, namespace="/gameplay")
                return {"ok": False, "error": error}

        return wrapper

    @guarded
    def join(data: dict[str, Any]) -> dict[str, object]:
        session_id = _session_id(data)
        join_room(session_id, namespace="/gameplay")
        return {"ok": True, "session": service.snapshot(session_id)}

    @guarded
    def calibration_observation(data: dict[str, Any]) -> dict[str, object]:
        # Full guided calibration is submitted over REST; this event acknowledges timing samples.
        return {"ok": True, "received": bool(data)}

    @guarded
    def playback_ready(data: dict[str, Any]) -> dict[str, object]:
        return {"ok": True, **service.start(_session_id(data), float(data.get("delaySeconds", 1.0)))}

    @guarded
    def playback_progress(data: dict[str, Any]) -> dict[str, object]:
        scores = service.progress(
            _session_id(data), float(data["videoTime"]), None if "serverTime" not in data else float(data["serverTime"])
        )
        return {"ok": True, "scores": [score.to_dict() for score in scores]}

    @guarded
    def abort(data: dict[str, Any]) -> dict[str, object]:
        return {"ok": True, "session": service.abort(_session_id(data)).snapshot()}

    socketio.on_event("session.join", join, namespace="/gameplay")
    socketio.on_event("calibration.observation", calibration_observation, namespace="/gameplay")
    socketio.on_event("playback.ready", playback_ready, namespace="/gameplay")
    socketio.on_event("playback.progress", playback_progress, namespace="/gameplay")
    socketio.on_event("session.abort", abort, namespace="/gameplay")


def _session_id(data: dict[str, Any]) -> str:
    value = data.get("sessionID")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("sessionID is required")
    return value
