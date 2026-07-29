"""Pure reducer for untrusted Socket.IO gameplay events."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

from dancify.domain import SessionState
from dancify.terminal.dto import JsonObject, MotionHealth, Score, Session, object_value
from dancify.terminal.errors import ProtocolError


@dataclass(slots=True)
class GameplayState:
    session: Session | None = None
    last_sequence: int = 0
    scores: dict[int, Score] = field(default_factory=dict[int, Score])
    last_error: str | None = None
    connected: bool = False
    motion_health: MotionHealth | None = None
    last_event: str | None = None

    def reconcile(self, session: Session) -> None:
        """Apply an authoritative join/REST snapshot without moving sequence backward."""

        self.session = session
        self.last_sequence = max(self.last_sequence, session.event_sequence)
        for score in session.scores:
            self.scores[score.window_index] = score
        if session.motion_health is not None:
            self.motion_health = session.motion_health

    def apply(self, event: str, raw: Any) -> bool:
        payload = object_value(raw, event)
        sequence = payload.get("sequence")
        if sequence is not None:
            if isinstance(sequence, bool) or not isinstance(sequence, int):
                raise ProtocolError("event sequence must be an integer")
            if sequence <= self.last_sequence:
                return False
            self.last_sequence = sequence
        self.last_event = event
        if event == "session.snapshot":
            self.reconcile(Session.from_dict(payload))
        elif event == "score.update":
            self._score(payload)
        elif event == "motion.health":
            self.motion_health = MotionHealth.from_dict(object_value(payload.get("motionHealth"), "motion health"))
        elif event == "session.completed":
            self._state(SessionState.COMPLETED, _optional_float(payload, "cumulativeScore", 0.0))
        elif event == "session.paused":
            self._state(SessionState.PAUSED)
        elif event == "session.error":
            self.last_error = str(payload.get("message", "unknown Socket.IO error"))
        return True

    def apply_ack_scores(self, raw: Any) -> None:
        payload = object_value(raw, "playback.progress acknowledgement")
        if payload.get("ok") is not True:
            error = payload.get("error")
            details = object_value(error, "socket error") if isinstance(error, dict) else {}
            raise ProtocolError(str(details.get("message", "Socket.IO playback.progress failed")))
        scores = payload.get("scores", [])
        if not isinstance(scores, list):
            raise ProtocolError("acknowledgement scores must be a list")
        for item in cast(list[Any], scores):  # type: ignore[redundant-cast]
            self._score(object_value(item, "score"))

    def _score(self, payload: JsonObject) -> bool:
        score = Score.from_dict(payload)
        if score.window_index in self.scores:
            return False
        self.scores[score.window_index] = score
        if self.session is not None:
            self.session = replace(
                self.session, current_window=score.window_index, cumulative_score=score.cumulative_score
            )
        return True

    def _state(self, state: SessionState, score: float | None = None) -> None:
        if self.session is not None:
            self.session = replace(
                self.session,
                state=state,
                cumulative_score=self.session.cumulative_score if score is None else score,
            )


def _optional_float(payload: JsonObject, key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(f"{key} must be numeric")
    return float(value)
