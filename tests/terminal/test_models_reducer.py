from __future__ import annotations

import pytest

from dancify.domain import SessionState, WristSide
from dancify.terminal.config import ClientConfig
from dancify.terminal.demo_data import calibration_payload, motion_features, routine_payload
from dancify.terminal.dto import (
    CalibrationResult,
    Routine,
    RoutineWindow,
    Score,
    Session,
    object_value,
)
from dancify.terminal.errors import APIError, ConfigurationError, ProtocolError
from dancify.terminal.reducer import GameplayState


def session_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "session",
        "routine_id": "routine",
        "player_id": "player",
        "state": "ready",
        "playback_start_time": None,
        "current_timestamp": 0.0,
        "current_window": 0,
        "cumulative_score": 0.0,
        "event_sequence": 2,
    }
    payload.update(changes)
    return payload


def score_payload(index: int = 0, value: float = 90.0) -> dict[str, object]:
    return {
        "windowIndex": index,
        "windowStartSeconds": float(index),
        "windowScore": value,
        "cumulativeScore": value,
        "valid": True,
    }


def test_client_config_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ClientConfig("http://localhost:5000/", 2.0, 10, 2)
    assert config.base_url == "http://localhost:5000"
    assert config.api_url.endswith("/api/v1")
    monkeypatch.delenv("DANCIFY_DSU_LEFT_SLOT", raising=False)
    right_only = ClientConfig.from_env()
    assert right_only.dsu_left_slot is None
    assert right_only.capture_config.left_slot is None
    assert right_only.capture_config.slots == {WristSide.RIGHT: 1}
    monkeypatch.setenv("DANCIFY_DSU_LEFT_SLOT", "")
    assert ClientConfig.from_env().dsu_left_slot is None
    monkeypatch.setenv("DANCIFY_URL", "https://dance.example")
    monkeypatch.setenv("DANCIFY_TIMEOUT", "3")
    monkeypatch.setenv("DANCIFY_DSU_HOST", "dsu.local")
    monkeypatch.setenv("DANCIFY_DSU_PORT", "26761")
    monkeypatch.setenv("DANCIFY_DSU_LEFT_SLOT", "2")
    monkeypatch.setenv("DANCIFY_DSU_RIGHT_SLOT", "3")
    environment = ClientConfig.from_env()
    assert environment.timeout_seconds == 3
    assert environment.capture_config.host == "dsu.local"
    assert environment.capture_config.port == 26761
    assert environment.capture_config.left_slot == 2
    assert environment.capture_config.right_slot == 3
    with pytest.raises(ConfigurationError):
        ClientConfig("file:///tmp/server")
    with pytest.raises(ConfigurationError):
        ClientConfig("http://localhost?bad=1")
    monkeypatch.setenv("DANCIFY_TIMEOUT", "bad")
    with pytest.raises(ConfigurationError, match="must be numeric"):
        ClientConfig.from_env()


def test_dtos_parse_backend_contract_and_reject_bad_values() -> None:
    routine = Routine.from_dict(
        {
            "routineID": "r",
            "title": "Dance",
            "sourceVideoURL": "demo.mp4",
            "duration": 2,
            "fps": 30,
            "schemaVersion": 1,
        }
    )
    assert routine.duration == 2
    assert RoutineWindow.from_dict({"index": 0, "startTime": 0, "endTime": 1, "scoreable": True}).scoreable
    session = Session.from_dict(session_payload())
    assert session.state is SessionState.READY
    assert session.active_wrists == ("left", "right")
    right_only = Session.from_dict(session_payload(activeWrists=["right"]))
    assert right_only.active_wrists == ("right",)
    assert CalibrationResult.from_dict({"timingOffsetSeconds": 0, "horizontalConfidence": 1}).horizontal_confidence == 1
    calibration = CalibrationResult.from_dict(
        {
            "timingOffsetSeconds": 0,
            "horizontalConfidence": 1,
            "schemaVersion": 2,
            "wrists": {"right": {"horizontalConfidence": 0.9}},
        }
    )
    assert calibration.wrist_confidence == {"right": 0.9}
    invalid_confidence = {"horizontalConfidence": 0.9}
    for wrists in ({}, {"left": invalid_confidence}, {"unknown": invalid_confidence}):
        with pytest.raises(ProtocolError):
            CalibrationResult.from_dict(
                {
                    "timingOffsetSeconds": 0,
                    "horizontalConfidence": 1,
                    "schemaVersion": 2,
                    "wrists": wrists,
                }
            )
    assert Score.from_dict(score_payload()).value == 90
    with pytest.raises(ProtocolError):
        object_value([])
    with pytest.raises(ProtocolError):
        Session.from_dict(session_payload(state="unknown"))
    with pytest.raises(ProtocolError):
        Session.from_dict(session_payload(activeWrists=["left", "left"]))
    with pytest.raises(ProtocolError):
        RoutineWindow.from_dict({"index": 0, "startTime": 0, "endTime": 1, "scoreable": 1})
    with pytest.raises(ProtocolError):
        Score.from_dict({**score_payload(), "valid": "yes"})


def test_generated_demo_data_matches_backend_shape() -> None:
    routine = routine_payload(2.0)
    assert len(routine["motion_signal"]) == 61
    assert routine["metadata"]["duration_seconds"] == 2.0
    assert len(motion_features(2.0)) == 200
    assert calibration_payload()["neutral"]
    with pytest.raises(ValueError):
        routine_payload(0.1)


def test_reducer_orders_events_and_deduplicates_scores() -> None:
    state = GameplayState()
    state.reconcile(Session.from_dict(session_payload()))
    event = {**session_payload(state="playing", event_sequence=2), "sequence": 3}
    assert state.apply("session.snapshot", event)
    assert state.session is not None and state.session.state is SessionState.PLAYING
    assert not state.apply("session.snapshot", event)
    assert state.apply("score.update", {**score_payload(), "sequence": 4})
    state.apply_ack_scores({"ok": True, "scores": [score_payload(), score_payload(1, 80)]})
    assert sorted(state.scores) == [0, 1]
    state.apply("session.paused", {"sequence": 5})
    assert state.session is not None and state.session.state is SessionState.PAUSED
    state.apply("session.completed", {"sequence": 6, "cumulativeScore": 85})
    assert state.session is not None and state.session.state is SessionState.COMPLETED
    state.apply("session.error", {"message": "bad"})
    assert state.last_error == "bad"
    with pytest.raises(ProtocolError):
        state.apply("score.update", {**score_payload(), "sequence": "bad"})
    with pytest.raises(ProtocolError):
        state.apply_ack_scores({"ok": False, "error": {"message": "no"}})


def test_actionable_api_error() -> None:
    error = APIError(404, "not_found", "missing")
    assert "Check that" in error.display()
