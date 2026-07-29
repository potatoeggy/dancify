from copy import deepcopy
from typing import Any, cast

import pytest
from conftest import performance_features
from flask import Flask
from flask.testing import FlaskClient

from dancify.calibration import ClockObservation
from dancify.domain import WristSide
from dancify.extensions import socketio
from dancify.service import GameplaySessionService


def _create_session(client: FlaskClient, routine_payload: dict[str, Any]) -> str:
    routine = client.post("/api/v1/routines", json=routine_payload)
    assert routine.status_code == 201
    created = client.post(
        "/api/v1/sessions",
        json={"routineID": routine.get_json()["routineID"], "playerID": "live-player"},
    )
    assert created.status_code == 201
    return cast(str, created.get_json()["id"])


def _v2_calibration(calibration_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": 2,
        "clockObservations": deepcopy(calibration_payload["clockObservations"]),
        "wrists": {
            "left": {
                "neutral": [[0, 0, 1], [0, 0, 1]],
                "upward": [[0, 1, 1], [0, 1, 1]],
                "outward": [[1, 0, 1], [1, 0, 1]],
            },
            "right": {
                "neutral": [[0, 0, 1], [0, 0, 1]],
                "upward": [[0, 1, 1], [0, 1, 1]],
                "outward": [[-1, 0, 1], [-1, 0, 1]],
            },
        },
    }


def _right_only_v2_calibration(calibration_payload: dict[str, Any]) -> dict[str, Any]:
    payload = _v2_calibration(calibration_payload)
    del payload["wrists"]["left"]
    return payload


def _raw(
    wrist: str,
    packet: int,
    capture_us: int,
    client_time: float,
    acceleration: list[float] | None = None,
) -> dict[str, object]:
    return {
        "wrist": wrist,
        "packetNumber": packet,
        "captureTimestampUs": capture_us,
        "clientTimestamp": client_time,
        "accelerationG": acceleration or [1.0, 1.0, 1.0],
        "angularVelocityDps": [0.0, 0.0, 0.0],
    }


def test_calibration_v1_is_retained_and_v2_profiles_are_per_wrist(
    app: Flask,
    client: FlaskClient,
    routine_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
) -> None:
    legacy_id = _create_session(client, routine_payload)
    legacy = client.post(f"/api/v1/sessions/{legacy_id}/calibration", json=calibration_payload)
    assert legacy.status_code == 200
    assert set(legacy.get_json()) == {"timingOffsetSeconds", "horizontalConfidence"}
    legacy_snapshot = client.get(f"/api/v1/sessions/{legacy_id}").get_json()
    assert legacy_snapshot["calibrationVersion"] == 1
    assert legacy_snapshot["activeWrists"] == ["left", "right"]

    v2_id = _create_session(client, routine_payload)
    calibrated = client.post(f"/api/v1/sessions/{v2_id}/calibration", json=_v2_calibration(calibration_payload))
    assert calibrated.status_code == 200
    body = calibrated.get_json()
    assert body["schemaVersion"] == 2
    assert set(body["wrists"]) == {"left", "right"}
    assert client.get(f"/api/v1/sessions/{v2_id}").get_json()["activeWrists"] == ["left", "right"]

    service = cast(GameplaySessionService, app.extensions["session_service"])
    left = service.calibration_profile(v2_id, WristSide.LEFT)
    right = service.calibration_profile(v2_id, WristSide.RIGHT)
    assert left is not right
    assert left.horizontal_axis.x == pytest.approx(1.0)
    assert right.horizontal_axis.x == pytest.approx(-1.0)

    invalid_id = _create_session(client, routine_payload)
    left_only = _v2_calibration(calibration_payload)
    del left_only["wrists"]["right"]
    unknown = _v2_calibration(calibration_payload)
    unknown["wrists"] = {"unknown": unknown["wrists"]["right"]}
    for invalid in (left_only, {**_v2_calibration(calibration_payload), "wrists": {}}, unknown):
        response = client.post(f"/api/v1/sessions/{invalid_id}/calibration", json=invalid)
        assert response.status_code == 400
        assert "right" in response.get_json()["error"]["message"]


def test_right_only_calibration_snapshot_and_inactive_left_rejection(
    client: FlaskClient,
    routine_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
) -> None:
    session_id = _create_session(client, routine_payload)
    calibrated = client.post(
        f"/api/v1/sessions/{session_id}/calibration",
        json=_right_only_v2_calibration(calibration_payload),
    )
    assert calibrated.status_code == 200
    assert list(calibrated.get_json()["wrists"]) == ["right"]

    snapshot = client.get(f"/api/v1/sessions/{session_id}").get_json()
    assert snapshot["calibrationVersion"] == 2
    assert snapshot["activeWrists"] == ["right"]

    start_at = client.post(f"/api/v1/sessions/{session_id}/start", json={"delaySeconds": 0}).get_json()["startAt"]
    uploaded = client.post(
        f"/api/v1/sessions/{session_id}/motion/raw",
        json={
            "samples": [
                _raw("left", 1, 1_000_000, start_at + 0.1),
                _raw("right", 1, 2_000_000, start_at + 0.1, [-1.0, 1.0, 1.0]),
            ]
        },
    )
    assert uploaded.status_code == 202
    result = uploaded.get_json()
    assert (result["accepted"], result["dropped"]) == (1, 1)
    assert result["errors"] == [
        {
            "index": 0,
            "code": "uncalibrated_wrist",
            "message": "left wrist is not calibrated",
        }
    ]
    assert result["motionHealth"]["wrists"]["left"]["dropped"] == 1
    assert result["motionHealth"]["wrists"]["right"]["accepted"] == 1


def test_aligned_right_only_raw_motion_scores_above_95(
    client: FlaskClient,
    routine_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
) -> None:
    from math import cos, sin

    session_id = _create_session(client, routine_payload)
    calibrated = client.post(
        f"/api/v1/sessions/{session_id}/calibration",
        json=_right_only_v2_calibration(calibration_payload),
    )
    assert calibrated.status_code == 200
    start_at = client.post(f"/api/v1/sessions/{session_id}/start", json={"delaySeconds": 0}).get_json()["startAt"]
    samples: list[dict[str, object]] = []
    for index in range(50):
        timestamp = index / 50
        horizontal, vertical = sin(timestamp * 4), cos(timestamp * 4)
        samples.append(
            _raw("right", index, 8_000_000 + index * 20_000, start_at + timestamp, [-horizontal, vertical, 1.0])
        )

    uploaded = client.post(f"/api/v1/sessions/{session_id}/motion/raw", json={"samples": samples})
    assert uploaded.status_code == 202
    assert uploaded.get_json()["accepted"] == 50
    scored = client.post(
        f"/api/v1/sessions/{session_id}/progress",
        json={"videoTime": 1.0, "serverTime": start_at + 1.0},
    ).get_json()["scores"]
    assert len(scored) == 1
    assert scored[0]["valid"] is True
    assert scored[0]["windowScore"] > 95


def test_raw_motion_maps_to_playback_time_and_reports_partial_health(
    app: Flask,
    client: FlaskClient,
    routine_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
) -> None:
    session_id = _create_session(client, routine_payload)
    assert (
        client.post(f"/api/v1/sessions/{session_id}/calibration", json=_v2_calibration(calibration_payload)).status_code
        == 200
    )
    start_at = client.post(f"/api/v1/sessions/{session_id}/start", json={"delaySeconds": 0}).get_json()["startAt"]

    socket = socketio.test_client(app, namespace="/gameplay")
    joined = socket.emit("session.join", {"sessionID": session_id}, namespace="/gameplay", callback=True)
    assert joined["ok"] is True
    socket.get_received("/gameplay")

    response = client.post(
        f"/api/v1/sessions/{session_id}/motion/raw",
        json={
            "samples": [
                _raw("left", 1, 1_000_000, start_at + 0.1),
                _raw("right", 7, 2_000_000, start_at + 0.1, [-1.0, 1.0, 1.0]),
                {"packetNumber": 2},
            ]
        },
    )
    assert response.status_code == 202
    result = response.get_json()
    assert (result["accepted"], result["dropped"]) == (2, 1)
    assert result["errors"] == [{"index": 2, "code": "invalid_sample", "message": "captureTimestampUs is required"}]
    assert result["motionHealth"]["wrists"]["left"]["accepted"] == 1
    assert result["motionHealth"]["wrists"]["right"]["accepted"] == 1

    second = client.post(
        f"/api/v1/sessions/{session_id}/motion/raw",
        json={
            "samples": [
                _raw("left", 1, 1_000_000, start_at + 0.1),
                _raw("left", 2, 1_100_000, start_at + 0.2),
                _raw("middle", 3, 1_200_000, start_at + 0.3),
            ]
        },
    ).get_json()
    assert (second["accepted"], second["dropped"]) == (1, 2)
    assert [error["code"] for error in second["errors"]] == ["invalid_sample", "duplicate_packet"]
    snapshot = client.get(f"/api/v1/sessions/{session_id}").get_json()
    assert snapshot["motionHealth"]["accepted"] == 3
    assert snapshot["motionHealth"]["dropped"] == 3
    assert snapshot["motionHealth"]["wrists"]["left"]["duplicates"] == 1

    events = socket.get_received("/gameplay")
    health_events = [event for event in events if event["name"] == "motion.health"]
    assert len(health_events) == 2
    latest = health_events[-1]["args"][0]
    assert latest["schemaVersion"] == 1
    assert latest["motionHealth"]["accepted"] == 3
    socket.disconnect(namespace="/gameplay")


def test_calibration_observation_uses_measured_server_timestamps(
    app: Flask, client: FlaskClient, routine_payload: dict[str, Any]
) -> None:
    session_id = _create_session(client, routine_payload)
    socket = socketio.test_client(app, namespace="/gameplay")
    result = socket.emit(
        "calibration.observation",
        {"sessionID": session_id, "clientSend": 10.0},
        namespace="/gameplay",
        callback=True,
    )
    assert result["ok"] is True
    assert result["clientSend"] == 10.0
    assert result["serverReceive"] <= result["serverSend"]
    assert result["observation"] == {
        "clientSend": result["clientSend"],
        "serverReceive": result["serverReceive"],
        "serverSend": result["serverSend"],
    }
    # Receipt on the client completes a valid four-timestamp observation.
    completed = ClockObservation(
        result["clientSend"], result["serverReceive"], result["serverSend"], result["serverSend"] + 0.01
    )
    assert completed.round_trip >= 0

    invalid = socket.emit(
        "calibration.observation",
        {"sessionID": session_id},
        namespace="/gameplay",
        callback=True,
    )
    assert invalid["ok"] is False
    assert invalid["error"]["code"] == "invalid_request"
    socket.disconnect(namespace="/gameplay")


def test_raw_capture_clock_mapping_scores_in_playback_coordinates(
    client: FlaskClient,
    routine_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
) -> None:
    from math import cos, sin

    session_id = _create_session(client, routine_payload)
    client.post(f"/api/v1/sessions/{session_id}/calibration", json=_v2_calibration(calibration_payload))
    start_at = client.post(f"/api/v1/sessions/{session_id}/start", json={"delaySeconds": 0}).get_json()["startAt"]
    samples: list[dict[str, object]] = []
    for index in range(50):
        timestamp = index / 50
        horizontal, vertical = sin(timestamp * 4), cos(timestamp * 4)
        samples.append(
            _raw("left", index, 5_000_000 + index * 20_000, start_at + timestamp, [horizontal, vertical, 1.0])
        )
        samples.append(
            _raw("right", index, 8_000_000 + index * 20_000, start_at + timestamp, [-horizontal, vertical, 1.0])
        )
    uploaded = client.post(f"/api/v1/sessions/{session_id}/motion/raw", json={"samples": samples})
    assert uploaded.status_code == 202
    assert uploaded.get_json()["accepted"] == 100
    scored = client.post(
        f"/api/v1/sessions/{session_id}/progress",
        json={"videoTime": 1.0, "serverTime": start_at + 1.0},
    ).get_json()["scores"]
    assert len(scored) == 1
    assert scored[0]["windowScore"] > 95


def test_scores_are_authoritative_in_rest_snapshot_and_socket_rejoin(
    app: Flask,
    client: FlaskClient,
    routine_payload: dict[str, Any],
    calibration_payload: dict[str, Any],
) -> None:
    session_id = _create_session(client, routine_payload)
    client.post(f"/api/v1/sessions/{session_id}/calibration", json=calibration_payload)
    start_at = client.post(f"/api/v1/sessions/{session_id}/start", json={"delaySeconds": 0}).get_json()["startAt"]
    client.post(f"/api/v1/sessions/{session_id}/motion", json={"features": performance_features()})
    scored = client.post(
        f"/api/v1/sessions/{session_id}/progress",
        json={"videoTime": 1.0, "serverTime": start_at + 1.0},
    ).get_json()["scores"]
    assert len(scored) == 1

    snapshot = client.get(f"/api/v1/sessions/{session_id}").get_json()
    assert snapshot["scores"] == scored
    socket = socketio.test_client(app, namespace="/gameplay")
    rejoined = socket.emit("session.join", {"sessionID": session_id}, namespace="/gameplay", callback=True)
    assert rejoined["session"]["scores"] == scored
    assert rejoined["session"]["cumulative_score"] == scored[0]["cumulativeScore"]
    socket.disconnect(namespace="/gameplay")
