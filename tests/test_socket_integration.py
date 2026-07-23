from typing import Any

from conftest import performance_features
from flask import Flask

from dancify.extensions import socketio


def test_gameplay_socket_protocol(
    app: Flask, routine_payload: dict[str, Any], calibration_payload: dict[str, Any]
) -> None:
    http = app.test_client()
    routine_id = http.post("/api/v1/routines", json=routine_payload).get_json()["routineID"]
    session_id = http.post("/api/v1/sessions", json={"routineID": routine_id, "playerID": "p"}).get_json()["id"]
    http.post(f"/api/v1/sessions/{session_id}/calibration", json=calibration_payload)
    client = socketio.test_client(app, namespace="/gameplay")
    joined = client.emit("session.join", {"sessionID": session_id}, namespace="/gameplay", callback=True)
    assert joined["ok"] is True and joined["session"]["state"] == "ready"
    ready = client.emit(
        "playback.ready", {"sessionID": session_id, "delaySeconds": 0}, namespace="/gameplay", callback=True
    )
    start_at = ready["startAt"]
    accepted = http.post(f"/api/v1/sessions/{session_id}/motion", json={"features": performance_features()})
    assert accepted.status_code == 202
    progress = client.emit(
        "playback.progress",
        {"sessionID": session_id, "videoTime": 1.0, "serverTime": start_at + 1.0},
        namespace="/gameplay",
        callback=True,
    )
    assert progress["ok"] is True and progress["scores"][0]["windowScore"] > 95
    names = [event["name"] for event in client.get_received("/gameplay")]
    assert "playback.scheduled" in names
    assert "score.update" in names
    client.disconnect(namespace="/gameplay")


def test_socket_errors_are_typed(app: Flask) -> None:
    client = socketio.test_client(app, namespace="/gameplay")
    result: dict[str, Any] = client.emit("session.join", {}, namespace="/gameplay", callback=True)
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_request"
    assert client.get_received("/gameplay")[-1]["name"] == "session.error"
