import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from conftest import performance_features
from flask.testing import FlaskClient

from dancify import create_app
from dancify.extensions import db
from dancify.models import SessionSummaryRecord


def create_routine(client: FlaskClient, payload: dict[str, Any]) -> str:
    response = client.post("/api/v1/routines", json=payload)
    assert response.status_code == 201
    return response.get_json()["routineID"]


def test_relative_sqlite_database_uses_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///local.db"})

    with app.app_context():
        db.create_all()
        db.engine.dispose()

    assert app.config["SQLALCHEMY_DATABASE_URI"] == f"sqlite:///{tmp_path / 'local.db'}"
    assert (tmp_path / "local.db").is_file()


def test_backend_loads_dotenv_with_safe_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "DATABASE_URL=sqlite:///dotenv.db\nDANCIFY_MAX_RAW_MOTION_BATCH=17\nSECRET_KEY=dotenv-secret\n"
    )

    with patch.dict(os.environ, {"SECRET_KEY": "process-secret"}, clear=True):
        app = create_app({"TESTING": True})
        assert app.config["SQLALCHEMY_DATABASE_URI"] == f"sqlite:///{tmp_path / 'dotenv.db'}"
        assert app.config["MAX_RAW_MOTION_BATCH"] == 17
        assert app.config["SECRET_KEY"] == "process-secret"

        explicit = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///explicit.db"})
        assert explicit.config["SQLALCHEMY_DATABASE_URI"] == f"sqlite:///{tmp_path / 'explicit.db'}"


def test_routine_api_and_end_to_end_gameplay(
    client: FlaskClient, routine_payload: dict[str, Any], calibration_payload: dict[str, Any]
) -> None:
    assert client.get("/health").get_json() == {"status": "ok"}
    routine_id = create_routine(client, routine_payload)
    assert client.get(f"/api/v1/routines/{routine_id}").get_json()["sourceVideoURL"] == "demo.mp4"
    assert len(client.get(f"/api/v1/routines/{routine_id}/windows").get_json()["windows"]) == 2

    created = client.post("/api/v1/sessions", json={"routineID": routine_id, "playerID": "player"})
    assert created.status_code == 201
    session_id = created.get_json()["id"]
    assert client.post(f"/api/v1/sessions/{session_id}/start", json={"delaySeconds": 0}).status_code == 409
    calibrated = client.post(f"/api/v1/sessions/{session_id}/calibration", json=calibration_payload)
    assert calibrated.status_code == 200
    scheduled = client.post(f"/api/v1/sessions/{session_id}/start", json={"delaySeconds": 0}).get_json()
    start_at = scheduled["startAt"]
    accepted = client.post(f"/api/v1/sessions/{session_id}/motion", json={"features": performance_features()})
    assert accepted.status_code == 202 and accepted.get_json()["accepted"] == 100

    scored = client.post(
        f"/api/v1/sessions/{session_id}/progress", json={"videoTime": 1.0, "serverTime": start_at + 1.0}
    )
    scores = scored.get_json()["scores"]
    assert len(scores) == 1 and scores[0]["windowScore"] > 95
    repeated = client.post(
        f"/api/v1/sessions/{session_id}/progress", json={"videoTime": 1.0, "serverTime": start_at + 1.0}
    )
    assert repeated.get_json()["scores"] == []

    completed = client.post(
        f"/api/v1/sessions/{session_id}/progress", json={"videoTime": 2.0, "serverTime": start_at + 2.0}
    )
    assert completed.status_code == 200
    assert client.get(f"/api/v1/sessions/{session_id}").get_json()["state"] == "completed"
    assert db.session.get(SessionSummaryRecord, session_id) is not None


def test_validation_not_found_drift_and_abort(
    client: FlaskClient, routine_payload: dict[str, Any], calibration_payload: dict[str, Any]
) -> None:
    assert client.get("/api/v1/routines/missing").status_code == 404
    assert client.post("/api/v1/routines", json={}).status_code == 400
    routine_id = create_routine(client, routine_payload)
    session_id = client.post("/api/v1/sessions", json={"routineID": routine_id, "playerID": "p"}).get_json()["id"]
    client.post(f"/api/v1/sessions/{session_id}/calibration", json=calibration_payload)
    start = client.post(f"/api/v1/sessions/{session_id}/start", json={"delaySeconds": 0}).get_json()["startAt"]
    drifted = client.post(f"/api/v1/sessions/{session_id}/progress", json={"videoTime": 0.1, "serverTime": start + 1.0})
    assert drifted.status_code == 200
    assert client.get(f"/api/v1/sessions/{session_id}").get_json()["state"] == "paused"
    aborted = client.post(f"/api/v1/sessions/{session_id}/abort")
    assert aborted.get_json()["state"] == "aborted"
