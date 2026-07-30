from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest
from flask import Flask
from flask.testing import FlaskClient

from dancify import create_app
from dancify.extensions import db
from dancify.service import GameplaySessionService, RoutineService


@pytest.fixture
def debug_app() -> Iterator[Flask]:
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "DANCIFY_ENVIRONMENT": "development",
            "DANCIFY_ENABLE_DEBUG_UI": True,
        }
    )
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()
        db.engine.dispose()


@pytest.fixture
def debug_client(debug_app: Flask) -> FlaskClient:
    return debug_app.test_client()


def create_routine(application: Flask, payload: dict[str, Any]) -> dict[str, object]:
    with application.app_context():
        service = cast(RoutineService, application.extensions["routine_service"])
        return service.create(payload)


def test_debug_routes_are_absent_by_default_and_config_fails_closed(client: FlaskClient) -> None:
    assert client.get("/_dev/scoring/").status_code == 404
    with pytest.raises(RuntimeError, match="exactly true or false"):
        create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
                "DANCIFY_ENABLE_DEBUG_UI": "yes",
            }
        )
    with pytest.raises(RuntimeError, match="allowed only in development"):
        create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
                "DANCIFY_ENVIRONMENT": "production",
                "DANCIFY_ENABLE_DEBUG_UI": True,
            }
        )


def test_debug_routes_require_loopback_and_send_security_headers(debug_client: FlaskClient) -> None:
    denied = debug_client.get("/_dev/scoring/api/routines", environ_overrides={"REMOTE_ADDR": "203.0.113.8"})
    assert denied.status_code == 403
    assert denied.json == {"error": {"code": "forbidden", "message": "debug scoring diagnostics are loopback-only"}}

    allowed = debug_client.get("/_dev/scoring/api/routines", environ_overrides={"REMOTE_ADDR": "::1"})
    assert allowed.status_code == 200
    assert allowed.headers["Cache-Control"] == "no-store"
    assert "default-src 'self'" in allowed.headers["Content-Security-Policy"]
    assert allowed.headers["X-Content-Type-Options"] == "nosniff"
    assert allowed.headers["X-Frame-Options"] == "DENY"


def test_list_select_and_reference_signal_contract(
    debug_app: Flask,
    debug_client: FlaskClient,
    routine_payload: dict[str, Any],
) -> None:
    first = create_routine(debug_app, {**routine_payload, "title": "First"})
    second = create_routine(debug_app, {**routine_payload, "title": "Second"})

    listed = debug_client.get("/_dev/scoring/api/routines")
    assert listed.status_code == 200
    routines = listed.json["routines"]
    assert [item["routineID"] for item in routines[:2]] == [second["routineID"], first["routineID"]]
    assert set(routines[0]) == {"routineID", "title", "sourceVideoURL", "duration", "fps", "schemaVersion"}

    routine_id = cast(str, second["routineID"])
    windows = debug_client.get(f"/_dev/scoring/api/routines/{routine_id}/windows")
    assert windows.status_code == 200
    assert windows.json["routine"] == {"routineID": routine_id, "title": "Second", "duration": 2.0}
    assert windows.json["windows"] == [
        {"index": 0, "startTime": 0.0, "endTime": 1.0, "scoreable": True},
        {"index": 1, "startTime": 1.0, "endTime": 2.0, "scoreable": True},
    ]

    reference = debug_client.get(f"/_dev/scoring/api/routines/{routine_id}/windows/0")
    assert reference.status_code == 200
    body = reference.json
    assert body["sampleRateHz"] == 50
    assert body["availableWrists"] == ["left", "right"]
    assert "not pose" in body["signalMeaning"]
    assert len(body["referenceSignals"]["left"]) == 50
    point = body["referenceSignals"]["left"][0]
    assert set(point) == {
        "offsetSeconds",
        "horizontalDirection",
        "verticalDirection",
        "linearIntensity",
        "horizontalComponent",
        "verticalComponent",
        "movementActive",
    }
    assert point["horizontalComponent"] == pytest.approx(point["horizontalDirection"] * point["linearIntensity"])


def test_attempts_are_bounded_typed_and_non_scoreable_windows_rejected(
    debug_app: Flask,
    debug_client: FlaskClient,
    routine_payload: dict[str, Any],
) -> None:
    routine = create_routine(debug_app, routine_payload)
    routine_id = routine["routineID"]
    endpoint = f"/_dev/scoring/api/routines/{routine_id}/windows/0/attempts"

    bad_payloads = [
        {"unknown": 1},
        {"activeWrists": []},
        {"activeWrists": ["left", "left"]},
        {"activeWrists": ["unknown"]},
        {"perturbation": {"directionRotationDegrees": True}},
        {"perturbation": {"intensityScale": 2.01}},
        {"perturbation": {"timeShiftMs": float("inf")}},
        {"perturbation": {"captureCoverage": -0.01}},
        {"perturbation": {"extra": 1}},
    ]
    for payload in bad_payloads:
        response = debug_client.post(endpoint, json=payload)
        assert response.status_code == 400, payload
        assert response.json["error"]["code"] == "invalid_request"

    wrong_type = debug_client.post(endpoint, data="[]", content_type="application/json")
    assert wrong_type.status_code == 400
    wrong_media = debug_client.post(endpoint, data="{}", content_type="text/plain")
    assert wrong_media.status_code == 400
    oversized = debug_client.post(
        endpoint,
        data='{"padding":"' + "x" * 5000 + '"}',
        content_type="application/json",
    )
    assert oversized.status_code == 413
    assert oversized.json["error"]["code"] == "request_too_large"

    partial_payload = cast(
        dict[str, Any],
        {**routine_payload, "metadata": {**routine_payload["metadata"], "duration_seconds": 2.2}},
    )
    partial = create_routine(debug_app, partial_payload)
    partial_response = debug_client.post(
        f"/_dev/scoring/api/routines/{partial['routineID']}/windows/2/attempts",
        json={},
    )
    assert partial_response.status_code == 400
    assert "not scoreable" in partial_response.json["error"]["message"]


def test_mock_attempts_are_deterministic_and_better_than_perturbed_attempts(
    debug_app: Flask,
    debug_client: FlaskClient,
    routine_payload: dict[str, Any],
) -> None:
    routine = create_routine(debug_app, routine_payload)
    endpoint = f"/_dev/scoring/api/routines/{routine['routineID']}/windows/0/attempts"

    baseline = debug_client.post(endpoint, json={})
    repeated = debug_client.post(endpoint, json={})
    reversed_attempt = debug_client.post(
        endpoint,
        json={"perturbation": {"directionRotationDegrees": 180.0}},
    )
    low_coverage = debug_client.post(
        endpoint,
        json={"perturbation": {"captureCoverage": 0.4, "sampleQuality": 0.5}},
    )
    right_only = debug_client.post(endpoint, json={"activeWrists": ["right"]})

    assert baseline.status_code == repeated.status_code == reversed_attempt.status_code == 200
    assert baseline.json == repeated.json
    assert baseline.json["mock"] is True
    assert baseline.json["metrics"]["score"] == pytest.approx(100.0)
    assert baseline.json["metrics"]["valid"] is True
    assert baseline.json["metrics"]["quality"] == pytest.approx(1.0)
    assert baseline.json["metrics"]["score"] > reversed_attempt.json["metrics"]["score"]
    assert reversed_attempt.json["metrics"]["breakdown"]["direction"] < 0.1
    assert low_coverage.json["metrics"]["valid"] is False
    assert low_coverage.json["metrics"]["coverage"] < 0.5
    assert low_coverage.json["metrics"]["quality"] < low_coverage.json["metrics"]["coverage"]
    assert right_only.json["activeWrists"] == ["right"]
    assert right_only.json["metrics"]["score"] > 99.0

    sessions = cast(GameplaySessionService, debug_app.extensions["session_service"])
    assert cast(Any, sessions)._window_evaluator is debug_app.extensions["window_scoring_evaluator"]


def test_debug_html_and_local_assets_smoke(debug_client: FlaskClient) -> None:
    page = debug_client.get("/_dev/scoring/")
    assert page.status_code == 200
    text = page.get_data(as_text=True)
    assert "Development-only MOCK diagnostics" in text
    assert "not a human pose" in text
    assert "does not read physical controllers" in text
    assert "cdn" not in text.lower()
    assert "Run MOCK attempt" in text

    script = debug_client.get("/_dev/scoring/assets/scoring_diagnostics.js")
    stylesheet = debug_client.get("/_dev/scoring/assets/scoring_diagnostics.css")
    assert script.status_code == stylesheet.status_code == 200
    script_text = script.get_data(as_text=True)
    assert "state.attempts.slice(0, 20)" in script_text
    assert "Solid = reference; dashed = selected MOCK attempt" in script_text
    assert "reference vs selected MOCK attempt" in script_text
    assert "selected MOCK attempt ${selectedAttempt.sequence} overlay" in script_text
    assert "innerHTML" not in script_text
