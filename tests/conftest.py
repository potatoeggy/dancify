from collections.abc import Iterator
from math import cos, sin
from typing import Any

import pytest
from flask import Flask
from flask.testing import FlaskClient

from dancify import create_app
from dancify.extensions import db


@pytest.fixture
def app() -> Iterator[Flask]:
    application = create_app({"TESTING": True, "SECRET_KEY": "test", "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()
        db.engine.dispose()


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    return app.test_client()


@pytest.fixture
def routine_payload() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index in range(61):
        timestamp = index / 30
        ax, ay = sin(timestamp * 4), -cos(timestamp * 4)
        wrist = {"x": 0.5, "y": 0.5, "vx": ax, "vy": ay, "ax": ax, "ay": ay}
        rows.append({"timestamp": timestamp, "left_wrist": wrist, "right_wrist": wrist})
    return {
        "title": "Demo",
        "metadata": {"source_video": "demo.mp4", "fps": 30.0, "duration_seconds": 2.0},
        "motion_signal": rows,
    }


@pytest.fixture
def calibration_payload() -> dict[str, Any]:
    return {
        "clockObservations": [
            {"clientSend": 0.0, "serverReceive": 0.01, "serverSend": 0.02, "clientReceive": 0.03},
            {"clientSend": 1.0, "serverReceive": 1.01, "serverSend": 1.02, "clientReceive": 1.03},
        ],
        "neutral": [[0, 0, 1], [0, 0, 1]],
        "upward": [[0, 1, 1], [0, 1, 1]],
        "outward": [[1, 0, 1], [1, 0, 1]],
    }


def performance_features(duration: float = 1.0, reversed_direction: bool = False) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    sign = -1.0 if reversed_direction else 1.0
    for index in range(int(duration * 50)):
        timestamp = index / 50
        ax, vertical = sin(timestamp * 4), cos(timestamp * 4)
        intensity = (ax * ax + vertical * vertical) ** 0.5
        for wrist in ("left", "right"):
            result.append(
                {
                    "timestamp": timestamp,
                    "wrist": wrist,
                    "verticalDirection": sign * vertical / intensity,
                    "horizontalDirection": sign * ax / intensity,
                    "horizontalConfidence": 1.0,
                    "linearIntensity": intensity,
                    "movementActive": True,
                    "sampleQuality": 1.0,
                }
            )
    return result
