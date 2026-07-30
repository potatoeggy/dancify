from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from dancify.terminal.config import ClientConfig
from dancify.terminal.errors import APIError, ConnectionFailure, ProtocolError
from dancify.terminal.rest import DancifyAPI


def response(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/health"):
        return httpx.Response(200, json={"status": "ok"})
    if path.endswith("/routines") and request.method == "POST":
        return httpx.Response(
            201,
            json={
                "routineID": "r",
                "title": "Demo",
                "sourceVideoURL": "demo.mp4",
                "duration": 2.0,
                "fps": 30.0,
                "schemaVersion": 1,
            },
        )
    if path.endswith("/routines/r/windows"):
        return httpx.Response(
            200,
            json={"windows": [{"index": 0, "startTime": 0, "endTime": 1, "scoreable": True}]},
        )
    if path.endswith("/routines/r"):
        return response(httpx.Request("POST", "http://test/api/v1/routines"))
    session = {
        "id": "s",
        "routine_id": "r",
        "player_id": "p",
        "state": "ready",
        "playback_start_time": None,
        "current_timestamp": 0.0,
        "current_window": 0,
        "cumulative_score": 0.0,
        "event_sequence": 0,
    }
    if path.endswith("/sessions"):
        return httpx.Response(201, json=session)
    if path.endswith("/sessions/s/calibration"):
        return httpx.Response(200, json={"timingOffsetSeconds": 0, "horizontalConfidence": 1})
    if path.endswith("/sessions/s/start"):
        return httpx.Response(200, json={"startAt": 10.0})
    if path.endswith("/sessions/s/motion"):
        body: dict[str, Any] = __import__("json").loads(request.content)
        return httpx.Response(202, json={"accepted": len(body["features"])})
    if path.endswith("/sessions/s/progress"):
        return httpx.Response(
            200,
            json={
                "scores": [
                    {
                        "windowIndex": 0,
                        "windowStartSeconds": 0,
                        "windowScore": 100,
                        "cumulativeScore": 100,
                        "valid": True,
                    }
                ]
            },
        )
    if path.endswith("/sessions/s/retry") and request.method == "POST":
        return httpx.Response(201, json={**session, "id": "s-retry", "state": "ready"})
    if path.endswith("/sessions/s/abort"):
        return httpx.Response(200, json={**session, "state": "aborted"})
    if path.endswith("/sessions/s"):
        return httpx.Response(200, json=session)
    return httpx.Response(404, json={"error": {"code": "not_found", "message": "missing"}})


def test_all_rest_operations() -> None:
    async def scenario() -> None:
        config = ClientConfig("http://test")
        client = httpx.AsyncClient(transport=httpx.MockTransport(response), base_url=config.api_url)
        api = DancifyAPI(config, client)
        assert await api.health()
        assert (await api.import_routine({})).id == "r"
        assert (await api.get_routine("r")).title == "Demo"
        assert len(await api.get_windows("r")) == 1
        assert (await api.create_session("r", "p")).id == "s"
        assert (await api.get_session("s")).player_id == "p"
        assert (await api.calibrate("s", {})).horizontal_confidence == 1
        assert await api.start("s", 0) == 10
        features = [{"timestamp": 0.0}]
        assert await api.upload_motion("s", features) == 1
        assert (await api.progress("s", 1.0, 11.0))[0].value == 100
        retried = await api.retry("s")
        assert retried.id == "s-retry"
        assert retried.state.value == "ready"
        assert (await api.abort("s")).state.value == "aborted"
        await api.close()
        await client.aclose()

    asyncio.run(scenario())


def test_rest_error_non_json_and_connection_failure() -> None:
    async def scenario() -> None:
        config = ClientConfig("http://test")
        missing = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(404, json={"error": {"code": "x", "message": "bad"}})
            ),
            base_url=config.api_url,
        )
        with pytest.raises(APIError):
            await DancifyAPI(config, missing).get_session("missing")
        broken = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, text="nope")),
            base_url=config.api_url,
        )
        with pytest.raises(ProtocolError):
            await DancifyAPI(config, broken).health()

        def fail(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline")

        offline = httpx.AsyncClient(transport=httpx.MockTransport(fail), base_url=config.api_url)
        with pytest.raises(ConnectionFailure):
            await DancifyAPI(config, offline).health()
        await missing.aclose()
        await broken.aclose()
        await offline.aclose()

    asyncio.run(scenario())
