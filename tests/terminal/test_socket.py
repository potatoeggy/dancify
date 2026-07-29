from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from dancify.terminal.config import ClientConfig
from dancify.terminal.errors import ProtocolError
from dancify.terminal.reducer import GameplayState
from dancify.terminal.socket import GameplaySocket


def session_payload(state: str = "ready") -> dict[str, object]:
    return {
        "id": "s",
        "routine_id": "r",
        "player_id": "p",
        "state": state,
        "playback_start_time": None,
        "current_timestamp": 0.0,
        "current_window": 0,
        "cumulative_score": 0.0,
        "event_sequence": 1,
    }


class FakeSocketClient:
    def __init__(self) -> None:
        self.handlers: dict[tuple[str, str], Any] = {}
        self.connected = False
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail_ready = False

    def on(self, event: str, handler: Any, namespace: str) -> None:
        self.handlers[(namespace, event)] = handler

    async def connect(self, *_: Any, **__: Any) -> None:
        self.connected = True
        handler = self.handlers[("/gameplay", "connect")]
        await handler()

    async def call(self, event: str, payload: dict[str, object], **_: Any) -> dict[str, Any]:
        self.calls.append((event, payload))
        if event == "session.join":
            return {"ok": True, "session": session_payload()}
        if event == "playback.ready":
            if self.fail_ready:
                return {"ok": False, "error": {"code": "bad", "message": "no"}}
            return {"ok": True, "startAt": 10.0}
        if event == "playback.progress":
            return {
                "ok": True,
                "scores": [
                    {
                        "windowIndex": 0,
                        "windowStartSeconds": 0,
                        "windowScore": 90,
                        "cumulativeScore": 90,
                        "valid": True,
                    }
                ],
            }
        if event == "session.abort":
            return {"ok": True, "session": session_payload("aborted")}
        raise AssertionError(event)

    async def disconnect(self) -> None:
        self.connected = False


def test_socket_workflow_and_reconnect() -> None:
    async def scenario() -> None:
        fake = FakeSocketClient()
        state = GameplayState()
        socket = GameplaySocket(ClientConfig("http://test"), "s", state, cast(Any, fake))
        joined = await socket.connect()
        assert joined.id == "s" and state.connected
        assert await socket.ready(0) == 10
        await socket.progress(1, 11)
        assert state.scores[0].value == 90
        assert (await socket.abort()).state.value == "aborted"
        await fake.handlers[("/gameplay", "score.update")](
            {
                "windowIndex": 1,
                "windowStartSeconds": 1,
                "windowScore": 80,
                "cumulativeScore": 85,
                "valid": True,
                "sequence": 3,
            }
        )
        assert state.scores[1].value == 80
        await fake.handlers[("/gameplay", "connect")]()
        assert [event for event, _ in fake.calls].count("session.join") == 2
        await socket.disconnect()
        assert not state.connected

    asyncio.run(scenario())


def test_socket_ack_error() -> None:
    async def scenario() -> None:
        fake = FakeSocketClient()
        fake.fail_ready = True
        socket = GameplaySocket(ClientConfig("http://test"), "s", client=cast(Any, fake))
        await socket.connect()
        with pytest.raises(ProtocolError, match="Socket.IO bad"):
            await socket.ready(0)
        await socket.disconnect()

    asyncio.run(scenario())
