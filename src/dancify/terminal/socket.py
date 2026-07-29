"""Reconnect/rejoin Socket.IO adapter for the ``/gameplay`` namespace."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import inspect
import time
from collections.abc import Awaitable, Callable
from math import ceil, isfinite
from types import TracebackType
from typing import Any, Self

import socketio  # type: ignore[import-untyped]
from socketio.exceptions import ConnectionError as SocketConnectionError  # type: ignore[import-untyped]
from socketio.exceptions import TimeoutError as SocketTimeoutError

from dancify.terminal.config import ClientConfig
from dancify.terminal.dto import ClockObservation, JsonObject, Session, object_value
from dancify.terminal.errors import ConnectionFailure, ProtocolError
from dancify.terminal.reducer import GameplayState

NAMESPACE = "/gameplay"
SERVER_EVENTS = (
    "session.snapshot",
    "calibration.result",
    "playback.scheduled",
    "score.update",
    "motion.health",
    "session.paused",
    "session.completed",
    "session.error",
)
type EventObserver = Callable[[str, JsonObject, GameplayState], Awaitable[None] | None]


class GameplaySocket:
    def __init__(
        self,
        config: ClientConfig,
        session_id: str,
        state: GameplayState | None = None,
        client: socketio.AsyncClient | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        observer: EventObserver | None = None,
    ) -> None:
        self.config = config
        self.session_id = session_id
        self.state = state or GameplayState()
        self._client = client or socketio.AsyncClient(reconnection=True, logger=False, engineio_logger=False)
        self._clock = clock
        self._observers: list[EventObserver] = [] if observer is None else [observer]
        self._joined_once = False
        self._register_handlers()

    def add_observer(self, observer: EventObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: EventObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    async def _notify(self, event: str, payload: JsonObject) -> None:
        for observer in tuple(self._observers):
            result = observer(event, payload, self.state)
            if inspect.isawaitable(result):
                await result

    def _register_handlers(self) -> None:
        async def connected() -> None:
            self.state.connected = True
            if self._joined_once:
                await self._join()

        async def disconnected() -> None:
            self.state.connected = False

        self._client.on("connect", connected, namespace=NAMESPACE)
        self._client.on("disconnect", disconnected, namespace=NAMESPACE)
        for event in SERVER_EVENTS:
            self._client.on(event, self._event_handler(event), namespace=NAMESPACE)

    def _event_handler(self, event: str) -> Any:
        async def handler(raw: Any) -> None:
            payload = object_value(raw, event)
            if self.state.apply(event, payload):
                await self._notify(event, payload)

        return handler

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self.disconnect()

    async def connect(self) -> Session:
        try:
            await self._client.connect(
                self.config.base_url,
                namespaces=[NAMESPACE],
                wait_timeout=max(1, ceil(self.config.timeout_seconds)),
            )
            session = await self._join()
        except (SocketConnectionError, SocketTimeoutError, TimeoutError, OSError) as exc:
            await self.disconnect()
            raise ConnectionFailure(
                f"cannot connect to Socket.IO namespace {NAMESPACE}",
                hint=f"Verify {self.config.base_url} is running and supports Socket.IO.",
            ) from exc
        self._joined_once = True
        return session

    async def _join(self) -> Session:
        result = await self._raw_call("session.join", {"sessionID": self.session_id})
        if result.get("ok") is not True:
            self._raise_ack(result)
        session = Session.from_dict(object_value(result.get("session"), "joined session"))
        self.state.reconcile(session)
        await self._notify("session.joined", object_value(result.get("session"), "joined session"))
        return session

    async def observe_clock(self) -> ClockObservation:
        """Complete one real four-timestamp NTP-style observation."""

        client_send = self._clock()
        result = await self._call("calibration.observation", {"clientSend": client_send})
        client_receive = self._clock()
        nested = result.get("observation")
        data = object_value(nested, "clock observation") if isinstance(nested, dict) else result
        echoed = _number(data, "clientSend")
        server_receive = _number(data, "serverReceive")
        server_send = _number(data, "serverSend")
        if abs(echoed - client_send) > 1e-6:
            raise ProtocolError("calibration observation did not echo clientSend")
        if client_receive < client_send or server_send < server_receive:
            raise ProtocolError("calibration observation timestamps are not ordered")
        return ClockObservation(client_send, server_receive, server_send, client_receive)

    async def ready(self, delay_seconds: float) -> float:
        result = await self._call("playback.ready", {"delaySeconds": delay_seconds})
        value = result.get("startAt")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ProtocolError("playback.ready acknowledgement has no numeric startAt")
        return float(value)

    async def progress(self, video_time: float, server_time: float | None = None) -> None:
        data: dict[str, object] = {"videoTime": video_time}
        if server_time is not None:
            data["serverTime"] = server_time
        result = await self._call("playback.progress", data)
        self.state.apply_ack_scores(result)
        await self._notify("playback.progress", result)

    async def abort(self) -> Session:
        result = await self._call("session.abort", {})
        session = Session.from_dict(object_value(result.get("session"), "aborted session"))
        self.state.reconcile(session)
        await self._notify("session.aborted", object_value(result.get("session"), "aborted session"))
        return session

    async def _call(self, event: str, data: dict[str, object]) -> dict[str, Any]:
        result = await self._raw_call(event, {"sessionID": self.session_id, **data})
        if result.get("ok") is not True:
            self._raise_ack(result)
        return result

    async def _raw_call(self, event: str, payload: dict[str, object]) -> dict[str, Any]:
        try:
            raw: Any = await self._client.call(
                event,
                payload,
                namespace=NAMESPACE,
                timeout=max(1, ceil(self.config.timeout_seconds)),
            )
        except (SocketTimeoutError, TimeoutError) as exc:
            raise ConnectionFailure(f"timed out waiting for {event} acknowledgement") from exc
        return object_value(raw, f"{event} acknowledgement")

    @staticmethod
    def _raise_ack(result: dict[str, Any]) -> None:
        error = result.get("error")
        details = object_value(error, "socket error") if isinstance(error, dict) else {}
        raise ProtocolError(
            f"Socket.IO {details.get('code', 'socket_error')}: {details.get('message', 'Socket.IO operation failed')}"
        )

    async def disconnect(self) -> None:
        if self._client.connected:
            await self._client.disconnect()
        self.state.connected = False


def _number(data: JsonObject, key: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProtocolError(f"clock observation {key} must be numeric")
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ProtocolError(f"clock observation {key} must be finite and non-negative")
    return result


async def probe_socket(config: ClientConfig) -> bool:
    """Connect to `/gameplay` without joining a session, then cleanly disconnect."""
    client = socketio.AsyncClient(reconnection=False, logger=False, engineio_logger=False)
    try:
        await client.connect(
            config.base_url,
            namespaces=[NAMESPACE],
            wait_timeout=max(1, ceil(config.timeout_seconds)),
        )
        return True
    except (SocketConnectionError, SocketTimeoutError, TimeoutError, OSError) as exc:
        raise ConnectionFailure(
            f"cannot connect to Socket.IO namespace {NAMESPACE}",
            hint=f"Verify {config.base_url} is running and supports Socket.IO.",
        ) from exc
    finally:
        if client.connected:
            await client.disconnect()
