"""Headless deterministic-demo and real live-gameplay orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import ceil
from types import TracebackType
from typing import Protocol, Self, cast

from dancify.domain import SessionState
from dancify.terminal.capture import CaptureError, DSUCapture
from dancify.terminal.config import ClientConfig
from dancify.terminal.demo_data import calibration_payload, routine_payload
from dancify.terminal.dto import ClockObservation, JsonObject, MotionHealth, RawUploadResult, Routine, Score, Session
from dancify.terminal.errors import GameplayAborted
from dancify.terminal.motion import (
    BoundedMotionUploader,
    GeneratedMotionSource,
    MotionSource,
    RawCaptureUploader,
    RawStreamResult,
)
from dancify.terminal.playback import PlaybackMode, PlaybackPort, PlaybackTimeline
from dancify.terminal.reducer import GameplayState
from dancify.terminal.rest import DancifyAPI
from dancify.terminal.socket import GameplaySocket


class GameplayConnection(Protocol):
    state: GameplayState

    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None: ...
    async def observe_clock(self) -> ClockObservation: ...
    async def ready(self, delay_seconds: float) -> float: ...
    async def progress(self, video_time: float, server_time: float | None = None) -> None: ...


SocketFactory = Callable[[ClientConfig, str, GameplayState], GameplayConnection]


@dataclass(frozen=True, slots=True)
class RunResult:
    session: Session
    accepted_features: int
    scores: tuple[Score, ...]
    dropped_samples: int = 0
    motion_health: MotionHealth | None = None


@dataclass(frozen=True, slots=True)
class DemoResult:
    routine: Routine
    run: RunResult


@dataclass(frozen=True, slots=True)
class LiveStatus:
    phase: str
    message: str
    video_time: float = 0.0
    state: GameplayState | None = None
    capture_healthy: bool | None = None
    accepted: int | None = None
    dropped: int | None = None


type LiveCallback = Callable[[LiveStatus], Awaitable[None] | None]


class HeadlessController:
    """Coordinate adapters while keeping deterministic demo and live paths explicit."""

    def __init__(self, api: DancifyAPI, config: ClientConfig, socket_factory: SocketFactory | None = None) -> None:
        self._api = api
        self._config = config
        self._socket_factory = socket_factory or _socket_factory

    async def run_session(
        self,
        session_id: str,
        duration: float,
        source: MotionSource,
        *,
        mode: PlaybackMode,
        delay_seconds: float = 1.0,
    ) -> RunResult:
        """Run an already-calibrated feature session; used by explicit deterministic mode."""

        if duration <= 0:
            raise ValueError("duration must be positive")
        state = GameplayState()
        connection = self._socket_factory(self._config, session_id, state)
        async with connection:
            effective_delay = 0.0 if mode is PlaybackMode.DETERMINISTIC else delay_seconds
            start_at = await connection.ready(effective_delay)
            uploader = BoundedMotionUploader(
                self._api,
                batch_size=self._config.motion_batch_size,
                queue_size=self._config.motion_queue_size,
            )
            # Deterministic data is intentionally uploaded before synthetic time
            # advances, preserving repeatable scoring independent of scheduling.
            accepted = await uploader.upload(session_id, source)
            timeline = PlaybackTimeline(
                mode,
                duration,
                start_at,
                effective_delay,
                progress_hz=self._config.progress_hz,
            )
            async for tick in timeline:
                await connection.progress(tick.video_time, tick.server_time)
        return await self._reconcile(session_id, state, accepted)

    async def run_with_player(
        self,
        session_id: str,
        duration: float,
        source: MotionSource,
        player: PlaybackPort,
        *,
        delay_seconds: float = 2.0,
    ) -> RunResult:
        """Legacy feature-source playback using mpv's actual position."""
        if duration <= 0 or delay_seconds < 0:
            raise ValueError("duration must be positive and delay non-negative")
        state = GameplayState()
        connection = self._socket_factory(self._config, session_id, state)
        accepted = 0
        result: RunResult | None = None
        primary: BaseException | None = None
        primary_traceback: TracebackType | None = None
        try:
            await player.prepare()
            await connection.__aenter__()
            await connection.ready(delay_seconds)
            uploader = BoundedMotionUploader(
                self._api,
                batch_size=self._config.motion_batch_size,
                queue_size=self._config.motion_queue_size,
            )
            accepted = await uploader.upload(session_id, source)
            await asyncio.sleep(delay_seconds)
            await player.play()
            interval = 1.0 / self._config.progress_hz
            last_position = -1.0
            while True:
                position = min(duration, await player.position())
                if position > last_position:
                    await connection.progress(position, None)
                    last_position = position
                if position >= duration:
                    break
                await asyncio.sleep(interval)
            result = await self._reconcile(session_id, state, accepted)
        except BaseException as exc:
            primary = exc
            primary_traceback = exc.__traceback__

        async def disconnect() -> None:
            await connection.__aexit__(
                None if primary is None else type(primary),
                primary,
                primary_traceback,
            )

        cleanup_errors = await _cleanup_independently((disconnect, player.close))
        primary, primary_traceback = _merge_failures(primary, primary_traceback, cleanup_errors)
        if primary is not None:
            abort_error = await _abort_nonterminal(self._api, session_id)
            if abort_error is not None:
                primary.add_note(f"backend abort also failed: {abort_error}")
            raise primary.with_traceback(primary_traceback)
        assert result is not None
        return result

    async def run_live(
        self,
        session_id: str,
        duration: float,
        capture: DSUCapture,
        player: PlaybackPort,
        *,
        delay_seconds: float = 2.0,
        cancel: asyncio.Event | None = None,
        update: LiveCallback | None = None,
    ) -> RunResult:
        """Run real capture, raw uploads, mpv heartbeats, and health monitoring concurrently."""

        if duration <= 0 or delay_seconds < 0:
            raise ValueError("duration must be positive and delay non-negative")
        cancellation = cancel or asyncio.Event()
        stop = asyncio.Event()
        upload_started = asyncio.Event()
        upload_finished = asyncio.Event()
        state = GameplayState()

        async def socket_observer(event: str, _payload: JsonObject, current: GameplayState) -> None:
            await _emit(update, LiveStatus("event", event, state=current))

        connection = self._socket_factory(self._config, session_id, state)
        if isinstance(connection, GameplaySocket):
            connection.add_observer(socket_observer)
        stream_result = RawStreamResult(0, 0, 0, None)
        result: RunResult | None = None
        primary: BaseException | None = None
        primary_traceback: TracebackType | None = None
        try:
            await player.prepare()
            if not capture.running:
                await capture.start()
            await _emit(
                update,
                LiveStatus("ready", "Controllers and media ready", state=state, capture_healthy=capture.health.healthy),
            )
            await connection.__aenter__()
            observation = await connection.observe_clock()
            start_at = await connection.ready(delay_seconds)
            local_start = observation.to_client_time(start_at)

            async def upload_motion() -> RawStreamResult:
                await upload_started.wait()

                async def uploaded(upload: RawUploadResult) -> None:
                    state.motion_health = upload.motion_health
                    await _emit(
                        update,
                        LiveStatus(
                            "motion",
                            "Raw motion uploaded",
                            state=state,
                            capture_healthy=capture.health.healthy,
                            accepted=upload.motion_health.accepted,
                            dropped=upload.motion_health.dropped,
                        ),
                    )

                uploader = RawCaptureUploader(self._api, batch_size=self._config.motion_batch_size)
                try:
                    return await uploader.stream(session_id, capture, stop, on_result=uploaded)
                finally:
                    upload_finished.set()

            async def playback() -> None:
                await _wait_for_start(local_start, cancellation)
                if cancellation.is_set():
                    raise GameplayAborted("gameplay aborted by operator")
                await _drain_capture(capture)
                await player.play()
                upload_started.set()
                interval = 1.0 / self._config.progress_hz
                loop = asyncio.get_running_loop()
                next_heartbeat = loop.time()
                last_position = -1.0
                while True:
                    await _wait_for_start(next_heartbeat, cancellation)
                    if cancellation.is_set():
                        raise GameplayAborted("gameplay aborted by operator")
                    position = min(duration, await player.position())
                    advanced = position > last_position
                    if position >= duration:
                        stop.set()
                        await upload_finished.wait()
                    await connection.progress(position, None)
                    if advanced:
                        last_position = position
                        await _emit(
                            update,
                            LiveStatus(
                                "playback",
                                "Playing",
                                video_time=position,
                                state=state,
                                capture_healthy=capture.health.healthy,
                            ),
                        )
                    if position >= duration:
                        return
                    next_heartbeat += interval
                    now = loop.time()
                    if next_heartbeat <= now:
                        next_heartbeat += max(1, ceil((now - next_heartbeat) / interval)) * interval

            async def monitor() -> None:
                stale_grace = asyncio.get_running_loop().time() + capture.config.stale_after
                while not stop.is_set():
                    if cancellation.is_set():
                        raise GameplayAborted("gameplay aborted by operator")
                    health = capture.health
                    stale_is_failure = asyncio.get_running_loop().time() >= stale_grace
                    failed = [slot for slot in health.slots if not slot.connected or (stale_is_failure and slot.stale)]
                    if not health.running or failed:
                        labels = ", ".join(f"{slot.wrist.value} slot {slot.slot}" for slot in failed)
                        raise CaptureError(f"DSU device stream is stale or disconnected: {labels or 'capture stopped'}")
                    await _emit(update, LiveStatus("health", "Controllers healthy", state=state, capture_healthy=True))
                    await asyncio.sleep(min(0.25, capture.config.stale_after / 2))

            try:
                async with asyncio.TaskGroup() as group:
                    upload_task = group.create_task(upload_motion())
                    group.create_task(playback())
                    group.create_task(monitor())
            except BaseExceptionGroup as exc:
                raise _first_exception(exc) from None
            stream_result = upload_task.result()
            result = await self._reconcile(
                session_id,
                state,
                stream_result.accepted,
                stream_result.dropped,
                stream_result.motion_health,
            )
            await _emit(
                update,
                LiveStatus(
                    "complete",
                    "Final backend state reconciled",
                    video_time=result.session.current_timestamp,
                    state=state,
                    accepted=result.accepted_features,
                    dropped=result.dropped_samples,
                ),
            )
        except BaseException as exc:
            primary = exc
            primary_traceback = exc.__traceback__
        finally:
            stop.set()

        async def disconnect() -> None:
            await connection.__aexit__(
                None if primary is None else type(primary),
                primary,
                primary_traceback,
            )

        cleanup_errors = await _cleanup_independently((disconnect, player.close, capture.stop))
        primary, primary_traceback = _merge_failures(primary, primary_traceback, cleanup_errors)
        if primary is not None:
            abort_error = await _abort_nonterminal(self._api, session_id)
            if abort_error is not None:
                primary.add_note(f"backend abort also failed: {abort_error}")
            raise primary.with_traceback(primary_traceback)
        assert result is not None
        return result

    async def demo(
        self,
        *,
        duration: float = 2.0,
        mode: PlaybackMode,
        player_id: str = "terminal-demo",
    ) -> DemoResult:
        """Explicit synthetic fixture workflow for CI and repeatable demonstrations."""

        routine = await self._api.import_routine(routine_payload(duration))
        session = await self._api.create_session(routine.id, player_id)
        await self._api.calibrate(session.id, calibration_payload())
        run = await self.run_session(
            session.id,
            routine.duration,
            GeneratedMotionSource(routine.duration),
            mode=mode,
        )
        return DemoResult(routine, run)

    async def _reconcile(
        self,
        session_id: str,
        state: GameplayState,
        accepted: int,
        dropped: int = 0,
        health: MotionHealth | None = None,
    ) -> RunResult:
        session = await self._api.get_session(session_id)
        state.reconcile(session)
        return RunResult(
            session,
            accepted,
            tuple(state.scores[index] for index in sorted(state.scores)),
            dropped,
            session.motion_health or health or state.motion_health,
        )


async def _emit(callback: LiveCallback | None, status: LiveStatus) -> None:
    if callback is None:
        return
    result = callback(status)
    if inspect.isawaitable(result):
        await result


async def _wait_for_start(deadline: float, cancellation: asyncio.Event) -> None:
    while True:
        if cancellation.is_set():
            return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(cancellation.wait(), min(0.1, remaining))


async def _drain_capture(capture: DSUCapture) -> None:
    for _ in range(capture.health.queue_depth):
        try:
            await capture.receive(0.001)
        except (TimeoutError, StopAsyncIteration):
            return


async def _cleanup_independently(actions: tuple[Callable[[], Awaitable[None]], ...]) -> list[BaseException]:
    failures: list[BaseException] = []
    for action in actions:
        try:
            await action()
        except BaseException as exc:
            failures.append(exc)
    return failures


def _merge_failures(
    primary: BaseException | None,
    traceback: TracebackType | None,
    cleanup_errors: list[BaseException],
) -> tuple[BaseException | None, TracebackType | None]:
    if primary is None and cleanup_errors:
        primary = cleanup_errors.pop(0)
        traceback = primary.__traceback__
    if primary is not None:
        for error in cleanup_errors:
            primary.add_note(f"additional cleanup failure: {error}")
    return primary, traceback


async def _abort_nonterminal(api: DancifyAPI, session_id: str) -> BaseException | None:
    try:
        snapshot = await api.get_session(session_id)
    except BaseException:
        try:
            await api.abort(session_id)
        except BaseException as abort_error:
            return abort_error
        return None
    if snapshot.state in {SessionState.COMPLETED, SessionState.ABORTED}:
        return None
    try:
        await api.abort(session_id)
    except BaseException as exc:
        return exc
    return None


def _first_exception(group: BaseExceptionGroup[BaseException]) -> BaseException:
    first = group.exceptions[0]
    if isinstance(first, BaseExceptionGroup):
        return _first_exception(cast(BaseExceptionGroup[BaseException], first))
    return first


def _socket_factory(config: ClientConfig, session_id: str, state: GameplayState) -> GameplayConnection:
    return GameplaySocket(config, session_id, state)
