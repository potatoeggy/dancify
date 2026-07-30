from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from dancify.domain import SessionState
from dancify.terminal.config import ClientConfig
from dancify.terminal.controller import HeadlessController, LiveStatus
from dancify.terminal.dto import CalibrationResult, Routine, Score, Session
from dancify.terminal.errors import PlaybackError
from dancify.terminal.motion import BoundedMotionUploader, GeneratedMotionSource
from dancify.terminal.playback import (
    DeterministicClock,
    MpvJsonIpcPlayer,
    PlaybackMode,
    PlaybackTimeline,
)
from dancify.terminal.reducer import GameplayState


class UploadAPI:
    def __init__(self) -> None:
        self.batches: list[int] = []

    async def upload_motion(self, _: str, features: list[dict[str, object]]) -> int:
        self.batches.append(len(features))
        return len(features)


def session(state: SessionState = SessionState.COMPLETED) -> Session:
    return Session("s", "r", "p", state, 0.0, 2.0, 1, 95.0, 4)


class FakeAPI(UploadAPI):
    async def get_session(self, _: str) -> Session:
        return session()

    async def import_routine(self, _: dict[str, Any]) -> Routine:
        return Routine("r", "Demo", "generated://demo", 2.0, 30.0, 1)

    async def create_session(self, *_: str) -> Session:
        return session(SessionState.CREATED)

    async def calibrate(self, *_: Any) -> CalibrationResult:
        return CalibrationResult(0.0, 1.0)


class FakeConnection:
    def __init__(self, state: GameplayState) -> None:
        self.state = state
        self.positions: list[float] = []

    async def __aenter__(self) -> FakeConnection:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def ready(self, _: float) -> float:
        return 10.0

    async def progress(self, video_time: float, _: float | None = None) -> None:
        self.positions.append(video_time)
        if video_time >= 1 and 0 not in self.state.scores:
            self.state.scores[0] = Score(0, 0.0, 95.0, 95.0, True)


def test_motion_uploader_timeline_and_controller() -> None:
    async def scenario() -> None:
        upload = UploadAPI()
        accepted = await BoundedMotionUploader(upload, batch_size=30, queue_size=1).upload(
            "s", GeneratedMotionSource(1.0)
        )
        assert accepted == 100
        assert upload.batches == [30, 30, 30, 10]

        deterministic = [tick async for tick in PlaybackTimeline(PlaybackMode.DETERMINISTIC, 1, 10, 0)]
        assert deterministic[0].server_time == 10
        assert deterministic[-1].video_time == 1

        clock = DeterministicClock()
        honest = [
            tick
            async for tick in PlaybackTimeline(
                PlaybackMode.HONEST,
                0.2,
                10,
                0.1,
                clock=clock,
                sleep=clock.sleep,
            )
        ]
        assert honest[-1].server_time is None

        api = FakeAPI()
        connections: list[FakeConnection] = []

        def factory(_: ClientConfig, __: str, state: GameplayState) -> FakeConnection:
            connection = FakeConnection(state)
            connections.append(connection)
            return connection

        controller = HeadlessController(cast(Any, api), ClientConfig("http://test"), factory)
        statuses: list[LiveStatus] = []
        result = await controller.run_session(
            "s",
            1.0,
            GeneratedMotionSource(1.0),
            mode=PlaybackMode.DETERMINISTIC,
            update=statuses.append,
        )
        assert result.session.state is SessionState.COMPLETED
        assert result.accepted_features == 100 and result.scores[0].value == 95
        assert any(status.state is not None and status.state.scores for status in statuses)
        demo = await controller.demo(duration=2.0, mode=PlaybackMode.DETERMINISTIC)
        assert demo.routine.id == "r" and demo.run.accepted_features == 200

    asyncio.run(scenario())


def fake_mpv_script(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import asyncio, json, os, sys
socket_path = next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('--input-ipc-server='))
stop = asyncio.Event()
async def handle(reader, writer):
    while line := await reader.readline():
        request = json.loads(line)
        command = request['command'][0]
        tail = request['command'][1:]
        if tail == ['duration']:
            data = 2.0
        elif tail == ['time-pos']:
            data = 0.5
        else:
            data = None
        response = {'request_id': request['request_id'], 'error': 'success', 'data': data}
        writer.write((json.dumps(response) + '\\n').encode())
        await writer.drain()
        if command == 'quit':
            stop.set()
            break
    writer.close()
async def main():
    try: os.unlink(socket_path)
    except FileNotFoundError: pass
    server = await asyncio.start_unix_server(handle, path=socket_path)
    await stop.wait()
    server.close()
    await server.wait_closed()
asyncio.run(main())
"""
    )
    path.chmod(path.stat().st_mode | 0o111)


def test_mpv_json_ipc_and_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        executable = tmp_path / "fake-mpv"
        media = tmp_path / "clip.mp4"
        media.write_bytes(b"demo")
        fake_mpv_script(executable)
        player = MpvJsonIpcPlayer(str(media), executable=str(executable), video=False)
        await player.prepare()
        assert "--vid=no" in player.command
        await player.play()
        assert await player.position() == 0.5
        temp_dir = player._temp_dir
        await player.close()
        assert temp_dir is not None and not temp_dir.exists()
        with pytest.raises(PlaybackError, match="does not exist"):
            await MpvJsonIpcPlayer(str(tmp_path / "missing.mp4")).prepare()

    asyncio.run(scenario())


def test_mpv_position_tolerates_transient_and_eof_property_unavailability() -> None:
    async def scenario() -> None:
        class ScriptedPlayer(MpvJsonIpcPlayer):
            def __init__(self) -> None:
                super().__init__("unused")
                self._duration = 2.0
                self._time_responses: list[float | PlaybackError] = [
                    PlaybackError("mpv command failed: property unavailable"),
                    0.5,
                    PlaybackError("mpv command failed: property unavailable"),
                ]
                self._idle_responses = iter((False, True))

            async def _command(self, *arguments: object) -> Any:
                if arguments == ("get_property", "time-pos"):
                    response = self._time_responses.pop(0)
                    if isinstance(response, PlaybackError):
                        raise response
                    return response
                if arguments == ("get_property", "idle-active"):
                    return next(self._idle_responses)
                raise AssertionError(arguments)

        player = ScriptedPlayer()
        assert player.media_duration == 2.0
        assert await player.position() == 0.0
        assert await player.position() == 0.5
        assert await player.position() == 2.0

    asyncio.run(scenario())
