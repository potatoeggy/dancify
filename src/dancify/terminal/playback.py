"""Playback clocks and optional external mpv JSON IPC integration."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

from dancify.terminal.errors import PlaybackError


class PlaybackMode(StrEnum):
    HONEST = "honest"
    DETERMINISTIC = "deterministic"


@dataclass(slots=True)
class DeterministicClock:
    """Manually advanced monotonic clock for deterministic runs and tests."""

    current: float = 0.0

    def __call__(self) -> float:
        return self.current

    async def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep duration must be non-negative")
        self.current += seconds


@dataclass(frozen=True, slots=True)
class PlaybackTick:
    video_time: float
    server_time: float | None


class PlaybackTimeline:
    def __init__(
        self,
        mode: PlaybackMode,
        duration: float,
        start_at: float,
        delay_seconds: float,
        *,
        progress_hz: int = 10,
        clock: Callable[[], float] = monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if duration <= 0 or delay_seconds < 0 or progress_hz != 10:
            raise ValueError("duration must be positive, delay non-negative, and progress exactly 10 Hz")
        self.mode = mode
        self.duration = duration
        self.start_at = start_at
        self.delay_seconds = delay_seconds
        self.progress_hz = progress_hz
        self._clock = clock
        self._sleep = sleep

    def __aiter__(self) -> AsyncIterator[PlaybackTick]:
        return self._ticks()

    async def _ticks(self) -> AsyncIterator[PlaybackTick]:
        steps = ceil(self.duration * self.progress_hz)
        if self.mode is PlaybackMode.DETERMINISTIC:
            for step in range(steps + 1):
                video_time = min(self.duration, step / self.progress_hz)
                yield PlaybackTick(video_time, self.start_at + video_time)
                await self._sleep(0)
            return

        await self._sleep(self.delay_seconds)
        local_start = self._clock()
        for step in range(steps + 1):
            target = min(self.duration, step / self.progress_hz)
            remaining = target - (self._clock() - local_start)
            if remaining > 0:
                await self._sleep(remaining)
            elapsed = self._clock() - local_start
            yield PlaybackTick(min(self.duration, max(target, elapsed)), None)


class PlaybackPort(Protocol):
    async def prepare(self) -> None: ...
    async def play(self) -> None: ...
    async def position(self) -> float: ...
    async def close(self) -> None: ...


class MpvJsonIpcPlayer:
    """Control a separate mpv process through newline-delimited JSON IPC."""

    def __init__(
        self,
        source: str,
        *,
        executable: str = "mpv",
        video: bool = True,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not source.strip():
            raise PlaybackError("media source is required")
        if timeout_seconds <= 0:
            raise PlaybackError("mpv timeout must be positive")
        self.source = source
        self.executable = executable
        self.video = video
        self.timeout_seconds = timeout_seconds
        self._temp_dir: Path | None = None
        self._ipc_path: Path | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id = 0
        self._duration: float | None = None
        self._position = 0.0
        self._lock = asyncio.Lock()

    @property
    def media_duration(self) -> float | None:
        return self._duration

    @property
    def command(self) -> tuple[str, ...]:
        if self._ipc_path is None:
            raise PlaybackError("mpv player has not been prepared")
        return (
            self.executable,
            "--idle=yes",
            "--pause=yes",
            "--no-terminal",
            f"--input-ipc-server={self._ipc_path}",
            "--force-window=yes" if self.video else "--vid=no",
            self.source,
        )

    async def prepare(self) -> None:
        if self._process is not None:
            raise PlaybackError("mpv player is already prepared")
        parsed = urlsplit(self.source)
        if not parsed.scheme and not Path(self.source).exists():
            raise PlaybackError(f"media file does not exist: {self.source}")
        self._temp_dir = Path(tempfile.mkdtemp(prefix="dancify-mpv-"))
        self._ipc_path = self._temp_dir / "ipc.sock"
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError) as exc:
            await self.close()
            raise PlaybackError(f"cannot start {self.executable}", hint="Install mpv or use --player clock.") from exc
        deadline = monotonic() + self.timeout_seconds
        while not self._ipc_path.exists():
            if self._process.returncode is not None:
                await self.close()
                raise PlaybackError("mpv exited before opening its IPC socket")
            if monotonic() >= deadline:
                await self.close()
                raise PlaybackError("timed out waiting for mpv IPC socket")
            await asyncio.sleep(0.02)
        try:
            assert self._ipc_path is not None
            self._reader, self._writer = await asyncio.open_unix_connection(str(self._ipc_path))
            media_deadline = monotonic() + self.timeout_seconds
            while True:
                try:
                    duration = await self._command("get_property", "duration")
                    if isinstance(duration, bool) or not isinstance(duration, int | float) or duration <= 0:
                        raise PlaybackError("mpv returned an invalid media duration")
                    self._duration = float(duration)
                    break
                except PlaybackError as exc:
                    if not _property_unavailable(exc) or monotonic() >= media_deadline:
                        raise
                    await asyncio.sleep(0.02)
        except (TimeoutError, OSError, PlaybackError) as exc:
            await self.close()
            if isinstance(exc, PlaybackError):
                raise
            raise PlaybackError("could not connect to mpv JSON IPC") from exc

    async def play(self) -> None:
        await self._command("set_property", "pause", False)

    async def position(self) -> float:
        try:
            value = await self._command("get_property", "time-pos")
        except PlaybackError as exc:
            if not _property_unavailable(exc):
                raise
            try:
                idle = await self._command("get_property", "idle-active")
            except PlaybackError as idle_error:
                if not _property_unavailable(idle_error):
                    raise
                idle = False
            if idle is True and self._duration is not None:
                self._position = self._duration
            return self._position
        if value is None:
            self._position = self._duration or self._position
            return self._position
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PlaybackError("mpv returned a non-numeric time-pos")
        self._position = max(0.0, float(value))
        return self._position

    async def close(self) -> None:
        if self._writer is not None:
            with suppress(TimeoutError, PlaybackError, OSError):
                await self._command("quit")
            self._writer.close()
            with suppress(OSError):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None
        if self._process is not None and self._process.returncode is None:
            try:
                await asyncio.wait_for(self._process.wait(), timeout=1.0)
            except TimeoutError:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=1.0)
                except TimeoutError:
                    self._process.kill()
                    await self._process.wait()
        self._process = None
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None
        self._ipc_path = None
        self._duration = None
        self._position = 0.0

    async def _command(self, *arguments: object) -> Any:
        if self._reader is None or self._writer is None:
            raise PlaybackError("mpv IPC is not connected")
        async with self._lock:
            self._request_id += 1
            request_id = self._request_id
            request = {"command": list(arguments), "request_id": request_id}
            self._writer.write((json.dumps(request) + "\n").encode())
            await self._writer.drain()
            while True:
                line = await asyncio.wait_for(self._reader.readline(), self.timeout_seconds)
                if not line:
                    raise PlaybackError("mpv IPC closed unexpectedly")
                try:
                    raw: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PlaybackError("mpv returned invalid JSON") from exc
                if not isinstance(raw, dict):
                    continue
                response = cast(dict[str, Any], raw)
                if response.get("request_id") != request_id:
                    continue
                if response.get("error") != "success":
                    raise PlaybackError(f"mpv command failed: {response.get('error', 'unknown error')}")
                return response.get("data")


def _property_unavailable(error: PlaybackError) -> bool:
    return "property unavailable" in error.message.casefold()
