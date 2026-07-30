"""Typer command surface for the real terminal frontend and deterministic demo."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Coroutine
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer

from dancify.terminal.calibration import CalibrationStatus, GuidedCalibrator
from dancify.terminal.capture import DSUCapture
from dancify.terminal.config import ClientConfig
from dancify.terminal.controller import HeadlessController, LiveStatus
from dancify.terminal.dto import Score, object_value
from dancify.terminal.errors import ClientError, ConfigurationError, ExitCode
from dancify.terminal.motion import GeneratedMotionSource, ListMotionSource
from dancify.terminal.playback import MpvJsonIpcPlayer, PlaybackMode
from dancify.terminal.rest import DancifyAPI
from dancify.terminal.socket import GameplaySocket, probe_socket
from dancify.terminal.tui import DancifyTerminalApp, score_cue

app = typer.Typer(no_args_is_help=True, help="Operate Dancify over REST, Socket.IO, DSU, and mpv.")
routine_app = typer.Typer(no_args_is_help=True, help="Import and inspect routines.")
session_app = typer.Typer(no_args_is_help=True, help="Create and manage gameplay sessions.")
app.add_typer(routine_app, name="routine")
app.add_typer(session_app, name="session")


class PlayerChoice(str, Enum):
    CLOCK = "clock"
    MPV = "mpv"


@dataclass(frozen=True, slots=True)
class CLISettings:
    config: ClientConfig


class _LiveScoreOutput:
    def __init__(self) -> None:
        self._printed_windows: set[int] = set()

    def __call__(self, status: LiveStatus) -> None:
        if status.state is None:
            return
        for window_index in sorted(status.state.scores):
            if window_index in self._printed_windows:
                continue
            typer.echo(_score_line(status.state.scores[window_index]), err=True)
            self._printed_windows.add(window_index)


def _score_line(score: Score) -> str:
    cue = score_cue(score.value, score.valid)
    no_data = " (no data)" if not score.valid else ""
    start = _score_time(score.window_start_seconds)
    end = _score_time(score.window_start_seconds + 1.0)
    breakdown = score.breakdown
    if breakdown is None:
        categories = "direction n/a | magnitude n/a | timing n/a | quality n/a"
    else:
        categories = (
            f"direction {breakdown.direction:.0%} | magnitude {breakdown.magnitude:.0%} "
            f"| timing {breakdown.timing:.0%} | quality {breakdown.quality:.0%}"
        )
    return f"[{start}-{end}] {cue}{no_data} {score.value:.1f} | {categories} | avg {score.cumulative_score:.1f}"


def _score_time(seconds: float) -> str:
    minutes, remaining = divmod(round(seconds), 60)
    return f"{minutes:02d}:{remaining:02d}"


@app.callback()
def configure(
    ctx: typer.Context,
    url: Annotated[str | None, typer.Option("--url", help="Backend base URL (or DANCIFY_URL).")] = None,
    timeout: Annotated[float | None, typer.Option("--timeout", help="Request timeout in seconds.")] = None,
    dsu_host: Annotated[str | None, typer.Option("--dsu-host", help="Cemuhook/DSU host.")] = None,
    dsu_port: Annotated[int | None, typer.Option("--dsu-port", min=1, max=65535)] = None,
    left_slot: Annotated[
        int | None,
        typer.Option("--left-slot", min=0, max=3, help="Left DSU slot; omit for right-handed mode."),
    ] = None,
    right_slot: Annotated[
        int | None, typer.Option("--right-slot", min=0, max=3, help="Required right-wrist DSU slot.")
    ] = None,
) -> None:
    """Configure backend and DSU wrist assignment."""

    ctx.obj = CLISettings(
        ClientConfig.from_env(
            base_url=url,
            timeout_seconds=timeout,
            dsu_host=dsu_host,
            dsu_port=dsu_port,
            dsu_left_slot=left_slot,
            dsu_right_slot=right_slot,
        )
    )


@app.command()
def discover(ctx: typer.Context) -> None:
    """Discover and verify every configured DSU controller slot."""

    async def operation() -> object:
        capture = DSUCapture(_settings(ctx).config.capture_config)
        try:
            await capture.start()
            return _capture_report(capture)
        finally:
            await capture.stop()

    _output(_run(operation()))


@app.command()
def doctor(
    ctx: typer.Context,
    player: Annotated[PlayerChoice, typer.Option("--player")] = PlayerChoice.CLOCK,
    capture: Annotated[bool, typer.Option("--capture", help="Also require all configured DSU slots.")] = False,
) -> None:
    """Check backend, Socket.IO, optional mpv, and optionally DSU readiness."""

    settings = _settings(ctx)

    async def operation() -> dict[str, object]:
        async with DancifyAPI(settings.config) as api:
            healthy = await api.health()
        socket_connected = await probe_socket(settings.config)
        mpv_path = shutil.which("mpv")
        capture_report: object = "not requested"
        capture_ok = True
        if capture:
            dsu = DSUCapture(settings.config.capture_config)
            try:
                await dsu.start()
                capture_report = _capture_report(dsu)
                capture_ok = dsu.health.healthy
            finally:
                await dsu.stop()
        return {
            "ok": healthy
            and socket_connected
            and capture_ok
            and (player is PlayerChoice.CLOCK or mpv_path is not None),
            "apiURL": settings.config.api_url,
            "socketNamespace": "/gameplay",
            "socketConnected": socket_connected,
            "player": player.value,
            "mpvAvailable": mpv_path is not None,
            "dsu": capture_report,
        }

    _output(_run(operation()))


@routine_app.command("import")
def routine_import(ctx: typer.Context, path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Import a video-analysis JSON payload."""
    payload = _json_file(path, "routine")

    async def operation() -> object:
        async with DancifyAPI(_settings(ctx).config) as api:
            return await api.import_routine(payload)

    _output(_run(operation()))


@routine_app.command("show")
def routine_show(ctx: typer.Context, routine_id: str) -> None:
    """Show routine metadata."""

    async def operation() -> object:
        async with DancifyAPI(_settings(ctx).config) as api:
            return await api.get_routine(routine_id)

    _output(_run(operation()))


@routine_app.command("windows")
def routine_windows(ctx: typer.Context, routine_id: str) -> None:
    """List generated score windows."""

    async def operation() -> object:
        async with DancifyAPI(_settings(ctx).config) as api:
            return await api.get_windows(routine_id)

    _output(_run(operation()))


@session_app.command("create")
def session_create(
    ctx: typer.Context,
    routine_id: str,
    player_id: Annotated[str, typer.Option("--player", help="Player identifier.")] = "terminal-player",
    scoring_algorithm: Annotated[str, typer.Option("--scorer")] = "weighted_dtw",
) -> None:
    """Create a gameplay session."""

    async def operation() -> object:
        async with DancifyAPI(_settings(ctx).config) as api:
            return await api.create_session(routine_id, player_id, scoring_algorithm)

    _output(_run(operation()))


@session_app.command("show")
def session_show(ctx: typer.Context, session_id: str) -> None:
    """Show the authoritative session snapshot."""

    async def operation() -> object:
        async with DancifyAPI(_settings(ctx).config) as api:
            return await api.get_session(session_id)

    _output(_run(operation()))


@session_app.command("calibrate")
def session_calibrate(
    ctx: typer.Context,
    session_id: str,
    path: Annotated[Path | None, typer.Option("--input", exists=True, dir_okay=False)] = None,
    samples: Annotated[int, typer.Option("--samples", min=2)] = 20,
    countdown: Annotated[int, typer.Option("--countdown", min=0)] = 3,
) -> None:
    """Measure configured wrists, or explicitly submit a supplied calibration JSON."""

    settings = _settings(ctx)

    async def operation() -> object:
        async with DancifyAPI(settings.config) as api:
            if path is not None:
                return await api.calibrate(session_id, _json_file(path, "calibration"))
            capture = DSUCapture(settings.config.capture_config)
            socket = GameplaySocket(settings.config, session_id)

            async def report(status: CalibrationStatus) -> None:
                typer.echo(status.message, err=True)

            async with socket:
                return await GuidedCalibrator(
                    api,
                    socket,
                    capture,
                    samples_per_wrist=samples,
                    countdown_seconds=countdown,
                    status=report,
                ).calibrate(session_id)

    _output(_run(operation()))


@session_app.command("retry")
def session_retry(ctx: typer.Context, session_id: str) -> None:
    """Clone an aborted source session into a new session.

    The source must be aborted. Valid calibration is reused when available, while
    gameplay state, scores, and motion are clean. Use the returned ID for `run`.
    """

    async def operation() -> object:
        async with DancifyAPI(_settings(ctx).config) as api:
            return await api.retry(session_id)

    _output(_run(operation()))


@session_app.command("abort")
def session_abort(ctx: typer.Context, session_id: str) -> None:
    """Abort a non-terminal session."""

    async def operation() -> object:
        async with DancifyAPI(_settings(ctx).config) as api:
            return await api.abort(session_id)

    _output(_run(operation()))


@app.command("run")
def run_command(
    ctx: typer.Context,
    session_id: str,
    duration: Annotated[float, typer.Option("--duration", min=0.1)],
    motion: Annotated[Path | None, typer.Option("--motion", exists=True, dir_okay=False)] = None,
    player: Annotated[PlayerChoice, typer.Option("--player")] = PlayerChoice.CLOCK,
    media: Annotated[str | None, typer.Option("--media", help="Local path or URL for mpv.")] = None,
    deterministic: Annotated[bool, typer.Option("--deterministic", help="Use explicit synthetic input/time.")] = False,
    delay: Annotated[float, typer.Option("--delay", min=0.0)] = 1.0,
    audio_only: Annotated[bool, typer.Option("--audio-only")] = False,
) -> None:
    """Run deterministic fixtures or real DSU capture with honest mpv timing."""

    settings = _settings(ctx)
    score_output = _LiveScoreOutput()
    if deterministic and player is PlayerChoice.MPV:
        raise ConfigurationError("mpv cannot be combined with --deterministic")
    if not deterministic and player is not PlayerChoice.MPV:
        raise ConfigurationError("live mode requires --player mpv; use --deterministic for synthetic clock mode")
    if player is PlayerChoice.MPV and media is None:
        raise ConfigurationError("--media is required when --player mpv is selected")

    async def operation() -> object:
        async with DancifyAPI(settings.config) as api:
            controller = HeadlessController(api, settings.config)
            if deterministic:
                source = GeneratedMotionSource(duration) if motion is None else ListMotionSource.from_file(motion)
                return await controller.run_session(
                    session_id,
                    duration,
                    source,
                    mode=PlaybackMode.DETERMINISTIC,
                    delay_seconds=0,
                    update=score_output,
                )
            if motion is not None:
                raise ConfigurationError("--motion is only valid with --deterministic")
            assert media is not None
            return await controller.run_live(
                session_id,
                duration,
                DSUCapture(settings.config.capture_config),
                MpvJsonIpcPlayer(media, video=not audio_only),
                delay_seconds=delay,
                update=score_output,
            )

    _output(_run(operation()))


@app.command()
def demo(
    ctx: typer.Context,
    duration: Annotated[float, typer.Option("--duration", min=0.5)] = 2.0,
    deterministic: Annotated[
        bool, typer.Option("--deterministic", help="Run synthetic time without sleeping.")
    ] = False,
    player_id: Annotated[str, typer.Option("--player")] = "terminal-demo",
) -> None:
    """Run the explicit generated-data demonstration (never used by live mode)."""

    settings = _settings(ctx)
    mode = PlaybackMode.DETERMINISTIC if deterministic else PlaybackMode.HONEST

    async def operation() -> object:
        async with DancifyAPI(settings.config) as api:
            return await HeadlessController(api, settings.config).demo(
                duration=duration, mode=mode, player_id=player_id
            )

    _output(_run(operation()))


@app.command()
def tui(ctx: typer.Context) -> None:
    """Open the complete keyboard-usable terminal operator."""
    DancifyTerminalApp(_settings(ctx).config).run()


def _capture_report(capture: DSUCapture) -> dict[str, object]:
    health = capture.health
    return {
        "host": capture.config.host,
        "port": capture.config.port,
        "healthy": health.healthy,
        "protocolVersion": health.protocol_version,
        "controllers": [
            {
                "wrist": slot.wrist.value,
                "slot": slot.slot,
                "connected": slot.connected,
                "stale": slot.stale,
                "quality": slot.quality,
                "sampleRateHz": slot.sample_rate_hz,
                "identity": identity.mac_address
                if (identity := capture.identities.get(slot.wrist)) is not None
                else None,
            }
            for slot in health.slots
        ],
    }


def _settings(ctx: typer.Context) -> CLISettings:
    if not isinstance(ctx.obj, CLISettings):
        raise ConfigurationError("client configuration was not initialized")
    return ctx.obj


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text())
    except OSError as exc:
        raise ConfigurationError(f"cannot read {label} file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{label} file {path} is not valid JSON: {exc}") from exc
    return object_value(value, f"{label} input")


def _run[T](awaitable: Coroutine[Any, Any, T]) -> T:
    try:
        return asyncio.run(awaitable)
    except KeyboardInterrupt as exc:
        typer.echo("interrupted", err=True)
        raise typer.Exit(int(ExitCode.INTERRUPTED)) from exc
    except ClientError as exc:
        typer.echo(exc.display(), err=True)
        raise typer.Exit(int(exc.exit_code)) from exc


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: getattr(value, item.name)
            for item in fields(value)
            if not (isinstance(value, Score) and item.name == "breakdown")
        }
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _output(value: object) -> None:
    typer.echo(json.dumps(value, default=_json_default, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
