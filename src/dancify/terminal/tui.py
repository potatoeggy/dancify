"""Complete Textual operator for real DSU/mpv gameplay and explicit demo mode."""
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, ClassVar, cast

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Log, Static

from dancify.domain import SessionState, WristSide
from dancify.terminal.calibration import CalibrationStatus, GuidedCalibrator
from dancify.terminal.capture import DSUCapture
from dancify.terminal.config import ClientConfig
from dancify.terminal.controller import DemoResult, HeadlessController, LiveStatus, SocketFactory
from dancify.terminal.dto import Routine, Session, object_value
from dancify.terminal.errors import ClientError, ConfigurationError, GameplayAborted
from dancify.terminal.playback import MpvJsonIpcPlayer, PlaybackMode, PlaybackPort
from dancify.terminal.reducer import GameplayState
from dancify.terminal.rest import DancifyAPI
from dancify.terminal.socket import GameplaySocket

APIFactory = Callable[[ClientConfig], DancifyAPI]
CaptureFactory = Callable[[ClientConfig], DSUCapture]
PlayerFactory = Callable[[str, bool], PlaybackPort]
CalibrationSocketFactory = Callable[[ClientConfig, str, GameplayState], GameplaySocket]


def _default_capture(config: ClientConfig) -> DSUCapture:
    return DSUCapture(config.capture_config)


def _default_player(source: str, video: bool) -> PlaybackPort:
    return MpvJsonIpcPlayer(source, video=video)


def _default_calibration_socket(
    config: ClientConfig,
    session_id: str,
    state: GameplayState,
) -> GameplaySocket:
    return GameplaySocket(config, session_id, state)


def _routine_json(path: Path) -> dict[str, Any]:
    try:
        value: Any = json.loads(path.read_text())
    except OSError as exc:
        raise ConfigurationError(f"cannot read routine file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"routine file {path} is not valid JSON: {exc}") from exc
    return object_value(value, "routine input")


def _wrist_assignment(config: ClientConfig) -> str:
    if config.dsu_left_slot is None:
        return f"R{config.dsu_right_slot} (right-only)"
    return f"L{config.dsu_left_slot}/R{config.dsu_right_slot}"


class SetupScreen(Screen[None]):
    """Single-screen operator flow; every step has a keyboard binding and status."""

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("b", "connect_backend", "Backend"),
        ("r", "load_routine", "Routine"),
        ("d", "detect", "Detect"),
        ("c", "calibrate", "Calibrate"),
        ("g", "go_live", "Go live"),
        Binding("ctrl+x", "abort", "Abort", priority=True),
        ("x", "abort", "Abort"),
        ("q", "app.quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.routine: Routine | None = None
        self.session: Session | None = None
        self.capture: DSUCapture | None = None
        self.cancel = asyncio.Event()
        self._live_active = False
        self._printed_score_windows: set[int] = set()
        self._active_workflow: asyncio.Task[None] | None = None
        self._workflow_pending = False
        self._abort_pending = False

    def compose(self) -> ComposeResult:
        app = cast(DancifyTerminalApp, self.app)
        yield Header()
        with Vertical(id="setup"):
            yield Label("Dancify terminal operator", id="title")
            yield Static(
                f"Backend {app.client_config.base_url} · DSU {app.client_config.dsu_host}:"
                f"{app.client_config.dsu_port} · "
                f"{_wrist_assignment(app.client_config)}",
                id="configuration",
            )
            with Horizontal(classes="fields"):
                yield Input(placeholder="Existing routine ID", id="routine-id")
                yield Input(placeholder="Routine JSON path (imports)", id="routine-json")
                yield Input(value="terminal-player", placeholder="Player ID", id="player-id")
                yield Input(placeholder="Existing session ID (optional)", id="session-id")
            with Horizontal(classes="fields"):
                yield Input(placeholder="Local media path or URL", id="media")
                yield Input(value="2.0", placeholder="Duration seconds", id="duration")
            with Horizontal(id="workflow-buttons"):
                yield Button("1 Backend", id="backend")
                yield Button("2 Routine/session", id="routine")
                yield Button("3 Detect controllers", id="detect")
                yield Button("4 Calibrate", id="calibrate")
                yield Button("5 Ready / play", id="ready", variant="success")
                yield Button("Abort", id="abort", variant="error")
            with Horizontal(id="demo-buttons"):
                yield Button("Explicit deterministic demo", id="run", variant="primary")
                yield Button("Quit", id="quit")
            yield Static("Step 1: check backend, or press Run for the explicit generated demo", id="status")
            yield Static("State: idle", id="state")
            yield Static("Controllers: not detected", id="device-health")
            yield Static("Score: —", id="live-score")
            yield Log(id="log")
        yield Footer()

    def action_connect_backend(self) -> None:
        self._start(self._connect_backend())

    def action_load_routine(self) -> None:
        self._start(self._load_routine())

    def action_detect(self) -> None:
        self._start(self._detect())

    def action_calibrate(self) -> None:
        self._start(self._calibrate())

    def action_go_live(self) -> None:
        self._start(self._go_live(), exclusive=True)

    def action_abort(self) -> None:
        self.abort_workflow()

    @on(Button.Pressed, "#backend")
    def connect_backend(self) -> None:
        self.action_connect_backend()

    @on(Button.Pressed, "#routine")
    def load_routine(self) -> None:
        self.action_load_routine()

    @on(Button.Pressed, "#detect")
    def detect(self) -> None:
        self.action_detect()

    @on(Button.Pressed, "#calibrate")
    def calibrate(self) -> None:
        self.action_calibrate()

    @on(Button.Pressed, "#ready")
    def go_live(self) -> None:
        self.action_go_live()

    @on(Button.Pressed, "#abort")
    def abort_workflow(self) -> None:
        if self._abort_pending:
            return
        self.cancel.set()
        self._abort_pending = True
        self._set_status("Abort requested; waiting for local cleanup before remote abort")
        self.run_worker(self._cancel_active_and_abort())

    async def _cancel_active_and_abort(self) -> None:
        try:
            active = self._active_workflow
            if active is not None and active is not asyncio.current_task() and not active.done():
                active.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await active
            await self._abort_session()
        finally:
            self._abort_pending = False

    @on(Button.Pressed, "#run")
    def run_demo(self) -> None:
        try:
            duration = self._duration(minimum=0.5)
        except ValueError:
            self._set_status("Duration must be at least 0.5 seconds")
            return
        self._set_status("Running explicit deterministic generated-data workflow…")
        self._log("Importing generated routine and creating deterministic session")
        self._start(self._execute_demo(duration), exclusive=True)

    @on(Button.Pressed, "#quit")
    def quit_app(self) -> None:
        self.app.exit()

    async def _connect_backend(self) -> None:
        app = cast(DancifyTerminalApp, self.app)
        try:
            async with app.api_factory(app.client_config) as api:
                healthy = await api.health()
            self._set_status("Backend ready; load/import a routine or existing session")
            self.query_one("#state", Static).update(f"Backend health: {'ok' if healthy else 'failed'}")
            self._log(f"Connected to {app.client_config.api_url}")
        except ClientError as exc:
            self._error(exc)

    async def _load_routine(self) -> None:
        app = cast(DancifyTerminalApp, self.app)
        routine_id = self.query_one("#routine-id", Input).value.strip()
        routine_path = self.query_one("#routine-json", Input).value.strip()
        session_id = self.query_one("#session-id", Input).value.strip()
        player_id = self.query_one("#player-id", Input).value.strip()
        try:
            async with app.api_factory(app.client_config) as api:
                if session_id:
                    self.session = await api.get_session(session_id)
                    self.routine = await api.get_routine(self.session.routine_id)
                else:
                    if routine_path:
                        self.routine = await api.import_routine(_routine_json(Path(routine_path)))
                    elif routine_id:
                        self.routine = await api.get_routine(routine_id)
                    else:
                        raise ConfigurationError("enter an existing routine ID or a routine JSON path")
                    if not player_id:
                        raise ConfigurationError("player ID is required")
                    self.session = await api.create_session(self.routine.id, player_id)
            assert self.routine is not None and self.session is not None
            self.query_one("#routine-id", Input).value = self.routine.id
            self.query_one("#session-id", Input).value = self.session.id
            self.query_one("#duration", Input).value = str(self.routine.duration)
            if not self.routine.source_video_url.startswith("generated://"):
                self.query_one("#media", Input).value = self.routine.source_video_url
            self._state(self.session)
            self._set_status("Routine/session ready; detect configured controller(s)")
            self._log(f"Routine {self.routine.id}; session {self.session.id}")
        except (ClientError, OSError) as exc:
            self._error(exc)

    async def _detect(self) -> None:
        app = cast(DancifyTerminalApp, self.app)
        try:
            if self.capture is not None:
                await self.capture.stop()
            self.capture = app.capture_factory(app.client_config)
            await self.capture.start()
            identities = self.capture.identities
            labels = ", ".join(
                f"{side.value}=slot {app.client_config.capture_config.slots[side]} {identity.mac_address}"
                for side, identity in identities.items()
            )
            self.query_one("#device-health", Static).update(f"Controllers: healthy · {labels}")
            self._set_status("Controllers assigned; perform measured calibration")
            self._log(labels)
        except ClientError as exc:
            self._error(exc)

    async def _calibrate(self) -> None:
        app = cast(DancifyTerminalApp, self.app)
        if self.session is None:
            self._set_status("Load a routine/session before calibration")
            return
        try:
            if self.capture is None or not self.capture.running:
                await self._detect()
            assert self.capture is not None
            capture = self.capture
            configured_slots = capture.config.slots
            state = GameplayState()
            socket = app.calibration_socket_factory(app.client_config, self.session.id, state)

            async def report(status: CalibrationStatus) -> None:
                self._set_status(status.message)
                if status.left_samples or status.right_samples:
                    counts: list[str] = []
                    if WristSide.LEFT in configured_slots:
                        counts.append(f"left {status.left_samples}")
                    if WristSide.RIGHT in configured_slots:
                        counts.append(f"right {status.right_samples}")
                    self.query_one("#device-health", Static).update(f"Calibration {status.stage}: {', '.join(counts)}")

            async with app.api_factory(app.client_config) as api:
                async with socket:
                    result = await GuidedCalibrator(
                        api,
                        socket,
                        capture,
                        cancel=self.cancel,
                        status=report,
                    ).calibrate(self.session.id)
                self.session = await api.get_session(self.session.id)
            self._state(self.session)
            self._set_status(f"Measured calibration ready ({result.horizontal_confidence:.0%}); choose media and play")
        except GameplayAborted:
            raise
        except ClientError as exc:
            self._error(exc)

    async def _go_live(self) -> None:
        app = cast(DancifyTerminalApp, self.app)
        if self.session is None or self.routine is None:
            self._set_status("Load a routine/session before gameplay")
            return
        if self.session.state is not SessionState.READY:
            self._set_status(f"Session is {self.session.state.value}; measured calibration must finish first")
            return
        media = self.query_one("#media", Input).value.strip()
        if not media:
            self._set_status("Media path or URL is required for honest mpv playback")
            return
        try:
            duration = self._duration(minimum=0.1)
        except ValueError:
            self._set_status("Duration must be at least 0.1 seconds")
            return
        try:
            if self.capture is None or not self.capture.running:
                await self._detect()
            assert self.capture is not None
            self.cancel = asyncio.Event()
            self._live_active = True
            self._printed_score_windows = {score.window_index for score in self.session.scores}

            async def update(status: LiveStatus) -> None:
                self._render_live_status(status, duration)

            async with app.api_factory(app.client_config) as api:
                result = await HeadlessController(api, app.client_config, app.socket_factory).run_live(
                    self.session.id,
                    duration,
                    self.capture,
                    app.player_factory(media, True),
                    cancel=self.cancel,
                    update=update,
                )
            self.session = result.session
            await self.app.push_screen(ResultsScreen(DemoResult(self.routine, result)))
        except GameplayAborted:
            raise
        except ClientError as exc:
            self._error(exc)
        finally:
            self._live_active = False

    async def _abort_session(self) -> None:
        app = cast(DancifyTerminalApp, self.app)
        try:
            if self.session is not None:
                async with app.api_factory(app.client_config) as api:
                    self.session = await api.get_session(self.session.id)
                    if self.session.state not in {SessionState.COMPLETED, SessionState.ABORTED}:
                        self.session = await api.abort(self.session.id)
                self._state(self.session)
            self._set_status("Session stopped and devices released")
        except ClientError as exc:
            self._error(exc)
        finally:
            if self.capture is not None:
                try:
                    await self.capture.stop()
                except ClientError as exc:
                    self._error(exc)

    async def _execute_demo(self, duration: float) -> None:
        app = cast(DancifyTerminalApp, self.app)
        try:
            async with app.api_factory(app.client_config) as api:
                result = await HeadlessController(api, app.client_config, app.socket_factory).demo(
                    duration=duration,
                    mode=PlaybackMode.DETERMINISTIC,
                    player_id="terminal-tui",
                )
        except ClientError as exc:
            self._error(exc)
            return
        await app.push_screen(ResultsScreen(result))

    async def on_unmount(self) -> None:
        self.cancel.set()
        active = self._active_workflow
        if active is not None and active is not asyncio.current_task() and not active.done():
            active.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await active
        if self.capture is not None:
            await self.capture.stop()

    def _duration(self, *, minimum: float) -> float:
        value = float(self.query_one("#duration", Input).value)
        if value < minimum:
            raise ValueError
        return value

    def _state(self, session: Session) -> None:
        self.query_one("#state", Static).update(
            f"State: {session.state.value} · window {session.current_window} · score {session.cumulative_score:.3f}"
        )

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _log(self, message: str) -> None:
        self.query_one("#log", Log).write_line(message)

    def _error(self, error: BaseException) -> None:
        message = error.display() if isinstance(error, ClientError) else str(error)
        self._set_status(message)
        self._log(f"ERROR: {message}")

    def _render_live_status(self, status: LiveStatus, duration: float) -> None:
        self._set_status(f"{status.message} · {status.video_time:.1f}/{duration:.1f}s")
        motion_health = None
        if status.state is not None:
            motion_health = status.state.motion_health
            if status.state.session is not None:
                self._state(status.state.session)
                scores = status.state.scores
                if scores:
                    for window_index in sorted(scores):
                        if window_index not in self._printed_score_windows:
                            score = scores[window_index]
                            self._log(
                                f"Score update · window {window_index}: {score.value:.3f} "
                                f"· cumulative {score.cumulative_score:.3f}"
                            )
                            self._printed_score_windows.add(window_index)
                    latest = scores[max(scores)]
                    self.query_one("#live-score", Static).update(
                        f"Score: {latest.value:.3f} · cumulative {latest.cumulative_score:.3f}"
                    )
        accepted = status.accepted if status.accepted is not None else getattr(motion_health, "accepted", None)
        dropped = status.dropped if status.dropped is not None else getattr(motion_health, "dropped", None)
        health_label = "healthy" if status.capture_healthy is not False else "unhealthy"
        details: list[str] = []
        if accepted is not None and dropped is not None:
            details.append(f"accepted {accepted}, dropped {dropped}")
        if motion_health is not None:
            active_wrists = (
                status.state.session.active_wrists
                if status.state is not None and status.state.session is not None
                else tuple(motion_health.wrists)
            )
            for side in active_wrists:
                wrist = motion_health.wrists.get(side)
                if wrist is not None:
                    details.append(
                        f"{side} accepted {wrist.accepted}, dropped {wrist.dropped}, quality {wrist.quality:.0%}"
                    )
        suffix = "" if not details else " · " + " · ".join(details)
        self.query_one("#device-health", Static).update(f"Controllers: {health_label}{suffix}")

    def _start(self, coroutine: Coroutine[Any, Any, None], *, exclusive: bool = False) -> None:
        del exclusive
        active = self._active_workflow
        if self._workflow_pending or self._abort_pending or (active is not None and not active.done()):
            coroutine.close()
            self._set_status("Another workflow is active; abort or wait for it to finish")
            return
        self.cancel = asyncio.Event()
        self._workflow_pending = True
        self.run_worker(self._run_workflow(coroutine))

    async def _run_workflow(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._active_workflow = task
        self._workflow_pending = False
        try:
            await coroutine
        except (GameplayAborted, asyncio.CancelledError):
            if self.is_mounted:
                self._set_status("Workflow stopped by operator")
        finally:
            if self._active_workflow is task:
                self._active_workflow = None


class ResultsScreen(Screen[None]):
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("escape", "app.pop_screen", "Back"),
        ("q", "app.quit", "Quit"),
    ]

    def __init__(self, result: DemoResult) -> None:
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        final = self.result.run.session
        yield Header()
        with Vertical(id="results"):
            yield Label("Gameplay complete", id="result-title")
            yield Static(f"State: {final.state.value}")
            yield Static(f"Routine: {self.result.routine.id}")
            yield Static(f"Session: {final.id}")
            yield Static(f"Accepted samples/features: {self.result.run.accepted_features}")
            yield Static(f"Dropped samples: {self.result.run.dropped_samples}")
            yield Static(f"Cumulative score: {final.cumulative_score:.3f}", id="score")
            for score in self.result.run.scores:
                yield Static(
                    f"Window {score.window_index}: {score.value:.3f} "
                    f"(cumulative {score.cumulative_score:.3f}, valid={score.valid})"
                )
            yield Button("Back", id="back")
        yield Footer()

    @on(Button.Pressed, "#back")
    def back(self) -> None:
        self.app.pop_screen()


class DancifyTerminalApp(App[None]):
    """Dependency-injectable full operator application."""

    CSS = """
    #setup, #results { padding: 1 2; }
    #title, #result-title { text-style: bold; margin-bottom: 1; }
    #status, #state, #device-health, #live-score { margin-top: 1; }
    .fields { height: auto; }
    Input { width: 1fr; min-width: 20; }
    Button { margin-right: 1; }
    #workflow-buttons, #demo-buttons { height: auto; margin-top: 1; }
    Log { height: 1fr; border: round $accent; margin-top: 1; }
    """

    def __init__(
        self,
        client_config: ClientConfig,
        *,
        api_factory: APIFactory = DancifyAPI,
        capture_factory: CaptureFactory | None = None,
        player_factory: PlayerFactory | None = None,
        socket_factory: SocketFactory | None = None,
        calibration_socket_factory: CalibrationSocketFactory | None = None,
    ) -> None:
        super().__init__()
        self.client_config = client_config
        self.api_factory = api_factory
        self.capture_factory = capture_factory or _default_capture
        self.player_factory = player_factory or _default_player
        self.socket_factory = socket_factory
        self.calibration_socket_factory = calibration_socket_factory or _default_calibration_socket

    def on_mount(self) -> None:
        self.push_screen(SetupScreen())
