from __future__ import annotations

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from textual.widgets import Log, Static

from dancify.domain import RawImuSample, SessionState, Vector3, WristSide
from dancify.terminal.calibration import CalibrationStatus, GuidedCalibrator
from dancify.terminal.capture import CaptureError
from dancify.terminal.config import ClientConfig
from dancify.terminal.controller import HeadlessController, LiveStatus
from dancify.terminal.dto import (
    CalibrationResult,
    ClockObservation,
    MotionHealth,
    RawUploadResult,
    Score,
    Session,
    WristMotionHealth,
)
from dancify.terminal.errors import GameplayAborted, PlaybackError
from dancify.terminal.reducer import GameplayState
from dancify.terminal.rest import DancifyAPI
from dancify.terminal.socket import GameplaySocket
from dancify.terminal.tui import DancifyTerminalApp, SetupScreen, score_cue


def _session(state: SessionState = SessionState.READY, score: float = 0.0) -> Session:
    return Session("s", "r", "p", state, 0.0, 1.0 if state is SessionState.COMPLETED else 0.0, 0, score, 3)


def _health(accepted: int = 0, dropped: int = 0) -> MotionHealth:
    wrist = WristMotionHealth(accepted // 2, dropped // 2, 0, 0, 0, 1.0)
    return MotionHealth(accepted, dropped, 0, 1.0, {"left": wrist, "right": wrist})


def _samples(vectors: tuple[Vector3, Vector3], packet: int) -> list[RawImuSample]:
    result: list[RawImuSample] = []
    for index, vector in enumerate(vectors):
        for wrist in WristSide:
            result.append(
                RawImuSample(
                    f"{wrist.value}:controller",
                    (packet + index) * 20_000,
                    packet + index,
                    vector,
                    Vector3(0.0, 0.0, 0.0),
                )
            )
    return result


class CalibrationCapture:
    def __init__(self, samples: list[RawImuSample], wrists: tuple[WristSide, ...] = tuple(WristSide)) -> None:
        self.running = True
        self._samples = samples
        self.starts = 0
        self.stops = 0
        self.config = SimpleNamespace(slots={wrist: index for index, wrist in enumerate(wrists)})
        self.health = SimpleNamespace(queue_depth=0, running=True, slots=())

    async def start(self) -> None:
        self.starts += 1
        self.running = True

    async def stop(self) -> None:
        self.stops += 1
        self.running = False

    async def receive(self, _timeout: float) -> RawImuSample:
        if not self._samples:
            raise TimeoutError
        return self._samples.pop(0)


class CalibrationSocket:
    def __init__(self) -> None:
        self.value = 0.0

    async def observe_clock(self) -> Any:
        from dancify.terminal.dto import ClockObservation

        self.value += 1.0
        return ClockObservation(self.value, self.value + 0.01, self.value + 0.02, self.value + 0.03)


class CalibrationAPI:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    async def calibrate(self, _session_id: str, payload: dict[str, Any]) -> CalibrationResult:
        self.payload = payload
        confidence = {wrist: 0.95 for wrist in cast(dict[str, Any], payload["wrists"])}
        return CalibrationResult(0.01, 0.9, 2, confidence)


def test_guided_calibration_measures_both_wrists_and_retries() -> None:
    async def scenario() -> None:
        # First neutral attempt is deliberately unstable; every accepted pose is measured.
        samples = (
            _samples((Vector3(0, 0, 1), Vector3(1, 0, 1)), 1)
            + _samples((Vector3(0, 0, 1), Vector3(0, 0, 1)), 3)
            + _samples((Vector3(0, 1, 1), Vector3(0, 1, 1)), 5)
            + _samples((Vector3(1, 0, 1), Vector3(1, 0, 1)), 7)
        )
        capture = CalibrationCapture(samples)
        api = CalibrationAPI()
        statuses: list[CalibrationStatus] = []
        result = await GuidedCalibrator(
            cast(Any, api),
            cast(Any, CalibrationSocket()),
            cast(Any, capture),
            samples_per_wrist=2,
            countdown_seconds=0,
            clock_observations=2,
            status=statuses.append,
        ).calibrate("s")
        assert result.schema_version == 2
        assert api.payload is not None and api.payload["schemaVersion"] == 2
        assert set(api.payload["wrists"]) == {"left", "right"}
        assert api.payload["wrists"]["right"]["outward"] == [[1, 0, 1], [1, 0, 1]]
        assert any("retrying neutral" in status.message for status in statuses)
        assert capture.starts == 0 and capture.stops == 0

    asyncio.run(scenario())


def test_guided_calibration_right_only_payload_and_status() -> None:
    async def scenario() -> None:
        vectors = (
            Vector3(0, 0, 1),
            Vector3(0, 1, 1),
            Vector3(1, 0, 1),
        )
        samples = [
            RawImuSample("right:controller", index * 20_000, index, vector, Vector3(0, 0, 0))
            for index, vector in enumerate(vectors, start=1)
            for _ in range(2)
        ]
        capture = CalibrationCapture(samples, (WristSide.RIGHT,))
        api = CalibrationAPI()
        statuses: list[CalibrationStatus] = []
        result = await GuidedCalibrator(
            cast(Any, api),
            cast(Any, CalibrationSocket()),
            cast(Any, capture),
            samples_per_wrist=2,
            countdown_seconds=0,
            clock_observations=2,
            status=statuses.append,
        ).calibrate("s")
        assert result.wrist_confidence == {"right": 0.95}
        assert api.payload is not None and set(api.payload["wrists"]) == {"right"}
        collecting = [status for status in statuses if status.message == "Collecting stable samples"]
        assert collecting and all(status.left_samples == 0 for status in collecting)
        assert collecting[-1].right_samples == 2
        assert any("your right wrist" in status.message for status in statuses)

    asyncio.run(scenario())


class ObservationClient:
    def __init__(self) -> None:
        self.handlers: dict[tuple[str, str], Any] = {}
        self.connected = False

    def on(self, event: str, handler: Any, namespace: str) -> None:
        self.handlers[(namespace, event)] = handler

    async def connect(self, *_: Any, **__: Any) -> None:
        self.connected = True
        await self.handlers[("/gameplay", "connect")]()

    async def disconnect(self) -> None:
        self.connected = False

    async def call(self, event: str, payload: dict[str, object], **_: Any) -> dict[str, Any]:
        if event == "session.join":
            return {
                "ok": True,
                "session": {
                    "id": "s",
                    "routine_id": "r",
                    "player_id": "p",
                    "state": "ready",
                    "playback_start_time": None,
                    "current_timestamp": 0.0,
                    "current_window": 0,
                    "cumulative_score": 0.0,
                    "event_sequence": 0,
                },
            }
        if event == "calibration.observation":
            sent = cast(float, payload["clientSend"])
            return {
                "ok": True,
                "observation": {"clientSend": sent, "serverReceive": 20.0, "serverSend": 20.01},
            }
        raise AssertionError(event)


def test_socket_real_clock_observation_and_event_observer() -> None:
    async def scenario() -> None:
        client = ObservationClient()
        times: Iterator[float] = iter((10.0, 10.04))
        events: list[str] = []

        async def observer(event: str, _payload: dict[str, Any], _state: GameplayState) -> None:
            events.append(event)

        socket = GameplaySocket(
            ClientConfig("http://test"),
            "s",
            client=cast(Any, client),
            clock=lambda: next(times),
            observer=observer,
        )
        await socket.connect()
        observation = await socket.observe_clock()
        assert observation.client_send == 10.0 and observation.client_receive == 10.04
        await client.handlers[("/gameplay", "score.update")](
            {
                "windowIndex": 0,
                "windowStartSeconds": 0.0,
                "windowScore": 91.0,
                "cumulativeScore": 91.0,
                "valid": True,
                "sequence": 1,
            }
        )
        assert events == ["session.joined", "score.update"]
        await socket.disconnect()

    asyncio.run(scenario())


def test_rest_raw_partial_acceptance_contract() -> None:
    async def scenario() -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/motion/raw")
            return httpx.Response(
                202,
                json={
                    "accepted": 1,
                    "dropped": 1,
                    "errors": [{"index": 1, "code": "duplicate_packet", "message": "duplicate packet"}],
                    "motionHealth": {
                        "accepted": 1,
                        "dropped": 1,
                        "malformed": 0,
                        "quality": 0.5,
                        "wrists": {
                            side: {
                                "accepted": int(side == "left"),
                                "dropped": int(side == "right"),
                                "duplicates": int(side == "right"),
                                "outOfOrder": 0,
                                "invalidTiming": 0,
                                "quality": float(side == "left"),
                            }
                            for side in ("left", "right")
                        },
                    },
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://test/api/v1")
        api = DancifyAPI(ClientConfig("http://test"), client)
        result = await api.upload_raw_motion("s", [{"packetNumber": 1}, {"packetNumber": 1}])
        assert result.accepted == 1 and result.dropped == 1
        assert result.errors[0].code == "duplicate_packet"
        await client.aclose()

    asyncio.run(scenario())


class Estimate:
    def to_monotonic_time(self, timestamp: int) -> float:
        return timestamp / 1_000_000


class LiveCapture:
    def __init__(self, play: asyncio.Event, count: int = 20, *, stale: bool = False) -> None:
        self.running = False
        self.play = play
        self.stale = stale
        self.samples = [
            RawImuSample(
                f"{'left' if index % 2 == 0 else 'right'}:pad",
                1_000_000 + index * 20_000,
                index + 1,
                Vector3(0.2, 0.3, 1.0),
                Vector3(0.0, 0.0, 0.0),
            )
            for index in range(count)
        ]
        self.starts = 0
        self.stops = 0
        self.config = SimpleNamespace(stale_after=0.01 if stale else 1.0)

    @property
    def health(self) -> Any:
        slots = tuple(
            SimpleNamespace(wrist=side, slot=index, connected=True, stale=self.stale)
            for index, side in enumerate(WristSide)
        )
        return SimpleNamespace(running=self.running, healthy=self.running, queue_depth=0, slots=slots)

    async def start(self) -> None:
        self.starts += 1
        self.running = True

    async def stop(self) -> None:
        self.stops += 1
        self.running = False

    async def receive(self, timeout: float) -> RawImuSample:
        if not self.play.is_set():
            try:
                await asyncio.wait_for(self.play.wait(), timeout)
            except TimeoutError:
                raise TimeoutError from None
        if self.samples:
            await asyncio.sleep(0)
            return self.samples.pop(0)
        await asyncio.sleep(timeout)
        raise TimeoutError

    def clock_estimate(self, _wrist: WristSide) -> Estimate:
        return Estimate()


class LivePlayer:
    def __init__(self, play: asyncio.Event) -> None:
        self.play_event = play
        self.positions = iter((0.0, 0.5, 1.0))
        self.prepared = self.closed = 0

    async def prepare(self) -> None:
        self.prepared += 1

    async def play(self) -> None:
        self.play_event.set()

    async def position(self) -> float:
        return next(self.positions)

    async def close(self) -> None:
        self.closed += 1


class LiveAPI:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.state = SessionState.READY
        self.accepted = 0
        self.aborts = 0

    async def upload_raw_motion(self, _session_id: str, samples: list[dict[str, Any]]) -> RawUploadResult:
        self.order.append("upload")
        self.accepted += len(samples)
        return RawUploadResult(len(samples), 0, (), _health(self.accepted))

    async def get_session(self, _session_id: str) -> Session:
        return _session(self.state, 88.0 if self.state is SessionState.COMPLETED else 0.0)

    async def abort(self, _session_id: str) -> Session:
        self.aborts += 1
        self.state = SessionState.ABORTED
        return _session(self.state)


class LiveConnection:
    def __init__(self, state: GameplayState, api: LiveAPI, order: list[str]) -> None:
        self.state = state
        self.api = api
        self.order = order

    async def __aenter__(self) -> LiveConnection:
        self.state.reconcile(_session())
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def observe_clock(self) -> ClockObservation:
        now = asyncio.get_running_loop().time()
        return ClockObservation(now, now, now, now)

    async def ready(self, delay: float) -> float:
        return asyncio.get_running_loop().time() + delay

    async def progress(self, video_time: float, _server_time: float | None = None) -> None:
        self.order.append(f"progress:{video_time}")
        if video_time >= 0.5 and not self.state.scores:
            self.state.scores[0] = Score(0, 0.0, 88.0, 88.0, True)
        if video_time >= 1.0:
            self.api.state = SessionState.COMPLETED


def test_live_controller_concurrency_completion_abort_and_cleanup() -> None:
    async def scenario() -> None:
        order: list[str] = []
        play = asyncio.Event()
        api = LiveAPI(order)
        capture = LiveCapture(play)
        player = LivePlayer(play)

        def factory(_config: ClientConfig, _session: str, state: GameplayState) -> LiveConnection:
            return LiveConnection(state, api, order)

        result = await HeadlessController(cast(Any, api), ClientConfig("http://test"), factory).run_live(
            "s", 1.0, cast(Any, capture), player, delay_seconds=0
        )
        assert result.session.state is SessionState.COMPLETED
        assert result.accepted_features == 20 and result.scores[0].value == 88
        assert order.index("upload") < order.index("progress:1.0")
        assert capture.starts == 1 and capture.stops == 1 and player.closed == 1

        cancel_order: list[str] = []
        cancel_api = LiveAPI(cancel_order)
        cancel_play = asyncio.Event()
        cancel_capture = LiveCapture(cancel_play)
        cancel_player = LivePlayer(cancel_play)
        cancelled = asyncio.Event()
        cancelled.set()

        def cancel_factory(_config: ClientConfig, _session: str, state: GameplayState) -> LiveConnection:
            return LiveConnection(state, cancel_api, cancel_order)

        with pytest.raises(GameplayAborted):
            await HeadlessController(cast(Any, cancel_api), ClientConfig("http://test"), cancel_factory).run_live(
                "s",
                1.0,
                cast(Any, cancel_capture),
                cancel_player,
                delay_seconds=0,
                cancel=cancelled,
            )
        assert cancel_api.aborts == 1
        assert cancel_capture.stops == 1 and cancel_player.closed == 1

        stale_order: list[str] = []
        stale_api = LiveAPI(stale_order)
        stale_play = asyncio.Event()
        stale_capture = LiveCapture(stale_play, stale=True)
        stale_player = LivePlayer(stale_play)

        def stale_factory(_config: ClientConfig, _session: str, state: GameplayState) -> LiveConnection:
            return LiveConnection(state, stale_api, stale_order)

        with pytest.raises(CaptureError, match="stale or disconnected"):
            await HeadlessController(cast(Any, stale_api), ClientConfig("http://test"), stale_factory).run_live(
                "s", 1.0, cast(Any, stale_capture), stale_player, delay_seconds=0
            )
        assert stale_api.aborts == 1
        assert stale_capture.stops == 1 and stale_player.closed == 1

        mismatch_order: list[str] = []
        mismatch_api = LiveAPI(mismatch_order)
        mismatch_play = asyncio.Event()
        mismatch_capture = LiveCapture(mismatch_play)

        class ShortPlayer(LivePlayer):
            @property
            def media_duration(self) -> float:
                return 0.5

        mismatch_player = ShortPlayer(mismatch_play)

        def mismatch_factory(_config: ClientConfig, _session: str, state: GameplayState) -> LiveConnection:
            return LiveConnection(state, mismatch_api, mismatch_order)

        with pytest.raises(PlaybackError, match="requested duration 1.000s exceeds media duration 0.500s"):
            await HeadlessController(cast(Any, mismatch_api), ClientConfig("http://test"), mismatch_factory).run_live(
                "s", 1.0, cast(Any, mismatch_capture), mismatch_player, delay_seconds=0
            )
        assert mismatch_player.prepared == 1 and mismatch_player.closed == 1
        assert mismatch_capture.starts == 0 and mismatch_api.aborts == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("value", "valid", "expected"),
    [
        (100.0, True, "PERFECT"),
        (90.0, True, "PERFECT"),
        (89.999, True, "GREAT"),
        (75.0, True, "GREAT"),
        (74.999, True, "GOOD"),
        (50.0, True, "GOOD"),
        (49.999, True, "KEEP GOING"),
        (0.0, True, "KEEP GOING"),
        (100.0, False, "MISS"),
    ],
)
def test_score_cue_thresholds(value: float, valid: bool, expected: str) -> None:
    assert score_cue(value, valid) == expected


def test_textual_pilot_live_score_health_and_abort_rendering() -> None:
    async def scenario() -> None:
        app = DancifyTerminalApp(ClientConfig("http://test"))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = cast(SetupScreen, app.screen)
            state = GameplayState()
            state.reconcile(_session(SessionState.PLAYING))
            state.scores[0] = Score(0, 0.0, 93.25, 93.25, True)
            screen._render_live_status(
                LiveStatus("playback", "Playing", 0.5, state, True, accepted=40, dropped=2),
                3.0,
            )
            screen._render_live_status(LiveStatus("health", "Healthy", 0.5, state, accepted=40, dropped=2), 3.0)
            state.scores[1] = Score(1, 1.0, 75.0, 84.125, True)
            screen._render_live_status(LiveStatus("event", "score.update", 1.5, state), 3.0)
            state.scores[2] = Score(2, 2.0, 100.0, 84.125, False)
            screen._render_live_status(LiveStatus("event", "score.update", 2.5, state), 3.0)
            screen._render_live_status(LiveStatus("playback", "Playing", 2.5, state, accepted=40, dropped=2), 3.0)
            await pilot.pause()

            score_widget = screen.query_one("#live-score", Static)
            assert str(score_widget.render()) == ("MISS (no data) · window 2 · score 100.0 · cumulative 84.1")
            assert score_widget.has_class("score-invalid")
            assert not score_widget.has_class("score-valid")
            assert "accepted 40, dropped 2" in str(screen.query_one("#device-health").render())
            live_log = screen.query_one("#log", Log)
            assert live_log.lines == [
                "Score update · PERFECT · window 0 · score 93.2 · cumulative 93.2",
                "Score update · GREAT · window 1 · score 75.0 · cumulative 84.1",
                "Score update · MISS (no data) · window 2 · score 100.0 · cumulative 84.1",
            ]
            await pilot.press("ctrl+x")
            await pilot.pause()
            assert screen.cancel.is_set()
            assert "stopped" in str(screen.query_one("#status").render())

    asyncio.run(scenario())


def test_resumed_score_history_is_displayed_but_not_announced() -> None:
    async def scenario() -> None:
        historical = Score(4, 4.0, 82.0, 82.0, True)
        session = Session(
            "s",
            "r",
            "p",
            SessionState.PLAYING,
            0.0,
            4.5,
            4,
            82.0,
            4,
            scores=(historical,),
        )
        app = DancifyTerminalApp(ClientConfig("http://test"))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = cast(SetupScreen, app.screen)
            screen.session = session
            screen._seed_score_history()
            state = GameplayState()
            state.reconcile(session)
            screen._render_live_status(LiveStatus("playback", "Resumed", 4.5, state), 5.0)
            screen._render_live_status(LiveStatus("health", "Healthy", 4.5, state), 5.0)
            await pilot.pause()

            assert str(screen.query_one("#live-score").render()) == ("GREAT · window 4 · score 82.0 · cumulative 82.0")
            assert screen.query_one("#log", Log).lines == []

    asyncio.run(scenario())


def test_live_maps_server_start_and_sends_repeated_ten_hz_heartbeats() -> None:
    async def scenario() -> None:
        order: list[str] = []
        play = asyncio.Event()
        api = LiveAPI(order)
        capture = LiveCapture(play, count=40)

        class RepeatingPlayer(LivePlayer):
            def __init__(self) -> None:
                super().__init__(play)
                self.positions = iter((0.0, 0.0, 0.25, 0.25, 0.5))
                self.played_at = 0.0

            async def play(self) -> None:
                self.played_at = asyncio.get_running_loop().time()
                await super().play()

        class ClockedConnection(LiveConnection):
            def __init__(self, state: GameplayState) -> None:
                super().__init__(state, api, order)
                self.expected_start = 0.0
                self.heartbeats: list[tuple[float, float]] = []

            async def observe_clock(self) -> ClockObservation:
                local = asyncio.get_running_loop().time()
                server = local + 100.0
                return ClockObservation(local, server, server, local)

            async def ready(self, delay: float) -> float:
                self.expected_start = asyncio.get_running_loop().time() + delay
                return self.expected_start + 100.0

            async def progress(self, video_time: float, _server_time: float | None = None) -> None:
                self.heartbeats.append((asyncio.get_running_loop().time(), video_time))
                if video_time >= 0.5:
                    api.state = SessionState.COMPLETED

        connections: list[ClockedConnection] = []

        def factory(_config: ClientConfig, _session: str, state: GameplayState) -> ClockedConnection:
            connection = ClockedConnection(state)
            connections.append(connection)
            return connection

        player = RepeatingPlayer()
        await HeadlessController(cast(Any, api), ClientConfig("http://test"), factory).run_live(
            "s", 0.5, cast(Any, capture), player, delay_seconds=0.12
        )
        connection = connections[0]
        assert player.played_at >= connection.expected_start - 0.01
        assert player.played_at < connection.expected_start + 0.06
        assert [position for _, position in connection.heartbeats] == [0.0, 0.0, 0.25, 0.25, 0.5]
        intervals = [
            current[0] - previous[0]
            for previous, current in zip(connection.heartbeats, connection.heartbeats[1:], strict=False)
        ]
        assert all(0.07 <= interval <= 0.14 for interval in intervals)

    asyncio.run(scenario())


def test_live_prepare_and_cleanup_failures_abort_and_cleanup_independently() -> None:
    async def scenario() -> None:
        order: list[str] = []
        api = LiveAPI(order)

        class FailingPlayer:
            def __init__(self) -> None:
                self.closed = 0

            async def prepare(self) -> None:
                raise RuntimeError("prepare failed")

            async def play(self) -> None:
                raise AssertionError("play must not run")

            async def position(self) -> float:
                raise AssertionError("position must not run")

            async def close(self) -> None:
                self.closed += 1
                raise RuntimeError("player close failed")

        class FailingCapture(LiveCapture):
            async def stop(self) -> None:
                self.stops += 1
                self.running = False
                raise RuntimeError("capture stop failed")

        class FailingConnection(LiveConnection):
            def __init__(self, state: GameplayState) -> None:
                super().__init__(state, api, order)
                self.exits = 0

            async def __aexit__(self, *_: Any) -> None:
                self.exits += 1
                raise RuntimeError("socket close failed")

        connections: list[FailingConnection] = []

        def factory(_config: ClientConfig, _session: str, state: GameplayState) -> FailingConnection:
            connection = FailingConnection(state)
            connections.append(connection)
            return connection

        play = asyncio.Event()
        capture = FailingCapture(play)
        player = FailingPlayer()
        with pytest.raises(RuntimeError, match="prepare failed") as raised:
            await HeadlessController(cast(Any, api), ClientConfig("http://test"), factory).run_live(
                "s", 1.0, cast(Any, capture), cast(Any, player), delay_seconds=0
            )
        assert api.aborts == 1
        assert player.closed == 1 and capture.stops == 1 and connections[0].exits == 1
        assert len(raised.value.__notes__) == 3

        api.state = SessionState.READY
        legacy_player = FailingPlayer()
        with pytest.raises(RuntimeError, match="prepare failed"):
            await HeadlessController(cast(Any, api), ClientConfig("http://test"), factory).run_with_player(
                "s", 1.0, cast(Any, SimpleNamespace()), cast(Any, legacy_player), delay_seconds=0
            )
        assert legacy_player.closed == 1 and api.aborts == 2

        class CleanupPlayer(LivePlayer):
            async def close(self) -> None:
                self.closed += 1
                raise RuntimeError("player close failed")

        class CleanupConnection(FailingConnection):
            async def progress(self, video_time: float, _server_time: float | None = None) -> None:
                self.order.append(f"progress:{video_time}")

        cleanup_connections: list[CleanupConnection] = []

        def cleanup_factory(_config: ClientConfig, _session: str, state: GameplayState) -> CleanupConnection:
            connection = CleanupConnection(state)
            cleanup_connections.append(connection)
            return connection

        api.state = SessionState.READY
        cleanup_play = asyncio.Event()
        cleanup_capture = FailingCapture(cleanup_play)
        cleanup_player = CleanupPlayer(cleanup_play)
        with pytest.raises(RuntimeError, match="socket close failed") as cleanup_raised:
            await HeadlessController(cast(Any, api), ClientConfig("http://test"), cleanup_factory).run_live(
                "s", 1.0, cast(Any, cleanup_capture), cleanup_player, delay_seconds=0
            )
        assert api.aborts == 3
        assert cleanup_connections[0].exits == 1
        assert cleanup_player.closed == 1 and cleanup_capture.stops == 1
        assert len(cleanup_raised.value.__notes__) == 2

    asyncio.run(scenario())


def test_live_health_persists_across_score_only_statuses_with_per_wrist_detail() -> None:
    async def scenario() -> None:
        app = DancifyTerminalApp(ClientConfig("http://test"))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = cast(SetupScreen, app.screen)
            state = GameplayState()
            state.reconcile(_session(SessionState.PLAYING))
            state.motion_health = _health(40, 2)
            screen._render_live_status(LiveStatus("motion", "Uploaded", state=state, capture_healthy=True), 1.0)
            state.scores[0] = Score(0, 0.0, 93.25, 93.25, True)
            screen._render_live_status(LiveStatus("event", "score.update", state=state), 1.0)
            await pilot.pause()
            rendered = str(screen.query_one("#device-health").render())
            assert "accepted 40, dropped 2" in rendered
            assert "left accepted 20, dropped 1" in rendered
            assert "right accepted 20, dropped 1" in rendered

    asyncio.run(scenario())


def test_tui_right_only_configuration_and_live_health_hide_left() -> None:
    async def scenario() -> None:
        app = DancifyTerminalApp(ClientConfig("http://test", dsu_right_slot=3))
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = cast(SetupScreen, app.screen)
            assert "R3 (right-only)" in str(screen.query_one("#configuration").render())
            state = GameplayState()
            state.reconcile(
                Session(
                    "s",
                    "r",
                    "p",
                    SessionState.PLAYING,
                    0.0,
                    0.0,
                    0,
                    0.0,
                    1,
                    active_wrists=("right",),
                )
            )
            state.motion_health = _health(20, 0)
            screen._render_live_status(LiveStatus("motion", "Uploaded", state=state, capture_healthy=True), 1.0)
            await pilot.pause()
            rendered = str(screen.query_one("#device-health").render())
            assert "right accepted" in rendered
            assert "left accepted" not in rendered

    asyncio.run(scenario())


def test_ctrl_x_cancels_and_awaits_blocking_calibration_before_abort() -> None:
    async def scenario() -> None:
        entered_receive = asyncio.Event()

        class BlockingCapture(CalibrationCapture):
            def __init__(self) -> None:
                super().__init__([])
                self.running = True

            async def receive(self, _timeout: float) -> RawImuSample:
                entered_receive.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        class TUIAPI(CalibrationAPI):
            def __init__(self) -> None:
                super().__init__()
                self.aborts = 0
                self.current = _session(SessionState.READY)

            async def __aenter__(self) -> TUIAPI:
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

            async def get_session(self, _session_id: str) -> Session:
                return self.current

            async def abort(self, _session_id: str) -> Session:
                self.aborts += 1
                self.current = _session(SessionState.ABORTED)
                return self.current

        class TUISocket(CalibrationSocket):
            async def __aenter__(self) -> TUISocket:
                return self

            async def __aexit__(self, *_: Any) -> None:
                return None

        capture = BlockingCapture()
        api = TUIAPI()
        socket = TUISocket()
        app = DancifyTerminalApp(
            ClientConfig("http://test"),
            api_factory=cast(Any, lambda _config: api),
            calibration_socket_factory=cast(Any, lambda *_args: socket),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = cast(SetupScreen, app.screen)
            screen.session = _session(SessionState.READY)
            screen.capture = cast(Any, capture)
            screen.action_calibrate()
            await asyncio.wait_for(entered_receive.wait(), 5.0)
            await pilot.press("ctrl+x")
            for _ in range(10):
                await pilot.pause()
                if api.aborts:
                    break
            assert api.aborts == 1
            assert capture.stops == 1
            assert screen._active_workflow is None
            assert "stopped" in str(screen.query_one("#status").render())

    asyncio.run(scenario())
