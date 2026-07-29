"""Bounded one- or two-wrist asyncio capture for Cemuhook/DSU UDP servers."""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Final

from dancify.domain import RawImuSample, WristSide
from dancify.terminal.dsu import (
    PROTOCOL_VERSION,
    ControllerDataResponse,
    ControllerIdentity,
    ControllerInfoResponse,
    DSUProtocolError,
    ProtocolVersionResponse,
    controller_data_request,
    controller_info_request,
    parse_response,
    version_request,
)
from dancify.terminal.errors import ConnectionFailure

_UINT32_MODULUS: Final = 1 << 32
_UINT32_HALF: Final = 1 << 31
_STOP: Final = object()


class CaptureError(ConnectionFailure):
    """The configured DSU source cannot provide every configured wrist stream."""


@dataclass(frozen=True, slots=True)
class DSUCaptureConfig:
    """Network, assignment, and resource limits for a DSU capture.

    The right slot is mandatory. Omitting the left slot enables right-handed
    mode. Assignment never depends on response arrival order or the server's
    implicit subscribe-to-all behavior.
    """

    host: str = "127.0.0.1"
    port: int = 26760
    left_slot: int | None = None
    right_slot: int = 1
    queue_size: int = 512
    discovery_timeout: float = 2.0
    refresh_interval: float = 1.0
    stale_after: float = 1.0
    client_id: int | None = None

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("DSU host is required")
        if isinstance(self.port, bool) or not 1 <= self.port <= 65535:
            raise ValueError("DSU port must be between 1 and 65535")
        if self.left_slot is not None and (isinstance(self.left_slot, bool) or not 0 <= self.left_slot <= 3):
            raise ValueError("left DSU slot must be between 0 and 3")
        if isinstance(self.right_slot, bool) or not 0 <= self.right_slot <= 3:
            raise ValueError("right DSU slot must be between 0 and 3")
        if self.left_slot is not None and self.left_slot == self.right_slot:
            raise ValueError("left and right wrists must use distinct DSU slots")
        if isinstance(self.queue_size, bool) or self.queue_size <= 0:
            raise ValueError("capture queue size must be positive")
        for name, value in (
            ("discovery timeout", self.discovery_timeout),
            ("refresh interval", self.refresh_interval),
            ("stale threshold", self.stale_after),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.client_id is not None and (isinstance(self.client_id, bool) or not 0 <= self.client_id <= 0xFFFFFFFF):
            raise ValueError("client_id must be an unsigned 32-bit integer")

    @property
    def slots(self) -> Mapping[WristSide, int]:
        slots = {WristSide.RIGHT: self.right_slot}
        if self.left_slot is not None:
            slots = {WristSide.LEFT: self.left_slot, **slots}
        return slots


@dataclass(frozen=True, slots=True)
class ClockEstimate:
    """Affine mapping from DSU device microseconds to local monotonic seconds."""

    scale: float
    offset_seconds: float
    observations: int

    def to_monotonic_time(self, device_timestamp_us: int) -> float:
        if device_timestamp_us < 0:
            raise ValueError("device timestamp must be non-negative")
        return self.scale * device_timestamp_us / 1_000_000.0 + self.offset_seconds


class SlotClockEstimator:
    """Bounded affine estimator using the lowest observed transport delay.

    Regression estimates clock rate, while the minimum residual estimates the
    offset without treating positive UDP queueing delay as clock offset.
    """

    def __init__(self, max_observations: int = 64) -> None:
        if max_observations < 2:
            raise ValueError("clock estimator needs room for at least two observations")
        self._points: deque[tuple[float, float]] = deque(maxlen=max_observations)
        self._estimate: ClockEstimate | None = None
        self._last_timestamp_us: int | None = None

    def reset(self) -> None:
        self._points.clear()
        self._estimate = None
        self._last_timestamp_us = None

    def observe(self, device_timestamp_us: int, received_at: float) -> ClockEstimate:
        if device_timestamp_us < 0 or not isfinite(received_at):
            raise ValueError("clock observations must be finite and non-negative")
        if self._last_timestamp_us is not None and device_timestamp_us < self._last_timestamp_us:
            self.reset()
        self._last_timestamp_us = device_timestamp_us
        device_seconds = device_timestamp_us / 1_000_000.0
        self._points.append((device_seconds, received_at))
        scale = 1.0
        if len(self._points) >= 3:
            mean_device = sum(point[0] for point in self._points) / len(self._points)
            mean_local = sum(point[1] for point in self._points) / len(self._points)
            variance = sum((device - mean_device) ** 2 for device, _ in self._points)
            if variance > 1e-12:
                fitted = sum((device - mean_device) * (local - mean_local) for device, local in self._points) / variance
                # Real clocks should be close; this also prevents network jitter
                # from creating a dangerous timestamp multiplier.
                scale = min(1.05, max(0.95, fitted))
        offset = min(local - scale * device for device, local in self._points)
        self._estimate = ClockEstimate(scale, offset, len(self._points))
        return self._estimate

    @property
    def estimate(self) -> ClockEstimate | None:
        return self._estimate


@dataclass(frozen=True, slots=True)
class PacketDecision:
    accepted: bool
    duplicate: bool = False
    out_of_order: bool = False
    estimated_loss: int = 0


def classify_packet(previous: int | None, current: int) -> PacketDecision:
    """Classify uint32 packet numbers using RFC-1982-style serial arithmetic."""

    if isinstance(current, bool) or not 0 <= current <= 0xFFFFFFFF:
        raise ValueError("packet number must be an unsigned 32-bit integer")
    if previous is None:
        return PacketDecision(True)
    if isinstance(previous, bool) or not 0 <= previous <= 0xFFFFFFFF:
        raise ValueError("previous packet number must be an unsigned 32-bit integer")
    delta = (current - previous) % _UINT32_MODULUS
    if delta == 0:
        return PacketDecision(False, duplicate=True)
    if delta >= _UINT32_HALF:
        return PacketDecision(False, out_of_order=True)
    return PacketDecision(True, estimated_loss=delta - 1)


@dataclass(frozen=True, slots=True)
class SlotStreamHealth:
    wrist: WristSide
    slot: int
    connected: bool
    stale: bool
    accepted: int
    duplicates: int
    out_of_order: int
    estimated_loss: int
    queue_dropped: int
    last_packet: int | None
    last_received_at: float | None
    sample_rate_hz: float
    clock: ClockEstimate | None

    @property
    def quality(self) -> float:
        attempted = self.accepted + self.duplicates + self.out_of_order + self.estimated_loss + self.queue_dropped
        return self.accepted / attempted if attempted else 1.0


@dataclass(frozen=True, slots=True)
class CaptureHealth:
    running: bool
    server_id: int | None
    protocol_version: int | None
    queue_depth: int
    invalid_packets: int
    transport_errors: int
    slots: tuple[SlotStreamHealth, ...]

    @property
    def healthy(self) -> bool:
        return self.running and all(slot.connected and not slot.stale for slot in self.slots)


@dataclass(slots=True)
class _SlotTracker:
    wrist: WristSide
    slot: int
    clock: SlotClockEstimator = field(default_factory=SlotClockEstimator)
    identity: ControllerIdentity | None = None
    last_packet: int | None = None
    last_received_at: float | None = None
    first_received_at: float | None = None
    accepted: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    estimated_loss: int = 0
    queue_dropped: int = 0

    def reset_stream(self) -> None:
        self.clock.reset()
        self.last_packet = None
        self.last_received_at = None
        self.first_received_at = None


class _CaptureProtocol(asyncio.DatagramProtocol):
    def __init__(self, owner: DSUCapture) -> None:
        self._owner = owner

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._owner.on_datagram(data)

    def error_received(self, exc: Exception) -> None:
        self._owner.on_transport_error(exc)

    def connection_lost(self, exc: Exception | None) -> None:
        self._owner.on_connection_lost(exc)


class DSUCapture:
    """A cleanly cancellable, bounded async stream of assigned DSU slots.

    ``start`` discovers every configured controller before returning. Iteration
    yields :class:`RawImuSample`; the wrist is encoded in its stable device ID
    (``left:<mac>`` / ``right:<mac>``).  ``clock_estimate`` provides the
    corresponding device-to-local monotonic mapping.
    """

    def __init__(
        self,
        config: DSUCaptureConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or DSUCaptureConfig()
        self._clock = clock
        self._client_id = self.config.client_id if self.config.client_id is not None else secrets.randbits(32)
        self._queue: asyncio.Queue[RawImuSample | object] = asyncio.Queue(self.config.queue_size)
        self._trackers = {slot: _SlotTracker(wrist, slot) for wrist, slot in self.config.slots.items()}
        self._transport: asyncio.DatagramTransport | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        self._running = False
        self._stopping = False
        self._signal = asyncio.Event()
        self._server_id: int | None = None
        self._protocol_version: int | None = None
        self._startup_error: CaptureError | None = None
        self._invalid_packets = 0
        self._transport_errors = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def client_id(self) -> int:
        return self._client_id

    @property
    def identities(self) -> Mapping[WristSide, ControllerIdentity]:
        return {tracker.wrist: tracker.identity for tracker in self._trackers.values() if tracker.identity is not None}

    def clock_estimate(self, wrist: WristSide) -> ClockEstimate | None:
        slot = self.config.slots.get(wrist)
        return None if slot is None else self._trackers[slot].clock.estimate

    async def __aenter__(self) -> DSUCapture:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    def __aiter__(self) -> AsyncIterator[RawImuSample]:
        return self.raw_samples()

    async def start(self) -> None:
        """Open UDP, discover configured slots, and begin registration refresh."""

        if self._running:
            return
        if self._transport is not None:
            raise CaptureError("DSU capture is stopping")
        # A capture may be restarted after normal shutdown or failed discovery;
        # no sentinel, old identity, packet epoch, or health counter leaks into
        # the new UDP run.
        self._queue = asyncio.Queue(self.config.queue_size)
        self._trackers = {slot: _SlotTracker(wrist, slot) for wrist, slot in self.config.slots.items()}
        self._signal = asyncio.Event()
        self._server_id = None
        self._protocol_version = None
        self._startup_error = None
        self._invalid_packets = 0
        self._transport_errors = 0
        self._stopping = False
        loop = asyncio.get_running_loop()
        try:
            transport, _ = await loop.create_datagram_endpoint(
                lambda: _CaptureProtocol(self),
                remote_addr=(self.config.host, self.config.port),
            )
            self._transport = transport
            self._running = True
            await self._discover()
            self._send_registrations()
            self._refresh_task = asyncio.create_task(self._refresh(), name="dancify-dsu-refresh")
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Cancel refresh, close UDP, and wake any waiting stream consumer."""

        if self._stopping:
            return
        self._stopping = True
        self._running = False
        task, self._refresh_task = self._refresh_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        transport, self._transport = self._transport, None
        if transport is not None:
            transport.close()
            await asyncio.sleep(0)
        self._put_stop()
        self._signal.set()
        self._stopping = False

    async def receive(self, timeout: float | None = None) -> RawImuSample:
        """Receive one accepted sample, optionally with a bounded wait."""

        if timeout is not None and (not isfinite(timeout) or timeout <= 0):
            raise ValueError("receive timeout must be finite and positive")
        get = self._queue.get()
        item = await get if timeout is None else await asyncio.wait_for(get, timeout)
        if item is _STOP:
            # Keep the sentinel available for another waiter/iterator.
            self._put_stop()
            raise StopAsyncIteration
        assert isinstance(item, RawImuSample)
        return item

    async def raw_samples(self) -> AsyncIterator[RawImuSample]:
        while True:
            try:
                yield await self.receive()
            except StopAsyncIteration:
                return

    @property
    def health(self) -> CaptureHealth:
        now = self._clock()
        slots: list[SlotStreamHealth] = []
        for tracker in sorted(self._trackers.values(), key=lambda item: item.wrist.value):
            identity = tracker.identity
            stale = tracker.last_received_at is None or now - tracker.last_received_at > self.config.stale_after
            elapsed = (
                tracker.last_received_at - tracker.first_received_at
                if tracker.last_received_at is not None and tracker.first_received_at is not None
                else 0.0
            )
            rate = (tracker.accepted - 1) / elapsed if tracker.accepted > 1 and elapsed > 0 else 0.0
            slots.append(
                SlotStreamHealth(
                    tracker.wrist,
                    tracker.slot,
                    identity is not None and identity.connected,
                    stale,
                    tracker.accepted,
                    tracker.duplicates,
                    tracker.out_of_order,
                    tracker.estimated_loss,
                    tracker.queue_dropped,
                    tracker.last_packet,
                    tracker.last_received_at,
                    rate,
                    tracker.clock.estimate,
                )
            )
        return CaptureHealth(
            self._running,
            self._server_id,
            self._protocol_version,
            self._queue.qsize(),
            self._invalid_packets,
            self._transport_errors,
            tuple(slots),
        )

    async def _discover(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.config.discovery_timeout
        self._send_discovery()
        while not self._all_connected():
            if self._startup_error is not None:
                raise self._startup_error
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                missing = ", ".join(
                    f"{tracker.wrist.value} slot {tracker.slot}"
                    for tracker in self._trackers.values()
                    if tracker.identity is None or not tracker.identity.connected
                )
                raise CaptureError(
                    f"timed out discovering DSU controllers: {missing}",
                    hint="Start the Cemuhook/DSU server and verify the configured wrist slot numbers.",
                )
            self._signal.clear()
            try:
                await asyncio.wait_for(self._signal.wait(), min(remaining, 0.25))
            except TimeoutError:
                self._send_discovery()
        if self._startup_error is not None:
            raise self._startup_error

    def _all_connected(self) -> bool:
        return all(tracker.identity is not None and tracker.identity.connected for tracker in self._trackers.values())

    def _send_discovery(self) -> None:
        transport = self._transport
        if transport is None:
            return
        transport.sendto(version_request(self._client_id))
        transport.sendto(controller_info_request(tuple(self._trackers), self._client_id))

    def _send_registrations(self) -> None:
        transport = self._transport
        if transport is not None:
            for slot in self._trackers:
                transport.sendto(controller_data_request(slot, self._client_id))

    async def _refresh(self) -> None:
        while True:
            await asyncio.sleep(self.config.refresh_interval)
            try:
                self._send_discovery()
                self._send_registrations()
            except OSError:
                # UDP routing errors can be transient; keep refreshing while
                # surfacing the incident through health instead of orphaning an
                # unobserved failed background task.
                self._transport_errors += 1

    def on_datagram(self, datagram: bytes) -> None:
        received_at = self._clock()
        try:
            response = parse_response(datagram)
        except (DSUProtocolError, ValueError):
            self._invalid_packets += 1
            return
        if self._server_id is None:
            self._server_id = response.server_id
        elif response.server_id != self._server_id:
            # A changed server ID denotes restart/replacement. Rediscover rather
            # than mixing identities, packet numbers, and clock epochs.
            self._server_id = response.server_id
            self._protocol_version = None
            for tracker in self._trackers.values():
                tracker.identity = None
                tracker.reset_stream()
        if isinstance(response, ProtocolVersionResponse):
            self._protocol_version = response.max_protocol_version
            if response.max_protocol_version < PROTOCOL_VERSION:
                self._startup_error = CaptureError(
                    f"DSU server supports protocol {response.max_protocol_version}, v{PROTOCOL_VERSION} is required"
                )
            self._signal.set()
            return
        if isinstance(response, ControllerInfoResponse):
            info_tracker = self._trackers.get(response.identity.slot)
            if info_tracker is None:
                return
            if info_tracker.identity != response.identity:
                info_tracker.reset_stream()
            info_tracker.identity = response.identity
            self._signal.set()
            return
        assert isinstance(response, ControllerDataResponse)
        state = response.state
        data_tracker = self._trackers.get(state.slot)
        if data_tracker is None or not state.connected or not state.identity.connected:
            return
        if data_tracker.identity is not None and data_tracker.identity.mac != state.identity.mac:
            self._invalid_packets += 1
            return
        decision = classify_packet(data_tracker.last_packet, state.packet_number)
        if decision.duplicate:
            data_tracker.duplicates += 1
            return
        if decision.out_of_order:
            data_tracker.out_of_order += 1
            return
        data_tracker.last_packet = state.packet_number
        data_tracker.estimated_loss += decision.estimated_loss
        data_tracker.accepted += 1
        data_tracker.last_received_at = received_at
        if data_tracker.first_received_at is None:
            data_tracker.first_received_at = received_at
        data_tracker.clock.observe(state.motion_timestamp_us, received_at)
        identity = data_tracker.identity or state.identity
        data_tracker.identity = identity
        suffix = identity.mac_address if any(identity.mac) else f"slot-{data_tracker.slot}"
        sample = RawImuSample(
            f"{data_tracker.wrist.value}:{suffix}",
            state.motion_timestamp_us,
            state.packet_number,
            state.acceleration_g,
            state.angular_velocity_dps,
        )
        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
            except asyncio.QueueEmpty:  # Defensive against a simultaneous consumer.
                pass
            else:
                if isinstance(dropped, RawImuSample):
                    dropped_wrist = next(
                        (wrist for wrist in WristSide if dropped.device_id.startswith(f"{wrist.value}:")),
                        None,
                    )
                    if dropped_wrist is not None:
                        self._trackers[self.config.slots[dropped_wrist]].queue_dropped += 1
        self._queue.put_nowait(sample)

    def on_transport_error(self, _exc: Exception) -> None:
        self._transport_errors += 1
        self._signal.set()

    def on_connection_lost(self, exc: Exception | None) -> None:
        if not self._stopping:
            self._running = False
            if exc is not None:
                self._transport_errors += 1
            self._put_stop()
        self._signal.set()

    def _put_stop(self) -> None:
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(_STOP)


# Explicit long-form alias for discoverability in controller/config integrations.
CemuhookCapture = DSUCapture

__all__ = [
    "CaptureError",
    "CaptureHealth",
    "CemuhookCapture",
    "ClockEstimate",
    "DSUCapture",
    "DSUCaptureConfig",
    "PacketDecision",
    "SlotClockEstimator",
    "SlotStreamHealth",
    "classify_packet",
]
