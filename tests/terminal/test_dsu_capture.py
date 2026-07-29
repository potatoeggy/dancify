from __future__ import annotations

import asyncio
import math
import struct
import zlib
from typing import cast

import pytest

from dancify.domain import MotionFeatures, RawImuSample, RawMotionSample, Vector3, WristSide
from dancify.motion import CircularMotionBuffer, DeterministicMotionSimulator, SimulationConfig
from dancify.terminal.capture import (
    CaptureError,
    CaptureHealth,
    ClockEstimate,
    DSUCapture,
    DSUCaptureConfig,
    SlotClockEstimator,
    SlotStreamHealth,
    classify_packet,
)
from dancify.terminal.dsu import (
    ControllerDataResponse,
    ControllerIdentity,
    ControllerInfoResponse,
    DSUProtocolError,
    MessageType,
    ProtocolVersionResponse,
    controller_data_request,
    controller_info_request,
    parse_response,
    version_request,
)

CLIENT_ID = 0x12345678
SERVER_ID = 0xA0B0C0D0
VERSION = 1001


def golden_packet(magic: bytes, sender_id: int, message_type: int, payload: bytes) -> bytes:
    """Independent packet constructor: no production codec constants/helpers."""

    body = struct.pack("<I", message_type) + payload
    packet = bytearray(struct.pack("<4sHHII", magic, VERSION, len(body), 0, sender_id) + body)
    struct.pack_into("<I", packet, 8, zlib.crc32(packet) & 0xFFFFFFFF)
    return bytes(packet)


def golden_identity(slot: int, mac: bytes, *, state: int = 2) -> bytes:
    return struct.pack("<BBBB6sB", slot, state, 2, 2, mac, 5)


def golden_info(slot: int, mac: bytes, *, state: int = 2) -> bytes:
    return golden_packet(
        b"DSUS",
        SERVER_ID,
        0x100001,
        golden_identity(slot, mac, state=state) + b"\0",
    )


def golden_data(
    slot: int,
    mac: bytes,
    packet_number: int,
    timestamp_us: int,
    *,
    acceleration: tuple[float, float, float] = (1.0, -2.0, 0.5),
    gyro: tuple[float, float, float] = (10.0, 20.0, -30.0),
    identity_state: int = 2,
    connected: int = 1,
    server_id: int = SERVER_ID,
) -> bytes:
    payload = bytearray(80)
    payload[:11] = golden_identity(slot, mac, state=identity_state)
    payload[11] = connected
    struct.pack_into("<I", payload, 12, packet_number)
    struct.pack_into("<Q", payload, 48, timestamp_us)
    struct.pack_into("<6f", payload, 56, *(acceleration + gyro))
    return golden_packet(b"DSUS", server_id, 0x100002, bytes(payload))


def decode_client_request(data: bytes) -> tuple[int, bytes]:
    """Validate requests independently inside the fake server."""

    assert len(data) >= 20
    magic, version, length, expected_crc, sender = struct.unpack_from("<4sHHII", data)
    assert magic == b"DSUC" and version == VERSION and sender == CLIENT_ID
    assert len(data) == 16 + length
    checking = bytearray(data)
    struct.pack_into("<I", checking, 8, 0)
    assert zlib.crc32(checking) & 0xFFFFFFFF == expected_crc
    return struct.unpack_from("<I", data, 16)[0], data[20:]


class FakeDSUServer(asyncio.DatagramProtocol):
    def __init__(
        self,
        identities: dict[int, bytes],
        *,
        answer: bool = True,
        max_protocol_version: int = VERSION,
    ) -> None:
        self.identities = identities
        self.answer = answer
        self.max_protocol_version = max_protocol_version
        self.transport: asyncio.DatagramTransport | None = None
        self.peer: tuple[str, int] | None = None
        self.requests: list[tuple[int, bytes]] = []
        self.registrations: list[int] = []

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.DatagramTransport, transport)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        message_type, payload = decode_client_request(data)
        self.requests.append((message_type, payload))
        self.peer = addr
        if not self.answer:
            return
        assert self.transport is not None
        if message_type == 0x100000:
            self.transport.sendto(
                golden_packet(
                    b"DSUS",
                    SERVER_ID,
                    0x100000,
                    struct.pack("<H", self.max_protocol_version),
                ),
                addr,
            )
        elif message_type == 0x100001:
            count = struct.unpack_from("<i", payload)[0]
            slots = payload[4 : 4 + count]
            assert len(slots) == count
            for slot in slots:
                mac = self.identities.get(slot)
                self.transport.sendto(
                    golden_info(slot, mac or b"\0" * 6, state=2 if mac else 0),
                    addr,
                )
        elif message_type == 0x100002:
            flags, slot = struct.unpack_from("<BB", payload)
            assert flags == 1 and payload[2:] == b"\0" * 6
            self.registrations.append(slot)

    def send_data(self, packet: bytes) -> None:
        assert self.transport is not None and self.peer is not None
        self.transport.sendto(packet, self.peer)


async def open_server(server: FakeDSUServer) -> tuple[asyncio.DatagramTransport, int]:
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: server,
        local_addr=("127.0.0.1", 0),
    )
    udp = cast(asyncio.DatagramTransport, transport)
    address = udp.get_extra_info("sockname")
    assert isinstance(address, tuple) and isinstance(address[1], int)
    return udp, address[1]


def test_request_codec_matches_independent_golden_packets() -> None:
    assert version_request(CLIENT_ID) == golden_packet(b"DSUC", CLIENT_ID, 0x100000, b"")
    assert controller_info_request([2, 0], CLIENT_ID) == golden_packet(
        b"DSUC", CLIENT_ID, 0x100001, struct.pack("<iBB", 2, 2, 0)
    )
    assert controller_data_request(2, CLIENT_ID) == golden_packet(
        b"DSUC", CLIENT_ID, 0x100002, struct.pack("<BB6x", 1, 2)
    )
    with pytest.raises(ValueError, match="distinct"):
        controller_info_request([0, 0], CLIENT_ID)
    with pytest.raises(ValueError, match="between 0 and 3"):
        controller_data_request(4, CLIENT_ID)


def test_response_parser_checks_length_crc_units_and_finite_motion() -> None:
    version = parse_response(golden_packet(b"DSUS", SERVER_ID, 0x100000, struct.pack("<H", VERSION)))
    assert version == ProtocolVersionResponse(SERVER_ID, VERSION)

    mac = bytes.fromhex("102030405060")
    info = parse_response(golden_info(2, mac) + b"ignored UDP trailer")
    assert isinstance(info, ControllerInfoResponse)
    assert info.identity.slot == 2 and info.identity.mac_address == "10:20:30:40:50:60"

    data = parse_response(golden_data(2, mac, 7, 1_234_567))
    assert isinstance(data, ControllerDataResponse)
    assert data.state.packet_number == 7 and data.state.motion_timestamp_us == 1_234_567
    assert data.state.acceleration_g.to_list() == [1.0, -2.0, 0.5]
    assert data.state.angular_velocity_dps.to_list() == [10.0, 20.0, -30.0]

    corrupt = bytearray(golden_info(2, mac))
    corrupt[-1] ^= 1
    with pytest.raises(DSUProtocolError, match="CRC mismatch"):
        parse_response(corrupt)

    short = bytearray(golden_info(2, mac))
    struct.pack_into("<H", short, 6, 100)
    with pytest.raises(DSUProtocolError, match="shorter than declared"):
        parse_response(short)

    nan_packet = golden_data(2, mac, 8, 2_000_000, acceleration=(math.nan, 0.0, 1.0))
    with pytest.raises(DSUProtocolError, match="finite"):
        parse_response(nan_packet)


def test_uint32_ordering_wrap_and_bounded_affine_clock() -> None:
    assert classify_packet(None, 0).accepted
    assert classify_packet(5, 5).duplicate
    assert classify_packet(5, 4).out_of_order
    assert classify_packet(0xFFFFFFFE, 1).estimated_loss == 2

    estimator = SlotClockEstimator(max_observations=4)
    estimator.observe(1_000_000, 11.001)
    estimator.observe(2_000_000, 12.001)
    estimate = estimator.observe(3_000_000, 13.001)
    assert estimate.scale == pytest.approx(1.0)
    assert estimate.offset_seconds == pytest.approx(10.001)
    assert estimate.to_monotonic_time(4_000_000) == pytest.approx(14.001)
    restarted = estimator.observe(10, 20.0)
    assert restarted.observations == 1


def test_strict_validation_and_malformed_response_matrix() -> None:
    with pytest.raises(ValueError, match="client_id"):
        version_request(-1)
    with pytest.raises(ValueError, match="one to four"):
        controller_info_request([], CLIENT_ID)
    with pytest.raises(ValueError, match="between 0 and 3"):
        controller_info_request([9], CLIENT_ID)
    with pytest.raises(DSUProtocolError, match="state"):
        ControllerIdentity(0, 9, 2, 2, b"123456", 5)
    with pytest.raises(DSUProtocolError, match="exactly 6"):
        ControllerIdentity(0, 2, 2, 2, b"short", 5)
    with pytest.raises(DSUProtocolError, match="model"):
        ControllerIdentity(0, 2, 256, 2, b"123456", 5)

    invalid_packets = [
        (b"short", "16-byte header"),
        (golden_packet(b"NOPE", SERVER_ID, 0x100000, b"\xe9\x03"), "magic"),
        (
            golden_packet(b"DSUS", SERVER_ID, 0x100000, b"\xe9\x03").replace(
                struct.pack("<H", VERSION), struct.pack("<H", 999), 1
            ),
            "protocol version",
        ),
        (golden_packet(b"DSUS", SERVER_ID, 0x999999, b""), "message type"),
        (golden_packet(b"DSUS", SERVER_ID, 0x100000, b""), "exactly 2"),
        (golden_packet(b"DSUS", SERVER_ID, 0x100001, b""), "exactly 12"),
        (
            golden_packet(
                b"DSUS",
                SERVER_ID,
                0x100001,
                golden_identity(0, b"123456") + b"x",
            ),
            "reserved byte",
        ),
        (golden_packet(b"DSUS", SERVER_ID, 0x100002, b""), "exactly 80"),
    ]
    # The version replacement invalidates CRC first; rebuild that one explicitly.
    invalid_packets[2] = (
        golden_packet(b"DSUS", SERVER_ID, 0x100000, b"\xe9\x03")[:4]
        + struct.pack("<H", 999)
        + golden_packet(b"DSUS", SERVER_ID, 0x100000, b"\xe9\x03")[6:],
        "protocol version",
    )
    for packet, message in invalid_packets:
        with pytest.raises(DSUProtocolError, match=message):
            parse_response(packet)

    too_small_declared = bytearray(16)
    struct.pack_into("<4sHHII", too_small_declared, 0, b"DSUS", VERSION, 3, 0, SERVER_ID)
    with pytest.raises(DSUProtocolError, match="does not include"):
        parse_response(too_small_declared)

    bad_connected = bytearray(80)
    bad_connected[:11] = golden_identity(0, b"123456")
    bad_connected[11] = 2
    with pytest.raises(DSUProtocolError, match="connected flag"):
        parse_response(golden_packet(b"DSUS", SERVER_ID, 0x100002, bytes(bad_connected)))

    bad_slot = bytearray(80)
    bad_slot[:11] = golden_identity(7, b"123456")
    bad_slot[11] = 1
    with pytest.raises(DSUProtocolError, match="between 0 and 3"):
        parse_response(golden_packet(b"DSUS", SERVER_ID, 0x100002, bytes(bad_slot)))


def test_capture_configuration_clock_and_health_contracts() -> None:
    invalid_configs = [
        ({"host": " "}, "host"),
        ({"port": 0}, "port"),
        ({"left_slot": 4}, "left"),
        ({"left_slot": 1, "right_slot": 1}, "distinct"),
        ({"queue_size": 0}, "queue"),
        ({"discovery_timeout": math.inf}, "timeout"),
        ({"refresh_interval": 0.0}, "refresh"),
        ({"client_id": -1}, "client_id"),
    ]
    for values, message in invalid_configs:
        with pytest.raises(ValueError, match=message):
            DSUCaptureConfig(**values)  # type: ignore[arg-type]

    estimate = ClockEstimate(1.0, 2.0, 1)
    with pytest.raises(ValueError, match="non-negative"):
        estimate.to_monotonic_time(-1)
    with pytest.raises(ValueError, match="at least two"):
        SlotClockEstimator(1)
    estimator = SlotClockEstimator()
    with pytest.raises(ValueError, match="finite"):
        estimator.observe(-1, math.nan)
    assert estimator.estimate is None

    with pytest.raises(ValueError, match="packet number"):
        classify_packet(None, -1)
    with pytest.raises(ValueError, match="previous"):
        classify_packet(-1, 0)

    slot = SlotStreamHealth(
        WristSide.LEFT,
        0,
        True,
        False,
        8,
        1,
        1,
        1,
        1,
        7,
        1.0,
        50.0,
        estimate,
    )
    assert slot.quality == pytest.approx(2 / 3)
    health = CaptureHealth(True, SERVER_ID, VERSION, 0, 0, 0, (slot,))
    assert health.healthy
    assert not CaptureHealth(False, None, None, 0, 0, 0, (slot,)).healthy


def test_live_two_slot_discovery_stream_health_refresh_and_cleanup() -> None:
    async def scenario() -> None:
        left_mac = bytes.fromhex("010203040506")
        right_mac = bytes.fromhex("a1a2a3a4a5a6")
        server = FakeDSUServer({2: left_mac, 0: right_mac})
        server_transport, port = await open_server(server)
        capture = DSUCapture(
            DSUCaptureConfig(
                port=port,
                left_slot=2,
                right_slot=0,
                queue_size=8,
                discovery_timeout=0.5,
                refresh_interval=0.02,
                stale_after=1.0,
                client_id=CLIENT_ID,
            )
        )
        try:
            await capture.start()
            await asyncio.sleep(0.03)
            assert capture.identities[WristSide.LEFT].slot == 2
            assert capture.identities[WristSide.RIGHT].slot == 0
            assert {0, 2}.issubset(server.registrations)
            assert sum(message == MessageType.CONTROLLER_INFO for message, _ in server.requests) >= 2

            packets = [
                golden_data(2, left_mac, 0xFFFFFFFE, 1_000_000),
                golden_data(2, left_mac, 0xFFFFFFFF, 1_010_000),
                golden_data(2, left_mac, 0, 1_020_000),
                golden_data(2, left_mac, 0, 1_020_000),  # duplicate
                golden_data(2, left_mac, 0xFFFFFFFF, 1_010_000),  # stale after wrap
                golden_data(2, left_mac, 2, 1_040_000),  # one estimated loss
                golden_data(0, right_mac, 10, 2_000_000),
            ]
            for packet in packets:
                server.send_data(packet)
            samples = [await capture.receive(0.5) for _ in range(5)]
            assert [sample.packet_number for sample in samples] == [0xFFFFFFFE, 0xFFFFFFFF, 0, 2, 10]
            assert samples[0].device_id == "left:01:02:03:04:05:06"
            assert samples[-1].device_id == "right:a1:a2:a3:a4:a5:a6"

            health = capture.health
            left = next(slot for slot in health.slots if slot.wrist is WristSide.LEFT)
            assert health.running and health.protocol_version == VERSION
            assert left.accepted == 4 and left.duplicates == 1
            assert left.out_of_order == 1 and left.estimated_loss == 1
            assert left.clock is not None and left.clock.observations == 4
            assert capture.clock_estimate(WristSide.LEFT) == left.clock
        finally:
            await capture.stop()
            server_transport.close()
            await asyncio.sleep(0)
        assert not capture.running
        with pytest.raises(StopAsyncIteration):
            await capture.receive(0.1)

    asyncio.run(scenario())


def test_live_right_only_discovery_registration_and_health() -> None:
    async def scenario() -> None:
        right_mac = bytes.fromhex("a1a2a3a4a5a6")
        server = FakeDSUServer({3: right_mac})
        server_transport, port = await open_server(server)
        capture = DSUCapture(
            DSUCaptureConfig(
                port=port,
                right_slot=3,
                discovery_timeout=0.5,
                refresh_interval=1.0,
                stale_after=1.0,
                client_id=CLIENT_ID,
            )
        )
        try:
            await capture.start()
            await asyncio.sleep(0.01)
            assert capture.config.left_slot is None
            assert capture.identities[WristSide.RIGHT].slot == 3
            assert server.registrations == [3]
            info_requests = [payload for message, payload in server.requests if message == MessageType.CONTROLLER_INFO]
            assert all(payload == struct.pack("<iB", 1, 3) for payload in info_requests)
            server.send_data(golden_data(3, right_mac, 1, 1_000_000))
            sample = await capture.receive(0.5)
            assert sample.device_id == "right:a1:a2:a3:a4:a5:a6"
            assert [slot.wrist for slot in capture.health.slots] == [WristSide.RIGHT]
            assert capture.clock_estimate(WristSide.LEFT) is None
        finally:
            await capture.stop()
            server_transport.close()
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_bounded_queue_discards_oldest_and_discovery_timeout_closes() -> None:
    async def scenario() -> None:
        left_mac = bytes.fromhex("010203040506")
        right_mac = bytes.fromhex("111213141516")
        server = FakeDSUServer({0: left_mac, 1: right_mac})
        transport, port = await open_server(server)
        capture = DSUCapture(
            DSUCaptureConfig(
                port=port,
                left_slot=0,
                right_slot=1,
                queue_size=2,
                discovery_timeout=0.3,
                client_id=CLIENT_ID,
            )
        )
        try:
            await capture.start()
            await asyncio.sleep(0)
            for number in range(4):
                server.send_data(golden_data(0, left_mac, number, 1_000_000 + number * 10_000))
            await asyncio.sleep(0.02)
            assert capture.health.queue_depth == 2
            assert capture.health.slots[0].queue_dropped == 2
            assert [await capture.receive(0.2), await capture.receive(0.2)][0].packet_number == 2
        finally:
            await capture.stop()
            transport.close()

        silent = FakeDSUServer({}, answer=False)
        silent_transport, silent_port = await open_server(silent)
        timed_out = DSUCapture(
            DSUCaptureConfig(
                port=silent_port,
                discovery_timeout=0.04,
                refresh_interval=1.0,
                client_id=CLIENT_ID,
            )
        )
        try:
            with pytest.raises(CaptureError, match="timed out discovering"):
                await timed_out.start()
            assert not timed_out.running
        finally:
            await timed_out.stop()
            silent_transport.close()
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_capture_public_callbacks_reject_bad_sources_and_handle_disconnect() -> None:
    async def scenario() -> None:
        capture = DSUCapture(DSUCaptureConfig(left_slot=0, right_slot=1, client_id=CLIENT_ID, queue_size=4))
        assert capture.client_id == CLIENT_ID
        capture.on_datagram(b"not DSU")
        capture.on_transport_error(OSError("unreachable"))
        assert capture.health.invalid_packets == 1
        assert capture.health.transport_errors == 1

        capture.on_datagram(golden_packet(b"DSUS", SERVER_ID, 0x100000, struct.pack("<H", VERSION)))
        # An unassigned slot is ignored.
        capture.on_datagram(golden_info(3, b"abcdef"))
        mac = b"123456"
        capture.on_datagram(golden_info(0, mac))
        # Disconnected data and identity mismatches never enter the queue.
        capture.on_datagram(golden_data(0, mac, 1, 1, identity_state=0, connected=0))
        capture.on_datagram(golden_data(0, b"654321", 1, 1))
        assert capture.health.invalid_packets == 2
        assert capture.health.queue_depth == 0

        # A server-ID change resets prior identities and packet/clock epochs.
        replacement_payload = golden_identity(1, b"ABCDEF") + b"\0"
        capture.on_datagram(golden_packet(b"DSUS", SERVER_ID + 1, 0x100001, replacement_payload))
        assert WristSide.LEFT not in capture.identities
        assert capture.identities[WristSide.RIGHT].mac == b"ABCDEF"

        zero_mac_capture = DSUCapture(DSUCaptureConfig(right_slot=0, client_id=CLIENT_ID))
        zero_mac_capture.on_datagram(golden_data(0, b"\0" * 6, 1, 50))
        sample = await zero_mac_capture.receive()
        assert sample.device_id == "right:slot-0"
        # Vector operations also prove parsed native units remain domain vectors.
        assert (sample.acceleration_g + sample.angular_velocity_dps).x == 11.0
        zero_mac_capture.on_connection_lost(RuntimeError("closed"))
        assert not zero_mac_capture.running
        assert zero_mac_capture.health.transport_errors == 1
        assert [item async for item in zero_mac_capture] == []

    asyncio.run(scenario())


def test_async_context_iteration_restart_and_old_protocol_cleanup() -> None:
    async def scenario() -> None:
        left_mac = bytes.fromhex("010203040506")
        right_mac = bytes.fromhex("111213141516")
        server = FakeDSUServer({0: left_mac, 1: right_mac})
        transport, port = await open_server(server)
        capture = DSUCapture(
            DSUCaptureConfig(port=port, left_slot=0, right_slot=1, discovery_timeout=0.3, client_id=CLIENT_ID)
        )
        try:
            async with capture as running:
                await running.start()  # Idempotent while already running.
                server.send_data(golden_data(1, right_mac, 1, 100))
                iterator = running.__aiter__()
                assert (await anext(iterator)).device_id.startswith("right:")
                with pytest.raises(ValueError, match="timeout"):
                    await running.receive(0)
            with pytest.raises(StopAsyncIteration):
                await anext(iterator)

            # A stopped instance starts with a fresh queue and discovery epoch.
            await capture.start()
            assert capture.running
            await asyncio.gather(capture.stop(), capture.stop())
            assert not capture.running
        finally:
            await capture.stop()
            transport.close()
            await asyncio.sleep(0)

        old = FakeDSUServer(
            {0: left_mac, 1: right_mac},
            max_protocol_version=VERSION - 1,
        )
        old_transport, old_port = await open_server(old)
        unsupported = DSUCapture(
            DSUCaptureConfig(port=old_port, left_slot=0, right_slot=1, discovery_timeout=0.3, client_id=CLIENT_ID)
        )
        try:
            with pytest.raises(CaptureError, match="supports protocol"):
                await unsupported.start()
            assert not unsupported.running
        finally:
            await unsupported.stop()
            old_transport.close()
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_decoded_samples_integrate_with_raw_motion_boundary() -> None:
    vector = Vector3.from_value([3, 4.0, 0])
    assert vector.norm() == 5
    assert vector.dot(Vector3(1, 0, 0)) == 3
    assert vector.scale(2).to_list() == [6.0, 8.0, 0.0]
    assert (vector - Vector3(1, 1, 0)).to_list() == [2.0, 3.0, 0.0]
    assert Vector3(0, 0, 0).normalized() == Vector3(0, 0, 0)
    for invalid in ("not-list", [1, 2], [1, True, 3]):
        with pytest.raises(ValueError, match="vector"):
            Vector3.from_value(invalid)
    with pytest.raises(ValueError, match="finite"):
        Vector3(math.inf, 0, 0)

    acceleration = Vector3(0, 0, 1)
    gyro = Vector3(1, 2, 3)
    boundary = RawMotionSample(WristSide.LEFT, 10, 2.5, 7, acceleration, gyro)
    raw = boundary.raw_sample()
    assert raw == RawImuSample("left", 10, 7, acceleration, gyro)
    with pytest.raises(ValueError, match="capture timestamp"):
        RawMotionSample(WristSide.LEFT, -1, 0, 0, acceleration, gyro)
    with pytest.raises(ValueError, match="clientTimestamp"):
        RawMotionSample(WristSide.LEFT, 0, math.nan, 0, acceleration, gyro)
    with pytest.raises(ValueError, match="device_id"):
        RawImuSample(" ", 0, 0, acceleration, gyro)
    with pytest.raises(ValueError, match="non-negative"):
        RawImuSample("left", -1, 0, acceleration, gyro)

    with pytest.raises(ValueError, match="finite"):
        MotionFeatures(math.nan, WristSide.LEFT, 0, 0, 1, 0, False)
    with pytest.raises(ValueError, match="horizontal direction"):
        MotionFeatures(0, WristSide.LEFT, 0, math.inf, 1, 0, False)
    with pytest.raises(ValueError, match="non-negative"):
        MotionFeatures(-1, WristSide.LEFT, 0, 0, 1, 0, False)
    with pytest.raises(ValueError, match="between zero and one"):
        MotionFeatures(0, WristSide.LEFT, 0, 0, 2, 0, False)

    with pytest.raises(ValueError, match="sample rate"):
        SimulationConfig(sample_rate_hz=0)
    with pytest.raises(ValueError, match="invalid simulation"):
        SimulationConfig(packet_loss=1.0)
    simulator = DeterministicMotionSimulator(SimulationConfig(duration_seconds=0.05, packet_loss=0.2, reorder_every=2))
    consumed: list[RawImuSample] = []

    def stop_after_one(sample: RawImuSample) -> None:
        consumed.append(sample)
        simulator.stop()

    simulator.start(stop_after_one)
    assert len(consumed) == 1
    with pytest.raises(ValueError, match="retention"):
        CircularMotionBuffer(0)
    buffer = CircularMotionBuffer()
    assert buffer.health.quality == 1.0
    assert buffer.add(raw)
    assert buffer.between(0, 20, "left") == (raw,)
    older = RawImuSample("left", 11, 6, acceleration, gyro)
    assert not buffer.add(older)
    assert buffer.health.out_of_order == 1
