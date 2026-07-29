"""Strict little-endian Cemuhook/DSU v1001 wire codec.

The functions in this module intentionally operate on complete UDP datagrams.  A
response with trailing bytes is parsed only up to its declared DSU length, as
required by the protocol, while short, malformed, or CRC-invalid packets are
rejected.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

from dancify.domain import Vector3

PROTOCOL_VERSION = 1001
HEADER_SIZE = 16
MAX_SLOT = 3
CLIENT_MAGIC = b"DSUC"
SERVER_MAGIC = b"DSUS"

_HEADER = struct.Struct("<4sHHII")
_MESSAGE_TYPE = struct.Struct("<I")
_IDENTITY = struct.Struct("<BBBB6sB")


class DSUProtocolError(ValueError):
    """A DSU datagram or request violates the v1001 wire contract."""


class MessageType(IntEnum):
    VERSION = 0x100000
    CONTROLLER_INFO = 0x100001
    CONTROLLER_DATA = 0x100002


@dataclass(frozen=True, slots=True)
class DSUPacket:
    """A CRC-checked DSUS packet, truncated to its declared wire length."""

    server_id: int
    message_type: MessageType
    payload: bytes


@dataclass(frozen=True, slots=True)
class ControllerIdentity:
    """Stable controller identity and connection metadata reported by DSU."""

    slot: int
    state: int
    model: int
    connection_type: int
    mac: bytes
    battery: int

    def __post_init__(self) -> None:
        _validate_slot(self.slot)
        if self.state not in (0, 1, 2):
            raise DSUProtocolError(f"invalid controller state {self.state}")
        if len(self.mac) != 6:
            raise DSUProtocolError("controller MAC must be exactly 6 bytes")
        for name, value in (
            ("model", self.model),
            ("connection type", self.connection_type),
            ("battery", self.battery),
        ):
            if not 0 <= value <= 0xFF:
                raise DSUProtocolError(f"invalid {name} {value}")

    @property
    def connected(self) -> bool:
        return self.state == 2

    @property
    def mac_address(self) -> str:
        return ":".join(f"{octet:02x}" for octet in self.mac)


@dataclass(frozen=True, slots=True)
class ControllerState:
    """One controller-data response with native DSU motion units."""

    identity: ControllerIdentity
    connected: bool
    packet_number: int
    motion_timestamp_us: int
    acceleration_g: Vector3
    angular_velocity_dps: Vector3

    @property
    def slot(self) -> int:
        return self.identity.slot


@dataclass(frozen=True, slots=True)
class ProtocolVersionResponse:
    server_id: int
    max_protocol_version: int


@dataclass(frozen=True, slots=True)
class ControllerInfoResponse:
    server_id: int
    identity: ControllerIdentity


@dataclass(frozen=True, slots=True)
class ControllerDataResponse:
    server_id: int
    state: ControllerState


DSUResponse = ProtocolVersionResponse | ControllerInfoResponse | ControllerDataResponse


def _validate_u32(value: int, name: str) -> None:
    if isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be an unsigned 32-bit integer")


def _validate_slot(slot: int) -> None:
    if isinstance(slot, bool) or not 0 <= slot <= MAX_SLOT:
        raise ValueError("DSU slot must be between 0 and 3")


def _request(message_type: MessageType, payload: bytes, client_id: int) -> bytes:
    _validate_u32(client_id, "client_id")
    body = _MESSAGE_TYPE.pack(message_type) + payload
    packet = bytearray(_HEADER.pack(CLIENT_MAGIC, PROTOCOL_VERSION, len(body), 0, client_id) + body)
    struct.pack_into("<I", packet, 8, zlib.crc32(packet) & 0xFFFFFFFF)
    return bytes(packet)


def version_request(client_id: int) -> bytes:
    """Build a protocol-version request."""

    return _request(MessageType.VERSION, b"", client_id)


def controller_info_request(slots: tuple[int, ...] | list[int], client_id: int) -> bytes:
    """Build an info request for one to four distinct, explicit slots."""

    requested = tuple(slots)
    if not 1 <= len(requested) <= 4:
        raise ValueError("controller info requires one to four slots")
    if len(set(requested)) != len(requested):
        raise ValueError("controller info slots must be distinct")
    for slot in requested:
        _validate_slot(slot)
    return _request(MessageType.CONTROLLER_INFO, struct.pack("<i", len(requested)) + bytes(requested), client_id)


def controller_data_request(slot: int, client_id: int) -> bytes:
    """Build a slot-based data registration request.

    Deliberately requiring a slot avoids DSU's implicit subscribe-to-all mode,
    which could silently swap wrists when controllers reconnect.
    """

    _validate_slot(slot)
    return _request(MessageType.CONTROLLER_DATA, struct.pack("<BB6x", 1, slot), client_id)


def parse_packet(datagram: bytes | bytearray | memoryview) -> DSUPacket:
    """Validate and decode the common DSUS envelope.

    DSU permits UDP datagrams to contain bytes beyond the packet's declared
    size.  Those bytes are ignored and are not included in CRC validation.
    """

    raw = bytes(datagram)
    if len(raw) < HEADER_SIZE:
        raise DSUProtocolError("DSU response is shorter than its 16-byte header")
    magic, version, declared_length, expected_crc, server_id = _HEADER.unpack_from(raw)
    if magic != SERVER_MAGIC:
        raise DSUProtocolError("DSU response has invalid magic")
    if version != PROTOCOL_VERSION:
        raise DSUProtocolError(f"unsupported DSU protocol version {version}")
    if declared_length < _MESSAGE_TYPE.size:
        raise DSUProtocolError("DSU declared length does not include a message type")
    packet_size = HEADER_SIZE + declared_length
    if len(raw) < packet_size:
        raise DSUProtocolError(f"DSU response is shorter than declared length ({len(raw)} < {packet_size})")
    packet = bytearray(raw[:packet_size])
    struct.pack_into("<I", packet, 8, 0)
    actual_crc = zlib.crc32(packet) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise DSUProtocolError(f"DSU response CRC mismatch (expected {expected_crc:#010x}, got {actual_crc:#010x})")
    raw_type = _MESSAGE_TYPE.unpack_from(packet, HEADER_SIZE)[0]
    try:
        message_type = MessageType(raw_type)
    except ValueError as exc:
        raise DSUProtocolError(f"unsupported DSU message type {raw_type:#x}") from exc
    return DSUPacket(server_id, message_type, bytes(packet[HEADER_SIZE + _MESSAGE_TYPE.size :]))


def parse_response(datagram: bytes | bytearray | memoryview) -> DSUResponse:
    """Parse a complete DSUS response into its semantic v1001 type."""

    packet = parse_packet(datagram)
    payload = packet.payload
    if packet.message_type is MessageType.VERSION:
        if len(payload) != 2:
            raise DSUProtocolError("version response payload must be exactly 2 bytes")
        return ProtocolVersionResponse(packet.server_id, struct.unpack("<H", payload)[0])
    if packet.message_type is MessageType.CONTROLLER_INFO:
        if len(payload) != 12:
            raise DSUProtocolError("controller-info payload must be exactly 12 bytes")
        if payload[11] != 0:
            raise DSUProtocolError("controller-info reserved byte must be zero")
        return ControllerInfoResponse(packet.server_id, _parse_identity(payload))
    if len(payload) != 80:
        raise DSUProtocolError("controller-data payload must be exactly 80 bytes")
    identity = _parse_identity(payload)
    connected = payload[11]
    if connected not in (0, 1):
        raise DSUProtocolError("controller-data connected flag must be zero or one")
    packet_number = struct.unpack_from("<I", payload, 12)[0]
    motion_timestamp_us = struct.unpack_from("<Q", payload, 48)[0]
    values = struct.unpack_from("<6f", payload, 56)
    if not all(isfinite(value) for value in values):
        raise DSUProtocolError("controller motion values must all be finite")
    try:
        acceleration = Vector3(*values[:3])
        angular_velocity = Vector3(*values[3:])
    except ValueError as exc:  # Preserve one stable protocol-level exception.
        raise DSUProtocolError(str(exc)) from exc
    return ControllerDataResponse(
        packet.server_id,
        ControllerState(
            identity,
            bool(connected),
            packet_number,
            motion_timestamp_us,
            acceleration,
            angular_velocity,
        ),
    )


def _parse_identity(payload: bytes) -> ControllerIdentity:
    slot, state, model, connection_type, mac, battery = _IDENTITY.unpack_from(payload)
    try:
        return ControllerIdentity(slot, state, model, connection_type, mac, battery)
    except ValueError as exc:
        if isinstance(exc, DSUProtocolError):
            raise
        raise DSUProtocolError(str(exc)) from exc


# Readable compatibility aliases for callers that prefer build/decode verbs.
build_version_request = version_request
build_controller_info_request = controller_info_request
build_controller_data_request = controller_data_request
decode_response = parse_response

__all__ = [
    "CLIENT_MAGIC",
    "HEADER_SIZE",
    "MAX_SLOT",
    "PROTOCOL_VERSION",
    "SERVER_MAGIC",
    "ControllerDataResponse",
    "ControllerIdentity",
    "ControllerInfoResponse",
    "ControllerState",
    "DSUPacket",
    "DSUProtocolError",
    "DSUResponse",
    "MessageType",
    "ProtocolVersionResponse",
    "build_controller_data_request",
    "build_controller_info_request",
    "build_version_request",
    "controller_data_request",
    "controller_info_request",
    "decode_response",
    "parse_packet",
    "parse_response",
    "version_request",
]
