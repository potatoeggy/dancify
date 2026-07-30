"""Timing and spatial calibration plus IMU-to-feature translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import exp, hypot, isfinite

from dancify.domain import MotionFeatures, RawImuSample, Vector3, WristSide


@dataclass(frozen=True, slots=True)
class ClockObservation:
    client_send: float
    server_receive: float
    server_send: float
    client_receive: float

    def __post_init__(self) -> None:
        values = (self.client_send, self.server_receive, self.server_send, self.client_receive)
        if not all(isfinite(value) and value >= 0 for value in values):
            raise ValueError("clock observation timestamps must be finite and non-negative")
        if self.client_receive < self.client_send or self.server_send < self.server_receive:
            raise ValueError("clock observation timestamps are not ordered")
        if self.round_trip < 0:
            raise ValueError("clock observation has negative network round trip")

    @property
    def round_trip(self) -> float:
        return (self.client_receive - self.client_send) - (self.server_send - self.server_receive)

    @property
    def offset(self) -> float:
        return ((self.server_receive - self.client_send) + (self.server_send - self.client_receive)) / 2.0


class TimingCalibrationService:
    @staticmethod
    def estimate_offset(observations: list[ClockObservation]) -> float:
        if not observations:
            raise ValueError("at least one clock observation is required")
        usable = sorted(observations, key=lambda item: item.round_trip)
        keep = usable[: max(1, len(usable) // 2)]
        return sum(item.offset for item in keep) / len(keep)


@dataclass(frozen=True, slots=True)
class AffineClockMapper:
    scale: float = 1.0
    offset_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.scale) or self.scale <= 0 or not isfinite(self.offset_seconds):
            raise ValueError("clock mapper scale and offset must be finite; scale must be positive")

    def to_server_time(self, device_timestamp_us: int) -> float:
        return self.scale * device_timestamp_us / 1_000_000.0 + self.offset_seconds


@dataclass(slots=True)
class CaptureClockMapper:
    """Estimate capture-clock to client-clock mapping from timestamp pairs.

    Each DSU sample carries its controller capture time and the client's monotonic
    time observed for that packet. A bounded least-squares fit accommodates normal
    clock drift while rejecting implausible fits caused by bursty arrival jitter.
    """

    max_observations: int = 64
    _observations: list[tuple[float, float]] = field(default_factory=list[tuple[float, float]])
    scale: float = 1.0
    offset_seconds: float = 0.0

    def observe(self, capture_timestamp_us: int, client_timestamp: float) -> float:
        if capture_timestamp_us < 0 or not isfinite(client_timestamp) or client_timestamp < 0:
            raise ValueError("capture/client timestamps must be finite and non-negative")
        capture = capture_timestamp_us / 1_000_000.0
        self._observations.append((capture, client_timestamp))
        if len(self._observations) > self.max_observations:
            del self._observations[: len(self._observations) - self.max_observations]
        self._fit()
        return self.to_client_time(capture_timestamp_us)

    def to_client_time(self, capture_timestamp_us: int) -> float:
        if capture_timestamp_us < 0:
            raise ValueError("capture timestamp must be non-negative")
        return self.scale * capture_timestamp_us / 1_000_000.0 + self.offset_seconds

    def _fit(self) -> None:
        if not self._observations:
            return
        if len(self._observations) == 1:
            capture, client = self._observations[0]
            self.scale = 1.0
            self.offset_seconds = client - capture
            return
        mean_capture = sum(item[0] for item in self._observations) / len(self._observations)
        mean_client = sum(item[1] for item in self._observations) / len(self._observations)
        variance = sum((item[0] - mean_capture) ** 2 for item in self._observations)
        covariance = sum((capture - mean_capture) * (client - mean_client) for capture, client in self._observations)
        fitted_scale = covariance / variance if variance > 1e-12 else 1.0
        # Independent monotonic clocks drift slowly. Reject a fit dominated by
        # network/queue jitter rather than allowing timestamps to jump wildly.
        self.scale = fitted_scale if 0.95 <= fitted_scale <= 1.05 else 1.0
        self.offset_seconds = sum(client - self.scale * capture for capture, client in self._observations) / len(
            self._observations
        )


@dataclass(frozen=True, slots=True)
class SpatialCalibrationProfile:
    gravity: Vector3
    vertical_axis: Vector3
    horizontal_axis: Vector3
    horizontal_confidence: float
    calibrated_at: float = 0.0

    @classmethod
    def from_gestures(
        cls,
        neutral: list[Vector3],
        upward: list[Vector3],
        outward: list[Vector3],
        calibrated_at: float = 0.0,
    ) -> SpatialCalibrationProfile:
        if not neutral or not upward or not outward:
            raise ValueError("calibration gestures require neutral, upward, and outward samples")
        gravity = _mean(neutral)
        vertical = (_mean(upward) - gravity).normalized()
        horizontal = (_mean(outward) - gravity).normalized()
        orthogonality = 1.0 - min(1.0, abs(vertical.dot(horizontal)))
        if vertical.norm() < 0.9 or horizontal.norm() < 0.9 or orthogonality < 0.2:
            raise ValueError("calibration gestures do not define reliable axes")
        return cls(gravity, vertical, horizontal, orthogonality, calibrated_at)

    def translate(
        self,
        sample: RawImuSample,
        wrist: WristSide,
        mapper: AffineClockMapper | None = None,
    ) -> MotionFeatures:
        if mapper is None:
            mapper = AffineClockMapper()
        linear = sample.acceleration_g - self.gravity
        vertical_value = linear.dot(self.vertical_axis)
        horizontal_value = linear.dot(self.horizontal_axis)
        intensity = hypot(vertical_value, horizontal_value)
        elapsed = max(0.0, mapper.to_server_time(sample.device_timestamp_us) - self.calibrated_at)
        confidence = self.horizontal_confidence * exp(-elapsed / 300.0)
        active = intensity >= 0.03
        if active:
            vertical_direction = vertical_value / intensity
            horizontal_direction: float | None = horizontal_value / intensity if confidence >= 0.25 else None
        else:
            vertical_direction = 0.0
            horizontal_direction = 0.0 if confidence >= 0.25 else None
        return MotionFeatures(
            synchronized_time=mapper.to_server_time(sample.device_timestamp_us),
            wrist=wrist,
            vertical_direction=vertical_direction,
            horizontal_direction=horizontal_direction,
            horizontal_confidence=max(0.0, min(1.0, confidence)),
            linear_intensity=intensity,
            movement_active=active,
        )


class ResamplingMode(StrEnum):
    """Choose emitted timestamps without changing target-grid sequence order."""

    TARGET_GRID = "target_grid"
    SOURCE_SAMPLE = "source_sample"


def resample_features(
    samples: tuple[MotionFeatures, ...],
    start: float,
    end: float,
    rate_hz: int = 50,
    max_gap_seconds: float = 0.05,
    timestamp_mode: ResamplingMode = ResamplingMode.TARGET_GRID,
) -> tuple[MotionFeatures, ...]:
    if rate_hz <= 0 or end <= start:
        raise ValueError("invalid resampling interval")
    streams = {
        wrist: sorted(
            (sample for sample in samples if sample.wrist == wrist),
            key=lambda item: item.synchronized_time,
        )
        for wrist in WristSide
    }
    output: list[MotionFeatures] = []
    for step in range(int((end - start) * rate_hz)):
        target_timestamp = start + step / rate_hz
        for wrist in WristSide:
            stream = streams[wrist]
            if not stream:
                continue
            nearest = min(stream, key=lambda item: abs(item.synchronized_time - target_timestamp))
            gap = abs(nearest.synchronized_time - target_timestamp)
            if gap > max_gap_seconds:
                continue
            quality = max(0.0, 1.0 - gap / max_gap_seconds)
            emitted_timestamp = (
                target_timestamp if timestamp_mode is ResamplingMode.TARGET_GRID else nearest.synchronized_time
            )
            output.append(
                MotionFeatures(
                    emitted_timestamp,
                    wrist,
                    nearest.vertical_direction,
                    nearest.horizontal_direction,
                    nearest.horizontal_confidence,
                    nearest.linear_intensity,
                    nearest.movement_active,
                    min(nearest.sample_quality, quality),
                )
            )
    return tuple(output)


def _mean(values: list[Vector3]) -> Vector3:
    count = float(len(values))
    return Vector3(
        sum(value.x for value in values) / count,
        sum(value.y for value in values) / count,
        sum(value.z for value in values) / count,
    )
