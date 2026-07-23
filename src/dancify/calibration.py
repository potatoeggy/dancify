"""Timing and spatial calibration plus IMU-to-feature translation."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, hypot

from dancify.domain import MotionFeatures, RawImuSample, Vector3, WristSide


@dataclass(frozen=True, slots=True)
class ClockObservation:
    client_send: float
    server_receive: float
    server_send: float
    client_receive: float

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

    def to_server_time(self, device_timestamp_us: int) -> float:
        return self.scale * device_timestamp_us / 1_000_000.0 + self.offset_seconds


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


def resample_features(
    samples: tuple[MotionFeatures, ...],
    start: float,
    end: float,
    rate_hz: int = 50,
    max_gap_seconds: float = 0.05,
) -> tuple[MotionFeatures, ...]:
    if rate_hz <= 0 or end <= start:
        raise ValueError("invalid resampling interval")
    output: list[MotionFeatures] = []
    for wrist in WristSide:
        stream = sorted(
            (sample for sample in samples if sample.wrist == wrist), key=lambda item: item.synchronized_time
        )
        if not stream:
            continue
        for step in range(int((end - start) * rate_hz)):
            timestamp = start + step / rate_hz
            nearest = min(stream, key=lambda item: abs(item.synchronized_time - timestamp))
            gap = abs(nearest.synchronized_time - timestamp)
            if gap > max_gap_seconds:
                continue
            quality = max(0.0, 1.0 - gap / max_gap_seconds)
            output.append(
                MotionFeatures(
                    timestamp,
                    wrist,
                    nearest.vertical_direction,
                    nearest.horizontal_direction,
                    nearest.horizontal_confidence,
                    nearest.linear_intensity,
                    nearest.movement_active,
                    min(nearest.sample_quality, quality),
                )
            )
    return tuple(sorted(output, key=lambda item: (item.synchronized_time, item.wrist.value)))


def _mean(values: list[Vector3]) -> Vector3:
    count = float(len(values))
    return Vector3(
        sum(value.x for value in values) / count,
        sum(value.y for value in values) / count,
        sum(value.z for value in values) / count,
    )
