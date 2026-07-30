import pytest

from dancify.calibration import (
    AffineClockMapper,
    ClockObservation,
    ResamplingMode,
    SpatialCalibrationProfile,
    TimingCalibrationService,
    resample_features,
)
from dancify.domain import MotionFeatures, RawImuSample, Vector3, WristSide
from dancify.motion import CircularMotionBuffer, DeterministicMotionSimulator, SimulationConfig


def test_timing_calibration_prefers_low_rtt_and_maps_device_time() -> None:
    observations = [ClockObservation(0, 0.11, 0.12, 0.30), ClockObservation(1, 1.01, 1.02, 1.03)]
    offset = TimingCalibrationService.estimate_offset(observations)
    assert offset == pytest.approx(0.0)
    assert AffineClockMapper(1.001, 2.0).to_server_time(1_000_000) == pytest.approx(3.001)
    with pytest.raises(ValueError):
        TimingCalibrationService.estimate_offset([])


def test_guided_projection_and_confidence_fallback() -> None:
    profile = SpatialCalibrationProfile.from_gestures([Vector3(0, 0, 1)], [Vector3(0, 1, 1)], [Vector3(1, 0, 1)])
    raw = RawImuSample("left", 1_000_000, 1, Vector3(1, 1, 1), Vector3(0, 0, 0))
    translated = profile.translate(raw, WristSide.LEFT)
    assert translated.horizontal_direction is not None
    assert translated.vertical_direction == pytest.approx(2**-0.5)
    late = profile.translate(raw, WristSide.LEFT, AffineClockMapper(scale=1000.0))
    assert late.horizontal_direction is None
    resampled = resample_features((translated,), 0.95, 1.05, rate_hz=10, max_gap_seconds=0.1)
    assert resampled
    with pytest.raises(ValueError, match="gestures"):
        SpatialCalibrationProfile.from_gestures([], [], [])


def test_simulator_buffer_order_loss_and_bound() -> None:
    simulator = DeterministicMotionSimulator(SimulationConfig(seed=4, duration_seconds=0.1))
    assert simulator.samples() == DeterministicMotionSimulator(SimulationConfig(seed=4, duration_seconds=0.1)).samples()
    buffer = CircularMotionBuffer(retention_seconds=0.05)
    samples = simulator.samples()
    accepted = buffer.extend(samples)
    assert accepted == len(samples)
    first = samples[-1]
    assert buffer.add(first) is False
    skipped = RawImuSample(
        first.device_id,
        first.device_timestamp_us + 1000,
        first.packet_number + 3,
        first.acceleration_g,
        first.angular_velocity_dps,
    )
    assert buffer.add(skipped)
    assert buffer.health.duplicates == 1
    assert buffer.health.estimated_loss >= 2
    assert len(buffer) <= len(samples)


def test_resampling_timestamp_modes_preserve_target_grid_sequence_order() -> None:
    def feature(timestamp: float, wrist: WristSide) -> MotionFeatures:
        return MotionFeatures(timestamp, wrist, 1.0, 0.0, 1.0, 1.0, True)

    samples = (
        feature(0.029, WristSide.RIGHT),
        feature(0.011, WristSide.LEFT),
        feature(0.009, WristSide.RIGHT),
        feature(0.031, WristSide.LEFT),
    )
    legacy = resample_features(samples, 0.0, 0.04, rate_hz=50, max_gap_seconds=0.02)
    assert [(item.synchronized_time, item.wrist) for item in legacy] == [
        (0.0, WristSide.LEFT),
        (0.0, WristSide.RIGHT),
        (0.02, WristSide.LEFT),
        (0.02, WristSide.RIGHT),
    ]

    source_timing = resample_features(
        samples,
        0.0,
        0.04,
        rate_hz=50,
        max_gap_seconds=0.02,
        timestamp_mode=ResamplingMode.SOURCE_SAMPLE,
    )
    assert [(item.synchronized_time, item.wrist) for item in source_timing] == [
        (0.011, WristSide.LEFT),
        (0.009, WristSide.RIGHT),
        (0.011, WristSide.LEFT),
        (0.029, WristSide.RIGHT),
    ]
