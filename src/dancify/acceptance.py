"""Repeatable synthetic acceptance and scoring benchmark."""

from __future__ import annotations

import json
from dataclasses import replace
from time import perf_counter

from dancify.calibration import AffineClockMapper, SpatialCalibrationProfile
from dancify.domain import MotionFeatures, MotionWindow, Vector3, WristSide
from dancify.motion import DeterministicMotionSimulator, SimulationConfig
from dancify.scoring import WeightedDtwScoringAlgorithm


def run_acceptance(iterations: int = 30) -> dict[str, float]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    profile = SpatialCalibrationProfile.from_gestures(
        [Vector3(0, 0, 1)],
        [Vector3(0, 1, 1)],
        [Vector3(1, 0, 1)],
    )
    simulator = DeterministicMotionSimulator(SimulationConfig(duration_seconds=1.0, sample_rate_hz=50))
    translated: list[MotionFeatures] = []
    for sample in simulator.samples():
        wrist = WristSide.LEFT if sample.device_id == "left" else WristSide.RIGHT
        translated.append(profile.translate(sample, wrist, AffineClockMapper()))
    reference = MotionWindow(0, 0.0, 1.0, tuple(translated))
    good = MotionWindow(0, 0.0, 1.0, tuple(translated))
    reversed_samples = tuple(
        replace(
            sample,
            vertical_direction=-sample.vertical_direction,
            horizontal_direction=(None if sample.horizontal_direction is None else -sample.horizontal_direction),
        )
        for sample in translated
    )
    reversed_window = MotionWindow(0, 0.0, 1.0, reversed_samples)
    missing = MotionWindow(0, 0.0, 1.0, tuple(translated[:20]), False, 0.0)
    scorer = WeightedDtwScoringAlgorithm()
    good_score = scorer.score(reference, good).value
    reversed_score = scorer.score(reference, reversed_window).value
    missing_score = scorer.score(reference, missing).value
    durations: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        scorer.score(reference, good)
        durations.append((perf_counter() - started) * 1000.0)
    durations.sort()
    p95 = durations[min(len(durations) - 1, int(len(durations) * 0.95))]
    if not good_score > reversed_score > missing_score:
        raise RuntimeError("synthetic score ordering failed")
    if p95 >= 20.0:
        raise RuntimeError(f"scoring p95 exceeded 20 ms: {p95:.3f}")
    return {
        "goodScore": good_score,
        "reversedScore": reversed_score,
        "missingScore": missing_score,
        "scoringP95Ms": round(p95, 3),
    }


def main() -> None:
    print(json.dumps(run_acceptance(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
