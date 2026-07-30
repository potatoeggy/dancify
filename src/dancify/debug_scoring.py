"""Shared canonical window evaluation used by gameplay and debug diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from dancify.calibration import resample_features
from dancify.domain import MotionFeatures, MotionWindow, ScoreResult, WristSide
from dancify.scoring import ScoringAlgorithm


@dataclass(frozen=True, slots=True)
class WindowEvaluation:
    """Prepared windows and the authoritative production score."""

    reference: MotionWindow
    performance: MotionWindow
    coverage: float
    result: ScoreResult


@dataclass(frozen=True, slots=True)
class WindowScoringEvaluator:
    """Own production window preparation so diagnostics cannot drift from gameplay."""

    sample_rate_hz: int = 50

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")

    def prepare_reference(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
        reference_features: tuple[MotionFeatures, ...],
        active_wrists: frozenset[WristSide],
    ) -> tuple[MotionFeatures, ...]:
        filtered = tuple(item for item in reference_features if item.wrist in active_wrists)
        return resample_features(filtered, start_seconds, end_seconds, self.sample_rate_hz)

    def evaluate(
        self,
        *,
        scorer: ScoringAlgorithm,
        index: int,
        start_seconds: float,
        end_seconds: float,
        reference_features: tuple[MotionFeatures, ...],
        performance_features: tuple[MotionFeatures, ...],
        active_wrists: frozenset[WristSide],
    ) -> WindowEvaluation:
        if not active_wrists:
            raise ValueError("at least one active wrist is required")
        reference_samples = self.prepare_reference(
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            reference_features=reference_features,
            active_wrists=active_wrists,
        )
        performance_raw = tuple(
            item
            for item in performance_features
            if item.wrist in active_wrists and start_seconds <= item.synchronized_time < end_seconds
        )
        performance_samples = resample_features(performance_raw, start_seconds, end_seconds, self.sample_rate_hz)
        expected = max(
            1,
            int((end_seconds - start_seconds) * self.sample_rate_hz) * len(active_wrists),
        )
        coverage = min(1.0, len(performance_samples) / expected)
        quality_values = [item.sample_quality for item in performance_samples]
        quality = coverage * (sum(quality_values) / len(quality_values) if quality_values else 0.0)
        valid = coverage >= 0.5
        reference_window = MotionWindow(index, start_seconds, end_seconds, reference_samples)
        performance_window = MotionWindow(
            index,
            start_seconds,
            end_seconds,
            performance_samples,
            valid,
            quality,
        )
        return WindowEvaluation(
            reference_window,
            performance_window,
            coverage,
            scorer.score(reference_window, performance_window),
        )
