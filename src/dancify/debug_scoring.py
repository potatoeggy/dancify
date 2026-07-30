"""Shared canonical window evaluation used by gameplay and debug diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from dancify.calibration import ResamplingMode, resample_features
from dancify.domain import MotionFeatures, MotionWindow, ScoreResult, WristSide
from dancify.scoring import ScoringAlgorithm, ScoringConfig


@dataclass(frozen=True, slots=True)
class WindowEvaluation:
    """Prepared windows and the authoritative production score."""

    reference: MotionWindow
    performance: MotionWindow
    coverage: float
    mean_sample_quality: float
    result: ScoreResult


@dataclass(frozen=True, slots=True)
class WindowScoringEvaluator:
    """Own production window preparation so diagnostics cannot drift from gameplay."""

    sample_rate_hz: int = 50
    resample_max_gap_seconds: float = 0.05
    minimum_valid_coverage: float = 0.5
    full_coverage: float = 1.0
    coverage_quality_floor: float = 0.5
    sample_quality_floor: float = 0.0
    smooth_coverage: bool = False
    resampling_mode: ResamplingMode = ResamplingMode.TARGET_GRID

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate_hz, bool) or self.sample_rate_hz <= 0:
            raise ValueError("sample rate must be positive")
        if not 0.0 < self.resample_max_gap_seconds <= 0.5:
            raise ValueError("resample max gap must be between zero and 0.5 seconds")
        if not 0.0 <= self.minimum_valid_coverage < self.full_coverage <= 1.0:
            raise ValueError("coverage bounds must satisfy 0 <= minimum < full <= 1")
        if not 0.0 <= self.coverage_quality_floor <= 1.0:
            raise ValueError("coverage quality floor must be between zero and one")
        if not 0.0 <= self.sample_quality_floor <= 1.0:
            raise ValueError("sample quality floor must be between zero and one")

    @classmethod
    def from_config(cls, config: ScoringConfig) -> WindowScoringEvaluator:
        return cls(
            config.sample_rate_hz,
            config.resample_max_gap_seconds,
            config.minimum_coverage,
            config.full_coverage,
            config.coverage_quality_floor,
            config.sample_quality_floor,
            config.smooth_coverage,
            config.resampling_mode,
        )

    def prepare_reference(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
        reference_features: tuple[MotionFeatures, ...],
        active_wrists: frozenset[WristSide],
    ) -> tuple[MotionFeatures, ...]:
        filtered = tuple(item for item in reference_features if item.wrist in active_wrists)
        return resample_features(
            filtered,
            start_seconds,
            end_seconds,
            rate_hz=self.sample_rate_hz,
            max_gap_seconds=self.resample_max_gap_seconds,
            timestamp_mode=self.resampling_mode,
        )

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
        performance_samples = resample_features(
            performance_raw,
            start_seconds,
            end_seconds,
            rate_hz=self.sample_rate_hz,
            max_gap_seconds=self.resample_max_gap_seconds,
            timestamp_mode=self.resampling_mode,
        )
        expected = max(
            1,
            int((end_seconds - start_seconds) * self.sample_rate_hz) * len(active_wrists),
        )
        coverage = min(1.0, len(performance_samples) / expected)
        quality_values = [item.sample_quality for item in performance_samples]
        mean_sample_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0
        quality = self._quality_adjustment(coverage, mean_sample_quality) if performance_samples else 0.0
        valid = bool(performance_samples) and coverage >= self.minimum_valid_coverage
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
            mean_sample_quality,
            scorer.score(reference_window, performance_window),
        )

    def _quality_adjustment(self, coverage: float, mean_sample_quality: float) -> float:
        if not self.smooth_coverage:
            coverage_factor = coverage
        elif coverage >= self.full_coverage:
            coverage_factor = 1.0
        elif coverage <= self.minimum_valid_coverage:
            coverage_factor = self.coverage_quality_floor
        else:
            progress = (coverage - self.minimum_valid_coverage) / (self.full_coverage - self.minimum_valid_coverage)
            smoothstep = progress * progress * (3.0 - 2.0 * progress)
            coverage_factor = self.coverage_quality_floor + (1.0 - self.coverage_quality_floor) * smoothstep
        sample_factor = self.sample_quality_floor + (1.0 - self.sample_quality_floor) * mean_sample_quality
        return max(0.0, min(1.0, coverage_factor * sample_factor))
