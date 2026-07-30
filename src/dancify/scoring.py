"""Pluggable window scheduling and constrained-DTW scoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from types import MappingProxyType
from typing import ClassVar, Protocol

from dancify.calibration import ResamplingMode
from dancify.domain import MotionFeatures, MotionWindow, ScoreBreakdown, ScoreResult, WristSide


class WindowingStrategy(Protocol):
    def completed_windows(self, session_time: float) -> tuple[int, ...]: ...


class ScoringAlgorithm(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, reference: MotionWindow, performance: MotionWindow) -> ScoreResult: ...


@dataclass(frozen=True, slots=True)
class FixedWindowingStrategy:
    duration_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("window duration must be positive")

    def completed_windows(self, session_time: float) -> tuple[int, ...]:
        if session_time <= 0:
            return ()
        return tuple(range(int(session_time / self.duration_seconds)))


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    """Immutable, bounded scoring and window-evaluation configuration."""

    profile: str
    direction_weight: float
    magnitude_weight: float
    timing_weight: float
    sakoe_chiba_radius: int
    timing_grace_seconds: float
    timing_falloff_seconds: float
    timing_path_cost_weight: float
    minimum_coverage: float
    full_coverage: float
    coverage_quality_floor: float
    sample_quality_floor: float
    resample_max_gap_seconds: float
    sample_rate_hz: int = 50
    use_sample_timing: bool = True
    smooth_coverage: bool = True
    resampling_mode: ResamplingMode = ResamplingMode.SOURCE_SAMPLE

    PROFILES: ClassVar[MappingProxyType[str, ScoringConfig]]

    def __post_init__(self) -> None:
        if self.profile not in {"generous", "balanced", "strict"}:
            raise ValueError("profile must be generous, balanced, or strict")
        numeric = (
            self.direction_weight,
            self.magnitude_weight,
            self.timing_weight,
            self.timing_grace_seconds,
            self.timing_falloff_seconds,
            self.timing_path_cost_weight,
            self.minimum_coverage,
            self.full_coverage,
            self.coverage_quality_floor,
            self.sample_quality_floor,
            self.resample_max_gap_seconds,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("scoring configuration values must be finite")
        weights = (self.direction_weight, self.magnitude_weight, self.timing_weight)
        if any(not 0.0 <= weight <= 1.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("weights must be between zero and one and sum to one")
        if isinstance(self.sakoe_chiba_radius, bool) or not 0 <= self.sakoe_chiba_radius <= 100:
            raise ValueError("Sakoe-Chiba radius must be between 0 and 100")
        if isinstance(self.sample_rate_hz, bool) or not 1 <= self.sample_rate_hz <= 240:
            raise ValueError("sample rate must be between 1 and 240 Hz")
        if not 0.0 <= self.timing_grace_seconds <= 1.0:
            raise ValueError("timing grace must be between 0 and 1 second")
        if not 0.0 < self.timing_falloff_seconds <= 2.0:
            raise ValueError("timing falloff must be greater than 0 and at most 2 seconds")
        if self.use_sample_timing and self.timing_falloff_seconds <= self.timing_grace_seconds:
            raise ValueError("timing falloff must be greater than timing grace")
        if not 0.0 <= self.timing_path_cost_weight <= 1.0:
            raise ValueError("timing path-cost weight must be between zero and one")
        if not 0.0 <= self.minimum_coverage <= self.full_coverage <= 1.0:
            raise ValueError("coverage bounds must satisfy 0 <= minimum <= full <= 1")
        if self.minimum_coverage == self.full_coverage:
            raise ValueError("full coverage must be greater than minimum coverage")
        if not 0.0 <= self.coverage_quality_floor <= 1.0:
            raise ValueError("coverage quality floor must be between zero and one")
        if not 0.0 <= self.sample_quality_floor <= 1.0:
            raise ValueError("sample quality floor must be between zero and one")
        if not 0.001 <= self.resample_max_gap_seconds <= 0.5:
            raise ValueError("resample max gap must be between 0.001 and 0.5 seconds")

    @classmethod
    def named(cls, name: str) -> ScoringConfig:
        try:
            return cls.PROFILES[name]
        except KeyError as exc:
            raise ValueError(f"unknown scoring profile {name!r}; expected generous, balanced, or strict") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "directionWeight": self.direction_weight,
            "magnitudeWeight": self.magnitude_weight,
            "timingWeight": self.timing_weight,
            "sakoeChibaRadius": self.sakoe_chiba_radius,
            "timingGraceSeconds": self.timing_grace_seconds,
            "timingFalloffSeconds": self.timing_falloff_seconds,
            "timingPathCostWeight": self.timing_path_cost_weight,
            "minimumCoverage": self.minimum_coverage,
            "fullCoverage": self.full_coverage,
            "coverageQualityFloor": self.coverage_quality_floor,
            "sampleQualityFloor": self.sample_quality_floor,
            "resampleMaxGapSeconds": self.resample_max_gap_seconds,
            "sampleRateHz": self.sample_rate_hz,
            "sampleSynchronizedTiming": self.use_sample_timing,
            "smoothCoverageRamp": self.smooth_coverage,
            "resamplingTimestampMode": self.resampling_mode.value,
        }


ScoringConfig.PROFILES = MappingProxyType(
    {
        "generous": ScoringConfig("generous", 0.45, 0.25, 0.30, 18, 0.150, 0.450, 0.35, 0.20, 0.65, 0.70, 0.85, 0.100),
        "balanced": ScoringConfig(
            "balanced", 0.475, 0.275, 0.250, 14, 0.075, 0.300, 0.20, 0.35, 0.80, 0.55, 0.65, 0.075
        ),
        "strict": ScoringConfig(
            "strict",
            0.50,
            0.30,
            0.20,
            10,
            0.0,
            1.0,
            0.0,
            0.50,
            1.0,
            0.50,
            0.0,
            0.050,
            use_sample_timing=False,
            smooth_coverage=False,
            resampling_mode=ResamplingMode.TARGET_GRID,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class WeightedDtwScoringAlgorithm:
    """Weighted DTW scorer; no arguments intentionally retain legacy strict behavior."""

    direction_weight: float = 0.5
    magnitude_weight: float = 0.3
    timing_weight: float = 0.2
    sakoe_chiba_radius: int = 10
    timing_grace_seconds: float = 0.0
    timing_falloff_seconds: float = 1.0
    timing_path_cost_weight: float = 0.0
    use_sample_timing: bool = False

    def __post_init__(self) -> None:
        weights = (self.direction_weight, self.magnitude_weight, self.timing_weight)
        if any(not isfinite(weight) or weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("weights must be finite, non-negative, and sum to one")
        if isinstance(self.sakoe_chiba_radius, bool) or not 0 <= self.sakoe_chiba_radius <= 100:
            raise ValueError("Sakoe-Chiba radius must be between 0 and 100")
        timing_values = (self.timing_grace_seconds, self.timing_falloff_seconds, self.timing_path_cost_weight)
        if not all(isfinite(value) for value in timing_values):
            raise ValueError("timing settings must be finite")
        if not 0.0 <= self.timing_grace_seconds <= 1.0:
            raise ValueError("timing grace must be between zero and one second")
        if not 0.0 < self.timing_falloff_seconds <= 2.0:
            raise ValueError("timing falloff must be greater than zero and at most two seconds")
        if self.use_sample_timing and self.timing_falloff_seconds <= self.timing_grace_seconds:
            raise ValueError("timing falloff must be greater than timing grace")
        if not 0.0 <= self.timing_path_cost_weight <= 1.0:
            raise ValueError("timing path-cost weight must be between zero and one")

    @classmethod
    def from_config(cls, config: ScoringConfig) -> WeightedDtwScoringAlgorithm:
        return cls(
            config.direction_weight,
            config.magnitude_weight,
            config.timing_weight,
            config.sakoe_chiba_radius,
            config.timing_grace_seconds,
            config.timing_falloff_seconds,
            config.timing_path_cost_weight,
            config.use_sample_timing,
        )

    @property
    def name(self) -> str:
        return "weighted_dtw"

    def score(self, reference: MotionWindow, performance: MotionWindow) -> ScoreResult:
        if not reference.samples or not performance.samples or not performance.valid:
            return _result(reference, 0.0, False, ScoreBreakdown(0.0, 0.0, 0.0, performance.quality))
        path = _dtw_path(
            reference.samples,
            performance.samples,
            self.sakoe_chiba_radius,
            self.timing_path_cost_weight,
            self.timing_grace_seconds,
            self.timing_falloff_seconds,
            self.use_sample_timing,
        )
        if not path:
            return _result(reference, 0.0, False, ScoreBreakdown(0.0, 0.0, 0.0, performance.quality))
        direction = sum(_direction(reference.samples[i], performance.samples[j]) for i, j in path) / len(path)
        magnitude = sum(_magnitude(reference.samples[i], performance.samples[j]) for i, j in path) / len(path)
        reference_positions, reference_lengths = _wrist_coordinates(reference.samples)
        performance_positions, performance_lengths = _wrist_coordinates(performance.samples)
        timing = sum(
            _timing(
                reference.samples[i],
                performance.samples[j],
                reference_positions[i],
                performance_positions[j],
                reference_lengths[reference.samples[i].wrist],
                performance_lengths[performance.samples[j].wrist],
                self.timing_grace_seconds,
                self.timing_falloff_seconds,
                self.use_sample_timing,
            )
            for i, j in path
        ) / len(path)
        quality = max(0.0, min(1.0, performance.quality))
        raw = self.direction_weight * direction + self.magnitude_weight * magnitude + self.timing_weight * timing
        value = round(max(0.0, min(100.0, 100.0 * raw * quality)), 3)
        return _result(reference, value, True, ScoreBreakdown(direction, magnitude, timing, quality))


class ScorerRegistry:
    def __init__(self) -> None:
        self._algorithms: dict[str, ScoringAlgorithm] = {}

    def register(self, algorithm: ScoringAlgorithm) -> None:
        if algorithm.name in self._algorithms:
            raise ValueError(f"scorer already registered: {algorithm.name}")
        self._algorithms[algorithm.name] = algorithm

    def get(self, name: str) -> ScoringAlgorithm:
        try:
            return self._algorithms[name]
        except KeyError as exc:
            raise ValueError(f"unknown scorer: {name}") from exc


@dataclass(slots=True)
class ArithmeticMeanScoreAggregator:
    total: float = 0.0
    count: int = 0

    def add(self, score: float) -> float:
        self.total += score
        self.count += 1
        return self.total / self.count


def _dtw_path(
    reference: tuple[MotionFeatures, ...],
    performance: tuple[MotionFeatures, ...],
    radius: int,
    timing_path_cost_weight: float = 0.0,
    timing_grace_seconds: float = 0.0,
    timing_falloff_seconds: float = 1.0,
    use_sample_timing: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Align each wrist independently and return only same-wrist index pairs."""

    combined: list[tuple[int, int]] = []
    for wrist in WristSide:
        reference_indices = tuple(index for index, sample in enumerate(reference) if sample.wrist is wrist)
        performance_indices = tuple(index for index, sample in enumerate(performance) if sample.wrist is wrist)
        if not reference_indices or not performance_indices:
            continue
        wrist_path = _single_wrist_dtw_path(
            tuple(reference[index] for index in reference_indices),
            tuple(performance[index] for index in performance_indices),
            radius,
            timing_path_cost_weight,
            timing_grace_seconds,
            timing_falloff_seconds,
            use_sample_timing,
        )
        combined.extend(
            (reference_indices[reference_index], performance_indices[performance_index])
            for reference_index, performance_index in wrist_path
        )
    return tuple(combined)


def _single_wrist_dtw_path(
    reference: tuple[MotionFeatures, ...],
    performance: tuple[MotionFeatures, ...],
    radius: int,
    timing_path_cost_weight: float,
    timing_grace_seconds: float,
    timing_falloff_seconds: float,
    use_sample_timing: bool,
) -> tuple[tuple[int, int], ...]:
    n, m = len(reference), len(performance)
    band = max(radius, abs(n - m))
    infinity = float("inf")
    cost = [[infinity] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            options = (
                (cost[i - 1][j], (i - 1, j)),
                (cost[i][j - 1], (i, j - 1)),
                (cost[i - 1][j - 1], (i - 1, j - 1)),
            )
            prior_cost, prior = min(options, key=lambda item: item[0])
            if prior_cost == infinity:
                continue
            timing = _timing(
                reference[i - 1],
                performance[j - 1],
                i - 1,
                j - 1,
                n,
                m,
                timing_grace_seconds,
                timing_falloff_seconds,
                use_sample_timing,
            )
            local = _local_cost(reference[i - 1], performance[j - 1])
            cost[i][j] = local + timing_path_cost_weight * (1.0 - timing) + prior_cost
            previous[(i, j)] = prior
    if cost[n][m] == infinity:
        return ()
    cursor = (n, m)
    path: list[tuple[int, int]] = []
    while cursor != (0, 0):
        path.append((cursor[0] - 1, cursor[1] - 1))
        cursor = previous[cursor]
    path.reverse()
    return tuple(path)


def _local_cost(left: MotionFeatures, right: MotionFeatures) -> float:
    if left.wrist is not right.wrist:
        return float("inf")
    return (1.0 - _direction(left, right)) + (1.0 - _magnitude(left, right))


def _wrist_coordinates(
    samples: tuple[MotionFeatures, ...],
) -> tuple[dict[int, int], dict[WristSide, int]]:
    positions: dict[int, int] = {}
    lengths: dict[WristSide, int] = {}
    for wrist in WristSide:
        indices = tuple(index for index, sample in enumerate(samples) if sample.wrist is wrist)
        lengths[wrist] = len(indices)
        positions.update((index, position) for position, index in enumerate(indices))
    return positions, lengths


def _direction(left: MotionFeatures, right: MotionFeatures) -> float:
    if not left.movement_active and not right.movement_active:
        return 1.0
    confidence = min(left.horizontal_confidence, right.horizontal_confidence)
    horizontal_available = left.horizontal_direction is not None and right.horizontal_direction is not None
    horizontal_weight = confidence if horizontal_available else 0.0
    left_h = left.horizontal_direction or 0.0
    right_h = right.horizontal_direction or 0.0
    dot = left.vertical_direction * right.vertical_direction + horizontal_weight * left_h * right_h
    left_norm = sqrt(left.vertical_direction**2 + horizontal_weight * left_h**2)
    right_norm = sqrt(right.vertical_direction**2 + horizontal_weight * right_h**2)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.5
    cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    return (cosine + 1.0) / 2.0


def _magnitude(left: MotionFeatures, right: MotionFeatures) -> float:
    if left.linear_intensity <= 1e-9 and right.linear_intensity <= 1e-9:
        return 1.0
    return exp(-abs(log((left.linear_intensity + 1e-6) / (right.linear_intensity + 1e-6))))


def _timing(
    left_sample: MotionFeatures,
    right_sample: MotionFeatures,
    i: int,
    j: int,
    n: int,
    m: int,
    grace_seconds: float,
    falloff_seconds: float,
    use_sample_timing: bool,
) -> float:
    if not use_sample_timing:
        left = i / max(1, n - 1)
        right = j / max(1, m - 1)
        return max(0.0, 1.0 - abs(left - right))
    delta = abs(left_sample.synchronized_time - right_sample.synchronized_time)
    if delta <= grace_seconds:
        return 1.0
    if delta >= falloff_seconds:
        return 0.0
    progress = (delta - grace_seconds) / (falloff_seconds - grace_seconds)
    smoothstep = progress * progress * (3.0 - 2.0 * progress)
    return 1.0 - smoothstep


def _result(reference: MotionWindow, value: float, valid: bool, breakdown: ScoreBreakdown) -> ScoreResult:
    return ScoreResult(reference.index, reference.start_seconds, value, 0.0, valid, breakdown)
