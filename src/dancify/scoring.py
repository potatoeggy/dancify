"""Pluggable window scheduling and constrained-DTW scoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Protocol

from dancify.domain import MotionFeatures, MotionWindow, ScoreBreakdown, ScoreResult


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
class WeightedDtwScoringAlgorithm:
    direction_weight: float = 0.5
    magnitude_weight: float = 0.3
    timing_weight: float = 0.2
    sakoe_chiba_radius: int = 10

    def __post_init__(self) -> None:
        weights = (self.direction_weight, self.magnitude_weight, self.timing_weight)
        if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("weights must be non-negative and sum to one")
        if self.sakoe_chiba_radius < 0:
            raise ValueError("Sakoe-Chiba radius must be non-negative")

    @property
    def name(self) -> str:
        return "weighted_dtw"

    def score(self, reference: MotionWindow, performance: MotionWindow) -> ScoreResult:
        if not reference.samples or not performance.samples or not performance.valid:
            return _result(reference, 0.0, False, ScoreBreakdown(0.0, 0.0, 0.0, performance.quality))
        path = _dtw_path(reference.samples, performance.samples, self.sakoe_chiba_radius)
        if not path:
            return _result(reference, 0.0, False, ScoreBreakdown(0.0, 0.0, 0.0, performance.quality))
        direction = sum(_direction(reference.samples[i], performance.samples[j]) for i, j in path) / len(path)
        magnitude = sum(_magnitude(reference.samples[i], performance.samples[j]) for i, j in path) / len(path)
        timing = sum(_timing(i, j, len(reference.samples), len(performance.samples)) for i, j in path) / len(path)
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
) -> tuple[tuple[int, int], ...]:
    n, m = len(reference), len(performance)
    band = max(radius, abs(n - m))
    infinity = float("inf")
    cost = [[infinity] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            options = ((cost[i - 1][j], (i - 1, j)), (cost[i][j - 1], (i, j - 1)), (cost[i - 1][j - 1], (i - 1, j - 1)))
            prior_cost, prior = min(options, key=lambda item: item[0])
            if prior_cost == infinity:
                continue
            cost[i][j] = _local_cost(reference[i - 1], performance[j - 1]) + prior_cost
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
    wrist_penalty = 2.0 if left.wrist != right.wrist else 0.0
    return wrist_penalty + (1.0 - _direction(left, right)) + (1.0 - _magnitude(left, right))


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


def _timing(i: int, j: int, n: int, m: int) -> float:
    left = i / max(1, n - 1)
    right = j / max(1, m - 1)
    return max(0.0, 1.0 - abs(left - right))


def _result(reference: MotionWindow, value: float, valid: bool, breakdown: ScoreBreakdown) -> ScoreResult:
    return ScoreResult(reference.index, reference.start_seconds, value, 0.0, valid, breakdown)
