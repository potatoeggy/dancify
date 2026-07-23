from math import sin
from time import perf_counter
from typing import Any

import pytest

from dancify.domain import MotionFeatures, MotionWindow, WristSide
from dancify.ingestion import parse_routine, reference_features
from dancify.scoring import FixedWindowingStrategy, ScorerRegistry, WeightedDtwScoringAlgorithm


def feature(index: int, *, opposite: bool = False, quality: float = 1.0) -> MotionFeatures:
    value = sin(index / 8)
    sign = -1 if opposite else 1
    return MotionFeatures(
        index / 50, WristSide.LEFT, sign * value, sign * (1 - abs(value)), 1.0, abs(value) + 0.2, True, quality
    )


def window(*, opposite: bool = False, valid: bool = True, quality: float = 1.0) -> MotionWindow:
    return MotionWindow(
        0, 0.0, 1.0, tuple(feature(index, opposite=opposite, quality=quality) for index in range(50)), valid, quality
    )


def test_ingestion_accepts_real_shape_and_generates_partial_window(routine_payload: dict[str, Any]) -> None:
    routine_payload["motion_signal"][0]["left_wrist"]["ax"] = None
    parsed = parse_routine(routine_payload)
    assert parsed.title == "Demo"
    assert len(parsed.reference_motion) == 61
    assert [item.scoreable for item in parsed.scoring_windows] == [True, True]
    assert reference_features(parsed.reference_motion, 0, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_routine({**routine_payload, "motion_signal": list(reversed(routine_payload["motion_signal"]))})


def test_dtw_scoring_is_bounded_ordered_and_fast() -> None:
    scorer = WeightedDtwScoringAlgorithm(sakoe_chiba_radius=10)
    reference = window()
    start = perf_counter()
    perfect = scorer.score(reference, window())
    elapsed = perf_counter() - start
    reversed_score = scorer.score(reference, window(opposite=True))
    invalid = scorer.score(reference, window(valid=False, quality=0.0))
    assert perfect.value > reversed_score.value > invalid.value
    assert perfect.value == pytest.approx(100.0)
    assert invalid.valid is False and invalid.value == 0
    assert elapsed < 0.02


def test_scoring_and_windowing_are_pluggable() -> None:
    fixed = FixedWindowingStrategy(1.0)
    assert fixed.completed_windows(0.99) == ()
    assert fixed.completed_windows(2.1) == (0, 1)
    with pytest.raises(ValueError):
        FixedWindowingStrategy(0)
    registry = ScorerRegistry()
    algorithm = WeightedDtwScoringAlgorithm()
    registry.register(algorithm)
    assert registry.get("weighted_dtw") is algorithm
    with pytest.raises(ValueError, match="already"):
        registry.register(algorithm)
    with pytest.raises(ValueError, match="unknown"):
        registry.get("missing")


def test_acceptance_benchmark_orders_scores_and_meets_budget() -> None:
    from dancify.acceptance import run_acceptance

    report = run_acceptance(iterations=5)
    assert report["goodScore"] > report["reversedScore"] > report["missingScore"]
    assert report["scoringP95Ms"] < 20.0
