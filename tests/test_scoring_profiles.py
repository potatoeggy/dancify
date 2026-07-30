from dataclasses import FrozenInstanceError
from math import sin
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dancify import create_app
from dancify.calibration import ResamplingMode
from dancify.debug_scoring import WindowScoringEvaluator
from dancify.domain import MotionFeatures, MotionWindow, WristSide
from dancify.scoring import ScoringConfig, WeightedDtwScoringAlgorithm, _dtw_path


def _feature(timestamp: float, wrist: WristSide = WristSide.RIGHT, phase: float | None = None) -> MotionFeatures:
    value = sin((timestamp if phase is None else phase) * 12.0)
    return MotionFeatures(timestamp, wrist, value, 1.0 - abs(value), 1.0, 1.0, True, 1.0)


def _stream(*, wrist: WristSide = WristSide.RIGHT, count: int = 50) -> tuple[MotionFeatures, ...]:
    return tuple(_feature(index / 50, wrist) for index in range(count))


def _evaluate(
    config: ScoringConfig,
    performance: tuple[MotionFeatures, ...],
    wrists: frozenset[WristSide] = frozenset({WristSide.RIGHT}),
) -> Any:
    reference = tuple(_stream(wrist=wrist) for wrist in wrists)
    flattened = tuple(
        sorted(
            (item for stream in reference for item in stream),
            key=lambda item: (item.synchronized_time, item.wrist.value),
        )
    )
    return WindowScoringEvaluator.from_config(config).evaluate(
        scorer=WeightedDtwScoringAlgorithm.from_config(config),
        index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        reference_features=flattened,
        performance_features=performance,
        active_wrists=wrists,
    )


def test_no_argument_scorer_and_evaluator_retain_strict_compatibility() -> None:
    strict = ScoringConfig.named("strict")
    direct = WeightedDtwScoringAlgorithm()
    configured = WeightedDtwScoringAlgorithm.from_config(strict)
    reference = MotionWindow(0, 0.0, 1.0, _stream())
    changed = MotionWindow(
        0,
        0.0,
        1.0,
        tuple(_feature(index / 50, phase=(49 - index) / 50) for index in range(50)),
        True,
        0.73,
    )
    assert direct == configured
    assert direct.score(reference, changed) == configured.score(reference, changed)
    assert (direct.direction_weight, direct.magnitude_weight, direct.timing_weight) == (0.5, 0.3, 0.2)
    assert direct.sakoe_chiba_radius == 10

    evaluation = WindowScoringEvaluator().evaluate(
        scorer=direct,
        index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        reference_features=_stream(),
        performance_features=_stream(count=25),
        active_wrists=frozenset({WristSide.RIGHT}),
    )
    assert evaluation.result.breakdown.quality == pytest.approx(evaluation.coverage * evaluation.mean_sample_quality)
    assert strict.minimum_coverage == 0.5
    assert strict.resample_max_gap_seconds == 0.05


def test_generous_timing_uses_sample_delta_with_grace_and_smooth_falloff() -> None:
    config = ScoringConfig.named("generous")
    scorer = WeightedDtwScoringAlgorithm.from_config(config)
    reference_samples = (_feature(0.5, phase=0.0),)
    reference = MotionWindow(0, 0.0, 1.5, reference_samples)

    scores: list[float] = []
    timings: list[float] = []
    for shift in (0.100, 0.200, 0.250):
        performance = MotionWindow(
            0,
            0.0,
            1.5,
            tuple(_feature(item.synchronized_time + shift, phase=0.0) for item in reference_samples),
        )
        result = scorer.score(reference, performance)
        scores.append(result.value)
        timings.append(result.breakdown.timing)

    assert timings[0] == pytest.approx(1.0)
    assert timings[1] == pytest.approx(0.925926, abs=1e-5)
    assert timings[2] == pytest.approx(0.740741, abs=1e-5)
    assert scores[0] == pytest.approx(100.0)
    assert scores[0] > scores[1] > scores[2] > 90.0


def test_generous_partial_coverage_is_valid_monotonic_and_forgiving() -> None:
    config = ScoringConfig.named("generous")
    evaluations = [_evaluate(config, _stream(count=count)) for count in (5, 13, 20)]
    coverages = [evaluation.coverage for evaluation in evaluations]
    assert coverages[0] == pytest.approx(0.20, abs=0.021)
    assert coverages[1] == pytest.approx(0.36, abs=0.021)
    assert coverages[2] == pytest.approx(0.50, abs=0.021)
    assert all(evaluation.result.valid for evaluation in evaluations)
    assert evaluations[0].result.breakdown.quality >= 0.59
    assert [evaluation.result.value for evaluation in evaluations] == sorted(
        evaluation.result.value for evaluation in evaluations
    )

    below_floor = _evaluate(config, _stream(count=1))
    no_data = _evaluate(config, ())
    assert below_floor.coverage < config.minimum_coverage
    assert below_floor.result.valid is False and below_floor.result.value == 0.0
    assert no_data.coverage == 0.0
    assert no_data.result.valid is False and no_data.result.value == 0.0
    assert no_data.result.breakdown.quality == 0.0


def test_order_direction_and_one_or_two_wrist_matching_remain_robust() -> None:
    config = ScoringConfig.named("generous")
    scorer = WeightedDtwScoringAlgorithm.from_config(config)
    reference_samples = _stream()
    reference = MotionWindow(0, 0.0, 1.0, reference_samples)
    reversed_order = MotionWindow(
        0,
        0.0,
        1.0,
        tuple(_feature(index / 50, phase=(49 - index) / 50) for index in range(50)),
    )
    opposite = MotionWindow(
        0,
        0.0,
        1.0,
        tuple(
            MotionFeatures(
                item.synchronized_time,
                item.wrist,
                -item.vertical_direction,
                -item.horizontal_direction if item.horizontal_direction is not None else None,
                item.horizontal_confidence,
                item.linear_intensity,
                item.movement_active,
            )
            for item in reference_samples
        ),
    )
    perfect = scorer.score(reference, reference)
    assert perfect.value > scorer.score(reference, reversed_order).value
    assert perfect.value > scorer.score(reference, opposite).value

    right = _evaluate(config, _stream())
    both_samples = tuple(
        sorted(
            (*_stream(wrist=WristSide.LEFT), *_stream()),
            key=lambda item: (item.synchronized_time, item.wrist.value),
        )
    )
    both = _evaluate(config, both_samples, frozenset(WristSide))
    assert right.coverage == both.coverage == 1.0
    assert right.result.value == both.result.value == 100.0


def test_named_profiles_are_frozen_bounded_and_ordered() -> None:
    generous = ScoringConfig.named("generous")
    balanced = ScoringConfig.named("balanced")
    strict = ScoringConfig.named("strict")
    assert generous.minimum_coverage < balanced.minimum_coverage < strict.minimum_coverage
    assert generous.sakoe_chiba_radius > balanced.sakoe_chiba_radius > strict.sakoe_chiba_radius
    assert generous.resample_max_gap_seconds > balanced.resample_max_gap_seconds > strict.resample_max_gap_seconds
    with pytest.raises(FrozenInstanceError):
        generous.minimum_coverage = 0.0  # type: ignore[misc]


def test_scoring_env_dotenv_and_flask_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DANCIFY_SCORING_PROFILE=strict\nDANCIFY_SCORING_TIMING_GRACE_SECONDS=0.01\n")
    process = {
        "DANCIFY_SCORING_PROFILE": "balanced",
        "DANCIFY_SCORING_TIMING_GRACE_SECONDS": "0.08",
    }
    with patch.dict("os.environ", process, clear=True):
        from_environment = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
        environment_config = from_environment.extensions["scoring_config"]
        assert environment_config.profile == "balanced"
        assert environment_config.timing_grace_seconds == pytest.approx(0.08)

        explicit = create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
                "DANCIFY_SCORING_PROFILE": "generous",
                "DANCIFY_SCORING_TIMING_GRACE_SECONDS": 0.2,
            }
        )
        explicit_config = explicit.extensions["scoring_config"]
        assert explicit_config.profile == "generous"
        assert explicit_config.timing_grace_seconds == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DANCIFY_SCORING_PROFILE", "easy"),
        ("DANCIFY_SCORING_TIMING_GRACE_SECONDS", True),
        ("DANCIFY_SCORING_TIMING_GRACE_SECONDS", float("nan")),
        ("DANCIFY_SCORING_TIMING_FALLOFF_SECONDS", float("inf")),
        ("DANCIFY_SCORING_MIN_COVERAGE", -0.01),
        ("DANCIFY_SCORING_FULL_COVERAGE", 1.01),
        ("DANCIFY_SCORING_RESAMPLE_MAX_GAP_SECONDS", 0.0),
        ("DANCIFY_SCORING_SAKOE_CHIBA_RADIUS", True),
        ("DANCIFY_SCORING_SAMPLE_RATE_HZ", 0),
    ],
)
def test_invalid_scoring_configuration_names_the_bad_key(key: str, value: object) -> None:
    with pytest.raises(RuntimeError, match=key):
        create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
                key: value,
            }
        )


def test_cross_field_scoring_bounds_are_rejected_actionably() -> None:
    with pytest.raises(RuntimeError, match="DANCIFY_SCORING_(MIN|FULL)_COVERAGE"):
        create_app(
            {
                "TESTING": True,
                "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
                "DANCIFY_SCORING_MIN_COVERAGE": 0.8,
                "DANCIFY_SCORING_FULL_COVERAGE": 0.3,
            }
        )


def test_dtw_never_aligns_opposite_wrists_and_missing_stream_uses_coverage_penalty() -> None:
    left = (_feature(0.0, WristSide.LEFT),)
    right = (_feature(0.0, WristSide.RIGHT),)
    assert _dtw_path(left, right, radius=18) == ()

    config = ScoringConfig.named("generous")
    scorer = WeightedDtwScoringAlgorithm.from_config(config)
    reference = tuple(
        sorted(
            (*_stream(wrist=WristSide.LEFT), *_stream(wrist=WristSide.RIGHT)),
            key=lambda item: (item.synchronized_time, item.wrist.value),
        )
    )
    evaluator = WindowScoringEvaluator.from_config(config)
    right_only = evaluator.evaluate(
        scorer=scorer,
        index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        reference_features=reference,
        performance_features=_stream(wrist=WristSide.RIGHT),
        active_wrists=frozenset(WristSide),
    )
    full = evaluator.evaluate(
        scorer=scorer,
        index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        reference_features=reference,
        performance_features=reference,
        active_wrists=frozenset(WristSide),
    )
    path = _dtw_path(
        right_only.reference.samples,
        right_only.performance.samples,
        config.sakoe_chiba_radius,
        config.timing_path_cost_weight,
        config.timing_grace_seconds,
        config.timing_falloff_seconds,
        config.use_sample_timing,
    )
    assert path
    assert all(
        right_only.reference.samples[reference_index].wrist is right_only.performance.samples[performance_index].wrist
        for reference_index, performance_index in path
    )
    assert right_only.coverage == pytest.approx(0.5)
    assert right_only.result.valid is True
    assert right_only.result.breakdown.timing == pytest.approx(1.0)
    assert 0.0 < right_only.result.value < full.result.value

    strict = ScoringConfig.named("strict")
    strict_right_only = WindowScoringEvaluator.from_config(strict).evaluate(
        scorer=WeightedDtwScoringAlgorithm.from_config(strict),
        index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        reference_features=reference,
        performance_features=_stream(wrist=WristSide.RIGHT),
        active_wrists=frozenset(WristSide),
    )
    assert strict_right_only.result.valid is True
    assert strict_right_only.result.breakdown.timing == pytest.approx(1.0)


def test_strict_two_wrist_tail_fixture_reproduces_exact_legacy_score() -> None:
    samples: list[MotionFeatures] = []
    for index in range(50):
        grid_timestamp = index / 50
        source_gap = 0.011725 if index == 49 else 0.001
        for wrist in WristSide:
            samples.append(_feature(grid_timestamp + source_gap, wrist, phase=grid_timestamp))

    strict = ScoringConfig.named("strict")
    direct_evaluator = WindowScoringEvaluator()
    configured_evaluator = WindowScoringEvaluator.from_config(strict)
    assert direct_evaluator.resampling_mode is ResamplingMode.TARGET_GRID
    assert configured_evaluator.resampling_mode is ResamplingMode.TARGET_GRID
    evaluation = configured_evaluator.evaluate(
        scorer=WeightedDtwScoringAlgorithm.from_config(strict),
        index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        reference_features=tuple(samples),
        performance_features=tuple(samples),
        active_wrists=frozenset(WristSide),
    )
    assert [(sample.synchronized_time, sample.wrist) for sample in evaluation.performance.samples] == [
        (index / 50, wrist) for index in range(50) for wrist in WristSide
    ]
    assert evaluation.result.value == pytest.approx(97.571)
    assert evaluation.result.breakdown.timing == pytest.approx(1.0)
