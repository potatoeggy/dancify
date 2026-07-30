"""Development-only, loopback-only MOCK scoring diagnostics UI."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from ipaddress import ip_address
from math import cos, floor, isfinite, radians, sin
from typing import Any, cast

from flask import Blueprint, Response, current_app, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from dancify.debug_scoring import WindowScoringEvaluator
from dancify.domain import MotionFeatures, WristSide
from dancify.ingestion import deserialize_reference, reference_features
from dancify.models import DanceRoutineRecord, RoutineWindowRecord
from dancify.scoring import ScorerRegistry
from dancify.service import NotFoundError, RoutineService

DEBUG_MAX_BODY_BYTES = 4096
_SIGNAL_MEANING = (
    "Signed camera-plane wrist acceleration direction components and intensity; "
    "not pose, animation, or video reconstruction."
)

scoring_debug = Blueprint(
    "scoring_debug",
    __name__,
    url_prefix="/_dev/scoring",
    template_folder="templates",
    static_folder="static",
    static_url_path="assets",
)


@dataclass(frozen=True, slots=True)
class Perturbation:
    direction_rotation_degrees: float = 0.0
    intensity_scale: float = 1.0
    time_shift_ms: float = 0.0
    capture_coverage: float = 1.0
    sample_quality: float = 1.0
    horizontal_confidence: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {
            "directionRotationDegrees": self.direction_rotation_degrees,
            "intensityScale": self.intensity_scale,
            "timeShiftMs": self.time_shift_ms,
            "captureCoverage": self.capture_coverage,
            "sampleQuality": self.sample_quality,
            "horizontalConfidence": self.horizontal_confidence,
        }


@dataclass(frozen=True, slots=True)
class ReferenceContext:
    record: DanceRoutineRecord
    window: RoutineWindowRecord
    raw_features: tuple[MotionFeatures, ...]
    reference_samples: tuple[MotionFeatures, ...]
    available_wrists: frozenset[WristSide]


def _routines() -> RoutineService:
    return cast(RoutineService, current_app.extensions["routine_service"])


def _evaluator() -> WindowScoringEvaluator:
    return cast(WindowScoringEvaluator, current_app.extensions["window_scoring_evaluator"])


def _scorer_registry() -> ScorerRegistry:
    return cast(ScorerRegistry, current_app.extensions["scorer_registry"])


@scoring_debug.before_request
def protect_debug_routes() -> tuple[Response, int] | None:
    remote = request.remote_addr
    try:
        loopback = remote is not None and ip_address(remote).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        return _error("forbidden", "debug scoring diagnostics are loopback-only", 403)
    if request.method == "POST":
        request.max_content_length = DEBUG_MAX_BODY_BYTES
        if request.content_length is not None and request.content_length > DEBUG_MAX_BODY_BYTES:
            return _error("request_too_large", f"JSON body exceeds {DEBUG_MAX_BODY_BYTES} bytes", 413)
    return None


@scoring_debug.after_request
def secure_debug_response(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@scoring_debug.errorhandler(RequestEntityTooLarge)
def oversized_request(_error_value: RequestEntityTooLarge) -> tuple[Response, int]:
    return _error("request_too_large", f"JSON body exceeds {DEBUG_MAX_BODY_BYTES} bytes", 413)


@scoring_debug.get("/")
def index() -> str:
    return render_template("scoring_diagnostics.html")


@scoring_debug.get("/api/routines")
def list_routines() -> Response:
    return jsonify({"routines": _routines().list(100)})


@scoring_debug.get("/api/routines/<routine_id>/windows")
def list_windows(routine_id: str) -> Response:
    metadata = _routines().metadata(routine_id)
    return jsonify(
        {
            "routine": {
                "routineID": metadata["routineID"],
                "title": metadata["title"],
                "duration": metadata["duration"],
            },
            "windows": _routines().windows(routine_id),
        }
    )


@scoring_debug.get("/api/routines/<routine_id>/windows/<int:window_index>")
def reference_window(routine_id: str, window_index: int) -> Response:
    context = _reference_context(routine_id, window_index)
    return jsonify(
        {
            "routineID": routine_id,
            "window": context.window.to_dict(),
            "sampleRateHz": _evaluator().sample_rate_hz,
            "availableWrists": [side.value for side in WristSide if side in context.available_wrists],
            "signalMeaning": _SIGNAL_MEANING,
            "referenceSignals": _signals(context.reference_samples, context.window.start_seconds),
        }
    )


@scoring_debug.post("/api/routines/<routine_id>/windows/<int:window_index>/attempts")
def score_attempt(routine_id: str, window_index: int) -> Response:
    context = _reference_context(routine_id, window_index)
    if not context.window.scoreable:
        raise ValueError("window is not scoreable")
    data = _json_object()
    _reject_unknown(data, {"activeWrists", "perturbation"}, "request")
    active_wrists = _active_wrists(data.get("activeWrists"), context.available_wrists)
    perturbation = _perturbation(data.get("perturbation", {}))
    performance = _mock_performance(context.reference_samples, active_wrists, perturbation)
    evaluation = _evaluator().evaluate(
        scorer=_scorer_registry().get("weighted_dtw"),
        index=context.window.window_index,
        start_seconds=context.window.start_seconds,
        end_seconds=context.window.end_seconds,
        reference_features=context.raw_features,
        performance_features=performance,
        active_wrists=active_wrists,
    )
    result = evaluation.result
    return jsonify(
        {
            "routineID": routine_id,
            "windowIndex": window_index,
            "mock": True,
            "activeWrists": [side.value for side in WristSide if side in active_wrists],
            "perturbation": perturbation.to_dict(),
            "metrics": {
                "score": result.value,
                "valid": result.valid,
                "coverage": evaluation.coverage,
                "quality": result.breakdown.quality,
                "breakdown": result.breakdown.to_dict(),
            },
            "performanceSignals": _signals(evaluation.performance.samples, context.window.start_seconds),
        }
    )


def _reference_context(routine_id: str, window_index: int) -> ReferenceContext:
    record = _routines().get_record(routine_id)
    window = next((item for item in record.windows if item.window_index == window_index), None)
    if window is None:
        raise NotFoundError(f"routine window not found: {routine_id}/{window_index}")
    raw = reference_features(deserialize_reference(record.reference_motion), window.start_seconds, window.end_seconds)
    initially_available = frozenset(item.wrist for item in raw)
    prepared = _evaluator().prepare_reference(
        start_seconds=window.start_seconds,
        end_seconds=window.end_seconds,
        reference_features=raw,
        active_wrists=initially_available,
    )
    available = frozenset(item.wrist for item in prepared)
    return ReferenceContext(record, window, raw, prepared, available)


def _json_object() -> dict[str, Any]:
    if not request.is_json:
        raise ValueError("Content-Type must be application/json")
    try:
        value: Any = json.loads(request.get_data(cache=False))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return cast(dict[str, Any], value)


def _reject_unknown(data: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(unknown)}")


def _active_wrists(value: Any, available: frozenset[WristSide]) -> frozenset[WristSide]:
    if value is None:
        if not available:
            raise ValueError("window has no reference wrist signals")
        return available
    if not isinstance(value, list) or not value:
        raise ValueError("activeWrists must be a non-empty list")
    raw = cast(list[Any], value)  # type: ignore[redundant-cast]
    if any(not isinstance(item, str) for item in raw):
        raise ValueError("activeWrists values must be strings")
    names = cast(list[str], raw)
    if len(names) != len(set(names)):
        raise ValueError("activeWrists values must be unique")
    try:
        selected = frozenset(WristSide(item) for item in names)
    except ValueError as exc:
        raise ValueError("activeWrists contains an unknown wrist") from exc
    if not selected <= available:
        raise ValueError("activeWrists must be a subset of availableWrists")
    return selected


def _perturbation(value: Any) -> Perturbation:
    if not isinstance(value, dict):
        raise ValueError("perturbation must be an object")
    data = cast(dict[str, Any], value)
    ranges = {
        "directionRotationDegrees": (-180.0, 180.0, 0.0),
        "intensityScale": (0.0, 2.0, 1.0),
        "timeShiftMs": (-500.0, 500.0, 0.0),
        "captureCoverage": (0.0, 1.0, 1.0),
        "sampleQuality": (0.0, 1.0, 1.0),
        "horizontalConfidence": (0.0, 1.0, 1.0),
    }
    _reject_unknown(data, set(ranges), "perturbation")
    parsed = {
        key: _bounded_number(data.get(key, default), key, lower, upper)
        for key, (lower, upper, default) in ranges.items()
    }
    return Perturbation(
        parsed["directionRotationDegrees"],
        parsed["intensityScale"],
        parsed["timeShiftMs"],
        parsed["captureCoverage"],
        parsed["sampleQuality"],
        parsed["horizontalConfidence"],
    )


def _bounded_number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be finite and between {lower:g} and {upper:g}")
    return result


def _mock_performance(
    reference: tuple[MotionFeatures, ...],
    active_wrists: frozenset[WristSide],
    perturbation: Perturbation,
) -> tuple[MotionFeatures, ...]:
    angle = radians(perturbation.direction_rotation_degrees)
    cosine = cos(angle)
    sine = sin(angle)
    shifted: list[MotionFeatures] = []
    for wrist in WristSide:
        if wrist not in active_wrists:
            continue
        stream = [item for item in reference if item.wrist is wrist]
        retained = floor(len(stream) * perturbation.capture_coverage + 1e-12)
        for item in stream[:retained]:
            horizontal = item.horizontal_direction or 0.0
            rotated_horizontal = horizontal * cosine - item.vertical_direction * sine
            rotated_vertical = horizontal * sine + item.vertical_direction * cosine
            timestamp = item.synchronized_time + perturbation.time_shift_ms / 1000.0
            if timestamp < 0:
                continue
            shifted.append(
                replace(
                    item,
                    synchronized_time=timestamp,
                    vertical_direction=rotated_vertical,
                    horizontal_direction=(rotated_horizontal if perturbation.horizontal_confidence >= 0.25 else None),
                    horizontal_confidence=perturbation.horizontal_confidence,
                    linear_intensity=item.linear_intensity * perturbation.intensity_scale,
                    sample_quality=perturbation.sample_quality,
                )
            )
    return tuple(sorted(shifted, key=lambda item: (item.synchronized_time, item.wrist.value)))


def _signals(samples: tuple[MotionFeatures, ...], start_seconds: float) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {side.value: [] for side in WristSide}
    for item in samples:
        horizontal_component = (
            None if item.horizontal_direction is None else item.horizontal_direction * item.linear_intensity
        )
        result[item.wrist.value].append(
            {
                "offsetSeconds": item.synchronized_time - start_seconds,
                "horizontalDirection": item.horizontal_direction,
                "verticalDirection": item.vertical_direction,
                "linearIntensity": item.linear_intensity,
                "horizontalComponent": horizontal_component,
                "verticalComponent": item.vertical_direction * item.linear_intensity,
                "movementActive": item.movement_active,
            }
        )
    return result


def _error(code: str, message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": {"code": code, "message": message}}), status
