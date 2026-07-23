"""Application services coordinating routines, calibration, sessions, and scoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from dancify.calibration import (
    AffineClockMapper,
    ClockObservation,
    SpatialCalibrationProfile,
    TimingCalibrationService,
    resample_features,
)
from dancify.domain import (
    GameSession,
    MotionFeatures,
    MotionWindow,
    RawImuSample,
    ScoreResult,
    SessionState,
    Vector3,
    WristSide,
)
from dancify.ingestion import (
    deserialize_reference,
    parse_routine,
    reference_features,
    serialize_reference,
)
from dancify.models import DanceRoutineRecord
from dancify.repositories import RoutineRepository, SessionSummaryRepository
from dancify.scoring import (
    ArithmeticMeanScoreAggregator,
    FixedWindowingStrategy,
    ScorerRegistry,
    ScoringAlgorithm,
)

EventPublisher = Callable[[str, str, dict[str, object]], None]


class NotFoundError(LookupError):
    pass


class ConflictError(ValueError):
    pass


@dataclass(slots=True)
class SessionRuntime:
    scorer: ScoringAlgorithm
    windowing: FixedWindowingStrategy = field(default_factory=FixedWindowingStrategy)
    aggregator: ArithmeticMeanScoreAggregator = field(default_factory=ArithmeticMeanScoreAggregator)
    clock_mapper: AffineClockMapper = field(default_factory=AffineClockMapper)
    spatial_profile: SpatialCalibrationProfile | None = None
    performance: list[MotionFeatures] = field(default_factory=list[MotionFeatures])


class RoutineService:
    def __init__(self, repository: RoutineRepository | None = None) -> None:
        self._repository = repository or RoutineRepository()

    def create(self, payload: dict[str, Any]) -> dict[str, object]:
        parsed = parse_routine(payload)
        routine_id = str(uuid4())
        record = self._repository.add(
            routine_id=routine_id,
            title=parsed.title,
            source_video_url=parsed.source_video_url,
            duration_seconds=parsed.duration_seconds,
            fps=parsed.fps,
            reference_motion=serialize_reference(parsed.reference_motion),
            windows=parsed.scoring_windows,
        )
        return record.metadata_dict()

    def get_record(self, routine_id: str) -> DanceRoutineRecord:
        record = self._repository.get(routine_id)
        if record is None:
            raise NotFoundError(f"routine not found: {routine_id}")
        return record

    def metadata(self, routine_id: str) -> dict[str, object]:
        return self.get_record(routine_id).metadata_dict()

    def windows(self, routine_id: str) -> list[dict[str, object]]:
        return [window.to_dict() for window in self.get_record(routine_id).windows]


class GameplaySessionService:
    def __init__(
        self,
        routines: RoutineService,
        scorers: ScorerRegistry,
        publisher: EventPublisher,
        summaries: SessionSummaryRepository | None = None,
        clock: Callable[[], float] = monotonic,
        sample_rate_hz: int = 50,
    ) -> None:
        self._routines = routines
        self._scorers = scorers
        self._publisher = publisher
        self._summaries = summaries or SessionSummaryRepository()
        self._clock = clock
        self._sample_rate_hz = sample_rate_hz
        self._sessions: dict[str, GameSession] = {}
        self._runtime: dict[str, SessionRuntime] = {}

    def create(self, routine_id: str, player_id: str, scorer_name: str = "weighted_dtw") -> GameSession:
        self._routines.get_record(routine_id)
        if not player_id.strip():
            raise ValueError("playerID is required")
        scorer = self._scorers.get(scorer_name)
        session = GameSession(str(uuid4()), routine_id, player_id.strip())
        self._sessions[session.id] = session
        self._runtime[session.id] = SessionRuntime(scorer)
        return session

    def get(self, session_id: str) -> GameSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise NotFoundError(f"session not found: {session_id}") from exc

    def snapshot(self, session_id: str) -> dict[str, object]:
        return self.get(session_id).snapshot()

    def calibrate(
        self,
        session_id: str,
        observations: list[ClockObservation],
        neutral: list[Vector3],
        upward: list[Vector3],
        outward: list[Vector3],
    ) -> dict[str, object]:
        session = self.get(session_id)
        if session.state == SessionState.CREATED:
            self._transition(session, SessionState.CALIBRATING)
        if session.state != SessionState.CALIBRATING:
            raise ConflictError("session must be calibrating")
        offset = TimingCalibrationService.estimate_offset(observations)
        runtime = self._runtime[session_id]
        runtime.clock_mapper = AffineClockMapper(offset_seconds=offset)
        runtime.spatial_profile = SpatialCalibrationProfile.from_gestures(neutral, upward, outward, self._clock())
        self._transition(session, SessionState.READY)
        result: dict[str, object] = {
            "timingOffsetSeconds": offset,
            "horizontalConfidence": runtime.spatial_profile.horizontal_confidence,
        }
        self._emit(session, "calibration.result", result)
        return result

    def start(self, session_id: str, delay_seconds: float = 1.0) -> dict[str, object]:
        session = self.get(session_id)
        if session.state != SessionState.READY:
            raise ConflictError("session must be ready before playback starts")
        if delay_seconds < 0:
            raise ValueError("delaySeconds must be non-negative")
        session.playback_start_time = self._clock() + delay_seconds
        self._transition(session, SessionState.SCHEDULED)
        payload: dict[str, object] = {"startAt": session.playback_start_time}
        self._emit(session, "playback.scheduled", payload)
        return payload

    def ingest_features(self, session_id: str, features: list[MotionFeatures]) -> int:
        session = self.get(session_id)
        if session.state not in {
            SessionState.SCHEDULED,
            SessionState.PLAYING,
            SessionState.PAUSED,
        }:
            raise ConflictError("session is not accepting motion")
        runtime = self._runtime[session_id]
        runtime.performance.extend(features)
        runtime.performance.sort(key=lambda item: (item.synchronized_time, item.wrist.value))
        return len(features)

    def ingest_raw_samples(self, session_id: str, samples: list[RawImuSample]) -> int:
        """Translate capture-port output without exposing hardware details to scoring."""
        runtime = self._runtime[self.get(session_id).id]
        profile = self.calibration_profile(session_id)
        features: list[MotionFeatures] = []
        for sample in samples:
            lowered = sample.device_id.lower()
            if "left" in lowered:
                wrist = WristSide.LEFT
            elif "right" in lowered:
                wrist = WristSide.RIGHT
            else:
                raise ValueError("device_id must identify the left or right wrist")
            features.append(profile.translate(sample, wrist, runtime.clock_mapper))
        return self.ingest_features(session_id, features)

    def progress(self, session_id: str, video_time: float, server_time: float | None = None) -> list[ScoreResult]:
        session = self.get(session_id)
        if video_time < 0:
            raise ValueError("videoTime must be non-negative")
        now = self._clock() if server_time is None else server_time
        if session.state == SessionState.SCHEDULED:
            if session.playback_start_time is None or now < session.playback_start_time:
                return []
            self._transition(session, SessionState.PLAYING)
        if session.state not in {SessionState.PLAYING, SessionState.PAUSED}:
            raise ConflictError("session is not playing")
        assert session.playback_start_time is not None
        drift = abs((now - session.playback_start_time) - video_time)
        if drift > 0.5:
            if session.state != SessionState.PAUSED:
                self._transition(session, SessionState.PAUSED)
                self._emit(session, "session.paused", {"reason": "synchronization_lost", "driftSeconds": drift})
            return []
        if session.state == SessionState.PAUSED:
            self._transition(session, SessionState.PLAYING)
        session.current_timestamp = video_time
        results = self._score_completed(session)
        routine = self._routines.get_record(session.routine_id)
        if video_time >= routine.duration_seconds:
            self._transition(session, SessionState.COMPLETED)
            self._summaries.save(session)
            self._emit(
                session,
                "session.completed",
                {"cumulativeScore": session.cumulative_score, "scoredWindows": len(session.scored_windows)},
            )
        return results

    def abort(self, session_id: str) -> GameSession:
        session = self.get(session_id)
        if session.state in {SessionState.COMPLETED, SessionState.ABORTED}:
            raise ConflictError("session is already terminal")
        self._transition(session, SessionState.ABORTED)
        self._summaries.save(session)
        return session

    def calibration_profile(self, session_id: str) -> SpatialCalibrationProfile:
        profile = self._runtime[self.get(session_id).id].spatial_profile
        if profile is None:
            raise ConflictError("session is not calibrated")
        return profile

    def _score_completed(self, session: GameSession) -> list[ScoreResult]:
        runtime = self._runtime[session.id]
        routine = self._routines.get_record(session.routine_id)
        raw_reference = deserialize_reference(routine.reference_motion)
        results: list[ScoreResult] = []
        for index in runtime.windowing.completed_windows(session.current_timestamp):
            if index in session.scored_windows or index >= len(routine.windows):
                continue
            window_record = routine.windows[index]
            session.scored_windows.add(index)
            session.current_window = index
            if not window_record.scoreable:
                continue
            reference_raw = reference_features(raw_reference, window_record.start_seconds, window_record.end_seconds)
            reference_samples = resample_features(
                reference_raw, window_record.start_seconds, window_record.end_seconds, self._sample_rate_hz
            )
            performance_raw = tuple(
                item
                for item in runtime.performance
                if window_record.start_seconds <= item.synchronized_time < window_record.end_seconds
            )
            performance_samples = resample_features(
                performance_raw, window_record.start_seconds, window_record.end_seconds, self._sample_rate_hz
            )
            expected = max(1, int((window_record.end_seconds - window_record.start_seconds) * self._sample_rate_hz) * 2)
            coverage = min(1.0, len(performance_samples) / expected)
            quality_values = [item.sample_quality for item in performance_samples]
            quality = coverage * (sum(quality_values) / len(quality_values) if quality_values else 0.0)
            valid = coverage >= 0.5
            reference_window = MotionWindow(
                index, window_record.start_seconds, window_record.end_seconds, reference_samples
            )
            performance_window = MotionWindow(
                index, window_record.start_seconds, window_record.end_seconds, performance_samples, valid, quality
            )
            scored = runtime.scorer.score(reference_window, performance_window)
            cumulative = runtime.aggregator.add(scored.value)
            result = ScoreResult(
                scored.window_index,
                scored.window_start_seconds,
                scored.value,
                cumulative,
                scored.valid,
                scored.breakdown,
            )
            session.cumulative_score = cumulative
            results.append(result)
            self._emit(session, "score.update", result.to_dict())
        return results

    def _transition(self, session: GameSession, target: SessionState) -> None:
        allowed: dict[SessionState, set[SessionState]] = {
            SessionState.CREATED: {SessionState.CALIBRATING, SessionState.ABORTED},
            SessionState.CALIBRATING: {SessionState.READY, SessionState.ABORTED},
            SessionState.READY: {SessionState.SCHEDULED, SessionState.ABORTED},
            SessionState.SCHEDULED: {SessionState.PLAYING, SessionState.ABORTED},
            SessionState.PLAYING: {SessionState.PAUSED, SessionState.COMPLETED, SessionState.ABORTED},
            SessionState.PAUSED: {SessionState.PLAYING, SessionState.ABORTED},
            SessionState.COMPLETED: set(),
            SessionState.ABORTED: set(),
        }
        if target not in allowed[session.state]:
            raise ConflictError(f"cannot transition from {session.state.value} to {target.value}")
        session.state = target
        self._emit(session, "session.snapshot", session.snapshot())

    def _emit(self, session: GameSession, event: str, payload: dict[str, object]) -> None:
        session.event_sequence += 1
        envelope = {
            "schemaVersion": 1,
            "sessionID": session.id,
            "sequence": session.event_sequence,
            "serverTimestamp": self._clock(),
            **payload,
        }
        self._publisher(session.id, event, envelope)


def require_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return cast(dict[str, Any], value)
