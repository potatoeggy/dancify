"""Application services coordinating routines, calibration, sessions, and scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from threading import RLock
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from dancify.calibration import (
    AffineClockMapper,
    CaptureClockMapper,
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
    RawMotionSample,
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
class WristStreamHealth:
    accepted: int = 0
    dropped: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    invalid_timing: int = 0

    def snapshot(self) -> dict[str, object]:
        attempted = self.accepted + self.dropped
        return {
            "accepted": self.accepted,
            "dropped": self.dropped,
            "duplicates": self.duplicates,
            "outOfOrder": self.out_of_order,
            "invalidTiming": self.invalid_timing,
            "quality": self.accepted / attempted if attempted else 1.0,
        }


@dataclass(slots=True)
class SessionRuntime:
    scorer: ScoringAlgorithm
    windowing: FixedWindowingStrategy = field(default_factory=FixedWindowingStrategy)
    aggregator: ArithmeticMeanScoreAggregator = field(default_factory=ArithmeticMeanScoreAggregator)
    clock_mapper: AffineClockMapper = field(default_factory=AffineClockMapper)
    spatial_profiles: dict[WristSide, SpatialCalibrationProfile] = field(
        default_factory=dict[WristSide, SpatialCalibrationProfile]
    )
    calibration_version: int = 0
    capture_mappers: dict[WristSide, CaptureClockMapper] = field(
        default_factory=lambda: {side: CaptureClockMapper() for side in WristSide}
    )
    raw_health: dict[WristSide, WristStreamHealth] = field(
        default_factory=lambda: {side: WristStreamHealth() for side in WristSide}
    )
    last_packet: dict[WristSide, int] = field(default_factory=dict[WristSide, int])
    malformed_samples: int = 0
    performance: list[MotionFeatures] = field(default_factory=list[MotionFeatures])
    lock: RLock = field(default_factory=RLock)

    def motion_health(self) -> dict[str, object]:
        wrists = {side.value: self.raw_health[side].snapshot() for side in WristSide}
        accepted = sum(self.raw_health[side].accepted for side in WristSide)
        dropped = self.malformed_samples + sum(self.raw_health[side].dropped for side in WristSide)
        attempted = accepted + dropped
        return {
            "accepted": accepted,
            "dropped": dropped,
            "malformed": self.malformed_samples,
            "quality": accepted / attempted if attempted else 1.0,
            "wrists": wrists,
        }


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
        max_raw_batch: int = 1000,
    ) -> None:
        if sample_rate_hz <= 0 or max_raw_batch <= 0:
            raise ValueError("sample rate and raw batch limit must be positive")
        self._routines = routines
        self._scorers = scorers
        self._publisher = publisher
        self._summaries = summaries or SessionSummaryRepository()
        self._clock = clock
        self._sample_rate_hz = sample_rate_hz
        self._max_raw_batch = max_raw_batch
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

    def retry(self, session_id: str) -> GameSession:
        """Clone an aborted session; retries without reusable calibration remain CREATED."""
        source = self.get(session_id)
        source_runtime = self._runtime[session_id]
        with source_runtime.lock:
            if source.state != SessionState.ABORTED:
                raise ConflictError("only aborted sessions can be retried")
            state = SessionState.READY if source_runtime.spatial_profiles else SessionState.CREATED
            retried = GameSession(str(uuid4()), source.routine_id, source.player_id, state=state)
            self._sessions[retried.id] = retried
            self._runtime[retried.id] = SessionRuntime(
                scorer=source_runtime.scorer,
                windowing=source_runtime.windowing,
                clock_mapper=source_runtime.clock_mapper,
                spatial_profiles=dict(source_runtime.spatial_profiles),
                calibration_version=source_runtime.calibration_version,
            )
            return retried

    def get(self, session_id: str) -> GameSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise NotFoundError(f"session not found: {session_id}") from exc

    def server_timestamp(self) -> float:
        return self._clock()

    def snapshot(self, session_id: str) -> dict[str, object]:
        session = self.get(session_id)
        runtime = self._runtime[session_id]
        with runtime.lock:
            snapshot = session.snapshot()
            snapshot["calibrationVersion"] = runtime.calibration_version
            snapshot["activeWrists"] = [side.value for side in WristSide if side in runtime.spatial_profiles]
            snapshot["motionHealth"] = runtime.motion_health()
            return snapshot

    def calibrate(
        self,
        session_id: str,
        observations: list[ClockObservation],
        neutral: list[Vector3],
        upward: list[Vector3],
        outward: list[Vector3],
        *,
        calibration_version: int = 1,
        wrist_gestures: Mapping[WristSide, tuple[list[Vector3], list[Vector3], list[Vector3]]] | None = None,
    ) -> dict[str, object]:
        session = self.get(session_id)
        runtime = self._runtime[session_id]
        with runtime.lock:
            if session.state == SessionState.CREATED:
                self._transition(session, SessionState.CALIBRATING)
            if session.state != SessionState.CALIBRATING:
                raise ConflictError("session must be calibrating")
            if calibration_version not in {1, 2}:
                raise ValueError("calibration schemaVersion must be 1 or 2")
            offset = TimingCalibrationService.estimate_offset(observations)
            calibrated_at = self._clock()
            if calibration_version == 1:
                gestures = (neutral, upward, outward)
                profiles = {
                    side: SpatialCalibrationProfile.from_gestures(*gestures, calibrated_at) for side in WristSide
                }
            else:
                if wrist_gestures is None:
                    raise ValueError("schemaVersion 2 calibration requires right or left and right wrists")
                wrist_set = set(wrist_gestures)
                if WristSide.RIGHT not in wrist_set or wrist_set - set(WristSide):
                    raise ValueError("schemaVersion 2 calibration requires right or left and right wrists")
                profiles = {
                    side: SpatialCalibrationProfile.from_gestures(*wrist_gestures[side], calibrated_at)
                    for side in WristSide
                    if side in wrist_gestures
                }
            runtime.clock_mapper = AffineClockMapper(offset_seconds=offset)
            runtime.spatial_profiles = profiles
            runtime.calibration_version = calibration_version
            self._transition(session, SessionState.READY)
            confidence = min(profile.horizontal_confidence for profile in profiles.values())
            result: dict[str, object] = {
                "timingOffsetSeconds": offset,
                "horizontalConfidence": confidence,
            }
            if calibration_version == 2:
                result.update(
                    {
                        "schemaVersion": 2,
                        "wrists": {
                            side.value: {"horizontalConfidence": profiles[side].horizontal_confidence}
                            for side in WristSide
                            if side in profiles
                        },
                    }
                )
            self._emit(session, "calibration.result", result)
            return result

    def start(self, session_id: str, delay_seconds: float = 1.0) -> dict[str, object]:
        session = self.get(session_id)
        runtime = self._runtime[session_id]
        with runtime.lock:
            if session.state != SessionState.READY:
                raise ConflictError("session must be ready before playback starts")
            if not isfinite(delay_seconds) or delay_seconds < 0:
                raise ValueError("delaySeconds must be finite and non-negative")
            session.playback_start_time = self._clock() + delay_seconds
            self._transition(session, SessionState.SCHEDULED)
            payload: dict[str, object] = {"startAt": session.playback_start_time}
            self._emit(session, "playback.scheduled", payload)
            return payload

    def ingest_features(self, session_id: str, features: list[MotionFeatures]) -> int:
        session = self.get(session_id)
        runtime = self._runtime[session_id]
        with runtime.lock:
            self._require_motion_state(session)
            runtime.performance.extend(features)
            runtime.performance.sort(key=lambda item: (item.synchronized_time, item.wrist.value))
            return len(features)

    def ingest_raw_samples(self, session_id: str, samples: list[RawImuSample]) -> int:
        """Translate legacy capture-port output; HTTP raw uploads require explicit wrists."""
        runtime = self._runtime[self.get(session_id).id]
        features: list[MotionFeatures] = []
        for sample in samples:
            lowered = sample.device_id.lower()
            if "left" in lowered:
                wrist = WristSide.LEFT
            elif "right" in lowered:
                wrist = WristSide.RIGHT
            else:
                raise ValueError("device_id must identify the left or right wrist")
            features.append(self.calibration_profile(session_id, wrist).translate(sample, wrist, runtime.clock_mapper))
        return self.ingest_features(session_id, features)

    def ingest_raw_motion(
        self,
        session_id: str,
        samples: list[tuple[int, RawMotionSample]],
        malformed: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """Ingest an item-wise tolerant live batch and return authoritative health."""
        session = self.get(session_id)
        runtime = self._runtime[session_id]
        parse_errors = list(malformed or [])
        if len(samples) + len(parse_errors) > self._max_raw_batch:
            raise ValueError(f"raw motion batch exceeds limit of {self._max_raw_batch}")
        with runtime.lock:
            self._require_motion_state(session)
            if session.playback_start_time is None:
                raise ConflictError("playback has not been scheduled")
            runtime.malformed_samples += len(parse_errors)
            errors = parse_errors
            accepted_features: list[MotionFeatures] = []
            routine = self._routines.get_record(session.routine_id)
            for index, upload in samples:
                health = runtime.raw_health[upload.wrist]
                if upload.wrist not in runtime.spatial_profiles:
                    health.dropped += 1
                    errors.append(
                        {
                            "index": index,
                            "code": "uncalibrated_wrist",
                            "message": f"{upload.wrist.value} wrist is not calibrated",
                        }
                    )
                    continue
                previous = runtime.last_packet.get(upload.wrist)
                if previous is not None and upload.packet_number <= previous:
                    health.dropped += 1
                    if upload.packet_number == previous:
                        health.duplicates += 1
                        code = "duplicate_packet"
                    else:
                        health.out_of_order += 1
                        code = "out_of_order_packet"
                    errors.append({"index": index, "code": code, "message": code.replace("_", " ")})
                    continue
                runtime.last_packet[upload.wrist] = upload.packet_number
                capture_mapper = runtime.capture_mappers[upload.wrist]
                client_capture_time = capture_mapper.observe(upload.capture_timestamp_us, upload.client_timestamp)
                server_capture_time = client_capture_time + runtime.clock_mapper.offset_seconds
                playback_time = server_capture_time - session.playback_start_time
                if playback_time < 0 or playback_time > routine.duration_seconds:
                    health.dropped += 1
                    health.invalid_timing += 1
                    code = "before_playback" if playback_time < 0 else "after_playback"
                    errors.append(
                        {
                            "index": index,
                            "code": code,
                            "message": "sample capture time is outside the playback interval",
                        }
                    )
                    continue
                server_mapper = AffineClockMapper(
                    capture_mapper.scale,
                    capture_mapper.offset_seconds + runtime.clock_mapper.offset_seconds,
                )
                translated = runtime.spatial_profiles[upload.wrist].translate(
                    upload.raw_sample(), upload.wrist, server_mapper
                )
                accepted_features.append(replace(translated, synchronized_time=playback_time))
                health.accepted += 1
            runtime.performance.extend(accepted_features)
            runtime.performance.sort(key=lambda item: (item.synchronized_time, item.wrist.value))
            motion_health = runtime.motion_health()
            self._emit(session, "motion.health", {"motionHealth": motion_health})
            return {
                "accepted": len(accepted_features),
                "dropped": len(errors),
                "errors": errors,
                "motionHealth": motion_health,
            }

    @staticmethod
    def _require_motion_state(session: GameSession) -> None:
        if session.state not in {SessionState.SCHEDULED, SessionState.PLAYING, SessionState.PAUSED}:
            raise ConflictError("session is not accepting motion")

    def progress(self, session_id: str, video_time: float, server_time: float | None = None) -> list[ScoreResult]:
        session = self.get(session_id)
        runtime = self._runtime[session_id]
        with runtime.lock:
            if not isfinite(video_time) or video_time < 0:
                raise ValueError("videoTime must be finite and non-negative")
            if server_time is not None and not isfinite(server_time):
                raise ValueError("serverTime must be finite")
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
        runtime = self._runtime[session_id]
        with runtime.lock:
            if session.state in {SessionState.COMPLETED, SessionState.ABORTED}:
                raise ConflictError("session is already terminal")
            self._transition(session, SessionState.ABORTED)
            self._summaries.save(session)
            return session

    def calibration_profile(self, session_id: str, wrist: WristSide = WristSide.LEFT) -> SpatialCalibrationProfile:
        runtime = self._runtime[self.get(session_id).id]
        try:
            return runtime.spatial_profiles[wrist]
        except KeyError as exc:
            raise ConflictError(f"{wrist.value} wrist is not calibrated") from exc

    def _score_completed(self, session: GameSession) -> list[ScoreResult]:
        runtime = self._runtime[session.id]
        routine = self._routines.get_record(session.routine_id)
        raw_reference = deserialize_reference(routine.reference_motion)
        active_wrists = set(runtime.spatial_profiles)
        results: list[ScoreResult] = []
        for index in runtime.windowing.completed_windows(session.current_timestamp):
            if index in session.scored_windows or index >= len(routine.windows):
                continue
            window_record = routine.windows[index]
            session.scored_windows.add(index)
            session.current_window = index
            if not window_record.scoreable:
                continue
            reference_raw = tuple(
                item
                for item in reference_features(raw_reference, window_record.start_seconds, window_record.end_seconds)
                if item.wrist in active_wrists
            )
            reference_samples = resample_features(
                reference_raw, window_record.start_seconds, window_record.end_seconds, self._sample_rate_hz
            )
            performance_raw = tuple(
                item
                for item in runtime.performance
                if item.wrist in active_wrists
                and window_record.start_seconds <= item.synchronized_time < window_record.end_seconds
            )
            performance_samples = resample_features(
                performance_raw, window_record.start_seconds, window_record.end_seconds, self._sample_rate_hz
            )
            expected = max(
                1,
                int((window_record.end_seconds - window_record.start_seconds) * self._sample_rate_hz)
                * len(active_wrists),
            )
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
            session.score_results[index] = result
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
        self._emit(session, "session.snapshot", self.snapshot(session.id))

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
