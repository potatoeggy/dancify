"""Persistence repositories."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dancify.domain import GameSession, ScoringWindow, SessionState
from dancify.extensions import db
from dancify.models import DanceRoutineRecord, RoutineWindowRecord, SessionSummaryRecord


class RoutineRepository:
    def add(
        self,
        *,
        routine_id: str,
        title: str,
        source_video_url: str,
        duration_seconds: float,
        fps: float,
        reference_motion: list[dict[str, Any]],
        windows: tuple[ScoringWindow, ...],
    ) -> DanceRoutineRecord:
        record = DanceRoutineRecord(
            id=routine_id,
            title=title,
            source_video_url=source_video_url,
            duration_seconds=duration_seconds,
            fps=fps,
            reference_motion=reference_motion,
        )
        record.windows = [
            RoutineWindowRecord(
                window_index=item.index,
                start_seconds=item.start_seconds,
                end_seconds=item.end_seconds,
                scoreable=item.scoreable,
            )
            for item in windows
        ]
        db.session.add(record)
        db.session.commit()
        return record

    def get(self, routine_id: str) -> DanceRoutineRecord | None:
        return db.session.get(DanceRoutineRecord, routine_id)

    def list(self, limit: int = 100) -> list[DanceRoutineRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("routine list limit must be between 1 and 100")
        statement = db.select(DanceRoutineRecord).order_by(DanceRoutineRecord.created_at.desc()).limit(limit)
        return list(db.session.scalars(statement))


class SessionSummaryRepository:
    def save(self, session: GameSession) -> SessionSummaryRecord:
        if session.state not in {SessionState.COMPLETED, SessionState.ABORTED}:
            raise ValueError("only terminal sessions can be persisted")
        existing = db.session.get(SessionSummaryRecord, session.id)
        if existing is not None:
            return existing
        record = SessionSummaryRecord(
            id=session.id,
            routine_id=session.routine_id,
            player_id=session.player_id,
            terminal_state=session.state.value,
            cumulative_score=session.cumulative_score,
            scored_windows=len(session.scored_windows),
            completed_at=datetime.now(UTC),
        )
        db.session.add(record)
        db.session.commit()
        return record

    def get(self, session_id: str) -> SessionSummaryRecord | None:
        return db.session.get(SessionSummaryRecord, session_id)
