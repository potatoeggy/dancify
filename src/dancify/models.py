"""Relational persistence models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from dancify.extensions import Base


class DanceRoutineRecord(Base):
    __tablename__ = "dance_routines"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    source_video_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    reference_motion: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    windows: Mapped[list[RoutineWindowRecord]] = relationship(
        back_populates="routine", cascade="all, delete-orphan", order_by="RoutineWindowRecord.window_index"
    )

    def metadata_dict(self) -> dict[str, object]:
        return {
            "routineID": self.id,
            "title": self.title,
            "sourceVideoURL": self.source_video_url,
            "duration": self.duration_seconds,
            "fps": self.fps,
            "schemaVersion": self.schema_version,
        }


class RoutineWindowRecord(Base):
    __tablename__ = "routine_windows"
    __table_args__ = (UniqueConstraint("routine_id", "window_index"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    routine_id: Mapped[str] = mapped_column(ForeignKey("dance_routines.id", ondelete="CASCADE"), nullable=False)
    window_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    scoreable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    routine: Mapped[DanceRoutineRecord] = relationship(back_populates="windows")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.window_index,
            "startTime": self.start_seconds,
            "endTime": self.end_seconds,
            "scoreable": self.scoreable,
        }


class SessionSummaryRecord(Base):
    __tablename__ = "session_summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    routine_id: Mapped[str] = mapped_column(ForeignKey("dance_routines.id"), nullable=False)
    player_id: Mapped[str] = mapped_column(String(120), nullable=False)
    terminal_state: Mapped[str] = mapped_column(String(20), nullable=False)
    cumulative_score: Mapped[float] = mapped_column(Float, nullable=False)
    scored_windows: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
