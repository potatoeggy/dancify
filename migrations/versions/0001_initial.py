"""Create routine, scoring-window, and session-summary tables.

Revision ID: 0001
Revises:
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "dance_routines",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("source_video_url", sa.String(length=2048), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("reference_motion", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "routine_windows",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("routine_id", sa.String(length=36), nullable=False),
        sa.Column("window_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("scoreable", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["routine_id"], ["dance_routines.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("routine_id", "window_index"),
    )
    op.create_table(
        "session_summaries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("routine_id", sa.String(length=36), nullable=False),
        sa.Column("player_id", sa.String(length=120), nullable=False),
        sa.Column("terminal_state", sa.String(length=20), nullable=False),
        sa.Column("cumulative_score", sa.Float(), nullable=False),
        sa.Column("scored_windows", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["routine_id"], ["dance_routines.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("session_summaries")
    op.drop_table("routine_windows")
    op.drop_table("dance_routines")
