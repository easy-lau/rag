"""add persistent RAG trace runs and events

Revision ID: 0025
Revises: 0024
Create Date: 2026-07-30
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0025"
down_revision: Union[str, None] = "0024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "rag_trace_runs",
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("request_kind", sa.String(length=32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("current_stage", sa.String(length=100), nullable=True),
        sa.Column("event_count", sa.Integer(), nullable=False),
        sa.Column("content_included", sa.Boolean(), nullable=False),
        sa.Column("input_preview", sa.Text(), nullable=True),
        sa.Column("output_preview", sa.Text(), nullable=True),
        sa.Column("evidence_status", sa.String(length=32), nullable=True),
        sa.Column("selected_kb_count", sa.Integer(), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("trace_id"),
    )
    op.create_index("ix_rag_trace_runs_started_at", "rag_trace_runs", ["started_at"])
    op.create_index(
        "ix_rag_trace_runs_status_started_at", "rag_trace_runs", ["status", "started_at"]
    )
    op.create_index(
        "ix_rag_trace_runs_user_started_at", "rag_trace_runs", ["user_id", "started_at"]
    )
    op.create_index(
        "ix_rag_trace_runs_conversation_started_at",
        "rag_trace_runs",
        ["conversation_id", "started_at"],
    )

    op.create_table(
        "rag_trace_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event", sa.String(length=100), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["trace_id"], ["rag_trace_runs.trace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("trace_id", "sequence", name="uq_rag_trace_event_sequence"),
    )
    op.create_index(
        "ix_rag_trace_events_event_created_at", "rag_trace_events", ["event", "created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_rag_trace_events_event_created_at", table_name="rag_trace_events")
    op.drop_table("rag_trace_events")
    op.drop_index("ix_rag_trace_runs_conversation_started_at", table_name="rag_trace_runs")
    op.drop_index("ix_rag_trace_runs_user_started_at", table_name="rag_trace_runs")
    op.drop_index("ix_rag_trace_runs_status_started_at", table_name="rag_trace_runs")
    op.drop_index("ix_rag_trace_runs_started_at", table_name="rag_trace_runs")
    op.drop_table("rag_trace_runs")
