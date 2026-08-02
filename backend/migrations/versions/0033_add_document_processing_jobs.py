"""add durable document-processing job ledger

Revision ID: 0033
Revises: 0032
Create Date: 2026-08-02

Document upload used to create an in-process asyncio task.  The task vanished
on process restart and could race a later document edit.  This migration turns
the work into a revision-owned, lease-claimed database job instead.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision: str = "0033"
down_revision: Union[str, None] = "0032"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "processing_revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.create_table(
        "document_processing_jobs",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", UUID(as_uuid=True), nullable=False),
        sa.Column("document_revision", sa.Integer(), nullable=False),
        sa.Column("job_type", sa.String(length=16), nullable=False),
        sa.Column("payload", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(length=16), nullable=False, server_default=sa.text("'queued'")),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("job_type IN ('file', 'text', 'image')", name="ck_document_processing_jobs_type"),
        sa.CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'superseded')", name="ck_document_processing_jobs_status"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_document_processing_jobs_attempts"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "document_revision", name="uq_document_processing_jobs_document_revision"),
    )
    op.create_index(
        "ix_document_processing_jobs_claim",
        "document_processing_jobs",
        ["status", "available_at", "lease_expires_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_processing_jobs_claim", table_name="document_processing_jobs")
    op.drop_table("document_processing_jobs")
    op.drop_column("documents", "processing_revision")
