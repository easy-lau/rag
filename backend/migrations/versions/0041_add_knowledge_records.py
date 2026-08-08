"""add source-addressable structured knowledge records

Revision ID: 0041
Revises: 0040
Create Date: 2026-08-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0041"
down_revision: Union[str, None] = "0040"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kb_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("doc_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("record_type", sa.String(length=24), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("predicate", sa.Text(), nullable=True),
        sa.Column("object_value", sa.Text(), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "record_type IN ('table_row', 'key_value')",
            name="ck_knowledge_records_type",
        ),
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["document_chunks.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_records_kb_id", "knowledge_records", ["kb_id"])
    op.create_index("ix_knowledge_records_doc_id", "knowledge_records", ["doc_id"])
    op.create_index("ix_knowledge_records_chunk_id", "knowledge_records", ["chunk_id"])
    op.execute(
        "CREATE INDEX ix_knowledge_records_search_fts ON knowledge_records "
        "USING gin (to_tsvector('simple', search_text))"
    )
    op.execute(
        "CREATE INDEX ix_knowledge_records_search_trgm ON knowledge_records "
        "USING gin (search_text gin_trgm_ops)"
    )


def downgrade() -> None:
    op.drop_table("knowledge_records")
