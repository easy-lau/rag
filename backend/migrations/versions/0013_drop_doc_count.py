"""drop knowledge_bases.doc_count (now derived from real document rows)

Revision ID: 0013
Revises: 0012
Create Date: 2026-06-14
"""
from alembic import op
import sqlalchemy as sa

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("knowledge_bases", "doc_count")


def downgrade() -> None:
    op.add_column(
        "knowledge_bases",
        sa.Column("doc_count", sa.Integer(), nullable=False, server_default="0"),
    )
