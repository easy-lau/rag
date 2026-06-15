"""drop knowledge_bases.tags (replaced by document-level tags)

Revision ID: 0012
Revises: 0011
Create Date: 2026-06-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("knowledge_bases", "tags")


def downgrade() -> None:
    op.add_column(
        "knowledge_bases",
        sa.Column("tags", JSONB, nullable=False, server_default=sa.text("'[]'")),
    )
