"""add is_active to documents

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-13
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"))


def downgrade() -> None:
    op.drop_column("documents", "is_active")
