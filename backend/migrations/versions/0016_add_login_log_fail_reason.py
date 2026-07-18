"""add fail_reason to login_logs

Revision ID: 0016
Revises: 0015
Create Date: 2026-06-24
"""
from alembic import op
import sqlalchemy as sa

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("login_logs", sa.Column("fail_reason", sa.String(128), nullable=True))


def downgrade() -> None:
    op.drop_column("login_logs", "fail_reason")
