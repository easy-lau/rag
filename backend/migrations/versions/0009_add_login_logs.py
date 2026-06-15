"""add login_logs

Revision ID: 0009
Revises: 0008
Create Date: 2026-06-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "login_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("success", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("ip", sa.String(64)),
        sa.Column("user_agent", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )
    op.create_index("ix_login_logs_created_at", "login_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_login_logs_created_at", table_name="login_logs")
    op.drop_table("login_logs")
