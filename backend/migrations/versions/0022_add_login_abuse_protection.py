"""add login abuse protection state and aggregated audit fields

Revision ID: 0022
Revises: 0021
Create Date: 2026-07-30

This revision is retained because the first account-locking draft may already
have been applied in a developer database.  Revision 0023 replaces the global
account state with source-scoped throttles without rewriting migration history.
"""

from alembic import op
import sqlalchemy as sa


revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "failed_login_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "failed_login_window_started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "users",
        sa.Column("last_failed_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("login_locked_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column(
            "login_lock_level",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )

    op.add_column(
        "login_logs",
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.add_column(
        "login_logs",
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute("UPDATE login_logs SET last_attempt_at = created_at WHERE last_attempt_at IS NULL")
    op.alter_column("login_logs", "last_attempt_at", nullable=False)

    op.create_index(
        "ix_login_logs_success_created_at", "login_logs", ["success", "created_at"]
    )
    op.create_index(
        "ix_login_logs_username_created_at", "login_logs", ["username", "created_at"]
    )
    op.create_index("ix_login_logs_ip_created_at", "login_logs", ["ip", "created_at"])
    op.create_index("ix_login_logs_last_attempt_at", "login_logs", ["last_attempt_at"])


def downgrade() -> None:
    op.drop_index("ix_login_logs_last_attempt_at", table_name="login_logs")
    op.drop_index("ix_login_logs_ip_created_at", table_name="login_logs")
    op.drop_index("ix_login_logs_username_created_at", table_name="login_logs")
    op.drop_index("ix_login_logs_success_created_at", table_name="login_logs")
    op.drop_column("login_logs", "last_attempt_at")
    op.drop_column("login_logs", "attempt_count")

    op.drop_column("users", "login_lock_level")
    op.drop_column("users", "login_locked_until")
    op.drop_column("users", "last_failed_login_at")
    op.drop_column("users", "failed_login_window_started_at")
    op.drop_column("users", "failed_login_count")
