"""replace global account lock with source-scoped login throttles

Revision ID: 0023
Revises: 0022
Create Date: 2026-07-30
"""

from alembic import op
import sqlalchemy as sa


revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_login_logs_failure_aggregation",
        "login_logs",
        ["success", "username", "ip", "fail_reason", "last_attempt_at"],
    )

    op.create_table(
        "login_throttles",
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("bucket_key", sa.String(length=64), nullable=False),
        sa.Column(
            "failure_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "window_started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("blocked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "last_failed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint(
            "scope IN ('pair', 'ip', 'account')",
            name="ck_login_throttles_scope",
        ),
        sa.PrimaryKeyConstraint("scope", "bucket_key"),
    )
    op.create_index(
        "ix_login_throttles_last_failed_at",
        "login_throttles",
        ["last_failed_at"],
    )
    op.create_index(
        "ix_login_throttles_blocked_until",
        "login_throttles",
        ["blocked_until"],
    )

    # 0022 的账号级字段可能已经包含失败状态。不能把它迁入来源桶，因为没有
    # 可信来源 IP；直接删除可避免升级后继续误锁 admin。
    op.drop_column("users", "login_lock_level")
    op.drop_column("users", "login_locked_until")
    op.drop_column("users", "last_failed_login_at")
    op.drop_column("users", "failed_login_window_started_at")
    op.drop_column("users", "failed_login_count")


def downgrade() -> None:
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

    op.drop_index("ix_login_throttles_blocked_until", table_name="login_throttles")
    op.drop_index("ix_login_throttles_last_failed_at", table_name="login_throttles")
    op.drop_table("login_throttles")
    op.drop_index("ix_login_logs_failure_aggregation", table_name="login_logs")
