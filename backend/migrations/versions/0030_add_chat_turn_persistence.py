"""add durable idempotent chat-turn persistence

Revision ID: 0030
Revises: 0029
Create Date: 2026-08-01

The transcript alone cannot distinguish a generated answer from a delivered
one.  ``chat_turns`` stages the generated payload before inserting the
assistant message and provides a conversation-scoped idempotency key.  Nullable
message metadata keeps existing transcript rows backward compatible while
allowing the history API to expose delivery and evidence state.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision: str = "0030"
down_revision: Union[str, None] = "0029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TURN_STATUSES = (
    "accepted",
    "generating",
    "generated",
    "completed",
    "persist_failed",
    "failed",
    "cancelled",
)


def upgrade() -> None:
    op.create_table(
        "chat_turns",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("question_hash", sa.String(length=64), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("request_context", JSONB(), nullable=False),
        sa.Column("resume_context", JSONB(), nullable=True),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column(
            "status",
            sa.String(length=24),
            nullable=False,
            server_default=sa.text("'accepted'"),
        ),
        sa.Column("lease_owner", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "execution_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column("evidence_status", sa.String(length=32), nullable=True),
        sa.Column("retrieval_executed", sa.Boolean(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("answer_content", sa.Text(), nullable=True),
        sa.Column("answer_sources", JSONB(), nullable=True),
        sa.Column("search_snapshot", JSONB(), nullable=True),
        sa.Column("tokens", sa.Integer(), nullable=True),
        # These are stable identifiers rather than foreign keys.  The turn is
        # the recovery ledger and must survive a missing/rolled-back transcript
        # row so a retry can repair it.
        sa.Column("user_message_id", UUID(as_uuid=True), nullable=True),
        sa.Column("assistant_message_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "persistence_attempts",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
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
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN (" + ", ".join(repr(value) for value in _TURN_STATUSES) + ")",
            name="ck_chat_turns_status",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "request_id",
            name="uq_chat_turn_conversation_request",
        ),
        sa.UniqueConstraint("user_id", "request_id", name="uq_chat_turn_user_request"),
    )
    op.create_index(
        "ix_chat_turns_conversation_created_at",
        "chat_turns",
        ["conversation_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_chat_turns_status_updated_at",
        "chat_turns",
        ["status", "updated_at"],
        unique=False,
    )
    op.create_index(
        "ix_chat_turns_user_created_at",
        "chat_turns",
        ["user_id", "created_at"],
        unique=False,
    )

    for column in (
        sa.Column("turn_id", UUID(as_uuid=True), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("turn_status", sa.String(length=24), nullable=True),
        sa.Column("trace_id", sa.String(length=64), nullable=True),
        sa.Column("evidence_status", sa.String(length=32), nullable=True),
        sa.Column("retrieval_executed", sa.Boolean(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("delivery_status", sa.String(length=24), nullable=True),
        sa.Column("persistence_status", sa.String(length=24), nullable=True),
        sa.Column("search_snapshot", JSONB(), nullable=True),
    ):
        op.add_column("messages", column)
    op.create_index("ix_messages_turn_id", "messages", ["turn_id"], unique=False)
    op.create_index(
        "ix_messages_request_id", "messages", ["request_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_messages_request_id", table_name="messages")
    op.drop_index("ix_messages_turn_id", table_name="messages")
    for column_name in (
        "persistence_status",
        "delivery_status",
        "error_code",
        "retrieval_executed",
        "evidence_status",
        "trace_id",
        "turn_status",
        "request_id",
        "turn_id",
        "search_snapshot",
    ):
        op.drop_column("messages", column_name)

    op.drop_index("ix_chat_turns_user_created_at", table_name="chat_turns")
    op.drop_index("ix_chat_turns_status_updated_at", table_name="chat_turns")
    op.drop_index(
        "ix_chat_turns_conversation_created_at", table_name="chat_turns"
    )
    op.drop_table("chat_turns")
