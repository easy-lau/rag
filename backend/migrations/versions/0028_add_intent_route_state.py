"""add persistent intent-route state and compact route diagnostics

Revision ID: 0028
Revises: 0027
Create Date: 2026-07-31

Pending clarification state belongs to its conversation and is consumed with an
optimistic revision.  Route logs retain only a compact, content-free summary and
an unbound trace identifier because RAG traces are written and expired by an
independent asynchronous store.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "0028"
down_revision: Union[str, None] = "0027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("pending_route_state", JSONB(), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "route_state_revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "intent_route_logs",
        sa.Column("trace_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "intent_route_logs",
        sa.Column("route_summary", JSONB(), nullable=True),
    )
    op.create_index(
        "ix_intent_route_logs_trace_id",
        "intent_route_logs",
        ["trace_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_intent_route_logs_trace_id",
        table_name="intent_route_logs",
    )
    op.drop_column("intent_route_logs", "route_summary")
    op.drop_column("intent_route_logs", "trace_id")
    op.drop_column("conversations", "route_state_revision")
    op.drop_column("conversations", "pending_route_state")
