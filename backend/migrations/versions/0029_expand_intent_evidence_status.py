"""expand intent route evidence status for clarification outcomes

Revision ID: 0029
Revises: 0028
Create Date: 2026-07-31

``needs_clarification`` is a persisted terminal evidence outcome and is longer
than the original VARCHAR(16) route-log column.  PostgreSQL otherwise rejects
the whole response-persistence transaction after generation has completed.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0029"
down_revision: Union[str, None] = "0028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "intent_route_logs",
        "evidence_status",
        existing_type=sa.String(length=16),
        type_=sa.String(length=32),
        existing_nullable=True,
    )


def downgrade() -> None:
    # A downgrade must make every existing value fit before narrowing the
    # column.  Historical clarification outcomes remain identifiable as an
    # error instead of making the schema downgrade fail midway.
    op.execute(
        """
        UPDATE intent_route_logs
        SET evidence_status = 'error'
        WHERE char_length(evidence_status) > 16
        """
    )
    op.alter_column(
        "intent_route_logs",
        "evidence_status",
        existing_type=sa.String(length=32),
        type_=sa.String(length=16),
        existing_nullable=True,
    )
