"""add authorization-neutral active conversation task state

Revision ID: 0034
Revises: 0033
Create Date: 2026-08-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "0034"
down_revision: Union[str, None] = "0033"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("active_task_state", JSONB(), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "active_task_revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("conversations", "active_task_revision")
    op.drop_column("conversations", "active_task_state")
