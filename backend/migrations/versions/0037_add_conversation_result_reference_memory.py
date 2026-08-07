"""add result reference memory to conversations

Revision ID: 0037
Revises: 0036
Create Date: 2026-08-07
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


revision: str = "0037"
down_revision: Union[str, None] = "0036"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("result_reference_memory", JSONB(), nullable=True),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "result_reference_revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("conversations", "result_reference_revision")
    op.drop_column("conversations", "result_reference_memory")
