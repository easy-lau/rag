"""add staged source path to documents

Revision ID: 0039
Revises: 0038
Create Date: 2026-08-07

Uploads become explicit two-phase operations: the upload endpoint only stages
the source file on a ``draft`` document row (``staged_path``), and the separate
``保存入库`` endpoint enqueues the ingestion job.  Drafts are never searchable
(chat retrieval already filters ``status == 'ready'``); deleting a draft must
also remove its staged source file.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0039"
down_revision: Union[str, None] = "0038"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("staged_path", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "staged_path")
