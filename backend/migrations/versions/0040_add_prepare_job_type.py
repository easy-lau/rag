"""allow prepare job type for draft content extraction

Revision ID: 0040
Revises: 0039
Create Date: 2026-08-07

Uploads now enqueue a ``prepare`` job that extracts reviewable content
(``raw_content``) while the document stays a ``draft``.  The separate
``保存入库`` endpoint later enqueues the real ``file``/``text`` ingestion job.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0040"
down_revision: Union[str, None] = "0039"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "ALTER TABLE document_processing_jobs "
            "DROP CONSTRAINT ck_document_processing_jobs_type"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE document_processing_jobs "
            "ADD CONSTRAINT ck_document_processing_jobs_type "
            "CHECK (job_type IN ('file', 'text', 'image', 'prepare'))"
        )
    )


def downgrade() -> None:
    op.execute(
        sa.text(
            "ALTER TABLE document_processing_jobs "
            "DROP CONSTRAINT ck_document_processing_jobs_type"
        )
    )
    op.execute(
        sa.text(
            "ALTER TABLE document_processing_jobs "
            "ADD CONSTRAINT ck_document_processing_jobs_type "
            "CHECK (job_type IN ('file', 'text', 'image'))"
        )
    )
