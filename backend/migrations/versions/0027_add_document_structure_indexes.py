"""add bounded document-structure lookup indexes

Revision ID: 0027
Revises: 0026
Create Date: 2026-07-31

Evidence expansion starts from at most four already-authorized seed chunks.  These
indexes let PostgreSQL resolve adjacent chunks, same-section chunks, and split
table siblings without materializing every chunk in the selected documents.
"""

from typing import Sequence, Union

from alembic import op


revision: str = "0027"
down_revision: Union[str, None] = "0026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_doc_chunk_index
        ON document_chunks (doc_id, chunk_index)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_section_key_position
        ON document_chunks (
            doc_id,
            (metadata->>'section_key'),
            chunk_index
        )
        WHERE metadata ? 'section_key'
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_heading_position
        ON document_chunks (
            doc_id,
            (metadata->>'heading'),
            chunk_index
        )
        WHERE metadata ? 'heading'
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_chunks_table_part_position
        ON document_chunks (
            doc_id,
            (metadata->>'table_id'),
            (
                CASE
                    WHEN COALESCE(metadata->>'table_part_index', '')
                         ~ '^[0-9]{1,9}$'
                    THEN (metadata->>'table_part_index')::integer
                    ELSE NULL
                END
            )
        )
        WHERE metadata ? 'table_id'
          AND metadata ? 'table_part_index'
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS ix_document_chunks_table_part_position"
    )
    op.execute(
        "DROP INDEX IF EXISTS ix_document_chunks_heading_position"
    )
    op.execute(
        "DROP INDEX IF EXISTS ix_document_chunks_section_key_position"
    )
    op.execute(
        "DROP INDEX IF EXISTS ix_document_chunks_doc_chunk_index"
    )
