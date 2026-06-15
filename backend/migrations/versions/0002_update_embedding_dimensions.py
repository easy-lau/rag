"""update embedding dimensions from 1536 to 2560

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-12
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(2560)")
    # ivfflat/hnsw on vector type max 2000 dims; cast to halfvec for indexing (pgvector >= 0.7)
    op.execute("""
        CREATE INDEX idx_chunks_embedding ON document_chunks
        USING hnsw ((embedding::halfvec(2560)) halfvec_cosine_ops)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(1536)")
    op.execute("""
        CREATE INDEX idx_chunks_embedding ON document_chunks
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    """)
