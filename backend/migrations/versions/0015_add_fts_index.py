"""add GIN full-text index on document_chunks.content

The keyword half of hybrid search uses to_tsvector('simple', content) @@ plainto_tsquery(...).
Without a matching expression index, every keyword/hybrid query does a full table scan and
recomputes tsvector per row. This expression index lets Postgres use the index instead.

Revision ID: 0015
Revises: 0014
Create Date: 2026-06-15
"""
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 表达式必须与查询里的 to_tsvector('simple', content) 完全一致才能命中索引。
    # 'simple' 配置常量 → to_tsvector 为 IMMUTABLE，可用于表达式索引。
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_content_fts "
        "ON document_chunks USING gin (to_tsvector('simple', content))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_content_fts")
