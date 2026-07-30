"""add RAG trace persistence integrity fields

Revision ID: 0026
Revises: 0025
Create Date: 2026-07-31
"""

from typing import Sequence, Union

from alembic import op


revision: str = "0026"
down_revision: Union[str, None] = "0025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 0025 已在部分开发/生产库执行，不能继续修改原迁移来补字段。曾经短暂
    # 修改过的 0025 又可能已经把字段建好，因此使用 PostgreSQL IF NOT EXISTS
    # 同时兼容“旧 0025”“短暂修改版 0025”和全新数据库三条升级路径。
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ADD COLUMN IF NOT EXISTS observed_event_count INTEGER"
    )
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ADD COLUMN IF NOT EXISTS storage_omitted_event_count INTEGER"
    )
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ADD COLUMN IF NOT EXISTS storage_truncated BOOLEAN"
    )
    op.execute(
        """
        UPDATE rag_trace_runs
        SET observed_event_count = COALESCE(observed_event_count, event_count),
            storage_omitted_event_count = COALESCE(storage_omitted_event_count, 0),
            storage_truncated = COALESCE(storage_truncated, FALSE)
        WHERE observed_event_count IS NULL
           OR storage_omitted_event_count IS NULL
           OR storage_truncated IS NULL
        """
    )
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ALTER COLUMN observed_event_count SET NOT NULL"
    )
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ALTER COLUMN storage_omitted_event_count SET NOT NULL"
    )
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ALTER COLUMN storage_truncated SET NOT NULL"
    )
    # 清理短暂修改版 0025 可能留下的数据库默认值；正常写入由应用层维护。
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ALTER COLUMN observed_event_count DROP DEFAULT"
    )
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ALTER COLUMN storage_omitted_event_count DROP DEFAULT"
    )
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "ALTER COLUMN storage_truncated DROP DEFAULT"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE rag_trace_runs DROP COLUMN IF EXISTS storage_truncated"
    )
    op.execute(
        "ALTER TABLE rag_trace_runs "
        "DROP COLUMN IF EXISTS storage_omitted_event_count"
    )
    op.execute(
        "ALTER TABLE rag_trace_runs DROP COLUMN IF EXISTS observed_event_count"
    )
