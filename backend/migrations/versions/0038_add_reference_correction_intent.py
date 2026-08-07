"""add reference_correction intent category

Revision ID: 0038
Revises: 0037
Create Date: 2026-08-07

Users who correct a previously displayed result list (for example ``第四个不是
《钉钉》吗`` / ``你刚才说错了，应该是第五个``) must be routed deterministically
instead of re-entering intent-model classification.  The execution layer then
resolves the ordinal against the persisted result list and reads the correct
document; the category makes that routing visible and auditable.
"""

import json
import uuid

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0038"
down_revision: Union[str, None] = "0037"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 同 0036：JSONB 列在 SQLAlchemy 2.x + asyncpg 在线执行时不能用
    # ``op.inline_literal``；改用 text INSERT + ``CAST(:examples AS JSONB)``。
    op.execute(
        sa.text(
            "INSERT INTO intent_categories "
            "(id, code, name, description, examples, action, enabled, priority) "
            "VALUES (:id, :code, :name, :description, CAST(:examples AS JSONB), "
            ":action, :enabled, :priority)"
        ).bindparams(
            sa.bindparam("id", value=uuid.uuid4(), type_=sa.Uuid()),
            sa.bindparam("code", value="reference_correction", type_=sa.String()),
            sa.bindparam("name", value="结果引用纠正", type_=sa.String()),
            sa.bindparam(
                "description",
                value=(
                    "用户纠正或质疑前面列出的结果序号（例如\u201c第四个不是《钉钉》吗\u201d"
                    "\u201c你刚才说错了，应该是第五个\u201d）。按已展示的结果列表重新解析序号，"
                    "直接读取正确文档，不再重新检索或要求用户选择。"
                ),
                type_=sa.Text(),
            ),
            sa.bindparam(
                "examples",
                value=json.dumps(
                    [
                        "第四个不是《钉钉》吗",
                        "你刚才说错了，应该是第五个",
                        "第五个才对吧",
                        "你返回错了吧，我想看第四个",
                    ],
                    ensure_ascii=False,
                ),
                type_=sa.Text(),
            ),
            sa.bindparam("action", value="retrieve", type_=sa.String()),
            sa.bindparam("enabled", value=True, type_=sa.Boolean()),
            sa.bindparam("priority", value=95, type_=sa.Integer()),
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("DELETE FROM intent_categories WHERE code = :code"),
        {"code": "reference_correction"},
    )
