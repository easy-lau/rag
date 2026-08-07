"""add conversation_repair intent category

Revision ID: 0036
Revises: 0035
Create Date: 2026-08-07

Users who complain about the system's own clarification/selection behaviour
(for example ``为什么要我选择，你刚刚不是已经回答了吗``) must not be routed
back into the knowledge-base retrieval loop.  This migration adds the
``conversation_repair`` chat category used by the deterministic repair rule;
existing deployments keep the rule disabled until this category exists.
"""

import json
import uuid

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0036"
down_revision: Union[str, None] = "0035"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # JSONB 列没有 Alembic 离线 literal renderer，且 ``op.inline_literal`` 在
    # SQLAlchemy 2.x + asyncpg 在线执行时会把 _literal_bindparam 交给 JSON
    # 绑定处理器而失败。这里用 text INSERT + 显式 ``CAST(:examples AS JSONB)``：
    # 参数按 text 传输，由 PostgreSQL 赋值转换，离线 SQL（literal_binds）与
    # 在线执行都稳定。
    op.execute(
        sa.text(
            "INSERT INTO intent_categories "
            "(id, code, name, description, examples, action, enabled, priority) "
            "VALUES (:id, :code, :name, :description, CAST(:examples AS JSONB), "
            ":action, :enabled, :priority)"
        ).bindparams(
            sa.bindparam("id", value=uuid.uuid4(), type_=sa.Uuid()),
            sa.bindparam("code", value="conversation_repair", type_=sa.String()),
            sa.bindparam("name", value="对话修复", type_=sa.String()),
            sa.bindparam(
                "description",
                value=(
                    "用户质疑、纠正或抱怨系统刚才的回答、澄清或选择行为（例如\u201c为什么要我选择\u201d"
                    "\u201c你刚刚不是已经回答了吗\u201d）。直接说明系统行为并修复对话，不再触发知识库检索；"
                    "真正的业务追问仍按知识库问答处理。"
                ),
                type_=sa.Text(),
            ),
            sa.bindparam(
                "examples",
                value=json.dumps(
                    [
                        "为什么要我选择，你刚刚不是已经回答了吗",
                        "你刚才为什么一直问我要哪个文档",
                        "不是已经回答过了吗，怎么又问我",
                    ],
                    ensure_ascii=False,
                ),
                type_=sa.Text(),
            ),
            sa.bindparam("action", value="chat", type_=sa.String()),
            sa.bindparam("enabled", value=True, type_=sa.Boolean()),
            sa.bindparam("priority", value=90, type_=sa.Integer()),
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("DELETE FROM intent_categories WHERE code = :code"),
        {"code": "conversation_repair"},
    )
