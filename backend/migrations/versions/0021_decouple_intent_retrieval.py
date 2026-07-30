"""decouple intent classification from retrieval execution

Revision ID: 0021
Revises: 0020
Create Date: 2026-07-30

The original route log stored only the category action, so a probabilistic
classification also became the final retrieval decision.  Preserve that raw
action while adding the backend policy decision and the eventual evidence
status as separate fields.
"""

import json

from alembic import op
import sqlalchemy as sa


revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


_OLD_SYSTEM_HELP_DESCRIPTION = (
    "询问本系统如何上传文档、创建知识库、进行检索或管理账号等使用方法。"
)
_NEW_SYSTEM_HELP_DESCRIPTION = (
    "仅限询问当前 RAG 问答平台自身如何上传文档、创建知识库、检索或进入管理后台。"
    "外部产品、业务系统的配置和使用问题不属于此类。"
)
_OLD_WRITING_DESCRIPTION = (
    "改写、润色、翻译、起草、总结用户提供内容等写作辅助请求，通常不需要检索知识库。"
)
_NEW_WRITING_DESCRIPTION = (
    "以改写、润色、翻译、起草或总结为主要目标。用户直接附带原文时无需检索；"
    "要求依据知识库资料写作时仍需先检索。"
)
_OLD_WRITING_EXAMPLES = ["帮我润色这段通知", "把下面内容翻译成英文", "起草一封会议邀请邮件"]
_NEW_WRITING_EXAMPLES = ["帮我润色这段通知", "起草一封会议邀请邮件", "根据员工手册总结请假规则"]
_OLD_SYSTEM_HELP_EXAMPLES = ["怎样上传文档？", "怎么创建知识库？", "系统如何检索？"]
_NEW_SYSTEM_HELP_EXAMPLES = [
    "当前 RAG 平台怎样上传文档？",
    "怎么创建知识库？",
    "在哪里查看检索结果？",
]


def upgrade() -> None:
    op.add_column("intent_route_logs", sa.Column("response_mode", sa.String(length=32)))
    op.add_column("intent_route_logs", sa.Column("retrieval_policy", sa.String(length=16)))
    op.add_column("intent_route_logs", sa.Column("need_retrieval", sa.Boolean()))
    op.add_column("intent_route_logs", sa.Column("decision_reason", sa.String(length=64)))
    op.add_column("intent_route_logs", sa.Column("retrieval_executed", sa.Boolean()))
    op.add_column("intent_route_logs", sa.Column("evidence_status", sa.String(length=16)))
    op.add_column("intent_route_logs", sa.Column("hit_count", sa.Integer()))

    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            UPDATE intent_route_logs
            SET
                response_mode = CASE action
                    WHEN 'chat' THEN 'general_chat'
                    WHEN 'writing' THEN 'writing'
                    WHEN 'system_help' THEN 'platform_help'
                    ELSE 'grounded_qa'
                END,
                retrieval_policy = CASE
                    WHEN action = 'retrieve' THEN 'required'
                    ELSE 'skip'
                END,
                need_retrieval = (action = 'retrieve'),
                decision_reason = 'legacy_action_mapping'
            """
        )
    )

    op.alter_column("intent_route_logs", "response_mode", nullable=False)
    op.alter_column("intent_route_logs", "retrieval_policy", nullable=False)
    op.alter_column("intent_route_logs", "need_retrieval", nullable=False)
    op.alter_column("intent_route_logs", "decision_reason", nullable=False)

    # Only update untouched built-in seed fields.  Administrators may edit the
    # category description/examples independently, and those customizations
    # must survive an application upgrade.
    conn.execute(
        sa.text(
            """
            UPDATE intent_categories
            SET description = :new_description
            WHERE code = 'system_help'
              AND description = :old_description
            """
        ),
        {
            "old_description": _OLD_SYSTEM_HELP_DESCRIPTION,
            "new_description": _NEW_SYSTEM_HELP_DESCRIPTION,
        },
    )
    conn.execute(
        sa.text(
            """
            UPDATE intent_categories
            SET examples = CAST(:new_examples AS jsonb)
            WHERE code = 'writing'
              AND examples = CAST(:old_examples AS jsonb)
            """
        ),
        {
            "old_examples": json.dumps(_OLD_WRITING_EXAMPLES, ensure_ascii=False),
            "new_examples": json.dumps(_NEW_WRITING_EXAMPLES, ensure_ascii=False),
        },
    )
    conn.execute(
        sa.text(
            """
            UPDATE intent_categories
            SET description = :new_description
            WHERE code = 'writing'
              AND description = :old_description
            """
        ),
        {
            "old_description": _OLD_WRITING_DESCRIPTION,
            "new_description": _NEW_WRITING_DESCRIPTION,
        },
    )
    conn.execute(
        sa.text(
            """
            UPDATE intent_categories
            SET examples = CAST(:new_examples AS jsonb)
            WHERE code = 'system_help'
              AND examples = CAST(:old_examples AS jsonb)
            """
        ),
        {
            "old_examples": json.dumps(_OLD_SYSTEM_HELP_EXAMPLES, ensure_ascii=False),
            "new_examples": json.dumps(_NEW_SYSTEM_HELP_EXAMPLES, ensure_ascii=False),
        },
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text(
            """
            UPDATE intent_categories
            SET examples = CAST(:old_examples AS jsonb)
            WHERE code = 'writing'
              AND examples = CAST(:new_examples AS jsonb)
            """
        ),
        {
            "old_examples": json.dumps(_OLD_WRITING_EXAMPLES, ensure_ascii=False),
            "new_examples": json.dumps(_NEW_WRITING_EXAMPLES, ensure_ascii=False),
        },
    )
    conn.execute(
        sa.text(
            """
            UPDATE intent_categories
            SET description = :old_description
            WHERE code = 'writing'
              AND description = :new_description
            """
        ),
        {
            "old_description": _OLD_WRITING_DESCRIPTION,
            "new_description": _NEW_WRITING_DESCRIPTION,
        },
    )
    conn.execute(
        sa.text(
            """
            UPDATE intent_categories
            SET description = :old_description
            WHERE code = 'system_help'
              AND description = :new_description
            """
        ),
        {
            "old_description": _OLD_SYSTEM_HELP_DESCRIPTION,
            "new_description": _NEW_SYSTEM_HELP_DESCRIPTION,
        },
    )
    conn.execute(
        sa.text(
            """
            UPDATE intent_categories
            SET examples = CAST(:old_examples AS jsonb)
            WHERE code = 'system_help'
              AND examples = CAST(:new_examples AS jsonb)
            """
        ),
        {
            "old_examples": json.dumps(_OLD_SYSTEM_HELP_EXAMPLES, ensure_ascii=False),
            "new_examples": json.dumps(_NEW_SYSTEM_HELP_EXAMPLES, ensure_ascii=False),
        },
    )

    op.drop_column("intent_route_logs", "hit_count")
    op.drop_column("intent_route_logs", "evidence_status")
    op.drop_column("intent_route_logs", "retrieval_executed")
    op.drop_column("intent_route_logs", "decision_reason")
    op.drop_column("intent_route_logs", "need_retrieval")
    op.drop_column("intent_route_logs", "retrieval_policy")
    op.drop_column("intent_route_logs", "response_mode")
