"""add configurable intent routing

Revision ID: 0018
Revises: 0017
Create Date: 2026-07-29
"""

import json
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "intent_router_configs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "mode", sa.String(length=32), nullable=False, server_default=sa.text("'rules_then_llm'")
        ),
        sa.Column("intent_model", sa.String(length=255), nullable=True),
        sa.Column(
            "confidence_threshold", sa.Float(), nullable=False, server_default=sa.text("0.65")
        ),
        sa.Column(
            "fallback_intent_code", sa.String(length=64), nullable=False, server_default=sa.text("'other'")
        ),
        sa.Column("allow_general_chat", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_table(
        "intent_categories",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("examples", JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("code", name="uq_intent_categories_code"),
    )
    op.create_index(
        "ix_intent_categories_enabled_priority",
        "intent_categories",
        ["enabled", "priority"],
    )
    op.create_table(
        "intent_route_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column(
            "conversation_id", UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="SET NULL")
        ),
        sa.Column("intent_code", sa.String(length=64), nullable=False),
        sa.Column("intent_name", sa.String(length=100), nullable=False),
        sa.Column("action", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("selected_kb_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("feedback", sa.String(length=16), nullable=True),
        sa.Column("feedback_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_intent_route_logs_created_at", "intent_route_logs", ["created_at"])
    op.create_index(
        "ix_intent_route_logs_user_id_created_at",
        "intent_route_logs",
        ["user_id", "created_at"],
    )

    # 预置单例配置和首版五个分类；应用层仍会补齐空表，防御人工清表/测试库。
    op.bulk_insert(
        sa.table(
            "intent_router_configs",
            sa.column("id", sa.Integer()),
            sa.column("enabled", sa.Boolean()),
            sa.column("mode", sa.String()),
            sa.column("confidence_threshold", sa.Float()),
            sa.column("fallback_intent_code", sa.String()),
            sa.column("allow_general_chat", sa.Boolean()),
        ),
        [{
            "id": 1,
            "enabled": True,
            "mode": "rules_then_llm",
            "confidence_threshold": 0.65,
            "fallback_intent_code": "other",
            "allow_general_chat": True,
        }],
    )
    categories = sa.table(
        "intent_categories",
        sa.column("id", UUID(as_uuid=True)),
        sa.column("code", sa.String()),
        sa.column("name", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("examples", JSONB()),
        sa.column("action", sa.String()),
        sa.column("enabled", sa.Boolean()),
        sa.column("priority", sa.Integer()),
    )
    op.bulk_insert(categories, [
        {
            "id": uuid.uuid4(),
            "code": "knowledge_qa",
            "name": "知识库问答",
            "description": "涉及企业制度、流程、业务资料、数据或已上传文档，需要先检索有权限的知识库再作答。",
            # JSONB 没有 Alembic 离线 literal renderer；用 inline text literal 让 Postgres
            # 在 INSERT 赋值时转换为 jsonb，同时保证 `alembic ... --sql` 可验证本迁移。
            "examples": op.inline_literal(
                json.dumps(["公司的报销流程是什么？", "请查询员工请假制度", "这份采购规范有什么要求？"], ensure_ascii=False),
                type_=sa.Text(),
            ),
            "action": "retrieve",
            "enabled": True,
            "priority": 100,
        },
        {
            "id": uuid.uuid4(),
            "code": "general_chat",
            "name": "通用交流",
            "description": "问候、感谢、一般常识或与企业资料无关的交流，不需要检索知识库。",
            "examples": op.inline_literal(
                json.dumps(["你好", "谢谢你的帮助", "今天上海天气怎么样？"], ensure_ascii=False),
                type_=sa.Text(),
            ),
            "action": "chat",
            "enabled": True,
            "priority": 80,
        },
        {
            "id": uuid.uuid4(),
            "code": "writing",
            "name": "写作润色",
            "description": "改写、润色、翻译、起草、总结用户提供内容等写作辅助请求，通常不需要检索知识库。",
            "examples": op.inline_literal(
                json.dumps(["帮我润色这段通知", "把下面内容翻译成英文", "起草一封会议邀请邮件"], ensure_ascii=False),
                type_=sa.Text(),
            ),
            "action": "writing",
            "enabled": True,
            "priority": 70,
        },
        {
            "id": uuid.uuid4(),
            "code": "system_help",
            "name": "系统使用帮助",
            "description": "询问本系统如何上传文档、创建知识库、进行检索或管理账号等使用方法。",
            "examples": op.inline_literal(
                json.dumps(["怎样上传文档？", "怎么创建知识库？", "系统如何检索？"], ensure_ascii=False),
                type_=sa.Text(),
            ),
            "action": "system_help",
            "enabled": True,
            "priority": 60,
        },
        {
            "id": uuid.uuid4(),
            "code": "other",
            "name": "未识别问题",
            "description": "无法可靠归类时的保守兜底。默认执行知识库检索，避免遗漏业务问题。",
            "examples": op.inline_literal(json.dumps([]), type_=sa.Text()),
            "action": "retrieve",
            "enabled": True,
            "priority": 0,
        },
    ], multiinsert=False)


def downgrade() -> None:
    op.drop_index("ix_intent_route_logs_user_id_created_at", table_name="intent_route_logs")
    op.drop_index("ix_intent_route_logs_created_at", table_name="intent_route_logs")
    op.drop_table("intent_route_logs")
    op.drop_index("ix_intent_categories_enabled_priority", table_name="intent_categories")
    op.drop_table("intent_categories")
    op.drop_table("intent_router_configs")
