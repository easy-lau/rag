"""add rbac: users / roles / role_permissions / role_knowledge_bases + conversations.user_id

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-14
"""
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


# 普通用户内建角色的默认权限：只能问答检索 + 看对话菜单
DEFAULT_USER_PERMISSIONS = ["menu:chat", "chat:use", "search:use", "kb:read"]


def upgrade() -> None:
    # ── 建表 ────────────────────────────────────────────────
    op.create_table(
        "roles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(50), nullable=False, unique=True),
        sa.Column("description", sa.Text),
        sa.Column("is_system", sa.Boolean, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(255)),
        sa.Column("display_name", sa.String(100)),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("true")),
        sa.Column("is_superadmin", sa.Boolean, server_default=sa.text("false")),
        sa.Column("role_id", UUID(as_uuid=True), sa.ForeignKey("roles.id", ondelete="SET NULL")),
        sa.Column("last_login_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    op.create_table(
        "role_permissions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("role_id", UUID(as_uuid=True), sa.ForeignKey("roles.id", ondelete="CASCADE")),
        sa.Column("permission_key", sa.String(50), nullable=False),
        sa.UniqueConstraint("role_id", "permission_key", name="uq_role_permission"),
    )

    op.create_table(
        "role_knowledge_bases",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("role_id", UUID(as_uuid=True), sa.ForeignKey("roles.id", ondelete="CASCADE")),
        sa.Column("kb_id", UUID(as_uuid=True), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE")),
        sa.UniqueConstraint("role_id", "kb_id", name="uq_role_kb"),
    )

    # conversations 增加 user_id（按用户隔离对话）
    op.add_column("conversations", sa.Column("user_id", UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_conversations_user_id", "conversations", "users", ["user_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])

    # ── Seed：内建角色 + admin 账号 ─────────────────────────
    # 延迟导入，复用应用内的权限目录与密码哈希，避免常量漂移
    from core.permissions import ALL_PERMISSIONS
    from core.security import hash_password
    from config import get_settings

    conn = op.get_bind()
    superadmin_role_id = str(uuid.uuid4())
    user_role_id = str(uuid.uuid4())
    admin_user_id = str(uuid.uuid4())

    conn.execute(
        sa.text(
            "INSERT INTO roles (id, name, description, is_system, created_at) "
            "VALUES (CAST(:id AS uuid), :name, :desc, true, NOW())"
        ),
        {"id": superadmin_role_id, "name": "超级管理员", "desc": "系统内建超级管理员，拥有全部权限"},
    )
    conn.execute(
        sa.text(
            "INSERT INTO roles (id, name, description, is_system, created_at) "
            "VALUES (CAST(:id AS uuid), :name, :desc, true, NOW())"
        ),
        {"id": user_role_id, "name": "普通用户", "desc": "系统内建普通用户，仅能问答检索"},
    )

    # 超级管理员：全部权限；普通用户：默认检索权限
    for key in ALL_PERMISSIONS:
        conn.execute(
            sa.text(
                "INSERT INTO role_permissions (id, role_id, permission_key) "
                "VALUES (CAST(:id AS uuid), CAST(:rid AS uuid), :k)"
            ),
            {"id": str(uuid.uuid4()), "rid": superadmin_role_id, "k": key},
        )
    for key in DEFAULT_USER_PERMISSIONS:
        conn.execute(
            sa.text(
                "INSERT INTO role_permissions (id, role_id, permission_key) "
                "VALUES (CAST(:id AS uuid), CAST(:rid AS uuid), :k)"
            ),
            {"id": str(uuid.uuid4()), "rid": user_role_id, "k": key},
        )

    # admin 账号：is_superadmin=true，密码取 ADMIN_INIT_PASSWORD
    conn.execute(
        sa.text(
            "INSERT INTO users (id, username, password_hash, display_name, is_active, is_superadmin, role_id, created_at) "
            "VALUES (CAST(:id AS uuid), :u, :ph, :dn, true, true, CAST(:rid AS uuid), NOW())"
        ),
        {
            "id": admin_user_id,
            "u": "admin",
            "ph": hash_password(get_settings().admin_init_password),
            "dn": "超级管理员",
            "rid": superadmin_role_id,
        },
    )

    # 普通用户角色默认授权现有全部知识库（保持当前“人人可搜”的连续性，管理员后续可收紧）
    kb_rows = conn.execute(sa.text("SELECT id FROM knowledge_bases")).fetchall()
    for (kb_id,) in kb_rows:
        conn.execute(
            sa.text(
                "INSERT INTO role_knowledge_bases (id, role_id, kb_id) "
                "VALUES (CAST(:id AS uuid), CAST(:rid AS uuid), CAST(:kb AS uuid))"
            ),
            {"id": str(uuid.uuid4()), "rid": user_role_id, "kb": str(kb_id)},
        )


def downgrade() -> None:
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_constraint("fk_conversations_user_id", "conversations", type_="foreignkey")
    op.drop_column("conversations", "user_id")
    op.drop_table("role_knowledge_bases")
    op.drop_table("role_permissions")
    op.drop_table("users")
    op.drop_table("roles")
