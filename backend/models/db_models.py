import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Integer, ForeignKey, DateTime, Boolean, UniqueConstraint, Index, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from database import Base


def now_utc():
    return datetime.now(timezone.utc)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    icon_color: Mapped[str] = mapped_column(String(20), default="#3B82F6")
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    documents: Mapped[list["Document"]] = relationship(back_populates="knowledge_base", cascade="all, delete-orphan", passive_deletes=True)
    creator: Mapped["User | None"] = relationship(foreign_keys=[created_by])


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str | None] = mapped_column(String(20))
    raw_content: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    image_url: Mapped[str | None] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="processing")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    tags: Mapped[list] = mapped_column(JSONB, default=list)  # 文档级标签，用于检索软加权（命中所选标签的文档排序分上浮）
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    knowledge_base: Mapped["KnowledgeBase"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan", passive_deletes=True)
    creator: Mapped["User | None"] = relationship(foreign_keys=[created_by])
    updater: Mapped["User | None"] = relationship(foreign_keys=[updated_by])


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(2560))
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(200))
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list | None] = mapped_column(JSONB)
    tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class SystemSetting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str | None] = mapped_column(Text)


class IntentRouterConfig(Base):
    """智能路由全局配置。

    该表有且仅有 id=1 的一行。使用独立表而不是 settings KV，避免类别和路由日志
    被序列化到字符串设置中，也便于后续增加版本和审计字段。
    """

    __tablename__ = "intent_router_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="rules_then_llm")
    intent_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.65)
    fallback_intent_code: Mapped[str] = mapped_column(String(64), nullable=False, default="other")
    allow_general_chat: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=now_utc, onupdate=now_utc
    )


class IntentCategory(Base):
    """可配置的意图分类及其固定后端动作。"""

    __tablename__ = "intent_categories"
    __table_args__ = (
        UniqueConstraint("code", name="uq_intent_categories_code"),
        Index("ix_intent_categories_enabled_priority", "enabled", "priority"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    examples: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=now_utc, onupdate=now_utc
    )


class IntentRouteLog(Base):
    """一次实际智能路由的决策快照。

    不存储问题正文，避免把用户聊天内容复制到另一张日志表；仅保留分类结论、
    耗时和可选人工反馈，供路由效果调优。
    """

    __tablename__ = "intent_route_logs"
    __table_args__ = (
        Index("ix_intent_route_logs_created_at", "created_at"),
        Index("ix_intent_route_logs_user_id_created_at", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    intent_code: Mapped[str] = mapped_column(String(64), nullable=False)
    intent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # ``action`` 保留分类所绑定的原始动作，便于判断模型究竟选中了什么；以下字段
    # 记录后端策略层给出的最终执行计划，不能再从 action 反向推导。
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    response_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    retrieval_policy: Mapped[str] = mapped_column(String(16), nullable=False)
    need_retrieval: Mapped[bool] = mapped_column(Boolean, nullable=False)
    decision_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    selected_kb_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 实际检索结果在流式管线运行后回填；旧日志和被用户提前中止的请求允许为空。
    retrieval_executed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    evidence_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    hit_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback: Mapped[str | None] = mapped_column(String(16), nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # 稳定代码只用于内建角色识别；自定义角色保持为空，不能依赖可修改的中文名称做安全判断。
    code: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)  # 内建角色禁止删除
    is_assignable: Mapped[bool] = mapped_column(Boolean, default=True)
    # none / selected / all；数据范围与功能 capability 分开保存。
    scope_mode: Mapped[str] = mapped_column(String(16), default="none", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    users: Mapped[list["User"]] = relationship(back_populates="role")
    permissions: Mapped[list["RolePermission"]] = relationship(
        back_populates="role", cascade="all, delete-orphan", passive_deletes=True
    )
    knowledge_bases: Mapped[list["RoleKnowledgeBase"]] = relationship(
        back_populates="role", cascade="all, delete-orphan", passive_deletes=True
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superadmin: Mapped[bool] = mapped_column(Boolean, default=False)  # 内建超管，绕过一切权限校验
    role_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("roles.id", ondelete="SET NULL"), nullable=True
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    role: Mapped["Role | None"] = relationship(back_populates="users")


class RolePermission(Base):
    __tablename__ = "role_permissions"
    __table_args__ = (UniqueConstraint("role_id", "permission_key", name="uq_role_permission"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"))
    permission_key: Mapped[str] = mapped_column(String(50))

    role: Mapped["Role"] = relationship(back_populates="permissions")


class RoleKnowledgeBase(Base):
    __tablename__ = "role_knowledge_bases"
    __table_args__ = (UniqueConstraint("role_id", "kb_id", name="uq_role_kb"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("roles.id", ondelete="CASCADE"))
    kb_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("knowledge_bases.id", ondelete="CASCADE"))

    role: Mapped["Role"] = relationship(back_populates="knowledge_bases")


class LoginLog(Base):
    __tablename__ = "login_logs"
    __table_args__ = (Index("ix_login_logs_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    username: Mapped[str] = mapped_column(String(64))
    success: Mapped[bool] = mapped_column(Boolean)
    # 登录失败原因（用户不存在 / 密码错误 / 账号已禁用）；成功时为空
    fail_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class OperationLog(Base):
    """操作审计日志：记录会改数据的关键动作（用户/角色/知识库/文档/设置/改密等）。

    target_* 与 username 均为冗余快照，确保被操作对象或操作人被删除后仍可追溯；
    detail 存结构化细节（改了哪些字段、批量数量等），绝不写入密码 / API Key 等敏感值。
    """

    __tablename__ = "operation_logs"
    __table_args__ = (Index("ix_operation_logs_created_at", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    username: Mapped[str] = mapped_column(String(64))          # 操作人（冗余，防删用户后失溯）
    action: Mapped[str] = mapped_column(String(64))            # 动作码，如 user.create / doc.delete
    target_type: Mapped[str | None] = mapped_column(String(32), nullable=True)   # 对象类型
    target_id: Mapped[str | None] = mapped_column(String(64), nullable=True)     # 对象 ID
    target_name: Mapped[str | None] = mapped_column(String(255), nullable=True)  # 对象名称（冗余）
    detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)            # 结构化细节
    ip: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
