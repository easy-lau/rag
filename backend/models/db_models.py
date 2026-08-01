import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
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
    tags: Mapped[list] = mapped_column(JSONB, default=list)  # 文档级标签，用于管理、展示与产品/版本约束识别
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
    # 每个会话最多保留一个版本化待澄清状态。状态只作为后续重新路由的输入，
    # 不能作为恢复执行授权的依据；revision 供调用层做乐观并发控制。
    pending_route_state: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    route_state_revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan")
    turns: Mapped[list["ChatTurn"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True
    )


class ChatTurn(Base):
    """Durable state for one client chat request.

    ``Message`` is the user-facing transcript, while this row is the
    idempotency and delivery ledger.  Keeping the generated answer on the turn
    before inserting the assistant message lets a later retry finish a
    ``persist_failed`` turn without running retrieval/LLM again.
    """

    __tablename__ = "chat_turns"
    __table_args__ = (
        CheckConstraint(
            "status IN ('accepted', 'generating', 'generated', 'completed', "
            "'persist_failed', 'failed', 'cancelled')",
            name="ck_chat_turns_status",
        ),
        UniqueConstraint(
            "conversation_id", "request_id", name="uq_chat_turn_conversation_request"
        ),
        UniqueConstraint("user_id", "request_id", name="uq_chat_turn_user_request"),
        Index("ix_chat_turns_conversation_created_at", "conversation_id", "created_at"),
        Index("ix_chat_turns_status_updated_at", "status", "updated_at"),
        Index("ix_chat_turns_user_created_at", "user_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # Client supplied idempotency key.  It is intentionally opaque; the
    # conversation scope prevents cross-user/cross-conversation collisions.
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    question_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Full canonical request envelope.  The digest is the comparison fast path;
    # the bounded JSON is retained so old clients that omit clarification
    # identity fields can still retry the exact original logical request.
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    request_context: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # Route identity after this request's own pre-generation cleanup.  Stale
    # recovery accepts input context or this checkpoint, but no unrelated
    # conversation revision.
    resume_context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="accepted")
    # accepted/generating are protected by a renewable execution lease.  A
    # process crash therefore becomes reclaimable instead of a permanent 202.
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    execution_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    evidence_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    retrieval_executed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # The complete answer is staged at ``generated`` and retained through a
    # ``persist_failed`` transition.  This field is not exposed as a separate
    # transcript until the assistant Message commit succeeds.
    answer_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_sources: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    # Bounded final search-panel snapshot.  It contains identities/roles and
    # counters only; current document rows are re-authorized when history is
    # read.
    search_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    assistant_message_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    persistence_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=now_utc, onupdate=now_utc
    )
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="turns")
    user: Mapped["User | None"] = relationship(foreign_keys=[user_id])


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list | None] = mapped_column(JSONB)
    tokens: Mapped[int | None] = mapped_column(Integer)
    # Turn metadata is duplicated on transcript rows so history can be served
    # in one query and older messages remain backward compatible (all fields
    # are nullable).  ``turn_id`` is deliberately unbound: cleanup/order of
    # transcript rows must not make an already durable turn undeletable.
    turn_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    turn_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    retrieval_executed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    delivery_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    persistence_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    # 从请求受理到回答持久化完成的端到端耗时（毫秒）。历史会话重载后仍可展示。
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    search_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class RagTraceRun(Base):
    """一次问答或检索测试的调用链摘要。

    列表只读取本表的短摘要；体积较大的逐阶段 JSON 存在 ``rag_trace_events``，
    仅在管理员打开详情时加载。生产环境默认不保存问题、回答和候选正文。
    """

    __tablename__ = "rag_trace_runs"
    __table_args__ = (
        Index("ix_rag_trace_runs_started_at", "started_at"),
        Index("ix_rag_trace_runs_status_started_at", "status", "started_at"),
        Index("ix_rag_trace_runs_user_started_at", "user_id", "started_at"),
        Index("ix_rag_trace_runs_conversation_started_at", "conversation_id", "started_at"),
    )

    trace_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_kind: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    # Trace 由独立异步队列写入，可能早于新会话事务提交，也需要在用户/会话
    # 删除后保留 30 天用于排障。因此这里只保存不可反查业务正文的 UUID 快照，
    # 不建立跨事务外键；详情事件仍通过 trace_id 外键随摘要级联清理。
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    current_stage: Mapped[str | None] = mapped_column(String(100), nullable=True)
    event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # observed counts records accepted by the persistence worker; queue drops
    # happen earlier and remain separately documented as an unknowable gap.
    observed_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    storage_omitted_event_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    storage_truncated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    content_included: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    input_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    selected_kb_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 审计命中数只统计 direct 回答证据；展示候选和 related 上下文不计入。
    hit_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=now_utc, onupdate=now_utc
    )

    events: Mapped[list["RagTraceEvent"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class RagTraceEvent(Base):
    """调用链中的一个阶段事件；payload 已经过正文开关与异常脱敏处理。"""

    __tablename__ = "rag_trace_events"
    __table_args__ = (
        UniqueConstraint("trace_id", "sequence", name="uq_rag_trace_event_sequence"),
        Index("ix_rag_trace_events_event_created_at", "event", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("rag_trace_runs.trace_id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    run: Mapped["RagTraceRun"] = relationship(back_populates="events")


class SystemSetting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str | None] = mapped_column(Text)


class IntentRouterConfig(Base):
    """智能路由全局配置。

    该表有且仅有 id=1 的一行，仅保存路由策略；意图模型由模型管理中的 settings
    统一维护，类别与路由日志仍使用各自的结构化表。
    """

    __tablename__ = "intent_router_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False, default="rules_then_llm")
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
        Index("ix_intent_route_logs_trace_id", "trace_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    # Trace 由异步存储独立维护并按保留期清理，因此这里只保存关联值，不建立外键。
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    intent_code: Mapped[str] = mapped_column(String(64), nullable=False)
    intent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    # ``action`` 保留分类所绑定的原始动作，便于判断模型究竟选中了什么；以下字段
    # 记录后端策略层给出的最终执行计划，不能再从 action 反向推导。
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    response_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    retrieval_policy: Mapped[str] = mapped_column(String(16), nullable=False)
    need_retrieval: Mapped[bool] = mapped_column(Boolean, nullable=False)
    decision_reason: Mapped[str] = mapped_column(String(64), nullable=False)
    # 只允许保存协议版本、枚举、数量等无正文摘要；完整语义合同仍进入受内容门禁
    # 保护的 RAG Trace，避免 intent:read 日志复制用户或知识库正文。
    route_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    selected_kb_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # 实际检索结果在流式管线运行后回填；hit_count 仅统计 direct 回答证据，
    # 旧日志和被用户提前中止的请求允许为空。
    retrieval_executed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # ``needs_clarification`` is 19 characters; keep this aligned with the
    # RAG trace status column so clarification responses can be committed in
    # the same transaction as the assistant message and pending route state.
    evidence_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
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
    __table_args__ = (
        Index("ix_login_logs_created_at", "created_at"),
        Index("ix_login_logs_success_created_at", "success", "created_at"),
        Index("ix_login_logs_username_created_at", "username", "created_at"),
        Index("ix_login_logs_ip_created_at", "ip", "created_at"),
        Index("ix_login_logs_last_attempt_at", "last_attempt_at"),
        Index(
            "ix_login_logs_failure_aggregation",
            "success",
            "username",
            "ip",
            "fail_reason",
            "last_attempt_at",
        ),
    )

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
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class LoginThrottle(Base):
    """登录失败来源限流状态。

    ``pair`` 仅限制某个来源对某个用户名的尝试，``ip`` 限制来源扫描多个账号，
    ``account`` 只用于发现集中攻击并告警，绝不据此锁定用户账号。
    bucket_key 使用 SHA-256 摘要，避免在状态表中重复保存用户名和 IP 明文；
    可读信息仍由 login_logs 审计表保存。
    """

    __tablename__ = "login_throttles"
    __table_args__ = (
        CheckConstraint(
            "scope IN ('pair', 'ip', 'account')",
            name="ck_login_throttles_scope",
        ),
        Index("ix_login_throttles_last_failed_at", "last_failed_at"),
        Index("ix_login_throttles_blocked_until", "blocked_until"),
    )

    scope: Mapped[str] = mapped_column(String(16), primary_key=True)
    bucket_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    window_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now_utc
    )
    blocked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_failed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now_utc
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now_utc
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=now_utc, onupdate=now_utc
    )


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
