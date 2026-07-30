import uuid
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


# ── Knowledge Base ──────────────────────────────────────────────
class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., max_length=100)
    description: str | None = None
    icon_color: str = "#3B82F6"


class KnowledgeBaseOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    icon_color: str
    doc_count: int = 0   # 非持久化字段，由接口按真实文档行数填充
    created_by: uuid.UUID | None = None
    created_by_name: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Document ─────────────────────────────────────────────────────
class DocumentOut(BaseModel):
    id: uuid.UUID
    kb_id: uuid.UUID
    filename: str
    file_type: str | None
    raw_content: str | None = None
    source_url: str | None = None
    image_url: str | None = None
    chunk_count: int
    status: str
    is_active: bool = True
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None
    created_by_name: str | None = None
    updated_by_name: str | None = None

    model_config = {"from_attributes": True}


# ── Chat ─────────────────────────────────────────────────────────
class SearchConfig(BaseModel):
    method: Literal["hybrid", "vector", "keyword"] = "hybrid"
    rerank: bool = True
    top_k: int = Field(5, ge=1, le=20)
    tags: list[str] = Field(default_factory=list, max_length=100)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=12000)
    conversation_id: uuid.UUID | None = None
    knowledge_base_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    search_config: SearchConfig = Field(default_factory=SearchConfig)


class MessageOut(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    sources: list | None
    tokens: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


# ── Search ───────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=12000)
    knowledge_base_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    method: Literal["hybrid", "vector", "keyword"] = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = True
    tags: list[str] = Field(default_factory=list, max_length=100)  # 标签软加权


class SearchResultItem(BaseModel):
    id: uuid.UUID
    content: str
    filename: str
    file_type: str | None
    score: float
    chunk_index: int
    metadata: dict | None
    tags: list[str] = Field(default_factory=list)
    kb_id: uuid.UUID | None = None
    doc_id: uuid.UUID | None = None
    retrieval_score: float | None = None
    vector_score: float | None = None
    vector_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    trigram_score: float | None = None
    trigram_rank: int | None = None
    active_channels: list[str] = Field(default_factory=list)
    rerank_status: str | None = None
    topic_relevance: float | None = None
    answer_support: float | None = None
    constraint_status: str | None = None
    evidence_role: str | None = None
    rerank_reason: str | None = None
    constraint_reason: str | None = None
    ranking_factors: dict | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int
    search_meta: dict[str, Any]


# ── Auth ─────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class MeOut(BaseModel):
    id: uuid.UUID
    username: str
    display_name: str | None = None
    is_superadmin: bool
    role_name: str | None = None
    kb_scope: Literal["none", "selected", "all"] = "none"
    permissions: list[str] = []
    menus: list[str] = []


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: MeOut


# ── User ─────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username: str
    password: str
    display_name: str | None = None
    role_id: uuid.UUID | None = None
    is_active: bool = True


class UserOut(BaseModel):
    id: uuid.UUID
    username: str
    display_name: str | None = None
    is_active: bool
    is_superadmin: bool
    role_id: uuid.UUID | None = None
    role_name: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdate(BaseModel):
    display_name: str | None = None
    role_id: uuid.UUID | None = None
    is_active: bool | None = None
    password: str | None = None


# ── Role ─────────────────────────────────────────────────────────
class RoleCreate(BaseModel):
    code: str | None = Field(None, max_length=64)
    name: str = Field(..., min_length=1, max_length=50)
    description: str | None = None
    is_assignable: bool = True
    permissions: list[str] = Field(default_factory=list)
    # None 仅用于兼容旧客户端：后端会根据 kb_ids 推断 selected / none。
    scope_mode: Literal["none", "selected", "all"] | None = None
    kb_ids: list[uuid.UUID] = Field(default_factory=list)


class RoleOut(BaseModel):
    id: uuid.UUID
    code: str | None = None
    name: str
    description: str | None = None
    is_system: bool
    is_assignable: bool = True
    permissions: list[str] = Field(default_factory=list)
    scope_mode: Literal["none", "selected", "all"] = "none"
    kb_ids: list[uuid.UUID] = Field(default_factory=list)
    created_at: datetime

    model_config = {"from_attributes": True}


class RoleUpdate(BaseModel):
    code: str | None = Field(None, max_length=64)
    name: str | None = Field(None, min_length=1, max_length=50)
    description: str | None = None
    # Omissible for PATCH-like PUT compatibility, but an explicit JSON null is
    # invalid because assignment state has only two meanings.
    is_assignable: bool = None
    permissions: list[str] | None = None
    scope_mode: Literal["none", "selected", "all"] | None = None
    kb_ids: list[uuid.UUID] | None = None


# ── Login Log ────────────────────────────────────────────────────
class LoginLogOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None = None
    username: str
    success: bool
    fail_reason: str | None = None
    ip: str | None = None
    user_agent: str | None = None
    attempt_count: int = 1
    last_attempt_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class LoginLogPage(BaseModel):
    items: list[LoginLogOut]
    total: int


class OperationLogOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None = None
    username: str
    action: str
    target_type: str | None = None
    target_id: str | None = None
    target_name: str | None = None
    detail: dict | None = None
    ip: str | None = None
    user_agent: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class OperationLogPage(BaseModel):
    items: list[OperationLogOut]
    total: int


# ── RAG Trace ───────────────────────────────────────────────────
class RagTraceRunOut(BaseModel):
    trace_id: str
    request_kind: str
    user_id: uuid.UUID | None = None
    username: str | None = None
    conversation_id: uuid.UUID | None = None
    status: str
    current_stage: str | None = None
    event_count: int
    observed_event_count: int
    storage_omitted_event_count: int
    storage_truncated: bool
    content_included: bool
    content_accessible: bool = True
    input_preview: str | None = None
    output_preview: str | None = None
    evidence_status: str | None = None
    selected_kb_count: int | None = None
    hit_count: int | None = None
    duration_ms: int | None = None
    started_at: datetime
    completed_at: datetime | None = None
    updated_at: datetime


class RagTraceRunPage(BaseModel):
    items: list[RagTraceRunOut]
    total: int


class RagTraceEventOut(BaseModel):
    id: uuid.UUID
    sequence: int
    event: str
    payload: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class RagTraceDetailOut(RagTraceRunOut):
    events: list[RagTraceEventOut] = Field(default_factory=list)


# ── Intent Routing ──────────────────────────────────────────────
IntentRouterMode = Literal["rules_then_llm", "llm_only", "off"]
IntentAction = Literal["retrieve", "chat", "writing", "system_help"]
IntentResponseMode = Literal["grounded_qa", "general_chat", "writing", "platform_help"]
IntentRetrievalPolicy = Literal["required", "optional", "skip"]
IntentEvidenceStatus = Literal[
    "skipped",
    "hit",
    "partial",
    "version_mismatch",
    "no_hit",
    "unverified",
    "error",
]
IntentFeedback = Literal["correct", "incorrect"]


class IntentRouterConfigOut(BaseModel):
    enabled: bool
    mode: IntentRouterMode
    confidence_threshold: float = Field(..., ge=0, le=1)
    fallback_intent_code: str = Field(..., min_length=1, max_length=64)
    allow_general_chat: bool


class IntentRouterConfigUpdate(BaseModel):
    enabled: bool | None = None
    mode: IntentRouterMode | None = None
    confidence_threshold: float | None = Field(None, ge=0, le=1)
    fallback_intent_code: str | None = Field(None, min_length=1, max_length=64)
    allow_general_chat: bool | None = None


class IntentCategoryCreate(BaseModel):
    code: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=4000)
    examples: list[str] = Field(default_factory=list, max_length=30)
    action: IntentAction
    enabled: bool = True
    priority: int = Field(0, ge=-10000, le=10000)


class IntentCategoryUpdate(BaseModel):
    # code 是分类器和路由配置引用的稳定标识，创建后禁止修改。
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=4000)
    examples: list[str] | None = Field(None, max_length=30)
    action: IntentAction | None = None
    enabled: bool | None = None
    priority: int | None = Field(None, ge=-10000, le=10000)


class IntentCategoryOut(BaseModel):
    id: uuid.UUID
    code: str
    name: str
    description: str
    examples: list[str] = Field(default_factory=list)
    action: IntentAction
    enabled: bool
    priority: int

    model_config = {"from_attributes": True}


class IntentDecisionOut(BaseModel):
    intent_code: str
    intent_name: str
    action: IntentAction
    response_mode: IntentResponseMode
    retrieval_policy: IntentRetrievalPolicy
    need_retrieval: bool
    decision_reason: str = Field(..., min_length=1, max_length=64)
    confidence: float = Field(..., ge=0, le=1)
    source: str


class IntentRouteTestRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=12000)
    knowledge_base_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)


class IntentRouteTestResponse(BaseModel):
    decision: IntentDecisionOut
    latency_ms: int = Field(..., ge=0)


class IntentRouteLogOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None = None
    conversation_id: uuid.UUID | None = None
    intent_code: str
    intent_name: str
    action: IntentAction
    response_mode: IntentResponseMode
    retrieval_policy: IntentRetrievalPolicy
    need_retrieval: bool
    decision_reason: str
    confidence: float
    source: str
    latency_ms: int
    selected_kb_count: int
    retrieval_executed: bool | None = None
    evidence_status: IntentEvidenceStatus | None = None
    hit_count: int | None = None
    feedback: IntentFeedback | None = None
    feedback_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class IntentRouteLogPage(BaseModel):
    items: list[IntentRouteLogOut]
    total: int


class IntentRouteFeedbackUpdate(BaseModel):
    feedback: IntentFeedback
