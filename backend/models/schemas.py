import uuid
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


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
class DocumentPermissions(BaseModel):
    read: bool
    update: bool
    delete: bool


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
    permissions: DocumentPermissions

    model_config = {"from_attributes": True}


# ── Controlled terminology registry ─────────────────────────────
#
# A terminology rule is deliberately modeled as concept + spelling + scope.
# These request/response models mirror that boundary rather than accepting a
# free-form JSON blob, so callers cannot accidentally create an unscoped alias
# or promote a retrieval-only spelling into a strict equivalence.
TerminologyMatchMode = Literal["strict_equivalent", "retrieval_only"]


class TerminologyTermCreate(BaseModel):
    term: str = Field(..., min_length=1, max_length=120)
    match_mode: TerminologyMatchMode = "strict_equivalent"
    is_active: bool = True


class TerminologyTermUpdate(BaseModel):
    term: str | None = Field(default=None, min_length=1, max_length=120)
    match_mode: TerminologyMatchMode | None = None
    is_active: bool | None = None


class TerminologyTermOut(BaseModel):
    id: uuid.UUID
    concept_id: uuid.UUID
    kb_id: uuid.UUID
    term: str
    normalized_term: str
    match_mode: TerminologyMatchMode
    is_active: bool
    created_by: uuid.UUID | None = None
    updated_by: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TerminologyScopeBindingCreate(BaseModel):
    concept_id: uuid.UUID
    document_id: uuid.UUID | None = None
    # Values are normalized by the management service with the same contract
    # function reserved for future runtime snapshot resolution.  Omitting all
    # three means the whole KB.
    scope_product_key: str | None = Field(default=None, max_length=160)
    scope_version_key: str | None = Field(default=None, max_length=160)
    scope_project_key: str | None = Field(default=None, max_length=160)
    is_active: bool = True


class TerminologyScopeBindingDraft(BaseModel):
    """Initial scope sent together with a newly created concept."""

    document_id: uuid.UUID | None = None
    scope_product_key: str | None = Field(default=None, max_length=160)
    scope_version_key: str | None = Field(default=None, max_length=160)
    scope_project_key: str | None = Field(default=None, max_length=160)
    is_active: bool = True


class TerminologyScopeBindingUpdate(BaseModel):
    # ``model_fields_set`` distinguishes omitted fields from explicit null,
    # allowing an administrator to intentionally widen a document/product
    # binding back to the containing knowledge base.
    document_id: uuid.UUID | None = None
    scope_product_key: str | None = Field(default=None, max_length=160)
    scope_version_key: str | None = Field(default=None, max_length=160)
    scope_project_key: str | None = Field(default=None, max_length=160)
    is_active: bool | None = None


class TerminologyConceptCreate(BaseModel):
    code: str = Field(..., min_length=1, max_length=64)
    canonical_term: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    terms: list[TerminologyTermCreate] = Field(default_factory=list, max_length=20)
    initial_binding: TerminologyScopeBindingDraft = Field(
        default_factory=TerminologyScopeBindingDraft
    )


class TerminologyConceptUpdate(BaseModel):
    # ``code`` is intentionally absent: it is a stable registry identity.
    canonical_term: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    is_active: bool | None = None


class TerminologyConceptOut(BaseModel):
    id: uuid.UUID
    kb_id: uuid.UUID
    code: str
    canonical_term: str
    description: str | None = None
    is_active: bool
    created_by: uuid.UUID | None = None
    updated_by: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime
    terms: list[TerminologyTermOut] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class TerminologyScopeBindingOut(BaseModel):
    id: uuid.UUID
    concept_id: uuid.UUID
    kb_id: uuid.UUID
    document_id: uuid.UUID | None = None
    scope_product_key: str | None = None
    scope_version_key: str | None = None
    scope_project_key: str | None = None
    is_active: bool
    created_by: uuid.UUID | None = None
    updated_by: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime
    concept: TerminologyConceptOut | None = None

    model_config = {"from_attributes": True}


class TerminologyRegistryOut(BaseModel):
    kb_id: uuid.UUID
    registry_revision: int
    # A graph response is explicit rather than inferred by callers from a
    # mutation's one changed row.  It is the exact graph at this revision.
    concepts: list[TerminologyConceptOut] = Field(default_factory=list)
    bindings: list[TerminologyScopeBindingOut] = Field(default_factory=list)


class TerminologyMutationOut(BaseModel):
    registry_revision: int
    registry: TerminologyRegistryOut
    concept: TerminologyConceptOut | None = None
    term: TerminologyTermOut | None = None
    binding: TerminologyScopeBindingOut | None = None


# ── Chat ─────────────────────────────────────────────────────────
class SearchConfig(BaseModel):
    method: Literal["hybrid", "vector", "keyword"] = "hybrid"
    rerank: bool = True
    top_k: int = Field(5, ge=1, le=20)


class ClarificationReplyCommand(BaseModel):
    action: Literal["select", "select_all", "refine", "cancel", "new_question"]
    choice_keys: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_action_shape(self):
        keys = [str(value or "").strip() for value in self.choice_keys]
        if any(not key or len(key) > 120 for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("clarification choice_keys are invalid")
        if self.action == "select" and len(keys) != 1:
            raise ValueError("select clarification requires exactly one choice")
        if self.action == "select_all" and len(keys) < 2:
            raise ValueError("select_all clarification requires multiple choices")
        if self.action in {"refine", "cancel", "new_question"} and keys:
            raise ValueError(f"{self.action} clarification cannot include choices")
        self.choice_keys = keys
        return self


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=12000)
    conversation_id: uuid.UUID | None = None
    # ``request_id`` is an opaque client idempotency key.  Older clients omit
    # it; the API generates one before routing and returns it in the SSE
    # headers/events.  ``turn_id`` is optional so a client can correlate a
    # durable turn across retries without having to know the server UUID.
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    turn_id: uuid.UUID | None = None
    # New clients echo these fields from a durable clarification ACK.  They are
    # optional for rolling compatibility, but when present the server binds
    # them into the request fingerprint and rejects a stale selection with 409.
    pending_route_revision: int | None = Field(default=None, ge=0)
    pending_state_id: str | None = Field(default=None, min_length=1, max_length=128)
    clarification_reply: ClarificationReplyCommand | None = None
    knowledge_base_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    search_config: SearchConfig = Field(default_factory=SearchConfig)


class MessageOut(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: str
    content: str
    sources: list | None
    clarification: dict | None = None
    tokens: int | None = None
    turn_id: uuid.UUID | None = None
    request_id: str | None = None
    # ``status`` is the durable turn status for this transcript row.  The
    # explicit ``turn_status`` alias is retained for clients that already use
    # the more descriptive name.
    status: str | None = None
    turn_status: str | None = None
    trace_id: str | None = None
    evidence_status: str | None = None
    retrieval_executed: bool | None = None
    error_code: str | None = None
    delivery_status: str | None = None
    persistence_status: str | None = None
    duration_ms: int | None = None
    search_snapshot: dict | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ConversationBatchDeleteRequest(BaseModel):
    conversation_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=100)


# ── Search ───────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=12000)
    knowledge_base_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    method: Literal["hybrid", "vector", "keyword"] = "hybrid"
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = True


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
    # Canonical producer/output spelling.  ``version_mismatch`` remains in
    # the read schema only for rolling upgrades of persisted route logs.
    "scope_mismatch",
    "version_mismatch",
    "no_hit",
    "insufficient_evidence",
    "unverified",
    "needs_clarification",
    "error",
]


class IntentRouterConfigOut(BaseModel):
    enabled: bool
    mode: IntentRouterMode
    confidence_threshold: float = Field(..., ge=0, le=1)
    fallback_intent_code: str = Field(..., min_length=1, max_length=64)
    allow_general_chat: bool
    route_schema_version: str
    contract_schema_version: str
    prompt_version: str


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


class IntentRouteContextMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class IntentRouteTestRequest(BaseModel):
    # ``question`` is retained for old clients; the v1 sandbox uses
    # ``current_input`` and never loads a real conversation or KB document.
    question: str | None = Field(None, min_length=1, max_length=12000)
    current_input: str | None = Field(None, min_length=1, max_length=12000)
    context_messages: list[IntentRouteContextMessage] = Field(
        default_factory=list,
        max_length=6,
    )
    selected_kb_count: int = Field(0, ge=0, le=100)
    knowledge_base_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_current_input(self):
        value = (self.current_input or self.question or "").strip()
        if not value:
            raise ValueError("current_input 或 question 不能为空")
        self.current_input = value
        self.question = value
        return self


class IntentRouteTestResponse(BaseModel):
    decision: IntentDecisionOut
    route_decision: dict[str, Any] | None = None
    task_contract: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(..., ge=0)
