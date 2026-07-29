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
    tags: list[str] = []
    created_at: datetime
    updated_at: datetime | None = None
    created_by_name: str | None = None
    updated_by_name: str | None = None

    model_config = {"from_attributes": True}


# ── Chat ─────────────────────────────────────────────────────────
class SearchConfig(BaseModel):
    method: str = "hybrid"       # hybrid | vector | keyword
    rerank: bool = True
    top_k: int = 5
    tags: list[str] = []         # 用户手动勾选的标签，对命中文档做检索软加权


class ChatRequest(BaseModel):
    question: str
    conversation_id: uuid.UUID | None = None
    knowledge_base_ids: list[uuid.UUID] = []
    search_config: SearchConfig = SearchConfig()


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
    query: str
    knowledge_base_ids: list[uuid.UUID] = []
    method: str = "hybrid"
    top_k: int = 5
    rerank: bool = True
    tags: list[str] = []         # 标签软加权（命中文档排序分上浮，不硬过滤）


class SearchResultItem(BaseModel):
    id: uuid.UUID
    content: str
    filename: str
    file_type: str | None
    score: float
    chunk_index: int
    metadata: dict | None
    tags: list[str] = []


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int
    search_meta: dict[str, Any]


# ── Auth ─────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class MeOut(BaseModel):
    id: uuid.UUID
    username: str
    display_name: str | None = None
    is_superadmin: bool
    role_name: str | None = None
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
    name: str
    description: str | None = None
    permissions: list[str] = []
    kb_ids: list[uuid.UUID] = []


class RoleOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None = None
    is_system: bool
    permissions: list[str] = []
    kb_ids: list[uuid.UUID] = []
    created_at: datetime

    model_config = {"from_attributes": True}


class RoleUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    permissions: list[str] | None = None
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


# ── Intent Routing ──────────────────────────────────────────────
IntentRouterMode = Literal["rules_then_llm", "llm_only", "off"]
IntentAction = Literal["retrieve", "chat", "writing", "system_help"]
IntentFeedback = Literal["correct", "incorrect"]


class IntentRouterConfigOut(BaseModel):
    enabled: bool
    mode: IntentRouterMode
    intent_model: str = ""
    confidence_threshold: float = Field(..., ge=0, le=1)
    fallback_intent_code: str = Field(..., min_length=1, max_length=64)
    allow_general_chat: bool


class IntentRouterConfigUpdate(BaseModel):
    enabled: bool | None = None
    mode: IntentRouterMode | None = None
    intent_model: str | None = Field(None, max_length=255)
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
    confidence: float
    source: str
    latency_ms: int
    selected_kb_count: int
    feedback: IntentFeedback | None = None
    feedback_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class IntentRouteLogPage(BaseModel):
    items: list[IntentRouteLogOut]
    total: int


class IntentRouteFeedbackUpdate(BaseModel):
    feedback: IntentFeedback
