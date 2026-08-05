from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path
from typing import Literal


ROOT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # 运行环境与日志。宿主机直接启动默认按开发环境处理；Docker Compose 会显式
    # 设为 production，避免生产日志默认记录用户问题、模型回答等业务正文。
    app_env: str = "development"
    # 发布镜像在构建阶段注入；本地直接运行默认 dev。结构化追踪会携带这两个
    # 字段，便于比较不同发版/提交上的检索算法指标。
    app_version: str = "dev"
    app_revision: str = ""
    log_level: str = "INFO"
    # 仅开发环境启用。聊天链路按 conversation_id 追加到独立日志文件；没有
    # 会话归属的启动/系统日志保留在 worker 系统文件。生产保持 stdout，由容器
    # 日志驱动、Loki 或 ELK 等集中收集系统接管。
    development_log_dir: str = str(
        Path(__file__).resolve().parent.parent / "logs" / "development"
    )
    rag_trace_enabled: bool = True
    rag_trace_content_enabled: bool | None = None
    rag_trace_candidate_details_enabled: bool | None = None
    rag_trace_content_max_chars: int = Field(50000, ge=1000, le=1000000)
    # 调用链事件异步写入数据库，供后台按 trace_id / 会话 / 时间检索。
    # 正文是否入库仍统一服从 rag_trace_include_content；生产环境默认只保存摘要、
    # 指标与对象 ID。保留期到期后由后台清理任务级联删除事件明细。
    rag_trace_persistence_enabled: bool = True
    rag_trace_retention_days: int = Field(30, ge=1, le=365)
    # 队列按事件数和单事件字节双重限界。500 条足以覆盖数轮完整候选链，
    # PostgreSQL 长时间不可用时也不会让可观测数据无上限挤占应用内存。
    rag_trace_queue_size: int = Field(500, ge=100, le=100000)
    rag_trace_max_event_bytes: int = Field(131072, ge=16384, le=1048576)
    rag_trace_max_events_per_run: int = Field(500, ge=20, le=5000)
    # V2 检索的各阶段期限还受整段工作流总期限约束；扩展超时只降级证据，
    # 不清空首轮召回；扩展失败只降级证据状态。
    rag_v2_retrieval_timeout_seconds: float = Field(15.0, ge=1.0, le=120.0)
    rag_v2_expansion_timeout_seconds: float = Field(8.0, ge=0.5, le=60.0)
    rag_v2_retrieval_workflow_timeout_seconds: float = Field(
        22.0,
        ge=1.0,
        le=180.0,
    )
    # 确定性证据图无法闭合时，才在已授权且限量的检索候选中使用结构化模型
    # 判断哪些片段真正支撑原问题。模型无权扩大 KB/文档范围、重写原问题或
    # 绕过最终证据校验；异常和超时保留确定性候选链。该开关用于在不回滚 V2
    # 的情况下快速降级为纯确定性模式。
    rag_v2_model_evidence_adjudication_enabled: bool = True
    # Evidence adjudication is an optional enhancement and must not consume the
    # full answer-model deadline.  Eight seconds bounds tail latency while a
    # deployment may still set ``None`` explicitly to inherit the ordinary LLM
    # timeout when accuracy is preferred over latency.
    rag_v2_model_evidence_adjudication_timeout_seconds: float | None = Field(
        8.0,
        ge=0.5,
        le=300.0,
    )
    # 任务图同一 wave 内的检索并发数。每个任务使用独立只读会话，避免并发
    # 复用请求事务；范围限制在连接池和单次工作流期限可承受的边界内。
    rag_v2_task_query_parallelism: int = Field(3, ge=1, le=8)
    # 包住建流、首分片前重试、退避等待和完整流读取，避免 max_attempts 倍增
    # llm_request_timeout_seconds。首个文本分片后的异常仍由流层直接抛出，不重放。
    rag_v2_generation_workflow_timeout_seconds: float = Field(
        60.0,
        ge=1.0,
        le=300.0,
    )
    # 语义理解入口由此开关单独决定。默认 V3：模型只能选择服务器签发的
    # source span，随后由后端编译
    # V2 任务图。``legacy`` 仅保留给已知回滚场景使用 query_analysis.v2。
    rag_semantic_entry: Literal["legacy", "v3"] = "v3"
    # V3 是当前生产语义入口：模型只能选择服务器签发的 source span，随后由
    # 后端编译为 V2 任务图。模型失败、schema 拒绝或容量不足时原子回退当前轮
    # 本地计划；本地 planner 的未知/不可运行不再能在 V3 前提前拦截请求。
    rag_query_understanding_v3_mode: Literal["off", "shadow", "active"] = "active"
    # ``None`` means that V3 has no separate short deadline and instead shares
    # the normal LLM request deadline.  Set a numeric value only when the
    # deployment explicitly needs to trade V3 understanding quality for a
    # tighter first-answer latency budget.
    rag_query_understanding_v3_active_timeout_seconds: float | None = Field(
        None,
        ge=0.5,
        le=300.0,
    )
    rag_query_understanding_v3_active_max_inflight: int = Field(2, ge=1, le=32)
    # 与 V3 模型理解并发的不可变 anchor 预取。超过这个短期限直接丢弃，由
    # 正常 V2 DAG 重新召回，不能为了缓存而延迟用户首个检索阶段。
    rag_query_understanding_v3_anchor_prefetch_enabled: bool = True
    rag_query_understanding_v3_anchor_prefetch_timeout_seconds: float = Field(
        2.5,
        ge=0.5,
        le=15.0,
    )
    # 旧 query_analysis.v2 仅为回滚/对照保留，不能与 V3 同时成为生产语义
    # authority；默认关闭，避免同一请求连续等待两次模型分析。
    rag_query_analyzer_mode: Literal["off", "shadow", "active"] = "off"
    # Direct/unit callers retain this legacy default.  Production active and
    # shadow execution below each declare their own bounded budget explicitly.
    rag_query_analyzer_timeout_seconds: float = Field(5.0, ge=0.5, le=30.0)
    # Active mode sits on the request path, so it has a separate short budget.
    # Shadow analysis remains asynchronous and may use the diagnostic budget.
    rag_query_analyzer_active_timeout_seconds: float = Field(
        1.5,
        ge=0.5,
        le=10.0,
    )
    rag_query_analyzer_active_max_inflight: int = Field(2, ge=1, le=32)
    rag_query_analyzer_shadow_timeout_seconds: float = Field(
        5.0,
        ge=0.5,
        le=30.0,
    )
    rag_query_analyzer_shadow_max_inflight: int = Field(2, ge=1, le=32)
    # Deterministic per-trace sampling; 0 disables background telemetry and 1
    # observes every eligible request without changing its retrieval result.
    rag_query_analyzer_shadow_sample_rate: float = Field(0.1, ge=0.0, le=1.0)

    # Docker Compose 会显式覆盖；该默认值仅用于宿主机本地开发的端口约定。
    database_url: str = "postgresql+asyncpg://rag:password@127.0.0.1:5433/rag_prod"

    # 后台可管理的运行时配置只允许由 settings 表加载；不能再被 .env 中遗留的
    # LLM_* / EMBEDDING_* / VISION_* / TOP_K 等覆盖。下面的别名故意不是公开
    # 环境变量，只让 BaseSettings 忽略同名环境变量，默认值供首次进入后台时显示。
    llm_api_key: str = Field("", validation_alias="__DATABASE_SETTINGS_ONLY_LLM_API_KEY")
    llm_base_url: str = Field(
        "https://api.openai.com/v1",
        validation_alias="__DATABASE_SETTINGS_ONLY_LLM_BASE_URL",
    )
    chat_model: str = Field("gpt-4o", validation_alias="__DATABASE_SETTINGS_ONLY_CHAT_MODEL")
    # 意图识别与对话共用 LLM 服务凭据；留空时运行时自动复用 chat_model。
    intent_model: str = Field(
        "", validation_alias="__DATABASE_SETTINGS_ONLY_INTENT_MODEL"
    )
    # 检索重排与对话共用 LLM 服务凭据；留空时运行时自动复用 chat_model。
    rerank_model: str = Field(
        "", validation_alias="__DATABASE_SETTINGS_ONLY_RERANK_MODEL"
    )
    temperature: float = Field(
        0.7, validation_alias="__DATABASE_SETTINGS_ONLY_TEMPERATURE"
    )
    max_tokens: int = Field(
        2048, validation_alias="__DATABASE_SETTINGS_ONLY_MAX_TOKENS"
    )
    # 聊天流在首个文本分片前发生的瞬时上游故障可安全重试。
    llm_request_timeout_seconds: float = 60.0
    # 语义路由只负责生成小型结构化合同，使用独立的整段工作流期限；超时后
    # 回退到需检索的安全合同，不能占用回答生成所允许的 60 秒预算。
    rag_route_timeout_seconds: float = Field(12.0, ge=1.0, le=60.0)
    llm_max_attempts: int = 3
    llm_retry_base_delay_seconds: float = 1.0

    # 向量模型
    embedding_api_key: str = Field(
        "", validation_alias="__DATABASE_SETTINGS_ONLY_EMBEDDING_API_KEY"
    )
    embedding_base_url: str = Field(
        "https://api.openai.com/v1",
        validation_alias="__DATABASE_SETTINGS_ONLY_EMBEDDING_BASE_URL",
    )
    embedding_model: str = Field(
        "text-embedding-3-small",
        validation_alias="__DATABASE_SETTINGS_ONLY_EMBEDDING_MODEL",
    )
    # pgvector 列已固定为 2560 维，不能通过 .env 或后台设置改写。
    embedding_dimensions: int = Field(
        2560, validation_alias="__DATABASE_SETTINGS_ONLY_EMBEDDING_DIMENSIONS"
    )
    # 向量服务偶发超时的容错参数。max_attempts 包含首次请求。
    embedding_request_timeout_seconds: float = 60.0
    embedding_max_attempts: int = 3
    embedding_retry_base_delay_seconds: float = 1.0
    embedding_batch_size: int = 64

    # 多模态模型（图片/截图识别）
    vision_api_key: str = Field("", validation_alias="__DATABASE_SETTINGS_ONLY_VISION_API_KEY")
    vision_base_url: str = Field(
        "https://api.openai.com/v1",
        validation_alias="__DATABASE_SETTINGS_ONLY_VISION_BASE_URL",
    )
    vision_model: str = Field(
        "gpt-4o", validation_alias="__DATABASE_SETTINGS_ONLY_VISION_MODEL"
    )

    # 检索参数
    top_k: int = Field(5, validation_alias="__DATABASE_SETTINGS_ONLY_TOP_K")
    rerank_enabled: bool = Field(
        True, validation_alias="__DATABASE_SETTINGS_ONLY_RERANK_ENABLED"
    )
    # Knowledge-base terminal states remain authoritative.  Administrators may
    # explicitly allow a separately labelled general-model answer, but that
    # answer never becomes a knowledge-base hit or acquires source authority.
    rag_general_fallback_mode: Literal[
        "off",
        "no_hit",
        "no_hit_or_insufficient",
    ] = Field(
        "off",
        validation_alias="__DATABASE_SETTINGS_ONLY_RAG_GENERAL_FALLBACK_MODE",
    )
    # 通用兜底可使用单独的低延迟模型；留空时沿用主对话模型。
    rag_general_fallback_model: str = Field(
        "",
        validation_alias="__DATABASE_SETTINGS_ONLY_RAG_GENERAL_FALLBACK_MODEL",
    )
    show_sources: bool = Field(
        True,
        validation_alias="__DATABASE_SETTINGS_ONLY_SHOW_SOURCES",
    )  # 是否在问答回答下方展示知识库来源 / 参考来源
    upload_dir: str = "uploads"
    # 文档入库由独立进程从 PostgreSQL 领取任务；API 进程只负责在同一事务中
    # 持久化文档和任务。租约过期后可由任一 worker 接手，避免重启丢任务。
    document_job_max_attempts: int = Field(3, ge=1, le=20)
    document_job_lease_seconds: int = Field(900, ge=30, le=86400)
    document_job_poll_seconds: float = Field(1.0, ge=0.1, le=60.0)
    # Compose/生产由 Supervisor 独立拉起 worker；宿主机 ``uvicorn`` 开发时
    # 默认同进程托管一个领取器，避免本地上传还要记住额外开终端。两种模式都
    # 使用同一张任务表和 lease，显式开启两者也不会重复消费同一任务。
    document_job_embedded_worker: bool | None = None

    @property
    def document_job_runs_embedded_worker(self) -> bool:
        if self.document_job_embedded_worker is not None:
            return self.document_job_embedded_worker
        return self.app_env.strip().lower() not in {"prod", "production"}

    # 站点品牌（公开可读，无需鉴权）
    site_title: str = Field(
        "RAG 检索系统", validation_alias="__DATABASE_SETTINGS_ONLY_SITE_TITLE"
    )  # 左上角标题 / 站点名称
    site_description: str = Field(
        "知识增强·精准问答",
        validation_alias="__DATABASE_SETTINGS_ONLY_SITE_DESCRIPTION",
    )  # 左上角副标题 / 站点描述
    site_logo: str = Field(
        "", validation_alias="__DATABASE_SETTINGS_ONLY_SITE_LOGO"
    )  # 图标 URL，空则前端用默认徽标
    browser_title: str = Field(
        "", validation_alias="__DATABASE_SETTINGS_ONLY_BROWSER_TITLE"
    )  # 浏览器标签标题，空则回退到 site_title
    site_copyright: str = Field(
        "", validation_alias="__DATABASE_SETTINGS_ONLY_SITE_COPYRIGHT"
    )  # 页面底部版权文字，空则不显示页脚

    # 认证与权限（JWT）
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 720
    admin_init_password: str = "admin12345"
    # 登录防护属于部署安全策略，只允许由服务端环境变量配置，不能下放到后台页面。
    # pair 只限制某一来源对某一用户名的尝试；ip 用于限制同一来源扫描多个账号；
    # account 只触发安全告警，不能据此锁定账号。
    login_pair_failure_threshold: int = Field(5, ge=3, le=20)
    login_pair_window_minutes: int = Field(15, ge=1, le=1440)
    login_pair_block_minutes: int = Field(15, ge=1, le=1440)
    login_ip_failure_threshold: int = Field(20, ge=5, le=1000)
    login_ip_window_minutes: int = Field(15, ge=1, le=1440)
    login_ip_block_minutes: int = Field(60, ge=1, le=10080)
    login_account_alert_threshold: int = Field(20, ge=5, le=1000)
    login_account_alert_window_minutes: int = Field(15, ge=1, le=1440)
    login_throttle_retention_hours: int = Field(48, ge=24, le=720)
    login_log_aggregate_seconds: int = Field(60, ge=10, le=3600)
    login_log_retention_days: int = Field(90, ge=7, le=3650)
    # 仅用于加密 settings 表中的模型 API Key；必须保留在部署环境，绝不写入数据库。
    config_encryption_key: str = ""

    @property
    def rag_trace_include_content(self) -> bool:
        """是否在算法追踪日志中记录完整问题、回答和候选片段。

        开发环境默认开启，生产环境默认关闭；部署者仍可通过
        ``RAG_TRACE_CONTENT_ENABLED`` 显式覆盖。
        """

        if self.rag_trace_content_enabled is not None:
            return self.rag_trace_content_enabled
        return self.app_env.strip().lower() not in {"prod", "production"}

    @property
    def rag_trace_include_candidate_details(self) -> bool:
        """逐候选事件开发默认开启、生产默认关闭，避免日志量随候选数线性膨胀。"""

        if self.rag_trace_candidate_details_enabled is not None:
            return self.rag_trace_candidate_details_enabled
        return self.app_env.strip().lower() not in {"prod", "production"}

    class Config:
        # 宿主机开发始终读取项目根目录的统一 .env；Docker 通过环境变量覆盖。
        env_file = ROOT_ENV_FILE
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
