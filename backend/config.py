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
    # 新主链默认启用；部署环境仍可显式设为 v1 并重启以无迁移回滚。
    rag_pipeline_version: Literal["v1", "v2"] = "v2"
    # V2 检索的各阶段期限还受整段工作流总期限约束；扩展超时只降级证据，
    # 不清空首轮召回。改回 rag_pipeline_version=v1 可整体回滚 V2 主链。
    rag_v2_retrieval_timeout_seconds: float = Field(15.0, ge=1.0, le=120.0)
    rag_v2_expansion_timeout_seconds: float = Field(8.0, ge=0.5, le=60.0)
    rag_v2_retrieval_workflow_timeout_seconds: float = Field(
        22.0,
        ge=1.0,
        le=180.0,
    )
    # 包住建流、首分片前重试、退避等待和完整流读取，避免 max_attempts 倍增
    # llm_request_timeout_seconds。首个文本分片后的异常仍由流层直接抛出，不重放。
    rag_v2_generation_workflow_timeout_seconds: float = Field(
        60.0,
        ge=1.0,
        le=300.0,
    )

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
    show_sources: bool = Field(
        True,
        validation_alias="__DATABASE_SETTINGS_ONLY_SHOW_SOURCES",
    )  # 是否在问答回答下方展示知识库来源 / 参考来源
    upload_dir: str = "uploads"

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
