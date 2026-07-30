from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


ROOT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
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
    temperature: float = Field(
        0.7, validation_alias="__DATABASE_SETTINGS_ONLY_TEMPERATURE"
    )
    max_tokens: int = Field(
        2048, validation_alias="__DATABASE_SETTINGS_ONLY_MAX_TOKENS"
    )
    # 聊天流在首个文本分片前发生的瞬时上游故障可安全重试。
    llm_request_timeout_seconds: float = 60.0
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

    class Config:
        # 宿主机开发始终读取项目根目录的统一 .env；Docker 通过环境变量覆盖。
        env_file = ROOT_ENV_FILE
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
