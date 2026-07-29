from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


ROOT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # Docker Compose 会显式覆盖；该默认值仅用于宿主机本地开发的端口约定。
    database_url: str = "postgresql+asyncpg://rag:password@127.0.0.1:5433/rag_prod"

    # 大语言模型
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048

    # 向量模型
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 2560

    # 多模态模型（图片/截图识别）
    vision_api_key: str = ""
    vision_base_url: str = "https://api.openai.com/v1"
    vision_model: str = "gpt-4o"

    # 检索参数
    top_k: int = 5
    rerank_enabled: bool = True
    show_sources: bool = True  # 是否在问答回答下方展示知识库来源 / 参考来源
    upload_dir: str = "uploads"

    # 站点品牌（公开可读，无需鉴权）
    site_title: str = "RAG 检索系统"           # 左上角标题 / 站点名称
    site_description: str = "知识增强·精准问答"  # 左上角副标题 / 站点描述
    site_logo: str = ""                         # 图标 URL，空则前端用默认徽标
    browser_title: str = ""                     # 浏览器标签标题，空则回退到 site_title
    site_copyright: str = ""                     # 页面底部版权文字，空则不显示页脚

    # 认证与权限（JWT）
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 720
    admin_init_password: str = "admin12345"

    class Config:
        # 宿主机开发始终读取项目根目录的统一 .env；Docker 通过环境变量覆盖。
        env_file = ROOT_ENV_FILE
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
