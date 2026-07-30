from openai import AsyncOpenAI
from config import get_settings


def _same_endpoint(left: str, right: str) -> bool:
    """仅在两个服务实际指向同一 OpenAI 兼容入口时复用凭据。"""
    return left.rstrip("/") == right.rstrip("/")


def get_service_credentials(service: str, settings=None) -> tuple[str, str]:
    """解析一个模型服务实际会使用的 ``(api_key, base_url)``。

    仅允许同一 Base URL 的服务组复用已有密钥；调用方不应自行拼接回退规则，
    这样连接测试与实际生产调用可以保持一致。
    """
    settings = settings or get_settings()
    if service == "llm":
        base_url = settings.llm_base_url
        api_key = settings.llm_api_key or (
            settings.embedding_api_key
            if _same_endpoint(base_url, settings.embedding_base_url)
            else ""
        )
        return api_key, base_url

    if service == "embedding":
        base_url = settings.embedding_base_url
        api_key = settings.embedding_api_key or (
            settings.llm_api_key
            if _same_endpoint(base_url, settings.llm_base_url)
            else ""
        )
        return api_key, base_url

    if service == "vision":
        base_url = settings.vision_base_url or settings.llm_base_url
        api_key = settings.vision_api_key or (
            settings.llm_api_key
            if _same_endpoint(base_url, settings.llm_base_url)
            else ""
        ) or (
            settings.embedding_api_key
            if _same_endpoint(base_url, settings.embedding_base_url)
            else ""
        )
        return api_key, base_url

    raise ValueError(f"未知模型服务：{service}")


def get_llm_client() -> AsyncOpenAI:
    api_key, base_url = get_service_credentials("llm")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def get_embedding_client() -> AsyncOpenAI:
    api_key, base_url = get_service_credentials("embedding")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        # Embedding 的重试由 core.embeddings 统一处理，便于记录批次和退避日志。
        max_retries=0,
    )


def get_vision_client() -> AsyncOpenAI:
    api_key, base_url = get_service_credentials("vision")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )


# backward compat
def get_client() -> AsyncOpenAI:
    return get_llm_client()
