from openai import AsyncOpenAI
from config import get_settings


def get_llm_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.llm_api_key or s.embedding_api_key,
        base_url=s.llm_base_url,
    )


def get_embedding_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.embedding_api_key or s.llm_api_key,
        base_url=s.embedding_base_url,
    )


def get_vision_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.vision_api_key or s.llm_api_key or s.embedding_api_key,
        base_url=s.vision_base_url or s.llm_base_url,
    )


# backward compat
def get_client() -> AsyncOpenAI:
    return get_llm_client()
