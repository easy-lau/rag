import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from config import get_settings
from core.openai_client import get_embedding_client


logger = logging.getLogger(__name__)


def _is_retryable_embedding_error(exc: Exception) -> bool:
    """只重试短暂性上游故障，避免把密钥、模型等配置错误无意义地拖慢。"""
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and 500 <= exc.status_code < 600


async def _create_embeddings_with_retry(
    client: Any,
    *,
    model: str,
    inputs: str | list[str],
    timeout_seconds: float,
    max_attempts: int,
    retry_base_delay_seconds: float,
) -> Any:
    texts = [inputs] if isinstance(inputs, str) else inputs
    attempt_limit = max(1, max_attempts)
    char_count = sum(len(text) for text in texts)

    for attempt in range(1, attempt_limit + 1):
        try:
            return await client.embeddings.create(
                model=model,
                input=inputs,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            can_retry = _is_retryable_embedding_error(exc) and attempt < attempt_limit
            if not can_retry:
                logger.error(
                    "[向量化] 请求失败 model=%s inputs=%d chars=%d attempt=%d/%d error=%s: %s",
                    model,
                    len(texts),
                    char_count,
                    attempt,
                    attempt_limit,
                    type(exc).__name__,
                    exc,
                )
                raise

            delay = max(0.0, retry_base_delay_seconds) * (2 ** (attempt - 1))
            logger.warning(
                "[向量化] 请求异常，%.1f 秒后重试 model=%s inputs=%d chars=%d attempt=%d/%d error=%s: %s",
                delay,
                model,
                len(texts),
                char_count,
                attempt,
                attempt_limit,
                type(exc).__name__,
                exc,
            )
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable")  # pragma: no cover


def _ordered_embeddings(response: Any, expected_count: int) -> list[list[float]]:
    data = sorted(response.data, key=lambda item: item.index)
    if len(data) != expected_count:
        raise RuntimeError(
            f"向量服务返回数量异常：期望 {expected_count} 条，实际 {len(data)} 条"
        )
    return [item.embedding for item in data]


async def embed_text(text: str) -> list[float]:
    s = get_settings()
    t0 = time.perf_counter()
    response = await _create_embeddings_with_retry(
        get_embedding_client(),
        model=s.embedding_model,
        inputs=text,
        timeout_seconds=s.embedding_request_timeout_seconds,
        max_attempts=s.embedding_max_attempts,
        retry_base_delay_seconds=s.embedding_retry_base_delay_seconds,
    )
    embedding = _ordered_embeddings(response, 1)[0]
    logger.info(
        "[向量化] 模型=%s 输入=%d字符 维度=%d 耗时=%.0fms",
        s.embedding_model,
        len(text),
        len(embedding),
        (time.perf_counter() - t0) * 1000,
    )
    return embedding


async def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    s = get_settings()
    client = get_embedding_client()
    batch_size = max(1, s.embedding_batch_size)
    out: list[list[float]] = []

    for offset in range(0, len(texts), batch_size):
        batch = list(texts[offset:offset + batch_size])
        t0 = time.perf_counter()
        response = await _create_embeddings_with_retry(
            client,
            model=s.embedding_model,
            inputs=batch,
            timeout_seconds=s.embedding_request_timeout_seconds,
            max_attempts=s.embedding_max_attempts,
            retry_base_delay_seconds=s.embedding_retry_base_delay_seconds,
        )
        embeddings = _ordered_embeddings(response, len(batch))
        out.extend(embeddings)
        logger.info(
            "[向量化] 模型=%s batch=%d-%d/%d 输入=%d条/%d字符 维度=%d 耗时=%.0fms",
            s.embedding_model,
            offset + 1,
            offset + len(batch),
            len(texts),
            len(batch),
            sum(len(text) for text in batch),
            len(embeddings[0]) if embeddings else 0,
            (time.perf_counter() - t0) * 1000,
        )

    return out
