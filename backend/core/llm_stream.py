import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from core.rag_trace import exception_log_text


logger = logging.getLogger(__name__)


def _is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and 500 <= exc.status_code < 600


def _has_text_delta(chunk: Any) -> bool:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return False
    delta = getattr(choices[0], "delta", None)
    return bool(getattr(delta, "content", None))


async def stream_with_retry_before_first_delta(
    open_stream: Callable[[], Awaitable[AsyncIterator[Any]]],
    *,
    model: str,
    prompt_chars: int,
    timeout_seconds: float,
    max_attempts: int,
    retry_base_delay_seconds: float,
) -> AsyncIterator[Any]:
    """重试尚未输出文本的聊天流，输出首个文本后绝不重放，防止回答重复。"""
    attempt_limit = max(1, max_attempts)

    for attempt in range(1, attempt_limit + 1):
        emitted_text = False
        try:
            stream = await open_stream()
            async for chunk in stream:
                if _has_text_delta(chunk):
                    emitted_text = True
                yield chunk
            return
        except Exception as exc:
            can_retry = (
                not emitted_text
                and _is_retryable_llm_error(exc)
                and attempt < attempt_limit
            )
            if not can_retry:
                logger.error(
                    "[聊天模型] 流请求失败 model=%s prompt_chars=%d timeout=%.1fs attempt=%d/%d emitted_text=%s error=%s",
                    model,
                    prompt_chars,
                    timeout_seconds,
                    attempt,
                    attempt_limit,
                    emitted_text,
                    exception_log_text(exc),
                )
                raise

            delay = max(0.0, retry_base_delay_seconds) * (2 ** (attempt - 1))
            logger.warning(
                "[聊天模型] 首个分片前请求异常，%.1f 秒后重试 model=%s prompt_chars=%d timeout=%.1fs attempt=%d/%d error=%s",
                delay,
                model,
                prompt_chars,
                timeout_seconds,
                attempt,
                attempt_limit,
                exception_log_text(exc),
            )
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable")  # pragma: no cover
