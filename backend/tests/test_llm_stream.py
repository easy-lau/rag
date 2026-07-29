import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APITimeoutError

from core.llm_stream import stream_with_retry_before_first_delta


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


def _chunk(content: str | None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


class LlmStreamRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_timeout_before_first_delta(self) -> None:
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")
        open_stream = AsyncMock(
            side_effect=[
                APITimeoutError(request=request),
                _stream(_chunk("你好")),
            ]
        )
        sleep = AsyncMock()

        with patch("core.llm_stream.asyncio.sleep", sleep):
            result = [
                chunk async for chunk in stream_with_retry_before_first_delta(
                    open_stream,
                    model="test-chat",
                    prompt_chars=12,
                    timeout_seconds=60,
                    max_attempts=3,
                    retry_base_delay_seconds=1,
                )
            ]

        self.assertEqual(len(result), 1)
        self.assertEqual(open_stream.await_count, 2)
        sleep.assert_awaited_once_with(1.0)

    async def test_does_not_retry_after_text_has_been_emitted(self) -> None:
        request = httpx.Request("POST", "https://example.test/v1/chat/completions")

        async def partial_stream():
            yield _chunk("已输出")
            raise APITimeoutError(request=request)

        open_stream = AsyncMock(return_value=partial_stream())
        sleep = AsyncMock()

        with patch("core.llm_stream.asyncio.sleep", sleep):
            with self.assertRaises(APITimeoutError):
                _ = [
                    chunk async for chunk in stream_with_retry_before_first_delta(
                        open_stream,
                        model="test-chat",
                        prompt_chars=12,
                        timeout_seconds=60,
                        max_attempts=3,
                        retry_base_delay_seconds=1,
                    )
                ]

        open_stream.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_does_not_retry_non_transient_error(self) -> None:
        open_stream = AsyncMock(side_effect=ValueError("无效模型"))
        sleep = AsyncMock()

        with patch("core.llm_stream.asyncio.sleep", sleep):
            with self.assertRaisesRegex(ValueError, "无效模型"):
                _ = [
                    chunk async for chunk in stream_with_retry_before_first_delta(
                        open_stream,
                        model="test-chat",
                        prompt_chars=12,
                        timeout_seconds=60,
                        max_attempts=3,
                        retry_base_delay_seconds=1,
                    )
                ]

        open_stream.assert_awaited_once()
        sleep.assert_not_awaited()
