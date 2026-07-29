import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APITimeoutError

from core.embeddings import embed_batch


class EmbeddingRetryTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, **overrides):
        values = {
            "embedding_model": "doubao-embedding-text-240715",
            "embedding_request_timeout_seconds": 60.0,
            "embedding_max_attempts": 3,
            "embedding_retry_base_delay_seconds": 1.0,
            "embedding_batch_size": 64,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _response(*embeddings):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=embedding)
                for index, embedding in enumerate(embeddings)
            ]
        )

    async def test_retries_timeout_then_returns_embeddings(self) -> None:
        request = httpx.Request("POST", "https://example.test/v1/embeddings")
        create = AsyncMock(
            side_effect=[
                APITimeoutError(request=request),
                self._response([0.1, 0.2], [0.3, 0.4]),
            ]
        )
        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        sleep = AsyncMock()

        with (
            patch("core.embeddings.get_settings", return_value=self._settings()),
            patch("core.embeddings.get_embedding_client", return_value=client),
            patch("core.embeddings.asyncio.sleep", sleep),
        ):
            result = await embed_batch(["第一段", "第二段"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(create.await_count, 2)
        sleep.assert_awaited_once_with(1.0)
        self.assertEqual(create.await_args.kwargs["timeout"], 60.0)

    async def test_does_not_retry_non_transient_failure(self) -> None:
        create = AsyncMock(side_effect=ValueError("模型配置错误"))
        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        sleep = AsyncMock()

        with (
            patch("core.embeddings.get_settings", return_value=self._settings()),
            patch("core.embeddings.get_embedding_client", return_value=client),
            patch("core.embeddings.asyncio.sleep", sleep),
        ):
            with self.assertRaisesRegex(ValueError, "模型配置错误"):
                await embed_batch(["第一段"])

        create.assert_awaited_once()
        sleep.assert_not_awaited()

    async def test_respects_configured_batch_size_and_response_indexes(self) -> None:
        create = AsyncMock(
            side_effect=[
                SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, embedding=[0.2]),
                        SimpleNamespace(index=0, embedding=[0.1]),
                    ]
                ),
                self._response([0.3]),
            ]
        )
        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))

        with (
            patch(
                "core.embeddings.get_settings",
                return_value=self._settings(embedding_batch_size=2),
            ),
            patch("core.embeddings.get_embedding_client", return_value=client),
        ):
            result = await embed_batch(["一", "二", "三"])

        self.assertEqual(result, [[0.1], [0.2], [0.3]])
        self.assertEqual(create.await_count, 2)
