import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.reranker import rerank, rerank_with_status


def _client_with_payload(payload: dict):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
    )


class RerankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_valid_scores_are_trusted_and_sorted(self) -> None:
        results = [
            {"id": "a", "content": "A", "score": 0.02},
            {"id": "b", "content": "B", "score": 0.01},
        ]
        client = _client_with_payload({
            "scores": [
                {"index": 1, "score": 0.2},
                {"index": 2, "score": 0.9},
            ]
        })

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            outcome = await rerank_with_status("query", results)

        self.assertTrue(outcome.succeeded)
        self.assertEqual([item["id"] for item in outcome.results], ["b", "a"])
        self.assertEqual([item["score"] for item in outcome.results], [0.9, 0.2])
        # 不原地覆盖检索器返回值，失败回退时才能保留原始分数语义。
        self.assertEqual(results[0]["score"], 0.02)

    async def test_incomplete_scores_are_untrusted_and_preserve_original_order(self) -> None:
        results = [
            {"id": "a", "content": "A", "score": 0.02},
            {"id": "b", "content": "B", "score": 0.01},
        ]
        client = _client_with_payload({
            "scores": [{"index": 1, "score": 0.8}],
        })

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            outcome = await rerank_with_status("query", results)
            compatible_results = await rerank("query", results)

        self.assertFalse(outcome.succeeded)
        self.assertIn("未覆盖全部候选", outcome.error or "")
        self.assertEqual(outcome.results, results)
        self.assertEqual(compatible_results, results)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
