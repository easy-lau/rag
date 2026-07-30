import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.rag_pipeline import run_rag_stream
from core.reranker import RerankOutcome


def _settings(**overrides):
    values = {
        "top_k": 5,
        "rerank_enabled": True,
        "chat_model": "test-chat",
        "temperature": 0,
        "max_tokens": 128,
        "llm_request_timeout_seconds": 10,
        "llm_max_attempts": 1,
        "llm_retry_base_delay_seconds": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


async def _empty_stream():
    if False:  # pragma: no cover - 保持该函数为异步生成器
        yield None


class _FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _empty_stream()


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def with_options(self, **_kwargs):
        return self


def _candidate(*, content: str, filename: str = "文档.md", score: float = 0.02):
    return {
        "id": uuid.uuid4(),
        "doc_id": uuid.uuid4(),
        "content": content,
        "filename": filename,
        "score": score,
        "doc_tags": [],
    }


def _event_payloads(chunks: list[str]) -> list[dict]:
    return [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: ")
    ]


def _search_event(chunks: list[str]) -> dict:
    return next(item for item in _event_payloads(chunks) if item["type"] == "search_results")


class RagPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        *,
        question: str = "云枢默认密码怎么修改",
        intent: dict,
        results: list[dict],
        rerank_outcome: RerankOutcome | None = None,
        rerank_enabled: bool = True,
    ) -> tuple[list[str], AsyncMock, _FakeClient]:
        search = AsyncMock(return_value=results)
        fake_client = _FakeClient()
        if rerank_outcome is None:
            rerank_outcome = RerankOutcome(results=results, succeeded=False, error="disabled")

        async def build_context(_db, selected):
            return "\n".join(item["content"] for item in selected)

        with (
            patch("core.rag_pipeline.get_settings", return_value=_settings()),
            patch("core.rag_pipeline.hybrid_search", new=search),
            patch("core.rag_pipeline.rerank_with_status", new=AsyncMock(return_value=rerank_outcome)),
            patch("core.rag_pipeline._build_context", new=build_context),
            patch("core.rag_pipeline.get_client", return_value=fake_client),
        ):
            chunks = [
                chunk
                async for chunk in run_rag_stream(
                    question=question,
                    kb_ids=[uuid.uuid4()],
                    search_config={"top_k": 5, "rerank": rerank_enabled},
                    conversation_id="test-conversation",
                    db=SimpleNamespace(),
                    intent=intent,
                )
            ]
        return chunks, search, fake_client

    async def test_explicit_need_retrieval_overrides_legacy_chat_action(self) -> None:
        result = _candidate(content="云枢 defaultPwd 配置")
        chunks, search, _client = await self._run(
            intent={
                "action": "chat",
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "retrieval_guard",
            },
            results=[result],
            rerank_outcome=RerankOutcome(results=[{**result, "score": 0.9}], succeeded=True),
        )

        search.assert_awaited_once()
        event = _search_event(chunks)
        self.assertTrue(event["retrieval_executed"])
        self.assertEqual(event["evidence_status"], "hit")
        self.assertEqual(event["decision_reason"], "retrieval_guard")

    async def test_explicit_skip_overrides_legacy_retrieve_action(self) -> None:
        chunks, search, _client = await self._run(
            question="你好",
            intent={
                "action": "retrieve",
                "response_mode": "general_chat",
                "retrieval_policy": "skip",
                "need_retrieval": False,
                "decision_reason": "exact_greeting",
            },
            results=[],
        )

        search.assert_not_awaited()
        event = _search_event(chunks)
        self.assertFalse(event["retrieval_executed"])
        self.assertEqual(event["evidence_status"], "skipped")
        self.assertEqual(event["total"], 0)

    async def test_required_retrieval_without_candidates_is_no_hit(self) -> None:
        chunks, _search, client = await self._run(
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[],
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "no_hit")
        prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("知识库中未找到相关内容", prompt)

    async def test_optional_unverified_candidates_fall_back_without_not_found_message(self) -> None:
        unrelated = _candidate(content="员工食堂本周菜单", filename="后勤通知.md")
        chunks, _search, client = await self._run(
            question="给我讲一个笑话",
            intent={
                "response_mode": "general_chat",
                "retrieval_policy": "optional",
                "need_retrieval": True,
                "decision_reason": "selected_kb_optional",
            },
            results=[unrelated],
            rerank_enabled=False,
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(event["results"], [])
        prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertNotIn("知识库中未找到相关内容", prompt)
        self.assertIn("专业的助手", prompt)

    async def test_writing_mode_can_use_retrieved_evidence(self) -> None:
        result = _candidate(content="云枢默认密码通过 defaultPwd 配置。", filename="云枢配置.md")
        ranked = {**result, "score": 0.95}
        chunks, _search, client = await self._run(
            question="根据云枢文档整理一份默认密码修改说明",
            intent={
                "response_mode": "writing",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "knowledge_based_writing",
            },
            results=[result],
            rerank_outcome=RerankOutcome(results=[ranked], succeeded=True),
        )

        self.assertEqual(_search_event(chunks)["evidence_status"], "hit")
        prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("基于企业知识库资料", prompt)
        self.assertIn("不可信参考资料", prompt)
        self.assertIn("defaultPwd", prompt)

    async def test_rerank_failure_does_not_clear_required_rrf_candidates(self) -> None:
        raw = _candidate(content="云枢 defaultPwd: Authine@123456", score=0.02)
        chunks, _search, _client = await self._run(
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[raw],
            rerank_outcome=RerankOutcome(
                results=[raw],
                succeeded=False,
                error="APITimeoutError",
            ),
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "unverified")
        self.assertEqual(event["total"], 1)
        self.assertEqual(event["results"][0]["score"], 0.02)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
