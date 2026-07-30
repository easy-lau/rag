import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.rag_pipeline import (
    _build_context,
    _merge_retrieval_candidates,
    _select_verified_evidence,
    run_rag_stream,
)
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
        standalone_query: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        carryover_sources: list[dict] | None = None,
        is_followup: bool = False,
    ) -> tuple[list[str], AsyncMock, _FakeClient]:
        search = AsyncMock(return_value=results)
        fake_client = _FakeClient()
        if rerank_outcome is None:
            rerank_outcome = RerankOutcome(results=results, succeeded=False, error="disabled")

        async def build_context(_db, selected, **_kwargs):
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
                    standalone_query=standalone_query,
                    conversation_history=conversation_history,
                    carryover_sources=carryover_sources,
                    is_followup=is_followup,
                )
            ]
        return chunks, search, fake_client

    async def test_explicit_need_retrieval_overrides_legacy_chat_action(self) -> None:
        result = _candidate(
            content="云枢 defaultPwd 配置",
            filename="云枢配置.md",
        )
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
        self.assertEqual(event["displayed_result_count"], 1)
        self.assertEqual(event["context_evidence_count"], 1)
        self.assertEqual(event["hit_count"], 1)
        self.assertEqual(len(event["answer_sources"]), 1)
        self.assertEqual(event["answer_sources"][0]["content"], result["content"])

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
        self.assertEqual(event["answer_sources"], [])
        self.assertEqual(event["context_evidence_count"], 0)
        self.assertEqual(event["hit_count"], 0)
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
        messages = client.completions.calls[0]["messages"]
        system_prompt = messages[0]["content"]
        context_message = messages[1]["content"]
        self.assertIn("基于企业知识库资料", system_prompt)
        self.assertIn("不可信参考资料", system_prompt)
        self.assertIn("defaultPwd", context_message)

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

    async def test_only_old_versions_are_related_even_when_rerank_is_disabled(self) -> None:
        old_results = [
            _candidate(
                filename=f"云枢{version}配置",
                content=f"产品版本：云枢{version}\nerror_reply_same: true",
                score=0.99,
            )
            for version in ("6", "7")
        ]
        chunks, _search, client = await self._run(
            question="解决登录用户名枚举 要配置什么 我是云枢8.6",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=old_results,
            rerank_enabled=False,
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "version_mismatch")
        self.assertEqual(event["direct_evidence_count"], 0)
        self.assertEqual(event["related_reference_count"], 2)
        self.assertTrue(all(item["evidence_role"] == "related" for item in event["results"]))
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("没有目标版本的直接证据", system_prompt)

    async def test_rerank_failure_still_applies_version_constraint(self) -> None:
        old = _candidate(
            filename="云枢7配置",
            content="产品版本：云枢7\nerror_reply_same1: true",
            score=0.03,
        )
        chunks, _search, _client = await self._run(
            question="我是云枢8.6，解决用户名枚举要配置什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[old],
            rerank_outcome=RerankOutcome(
                results=[old],
                succeeded=False,
                error="APITimeoutError",
            ),
        )
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "version_mismatch")
        self.assertEqual(event["results"][0]["constraint_status"], "mismatch")
        self.assertEqual(event["results"][0]["evidence_role"], "related")

    async def test_exact_version_beats_higher_scored_old_versions(self) -> None:
        old = _candidate(filename="云枢7配置", content="产品版本：云枢7", score=0.99)
        exact = _candidate(filename="云枢8.6配置", content="产品版本：云枢8.6", score=0.1)
        malicious_ranked = [
            {
                **old,
                "topic_relevance": 0.99,
                "answer_support": 0.99,
                "constraint_status": "exact",
                "evidence_role": "direct",
                "rerank_status": "verified",
            },
            {
                **exact,
                "topic_relevance": 0.8,
                "answer_support": 0.8,
                "constraint_status": "exact",
                "evidence_role": "direct",
                "rerank_status": "verified",
            },
        ]
        chunks, _search, client = await self._run(
            question="云枢8.6登录安全怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[old, exact],
            rerank_outcome=RerankOutcome(results=malicious_ranked, succeeded=True),
        )
        event = _search_event(chunks)
        self.assertEqual(event["results"][0]["filename"], "云枢8.6配置")
        self.assertEqual(event["direct_evidence_count"], 1)
        context_message = client.completions.calls[0]["messages"][1]["content"]
        self.assertIn("产品版本：云枢8.6", context_message)
        self.assertNotIn("产品版本：云枢7", context_message)

    async def test_negated_compatibility_cannot_become_direct_context(self) -> None:
        old = _candidate(
            filename="云枢7配置.md",
            content="产品版本：云枢7\n本参数不再兼容云枢8.6",
            score=0.99,
        )
        # 模拟重排模型错误地把否定句判成 direct；流水线的确定性门控必须覆盖它。
        malicious = {
            **old,
            "topic_relevance": 0.99,
            "answer_support": 0.99,
            "constraint_status": "compatible",
            "evidence_role": "direct",
            "rerank_status": "verified",
        }
        chunks, _search, client = await self._run(
            question="我是云枢8.6，登录安全怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[old],
            rerank_outcome=RerankOutcome(results=[malicious], succeeded=True),
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "version_mismatch")
        self.assertEqual(event["direct_evidence_count"], 0)
        self.assertEqual(event["results"][0]["constraint_status"], "mismatch")
        self.assertEqual(event["results"][0]["evidence_role"], "related")
        prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("没有目标版本的直接证据", prompt)

    async def test_hard_constraint_disables_whole_document_expansion(self) -> None:
        result = _candidate(filename="云枢8.6配置", content="命中片段")
        fake_db = SimpleNamespace()
        with patch(
            "core.rag_pipeline._fetch_doc_text",
            new=AsyncMock(return_value="命中片段\n\n未评估的其它版本章节"),
        ):
            context = await _build_context(
                fake_db,
                [result],
                allow_whole_document=False,
            )
        self.assertIn("命中片段", context)
        self.assertNotIn("未评估的其它版本章节", context)

    def test_mismatch_direct_is_defensively_downgraded_and_truncation_is_counted(self) -> None:
        mismatch = {
            **_candidate(filename="云枢7配置", content="旧版本"),
            "topic_relevance": 0.99,
            "answer_support": 0.99,
            "constraint_status": "mismatch",
            "query_has_constraint": True,
            "query_has_hard_constraint": True,
            "evidence_role": "direct",
        }
        selected = _select_verified_evidence([mismatch], 5)
        self.assertEqual(selected[2], "version_mismatch")
        self.assertEqual(selected[0][0]["evidence_role"], "related")

        many = [
            {
                **_candidate(content=f"direct-{index}"),
                "topic_relevance": 0.9,
                "answer_support": 0.9,
                "constraint_status": "neutral",
                "evidence_role": "direct",
            }
            for index in range(10)
        ]
        selected = _select_verified_evidence(many, 3)
        self.assertEqual(len(selected[0]), 3)
        self.assertEqual(selected[5], 7)
        self.assertEqual(selected[7], 7)

    def test_direct_evidence_must_pass_topic_and_answer_support_thresholds(self) -> None:
        misleading = {
            **_candidate(content="与问题主题无关"),
            "topic_relevance": 0.0,
            "answer_support": 0.99,
            "constraint_status": "neutral",
            "evidence_role": "direct",
        }

        selected = _select_verified_evidence([misleading], 5)

        self.assertEqual(selected[2], "no_hit")
        self.assertEqual(selected[0], [])
        self.assertEqual(selected[1], [])

    def test_product_only_constraint_mismatch_is_partial_not_version_mismatch(self) -> None:
        mismatch = {
            **_candidate(content="其他产品配置"),
            "topic_relevance": 0.9,
            "answer_support": 0.9,
            "constraint_status": "mismatch",
            "query_has_constraint": True,
            "query_has_hard_constraint": False,
            "evidence_role": "related",
        }

        selected = _select_verified_evidence([mismatch], 5)

        self.assertEqual(selected[2], "partial")

    def test_low_answer_support_is_rejected_from_display_and_generation(self) -> None:
        for support in (0.0, 0.09):
            with self.subTest(support=support):
                related_without_support = {
                    **_candidate(content="云枢7配置 › 二、问题描述：无"),
                    "topic_relevance": 0.92,
                    "answer_support": support,
                    "constraint_status": "neutral",
                    "evidence_role": "related",
                }

                selected = _select_verified_evidence([related_without_support], 5)

                self.assertEqual(selected[0], [])
                self.assertEqual(selected[1], [])
                self.assertEqual(selected[2], "no_hit")
                self.assertEqual(selected[6], 1)

    async def test_no_hit_related_candidates_stay_in_results_not_answer_sources(self) -> None:
        useful_raw = _candidate(content="云枢7包含登录失败锁定参数，但未解释401")
        placeholder_raw = _candidate(content="云枢7配置 › 原因分析：无")
        useful_related = {
            **useful_raw,
            "topic_relevance": 0.92,
            "answer_support": 0.2,
            "constraint_status": "neutral",
            "evidence_role": "related",
            "rerank_status": "verified",
        }
        placeholder_without_support = {
            **placeholder_raw,
            "topic_relevance": 0.92,
            "answer_support": 0.0,
            "constraint_status": "neutral",
            "evidence_role": "related",
            "rerank_status": "verified",
        }

        chunks, _search, client = await self._run(
            question="我登录后报401最可能是什么原因",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            results=[useful_raw, placeholder_raw],
            rerank_outcome=RerankOutcome(
                results=[useful_related, placeholder_without_support],
                succeeded=True,
            ),
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(len(event["results"]), 1)
        self.assertIn("锁定参数", event["results"][0]["content"])
        self.assertNotIn("原因分析：无", event["results"][0]["content"])
        self.assertEqual(event["displayed_result_count"], 1)
        self.assertEqual(event["related_reference_count"], 1)
        self.assertEqual(event["answer_sources"], [])
        self.assertEqual(event["context_evidence_count"], 0)
        self.assertEqual(event["hit_count"], 0)
        all_prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("原因分析：无", all_prompt_content)

    async def test_partial_related_context_is_exposed_as_answer_source_but_not_direct_hit(self) -> None:
        raw = _candidate(content="旧版本登录安全配置")
        supported_related = {
            **raw,
            "topic_relevance": 0.9,
            "answer_support": 0.8,
            "constraint_status": "neutral",
            "evidence_role": "related",
            "rerank_status": "verified",
        }

        chunks, _search, client = await self._run(
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            results=[raw],
            rerank_outcome=RerankOutcome(
                results=[supported_related],
                succeeded=True,
            ),
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "partial")
        self.assertEqual(len(event["results"]), 1)
        self.assertEqual(len(event["answer_sources"]), 1)
        self.assertEqual(event["context_evidence_count"], 1)
        self.assertEqual(event["direct_evidence_count"], 0)
        self.assertEqual(event["hit_count"], 0)
        all_prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("旧版本登录安全配置", all_prompt_content)

    def test_optional_policy_keeps_supported_related_as_display_only(self) -> None:
        related = {
            **_candidate(content="主题相近但不能直接支撑当前问题"),
            "topic_relevance": 0.92,
            "answer_support": 0.88,
            "constraint_status": "neutral",
            "evidence_role": "related",
        }

        selected = _select_verified_evidence(
            [related],
            5,
            allow_related_context=False,
        )

        self.assertEqual(len(selected[0]), 1)
        self.assertEqual(selected[1], [])
        self.assertEqual(selected[2], "partial")

    def test_carryover_and_fresh_candidate_are_deduplicated(self) -> None:
        chunk_id = uuid.uuid4()
        carryover = {
            **_candidate(content="旧快照"),
            "id": chunk_id,
            "score": 0.0,
            "candidate_origin": "carryover_previous_turn",
        }
        fresh = {
            **carryover,
            "content": "数据库当前片段",
            "score": 0.82,
            "active_channels": ["vector"],
        }

        merged = _merge_retrieval_candidates([fresh], [carryover])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["content"], "数据库当前片段")
        self.assertEqual(merged[0]["score"], 0.82)
        self.assertEqual(merged[0]["candidate_origin"], "carryover_and_current_retrieval")
        self.assertEqual(merged[0]["active_channels"], ["vector", "carryover"])

    async def test_followup_uses_standalone_query_for_retrieval_and_rerank(self) -> None:
        result = _candidate(content="error_reply_same: true")
        standalone = (
            "围绕云枢 8.6 登录用户名枚举，配置项 error_reply_same 有什么影响"
        )
        verified = {
            **result,
            "topic_relevance": 0.9,
            "answer_support": 0.9,
            "constraint_status": "unknown",
            "evidence_role": "related",
            "rerank_status": "verified",
        }
        chunks, search, client = await self._run(
            question="这些配置会对程序有什么影响",
            standalone_query=standalone,
            conversation_history=[
                {"role": "user", "content": "云枢 8.6 怎么解决用户名枚举"},
                {"role": "assistant", "content": "资料中提到了旧版本配置。"},
            ],
            is_followup=True,
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[result],
            rerank_outcome=RerankOutcome(results=[verified], succeeded=True),
        )

        self.assertEqual(search.await_args.kwargs["query"], standalone)
        event = _search_event(chunks)
        self.assertTrue(event["is_followup"])
        messages = client.completions.calls[0]["messages"]
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("云枢 8.6", messages[1]["content"])
        self.assertEqual(messages[-1]["content"], "这些配置会对程序有什么影响")

    async def test_zero_support_carryover_does_not_bypass_gate_when_rerank_is_off(self) -> None:
        carryover = {
            **_candidate(content="问题描述：无"),
            "candidate_origin": "carryover_previous_turn",
            "carryover_previous_support": 0.0,
            "score": 0.0,
        }
        chunks, _search, client = await self._run(
            question="这些配置会对程序有什么影响",
            standalone_query="云枢登录安全配置有什么影响",
            carryover_sources=[carryover],
            is_followup=True,
            rerank_enabled=False,
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
        self.assertEqual(event["context_evidence_count"], 0)
        all_prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("问题描述：无", all_prompt_content)

    async def test_optional_verified_related_does_not_hijack_general_chat(self) -> None:
        result = _candidate(content="云枢旧版本相近资料，不应进入通用聊天上下文")
        verified = {
            **result,
            "topic_relevance": 0.9,
            "answer_support": 0.9,
            "constraint_status": "neutral",
            "evidence_role": "related",
            "rerank_status": "verified",
        }
        chunks, _search, client = await self._run(
            question="给我一个部署建议",
            intent={
                "response_mode": "general_chat",
                "retrieval_policy": "optional",
                "need_retrieval": True,
                "decision_reason": "selected_knowledge_context",
            },
            results=[result],
            rerank_outcome=RerankOutcome(results=[verified], succeeded=True),
        )

        event = _search_event(chunks)
        self.assertEqual(event["related_reference_count"], 1)
        self.assertEqual(event["context_evidence_count"], 0)
        self.assertEqual(event["answer_sources"], [])
        self.assertEqual(event["hit_count"], 0)
        all_prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("云枢旧版本相近资料", all_prompt_content)

    async def test_trace_records_reproducible_algorithm_and_generation_config(self) -> None:
        result = _candidate(content="cloudpivot.organization.login.error_reply_same: true")
        verified = {
            **result,
            "topic_relevance": 0.98,
            "answer_support": 0.96,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
        }

        with patch("core.rag_pipeline.trace_event") as trace_mock:
            await self._run(
                question="云枢中如何配置登录用户名枚举",
                intent={
                    "response_mode": "grounded_qa",
                    "retrieval_policy": "required",
                    "need_retrieval": True,
                    "decision_reason": "enterprise_operation_guard",
                },
                results=[result],
                rerank_outcome=RerankOutcome(results=[verified], succeeded=True),
            )

        events = {
            call.args[0]: call.kwargs
            for call in trace_mock.call_args_list
            if call.args
        }
        plan = events["retrieval.plan"]
        self.assertEqual(plan["retrieval_algorithm"], "vector_fts_trigram_rrf")
        self.assertEqual(plan["rrf_k"], 60)
        self.assertEqual(plan["candidate_chunks_per_document"], 3)
        self.assertEqual(plan["rerank_candidate_multiplier"], 3)

        rerank = events["rerank.completed"]
        self.assertEqual(rerank["model"], "test-chat")
        self.assertTrue(rerank["prompt_version"])
        self.assertEqual(rerank["answer_support_threshold"], 0.3)

        evidence = events["evidence.selection"]
        self.assertEqual(evidence["topic_relevance_threshold"], 0.3)
        self.assertEqual(evidence["answer_support_threshold"], 0.3)
        self.assertEqual(evidence["related_reference_min_support"], 0.1)
        self.assertIn("相近资料支撑阈值 0.1", evidence["mode"])
        self.assertEqual(evidence["displayed_result_count"], 1)
        self.assertEqual(evidence["context_evidence_count"], 1)
        self.assertEqual(evidence["hit_count"], 1)
        self.assertEqual(len(evidence["answer_sources"]), 1)

        generation = events["generation.context"]
        self.assertEqual(generation["model"], "test-chat")
        self.assertEqual(generation["request_timeout_seconds"], 10)
        self.assertEqual(generation["max_attempts"], 1)
        self.assertEqual(len(generation["system_prompt_sha256"]), 64)
        self.assertNotIn("system_prompt", generation)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
