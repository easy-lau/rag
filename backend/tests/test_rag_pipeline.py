import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.rag_pipeline import (
    _apply_joint_context_budget,
    _bounded_initial_expansion_candidates,
    _build_context,
    _generation_coverage_payload,
    _knowledge_context_message,
    _merge_retrieval_candidates,
    _select_verified_evidence,
    run_rag_stream,
)
from core.evidence_expansion import ExpansionOutcome
from core.reranker import AnswerRequirement, ExpansionPlan, RerankOutcome


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


def _expanded_outcome(initial: list[dict], added: list[dict]) -> ExpansionOutcome:
    return ExpansionOutcome(
        candidates=[*initial, *added],
        seed_candidates=initial[:1],
        scoped_candidates=added,
        structural_candidates=[],
        counts_by_origin={"global_retrieval": len(initial), "document_scoped": len(added)},
        added_candidate_count=len(added),
        added_chars=sum(len(str(item.get("content") or "")) for item in added),
        deduplicated_count=0,
        budget_dropped_count=0,
        expanded=True,
    )


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
        expansion_outcome: ExpansionOutcome | None = None,
        joint_outcome: RerankOutcome | None = None,
        expansion_mock: AsyncMock | None = None,
        joint_mock: AsyncMock | None = None,
    ) -> tuple[list[str], AsyncMock, _FakeClient]:
        search = AsyncMock(return_value=results)
        fake_client = _FakeClient()
        if rerank_outcome is None:
            rerank_outcome = RerankOutcome(results=results, succeeded=False, error="disabled")
        expansion_call = expansion_mock or AsyncMock(
            side_effect=AssertionError("当前场景不应触发文档内证据扩展")
        )
        if expansion_outcome is not None:
            expansion_call.return_value = expansion_outcome
            expansion_call.side_effect = None
        joint_call = joint_mock or AsyncMock(
            side_effect=AssertionError("当前场景不应触发联合重排")
        )
        if joint_outcome is not None:
            joint_call.return_value = joint_outcome
            joint_call.side_effect = None

        async def build_context(_db, selected, **_kwargs):
            return "\n".join(item["content"] for item in selected)

        with (
            patch("core.rag_pipeline.get_settings", return_value=_settings()),
            patch("core.rag_pipeline.hybrid_search", new=search),
            patch("core.rag_pipeline.rerank_with_status", new=AsyncMock(return_value=rerank_outcome)),
            patch("core.rag_pipeline.expand_evidence_candidates", new=expansion_call),
            patch("core.rag_pipeline.joint_rerank_with_coverage", new=joint_call),
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

    def test_expansion_pool_keeps_target_document_and_real_competitor(self) -> None:
        target_doc = uuid.uuid4()
        competitor_doc = uuid.uuid4()
        unrelated_doc = uuid.uuid4()
        target_candidates = [
            {
                **_candidate(content=f"目标文档片段 {index}"),
                "doc_id": target_doc,
                "rerank_candidate_index": index,
                "topic_relevance": 0.8,
                "answer_support": 0.1,
                "constraint_status": "neutral",
                "evidence_role": "related",
                "contribution_role": "background",
            }
            for index in range(1, 4)
        ]
        competitor = {
            **_candidate(content="另一份真实相关的竞争证据"),
            "doc_id": competitor_doc,
            "rerank_candidate_index": 4,
            "topic_relevance": 0.9,
            "answer_support": 0.7,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "contribution_role": "standalone_answer",
        }
        unrelated = [
            {
                **_candidate(content=f"明显无关候选 {index}"),
                "doc_id": unrelated_doc,
                "rerank_candidate_index": index,
                "topic_relevance": 0.01,
                "answer_support": 0.0,
                "constraint_status": "neutral",
                "evidence_role": "irrelevant",
                "contribution_role": "irrelevant",
            }
            for index in range(5, 13)
        ]

        bounded = _bounded_initial_expansion_candidates(
            [*target_candidates, competitor, *unrelated],
            ExpansionPlan(
                needed=True,
                target_candidate_indexes=(1,),
                queries=("目标文档完整标准",),
            ),
        )

        self.assertEqual(len(bounded), 4)
        self.assertEqual(
            {item["doc_id"] for item in bounded},
            {target_doc, competitor_doc},
        )
        self.assertNotIn(unrelated_doc, {item["doc_id"] for item in bounded})

    async def test_missing_helpful_requirement_is_promoted_for_joint_coverage(self) -> None:
        document_id = uuid.uuid4()
        requirements = (
            AnswerRequirement("grade", "确定普通员工适用职级"),
            AnswerRequirement(
                "standard",
                "职级对应的交通住宿和补贴标准",
                importance="helpful",
                source="inferred",
            ),
        )
        mapping = {
            **_candidate(content="普通员工属于 D级", filename="公司出差管理标准.md"),
            "doc_id": document_id,
            "chunk_index": 2,
            "rerank_candidate_index": 1,
            "topic_relevance": 1.0,
            "answer_support": 0.9,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "standalone_answer",
            "supports_requirement_ids": ["grade"],
        }
        detail = {
            **_candidate(
                content="D级交通、住宿和补贴标准明细",
                filename="公司出差管理标准.md",
            ),
            "doc_id": document_id,
            "chunk_index": 3,
            "candidate_origin": "document_scoped",
        }
        selected = [
            {
                **candidate,
                "rerank_candidate_index": index,
                "topic_relevance": 0.98,
                "answer_support": 0.9,
                "constraint_status": "neutral",
                "evidence_role": "direct",
                "rerank_status": "verified_joint",
                "contribution_role": (
                    "bridge" if index == 1 else "complement"
                ),
                "supports_requirement_ids": (
                    ["grade"] if index == 1 else ["standard"]
                ),
                "jointly_selected": True,
                "evidence_set_id": "set_1",
                "joint_support_score": 0.93,
                "coverage_status": "complete",
            }
            for index, candidate in enumerate((mapping, detail), start=1)
        ]
        expansion_mock = AsyncMock()
        joint_mock = AsyncMock()

        chunks, _search, _client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[mapping],
            rerank_outcome=RerankOutcome(
                results=[mapping],
                succeeded=True,
                requirements=requirements,
                expansion_plan=ExpansionPlan(
                    needed=True,
                    target_candidate_indexes=(1,),
                    queries=("D级完整出差标准",),
                    missing_requirement_ids=("standard",),
                ),
            ),
            expansion_outcome=_expanded_outcome([mapping], [detail]),
            joint_outcome=RerankOutcome(
                results=selected,
                succeeded=True,
                requirements=requirements,
                coverage_status="complete",
                selected_evidence_set_id="set_1",
                selected_candidate_indexes=(1, 2),
                joint_support_score=0.93,
            ),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_awaited_once()
        joint_mock.assert_awaited_once()
        promoted = {item.id: item for item in joint_mock.await_args.args[2]}
        self.assertEqual(promoted["standard"].importance, "required")
        self.assertEqual(promoted["standard"].source, "explicit")
        self.assertEqual(_search_event(chunks)["coverage_status"], "complete")

    async def test_failed_initial_rerank_can_use_safe_document_expansion(self) -> None:
        document_id = uuid.uuid4()
        initial = [
            {
                **_candidate(
                    content=content,
                    filename="公司出差管理标准.md",
                ),
                "doc_id": document_id,
                "chunk_index": index,
                "active_channels": channels,
            }
            for index, content, channels in (
                (1, "普通员工属于 D级", ["vector", "trigram"]),
                (2, "出差审批和报销说明", ["trigram"]),
                (3, "差旅等级说明", ["vector"]),
            )
        ]
        detail = {
            **_candidate(
                content="D级住宿450/350/250元，餐饮100元，通讯50元，出差补贴100元",
                filename="公司出差管理标准.md",
            ),
            "doc_id": document_id,
            "chunk_index": 4,
            "candidate_origin": "document_scoped",
        }
        selected = [
            {
                **candidate,
                "rerank_candidate_index": index,
                "topic_relevance": 0.98,
                "answer_support": 0.9,
                "constraint_status": "neutral",
                "evidence_role": "direct",
                "rerank_status": "verified_joint",
                "contribution_role": (
                    "bridge" if index == 1 else "complement"
                ),
                "supports_requirement_ids": ["answer"],
                "jointly_selected": True,
                "evidence_set_id": "set_1",
                "joint_support_score": 0.92,
                "coverage_status": "complete",
            }
            for index, candidate in enumerate((initial[0], detail), start=1)
        ]
        expansion_mock = AsyncMock()
        joint_mock = AsyncMock()

        chunks, _search, client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=initial,
            rerank_outcome=RerankOutcome(
                results=initial,
                succeeded=False,
                error="InternalServerError: 500 websocket EOF",
            ),
            expansion_outcome=_expanded_outcome(initial, [detail]),
            joint_outcome=RerankOutcome(
                results=selected,
                succeeded=True,
                coverage_status="complete",
                selected_evidence_set_id="set_1",
                selected_candidate_indexes=(1, 2),
                joint_support_score=0.92,
            ),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_awaited_once()
        expansion_inputs = expansion_mock.await_args.kwargs["initial_candidates"]
        self.assertEqual(len(expansion_inputs), 3)
        self.assertEqual({item["doc_id"] for item in expansion_inputs}, {document_id})
        self.assertEqual(
            expansion_mock.await_args.kwargs["plan"].target_candidate_indexes,
            (1, 2, 3),
        )
        joint_mock.assert_awaited_once()
        self.assertEqual(
            tuple(item.id for item in joint_mock.await_args.args[2]),
            ("answer",),
        )
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "hit")
        self.assertEqual(event["coverage_status"], "complete")
        self.assertTrue(event["expansion_attempted"])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("住宿450/350/250元", prompt_content)

    async def test_failed_initial_rerank_vector_only_does_not_expand(self) -> None:
        document_id = uuid.uuid4()
        initial = [
            {
                **_candidate(content=f"同文档向量候选 {index}"),
                "doc_id": document_id,
                "active_channels": ["vector"],
            }
            for index in range(3)
        ]
        expansion_mock = AsyncMock(
            side_effect=AssertionError("纯向量召回不得触发失败后扩展")
        )
        joint_mock = AsyncMock(
            side_effect=AssertionError("纯向量召回不得触发联合重排")
        )

        chunks, _search, _client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=initial,
            rerank_outcome=RerankOutcome(
                results=initial,
                succeeded=False,
                error="InternalServerError: 500",
            ),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_not_awaited()
        joint_mock.assert_not_awaited()
        self.assertFalse(_search_event(chunks)["expansion_attempted"])

    async def test_failed_initial_rerank_without_dominant_document_does_not_expand(self) -> None:
        first_doc = uuid.uuid4()
        second_doc = uuid.uuid4()
        initial = [
            {
                **_candidate(content="第一文档词面命中"),
                "doc_id": first_doc,
                "active_channels": ["trigram"],
            },
            {
                **_candidate(content="第二文档候选"),
                "doc_id": second_doc,
                "active_channels": ["vector"],
            },
            {
                **_candidate(content="第一文档另一候选"),
                "doc_id": first_doc,
                "active_channels": ["vector"],
            },
        ]
        expansion_mock = AsyncMock(
            side_effect=AssertionError("前三条不属于同一文档时不得扩展")
        )
        joint_mock = AsyncMock(
            side_effect=AssertionError("前三条不属于同一文档时不得联合重排")
        )

        chunks, _search, _client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=initial,
            rerank_outcome=RerankOutcome(
                results=initial,
                succeeded=False,
                error="InternalServerError: 500",
            ),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_not_awaited()
        joint_mock.assert_not_awaited()
        self.assertFalse(_search_event(chunks)["expansion_attempted"])

    async def test_failed_initial_and_joint_rerank_discards_all_unverified_context(self) -> None:
        target_doc = uuid.uuid4()
        unrelated_doc = uuid.uuid4()
        dominant = [
            {
                **_candidate(
                    content=f"目标文档召回片段 {index}",
                    filename="公司出差管理标准.md",
                ),
                "doc_id": target_doc,
                "active_channels": ["trigram"] if index == 1 else ["vector"],
            }
            for index in range(1, 4)
        ]
        unrelated = {
            **_candidate(content="员工请假制度中的无关内容", filename="员工请假制度.md"),
            "doc_id": unrelated_doc,
            "active_channels": ["vector"],
        }
        detail = {
            **_candidate(content="未经联合验证的D级住宿标准"),
            "doc_id": target_doc,
            "candidate_origin": "document_scoped",
        }
        expansion_mock = AsyncMock()
        joint_mock = AsyncMock()

        chunks, _search, client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[*dominant, unrelated],
            rerank_outcome=RerankOutcome(
                results=[*dominant, unrelated],
                succeeded=False,
                error="InternalServerError: 500 websocket EOF",
            ),
            expansion_outcome=_expanded_outcome(dominant, [detail]),
            joint_outcome=RerankOutcome(
                results=[*dominant, detail],
                succeeded=False,
                error="TimeoutError",
                coverage_status="insufficient",
            ),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_awaited_once()
        expansion_inputs = expansion_mock.await_args.kwargs["initial_candidates"]
        self.assertEqual({item["doc_id"] for item in expansion_inputs}, {target_doc})
        joint_mock.assert_awaited_once()
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("员工请假制度中的无关内容", prompt_content)
        self.assertNotIn("未经联合验证的D级住宿标准", prompt_content)

    async def test_cross_chunk_bridge_expands_and_uses_complete_joint_evidence(self) -> None:
        document_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        requirements = (
            AnswerRequirement("grade", "确定普通员工适用职级"),
            AnswerRequirement("traffic", "交通标准"),
            AnswerRequirement("lodging", "住宿标准"),
            AnswerRequirement("allowance", "餐饮和其他补贴"),
        )
        mapping = {
            **_candidate(content="普通员工、专员属于 D级", filename="公司出差管理标准.md"),
            "doc_id": document_id,
            "kb_id": kb_id,
            "chunk_index": 2,
            "rerank_candidate_index": 1,
            "topic_relevance": 1.0,
            "answer_support": 0.55,
            "constraint_status": "neutral",
            "evidence_role": "related",
            "rerank_status": "verified",
            "contribution_role": "bridge",
            "supports_requirement_ids": ["grade"],
            "bridge_facts": [
                {"subject": "普通员工", "relation": "属于", "object": "D级"}
            ],
        }
        details = [
            {
                **_candidate(content=content, filename="公司出差管理标准.md"),
                "doc_id": document_id,
                "kb_id": kb_id,
                "chunk_index": index,
                "candidate_origin": "document_scoped",
            }
            for index, content in (
                (3, "D级：飞机经济舱；高铁/动车二等座；普通火车硬卧"),
                (6, "D级住宿：一线城市450元，二线350元，其他城市250元"),
                (7, "D级餐饮100元/天；通讯50元/天；出差补贴100元/天"),
            )
        ]
        requirement_ids = ("grade", "traffic", "lodging", "allowance")
        supports = (["grade"], ["traffic"], ["lodging"], ["allowance"])
        joint_results = []
        for index, (candidate, supported) in enumerate(
            zip([mapping, *details], supports),
            start=1,
        ):
            joint_results.append({
                **candidate,
                "rerank_candidate_index": index,
                "topic_relevance": 0.98,
                "answer_support": 0.9 if index > 1 else 0.55,
                "constraint_status": "neutral",
                "evidence_role": "direct",
                "rerank_status": "verified_joint",
                "joint_rerank_status": "verified",
                "contribution_role": "bridge" if index == 1 else "complement",
                "supports_requirement_ids": supported,
                "jointly_selected": True,
                "evidence_set_id": "set_1",
                "joint_support_score": 0.93,
                "coverage_status": "complete",
            })

        plan = ExpansionPlan(
            needed=True,
            target_candidate_indexes=(1,),
            queries=("D级交通住宿餐饮补贴标准",),
            missing_requirement_ids=("traffic", "lodging", "allowance"),
            reason="已找到职级映射，但缺少标准明细",
        )
        first_outcome = RerankOutcome(
            results=[mapping],
            succeeded=True,
            requirements=requirements,
            expansion_plan=plan,
        )
        final_outcome = RerankOutcome(
            results=joint_results,
            succeeded=True,
            requirements=requirements,
            coverage_status="complete",
            joint_support_score=0.93,
            selected_evidence_set_id="set_1",
            selected_candidate_indexes=(1, 2, 3, 4),
            missing_requirement_ids=(),
            model="test-chat",
            prompt_version="joint-test",
            elapsed_ms=12,
            candidate_count=4,
        )
        expansion_mock = AsyncMock()
        joint_mock = AsyncMock()

        chunks, _search, client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[mapping],
            rerank_outcome=first_outcome,
            expansion_outcome=_expanded_outcome([mapping], details),
            joint_outcome=final_outcome,
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_awaited_once()
        joint_mock.assert_awaited_once()
        self.assertEqual(joint_mock.await_args.args[0], "普通员工的出差标准是什么")
        self.assertEqual(
            tuple(item.id for item in joint_mock.await_args.args[2]),
            requirement_ids,
        )
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "hit")
        self.assertEqual(event["coverage_status"], "complete")
        self.assertTrue(event["expansion_attempted"])
        self.assertEqual(len(event["answer_sources"]), 4)
        context_message = client.completions.calls[0]["messages"][1]["content"]
        for expected in ("普通员工", "经济舱", "450元", "餐饮100元"):
            self.assertIn(expected, context_message)

    async def test_complete_single_chunk_uses_fast_path_without_expansion(self) -> None:
        requirement = AnswerRequirement("answer", "默认密码修改方法")
        result = {
            **_candidate(content="修改 defaultPwd 后重启服务"),
            "rerank_candidate_index": 1,
            "topic_relevance": 0.98,
            "answer_support": 0.95,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "standalone_answer",
            "supports_requirement_ids": ["answer"],
        }
        expansion_mock = AsyncMock()
        joint_mock = AsyncMock()
        chunks, _search, _client = await self._run(
            question="系统默认密码怎么修改",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[result],
            rerank_outcome=RerankOutcome(
                results=[result],
                succeeded=True,
                requirements=(requirement,),
                expansion_plan=ExpansionPlan(needed=False),
            ),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_not_awaited()
        joint_mock.assert_not_awaited()
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "hit")
        self.assertFalse(event["expansion_attempted"])

    async def test_version_mismatch_cannot_seed_document_expansion(self) -> None:
        old = {
            **_candidate(
                filename="云枢7配置.md",
                content="云枢7：error_reply_same1: true",
            ),
            "rerank_candidate_index": 1,
            "topic_relevance": 0.98,
            "answer_support": 0.9,
            "constraint_status": "mismatch",
            "evidence_role": "related",
            "rerank_status": "verified",
            "contribution_role": "bridge",
            "supports_requirement_ids": ["answer"],
        }
        expansion_mock = AsyncMock()
        joint_mock = AsyncMock()
        chunks, _search, _client = await self._run(
            question="我是云枢8.6，用户名枚举怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[old],
            rerank_outcome=RerankOutcome(
                results=[old],
                succeeded=True,
                requirements=(AnswerRequirement("answer", "目标版本配置"),),
                expansion_plan=ExpansionPlan(
                    needed=True,
                    target_candidate_indexes=(1,),
                    queries=("云枢7用户名枚举配置",),
                    missing_requirement_ids=("answer",),
                ),
            ),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_not_awaited()
        joint_mock.assert_not_awaited()
        self.assertEqual(_search_event(chunks)["evidence_status"], "version_mismatch")

    async def test_joint_failure_discards_new_chunks_and_fails_closed(self) -> None:
        document_id = uuid.uuid4()
        requirement = AnswerRequirement("answer", "员工适用的完整出差标准")
        mapping = {
            **_candidate(content="普通员工属于D级"),
            "doc_id": document_id,
            "chunk_index": 2,
            "rerank_candidate_index": 1,
            "topic_relevance": 1.0,
            "answer_support": 0.5,
            "constraint_status": "neutral",
            # 即使首轮模型把桥接片段误标成 direct，补检失败后也不能把它当作
            # 完整答案依据。这是 fail-closed 回归测试的关键前提。
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "bridge",
            "supports_requirement_ids": ["answer"],
            "bridge_facts": [
                {"subject": "普通员工", "relation": "属于", "object": "D级"}
            ],
        }
        unverified_detail = {
            **_candidate(content="未经联合验证的D级标准"),
            "doc_id": document_id,
            "chunk_index": 3,
            "candidate_origin": "document_scoped",
        }
        chunks, _search, client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[mapping],
            rerank_outcome=RerankOutcome(
                results=[mapping],
                succeeded=True,
                requirements=(requirement,),
                expansion_plan=ExpansionPlan(
                    needed=True,
                    target_candidate_indexes=(1,),
                    queries=("D级出差标准",),
                    missing_requirement_ids=("answer",),
                ),
            ),
            expansion_outcome=_expanded_outcome([mapping], [unverified_detail]),
            joint_outcome=RerankOutcome(
                results=[mapping, unverified_detail],
                succeeded=False,
                error="TimeoutError",
                coverage_status="insufficient",
            ),
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("未经联合验证", prompt_content)

    async def test_expansion_exception_with_direct_bridge_fails_closed(self) -> None:
        mapping = {
            **_candidate(content="普通员工属于D级"),
            "chunk_index": 2,
            "rerank_candidate_index": 1,
            "topic_relevance": 1.0,
            "answer_support": 0.9,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "bridge",
            "supports_requirement_ids": ["answer"],
        }
        expansion_mock = AsyncMock(side_effect=RuntimeError("scoped search failed"))
        joint_mock = AsyncMock()
        chunks, _search, client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[mapping],
            rerank_outcome=RerankOutcome(
                results=[mapping],
                succeeded=True,
                requirements=(AnswerRequirement("answer", "完整出差标准"),),
                expansion_plan=ExpansionPlan(
                    needed=True,
                    target_candidate_indexes=(1,),
                    queries=("D级出差标准",),
                    missing_requirement_ids=("answer",),
                ),
            ),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
        )

        expansion_mock.assert_awaited_once()
        joint_mock.assert_not_awaited()
        event = _search_event(chunks)
        self.assertEqual(event["coverage_status"], "insufficient")
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("普通员工属于D级", prompt_content)

    async def test_joint_exception_with_direct_bridge_fails_closed(self) -> None:
        mapping = {
            **_candidate(content="普通员工属于D级"),
            "chunk_index": 2,
            "rerank_candidate_index": 1,
            "topic_relevance": 1.0,
            "answer_support": 0.9,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "bridge",
            "supports_requirement_ids": ["answer"],
        }
        unverified_detail = {
            **_candidate(content="未经联合验证的D级住宿标准"),
            "doc_id": mapping["doc_id"],
            "chunk_index": 3,
        }
        joint_mock = AsyncMock(side_effect=RuntimeError("joint model failed"))
        chunks, _search, client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[mapping],
            rerank_outcome=RerankOutcome(
                results=[mapping],
                succeeded=True,
                requirements=(AnswerRequirement("answer", "完整出差标准"),),
                expansion_plan=ExpansionPlan(
                    needed=True,
                    target_candidate_indexes=(1,),
                    queries=("D级出差标准",),
                    missing_requirement_ids=("answer",),
                ),
            ),
            expansion_outcome=_expanded_outcome([mapping], [unverified_detail]),
            joint_mock=joint_mock,
        )

        joint_mock.assert_awaited_once()
        event = _search_event(chunks)
        self.assertEqual(event["coverage_status"], "insufficient")
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("普通员工属于D级", prompt_content)
        self.assertNotIn("未经联合验证", prompt_content)

    async def test_no_new_expansion_candidates_with_direct_bridge_fails_closed(self) -> None:
        mapping = {
            **_candidate(content="普通员工属于D级"),
            "chunk_index": 2,
            "rerank_candidate_index": 1,
            "topic_relevance": 1.0,
            "answer_support": 0.9,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "bridge",
            "supports_requirement_ids": ["answer"],
        }
        empty_expansion = ExpansionOutcome(
            candidates=[mapping],
            seed_candidates=[mapping],
            scoped_candidates=[],
            structural_candidates=[],
            counts_by_origin={"global_retrieval": 1},
            added_candidate_count=0,
            added_chars=0,
            deduplicated_count=0,
            budget_dropped_count=0,
            expanded=False,
        )
        joint_mock = AsyncMock()
        chunks, _search, client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[mapping],
            rerank_outcome=RerankOutcome(
                results=[mapping],
                succeeded=True,
                requirements=(AnswerRequirement("answer", "完整出差标准"),),
                expansion_plan=ExpansionPlan(
                    needed=True,
                    target_candidate_indexes=(1,),
                    queries=("D级出差标准",),
                    missing_requirement_ids=("answer",),
                ),
            ),
            expansion_outcome=empty_expansion,
            joint_mock=joint_mock,
        )

        joint_mock.assert_not_awaited()
        event = _search_event(chunks)
        self.assertEqual(event["coverage_status"], "insufficient")
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("普通员工属于D级", prompt_content)

    def test_context_budget_downgrades_complete_when_required_facet_is_dropped(self) -> None:
        requirements = (
            AnswerRequirement("mapping", "人员到等级映射"),
            AnswerRequirement("standard", "等级对应标准"),
        )
        first = {
            **_candidate(content="甲" * 16000),
            "jointly_selected": True,
            "supports_requirement_ids": ["mapping"],
            "contribution_role": "bridge",
        }
        second = {
            **_candidate(content="标准明细"),
            "jointly_selected": True,
            "supports_requirement_ids": ["standard"],
            "contribution_role": "complement",
        }

        bounded, status, missing, dropped, used_chars = _apply_joint_context_budget(
            [first, second],
            "complete",
            requirements,
        )

        self.assertEqual(status, "partial")
        self.assertEqual(missing, ("standard",))
        self.assertEqual(dropped, 1)
        self.assertEqual(used_chars, 16000)
        self.assertEqual(sum(bool(item.get("jointly_selected")) for item in bounded), 1)

    def test_partial_coverage_tells_generation_which_generic_requirement_is_missing(self) -> None:
        requirements = (
            AnswerRequirement("tier", "确定项目采用的服务等级"),
            AnswerRequirement("sla", "给出该等级的响应时限"),
        )

        coverage = _generation_coverage_payload(
            "partial",
            requirements,
            ("sla",),
        )
        message = _knowledge_context_message(
            "项目星河采用银牌服务。",
            evidence_coverage=coverage,
        )
        payload = json.loads(message.split("\n", 1)[1])

        self.assertTrue(payload["untrusted"])
        self.assertEqual(payload["evidence_coverage"]["status"], "partial")
        self.assertEqual(
            payload["evidence_coverage"]["missing_requirements"],
            [{"id": "sla", "description": "给出该等级的响应时限"}],
        )
        self.assertEqual(
            len(payload["evidence_coverage"]["required_requirements"]),
            2,
        )

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
