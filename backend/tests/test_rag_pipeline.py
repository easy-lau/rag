import json
import unittest
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.rag_pipeline import (
    _apply_joint_context_budget,
    _bounded_initial_expansion_candidates,
    _build_context,
    _clarification_trace_payload,
    _generation_coverage_payload,
    _fallback_to_initial_verified_evidence,
    _knowledge_context_message,
    _merge_retrieval_candidates,
    _normalize_evidence_scope_filter,
    _restrict_candidates_to_scope,
    _scope_anchor_coverage,
    _resolve_document_expansion_plan,
    _rescue_missing_joint_evidence,
    _select_unverified_evidence,
    _select_verified_evidence,
    annotate_deterministic_constraints,
    run_rag_stream,
)
from core.evidence_ambiguity import EvidenceAmbiguityDecision, EvidenceScopeChoice
from core.evidence_expansion import ExpansionOutcome
from core.query_route_compiler import (
    RouteCategoryPolicy,
    RouteCompilerConfig,
    TaskContractDispatchError,
    compile_rag_task_contract,
)
from core.query_route_contract import parse_rag_route_decision
from core.query_constraints import (
    evaluate_candidate_constraints,
    extract_query_constraints,
    inherit_document_constraint_metadata,
)
from core.reranker import AnswerRequirement, ExpansionPlan, RerankOutcome


def _settings(**overrides):
    values = {
        "top_k": 5,
        "rerank_enabled": True,
        "chat_model": "test-chat",
        "rerank_model": "",
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


class EvidenceScopeSliceFilterTests(unittest.TestCase):
    def test_selected_section_drops_sibling_section_in_same_document(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        selected_chunk_id = uuid.uuid4()
        sibling_chunk_id = uuid.uuid4()
        payload = {
            "mode": "single",
            "kb_ids": [str(kb_id)],
            "doc_ids": [str(doc_id)],
            "choices": [{
                "key": "c2",
                "label": "公司差旅制度 2025版",
                "products": [],
                "canonical_products": [],
                "versions": ["2025"],
                "projects": [],
                "filenames": ["公司差旅制度.docx"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(doc_id)],
                "anchor_doc_ids": [str(doc_id)],
                "companion_doc_ids": [],
                "scope_slices": [{
                    "kb_id": str(kb_id),
                    "doc_id": str(doc_id),
                    "section_key": "section-2025",
                    "chunk_ids": [str(selected_chunk_id)],
                    "is_anchor": True,
                }],
            }],
        }
        normalized = _normalize_evidence_scope_filter(
            payload,
            authorized_kb_ids=[kb_id],
        )
        self.assertIsNotNone(normalized)
        self.assertTrue(normalized.valid)
        candidates = [
            {
                "id": str(selected_chunk_id),
                "kb_id": str(kb_id),
                "doc_id": str(doc_id),
                "metadata": {"section_key": "section-2025"},
            },
            {
                "id": str(sibling_chunk_id),
                "kb_id": str(kb_id),
                "doc_id": str(doc_id),
                "metadata": {"section_key": "section-2024"},
            },
        ]

        selected, dropped = _restrict_candidates_to_scope(
            candidates,
            normalized,
        )

        self.assertEqual([item["id"] for item in selected], [str(selected_chunk_id)])
        self.assertEqual(dropped, 1)
        self.assertEqual(
            _scope_anchor_coverage(selected, normalized),
            (True, (str(doc_id),)),
        )
        self.assertEqual(
            _scope_anchor_coverage(candidates[1:], normalized),
            (False, ()),
        )


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


def _task_contract(
    *,
    intent_code: str,
    action: str,
    evidence_scope: str,
    selected_kb_count: int = 1,
    requirements: list[dict] | None = None,
    source: str = "llm",
    question: str = "回答用户当前问题",
):
    route = parse_rag_route_decision(
        {
            "schema_version": "rag_route_decision.v1",
            "readiness": "ready",
            "intent_code": intent_code,
            "relation": "new",
            "evidence_scope": evidence_scope,
            "query_resolution": {"mode": "current", "context_turn_keys": []},
            "requirements": (
                requirements
                if requirements is not None
                else [
                    {
                        "role": "answer",
                        "origin": "user_text",
                        "description": "回答用户当前问题",
                    }
                ]
            ),
            "clarification": {"question": "", "unresolved": []},
            "confidence": 0.96,
            "rationale": "pipeline regression",
        },
        allowed_intent_codes=[intent_code],
    )
    return compile_rag_task_contract(
        route,
        RouteCategoryPolicy(
            code=intent_code,
            name=intent_code,
            action=action,
        ),
        RouteCompilerConfig(),
        question=question,
        selected_kb_count=selected_kb_count,
        source=source,
    )


def _event_payloads(chunks: list[str]) -> list[dict]:
    return [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: ")
    ]


def _scope_choice(
    *,
    key: str,
    label: str,
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    product: str = "云枢",
    version: str,
    project: str | None = None,
) -> dict:
    return {
        "key": key,
        "label": label,
        "products": [product],
        "canonical_products": [product],
        "versions": [version],
        "projects": [project] if project else [],
        "kb_ids": [str(kb_id)],
        "doc_ids": [str(doc_id)],
        "anchor_doc_ids": [str(doc_id)],
        "companion_doc_ids": [],
        "filenames": [f"{label}.md"],
    }


def _scope_filter(mode: str, choices: list[dict]) -> dict:
    return {
        "mode": mode,
        "kb_ids": list(dict.fromkeys(
            kb_id
            for choice in choices
            for kb_id in choice["kb_ids"]
        )),
        "doc_ids": list(dict.fromkeys(
            doc_id
            for choice in choices
            for doc_id in choice["doc_ids"]
        )),
        "choices": choices,
    }


def _search_event(chunks: list[str]) -> dict:
    return next(item for item in _event_payloads(chunks) if item["type"] == "search_results")


def _trace_event(trace_mock, name: str) -> dict:
    return next(
        call.kwargs
        for call in trace_mock.call_args_list
        if call.args and call.args[0] == name
    )


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


def _travel_small_document(
    kb_id: uuid.UUID,
    document_id: uuid.UUID,
) -> tuple[list[dict], list[dict]]:
    contents = (
        "公司出差管理标准",
        "总则：规范员工因公出差的交通、住宿和餐饮费用。",
        "职级分类：D级适用于普通员工、专员。",
        "飞机：D级国内和国际航班均为经济舱。",
        "火车：D级高铁动车二等座、火车硬卧。",
        "市内交通：D级以公共交通为主，特殊情况可乘出租车。",
        "住宿：D级一线城市450元/天、二线350元/天、其他250元/天。",
        "餐饮补贴：D级100元/天。",
        "通讯补贴：所有职级50元/天。",
        "出差补贴：所有职级100元/天。",
        "特殊地区补贴按流程另行审批。",
        "出差前填写申请单并完成上级审批。",
        "出差结束后5个工作日内提交费用报销。",
        "超出标准部分需说明原因并另行审批。",
        "本标准自发布之日起执行。",
    )
    total_chars = sum(len(content) for content in contents)
    full_document: list[dict] = []
    for index, content in enumerate(contents):
        item = {
            **_candidate(
                content=content,
                filename="公司出差管理标准.docx",
                score=0.0,
            ),
            "doc_id": document_id,
            "kb_id": kb_id,
            "chunk_index": index,
            "retrieval_score": None,
            "score": None,
            "candidate_origin": "small_document_full",
            "candidate_origins": ["small_document_full"],
            "full_document_chunk_count": len(contents),
            "full_document_char_count": total_chars,
        }
        full_document.append(item)

    initial: list[dict] = []
    for rank, document_index in enumerate((0, 1, 2), start=1):
        item = {
            **full_document[document_index],
            "candidate_origin": "current_retrieval",
            "candidate_origins": [],
            "retrieval_score": 0.08 - rank * 0.001,
            "score": 0.08 - rank * 0.001,
            "active_channels": (
                ["vector", "trigram"] if rank == 2 else ["vector"]
            ),
        }
        initial.append(item)
    return initial, full_document


class RagPipelineTests(unittest.IsolatedAsyncioTestCase):
    def test_clarification_trace_payload_redacts_choice_content(self) -> None:
        choice_doc_id = str(uuid.uuid4())
        decision = EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension="version",
            question="内部项目的敏感澄清正文",
            reason="multiple_mutually_exclusive_relevant_scopes",
            choices=(EvidenceScopeChoice(
                key="c1",
                label="敏感产品 8.2.75 —《内部文件》",
                products=("敏感产品",),
                canonical_products=("敏感产品",),
                versions=("8.2.75",),
                projects=("内部项目",),
                kb_ids=(str(uuid.uuid4()),),
                doc_ids=(choice_doc_id,),
                anchor_doc_ids=(choice_doc_id,),
                companion_doc_ids=(),
                filenames=("内部文件",),
                max_topic_relevance=0.9,
                max_answer_support=0.8,
            ),),
            relevant_document_count=1,
        )

        with patch(
            "core.rag_pipeline.content_fields",
            return_value={
                "question": decision.question,
                "question_chars": len(decision.question),
                "question_sha256": "digest",
            },
        ):
            payload = _clarification_trace_payload(
                decision,
                include_content=False,
            )

        self.assertNotIn("question", payload)
        self.assertNotIn("label", payload["choices"][0])
        self.assertNotIn("products", payload["choices"][0])
        self.assertNotIn("versions", payload["choices"][0])
        self.assertEqual(payload["choices"][0]["document_count"], 1)

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
        rerank_mock: AsyncMock | None = None,
        full_document_mock: AsyncMock | None = None,
        scoped_search_mock: AsyncMock | None = None,
        evidence_scope_filter: dict | None = None,
        task_contract=None,
        kb_ids: list[uuid.UUID] | None = None,
        settings_overrides: dict | None = None,
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
        rerank_call = rerank_mock or AsyncMock(return_value=rerank_outcome)
        full_document_call = full_document_mock or AsyncMock(
            side_effect=AssertionError("当前场景不应触发首轮前小文档全文探测")
        )
        scoped_search_call = scoped_search_mock or AsyncMock(
            side_effect=AssertionError("当前场景不应触发澄清范围内重检索")
        )

        async def build_context(_db, selected, **_kwargs):
            return "\n".join(item["content"] for item in selected)

        with (
            patch(
                "core.rag_pipeline.get_settings",
                return_value=_settings(**(settings_overrides or {})),
            ),
            patch("core.rag_pipeline.hybrid_search", new=search),
            patch(
                "core.rag_pipeline.search_within_documents",
                new=scoped_search_call,
            ),
            patch("core.rag_pipeline.rerank_with_status", new=rerank_call),
            patch(
                "core.rag_pipeline.fetch_small_document_candidates",
                new=full_document_call,
            ),
            patch("core.rag_pipeline.expand_evidence_candidates", new=expansion_call),
            patch("core.rag_pipeline.joint_rerank_with_coverage", new=joint_call),
            patch(
                "core.rag_pipeline.select_small_document_evidence_with_coverage",
                new=joint_call,
            ),
            patch("core.rag_pipeline._build_context", new=build_context),
            patch("core.rag_pipeline.get_client", return_value=fake_client),
        ):
            chunks = [
                chunk
                async for chunk in run_rag_stream(
                    question=question,
                    kb_ids=kb_ids or [uuid.uuid4()],
                    search_config={"top_k": 5, "rerank": rerank_enabled},
                    conversation_id="test-conversation",
                    db=SimpleNamespace(),
                    intent=intent,
                    standalone_query=standalone_query,
                    conversation_history=conversation_history,
                    carryover_sources=carryover_sources,
                    is_followup=is_followup,
                    task_contract=task_contract,
                    evidence_scope_filter=evidence_scope_filter,
                )
            ]
        return chunks, search, fake_client

    async def test_single_requirement_chat_uses_bounded_simple_candidate_pool(self) -> None:
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
        )

        # top_k=5 previously expanded every simple question to 15 candidates.
        # Ten preserves a 2x recall pool while bounding model input/output.
        _chunks, search, _client = await self._run(
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=contract,
            results=[],
        )
        self.assertEqual(search.await_args.kwargs["top_k"], 10)

    async def test_multiple_relevant_document_versions_clarify_before_generation(self) -> None:
        kb_id = uuid.uuid4()
        v6_doc_id = uuid.uuid4()
        v8_doc_id = uuid.uuid4()
        v6 = {
            **_candidate(
                content="所属产品：云枢6>> 产品版本：6.0.1>>",
                filename="钉钉",
            ),
            "kb_id": kb_id,
            "doc_id": v6_doc_id,
            "topic_relevance": 0.7,
            "answer_support": 0.1,
            "evidence_role": "related",
            "rerank_status": "verified",
        }
        v8_basic = {
            **_candidate(
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：中青建安>"
                ),
                filename="二开发送钉钉工作通知",
            ),
            "kb_id": kb_id,
            "doc_id": v8_doc_id,
            "topic_relevance": 0.9,
            "answer_support": 0.1,
            "evidence_role": "related",
            "rerank_status": "verified",
        }
        v8_solution = {
            **_candidate(
                content="调用 DingTalkMessageServiceImpl 发送钉钉工作通知",
                filename="二开发送钉钉工作通知",
            ),
            "kb_id": kb_id,
            "doc_id": v8_doc_id,
            "topic_relevance": 1.0,
            "answer_support": 0.98,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.98,
        }
        candidates = [v6, v8_basic, v8_solution]
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
        )

        chunks, _search, client = await self._run(
            question="云枢中想二开钉钉消息可以吗",
            standalone_query="云枢中想二开钉钉消息可以吗",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=contract,
            results=candidates,
            rerank_outcome=RerankOutcome(
                results=candidates,
                succeeded=True,
            ),
            kb_ids=[kb_id],
        )

        payloads = _event_payloads(chunks)
        search_event = next(
            item for item in payloads if item["type"] == "search_results"
        )
        clarification = next(
            item for item in payloads if item["type"] == "evidence_clarification"
        )
        answer = next(
            item for item in payloads if item["type"] == "text_delta"
        )

        self.assertEqual(search_event["evidence_status"], "needs_clarification")
        self.assertEqual(search_event["decision_reason"], "evidence_scope_ambiguous")
        self.assertEqual(search_event["answer_sources"], [])
        self.assertEqual(search_event["direct_evidence_count"], 0)
        self.assertEqual(clarification["dimension"], "version")
        self.assertEqual(len(clarification["choices"]), 2)
        self.assertIn("6.0.1", answer["content"])
        self.assertIn("8.2.75", answer["content"])
        self.assertFalse(any(
            item.get("type") == "search_step"
            and item.get("step") == "generate"
            and item.get("status") == "active"
            for item in payloads
        ))
        self.assertEqual(payloads[-1]["type"], "done")
        self.assertEqual(client.completions.calls, [])

    async def test_unverified_multiple_versions_still_clarify_without_generation(self) -> None:
        kb_id = uuid.uuid4()
        candidates = [
            {
                **_candidate(
                    content="所属产品：产品A；产品版本：1.0。配置项为旧值。",
                    filename="产品A旧版配置",
                ),
                "kb_id": kb_id,
                "doc_id": uuid.uuid4(),
            },
            {
                **_candidate(
                    content="所属产品：产品A；产品版本：2.0。配置项为新值。",
                    filename="产品A新版配置",
                ),
                "kb_id": kb_id,
                "doc_id": uuid.uuid4(),
            },
        ]

        chunks, _search, client = await self._run(
            question="产品A怎么配置",
            standalone_query="产品A怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=candidates,
            rerank_enabled=False,
            kb_ids=[kb_id],
        )

        payloads = _event_payloads(chunks)
        search_event = _search_event(chunks)
        self.assertEqual(search_event["evidence_status"], "needs_clarification")
        self.assertEqual(search_event["answer_sources"], [])
        self.assertEqual(search_event["direct_evidence_count"], 0)
        self.assertTrue(any(
            item["type"] == "evidence_clarification" for item in payloads
        ))
        self.assertFalse(any(
            item.get("type") == "search_step"
            and item.get("step") == "generate"
            and item.get("status") == "active"
            for item in payloads
        ))
        self.assertEqual(client.completions.calls, [])

    async def test_failed_rerank_multiple_versions_still_clarify_without_generation(self) -> None:
        kb_id = uuid.uuid4()
        candidates = [
            {
                **_candidate(
                    content="所属产品：产品A；产品版本：1.0。配置项为旧值。",
                    filename="产品A旧版配置",
                ),
                "kb_id": kb_id,
                "doc_id": uuid.uuid4(),
            },
            {
                **_candidate(
                    content="所属产品：产品A；产品版本：2.0。配置项为新值。",
                    filename="产品A新版配置",
                ),
                "kb_id": kb_id,
                "doc_id": uuid.uuid4(),
            },
        ]

        chunks, _search, client = await self._run(
            question="产品A怎么配置",
            standalone_query="产品A怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=candidates,
            rerank_outcome=RerankOutcome(
                results=candidates,
                succeeded=False,
                error="APITimeoutError: rerank unavailable",
            ),
            kb_ids=[kb_id],
        )

        search_event = _search_event(chunks)
        self.assertEqual(search_event["evidence_status"], "needs_clarification")
        self.assertEqual(search_event["answer_sources"], [])
        self.assertTrue(any(
            item["type"] == "evidence_clarification"
            for item in _event_payloads(chunks)
        ))
        self.assertEqual(client.completions.calls, [])

    async def test_more_than_six_explicit_versions_require_broad_refinement(self) -> None:
        kb_id = uuid.uuid4()
        candidates = [
            {
                **_candidate(
                    content=f"所属产品：平台A；产品版本：{version}。配置说明。",
                    filename=f"平台A {version} 版本说明",
                ),
                "kb_id": kb_id,
                "metadata": {
                    "product": "平台A",
                    "version": version,
                },
            }
            for version in ("1", "2", "3", "4", "5", "6", "7")
        ]

        chunks, _search, client = await self._run(
            question="平台A所有版本都对比",
            standalone_query="平台A所有版本都对比",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=candidates,
            rerank_enabled=False,
            kb_ids=[kb_id],
        )

        payloads = _event_payloads(chunks)
        event = _search_event(chunks)
        clarification = next(
            item for item in payloads if item["type"] == "evidence_clarification"
        )
        self.assertEqual(event["evidence_status"], "needs_clarification")
        self.assertEqual(event["decision_reason"], "evidence_scope_ambiguous")
        self.assertEqual(event["answer_sources"], [])
        self.assertEqual(clarification["dimension"], "version")
        self.assertEqual(clarification["reason"], "too_many_mutually_exclusive_scopes")
        self.assertEqual(clarification["choices"], [])
        self.assertIn("缩小查询范围", clarification["question"])
        self.assertFalse(any(
            item.get("type") == "search_step"
            and item.get("step") == "generate"
            and item.get("status") == "active"
            for item in payloads
        ))
        self.assertEqual(client.completions.calls, [])

    async def test_first_turn_multi_version_comparison_keeps_every_version(self) -> None:
        kb_id = uuid.uuid4()
        v6_doc_id = uuid.uuid4()
        v7_doc_id = uuid.uuid4()
        v8_doc_id = uuid.uuid4()
        v6 = {
            **_candidate(
                content="所属产品：云枢6；产品版本：6.0.1。旧版使用接口A。",
                filename="旧版接口",
            ),
            "kb_id": kb_id,
            "doc_id": v6_doc_id,
            "topic_relevance": 0.95,
            "answer_support": 0.9,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.9,
        }
        v7 = {
            **_candidate(
                content="所属产品：云枢7；产品版本：7.0。中间版本使用接口C。",
                filename="中间版本接口",
            ),
            "kb_id": kb_id,
            "doc_id": v7_doc_id,
            "topic_relevance": 0.99,
            "answer_support": 0.99,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.99,
        }
        v8 = {
            **_candidate(
                content="所属产品：云枢8；产品版本：8.2.75。新版使用接口B。",
                filename="新版接口",
            ),
            "kb_id": kb_id,
            "doc_id": v8_doc_id,
            "topic_relevance": 0.96,
            "answer_support": 0.92,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.92,
        }
        for query in (
            "请对比云枢6和云枢8的消息接口",
            "云枢6和云枢8都要配置登录安全",
        ):
            with self.subTest(query=query):
                rerank = AsyncMock(return_value=RerankOutcome(
                    # A faulty/custom reranker is allowed to echo the original
                    # third scope. Pipeline must enforce the source-derived
                    # allow-list again.
                    results=[v7, v6, v8],
                    succeeded=True,
                ))

                chunks, _search, client = await self._run(
                    question=query,
                    standalone_query=query,
                    intent={
                        "response_mode": "grounded_qa",
                        "retrieval_policy": "required",
                        "need_retrieval": True,
                        "decision_reason": "classified_retrieval",
                    },
                    task_contract=_task_contract(
                        intent_code="knowledge_qa",
                        action="retrieve",
                        evidence_scope="enterprise_kb",
                    ),
                    results=[v6, v7, v8],
                    rerank_outcome=RerankOutcome(
                        results=[v6, v7, v8],
                        succeeded=True,
                    ),
                    rerank_mock=rerank,
                    kb_ids=[kb_id],
                )

                self.assertEqual(
                    {str(item["doc_id"]) for item in rerank.await_args.args[1]},
                    {str(v6_doc_id), str(v8_doc_id)},
                )
                search_event = _search_event(chunks)
                self.assertIsNone(search_event["query_constraints"]["product"])
                self.assertIsNone(search_event["query_constraints"]["version"])
                self.assertEqual(
                    {item["doc_id"] for item in search_event["answer_sources"]},
                    {str(v6_doc_id), str(v8_doc_id)},
                )
                self.assertNotIn(
                    str(v7_doc_id),
                    {item["doc_id"] for item in search_event["results"]},
                )
                self.assertFalse(any(
                    item["type"] == "evidence_clarification"
                    for item in _event_payloads(chunks)
                ))
                generation_payload = "\n".join(
                    str(message.get("content") or "")
                    for message in client.completions.calls[0]["messages"]
                )
                self.assertIn("6.0.1", generation_payload)
                self.assertIn("8.2.75", generation_payload)
                self.assertNotIn("7.0", generation_payload)
                self.assertIn(
                    "按产品、版本或项目分别组织答案",
                    client.completions.calls[0]["messages"][0]["content"],
                )

    async def test_first_turn_all_projects_compares_without_reclarifying(self) -> None:
        kb_id = uuid.uuid4()
        project_a_doc_id = uuid.uuid4()
        project_b_doc_id = uuid.uuid4()
        project_a = {
            **_candidate(
                content=(
                    "所属产品：产品A；产品版本：1.0；所属项目：甲项目。"
                    "登录配置使用参数A。"
                ),
                filename="甲项目登录配置",
            ),
            "kb_id": kb_id,
            "doc_id": project_a_doc_id,
            "topic_relevance": 0.96,
            "answer_support": 0.9,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.9,
        }
        project_b = {
            **_candidate(
                content=(
                    "所属产品：产品A；产品版本：1.0；所属项目：乙项目。"
                    "登录配置使用参数B。"
                ),
                filename="乙项目登录配置",
            ),
            "kb_id": kb_id,
            "doc_id": project_b_doc_id,
            "topic_relevance": 0.95,
            "answer_support": 0.88,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.88,
        }
        query = "所有项目的登录配置都对比"

        chunks, _search, client = await self._run(
            question=query,
            standalone_query=query,
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=[project_a, project_b],
            rerank_outcome=RerankOutcome(
                results=[project_a, project_b],
                succeeded=True,
            ),
            kb_ids=[kb_id],
        )

        payloads = _event_payloads(chunks)
        self.assertFalse(any(
            item["type"] == "evidence_clarification" for item in payloads
        ))
        self.assertEqual(
            {item["doc_id"] for item in _search_event(chunks)["answer_sources"]},
            {str(project_a_doc_id), str(project_b_doc_id)},
        )
        self.assertIn(
            "按产品、版本或项目分别组织答案",
            client.completions.calls[0]["messages"][0]["content"],
        )

    async def test_single_scope_selection_retrieves_only_selected_documents(self) -> None:
        kb_id = uuid.uuid4()
        selected_doc_id = uuid.uuid4()
        other_doc_id = uuid.uuid4()
        selected = {
            **_candidate(
                content=(
                    "所属产品：云枢8；产品版本：8.2.75。"
                    "工作通知通过 DingTalkMessageServiceImpl 发送。"
                ),
                filename="二开发送钉钉工作通知",
            ),
            "kb_id": kb_id,
            "doc_id": selected_doc_id,
            "topic_relevance": 0.98,
            "answer_support": 0.95,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.95,
        }
        choice = _scope_choice(
            key="c2",
            label="云枢 8.2.75（中青建安）—《二开发送钉钉工作通知》",
            kb_id=kb_id,
            doc_id=selected_doc_id,
            version="8.2.75",
            project="中青建安",
        )
        scoped_search = AsyncMock(return_value=[selected])
        rerank = AsyncMock(return_value=RerankOutcome(
            results=[selected],
            succeeded=True,
        ))

        chunks, global_search, client = await self._run(
            question="2",
            standalone_query="云枢中想二开钉钉消息可以吗",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=[selected],
            rerank_outcome=RerankOutcome(results=[selected], succeeded=True),
            rerank_mock=rerank,
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("single", [choice]),
            kb_ids=[kb_id],
            carryover_sources=[{
                **selected,
                "doc_id": other_doc_id,
            }],
        )

        global_search.assert_not_awaited()
        self.assertEqual(scoped_search.await_count, 1)
        call = scoped_search.await_args.kwargs
        self.assertEqual(call["kb_ids"], [kb_id])
        self.assertEqual(call["doc_ids"], [selected_doc_id])
        self.assertEqual(call["max_document_count"], 1)
        self.assertEqual(call["queries"][0], "云枢中想二开钉钉消息可以吗")
        self.assertIn("8.2.75", call["queries"][1])
        self.assertIn("8.2.75", rerank.await_args.args[0])
        self.assertFalse(any(
            item["type"] == "evidence_clarification"
            for item in _event_payloads(chunks)
        ))
        search_event = _search_event(chunks)
        self.assertEqual(search_event["evidence_status"], "hit")
        self.assertEqual(
            [item["doc_id"] for item in search_event["answer_sources"]],
            [str(selected_doc_id)],
        )
        generation_question = client.completions.calls[0]["messages"][-1]["content"]
        self.assertIn("云枢中想二开钉钉消息可以吗", generation_question)
        self.assertIn("8.2.75", generation_question)

    async def test_scope_selection_forces_retrieval_when_route_drifted_to_chat(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        selected = {
            **_candidate(
                content="所属产品：云枢8；产品版本：8.2.75。配置项为B。",
                filename="版本配置",
            ),
            "kb_id": kb_id,
            "doc_id": doc_id,
            "topic_relevance": 0.98,
            "answer_support": 0.95,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.95,
        }
        choice = _scope_choice(
            key="c1",
            label="云枢 8.2.75 —《版本配置》",
            kb_id=kb_id,
            doc_id=doc_id,
            version="8.2.75",
        )
        scoped_search = AsyncMock(return_value=[selected])

        chunks, global_search, _client = await self._run(
            question="1",
            standalone_query="登录安全怎么配置",
            intent={
                "response_mode": "general_chat",
                "retrieval_policy": "skip",
                "need_retrieval": False,
                "decision_reason": "classified_general_chat",
            },
            task_contract=_task_contract(
                intent_code="general_chat",
                action="chat",
                evidence_scope="general_world",
            ),
            results=[selected],
            rerank_outcome=RerankOutcome(results=[selected], succeeded=True),
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("single", [choice]),
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        scoped_search.assert_awaited_once()
        search_event = _search_event(chunks)
        self.assertTrue(search_event["retrieval_executed"])
        self.assertEqual(search_event["decision_reason"], "evidence_scope_selected")
        self.assertEqual(search_event["evidence_status"], "hit")

    async def test_single_scope_companion_cannot_replace_anchor_hit(self) -> None:
        kb_id = uuid.uuid4()
        anchor_doc_id = uuid.uuid4()
        companion_doc_id = uuid.uuid4()
        anchor = {
            **_candidate(
                content="所属产品：云枢8；产品版本：8.2.75。该版本使用接口B。",
                filename="8.2.75 版本说明",
            ),
            "kb_id": kb_id,
            "doc_id": anchor_doc_id,
            "topic_relevance": 0.98,
            "answer_support": 0.95,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.95,
        }
        companion = {
            **_candidate(
                content="所属产品：云枢。消息功能的通用前置条件。",
                filename="通用前置条件",
            ),
            "kb_id": kb_id,
            "doc_id": companion_doc_id,
            "topic_relevance": 0.9,
            "answer_support": 0.8,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.8,
        }
        choice = _scope_choice(
            key="c1",
            label="云枢 8.2.75 —《8.2.75 版本说明》等2篇",
            kb_id=kb_id,
            doc_id=anchor_doc_id,
            version="8.2.75",
        )
        choice["doc_ids"].append(str(companion_doc_id))
        choice["companion_doc_ids"] = [str(companion_doc_id)]
        scoped_search = AsyncMock(side_effect=[[companion], [anchor]])
        rerank = AsyncMock(return_value=RerankOutcome(
            results=[anchor, companion],
            succeeded=True,
        ))

        chunks, global_search, _client = await self._run(
            question="1",
            standalone_query="消息接口怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=[companion],
            rerank_outcome=RerankOutcome(results=[anchor, companion], succeeded=True),
            rerank_mock=rerank,
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("single", [choice]),
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        self.assertEqual(scoped_search.await_count, 2)
        self.assertEqual(
            scoped_search.await_args_list[1].kwargs["doc_ids"],
            [anchor_doc_id],
        )
        event = _search_event(chunks)
        self.assertTrue(event["evidence_scope_anchor_hit"])
        self.assertEqual(
            event["evidence_scope_anchor_doc_ids"],
            [str(anchor_doc_id)],
        )

    async def test_rerank_dropped_anchor_blocks_companion_only_answer(self) -> None:
        kb_id = uuid.uuid4()
        anchor_doc_id = uuid.uuid4()
        companion_doc_id = uuid.uuid4()
        anchor = {
            **_candidate(
                content="所属产品：云枢8；产品版本：8.2.75。该版本使用接口B。",
                filename="8.2.75 版本说明",
            ),
            "kb_id": kb_id,
            "doc_id": anchor_doc_id,
        }
        companion = {
            **_candidate(
                content="所属产品：云枢。消息功能的通用前置条件。",
                filename="通用前置条件",
            ),
            "kb_id": kb_id,
            "doc_id": companion_doc_id,
        }
        rerank_rejected_anchor = {
            **anchor,
            "topic_relevance": 0.1,
            "answer_support": 0.0,
            "evidence_role": "irrelevant",
            "rerank_status": "verified",
            "score": 0.0,
        }
        rerank_direct_companion = {
            **companion,
            "topic_relevance": 0.98,
            "answer_support": 0.95,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.95,
        }
        choice = _scope_choice(
            key="c1",
            label="云枢 8.2.75 —《8.2.75 版本说明》等2篇",
            kb_id=kb_id,
            doc_id=anchor_doc_id,
            version="8.2.75",
        )
        choice["doc_ids"].append(str(companion_doc_id))
        choice["companion_doc_ids"] = [str(companion_doc_id)]
        scoped_search = AsyncMock(return_value=[anchor, companion])

        chunks, global_search, client = await self._run(
            question="2",
            standalone_query="消息接口怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=[anchor, companion],
            rerank_outcome=RerankOutcome(
                results=[rerank_rejected_anchor, rerank_direct_companion],
                succeeded=True,
            ),
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("single", [choice]),
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        scoped_search.assert_awaited_once()
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(
            event["decision_reason"],
            "evidence_scope_answer_anchor_incomplete",
        )
        self.assertFalse(event["evidence_scope_anchor_hit"])
        self.assertEqual(event["evidence_scope_anchor_doc_ids"], [])
        self.assertEqual(event["answer_sources"], [])
        self.assertEqual(event["direct_evidence_count"], 0)
        companion_result = next(
            item
            for item in event["results"]
            if item["doc_id"] == str(companion_doc_id)
        )
        self.assertEqual(companion_result["evidence_role"], "related")
        self.assertNotIn(
            "消息功能的通用前置条件",
            json.dumps(client.completions.calls, ensure_ascii=False),
        )

    async def test_compare_scope_shared_companion_cannot_cover_both_anchors(self) -> None:
        kb_id = uuid.uuid4()
        v6_doc_id = uuid.uuid4()
        v8_doc_id = uuid.uuid4()
        companion_doc_id = uuid.uuid4()

        def scoped_candidate(doc_id: uuid.UUID, content: str, filename: str) -> dict:
            return {
                **_candidate(content=content, filename=filename),
                "kb_id": kb_id,
                "doc_id": doc_id,
                "topic_relevance": 0.95,
                "answer_support": 0.9,
                "evidence_role": "direct",
                "rerank_status": "verified",
                "score": 0.9,
            }

        v6 = scoped_candidate(
            v6_doc_id,
            "所属产品：云枢6；产品版本：6.0.1。旧版使用接口A。",
            "旧版接口",
        )
        v8 = scoped_candidate(
            v8_doc_id,
            "所属产品：云枢8；产品版本：8.2.75。新版使用接口B。",
            "新版接口",
        )
        companion = scoped_candidate(
            companion_doc_id,
            "所属产品：云枢。两个版本共同使用消息服务。",
            "通用消息说明",
        )
        choices = [
            _scope_choice(
                key="c1",
                label="云枢 6.0.1 —《旧版接口》等2篇",
                kb_id=kb_id,
                doc_id=v6_doc_id,
                version="6.0.1",
            ),
            _scope_choice(
                key="c2",
                label="云枢 8.2.75 —《新版接口》等2篇",
                kb_id=kb_id,
                doc_id=v8_doc_id,
                version="8.2.75",
            ),
        ]
        for choice in choices:
            choice["doc_ids"].append(str(companion_doc_id))
            choice["companion_doc_ids"] = [str(companion_doc_id)]
        scoped_search = AsyncMock(side_effect=[[companion], [v6], [v8]])
        rerank = AsyncMock(return_value=RerankOutcome(
            results=[v6, v8, companion],
            succeeded=True,
        ))

        chunks, global_search, _client = await self._run(
            question="都对比",
            standalone_query="消息接口怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=[companion],
            rerank_outcome=RerankOutcome(
                results=[v6, v8, companion],
                succeeded=True,
            ),
            rerank_mock=rerank,
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("compare_all", choices),
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        self.assertEqual(scoped_search.await_count, 3)
        self.assertEqual(
            scoped_search.await_args_list[1].kwargs["doc_ids"],
            [v6_doc_id],
        )
        self.assertEqual(
            scoped_search.await_args_list[2].kwargs["doc_ids"],
            [v8_doc_id],
        )
        event = _search_event(chunks)
        self.assertTrue(event["evidence_scope_anchor_hit"])
        self.assertEqual(
            set(event["evidence_scope_anchor_doc_ids"]),
            {str(v6_doc_id), str(v8_doc_id)},
        )
        self.assertNotIn(
            str(companion_doc_id),
            event["evidence_scope_anchor_doc_ids"],
        )

    async def test_six_scope_comparison_reserves_one_answer_slot_per_choice(self) -> None:
        kb_id = uuid.uuid4()
        candidates: list[dict] = []
        choices: list[dict] = []
        for version in ("1", "2", "3", "4", "5", "6"):
            doc_id = uuid.uuid4()
            candidates.append({
                **_candidate(
                    content=f"所属产品：平台A；产品版本：{version}。配置值为{version}。",
                    filename=f"平台A {version} 版本配置",
                ),
                "kb_id": kb_id,
                "doc_id": doc_id,
                "topic_relevance": 0.95,
                "answer_support": 0.9,
                "evidence_role": "direct",
                "rerank_status": "verified",
                "score": 0.9,
            })
            choices.append(_scope_choice(
                key=f"c{version}",
                label=f"平台A {version} —《版本配置》",
                kb_id=kb_id,
                doc_id=doc_id,
                product="平台A",
                version=version,
            ))
        scoped_search = AsyncMock(return_value=candidates)

        chunks, global_search, client = await self._run(
            question="都对比",
            standalone_query="平台A登录安全怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=candidates,
            rerank_outcome=RerankOutcome(results=candidates, succeeded=True),
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("compare_all", choices),
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        self.assertGreaterEqual(scoped_search.await_args.kwargs["total_limit"], 6)
        event = _search_event(chunks)
        self.assertTrue(event["evidence_scope_anchor_hit"])
        self.assertEqual(len(event["answer_sources"]), 6)
        self.assertEqual(
            {item["doc_id"] for item in event["answer_sources"]},
            {str(item["doc_id"]) for item in candidates},
        )
        self.assertFalse(any(
            item["type"] == "evidence_clarification"
            for item in _event_payloads(chunks)
        ))
        self.assertEqual(len(client.completions.calls), 1)

    async def test_single_compatibility_choice_does_not_bind_first_label_version(self) -> None:
        kb_id = uuid.uuid4()
        v6_doc_id = uuid.uuid4()
        v8_doc_id = uuid.uuid4()
        candidates = [
            {
                **_candidate(
                    content="所属产品：云枢；产品版本：6.0.1。兼容配置A。",
                    filename="兼容配置旧版",
                ),
                "kb_id": kb_id,
                "doc_id": v6_doc_id,
                "topic_relevance": 0.95,
                "answer_support": 0.9,
                "evidence_role": "direct",
                "rerank_status": "verified",
                "score": 0.9,
            },
            {
                **_candidate(
                    content="所属产品：云枢；产品版本：8.2.75。兼容配置B。",
                    filename="兼容配置新版",
                ),
                "kb_id": kb_id,
                "doc_id": v8_doc_id,
                "topic_relevance": 0.94,
                "answer_support": 0.88,
                "evidence_role": "direct",
                "rerank_status": "verified",
                "score": 0.88,
            },
        ]
        choice = _scope_choice(
            key="c1",
            label="云枢兼容版本 6.0.1 / 8.2.75 —《兼容配置》",
            kb_id=kb_id,
            doc_id=v6_doc_id,
            version="6.0.1",
        )
        choice["versions"].append("8.2.75")
        choice["doc_ids"].append(str(v8_doc_id))
        choice["anchor_doc_ids"].append(str(v8_doc_id))
        scoped_search = AsyncMock(return_value=candidates)

        chunks, global_search, _client = await self._run(
            question="1",
            standalone_query="消息接口怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=candidates,
            rerank_outcome=RerankOutcome(results=candidates, succeeded=True),
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("single", [choice]),
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        event = _search_event(chunks)
        self.assertIsNone(event["query_constraints"]["version"])
        self.assertEqual(
            {item["doc_id"] for item in event["answer_sources"]},
            {str(v6_doc_id), str(v8_doc_id)},
        )

    async def test_scope_selection_fails_closed_when_anchor_stays_missing(self) -> None:
        kb_id = uuid.uuid4()
        anchor_doc_id = uuid.uuid4()
        companion_doc_id = uuid.uuid4()
        companion = {
            **_candidate(
                content="所属产品：云枢。消息功能的通用说明。",
                filename="通用消息说明",
            ),
            "kb_id": kb_id,
            "doc_id": companion_doc_id,
        }
        choice = _scope_choice(
            key="c1",
            label="云枢 8.2.75 —《版本说明》等2篇",
            kb_id=kb_id,
            doc_id=anchor_doc_id,
            version="8.2.75",
        )
        choice["doc_ids"].append(str(companion_doc_id))
        choice["companion_doc_ids"] = [str(companion_doc_id)]
        scoped_search = AsyncMock(side_effect=[[companion], []])

        chunks, global_search, _client = await self._run(
            question="1",
            standalone_query="消息接口怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=[companion],
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("single", [choice]),
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        self.assertEqual(scoped_search.await_count, 2)
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "error")
        self.assertFalse(event["evidence_scope_anchor_hit"])
        self.assertEqual(event["evidence_scope_anchor_doc_ids"], [])

    async def test_compare_all_scope_keeps_both_versions_and_separates_answer(self) -> None:
        kb_id = uuid.uuid4()
        v6_doc_id = uuid.uuid4()
        v8_doc_id = uuid.uuid4()
        v6 = {
            **_candidate(
                content="所属产品：云枢6；产品版本：6.0.1。旧版使用接口A。",
                filename="钉钉",
            ),
            "kb_id": kb_id,
            "doc_id": v6_doc_id,
            "topic_relevance": 0.95,
            "answer_support": 0.9,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.9,
        }
        v8 = {
            **_candidate(
                content="所属产品：云枢8；产品版本：8.2.75。新版使用接口B。",
                filename="二开发送钉钉工作通知",
            ),
            "kb_id": kb_id,
            "doc_id": v8_doc_id,
            "topic_relevance": 0.97,
            "answer_support": 0.92,
            "evidence_role": "direct",
            "rerank_status": "verified",
            "score": 0.92,
        }
        choices = [
            _scope_choice(
                key="c1",
                label="云枢 6.0.1 —《钉钉》",
                kb_id=kb_id,
                doc_id=v6_doc_id,
                version="6.0.1",
            ),
            _scope_choice(
                key="c2",
                label="云枢 8.2.75 —《二开发送钉钉工作通知》",
                kb_id=kb_id,
                doc_id=v8_doc_id,
                version="8.2.75",
            ),
        ]
        # The first globally-ranked scoped lookup is intentionally crowded by
        # v6.  Pipeline must issue one bounded v8-only supplement before it is
        # allowed to generate a comparison.
        scoped_search = AsyncMock(side_effect=[[v6], [v8]])
        rerank = AsyncMock(return_value=RerankOutcome(
            results=[v6, v8],
            succeeded=True,
        ))

        chunks, global_search, client = await self._run(
            question="都对比",
            standalone_query="云枢中想二开钉钉消息可以吗",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=[v6, v8],
            rerank_outcome=RerankOutcome(results=[v6, v8], succeeded=True),
            rerank_mock=rerank,
            scoped_search_mock=scoped_search,
            evidence_scope_filter=_scope_filter("compare_all", choices),
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        self.assertEqual(scoped_search.await_count, 2)
        broad_call = scoped_search.await_args_list[0].kwargs
        supplement_call = scoped_search.await_args_list[1].kwargs
        self.assertEqual(set(broad_call["doc_ids"]), {v6_doc_id, v8_doc_id})
        self.assertIn("6.0.1", broad_call["queries"][1])
        self.assertIn("8.2.75", broad_call["queries"][1])
        self.assertEqual(supplement_call["doc_ids"], [v8_doc_id])
        self.assertEqual(supplement_call["kb_ids"], [kb_id])
        rerank_query = rerank.await_args.args[0]
        self.assertIn("全部适用范围", rerank_query)
        self.assertNotIn("6.0.1", rerank_query)
        self.assertNotIn("8.2.75", rerank_query)
        payloads = _event_payloads(chunks)
        self.assertFalse(any(
            item["type"] == "evidence_clarification" for item in payloads
        ))
        search_event = _search_event(chunks)
        self.assertEqual(search_event["answer_source_count"], 2)
        self.assertFalse(search_event["query_constraints"]["explicit_version"])
        self.assertIsNone(search_event["query_constraints"]["version"])
        self.assertEqual(
            {item["doc_id"] for item in search_event["answer_sources"]},
            {str(v6_doc_id), str(v8_doc_id)},
        )
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("按产品、版本或项目分别组织答案", system_prompt)

    async def test_malformed_scope_filter_fails_closed_without_global_search(self) -> None:
        kb_id = uuid.uuid4()
        choice_doc_id = uuid.uuid4()
        unrelated_doc_id = uuid.uuid4()
        choice = _scope_choice(
            key="c1",
            label="云枢 8.2.75 —《配置说明》",
            kb_id=kb_id,
            doc_id=choice_doc_id,
            version="8.2.75",
        )
        malformed = _scope_filter("single", [choice])
        malformed["doc_ids"] = [str(unrelated_doc_id)]

        chunks, global_search, client = await self._run(
            question="1",
            standalone_query="登录安全怎么配置",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
            task_contract=_task_contract(
                intent_code="knowledge_qa",
                action="retrieve",
                evidence_scope="enterprise_kb",
            ),
            results=[],
            evidence_scope_filter=malformed,
            kb_ids=[kb_id],
        )

        global_search.assert_not_awaited()
        search_event = _search_event(chunks)
        self.assertEqual(search_event["evidence_status"], "error")
        prompt_content = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("检索或证据验证暂时失败", prompt_content)
        self.assertNotIn("知识库中未找到相关内容", prompt_content)

    async def test_v1_task_contract_is_authority_over_legacy_execution_fields(self) -> None:
        contract = _task_contract(
            intent_code="general_chat",
            action="chat",
            evidence_scope="general_world",
        )

        chunks, search, _client = await self._run(
            intent={
                "action": "retrieve",
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "stale_legacy_projection",
            },
            task_contract=contract,
            results=[],
        )

        search.assert_not_awaited()
        event = _search_event(chunks)
        self.assertFalse(event["retrieval_executed"])
        self.assertEqual(event["decision_reason"], contract.decision_reason)

    async def test_v1_task_contract_internal_execution_drift_is_rejected(self) -> None:
        contract = _task_contract(
            intent_code="general_chat",
            action="chat",
            evidence_scope="general_world",
        )
        inconsistent = replace(contract, need_retrieval=True)

        with self.assertRaises(TaskContractDispatchError):
            await self._run(
                intent=None,
                task_contract=inconsistent,
                results=[],
            )

    async def test_pipeline_passes_route_locked_requirements_to_initial_reranker(self) -> None:
        requirements = [
            {
                "role": "bridge",
                "origin": "semantically_entailed",
                "description": "确定普通员工对应的职级",
            },
            {
                "role": "answer",
                "origin": "user_text",
                "description": "取得完整出差标准",
            },
        ]
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            requirements=requirements,
        )
        result = _candidate(content="普通员工为 D 级，按 D 级标准执行。")
        rerank = AsyncMock(
            return_value=RerankOutcome(
                results=[result],
                succeeded=False,
                error="test fallback",
            )
        )

        await self._run(
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
            },
            task_contract=contract,
            results=[result],
            rerank_mock=rerank,
        )

        locked = rerank.await_args.args[2]
        self.assertEqual([item.id for item in locked], ["r1", "r2"])
        self.assertEqual(
            [(item.importance, item.source) for item in locked],
            [("helpful", "inferred"), ("required", "explicit")],
        )

    async def test_contract_kb_count_uses_unique_ids_at_pipeline_gate(self) -> None:
        kb_id = uuid.uuid4()
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            selected_kb_count=1,
        )

        chunks, search, _client = await self._run(
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
            },
            task_contract=contract,
            results=[],
            kb_ids=[kb_id, kb_id],
        )

        search.assert_awaited_once()
        self.assertTrue(_search_event(chunks)["retrieval_executed"])

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

    async def test_rerank_failure_hides_unknown_scope_for_explicit_version(self) -> None:
        unrelated = _candidate(
            filename="公司出差管理标准.docx",
            content="公司出差管理标准",
            score=0.99,
        )
        chunks, _search, client = await self._run(
            question="云枢8.6呢",
            standalone_query="云枢8.6 解决登录用户名枚举要配置什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=[unrelated],
            rerank_outcome=RerankOutcome(
                results=[unrelated],
                succeeded=False,
                error="ValueError: expansion.target_candidate_indexes 数量无效",
            ),
        )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "no_hit")
        self.assertEqual(event["results"], [])
        self.assertEqual(event["answer_sources"], [])
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("公司出差管理标准", prompt)

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

    async def test_inherited_document_version_is_visible_in_generation_context(self) -> None:
        basic = {
            "id": "basic",
            "kb_id": "kb",
            "doc_id": "doc",
            "filename": "二开发送钉钉工作通知",
            "content": "所属产品：云枢8>> 产品版本：8.2.75>>",
        }
        solution = {
            "id": "solution",
            "kb_id": "kb",
            "doc_id": "doc",
            "filename": "二开发送钉钉工作通知",
            "content": "调用 DingTalkMessageServiceImpl 发送钉钉工作通知",
            "evidence_role": "direct",
        }
        enriched = inherit_document_constraint_metadata([basic, solution])
        answer = enriched[1]
        evaluation = evaluate_candidate_constraints(
            extract_query_constraints("云枢中想二开钉钉消息可以吗"),
            answer,
        )
        answer["constraint_reason"] = evaluation.reason

        context = await _build_context(
            SimpleNamespace(),
            [answer],
            allow_whole_document=False,
        )

        self.assertIn("8.2.75", context)
        self.assertIn("DingTalkMessageServiceImpl", context)

    async def test_compare_context_marks_each_document_scope(self) -> None:
        first = {
            **_candidate(content="旧版使用接口A", filename="旧版说明"),
            "doc_id": "doc-v6",
            "evidence_role": "direct",
        }
        second = {
            **_candidate(content="新版使用接口B", filename="新版说明"),
            "doc_id": "doc-v8",
            "evidence_role": "direct",
        }

        context = await _build_context(
            SimpleNamespace(),
            [first, second],
            allow_whole_document=False,
            scope_labels_by_document={
                "doc-v6": "云枢 6.0.1",
                "doc-v8": "云枢 8.2.75",
            },
        )

        self.assertIn("适用范围：云枢 6.0.1", context)
        self.assertIn("适用范围：云枢 8.2.75", context)

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

    def test_productless_version_mismatch_is_diagnostic_only(self) -> None:
        constraints = extract_query_constraints("版本8.6登录配置")
        results = annotate_deterministic_constraints(
            [
                _candidate(filename="登录制度7版", content="旧版配置", score=0.9),
                _candidate(filename="通用登录制度", content="通用配置", score=0.8),
            ],
            constraints,
        )

        selected = _select_unverified_evidence(results, 5, constraints)

        self.assertEqual(selected[2], "version_mismatch")
        self.assertEqual(len(selected[0]), 1)
        self.assertEqual(selected[0][0]["constraint_status"], "mismatch")
        self.assertEqual(selected[1], [])

    def test_productless_version_unknown_never_enters_verified_context(self) -> None:
        unknown = {
            **_candidate(content="未声明适用版本的通用配置"),
            "topic_relevance": 0.99,
            "answer_support": 0.99,
            "constraint_status": "unknown",
            "query_has_constraint": True,
            "query_has_product_constraint": False,
            "query_has_hard_constraint": False,
            "query_has_version_constraint": True,
            "evidence_role": "direct",
        }

        selected = _select_verified_evidence([unknown], 5)

        self.assertEqual(selected[0], [])
        self.assertEqual(selected[1], [])
        self.assertEqual(selected[2], "no_hit")

    def test_productless_version_mismatch_never_enters_verified_context(self) -> None:
        mismatch = {
            **_candidate(filename="登录制度7版", content="旧版配置"),
            "topic_relevance": 0.99,
            "answer_support": 0.99,
            "constraint_status": "mismatch",
            "query_has_constraint": True,
            "query_has_product_constraint": False,
            "query_has_hard_constraint": False,
            "query_has_version_constraint": True,
            "evidence_role": "direct",
        }

        selected = _select_verified_evidence([mismatch], 5)

        self.assertEqual(selected[2], "version_mismatch")
        self.assertEqual(len(selected[0]), 1)
        self.assertEqual(selected[0][0]["evidence_role"], "related")
        self.assertEqual(selected[1], [])

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

    async def test_dominant_small_document_uses_only_one_joint_rerank(self) -> None:
        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        initial, full_document = _travel_small_document(kb_id, document_id)
        competitors = [
            {
                **_candidate(
                    content=f"竞争文档 {index}",
                    filename=f"其他制度{index}.md",
                ),
                "kb_id": kb_id,
                "active_channels": ["vector"],
            }
            for index in range(1, 4)
        ]
        out_of_scope_competitor = {
            **_candidate(
                content="不在本次授权知识库范围内的制度",
                filename="越权候选.md",
            ),
            "kb_id": uuid.uuid4(),
            "active_channels": ["trigram"],
        }
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            requirements=[
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "确定普通员工适用职级",
                },
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "回答交通和市内交通标准",
                },
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "回答住宿标准",
                },
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "回答餐饮及其他补贴",
                },
            ],
        )
        requirement_ids = tuple(item.id for item in contract.requirements)
        selected_indexes = (3, 4, 6, 7, 8, 9, 10)

        async def joint_once(_query, candidates, requirements, **kwargs):
            self.assertEqual(tuple(item.id for item in requirements), requirement_ids)
            self.assertEqual(kwargs["bridge_requirement_ids"], ())
            self.assertEqual(
                kwargs["eligible_candidate_indexes"],
                tuple(range(1, 16)),
            )
            self.assertEqual(kwargs["anchor_candidate_indexes"], (1, 2, 3))
            selected: list[dict] = []
            support_map = (
                [requirement_ids[0]],
                [requirement_ids[1]],
                [requirement_ids[1]],
                [requirement_ids[2]],
                [requirement_ids[3]],
                [requirement_ids[3]],
                [requirement_ids[3]],
            )
            for candidate_index, supported in zip(selected_indexes, support_map):
                candidate = candidates[candidate_index - 1]
                selected.append({
                    **candidate,
                    "rerank_candidate_index": candidate_index,
                    "topic_relevance": 0.99,
                    "answer_support": 0.94,
                    "constraint_status": "neutral",
                    "evidence_role": "direct",
                    "rerank_status": "verified_joint",
                    "joint_rerank_status": "verified",
                    "contribution_role": (
                        "bridge" if candidate_index == 3 else "complement"
                    ),
                    "supports_requirement_ids": supported,
                    "jointly_selected": True,
                    "evidence_set_id": "set_1",
                    "joint_support_score": 0.96,
                    "coverage_status": "complete",
                })
            return RerankOutcome(
                results=selected,
                succeeded=True,
                requirements=tuple(requirements),
                coverage_status="complete",
                covered_requirement_ids=requirement_ids,
                missing_requirement_ids=(),
                selected_evidence_set_id="set_1",
                selected_candidate_indexes=selected_indexes,
                joint_support_score=0.96,
                model="test-chat",
                prompt_version="joint-test",
                elapsed_ms=17,
                candidate_count=len(candidates),
            )

        rerank_mock = AsyncMock(
            side_effect=AssertionError("小文档快速路径不得调用首轮重排")
        )
        expansion_mock = AsyncMock(
            side_effect=AssertionError("全文已加载后不得再次执行证据扩展")
        )
        full_document_mock = AsyncMock(return_value=full_document)
        joint_mock = AsyncMock(side_effect=joint_once)
        with patch("core.rag_pipeline.trace_event") as trace_mock:
            chunks, _search, client = await self._run(
                question="普通员工的出差标准是什么",
                intent={
                    "response_mode": "grounded_qa",
                    "retrieval_policy": "required",
                    "need_retrieval": True,
                    "decision_reason": "business_question",
                },
                results=[*initial, *competitors, out_of_scope_competitor],
                rerank_mock=rerank_mock,
                expansion_mock=expansion_mock,
                joint_mock=joint_mock,
                full_document_mock=full_document_mock,
                task_contract=contract,
                kb_ids=[kb_id],
            )

        rerank_mock.assert_not_awaited()
        expansion_mock.assert_not_awaited()
        full_document_mock.assert_awaited_once()
        self.assertEqual(full_document_mock.await_args.kwargs["max_chunks"], 18)
        self.assertEqual(full_document_mock.await_args.kwargs["max_chars"], 12_000)
        joint_mock.assert_awaited_once()
        joint_candidates = joint_mock.await_args.args[1]
        self.assertEqual(len(joint_candidates), 18)
        self.assertEqual(
            sum(str(item.get("doc_id")) == str(document_id) for item in joint_candidates),
            15,
        )
        self.assertEqual(
            len({
                str(item.get("doc_id"))
                for item in joint_candidates
                if str(item.get("doc_id")) != str(document_id)
            }),
            3,
        )
        self.assertNotIn(
            "不在本次授权知识库范围内的制度",
            {str(item.get("content") or "") for item in joint_candidates},
        )
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "hit")
        self.assertEqual(event["coverage_status"], "complete")
        self.assertTrue(event["expansion_attempted"])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        for expected in ("普通员工", "经济舱", "450元", "100元/天"):
            self.assertIn(expected, prompt_content)
        initial_trace = _trace_event(trace_mock, "rerank.completed")
        self.assertFalse(initial_trace["attempted"])
        self.assertEqual(
            initial_trace["reason"],
            "pre_rerank_dominant_small_document",
        )
        joint_trace = _trace_event(trace_mock, "rerank.joint_completed")
        self.assertEqual(joint_trace["pass_name"], "joint_initial")
        selection_trace = _trace_event(trace_mock, "evidence.selection")
        self.assertIsNone(selection_trace["initial_rerank_succeeded"])
        self.assertTrue(selection_trace["joint_rerank_succeeded"])
        self.assertTrue(selection_trace["rerank_succeeded"])

    async def test_broad_rule_question_uses_selector_without_dedicated_model(self) -> None:
        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        initial, full_document = _travel_small_document(kb_id, document_id)
        competitors = [
            {
                **_candidate(
                    content="请假审批按天数分级处理。",
                    filename="员工请假管理办法.docx",
                ),
                "kb_id": kb_id,
                "active_channels": ["vector"],
            }
        ]
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            source="rule",
            requirements=[
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "普通员工的出差标准是什么",
                },
            ],
        )
        answer_requirement_id = contract.requirements[0].id
        bridge_requirement_id = contract.requirements[1].id
        rerank_mock = AsyncMock(
            side_effect=AssertionError("小文档快速路径不得调用首轮模型重排")
        )
        expansion_mock = AsyncMock(
            side_effect=AssertionError("完整小文档加载后不得再次扩展")
        )

        async def select_once(_query, candidates, requirements, **kwargs):
            self.assertEqual(
                [item.id for item in requirements],
                [answer_requirement_id, bridge_requirement_id],
            )
            self.assertEqual(
                kwargs["bridge_requirement_ids"],
                (bridge_requirement_id,),
            )
            self.assertEqual(kwargs["eligible_candidate_indexes"], tuple(range(1, 16)))
            self.assertEqual(kwargs["anchor_candidate_indexes"], (1, 2, 3))
            selected_indexes = (3, 4, 5, 6, 7, 8, 9, 10)
            selected: list[dict] = []
            for candidate_index in selected_indexes:
                candidate = candidates[candidate_index - 1]
                selected.append({
                    **candidate,
                    "rerank_candidate_index": candidate_index,
                    "topic_relevance": 0.9,
                    "answer_support": 0.9,
                    "constraint_status": "neutral",
                    "evidence_role": "direct",
                    "rerank_status": "verified_joint",
                    "joint_rerank_status": "verified_joint",
                    "contribution_role": (
                        "bridge" if candidate_index == 3 else "complement"
                    ),
                    "supports_requirement_ids": [
                        (
                            bridge_requirement_id
                            if candidate_index == 3
                            else answer_requirement_id
                        )
                    ],
                    "bridge_facts": (
                        [{
                            "subject": "普通员工",
                            "relation": "属于",
                            "object": "D级",
                        }]
                        if candidate_index == 3
                        else []
                    ),
                    "jointly_selected": True,
                    "evidence_set_id": "small_document_set",
                    "joint_support_score": 0.9,
                    "coverage_status": "complete",
                })
            return RerankOutcome(
                results=selected,
                succeeded=True,
                requirements=tuple(requirements),
                coverage_status="complete",
                covered_requirement_ids=(
                    answer_requirement_id,
                    bridge_requirement_id,
                ),
                selected_evidence_set_id="small_document_set",
                selected_candidate_indexes=selected_indexes,
                joint_support_score=0.9,
                model="test-chat",
                prompt_version="small-document-test",
                elapsed_ms=12,
                candidate_count=len(candidates),
            )

        joint_mock = AsyncMock(side_effect=select_once)
        with patch("core.rag_pipeline.trace_event") as trace_mock:
            chunks, _search, client = await self._run(
                question="普通员工的出差标准是什么",
                intent={
                    "response_mode": "grounded_qa",
                    "retrieval_policy": "required",
                    "need_retrieval": True,
                    "decision_reason": "business_question",
                },
                results=[*initial, *competitors],
                rerank_mock=rerank_mock,
                joint_mock=joint_mock,
                expansion_mock=expansion_mock,
                full_document_mock=AsyncMock(return_value=full_document),
                task_contract=contract,
                kb_ids=[kb_id],
            )

        rerank_mock.assert_not_awaited()
        joint_mock.assert_awaited_once()
        expansion_mock.assert_not_awaited()
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "hit")
        self.assertEqual(event["coverage_status"], "complete")
        self.assertGreater(event["direct_evidence_count"], 0)
        self.assertEqual(len(event["answer_sources"]), 8)
        self.assertEqual(
            {item["doc_id"] for item in event["answer_sources"]},
            {str(document_id)},
        )
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        for expected in ("普通员工", "经济舱", "450元", "100元/天"):
            self.assertIn(expected, prompt_content)
        self.assertNotIn("请假审批按天数分级处理", prompt_content)

        rerank_trace = _trace_event(trace_mock, "rerank.completed")
        self.assertFalse(rerank_trace["attempted"])
        self.assertIsNone(rerank_trace["succeeded"])
        self.assertEqual(
            rerank_trace["reason"],
            "pre_rerank_dominant_small_document",
        )
        selection_trace = _trace_event(trace_mock, "evidence.selection")
        self.assertTrue(selection_trace["rerank_succeeded"])
        self.assertEqual(selection_trace["evidence_status"], "hit")
        self.assertEqual(selection_trace["context_count"], 8)

    async def test_invalid_small_document_probe_falls_back_to_initial_rerank(self) -> None:
        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        initial, full_document = _travel_small_document(kb_id, document_id)
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
        )
        requirement_id = contract.requirements[0].id
        verified = {
            **initial[2],
            "rerank_candidate_index": 3,
            "topic_relevance": 0.99,
            "answer_support": 0.95,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "standalone_answer",
            "supports_requirement_ids": [requirement_id],
        }
        invalid_scope = [{**full_document[0], "kb_id": uuid.uuid4()}]
        for label, probe_result in (
            ("empty_or_over_budget", []),
            ("scope_mismatch", invalid_scope),
        ):
            with self.subTest(label=label):
                rerank_mock = AsyncMock(return_value=RerankOutcome(
                    results=[verified],
                    succeeded=True,
                    requirements=(AnswerRequirement(
                        requirement_id,
                        contract.requirements[0].description,
                    ),),
                    expansion_plan=ExpansionPlan(needed=False),
                ))
                full_document_mock = AsyncMock(return_value=probe_result)

                chunks, _search, _client = await self._run(
                    question="普通员工的出差标准是什么",
                    intent={
                        "response_mode": "grounded_qa",
                        "retrieval_policy": "required",
                        "need_retrieval": True,
                        "decision_reason": "business_question",
                    },
                    results=initial,
                    rerank_mock=rerank_mock,
                    full_document_mock=full_document_mock,
                    task_contract=contract,
                    kb_ids=[kb_id],
                )

                full_document_mock.assert_awaited_once()
                rerank_mock.assert_awaited_once()
                self.assertEqual(_search_event(chunks)["evidence_status"], "hit")

    async def test_incomplete_anchor_probe_is_not_reused_after_expansion_error(self) -> None:
        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        initial, full_document = _travel_small_document(kb_id, document_id)
        # 用 18 条优先全文候选占满 merge 预算，并让第一个原始 seed id 消失，
        # 精确模拟异常 loader/并发边界下 anchor 映射不完整的情况。
        probed_document = [dict(item) for item in full_document]
        probed_document[0] = {**probed_document[0], "id": uuid.uuid4()}
        for offset in range(3):
            probed_document.append({
                **full_document[-1],
                "id": uuid.uuid4(),
                "chunk_index": len(full_document) + offset,
                "content": f"附录{offset}",
            })
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
        )
        rerank_mock = AsyncMock(return_value=RerankOutcome(
            results=initial,
            succeeded=False,
            error="TimeoutError: initial rerank timeout",
        ))
        expansion_mock = AsyncMock(
            side_effect=RuntimeError("scoped expansion unavailable")
        )
        joint_mock = AsyncMock(
            side_effect=AssertionError("废弃的全文探测结果不得进入联合重排")
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
            rerank_mock=rerank_mock,
            full_document_mock=AsyncMock(return_value=probed_document),
            expansion_mock=expansion_mock,
            joint_mock=joint_mock,
            task_contract=contract,
            kb_ids=[kb_id],
        )

        rerank_mock.assert_awaited_once()
        expansion_mock.assert_awaited_once()
        joint_mock.assert_not_awaited()
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "error")
        self.assertEqual(event["answer_sources"], [])

    async def test_fast_joint_failure_is_error_without_initial_retry(self) -> None:
        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        initial, full_document = _travel_small_document(kb_id, document_id)
        contract = _task_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
        )
        rerank_mock = AsyncMock(
            side_effect=AssertionError("联合失败后不得补打首轮模型")
        )
        joint_mock = AsyncMock(return_value=RerankOutcome(
            results=full_document,
            succeeded=False,
            error="ValueError: 非法 evidence_role 且 json 修复失败",
            coverage_status="insufficient",
        ))

        chunks, _search, client = await self._run(
            question="普通员工的出差标准是什么",
            intent={
                "response_mode": "grounded_qa",
                "retrieval_policy": "required",
                "need_retrieval": True,
                "decision_reason": "business_question",
            },
            results=initial,
            rerank_mock=rerank_mock,
            full_document_mock=AsyncMock(return_value=full_document),
            joint_mock=joint_mock,
            task_contract=contract,
            kb_ids=[kb_id],
        )

        rerank_mock.assert_not_awaited()
        joint_mock.assert_awaited_once()
        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "error")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("检索或证据验证暂时失败", prompt_content)
        self.assertNotIn("知识库中未找到相关内容", prompt_content)
        self.assertNotIn("D级一线城市450元", prompt_content)

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
        self.assertEqual(event["evidence_status"], "error")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("检索或证据验证暂时失败", prompt_content)
        self.assertNotIn("知识库中未找到相关内容", prompt_content)
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

    def test_incomplete_standalone_direct_does_not_skip_expansion(self) -> None:
        requirements = (
            AnswerRequirement("channel", "确认提交渠道"),
            AnswerRequirement("format", "确认文件格式"),
        )
        partial = {
            **_candidate(content="材料应通过门户提交"),
            "rerank_candidate_index": 1,
            "topic_relevance": 0.98,
            "answer_support": 0.95,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "standalone_answer",
            "supports_requirement_ids": ["channel"],
        }

        plan, resolved_requirements, trigger = _resolve_document_expansion_plan(
            question="材料应从哪里提交，需要什么格式",
            results=[partial],
            outcome=RerankOutcome(
                results=[partial],
                succeeded=True,
                requirements=requirements,
                expansion_plan=ExpansionPlan(needed=False),
            ),
            constraints=extract_query_constraints(
                "材料应从哪里提交，需要什么格式"
            ),
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.target_candidate_indexes, (1,))
        self.assertEqual(plan.missing_requirement_ids, ("format",))
        self.assertEqual(
            tuple(item.id for item in resolved_requirements),
            ("channel", "format"),
        )
        self.assertEqual(trigger, "incomplete_direct_coverage")

    async def test_initial_coverage_is_partial_when_direct_misses_required_id(
        self,
    ) -> None:
        requirements = (
            AnswerRequirement("channel", "确认提交渠道"),
            AnswerRequirement("format", "确认文件格式"),
        )
        partial = {
            **_candidate(content="材料应通过门户提交"),
            # 不提供候选序号，使本用例只验证 coverage 推导而不实际执行扩展。
            "topic_relevance": 0.98,
            "answer_support": 0.95,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "standalone_answer",
            "supports_requirement_ids": ["channel"],
        }

        with patch("core.rag_pipeline.trace_event") as trace_mock:
            await self._run(
                question="材料应从哪里提交，需要什么格式",
                intent={
                    "response_mode": "grounded_qa",
                    "retrieval_policy": "required",
                    "need_retrieval": True,
                    "decision_reason": "business_question",
                },
                results=[partial],
                rerank_outcome=RerankOutcome(
                    results=[partial],
                    succeeded=True,
                    requirements=requirements,
                    expansion_plan=ExpansionPlan(needed=False),
                ),
            )

        coverage = _trace_event(trace_mock, "evidence.coverage_assessed")
        self.assertEqual(coverage["coverage_status"], "partial")
        self.assertEqual(coverage["required_requirement_count"], 2)
        self.assertEqual(coverage["covered_requirement_count"], 1)
        self.assertEqual(coverage["missing_requirement_count"], 1)

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
        with patch("core.rag_pipeline.trace_event") as trace_mock:
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
        self.assertEqual(event["evidence_status"], "error")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("检索或证据验证暂时失败", prompt_content)
        self.assertNotIn("知识库中未找到相关内容", prompt_content)
        self.assertNotIn("普通员工属于D级", prompt_content)
        self.assertNotIn("未经联合验证", prompt_content)
        selection_trace = _trace_event(trace_mock, "evidence.selection")
        self.assertFalse(selection_trace["rerank_succeeded"])
        self.assertTrue(selection_trace["initial_rerank_succeeded"])
        self.assertFalse(selection_trace["joint_rerank_succeeded"])
        self.assertEqual(selection_trace["rerank_error"], "TimeoutError")
        self.assertEqual(selection_trace["joint_rerank_error"], "TimeoutError")
        self.assertEqual(selection_trace["evidence_error_stage"], "joint_rerank")

    async def test_joint_timeout_keeps_verified_initial_answer(self) -> None:
        document_id = uuid.uuid4()
        requirement = AnswerRequirement("answer", "普通员工餐补金额")
        answer = {
            **_candidate(content="D级餐饮补贴为100元/天"),
            "doc_id": document_id,
            "chunk_index": 8,
            "rerank_candidate_index": 1,
            "topic_relevance": 0.99,
            "answer_support": 0.9,
            "constraint_status": "neutral",
            "evidence_role": "direct",
            "rerank_status": "verified",
            "contribution_role": "complement",
            "supports_requirement_ids": ["answer"],
        }
        expansion_detail = {
            **_candidate(content="未经联合验证的补充内容"),
            "doc_id": document_id,
            "candidate_origin": "document_scoped",
        }
        with patch("core.rag_pipeline.trace_event") as trace_mock:
            chunks, _search, client = await self._run(
                question="普通员工的餐补标准是多少",
                intent={
                    "response_mode": "grounded_qa",
                    "retrieval_policy": "required",
                    "need_retrieval": True,
                    "decision_reason": "business_question",
                },
                results=[answer],
                rerank_outcome=RerankOutcome(
                    results=[answer],
                    succeeded=True,
                    requirements=(requirement,),
                    expansion_plan=ExpansionPlan(
                        needed=True,
                        target_candidate_indexes=(1,),
                        queries=("D级餐补标准",),
                        missing_requirement_ids=("answer",),
                    ),
                ),
                expansion_outcome=_expanded_outcome([answer], [expansion_detail]),
                joint_outcome=RerankOutcome(
                    results=[answer, expansion_detail],
                    succeeded=False,
                    error="APITimeoutError: Request timed out.",
                    coverage_status="insufficient",
                ),
            )

        event = _search_event(chunks)
        self.assertEqual(event["evidence_status"], "partial")
        self.assertEqual(event["context_evidence_count"], 1)
        self.assertEqual(event["answer_sources"][0]["chunk_index"], 8)
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("D级餐饮补贴为100元/天", prompt_content)
        self.assertNotIn("未经联合验证的补充内容", prompt_content)
        selection_trace = _trace_event(trace_mock, "evidence.selection")
        self.assertEqual(selection_trace["evidence_status"], "partial")
        self.assertTrue(selection_trace["initial_verified_fallback_used"])
        self.assertFalse(selection_trace["joint_rerank_succeeded"])
        self.assertEqual(
            selection_trace["joint_rerank_error"],
            "APITimeoutError: Request timed out.",
        )

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
        with patch("core.rag_pipeline.trace_event") as trace_mock:
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
        self.assertEqual(event["evidence_status"], "error")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("检索或证据验证暂时失败", prompt_content)
        self.assertNotIn("知识库中未找到相关内容", prompt_content)
        self.assertNotIn("普通员工属于D级", prompt_content)
        selection_trace = _trace_event(trace_mock, "evidence.selection")
        self.assertFalse(selection_trace["rerank_succeeded"])
        self.assertEqual(
            selection_trace["rerank_error"],
            "RuntimeError: scoped search failed",
        )
        self.assertIsNone(selection_trace["joint_rerank_succeeded"])
        self.assertEqual(selection_trace["evidence_error_stage"], "expansion")

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
        self.assertEqual(event["evidence_status"], "error")
        self.assertEqual(event["answer_sources"], [])
        prompt_content = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("检索或证据验证暂时失败", prompt_content)
        self.assertNotIn("知识库中未找到相关内容", prompt_content)
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

    def test_joint_repair_rescues_verified_answer_chunk_omitted_by_model(self) -> None:
        doc_id = uuid.uuid4()
        bridge_id = uuid.uuid4()
        answer_id = uuid.uuid4()
        bridge = {
            **_candidate(content="普通员工属于D级"),
            "id": bridge_id,
            "doc_id": doc_id,
            "rerank_candidate_index": 1,
            "jointly_selected": True,
            "supports_requirement_ids": ["r2"],
            "contribution_role": "bridge",
        }
        omitted_answer = {
            **_candidate(content="D级餐饮补贴为100元/天"),
            "id": answer_id,
            "doc_id": doc_id,
            "rerank_candidate_index": 2,
            "jointly_selected": False,
        }
        initial_answer = {
            **omitted_answer,
            "rerank_status": "verified",
            "topic_relevance": 1.0,
            "answer_support": 0.9,
            "constraint_status": "neutral",
            "evidence_role": "related",
            "contribution_role": "complement",
            "supports_requirement_ids": ["r1"],
        }
        outcome = RerankOutcome(
            results=[bridge, omitted_answer],
            succeeded=True,
            coverage_status="partial",
            selected_evidence_set_id="set_1",
            selected_candidate_indexes=(1,),
            covered_requirement_ids=("r2",),
            missing_requirement_ids=("r1",),
            joint_support_score=0.95,
        )

        rescued = _rescue_missing_joint_evidence(
            outcome,
            [initial_answer],
            (
                AnswerRequirement("r1", "查询餐补金额"),
                AnswerRequirement("r2", "确认普通员工对应D级", "helpful", "inferred"),
            ),
        )

        self.assertEqual(rescued.coverage_status, "complete")
        self.assertEqual(rescued.missing_requirement_ids, ())
        self.assertEqual(rescued.selected_candidate_indexes, (1, 2))
        selected_answer = rescued.results[1]
        self.assertTrue(selected_answer["jointly_selected"])
        self.assertEqual(selected_answer["evidence_role"], "direct")
        self.assertEqual(selected_answer["supports_requirement_ids"], ["r1"])

    def test_joint_timeout_fallback_accepts_verified_bridge_and_complement_set(self) -> None:
        doc_id = uuid.uuid4()
        bridge = {
            **_candidate(content="普通员工属于D级"),
            "doc_id": doc_id,
            "rerank_status": "verified",
            "topic_relevance": 0.98,
            "answer_support": 0.72,
            "constraint_status": "neutral",
            "evidence_role": "related",
            "contribution_role": "bridge",
            "supports_requirement_ids": ["r2"],
        }
        answer = {
            **_candidate(content="D级餐饮补贴为100元/天"),
            "doc_id": doc_id,
            "rerank_status": "verified",
            "topic_relevance": 1.0,
            "answer_support": 0.9,
            "constraint_status": "neutral",
            "evidence_role": "related",
            "contribution_role": "complement",
            "supports_requirement_ids": ["r1"],
        }

        fallback, available = _fallback_to_initial_verified_evidence(
            [bridge, answer],
            (
                AnswerRequirement("r1", "查询普通员工餐补金额"),
                AnswerRequirement("r2", "确认普通员工对应D级", "helpful", "inferred"),
            ),
            bridge_requirement_ids=("r2",),
        )

        self.assertTrue(available)
        self.assertEqual(sum(bool(item.get("jointly_selected")) for item in fallback), 2)
        self.assertEqual(
            {item["contribution_role"] for item in fallback if item.get("jointly_selected")},
            {"bridge", "complement"},
        )
        _missing_bridge, available_without_bridge = (
            _fallback_to_initial_verified_evidence(
                [answer],
                (
                    AnswerRequirement("r1", "查询普通员工餐补金额"),
                    AnswerRequirement(
                        "r2",
                        "确认普通员工对应D级",
                        "helpful",
                        "inferred",
                    ),
                ),
                bridge_requirement_ids=("r2",),
            )
        )
        self.assertFalse(available_without_bridge)

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
