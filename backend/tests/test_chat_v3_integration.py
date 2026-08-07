"""Regression coverage for the V3 semantic entry at the chat boundary.

These tests intentionally exercise ``api.chat.send_message`` with in-memory
sessions only.  The V3 model result is compiled from an in-process source
catalog, so the assertions cover the production hand-off without a database,
network model call, or retrieval backend.
"""

from __future__ import annotations

import asyncio
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api.chat import _v3_active_timeout_seconds, send_message
from core.active_task_state import ResolvedActiveTask, build_active_task_state
from core.active_task_state import parse_active_task_state
from core.semantic_memory import extract_resolved_entity_memory
from core.clarification import ClarificationContract, build_clarification_state
from core.conversation_context import ConversationContext, RouteTurnCandidate
from core.query_analysis_execution import (
    build_execution_baseline,
    build_execution_clarification_baseline,
    evaluate_query_execution_gate,
)
from core.query_route_compiler import (
    RouteCategoryPolicy,
    RouteCompilerConfig,
    compile_rag_task_contract,
)
from core.query_route_contract import parse_rag_route_decision
from core.query_understanding_v3_analyzer import QueryUnderstandingV3RunResult
from core.query_understanding_v3_catalog import SourceSpanCatalog
from core.query_understanding_v3_compiler import (
    QueryUnderstandingV3ExecutionValidation,
    compile_query_understanding,
)
from core.query_understanding_v3_contract import parse_query_understanding
from core.query_understanding_v3_execution import (
    QueryUnderstandingV3ContextSelection,
    QueryUnderstandingV3ExecutionResult,
)
from core.rag_v2.contracts import QueryPlanV2
from core.rag_v2.pipeline import AnchorRetrievalSnapshot
from models.schemas import ChatRequest


class _RequestDB:
    """Small non-durable request-session fake used by legacy chat tests too."""

    def __init__(self, conversation: SimpleNamespace) -> None:
        self.conversation = conversation
        self.added: list[object] = []
        self.commit_count = 0

    async def get(self, _model: object, _identity: object) -> SimpleNamespace:
        return self.conversation

    def add(self, value: object) -> None:
        self.added.append(value)

    def add_all(self, values: list[object]) -> None:
        self.added.extend(values)

    async def commit(self) -> None:
        self.commit_count += 1


class _SaveDB:
    """Independent response-save session; it never talks to a real database."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.commit_count = 0

    async def __aenter__(self) -> "_SaveDB":
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> bool:
        return False

    async def get(self, _model: object, _identity: object) -> None:
        return None

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        self.commit_count += 1


class _PersistingSaveDB:
    """Save session whose ``get`` returns the mutable conversation object."""

    def __init__(self, conversation: SimpleNamespace) -> None:
        self.conversation = conversation
        self.added: list[object] = []

    async def __aenter__(self) -> "_PersistingSaveDB":
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> bool:
        return False

    async def get(self, _model: object, _identity: object) -> SimpleNamespace:
        return self.conversation

    def add(self, value: object) -> None:
        self.added.append(value)

    async def commit(self) -> None:
        return None


def _routing_result(
    *,
    selected_kb_count: int,
    readiness: str = "ready",
    clarification: dict | None = None,
) -> SimpleNamespace:
    """Build a route/RBAC-authorized knowledge-QA contract without routing."""

    if clarification is None:
        clarification = {"question": "", "unresolved": []}

    route = parse_rag_route_decision(
        {
            "schema_version": "rag_route_decision.v1",
            "readiness": readiness,
            "intent_code": "knowledge_qa",
            "relation": "new",
            "evidence_scope": "enterprise_kb",
            "query_resolution": {"mode": "current", "context_turn_keys": []},
            "requirements": [
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "回答当前输入",
                }
            ],
            "clarification": clarification,
            "confidence": 0.98,
            "rationale": "V3 chat integration fixture",
        },
        allowed_intent_codes=["knowledge_qa"],
    )
    contract = compile_rag_task_contract(
        route,
        RouteCategoryPolicy(
            code="knowledge_qa",
            name="知识问答",
            action="retrieve",
        ),
        RouteCompilerConfig(),
        question="当前输入",
        selected_kb_count=selected_kb_count,
        source="test",
    )
    decision_payload = {
        "intent_code": "knowledge_qa",
        "intent_name": "知识问答",
        "action": "retrieve",
        "confidence": 0.98,
        "source": "test",
        "response_mode": contract.response_mode,
        "retrieval_policy": contract.retrieval_policy,
        "need_retrieval": contract.need_retrieval,
        "decision_reason": contract.decision_reason,
    }
    return SimpleNamespace(
        decision=SimpleNamespace(
            need_retrieval=True,
            decision_reason=contract.decision_reason,
            to_dict=lambda: dict(decision_payload),
        ),
        route_decision=route,
        task_contract=contract,
        diagnostics={},
        route_log_id=None,
    )


def _v3_settings(*, anchor_prefetch_enabled: bool) -> SimpleNamespace:
    """Keep legacy analysis enabled in config to prove V3 does not invoke it."""

    return SimpleNamespace(
        rag_trace_include_content=False,
        rag_pipeline_version="v2",
        rag_semantic_entry="v3",
        rag_query_understanding_v3_mode="active",
        rag_query_understanding_v3_active_timeout_seconds=0.5,
        rag_query_understanding_v3_active_max_inflight=1,
        rag_query_understanding_v3_anchor_prefetch_enabled=anchor_prefetch_enabled,
        rag_query_understanding_v3_anchor_prefetch_timeout_seconds=0.5,
        # Deliberately active: V3 must still be the only semantic authority.
        rag_query_analyzer_mode="active",
        rag_query_analyzer_active_timeout_seconds=0.5,
        rag_query_analyzer_active_max_inflight=1,
        rag_query_analyzer_shadow_timeout_seconds=0.5,
        rag_query_analyzer_shadow_max_inflight=1,
        rag_query_analyzer_shadow_sample_rate=1.0,
    )


def _span_id(catalog: SourceSpanCatalog, text: str, *, source_key: str) -> str:
    return next(
        item.span_id
        for item in catalog.entries
        if item.source_key == source_key and item.text == text
    )


def _compiled_v3_result(
    *,
    baseline,
    target_texts: tuple[str, ...],
    qualifier_text: str | None = None,
    qualifier_source_key: str = "current",
    knowledge_request: dict | None = None,
) -> QueryUnderstandingV3ExecutionResult:
    """Compile a real catalog-bound V3 result for the supplied chat baseline."""

    question = baseline.fallback.question
    catalog = SourceSpanCatalog.build(
        current_question=question,
        route_context=baseline.fallback.route_context,
    )
    qualifier_span_ids = (
        [_span_id(catalog, qualifier_text, source_key=qualifier_source_key)]
        if qualifier_text is not None
        else []
    )
    raw = {
        "schema_version": "query_understanding.v3",
        "answer_candidates": [
            {
                "id": f"a{index}",
                "target_span_id": _span_id(
                    catalog,
                    target,
                    source_key="current",
                ),
                "qualifier_span_ids": list(qualifier_span_ids),
            }
            for index, target in enumerate(target_texts, start=1)
        ],
        "knowledge_request": knowledge_request or {
            "resource": "document_content",
            "operation": "answer",
            "filter_span_ids": [],
            "group_by": "none",
            "status_filter": "any",
            "result_handles": [],
        },
    }
    understanding = parse_query_understanding(
        json.dumps(raw, ensure_ascii=False),
        catalog=catalog,
    )
    compilation = compile_query_understanding(
        catalog=catalog,
        understanding=understanding,
        baseline_floor=baseline.floor,
    )
    if compilation.used_fallback:  # pragma: no cover - fixture invariant
        raise AssertionError(
            f"fixture V3 candidate unexpectedly rejected: "
            f"{compilation.validation.reason}"
        )
    selected_baseline = build_execution_baseline(
        plan=compilation.plan,
        local_surface_plan=baseline.fallback.local_surface_plan,
        contextual_plan=baseline.fallback.contextual_plan,
        question=baseline.fallback.question,
        standalone_query=baseline.fallback.standalone_query,
        route_context=baseline.fallback.route_context,
        deterministic_is_followup=baseline.fallback.deterministic_is_followup,
        execution_bundle=compilation.execution_bundle,
    )
    execution_gate = evaluate_query_execution_gate(selected_baseline)
    if execution_gate.needs_clarification:  # pragma: no cover - invariant
        raise AssertionError("compiled V3 fixture must be executable")
    analysis_result = QueryUnderstandingV3RunResult(
        mode="active",
        catalog=catalog,
        analysis=understanding,
        model="test-v3-model",
        latency_ms=1,
    )
    return QueryUnderstandingV3ExecutionResult(
        decision="applied",
        reason="catalog_bound_candidate_compiled",
        request_baseline=baseline,
        selected_baseline=selected_baseline,
        query_execution_gate=execution_gate,
        analysis_result=analysis_result,
        validation=compilation.validation,
        compilation=compilation,
        context_selection=QueryUnderstandingV3ContextSelection(
            current_question=question,
            selected_context_turn_keys=understanding.referenced_context_keys,
        ),
    )


def _fallback_v3_result(*, baseline) -> QueryUnderstandingV3ExecutionResult:
    """Return a normal V3 fallback, preserving the local execution floor."""

    gate = evaluate_query_execution_gate(baseline.fallback)
    return QueryUnderstandingV3ExecutionResult(
        decision=("clarification" if gate.needs_clarification else "fallback"),
        reason="model_unavailable",
        request_baseline=baseline,
        selected_baseline=baseline.fallback,
        query_execution_gate=gate,
    )


def _not_ready_plan(question: str) -> QueryPlanV2:
    """A deliberately unresolved local floor for the fallback regression."""

    return QueryPlanV2(
        original_query=question,
        answer_shape="unknown",
        retrieval_queries=(),
        requirements=(),
        confidence=0.0,
        source="fallback",
        reason="test_local_not_ready",
        needs_clarification=True,
        clarification_question="请补充需要查询的具体对象。",
    )


async def _sse_events(response) -> list[dict]:
    events: list[dict] = []
    async for chunk in response.body_iterator:
        payload = json.loads(str(chunk).removeprefix("data: ").strip())
        events.append(payload)
    return events


class ChatV3IntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_v3_active_timeout_uses_global_llm_timeout_when_unset(self) -> None:
        settings = SimpleNamespace(
            rag_query_understanding_v3_active_timeout_seconds=None,
            llm_request_timeout_seconds=60.0,
        )

        self.assertEqual(_v3_active_timeout_seconds(settings), 60.0)

    async def test_catalog_capability_skips_vector_prefetch_and_v2_runner(self) -> None:
        question = "我现在有关于云枢配置的知识库有几个文章"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=question,
            history_messages=(),
            carryover_sources=(),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
            active_task_state=None,
            active_task_revision=0,
        )
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        catalog_calls: list[dict] = []

        class V3Service:
            async def run_active(self, **kwargs):
                catalog = SourceSpanCatalog.build(
                    current_question=kwargs["baseline"].fallback.question,
                    route_context=kwargs["baseline"].fallback.route_context,
                )
                return _compiled_v3_result(
                    baseline=kwargs["baseline"],
                    target_texts=("知识库有几个文章",),
                    knowledge_request={
                        "resource": "document_catalog",
                        "operation": "count",
                        "filter_span_ids": [
                            _span_id(catalog, "云枢配置", source_key="current")
                        ],
                        "group_by": "none",
                        "status_filter": "any",
                    },
                )

        async def catalog_stream(**kwargs):
            catalog_calls.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": True,
                "evidence_status": "no_hit",
                "displayed_result_count": 0,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
            }, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({
                "type": "text_delta",
                "content": "当前授权范围内没有匹配文章。",
            }, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({
                "type": "done",
                "conversation_id": str(conversation_id),
            }) + "\n\n"

        no_prefetch = AsyncMock(
            side_effect=AssertionError("catalog request must not prefetch vectors")
        )

        async def no_v2_stream(**_kwargs):
            raise AssertionError("catalog request must not enter V2 retrieval")
            yield  # pragma: no cover

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=_routing_result(selected_kb_count=1)),
            ),
            patch(
                "api.chat.get_settings",
                return_value=_v3_settings(anchor_prefetch_enabled=True),
            ),
            patch(
                "api.chat.get_query_analysis_execution_service",
                side_effect=AssertionError("V3 request must not invoke legacy analysis"),
            ),
            patch(
                "api.chat.get_query_understanding_v3_execution_service",
                return_value=V3Service(),
            ),
            patch(
                "api.chat.apply_v3_catalog_context_selection",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.retrieve_anchor_retrieval_snapshot", new=no_prefetch),
            patch("api.chat.run_rag_v2_stream", new=no_v2_stream),
            patch("api.chat.run_knowledge_catalog_stream", new=catalog_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        no_prefetch.assert_not_awaited()
        self.assertEqual(len(catalog_calls), 1)
        self.assertEqual(catalog_calls[0]["kb_ids"], [kb_id])
        self.assertNotIn("execution_bundle", catalog_calls[0])
        request = catalog_calls[0]["knowledge_request"]
        self.assertTrue(request.is_catalog_operation)
        self.assertEqual(request.operation, "count")
        self.assertEqual(request.filter_terms, ("云枢配置",))

    async def test_result_reference_dispatches_only_the_selected_prior_document(self) -> None:
        question = "我想看第一个文章"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        first_doc_id = uuid.uuid4()
        second_doc_id = uuid.uuid4()
        prior_sources = tuple({
            "source_kind": "document_metadata",
            "id": str(doc_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
            "filename": filename,
            "status": "ready",
            "evidence_role": "direct",
        } for doc_id, filename in (
            (first_doc_id, "第一篇.md"),
            (second_doc_id, "第二篇.md"),
        ))
        candidate = RouteTurnCandidate(
            candidate_key="t1",
            user_question="列出当前文章",
            assistant_answer="1. 第一篇.md\n2. 第二篇.md",
            raw_sources=prior_sources,
        )
        context = ConversationContext(
            is_followup=True,
            followup_reason="result_reference",
            standalone_query=question,
            history_messages=(),
            carryover_sources=prior_sources,
            route_turn_candidates=(candidate,),
            relation="followup",
            query_resolution_mode="v3_catalog",
            context_turn_keys=("t1",),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
            active_task_state=None,
            active_task_revision=0,
        )
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        result_calls: list[dict] = []

        class V3Service:
            async def run_active(self, **kwargs):
                if not kwargs["baseline"].fallback.route_context[0].get("result_items"):
                    raise AssertionError(kwargs["baseline"].fallback.route_context)
                return _compiled_v3_result(
                    baseline=kwargs["baseline"],
                    target_texts=(question,),
                    knowledge_request={
                        "resource": "document_result",
                        "operation": "read",
                        "filter_span_ids": [],
                        "group_by": "none",
                        "status_filter": "any",
                        "result_handles": ["r_t1_001"],
                    },
                )

        async def result_stream(**kwargs):
            result_calls.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": True,
                "evidence_status": "no_hit",
                "displayed_result_count": 0,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
            }) + "\n\n"
            yield "data: " + json.dumps({
                "type": "text_delta",
                "content": "文档当前不可读。",
            }, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({
                "type": "done",
                "conversation_id": str(conversation_id),
            }) + "\n\n"

        async def no_v2_stream(**_kwargs):
            raise AssertionError("result reference must not enter vector retrieval")
            yield  # pragma: no cover

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=AsyncMock(return_value=context)),
            patch("api.chat.classify_intent_result", new=AsyncMock(return_value=_routing_result(selected_kb_count=1))),
            patch("api.chat.get_settings", return_value=_v3_settings(anchor_prefetch_enabled=False)),
            patch("api.chat.get_query_understanding_v3_execution_service", return_value=V3Service()),
            patch("api.chat.apply_v3_catalog_context_selection", new=AsyncMock(return_value=context)),
            patch("api.chat.run_rag_v2_stream", new=no_v2_stream),
            patch("api.chat.run_knowledge_result_stream", new=result_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        self.assertEqual(len(result_calls), 1, trace.call_args_list)
        call = result_calls[0]
        self.assertTrue(call["knowledge_request"].is_result_operation)
        self.assertEqual(call["knowledge_request"].result_handles, ("r_t1_001",))
        self.assertEqual(len(call["result_sources"]), 1)
        self.assertEqual(call["result_sources"][0]["doc_id"], str(first_doc_id))

    async def test_verified_missing_object_followup_does_not_wait_for_v3_model(self) -> None:
        question = "应该如何配置"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        source = {
            "id": str(uuid.uuid4()),
            "doc_id": str(uuid.uuid4()),
            "kb_id": str(kb_id),
            "content": "force_change_default_password: true # 默认密码强制修改",
        }
        context = ConversationContext(
            is_followup=True,
            followup_reason="missing_action_object",
            standalone_query="应该如何配置默认密码强制修改",
            history_messages=(
                {"role": "user", "content": "默认密码强制修改"},
                {"role": "assistant", "content": "已找到对应资料"},
            ),
            carryover_sources=(source,),
            relation="followup",
            query_resolution_mode="contextualize",
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
            active_task_state=None,
            active_task_revision=0,
        )
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        routing_result = _routing_result(selected_kb_count=1)
        v2_calls: list[dict] = []

        async def v2_stream(**kwargs):
            v2_calls.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": True,
                "evidence_status": "no_hit",
                "displayed_result_count": 0,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
            }) + "\n\n"
            yield "data: " + json.dumps({
                "type": "done",
                "conversation_id": str(conversation_id),
            }) + "\n\n"

        v3_service = SimpleNamespace(
            run_active=AsyncMock(
                side_effect=AssertionError("verified follow-up must skip V3")
            )
        )
        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch(
                "api.chat.get_settings",
                return_value=_v3_settings(anchor_prefetch_enabled=False),
            ),
            patch(
                "api.chat.get_query_understanding_v3_execution_service",
                return_value=v3_service,
            ),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        v3_service.run_active.assert_not_awaited()
        self.assertEqual(len(v2_calls), 1)
        self.assertEqual(
            v2_calls[0]["standalone_query"],
            "应该如何配置默认密码强制修改",
        )
        self.assertEqual(v2_calls[0]["conversation_history"], [])
        self.assertEqual(v2_calls[0]["carryover_sources"], [source])
        self.assertTrue(v2_calls[0]["is_followup"])

    async def test_same_active_result_reference_stays_on_selected_document(self) -> None:
        question = "第四个里面这么点东西吗"
        root_query = "我要看第四个"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        source_turn_id = uuid.uuid4()
        source = {
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
            "filename": "已选择的文档.md",
            "content": "该文档的已授权正文",
            "evidence_role": "direct",
            "candidate_origin": "active_task_state",
        }
        state = build_active_task_state(
            root_query=root_query,
            answer_shape="overview",
            sources=[source],
            source_turn_id=source_turn_id,
            trace_id=uuid.uuid4().hex,
        )
        resolved_task = ResolvedActiveTask(
            state=state,
            sources=(source,),
            kb_ids=(kb_id,),
            doc_ids=(doc_id,),
        )
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=question,
            history_messages=(),
            carryover_sources=(),
            route_turn_candidates=(RouteTurnCandidate(
                candidate_key="t1",
                user_question=root_query,
                assistant_answer="已打开第 4 篇文档。",
                raw_sources=(source,),
                assistant_turn_id=source_turn_id,
            ),),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
            active_task_state=state.to_dict(),
            active_task_revision=state.revision,
        )
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        v2_calls: list[dict] = []

        async def v2_stream(**kwargs):
            v2_calls.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": True,
                "evidence_status": "no_hit",
                "displayed_result_count": 0,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
            }) + "\n\n"
            yield "data: " + json.dumps({
                "type": "done",
                "conversation_id": str(conversation_id),
            }) + "\n\n"

        v3_service = SimpleNamespace(
            run_active=AsyncMock(
                side_effect=AssertionError(
                    "an authorized active result reference must skip V3"
                )
            )
        )
        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.resolve_active_task_state",
                new=AsyncMock(return_value=resolved_task),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=_routing_result(selected_kb_count=1)),
            ),
            patch(
                "api.chat.get_settings",
                return_value=_v3_settings(anchor_prefetch_enabled=False),
            ),
            patch(
                "api.chat.get_query_understanding_v3_execution_service",
                return_value=v3_service,
            ),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        v3_service.run_active.assert_not_awaited()
        self.assertEqual(len(v2_calls), 1)
        self.assertIs(v2_calls[0]["active_task_scope"], resolved_task)
        self.assertEqual(v2_calls[0]["active_task_scope"].doc_ids, (doc_id,))
        self.assertEqual(v2_calls[0]["carryover_sources"], [source])
        self.assertEqual(
            v2_calls[0]["followup_reason"],
            "active_task_state:active_result_reference",
        )
        self.assertIn(root_query, v2_calls[0]["standalone_query"])

    async def test_partial_unverified_answer_persists_entity_memory(self) -> None:
        # 上一轮是 unverified 自动回答（coverage_sufficient_answer）也必须写入
        # active_task_state + semantic_memory；否则下一轮实体复用无记忆可用，
        # 又回到“检索不到就澄清”的旧循环。
        question = "普通员工出差时可以乘坐的交通工具有哪些"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=question,
            history_messages=(),
            carryover_sources=(),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
            active_task_state=None,
            active_task_revision=0,
        )
        request_db = _RequestDB(conversation)
        save_db = _PersistingSaveDB(conversation)
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        source_item = {
            "source_kind": "document_chunk",
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
            "content": (
                "| 职级 | 适用人员 |\n"
                "| --- | --- |\n"
                "| D级 | 普通员工、专员 |"
            ),
            "chunk_index": 2,
            "metadata": {},
            "filename": "公司出差管理标准.docx",
            "file_type": "docx",
            "evidence_role": "unverified",
            "source_verification": "unverified",
        }

        class V3Service:
            async def run_active(self, **kwargs):
                return _compiled_v3_result(
                    baseline=kwargs["baseline"],
                    target_texts=("普通员工出差",),
                )

        async def v2_stream(**kwargs):
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [{**source_item}],
                "answer_sources": [{**source_item}],
                "retrieval_executed": True,
                "evidence_status": "partial",
                "coverage_status": "insufficient",
                "unverified_generation": True,
                "source_verification": "unverified",
                "displayed_result_count": 1,
                "direct_evidence_count": 0,
                "related_reference_count": 1,
            }, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({
                "type": "text_delta",
                "content": "D级适用普通员工、专员。",
            }, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({
                "type": "done",
                "conversation_id": str(conversation_id),
            }) + "\n\n"

        from datetime import datetime, timezone as _tz

        fake_turn = SimpleNamespace(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            user_id=user_id,
            request_id="req-entity-memory",
            trace_id="trace-entity-memory",
            status="accepted",
            evidence_status=None,
            retrieval_executed=None,
            error_code=None,
            answer_content=None,
            answer_sources=None,
            search_snapshot=None,
            tokens=None,
            user_message_id=None,
            assistant_message_id=None,
            execution_attempts=1,
            lease_owner=None,
            lease_expires_at=None,
            created_at=datetime.now(_tz.utc),
            updated_at=datetime.now(_tz.utc),
            generated_at=None,
            completed_at=None,
        )

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.resolve_active_task_state",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=_routing_result(selected_kb_count=1)),
            ),
            patch(
                "api.chat.get_settings",
                return_value=_v3_settings(anchor_prefetch_enabled=False),
            ),
            patch(
                "api.chat.get_query_understanding_v3_execution_service",
                return_value=V3Service(),
            ),
            patch(
                "api.chat._validate_stream_answer_sources",
                new=AsyncMock(return_value=(
                    [dict(source_item)],
                    {(doc_id, chunk_id)},
                    None,
                )),
            ),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat._durable_turn_supported", return_value=True),
            patch(
                "api.chat.reserve_turn",
                new=AsyncMock(return_value=(fake_turn, True)),
            ),
            patch("api.chat.find_turn_for_user", new=AsyncMock(return_value=None)),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        persisted = conversation.active_task_state
        self.assertIsNotNone(persisted)
        assert persisted is not None
        parsed = parse_active_task_state(persisted)
        self.assertIsNotNone(parsed)
        assert parsed is not None and parsed.semantic_memory is not None
        self.assertEqual(
            [(f.mention, f.attribute, f.value) for f in parsed.semantic_memory.facts],
            [("普通员工", "职级", "D级")],
        )
        self.assertEqual(parsed.root_query, question)
        self.assertEqual(parsed.selected_chunk_ids, (chunk_id,))

    async def test_standalone_question_reusing_resolved_entity_skips_v3(self) -> None:
        # 上一轮已确认 普通员工=D级（来源：公司出差管理标准.docx）。
        # 下一轮“普通员工可以乘坐头等舱吗”是独立问题，但复用该实体：锚定来源、
        # 保持当前问题为检索锚点，并且不再进入 V3/澄清循环。
        question = "普通员工可以乘坐头等舱吗"
        root_query = "普通员工出差时可以乘坐的交通工具有哪些"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        source_turn_id = uuid.uuid4()
        source = {
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
            "filename": "公司出差管理标准.docx",
            "content": (
                "【公司出差管理标准.docx › 二、职级分类】\n"
                "| 职级 | 适用人员 |\n"
                "| --- | --- |\n"
                "| D级 | 普通员工、专员 |"
            ),
            "evidence_role": "direct",
            "candidate_origin": "active_task_state",
        }
        memory = extract_resolved_entity_memory(
            sources=[source],
            question=root_query,
            source_turn_id=source_turn_id,
            trace_id=uuid.uuid4().hex,
        )
        self.assertIsNotNone(memory)
        state = build_active_task_state(
            root_query=root_query,
            answer_shape="fact",
            sources=[source],
            source_turn_id=source_turn_id,
            trace_id=uuid.uuid4().hex,
            semantic_memory=memory,
        )
        resolved_task = ResolvedActiveTask(
            state=state,
            sources=(source,),
            kb_ids=(kb_id,),
            doc_ids=(doc_id,),
        )
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=question,
            history_messages=(),
            carryover_sources=(),
            route_turn_candidates=(RouteTurnCandidate(
                candidate_key="t1",
                user_question=root_query,
                assistant_answer="普通员工属于D级，国内外航班均为经济舱。",
                raw_sources=(source,),
                assistant_turn_id=source_turn_id,
            ),),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
            active_task_state=state.to_dict(),
            active_task_revision=state.revision,
        )
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        v2_calls: list[dict] = []

        async def v2_stream(**kwargs):
            v2_calls.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": True,
                "evidence_status": "no_hit",
                "displayed_result_count": 0,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
            }) + "\n\n"
            yield "data: " + json.dumps({
                "type": "done",
                "conversation_id": str(conversation_id),
            }) + "\n\n"

        v3_service = SimpleNamespace(
            run_active=AsyncMock(
                side_effect=AssertionError(
                    "entity reuse binds to the active task and must skip V3"
                )
            )
        )
        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.resolve_active_task_state",
                new=AsyncMock(return_value=resolved_task),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=_routing_result(selected_kb_count=1)),
            ),
            patch(
                "api.chat.get_settings",
                return_value=_v3_settings(anchor_prefetch_enabled=False),
            ),
            patch(
                "api.chat.get_query_understanding_v3_execution_service",
                return_value=v3_service,
            ),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        v3_service.run_active.assert_not_awaited()
        self.assertEqual(len(v2_calls), 1)
        self.assertIs(v2_calls[0]["active_task_scope"], resolved_task)
        self.assertEqual(v2_calls[0]["carryover_sources"], [source])
        self.assertEqual(
            v2_calls[0]["followup_reason"],
            "active_task_state:semantic_entity_reuse",
        )
        self.assertEqual(v2_calls[0]["standalone_query"], question)
        self.assertTrue(v2_calls[0]["is_followup"])

    async def test_pending_route_reply_keeps_original_task_when_v3_falls_back(self) -> None:
        """A clarification value must never become the fallback retrieval query."""

        reply = "云枢的"
        original_query = "我现在想改验证码有效期时间"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = build_clarification_state(
            contract=ClarificationContract(
                adapter="semantic",
                dimension="system_or_product",
                reason_code="missing_context",
                selection_mode="refine",
            ),
            original_query=original_query,
            selected_kb_ids=[kb_id],
            base_user_message_id=uuid.uuid4(),
            clarification_message_id=uuid.uuid4(),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=1,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=reply,
            history_messages=(),
            carryover_sources=(),
            pending_route_state=pending,
        )
        v2_calls: list[dict] = []
        route_calls: list[str] = []

        class FallbackService:
            async def run_active(self, **kwargs):
                return _fallback_v3_result(baseline=kwargs["baseline"])

        async def classify(_db, question, **_kwargs):
            route_calls.append(question)
            return _routing_result(selected_kb_count=1)

        async def v2_stream(**kwargs):
            v2_calls.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": True,
                "evidence_status": "no_hit",
                "displayed_result_count": 0,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
            }) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=AsyncMock(return_value=context)),
            patch("api.chat.classify_intent_result", new=classify),
            patch("api.chat.get_settings", return_value=_v3_settings(anchor_prefetch_enabled=False)),
            patch(
                "api.chat.get_query_analysis_execution_service",
                side_effect=AssertionError("V3 request must not invoke legacy analysis"),
            ),
            patch(
                "api.chat.get_query_understanding_v3_execution_service",
                return_value=FallbackService(),
            ),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=reply,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        self.assertEqual(len(route_calls), 1)
        self.assertIn(original_query, route_calls[0])
        self.assertIn(reply, route_calls[0])
        self.assertEqual(len(v2_calls), 1)
        handoff = v2_calls[0]
        self.assertEqual(handoff["question"], route_calls[0])
        self.assertEqual(handoff["standalone_query"], route_calls[0])
        planned_query = handoff["execution_bundle"].plan.original_query
        self.assertEqual(planned_query, original_query)
        self.assertNotIn(reply, planned_query)
        self.assertNotEqual(handoff["question"], reply)

    async def test_v3_defers_route_semantic_clarification_but_keeps_hard_contract_boundary(self) -> None:
        """A route model cannot preempt the source-bound V3 semantic entry."""

        question = "普通员工的餐补是多少？"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=question,
            history_messages=(),
            carryover_sources=(),
        )
        routing_result = _routing_result(
            selected_kb_count=1,
            readiness="needs_clarification",
            clarification={
                "question": "请补充你要查询的对象。",
                "unresolved": [
                    {"role": "subject", "reason": "missing", "candidate_keys": []}
                ],
            },
        )
        v2_calls: list[dict] = []

        class V3Service:
            async def run_active(self, **kwargs):
                return _compiled_v3_result(
                    baseline=kwargs["baseline"],
                    target_texts=("餐补",),
                    qualifier_text="普通员工",
                )

        async def v2_stream(**kwargs):
            v2_calls.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": True,
                "evidence_status": "no_hit",
                "displayed_result_count": 0,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
            }) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "text_delta", "content": "已交由 V3/V2 处理"},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=AsyncMock(return_value=context)),
            patch("api.chat.classify_intent_result", new=AsyncMock(return_value=routing_result)),
            patch("api.chat.get_settings", return_value=_v3_settings(anchor_prefetch_enabled=False)),
            patch(
                "api.chat.get_query_analysis_execution_service",
                side_effect=AssertionError("V3 request must not invoke legacy analysis"),
            ),
            patch("api.chat.get_query_understanding_v3_execution_service", return_value=V3Service()),
            patch("api.chat.apply_v3_catalog_context_selection", new=AsyncMock(return_value=context)),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        self.assertEqual(len(v2_calls), 1)
        handoff = v2_calls[0]
        self.assertTrue(handoff["task_contract"].dispatch_authorized)
        self.assertEqual(
            handoff["task_contract"].decision_reason,
            "v3_semantic_entry_deferred",
        )
        # The runner sees a current-turn policy shell; the original route
        # semantic clarification remains diagnostic-only in the intent event.
        self.assertEqual(handoff["task_contract"].context_turn_keys, ())
        self.assertEqual(handoff["task_contract"].relation, "new")
        self.assertFalse(
            handoff["intent"]["route_task_contract"]["dispatch_authorized"]
        )
        self.assertEqual(
            handoff["intent"]["semantic_entry_gate"]["disposition"],
            "defer_to_v3",
        )

    async def test_v3_multi_target_context_handoff_skips_legacy_analysis(self) -> None:
        """Only catalog-bound V3 may apply semantics and history to V2."""

        question = "住宿标准、餐补和出差补贴分别是多少"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        history_source = {
            "id": str(uuid.uuid4()),
            "doc_id": str(uuid.uuid4()),
            "kb_id": str(kb_id),
            "content": "普通员工对应D级。",
        }
        candidate = RouteTurnCandidate(
            candidate_key="t1",
            user_question="普通员工的出差标准是什么",
            assistant_answer="普通员工对应D级。",
            raw_sources=(history_source,),
        )
        initial_context = ConversationContext(
            is_followup=True,
            followup_reason="legacy_route_projection",
            standalone_query="普通员工的住宿标准、餐补和出差补贴分别是多少",
            history_messages=(
                {"role": "user", "content": candidate.user_question},
                {"role": "assistant", "content": candidate.assistant_answer or ""},
            ),
            carryover_sources=(history_source,),
            route_turn_candidates=(candidate,),
            relation="continuation",
            query_resolution_mode="contextualize",
            context_turn_keys=("t1",),
        )
        selected_context = ConversationContext(
            is_followup=True,
            followup_reason="query_understanding_v3_contextual",
            standalone_query=question,
            history_messages=(
                {"role": "user", "content": candidate.user_question},
                {"role": "assistant", "content": candidate.assistant_answer or ""},
            ),
            carryover_sources=(history_source,),
            previous_user_question=candidate.user_question,
            route_turn_candidates=(candidate,),
            relation="followup",
            query_resolution_mode="v3_catalog",
            context_turn_keys=("t1",),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        routing_result = _routing_result(selected_kb_count=1)
        v2_calls: list[dict] = []

        class V3Service:
            def __init__(self) -> None:
                self.calls: list[dict] = []
                self.results: list[QueryUnderstandingV3ExecutionResult] = []

            async def run_active(self, **kwargs):
                self.calls.append(kwargs)
                result = _compiled_v3_result(
                    baseline=kwargs["baseline"],
                    target_texts=("住宿标准", "餐补", "出差补贴"),
                    qualifier_text="普通员工",
                    qualifier_source_key="t1",
                )
                self.results.append(result)
                return result

        v3_service = V3Service()

        async def v2_stream(**kwargs):
            v2_calls.append(kwargs)
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    "evidence_status": "no_hit",
                    "displayed_result_count": 0,
                    "direct_evidence_count": 0,
                    "related_reference_count": 0,
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "text_delta", "content": "V3 已交给 V2"},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        legacy_service = AsyncMock(
            side_effect=AssertionError("V3 请求不得调用 legacy query-analysis"),
        )
        apply_context = AsyncMock(return_value=selected_context)
        task_read_session_factory = object()
        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=initial_context),
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(
                    side_effect=AssertionError("V3 不得采用 legacy context projection")
                ),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch("api.chat.get_settings", return_value=_v3_settings(anchor_prefetch_enabled=False)),
            patch("api.chat.get_query_analysis_execution_service", new=legacy_service),
            patch("api.chat.get_query_understanding_v3_execution_service", return_value=v3_service),
            patch("api.chat.apply_v3_catalog_context_selection", new=apply_context),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("api.chat.TaskReadSessionLocal", new=task_read_session_factory),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            events = await _sse_events(response)

        self.assertTrue(events)
        legacy_service.assert_not_called()
        self.assertEqual(len(v3_service.calls), 1)
        apply_context.assert_awaited_once()
        self.assertEqual(
            apply_context.await_args.kwargs["selected_context_turn_keys"],
            ("t1",),
        )
        self.assertEqual(len(v2_calls), 1)
        handoff = v2_calls[0]
        self.assertEqual(handoff["question"], question)
        self.assertEqual(handoff["standalone_query"], question)
        self.assertTrue(handoff["is_followup"])
        self.assertEqual(handoff["followup_reason"], "query_understanding_v3_contextual")
        self.assertEqual(handoff["conversation_history"], list(selected_context.history_messages))
        self.assertEqual(handoff["carryover_sources"], list(selected_context.carryover_sources))
        self.assertIs(
            handoff["execution_bundle"],
            v3_service.results[0].execution_bundle,
            "V2 must receive the V3 compiler's exact immutable bundle",
        )
        # The V3 result is compiled from three selected targets, not a hard-coded
        # business-question branch.  Its one classification augmentation is
        # also preserved in the executable V2 ledger.
        answers = [
            item
            for item in handoff["execution_bundle"].plan.requirements
            if item.role == "answer"
        ]
        self.assertEqual(
            [item.description for item in answers],
            ["普通员工 住宿标准", "普通员工 餐补", "普通员工 出差补贴"],
        )
        self.assertTrue(all(item.augmentation_requirement_ids for item in answers))
        self.assertIs(handoff["task_read_session_factory"], task_read_session_factory)

    async def test_v3_fallback_with_not_ready_local_floor_returns_sse_clarification(self) -> None:
        """A model fallback must fail closed before V2 retrieval, not crash."""

        question = "请帮我查一下"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=question,
            history_messages=(),
            carryover_sources=(),
        )

        class FallbackService:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def run_active(self, **kwargs):
                self.calls.append(kwargs)
                return _fallback_v3_result(baseline=kwargs["baseline"])

        v3_service = FallbackService()

        def no_v2_retrieval(**_kwargs):
            raise AssertionError("not-ready V3 fallback must not enter V2 retrieval")

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=AsyncMock(return_value=context)),
            patch("api.chat.classify_intent_result", new=AsyncMock(return_value=_routing_result(selected_kb_count=1))),
            patch("api.chat.get_settings", return_value=_v3_settings(anchor_prefetch_enabled=False)),
            patch("api.chat.plan_query_locally", return_value=_not_ready_plan(question)),
            patch(
                "api.chat.get_query_analysis_execution_service",
                side_effect=AssertionError("V3 request must not invoke legacy analysis"),
            ),
            patch("api.chat.get_query_understanding_v3_execution_service", return_value=v3_service),
            patch("api.chat.run_rag_v2_stream", new=no_v2_retrieval),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            events = await _sse_events(response)

        self.assertEqual(len(v3_service.calls), 1)
        intent = next(event for event in events if event.get("type") == "intent")
        self.assertFalse(intent["decision"]["dispatch_authorized"])
        self.assertEqual(
            intent["decision"]["query_execution"]["state"],
            "needs_clarification",
        )
        self.assertFalse(
            next(event for event in events if event.get("type") == "search_results")[
                "retrieval_executed"
            ]
        )
        self.assertTrue(any(
            event.get("type") == "clarification_state"
            and event.get("status") == "active"
            for event in events
        ))
        self.assertFalse(any(event.get("type") == "text_delta" for event in events))
        self.assertIsNotNone(conversation.pending_route_state)

    async def test_v3_contextual_safety_closure_is_adopted_before_v2_dispatch(self) -> None:
        """A model-selected unsafe history closes the gate, never bare-retrieves."""

        question = "餐补呢"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        candidate = RouteTurnCandidate(
            candidate_key="t1",
            user_question="普通员工在云枢8.6中的餐饮补贴是多少",
            assistant_answer="历史回答不应被当作适用范围来源。",
        )
        context = ConversationContext(
            is_followup=True,
            followup_reason="short_elliptical_question",
            standalone_query=question,
            history_messages=(
                {"role": "user", "content": candidate.user_question},
                {"role": "assistant", "content": candidate.assistant_answer or ""},
            ),
            carryover_sources=(),
            route_turn_candidates=(candidate,),
            relation="followup",
            query_resolution_mode="contextualize",
            context_turn_keys=("t1",),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()

        class ContextSafetyClosureService:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            async def run_active(self, **kwargs):
                self.calls.append(kwargs)
                baseline = kwargs["baseline"]
                closed = build_execution_clarification_baseline(
                    baseline=baseline.fallback,
                    reason="historical_context_requires_clarification",
                    clarification_question="请在本轮重新明确适用范围。",
                )
                return QueryUnderstandingV3ExecutionResult(
                    decision="clarification",
                    reason="historical_context_requires_clarification",
                    request_baseline=baseline,
                    selected_baseline=closed,
                    query_execution_gate=evaluate_query_execution_gate(closed),
                    validation=QueryUnderstandingV3ExecutionValidation(
                        accepted=False,
                        reason="historical_context_not_inheritable_explicit_scope",
                        current_target_count=1,
                        candidate_target_count=1,
                        explicit_scope_partition_count=0,
                        requires_clarification=True,
                    ),
                )

        v3_service = ContextSafetyClosureService()

        def no_v2_retrieval(**_kwargs):
            raise AssertionError("contextual safety closure must not enter V2 retrieval")

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=AsyncMock(return_value=context)),
            patch("api.chat.classify_intent_result", new=AsyncMock(return_value=_routing_result(selected_kb_count=1))),
            patch("api.chat.get_settings", return_value=_v3_settings(anchor_prefetch_enabled=False)),
            patch(
                "api.chat.get_query_analysis_execution_service",
                side_effect=AssertionError("V3 request must not invoke legacy analysis"),
            ),
            patch("api.chat.get_query_understanding_v3_execution_service", return_value=v3_service),
            patch("api.chat.run_rag_v2_stream", new=no_v2_retrieval),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            events = await _sse_events(response)

        self.assertEqual(len(v3_service.calls), 1)
        intent = next(event for event in events if event.get("type") == "intent")
        self.assertEqual(
            intent["decision"]["query_execution"]["state"],
            "needs_clarification",
        )
        self.assertFalse(
            next(event for event in events if event.get("type") == "search_results")[
                "retrieval_executed"
            ]
        )
        self.assertIsNotNone(conversation.pending_route_state)

    async def test_anchor_prefetch_uses_original_question_and_same_revision_as_v2(self) -> None:
        """The concurrent cache is bound to the immutable current question."""

        question = "那餐补呢"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        candidate = RouteTurnCandidate(
            candidate_key="t1",
            user_question="普通员工的住宿标准是多少",
            assistant_answer="普通员工对应D级。",
        )
        initial_context = ConversationContext(
            is_followup=True,
            followup_reason="legacy_route_projection",
            standalone_query="普通员工的餐补是多少",
            history_messages=(
                {"role": "user", "content": candidate.user_question},
                {"role": "assistant", "content": candidate.assistant_answer or ""},
            ),
            carryover_sources=(),
            route_turn_candidates=(candidate,),
            relation="continuation",
            query_resolution_mode="contextualize",
            context_turn_keys=("t1",),
        )
        selected_context = ConversationContext(
            is_followup=True,
            followup_reason="query_understanding_v3_contextual",
            standalone_query=question,
            history_messages=initial_context.history_messages,
            carryover_sources=(),
            previous_user_question=candidate.user_question,
            route_turn_candidates=(candidate,),
            relation="followup",
            query_resolution_mode="v3_catalog",
            context_turn_keys=("t1",),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        anchor_calls: list[dict] = []
        v2_calls: list[dict] = []

        class V3Service:
            async def run_active(self, **kwargs):
                # Let the concurrently scheduled prefetch finish before the
                # semantic barrier checks ``Task.done()``.
                await asyncio.sleep(0)
                return _compiled_v3_result(
                    baseline=kwargs["baseline"],
                    target_texts=("餐补",),
                    qualifier_text="普通员工",
                    qualifier_source_key="t1",
                )

        async def retrieve_anchor(**kwargs):
            anchor_calls.append(kwargs)
            return AnchorRetrievalSnapshot(
                revision=kwargs["revision"],
                query=kwargs["query"],
                kb_ids=tuple(kwargs["kb_ids"]),
                document_ids=None,
                method=kwargs["method"],
                candidate_limit=kwargs["candidate_limit"],
            )

        async def v2_stream(**kwargs):
            v2_calls.append(kwargs)
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    "evidence_status": "no_hit",
                    "displayed_result_count": 0,
                    "direct_evidence_count": 0,
                    "related_reference_count": 0,
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "text_delta", "content": "V2"}, ensure_ascii=False
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=initial_context),
            ),
            patch("api.chat.classify_intent_result", new=AsyncMock(return_value=_routing_result(selected_kb_count=1))),
            patch("api.chat.get_settings", return_value=_v3_settings(anchor_prefetch_enabled=True)),
            patch(
                "api.chat.get_query_analysis_execution_service",
                side_effect=AssertionError("V3 request must not invoke legacy analysis"),
            ),
            patch("api.chat.get_query_understanding_v3_execution_service", return_value=V3Service()),
            patch("api.chat.apply_v3_catalog_context_selection", new=AsyncMock(return_value=selected_context)),
            patch("api.chat.retrieve_anchor_retrieval_snapshot", new=retrieve_anchor),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            await _sse_events(response)

        self.assertEqual(len(anchor_calls), 1)
        self.assertEqual(anchor_calls[0]["query"], question)
        # The history-derived answer task is deliberately different.  It must
        # never become the prefetch query or overwrite the immutable anchor.
        self.assertNotEqual(anchor_calls[0]["query"], "普通员工 餐补")
        self.assertEqual(len(v2_calls), 1)
        self.assertEqual(v2_calls[0]["question"], question)
        self.assertEqual(v2_calls[0]["anchor_retrieval_revision"], anchor_calls[0]["revision"])
        snapshot = v2_calls[0]["anchor_retrieval_snapshot"]
        self.assertEqual(snapshot.query, question)
        self.assertEqual(snapshot.revision, anchor_calls[0]["revision"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
