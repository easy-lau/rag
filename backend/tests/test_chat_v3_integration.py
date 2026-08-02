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

from api.chat import send_message
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
    async def test_pending_route_reply_keeps_original_task_when_v3_falls_back(self) -> None:
        """A clarification value must never become the fallback retrieval query."""

        reply = "云枢的"
        original_query = "我现在想改验证码有效期时间"
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = {
            "schema_version": "rag_pending_clarification.v1",
            "state_id": "route-pending",
            "base_user_message_id": str(uuid.uuid4()),
            "clarification_message_id": str(uuid.uuid4()),
            "intent_code": "knowledge_qa",
            "original_query": original_query,
            "clarification_answers": [],
            "unresolved": [
                {"role": "system_or_product", "reason": "missing", "candidate_count": 0}
            ],
            "selected_kb_ids_snapshot": [str(kb_id)],
            "created_at": "2026-08-02T00:00:00+00:00",
            "expires_at": "2099-08-02T00:00:00+00:00",
            "dispatch_authorized": False,
        }
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
        self.assertIn(original_query, planned_query)
        self.assertIn(reply, planned_query)
        self.assertNotEqual(planned_query, reply)
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
        self.assertTrue(any(event.get("type") == "text_delta" for event in events))
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
