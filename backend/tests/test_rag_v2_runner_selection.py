import json
import os
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from api.chat import _select_rag_pipeline_version, send_message
from config import Settings
from core.conversation_context import ConversationContext
from core.query_analysis_execution import QueryAnalysisExecutionResult
from core.query_route_compiler import (
    CompiledAnswerRequirement,
    RagTaskContract,
)
from core.query_route_contract import RouteClarification
from core.rag_v2.pipeline import run_rag_v2_stream
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.task_graph import RagExecutionBundle, compile_rag_execution_bundle
from models.schemas import ChatRequest


def _task_contract(
    *,
    response_mode: str = "grounded_qa",
    need_retrieval: bool = True,
    dispatch_authorized: bool = True,
    selected_kb_count: int = 1,
) -> RagTaskContract:
    return RagTaskContract(
        schema_version="rag_task_contract.v1",
        route_schema_version="rag_route_decision.v1",
        readiness="ready" if dispatch_authorized else "needs_clarification",
        intent_code="knowledge_qa",
        intent_name="知识问答",
        action="retrieve",
        confidence=0.95,
        source="llm",
        relation="new",
        evidence_scope="enterprise_kb",
        query_mode="current",
        context_turn_keys=(),
        response_mode=response_mode,
        retrieval_policy="required" if need_retrieval else "skip",
        need_retrieval=need_retrieval,
        dispatch_authorized=dispatch_authorized,
        decision_reason="test_contract",
        selected_kb_count=selected_kb_count,
        requirements=(
            CompiledAnswerRequirement(
                id="r1",
                role="answer",
                origin="user_text",
                description="回答当前问题",
                importance="required",
                source="explicit",
            ),
        ),
        clarification=RouteClarification(question=""),
    )


def _ledgered_bundle(question: str) -> RagExecutionBundle:
    """Build the production-shaped V2 hand-off required by direct tests."""

    plan = QueryPlanV2(
        original_query=question,
        answer_shape="fact",
        retrieval_queries=(question,),
        requirements=(
            AnswerRequirementV2(
                id="r1",
                description=question,
                role="answer",
                importance="required",
                source="explicit",
                # The executable ledger distinguishes a deliberately direct
                # fact (no proof bridge) from an uncompiled legacy answer.
                # Use an explicit empty edge set so this helper exercises the
                # same invariant as production's plan compiler.
                depends_on_requirement_ids=(),
            ),
        ),
        confidence=0.9,
        source="local",
        reason="runner_selection_test_ledgered_bundle",
    )
    bundle = compile_rag_execution_bundle(plan)
    if bundle.mode != "ledgered":  # pragma: no cover - test helper invariant
        raise AssertionError("runner selection test requires a ledgered bundle")
    return bundle


def _evidence_pending_state(*, kb_id: uuid.UUID) -> dict:
    first_doc_id = uuid.uuid4()
    second_doc_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    return {
        "schema_version": "rag_pending_clarification.v2",
        "kind": "evidence_scope",
        "state_id": str(uuid.uuid4()),
        "base_user_message_id": str(uuid.uuid4()),
        "clarification_message_id": str(uuid.uuid4()),
        "original_query": "登录用户名枚举要配置什么",
        "dimension": "version",
        "selection_mode": "choice",
        "choices": [
            {
                "key": "c1",
                "label": "云枢 6.0.1",
                "products": ["云枢"],
                "canonical_products": ["云枢"],
                "versions": ["6.0.1"],
                "projects": [],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(first_doc_id)],
                "anchor_doc_ids": [str(first_doc_id)],
                "companion_doc_ids": [],
                "filenames": ["云枢6.md"],
            },
            {
                "key": "c2",
                "label": "云枢 8.2.75",
                "products": ["云枢"],
                "canonical_products": ["云枢"],
                "versions": ["8.2.75"],
                "projects": [],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(second_doc_id)],
                "anchor_doc_ids": [str(second_doc_id)],
                "companion_doc_ids": [],
                "filenames": ["云枢8.md"],
            },
        ],
        "clarification_message": "请问需要查询云枢 6.0.1 还是 8.2.75？",
        "selected_kb_ids_snapshot": [str(kb_id)],
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "dispatch_authorized": False,
    }


class RagPipelineVersionConfigurationTests(unittest.TestCase):
    def test_default_version_is_v2_and_invalid_values_are_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings(_env_file=None).rag_pipeline_version, "v2")

        with self.assertRaises(ValidationError):
            Settings(rag_pipeline_version="v3", _env_file=None)


class RagPipelineSelectionTests(unittest.TestCase):
    def _select(self, **overrides):
        values = {
            "configured_version": "v2",
            "task_contract": _task_contract(),
            "evidence_scope_filter": None,
            "evidence_scope_refinement_active": False,
            "is_followup": False,
            "carryover_sources": (),
        }
        values.update(overrides)
        return _select_rag_pipeline_version(**values)

    def test_configured_v1_remains_on_v1(self) -> None:
        self.assertEqual(
            self._select(configured_version="v1"),
            ("v1", "configured_v1"),
        )

    def test_eligible_grounded_qa_selects_v2(self) -> None:
        self.assertEqual(self._select(), ("v2", "eligible_grounded_qa"))

    def test_v2_routes_direct_modes_without_legacy_fallback(self) -> None:
        cases = {
            "general_chat": (
                _task_contract(response_mode="general_chat", need_retrieval=False),
                "verified_general_chat",
            ),
            "inline_writing": (
                _task_contract(response_mode="writing", need_retrieval=False),
                "verified_writing",
            ),
            "platform_help": (
                _task_contract(response_mode="platform_help", need_retrieval=False),
                "verified_platform_help",
            ),
        }
        for name, (contract, expected_reason) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    self._select(task_contract=contract),
                    ("direct", expected_reason),
                )

    def test_v2_rejects_invalid_contracts_instead_of_using_v1(self) -> None:
        cases = {
            "missing_contract": (None, "missing_or_invalid_task_contract"),
            "dispatch_not_authorized": (
                _task_contract(dispatch_authorized=False),
                "dispatch_not_authorized",
            ),
            "invalid_direct": (
                _task_contract(response_mode="grounded_qa", need_retrieval=False),
                "invalid_task_contract:grounded_mode_requires_retrieval",
            ),
        }
        for name, (contract, expected_reason) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    self._select(task_contract=contract),
                    ("reject", expected_reason),
                )

    def test_knowledge_grounded_writing_selects_v2(self) -> None:
        self.assertEqual(
            self._select(
                task_contract=_task_contract(
                    response_mode="writing",
                    need_retrieval=True,
                )
            ),
            ("v2", "eligible_knowledge_writing"),
        )

    def test_evidence_scope_selection_enters_v2_with_followup_context(self) -> None:
        self.assertEqual(
            self._select(
                evidence_scope_filter={"mode": "choice"},
                is_followup=True,
                carryover_sources=({"doc_id": "doc-1"},),
            ),
            ("v2", "eligible_evidence_scope_selection"),
        )

    def test_refinement_and_ordinary_context_continue_on_v2(self) -> None:
        cases = {
            "scope_refinement": (
                None,
                True,
                False,
                (),
                "eligible_evidence_scope_refinement",
            ),
            "followup": (
                None,
                False,
                True,
                (),
                "eligible_grounded_followup",
            ),
            "carryover": (
                None,
                False,
                False,
                ({"doc_id": "doc-1"},),
                "eligible_grounded_followup",
            ),
        }
        for name, (
            scope_filter,
            refinement_active,
            is_followup,
            carryover,
            reason,
        ) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    self._select(
                        evidence_scope_filter=scope_filter,
                        evidence_scope_refinement_active=refinement_active,
                        is_followup=is_followup,
                        carryover_sources=carryover,
                    ),
                    ("v2", reason),
                )

class RagV2ScopeFilterSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_or_unauthorized_scope_filter_fails_closed(self) -> None:
        authorized_kb_id = uuid.uuid4()
        unauthorized_kb_id = uuid.uuid4()
        unauthorized_doc_id = uuid.uuid4()
        valid_unauthorized_filter = {
            "mode": "single",
            "kb_ids": [str(unauthorized_kb_id)],
            "doc_ids": [str(unauthorized_doc_id)],
            "choices": [{
                "key": "c1",
                "label": "未授权范围",
                "products": [],
                "canonical_products": [],
                "versions": [],
                "projects": [],
                "filenames": ["未授权文档.md"],
                "kb_ids": [str(unauthorized_kb_id)],
                "doc_ids": [str(unauthorized_doc_id)],
                "anchor_doc_ids": [str(unauthorized_doc_id)],
                "companion_doc_ids": [],
            }],
        }
        malformed_filter = {
            **valid_unauthorized_filter,
            "kb_ids": [str(authorized_kb_id)],
            "choices": [],
        }

        for name, scope_filter in (
            ("unauthorized_kb", valid_unauthorized_filter),
            ("malformed_shape", malformed_filter),
        ):
            with self.subTest(name=name):
                global_search = AsyncMock(
                    side_effect=AssertionError(
                        "无效范围过滤不得回退到全局检索"
                    )
                )
                scoped_search = AsyncMock(
                    side_effect=AssertionError(
                        "无效范围过滤不得查询任何文档"
                    )
                )
                stream = run_rag_v2_stream(
                    question="查询未授权范围",
                    kb_ids=[authorized_kb_id],
                    search_config={
                        "top_k": 5,
                        "method": "hybrid",
                        "rerank": False,
                    },
                    conversation_id="scope-filter-safety",
                    db=SimpleNamespace(),
                    execution_bundle=_ledgered_bundle("查询未授权范围"),
                    intent={"intent_code": "knowledge_qa"},
                    task_contract=_task_contract(),
                    evidence_scope_filter=scope_filter,
                )
                search_event = None
                with (
                    patch(
                        "core.rag_v2.pipeline.hybrid_search",
                        new=global_search,
                    ),
                    patch(
                        "core.rag_v2.pipeline.search_within_documents",
                        new=scoped_search,
                    ),
                    patch("core.rag_v2.pipeline.trace_event"),
                ):
                    try:
                        async for chunk in stream:
                            payload = json.loads(
                                chunk.removeprefix("data: ").strip()
                            )
                            if payload.get("type") == "search_results":
                                search_event = payload
                                break
                    finally:
                        await stream.aclose()

                self.assertIsNotNone(search_event)
                self.assertEqual(search_event["evidence_status"], "error")
                self.assertEqual(search_event["results"], [])
                self.assertEqual(search_event["answer_sources"], [])
                global_search.assert_not_awaited()
                scoped_search.assert_not_awaited()


class _RequestDB:
    def __init__(self, conversation):
        self.conversation = conversation
        self.added = []
        self.commit_count = 0

    async def get(self, _model, _identity):
        return self.conversation

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commit_count += 1
        return None


class _SaveDB:
    def __init__(self, conversation=None):
        self.conversation = conversation
        self.added = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def get(self, _model, _identity):
        return self.conversation

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        return None


class RagPipelineDispatchTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _ready_analysis_routing_result():
        contract = _task_contract()
        decision = SimpleNamespace(
            need_retrieval=True,
            decision_reason="classified_retrieval",
            to_dict=lambda: {
                "intent_code": "knowledge_qa",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
        )
        route_decision = SimpleNamespace(
            schema_version="rag_route_decision.v1",
            relation="new",
            evidence_scope="enterprise_kb",
            to_dict=lambda: {"schema_version": "rag_route_decision.v1"},
        )
        return SimpleNamespace(
            decision=decision,
            route_decision=route_decision,
            task_contract=contract,
            diagnostics={},
            route_log_id=None,
        )

    async def _run_mocked_v2_dispatch(
        self,
        *,
        question: str,
        kb_id: uuid.UUID,
        conversation: SimpleNamespace,
        user: SimpleNamespace,
        context: ConversationContext,
        routing_result: SimpleNamespace,
    ):
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        received_kwargs = []

        async def v2_stream(**kwargs):
            received_kwargs.append(kwargs)
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    # Keep pending route state unchanged in dispatch tests;
                    # state resolution is exercised by route-state tests.
                    "evidence_status": "error",
                    "displayed_result_count": 0,
                    "direct_evidence_count": 0,
                    "related_reference_count": 0,
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "text_delta", "content": "V2 dispatch"},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation.id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch(
                "api.chat.get_settings",
                return_value=SimpleNamespace(
                    rag_trace_include_content=False,
                    rag_pipeline_version="v2",
                    rag_query_analyzer_mode="off",
                ),
            ),
            patch("api.chat.run_rag_stream") as v1_stream,
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation.id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        return received_kwargs, v1_stream, trace, chunks

    async def _consume_analyzer_timing_dispatch(
        self,
        *,
        analyzer_mode: str,
        service,
    ):
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
            standalone_query="普通员工的住宿标准和餐补是多少",
            history_messages=(),
            carryover_sources=(),
        )
        routing_result = self._ready_analysis_routing_result()
        runner_started: list[int] = []

        async def v2_stream(**_kwargs):
            runner_started.append(request_db.commit_count)
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    "evidence_status": "error",
                    "displayed_result_count": 0,
                    "direct_evidence_count": 0,
                    "related_reference_count": 0,
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "text_delta", "content": "V2 dispatch"},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch(
                "api.chat.get_settings",
                return_value=SimpleNamespace(
                    rag_trace_include_content=False,
                    rag_pipeline_version="v2",
                    rag_query_analyzer_mode=analyzer_mode,
                    rag_query_analyzer_active_timeout_seconds=0.5,
                    rag_query_analyzer_active_max_inflight=1,
                    rag_query_analyzer_shadow_timeout_seconds=0.5,
                    rag_query_analyzer_shadow_max_inflight=1,
                    rag_query_analyzer_shadow_sample_rate=1.0,
                ),
            ),
            patch("api.chat.get_query_analysis_execution_service", return_value=service),
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question="普通员工的住宿标准和餐补是多少",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            iterator = response.body_iterator
            first = await anext(iterator)
            before_second_event = request_db.commit_count
            service_calls_after_first = list(getattr(service, "calls", ()))
            second = await anext(iterator)
            remaining = [chunk async for chunk in iterator]
        return {
            "first": first,
            "second": second,
            "remaining": remaining,
            "request_db": request_db,
            "runner_started": runner_started,
            "commit_count_after_first": before_second_event,
            "service_calls_after_first": service_calls_after_first,
        }

    async def test_eligible_request_calls_only_v2_and_traces_selection(self) -> None:
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
            standalone_query="普通员工的出差标准是什么",
            history_messages=(),
            carryover_sources=(),
        )
        contract = _task_contract()
        decision = SimpleNamespace(
            need_retrieval=True,
            decision_reason="classified_retrieval",
            to_dict=lambda: {
                "intent_code": "knowledge_qa",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
        )
        route_decision = SimpleNamespace(
            schema_version="rag_route_decision.v1",
            relation="new",
            evidence_scope="enterprise_kb",
            to_dict=lambda: {"schema_version": "rag_route_decision.v1"},
        )
        routing_result = SimpleNamespace(
            decision=decision,
            route_decision=route_decision,
            task_contract=contract,
            diagnostics={},
            route_log_id=None,
        )
        received_kwargs = []

        def task_read_session_factory():
            raise AssertionError("mock V2 stream must not open a read session")

        async def v2_stream(**kwargs):
            received_kwargs.append(kwargs)
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
                {"type": "text_delta", "content": "未找到相关内容"},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch(
                "api.chat.get_settings",
                return_value=SimpleNamespace(
                    rag_trace_include_content=False,
                    rag_pipeline_version="v2",
                    rag_query_analyzer_mode="off",
                ),
            ),
            patch("api.chat.run_rag_stream") as v1_stream,
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch(
                "api.chat.TaskReadSessionLocal",
                new=task_read_session_factory,
            ),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="普通员工的出差标准是什么",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertTrue(chunks)
        v1_stream.assert_not_called()
        self.assertEqual(len(received_kwargs), 1)
        self.assertEqual(
            set(received_kwargs[0]),
            {
                "question",
                "kb_ids",
                "search_config",
                "conversation_id",
                "db",
                "intent",
                "task_contract",
                "trace_id",
                "standalone_query",
                "conversation_history",
                "carryover_sources",
                "is_followup",
                "followup_reason",
                "evidence_scope_filter",
                "execution_bundle",
                "task_read_session_factory",
            },
        )
        self.assertIs(
            received_kwargs[0]["task_read_session_factory"],
            task_read_session_factory,
        )
        self.assertIsInstance(
            received_kwargs[0]["execution_bundle"],
            RagExecutionBundle,
        )
        self.assertEqual(
            received_kwargs[0]["execution_bundle"].mode,
            "ledgered",
        )
        self.assertEqual(
            received_kwargs[0]["execution_bundle"].plan.requirements,
            received_kwargs[0]["execution_bundle"].task_graph.requirements,
        )
        selection = next(
            call
            for call in trace.call_args_list
            if call.args and call.args[0] == "chat.pipeline_selected"
        )
        self.assertEqual(selection.kwargs["version"], "v2")
        self.assertEqual(selection.kwargs["reason"], "eligible_grounded_qa")

    async def test_active_analysis_runs_only_after_commit_and_first_sse(self) -> None:
        class ActiveService:
            def __init__(self):
                self.calls = []

            async def run_active(self, **kwargs):
                self.calls.append(kwargs)
                baseline = kwargs["baseline"]
                return QueryAnalysisExecutionResult(
                    mode="active",
                    decision="fallback",
                    reason="analysis_timeout",
                    baseline=baseline,
                    execution_bundle=baseline.execution_bundle,
                    analysis_latency_ms=0,
                )

        service = ActiveService()
        result = await self._consume_analyzer_timing_dispatch(
            analyzer_mode="active",
            service=service,
        )
        first_payload = json.loads(result["first"].removeprefix("data: ").strip())
        self.assertEqual(first_payload["type"], "conversation_started")
        self.assertEqual(result["service_calls_after_first"], [])
        self.assertEqual(len(service.calls), 1)
        self.assertGreaterEqual(result["commit_count_after_first"], 1)
        self.assertGreaterEqual(service.calls[0]["maximum_inflight"], 1)
        self.assertEqual(service.calls[0]["timeout_seconds"], 0.5)
        self.assertEqual(result["runner_started"], [result["request_db"].commit_count])

    async def test_shadow_submission_runs_only_after_commit_and_first_sse(self) -> None:
        class ShadowService:
            def __init__(self):
                self.calls = []

            def submit_shadow(self, **kwargs):
                self.calls.append(kwargs)
                return True

        service = ShadowService()
        result = await self._consume_analyzer_timing_dispatch(
            analyzer_mode="shadow",
            service=service,
        )
        first_payload = json.loads(result["first"].removeprefix("data: ").strip())
        self.assertEqual(first_payload["type"], "conversation_started")
        self.assertEqual(result["service_calls_after_first"], [])
        self.assertEqual(len(service.calls), 1)
        submission = service.calls[0]
        self.assertEqual(submission["submission_phase"], "post_commit_post_first_sse")
        self.assertEqual(submission["sample_rate"], 1.0)
        self.assertGreaterEqual(result["commit_count_after_first"], 1)

    async def test_verified_general_chat_calls_only_direct_runner(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
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
            standalone_query="你好",
            history_messages=(),
            carryover_sources=(),
        )
        contract = _task_contract(
            response_mode="general_chat",
            need_retrieval=False,
            selected_kb_count=0,
        )
        decision = SimpleNamespace(
            need_retrieval=False,
            decision_reason="exact_greeting",
            to_dict=lambda: {
                "intent_code": "general_chat",
                "need_retrieval": False,
                "decision_reason": "exact_greeting",
            },
        )
        route_decision = SimpleNamespace(
            schema_version="rag_route_decision.v1",
            relation="new",
            evidence_scope="general_world",
            to_dict=lambda: {"schema_version": "rag_route_decision.v1"},
        )
        routing_result = SimpleNamespace(
            decision=decision,
            route_decision=route_decision,
            task_contract=contract,
            diagnostics={},
            route_log_id=None,
        )
        received_kwargs = []

        async def direct_stream(**kwargs):
            received_kwargs.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": False,
                "evidence_status": "skipped",
                "displayed_result_count": 0,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
            }) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "text_delta", "content": "你好，有什么可以帮你？"},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch(
                "api.chat.get_settings",
                return_value=SimpleNamespace(
                    rag_trace_include_content=False,
                    rag_pipeline_version="v2",
                ),
            ),
            patch("api.chat.run_rag_stream") as v1_stream,
            patch("api.chat.run_rag_v2_stream") as v2_stream,
            patch("api.chat.run_direct_response_stream", new=direct_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="你好",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[],
                ),
                db=request_db,
                user=user,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertTrue(chunks)
        self.assertEqual(len(received_kwargs), 1)
        v1_stream.assert_not_called()
        v2_stream.assert_not_called()
        selection = next(
            call
            for call in trace.call_args_list
            if call.args and call.args[0] == "chat.pipeline_selected"
        )
        self.assertEqual(selection.kwargs["version"], "direct")
        self.assertEqual(selection.kwargs["reason"], "verified_general_chat")

    async def test_runner_contract_rejection_503_requires_new_request_id(
        self,
    ) -> None:
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
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="查询采购审批额度",
            history_messages=(),
            carryover_sources=(),
        )
        decision = SimpleNamespace(
            need_retrieval=True,
            decision_reason="classified_retrieval",
            to_dict=lambda: {
                "intent_code": "knowledge_qa",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
        )
        routing_result = SimpleNamespace(
            decision=decision,
            route_decision=None,
            task_contract=None,
            diagnostics={},
            route_log_id=None,
        )

        with (
            patch(
                "api.chat.get_accessible_kb_ids",
                new=AsyncMock(return_value=None),
            ),
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
                return_value=SimpleNamespace(
                    rag_trace_include_content=False,
                    rag_pipeline_version="v2",
                ),
            ),
            patch("api.chat.trace_event"),
        ):
            with self.assertRaises(HTTPException) as raised:
                await send_message(
                    ChatRequest(
                        question="查询采购审批额度",
                        conversation_id=conversation_id,
                        knowledge_base_ids=[kb_id],
                        request_id="runner-rejected-request",
                    ),
                    db=request_db,
                    user=user,
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            {
                "message": "请求执行合同校验失败，请重新发送",
                "error_code": "runner_contract_rejected",
                "same_request_recoverable": False,
                "retry_with_new_request_id": True,
            },
        )

    async def test_evidence_scope_selection_skips_router_and_calls_v2(
        self,
    ) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=3,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RequestDB(conversation)
        save_db = _SaveDB()
        carryover_source = {
            "id": str(uuid.uuid4()),
            "doc_id": pending["choices"][0]["doc_ids"][0],
            "kb_id": str(kb_id),
            "content": "上一轮范围候选",
        }
        context = ConversationContext(
            is_followup=True,
            followup_reason="route_followup_current",
            standalone_query=pending["original_query"],
            history_messages=(),
            carryover_sources=(carryover_source,),
            pending_route_state=pending,
        )
        classify = AsyncMock(
            side_effect=AssertionError("有效证据序号不应再次调用意图模型")
        )
        received_kwargs = []

        async def v2_stream(**kwargs):
            received_kwargs.append(kwargs)
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    # Keep the pending selection active in this dispatch-only
                    # test; scope-state resolution is covered separately.
                    "evidence_status": "error",
                    "displayed_result_count": 0,
                    "direct_evidence_count": 0,
                    "related_reference_count": 0,
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "text_delta", "content": "范围检索暂时不可用"},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=classify,
            ),
            patch(
                "api.chat.get_settings",
                return_value=SimpleNamespace(
                    rag_trace_include_content=False,
                    rag_pipeline_version="v2",
                ),
            ),
            patch("api.chat.run_rag_stream") as v1_stream,
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="2",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertTrue(chunks)
        classify.assert_not_awaited()
        v1_stream.assert_not_called()
        self.assertEqual(len(received_kwargs), 1)
        scope_filter = received_kwargs[0]["evidence_scope_filter"]
        self.assertEqual(scope_filter["mode"], "single")
        self.assertEqual(scope_filter["choices"][0]["key"], "c2")
        self.assertEqual(received_kwargs[0]["question"], pending["original_query"])
        self.assertEqual(
            received_kwargs[0]["task_contract"].decision_reason,
            "evidence_scope_selected",
        )
        # A pending evidence choice is not itself a resolved semantic binding.
        # V2 receives the original selected question but no route-candidate
        # history/carry-over material until a source-anchored contract exists.
        self.assertFalse(received_kwargs[0]["is_followup"])
        self.assertEqual(received_kwargs[0]["carryover_sources"], [])
        selection = next(
            call
            for call in trace.call_args_list
            if call.args and call.args[0] == "chat.pipeline_selected"
        )
        self.assertEqual(selection.kwargs["version"], "v2")
        self.assertEqual(
            selection.kwargs["reason"],
            "eligible_evidence_scope_selection",
        )
        # An error/empty scope result must not consume the pending choice;
        # retry remains possible and the UI must not silently resume globally.
        self.assertIs(conversation.pending_route_state, pending)
        self.assertEqual(conversation.route_state_revision, 3)

    async def test_evidence_scope_selection_clears_pending_only_after_anchor_hit(
        self,
    ) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=3,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RequestDB(conversation)
        save_db = _SaveDB(conversation)
        context = ConversationContext(
            is_followup=True,
            followup_reason="route_followup_current",
            standalone_query=pending["original_query"],
            history_messages=(),
            carryover_sources=(),
            pending_route_state=pending,
        )
        classify = AsyncMock(
            side_effect=AssertionError("有效证据序号不应再次调用意图模型")
        )
        anchor_doc_id = pending["choices"][1]["anchor_doc_ids"][0]
        anchor_chunk_id = str(uuid.uuid4())
        answer_source = {
            "id": anchor_chunk_id,
            "chunk_id": anchor_chunk_id,
            "doc_id": anchor_doc_id,
            "kb_id": str(kb_id),
            "content": "范围证据",
            "evidence_role": "direct",
        }

        async def v2_stream(**_kwargs):
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [answer_source],
                "answer_sources": [answer_source],
                "retrieval_executed": True,
                "evidence_status": "hit",
                "displayed_result_count": 1,
                "direct_evidence_count": 1,
                "related_reference_count": 0,
                "evidence_scope_anchor_hit": True,
                "evidence_scope_anchor_doc_ids": [anchor_doc_id],
            }) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "text_delta", "content": "范围回答"},
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        async def trusted_source_refresh(
            _db,
            *,
            raw_sources,
            raw_results,
            selected_kb_ids,
            read_session_factory=None,
        ):
            del raw_results, selected_kb_ids, read_session_factory
            return (
                list(raw_sources or []),
                {
                    (
                        uuid.UUID(str(source["kb_id"])),
                        uuid.UUID(str(source["doc_id"])),
                    )
                    for source in (raw_sources or [])
                },
                None,
            )

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=AsyncMock(return_value=context)),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.classify_intent_result", new=classify),
            patch(
                "api.chat.get_settings",
                return_value=SimpleNamespace(
                    rag_trace_include_content=False,
                    rag_pipeline_version="v2",
                ),
            ),
            patch("api.chat.run_rag_stream") as v1_stream,
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
            patch(
                "api.chat._validate_stream_answer_sources",
                new=trusted_source_refresh,
            ),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question="2",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        self.assertTrue(chunks)
        classify.assert_not_awaited()
        v1_stream.assert_not_called()
        self.assertIsNone(conversation.pending_route_state)
        self.assertEqual(conversation.route_state_revision, 4)

    async def test_grounded_followup_without_resolved_semantics_uses_current_turn_baseline(
        self,
    ) -> None:
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
        history = (
            {"role": "user", "content": "普通员工对应什么职级"},
            {"role": "assistant", "content": "普通员工对应D级。"},
        )
        carryover = ({
            "id": str(uuid.uuid4()),
            "doc_id": str(uuid.uuid4()),
            "kb_id": str(kb_id),
            "content": "普通员工对应D级。",
        },)
        standalone_query = "普通员工的餐补标准是多少"
        context = ConversationContext(
            is_followup=True,
            followup_reason="route_contextualized",
            standalone_query=standalone_query,
            history_messages=history,
            carryover_sources=carryover,
            relation="continuation",
            query_resolution_mode="contextualize",
            context_turn_keys=("t1",),
        )
        contract = _task_contract()
        decision = SimpleNamespace(
            need_retrieval=True,
            decision_reason="classified_retrieval",
            to_dict=lambda: {"intent_code": "knowledge_qa", "need_retrieval": True},
        )
        route_decision = SimpleNamespace(
            schema_version="rag_route_decision.v1",
            relation="continuation",
            evidence_scope="enterprise_kb",
            to_dict=lambda: {"schema_version": "rag_route_decision.v1"},
        )
        routing_result = SimpleNamespace(
            decision=decision,
            route_decision=route_decision,
            task_contract=contract,
            diagnostics={},
            route_log_id=None,
        )

        received, v1_stream, trace, chunks = await self._run_mocked_v2_dispatch(
            question="那餐补呢",
            kb_id=kb_id,
            conversation=conversation,
            user=user,
            context=context,
            routing_result=routing_result,
        )

        self.assertTrue(chunks)
        v1_stream.assert_not_called()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0]["question"], "那餐补呢")
        self.assertEqual(received[0]["standalone_query"], "那餐补呢")
        self.assertEqual(received[0]["conversation_history"], [])
        self.assertEqual(received[0]["carryover_sources"], [])
        self.assertFalse(received[0]["is_followup"])
        selection = next(
            call
            for call in trace.call_args_list
            if call.args and call.args[0] == "chat.pipeline_selected"
        )
        self.assertEqual(selection.kwargs["version"], "v2")
        self.assertEqual(selection.kwargs["reason"], "eligible_grounded_followup")

    async def test_evidence_scope_refinement_dispatches_v2_with_refined_query(
        self,
    ) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        pending["selection_mode"] = "refine"
        pending["choices"] = []
        pending["clarification_message"] = "请补充具体产品或版本。"
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=4,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=pending["original_query"],
            history_messages=(),
            carryover_sources=(),
            pending_route_state=pending,
        )
        contract = _task_contract()
        decision = SimpleNamespace(
            need_retrieval=True,
            decision_reason="classified_retrieval",
            to_dict=lambda: {"intent_code": "knowledge_qa", "need_retrieval": True},
        )
        route_decision = SimpleNamespace(
            schema_version="rag_route_decision.v1",
            relation="continuation",
            evidence_scope="enterprise_kb",
            to_dict=lambda: {"schema_version": "rag_route_decision.v1"},
        )
        routing_result = SimpleNamespace(
            decision=decision,
            route_decision=route_decision,
            task_contract=contract,
            diagnostics={},
            route_log_id=None,
        )

        received, v1_stream, trace, chunks = await self._run_mocked_v2_dispatch(
            question="云枢8.2.75版本",
            kb_id=kb_id,
            conversation=conversation,
            user=user,
            context=context,
            routing_result=routing_result,
        )

        self.assertTrue(chunks)
        v1_stream.assert_not_called()
        self.assertEqual(len(received), 1)
        self.assertIsNone(received[0]["evidence_scope_filter"])
        self.assertIn(pending["original_query"], received[0]["question"])
        self.assertIn("云枢8.2.75版本", received[0]["question"])
        self.assertEqual(received[0]["standalone_query"], received[0]["question"])
        self.assertTrue(received[0]["is_followup"] is False)
        selection = next(
            call
            for call in trace.call_args_list
            if call.args and call.args[0] == "chat.pipeline_selected"
        )
        self.assertEqual(selection.kwargs["version"], "v2")
        self.assertEqual(
            selection.kwargs["reason"],
            "eligible_evidence_scope_refinement",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
