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
from core.query_route_compiler import (
    CompiledAnswerRequirement,
    RagTaskContract,
)
from core.query_route_contract import RouteClarification
from core.rag_v2.pipeline import run_rag_v2_stream
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
            "selected_tags": [],
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
                [],
                False,
                (),
                "eligible_evidence_scope_refinement",
            ),
            "followup": (
                None,
                False,
                [],
                True,
                (),
                "eligible_grounded_followup",
            ),
            "carryover": (
                None,
                False,
                [],
                False,
                ({"doc_id": "doc-1"},),
                "eligible_grounded_followup",
            ),
        }
        for name, (
            scope_filter,
            refinement_active,
            tags,
            is_followup,
            carryover,
            reason,
        ) in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    self._select(
                        evidence_scope_filter=scope_filter,
                        evidence_scope_refinement_active=refinement_active,
                        selected_tags=tags,
                        is_followup=is_followup,
                        carryover_sources=carryover,
                    ),
                    ("v2", reason),
                )

    def test_tags_use_v2_soft_boost(self) -> None:
        self.assertEqual(
            self._select(selected_tags=["制度"]),
            ("v2", "eligible_grounded_qa"),
        )

    def test_tags_keep_scope_selection_on_v2(self) -> None:
        self.assertEqual(
            self._select(
                evidence_scope_filter={"mode": "choice"},
                selected_tags=["制度"],
                is_followup=True,
                carryover_sources=({"doc_id": "doc-1"},),
            ),
            ("v2", "eligible_evidence_scope_selection"),
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

    async def get(self, _model, _identity):
        return self.conversation

    def add(self, value):
        self.added.append(value)

    async def commit(self):
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
                ),
            ),
            patch("api.chat.run_rag_stream") as v1_stream,
            patch("api.chat.run_rag_v2_stream", new=v2_stream),
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
            },
        )
        selection = next(
            call
            for call in trace.call_args_list
            if call.args and call.args[0] == "chat.pipeline_selected"
        )
        self.assertEqual(selection.kwargs["version"], "v2")
        self.assertEqual(selection.kwargs["reason"], "eligible_grounded_qa")

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
        self.assertTrue(received_kwargs[0]["is_followup"])
        self.assertEqual(received_kwargs[0]["carryover_sources"], [carryover_source])
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
        ):
            del raw_results, selected_kb_ids
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

    async def test_grounded_followup_dispatches_v2_with_contextualized_query_and_history(
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
        self.assertEqual(received[0]["standalone_query"], standalone_query)
        self.assertEqual(received[0]["conversation_history"], list(history))
        self.assertEqual(received[0]["carryover_sources"], list(carryover))
        self.assertTrue(received[0]["is_followup"])
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
