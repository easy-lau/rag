import asyncio
import json
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api.chat import (
    _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE,
    _bounded_search_snapshot,
    _messages_with_current_source_scope,
    _parse_sse_payload,
    _public_stream_error_message,
    send_message,
)
from core.conversation_context import ConversationContext, RouteTurnCandidate
from core.clarification import ClarificationContract, build_clarification_state
from core.query_route_compiler import (
    RouteCategoryPolicy,
    RouteCompilerConfig,
    compile_rag_task_contract,
)
from core.query_route_contract import parse_rag_route_decision
from models.db_models import Message
from models.schemas import ChatRequest


class _ChatDB:
    def __init__(self, conversation):
        self.conversation = conversation
        self.added = []
        self.commits = 0

    async def get(self, _model, _identity):
        return self.conversation

    def add_all(self, values):
        self.added.extend(values)

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commits += 1


class _SaveDB:
    def __init__(self, route_log=None):
        self.route_log = route_log
        self.added = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def get(self, _model, _identity):
        return self.route_log

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commits += 1


def _history_pending_state(
    *,
    clarification_message_id: uuid.UUID,
    kb_id: uuid.UUID,
    first_doc_id: uuid.UUID,
    second_doc_id: uuid.UUID,
) -> dict:
    return build_clarification_state(
        contract=ClarificationContract(
            adapter="evidence",
            dimension="version",
            reason_code="multiple_authorized_versions",
            selection_mode="choice",
            choices=(
                {
                    "key": "c1",
                    "label": "2025 版差旅标准",
                    "products": ["差旅制度"],
                    "canonical_products": ["差旅制度"],
                    "versions": ["2025"],
                    "projects": [],
                    "filenames": ["差旅标准-2025.md"],
                    "kb_ids": [str(kb_id)],
                    "doc_ids": [str(first_doc_id)],
                    "anchor_doc_ids": [str(first_doc_id)],
                    "companion_doc_ids": [],
                },
                {
                    "key": "c2",
                    "label": "2026 版差旅标准",
                    "products": ["差旅制度"],
                    "canonical_products": ["差旅制度"],
                    "versions": ["2026"],
                    "projects": [],
                    "filenames": ["差旅标准-2026.md"],
                    "kb_ids": [str(kb_id)],
                    "doc_ids": [str(second_doc_id)],
                    "anchor_doc_ids": [str(second_doc_id)],
                    "companion_doc_ids": [],
                },
            ),
        ),
        original_query="普通员工的出差标准是什么",
        selected_kb_ids=[kb_id],
        base_user_message_id=uuid.uuid4(),
        clarification_message_id=clarification_message_id,
    )


def _routing_result(
    *,
    selected_kb_count: int,
    need_retrieval: bool = True,
    relation: str = "new",
    route_log_id: uuid.UUID | None = None,
):
    intent_code = "knowledge_qa" if need_retrieval else "general_chat"
    action = "retrieve" if need_retrieval else "chat"
    evidence_scope = "enterprise_kb" if need_retrieval else "general_world"
    route = parse_rag_route_decision(
        {
            "schema_version": "rag_route_decision.v1",
            "readiness": "ready",
            "intent_code": intent_code,
            "relation": relation,
            "evidence_scope": evidence_scope,
            "query_resolution": {"mode": "current", "context_turn_keys": []},
            "requirements": [
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "回答用户当前输入",
                }
            ],
            "clarification": {"question": "", "unresolved": []},
            "confidence": 0.96,
            "rationale": "chat stream regression fixture",
        },
        allowed_intent_codes=[intent_code],
    )
    contract = compile_rag_task_contract(
        route,
        RouteCategoryPolicy(code=intent_code, name=intent_code, action=action),
        RouteCompilerConfig(),
        question="用户当前输入",
        selected_kb_count=selected_kb_count,
        source="test",
    )
    decision_payload = {
        "intent_code": intent_code,
        "intent_name": intent_code,
        "action": action,
        "confidence": route.confidence,
        "source": "test",
        "response_mode": contract.response_mode,
        "retrieval_policy": contract.retrieval_policy,
        "need_retrieval": contract.need_retrieval,
        "decision_reason": contract.decision_reason,
    }
    decision = SimpleNamespace(
        need_retrieval=contract.need_retrieval,
        decision_reason=contract.decision_reason,
        to_dict=lambda: dict(decision_payload),
    )
    return SimpleNamespace(
        decision=decision,
        route_log_id=route_log_id,
        route_decision=route,
        task_contract=contract,
        diagnostics={},
    )


class ChatStreamParsingTests(unittest.TestCase):
    def test_search_snapshot_persists_general_answer_provenance(self) -> None:
        snapshot = _bounded_search_snapshot({
            "results": [],
            "answer_sources": [],
            "evidence_status": "no_hit",
            "answer_provenance": "general_model",
            "general_fallback_mode": "no_hit",
        })

        self.assertEqual(
            snapshot["counters"]["answer_provenance"],
            "general_model",
        )
        self.assertEqual(
            snapshot["counters"]["general_fallback_mode"],
            "no_hit",
        )

    def test_search_snapshot_drops_unknown_general_fallback_metadata(self) -> None:
        snapshot = _bounded_search_snapshot({
            "answer_provenance": "producer_defined",
            "general_fallback_mode": "always",
        })

        self.assertNotIn("answer_provenance", snapshot["counters"])
        self.assertNotIn("general_fallback_mode", snapshot["counters"])

    def test_search_snapshot_persists_bounded_evidence_quality(self) -> None:
        quality = {
            "coverage": "medium",
            "coverage_ratio": 0.5,
            "reliability": "high",
            "freshness": "unknown",
            "consistency": "high",
            "completeness": "partial",
            "missing_requirement_ids": ["r2"],
        }
        snapshot = _bounded_search_snapshot({"evidence_quality": quality})

        self.assertEqual(snapshot["counters"]["evidence_quality"], quality)

    def test_search_snapshot_rejects_malformed_evidence_quality(self) -> None:
        snapshot = _bounded_search_snapshot({
            "evidence_quality": {
                "coverage": "excellent",
                "coverage_ratio": 9,
                "reliability": "high",
                "freshness": "unknown",
                "consistency": "high",
                "completeness": "partial",
                "missing_requirement_ids": ["r2"],
            },
        })

        self.assertNotIn("evidence_quality", snapshot["counters"])

    def test_text_delta_content_cannot_spoof_search_results_event(self) -> None:
        content = '下面只是正文示例：{"type": "search_results"}，不是 SSE 事件。'
        chunk = (
            "data: "
            + json.dumps(
                {"type": "text_delta", "content": content},
                ensure_ascii=False,
            )
            + "\n\n"
        )

        payload = _parse_sse_payload(chunk)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["type"], "text_delta")
        self.assertEqual(payload["content"], content)

    def test_public_stream_error_hides_upstream_details(self) -> None:
        message = _public_stream_error_message(
            RuntimeError("POST https://provider.example/v1/chat secret response")
        )

        self.assertEqual(message, "回答生成失败，请稍后重试")
        self.assertNotIn("provider", message)

    def test_public_stream_timeout_keeps_actionable_category(self) -> None:
        class APITimeoutError(RuntimeError):
            pass

        message = _public_stream_error_message(APITimeoutError("Request timed out"))

        self.assertEqual(message, "模型服务响应超时，请稍后重试")


class ChatHistoricalSourceScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_search_snapshot_is_reauthorized_and_rehydrated(self) -> None:
        conversation_id = uuid.uuid4()
        allowed_kb_id = uuid.uuid4()
        revoked_kb_id = uuid.uuid4()
        allowed_doc_id = uuid.uuid4()
        revoked_doc_id = uuid.uuid4()
        allowed_chunk_id = uuid.uuid4()
        revoked_chunk_id = uuid.uuid4()
        row = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="历史回答",
            sources=[],
            search_snapshot={
                "schema_version": "rag_search_snapshot.v1",
                "candidates": [
                    {
                        "kb_id": str(allowed_kb_id),
                        "doc_id": str(allowed_doc_id),
                        "id": str(allowed_chunk_id),
                        "content": "不得返回的旧正文",
                        "source_url": "https://stale.example/private",
                        "evidence_role": "direct",
                        "score": 0.9,
                    },
                    {
                        "kb_id": str(revoked_kb_id),
                        "doc_id": str(revoked_doc_id),
                        "id": str(revoked_chunk_id),
                        "content": "已经撤权的候选正文",
                        "evidence_role": "related",
                    },
                ],
                "answer_sources": [],
                "counters": {
                    "evidence_status": "hit",
                    "retrieval_executed": True,
                    "hit_count": 1,
                },
            },
            created_at=datetime.now(UTC),
        )
        current_chunk = SimpleNamespace(
            id=allowed_chunk_id,
            doc_id=allowed_doc_id,
            kb_id=allowed_kb_id,
            content="当前授权正文",
            chunk_index=1,
            metadata_={},
        )
        current_document = SimpleNamespace(
            filename="当前授权文档.md",
            file_type="md",
            source_url="https://current.example/doc",
            image_url=None,
            tags=[],
        )
        db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(
                    all=lambda: [(current_chunk, current_document)]
                )
            )
        )
        with patch(
            "api.chat.get_accessible_kb_ids",
            new=AsyncMock(return_value=[allowed_kb_id]),
        ):
            messages = await _messages_with_current_source_scope(
                [row],
                user=SimpleNamespace(id=uuid.uuid4()),
                db=db,
            )

        snapshot = messages[0].search_snapshot
        self.assertEqual(snapshot["counters"]["evidence_status"], "hit")
        self.assertEqual(len(snapshot["candidates"]), 1)
        self.assertEqual(snapshot["candidates"][0]["content"], "当前授权正文")
        serialized = json.dumps(snapshot, ensure_ascii=False)
        self.assertNotIn("不得返回的旧正文", serialized)
        self.assertNotIn("已经撤权", serialized)
        self.assertNotIn("stale.example", serialized)

    async def test_history_sources_are_filtered_by_current_scope_and_document_state(self) -> None:
        conversation_id = uuid.uuid4()
        allowed_kb_id = uuid.uuid4()
        revoked_kb_id = uuid.uuid4()
        allowed_doc_id = uuid.uuid4()
        revoked_doc_id = uuid.uuid4()
        allowed_chunk_id = uuid.uuid4()
        revoked_chunk_id = uuid.uuid4()
        stale_chunk_id = uuid.uuid4()
        row = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="历史回答",
            sources=[
                {
                    "kb_id": str(allowed_kb_id),
                    "doc_id": str(allowed_doc_id),
                    "id": str(allowed_chunk_id),
                    "content": "旧快照内容",
                },
                {
                    "kb_id": str(revoked_kb_id),
                    "doc_id": str(revoked_doc_id),
                    "id": str(revoked_chunk_id),
                    "content": "已经撤权的敏感片段",
                    "source_url": "https://private.example/doc",
                },
                {
                    "kb_id": str(allowed_kb_id),
                    "doc_id": str(allowed_doc_id),
                    "id": str(stale_chunk_id),
                    "content": "已重新分块删除的旧片段",
                },
            ],
            created_at=datetime.now(UTC),
        )
        current_chunk = SimpleNamespace(
            id=allowed_chunk_id,
            doc_id=allowed_doc_id,
            kb_id=allowed_kb_id,
            content="当前仍有权查看的片段",
            chunk_index=2,
            metadata_={"section": "current"},
        )
        current_document = SimpleNamespace(
            filename="当前文档.md",
            file_type="md",
            source_url="https://current.example/doc",
            image_url=None,
            tags=["当前"],
        )
        db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(
                    all=lambda: [(current_chunk, current_document)],
                )
            )
        )
        user = SimpleNamespace(id=uuid.uuid4())

        with patch(
            "api.chat.get_accessible_kb_ids",
            new=AsyncMock(return_value=[allowed_kb_id]),
        ):
            messages = await _messages_with_current_source_scope(
                [row],
                user=user,
                db=db,
            )

        self.assertEqual(len(messages[0].sources), 1)
        self.assertEqual(messages[0].sources[0]["doc_id"], str(allowed_doc_id))
        self.assertEqual(messages[0].sources[0]["content"], "当前仍有权查看的片段")
        self.assertEqual(messages[0].sources[0]["filename"], "当前文档.md")
        self.assertNotIn("敏感片段", json.dumps(messages[0].sources, ensure_ascii=False))
        self.assertNotIn("已重新分块", json.dumps(messages[0].sources, ensure_ascii=False))

    async def test_history_with_non_list_sources_does_not_break_the_conversation(self) -> None:
        row = Message(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            role="assistant",
            content="历史回答",
            sources={"unexpected": "legacy-shape"},
            created_at=datetime.now(UTC),
        )
        db = SimpleNamespace(execute=AsyncMock())
        with patch(
            "api.chat.get_accessible_kb_ids",
            new=AsyncMock(return_value=[]),
        ):
            messages = await _messages_with_current_source_scope(
                [row],
                user=SimpleNamespace(id=uuid.uuid4()),
                db=db,
            )

        self.assertIsNone(messages[0].sources)
        db.execute.assert_not_awaited()

    async def test_legacy_non_answer_sources_are_not_returned_as_citations(self) -> None:
        conversation_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        for evidence_status in (
            "no_hit",
            "insufficient_evidence",
            "skipped",
            "error",
            "needs_clarification",
            "version_mismatch",
        ):
            with self.subTest(evidence_status=evidence_status):
                row = Message(
                    id=uuid.uuid4(),
                    conversation_id=conversation_id,
                    role="assistant",
                    content="历史回答",
                    sources=[{
                        "kb_id": str(kb_id),
                        "doc_id": str(doc_id),
                        "id": str(uuid.uuid4()),
                        "content": "旧版本宽检索候选",
                        "evidence_status": evidence_status,
                    }],
                    created_at=datetime.now(UTC),
                )
                db = SimpleNamespace(execute=AsyncMock())
                with patch(
                    "api.chat.get_accessible_kb_ids",
                    new=AsyncMock(return_value=None),
                ):
                    messages = await _messages_with_current_source_scope(
                        [row],
                        user=SimpleNamespace(id=uuid.uuid4()),
                        db=db,
                    )

                self.assertEqual(messages[0].sources, [])
                db.execute.assert_not_awaited()


class ChatHistoricalClarificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_pending_clarification_is_restored_with_public_choices(self) -> None:
        conversation_id = uuid.uuid4()
        message_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        first_doc_id = uuid.uuid4()
        second_doc_id = uuid.uuid4()
        row = Message(
            id=message_id,
            conversation_id=conversation_id,
            role="assistant",
            content="检索到两个版本，请选择 2025 版或 2026 版。",
            sources=[],
            created_at=datetime.now(UTC),
        )
        pending = _history_pending_state(
            clarification_message_id=message_id,
            kb_id=kb_id,
            first_doc_id=first_doc_id,
            second_doc_id=second_doc_id,
        )
        db = SimpleNamespace(
            execute=AsyncMock(
                return_value=SimpleNamespace(
                    all=lambda: [
                        (first_doc_id, kb_id),
                        (second_doc_id, kb_id),
                    ]
                )
            )
        )

        with patch(
            "api.chat.get_accessible_kb_ids",
            new=AsyncMock(return_value=[kb_id]),
        ):
            messages = await _messages_with_current_source_scope(
                [row],
                user=SimpleNamespace(id=uuid.uuid4()),
                db=db,
                pending_route_state=pending,
                route_state_revision=7,
            )

        clarification = messages[0].clarification
        self.assertIsNotNone(clarification)
        self.assertEqual(clarification["status"], "active")
        self.assertTrue(clarification["persisted"])
        self.assertEqual(clarification["pending_state_id"], pending["state_id"])
        self.assertEqual(clarification["clarification_message_id"], str(message_id))
        self.assertEqual(clarification["route_state_revision"], 7)
        self.assertEqual(
            set(clarification["choices"][0]),
            {
                "key",
                "label",
                "value",
                "products",
                "canonical_products",
                "versions",
                "projects",
                "filenames",
            },
        )
        for choice in clarification["choices"]:
            self.assertNotIn("doc_ids", choice)
            self.assertNotIn("kb_ids", choice)
            self.assertNotIn("anchor_doc_ids", choice)

    async def test_history_does_not_restore_invalid_or_unauthorized_clarification(self) -> None:
        conversation_id = uuid.uuid4()
        message_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        first_doc_id = uuid.uuid4()
        second_doc_id = uuid.uuid4()
        base_pending = _history_pending_state(
            clarification_message_id=message_id,
            kb_id=kb_id,
            first_doc_id=first_doc_id,
            second_doc_id=second_doc_id,
        )
        expired = dict(base_pending)
        expired["created_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        expired["expires_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        mismatched = dict(base_pending)
        mismatched["clarification_message_id"] = str(uuid.uuid4())
        cases = (
            (
                "expired",
                expired,
                [kb_id],
                [(first_doc_id, kb_id), (second_doc_id, kb_id)],
            ),
            (
                "message_mismatch",
                mismatched,
                [kb_id],
                [(first_doc_id, kb_id), (second_doc_id, kb_id)],
            ),
            (
                "kb_scope_revoked",
                base_pending,
                [],
                [(first_doc_id, kb_id), (second_doc_id, kb_id)],
            ),
            (
                "document_inactive_or_unready",
                base_pending,
                [kb_id],
                [(first_doc_id, kb_id)],
            ),
        )

        for case_name, pending, accessible, document_rows in cases:
            with self.subTest(case=case_name):
                row = Message(
                    id=message_id,
                    conversation_id=conversation_id,
                    role="assistant",
                    content="请选择版本",
                    sources=[],
                    created_at=datetime.now(UTC),
                )
                db = SimpleNamespace(
                    execute=AsyncMock(
                        return_value=SimpleNamespace(
                            all=lambda rows=document_rows: rows
                        )
                    )
                )
                with patch(
                    "api.chat.get_accessible_kb_ids",
                    new=AsyncMock(return_value=accessible),
                ):
                    messages = await _messages_with_current_source_scope(
                        [row],
                        user=SimpleNamespace(id=uuid.uuid4()),
                        db=db,
                        pending_route_state=pending,
                        route_state_revision=3,
                    )

                self.assertIsNone(messages[0].clarification)


class ChatAnswerSourcePersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def _run_successful_stream(
        self,
        search_event: dict,
        *,
        before_search_content: str | None = None,
        producer_state: dict[str, bool] | None = None,
    ):
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        route_log_id = uuid.uuid4()
        conversation = SimpleNamespace(id=conversation_id, user_id=user_id)
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _ChatDB(conversation)
        route_log = SimpleNamespace(
            retrieval_executed=None,
            evidence_status=None,
            hit_count=None,
        )
        save_db = _SaveDB(route_log)
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="云枢登录问题",
            history_messages=(),
            carryover_sources=(),
        )
        routing_result = _routing_result(
            selected_kb_count=1,
            route_log_id=route_log_id,
        )

        async def successful_stream(**_kwargs):
            try:
                if before_search_content is not None:
                    yield f"data: {json.dumps({'type': 'text_delta', 'content': before_search_content}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(search_event, ensure_ascii=False)}\n\n"
                if producer_state is not None:
                    producer_state["resumed_after_search"] = True
                yield f"data: {json.dumps({'type': 'text_delta', 'content': '测试回答'}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'usage', 'total_tokens': 12})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conversation_id)})}\n\n"
            finally:
                if producer_state is not None:
                    producer_state["closed"] = True

        async def trusted_source_refresh(
            _db,
            *,
            raw_sources,
            raw_results,
            selected_kb_ids,
            read_session_factory=None,
            allow_unverified=False,
        ):
            # These tests exercise persistence/event accounting.  The dedicated
            # source-validation tests cover the real DB refresh boundary; use a
            # trusted fixture here so arbitrary UUID snapshots do not need a
            # database row for every case.
            del raw_results, selected_kb_ids, read_session_factory, allow_unverified
            sources = list(raw_sources or [])
            pairs = {
                (uuid.UUID(str(source["kb_id"])), uuid.UUID(str(source["doc_id"])))
                for source in sources
                if isinstance(source, dict)
            }
            return sources, pairs, None

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
            patch("api.chat.run_rag_v2_stream", new=successful_stream),
            patch(
                "api.chat._validate_stream_answer_sources",
                new=trusted_source_refresh,
            ),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="云枢登录问题",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[uuid.uuid4()],
                ),
                db=request_db,
                user=user,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        payloads = [
            _parse_sse_payload(chunk.decode() if isinstance(chunk, bytes) else chunk)
            for chunk in chunks
        ]
        assistant = next(message for message in save_db.added if message.role == "assistant")
        response_trace = next(
            call
            for call in reversed(trace.call_args_list)
            if call.args and call.args[0] == "chat.response"
        )
        return payloads, assistant, route_log, response_trace

    @staticmethod
    def _source(*, role: str, filename: str) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "chunk_id": str(uuid.uuid4()),
            "doc_id": str(uuid.uuid4()),
            "kb_id": str(uuid.uuid4()),
            "content": f"{filename}内容",
            "filename": filename,
            "evidence_role": role,
            "constraint_status": "neutral",
            "answer_support": 0.2 if role == "related" else 0.9,
        }

    async def test_no_hit_persists_no_sources_but_keeps_panel_results(self) -> None:
        related = self._source(role="related", filename="相近资料.md")
        event = {
            "type": "search_results",
            "results": [related],
            "answer_sources": [],
            "total": 1,
            "displayed_result_count": 1,
            "context_evidence_count": 0,
            "hit_count": 0,
            "retrieval_executed": True,
            "evidence_status": "no_hit",
            "direct_evidence_count": 0,
            "related_reference_count": 1,
        }

        payloads, assistant, route_log, response_trace = (
            await self._run_successful_stream(event)
        )

        panel_event = next(item for item in payloads if item and item["type"] == "search_results")
        self.assertEqual(len(panel_event["results"]), 1)
        self.assertEqual(panel_event["answer_sources"], [])
        self.assertEqual(assistant.sources, [])
        self.assertEqual(route_log.hit_count, 0)
        self.assertEqual(response_trace.kwargs["displayed_result_count"], 1)
        self.assertEqual(response_trace.kwargs["context_evidence_count"], 0)
        self.assertEqual(response_trace.kwargs["hit_count"], 0)
        self.assertEqual(response_trace.kwargs["sources"], [])

    async def test_non_answer_statuses_fail_closed_on_claimed_answer_sources(self) -> None:
        claimed_source = self._source(role="direct", filename="错误携带的证据.md")
        for evidence_status in (
            "no_hit",
            "insufficient_evidence",
            "skipped",
            "error",
            "needs_clarification",
            "version_mismatch",
        ):
            with self.subTest(evidence_status=evidence_status):
                event = {
                    "type": "search_results",
                    "results": [claimed_source],
                    # 模拟旧版、插件或异常上游产生自相矛盾的协议：状态明确
                    # 表示没有回答依据，却仍携带来源与非零命中数。
                    "answer_sources": [claimed_source],
                    "total": 1,
                    "displayed_result_count": 1,
                    "context_evidence_count": 1,
                    "hit_count": 1,
                    "retrieval_executed": evidence_status != "skipped",
                    "evidence_status": evidence_status,
                    "direct_evidence_count": 1,
                    "related_reference_count": 0,
                }

                _payloads, assistant, route_log, response_trace = (
                    await self._run_successful_stream(event)
                )

                self.assertEqual(assistant.sources, [])
                self.assertEqual(route_log.hit_count, 0)
                self.assertEqual(
                    response_trace.kwargs["displayed_result_count"],
                    0,
                )
                self.assertEqual(
                    response_trace.kwargs["context_evidence_count"],
                    0,
                )
                self.assertEqual(response_trace.kwargs["hit_count"], 0)
                self.assertEqual(response_trace.kwargs["sources"], [])

    async def test_hit_and_partial_persist_exact_generation_sources(self) -> None:
        direct = self._source(role="direct", filename="直接证据.md")
        related = self._source(role="related", filename="相近资料.md")
        cases = (
            ("hit", [direct, related], [direct], 1),
            ("partial", [related], [related], 0),
        )
        for evidence_status, results, answer_sources, direct_count in cases:
            with self.subTest(evidence_status=evidence_status):
                event = {
                    "type": "search_results",
                    "results": results,
                    "answer_sources": answer_sources,
                    "total": len(results),
                    "displayed_result_count": len(results),
                    "context_evidence_count": len(answer_sources),
                    "hit_count": direct_count,
                    "retrieval_executed": True,
                    "evidence_status": evidence_status,
                    "direct_evidence_count": direct_count,
                    "related_reference_count": sum(
                        source["evidence_role"] == "related" for source in results
                    ),
                }

                _payloads, assistant, route_log, response_trace = (
                    await self._run_successful_stream(event)
                )

                self.assertEqual(
                    [source["filename"] for source in assistant.sources],
                    [source["filename"] for source in answer_sources],
                )
                self.assertEqual(route_log.hit_count, direct_count)
                self.assertEqual(
                    response_trace.kwargs["displayed_result_count"],
                    len(results),
                )
                self.assertEqual(
                    response_trace.kwargs["context_evidence_count"],
                    len(answer_sources),
                )
                self.assertEqual(response_trace.kwargs["hit_count"], direct_count)
                self.assertEqual(
                    len(response_trace.kwargs["sources"]),
                    len(answer_sources),
                )

    async def test_unverified_partial_sources_keep_their_label_in_history(self) -> None:
        source = self._source(role="unverified", filename="待验证参考.md")
        source["id"] = source["chunk_id"]
        source["source_verification"] = "unverified"
        event = {
            "type": "search_results",
            "results": [source],
            "answer_sources": [source],
            "total": 1,
            "displayed_result_count": 1,
            "context_evidence_count": 1,
            "hit_count": 0,
            "retrieval_executed": True,
            "evidence_status": "partial",
            "direct_evidence_count": 0,
            "related_reference_count": 0,
            "unverified_reference_count": 1,
            "unverified_generation": True,
            "source_verification": "unverified",
        }

        payloads, assistant, route_log, response_trace = (
            await self._run_successful_stream(event)
        )

        search_event = next(
            item for item in payloads if item and item["type"] == "search_results"
        )
        self.assertEqual(search_event["evidence_status"], "partial")
        self.assertEqual(search_event["direct_evidence_count"], 0)
        self.assertEqual(search_event["hit_count"], 0)
        self.assertEqual(len(assistant.sources), 1)
        self.assertEqual(
            assistant.sources[0]["source_verification"],
            "unverified",
        )
        snapshot = _bounded_search_snapshot(search_event)
        self.assertTrue(snapshot["counters"]["unverified_generation"])
        self.assertEqual(
            snapshot["counters"]["source_verification"],
            "unverified",
        )
        self.assertEqual(
            snapshot["answer_sources"][0]["evidence_role"],
            "unverified",
        )
        self.assertEqual(route_log.evidence_status, "partial")
        self.assertEqual(response_trace.kwargs["hit_count"], 0)

    async def test_positive_status_without_valid_sources_locks_generation(self) -> None:
        event = {
            "type": "search_results",
            "results": [],
            "answer_sources": [],
            "retrieval_executed": True,
            "evidence_status": "hit",
            "direct_evidence_count": 1,
            "related_reference_count": 0,
        }

        producer_state = {
            "resumed_after_search": False,
            "closed": False,
        }
        payloads, assistant, route_log, response_trace = (
            await self._run_successful_stream(
                event,
                before_search_content="不应被保存的前置模型片段",
                producer_state=producer_state,
            )
        )

        text_deltas = [
            item["content"]
            for item in payloads
            if item and item.get("type") == "text_delta"
        ]
        # SSE cannot retract a custom producer's already-sent prefix, but the
        # API clears its persistence buffer and suppresses every later model
        # delta once the invalid evidence event arrives.
        self.assertIn("不应被保存的前置模型片段", text_deltas)
        self.assertIn(
            _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE,
            text_deltas,
        )
        self.assertNotIn("测试回答", text_deltas)
        self.assertEqual(
            assistant.content,
            _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE,
        )
        self.assertEqual(assistant.sources, [])
        self.assertEqual(route_log.evidence_status, "error")
        self.assertTrue(
            response_trace.kwargs["evidence_source_validation_locked"]
        )
        self.assertFalse(producer_state["resumed_after_search"])
        self.assertTrue(producer_state["closed"])

    async def test_non_answer_status_with_claimed_sources_stops_producer(self) -> None:
        claimed_source = self._source(
            role="direct",
            filename="不应出现的证据.md",
        )
        event = {
            "type": "search_results",
            "results": [claimed_source],
            "answer_sources": [claimed_source],
            "retrieval_executed": True,
            "evidence_status": "no_hit",
            "direct_evidence_count": 1,
            "related_reference_count": 0,
        }
        producer_state = {
            "resumed_after_search": False,
            "closed": False,
        }

        payloads, assistant, route_log, response_trace = (
            await self._run_successful_stream(
                event,
                producer_state=producer_state,
            )
        )

        text_deltas = [
            item["content"]
            for item in payloads
            if item and item.get("type") == "text_delta"
        ]
        self.assertEqual(
            text_deltas,
            [_EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE],
        )
        self.assertEqual(
            assistant.content,
            _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE,
        )
        self.assertEqual(assistant.sources, [])
        self.assertEqual(route_log.evidence_status, "error")
        self.assertTrue(
            response_trace.kwargs["evidence_source_validation_locked"]
        )
        self.assertFalse(producer_state["resumed_after_search"])
        self.assertTrue(producer_state["closed"])

    async def test_unknown_evidence_status_is_fail_closed(self) -> None:
        event = {
            "type": "search_results",
            "results": [],
            "answer_sources": [],
            "retrieval_executed": True,
            "evidence_status": "future_status",
            "direct_evidence_count": 0,
            "related_reference_count": 0,
        }

        payloads, assistant, route_log, _trace = (
            await self._run_successful_stream(event)
        )
        search_event = next(
            item for item in payloads if item and item["type"] == "search_results"
        )
        self.assertEqual(search_event["evidence_status"], "error")
        self.assertEqual(search_event["results"], [])
        self.assertEqual(
            assistant.content,
            _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE,
        )
        self.assertEqual(route_log.evidence_status, "error")


class ChatUnresolvedReferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_unresolved_reference_returns_clarification_without_model_or_retrieval(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        conversation = SimpleNamespace(id=conversation_id, user_id=user_id)
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _ChatDB(conversation)
        context = ConversationContext(
            is_followup=False,
            followup_reason="unresolved_reference:这些",
            standalone_query="这些配置有什么影响",
            history_messages=(),
            carryover_sources=(),
            unresolved_reference=True,
        )

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.classify_intent_result", new=AsyncMock()) as classify,
            patch("api.chat.run_rag_stream") as rag_stream,
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="这些配置有什么影响",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[uuid.uuid4()],
                ),
                db=db,
                user=user,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        payloads = [
            _parse_sse_payload(
                chunk.decode() if isinstance(chunk, bytes) else chunk
            )
            for chunk in chunks
        ]
        payloads = [payload for payload in payloads if payload]
        classify.assert_not_awaited()
        rag_stream.assert_not_called()
        self.assertEqual(
            [call.args[0] for call in trace.call_args_list],
            [
                "chat.request",
                "conversation.context_resolved",
                "conversation.context_candidates",
                "conversation.reference_unresolved",
                "chat.response",
            ],
        )
        self.assertEqual(db.commits, 1)
        self.assertEqual([message.role for message in db.added], ["user", "assistant"])
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertFalse(result["retrieval_executed"])
        self.assertEqual(result["decision_reason"], "unresolved_reference")
        clarification = next(
            item
            for item in payloads
            if item["type"] == "clarification_state" and item["status"] == "active"
        )
        self.assertEqual(clarification["adapter"], "semantic")
        self.assertFalse(any(item["type"] == "text_delta" for item in payloads))

    async def test_missing_object_followup_passes_current_turn_and_route_candidate_separately(
        self,
    ) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        conversation = SimpleNamespace(id=conversation_id, user_id=user_id)
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _ChatDB(conversation)
        standalone_query = "云枢中如何配置登录用户名枚举"
        context = ConversationContext(
            is_followup=True,
            followup_reason="missing_action_object",
            standalone_query=standalone_query,
            history_messages=(
                {"role": "user", "content": "登录用户名枚举是什么"},
                {"role": "assistant", "content": "它是一类登录信息泄露风险。"},
            ),
            carryover_sources=(),
            route_turn_candidates=(
                RouteTurnCandidate(
                    candidate_key="t1",
                    user_question="登录用户名枚举是什么",
                    assistant_answer="它是一类登录信息泄露风险。",
                ),
            ),
        )
        routing_result = _routing_result(
            selected_kb_count=1,
            relation="followup",
        )
        classify = AsyncMock(return_value=routing_result)

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.classify_intent_result", new=classify),
            patch("api.chat.trace_event"),
        ):
            await send_message(
                ChatRequest(
                    question="云枢中如何配置",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=db,
                user=user,
            )

        # V2 routing receives only the current user text.  Historical wording
        # is an explicit, bounded ``t1`` candidate rather than being merged
        # into a synthetic standalone question before the route contract is
        # validated.
        self.assertEqual(classify.await_args.args[1], "云枢中如何配置")
        self.assertNotEqual(classify.await_args.args[1], standalone_query)
        self.assertEqual(
            classify.await_args.kwargs["route_context"],
            ({
                "candidate_key": "t1",
                "user_input": "登录用户名枚举是什么",
                "assistant_answer": "它是一类登录信息泄露风险。",
                "reusable_source_count": 0,
            },),
        )
        self.assertEqual(classify.await_args.kwargs["selected_kb_ids"], [kb_id])

    async def test_required_retrieval_without_kb_finishes_trace_as_validation_error(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        conversation = SimpleNamespace(id=conversation_id, user_id=user_id)
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _ChatDB(conversation)
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="云枢默认密码怎么配置",
            history_messages=(),
            carryover_sources=(),
        )
        decision = SimpleNamespace(
            need_retrieval=True,
            decision_reason="intent_requires_retrieval",
            to_dict=lambda: {
                "intent_code": "knowledge_qa",
                "need_retrieval": True,
                "decision_reason": "intent_requires_retrieval",
            },
        )
        routing_result = SimpleNamespace(decision=decision, route_log_id=None)

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
            patch("api.chat.trace_event") as trace,
        ):
            with self.assertRaises(HTTPException) as raised:
                await send_message(
                    ChatRequest(
                        question="云枢默认密码怎么配置",
                        conversation_id=conversation_id,
                        knowledge_base_ids=[],
                    ),
                    db=db,
                    user=user,
                )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(
            [call.args[0] for call in trace.call_args_list],
            [
                "chat.request",
                "conversation.context_resolved",
                "conversation.context_candidates",
                "intent.routing_decision",
                "chat.error",
            ],
        )
        self.assertEqual(trace.call_args_list[-1].kwargs["stage"], "request_validation")

    async def test_stream_runtime_error_keeps_safe_sse_and_terminal_trace(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        conversation = SimpleNamespace(id=conversation_id, user_id=user_id)
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _ChatDB(conversation)
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="云枢默认密码怎么配置",
            history_messages=(),
            carryover_sources=(),
        )
        routing_result = _routing_result(selected_kb_count=1)

        received_kwargs = []

        async def failing_stream(**kwargs):
            received_kwargs.append(kwargs)
            if False:  # pragma: no cover - makes this an async generator
                yield ""
            raise RuntimeError("https://provider.example/private response")

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
            patch("api.chat.run_rag_v2_stream", new=failing_stream),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="云枢默认密码怎么配置",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=db,
                user=user,
            )
            chunks = [chunk async for chunk in response.body_iterator]

        payloads = [
            _parse_sse_payload(chunk.decode() if isinstance(chunk, bytes) else chunk)
            for chunk in chunks
        ]
        error = next(item for item in payloads if item and item["type"] == "error")
        self.assertEqual(error["message"], "回答生成失败，请稍后重试")
        self.assertNotIn("provider", error["message"])
        self.assertEqual(trace.call_args_list[-1].args[0], "chat.error")
        self.assertEqual(received_kwargs[0]["conversation_history"], [])

    async def test_cancelled_stream_is_traced_and_re_raised(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        conversation = SimpleNamespace(id=conversation_id, user_id=user_id)
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _ChatDB(conversation)
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="普通问题",
            history_messages=(),
            carryover_sources=(),
        )
        routing_result = _routing_result(
            selected_kb_count=0,
            need_retrieval=False,
        )

        async def cancelled_stream(**_kwargs):
            yield "data: " + json.dumps(
                {
                    "type": "clarification_state",
                    "schema_version": "rag_clarification_state.v1",
                    "status": "proposed",
                    "persisted": False,
                    "needs_clarification": True,
                    "adapter": "semantic",
                    "dimension": "document",
                    "reason_code": "missing_document",
                    "selection_mode": "refine",
                    "choices": [],
                    "allowed_actions": ["refine", "cancel", "new_question"],
                    "pending_state_id": None,
                    "clarification_message_id": None,
                    "route_state_revision": None,
                },
                ensure_ascii=False,
            ) + "\n\n"
            raise asyncio.CancelledError

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
            patch("api.chat.run_direct_response_stream", new=cancelled_stream),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="普通问题",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[],
                ),
                db=db,
                user=user,
            )
            chunks = []
            with self.assertRaises(asyncio.CancelledError):
                async for chunk in response.body_iterator:
                    chunks.append(chunk)

        payloads = [
            _parse_sse_payload(chunk.decode() if isinstance(chunk, bytes) else chunk)
            for chunk in chunks
        ]
        event_types = [item["type"] for item in payloads if item]
        self.assertIn("clarification_state", event_types)
        self.assertEqual(trace.call_args_list[-1].args[0], "chat.cancelled")
        self.assertEqual(trace.call_args_list[-1].kwargs["stage"], "streaming")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
