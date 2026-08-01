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
    _messages_with_current_source_scope,
    _parse_sse_payload,
    _public_stream_error_message,
    send_message,
)
from core.conversation_context import ConversationContext
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
    now = datetime.now(UTC)
    return {
        "schema_version": "rag_pending_clarification.v2",
        "kind": "evidence_scope",
        "state_id": str(uuid.uuid4()),
        "base_user_message_id": str(uuid.uuid4()),
        "clarification_message_id": str(clarification_message_id),
        "original_query": "普通员工的出差标准是什么",
        "dimension": "version",
        "selection_mode": "choice",
        "choices": [
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
        ],
        "clarification_message": "检索到两个版本，请选择 2025 版或 2026 版。",
        "selected_kb_ids_snapshot": [str(kb_id)],
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "dispatch_authorized": False,
    }


class ChatStreamParsingTests(unittest.TestCase):
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
        self.assertTrue(clarification["acknowledged"])
        self.assertTrue(clarification["persisted"])
        self.assertEqual(clarification["pending_state_id"], pending["state_id"])
        self.assertEqual(clarification["clarification_message_id"], str(message_id))
        self.assertEqual(clarification["route_state_revision"], 7)
        self.assertEqual(
            set(clarification["choices"][0]),
            {
                "key",
                "label",
                "products",
                "versions",
                "projects",
                "filenames",
            },
        )
        public_payload = json.dumps(clarification, ensure_ascii=False)
        self.assertNotIn("doc_ids", public_payload)
        self.assertNotIn("kb_ids", public_payload)
        self.assertNotIn("anchor_doc_ids", public_payload)
        self.assertNotIn("canonical_products", public_payload)

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

        async def trusted_source_refresh(_db, *, raw_sources, raw_results, selected_kb_ids):
            # These tests exercise persistence/event accounting.  The dedicated
            # source-validation tests cover the real DB refresh boundary; use a
            # trusted fixture here so arbitrary UUID snapshots do not need a
            # database row for every case.
            del raw_results, selected_kb_ids
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
            patch("api.chat.run_rag_stream", new=successful_stream),
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
        answer = next(item for item in payloads if item["type"] == "text_delta")
        self.assertIn("无法确定", answer["content"])

    async def test_missing_object_followup_uses_standalone_query_for_intent_router(self) -> None:
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
        )
        decision = SimpleNamespace(
            need_retrieval=True,
            decision_reason="safe_fallback",
            to_dict=lambda: {
                "intent_code": "other",
                "need_retrieval": True,
                "decision_reason": "safe_fallback",
            },
        )
        routing_result = SimpleNamespace(decision=decision, route_log_id=None)
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

        self.assertEqual(classify.await_args.args[1], standalone_query)
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
        decision = SimpleNamespace(
            need_retrieval=True,
            decision_reason="classified_retrieval",
            to_dict=lambda: {
                "intent_code": "knowledge_qa",
                "need_retrieval": True,
                "decision_reason": "classified_retrieval",
            },
        )
        routing_result = SimpleNamespace(decision=decision, route_log_id=None)

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
            patch("api.chat.run_rag_stream", new=failing_stream),
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
        decision = SimpleNamespace(
            need_retrieval=False,
            decision_reason="general_chat",
            to_dict=lambda: {
                "intent_code": "general_chat",
                "need_retrieval": False,
                "decision_reason": "general_chat",
            },
        )
        routing_result = SimpleNamespace(decision=decision, route_log_id=None)

        async def cancelled_stream(**_kwargs):
            yield "data: " + json.dumps(
                {
                    "type": "evidence_clarification",
                    "schema_version": "rag_evidence_clarification.v1",
                    "needs_clarification": True,
                    "dimension": "document",
                    "question": "请选择需要查询的资料。",
                    "choices": [],
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
            patch("api.chat.run_rag_stream", new=cancelled_stream),
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
        self.assertIn("evidence_clarification", event_types)
        self.assertNotIn("evidence_clarification_ack", event_types)
        self.assertEqual(trace.call_args_list[-1].args[0], "chat.cancelled")
        self.assertEqual(trace.call_args_list[-1].kwargs["stage"], "streaming")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
