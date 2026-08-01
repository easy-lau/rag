import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api.chat import (
    _mark_turn_persist_failed,
    _route_clarification_response,
    _turn_replay_response,
    send_message,
)
from core.conversation_context import ConversationContext
from core.chat_turns import (
    TurnRequestConflict,
    assert_turn_request_matches,
    build_turn_request_context,
    commit_with_retry,
    normalize_request_id,
    question_digest,
    reclaim_stale_turn,
    request_context_fingerprint,
    reserve_turn,
    transition_turn,
    turn_duration_ms,
    turn_lease_expired,
)
from models.db_models import ChatTurn, Conversation, Message
from models.schemas import ChatRequest


class _CommitDB:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.commits = 0
        self.rollbacks = 0

    async def commit(self) -> None:
        self.commits += 1
        if self.commits <= self.failures:
            raise RuntimeError("transient commit failure")

    async def rollback(self) -> None:
        self.rollbacks += 1


class _RollbackAwareCommitDB:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.staged = True
        self.reapplications = 0

    async def commit(self) -> None:
        self.commits += 1
        if self.commits == 1:
            raise RuntimeError("ambiguous transient failure")
        if not self.staged:
            raise RuntimeError("empty commit after rollback")

    async def rollback(self) -> None:
        self.rollbacks += 1
        self.staged = False

    async def reapply(self, _session) -> None:
        self.reapplications += 1
        self.staged = True


class _TurnSaveDB:
    def __init__(self, turn) -> None:
        self.turn = turn
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def get(self, model, identity):
        if model is ChatTurn and identity == self.turn.id:
            return self.turn
        return None

    async def commit(self):
        self.commits += 1


class _RetryingImmediateDB:
    """Fake the ORM state loss that follows a real SQLAlchemy rollback."""

    def __init__(self, turn: ChatTurn, conversation: Conversation) -> None:
        self.turn = turn
        self.conversation = conversation
        self.messages: dict[uuid.UUID, Message] = {}
        self.commits = 0
        self.rollbacks = 0

    async def get(self, model, identity, **_kwargs):
        if model is ChatTurn and identity == self.turn.id:
            return self.turn
        if model is Conversation and identity == self.conversation.id:
            return self.conversation
        if model is Message:
            return self.messages.get(identity)
        return None

    def add(self, value):
        if isinstance(value, Message):
            self.messages[value.id] = value

    def add_all(self, values):
        for value in values:
            self.add(value)

    async def commit(self):
        self.commits += 1
        if self.commits == 1:
            raise RuntimeError("first commit failed")

    async def rollback(self):
        self.rollbacks += 1
        self.messages.clear()
        self.conversation.pending_route_state = None
        self.conversation.route_state_revision = 0
        self.turn.status = "accepted"
        self.turn.trace_id = "initial-trace"
        self.turn.evidence_status = None
        self.turn.retrieval_executed = None
        self.turn.error_code = None
        self.turn.answer_content = None
        self.turn.answer_sources = None
        self.turn.search_snapshot = None
        self.turn.tokens = None
        self.turn.user_message_id = None
        self.turn.assistant_message_id = None
        self.turn.generated_at = None
        self.turn.completed_at = None


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _ReservationDB:
    def __init__(self, existing=None) -> None:
        self.existing = existing
        self.added = []
        self.flushes = 0

    async def execute(self, _statement):
        return _Result(self.existing)

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        self.flushes += 1


class _ExistingRequestDB:
    def __init__(self, turn, conversation) -> None:
        self.turn = turn
        self.conversation = conversation
        self.added = []

    async def execute(self, _statement):
        return _Result(self.turn)

    async def get(self, _model, identity):
        if identity == self.conversation.id:
            return self.conversation
        return None

    async def flush(self):
        return None

    def add(self, value):
        self.added.append(value)


class _DurableExistingDB(_ExistingRequestDB):
    def __init__(self, turn, conversation) -> None:
        super().__init__(turn, conversation)
        self.commits = 0
        self.refreshed = []

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None

    async def refresh(self, value, attribute_names=None):
        self.refreshed.append((value, attribute_names))

    def add_all(self, values):
        self.added.extend(values)


class _FailingReservationCommitDB:
    def __init__(self, conversation) -> None:
        self.conversation = conversation
        self.added = []
        self.flushes = 0
        self.rollbacks = 0

    async def execute(self, _statement):
        return _Result(None)

    async def get(self, model, identity):
        if model is Conversation and identity == self.conversation.id:
            return self.conversation
        return None

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        self.flushes += 1

    async def commit(self):
        raise RuntimeError("reservation commit outcome is uncertain")

    async def rollback(self):
        self.rollbacks += 1


class ChatTurnProtocolTests(unittest.IsolatedAsyncioTestCase):
    def test_old_client_can_omit_request_and_turn_ids(self) -> None:
        payload = ChatRequest(question="测试")
        self.assertIsNone(payload.request_id)
        self.assertIsNone(payload.turn_id)
        self.assertEqual(len(normalize_request_id(None)), 32)

    def test_request_id_accepts_only_header_safe_opaque_tokens(self) -> None:
        for value in (
            "550e8400-e29b-41d4-a716-446655440000",
            "0123456789abcdef",
            "client.request_id:retry-2",
            "A" * 128,
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_request_id(value), value)

        for value in (
            "-leading-dash",
            "contains space",
            "request\r\nX-Injected: true",
            "请求标识",
            "A" * 129,
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_request_id(value)

    def test_state_machine_covers_generated_recovery(self) -> None:
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            request_id="req-1",
            question_hash=question_digest("问题"),
            status="accepted",
        )
        transition_turn(turn, "generating")
        transition_turn(turn, "generated", answer_content="答案")
        transition_turn(turn, "persist_failed", error_code="commit_failed")
        transition_turn(turn, "generated")
        transition_turn(turn, "completed", assistant_message_id=uuid.uuid4())
        self.assertEqual(turn.status, "completed")
        self.assertEqual(turn.answer_content, "答案")

    def test_completed_turn_duration_uses_server_timestamps(self) -> None:
        started_at = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)
        turn = SimpleNamespace(
            created_at=started_at,
            completed_at=started_at + timedelta(seconds=12, milliseconds=345),
        )
        self.assertEqual(turn_duration_ms(turn), 12_345)

    async def test_bare_commit_is_not_retried_after_rollback(self) -> None:
        db = _CommitDB(failures=2)
        with self.assertRaises(RuntimeError):
            await commit_with_retry(db, attempts=3)
        self.assertEqual(db.commits, 1)
        self.assertEqual(db.rollbacks, 1)

    async def test_commit_retry_reapplies_transaction_after_rollback(self) -> None:
        db = _RollbackAwareCommitDB()
        await commit_with_retry(db, attempts=3, reapply=db.reapply)
        self.assertEqual(db.commits, 2)
        self.assertEqual(db.rollbacks, 1)
        self.assertEqual(db.reapplications, 1)
        self.assertTrue(db.staged)

    async def test_clarification_retry_reloads_and_replays_orm_transaction(self) -> None:
        user_id = uuid.uuid4()
        conversation = Conversation(
            id=uuid.uuid4(),
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        context = build_turn_request_context(
            question="那这个指什么",
            conversation_id=conversation.id,
            knowledge_base_ids=[],
            search_config={},
            pending_route_revision=0,
            pending_state_id=None,
        )
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            user_id=user_id,
            request_id="clarification-retry",
            question_hash=question_digest("那这个指什么"),
            request_fingerprint=request_context_fingerprint(context),
            request_context=context,
            trace_id="initial-trace",
            status="accepted",
            execution_attempts=1,
        )
        db = _RetryingImmediateDB(turn, conversation)
        with patch("api.chat.trace_event"):
            response = await _route_clarification_response(
                db=db,
                conv=conversation,
                user=SimpleNamespace(id=user_id, is_superadmin=False),
                question="那这个指什么",
                clarification_message="请说明你指的是哪一项配置。",
                decision_reason="unresolved_reference",
                trace_id="retry-trace",
                selected_kb_ids=[],
                task_contract=None,
                turn=turn,
            )
            body = "".join(
                [
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                    async for chunk in response.body_iterator
                ]
            )

        self.assertEqual(db.commits, 2)
        self.assertEqual(db.rollbacks, 1)
        self.assertEqual(turn.status, "completed")
        self.assertEqual(conversation.route_state_revision, 1)
        self.assertEqual(
            sorted(message.role for message in db.messages.values()),
            ["assistant", "user"],
        )
        self.assertIn("请说明你指的是哪一项配置", body)

    def test_request_fingerprint_binds_scope_config_and_pending_identity(self) -> None:
        conversation_id = uuid.uuid4()
        first_kb = uuid.uuid4()
        second_kb = uuid.uuid4()
        base = build_turn_request_context(
            question=" 普通员工出差标准 ",
            conversation_id=conversation_id,
            knowledge_base_ids=[second_kb, first_kb, first_kb],
            search_config={
                "method": "hybrid",
                "rerank": True,
                "top_k": 5,
            },
            pending_route_revision=3,
            pending_state_id="state-3",
        )
        reordered = build_turn_request_context(
            question="普通员工出差标准",
            conversation_id=conversation_id,
            knowledge_base_ids=[first_kb, second_kb],
            search_config={
                "method": "hybrid",
                "rerank": True,
                "top_k": 5,
            },
            pending_route_revision=3,
            pending_state_id="state-3",
        )
        self.assertEqual(
            request_context_fingerprint(base),
            request_context_fingerprint(reordered),
        )
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            user_id=uuid.uuid4(),
            request_id="fingerprint-request",
            question_hash=question_digest("普通员工出差标准"),
            request_fingerprint=request_context_fingerprint(base),
            request_context=base,
            status="completed",
        )
        assert_turn_request_matches(turn, reordered)
        for drift in (
            {**reordered, "knowledge_base_ids": [str(first_kb)]},
            {
                **reordered,
                "search_config": {**reordered["search_config"], "top_k": 10},
            },
            {
                **reordered,
                "pending_route": {"revision": 4, "state_id": "state-4"},
            },
        ):
            with self.assertRaises(TurnRequestConflict):
                assert_turn_request_matches(turn, drift)

    def test_expired_execution_lease_can_be_reclaimed_once(self) -> None:
        now = datetime.now(timezone.utc)
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            request_id="stale-request",
            question_hash=question_digest("问题"),
            request_fingerprint="a" * 64,
            request_context={},
            status="generating",
            lease_owner="dead-worker",
            lease_expires_at=now - timedelta(seconds=1),
            execution_attempts=1,
            updated_at=now - timedelta(minutes=10),
        )
        self.assertTrue(turn_lease_expired(turn, now=now))
        self.assertTrue(reclaim_stale_turn(turn, owner="retry", now=now))
        self.assertEqual(turn.status, "accepted")
        self.assertEqual(turn.lease_owner, "retry")
        self.assertEqual(turn.execution_attempts, 2)
        self.assertFalse(turn_lease_expired(turn, now=now))
        self.assertFalse(reclaim_stale_turn(turn, owner="second", now=now))

    async def test_unstaged_generated_payload_is_not_advertised_recoverable(self) -> None:
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            request_id="unstaged-answer",
            question_hash=question_digest("问题"),
            request_fingerprint="b" * 64,
            request_context={},
            status="generating",
            answer_content=None,
        )
        save_db = _TurnSaveDB(turn)
        with patch("database.AsyncSessionLocal", return_value=save_db):
            recoverable = await _mark_turn_persist_failed(
                turn=turn,
                trace_id="trace",
            )
        self.assertFalse(recoverable)
        self.assertEqual(turn.status, "failed")
        self.assertEqual(turn.error_code, "generated_payload_not_persisted")
        self.assertEqual(save_db.commits, 1)

    async def test_failed_duplicate_replay_emits_new_request_guidance(self) -> None:
        conversation_id = uuid.uuid4()
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            user_id=uuid.uuid4(),
            request_id="failed-replay",
            question_hash=question_digest("问题"),
            request_fingerprint="c" * 64,
            request_context={},
            status="failed",
            error_code="generated_payload_not_persisted",
            answer_content=None,
        )
        response = _turn_replay_response(
            conv=SimpleNamespace(id=conversation_id),
            turn=turn,
        )
        body = "".join(
            [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
        )

        self.assertIn('"type": "error"', body)
        self.assertIn('"persistence_status": "failed"', body)
        self.assertIn('"same_request_recoverable": false', body)
        self.assertIn('"retry_with_new_request_id": true', body)
        self.assertIn("新的 request_id", body)

    async def test_duplicate_user_request_reuses_original_turn(self) -> None:
        existing = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            request_id="stable-request",
            question_hash=question_digest("同一个问题"),
            status="completed",
        )
        db = _ReservationDB(existing=existing)

        turn, created = await reserve_turn(
            db,
            conversation_id=existing.conversation_id,
            user_id=existing.user_id,
            request_id=existing.request_id,
            turn_id=None,
            question="同一个问题",
            trace_id="trace",
        )

        self.assertFalse(created)
        self.assertIs(turn, existing)
        self.assertEqual(db.added, [])
        self.assertEqual(db.flushes, 0)

    async def test_same_request_id_cannot_change_question(self) -> None:
        existing = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            request_id="stable-request",
            question_hash=question_digest("原问题"),
            status="completed",
        )
        db = _ReservationDB(existing=existing)
        with self.assertRaises(TurnRequestConflict):
            await reserve_turn(
                db,
                conversation_id=existing.conversation_id,
                user_id=existing.user_id,
                request_id=existing.request_id,
                turn_id=None,
                question="另一个问题",
                trace_id="trace",
            )

    async def test_retry_without_conversation_id_replays_original_conversation(self) -> None:
        user_id = uuid.uuid4()
        conversation_id = uuid.uuid4()
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            user_id=user_id,
            request_id="new-chat-stable-request",
            question_hash=question_digest("同一个问题"),
            trace_id="trace-1",
            status="completed",
            answer_content="已保存答案",
            answer_sources=[],
        )
        conversation = SimpleNamespace(id=conversation_id, user_id=user_id)
        db = _ExistingRequestDB(turn, conversation)
        user = SimpleNamespace(id=user_id, is_superadmin=False)

        with patch(
            "api.chat.get_accessible_kb_ids",
            new=AsyncMock(return_value=None),
        ), patch("api.chat.classify_intent_result") as classify:
            response = await send_message(
                ChatRequest(
                    question="同一个问题",
                    request_id=turn.request_id,
                ),
                db=db,
                user=user,
            )
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]

        self.assertIn(str(conversation_id), "".join(chunks))
        self.assertIn("已保存答案", "".join(chunks))
        self.assertEqual(db.added, [])
        classify.assert_not_called()

    async def test_same_request_id_rejects_kb_or_search_parameter_drift(self) -> None:
        user_id = uuid.uuid4()
        conversation_id = uuid.uuid4()
        original_kb = uuid.uuid4()
        context = build_turn_request_context(
            question="普通员工住宿标准",
            conversation_id=conversation_id,
            knowledge_base_ids=[original_kb],
            search_config={
                "method": "hybrid",
                "rerank": True,
                "top_k": 5,
            },
            pending_route_revision=0,
            pending_state_id=None,
        )
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            user_id=user_id,
            request_id="drift-request",
            question_hash=question_digest("普通员工住宿标准"),
            request_fingerprint=request_context_fingerprint(context),
            request_context=context,
            status="completed",
            answer_content="原回答",
            answer_sources=[],
        )
        db = _ExistingRequestDB(
            turn,
            SimpleNamespace(id=conversation_id, user_id=user_id),
        )
        with patch(
            "api.chat.get_accessible_kb_ids",
            new=AsyncMock(return_value=None),
        ):
            for payload in (
                ChatRequest(
                    question="普通员工住宿标准",
                    request_id=turn.request_id,
                    knowledge_base_ids=[uuid.uuid4()],
                ),
                ChatRequest(
                    question="普通员工住宿标准",
                    request_id=turn.request_id,
                    knowledge_base_ids=[original_kb],
                    search_config={
                        "method": "keyword",
                        "rerank": False,
                        "top_k": 10,
                    },
                ),
            ):
                with self.subTest(payload=payload.model_dump()):
                    with self.assertRaises(HTTPException) as raised:
                        await send_message(
                            payload,
                            db=db,
                            user=SimpleNamespace(
                                id=user_id,
                                is_superadmin=False,
                            ),
                        )
                    self.assertEqual(raised.exception.status_code, 409)

    async def test_reservation_commit_503_requires_same_request_id(self) -> None:
        user_id = uuid.uuid4()
        conversation = Conversation(
            id=uuid.uuid4(),
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        db = _FailingReservationCommitDB(conversation)

        with patch(
            "api.chat.get_accessible_kb_ids",
            new=AsyncMock(return_value=None),
        ):
            with self.assertRaises(HTTPException) as raised:
                await send_message(
                    ChatRequest(
                        question="查询报销时限",
                        conversation_id=conversation.id,
                        request_id="uncertain-reservation-request",
                    ),
                    db=db,
                    user=SimpleNamespace(id=user_id, is_superadmin=False),
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(
            raised.exception.detail,
            {
                "message": "请求已接收但暂时无法保存，请使用相同 request_id 重试",
                "error_code": "turn_reservation_persistence_uncertain",
                "same_request_recoverable": True,
                "retry_with_new_request_id": False,
            },
        )
        self.assertEqual(db.rollbacks, 1)

    async def test_stale_generating_turn_is_reclaimed_instead_of_permanent_202(self) -> None:
        user_id = uuid.uuid4()
        conversation_id = uuid.uuid4()
        context = build_turn_request_context(
            question="这些配置有什么影响",
            conversation_id=conversation_id,
            knowledge_base_ids=[],
            search_config={
                "method": "hybrid",
                "rerank": True,
                "top_k": 5,
            },
            pending_route_revision=0,
            pending_state_id=None,
        )
        now = datetime.now(timezone.utc)
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            user_id=user_id,
            request_id="reclaim-through-api",
            question_hash=question_digest("这些配置有什么影响"),
            request_fingerprint=request_context_fingerprint(context),
            request_context=context,
            status="generating",
            trace_id="dead-trace",
            lease_owner="dead-worker",
            lease_expires_at=now - timedelta(seconds=1),
            execution_attempts=1,
            updated_at=now - timedelta(minutes=20),
        )
        conversation = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        db = _DurableExistingDB(turn, conversation)
        unresolved = ConversationContext(
            is_followup=False,
            followup_reason="unresolved_reference:这些",
            standalone_query="这些配置有什么影响",
            history_messages=(),
            carryover_sources=(),
            unresolved_reference=True,
        )
        with (
            patch(
                "api.chat.get_accessible_kb_ids",
                new=AsyncMock(return_value=None),
            ),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=unresolved),
            ),
            patch("api.chat.classify_intent_result", new=AsyncMock()) as classify,
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question="这些配置有什么影响",
                    request_id=turn.request_id,
                ),
                db=db,
                user=SimpleNamespace(id=user_id, is_superadmin=False),
            )
            body = "".join(
                [
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                    async for chunk in response.body_iterator
                ]
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(turn.status, "completed")
        self.assertEqual(turn.execution_attempts, 2)
        self.assertIn("无法确定", body)
        classify.assert_not_awaited()
        self.assertGreaterEqual(db.commits, 2)

    async def test_duplicate_replay_drops_sources_after_kb_access_is_revoked(self) -> None:
        user_id = uuid.uuid4()
        conversation_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        source = {
            "kb_id": str(kb_id),
            "doc_id": str(doc_id),
            "id": str(chunk_id),
            "content": "撤权后不得重放的正文",
            "filename": "私密资料.md",
            "evidence_role": "direct",
        }
        turn = ChatTurn(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            user_id=user_id,
            request_id="revoked-replay",
            question_hash=question_digest("原问题"),
            trace_id="trace-2",
            status="completed",
            evidence_status="hit",
            retrieval_executed=True,
            answer_content="回答正文仍可保留",
            answer_sources=[source],
            search_snapshot={
                "schema_version": "rag_search_snapshot.v1",
                "candidates": [source],
                "answer_sources": [source],
                "counters": {"evidence_status": "hit"},
            },
        )
        conversation = SimpleNamespace(id=conversation_id, user_id=user_id)
        db = _ExistingRequestDB(turn, conversation)
        with patch(
            "api.chat.get_accessible_kb_ids",
            new=AsyncMock(return_value=[]),
        ):
            response = await send_message(
                ChatRequest(question="原问题", request_id=turn.request_id),
                db=db,
                user=SimpleNamespace(id=user_id, is_superadmin=False),
            )
            body = "".join([
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ])

        self.assertIn("回答正文仍可保留", body)
        self.assertNotIn("撤权后不得重放", body)
        self.assertNotIn("私密资料.md", body)


if __name__ == "__main__":
    unittest.main()
