import unittest
import uuid
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock, patch

from api.chat import (
    _active_pending_route_state,
    _evidence_event_pending_state,
    _evidence_scope_filter,
    _parse_sse_payload,
    _parse_evidence_scope_reply,
    _route_clarification_response,
    _refined_evidence_query,
    _scope_anchor_coverage_from_sources,
    _validate_stream_answer_sources,
    _scoped_evidence_query,
    send_message,
)
from core.conversation_context import ConversationContext
from core.query_route_compiler import (
    RouteCategoryPolicy,
    RouteCompilerConfig,
    compile_rag_task_contract,
)
from core.query_route_contract import parse_rag_route_decision
from models.db_models import Document, DocumentChunk, IntentRouteLog
from models.schemas import ChatRequest, IntentEvidenceStatus


class _RouteStateDB:
    def __init__(self, conversation):
        self.conversation = conversation
        self.added = []
        self.commits = 0

    async def get(self, _model, _identity):
        return self.conversation

    def add(self, value):
        self.added.append(value)

    def add_all(self, values):
        self.added.extend(values)

    async def commit(self):
        self.commits += 1


class _RouteStateSourceDB(_RouteStateDB):
    def __init__(self, conversation, *, source_rows):
        super().__init__(conversation)
        self.source_rows = source_rows

    async def execute(self, _statement):
        return _RowsResult(self.source_rows)


class _SaveStateDB:
    def __init__(self, conversation, *, route_logs=None):
        self.conversation = conversation
        self.route_logs = route_logs or {}
        self.added = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def get(self, model, _identity):
        from models.db_models import Conversation, IntentRouteLog

        if model is Conversation:
            return self.conversation
        if model is IntentRouteLog:
            return self.route_logs.get(_identity)
        return None

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        # Exercise the PostgreSQL VARCHAR boundary that a plain async fake
        # would otherwise skip.  This caught the production-only failure where
        # ``needs_clarification`` did not fit the route-log column.
        if self.route_logs:
            from models.db_models import IntentRouteLog

            column = IntentRouteLog.__table__.c.evidence_status
            max_length = column.type.length
            for route_log in self.route_logs.values():
                value = getattr(route_log, "evidence_status", None)
                if value is not None and len(value) > max_length:
                    raise ValueError(
                        f"value too long for evidence_status VARCHAR({max_length})"
                    )
        self.commits += 1


class _FailingSaveStateDB(_SaveStateDB):
    def __init__(self, conversation):
        super().__init__(conversation)
        self._pending_snapshot = json.loads(
            json.dumps(conversation.pending_route_state, ensure_ascii=False)
        )
        self._revision_snapshot = conversation.route_state_revision

    async def __aexit__(self, exc_type, _exc, _traceback):
        if exc_type is not None:
            # Mirror the rollback performed when a real AsyncSession closes an
            # uncommitted transaction.  The request-session object is separate
            # below, so a failed response save can never appear to resolve the
            # user's outstanding evidence choice.
            self.conversation.pending_route_state = self._pending_snapshot
            self.conversation.route_state_revision = self._revision_snapshot
        return False

    async def commit(self):
        raise RuntimeError("simulated response persistence failure")


def _evidence_pending_state(
    *,
    kb_id: uuid.UUID | None = None,
    first_doc_id: uuid.UUID | None = None,
    second_doc_id: uuid.UUID | None = None,
):
    kb_id = kb_id or uuid.uuid4()
    first_doc_id = first_doc_id or uuid.uuid4()
    second_doc_id = second_doc_id or uuid.uuid4()
    now = datetime.now(timezone.utc)
    return {
        "schema_version": "rag_pending_clarification.v2",
        "kind": "evidence_scope",
        "state_id": str(uuid.uuid4()),
        "base_user_message_id": str(uuid.uuid4()),
        "clarification_message_id": str(uuid.uuid4()),
        "original_query": "解决登录用户名枚举要配置什么",
        "dimension": "version",
        "selection_mode": "choice",
        "choices": [
            {
                "key": "c1",
                "label": "云枢 6.0.1 —《钉钉》",
                "products": ["云枢"],
                "canonical_products": ["云枢"],
                "versions": ["6.0.1"],
                "projects": [],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(first_doc_id)],
                "anchor_doc_ids": [str(first_doc_id)],
                "companion_doc_ids": [],
                "filenames": ["钉钉.md"],
            },
            {
                "key": "c2",
                "label": "云枢 8.2.75（中青建安）—《二开发送钉钉工作通知》",
                "products": ["云枢"],
                "canonical_products": ["云枢"],
                "versions": ["8.2.75"],
                "projects": ["中青建安"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(second_doc_id)],
                "anchor_doc_ids": [str(second_doc_id)],
                "companion_doc_ids": [],
                "filenames": ["二开发送钉钉工作通知.md"],
            },
        ],
        "clarification_message": (
            "检索到与当前问题相关、但适用范围不同的资料：\n"
            "1. 云枢 6.0.1 —《钉钉》\n"
            "2. 云枢 8.2.75（中青建安）—《二开发送钉钉工作通知》\n"
            "请问需要查询哪一项？也可以回复“都对比”。"
        ),
        "selected_kb_ids_snapshot": [str(kb_id)],
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
        "dispatch_authorized": False,
    }


def _broad_evidence_pending_state(*, kb_id: uuid.UUID | None = None):
    state = _evidence_pending_state(kb_id=kb_id)
    state["selection_mode"] = "refine"
    state["choices"] = []
    state["clarification_message"] = (
        "检索到多个互不相同的适用范围，请补充具体产品和版本。"
    )
    return state


def _route_and_contract(
    *,
    intent_code: str,
    action: str,
    evidence_scope: str,
    selected_kb_count: int,
    relation: str = "new",
):
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
            "rationale": "route-state regression",
        },
        allowed_intent_codes=[intent_code],
    )
    contract = compile_rag_task_contract(
        route,
        RouteCategoryPolicy(code=intent_code, name=intent_code, action=action),
        RouteCompilerConfig(),
        question="用户当前输入",
        selected_kb_count=selected_kb_count,
        source="llm",
    )
    decision_payload = {
        "intent_code": intent_code,
        "intent_name": intent_code,
        "action": action,
        "confidence": route.confidence,
        "source": "llm",
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
    return route, contract, SimpleNamespace(
        decision=decision,
        route_decision=route,
        task_contract=contract,
        diagnostics={},
        route_log_id=None,
    )


def _context(*, pending_route_state=None):
    return ConversationContext(
        is_followup=False,
        followup_reason="standalone_question",
        standalone_query="用户当前输入",
        history_messages=(),
        carryover_sources=(),
        pending_route_state=pending_route_state,
    )


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _SourceValidationDB:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, _statement):
        return _RowsResult(self.rows)


class EvidenceSourceValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_refreshes_only_active_ready_current_chunk(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        document = Document(
            id=doc_id,
            kb_id=kb_id,
            filename="当前制度.md",
            status="ready",
            is_active=True,
            file_type="md",
        )
        chunk = DocumentChunk(
            id=chunk_id,
            doc_id=doc_id,
            kb_id=kb_id,
            content="数据库中的真实内容",
            chunk_index=3,
            metadata_={"section": "交通"},
        )
        source = {
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
            "content": "producer 伪造内容",
            "evidence_role": "direct",
        }
        refreshed, pairs, error = await _validate_stream_answer_sources(
            _SourceValidationDB([(chunk, document)]),
            raw_sources=[source],
            raw_results=[source],
            selected_kb_ids=[kb_id],
        )
        self.assertIsNone(error)
        self.assertEqual(pairs, {(kb_id, doc_id)})
        self.assertEqual(refreshed[0]["content"], "数据库中的真实内容")
        self.assertEqual(refreshed[0]["filename"], "当前制度.md")

    async def test_refresh_failure_clears_sources_fail_closed(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        source = {
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
            "content": "不应被保存",
            "evidence_role": "direct",
        }
        refreshed, pairs, error = await _validate_stream_answer_sources(
            object(),
            raw_sources=[source],
            raw_results=[source],
            selected_kb_ids=[kb_id],
        )
        self.assertEqual(refreshed, [])
        self.assertEqual(pairs, set())
        self.assertTrue(str(error).startswith("source_refresh_failed:"))

    async def test_inactive_document_is_not_accepted_from_adapter_rows(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        document = Document(
            id=doc_id,
            kb_id=kb_id,
            filename="已停用.md",
            status="ready",
            is_active=False,
        )
        chunk = DocumentChunk(
            id=chunk_id,
            doc_id=doc_id,
            kb_id=kb_id,
            content="不应进入上下文",
        )
        source = {
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "kb_id": str(kb_id),
            "content": "旧快照",
        }
        refreshed, pairs, error = await _validate_stream_answer_sources(
            _SourceValidationDB([(chunk, document)]),
            raw_sources=[source],
            raw_results=[source],
            selected_kb_ids=[kb_id],
        )
        self.assertEqual(refreshed, [])
        self.assertEqual(pairs, set())
        self.assertEqual(error, "answer_source_not_current")

    def test_scope_anchor_uses_actual_answer_pairs_not_producer_boolean(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        scope_filter = {
            "choices": [
                {
                    "kb_ids": [str(kb_id)],
                    "anchor_doc_ids": [str(doc_id)],
                }
            ]
        }
        self.assertEqual(
            _scope_anchor_coverage_from_sources(scope_filter, set()),
            (False, []),
        )
        self.assertEqual(
            _scope_anchor_coverage_from_sources(
                scope_filter,
                {(kb_id, doc_id)},
            ),
            (True, [str(doc_id)]),
        )


class PendingRouteStateTests(unittest.IsolatedAsyncioTestCase):
    def test_all_public_evidence_statuses_fit_route_log_column(self) -> None:
        statuses = get_args(IntentEvidenceStatus)
        self.assertIn("needs_clarification", statuses)
        self.assertLessEqual(
            max(len(status) for status in statuses),
            IntentRouteLog.__table__.c.evidence_status.type.length,
        )

    def test_active_pending_state_rejects_expired_or_malformed_expiry(self) -> None:
        now = datetime.now(timezone.utc)
        active = {
            "schema_version": "rag_pending_clarification.v1",
            "state_id": "active",
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "dispatch_authorized": False,
        }
        expired = {
            **active,
            "state_id": "expired",
            "expires_at": (now - timedelta(seconds=1)).isoformat(),
        }
        malformed = {**active, "state_id": "malformed", "expires_at": "not-a-date"}
        missing_expiry = {key: value for key, value in active.items() if key != "expires_at"}
        unknown_version = {**active, "schema_version": "rag_pending_clarification.v3"}
        authorized = {**active, "dispatch_authorized": True}
        missing_authorization = {
            key: value for key, value in active.items() if key != "dispatch_authorized"
        }

        self.assertIs(_active_pending_route_state(active), active)
        for value in (
            expired,
            malformed,
            missing_expiry,
            unknown_version,
            authorized,
            missing_authorization,
        ):
            with self.subTest(value=value):
                self.assertIsNone(_active_pending_route_state(value))
        self.assertIsNone(_active_pending_route_state("not-an-object"))

    def test_active_v2_state_is_strictly_validated_and_non_executable(self) -> None:
        active = _evidence_pending_state()

        self.assertEqual(_active_pending_route_state(active), active)
        with_untrusted_extras = {
            **active,
            "unexpected": "do not persist",
            "choices": [
                {**active["choices"][0], "raw_model_payload": "private"},
                active["choices"][1],
            ],
        }
        normalized = _active_pending_route_state(with_untrusted_extras)
        self.assertNotIn("unexpected", normalized)
        self.assertNotIn("raw_model_payload", normalized["choices"][0])

        malformed_values = (
            {**active, "kind": "intent_slot"},
            {**active, "dispatch_authorized": True},
            {**active, "base_user_message_id": "not-a-uuid"},
            {**active, "clarification_message_id": "not-a-uuid"},
            {**active, "created_at": "not-a-date"},
            {**active, "original_query": ""},
            {**active, "dimension": "unknown"},
            {**active, "selected_kb_ids_snapshot": ["not-a-uuid"]},
            {**active, "choices": active["choices"][:1]},
            {
                **active,
                "choices": [
                    {**active["choices"][0], "doc_ids": ["not-a-uuid"]},
                    active["choices"][1],
                ],
            },
            {
                **active,
                "choices": [
                    {
                        **active["choices"][0],
                        "kb_ids": [str(uuid.uuid4())],
                    },
                    active["choices"][1],
                ],
            },
        )
        for malformed in malformed_values:
            with self.subTest(malformed=malformed):
                self.assertIsNone(_active_pending_route_state(malformed))

    def test_evidence_choice_text_up_to_500_chars_can_be_persisted(self) -> None:
        kb_id = uuid.uuid4()
        template = _evidence_pending_state(kb_id=kb_id)
        choices = json.loads(json.dumps(template["choices"], ensure_ascii=False))
        choices[0]["label"] = "范" * 500
        choices[0]["products"] = ["产" * 500]
        event = {
            "schema_version": "rag_evidence_clarification.v1",
            "needs_clarification": True,
            "dimension": "version",
            "question": template["clarification_message"],
            "choices": choices,
        }

        state = _evidence_event_pending_state(
            event,
            original_query=template["original_query"],
            selected_kb_ids=[kb_id],
            base_user_message_id=uuid.uuid4(),
            clarification_message_id=uuid.uuid4(),
        )

        self.assertIsNotNone(state)
        self.assertEqual(len(state["choices"][0]["label"]), 500)
        self.assertEqual(len(state["choices"][0]["products"][0]), 500)

        for field, value in (
            ("label", "范" * 501),
            ("products", ["产" * 501]),
        ):
            with self.subTest(field=field):
                malformed_choices = json.loads(
                    json.dumps(choices, ensure_ascii=False)
                )
                malformed_choices[0][field] = value
                malformed = _evidence_event_pending_state(
                    {**event, "choices": malformed_choices},
                    original_query=template["original_query"],
                    selected_kb_ids=[kb_id],
                    base_user_message_id=uuid.uuid4(),
                    clarification_message_id=uuid.uuid4(),
                )
                self.assertIsNone(malformed)

    def test_legacy_choices_derive_anchor_and_companion_documents(self) -> None:
        kb_id = uuid.uuid4()
        first_anchor = uuid.uuid4()
        second_anchor = uuid.uuid4()
        shared_companion = uuid.uuid4()
        state = _evidence_pending_state(
            kb_id=kb_id,
            first_doc_id=first_anchor,
            second_doc_id=second_anchor,
        )
        state.pop("selection_mode")
        for choice, anchor in zip(
            state["choices"],
            (first_anchor, second_anchor),
            strict=True,
        ):
            choice["doc_ids"] = [str(anchor), str(shared_companion)]
            choice.pop("anchor_doc_ids")
            choice.pop("companion_doc_ids")

        normalized = _active_pending_route_state(state)

        self.assertEqual(normalized["selection_mode"], "choice")
        self.assertEqual(
            normalized["choices"][0]["anchor_doc_ids"],
            [str(first_anchor)],
        )
        self.assertEqual(
            normalized["choices"][1]["anchor_doc_ids"],
            [str(second_anchor)],
        )
        self.assertEqual(
            normalized["choices"][0]["companion_doc_ids"],
            [str(shared_companion)],
        )
        self.assertEqual(
            normalized["choices"][1]["companion_doc_ids"],
            [str(shared_companion)],
        )

        malformed = json.loads(json.dumps(normalized, ensure_ascii=False))
        malformed["choices"][0]["anchor_doc_ids"] = [str(shared_companion)]
        malformed["choices"][0]["companion_doc_ids"] = [str(first_anchor)]
        self.assertIsNone(_active_pending_route_state(malformed))

    def test_evidence_scope_reply_parser_supports_generic_choice_forms(self) -> None:
        pending = _evidence_pending_state()
        cases = {
            "1": ("single", "c1"),
            "c2": ("single", "c2"),
            "2吧": ("single", "c2"),
            "c2。": ("single", "c2"),
            "第二个吧": ("single", "c2"),
            "8.2.75": ("single", "c2"),
            "选 8.2.75 版本吧": ("single", "c2"),
            pending["choices"][0]["label"]: ("single", "c1"),
            "都要": ("compare_all", "c1"),
            "都对比": ("compare_all", "c1"),
        }
        for reply, (action, first_key) in cases.items():
            with self.subTest(reply=reply):
                parsed = _parse_evidence_scope_reply(reply, pending)
                self.assertEqual(parsed.action, action)
                self.assertEqual(parsed.choices[0]["key"], first_key)

        self.assertEqual(
            _parse_evidence_scope_reply("随便一个", pending).action,
            "repeat",
        )
        self.assertEqual(
            _parse_evidence_scope_reply("哪个好", pending).action,
            "repeat",
        )
        self.assertEqual(
            _parse_evidence_scope_reply("哪个都行", pending).action,
            "repeat",
        )
        self.assertEqual(
            _parse_evidence_scope_reply("普通员工的出差标准", pending).action,
            "new_question",
        )
        self.assertEqual(
            _parse_evidence_scope_reply("今天天气", pending).action,
            "new_question",
        )
        self.assertEqual(
            _parse_evidence_scope_reply(
                "8.2.75版本的密码策略怎么配置",
                pending,
            ).action,
            "new_question",
        )
        self.assertEqual(
            _parse_evidence_scope_reply("取消", pending).action,
            "cancel",
        )

        broad_pending = {
            **pending,
            "selection_mode": "refine",
            "choices": [],
        }
        self.assertEqual(
            _parse_evidence_scope_reply("2025版", broad_pending).action,
            "refine",
        )
        self.assertEqual(
            _parse_evidence_scope_reply("随便一个", broad_pending).action,
            "repeat",
        )
        self.assertEqual(
            _parse_evidence_scope_reply(
                "普通员工的出差标准",
                broad_pending,
            ).action,
            "new_question",
        )
        self.assertEqual(
            _parse_evidence_scope_reply("取消", broad_pending).action,
            "cancel",
        )
        self.assertEqual(
            _parse_evidence_scope_reply(
                "8.2.75版本的密码策略怎么配置",
                broad_pending,
            ).action,
            "new_question",
        )
        self.assertEqual(
            _parse_evidence_scope_reply(
                "我使用的是云枢 8.2.75 版本，属于中青建安项目，请按这个具体范围继续查询",
                broad_pending,
            ).action,
            "refine",
        )

    def test_explicit_subset_comparison_does_not_include_unmentioned_choice(self) -> None:
        pending = _evidence_pending_state()
        third_doc_id = uuid.uuid4()
        pending["choices"].insert(1, {
            **pending["choices"][0],
            "key": "c3",
            "label": "云枢 7.0 —《中间版本配置》",
            "versions": ["7.0"],
            "doc_ids": [str(third_doc_id)],
            "anchor_doc_ids": [str(third_doc_id)],
            "filenames": ["中间版本配置.md"],
        })

        reply = _parse_evidence_scope_reply(
            "对比 6.0.1 和 8.2.75 的差异",
            pending,
        )

        self.assertEqual(reply.action, "compare_all")
        self.assertEqual(
            [choice["key"] for choice in reply.choices],
            ["c1", "c2"],
        )

    def test_evidence_filter_is_current_scope_bounded(self) -> None:
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        reply = _parse_evidence_scope_reply("2", pending)

        scope_filter = _evidence_scope_filter(
            reply,
            current_kb_ids=[kb_id],
        )

        self.assertEqual(scope_filter["mode"], "single")
        self.assertEqual(scope_filter["choices"][0]["key"], "c2")
        self.assertEqual(scope_filter["kb_ids"], [str(kb_id)])
        self.assertEqual(
            scope_filter["doc_ids"],
            pending["choices"][1]["doc_ids"],
        )
        self.assertEqual(
            scope_filter["choices"][0]["anchor_doc_ids"],
            pending["choices"][1]["anchor_doc_ids"],
        )
        self.assertEqual(
            scope_filter["choices"][0]["companion_doc_ids"],
            pending["choices"][1]["companion_doc_ids"],
        )
        query = _scoped_evidence_query(pending["original_query"], scope_filter)
        self.assertIn(pending["original_query"], query)
        self.assertIn("8.2.75", query)
        self.assertIsNone(
            _evidence_scope_filter(reply, current_kb_ids=[uuid.uuid4()])
        )

    def test_broad_clarification_creates_strict_refinement_pending_state(self) -> None:
        kb_id = uuid.uuid4()
        state = _evidence_event_pending_state(
            {
                "type": "evidence_clarification",
                "schema_version": "rag_evidence_clarification.v1",
                "needs_clarification": True,
                "dimension": "version",
                "question": "检索到过多适用范围，请补充具体产品和版本。",
                "choices": [],
            },
            original_query="如何配置登录安全",
            selected_kb_ids=[kb_id],
            base_user_message_id=uuid.uuid4(),
            clarification_message_id=uuid.uuid4(),
        )

        self.assertIsNotNone(state)
        self.assertEqual(state["selection_mode"], "refine")
        self.assertEqual(state["choices"], [])
        self.assertFalse(state["dispatch_authorized"])
        self.assertEqual(state["selected_kb_ids_snapshot"], [str(kb_id)])
        self.assertEqual(_active_pending_route_state(state), state)

        for malformed in (
            {**state, "selection_mode": "unknown"},
            {**state, "selection_mode": "choice"},
            {
                **state,
                "choices": _evidence_pending_state(kb_id=kb_id)["choices"],
            },
        ):
            with self.subTest(malformed=malformed):
                self.assertIsNone(_active_pending_route_state(malformed))

    async def test_clarification_creation_persists_non_executable_versioned_state(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)
        _route, contract, _result = _route_and_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            selected_kb_count=0,
        )
        self.assertFalse(contract.dispatch_authorized)

        with patch("api.chat.trace_event") as trace:
            response = await _route_clarification_response(
                db=db,
                conv=conv,
                user=user,
                question="普通员工的出差标准",
                clarification_message=contract.clarification.question,
                decision_reason=contract.decision_reason,
                trace_id="trace-create",
                selected_kb_ids=[],
                task_contract=contract,
            )
            payloads = [
                _parse_sse_payload(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
                async for chunk in response.body_iterator
            ]
        payloads = [payload for payload in payloads if payload]

        state = conv.pending_route_state
        self.assertEqual(state["schema_version"], "rag_pending_clarification.v1")
        self.assertFalse(state["dispatch_authorized"])
        self.assertEqual(state["intent_code"], "knowledge_qa")
        self.assertEqual(state["unresolved"][0]["role"], "knowledge_base")
        self.assertEqual(state["selected_kb_ids_snapshot"], [])
        self.assertIsNotNone(_active_pending_route_state(state))
        self.assertEqual(conv.route_state_revision, 1)
        self.assertEqual(db.commits, 1)
        self.assertEqual([item.role for item in db.added], ["user", "assistant"])
        self.assertIn(
            "intent.clarification_created",
            [call.args[0] for call in trace.call_args_list],
        )
        event_types = [payload["type"] for payload in payloads]
        self.assertEqual(event_types[0], "conversation_started")
        self.assertLess(event_types.index("intent"), event_types.index("search_results"))
        intent_payload = next(
            payload["decision"] for payload in payloads if payload["type"] == "intent"
        )
        self.assertNotIn("route_decision", intent_payload)
        self.assertEqual(intent_payload["readiness"], "needs_clarification")
        self.assertFalse(intent_payload["dispatch_authorized"])
        self.assertEqual(
            intent_payload["task_contract"]["schema_version"],
            "rag_task_contract.v1",
        )
        self.assertFalse(
            intent_payload["task_contract"]["dispatch_authorized"]
        )

    async def test_route_clarification_with_missing_context_candidate_stops_rag(self) -> None:
        """A valid missing-slot clarification must not fall back into retrieval."""

        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        context = _context()
        route = parse_rag_route_decision(
            {
                "schema_version": "rag_route_decision.v1",
                "readiness": "needs_clarification",
                "intent_code": "knowledge_qa",
                "relation": "continuation",
                "evidence_scope": "enterprise_kb",
                "query_resolution": {
                    "mode": "contextualize",
                    "context_turn_keys": ["t2", "t3"],
                },
                "requirements": [
                    {
                        "role": "answer",
                        "origin": "user_text",
                        "description": "确认用户对应职级的餐补金额",
                    }
                ],
                "clarification": {
                    "question": (
                        "请确认你的职级；如果你是普通员工（D级），"
                        "按现有标准餐补为100元/天。"
                    ),
                    "unresolved": [
                        {
                            "role": "user_grade",
                            "reason": "missing",
                            "candidate_keys": ["t2"],
                        }
                    ],
                },
                "confidence": 0.96,
                "rationale": "历史只说明普通员工为D级，未确认是否为用户本人。",
            },
            allowed_intent_codes=["knowledge_qa"],
            available_turn_keys=["t2", "t3"],
        )
        contract = compile_rag_task_contract(
            route,
            RouteCategoryPolicy(
                code="knowledge_qa",
                name="知识库问答",
                action="retrieve",
            ),
            RouteCompilerConfig(),
            question="那我的补餐补是多少",
            selected_kb_count=1,
            source="llm",
            available_turn_keys=["t2", "t3"],
        )
        self.assertFalse(contract.dispatch_authorized)
        routing_result = SimpleNamespace(
            decision=SimpleNamespace(
                need_retrieval=True,
                decision_reason="semantic_clarification",
                to_dict=lambda: {
                    "intent_code": "knowledge_qa",
                    "need_retrieval": True,
                    "decision_reason": "semantic_clarification",
                },
            ),
            route_decision=route,
            task_contract=contract,
            diagnostics={},
            route_log_id=None,
        )
        db = _RouteStateDB(conv)
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
            patch("api.chat.run_rag_stream", new=AsyncMock()) as rag_stream,
            patch("api.chat.trace_event"),
        ):
            response = await send_message(
                ChatRequest(
                    question="那我的补餐补是多少",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=db,
                user=user,
            )
            payloads = [
                _parse_sse_payload(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
                async for chunk in response.body_iterator
            ]

        rag_stream.assert_not_awaited()
        self.assertIsNotNone(conv.pending_route_state)
        self.assertEqual(
            conv.pending_route_state["schema_version"],
            "rag_pending_clarification.v1",
        )
        self.assertIn(
            "请确认你的职级",
            "".join(
                str(item.get("content") or "")
                for item in payloads
                if item and item["type"] == "text_delta"
            ),
        )
        self.assertNotIn("error", [item["type"] for item in payloads if item])

    async def test_local_query_plan_clarification_stops_v2_before_retrieval(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        question = "该值取决于前一项"
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=question,
            history_messages=(),
            carryover_sources=(),
        )
        _route, _contract, routing_result = _route_and_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            selected_kb_count=1,
        )
        db = _RouteStateDB(conv)

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
            patch("api.chat.run_rag_v2_stream", new=AsyncMock()) as rag_v2,
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question=question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=db,
                user=user,
            )
            payloads = [
                _parse_sse_payload(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
                async for chunk in response.body_iterator
            ]

        rag_v2.assert_not_awaited()
        self.assertEqual(
            conv.pending_route_state["schema_version"],
            "rag_pending_clarification.v1",
        )
        self.assertEqual(
            conv.pending_route_state["unresolved"][0]["role"],
            "query_plan",
        )
        intent = next(
            item["decision"] for item in payloads if item and item["type"] == "intent"
        )
        self.assertEqual(intent["readiness"], "needs_clarification")
        self.assertFalse(intent["dispatch_authorized"])
        answer = "".join(
            str(item.get("content") or "")
            for item in payloads
            if item and item["type"] == "text_delta"
        )
        self.assertIn("对应关系", answer)
        self.assertIn("标准或数值", answer)
        trace_events = [call.args[0] for call in trace.call_args_list if call.args]
        self.assertNotIn("retrieval.plan", trace_events)
        self.assertIn("query.plan", trace_events)

    def test_broad_refinement_remains_one_query_plan(self) -> None:
        refined = _refined_evidence_query("员工标准是什么", "2025版")

        self.assertNotIn("\n", refined)
        self.assertNotIn("；", refined)
        self.assertIn("员工标准是什么", refined)
        self.assertIn("用户补充的适用范围：2025版", refined)

    async def test_pipeline_evidence_clarification_creates_v2_pending_state(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        route_log_id = uuid.uuid4()
        pending_template = _evidence_pending_state(kb_id=kb_id)
        raw_followup_question = "那这个要怎么配置"
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateDB(conv)
        route_log = SimpleNamespace(
            retrieval_executed=None,
            evidence_status=None,
            hit_count=None,
        )
        save_db = _SaveStateDB(
            conv,
            route_logs={route_log_id: route_log},
        )
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=pending_template["original_query"],
            history_messages=(),
            carryover_sources=(),
        )
        _route, _contract, routing_result = _route_and_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            selected_kb_count=1,
        )
        routing_result.route_log_id = route_log_id
        contradictory_source = {
            "id": str(uuid.uuid4()),
            "doc_id": pending_template["choices"][1]["doc_ids"][0],
            "kb_id": str(kb_id),
            "filename": "不应恢复为依据.md",
            "content": "只能作为候选展示",
            "evidence_role": "direct",
            "score": 0.99,
        }

        async def clarification_stream(**_kwargs):
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    "evidence_status": "needs_clarification",
                    "direct_evidence_count": 0,
                    "related_reference_count": 2,
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "type": "evidence_clarification",
                    "schema_version": "rag_evidence_clarification.v1",
                    "needs_clarification": True,
                    "dimension": pending_template["dimension"],
                    "question": pending_template["clarification_message"],
                    "reason": "multiple_mutually_exclusive_relevant_scopes",
                    "choices": pending_template["choices"],
                },
                ensure_ascii=False,
            ) + "\n\n"
            # 模拟异常/滚动升级生产者在最终澄清门禁之后又发出 hit。
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [contradictory_source],
                    "answer_sources": [contradictory_source],
                    "total": 1,
                    "retrieval_executed": True,
                    "evidence_status": "hit",
                    "hit_count": 1,
                    "direct_evidence_count": 1,
                    "context_evidence_count": 1,
                    "answer_source_count": 1,
                    "related_reference_count": 0,
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "type": "text_delta",
                    "content": pending_template["clarification_message"],
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "type": "text_delta",
                    "content": "这是澄清门禁后绝不能输出或保存的答案。",
                },
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
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch("api.chat.run_rag_v2_stream", new=clarification_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question=raw_followup_question,
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            self.assertIsNone(conv.pending_route_state)
            payloads = [
                _parse_sse_payload(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
                async for chunk in response.body_iterator
            ]

        state = conv.pending_route_state
        self.assertIsNotNone(state)
        self.assertEqual(state["schema_version"], "rag_pending_clarification.v2")
        self.assertEqual(state["kind"], "evidence_scope")
        self.assertFalse(state["dispatch_authorized"])
        self.assertEqual(state["original_query"], pending_template["original_query"])
        self.assertNotEqual(state["original_query"], raw_followup_question)
        self.assertEqual([item["key"] for item in state["choices"]], ["c1", "c2"])
        self.assertEqual(conv.route_state_revision, 1)
        self.assertEqual(_active_pending_route_state(state), state)
        assistant = next(item for item in save_db.added if item.role == "assistant")
        self.assertEqual(assistant.sources, [])
        self.assertEqual(
            assistant.content,
            pending_template["clarification_message"],
        )
        streamed_text = "".join(
            str(item.get("content") or "")
            for item in payloads
            if item and item["type"] == "text_delta"
        )
        self.assertEqual(streamed_text, pending_template["clarification_message"])
        self.assertNotIn("绝不能输出", streamed_text)
        self.assertNotIn("error", [item["type"] for item in payloads if item])
        self.assertTrue(route_log.retrieval_executed)
        self.assertEqual(route_log.evidence_status, "needs_clarification")
        self.assertEqual(route_log.hit_count, 0)
        search_payloads = [
            item for item in payloads if item and item["type"] == "search_results"
        ]
        locked_payload = search_payloads[-1]
        self.assertEqual(locked_payload["evidence_status"], "needs_clarification")
        self.assertEqual(locked_payload["answer_sources"], [])
        self.assertEqual(locked_payload["direct_evidence_count"], 0)
        self.assertEqual(locked_payload["context_evidence_count"], 0)
        self.assertEqual(locked_payload["hit_count"], 0)
        self.assertEqual(locked_payload["results"][0]["evidence_role"], "related")
        event_types = [item["type"] for item in payloads if item]
        self.assertLess(
            event_types.index("evidence_clarification"),
            event_types.index("evidence_clarification_ack"),
        )
        self.assertLess(
            event_types.index("evidence_clarification_ack"),
            event_types.index("done"),
        )
        ack = next(
            item
            for item in payloads
            if item and item["type"] == "evidence_clarification_ack"
        )
        self.assertEqual(
            ack["schema_version"],
            "rag_evidence_clarification_ack.v1",
        )
        self.assertTrue(ack["persisted"])
        self.assertEqual(ack["pending_state_id"], state["state_id"])
        self.assertEqual(ack["clarification_message_id"], str(assistant.id))
        self.assertEqual(ack["route_state_revision"], 1)
        self.assertEqual(ack["conversation_id"], str(conversation_id))
        events = [call.args[0] for call in trace.call_args_list]
        self.assertIn("evidence.clarification_created", events)
        self.assertNotIn("evidence.clarification_resolved", events)

    async def test_pipeline_evidence_clarification_save_failure_has_no_ack(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending_template = _evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=None,
            route_state_revision=0,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateDB(conv)
        save_db = _FailingSaveStateDB(conv)
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=pending_template["original_query"],
            history_messages=(),
            carryover_sources=(),
        )
        _route, _contract, routing_result = _route_and_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            selected_kb_count=1,
        )

        async def clarification_stream(**_kwargs):
            yield "data: " + json.dumps(
                {
                    "type": "evidence_clarification",
                    "schema_version": "rag_evidence_clarification.v1",
                    "needs_clarification": True,
                    "dimension": pending_template["dimension"],
                    "question": pending_template["clarification_message"],
                    "choices": pending_template["choices"],
                },
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
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch("api.chat.run_rag_v2_stream", new=clarification_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="解决登录用户名枚举要配置什么",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            payloads = [
                _parse_sse_payload(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
                async for chunk in response.body_iterator
            ]

        event_types = [item["type"] for item in payloads if item]
        self.assertIn("evidence_clarification", event_types)
        self.assertNotIn("evidence_clarification_ack", event_types)
        self.assertEqual(event_types[-2:], ["error", "done"])
        self.assertIsNone(conv.pending_route_state)
        self.assertEqual(conv.route_state_revision, 0)
        events = [call.args[0] for call in trace.call_args_list]
        self.assertIn("chat.persistence_error", events)
        self.assertNotIn("evidence.clarification_created", events)

    async def test_v2_selection_keeps_pending_when_answer_context_is_empty(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=4,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateDB(conv)
        save_db = _SaveStateDB(conv)
        context = _context(pending_route_state=pending)
        classify = AsyncMock(
            side_effect=AssertionError("证据范围选择不应调用意图模型")
        )
        received_kwargs = []

        async def answer_stream(**kwargs):
            received_kwargs.append(kwargs)
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    "evidence_status": "hit",
                    "direct_evidence_count": 1,
                    "related_reference_count": 0,
                    "evidence_scope_anchor_hit": True,
                    "evidence_scope_anchor_doc_ids": pending["choices"][1][
                        "anchor_doc_ids"
                    ],
                }
            ) + "\n\n"
            yield 'data: {"type":"text_delta","content":"8.2.75 配置答案"}\n\n'
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.classify_intent_result", new=classify),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.run_rag_v2_stream", new=answer_stream),
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
            # 用户选择已保存，但回答尚未完成；v2 不能提前清除。
            self.assertIs(conv.pending_route_state, pending)
            self.assertEqual(conv.route_state_revision, 4)
            [chunk async for chunk in response.body_iterator]

        classify.assert_not_awaited()
        scope_filter = received_kwargs[0]["evidence_scope_filter"]
        self.assertEqual(scope_filter["mode"], "single")
        self.assertEqual(scope_filter["choices"][0]["key"], "c2")
        # 范围标签只用于重新路由/合同；Pipeline 以原问题为 base query，
        # 再由 evidence_scope_filter 构造单选或全量对比查询。
        self.assertEqual(received_kwargs[0]["question"], pending["original_query"])
        self.assertEqual(
            received_kwargs[0]["standalone_query"],
            pending["original_query"],
        )
        self.assertEqual(
            received_kwargs[0]["task_contract"].decision_reason,
            "evidence_scope_selected",
        )
        # A producer-advertised anchor cannot resolve the scope when the
        # actual answer context is empty.  The pending choice remains
        # available for a retry after retrieval is healthy.
        self.assertIs(conv.pending_route_state, pending)
        self.assertEqual(conv.route_state_revision, 4)
        self.assertNotIn(
            "evidence.clarification_resolved",
            [call.args[0] for call in trace.call_args_list],
        )

    async def test_v2_selection_clears_pending_after_valid_source_commit(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        selected_doc_id = uuid.UUID(pending["choices"][1]["doc_ids"][0])
        chunk_id = uuid.uuid4()
        document = Document(
            id=selected_doc_id,
            kb_id=kb_id,
            filename="二开发送钉钉工作通知.md",
            status="ready",
            is_active=True,
            file_type="md",
        )
        chunk = DocumentChunk(
            id=chunk_id,
            doc_id=selected_doc_id,
            kb_id=kb_id,
            content="数据库刷新后的 8.2.75 配置依据",
            chunk_index=2,
            metadata_={"section": "用户名枚举"},
        )
        source = {
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(selected_doc_id),
            "kb_id": str(kb_id),
            "filename": "producer 旧标题.md",
            "content": "producer 旧正文",
            "evidence_role": "direct",
            "score": 0.98,
        }
        request_conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=4,
        )
        persisted_conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=json.loads(
                json.dumps(pending, ensure_ascii=False)
            ),
            route_state_revision=4,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateSourceDB(
            request_conv,
            source_rows=[(chunk, document)],
        )
        save_db = _SaveStateDB(persisted_conv)
        context = _context(pending_route_state=pending)
        _route, _contract, routing_result = _route_and_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            selected_kb_count=1,
            relation="continuation",
        )

        async def answer_stream(**_kwargs):
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [source],
                    "answer_sources": [source],
                    "total": 1,
                    "displayed_result_count": 1,
                    "retrieval_executed": True,
                    "evidence_status": "hit",
                    "direct_evidence_count": 1,
                    "related_reference_count": 0,
                    # These producer claims are intentionally ignored.  The
                    # API must recompute anchor coverage from refreshed rows.
                    "evidence_scope_anchor_hit": False,
                    "evidence_scope_anchor_doc_ids": [],
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield 'data: {"type":"text_delta","content":"8.2.75 的配置答案"}\n\n'
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
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.run_rag_v2_stream", new=answer_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="c2",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            # The request transaction only saves the user's choice.  Pending
            # scope remains until the streamed answer transaction commits.
            self.assertIs(request_conv.pending_route_state, pending)
            self.assertEqual(request_conv.route_state_revision, 4)
            payloads = [
                _parse_sse_payload(
                    item.decode() if isinstance(item, bytes) else item
                )
                async for item in response.body_iterator
            ]

        self.assertIsNone(persisted_conv.pending_route_state)
        self.assertEqual(persisted_conv.route_state_revision, 5)
        assistant = next(item for item in save_db.added if item.role == "assistant")
        self.assertEqual(assistant.content, "8.2.75 的配置答案")
        self.assertEqual(
            assistant.sources[0]["content"],
            "数据库刷新后的 8.2.75 配置依据",
        )
        self.assertEqual(
            assistant.sources[0]["filename"],
            "二开发送钉钉工作通知.md",
        )
        search_result = next(
            item for item in payloads if item and item["type"] == "search_results"
        )
        self.assertEqual(
            search_result["answer_sources"][0]["content"],
            "数据库刷新后的 8.2.75 配置依据",
        )
        self.assertTrue(search_result["evidence_scope_anchor_hit"])
        self.assertEqual(
            search_result["evidence_scope_anchor_doc_ids"],
            [str(selected_doc_id)],
        )
        resolved = next(
            call
            for call in trace.call_args_list
            if call.args[0] == "evidence.clarification_resolved"
        )
        self.assertEqual(resolved.kwargs["resolution"], "selected")
        self.assertEqual(resolved.kwargs["selected_choice_keys"], ["c2"])

    async def test_v2_selection_keeps_pending_when_answer_save_fails(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        selected_doc_id = uuid.UUID(pending["choices"][1]["doc_ids"][0])
        chunk_id = uuid.uuid4()
        document = Document(
            id=selected_doc_id,
            kb_id=kb_id,
            filename="二开发送钉钉工作通知.md",
            status="ready",
            is_active=True,
            file_type="md",
        )
        chunk = DocumentChunk(
            id=chunk_id,
            doc_id=selected_doc_id,
            kb_id=kb_id,
            content="数据库中的有效依据",
            chunk_index=2,
            metadata_={},
        )
        source = {
            "id": str(chunk_id),
            "chunk_id": str(chunk_id),
            "doc_id": str(selected_doc_id),
            "kb_id": str(kb_id),
            "filename": document.filename,
            "content": "producer 快照",
            "evidence_role": "direct",
        }
        request_conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=12,
        )
        persisted_conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=json.loads(
                json.dumps(pending, ensure_ascii=False)
            ),
            route_state_revision=12,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateSourceDB(
            request_conv,
            source_rows=[(chunk, document)],
        )
        save_db = _FailingSaveStateDB(persisted_conv)
        context = _context(pending_route_state=pending)
        _route, _contract, routing_result = _route_and_contract(
            intent_code="knowledge_qa",
            action="retrieve",
            evidence_scope="enterprise_kb",
            selected_kb_count=1,
            relation="continuation",
        )

        async def answer_stream(**_kwargs):
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [source],
                    "answer_sources": [source],
                    "total": 1,
                    "retrieval_executed": True,
                    "evidence_status": "hit",
                    "direct_evidence_count": 1,
                    "related_reference_count": 0,
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield 'data: {"type":"text_delta","content":"已经生成的答案"}\n\n'
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
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.run_rag_v2_stream", new=answer_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="c2",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            payloads = [
                _parse_sse_payload(
                    item.decode() if isinstance(item, bytes) else item
                )
                async for item in response.body_iterator
            ]

        self.assertIs(request_conv.pending_route_state, pending)
        self.assertEqual(request_conv.route_state_revision, 12)
        self.assertEqual(
            persisted_conv.pending_route_state["state_id"],
            pending["state_id"],
        )
        self.assertEqual(persisted_conv.route_state_revision, 12)
        error = next(item for item in payloads if item and item["type"] == "error")
        self.assertEqual(error["message"], "回答已生成，但保存失败，请重试")
        self.assertEqual(
            [item["type"] for item in payloads if item][-1],
            "done",
        )
        events = [call.args[0] for call in trace.call_args_list]
        self.assertIn("chat.persistence_error", events)
        self.assertNotIn("evidence.clarification_resolved", events)
        self.assertNotIn("chat.response", events)

    async def test_v2_selection_builds_route_without_intent_model(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=8,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateDB(conv)
        save_db = _SaveStateDB(conv)
        context = _context(pending_route_state=pending)
        classify = AsyncMock(
            side_effect=AssertionError("证据范围选择不应调用意图模型")
        )
        received_kwargs = []

        async def answer_stream(**kwargs):
            received_kwargs.append(kwargs)
            yield "data: " + json.dumps({
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "retrieval_executed": True,
                "evidence_status": "hit",
                "direct_evidence_count": 1,
                "related_reference_count": 0,
                "evidence_scope_anchor_hit": True,
                "evidence_scope_anchor_doc_ids": pending["choices"][1][
                    "anchor_doc_ids"
                ],
            }) + "\n\n"
            yield 'data: {"type":"text_delta","content":"恢复后的范围答案"}\n\n'
            yield "data: " + json.dumps({
                "type": "done",
                "conversation_id": str(conversation_id),
            }) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch(
                "api.chat.classify_intent_result",
                new=classify,
            ),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.run_rag_v2_stream", new=answer_stream),
            patch(
                "api.chat._route_clarification_response",
                new=AsyncMock(),
            ) as route_clarification,
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="c2",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            self.assertIs(conv.pending_route_state, pending)
            [chunk async for chunk in response.body_iterator]

        route_clarification.assert_not_awaited()
        classify.assert_not_awaited()
        self.assertEqual(len(received_kwargs), 1)
        recovered_contract = received_kwargs[0]["task_contract"]
        self.assertTrue(recovered_contract.dispatch_authorized)
        self.assertEqual(recovered_contract.readiness, "ready")
        self.assertEqual(recovered_contract.action, "retrieve")
        self.assertEqual(recovered_contract.source, "evidence_pending_rule")
        self.assertEqual(recovered_contract.response_mode, "grounded_qa")
        self.assertEqual(recovered_contract.retrieval_policy, "required")
        self.assertEqual(recovered_contract.decision_reason, "evidence_scope_selected")
        self.assertEqual(
            received_kwargs[0]["evidence_scope_filter"]["choices"][0]["key"],
            "c2",
        )
        self.assertIs(conv.pending_route_state, pending)
        self.assertEqual(conv.route_state_revision, 8)
        self.assertIn(
            "evidence.route_contract_built",
            [call.args[0] for call in trace.call_args_list],
        )

    async def test_broad_refinement_combines_query_and_clears_after_success(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _broad_evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=6,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateDB(conv)
        save_db = _SaveStateDB(conv)
        context = _context(pending_route_state=pending)
        classify = AsyncMock(
            side_effect=AssertionError("证据范围补充不应调用意图模型")
        )
        received_kwargs = []

        async def answer_stream(**kwargs):
            received_kwargs.append(kwargs)
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    "evidence_status": "no_hit",
                    "direct_evidence_count": 0,
                    "related_reference_count": 0,
                }
            ) + "\n\n"
            yield 'data: {"type":"text_delta","content":"2025版暂无直接资料"}\n\n'
            yield "data: " + json.dumps(
                {"type": "done", "conversation_id": str(conversation_id)}
            ) + "\n\n"

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.classify_intent_result", new=classify),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.run_rag_v2_stream", new=answer_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="2025版",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            self.assertIs(conv.pending_route_state, pending)
            [chunk async for chunk in response.body_iterator]

        classify.assert_not_awaited()
        refined_query = received_kwargs[0]["question"]
        self.assertIn(pending["original_query"], refined_query)
        self.assertIn("用户补充的适用范围：2025版", refined_query)
        self.assertEqual(received_kwargs[0]["standalone_query"], refined_query)
        self.assertEqual(
            received_kwargs[0]["task_contract"].decision_reason,
            "evidence_scope_refined",
        )
        self.assertIsNone(received_kwargs[0]["evidence_scope_filter"])
        self.assertEqual(received_kwargs[0]["kb_ids"], [kb_id])
        self.assertIsNone(conv.pending_route_state)
        self.assertEqual(conv.route_state_revision, 7)
        resolved = next(
            call
            for call in trace.call_args_list
            if call.args[0] == "evidence.clarification_resolved"
        )
        self.assertEqual(resolved.kwargs["resolution"], "refined")

    async def test_new_bounded_clarification_replaces_broad_pending_state(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _broad_evidence_pending_state(kb_id=kb_id)
        bounded = _evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=9,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateDB(conv)
        save_db = _SaveStateDB(conv)
        context = _context(pending_route_state=pending)
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

        async def clarification_stream(**_kwargs):
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "retrieval_executed": True,
                    "evidence_status": "needs_clarification",
                    "direct_evidence_count": 0,
                    "related_reference_count": 2,
                }
            ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "type": "evidence_clarification",
                    "schema_version": "rag_evidence_clarification.v1",
                    "needs_clarification": True,
                    "dimension": "version",
                    "question": bounded["clarification_message"],
                    "choices": bounded["choices"],
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield "data: " + json.dumps(
                {
                    "type": "text_delta",
                    "content": bounded["clarification_message"],
                },
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
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch("api.chat.run_rag_v2_stream", new=clarification_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="2025版",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            [chunk async for chunk in response.body_iterator]

        replacement = conv.pending_route_state
        self.assertEqual(replacement["selection_mode"], "choice")
        self.assertEqual([item["key"] for item in replacement["choices"]], ["c1", "c2"])
        self.assertNotEqual(replacement["state_id"], pending["state_id"])
        self.assertEqual(conv.route_state_revision, 10)
        events = [call.args[0] for call in trace.call_args_list]
        self.assertIn("evidence.clarification_resolved", events)
        self.assertIn("evidence.clarification_created", events)

    async def test_invalid_short_v2_reply_repeats_choices_without_dispatch(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=2,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)
        context = _context(pending_route_state=pending)

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
                    question="随便一个",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=db,
                user=user,
            )
            payloads = [
                _parse_sse_payload(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
                async for chunk in response.body_iterator
            ]

        classify.assert_not_awaited()
        rag_stream.assert_not_called()
        self.assertIs(conv.pending_route_state, pending)
        self.assertEqual(conv.route_state_revision, 2)
        event_types = [item["type"] for item in payloads if item]
        self.assertLess(
            event_types.index("evidence_clarification"),
            event_types.index("evidence_clarification_ack"),
        )
        self.assertLess(
            event_types.index("evidence_clarification_ack"),
            event_types.index("done"),
        )
        ack = next(
            item
            for item in payloads
            if item and item["type"] == "evidence_clarification_ack"
        )
        self.assertTrue(ack["persisted"])
        self.assertEqual(ack["pending_state_id"], pending["state_id"])
        self.assertEqual(
            ack["clarification_message_id"],
            pending["clarification_message_id"],
        )
        self.assertEqual(ack["route_state_revision"], 2)
        self.assertIn(
            "evidence.clarification_repeated",
            [call.args[0] for call in trace.call_args_list],
        )

    async def test_v2_selection_keeps_pending_state_when_pipeline_fails(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=7,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)
        context = _context(pending_route_state=pending)
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

        async def failing_stream(**_kwargs):
            if False:  # pragma: no cover - keep async-generator semantics
                yield ""
            raise RuntimeError("temporary retrieval failure")

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
                    question="8.2.75",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=db,
                user=user,
            )
            [chunk async for chunk in response.body_iterator]

        self.assertIs(conv.pending_route_state, pending)
        self.assertEqual(conv.route_state_revision, 7)
        self.assertNotIn(
            "evidence.clarification_resolved",
            [call.args[0] for call in trace.call_args_list],
        )

    async def test_broad_refinement_keeps_pending_on_handled_error_or_skipped_status(self) -> None:
        for terminal_status in ("error", "skipped"):
            with self.subTest(terminal_status=terminal_status):
                conversation_id = uuid.uuid4()
                user_id = uuid.uuid4()
                kb_id = uuid.uuid4()
                pending = _broad_evidence_pending_state(kb_id=kb_id)
                conv = SimpleNamespace(
                    id=conversation_id,
                    user_id=user_id,
                    pending_route_state=pending,
                    route_state_revision=11,
                )
                user = SimpleNamespace(id=user_id, is_superadmin=False)
                request_db = _RouteStateDB(conv)
                save_db = _SaveStateDB(conv)
                context = _context(pending_route_state=pending)
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
                    route_log_id=None,
                )

                async def handled_failure_stream(**_kwargs):
                    yield "data: " + json.dumps(
                        {
                            "type": "search_results",
                            "results": [],
                            "answer_sources": [],
                            "retrieval_executed": terminal_status != "skipped",
                            "evidence_status": terminal_status,
                            "direct_evidence_count": 0,
                            "related_reference_count": 0,
                        }
                    ) + "\n\n"
                    yield "data: " + json.dumps(
                        {
                            "type": "text_delta",
                            "content": "本次证据处理失败，请重试。",
                        },
                        ensure_ascii=False,
                    ) + "\n\n"
                    yield "data: " + json.dumps(
                        {"type": "done", "conversation_id": str(conversation_id)}
                    ) + "\n\n"

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
                        "api.chat.run_rag_v2_stream",
                        new=handled_failure_stream,
                    ),
                    patch("database.AsyncSessionLocal", return_value=save_db),
                    patch("api.chat.trace_event") as trace,
                ):
                    response = await send_message(
                        ChatRequest(
                            question="2025版",
                            conversation_id=conversation_id,
                            knowledge_base_ids=[kb_id],
                        ),
                        db=request_db,
                        user=user,
                    )
                    [chunk async for chunk in response.body_iterator]

                self.assertIs(conv.pending_route_state, pending)
                self.assertEqual(conv.route_state_revision, 11)
                assistant = next(
                    item for item in save_db.added if item.role == "assistant"
                )
                self.assertEqual(assistant.sources, [])
                self.assertNotIn(
                    "evidence.clarification_resolved",
                    [call.args[0] for call in trace.call_args_list],
                )

    async def test_companion_only_hit_does_not_resolve_bounded_selection(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        shared_companion = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        for choice in pending["choices"]:
            choice["doc_ids"].append(str(shared_companion))
            choice["companion_doc_ids"] = [str(shared_companion)]
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=14,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        request_db = _RouteStateDB(conv)
        save_db = _SaveStateDB(conv)
        context = _context(pending_route_state=pending)
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
        companion_source = {
            "id": str(uuid.uuid4()),
            "doc_id": str(shared_companion),
            "kb_id": str(kb_id),
            "filename": "跨范围通用说明.md",
            "content": "通用内容",
            "evidence_role": "related",
        }

        async def companion_stream(**_kwargs):
            yield "data: " + json.dumps(
                {
                    "type": "search_results",
                    "results": [companion_source],
                    "answer_sources": [companion_source],
                    "retrieval_executed": True,
                    "evidence_status": "hit",
                    "direct_evidence_count": 1,
                    "related_reference_count": 1,
                    "evidence_scope_anchor_hit": False,
                    "evidence_scope_anchor_doc_ids": [],
                },
                ensure_ascii=False,
            ) + "\n\n"
            yield 'data: {"type":"text_delta","content":"仅命中通用说明"}\n\n'
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
                "api.chat.classify_intent_result",
                new=AsyncMock(return_value=routing_result),
            ),
            patch("api.chat.run_rag_v2_stream", new=companion_stream),
            patch("database.AsyncSessionLocal", return_value=save_db),
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="c2",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=request_db,
                user=user,
            )
            [chunk async for chunk in response.body_iterator]

        self.assertIs(conv.pending_route_state, pending)
        self.assertEqual(conv.route_state_revision, 14)
        self.assertNotIn(
            "evidence.clarification_resolved",
            [call.args[0] for call in trace.call_args_list],
        )

    async def test_selection_does_not_resume_after_kb_scope_expands(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        old_kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=old_kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=2,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=_context(pending_route_state=pending)),
            ),
            patch("api.chat.classify_intent_result", new=AsyncMock()) as classify,
        ):
            response = await send_message(
                ChatRequest(
                    question="c2",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[old_kb_id, uuid.uuid4()],
                ),
                db=db,
                user=user,
            )
            payloads = [
                _parse_sse_payload(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
                async for chunk in response.body_iterator
            ]

        classify.assert_not_awaited()
        serialized = json.dumps(payloads, ensure_ascii=False)
        self.assertNotIn("8.2.75", serialized)
        self.assertNotIn("二开发送钉钉工作通知", serialized)
        self.assertNotIn(
            "evidence_clarification",
            [item["type"] for item in payloads if item],
        )
        self.assertIs(conv.pending_route_state, pending)

    async def test_broad_refinement_does_not_search_newly_added_kb(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        old_kb_id = uuid.uuid4()
        pending = _broad_evidence_pending_state(kb_id=old_kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=3,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=_context(pending_route_state=pending)),
            ),
            patch("api.chat.classify_intent_result", new=AsyncMock()) as classify,
            patch("api.chat.run_rag_stream") as rag_stream,
        ):
            response = await send_message(
                ChatRequest(
                    question="2025版",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[old_kb_id, uuid.uuid4()],
                ),
                db=db,
                user=user,
            )
            payloads = [
                _parse_sse_payload(
                    chunk.decode() if isinstance(chunk, bytes) else chunk
                )
                async for chunk in response.body_iterator
            ]

        classify.assert_not_awaited()
        rag_stream.assert_not_called()
        self.assertIs(conv.pending_route_state, pending)
        self.assertEqual(conv.route_state_revision, 3)
        self.assertIn(
            "知识库范围不一致",
            "".join(
                str(item.get("content") or "")
                for item in payloads
                if item and item.get("type") == "text_delta"
            ),
        )

    async def test_v2_cancel_clears_state_without_router_dispatch(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=3,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch(
                "api.chat.prepare_conversation_context",
                new=AsyncMock(return_value=_context(pending_route_state=pending)),
            ),
            patch("api.chat.classify_intent_result", new=AsyncMock()) as classify,
            patch("api.chat.trace_event") as trace,
        ):
            response = await send_message(
                ChatRequest(
                    question="取消",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=db,
                user=user,
            )
            [chunk async for chunk in response.body_iterator]

        classify.assert_not_awaited()
        self.assertIsNone(conv.pending_route_state)
        self.assertEqual(conv.route_state_revision, 4)
        resolved = next(
            call
            for call in trace.call_args_list
            if call.args[0] == "evidence.clarification_resolved"
        )
        self.assertEqual(resolved.kwargs["resolution"], "cancelled")

    async def test_explicit_new_question_clears_v2_before_fresh_routing(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        pending = _evidence_pending_state(kb_id=kb_id)
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=5,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="8.2.75版本的密码策略怎么配置",
            history_messages=(),
            carryover_sources=(),
            pending_route_state=None,
        )
        _route, _contract, routing_result = _route_and_contract(
            intent_code="general_chat",
            action="chat",
            evidence_scope="general_world",
            selected_kb_count=1,
        )
        prepare = AsyncMock(return_value=context)
        classify = AsyncMock(return_value=routing_result)

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=prepare),
            patch("api.chat.classify_intent_result", new=classify),
            patch("api.chat.trace_event") as trace,
        ):
            await send_message(
                ChatRequest(
                    question="8.2.75版本的密码策略怎么配置",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[kb_id],
                ),
                db=db,
                user=user,
            )

        self.assertIsNone(prepare.await_args.kwargs["pending_route_state"])
        self.assertEqual(
            prepare.await_args.kwargs["question"],
            "8.2.75版本的密码策略怎么配置",
        )
        self.assertFalse(classify.await_args.kwargs["has_pending_clarification"])
        self.assertEqual(
            classify.await_args.args[1],
            "8.2.75版本的密码策略怎么配置",
        )
        self.assertIsNone(conv.pending_route_state)
        self.assertEqual(conv.route_state_revision, 6)
        resolved = next(
            call
            for call in trace.call_args_list
            if call.args[0] == "evidence.clarification_resolved"
        )
        self.assertEqual(resolved.kwargs["resolution"], "new_question")

    async def test_pending_reply_reenters_full_router_then_clears_state(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        pending = {
            "schema_version": "rag_pending_clarification.v1",
            "state_id": "pending-state",
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
            "dispatch_authorized": False,
        }
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=pending,
            route_state_revision=4,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)
        context = _context(pending_route_state=pending)
        _route, _contract, routing_result = _route_and_contract(
            intent_code="general_chat",
            action="chat",
            evidence_scope="general_world",
            selected_kb_count=0,
            relation="continuation",
        )
        prepare = AsyncMock(return_value=context)
        classify = AsyncMock(return_value=routing_result)

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=prepare),
            patch("api.chat.classify_intent_result", new=classify),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.trace_event") as trace,
        ):
            await send_message(
                ChatRequest(
                    question="补充后的完整回答",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[],
                ),
                db=db,
                user=user,
            )

        self.assertIs(prepare.await_args.kwargs["pending_route_state"], pending)
        self.assertTrue(classify.await_args.kwargs["has_pending_clarification"])
        self.assertIsNone(conv.pending_route_state)
        self.assertEqual(conv.route_state_revision, 5)
        self.assertEqual(db.commits, 1)
        resolved = next(
            call
            for call in trace.call_args_list
            if call.args[0] == "intent.clarification_resolved"
        )
        self.assertEqual(resolved.kwargs["pending_state_id"], "pending-state")
        self.assertEqual(resolved.kwargs["relation"], "continuation")

    async def test_expired_pending_state_is_cleared_before_routing(self) -> None:
        conversation_id = uuid.uuid4()
        user_id = uuid.uuid4()
        expired = {
            "schema_version": "rag_pending_clarification.v1",
            "state_id": "expired-state",
            "expires_at": (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat(),
            "dispatch_authorized": False,
        }
        conv = SimpleNamespace(
            id=conversation_id,
            user_id=user_id,
            pending_route_state=expired,
            route_state_revision=8,
        )
        user = SimpleNamespace(id=user_id, is_superadmin=False)
        db = _RouteStateDB(conv)
        context = _context(pending_route_state=None)
        _route, _contract, routing_result = _route_and_contract(
            intent_code="general_chat",
            action="chat",
            evidence_scope="general_world",
            selected_kb_count=0,
        )
        prepare = AsyncMock(return_value=context)
        classify = AsyncMock(return_value=routing_result)

        with (
            patch("api.chat.get_accessible_kb_ids", new=AsyncMock(return_value=None)),
            patch("api.chat.prepare_conversation_context", new=prepare),
            patch("api.chat.classify_intent_result", new=classify),
            patch(
                "api.chat.resolve_routed_conversation_context",
                new=AsyncMock(return_value=context),
            ),
            patch("api.chat.trace_event") as trace,
        ):
            await send_message(
                ChatRequest(
                    question="新的独立问题",
                    conversation_id=conversation_id,
                    knowledge_base_ids=[],
                ),
                db=db,
                user=user,
            )

        self.assertIsNone(prepare.await_args.kwargs["pending_route_state"])
        self.assertFalse(classify.await_args.kwargs["has_pending_clarification"])
        self.assertIsNone(conv.pending_route_state)
        self.assertEqual(conv.route_state_revision, 9)
        events = [call.args[0] for call in trace.call_args_list]
        self.assertEqual(events[0], "chat.request")
        self.assertIn("intent.clarification_expired", events)
        self.assertGreater(events.index("intent.clarification_expired"), 0)
        self.assertNotIn("intent.clarification_resolved", events)


if __name__ == "__main__":
    unittest.main()
