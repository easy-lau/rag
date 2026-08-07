import unittest
import uuid
from contextlib import asynccontextmanager

from core.active_task_state import (
    ACTIVE_TASK_STATE_SCHEMA_VERSION,
    ResolvedActiveTask,
    build_active_task_state,
    parse_active_task_state,
    resolve_active_task_state,
)
from core.semantic_memory import extract_resolved_entity_memory
from core.conversation_context import (
    ConversationContext,
    RouteTurnCandidate,
    apply_active_task_context,
    build_active_task_v2_execution_context,
)
from models.db_models import Document, DocumentChunk


class _Rows:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class _DB:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, _statement):
        return _Rows(self.rows)


def _source(*, kb_id, doc_id, chunk_id):
    return {
        "kb_id": str(kb_id),
        "doc_id": str(doc_id),
        "id": str(chunk_id),
        "chunk_id": str(chunk_id),
        "evidence_role": "direct",
    }


class ActiveTaskStateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.kb_id = uuid.uuid4()
        self.doc_id = uuid.uuid4()
        self.chunk_id = uuid.uuid4()
        self.turn_id = uuid.uuid4()
        self.state = build_active_task_state(
            root_query="默认密码强制修改",
            answer_shape="fact",
            sources=[_source(
                kb_id=self.kb_id,
                doc_id=self.doc_id,
                chunk_id=self.chunk_id,
            )],
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
        )

    def test_state_round_trip_is_strict_and_content_free(self):
        payload = self.state.to_dict()

        parsed = parse_active_task_state(payload)

        self.assertEqual(parsed, self.state)
        self.assertNotIn("content", payload)
        self.assertNotIn("filename", payload)
        self.assertIsNone(parse_active_task_state({**payload, "dispatch_authorized": True}))


    def test_v1_state_without_semantic_memory_parses_for_backward_compatibility(self):
        payload = self.state.to_dict()
        v1_payload = {
            key: value
            for key, value in payload.items()
            if key != "semantic_memory"
        }
        v1_payload["schema_version"] = "rag_active_task.v1"

        parsed = parse_active_task_state(v1_payload)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.schema_version, "rag_active_task.v1")
        self.assertIsNone(parsed.semantic_memory)
        self.assertEqual(parsed.revision, self.state.revision)
        self.assertEqual(parsed.selected_chunk_ids, self.state.selected_chunk_ids)

    def test_v2_state_round_trip_preserves_semantic_memory(self):
        memory = extract_resolved_entity_memory(
            sources=[{
                **dict(_source(
                    kb_id=self.kb_id,
                    doc_id=self.doc_id,
                    chunk_id=self.chunk_id,
                )),
                "content": (
                    "| 职级 | 适用人员 |\n"
                    "| --- | --- |\n"
                    "| D级 | 普通员工、专员 |"
                ),
                "filename": "公司出差管理标准.docx",
            }],
            question="普通员工出差时可以乘坐的交通工具有哪些",
            source_turn_id=self.turn_id,
            trace_id=self.state.trace_id,
        )
        state = build_active_task_state(
            root_query="普通员工出差时可以乘坐的交通工具有哪些",
            answer_shape="fact",
            sources=[_source(
                kb_id=self.kb_id,
                doc_id=self.doc_id,
                chunk_id=self.chunk_id,
            )],
            source_turn_id=self.turn_id,
            trace_id=self.state.trace_id,
            semantic_memory=memory,
        )

        self.assertEqual(state.schema_version, ACTIVE_TASK_STATE_SCHEMA_VERSION)
        parsed = parse_active_task_state(state.to_dict())

        self.assertEqual(parsed, state)
        assert parsed is not None and parsed.semantic_memory is not None
        self.assertEqual(parsed.semantic_memory, memory)
        self.assertNotIn("content", str(state.to_dict()))

    def test_corrupt_semantic_memory_fails_closed(self):
        payload = self.state.to_dict()
        payload["semantic_memory"] = {
            "schema_version": "rag_semantic_memory.v1",
            "facts": "corrupt",
            "source_chunk_ids": [],
        }

        self.assertIsNone(parse_active_task_state(payload))

    async def test_resolution_reloads_only_current_ready_document_scope(self):
        document = Document(
            id=self.doc_id,
            kb_id=self.kb_id,
            filename="云枢6配置参数说明",
            status="ready",
            is_active=True,
            tags=["配置"],
        )
        chunk = DocumentChunk(
            id=self.chunk_id,
            doc_id=self.doc_id,
            kb_id=self.kb_id,
            content="force_change_default_password: true # 默认密码强制修改",
            chunk_index=3,
            metadata_={"heading": "解决方案"},
        )

        resolved = await resolve_active_task_state(
            _DB([(chunk, document)]),
            value=self.state.to_dict(),
            selected_kb_ids=[self.kb_id],
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.doc_ids, (self.doc_id,))
        self.assertEqual(resolved.sources[0]["candidate_origin"], "active_task_state")
        forbidden = await resolve_active_task_state(
            _DB([(chunk, document)]),
            value=self.state.to_dict(),
            selected_kb_ids=[uuid.uuid4()],
        )
        self.assertIsNone(forbidden)

    async def test_owned_resolution_projects_rows_before_session_rollback(self):
        """Owned read sessions must never leak detached ORM instances."""

        document = Document(
            id=self.doc_id,
            kb_id=self.kb_id,
            filename="云枢6配置参数说明.md",
            status="ready",
            is_active=True,
            file_type="md",
            source_url="https://example.test/config",
            tags=["配置", "安全"],
        )
        chunk = DocumentChunk(
            id=self.chunk_id,
            doc_id=self.doc_id,
            kb_id=self.kb_id,
            content="force_change_default_password: true # 默认密码强制修改",
            chunk_index=3,
            metadata_={"heading": "解决方案"},
        )
        session_closed = {"value": False}

        class ExpiringRow:
            def __init__(self, value):
                self._value = value

            def __getattr__(self, name):
                if session_closed["value"]:
                    raise AssertionError(
                        "ORM attribute accessed after owned read-session rollback"
                    )
                return getattr(self._value, name)

        owned_sessions = []

        @asynccontextmanager
        async def read_session_factory():
            session = _DB([
                (
                    ExpiringRow(chunk),
                    ExpiringRow(document),
                )
            ])
            session.rollback_calls = 0

            async def rollback():
                session.rollback_calls += 1
                session_closed["value"] = True

            session.rollback = rollback
            owned_sessions.append(session)
            yield session

        class RequestDB:
            async def execute(self, _statement):
                raise AssertionError("resolution must use the owned read session")

        resolved = await resolve_active_task_state(
            RequestDB(),
            value=self.state.to_dict(),
            selected_kb_ids=[self.kb_id],
            read_session_factory=read_session_factory,
        )

        self.assertIsNotNone(resolved)
        self.assertTrue(session_closed["value"])
        self.assertEqual(len(owned_sessions), 1)
        self.assertEqual(owned_sessions[0].rollback_calls, 1)
        self.assertEqual(resolved.kb_ids, (self.kb_id,))
        self.assertEqual(resolved.doc_ids, (self.doc_id,))
        self.assertEqual(resolved.sources[0]["id"], self.chunk_id)
        self.assertEqual(resolved.sources[0]["content"], chunk.content)
        self.assertEqual(
            resolved.sources[0]["metadata"],
            {"heading": "解决方案"},
        )
        self.assertEqual(resolved.sources[0]["filename"], document.filename)
        self.assertEqual(resolved.sources[0]["doc_tags"], ["配置", "安全"])

    async def test_missing_action_object_uses_task_root_and_skips_model_dependency(self):
        document = Document(
            id=self.doc_id,
            kb_id=self.kb_id,
            filename="云枢6配置参数说明",
            status="ready",
            is_active=True,
            tags=[],
        )
        chunk = DocumentChunk(
            id=self.chunk_id,
            doc_id=self.doc_id,
            kb_id=self.kb_id,
            content="force_change_default_password: true # 默认密码强制修改",
            chunk_index=3,
            metadata_={},
        )
        resolved = await resolve_active_task_state(
            _DB([(chunk, document)]),
            value=self.state.to_dict(),
            selected_kb_ids=[self.kb_id],
        )
        candidate = RouteTurnCandidate(
            candidate_key="t1",
            user_question="选择云枢6配置参数说明",
            assistant_answer="资料说明了该配置。",
            raw_sources=(_source(
                kb_id=self.kb_id,
                doc_id=self.doc_id,
                chunk_id=self.chunk_id,
            ),),
            assistant_turn_id=self.turn_id,
        )
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="应该如何配置",
            history_messages=(),
            carryover_sources=(),
            route_turn_candidates=(candidate,),
        )

        applied = apply_active_task_context(
            context=context,
            question="应该如何配置",
            resolved_task=resolved,
        )
        execution = build_active_task_v2_execution_context(context=applied)

        self.assertEqual(applied.query_resolution_mode, "active_task_state")
        self.assertIn("默认密码强制修改", applied.standalone_query)
        self.assertEqual(execution.mode, "active_task_state")
        self.assertTrue(execution.semantic_context_applied)
        self.assertEqual(len(execution.carryover_sources), 1)

    async def test_newer_turn_cannot_revive_stale_active_task(self):
        context = ConversationContext(
            is_followup=True,
            followup_reason="missing_action_object",
            standalone_query="应该如何配置新问题",
            history_messages=(),
            carryover_sources=(),
            route_turn_candidates=(RouteTurnCandidate(
                candidate_key="t1",
                user_question="新问题",
                assistant_answer=None,
                assistant_turn_id=None,
            ),),
        )

        applied = apply_active_task_context(
            context=context,
            question="应该如何配置",
            resolved_task=None,
        )

        self.assertIs(applied, context)

    def test_same_ordinal_continues_the_reauthorized_selected_document(self):
        state = build_active_task_state(
            root_query="我要看第四个",
            answer_shape="overview",
            sources=[_source(
                kb_id=self.kb_id,
                doc_id=self.doc_id,
                chunk_id=self.chunk_id,
            )],
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
        )
        source = _source(
            kb_id=self.kb_id,
            doc_id=self.doc_id,
            chunk_id=self.chunk_id,
        )
        resolved = ResolvedActiveTask(
            state=state,
            sources=(source,),
            kb_ids=(self.kb_id,),
            doc_ids=(self.doc_id,),
        )
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="第四个里面这么点东西吗",
            history_messages=(),
            carryover_sources=(),
            route_turn_candidates=(RouteTurnCandidate(
                candidate_key="t1",
                user_question=state.root_query,
                assistant_answer="所选文档正文",
                raw_sources=(source,),
                assistant_turn_id=self.turn_id,
            ),),
        )

        applied = apply_active_task_context(
            context=context,
            question="第四个里面这么点东西吗",
            resolved_task=resolved,
        )

        self.assertIs(applied.active_task, resolved)
        self.assertEqual(applied.active_task_scope_mode, "document")
        self.assertEqual(
            applied.followup_reason,
            "active_task_state:active_result_reference",
        )
        self.assertEqual(applied.carryover_sources, (source,))
        self.assertIn("我要看第四个", applied.standalone_query)

    def _entity_memory_state(self, *, question: str = "普通员工出差时可以乘坐的交通工具有哪些"):
        memory = extract_resolved_entity_memory(
            sources=[{
                **dict(_source(
                    kb_id=self.kb_id,
                    doc_id=self.doc_id,
                    chunk_id=self.chunk_id,
                )),
                "content": (
                    "【公司出差管理标准.docx › 二、职级分类】\n"
                    "| 职级 | 适用人员 |\n"
                    "| --- | --- |\n"
                    "| D级 | 普通员工、专员 |"
                ),
                "filename": "公司出差管理标准.docx",
            }],
            question=question,
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
        )
        return build_active_task_state(
            root_query=question,
            answer_shape="fact",
            sources=[_source(
                kb_id=self.kb_id,
                doc_id=self.doc_id,
                chunk_id=self.chunk_id,
            )],
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
            semantic_memory=memory,
        )

    def _resolved_context(self, *, question: str = "普通员工出差时可以乘坐的交通工具有哪些"):
        state = self._entity_memory_state(question=question)
        source = _source(
            kb_id=self.kb_id,
            doc_id=self.doc_id,
            chunk_id=self.chunk_id,
        )
        resolved = ResolvedActiveTask(
            state=state,
            sources=(source,),
            kb_ids=(self.kb_id,),
            doc_ids=(self.doc_id,),
        )
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query=question,
            history_messages=(),
            carryover_sources=(),
            route_turn_candidates=(RouteTurnCandidate(
                candidate_key="t1",
                user_question=state.root_query,
                assistant_answer="D级适用普通员工、专员；国内/国际航班均为经济舱。",
                raw_sources=(source,),
                assistant_turn_id=self.turn_id,
            ),),
        )
        return resolved, source, context

    def test_standalone_question_reusing_resolved_entity_carries_sources(self):
        resolved, source, context = self._resolved_context()

        applied = apply_active_task_context(
            context=context,
            question="普通员工可以乘坐头等舱吗",
            resolved_task=resolved,
        )

        self.assertIs(applied.active_task, resolved)
        self.assertEqual(applied.active_task_scope_mode, "document")
        self.assertEqual(
            applied.followup_reason,
            "active_task_state:semantic_entity_reuse",
        )
        self.assertEqual(applied.carryover_sources, (source,))
        # The question itself is complete: the retrieval anchor stays the
        # literal current question and is not concatenated with history.
        self.assertEqual(applied.standalone_query, "普通员工可以乘坐头等舱吗")
        self.assertTrue(applied.is_followup)
        self.assertEqual(applied.relation, "followup")

    def test_standalone_question_without_entity_reuse_stays_independent(self):
        resolved, _source_row, context = self._resolved_context()

        applied = apply_active_task_context(
            context=context,
            question="员工请假流程是什么",
            resolved_task=resolved,
        )

        self.assertIs(applied, context)
        self.assertIsNone(applied.active_task)
        self.assertFalse(applied.is_followup)

    def test_entity_reuse_requires_semantic_memory(self):
        state = build_active_task_state(
            root_query="普通员工出差时可以乘坐的交通工具有哪些",
            answer_shape="fact",
            sources=[_source(
                kb_id=self.kb_id,
                doc_id=self.doc_id,
                chunk_id=self.chunk_id,
            )],
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
        )
        source = _source(
            kb_id=self.kb_id,
            doc_id=self.doc_id,
            chunk_id=self.chunk_id,
        )
        resolved = ResolvedActiveTask(
            state=state,
            sources=(source,),
            kb_ids=(self.kb_id,),
            doc_ids=(self.doc_id,),
        )
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="普通员工出差时可以乘坐的交通工具有哪些",
            history_messages=(),
            carryover_sources=(),
            route_turn_candidates=(RouteTurnCandidate(
                candidate_key="t1",
                user_question=state.root_query,
                assistant_answer="D级适用普通员工、专员",
                raw_sources=(source,),
                assistant_turn_id=self.turn_id,
            ),),
        )

        applied = apply_active_task_context(
            context=context,
            question="普通员工可以乘坐头等舱吗",
            resolved_task=resolved,
        )

        self.assertIs(applied, context)
        self.assertFalse(applied.is_followup)

    def test_different_or_multiple_ordinals_do_not_inherit_active_document(self):
        state = build_active_task_state(
            root_query="我要看第四个",
            answer_shape="overview",
            sources=[_source(
                kb_id=self.kb_id,
                doc_id=self.doc_id,
                chunk_id=self.chunk_id,
            )],
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
        )
        source = _source(
            kb_id=self.kb_id,
            doc_id=self.doc_id,
            chunk_id=self.chunk_id,
        )
        resolved = ResolvedActiveTask(
            state=state,
            sources=(source,),
            kb_ids=(self.kb_id,),
            doc_ids=(self.doc_id,),
        )
        for question in (
            "第五个里面还有什么",
            "比较第四个和第五个",
            "换个问题，第四个选项是什么意思",
        ):
            with self.subTest(question=question):
                context = ConversationContext(
                    is_followup=False,
                    followup_reason="standalone_question",
                    standalone_query=question,
                    history_messages=(),
                    carryover_sources=(),
                    route_turn_candidates=(RouteTurnCandidate(
                        candidate_key="t1",
                        user_question=state.root_query,
                        assistant_answer="所选文档正文",
                        raw_sources=(source,),
                        assistant_turn_id=self.turn_id,
                    ),),
                )

                applied = apply_active_task_context(
                    context=context,
                    question=question,
                    resolved_task=resolved,
                )

                self.assertIsNone(applied.active_task)
                self.assertEqual(applied.carryover_sources, ())


if __name__ == "__main__":
    unittest.main()
