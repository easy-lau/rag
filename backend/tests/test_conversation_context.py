import unittest
import uuid
from datetime import datetime, timedelta, timezone

from core.conversation_context import (
    UNRESOLVED_REFERENCE_MESSAGE,
    build_standalone_query,
    detect_followup,
    prepare_conversation_context,
    resolve_routed_conversation_context,
    route_context_payloads,
)
from core.query_constraints import extract_query_constraints
from core.query_route_contract import parse_rag_route_decision
from models.db_models import Document, DocumentChunk, Message


class _ScalarResult:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return self

    def all(self):
        return self._values


class _RowsResult:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class _FakeDB:
    def __init__(self, history, source_rows):
        self._results = [_ScalarResult(history), _RowsResult(source_rows)]
        self.execute_count = 0
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        result = self._results[self.execute_count]
        self.execute_count += 1
        return result


class ConversationContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_followup_can_use_current_query_and_reuse_evidence(self) -> None:
        conversation_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="普通员工的出差标准是什么",
            created_at=now - timedelta(seconds=2),
        )
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="普通员工属于 D 级，以下是出差标准。",
            sources=[{"id": str(chunk_id), "evidence_role": "direct"}],
            created_at=now - timedelta(seconds=1),
        )
        document = Document(
            id=document_id,
            kb_id=kb_id,
            filename="出差管理制度.md",
            status="ready",
            is_active=True,
        )
        chunk = DocumentChunk(
            id=chunk_id,
            doc_id=document_id,
            kb_id=kb_id,
            content="D 级员工住宿标准按城市类别执行。",
            chunk_index=2,
        )
        db = _FakeDB(
            [assistant_message, user_message],
            [(chunk, document)],
        )

        prepared = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="普通员工出差的住宿标准",
            kb_ids=[kb_id],
        )
        self.assertFalse(prepared.is_followup)  # legacy syntax heuristic
        self.assertEqual(route_context_payloads(prepared)[0]["candidate_key"], "t1")

        resolved = await resolve_routed_conversation_context(
            db,
            context=prepared,
            question="普通员工出差的住宿标准",
            kb_ids=[kb_id],
            route_decision={
                "readiness": "ready",
                "relation": "followup",
                "query_resolution": {
                    "mode": "current",
                    "context_turn_keys": ["t1"],
                },
            },
        )

        self.assertTrue(resolved.is_followup)
        self.assertEqual(resolved.relation, "followup")
        self.assertEqual(resolved.query_resolution_mode, "current")
        self.assertEqual(resolved.standalone_query, "普通员工出差的住宿标准")
        self.assertEqual(resolved.context_turn_keys, ("t1",))
        self.assertEqual(len(resolved.carryover_sources), 1)
        self.assertEqual(resolved.carryover_sources[0]["id"], chunk_id)
        self.assertEqual(
            [item["role"] for item in resolved.history_messages],
            ["user", "assistant"],
        )

    async def test_semantic_elliptical_followup_contextualizes_query(self) -> None:
        conversation_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="普通员工的出差标准是什么",
            created_at=now - timedelta(seconds=2),
        )
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="普通员工属于 D 级。",
            sources=[],
            created_at=now - timedelta(seconds=1),
        )
        db = _FakeDB([assistant_message, user_message], [])
        prepared = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="住宿呢",
            kb_ids=[],
        )
        route_decision = parse_rag_route_decision(
            {
                "schema_version": "rag_route_decision.v1",
                "readiness": "ready",
                "intent_code": "knowledge_qa",
                "relation": "followup",
                "evidence_scope": "enterprise_kb",
                "query_resolution": {
                    "mode": "contextualize",
                    "context_turn_keys": ["t1"],
                },
                "requirements": [
                    {
                        "role": "answer",
                        "origin": "user_text",
                        "description": "取得住宿标准",
                    }
                ],
                "clarification": {"question": "", "unresolved": []},
                "confidence": 0.95,
                "rationale": "省略宾语的语义追问",
            },
            allowed_intent_codes=["knowledge_qa"],
            available_turn_keys=["t1"],
        )
        resolved = await resolve_routed_conversation_context(
            db,
            context=prepared,
            question="住宿呢",
            kb_ids=[],
            route_decision=route_decision,
        )

        self.assertTrue(resolved.is_followup)
        self.assertEqual(resolved.query_resolution_mode, "contextualize")
        self.assertEqual(resolved.context_turn_keys, ("t1",))
        self.assertIn("住宿呢", resolved.standalone_query)
        self.assertIn("普通员工", resolved.standalone_query)

    def test_explicit_reference_is_followup_but_new_topic_is_not(self) -> None:
        self.assertTrue(
            detect_followup(
                "这些配置会对程序有什么影响",
                has_previous_turn=True,
            )[0]
        )
        self.assertFalse(
            detect_followup(
                "换个问题，这些配置是谁写的",
                has_previous_turn=True,
            )[0]
        )
        for standalone in ("这个月有哪些节日", "这个新项目怎么部署"):
            with self.subTest(standalone=standalone):
                self.assertEqual(
                    detect_followup(
                        standalone,
                        has_previous_turn=True,
                        previous_user_question="云枢默认密码如何设置",
                    ),
                    (False, "standalone_question"),
                )
        for question in ("那云枢7呢", "那8.6呢", "那这个版本呢", "云枢8.6呢"):
            with self.subTest(question=question):
                self.assertTrue(
                    detect_followup(question, has_previous_turn=True)[0]
                )
        self.assertFalse(
            detect_followup(
                "这些配置会对程序有什么影响",
                has_previous_turn=False,
            )[0]
        )

    def test_complete_conditional_question_does_not_inherit_previous_topic(self) -> None:
        questions = (
            "如果 Redis 宕机怎么办",
            "如果 Redis 宕机会怎么样",
            "如果 Redis 宕机怎么样",
            "如果业务流量翻倍呢",
            "如果 Redis 不可用如何",
        )
        for question in questions:
            for has_previous_turn in (False, True):
                with self.subTest(
                    question=question,
                    has_previous_turn=has_previous_turn,
                ):
                    is_followup, reason = detect_followup(
                        question,
                        has_previous_turn=has_previous_turn,
                    )
                    self.assertFalse(is_followup)
                    self.assertNotIn("unresolved_reference", reason)

        for followup in ("如果是8.6呢", "如果是 Redis 呢", "改成 Redis 呢", "换成 Redis 可以吗"):
            with self.subTest(followup=followup):
                self.assertTrue(
                    detect_followup(followup, has_previous_turn=True)[0]
                )
        unresolved = detect_followup("如果是8.6呢", has_previous_turn=False)
        self.assertFalse(unresolved[0])
        self.assertTrue(unresolved[1].startswith("unresolved_reference:"))

        scoped_unresolved = detect_followup("云枢8.6呢", has_previous_turn=False)
        self.assertFalse(scoped_unresolved[0])
        self.assertTrue(scoped_unresolved[1].startswith("unresolved_reference:"))

    def test_missing_action_object_inherits_only_when_history_exists(self) -> None:
        followups = (
            "云枢中如何配置",
            "云枢 8.6 中如何配置",
            "CloudPivot 8.6 中如何配置",
            "在 云枢 中怎么设置",
            "云枢里如何设置呢",
            "云枢中具体怎么配置",
            "云枢中该如何配置",
            "云枢里面怎么配置",
            "云枢中怎么配",
            "那在云枢里怎么设置",
            "那要怎么配置",
            "如何处理",
            "云枢怎么配置",
        )
        for question in followups:
            with self.subTest(question=question):
                self.assertEqual(
                    detect_followup(question, has_previous_turn=True),
                    (True, "missing_action_object"),
                )
                without_history = detect_followup(
                    question,
                    has_previous_turn=False,
                )
                self.assertFalse(without_history[0])
                self.assertEqual(
                    without_history[1],
                    "unresolved_reference:missing_action_object",
                )

        complete_questions = (
            "登录用户名枚举在云枢中如何配置",
            "默认密码在云枢中怎么设置",
            "持久化在 Redis 中如何配置",
            "云枢默认密码怎么配置",
            "云枢的默认密码怎么配置",
            "云枢登录用户名枚举怎么配置",
            "云枢8.6默认密码怎么配置",
            "CloudPivot defaultPwd 怎么配置",
            "Redis的持久化怎么配置",
            "PostgreSQL高可用怎么配置",
        )
        for question in complete_questions:
            with self.subTest(question=question):
                for has_previous_turn in (False, True):
                    with self.subTest(has_previous_turn=has_previous_turn):
                        is_followup, reason = detect_followup(
                            question,
                            has_previous_turn=has_previous_turn,
                            previous_user_question="登录用户名枚举是什么",
                        )
                        self.assertFalse(is_followup)
                        self.assertEqual(
                            reason,
                            (
                                "standalone_question"
                                if has_previous_turn
                                else "no_previous_turn"
                            ),
                        )

    def test_complete_or_new_topic_action_question_does_not_inherit(self) -> None:
        independent_questions = (
            "云枢中如何配置登录用户名枚举",
            "云枢中如何配置默认密码",
            "如何配置 PostgreSQL 高可用",
            "Redis 中如何配置持久化",
            "云枢怎么部署集群",
            "如果 Redis 宕机怎么办",
            "换个问题，云枢中如何配置",
        )
        for question in independent_questions:
            with self.subTest(question=question):
                self.assertFalse(
                    detect_followup(question, has_previous_turn=True)[0]
                )

        self.assertEqual(
            detect_followup(
                "换个问题，云枢中如何配置",
                has_previous_turn=True,
                previous_user_question="登录用户名枚举是什么",
            ),
            (
                False,
                "unresolved_reference:missing_action_object:explicit_new_topic",
            ),
        )

    def test_missing_object_with_different_explicit_scope_starts_new_topic(self) -> None:
        previous = "云枢中如何配置登录用户名枚举"
        for question in ("Redis 应该如何设置", "Python 怎么配置"):
            with self.subTest(question=question):
                self.assertEqual(
                    detect_followup(
                        question,
                        has_previous_turn=True,
                        previous_user_question=previous,
                    ),
                    (
                        False,
                        "unresolved_reference:missing_action_object:explicit_new_scope",
                    ),
                )

        self.assertEqual(
            detect_followup(
                "Redis 应该如何设置",
                has_previous_turn=True,
                previous_user_question="Redis 持久化是什么",
            ),
            (True, "missing_action_object"),
        )
        self.assertEqual(
            detect_followup(
                "云枢中如何配置",
                has_previous_turn=True,
                previous_user_question="登录用户名枚举是什么",
            ),
            (True, "missing_action_object"),
        )
        self.assertEqual(
            detect_followup(
                "Redis 应该如何设置",
                has_previous_turn=True,
                previous_user_question="登录用户名枚举是什么",
            ),
            (
                False,
                "unresolved_reference:missing_action_object:explicit_new_scope",
            ),
        )

    def test_missing_object_same_latin_scope_without_spaces_inherits_topic(self) -> None:
        self.assertEqual(
            detect_followup(
                "Redis应该如何设置",
                has_previous_turn=True,
                previous_user_question="Redis持久化是什么",
            ),
            (True, "missing_action_object"),
        )
        self.assertEqual(
            build_standalone_query(
                "Redis应该如何设置",
                previous_user_question="Redis持久化是什么",
                previous_assistant_answer="",
                followup_reason="missing_action_object",
            ),
            "Redis应该如何设置持久化",
        )

    def test_missing_object_does_not_inherit_unrelated_previous_topic(self) -> None:
        for previous in ("你好", "什么是量子纠缠", "今天天气怎么样"):
            with self.subTest(previous=previous):
                self.assertEqual(
                    detect_followup(
                        "云枢中如何配置",
                        has_previous_turn=True,
                        previous_user_question=previous,
                    ),
                    (False, "unresolved_reference:missing_action_object"),
                )

        for previous in (
            "登录用户名枚举是什么",
            "默认密码是什么",
        ):
            with self.subTest(previous=previous):
                self.assertEqual(
                    detect_followup(
                        "云枢中如何配置",
                        has_previous_turn=True,
                        previous_user_question=previous,
                    ),
                    (True, "missing_action_object"),
                )
        self.assertEqual(
            detect_followup(
                "Redis 中如何配置",
                has_previous_turn=True,
                previous_user_question="Redis 持久化有什么作用",
            ),
            (True, "missing_action_object"),
        )
        for previous in (
            "SSO 是什么",
            "OAuth 是什么",
            "defaultPwd 是什么",
            "error_reply_same 是什么",
        ):
            with self.subTest(technical_topic=previous):
                is_followup, reason = detect_followup(
                    "云枢中如何配置",
                    has_previous_turn=True,
                    previous_user_question=previous,
                )
                self.assertTrue(is_followup)
                self.assertEqual(reason, "missing_action_object")

    def test_reference_with_concrete_postfix_object_is_standalone(self) -> None:
        complete_questions = (
            "这个问题怎么解决：Redis 连接超时",
            "解释这个配置：spring.datasource.url",
            "该参数是什么：error_reply_same",
        )
        for question in complete_questions:
            for has_previous_turn in (False, True):
                with self.subTest(
                    question=question,
                    has_previous_turn=has_previous_turn,
                ):
                    self.assertEqual(
                        detect_followup(
                            question,
                            has_previous_turn=has_previous_turn,
                            previous_user_question="云枢默认密码如何配置",
                        ),
                        (False, "explicit_postfix_object"),
                    )

        self.assertTrue(
            detect_followup(
                "这个配置：怎么改",
                has_previous_turn=True,
                previous_user_question="云枢默认密码如何配置",
            )[0]
        )

    def test_standalone_query_uses_topic_and_technical_terms(self) -> None:
        rewritten = build_standalone_query(
            "这些配置有什么影响",
            previous_user_question="云枢 8.6 如何防止登录用户名枚举",
            previous_assistant_answer="旧资料提到了 error_reply_same1 和 defaultPwd。",
            carryover_sources=[],
        )

        self.assertIn("云枢 8.6", rewritten)
        self.assertIn("error_reply_same1", rewritten)
        self.assertIn("defaultPwd", rewritten)
        self.assertIn("这些配置有什么影响", rewritten)

    def test_current_explicit_version_overrides_previous_version(self) -> None:
        rewritten = build_standalone_query(
            "那云枢7呢",
            previous_user_question="云枢8.6 如何防止登录用户名枚举",
            previous_assistant_answer="旧资料提到了 error_reply_same1。",
        )

        constraints = extract_query_constraints(rewritten)

        self.assertEqual(constraints.product, "云枢")
        self.assertEqual(constraints.version, "7")
        self.assertTrue(constraints.explicit_version)
        self.assertNotIn("8.6", rewritten)
        self.assertNotIn("原始追问", rewritten)

    def test_missing_action_object_builds_clean_topic_query(self) -> None:
        rewritten = build_standalone_query(
            "云枢中如何配置",
            previous_user_question="登录用户名枚举 是什么",
            previous_assistant_answer="可通过 error_reply_same 统一失败提示。",
            followup_reason="missing_action_object",
        )

        self.assertEqual(rewritten, "云枢中如何配置登录用户名枚举")
        self.assertNotIn("上一轮", rewritten)
        self.assertNotIn("继承", rewritten)

    def test_missing_action_object_extracts_prefix_question_topic_cleanly(self) -> None:
        cases = (
            ("什么是登录用户名枚举", "云枢中如何配置登录用户名枚举"),
            ("请问什么是登录用户名枚举？", "云枢中如何配置登录用户名枚举"),
            ("登录用户名枚举指的是什么", "云枢中如何配置登录用户名枚举"),
            ("解释一下登录用户名枚举", "云枢中如何配置登录用户名枚举"),
            ("登录用户名枚举是啥", "云枢中如何配置登录用户名枚举"),
            ("为什么登录会失败", "云枢中如何配置登录会失败"),
        )
        for previous, expected in cases:
            with self.subTest(previous=previous):
                rewritten = build_standalone_query(
                    "云枢中如何配置",
                    previous_user_question=previous,
                    previous_assistant_answer="",
                    followup_reason="missing_action_object",
                )
                self.assertEqual(rewritten, expected)

    def test_missing_action_object_avoids_duplicate_product_name(self) -> None:
        cases = (
            (
                "那在云枢里怎么设置",
                "云枢登录用户名枚举是什么",
                "在云枢里怎么设置登录用户名枚举",
            ),
            (
                "云枢中如何配置",
                "云枢中登录用户名枚举是什么",
                "云枢中如何配置登录用户名枚举",
            ),
            (
                "那在云枢里怎么设置",
                "在云枢中登录用户名枚举是什么",
                "在云枢里怎么设置登录用户名枚举",
            ),
            (
                "云枢里如何设置呢",
                "登录用户名枚举是什么",
                "云枢里如何设置登录用户名枚举",
            ),
            (
                "在云枢中该怎么配置呢",
                "登录用户名枚举是什么",
                "在云枢中该怎么配置登录用户名枚举",
            ),
        )
        for current, previous, expected in cases:
            with self.subTest(current=current, previous=previous):
                rewritten = build_standalone_query(
                    current,
                    previous_user_question=previous,
                    previous_assistant_answer="可通过 error_reply_same 统一失败提示。",
                    followup_reason="missing_action_object",
                )
                self.assertEqual(rewritten, expected)

    def test_regular_followup_query_contains_no_diagnostic_meta_language(self) -> None:
        rewritten = build_standalone_query(
            "这些配置有什么影响",
            previous_user_question="云枢 8.6 如何防止登录用户名枚举",
            previous_assistant_answer="旧资料提到了 error_reply_same1。",
        )

        self.assertNotIn("当前追问", rewritten)
        self.assertNotIn("用于消解指代", rewritten)
        self.assertIn("云枢 8.6", rewritten)
        self.assertIn("error_reply_same1", rewritten)

    def test_version_only_followup_inherits_product_but_overrides_version(self) -> None:
        rewritten = build_standalone_query(
            "那8.6呢",
            previous_user_question="云枢7 如何防止登录用户名枚举",
            previous_assistant_answer="旧资料提到了 error_reply_same1。",
        )

        constraints = extract_query_constraints(rewritten)

        self.assertEqual(constraints.product, "云枢")
        self.assertEqual(constraints.version, "8.6")
        self.assertNotIn("云枢7", rewritten)
        self.assertNotIn("原始追问", rewritten)

    async def test_prepare_context_marks_reference_without_history_as_unresolved(self) -> None:
        db = _FakeDB([], [])

        context = await prepare_conversation_context(
            db,
            conversation_id=uuid.uuid4(),
            question="这些配置会对程序有什么影响",
            kb_ids=[uuid.uuid4()],
        )

        self.assertFalse(context.is_followup)
        self.assertTrue(context.unresolved_reference)
        self.assertTrue(context.followup_reason.startswith("unresolved_reference:"))
        self.assertEqual(db.execute_count, 1)

    async def test_prepare_context_current_version_overrides_history(self) -> None:
        conversation_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="云枢8.6 如何防止登录用户名枚举",
            created_at=now - timedelta(seconds=2),
        )
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="当前资料不足。",
            sources=[],
            created_at=now - timedelta(seconds=1),
        )
        db = _FakeDB([assistant_message, user_message], [])

        context = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="那云枢7呢",
            kb_ids=[uuid.uuid4()],
        )
        constraints = extract_query_constraints(context.standalone_query)

        self.assertTrue(context.is_followup)
        self.assertFalse(context.unresolved_reference)
        self.assertEqual(constraints.version, "7")

    async def test_prepare_context_resolves_missing_action_object_and_reuses_source(self) -> None:
        conversation_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="登录用户名枚举 是什么",
            created_at=now - timedelta(seconds=2),
        )
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="该风险可通过统一登录失败提示降低。",
            sources=[{
                "id": str(chunk_id),
                "evidence_role": "related",
                "answer_support": 0.65,
            }],
            created_at=now - timedelta(seconds=1),
        )
        document = Document(
            id=document_id,
            kb_id=kb_id,
            filename="云枢配置.md",
            status="ready",
            is_active=True,
        )
        chunk = DocumentChunk(
            id=chunk_id,
            doc_id=document_id,
            kb_id=kb_id,
            content="cloudpivot.organization.login.error_reply_same: true",
            chunk_index=3,
        )
        db = _FakeDB(
            [assistant_message, user_message],
            [(chunk, document)],
        )

        context = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="云枢中如何配置",
            kb_ids=[kb_id],
        )

        self.assertTrue(context.is_followup)
        self.assertEqual(context.followup_reason, "missing_action_object")
        self.assertFalse(context.unresolved_reference)
        self.assertIn("登录用户名枚举", context.standalone_query)
        self.assertIn("云枢中如何配置", context.standalone_query)
        self.assertEqual(len(context.carryover_sources), 1)
        self.assertEqual(context.carryover_sources[0]["id"], chunk_id)

    async def test_prepare_context_without_history_marks_missing_object_unresolved(self) -> None:
        db = _FakeDB([], [])

        context = await prepare_conversation_context(
            db,
            conversation_id=uuid.uuid4(),
            question="云枢中如何配置",
            kb_ids=[uuid.uuid4()],
        )

        self.assertFalse(context.is_followup)
        self.assertTrue(context.unresolved_reference)
        self.assertEqual(
            context.followup_reason,
            "unresolved_reference:missing_action_object",
        )
        self.assertEqual(db.execute_count, 1)

    async def test_clarification_answer_fills_previous_missing_object(self) -> None:
        conversation_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="云枢中如何配置",
            created_at=now - timedelta(seconds=2),
        )
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=UNRESOLVED_REFERENCE_MESSAGE,
            sources=[],
            created_at=now - timedelta(seconds=1),
        )
        db = _FakeDB([assistant_message, user_message], [])

        context = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="登录用户名枚举",
            kb_ids=[uuid.uuid4()],
        )

        self.assertTrue(context.is_followup)
        self.assertFalse(context.unresolved_reference)
        self.assertEqual(
            context.followup_reason,
            "clarification_answer:missing_action_object",
        )
        self.assertEqual(context.standalone_query, "云枢中如何配置登录用户名枚举")
        self.assertEqual(db.execute_count, 1)

    async def test_clarification_does_not_swallow_a_new_complete_question(self) -> None:
        conversation_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="云枢中如何配置",
            created_at=now - timedelta(seconds=2),
        )
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content=UNRESOLVED_REFERENCE_MESSAGE,
            sources=[],
            created_at=now - timedelta(seconds=1),
        )
        db = _FakeDB([assistant_message, user_message], [])

        context = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="Redis 为什么连接超时？",
            kb_ids=[uuid.uuid4()],
        )

        self.assertFalse(context.is_followup)
        self.assertEqual(context.standalone_query, "Redis 为什么连接超时？")
        self.assertEqual(db.execute_count, 1)

    async def test_clarification_does_not_treat_cancellation_as_slot_answer(self) -> None:
        conversation_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        for answer in ("算了", "不了，谢谢", "暂时不需要", "换一个方向"):
            with self.subTest(answer=answer):
                user_message = Message(
                    id=uuid.uuid4(),
                    conversation_id=conversation_id,
                    role="user",
                    content="云枢中如何配置",
                    created_at=now - timedelta(seconds=2),
                )
                assistant_message = Message(
                    id=uuid.uuid4(),
                    conversation_id=conversation_id,
                    role="assistant",
                    content=UNRESOLVED_REFERENCE_MESSAGE,
                    sources=[],
                    created_at=now - timedelta(seconds=1),
                )
                db = _FakeDB([assistant_message, user_message], [])

                context = await prepare_conversation_context(
                    db,
                    conversation_id=conversation_id,
                    question=answer,
                    kb_ids=[uuid.uuid4()],
                )

                self.assertFalse(context.is_followup)
                self.assertFalse(context.unresolved_reference)
                self.assertEqual(context.standalone_query, answer)
                self.assertNotIn("云枢中如何配置", context.standalone_query)

    async def test_prepare_context_does_not_reuse_sources_after_explicit_scope_change(self) -> None:
        conversation_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="云枢中如何配置登录用户名枚举",
            created_at=now - timedelta(seconds=2),
        )
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="请设置 error_reply_same。",
            sources=[{"id": str(uuid.uuid4()), "evidence_role": "direct"}],
            created_at=now - timedelta(seconds=1),
        )
        db = _FakeDB([assistant_message, user_message], [])

        context = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="Redis 应该如何设置",
            kb_ids=[uuid.uuid4()],
        )

        self.assertFalse(context.is_followup)
        self.assertTrue(context.unresolved_reference)
        self.assertEqual(
            context.followup_reason,
            "unresolved_reference:missing_action_object:explicit_new_scope",
        )
        self.assertEqual(context.standalone_query, "Redis 应该如何设置")
        self.assertEqual(context.carryover_sources, ())
        self.assertEqual(db.execute_count, 1)

    async def test_prepare_context_anchors_on_latest_unanswered_user_message(self) -> None:
        conversation_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        older_user = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="云枢默认密码是什么",
            created_at=now - timedelta(seconds=3),
        )
        older_assistant = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="默认密码与 defaultPwd 有关。",
            sources=[{"id": str(uuid.uuid4()), "evidence_role": "direct"}],
            created_at=now - timedelta(seconds=2),
        )
        latest_unanswered_user = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="登录用户名枚举是什么",
            created_at=now - timedelta(seconds=1),
        )
        db = _FakeDB(
            [latest_unanswered_user, older_assistant, older_user],
            [],
        )

        context = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="云枢中如何配置",
            kb_ids=[uuid.uuid4()],
        )

        self.assertTrue(context.is_followup)
        self.assertEqual(context.followup_reason, "missing_action_object")
        self.assertEqual(
            context.previous_user_question,
            "登录用户名枚举是什么",
        )
        self.assertEqual(
            context.standalone_query,
            "云枢中如何配置登录用户名枚举",
        )
        self.assertEqual(context.carryover_sources, ())
        self.assertEqual(db.execute_count, 1)

    async def test_prepare_context_reloads_previous_source_in_current_kb_scope(self) -> None:
        conversation_id = uuid.uuid4()
        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        chunk_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="user",
            content="解决登录用户名枚举要配置什么，我是云枢 8.6",
            created_at=now - timedelta(seconds=2),
        )
        assistant_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            role="assistant",
            content="云枢 6/7 资料提到了 error_reply_same 和 error_reply_same1。",
            sources=[
                {
                    "id": str(chunk_id),
                    "evidence_role": "related",
                    "topic_relevance": 0.95,
                    # Previous-turn support is not reused as a current score.
                    # Even zero-support displayed evidence can be what “这些配置” refers to.
                    "answer_support": 0.0,
                }
            ],
            created_at=now - timedelta(seconds=1),
        )
        document = Document(
            id=document_id,
            kb_id=kb_id,
            filename="云枢7配置.md",
            status="ready",
            is_active=True,
            tags=["登录安全"],
        )
        chunk = DocumentChunk(
            id=chunk_id,
            doc_id=document_id,
            kb_id=kb_id,
            content="error_reply_same1: true",
            chunk_index=3,
            metadata_={"section": "login"},
        )
        db = _FakeDB(
            # Query is descending; the context loader restores chronological order.
            [assistant_message, user_message],
            [(chunk, document)],
        )

        context = await prepare_conversation_context(
            db,
            conversation_id=conversation_id,
            question="这些配置会对程序有什么影响",
            kb_ids=[kb_id],
        )

        self.assertTrue(context.is_followup)
        self.assertEqual(len(context.carryover_sources), 1)
        self.assertEqual(context.carryover_sources[0]["id"], chunk_id)
        self.assertEqual(
            context.carryover_sources[0]["candidate_origin"],
            "carryover_previous_turn",
        )
        self.assertIn("云枢 8.6", context.standalone_query)
        self.assertIn("error_reply_same1", context.standalone_query)
        self.assertEqual([item["role"] for item in context.history_messages], ["user", "assistant"])
        self.assertEqual(db.execute_count, 2)
        source_sql = str(db.statements[1])
        self.assertIn(
            "documents.kb_id = document_chunks.kb_id",
            source_sql,
        )
        self.assertIn("documents.status =", source_sql)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
