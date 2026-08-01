from dataclasses import replace
import unittest

from core.query_route_compiler import (
    RouteCategoryPolicy,
    RouteCompilerConfig,
    TaskContractCompilationError,
    TaskContractDispatchError,
    compile_rag_task_contract,
    is_rag_task_contract_dispatchable,
    rag_task_contract_gate_reason,
    require_rag_task_contract_dispatchable,
    safe_rag_task_contract_summary,
)
from core.query_route_contract import parse_rag_route_decision


CATEGORIES = {
    "knowledge_qa": RouteCategoryPolicy(
        code="knowledge_qa",
        name="知识库问答",
        action="retrieve",
    ),
    "general_chat": RouteCategoryPolicy(
        code="general_chat",
        name="通用交流",
        action="chat",
    ),
    "writing": RouteCategoryPolicy(
        code="writing",
        name="写作润色",
        action="writing",
    ),
    "system_help": RouteCategoryPolicy(
        code="system_help",
        name="系统使用帮助",
        action="system_help",
    ),
}


def _route(
    *,
    intent_code="knowledge_qa",
    evidence_scope="enterprise_kb",
    confidence=0.94,
    relation="new",
    mode="current",
    turn_keys=(),
    requirements=None,
    readiness="ready",
    clarification=None,
):
    if requirements is None:
        requirements = [
            {
                "role": "answer",
                "origin": "user_text",
                "description": "回答用户当前问题",
            }
        ]
    if clarification is None:
        clarification = {"question": "", "unresolved": []}
    payload = {
        "schema_version": "rag_route_decision.v1",
        "readiness": readiness,
        "intent_code": intent_code,
        "relation": relation,
        "evidence_scope": evidence_scope,
        "query_resolution": {
            "mode": mode,
            "context_turn_keys": list(turn_keys),
        },
        "requirements": requirements,
        "clarification": clarification,
        "confidence": confidence,
        "rationale": "测试路由",
    }
    return parse_rag_route_decision(
        payload,
        allowed_intent_codes=list(CATEGORIES),
        available_turn_keys=["t1", "t2", "t3"],
    )


def _compile(
    route,
    *,
    selected_kb_count=1,
    config=None,
    question="普通员工的出差标准是什么？",
    **guards,
):
    return compile_rag_task_contract(
        route,
        CATEGORIES[route.intent_code],
        config or RouteCompilerConfig(),
        question=question,
        selected_kb_count=selected_kb_count,
        available_turn_keys=["t1", "t2", "t3"],
        source="llm",
        **guards,
    )


class RagTaskCompilerPolicyTests(unittest.TestCase):
    def test_compiler_mapping_table_is_deterministic(self) -> None:
        cases = (
            (
                _route(),
                {},
                ("grounded_qa", "required", True, "classified_retrieval"),
            ),
            (
                _route(intent_code="general_chat", evidence_scope="general_world"),
                {},
                ("general_chat", "skip", False, "classified_general_chat"),
            ),
            (
                _route(intent_code="general_chat", evidence_scope="enterprise_kb"),
                {},
                ("grounded_qa", "required", True, "knowledge_scope_guard"),
            ),
            (
                _route(intent_code="writing", evidence_scope="current_input"),
                {},
                ("writing", "skip", False, "classified_writing"),
            ),
            (
                _route(intent_code="writing", evidence_scope="mixed"),
                {},
                ("writing", "required", True, "knowledge_dependent_writing"),
            ),
            (
                _route(intent_code="system_help", evidence_scope="platform_self"),
                {},
                ("grounded_qa", "required", True, "platform_help_scope_guard"),
            ),
            (
                _route(intent_code="system_help", evidence_scope="platform_self"),
                {"explicit_platform_help": True},
                ("platform_help", "skip", False, "explicit_platform_help"),
            ),
        )

        for route, guards, expected in cases:
            with self.subTest(intent=route.intent_code, scope=route.evidence_scope, guards=guards):
                contract = _compile(route, **guards)
                self.assertEqual(
                    (
                        contract.response_mode,
                        contract.retrieval_policy,
                        contract.need_retrieval,
                        contract.decision_reason,
                    ),
                    expected,
                )
                self.assertTrue(contract.dispatch_authorized)
                self.assertTrue(is_rag_task_contract_dispatchable(contract))

    def test_high_certainty_local_guards_can_correct_category(self) -> None:
        greeting = _compile(
            _route(evidence_scope="general_world"),
            explicit_greeting=True,
        )
        writing = _compile(
            _route(intent_code="general_chat", evidence_scope="current_input"),
            inline_writing=True,
        )
        knowledge = _compile(
            _route(intent_code="general_chat", evidence_scope="general_world"),
            requires_knowledge=True,
        )
        knowledge_inline_writing = _compile(
            _route(intent_code="writing", evidence_scope="current_input"),
            inline_writing=True,
            knowledge_writing=True,
        )

        self.assertEqual((greeting.response_mode, greeting.retrieval_policy), ("general_chat", "skip"))
        self.assertEqual((writing.response_mode, writing.retrieval_policy), ("writing", "skip"))
        self.assertEqual((knowledge.response_mode, knowledge.retrieval_policy), ("grounded_qa", "required"))
        self.assertEqual(
            (
                knowledge_inline_writing.response_mode,
                knowledge_inline_writing.retrieval_policy,
                knowledge_inline_writing.decision_reason,
            ),
            ("writing", "required", "knowledge_dependent_writing"),
        )

    def test_general_chat_disabled_overrides_direct_local_guards(self) -> None:
        contract = _compile(
            _route(intent_code="general_chat", evidence_scope="general_world"),
            config=RouteCompilerConfig(allow_general_chat=False),
            explicit_greeting=True,
        )
        self.assertEqual(contract.response_mode, "grounded_qa")
        self.assertEqual(contract.retrieval_policy, "required")
        self.assertEqual(contract.decision_reason, "general_chat_disabled")

    def test_low_confidence_only_moves_skip_toward_safe_retrieval(self) -> None:
        route = _route(
            intent_code="general_chat",
            evidence_scope="general_world",
            confidence=0.4,
        )
        contract = _compile(route, selected_kb_count=1)

        self.assertEqual(contract.response_mode, "grounded_qa")
        self.assertEqual(contract.retrieval_policy, "required")
        self.assertTrue(contract.need_retrieval)
        self.assertEqual(contract.decision_reason, "low_confidence_safe_retrieval")

        already_required = _compile(_route(confidence=0.4), selected_kb_count=1)
        self.assertEqual(already_required.decision_reason, "classified_retrieval")
        self.assertEqual(already_required.retrieval_policy, "required")

    def test_low_confidence_does_not_override_deterministic_local_direct_guards(self) -> None:
        cases = (
            (
                _route(
                    intent_code="general_chat",
                    evidence_scope="general_world",
                    confidence=0.0,
                ),
                {"explicit_greeting": True},
                "exact_greeting",
                "general_chat",
            ),
            (
                _route(
                    intent_code="system_help",
                    evidence_scope="platform_self",
                    confidence=0.0,
                ),
                {"explicit_platform_help": True},
                "explicit_platform_help",
                "platform_help",
            ),
            (
                _route(
                    intent_code="general_chat",
                    evidence_scope="current_input",
                    confidence=0.0,
                ),
                {"inline_writing": True},
                "inline_writing_content",
                "writing",
            ),
        )

        for route, guards, reason, response_mode in cases:
            with self.subTest(reason=reason):
                contract = _compile(route, selected_kb_count=0, **guards)
                self.assertEqual(contract.response_mode, response_mode)
                self.assertEqual(contract.retrieval_policy, "skip")
                self.assertFalse(contract.need_retrieval)
                self.assertEqual(contract.decision_reason, reason)
                self.assertTrue(contract.dispatch_authorized)

    def test_required_retrieval_without_kb_becomes_non_executable_clarification(self) -> None:
        contract = _compile(_route(), selected_kb_count=0)

        self.assertEqual(contract.readiness, "needs_clarification")
        self.assertFalse(contract.dispatch_authorized)
        self.assertTrue(contract.need_retrieval)
        self.assertEqual(contract.decision_reason, "knowledge_base_required")
        self.assertEqual(contract.clarification.unresolved[0].role, "knowledge_base")
        self.assertEqual(contract.clarification.unresolved[0].reason, "missing")
        self.assertFalse(is_rag_task_contract_dispatchable(contract))

    def test_model_clarification_remains_terminal_and_is_not_rewritten(self) -> None:
        clarification = {
            "question": "请补充要查询的制度主题。",
            "unresolved": [
                {"role": "subject", "reason": "missing", "candidate_keys": []}
            ],
        }
        route = _route(
            readiness="needs_clarification",
            clarification=clarification,
            requirements=[],
        )
        contract = _compile(route, selected_kb_count=1)

        self.assertEqual(contract.readiness, "needs_clarification")
        self.assertEqual(contract.clarification.to_dict(), clarification)
        self.assertEqual(contract.decision_reason, "semantic_clarification")
        self.assertFalse(contract.dispatch_authorized)


class RagTaskCompilerContractTests(unittest.TestCase):
    def test_implicit_mapping_gets_local_bridge_when_route_model_omits_it(self) -> None:
        contract = _compile(_route(requirements=[{
            "role": "answer",
            "origin": "user_text",
            "description": "普通员工的出差标准是什么？",
        }]))

        self.assertEqual(
            [item.role for item in contract.requirements],
            ["answer", "bridge"],
        )
        self.assertEqual(contract.requirements[1].source, "inferred")
        removed_bridge = replace(
            contract,
            requirements=(contract.requirements[0],),
        )
        self.assertEqual(
            rag_task_contract_gate_reason(removed_bridge),
            "implicit_mapping_missing_bridge",
        )

    def test_original_question_restores_bridge_after_model_summary_drops_qualifier(self) -> None:
        contract = _compile(_route(requirements=[{
            "role": "answer",
            "origin": "user_text",
            "description": "查询出差标准",
        }]))

        self.assertEqual(
            [item.role for item in contract.requirements],
            ["answer", "bridge"],
        )
        self.assertIn("普通员工", contract.requirements[1].description)
        self.assertNotIn("D级", contract.requirements[1].description)

    def test_requirements_receive_stable_ids_and_safe_coverage_semantics(self) -> None:
        route = _route(
            requirements=[
                {
                    "role": "bridge",
                    "origin": "semantically_entailed",
                    "description": "确定普通员工对应的职级",
                },
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "取得完整出差标准",
                },
                {
                    "role": "bridge",
                    "origin": "user_text",
                    "description": "保留用户明确给出的中间条件",
                },
            ]
        )
        contract = _compile(route)

        self.assertEqual([item.id for item in contract.requirements], ["r1", "r2", "r3"])
        self.assertEqual(
            [(item.importance, item.source) for item in contract.requirements],
            [("helpful", "inferred"), ("required", "explicit"), ("helpful", "explicit")],
        )
        self.assertEqual(contract.to_dict()["requirements"][1]["description"], "取得完整出差标准")

    def test_followup_context_binding_is_rechecked_at_compile_and_dispatch(self) -> None:
        route = _route(
            relation="followup",
            mode="current",
            turn_keys=["t1"],
        )
        contract = _compile(route)

        self.assertTrue(
            is_rag_task_contract_dispatchable(
                contract,
                selected_kb_count=1,
                available_turn_keys=["t1"],
            )
        )
        self.assertEqual(
            rag_task_contract_gate_reason(
                contract,
                selected_kb_count=1,
                available_turn_keys=["t2"],
            ),
            "context_turn_unavailable",
        )
        with self.assertRaises(TaskContractDispatchError):
            require_rag_task_contract_dispatchable(
                contract,
                selected_kb_count=1,
                available_turn_keys=["t2"],
            )

    def test_self_contained_followup_current_without_binding_is_dispatchable(self) -> None:
        route = _route(
            relation="followup",
            mode="current",
            turn_keys=[],
        )
        contract = _compile(route)

        self.assertEqual(contract.relation, "followup")
        self.assertEqual(contract.query_mode, "current")
        self.assertEqual(contract.context_turn_keys, ())
        self.assertTrue(
            is_rag_task_contract_dispatchable(
                contract,
                selected_kb_count=1,
                available_turn_keys=[],
            )
        )

    def test_gate_rejects_execution_field_drift_and_runtime_kb_change(self) -> None:
        contract = _compile(_route())
        inconsistent = replace(contract, need_retrieval=False)
        wrong_mode = replace(contract, response_mode="general_chat")
        missing_answer_target = replace(contract, requirements=())

        self.assertEqual(
            rag_task_contract_gate_reason(inconsistent),
            "retrieval_fields_inconsistent",
        )
        self.assertEqual(
            rag_task_contract_gate_reason(wrong_mode),
            "direct_mode_requires_skip",
        )
        self.assertEqual(
            rag_task_contract_gate_reason(contract, selected_kb_count=2),
            "selected_kb_count_changed",
        )
        self.assertEqual(
            rag_task_contract_gate_reason(missing_answer_target),
            "missing_answer_requirement",
        )
        self.assertFalse(is_rag_task_contract_dispatchable(missing_answer_target))

    def test_safe_summary_contains_no_semantic_prose(self) -> None:
        contract = _compile(
            _route(
                requirements=[
                    {
                        "role": "answer",
                        "origin": "user_text",
                        "description": "高度敏感的业务问题正文",
                    }
                ]
            ),
            question="请查询当前业务问题",
        )
        summary = safe_rag_task_contract_summary(contract)
        serialized = repr(summary)

        self.assertNotIn("高度敏感", serialized)
        self.assertNotIn("question", summary)
        self.assertEqual(summary["requirement_count"], 1)
        self.assertEqual(summary["required_requirement_count"], 1)
        self.assertEqual(summary["bridge_requirement_count"], 0)
        self.assertEqual(contract.safe_summary(), summary)

    def test_category_and_source_mismatch_fail_before_contract_creation(self) -> None:
        route = _route()
        wrong_category = RouteCategoryPolicy(
            code="general_chat",
            name="通用交流",
            action="chat",
        )
        disabled_category = replace(CATEGORIES["knowledge_qa"], enabled=False)

        for category, source in (
            (wrong_category, "llm"),
            (disabled_category, "llm"),
            (CATEGORIES["knowledge_qa"], "unsafe source"),
        ):
            with self.subTest(category=category, source=source):
                with self.assertRaises(TaskContractCompilationError):
                    compile_rag_task_contract(
                        route,
                        category,
                        RouteCompilerConfig(),
                        question="问题",
                        selected_kb_count=1,
                        source=source,
                    )


if __name__ == "__main__":
    unittest.main()
