import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.intent_router import (
    DEFAULT_INTENT_CATEGORIES,
    _apply_routing_policy,
    _classification_prompt,
    _classify_with_llm,
    _default_config,
    _fallback_decision,
    _make_decision,
    _rule_match,
)
from models.db_models import IntentCategory


def _categories() -> list[IntentCategory]:
    return [IntentCategory(**item) for item in DEFAULT_INTENT_CATEGORIES]


def _category(code: str) -> IntentCategory:
    return next(item for item in _categories() if item.code == code)


class IntentRoutingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _default_config()

    def test_external_product_help_misclassification_is_forced_to_retrieve(self) -> None:
        classified = _make_decision(_category("system_help"), 0.99, "llm")

        decision = _apply_routing_policy(
            "我现在想改云枢的默认密码怎么办",
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.intent_code, "system_help")
        self.assertEqual(decision.action, "system_help")
        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.response_mode, "grounded_qa")
        self.assertEqual(decision.retrieval_policy, "required")
        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "platform_help_scope_guard")

    def test_external_product_help_guard_requires_retrieval_without_selected_kb(self) -> None:
        classified = _make_decision(_category("system_help"), 0.92, "llm")

        decision = _apply_routing_policy(
            "某业务产品的接口地址在哪里配置？",
            classified,
            self.config,
            selected_kb_count=0,
        )

        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.retrieval_policy, "required")
        self.assertEqual(decision.decision_reason, "platform_help_scope_guard")

    def test_explicit_current_rag_platform_help_skips_retrieval(self) -> None:
        classified = _make_decision(_category("system_help"), 0.98, "rule")

        decision = _apply_routing_policy(
            "当前 RAG 平台怎么上传文档？",
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.response_mode, "platform_help")
        self.assertEqual(decision.retrieval_policy, "skip")
        self.assertFalse(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "explicit_platform_help")

    def test_exact_greeting_skips_retrieval_even_when_kb_is_selected(self) -> None:
        classified = _rule_match("你好！", _categories())
        self.assertIsNotNone(classified)

        decision = _apply_routing_policy(
            "你好！",
            classified,
            self.config,
            selected_kb_count=2,
        )

        self.assertEqual(decision.response_mode, "general_chat")
        self.assertEqual(decision.retrieval_policy, "skip")
        self.assertFalse(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "exact_greeting")

    def test_exact_greeting_rule_corrects_retrieve_misclassification(self) -> None:
        classified = _make_decision(_category("knowledge_qa"), 0.93, "llm")

        decision = _apply_routing_policy(
            "你好",
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.intent_code, "knowledge_qa")
        self.assertEqual(decision.response_mode, "general_chat")
        self.assertFalse(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "exact_greeting")

    def test_explicit_platform_help_corrects_retrieve_misclassification(self) -> None:
        classified = _make_decision(_category("knowledge_qa"), 0.9, "llm")

        decision = _apply_routing_policy(
            "在哪里查看检索结果？",
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.response_mode, "platform_help")
        self.assertEqual(decision.retrieval_policy, "skip")
        self.assertFalse(decision.need_retrieval)

    def test_platform_reference_with_business_query_still_retrieves(self) -> None:
        classified = _make_decision(_category("system_help"), 0.96, "llm")

        decision = _apply_routing_policy(
            "如何在当前 RAG 平台查询某业务产品的默认密码？",
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.response_mode, "grounded_qa")
        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "platform_help_scope_guard")

    def test_inline_writing_with_actual_content_skips_retrieval(self) -> None:
        question = "请润色以下内容：明天上午十点开会。"
        classified = _rule_match(question, _categories())
        self.assertIsNotNone(classified)

        decision = _apply_routing_policy(
            question,
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.action, "writing")
        self.assertEqual(decision.response_mode, "writing")
        self.assertEqual(decision.retrieval_policy, "skip")
        self.assertFalse(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "inline_writing_content")

    def test_inline_writing_rule_corrects_retrieve_misclassification(self) -> None:
        question = "请翻译以下内容：Hello world"
        classified = _make_decision(_category("knowledge_qa"), 0.9, "llm")

        decision = _apply_routing_policy(
            question,
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.response_mode, "writing")
        self.assertFalse(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "inline_writing_content")

    def test_writing_based_on_knowledge_requires_retrieval_with_selected_kb(self) -> None:
        classified = _make_decision(_category("writing"), 0.88, "llm")

        decision = _apply_routing_policy(
            "请根据员工手册总结请假规则",
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.response_mode, "writing")
        self.assertEqual(decision.retrieval_policy, "required")
        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "knowledge_dependent_writing")

    def test_writing_based_on_knowledge_requires_kb_selection(self) -> None:
        classified = _make_decision(_category("writing"), 0.88, "llm")

        decision = _apply_routing_policy(
            "总结云枢配置文档",
            classified,
            self.config,
            selected_kb_count=0,
        )

        self.assertEqual(decision.response_mode, "writing")
        self.assertEqual(decision.retrieval_policy, "required")
        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "knowledge_dependent_writing")

    def test_chat_with_selected_kb_uses_optional_retrieval(self) -> None:
        classified = _make_decision(_category("general_chat"), 0.81, "llm")

        decision = _apply_routing_policy(
            "这个产品支持哪些部署方式？",
            classified,
            self.config,
            selected_kb_count=1,
        )

        self.assertEqual(decision.response_mode, "general_chat")
        self.assertEqual(decision.retrieval_policy, "optional")
        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "selected_knowledge_context")

    def test_chat_without_selected_kb_keeps_optional_policy_but_does_not_retrieve(self) -> None:
        classified = _make_decision(_category("general_chat"), 0.81, "llm")

        decision = _apply_routing_policy(
            "介绍一下向量数据库",
            classified,
            self.config,
            selected_kb_count=0,
        )

        self.assertEqual(decision.retrieval_policy, "optional")
        self.assertFalse(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "no_selected_knowledge")

    def test_disabling_general_chat_forces_retrieval_without_rewriting_source(self) -> None:
        config = _default_config()
        config.allow_general_chat = False
        classified = _make_decision(_category("general_chat"), 0.91, "llm")

        decision = _apply_routing_policy(
            "介绍一下向量数据库",
            classified,
            config,
            selected_kb_count=0,
        )

        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.response_mode, "grounded_qa")
        self.assertEqual(decision.retrieval_policy, "required")
        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "general_chat_disabled")

    def test_fallback_always_requires_retrieval(self) -> None:
        decision = _fallback_decision(self.config, _categories())

        self.assertEqual(decision.action, "retrieve")
        self.assertEqual(decision.response_mode, "grounded_qa")
        self.assertEqual(decision.retrieval_policy, "required")
        self.assertTrue(decision.need_retrieval)
        self.assertEqual(decision.decision_reason, "safe_fallback")

    def test_prompt_uses_selected_knowledge_as_weak_prior_and_narrows_platform_help(self) -> None:
        prompt = _classification_prompt(
            "某业务产品的默认密码如何配置？",
            _categories(),
            selected_kb_count=2,
        )

        self.assertIn("选择了 2 个知识库", prompt)
        self.assertIn("只是弱先验", prompt)
        self.assertIn("外部产品、业务系统", prompt)
        self.assertIn("都不是 system_help", prompt)
        self.assertIn("仍应选择 writing", prompt)

    def test_to_dict_contains_classification_and_execution_plan(self) -> None:
        classified = _make_decision(_category("knowledge_qa"), 0.9, "llm")
        decision = _apply_routing_policy("报销流程是什么？", classified, self.config)

        self.assertEqual(
            set(decision.to_dict()),
            {
                "intent_code",
                "intent_name",
                "action",
                "confidence",
                "source",
                "response_mode",
                "retrieval_policy",
                "need_retrieval",
                "decision_reason",
            },
        )


class IntentRoutingModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_uses_model_management_intent_model(self) -> None:
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"intent_code":"general_chat","confidence":0.9}'
                        )
                    )
                ]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with (
            patch("core.intent_router.get_client", return_value=client),
            patch(
                "core.intent_router.get_settings",
                return_value=SimpleNamespace(
                    intent_model="intent-model",
                    chat_model="chat-model",
                ),
            ),
        ):
            decision = await _classify_with_llm(
                "介绍一下向量数据库",
                _default_config(),
                _categories(),
            )

        self.assertIsNotNone(decision)
        self.assertEqual(create.await_args.kwargs["model"], "intent-model")

    async def test_classifier_falls_back_to_chat_model_when_intent_model_is_empty(self) -> None:
        create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"intent_code":"general_chat","confidence":0.9}'
                        )
                    )
                ]
            )
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with (
            patch("core.intent_router.get_client", return_value=client),
            patch(
                "core.intent_router.get_settings",
                return_value=SimpleNamespace(intent_model="", chat_model="chat-model"),
            ),
        ):
            await _classify_with_llm(
                "介绍一下向量数据库",
                _default_config(),
                _categories(),
            )

        self.assertEqual(create.await_args.kwargs["model"], "chat-model")


if __name__ == "__main__":
    unittest.main()
