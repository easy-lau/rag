import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.query_constraints import (
    evaluate_candidate_constraints,
    extract_query_constraints,
    inherit_document_constraint_metadata,
)
from core.reranker import (
    SIMPLE_RERANK_PROMPT_VERSION,
    AnswerRequirement,
    rerank,
    rerank_with_status,
)


def _client_with_payload(payload: dict):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))
    )


def _assessment(
    index: int,
    *,
    topic: float,
    support: float,
    constraint: str = "neutral",
    role: str = "direct",
    reason: str = "候选能够支撑查询",
) -> dict:
    return {
        "index": index,
        "topic_relevance": topic,
        "answer_support": support,
        "constraint_status": constraint,
        "evidence_role": role,
        "reason": reason,
    }


class QueryConstraintTests(unittest.TestCase):
    def test_action_word_adjacent_to_version_is_not_product_identity(self) -> None:
        constraints = extract_query_constraints("升级8.6的步骤")

        self.assertIsNone(constraints.product)
        self.assertEqual(constraints.version, "8.6")
        self.assertTrue(constraints.explicit_version)

    def test_extracts_product_and_explicit_version_from_user_context(self) -> None:
        constraints = extract_query_constraints(
            "解决登录用户名枚举 要配置什么 我是云枢8.6"
        )

        self.assertEqual(constraints.product, "云枢")
        self.assertEqual(constraints.version, "8.6")
        self.assertTrue(constraints.explicit_version)
        self.assertEqual(constraints.matched_text, "我是云枢8.6")
        self.assertIn("显式版本", constraints.extraction_reason)

    def test_obvious_old_versions_are_deterministic_mismatches(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6，用户名枚举怎么配置")

        for version in ("6", "7"):
            evaluation = evaluate_candidate_constraints(
                constraints,
                {
                    "filename": f"云枢{version}配置参数说明",
                    "content": f"产品版本：云枢{version}全系\nerror_reply_same: true",
                },
            )
            self.assertEqual(evaluation.status, "mismatch")
            self.assertIn("版本冲突", evaluation.reason)
            self.assertEqual(evaluation.candidate_versions, (version,))

    def test_explicit_compatibility_is_not_misclassified_as_exact(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6，查询登录安全配置")
        evaluation = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "跨版本安全说明",
                "content": "本配置已验证兼容云枢8.6，可用于统一登录失败提示。",
            },
        )

        self.assertEqual(evaluation.status, "compatible")
        self.assertIn("兼容", evaluation.reason)

    def test_negated_compatibility_and_version_prefix_are_not_positive_matches(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6")
        for text in (
            "本功能不兼容云枢8.6",
            "该参数不支持云枢8.6",
            "本配置并 不 支持 云枢 8.6",
            "本配置不适用于云枢8.6",
            "该能力不再兼容云枢8.6",
            "当前无法适用于云枢8.6",
            "该方案不能适用于云枢8.6",
            "此插件未适配云枢8.6",
            "这里只是对比，不代表支持云枢8.6",
            "云枢8.6不再兼容该配置",
        ):
            with self.subTest(text=text):
                evaluation = evaluate_candidate_constraints(
                    constraints,
                    {"filename": "说明", "content": text},
                )
                self.assertEqual(evaluation.status, "mismatch")

        prefix = evaluate_candidate_constraints(
            constraints,
            {"filename": "说明", "content": "本功能支持云枢8.6.1"},
        )
        self.assertNotEqual(prefix.status, "compatible")

    def test_declared_old_version_wins_over_incidental_target_version_mention(self) -> None:
        constraints = extract_query_constraints("CloudPivot v8.6 登录安全")
        evaluation = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "云枢6配置说明",
                "metadata": {"product": "云枢", "version": "6"},
                "content": "本文比较云枢8.6的差异，不代表兼容。",
            },
        )
        self.assertEqual(evaluation.status, "mismatch")
        self.assertIn("6", evaluation.candidate_versions)

    def test_exact_product_identity_does_not_use_implicit_aliases(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6，查询登录安全")
        exact = evaluate_candidate_constraints(
            constraints,
            {"metadata": {"product": "云枢", "version": "8.6"}, "content": "配置说明"},
        )
        conflicting = evaluate_candidate_constraints(
            constraints,
            {"metadata": {"product": "非云枢", "version": "8.6"}, "content": "配置说明"},
        )
        self.assertEqual(exact.status, "exact")
        self.assertEqual(conflicting.status, "mismatch")

    def test_node_count_does_not_create_product_or_version_scope(self) -> None:
        constraints = extract_query_constraints("使用云枢 8 个节点")
        self.assertIsNone(constraints.product)
        self.assertIsNone(constraints.version)
        self.assertFalse(constraints.has_hard_constraint)
        self.assertFalse(constraints.has_product_constraint)

    def test_productless_explicit_version_labels_are_extracted(self) -> None:
        for query, version in (
            ("版本8.6登录配置", "8.6"),
            ("v8.6登录配置", "8.6"),
            ("普通员工的2025版出差标准", "2025"),
        ):
            with self.subTest(query=query):
                constraints = extract_query_constraints(query)
                self.assertIsNone(constraints.product)
                self.assertEqual(constraints.version, version)
                self.assertTrue(constraints.explicit_version)
                self.assertTrue(constraints.has_version_constraint)
                self.assertTrue(constraints.has_scope_constraint)
                self.assertFalse(constraints.has_hard_constraint)

        quantity = extract_query_constraints("出差需要2天怎么报销")
        self.assertIsNone(quantity.version)
        self.assertFalse(quantity.explicit_version)

    def test_productless_version_is_checked_against_filename_identity(self) -> None:
        constraints = extract_query_constraints("版本8.6登录配置")
        exact = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "制度8.6配置",
                "metadata": {"source": "制度8.6配置"},
                "content": "登录安全说明",
            },
        )
        mismatch = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "制度7配置",
                "metadata": {"source": "制度7配置"},
                "content": "登录安全说明",
            },
        )
        unknown = evaluate_candidate_constraints(
            constraints,
            {"filename": "通用登录配置", "content": "登录安全说明"},
        )

        self.assertEqual(exact.status, "exact")
        self.assertEqual(mismatch.status, "mismatch")
        self.assertEqual(unknown.status, "unknown")

    def test_component_version_cannot_override_declared_product_version(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6，用户名枚举怎么配置")
        evaluation = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "云枢7配置.md",
                "content": "Java版本：8.6\nerror_reply_same1: true",
            },
        )

        self.assertEqual(evaluation.status, "mismatch")
        self.assertEqual(evaluation.candidate_versions, ("7",))

    def test_bare_component_compatibility_cannot_upgrade_old_product_document(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6，查询登录安全")
        evaluation = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "云枢7配置.md",
                "content": "浏览器组件支持8.6，云枢产品版本仍为7。",
            },
        )

        self.assertEqual(evaluation.status, "mismatch")

    def test_version_metadata_without_product_identity_is_unknown(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6，查询登录安全")
        evaluation = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "其他系统配置.md",
                "metadata": {"version": "8.6"},
                "content": "登录安全说明",
            },
        )

        self.assertEqual(evaluation.status, "unknown")

    def test_conflicting_declared_versions_are_not_exact(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6，查询登录安全")
        evaluation = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "云枢7配置.md",
                "content": "所属产品：云枢\n产品版本：云枢8.6",
            },
        )

        self.assertEqual(evaluation.status, "mismatch")

    def test_single_number_tag_is_not_treated_as_product_version(self) -> None:
        constraints = extract_query_constraints("我是云枢8.6，查询登录安全")
        evaluation = evaluate_candidate_constraints(
            constraints,
            {
                "filename": "云枢配置.md",
                "doc_tags": ["8"],
                "content": "登录安全说明",
            },
        )

        self.assertEqual(evaluation.status, "unknown")

    def test_product_generation_suffix_and_flattened_markdown_fields_are_normalized(self) -> None:
        product_only = extract_query_constraints("产品：云枢；想二开消息可以吗")
        candidate = {
            "filename": "二开发送钉钉工作通知",
            "content": (
                "【一、基本信息】\n"
                "> 所属产品：云枢>> 产品版本：8.2.75>> 所属项目：中青建安>"
            ),
        }

        evaluation = evaluate_candidate_constraints(product_only, candidate)

        self.assertEqual(evaluation.status, "neutral")
        self.assertIn("8.2.75", evaluation.reason)
        self.assertNotIn("所属项目", evaluation.candidate_products)

        explicit_version = evaluate_candidate_constraints(
            extract_query_constraints("我是云枢8.6，如何发送消息"),
            candidate,
        )
        self.assertEqual(explicit_version.status, "mismatch")
        self.assertIn("8.2.75", explicit_version.reason)

    def test_exact_product_label_is_not_arbitrary_substring_match(self) -> None:
        constraints = extract_query_constraints("产品：云枢；如何配置消息")
        same_product = evaluate_candidate_constraints(
            constraints,
            {
                "metadata": {"product": "云枢", "version": "8.2.75"},
                "content": "消息配置",
            },
        )
        conflicting = evaluate_candidate_constraints(
            constraints,
            {
                "metadata": {"product": "非云枢8", "version": "8.2.75"},
                "content": "消息配置",
            },
        )

        self.assertEqual(same_product.status, "neutral")
        self.assertEqual(conflicting.status, "mismatch")

    def test_document_identity_is_inherited_only_within_same_kb_and_document(self) -> None:
        candidates = [
            {
                "id": "basic",
                "kb_id": "kb-a",
                "doc_id": "doc-a",
                "content": "所属产品：云枢>> 产品版本：8.2.75>>",
            },
            {
                "id": "solution",
                "kb_id": "kb-a",
                "doc_id": "doc-a",
                "content": "调用 DingTalkMessageServiceImpl 发送消息",
            },
            {
                "id": "other-document",
                "kb_id": "kb-a",
                "doc_id": "doc-b",
                "content": "调用 DingTalkMessageServiceImpl 发送消息",
            },
            {
                "id": "same-doc-id-other-kb",
                "kb_id": "kb-b",
                "doc_id": "doc-a",
                "content": "调用 DingTalkMessageServiceImpl 发送消息",
            },
        ]
        enriched = inherit_document_constraint_metadata(candidates)
        constraints = extract_query_constraints("产品：云枢；想二开消息可以吗")
        evaluations = {
            item["id"]: evaluate_candidate_constraints(constraints, item)
            for item in enriched
        }

        self.assertEqual(evaluations["basic"].status, "neutral")
        self.assertEqual(evaluations["solution"].status, "neutral")
        self.assertIn("8.2.75", evaluations["solution"].reason)
        self.assertEqual(evaluations["other-document"].status, "unknown")
        self.assertEqual(evaluations["same-doc-id-other-kb"].status, "unknown")


class RerankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_locked_requirement_uses_compact_rerank_contract(self) -> None:
        results = [{"id": "a", "content": "报销需先提交申请", "score": 0.02}]
        client = _client_with_payload(
            {"results": [_assessment(1, topic=0.9, support=0.9)]}
        )

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            outcome = await rerank_with_status(
                "公司的报销流程是什么",
                results,
                [AnswerRequirement("r1", "回答公司的报销流程")],
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.prompt_version, SIMPLE_RERANK_PROMPT_VERSION)
        request = client.chat.completions.create.await_args.kwargs
        self.assertEqual(request["max_tokens"], 900)
        self.assertIn("不要规划证据扩展", request["messages"][0]["content"])
        self.assertNotIn("target_candidate_indexes", request["messages"][0]["content"])

    async def test_initial_rerank_uses_dedicated_model_when_configured(self) -> None:
        results = [{"id": "a", "content": "A", "score": 0.02}]
        client = _client_with_payload(
            {"results": [_assessment(1, topic=0.9, support=0.9)]}
        )

        with (
            patch("core.reranker.get_client", return_value=client),
            patch(
                "core.reranker.get_settings",
                return_value=SimpleNamespace(
                    chat_model="chat-model",
                    rerank_model=" fast-reranker ",
                ),
            ),
        ):
            outcome = await rerank_with_status("普通查询", results)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.model, "fast-reranker")
        self.assertEqual(
            client.chat.completions.create.await_args.kwargs["model"],
            "fast-reranker",
        )

    async def test_legacy_rerank_negotiates_response_format_downgrade(self) -> None:
        """The legacy rerank path reuses the structured-output negotiation.

        A provider that rejects ``response_format`` must degrade to plain JSON
        instead of failing the whole rerank with a 400.
        """

        class ProviderContractError(Exception):
            status_code = 400

        payload = {"results": [_assessment(1, topic=0.9, support=0.9)]}

        async def create(**kwargs):
            if "response_format" in kwargs:
                raise ProviderContractError(
                    "Request failed: Bad Request, error: "
                    "This response_format type is unavailable now"
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )]
            )

        create_mock = AsyncMock(side_effect=create)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        results = [{"id": "a", "content": "报销需先提交申请", "score": 0.02}]
        with (
            patch("core.reranker.get_client", return_value=client),
            patch(
                "core.reranker.get_settings",
                return_value=SimpleNamespace(
                    chat_model="test-chat",
                    llm_base_url="https://provider.example/v1",
                ),
            ),
        ):
            outcome = await rerank_with_status(
                "公司的报销流程是什么",
                results,
                [AnswerRequirement("r1", "回答公司的报销流程")],
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.structured_output_mode, "plain_json")
        self.assertEqual(
            outcome.structured_output_attempted_modes,
            ("json_schema", "json_object", "plain_json"),
        )
        self.assertNotIn("response_format", create_mock.await_args_list[-1].kwargs)

    async def test_legacy_rerank_uses_adjudication_timeout_not_llm_timeout(
        self,
    ) -> None:
        """Legacy rerank must not fall back to the global LLM timeout."""

        results = [{"id": "a", "content": "A", "score": 0.02}]
        client = _client_with_payload(
            {"results": [_assessment(1, topic=0.9, support=0.9)]}
        )
        with (
            patch("core.reranker.get_client", return_value=client),
            patch(
                "core.reranker.get_settings",
                return_value=SimpleNamespace(
                    chat_model="test-chat",
                    llm_request_timeout_seconds=60,
                    rerank_timeout_seconds=15,
                ),
            ),
        ):
            outcome = await rerank_with_status("普通查询", results)

        self.assertTrue(outcome.succeeded)
        request_timeout = client.chat.completions.create.await_args.kwargs["timeout"]
        self.assertGreater(request_timeout, 0.0)
        self.assertLessEqual(request_timeout, 15.0)

    async def test_complete_valid_assessments_are_trusted_and_sorted(self) -> None:
        results = [
            {"id": "a", "content": "A", "score": 0.02},
            {"id": "b", "content": "B", "score": 0.01},
        ]
        client = _client_with_payload(
            {
                "results": [
                    _assessment(1, topic=0.4, support=0.2),
                    _assessment(2, topic=0.9, support=0.9),
                ]
            }
        )

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            outcome = await rerank_with_status("普通查询", results)

        self.assertTrue(outcome.succeeded)
        self.assertEqual([item["id"] for item in outcome.results], ["b", "a"])
        self.assertEqual([item["score"] for item in outcome.results], [0.9, 0.2])
        self.assertEqual([item["retrieval_score"] for item in outcome.results], [0.01, 0.02])
        self.assertTrue(all(item["rerank_status"] == "verified" for item in outcome.results))
        # 不原地覆盖检索器返回值，失败回退时才能保留原始分数语义。
        self.assertEqual(results[0]["score"], 0.02)
        self.assertNotIn("retrieval_score", results[0])

    async def test_disabled_expansion_noise_does_not_invalidate_assessments(self) -> None:
        results = [
            {"id": str(index), "content": f"候选{index}", "score": 0.01}
            for index in range(1, 16)
        ]
        client = _client_with_payload({
            "results": [
                _assessment(index, topic=0.8, support=0.8)
                for index in range(1, 16)
            ],
            "expansion": {
                "needed": False,
                "target_candidate_indexes": list(range(1, 16)),
                "queries": [],
                "missing_requirement_ids": [],
                "reason": "不需要扩展",
            },
        })

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            outcome = await rerank_with_status("普通查询", results)

        self.assertTrue(outcome.succeeded)
        self.assertIsNotNone(outcome.expansion_plan)
        self.assertFalse(outcome.expansion_plan.needed)

    async def test_high_semantic_score_cannot_override_version_mismatch(self) -> None:
        results = [
            {
                "id": "v6",
                "filename": "云枢6配置参数说明",
                "doc_tags": ["云枢", "配置"],
                "metadata": {"heading": "登录安全"},
                "content": "产品版本：云枢6全系\nerror_reply_same: true",
                "score": 0.03,
            },
            {
                "id": "v7",
                "filename": "云枢7配置",
                "content": "云枢7：error_reply_same1: true",
                "score": 0.03,
            },
            {
                "id": "v86",
                "filename": "云枢8.6安全配置",
                "content": "云枢8.6：security.login.error-reply-same: true",
                "score": 0.01,
            },
        ]
        # 故意模拟模型把旧版本全部误判为“精确直接证据”。代码规则必须覆盖它。
        client = _client_with_payload(
            {
                "results": [
                    _assessment(1, topic=0.99, support=0.99, constraint="exact"),
                    _assessment(2, topic=0.99, support=0.99, constraint="exact"),
                    _assessment(3, topic=0.80, support=0.80, constraint="exact"),
                ]
            }
        )

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            outcome = await rerank_with_status(
                "解决登录用户名枚举 要配置什么 我是云枢8.6",
                results,
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual([item["id"] for item in outcome.results], ["v86", "v6", "v7"])
        exact, version6, version7 = outcome.results
        self.assertEqual(exact["constraint_status"], "exact")
        for item in (version6, version7):
            self.assertEqual(item["constraint_status"], "mismatch")
            self.assertEqual(item["evidence_role"], "related")
            self.assertEqual(item["score"], 0.0)
            self.assertEqual(item["answer_support"], 0.99)
            self.assertTrue(item["constraint_overridden"])
            self.assertIn("版本冲突", item["constraint_reason"])

    async def test_productless_version_still_applies_hard_rerank_gate(self) -> None:
        results = [
            {
                "id": "v7",
                "filename": "登录制度7版",
                "content": "旧版登录配置",
                "score": 0.9,
            },
            {
                "id": "v86",
                "filename": "登录制度8.6版",
                "content": "目标版登录配置",
                "score": 0.2,
            },
        ]
        client = _client_with_payload({
            "results": [
                _assessment(1, topic=0.99, support=0.99, constraint="exact"),
                _assessment(2, topic=0.85, support=0.85, constraint="exact"),
            ]
        })

        with (
            patch("core.reranker.get_client", return_value=client),
            patch(
                "core.reranker.get_settings",
                return_value=SimpleNamespace(chat_model="test"),
            ),
        ):
            outcome = await rerank_with_status("版本8.6登录配置", results)

        self.assertTrue(outcome.succeeded)
        self.assertEqual([item["id"] for item in outcome.results], ["v86", "v7"])
        exact, mismatch = outcome.results
        self.assertEqual(exact["constraint_status"], "exact")
        self.assertEqual(exact["evidence_role"], "direct")
        self.assertEqual(mismatch["constraint_status"], "mismatch")
        self.assertEqual(mismatch["evidence_role"], "related")
        self.assertEqual(mismatch["score"], 0.0)
        self.assertTrue(mismatch["query_has_constraint"])
        self.assertFalse(mismatch["query_has_product_constraint"])
        self.assertTrue(mismatch["query_has_version_constraint"])
        self.assertFalse(mismatch["query_has_hard_constraint"])

    async def test_rerank_inherits_product_scope_for_answer_chunk(self) -> None:
        results = [
            {
                "id": "basic",
                "kb_id": "kb",
                "doc_id": "doc",
                "filename": "二开发送钉钉工作通知",
                "content": "所属产品：云枢>> 产品版本：8.2.75>>",
                "score": 0.03,
            },
            {
                "id": "solution",
                "kb_id": "kb",
                "doc_id": "doc",
                "filename": "二开发送钉钉工作通知",
                "content": "调用 DingTalkMessageServiceImpl 发送钉钉工作通知",
                "score": 0.01,
            },
        ]
        client = _client_with_payload({
            "results": [
                _assessment(
                    1,
                    topic=0.8,
                    support=0.1,
                    role="related",
                ),
                _assessment(2, topic=1.0, support=0.98, role="direct"),
            ]
        })

        with (
            patch("core.reranker.get_client", return_value=client),
            patch(
                "core.reranker.get_settings",
                return_value=SimpleNamespace(chat_model="test"),
            ),
        ):
            outcome = await rerank_with_status(
                "产品：云枢；想二开消息可以吗",
                results,
                [AnswerRequirement("r1", "确认是否支持二开发送钉钉消息")],
            )

        self.assertTrue(outcome.succeeded)
        answer = next(item for item in outcome.results if item["id"] == "solution")
        self.assertEqual(answer["constraint_status"], "neutral")
        self.assertEqual(answer["evidence_role"], "direct")
        self.assertIn("8.2.75", answer["constraint_reason"])
        self.assertEqual(
            answer["metadata"]["inherited_document_identity"]["version"],
            ["8.2.75"],
        )

    async def test_prompt_contains_filename_tags_metadata_and_more_than_300_chars(self) -> None:
        long_content = "甲" * 360 + "CONTENT_END_MARKER"
        results = [
            {
                "id": "a",
                "filename": "云枢8.6配置.md",
                "doc_tags": ["登录安全"],
                "metadata": {"heading": "用户名枚举"},
                "content": long_content,
                "score": 0.02,
            }
        ]
        client = _client_with_payload(
            {"results": [_assessment(1, topic=0.9, support=0.9, constraint="exact")]}
        )

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            await rerank_with_status("云枢8.6如何配置", results)

        messages = client.chat.completions.create.await_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertNotIn("CONTENT_END_MARKER", messages[0]["content"])
        prompt = messages[1]["content"]
        self.assertIn("云枢8.6配置.md", prompt)
        self.assertIn("登录安全", prompt)
        self.assertIn("用户名枚举", prompt)
        self.assertIn("CONTENT_END_MARKER", prompt)

    async def test_incomplete_assessments_are_unverified_and_preserve_recall(self) -> None:
        results = [
            {"id": "a", "content": "A", "score": 0.02},
            {"id": "b", "content": "B", "score": 0.01},
        ]
        client = _client_with_payload(
            {"results": [_assessment(1, topic=0.8, support=0.8)]}
        )

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            outcome = await rerank_with_status("query", results)
            compatible_results = await rerank("query", results)

        self.assertFalse(outcome.succeeded)
        self.assertIn("未覆盖全部候选", outcome.error or "")
        self.assertEqual([item["id"] for item in outcome.results], ["a", "b"])
        self.assertEqual([item["score"] for item in outcome.results], [0.02, 0.01])
        self.assertEqual(
            [item["retrieval_score"] for item in outcome.results], [0.02, 0.01]
        )
        self.assertTrue(
            all(item["rerank_status"] == "unverified" for item in outcome.results)
        )
        self.assertTrue(all(item["topic_relevance"] is None for item in outcome.results))
        self.assertEqual([item["id"] for item in compatible_results], ["a", "b"])

    async def test_invalid_constraint_field_uses_backend_status_without_batch_failure(self) -> None:
        results = [{"id": "a", "content": "A", "score": 0.015625}]
        client = _client_with_payload(
            {
                "results": [
                    _assessment(
                        1,
                        topic=0.9,
                        support=0.9,
                        constraint="made_up",
                    )
                ]
            }
        )

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=SimpleNamespace(chat_model="test")),
        ):
            outcome = await rerank_with_status("query", results)

        self.assertTrue(outcome.succeeded)
        self.assertIsNone(outcome.error)
        self.assertEqual(outcome.results[0]["constraint_status"], "neutral")
        self.assertEqual(
            outcome.results[0]["constraint_status_resolution"],
            "deterministic_fallback",
        )
        self.assertEqual(outcome.results[0]["rerank_status"], "verified")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
