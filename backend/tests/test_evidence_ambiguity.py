import unittest

from core.evidence_ambiguity import (
    _topic_document_rows,
    _version_key,
    detect_evidence_scope_ambiguity,
    query_requests_all_scopes,
    resolve_explicit_scope_comparison,
)
from core.query_constraints import extract_query_constraints
from core.rag_v2.query_plan import plan_query_locally


def _candidate(
    *,
    chunk_id: str,
    doc_id: str,
    filename: str,
    content: str,
    topic: float,
    support: float,
    role: str = "related",
    metadata: dict | None = None,
    rerank_status: str = "verified",
) -> dict:
    return {
        "id": chunk_id,
        "kb_id": "kb-1",
        "doc_id": doc_id,
        "filename": filename,
        "content": content,
        "metadata": metadata or {},
        "topic_relevance": topic,
        "answer_support": support,
        "evidence_role": role,
        "rerank_status": rerank_status,
    }


class EvidenceAmbiguityTests(unittest.TestCase):
    def test_topic_document_rows_tolerates_non_dict_metadata(self) -> None:
        for metadata in (None, "legacy source", ["legacy source"]):
            with self.subTest(metadata=metadata):
                rows = _topic_document_rows(
                    [
                        {
                            "kb_id": "kb-1",
                            "doc_id": "doc-legacy",
                            "metadata": metadata,
                        }
                    ]
                )
                self.assertEqual(rows, [])

        rows = _topic_document_rows(
            [
                {
                    "kb_id": "kb-1",
                    "doc_id": "doc-mapping",
                    "metadata": {"source": "员工制度.docx"},
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["filename"], "员工制度.docx")

    def test_broad_unscoped_query_clarifies_independent_documents(self) -> None:
        candidates = [
            _candidate(
                chunk_id="leave-1",
                doc_id="doc-leave",
                filename="员工请假管理办法.docx",
                content="员工请假制度、审批流程和休假要求。",
                topic=0.90,
                support=0.80,
            ),
            _candidate(
                chunk_id="travel-1",
                doc_id="doc-travel",
                filename="公司出差管理标准.docx",
                content="员工出差交通、住宿和餐饮标准。",
                topic=0.88,
                support=0.78,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="员工标准是什么",
            constraints=extract_query_constraints("员工标准是什么"),
            candidates=candidates,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "document")
        self.assertEqual(decision.reason, "multiple_mutually_relevant_documents")
        self.assertEqual(
            {choice.doc_ids for choice in decision.choices},
            {("doc-leave",), ("doc-travel",)},
        )
        self.assertIn("员工请假管理办法.docx", decision.question)
        self.assertIn("公司出差管理标准.docx", decision.question)

    def test_document_topic_gate_keeps_specific_query_answerable(self) -> None:
        candidates = [
            _candidate(
                chunk_id="leave-1",
                doc_id="doc-leave",
                filename="员工请假管理办法.docx",
                content="员工请假制度、审批流程和休假要求。",
                topic=0.35,
                support=0.20,
            ),
            _candidate(
                chunk_id="travel-1",
                doc_id="doc-travel",
                filename="公司出差管理标准.docx",
                content="员工出差交通、住宿和餐饮标准。",
                topic=0.90,
                support=0.88,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="员工的出差标准是什么",
            constraints=extract_query_constraints("员工的出差标准是什么"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)

    def test_cross_document_employee_classification_and_travel_rule_are_complementary(
        self,
    ) -> None:
        query = "普通员工的出差标准是什么"
        plan = plan_query_locally(query)
        candidates = [
            _candidate(
                chunk_id="employee-level",
                doc_id="doc-employee-level",
                filename="员工职级分类.docx",
                content="普通员工对应D级。",
                topic=0.92,
                support=0.82,
            ),
            _candidate(
                chunk_id="travel-rule",
                doc_id="doc-travel-rule",
                filename="公司出差管理标准.docx",
                content="D级出差标准包括交通、住宿和餐补。",
                topic=0.90,
                support=0.86,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
            requirements=plan.requirements,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "single_or_overlapping_scope")

    def test_cross_document_worker_classification_and_lodging_rule_are_complementary(
        self,
    ) -> None:
        query = "合同工住宿标准是多少"
        plan = plan_query_locally(query)
        candidates = [
            _candidate(
                chunk_id="worker-class",
                doc_id="doc-worker-class",
                filename="人员分类.docx",
                content="合同工属于L2类。",
                topic=0.91,
                support=0.80,
            ),
            _candidate(
                chunk_id="lodging-rule",
                doc_id="doc-lodging-rule",
                filename="住宿标准.docx",
                content="L2类住宿标准为300元/天。",
                topic=0.89,
                support=0.87,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
            requirements=plan.requirements,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "single_or_overlapping_scope")

    def test_cross_document_risk_explanation_and_configuration_are_complementary(
        self,
    ) -> None:
        query = "登录用户名枚举要配置什么"
        plan = plan_query_locally(query)
        candidates = [
            _candidate(
                chunk_id="risk-explanation",
                doc_id="doc-risk-explanation",
                filename="安全风险说明.docx",
                content="登录用户名枚举属于账号安全风险。",
                topic=0.91,
                support=0.76,
            ),
            _candidate(
                chunk_id="security-config",
                doc_id="doc-security-config",
                filename="安全配置手册.docx",
                content="用户名枚举防护需配置统一错误提示。",
                topic=0.90,
                support=0.88,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
            requirements=plan.requirements,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "single_or_overlapping_scope")

    def test_multi_part_requirements_anchor_complementary_documents(self) -> None:
        query = "报销提交时限是多久？需要提供哪些凭证？"
        plan = plan_query_locally(query)
        candidates = [
            _candidate(
                chunk_id="deadline",
                doc_id="doc-deadline",
                filename="制度附件一.docx",
                content="费用报销时限：出差结束后5个工作日内提交。",
                topic=0.90,
                support=0.84,
            ),
            _candidate(
                chunk_id="receipts",
                doc_id="doc-receipts",
                filename="制度附件二.docx",
                content="报销凭证：必须提供正规发票、行程单及住宿发票。",
                topic=0.89,
                support=0.85,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
            requirements=plan.requirements,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "single_or_overlapping_scope")

    def test_vague_employee_standard_still_clarifies_with_plan_requirements(
        self,
    ) -> None:
        query = "员工标准是什么"
        plan = plan_query_locally(query)
        candidates = [
            _candidate(
                chunk_id="leave-vague",
                doc_id="doc-leave-vague",
                filename="员工请假管理办法.docx",
                content="员工请假制度、审批流程和休假要求。",
                topic=0.90,
                support=0.80,
            ),
            _candidate(
                chunk_id="travel-vague",
                doc_id="doc-travel-vague",
                filename="公司出差管理标准.docx",
                content="员工出差交通、住宿和餐饮标准。",
                topic=0.88,
                support=0.78,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
            requirements=plan.requirements,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "document")

    def test_version_conflict_still_clarifies_with_complementary_requirement_signal(
        self,
    ) -> None:
        query = "普通员工的出差标准是什么"
        plan = plan_query_locally(query)
        candidates = [
            _candidate(
                chunk_id="travel-2024",
                doc_id="doc-travel-2024",
                filename="差旅制度2024版.docx",
                content="普通员工住宿标准。",
                metadata={"version": "2024"},
                topic=0.94,
                support=0.88,
            ),
            _candidate(
                chunk_id="travel-2025",
                doc_id="doc-travel-2025",
                filename="差旅制度2025版.docx",
                content="普通员工住宿标准。",
                metadata={"version": "2025"},
                topic=0.95,
                support=0.89,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
            requirements=plan.requirements,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "version")

    def test_document_topic_gate_respects_explicit_all_documents_request(self) -> None:
        candidates = [
            _candidate(
                chunk_id="leave-all",
                doc_id="doc-leave",
                filename="员工请假管理办法.docx",
                content="员工请假制度。",
                topic=0.90,
                support=0.80,
            ),
            _candidate(
                chunk_id="travel-all",
                doc_id="doc-travel",
                filename="公司出差管理标准.docx",
                content="员工出差标准。",
                topic=0.88,
                support=0.78,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="请汇总所有员工制度",
            constraints=extract_query_constraints("请汇总所有员工制度"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)

    def test_log9_two_relevant_versions_require_clarification(self) -> None:
        candidates = [
            _candidate(
                chunk_id="v6-basic",
                doc_id="doc-v6",
                filename="钉钉",
                content="所属产品：云枢6>> 产品版本：6.0.1>>",
                topic=0.7,
                support=0.1,
            ),
            _candidate(
                chunk_id="v8-basic",
                doc_id="doc-v8",
                filename="二开发送钉钉工作通知",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：中青建安>"
                ),
                topic=0.9,
                support=0.1,
            ),
            _candidate(
                chunk_id="v8-solution",
                doc_id="doc-v8",
                filename="二开发送钉钉工作通知",
                content="调用 DingTalkMessageServiceImpl 发送钉钉工作通知",
                topic=1.0,
                support=0.98,
                role="direct",
            ),
            _candidate(
                chunk_id="v7-config",
                doc_id="doc-v7-config",
                filename="云枢7配置",
                content="所属产品：云枢>> 产品版本：云枢7全系>>",
                topic=0.2,
                support=0.0,
                role="irrelevant",
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢中想二开钉钉消息可以吗",
            constraints=extract_query_constraints(
                "云枢中想二开钉钉消息可以吗"
            ),
            candidates=candidates,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "version")
        self.assertEqual(len(decision.choices), 2)
        self.assertEqual(
            {choice.versions for choice in decision.choices},
            {("6.0.1",), ("8.2.75",)},
        )
        self.assertIn("中青建安", decision.question)
        self.assertIn("都对比", decision.question)
        self.assertNotIn("云枢7配置", decision.question)
        self.assertEqual(decision.allowed_doc_ids, ())
        self.assertEqual(decision.to_dict()["allowed_doc_ids"], [])

    def test_unrelated_version_metadata_does_not_hijack_meal_allowance_query(
        self,
    ) -> None:
        """A policy question must not become a product-version picker.

        This mirrors the live regression where a rerank timeout left the
        versioned DingTalk documents unverified.  Their source labels are
        valid scope metadata, but neither document contains an anchor for the
        employee meal-allowance question, so they cannot create choices.
        """

        query = "普通员工餐补标准是多少？请说明对应职级和每日金额。"
        candidates = [
            _candidate(
                chunk_id="travel-meal",
                doc_id="doc-travel",
                filename="公司出差管理标准.docx",
                content="普通员工、专员属于D级，餐饮补贴为100元/天。",
                topic=0.0,
                support=0.0,
                metadata={},
                rerank_status="error",
            ),
            _candidate(
                chunk_id="dingtalk-v6",
                doc_id="doc-dingtalk-v6",
                filename="钉钉 6.0.1 工作通知配置.md",
                content="通过工作通知接口发送钉钉消息。",
                topic=0.0,
                support=0.0,
                metadata={"product": "钉钉", "version": "6.0.1"},
                rerank_status="error",
            ),
            _candidate(
                chunk_id="dingtalk-v8",
                doc_id="doc-dingtalk-v8",
                filename="钉钉 8.2.75 工作通知配置.md",
                content="使用新版工作通知接口发送消息。",
                topic=0.0,
                support=0.0,
                metadata={"product": "钉钉", "version": "8.2.75"},
                rerank_status="error",
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "single_or_overlapping_scope")
        self.assertEqual(decision.allowed_doc_ids, ())

    def test_same_version_complementary_documents_do_not_clarify(self) -> None:
        candidates = [
            _candidate(
                chunk_id="a",
                doc_id="doc-a",
                filename="配置说明",
                content="所属产品：云枢8>> 产品版本：8.2.75>>",
                topic=0.9,
                support=0.5,
            ),
            _candidate(
                chunk_id="b",
                doc_id="doc-b",
                filename="接口补充",
                content="所属产品：云枢8>> 产品版本：8.2.75>>",
                topic=0.85,
                support=0.8,
                role="direct",
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢消息接口怎么配置",
            constraints=extract_query_constraints("云枢消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "single_or_overlapping_scope")

    def test_same_version_explicit_different_projects_require_clarification(self) -> None:
        candidates = [
            _candidate(
                chunk_id="project-a",
                doc_id="doc-a",
                filename="甲项目配置",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：中青建安>"
                ),
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="project-b",
                doc_id="doc-b",
                filename="乙项目配置",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：华东示范项目>"
                ),
                topic=0.88,
                support=0.75,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢消息接口怎么配置",
            constraints=extract_query_constraints("云枢消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "project")
        self.assertEqual(
            {choice.projects for choice in decision.choices},
            {("中青建安",), ("华东示范项目",)},
        )

    def test_same_version_same_project_documents_remain_complementary(self) -> None:
        candidates = [
            _candidate(
                chunk_id="project-a-one",
                doc_id="doc-a-one",
                filename="接口配置",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：中青建安>"
                ),
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="project-a-two",
                doc_id="doc-a-two",
                filename="接口参数补充",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：中青建安>"
                ),
                topic=0.85,
                support=0.75,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢消息接口怎么配置",
            constraints=extract_query_constraints("云枢消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)

    def test_project_template_placeholder_does_not_create_false_scope(self) -> None:
        candidates = [
            _candidate(
                chunk_id="template",
                doc_id="doc-template",
                filename="通用问题模板",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：<出现问题的项目，非必填>>"
                ),
                topic=0.9,
                support=0.6,
            ),
            _candidate(
                chunk_id="real-project",
                doc_id="doc-real-project",
                filename="项目配置",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：中青建安>"
                ),
                topic=0.92,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢消息接口怎么配置",
            constraints=extract_query_constraints("云枢消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "single_or_overlapping_scope")

    def test_explicit_version_filters_other_versions_without_clarifying(self) -> None:
        candidates = [
            _candidate(
                chunk_id="v6",
                doc_id="doc-v6",
                filename="旧版",
                content="所属产品：云枢6>> 产品版本：6.0.1>>",
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="v8",
                doc_id="doc-v8",
                filename="新版",
                content="所属产品：云枢8>> 产品版本：8.2.75>>",
                topic=0.9,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢8.2.75消息接口怎么配置",
            constraints=extract_query_constraints("云枢8.2.75消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "single_or_overlapping_scope")

    def test_explicit_version_still_clarifies_different_projects(self) -> None:
        candidates = [
            _candidate(
                chunk_id="project-a-v8",
                doc_id="doc-a-v8",
                filename="甲项目配置",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：中青建安>"
                ),
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="project-b-v8",
                doc_id="doc-b-v8",
                filename="乙项目配置",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：华东示范项目>"
                ),
                topic=0.9,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢8.2.75消息接口怎么配置",
            constraints=extract_query_constraints("云枢8.2.75消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "project")

    def test_explicit_project_filters_other_projects_without_clarifying(self) -> None:
        candidates = [
            _candidate(
                chunk_id="project-a-explicit",
                doc_id="doc-a-explicit",
                filename="甲项目配置",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：中青建安>"
                ),
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="project-b-explicit",
                doc_id="doc-b-explicit",
                filename="乙项目配置",
                content=(
                    "所属产品：云枢8>> 产品版本：8.2.75>> "
                    "所属项目：华东示范项目>"
                ),
                topic=0.9,
                support=0.8,
            ),
        ]

        query = "中青建安的云枢8.2.75消息接口怎么配置"
        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.relevant_document_count, 1)
        self.assertEqual(decision.allowed_doc_ids, ("doc-a-explicit",))

    def test_explicit_all_versions_request_does_not_clarify(self) -> None:
        candidates = [
            _candidate(
                chunk_id="v6",
                doc_id="doc-v6",
                filename="旧版",
                content="所属产品：云枢6>> 产品版本：6.0.1>>",
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="v8",
                doc_id="doc-v8",
                filename="新版",
                content="所属产品：云枢8>> 产品版本：8.2.75>>",
                topic=0.9,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="请分别对比云枢所有版本的消息接口",
            constraints=extract_query_constraints(
                "请分别对比云枢所有版本的消息接口"
            ),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "query_requests_all_scopes")

    def test_colloquial_all_scope_requests_do_not_clarify(self) -> None:
        candidates = [
            _candidate(
                chunk_id="v6",
                doc_id="doc-v6",
                filename="旧版",
                content="所属产品：云枢6>> 产品版本：6.0.1>>",
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="v8",
                doc_id="doc-v8",
                filename="新版",
                content="所属产品：云枢8>> 产品版本：8.2.75>>",
                topic=0.9,
                support=0.8,
            ),
        ]

        for query in ("两个都要", "都查一下", "都看看", "都对比"):
            with self.subTest(query=query):
                decision = detect_evidence_scope_ambiguity(
                    query=query,
                    constraints=extract_query_constraints(query),
                    candidates=candidates,
                )
                self.assertFalse(decision.needs_clarification)
                self.assertEqual(decision.reason, "query_requests_all_scopes")

    def test_feature_list_with_douyao_does_not_disable_scope_clarification(self) -> None:
        candidates = [
            _candidate(
                chunk_id="feature-v1",
                doc_id="doc-feature-v1",
                filename="旧版安全配置",
                content="所属产品：产品A；产品版本：1.0",
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="feature-v2",
                doc_id="doc-feature-v2",
                filename="新版安全配置",
                content="所属产品：产品A；产品版本：2.0",
                topic=0.9,
                support=0.8,
            ),
        ]

        for query in (
            "账号锁定和密码策略都要怎么配置",
            "版本控制和账号策略都要配置",
            "项目管理和登录安全都要配置",
            "产品编码和权限策略都要配置",
        ):
            with self.subTest(query=query):
                self.assertFalse(query_requests_all_scopes(query))
                decision = detect_evidence_scope_ambiguity(
                    query=query,
                    constraints=extract_query_constraints(query),
                    candidates=candidates,
                )
                self.assertTrue(decision.needs_clarification)

    def test_verified_scope_uses_same_topic_gate_as_pipeline(self) -> None:
        candidates = [
            _candidate(
                chunk_id="threshold-v1",
                doc_id="doc-threshold-v1",
                filename="版本一",
                content="所属产品：产品A；产品版本：1.0",
                topic=0.55,
                support=0.9,
                role="direct",
            ),
            _candidate(
                chunk_id="threshold-v2",
                doc_id="doc-threshold-v2",
                filename="版本二",
                content="所属产品：产品A；产品版本：2.0",
                topic=0.55,
                support=0.9,
                role="direct",
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="产品A的安全配置",
            constraints=extract_query_constraints("产品A的安全配置"),
            candidates=candidates,
        )
        self.assertTrue(decision.needs_clarification)

    def test_productless_versioned_policies_still_clarify(self) -> None:
        candidates = [
            _candidate(
                chunk_id="policy-2024",
                doc_id="doc-policy-2024",
                filename="差旅制度2024版",
                content="普通员工住宿标准",
                metadata={"version": "2024"},
                topic=0.95,
                support=0.9,
                role="direct",
            ),
            _candidate(
                chunk_id="policy-2025",
                doc_id="doc-policy-2025",
                filename="差旅制度2025版",
                content="普通员工住宿标准",
                metadata={"version": "2025"},
                topic=0.95,
                support=0.9,
                role="direct",
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="普通员工的出差标准是什么",
            constraints=extract_query_constraints("普通员工的出差标准是什么"),
            candidates=candidates,
        )
        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "version")

        explicit = detect_evidence_scope_ambiguity(
            query="普通员工的2025版出差标准是什么",
            constraints=extract_query_constraints("普通员工的2025版出差标准是什么"),
            candidates=candidates,
        )
        self.assertFalse(explicit.needs_clarification)
        self.assertEqual(explicit.relevant_document_count, 1)
        self.assertEqual(
            explicit.allowed_doc_ids,
            ("doc-policy-2025",),
        )
        self.assertEqual(
            explicit.to_dict()["allowed_doc_ids"],
            ["doc-policy-2025"],
        )

        arbitrary_number = detect_evidence_scope_ambiguity(
            query="出差需要2天怎么报销",
            constraints=extract_query_constraints("出差需要2天怎么报销"),
            candidates=candidates,
        )
        self.assertTrue(arbitrary_number.needs_clarification)

    def test_filename_and_source_labels_anchor_version_ambiguity(self) -> None:
        # Ingestion often places applicability only in the filename/source
        # label; the answer chunks themselves may not repeat a header field.
        candidates = [
            _candidate(
                chunk_id="filename-v6",
                doc_id="doc-filename-v6",
                filename="云枢6配置参数说明",
                content="登录用户名枚举配置说明",
                metadata={
                    "source": "云枢6配置参数说明",
                    "heading": "云枢6配置参数说明 › 四、解决方案",
                },
                topic=0.9,
                support=0.8,
                role="direct",
            ),
            _candidate(
                chunk_id="filename-v7",
                doc_id="doc-filename-v7",
                filename="云枢7配置",
                content="登录用户名枚举配置说明",
                metadata={"source": "云枢7配置"},
                topic=0.9,
                support=0.8,
                role="direct",
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="登录用户名枚举要配置什么",
            constraints=extract_query_constraints("登录用户名枚举要配置什么"),
            candidates=candidates,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "version")
        self.assertEqual(
            {choice.versions for choice in decision.choices},
            {("6",), ("7",)},
        )

    def test_filename_only_versions_are_hard_limited_in_comparison(self) -> None:
        candidates = [
            _candidate(
                chunk_id=f"filename-cmp-{version}",
                doc_id=f"doc-filename-cmp-{version}",
                filename=f"云枢{version}配置",
                content="登录安全配置说明",
                metadata={"source": f"云枢{version}配置"},
                topic=0.9,
                support=0.8,
                role="direct",
            )
            for version in ("6", "7", "8")
        ]
        query = "比较云枢6和云枢7的登录安全配置"
        plan = resolve_explicit_scope_comparison(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )

        self.assertTrue(plan.matched)
        self.assertEqual(
            set(plan.allowed_doc_ids),
            {"doc-filename-cmp-6", "doc-filename-cmp-7"},
        )
        self.assertNotIn("doc-filename-cmp-8", plan.allowed_doc_ids)

    def test_overlapping_multi_product_document_bridges_product_groups(self) -> None:
        candidates = [
            _candidate(
                chunk_id="product-a",
                doc_id="doc-product-a",
                filename="产品A配置",
                content="配置说明",
                metadata={"product": "产品A"},
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="product-b",
                doc_id="doc-product-b",
                filename="产品B配置",
                content="配置说明",
                metadata={"product": "产品B"},
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="product-ab",
                doc_id="doc-product-ab",
                filename="产品兼容说明",
                content="兼容说明",
                metadata={"product": ["产品A", "产品B"]},
                topic=0.9,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="账号安全如何配置",
            constraints=extract_query_constraints("账号安全如何配置"),
            candidates=candidates,
        )
        self.assertFalse(decision.needs_clarification)

    def test_unversioned_companion_is_shared_by_every_version_choice(self) -> None:
        candidates = [
            _candidate(
                chunk_id="shared",
                doc_id="doc-shared",
                filename="通用前置条件",
                content="所属产品：产品A；通用前置条件",
                topic=0.9,
                support=0.7,
            ),
            _candidate(
                chunk_id="shared-v1",
                doc_id="doc-shared-v1",
                filename="版本一",
                content="所属产品：产品A；产品版本：1.0",
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="shared-v2",
                doc_id="doc-shared-v2",
                filename="版本二",
                content="所属产品：产品A；产品版本：2.0",
                topic=0.9,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="产品A如何配置",
            constraints=extract_query_constraints("产品A如何配置"),
            candidates=candidates,
        )
        self.assertTrue(decision.needs_clarification)
        self.assertTrue(all("doc-shared" in choice.doc_ids for choice in decision.choices))
        self.assertTrue(
            all("doc-shared" in choice.companion_doc_ids for choice in decision.choices)
        )
        self.assertTrue(all(choice.anchor_doc_ids for choice in decision.choices))

    def test_numeric_and_text_versions_have_stable_mixed_sorting(self) -> None:
        self.assertEqual(
            sorted(["legacy", "8.2.75", "6.0.1"], key=_version_key),
            ["6.0.1", "8.2.75", "legacy"],
        )

    def test_choice_display_text_is_bounded_for_pending_protocol(self) -> None:
        long_filename = "内部配置" * 120
        candidates = [
            _candidate(
                chunk_id="bounded-v6",
                doc_id="bounded-doc-v6",
                filename=long_filename,
                content="所属产品：云枢6>> 产品版本：6.0.1>>",
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="bounded-v8",
                doc_id="bounded-doc-v8",
                filename=long_filename,
                content="所属产品：云枢8>> 产品版本：8.2.75>>",
                topic=0.9,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢消息接口怎么配置",
            constraints=extract_query_constraints("云枢消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertTrue(decision.needs_clarification)
        for choice in decision.choices:
            self.assertLessEqual(len(choice.label), 500)
            self.assertTrue(all(len(item) <= 500 for item in choice.filenames))

    def test_irrelevant_other_version_does_not_create_false_choice(self) -> None:
        candidates = [
            _candidate(
                chunk_id="answer",
                doc_id="doc-answer",
                filename="消息接口",
                content="所属产品：云枢8>> 产品版本：8.2.75>>",
                topic=0.95,
                support=0.9,
                role="direct",
            ),
            _candidate(
                chunk_id="noise",
                doc_id="doc-noise",
                filename="云枢6数据库配置",
                content="所属产品：云枢6>> 产品版本：6.0.1>>",
                topic=0.2,
                support=0.0,
                role="irrelevant",
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢消息接口怎么配置",
            constraints=extract_query_constraints("云枢消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)

    def test_product_ambiguity_is_driven_by_document_identity(self) -> None:
        candidates = [
            _candidate(
                chunk_id="cloudpivot",
                doc_id="doc-cloudpivot",
                filename="账号安全配置",
                content="账号安全配置",
                metadata={"product": "云枢", "version": "8.2.75"},
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="weaver",
                doc_id="doc-weaver",
                filename="账号安全配置",
                content="账号安全配置",
                metadata={"product": "泛微OA", "version": "10.0"},
                topic=0.9,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="解决登录用户名枚举要配置什么",
            constraints=extract_query_constraints(
                "解决登录用户名枚举要配置什么"
            ),
            candidates=candidates,
        )

        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.dimension, "product_version")
        self.assertEqual(len(decision.choices), 2)

    def test_overlapping_multi_version_documents_are_one_scope_group(self) -> None:
        candidates = [
            _candidate(
                chunk_id="multi",
                doc_id="doc-multi",
                filename="兼容说明",
                content="兼容说明",
                metadata={"product": "云枢", "version": ["7", "8"]},
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="v8",
                doc_id="doc-v8",
                filename="云枢8补充说明",
                content="补充说明",
                metadata={"product": "云枢", "version": "8"},
                topic=0.9,
                support=0.8,
            ),
        ]

        decision = detect_evidence_scope_ambiguity(
            query="云枢消息接口怎么配置",
            constraints=extract_query_constraints("云枢消息接口怎么配置"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)

    def test_enumerated_product_generations_exclude_unmentioned_version(self) -> None:
        candidates = [
            _candidate(
                chunk_id="shared",
                doc_id="doc-shared",
                filename="通用前置条件",
                content="通用说明",
                metadata={"product": "云枢"},
                topic=0.9,
                support=0.7,
            ),
            *[
                _candidate(
                    chunk_id=f"version-{version}",
                    doc_id=f"doc-{generation}",
                    filename=f"云枢{generation}说明",
                    content="产品说明",
                    metadata={"product": "云枢", "version": version},
                    topic=0.9,
                    support=0.8,
                )
                for generation, version in (
                    ("6", "6.0.1"),
                    ("7", "7.1.0"),
                    ("8", "8.2.75"),
                )
            ],
        ]

        query = "对比云枢6和云枢8的配置差异"
        plan = resolve_explicit_scope_comparison(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )

        self.assertTrue(plan.matched)
        self.assertEqual(plan.reason, "explicit_enumerated_scopes")
        self.assertEqual(plan.dimension, "version")
        self.assertEqual(
            {choice.versions for choice in plan.choices},
            {("6.0.1",), ("8.2.75",)},
        )
        self.assertEqual(
            set(plan.allowed_doc_ids),
            {"doc-6", "doc-8", "doc-shared"},
        )
        self.assertNotIn("doc-7", plan.allowed_doc_ids)
        self.assertTrue(
            all(
                choice.companion_doc_ids == ("doc-shared",)
                for choice in plan.choices
            )
        )
        self.assertEqual(
            {choice.anchor_doc_ids for choice in plan.choices},
            {("doc-6",), ("doc-8",)},
        )

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )
        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "explicit_enumerated_scopes")
        self.assertEqual(len(decision.choices), 2)
        self.assertEqual(
            set(decision.allowed_doc_ids),
            {"doc-6", "doc-8", "doc-shared"},
        )

    def test_v6_v7_comparison_decision_carries_allowed_documents(self) -> None:
        candidates = [
            _candidate(
                chunk_id=f"cloudpivot-{version}",
                doc_id=f"doc-cloudpivot-{version}",
                filename=f"CloudPivot {version} 安全配置",
                content="安全配置说明",
                metadata={"product": "CloudPivot", "version": version},
                topic=0.9,
                support=0.8,
            )
            for version in ("6", "7", "8")
        ]
        query = "比较 CloudPivot 6 和 CloudPivot 7 的安全配置"

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "explicit_enumerated_scopes")
        self.assertEqual(
            {choice.versions for choice in decision.choices},
            {("6",), ("7",)},
        )
        self.assertEqual(
            decision.allowed_doc_ids,
            ("doc-cloudpivot-6", "doc-cloudpivot-7"),
        )
        self.assertEqual(
            decision.to_dict()["allowed_doc_ids"],
            ["doc-cloudpivot-6", "doc-cloudpivot-7"],
        )

    def test_source_versions_generate_only_unique_prefix_aliases(self) -> None:
        candidates = [
            _candidate(
                chunk_id="v6",
                doc_id="doc-v6",
                filename="版本六",
                content="配置说明",
                metadata={"product": "云枢", "version": "6.0.1"},
                topic=0.9,
                support=0.8,
            ),
            _candidate(
                chunk_id="v8",
                doc_id="doc-v8",
                filename="版本八",
                content="配置说明",
                metadata={"product": "云枢", "version": "8.2.75"},
                topic=0.9,
                support=0.8,
            ),
        ]

        for query in (
            "对比云枢6和云枢8",
            "比较云枢6.0和云枢8.2",
            "版本6.0.1与8.2.75版有什么区别",
            "v6 vs v8",
            "云枢6和云枢8都要配置登录安全",
            "云枢6与云枢8都需要配置登录安全",
        ):
            with self.subTest(query=query):
                plan = resolve_explicit_scope_comparison(
                    query=query,
                    constraints=extract_query_constraints(query),
                    candidates=candidates,
                )
                self.assertTrue(plan.matched)
                self.assertEqual(
                    {choice.versions for choice in plan.choices},
                    {("6.0.1",), ("8.2.75",)},
                )

    def test_ambiguous_version_prefix_is_not_guessed(self) -> None:
        candidates = [
            _candidate(
                chunk_id=f"v-{version}",
                doc_id=f"doc-{version}",
                filename=f"版本{version}",
                content="配置说明",
                metadata={"product": "云枢", "version": version},
                topic=0.9,
                support=0.8,
            )
            for version in ("8.2.1", "8.2.9", "8.6.1")
        ]

        plan = resolve_explicit_scope_comparison(
            query="对比v8.2和v8.6",
            constraints=extract_query_constraints("对比v8.2和v8.6"),
            candidates=candidates,
        )

        self.assertFalse(plan.matched)
        self.assertEqual(plan.reason, "enumerated_scope_aliases_not_unique")

    def test_enumerated_projects_exclude_unmentioned_project(self) -> None:
        candidates = [
            _candidate(
                chunk_id="project-shared",
                doc_id="doc-project-shared",
                filename="通用配置",
                content="通用配置",
                metadata={"product": "平台A", "version": "2.0"},
                topic=0.9,
                support=0.7,
            ),
            *[
                _candidate(
                    chunk_id=f"project-{project}",
                    doc_id=f"doc-{project}",
                    filename=f"{project}配置",
                    content="项目配置",
                    metadata={
                        "product": "平台A",
                        "version": "2.0",
                        "project": project,
                    },
                    topic=0.9,
                    support=0.8,
                )
                for project in ("甲项目", "乙项目", "丙项目")
            ],
        ]

        query = "对比甲项目和乙项目的配置差异"
        plan = resolve_explicit_scope_comparison(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )

        self.assertTrue(plan.matched)
        self.assertEqual(plan.dimension, "project")
        self.assertEqual(
            {choice.projects for choice in plan.choices},
            {("甲项目",), ("乙项目",)},
        )
        self.assertNotIn("doc-丙项目", plan.allowed_doc_ids)
        self.assertIn("doc-project-shared", plan.allowed_doc_ids)
        self.assertTrue(
            all(
                "doc-project-shared" in choice.companion_doc_ids
                for choice in plan.choices
            )
        )

    def test_all_versions_are_scoped_by_named_source_product(self) -> None:
        candidates = [
            *[
                _candidate(
                    chunk_id=f"cloud-{version}",
                    doc_id=f"doc-cloud-{version}",
                    filename=f"云枢{version}",
                    content="配置说明",
                    metadata={"product": "云枢", "version": version},
                    topic=0.9,
                    support=0.8,
                )
                for version in ("6.0.1", "7.1.0", "8.2.75")
            ],
            _candidate(
                chunk_id="other-product",
                doc_id="doc-other-product",
                filename="其他产品",
                content="配置说明",
                metadata={"product": "其他平台", "version": "10.0"},
                topic=0.9,
                support=0.8,
            ),
        ]

        query = "请对比云枢所有版本的配置"
        plan = resolve_explicit_scope_comparison(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )

        self.assertTrue(plan.matched)
        self.assertEqual(plan.reason, "explicit_all_scopes")
        self.assertEqual(plan.dimension, "version")
        self.assertEqual(len(plan.choices), 3)
        self.assertNotIn("doc-other-product", plan.allowed_doc_ids)

        decision = detect_evidence_scope_ambiguity(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )
        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.reason, "query_requests_all_scopes")
        self.assertEqual(decision.dimension, "version")
        self.assertEqual(len(decision.choices), 3)

    def test_all_projects_keep_every_project_in_named_product_version(self) -> None:
        candidates = [
            *[
                _candidate(
                    chunk_id=f"project-all-{project}",
                    doc_id=f"doc-project-all-{project}",
                    filename=f"{project}配置",
                    content="项目配置",
                    metadata={
                        "product": "云枢",
                        "version": version,
                        "project": project,
                    },
                    topic=0.9,
                    support=0.8,
                )
                for project, version in (
                    ("甲项目", "8.2.75"),
                    ("乙项目", "8.2.75"),
                    ("丙项目", "8.2.75"),
                    ("丁项目", "9.0"),
                )
            ],
        ]

        query = "云枢8.2的所有项目都对比"
        plan = resolve_explicit_scope_comparison(
            query=query,
            constraints=extract_query_constraints(query),
            candidates=candidates,
        )

        self.assertTrue(plan.matched)
        self.assertEqual(plan.reason, "explicit_all_scopes")
        self.assertEqual(plan.dimension, "project")
        self.assertEqual(
            {choice.projects for choice in plan.choices},
            {('甲项目',), ('乙项目',), ('丙项目',)},
        )
        self.assertNotIn("doc-project-all-丁项目", plan.allowed_doc_ids)

    def test_more_than_six_explicit_scopes_never_returns_partial_plan(self) -> None:
        candidates = [
            _candidate(
                chunk_id=f"many-{version}",
                doc_id=f"doc-many-{version}",
                filename=f"版本{version}",
                content="配置说明",
                metadata={"product": "平台A", "version": version},
                topic=0.9,
                support=0.8,
            )
            for version in ("1", "2", "3", "4", "5", "6", "7")
        ]

        plan = resolve_explicit_scope_comparison(
            query="平台A所有版本都对比",
            constraints=extract_query_constraints("平台A所有版本都对比"),
            candidates=candidates,
        )

        self.assertFalse(plan.matched)
        self.assertEqual(
            plan.reason,
            "too_many_explicit_scopes_for_complete_plan",
        )
        self.assertEqual(plan.choices, ())
        self.assertEqual(plan.allowed_doc_ids, ())
        all_decision = detect_evidence_scope_ambiguity(
            query="平台A所有版本都对比",
            constraints=extract_query_constraints("平台A所有版本都对比"),
            candidates=candidates,
        )
        self.assertTrue(all_decision.needs_clarification)
        self.assertEqual(all_decision.dimension, "version")
        self.assertEqual(
            all_decision.reason,
            "too_many_mutually_exclusive_scopes",
        )
        self.assertEqual(all_decision.choices, ())

        enumerated_query = "对比" + "、".join(
            f"v{version}" for version in ("1", "2", "3", "4", "5", "6", "7")
        )
        enumerated_plan = resolve_explicit_scope_comparison(
            query=enumerated_query,
            constraints=extract_query_constraints(enumerated_query),
            candidates=candidates,
        )
        self.assertFalse(enumerated_plan.matched)
        self.assertEqual(
            enumerated_plan.reason,
            "too_many_explicit_scopes_for_complete_plan",
        )
        self.assertEqual(enumerated_plan.choices, ())
        enumerated_decision = detect_evidence_scope_ambiguity(
            query=enumerated_query,
            constraints=extract_query_constraints(enumerated_query),
            candidates=candidates,
        )
        self.assertTrue(enumerated_decision.needs_clarification)
        self.assertEqual(enumerated_decision.choices, ())

    def test_duration_numbers_do_not_select_real_version_one_or_two(self) -> None:
        candidates = [
            _candidate(
                chunk_id=f"policy-{version}",
                doc_id=f"doc-policy-{version}",
                filename=f"差旅制度{version}版",
                content="出差报销规定",
                metadata={"version": version},
                topic=0.95,
                support=0.9,
                role="direct",
            )
            for version in ("1", "2")
        ]

        for query in ("出差需要2天怎么报销", "对比出差2天和3天的差异"):
            with self.subTest(query=query):
                plan = resolve_explicit_scope_comparison(
                    query=query,
                    constraints=extract_query_constraints(query),
                    candidates=candidates,
                )
                self.assertFalse(plan.matched)
                decision = detect_evidence_scope_ambiguity(
                    query=query,
                    constraints=extract_query_constraints(query),
                    candidates=candidates,
                )
                self.assertTrue(decision.needs_clarification)
                self.assertEqual(len(decision.choices), 2)

    def test_explicit_year_version_still_keeps_only_that_source_scope(self) -> None:
        candidates = [
            _candidate(
                chunk_id=f"year-{version}",
                doc_id=f"doc-year-{version}",
                filename=f"差旅制度{version}版",
                content="出差标准",
                metadata={"version": version},
                topic=0.95,
                support=0.9,
                role="direct",
            )
            for version in ("2024", "2025")
        ]

        decision = detect_evidence_scope_ambiguity(
            query="2025版出差标准",
            constraints=extract_query_constraints("2025版出差标准"),
            candidates=candidates,
        )

        self.assertFalse(decision.needs_clarification)
        self.assertEqual(decision.relevant_document_count, 1)
        self.assertEqual(decision.allowed_doc_ids, ("doc-year-2025",))


if __name__ == "__main__":
    unittest.main()
