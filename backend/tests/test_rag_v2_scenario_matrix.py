"""Cross-domain acceptance matrix for the RAG v2 safety contracts.

The cases in this module intentionally span several policy domains.  They are
kept together so a future implementation cannot make the happy path pass by
special-casing one phrase such as ``普通员工餐补``.
"""

from dataclasses import dataclass, replace
import unittest

from core.query_route_compiler import (
    RouteCategoryPolicy,
    RouteCompilerConfig,
    compile_rag_task_contract,
    is_rag_task_contract_dispatchable,
    rag_task_contract_gate_reason,
)
from core.query_route_contract import parse_rag_route_decision
from core.rag_v2.evidence import assemble_evidence_bundle
from core.rag_v2.query_plan import plan_query_locally


@dataclass(frozen=True)
class MappingScenario:
    name: str
    question: str
    resolved_value: str
    bridge_content: str
    answer_content: str
    wrong_answer_content: str


@dataclass(frozen=True)
class BusinessEvidenceScenario:
    name: str
    question: str
    answer_shape: str
    supporting_contents: tuple[str, ...]
    unrelated_content: str


MAPPING_SCENARIOS = (
    MappingScenario(
        name="ordinary_employee_trip",
        question="普通员工的出差标准是什么",
        resolved_value="D级",
        bridge_content="职级分类：普通员工对应D级。",
        answer_content=(
            "D级出差标准：飞机经济舱、高铁二等座，住宿一线城市不超过"
            "450元/天，餐补100元/天。"
        ),
        wrong_answer_content="A级出差标准：飞机公务舱，住宿不超过1200元/天。",
    ),
    MappingScenario(
        name="ordinary_employee_lodging",
        question="普通员工住宿标准是多少",
        resolved_value="D级",
        bridge_content="职级分类：普通员工对应D级。",
        answer_content="住宿标准：D级一线城市不超过450元/天。",
        wrong_answer_content="住宿标准：A级一线城市不超过1200元/天。",
    ),
    MappingScenario(
        name="ordinary_employee_transport",
        question="普通员工交通标准是什么",
        resolved_value="D级",
        bridge_content="职级分类：普通员工对应D级。",
        answer_content="交通标准：D级乘飞机经济舱、高铁二等座。",
        wrong_answer_content="交通标准：A级可乘飞机公务舱、高铁一等座。",
    ),
    MappingScenario(
        name="contractor_lodging",
        question="合同工住宿标准是多少",
        resolved_value="L2类",
        bridge_content="用工分类：合同工属于L2类。",
        answer_content="住宿标准：L2类不超过300元/天。",
        wrong_answer_content="住宿标准：L1类不超过800元/天。",
    ),
    MappingScenario(
        name="probation_annual_leave",
        question="试用期年假天数是多少",
        resolved_value="P0阶段",
        bridge_content="员工阶段：试用期属于P0阶段。",
        answer_content="年假天数：P0阶段为0天。",
        wrong_answer_content="年假天数：P1阶段为5天。",
    ),
    MappingScenario(
        name="regional_lodging",
        question="华东地区住宿标准是多少",
        resolved_value="R2档",
        bridge_content="地区分类：华东地区适用R2档。",
        answer_content="住宿标准：R2档不超过500元/天。",
        wrong_answer_content="住宿标准：R1档不超过900元/天。",
    ),
    MappingScenario(
        name="department_grade_approval_limit",
        question="研发部门专员的审批额度是多少",
        resolved_value="E2级",
        bridge_content="职级分类：研发部门专员对应E2级。",
        answer_content="审批额度：E2级单笔上限5000元。",
        wrong_answer_content="审批额度：E1级单笔上限1000元。",
    ),
    MappingScenario(
        name="department_manager_lodging",
        question="部门经理的住宿标准是多少",
        resolved_value="C级",
        bridge_content="职级分类：部门经理对应C级。",
        answer_content="住宿标准：C级一线城市不超过600元/天。",
        wrong_answer_content="住宿标准：A级一线城市不超过1200元/天。",
    ),
)


BUSINESS_EVIDENCE_SCENARIOS = (
    BusinessEvidenceScenario(
        name="purchase_approval_limit",
        question="采购申请单笔审批额度是多少",
        answer_shape="fact",
        supporting_contents=(
            "采购审批制度：单笔采购申请金额不超过5000元的，由部门经理审批。",
        ),
        unrelated_content="公司印章由行政部门统一保管。",
    ),
    BusinessEvidenceScenario(
        name="reimbursement_deadline_and_receipts",
        question="报销提交时限是多久？需要提供哪些凭证？",
        answer_shape="multi_part",
        supporting_contents=(
            "费用报销时限：出差结束后5个工作日内提交。",
            "报销凭证：必须提供正规发票、行程单及住宿发票。",
        ),
        unrelated_content="固定资产每年盘点一次。",
    ),
    BusinessEvidenceScenario(
        name="leave_approval_process",
        question="员工请假审批流程是什么",
        answer_shape="process",
        supporting_contents=(
            "请假审批流程：员工提交申请，直属主管审批，三天以上再由部门负责人审批。",
        ),
        unrelated_content="办公用品由行政部门按季度统一采购。",
    ),
    BusinessEvidenceScenario(
        name="login_username_enumeration",
        question="如何配置登录用户名枚举防护",
        answer_shape="process",
        supporting_contents=(
            "登录用户名枚举防护配置：将 error_reply_same 设置为 true，"
            "使账号不存在与密码错误返回相同提示。",
        ),
        unrelated_content="数据库备份任务每天凌晨执行。",
    ),
)


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


def _candidate(
    chunk_id: str,
    content: str,
    *,
    chunk_index: int = 0,
    doc_id: str = "scenario-document",
    **values,
) -> dict:
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "kb_id": "authorized-knowledge-base",
        "chunk_index": chunk_index,
        "content": content,
        **values,
    }


def _route(question: str, *, intent_code: str, evidence_scope: str):
    return parse_rag_route_decision(
        {
            "schema_version": "rag_route_decision.v1",
            "readiness": "ready",
            "intent_code": intent_code,
            "relation": "new",
            "evidence_scope": evidence_scope,
            "query_resolution": {
                "mode": "current",
                "context_turn_keys": [],
            },
            "requirements": [
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": question,
                }
            ],
            "clarification": {"question": "", "unresolved": []},
            "confidence": 0.95,
            "rationale": "跨场景测试路由",
        },
        allowed_intent_codes=list(CATEGORIES),
        available_turn_keys=[],
    )


def _compile(
    question: str,
    *,
    intent_code: str = "knowledge_qa",
    evidence_scope: str = "enterprise_kb",
    selected_kb_count: int = 1,
    **guards,
):
    route = _route(
        question,
        intent_code=intent_code,
        evidence_scope=evidence_scope,
    )
    return compile_rag_task_contract(
        route,
        CATEGORIES[intent_code],
        RouteCompilerConfig(),
        question=question,
        selected_kb_count=selected_kb_count,
        available_turn_keys=[],
        source="test_matrix",
        **guards,
    )


class RagV2CrossDomainPlanningMatrixTests(unittest.TestCase):
    def test_implicit_mapping_questions_always_plan_answer_and_bridge(self) -> None:
        for scenario in MAPPING_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                plan = plan_query_locally(scenario.question)

                self.assertEqual(plan.answer_shape, "multi_hop")
                self.assertEqual(
                    [item.role for item in plan.requirements],
                    ["answer", "bridge"],
                )
                self.assertTrue(plan.requirements[0].is_required_answer)
                self.assertEqual(plan.requirements[1].importance, "helpful")
                self.assertEqual(plan.requirements[1].source, "inferred")
                self.assertEqual(len(plan.retrieval_queries), 2)
                self.assertNotEqual(
                    plan.retrieval_queries[0],
                    plan.retrieval_queries[1],
                )
                # Planning may require a relationship, but it must never guess
                # the concrete intermediate classification value.
                self.assertNotIn(
                    scenario.resolved_value,
                    plan.requirements[1].description,
                )

    def test_policy_and_security_queries_remain_domain_neutral(self) -> None:
        cases = (
            ("报销需要满足什么条件", "unknown"),
            ("单笔报销额度是多少", "fact"),
            ("报销审批流程是什么", "process"),
            ("如何配置登录用户名枚举防护", "process"),
        )

        for question, expected_shape in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertEqual(plan.answer_shape, expected_shape)
                self.assertEqual(len(plan.requirements), 1)
                self.assertEqual(plan.requirements[0].role, "answer")
                self.assertFalse(any(
                    item.role == "bridge" for item in plan.requirements
                ))

    def test_possessive_owner_phrases_do_not_invent_a_taxonomy(self) -> None:
        cases = (
            "公司的出差管理标准是什么",
            "企业的费用标准是什么",
            "集团的审批规则是什么",
            "组织的访问控制规则是什么",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertNotEqual(plan.answer_shape, "multi_hop")
                self.assertFalse(any(
                    item.role == "bridge" for item in plan.requirements
                ))

    def test_explicit_scope_and_classifier_phrases_stay_single_requirement(
        self,
    ) -> None:
        cases = (
            "云枢8.6版本的参数上限是多少",
            "D级的餐补是多少",
            "专业版产品的并发上限是多少",
            "重点项目的审批额度是多少",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertNotEqual(plan.answer_shape, "multi_hop")
                self.assertEqual(len(plan.requirements), 1)
                self.assertEqual(plan.requirements[0].role, "answer")


class RagV2CrossDomainDispatchMatrixTests(unittest.TestCase):
    def test_response_modes_do_not_conflate_knowledge_and_direct_requests(self) -> None:
        cases = (
            {
                "name": "login_security_configuration",
                "question": "如何配置登录用户名枚举防护",
                "intent_code": "knowledge_qa",
                "evidence_scope": "enterprise_kb",
                "selected_kb_count": 1,
                "guards": {"requires_knowledge": True},
                "expected": ("grounded_qa", "required", True),
            },
            {
                "name": "knowledge_dependent_writing",
                "question": "请根据公司的出差制度写一份普通员工出差注意事项",
                "intent_code": "writing",
                "evidence_scope": "mixed",
                "selected_kb_count": 1,
                "guards": {
                    "requires_knowledge": True,
                    "knowledge_writing": True,
                },
                "expected": ("writing", "required", True),
            },
            {
                "name": "inline_writing",
                "question": "请把“今天完成接口测试”改写得正式一些",
                "intent_code": "writing",
                "evidence_scope": "current_input",
                "selected_kb_count": 0,
                "guards": {"inline_writing": True},
                "expected": ("writing", "skip", False),
            },
            {
                "name": "ordinary_greeting",
                "question": "你好",
                "intent_code": "general_chat",
                "evidence_scope": "general_world",
                "selected_kb_count": 0,
                "guards": {"explicit_greeting": True},
                "expected": ("general_chat", "skip", False),
            },
            {
                "name": "platform_help",
                "question": "如何创建知识库",
                "intent_code": "system_help",
                "evidence_scope": "platform_self",
                "selected_kb_count": 0,
                "guards": {"explicit_platform_help": True},
                "expected": ("platform_help", "skip", False),
            },
        )

        for case in cases:
            with self.subTest(scenario=case["name"]):
                contract = _compile(
                    case["question"],
                    intent_code=case["intent_code"],
                    evidence_scope=case["evidence_scope"],
                    selected_kb_count=case["selected_kb_count"],
                    **case["guards"],
                )

                self.assertEqual(
                    (
                        contract.response_mode,
                        contract.retrieval_policy,
                        contract.need_retrieval,
                    ),
                    case["expected"],
                )
                self.assertTrue(contract.dispatch_authorized)
                self.assertTrue(is_rag_task_contract_dispatchable(contract))

    def test_compiled_mapping_contract_cannot_drop_its_bridge(self) -> None:
        for scenario in MAPPING_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                contract = _compile(scenario.question)

                self.assertEqual(
                    [item.role for item in contract.requirements],
                    ["answer", "bridge"],
                )
                without_bridge = replace(
                    contract,
                    requirements=tuple(
                        item
                        for item in contract.requirements
                        if item.role != "bridge"
                    ),
                )
                self.assertEqual(
                    rag_task_contract_gate_reason(without_bridge),
                    "implicit_mapping_missing_bridge",
                )
                self.assertFalse(
                    is_rag_task_contract_dispatchable(without_bridge)
                )


class RagV2CrossDomainEvidenceMatrixTests(unittest.TestCase):
    def test_coordinated_employee_standards_require_every_item_and_bridge(
        self,
    ) -> None:
        question = "普通员工出差的住宿、交通和餐补标准分别是多少？"
        plan = plan_query_locally(question)
        candidates = (
            _candidate(
                "lodging",
                "出差住宿标准：D级一线城市不超过450元/天。",
                doc_id="travel-policy",
                expansion_query_indexes=[0],
            ),
            _candidate(
                "transport",
                "出差交通标准：D级飞机经济舱、高铁二等座。",
                chunk_index=1,
                doc_id="travel-policy",
                expansion_query_indexes=[1],
            ),
            _candidate(
                "meal",
                "出差餐补标准：D级每天100元。",
                chunk_index=2,
                doc_id="travel-policy",
                expansion_query_indexes=[2],
            ),
            _candidate(
                "classification",
                "职级分类：普通员工对应D级。",
                doc_id="employee-classification",
                expansion_query_indexes=[3],
            ),
        )

        # Presentation shape and dependency topology are independent: this is
        # a three-part answer whose items all depend on one bridge, not one
        # monolithic multi-hop answer.
        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertTrue(plan.has_bridge_dependencies)
        self.assertEqual(
            [item.role for item in plan.requirements],
            ["answer", "answer", "answer", "bridge"],
        )
        self.assertEqual(len(plan.retrieval_queries), len(plan.requirements))

        complete = assemble_evidence_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            candidates=candidates,
            requirements=plan.requirements,
            retrieval_queries=plan.retrieval_queries,
            completeness="complete",
        )
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(complete.state.completeness, "complete")
        self.assertEqual(
            set(complete.answer_source_ids),
            {"lodging", "transport", "meal", "classification"},
        )

        missing_meal = assemble_evidence_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            candidates=(candidates[0], candidates[1], candidates[3]),
            requirements=plan.requirements,
            retrieval_queries=plan.retrieval_queries,
            completeness="complete",
        )
        self.assertEqual(missing_meal.missing_requirement_ids, ("r3",))
        self.assertEqual(missing_meal.state.completeness, "partial")

        missing_bridge = assemble_evidence_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            candidates=candidates[:3],
            requirements=plan.requirements,
            retrieval_queries=plan.retrieval_queries,
            completeness="complete",
        )
        self.assertEqual(
            missing_bridge.missing_requirement_ids,
            ("r1", "r2", "r3", "r4"),
        )
        self.assertEqual(missing_bridge.state.completeness, "unknown")

    def test_correct_intermediate_value_joins_mapping_and_answer(self) -> None:
        for scenario in MAPPING_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                plan = plan_query_locally(scenario.question)
                bundle = assemble_evidence_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    candidates=(
                        _candidate(
                            "answer",
                            scenario.answer_content,
                            doc_id=f"{scenario.name}-answer-policy",
                            candidate_origins=["initial_retrieval"],
                        ),
                        _candidate(
                            "bridge",
                            scenario.bridge_content,
                            chunk_index=1,
                            doc_id=f"{scenario.name}-classification-table",
                        ),
                    ),
                    requirements=plan.requirements,
                    retrieval_queries=plan.retrieval_queries,
                    completeness="complete",
                )

                by_id = {item.chunk_id: item for item in bundle.items}
                self.assertNotEqual(
                    by_id["answer"].doc_id,
                    by_id["bridge"].doc_id,
                )
                self.assertEqual(bundle.missing_requirement_ids, ())
                self.assertEqual(bundle.state.completeness, "complete")
                self.assertEqual(
                    set(bundle.answer_source_ids),
                    {"answer", "bridge"},
                )
                self.assertIn("r1", by_id["answer"].supports_requirement_ids)
                self.assertIn("r2", by_id["bridge"].supports_requirement_ids)

    def test_bridge_only_is_partial_and_cannot_answer_the_final_target(self) -> None:
        for scenario in MAPPING_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                plan = plan_query_locally(scenario.question)
                bundle = assemble_evidence_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    candidates=(
                        _candidate("bridge", scenario.bridge_content),
                    ),
                    requirements=plan.requirements,
                    retrieval_queries=plan.retrieval_queries,
                    completeness="complete",
                )

                item = bundle.items[0]
                self.assertEqual(item.role, "bridge")
                self.assertEqual(item.supports_requirement_ids, ("r2",))
                self.assertEqual(bundle.missing_requirement_ids, ("r1",))
                self.assertEqual(bundle.state.completeness, "partial")

    def test_mismatched_intermediate_value_cannot_complete_answer(self) -> None:
        for scenario in MAPPING_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                plan = plan_query_locally(scenario.question)
                bundle = assemble_evidence_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    candidates=(
                        _candidate(
                            "wrong-answer",
                            scenario.wrong_answer_content,
                            doc_id=f"{scenario.name}-wrong-answer-policy",
                            candidate_origins=["initial_retrieval"],
                        ),
                        _candidate(
                            "bridge",
                            scenario.bridge_content,
                            chunk_index=1,
                            doc_id=f"{scenario.name}-classification-table",
                        ),
                    ),
                    requirements=plan.requirements,
                    retrieval_queries=plan.retrieval_queries,
                    completeness="complete",
                )

                by_id = {item.chunk_id: item for item in bundle.items}
                self.assertNotEqual(
                    by_id["wrong-answer"].doc_id,
                    by_id["bridge"].doc_id,
                )
                self.assertEqual(bundle.missing_requirement_ids, ("r1",))
                self.assertEqual(bundle.state.completeness, "partial")
                self.assertNotIn(
                    "r1",
                    by_id["wrong-answer"].supports_requirement_ids,
                )
                self.assertNotIn("wrong-answer", bundle.answer_source_ids)

    def test_business_evidence_sources_exclude_unrelated_fragments(self) -> None:
        for scenario in BUSINESS_EVIDENCE_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                plan = plan_query_locally(scenario.question)
                support_ids = tuple(
                    f"{scenario.name}-support-{index}"
                    for index in range(1, len(scenario.supporting_contents) + 1)
                )
                support_candidates = tuple(
                    _candidate(
                        chunk_id,
                        content,
                        doc_id=f"{scenario.name}-policy-{index}",
                    )
                    for index, (chunk_id, content) in enumerate(
                        zip(support_ids, scenario.supporting_contents),
                        start=1,
                    )
                )
                distractor_id = f"{scenario.name}-unrelated"
                distractor = _candidate(
                    distractor_id,
                    scenario.unrelated_content,
                    doc_id=f"{scenario.name}-unrelated-document",
                )

                self.assertEqual(plan.answer_shape, scenario.answer_shape)
                self.assertEqual(
                    len(plan.requirements),
                    len(scenario.supporting_contents),
                )

                bundle = assemble_evidence_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    candidates=(*support_candidates, distractor),
                    requirements=plan.requirements,
                    retrieval_queries=plan.retrieval_queries,
                    completeness="complete",
                )

                by_id = {item.chunk_id: item for item in bundle.items}
                self.assertEqual(bundle.missing_requirement_ids, ())
                self.assertEqual(bundle.state.completeness, "complete")
                self.assertEqual(bundle.answer_source_ids, support_ids)
                for index, chunk_id in enumerate(support_ids, start=1):
                    self.assertEqual(
                        by_id[chunk_id].supports_requirement_ids,
                        (f"r{index}",),
                    )
                self.assertEqual(
                    by_id[distractor_id].supports_requirement_ids,
                    (),
                )
                self.assertEqual(by_id[distractor_id].role, "background")
                self.assertNotIn(distractor_id, bundle.answer_source_ids)

    def test_business_evidence_missing_required_fragment_is_partial(self) -> None:
        for scenario in BUSINESS_EVIDENCE_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                plan = plan_query_locally(scenario.question)
                retained_contents = scenario.supporting_contents[:-1]
                candidates = tuple(
                    _candidate(
                        f"{scenario.name}-support-{index}",
                        content,
                        doc_id=f"{scenario.name}-policy-{index}",
                    )
                    for index, content in enumerate(retained_contents, start=1)
                ) + (
                    _candidate(
                        f"{scenario.name}-unrelated",
                        scenario.unrelated_content,
                        doc_id=f"{scenario.name}-unrelated-document",
                    ),
                )

                bundle = assemble_evidence_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    candidates=candidates,
                    requirements=plan.requirements,
                    retrieval_queries=plan.retrieval_queries,
                    completeness="complete",
                )

                self.assertEqual(
                    bundle.answer_source_ids,
                    tuple(
                        f"{scenario.name}-support-{index}"
                        for index in range(1, len(scenario.supporting_contents))
                    ),
                )
                self.assertEqual(
                    bundle.missing_requirement_ids,
                    (f"r{len(scenario.supporting_contents)}",),
                )
                self.assertEqual(
                    bundle.state.completeness,
                    (
                        "unknown"
                        if len(scenario.supporting_contents) == 1
                        else "partial"
                    ),
                )

    def test_multi_part_reimbursement_coverage_is_all_or_partial_per_item(self) -> None:
        question = (
            "报销需要满足什么条件？单笔报销额度是多少？"
            "报销审批流程是什么？"
        )
        plan = plan_query_locally(question)
        candidates = (
            _candidate(
                "condition",
                "报销条件：需要正规发票和费用明细。",
                expansion_query_indexes=[0],
            ),
            _candidate(
                "limit",
                "单笔报销额度：不超过5000元。",
                chunk_index=1,
                expansion_query_indexes=[1],
            ),
            _candidate(
                "approval",
                "报销审批流程：直属主管审批后由财务复核。",
                chunk_index=2,
                expansion_query_indexes=[2],
            ),
        )

        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertEqual(len(plan.requirements), 3)

        complete = assemble_evidence_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            candidates=candidates,
            requirements=plan.requirements,
            retrieval_queries=plan.retrieval_queries,
            completeness="complete",
        )
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(complete.state.completeness, "complete")
        self.assertEqual(
            set(complete.answer_source_ids),
            {"condition", "limit", "approval"},
        )

        missing_approval = assemble_evidence_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            candidates=candidates[:2],
            requirements=plan.requirements,
            retrieval_queries=plan.retrieval_queries,
            completeness="complete",
        )
        self.assertEqual(missing_approval.missing_requirement_ids, ("r3",))
        self.assertEqual(missing_approval.state.completeness, "partial")
        self.assertNotIn("approval", missing_approval.answer_source_ids)


if __name__ == "__main__":
    unittest.main()
