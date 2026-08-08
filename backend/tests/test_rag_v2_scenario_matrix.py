"""Cross-domain acceptance matrix for the RAG v2 safety contracts.

The cases in this module intentionally span several policy domains.  They are
kept together so a future implementation cannot make the happy path pass by
special-casing one phrase such as ``普通员工餐补``.
"""

from dataclasses import dataclass, replace
from itertools import product
import re
import unittest

from core.query_route_compiler import (
    RouteCategoryPolicy,
    RouteCompilerConfig,
    compile_rag_task_contract,
    is_rag_task_contract_dispatchable,
    rag_task_contract_gate_reason,
)
from core.query_route_contract import parse_rag_route_decision
from core.rag_v2.bridge_resolution import (
    partition_bridge_facts,
    resolve_bridge_facts,
)
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.evidence import assemble_evidence_bundle
from core.rag_v2.query_plan import plan_query_locally
from core.rag_v2.task_execution import BridgeResolution, TaskExecutionLedger
from core.rag_v2.task_graph import compile_rag_execution_bundle


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


def _explicit_requirement_edges(
    requirements: tuple[AnswerRequirementV2, ...],
) -> tuple[AnswerRequirementV2, ...]:
    """Make the test fixture's bridge-edge choice explicit.

    The runtime rejects a V2 plan that leaves an answer's proof or optional
    augmentation edge undecided.  Older fixture data occasionally omitted the
    empty tuple; normalising that legacy omission here keeps the scenario
    matrix focused on evidence behaviour, without reintroducing an unbound
    production path.
    """

    normalized: list[AnswerRequirementV2] = []
    for requirement in requirements:
        if requirement.role == "bridge":
            normalized.append(requirement)
            continue
        normalized.append(replace(
            requirement,
            depends_on_requirement_ids=(
                ()
                if requirement.depends_on_requirement_ids is None
                else requirement.depends_on_requirement_ids
            ),
            augmentation_requirement_ids=(
                ()
                if requirement.augmentation_requirement_ids is None
                else requirement.augmentation_requirement_ids
            ),
        ))
    return tuple(normalized)


def _record_observation(
    ledger: TaskExecutionLedger,
    *,
    kind: str,
    query: str,
    task_ids: tuple[str, ...],
    candidates: tuple[dict, ...],
    parent_task_ids: tuple[str, ...] = (),
    parent_chunk_ids: tuple[str, ...] = (),
    route_kind: str = "static",
    bridge_edge_mode: str | None = None,
) -> tuple[dict, ...]:
    """Observe one simulated physical retrieval through the real ledger.

    Candidate metadata never decides task ownership in this matrix.  This
    helper deliberately passes raw retriever rows to ``observe_candidates``;
    the request-local ledger sanitises them and records the only provenance
    which evidence assembly is allowed to trust.
    """

    execution_id = ledger.begin_execution(
        kind=kind,
        query=query,
        task_ids=task_ids,
        parent_task_ids=parent_task_ids,
        parent_chunk_ids=parent_chunk_ids,
        route_kind=route_kind,
        bridge_edge_mode=bridge_edge_mode,
    )
    observed = tuple(ledger.observe_candidates(
        candidates,
        execution_id=execution_id,
        parent_task_ids=parent_task_ids,
        parent_chunk_ids=parent_chunk_ids,
    ))
    ledger.finish_execution(
        execution_id,
        status="succeeded",
        candidate_count=len(observed),
    )
    return observed


def _ledgered_bundle(
    *,
    query: str,
    requirements: tuple[AnswerRequirementV2, ...],
    answer_shape: str,
    initial_candidates_by_task: dict[str, tuple[dict, ...]] | None = None,
    second_hop_candidates_by_answer: dict[str, tuple[dict, ...]] | None = None,
    constraints=None,
) -> tuple[object, object, TaskExecutionLedger]:
    """Execute a scenario through ``plan -> graph -> ledger -> evidence``.

    This is intentionally stricter than the historical scenario fixture:

    * a source is visible only if a concrete task retrieved it in this run;
    * a bridge fact is resolved from its bridge task, not from a generic pool;
    * a bridge second hop is bound to the exact source chunks that supplied
      the resolved fact; and
    * direct evidence stays a separate route from optional augmentation.

    ``initial_candidates_by_task`` and ``second_hop_candidates_by_answer``
    model retriever responses, keyed by stable task id.  They make it
    impossible for a test to accidentally prove an answer with a candidate
    returned for a different task.
    """

    normalized_requirements = _explicit_requirement_edges(requirements)
    plan = QueryPlanV2(
        original_query=query,
        answer_shape=answer_shape,
        retrieval_queries=tuple(
            item.description
            for item in normalized_requirements
            if item.role == "answer"
        ),
        requirements=normalized_requirements,
        confidence=0.95,
        source="local",
    )
    execution_bundle = compile_rag_execution_bundle(plan)
    if (
        not execution_bundle.uses_task_ledger
        or execution_bundle.task_graph is None
    ):
        raise AssertionError("scenario requires a ledgered V2 execution bundle")
    graph = execution_bundle.task_graph
    ledger = TaskExecutionLedger(graph, run_id="scenario-matrix")
    initial = initial_candidates_by_task or {}
    second_hop = second_hop_candidates_by_answer or {}
    observed_pools: list[tuple[dict, ...]] = []

    # The anchor is a real execution even when this scenario has no broad
    # recall rows.  It prevents a test-only shortcut from skipping graph
    # topology altogether.
    anchor = graph.task_by_id["anchor_root"]
    observed_pools.append(_record_observation(
        ledger,
        kind="scenario_anchor_query",
        query=anchor.query,
        task_ids=(anchor.task_id,),
        candidates=tuple(initial.get(anchor.task_id, ())),
    ))

    bridge_facts_by_task: dict[str, tuple] = {}
    bridge_resolution_by_task: dict[str, BridgeResolution] = {}
    for task in graph.tasks:
        if task.role != "bridge":
            continue
        candidates = tuple(initial.get(task.task_id, ()))
        execution_id = ledger.begin_execution(
            kind="scenario_bridge_query",
            query=task.query,
            task_ids=(task.task_id,),
        )
        observed = tuple(ledger.observe_candidates(
            candidates,
            execution_id=execution_id,
        ))
        ledger.finish_execution(
            execution_id,
            status="succeeded",
            candidate_count=len(observed),
        )
        observed_pools.append(observed)
        requirement = next(
            item
            for item in normalized_requirements
            if item.id == task.target_requirement_ids[0]
        )
        facts, conflicts = partition_bridge_facts(
            resolve_bridge_facts((requirement,), observed)
        )
        if conflicts:
            resolution = BridgeResolution(
                bridge_task_id=task.task_id,
                status="conflict",
                conflicts=conflicts,
                source_execution_ids=(execution_id,),
                source_chunk_ids=tuple(
                    chunk_id
                    for conflict in conflicts
                    for chunk_id in conflict.source_chunk_ids
                ),
                reason="scenario_conflicting_bridge_facts",
            )
        elif facts:
            resolution = BridgeResolution(
                bridge_task_id=task.task_id,
                status="resolved",
                facts=facts,
                source_execution_ids=(execution_id,),
                source_chunk_ids=tuple(fact.source_chunk_id for fact in facts),
            )
            bridge_facts_by_task[task.task_id] = facts
        else:
            resolution = BridgeResolution(
                bridge_task_id=task.task_id,
                status="no_fact",
                source_execution_ids=(execution_id,),
                reason="scenario_bridge_no_fact",
            )
        ledger.record_bridge_resolution(resolution)
        bridge_resolution_by_task[task.task_id] = resolution

    # The literal answer query always runs.  A proof bridge controls claim
    # applicability, not whether directly stated source evidence can be
    # retrieved and audited.
    for task in graph.tasks:
        if task.role != "answer":
            continue
        observed_pools.append(_record_observation(
            ledger,
            kind="scenario_answer_query",
            query=task.query,
            task_ids=(task.task_id,),
            candidates=tuple(initial.get(task.task_id, ())),
        ))

    recorded_augmentation_answers: set[str] = set()
    for mode in ("proof", "augmentation"):
        for path in graph.answer_bridge_paths(mode=mode):
            unresolved_parent_ids = tuple(
                task_id
                for task_id in path.bridge_task_ids
                if bridge_resolution_by_task[task_id].status != "resolved"
            )
            if unresolved_parent_ids:
                if path.edge_mode == "proof":
                    ledger.mark_tasks_blocked_by_dependency(
                        (path.answer_task_id,),
                        blocked_by_task_ids=unresolved_parent_ids,
                        reason="scenario_bridge_proof_unresolved",
                    )
                elif path.answer_task_id not in recorded_augmentation_answers:
                    ledger.record_answer_bridge_augmentation(
                        (path.answer_task_id,),
                        status="skipped_no_fact",
                        reason="scenario_bridge_augmentation_unresolved",
                    )
                    recorded_augmentation_answers.add(path.answer_task_id)
                continue

            parent_fact_sets = tuple(
                bridge_facts_by_task[parent_task_id]
                for parent_task_id in path.bridge_task_ids
            )
            routed_candidates = tuple(second_hop.get(path.answer_task_id, ()))
            for facts in product(*parent_fact_sets):
                observed_pools.append(_record_observation(
                    ledger,
                    kind="scenario_bridge_second_hop",
                    query=graph.task_by_id[path.answer_task_id].query,
                    task_ids=(path.answer_task_id,),
                    candidates=routed_candidates,
                    parent_task_ids=path.bridge_task_ids,
                    parent_chunk_ids=tuple(
                        fact.source_chunk_id for fact in facts
                    ),
                    route_kind="bridge_second_hop",
                    bridge_edge_mode=path.edge_mode,
                ))
            if (
                path.edge_mode == "augmentation"
                and path.answer_task_id not in recorded_augmentation_answers
            ):
                ledger.record_answer_bridge_augmentation(
                    (path.answer_task_id,),
                    status=(
                        "released"
                        if routed_candidates
                        else "skipped_not_materializable"
                    ),
                    reason="scenario_bridge_second_hop",
                )
                recorded_augmentation_answers.add(path.answer_task_id)

    candidates = tuple(ledger.merge_candidate_pools(*observed_pools))
    bundle = assemble_evidence_bundle(
        query=query,
        candidates=candidates,
        requirements=normalized_requirements,
        retrieval_queries=plan.retrieval_queries,
        task_graph=graph,
        task_ledger=ledger,
        answer_shape=answer_shape,
        constraints=constraints,
    )
    return bundle, graph, ledger


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


def _explicit_mapping_plan(scenario: MappingScenario) -> QueryPlanV2:
    """Build a proof graph only for tests that explicitly exercise that graph.

    Ordinary local planning no longer invents this edge.  These evidence tests
    provide it as an already-authorized semantic contract so the generic
    bridge executor remains covered without changing direct-query semantics.
    """

    mapping_clause = scenario.bridge_content.split("：", 1)[-1]
    subject = re.split(r"对应|属于|适用", mapping_clause, maxsplit=1)[0].strip()
    return QueryPlanV2(
        original_query=scenario.question,
        answer_shape="multi_hop",
        retrieval_queries=(
            scenario.question,
            f"确认{subject}对应的分类",
        ),
        requirements=(
            AnswerRequirementV2(
                id="r1",
                description=scenario.question,
                depends_on_requirement_ids=("r2",),
                augmentation_requirement_ids=(),
            ),
            AnswerRequirementV2(
                id="r2",
                description=f"确认{subject}对应的分类",
                role="bridge",
                importance="helpful",
                source="explicit",
                bridge_subject=subject,
                bridge_kind="classification",
            ),
        ),
        confidence=1.0,
        source="local",
        reason="explicit_mapping_test_contract",
    )


class RagV2CrossDomainPlanningMatrixTests(unittest.TestCase):
    def test_subject_specific_questions_never_invent_a_classification_task(
        self,
    ) -> None:
        """The local plan preserves the literal question as its only target."""

        for scenario in MAPPING_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                plan = plan_query_locally(scenario.question)

                self.assertEqual([item.role for item in plan.requirements], ["answer"])
                self.assertTrue(plan.requirements[0].is_required_answer)
                self.assertEqual(plan.requirements[0].depends_on_requirement_ids, ())
                self.assertEqual(plan.requirements[0].augmentation_requirement_ids, ())
                self.assertEqual(plan.retrieval_queries, (scenario.question,))

    def test_local_fallback_does_not_invent_a_condition_mapping(self) -> None:
        """Business-specific taxonomy belongs to the semantic contract, not regex.

        ``试用期 -> P0`` and ``华东地区 -> R2`` are valid source facts in a
        particular policy, but they are not universal language rules.  The
        local fallback must retain a directly retrievable fact request instead
        of manufacturing an unsupported taxonomy bridge.  A model-derived
        semantic contract may add a typed bridge only after its source span is
        validated by the backend.
        """

        for scenario in (
            item
            for item in MAPPING_SCENARIOS
            if item.name in {"probation_annual_leave", "regional_lodging"}
        ):
            with self.subTest(scenario=scenario.name):
                plan = plan_query_locally(scenario.question)

                self.assertEqual(plan.answer_shape, "fact")
                self.assertEqual([item.role for item in plan.requirements], ["answer"])
                self.assertEqual(plan.requirements[0].depends_on_requirement_ids, ())
                self.assertEqual(plan.requirements[0].augmentation_requirement_ids, ())

    def test_policy_and_security_queries_remain_domain_neutral(self) -> None:
        cases = (
            ("报销需要满足什么条件", "fact"),
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

    def test_compiled_direct_contract_never_requires_an_inferred_bridge(self) -> None:
        for scenario in MAPPING_SCENARIOS:
            with self.subTest(scenario=scenario.name):
                contract = _compile(scenario.question)

                self.assertEqual(
                    [item.role for item in contract.requirements],
                    ["answer"],
                )
                self.assertIsNone(rag_task_contract_gate_reason(contract))
                self.assertTrue(is_rag_task_contract_dispatchable(contract))

    def test_direct_contract_without_taxonomy_remains_dispatchable(self) -> None:
        for scenario in (
            item
            for item in MAPPING_SCENARIOS
            if item.name in {"probation_annual_leave", "regional_lodging"}
        ):
            with self.subTest(scenario=scenario.name):
                contract = _compile(scenario.question)

                self.assertEqual([item.role for item in contract.requirements], ["answer"])
                self.assertIsNone(rag_task_contract_gate_reason(contract))
                self.assertTrue(is_rag_task_contract_dispatchable(contract))


class RagV2CrossDomainEvidenceMatrixTests(unittest.TestCase):
    def test_coordinated_employee_standards_require_every_explicit_item(
        self,
    ) -> None:
        question = "普通员工出差的住宿、交通和餐补标准分别是多少？"
        plan = plan_query_locally(question)
        lodging = _candidate(
            "lodging",
            "普通员工出差住宿标准：一线城市不超过450元/天。",
            doc_id="travel-policy",
        )
        transport = _candidate(
            "transport",
            "普通员工出差交通标准：飞机经济舱、高铁二等座。",
            chunk_index=1,
            doc_id="travel-policy",
        )
        meal = _candidate(
            "meal",
            "普通员工出差餐补标准：每天100元。",
            chunk_index=2,
            doc_id="travel-policy",
        )

        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertFalse(plan.has_bridge_dependencies)
        self.assertFalse(plan.has_bridge_augmentations)
        self.assertEqual(
            [item.role for item in plan.requirements],
            ["answer", "answer", "answer"],
        )
        self.assertEqual(
            [item.augmentation_requirement_ids for item in plan.requirements[:3]],
            [(), (), ()],
        )

        complete, _graph, ledger = _ledgered_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            requirements=plan.requirements,
            initial_candidates_by_task={
                "answer_r1": (lodging,),
                "answer_r2": (transport,),
                "answer_r3": (meal,),
            },
        )
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(complete.state.completeness, "complete")
        self.assertEqual(
            set(complete.answer_source_ids),
            {"lodging", "transport", "meal"},
        )
        self.assertTrue(all(
            ledger.task_state_summary()[task_id]["bridge_augmentation_status"]
            == "not_applicable"
            for task_id in ("answer_r1", "answer_r2", "answer_r3")
        ))

        missing_meal, _graph, _ledger = _ledgered_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            requirements=plan.requirements,
            initial_candidates_by_task={
                "answer_r1": (lodging,),
                "answer_r2": (transport,),
            },
        )
        self.assertEqual(missing_meal.missing_requirement_ids, ("r3",))
        self.assertEqual(missing_meal.state.completeness, "partial")

        complete_without_mapping, _graph, ledger = _ledgered_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            requirements=plan.requirements,
            initial_candidates_by_task={
                "answer_r1": (lodging,),
                "answer_r2": (transport,),
                "answer_r3": (meal,),
            },
        )
        self.assertEqual(complete_without_mapping.missing_requirement_ids, ())
        self.assertEqual(complete_without_mapping.state.completeness, "complete")
        self.assertTrue(all(
            ledger.task_state_summary()[task_id]["bridge_augmentation_status"]
            == "not_applicable"
            for task_id in ("answer_r1", "answer_r2", "answer_r3")
        ))

    def test_unmarked_coordinated_targets_close_from_direct_chunks(
        self,
    ) -> None:
        question = "普通员工的住宿标准和餐补是多少"
        plan = plan_query_locally(question)
        lodging = _candidate(
            "lodging",
            "普通员工住宿费用标准：一线城市不超过450元/天。",
            doc_id="travel-policy",
        )
        meal = _candidate(
            "meal",
            "普通员工餐补标准：100元/天。",
            chunk_index=1,
            doc_id="travel-policy",
        )
        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertEqual(
            [item.role for item in plan.requirements],
            ["answer", "answer"],
        )
        self.assertEqual(
            [item.depends_on_requirement_ids for item in plan.requirements[:2]],
            [(), ()],
        )
        self.assertEqual(
            [item.augmentation_requirement_ids for item in plan.requirements[:2]],
            [(), ()],
        )

        bundle, _graph, ledger = _ledgered_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            requirements=plan.requirements,
            initial_candidates_by_task={
                "answer_r1": (lodging,),
                "answer_r2": (meal,),
            },
        )

        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual(
            set(bundle.answer_source_ids),
            {"lodging", "meal"},
        )
        self.assertEqual(
            ledger.task_state_summary()["answer_r1"]["bridge_augmentation_status"],
            "not_applicable",
        )

    def test_optional_bridge_failure_never_blocks_a_direct_subject_clause(
        self,
    ) -> None:
        """Direct and bridge-augmented routes are independent proof paths."""

        question = "普通员工的餐补是多少"
        plan = plan_query_locally(question)
        direct = _candidate("direct", "普通员工餐补标准为100元/天。")

        bundle, _graph, ledger = _ledgered_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            requirements=plan.requirements,
            initial_candidates_by_task={"answer_r1": (direct,)},
        )

        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual(bundle.answer_source_ids, ("direct",))
        self.assertEqual(
            ledger.task_state_summary()["answer_r1"]["bridge_augmentation_status"],
            "not_applicable",
        )

    def test_direct_subject_clause_accepts_literal_compact_claims(self) -> None:
        """Legacy evidence assembly still accepts direct literal clauses."""

        question = "普通员工的餐补是多少"
        plan = plan_query_locally(question)
        cases = (
            ("compact_direct", "普通员工餐补标准为100元/天。", True),
            ("possessive_direct", "普通员工的餐补标准为100元/天。", True),
        )

        for chunk_id, content, should_close in cases:
            with self.subTest(chunk_id=chunk_id):
                bundle, _graph, _ledger = _ledgered_bundle(
                    query=question,
                    answer_shape=plan.answer_shape,
                    requirements=plan.requirements,
                    initial_candidates_by_task={
                        "answer_r1": (_candidate(chunk_id, content),),
                    },
                )

                self.assertEqual(
                    bundle.missing_requirement_ids == (),
                    should_close,
                )
                self.assertEqual(
                    chunk_id in bundle.answer_source_ids,
                    should_close,
                )

    def test_explicit_proof_bridge_requires_the_resolved_value_and_route(
        self,
    ) -> None:
        """A hard relation request may use D-level evidence only with its proof."""

        question = "根据普通员工对应的职级，住宿标准是多少"
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description=question,
                depends_on_requirement_ids=("r2",),
                augmentation_requirement_ids=(),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通员工对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        mapping = _candidate("mapping", "普通员工对应D级。")
        correct = _candidate("correct", "D级住宿标准为450元/天。")
        wrong = _candidate("wrong", "A级住宿标准为1200元/天。")

        complete, _graph, _ledger = _ledgered_bundle(
            query=question,
            requirements=requirements,
            answer_shape="multi_hop",
            initial_candidates_by_task={"bridge_r2": (mapping,)},
            second_hop_candidates_by_answer={"answer_r1": (correct,)},
        )
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(
            set(complete.answer_source_ids),
            {"mapping", "correct"},
        )

        wrong_value, _graph, _ledger = _ledgered_bundle(
            query=question,
            requirements=requirements,
            answer_shape="multi_hop",
            initial_candidates_by_task={"bridge_r2": (mapping,)},
            second_hop_candidates_by_answer={"answer_r1": (wrong,)},
        )
        self.assertEqual(wrong_value.missing_requirement_ids, ("r1",))
        self.assertNotIn("wrong", wrong_value.answer_source_ids)

        no_bridge, _graph, ledger = _ledgered_bundle(
            query=question,
            requirements=requirements,
            answer_shape="multi_hop",
            initial_candidates_by_task={"answer_r1": (correct,)},
        )
        self.assertEqual(no_bridge.missing_requirement_ids, ("r1", "r2"))
        self.assertNotIn("correct", no_bridge.answer_source_ids)
        self.assertEqual(
            ledger.task_state_summary()["answer_r1"]["blocked_dependency"],
            1,
        )

    def test_correct_intermediate_value_joins_mapping_and_answer(self) -> None:
        scenarios = tuple(
            item
            for item in MAPPING_SCENARIOS
            if item.name not in {"probation_annual_leave", "regional_lodging"}
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario.name):
                plan = _explicit_mapping_plan(scenario)
                bundle, _graph, ledger = _ledgered_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    requirements=plan.requirements,
                    initial_candidates_by_task={
                        "bridge_r2": (
                            _candidate(
                                "bridge",
                                scenario.bridge_content,
                                chunk_index=1,
                                doc_id=f"{scenario.name}-classification-table",
                            ),
                        ),
                    },
                    second_hop_candidates_by_answer={
                        "answer_r1": (
                            _candidate(
                                "answer",
                                scenario.answer_content,
                                doc_id=f"{scenario.name}-answer-policy",
                            ),
                        ),
                    },
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
                self.assertEqual(
                    ledger.task_state_summary()["answer_r1"]["bridge_augmentation_status"],
                    "not_applicable",
                )

    def test_bridge_only_is_partial_and_cannot_answer_the_final_target(self) -> None:
        scenarios = tuple(
            item
            for item in MAPPING_SCENARIOS
            if item.name not in {"probation_annual_leave", "regional_lodging"}
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario.name):
                plan = _explicit_mapping_plan(scenario)
                bundle, _graph, _ledger = _ledgered_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    requirements=plan.requirements,
                    initial_candidates_by_task={
                        "bridge_r2": (_candidate("bridge", scenario.bridge_content),),
                    },
                )

                item = bundle.items[0]
                self.assertEqual(item.role, "bridge")
                self.assertEqual(item.supports_requirement_ids, ("r2",))
                self.assertEqual(bundle.missing_requirement_ids, ("r1",))
                # The provisional bundle may retain the bridge for trace
                # diagnostics, but it has no answer claim for r1 and must
                # therefore be blocked before final visible context/generation.
                self.assertNotIn("r1", item.supports_requirement_ids)
                self.assertIn(bundle.state.completeness, {"partial", "unknown"})

    def test_mismatched_intermediate_value_cannot_complete_answer(self) -> None:
        scenarios = tuple(
            item
            for item in MAPPING_SCENARIOS
            if item.name not in {"probation_annual_leave", "regional_lodging"}
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario.name):
                plan = _explicit_mapping_plan(scenario)
                bundle, _graph, _ledger = _ledgered_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    requirements=plan.requirements,
                    initial_candidates_by_task={
                        "bridge_r2": (
                            _candidate(
                                "bridge",
                                scenario.bridge_content,
                                chunk_index=1,
                                doc_id=f"{scenario.name}-classification-table",
                            ),
                        ),
                    },
                    second_hop_candidates_by_answer={
                        "answer_r1": (
                            _candidate(
                                "wrong-answer",
                                scenario.wrong_answer_content,
                                doc_id=f"{scenario.name}-wrong-answer-policy",
                            ),
                        ),
                    },
                )

                by_id = {item.chunk_id: item for item in bundle.items}
                self.assertNotEqual(
                    by_id["wrong-answer"].doc_id,
                    by_id["bridge"].doc_id,
                )
                self.assertEqual(bundle.missing_requirement_ids, ("r1",))
                self.assertIn(bundle.state.completeness, {"partial", "unknown"})
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

                bundle, _graph, _ledger = _ledgered_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    requirements=plan.requirements,
                    initial_candidates_by_task={
                        **{
                            f"answer_r{index}": (candidate,)
                            for index, candidate in enumerate(
                                support_candidates,
                                start=1,
                            )
                        },
                        # An irrelevant retrieval row is deliberately bound to
                        # a real answer query.  Its exclusion must come from
                        # semantic claim adjudication, never from an omitted
                        # test input.
                        "answer_r1": (support_candidates[0], distractor),
                    },
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

                bundle, _graph, _ledger = _ledgered_bundle(
                    query=scenario.question,
                    answer_shape=plan.answer_shape,
                    requirements=plan.requirements,
                    initial_candidates_by_task={
                        f"answer_r{index}": (candidate,)
                        for index, candidate in enumerate(candidates, start=1)
                        if index <= len(retained_contents)
                    },
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

        complete, _graph, _ledger = _ledgered_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            requirements=plan.requirements,
            initial_candidates_by_task={
                "answer_r1": (candidates[0],),
                "answer_r2": (candidates[1],),
                "answer_r3": (candidates[2],),
            },
        )
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(complete.state.completeness, "complete")
        self.assertEqual(
            set(complete.answer_source_ids),
            {"condition", "limit", "approval"},
        )

        missing_approval, _graph, _ledger = _ledgered_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            requirements=plan.requirements,
            initial_candidates_by_task={
                "answer_r1": (candidates[0],),
                "answer_r2": (candidates[1],),
            },
        )
        self.assertEqual(missing_approval.missing_requirement_ids, ("r3",))
        self.assertEqual(missing_approval.state.completeness, "partial")
        self.assertNotIn("approval", missing_approval.answer_source_ids)


if __name__ == "__main__":
    unittest.main()
