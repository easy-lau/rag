import unittest
from dataclasses import replace
from itertools import product

from core.query_constraints import (
    extract_applicability_scope,
    extract_applicability_scopes,
    extract_query_constraints,
)
from core.rag_v2.bridge_resolution import (
    ResolvedBridgeFact,
    adjudicate_answer_claims,
    answer_target_terms,
    bridge_fact_matches_candidate_scope,
    build_bridge_expansion_queries,
    candidate_supports_resolved_answer_set,
    content_contains_bridge_value,
    content_contains_positive_subject,
    content_matches_answer_target,
    extract_bridge_values,
    extract_bridge_subject,
    partition_bridge_facts,
    resolve_bridge_facts,
)
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.evidence import (
    assemble_evidence_bundle as _assemble_evidence_bundle,
    finalize_visible_evidence_bundle,
)
from core.rag_v2.task_execution import BridgeResolution, TaskExecutionLedger
from core.rag_v2.task_graph import compile_rag_execution_bundle


def _candidate(
    chunk_id: str,
    *,
    doc_id: str = "doc-a",
    kb_id: str = "kb-a",
    chunk_index: int = 0,
    content: str = "具体条款：住宿标准为450元/天。",
    **values,
) -> dict:
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "chunk_index": chunk_index,
        "content": content,
        **values,
    }


def _multi_hop_requirements(
    answer_description: str,
    bridge_description: str,
) -> tuple[AnswerRequirementV2, AnswerRequirementV2]:
    """Build an explicitly declared proof fixture for join-specific tests.

    This helper is intentionally for evidence tests that exercise a typed
    proof edge.  It is not a model of ordinary implicit classification in a
    user question; production planning represents that path as a separate
    optional augmentation edge.
    """

    return (
        AnswerRequirementV2(
            id="r1",
            description=answer_description,
            depends_on_requirement_ids=("r2",),
            augmentation_requirement_ids=(),
        ),
        AnswerRequirementV2(
            id="r2",
            description=bridge_description,
            role="bridge",
            importance="helpful",
            source="inferred",
            bridge_subject=extract_bridge_subject(bridge_description),
            bridge_kind="classification",
        ),
    )


def _classification_augmentation_requirements(
    answer_description: str,
    bridge_description: str,
    *,
    subject: str,
) -> tuple[AnswerRequirementV2, AnswerRequirementV2]:
    """Build the production-shaped optional classification fixture.

    A direct source claim may satisfy ``r1`` without resolving ``r2``.  The
    bridge is still available to improve recall/precision in the execution
    graph, but it must never become a proof precondition by test accident.
    """

    return (
        AnswerRequirementV2(
            id="r1",
            description=answer_description,
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=("r2",),
        ),
        AnswerRequirementV2(
            id="r2",
            description=bridge_description,
            role="bridge",
            importance="helpful",
            source="inferred",
            bridge_subject=subject,
            bridge_kind="classification",
        ),
    )


def _explicit_test_requirements(
    requirements: tuple[AnswerRequirementV2, ...],
) -> tuple[AnswerRequirementV2, ...]:
    """Reject legacy bridge fixtures and make answer-edge decisions explicit.

    Production plans are required to carry explicit answer dependency choices.
    Older evidence unit tests intentionally omitted ``()`` for independent
    answers, so this test-only fixture makes that absence explicit without
    inferring any bridge relation.  A bridge still has to declare its subject
    and kind in the individual test data.
    """

    explicit: list[AnswerRequirementV2] = []
    for requirement in requirements:
        if requirement.role == "bridge":
            if not requirement.bridge_subject or not requirement.bridge_kind:
                raise AssertionError(
                    "ledgered evidence fixtures require a typed bridge subject and kind"
                )
            explicit.append(requirement)
            continue
        explicit.append(replace(
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
    return tuple(explicit)


def _ledgered_evidence_bundle(*, return_execution_state: bool = False, **kwargs):
    """Assemble valid V2 evidence through one request-local execution ledger.

    The old tests attached task ids, support ids, or query indexes to candidate
    metadata.  This fixture deliberately removes those fields through
    ``observe_candidates`` and records every candidate under an actual task
    execution.  Proof-bridge answers additionally receive a real
    ``bridge_second_hop`` binding with the exact bridge parent source ids.
    """

    raw_requirements = tuple(kwargs.get("requirements") or ())
    requirements = _explicit_test_requirements(raw_requirements)
    query = str(kwargs.get("query") or "")
    answer_shape = kwargs.get("answer_shape")
    if answer_shape is None:
        answer_shape = "multi_part" if sum(
            item.role == "answer" for item in requirements
        ) > 1 else "fact"
    retrieval_queries = tuple(kwargs.get("retrieval_queries") or tuple(
        item.description for item in requirements if item.role == "answer"
    ))
    plan = QueryPlanV2(
        original_query=query,
        answer_shape=answer_shape,
        retrieval_queries=retrieval_queries,
        requirements=requirements,
        confidence=0.95,
        source="local",
    )
    execution_bundle = compile_rag_execution_bundle(plan)
    if (
        not execution_bundle.uses_task_ledger
        or execution_bundle.task_graph is None
    ):
        raise AssertionError("valid evidence fixture must compile a ledgered bundle")
    graph = execution_bundle.task_graph
    ledger = TaskExecutionLedger(graph, run_id="evidence-test-run")

    raw_candidates = tuple(kwargs.get("candidates") or ())
    raw_overview_candidates = tuple(kwargs.get("overview_candidates") or ())
    # Candidates in this file are mappings.  Let the ledger own sanitisation;
    # later observations use its safe copies, never raw task annotations.
    all_candidates: list[dict] = []
    for candidate in (*raw_candidates, *raw_overview_candidates):
        if isinstance(candidate, dict):
            all_candidates.append(dict(candidate))
        else:
            all_candidates.append(candidate.to_dict())

    bridge_facts_by_task: dict[str, tuple[ResolvedBridgeFact, ...]] = {}
    for task in graph.tasks:
        if task.role != "bridge":
            continue
        execution_id = ledger.begin_execution(
            kind="test_bridge_query",
            query=task.query,
            task_ids=(task.task_id,),
        )
        observed = ledger.observe_candidates(
            all_candidates,
            execution_id=execution_id,
        )
        ledger.finish_execution(
            execution_id,
            status="succeeded",
            candidate_count=len(observed),
        )
        requirement = next(
            item
            for item in requirements
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
                reason="test_conflicting_bridge_facts",
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
                reason="test_bridge_no_fact",
            )
        ledger.record_bridge_resolution(resolution)

    for task in graph.tasks:
        if task.role != "answer":
            continue
        # Every answer has its literal first-wave retrieval.  It cannot close
        # a proof route by itself, but keeping it in the ledger mirrors the
        # production schedule and preserves direct/augmentation coverage.
        execution_id = ledger.begin_execution(
            kind="test_answer_query",
            query=task.query,
            task_ids=(task.task_id,),
        )
        ledger.observe_candidates(
            all_candidates,
            execution_id=execution_id,
        )
        ledger.finish_execution(
            execution_id,
            status="succeeded",
            candidate_count=len(all_candidates),
        )

        paths = tuple(
            path
            for mode in ("proof", "augmentation")
            for path in graph.answer_bridge_paths(mode=mode)
            if path.answer_task_id == task.task_id
        )
        for path in paths:
            parent_fact_sets = tuple(
                bridge_facts_by_task.get(parent_task_id, ())
                for parent_task_id in path.bridge_task_ids
            )
            if not parent_fact_sets or any(not facts for facts in parent_fact_sets):
                continue
            # A request may resolve distinct, scope-compatible bridge facts in
            # different documents.  Bind every physical second-hop response to
            # one exact fact combination; a flattened union would be the same
            # provenance bug the production ledger is designed to reject.
            for path_facts in product(*parent_fact_sets):
                execution_id = ledger.begin_execution(
                    kind="test_bridge_second_hop",
                    query=task.query,
                    task_ids=(task.task_id,),
                    parent_task_ids=path.bridge_task_ids,
                    parent_chunk_ids=tuple(
                        fact.source_chunk_id for fact in path_facts
                    ),
                    route_kind="bridge_second_hop",
                    bridge_edge_mode=path.edge_mode,
                )
                ledger.observe_candidates(
                    all_candidates,
                    execution_id=execution_id,
                    parent_task_ids=path.bridge_task_ids,
                    parent_chunk_ids=tuple(
                        fact.source_chunk_id for fact in path_facts
                    ),
                )
                ledger.finish_execution(
                    execution_id,
                    status="succeeded",
                    candidate_count=len(all_candidates),
                )

    assembled_kwargs = dict(kwargs)
    assembled_kwargs.update(
        requirements=requirements,
        task_graph=graph,
        task_ledger=ledger,
    )
    bundle = _assemble_evidence_bundle(**assembled_kwargs)
    if return_execution_state:
        return bundle, graph, ledger
    return bundle


def assemble_evidence_bundle(**kwargs):
    """Route every valid V2 evidence test through the ledgered fixture."""

    requirements = tuple(kwargs.get("requirements") or ())
    answer_shape = kwargs.get("answer_shape")
    # These two tests intentionally validate malformed public input before a
    # plan can exist, so a ledgered handoff would be nonsensical.
    if not requirements or (
        answer_shape == "multi_hop"
        and not any(item.role == "bridge" for item in requirements)
    ):
        return _assemble_evidence_bundle(**kwargs)
    return _ledgered_evidence_bundle(**kwargs)


class EvidenceBundleAssemblyTests(unittest.TestCase):
    def test_normal_v2_shape_requires_typed_requirements(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty requirements"):
            assemble_evidence_bundle(
                query="查询标准",
                answer_shape="fact",
                candidates=[_candidate("seed")],
            )

    def test_multi_hop_shape_requires_bridge_requirement(self) -> None:
        with self.assertRaisesRegex(ValueError, "answer-to-bridge dependency"):
            assemble_evidence_bundle(
                query="对象对应的额度是多少",
                answer_shape="multi_hop",
                candidates=[_candidate("seed")],
                requirements=(
                    AnswerRequirementV2(
                        id="r1",
                        description="对象对应的额度是多少",
                        depends_on_requirement_ids=(),
                    ),
                ),
            )

    def test_single_requirement_retrieval_seed_is_not_requirement_support(self) -> None:
        requirement = AnswerRequirementV2(id="r1", description="某项标准")
        bundle = assemble_evidence_bundle(
            query="某项标准",
            candidates=[
                _candidate(
                    "seed",
                    content="明确条款：上限为450元/天。",
                    candidate_origins=["initial_retrieval"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=("某项标准",),
        )

        self.assertEqual(bundle.items[0].role, "background")
        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_document_policy_overview_maps_only_a_complete_rooted_snapshot(self) -> None:
        """A policy overview requires a real root plus every source chunk.

        The historical version of this regression used ``制度标题`` and one
        unrelated ``具体章节内容`` but expected both to support an overview.
        That fixture could not prove either the governing policy or an
        exhaustive source snapshot, and would reward exactly the unsafe
        behaviour the V2 evidence graph is meant to reject.  Keep the
        coverage test at the execution boundary instead: a current-query
        title/classification seed roots one document, and each bounded member
        carries the retriever-declared full-document cardinality.
        """

        requirement = AnswerRequirementV2(
            id="r1",
            description="普通岗位的管理标准是什么",
            coverage_mode="collection",
            coverage_contract="document_policy",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        )
        filename = "公司管理标准.docx"
        contents = (
            "公司制度",
            "一、总则：规范管理。",
            "二、分类：普通岗位对应D级。",
            "三、交通：D级乘坐经济舱和高铁二等座。",
            "四、住宿：D级上限450元/天。",
            "五、餐饮：D级补贴100元/天。",
        )
        candidates = [
            _candidate(
                f"policy-{index}",
                chunk_index=index,
                content=content,
                filename=filename,
                candidate_origins=(
                    ["initial_retrieval"]
                    if index in {0, 2}
                    else ["small_document_full"]
                ),
                full_document_chunk_count=len(contents),
            )
            for index, content in enumerate(contents)
        ]

        provisional, task_graph, ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query=requirement.description,
            answer_shape="overview",
            # Mirror the production boundary: only title/classification rows
            # are first-wave retrieval anchors; the bounded full snapshot is
            # admitted through the dedicated overview-expansion channel.
            candidates=[candidates[0], candidates[2]],
            overview_candidates=candidates,
            requirements=(requirement,),
            retrieval_queries=(requirement.description,),
        )
        bundle = finalize_visible_evidence_bundle(
            provisional,
            requirements=(requirement,),
            task_graph=task_graph,
            task_ledger=ledger,
        ).bundle

        expected_ids = {f"policy-{index}" for index in range(len(contents))}
        self.assertEqual(set(bundle.answer_source_ids), expected_ids)
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertTrue(all(
            item.supports_requirement_ids == ("r1",)
            for item in bundle.answer_sources
        ))

    def test_query_indexes_preserve_visible_per_requirement_mapping(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="交通要求"),
            AnswerRequirementV2(id="r2", description="住宿要求"),
        )
        bundle = assemble_evidence_bundle(
            query="请分别查询交通和住宿要求",
            candidates=[
                _candidate(
                    "lodging",
                    content="住宿要求：每天不超过450元。",
                    expansion_query_indexes=[1],
                ),
                _candidate(
                    "transport",
                    chunk_index=1,
                    content="交通要求：乘坐高铁二等座。",
                    expansion_query_indexes=[0],
                ),
            ],
            requirements=requirements,
            retrieval_queries=("交通要求", "住宿要求"),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["transport"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["transport"].role, "direct")
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(set(bundle.answer_source_ids), {"transport", "lodging"})
        self.assertEqual(bundle.state.completeness, "complete")

    def test_merged_query_indexes_cannot_manufacture_missing_answer(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工出差的住宿标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工出差的交通标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r3",
                description="普通员工出差的餐补标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r4",
                description=(
                    "确认普通员工对应的适用分类、等级、类别或阶段"
                    "（用于确定住宿标准）"
                ),
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        all_query_indexes = [0, 1, 2, 3]
        bundle = assemble_evidence_bundle(
            query="普通员工出差的住宿、交通和餐补标准分别是多少？",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "lodging",
                    content="D级住宿标准：一线城市不超过450元/天。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "transport",
                    chunk_index=1,
                    content="D级交通标准：飞机经济舱、高铁二等座。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "classification",
                    chunk_index=2,
                    content="职级分类：普通员工对应D级。",
                    expansion_query_indexes=all_query_indexes,
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(
                requirement.description for requirement in requirements
            ),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["transport"].supports_requirement_ids, ("r2",))
        self.assertEqual(
            by_id["classification"].supports_requirement_ids,
            ("r4",),
        )
        self.assertEqual(bundle.missing_requirement_ids, ("r3",))
        self.assertEqual(bundle.state.completeness, "partial")

    def test_common_subject_terms_cannot_cover_coordinated_answers(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="普通员工住宿标准"),
            AnswerRequirementV2(id="r2", description="普通员工交通标准"),
            AnswerRequirementV2(id="r3", description="普通员工餐补标准"),
        )
        bundle = assemble_evidence_bundle(
            query="普通员工住宿、交通和餐补标准分别是多少？",
            answer_shape="multi_part",
            candidates=[
                _candidate(
                    "common-subject",
                    content="普通员工制度适用于公司全体员工。",
                    expansion_query_indexes=[0, 1, 2],
                )
            ],
            requirements=requirements,
            retrieval_queries=tuple(
                requirement.description for requirement in requirements
            ),
            completeness="complete",
        )

        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1", "r2", "r3"))
        self.assertEqual(bundle.state.completeness, "unknown")

    def test_merged_query_indexes_keep_each_visible_coordinated_answer(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工出差的住宿标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工出差的交通标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r3",
                description="普通员工出差的餐补标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r4",
                description=(
                    "确认普通员工对应的适用分类、等级、类别或阶段"
                    "（用于确定住宿标准）"
                ),
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        all_query_indexes = [0, 1, 2, 3]
        bundle = assemble_evidence_bundle(
            query="普通员工出差的住宿、交通和餐补标准分别是多少？",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "lodging",
                    content="D级住宿标准：一线城市不超过450元/天。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "transport",
                    chunk_index=1,
                    content="D级交通标准：飞机经济舱、高铁二等座。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "meal",
                    chunk_index=2,
                    content="D级餐补标准：每天100元。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "classification",
                    chunk_index=3,
                    content="职级分类：普通员工对应D级。",
                    expansion_query_indexes=all_query_indexes,
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(
                requirement.description for requirement in requirements
            ),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["transport"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["meal"].supports_requirement_ids, ("r3",))
        self.assertEqual(
            by_id["classification"].supports_requirement_ids,
            ("r4",),
        )
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_reimbursement_fragments_cannot_cover_each_other(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="报销提交时限是多久",
            ),
            AnswerRequirementV2(
                id="r2",
                description="需要提供哪些凭证",
            ),
        )
        all_query_indexes = [0, 1]
        deadline = _candidate(
            "deadline",
            content="费用报销时限：出差结束后5个工作日内提交。",
            expansion_query_indexes=all_query_indexes,
        )
        receipts = _candidate(
            "receipts",
            chunk_index=1,
            content="报销凭证：必须提供正规发票、行程单及住宿发票。",
            expansion_query_indexes=all_query_indexes,
        )
        values = {
            "query": "报销提交时限是多久？需要提供哪些凭证？",
            "answer_shape": "multi_part",
            "requirements": requirements,
            "retrieval_queries": tuple(
                requirement.description for requirement in requirements
            ),
            "completeness": "complete",
        }

        complete = assemble_evidence_bundle(
            candidates=[deadline, receipts],
            **values,
        )
        by_id = {item.chunk_id: item for item in complete.items}
        self.assertEqual(by_id["deadline"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["receipts"].supports_requirement_ids, ("r2",))
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(complete.state.completeness, "complete")

        deadline_only = assemble_evidence_bundle(
            candidates=[deadline],
            **values,
        )
        self.assertEqual(deadline_only.missing_requirement_ids, ("r2",))
        self.assertEqual(deadline_only.state.completeness, "partial")

        receipts_only = assemble_evidence_bundle(
            candidates=[receipts],
            **values,
        )
        self.assertEqual(receipts_only.missing_requirement_ids, ("r1",))
        self.assertEqual(receipts_only.state.completeness, "partial")

    def test_bridge_query_index_cannot_promote_value_only_chunk(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐补金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通岗位",
                bridge_kind="classification",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "amount",
                    content="餐饮补贴：D级为100元/天。",
                    candidate_origins=["initial_retrieval"],
                    expansion_query_indexes=[0, 1],
                ),
                _candidate(
                    "classification",
                    chunk_index=1,
                    content="职级分类：普通岗位对应D级。",
                    expansion_query_indexes=[1],
                ),
            ],
            requirements=requirements,
            retrieval_queries=(
                "普通岗位的餐饮补贴是多少",
                "普通岗位 对应的适用分类 等级 类别 阶段",
            ),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["amount"].supports_requirement_ids, ("r1",))
        self.assertEqual(
            by_id["classification"].supports_requirement_ids,
            ("r2",),
        )
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_explicit_support_ids_are_filtered_to_known_requirements(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="目标一"),
            AnswerRequirementV2(id="r2", description="目标二"),
        )
        bundle = assemble_evidence_bundle(
            query="查询目标",
            candidates=[
                _candidate(
                    "explicit",
                    content="目标一：已完成",
                    role="direct",
                    supports_requirement_ids=["r1", "unknown", "INVALID"],
                )
            ],
            requirements=requirements,
            retrieval_queries=("其他查询",),
            rerank_succeeded=True,
            completeness="complete",
        )

        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ("r1",))
        self.assertEqual(item.metadata["supports_requirement_ids"], ["r1"])
        self.assertEqual(item.role, "direct")
        self.assertEqual(bundle.answer_source_ids, ("explicit",))
        self.assertEqual(bundle.missing_requirement_ids, ("r2",))
        self.assertEqual(bundle.state.completeness, "partial")

    def test_lexical_coverage_is_assessed_per_chunk_not_concatenated(self) -> None:
        requirement = AnswerRequirementV2(
            id="r1",
            description="alpha beta",
        )
        bundle = assemble_evidence_bundle(
            query="alpha beta",
            candidates=[
                _candidate("alpha", content="alpha"),
                _candidate("beta", chunk_index=1, content="beta"),
            ],
            requirements=(requirement,),
            retrieval_queries=(),
            completeness="complete",
        )

        self.assertTrue(all(not item.supports_requirement_ids for item in bundle.items))
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))
        self.assertEqual(bundle.state.completeness, "unknown")

    def test_entity_overlap_alone_cannot_satisfy_compound_requirement(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通岗位",
                bridge_kind="classification",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "bridge",
                    content="职级分类：普通岗位对应D级。",
                )
            ],
            requirements=requirements,
            retrieval_queries=("普通岗位的餐饮补贴是多少",),
            completeness="complete",
        )

        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ("r2",))
        self.assertEqual(item.role, "bridge")
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_answer_query_index_cannot_promote_bridge_only_chunk(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通岗位",
                bridge_kind="classification",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "bridge",
                    content="职级分类：普通岗位对应D级。",
                    expansion_query_indexes=[0],
                )
            ],
            requirements=requirements,
            retrieval_queries=("普通岗位的餐饮补贴是多少",),
            completeness="complete",
        )

        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ("r2",))
        self.assertEqual(item.role, "bridge")
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_multi_hop_joins_answer_seed_to_resolved_bridge_value(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通岗位",
                bridge_kind="classification",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "answer",
                    content="餐饮补贴：D级为100元/天。",
                    candidate_origins=["initial_retrieval"],
                ),
                _candidate(
                    "bridge",
                    chunk_index=1,
                    content="职级分类：普通岗位对应D级。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=("普通岗位的餐饮补贴是多少",),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["answer"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["answer"].role, "complement")
        self.assertEqual(by_id["bridge"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["bridge"].role, "bridge")
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(set(bundle.answer_source_ids), {"answer", "bridge"})

    def test_multi_hop_rejects_unjoined_answer_value(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通岗位",
                bridge_kind="classification",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "wrong-answer",
                    content="餐饮补贴：A级为200元/天。",
                    candidate_origins=["initial_retrieval"],
                ),
                _candidate(
                    "bridge",
                    chunk_index=1,
                    content="职级分类：普通岗位对应D级。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=("普通岗位的餐饮补贴是多少",),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["wrong-answer"].supports_requirement_ids, ())
        self.assertEqual(by_id["wrong-answer"].role, "background")
        self.assertEqual(bundle.answer_source_ids, ("bridge",))
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_cross_domain_implicit_mappings_join_only_the_resolved_value(self) -> None:
        cases = (
            (
                "合同工住宿标准",
                "合同工",
                "合同工属于L2类。",
                "住宿标准：L2类为300元/天。",
                {"answer", "bridge"},
            ),
            (
                "试用期年假天数",
                "试用期",
                "试用期属于入职阶段P0。",
                "试用期年假天数：P0为0天。",
                {"answer", "bridge"},
            ),
            (
                "外包人员的系统权限是什么",
                "外包人员",
                "外包人员归属于访客角色R1。",
                "系统权限：R1仅可查看公开数据。",
                {"answer", "bridge"},
            ),
        )

        for (
            question,
            bridge_subject,
            bridge_content,
            answer_content,
            expected_source_ids,
        ) in cases:
            with self.subTest(question=question):
                requirements = _multi_hop_requirements(
                    question,
                    f"确认{bridge_subject}对应的适用分类",
                )
                bundle = assemble_evidence_bundle(
                    query=question,
                    answer_shape="multi_hop",
                    candidates=[
                        _candidate(
                            "answer",
                            content=answer_content,
                            candidate_origins=[
                                "initial_retrieval",
                                "small_document_full",
                            ],
                            full_document_chunk_count=2,
                        ),
                        _candidate(
                            "bridge",
                            chunk_index=1,
                            content=bridge_content,
                        ),
                    ],
                    requirements=requirements,
                    retrieval_queries=tuple(
                        item.description for item in requirements
                    ),
                    completeness="complete",
                )

                self.assertEqual(bundle.missing_requirement_ids, ())
                self.assertEqual(
                    set(bundle.answer_source_ids),
                    expected_source_ids,
                )

    def test_table_bridge_uses_only_subject_row_and_ignores_leave_approval(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的住宿标准是多少",
            "确认总经理对应的适用分类、等级、类别或阶段（用于确定住宿标准）",
        )
        classification = (
            "| 职级 | 适用人员 |\n"
            "| --- | --- |\n"
            "| A级 | 董事长、总经理、副总经理 |\n"
            "| B级 | 部门总监、高级经理 |\n"
            "| C级 | 部门经理、主管 |\n"
            "| D级 | 普通员工、专员 |"
        )
        leave_approval = (
            "| 请假时长 | 审批人 |\n"
            "| --- | --- |\n"
            "| 1天以内 | 直属主管 |\n"
            "| 5天以上 | 总经理 |"
        )

        self.assertEqual(
            extract_bridge_values(requirements[1].description, classification),
            ("A级",),
        )
        self.assertEqual(
            extract_bridge_values(requirements[1].description, leave_approval),
            (),
        )

        bundle = assemble_evidence_bundle(
            query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("classification", content=classification),
                _candidate(
                    "leave-approval",
                    doc_id="doc-leave",
                    content=leave_approval,
                ),
                _candidate(
                    "grade-a",
                    chunk_index=1,
                    content="住宿标准：A级一线城市不超过1200元/天。",
                ),
                _candidate(
                    "grade-d",
                    chunk_index=2,
                    content="住宿标准：D级一线城市不超过450元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["classification"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["grade-a"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["grade-d"].supports_requirement_ids, ())
        self.assertEqual(by_id["leave-approval"].supports_requirement_ids, ())
        self.assertEqual(
            set(bundle.answer_source_ids),
            {"classification", "grade-a"},
        )
        # Final typed evidence coverage is authoritative over a pipeline-time
        # partial ceiling once the complete bridge path is visible.
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_classification_table_requires_an_explicit_applicable_entity_column(
        self,
    ) -> None:
        description = "确认总经理对应的适用分类、等级、类别或阶段"
        valid_tables = (
            (
                "| 职级 | 适用人员（含实习生） |\n"
                "| --- | --- |\n"
                "| A级 | 总经理 |"
            ),
            (
                "| 适用对象 | 分类 |\n"
                "| --- | --- |\n"
                "| 总经理 | A级 |"
            ),
        )
        for content in valid_tables:
            with self.subTest(content=content):
                self.assertEqual(
                    extract_bridge_values(
                        description,
                        content,
                        bridge_kind="classification",
                    ),
                    ("A级",),
                )

        for header in (
            "审批人员",
            "申请人员",
            "审核人员",
            "负责人",
            "经办人员",
            "适用审批人员",
        ):
            content = (
                f"| 职级 | {header} |\n"
                "| --- | --- |\n"
                "| A级 | 总经理 |"
            )
            with self.subTest(header=header):
                self.assertEqual(
                    extract_bridge_values(
                        description,
                        content,
                        bridge_kind="classification",
                    ),
                    (),
                )

        # An unlabeled two-cell row is not a verifiable classification schema.
        self.assertEqual(
            extract_bridge_values(
                description,
                "总经理 | A级",
                bridge_kind="classification",
            ),
            (),
        )

    def test_mapping_bridge_does_not_borrow_the_classification_table_schema(
        self,
    ) -> None:
        classification_table = (
            "| 职级 | 适用人员 |\n"
            "| --- | --- |\n"
            "| A级 | 总经理 |"
        )

        # A generic mapping request must not inherit the typed
        # ``applicable entity -> taxonomy`` shortcut merely because the source
        # happens to be a classification table.
        self.assertEqual(
            extract_bridge_values(
                "确认总经理对应关系",
                classification_table,
                bridge_kind="mapping",
            ),
            (),
        )
        # Mapping can still use an explicitly named source column, or a prose
        # relation, through its own conservative resolver.
        self.assertEqual(
            extract_bridge_values(
                "确认总经理对应职级",
                classification_table,
                bridge_kind="mapping",
            ),
            ("A级",),
        )
        self.assertEqual(
            extract_bridge_values(
                "确认总经理对应关系",
                "总经理对应A级。",
                bridge_kind="mapping",
            ),
            ("A级",),
        )

    def test_manager_mentions_do_not_manufacture_a_grade_mapping(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的住宿标准是多少",
            "确认总经理对应的适用分类、等级、类别或阶段（用于确定住宿标准）",
        )
        bundle = assemble_evidence_bundle(
            query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "approval",
                    content="A级、B级人员出差需总经理审批。",
                ),
                _candidate(
                    "appendix",
                    chunk_index=1,
                    content="本标准未尽事宜，由总经理办公会研究决定。",
                ),
                _candidate(
                    "grade-a",
                    chunk_index=2,
                    content="住宿标准：A级一线城市不超过1200元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        self.assertTrue(all(
            "r2" not in item.supports_requirement_ids for item in bundle.items
        ))
        self.assertEqual(bundle.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertNotEqual(bundle.state.completeness, "complete")

    def test_cross_domain_named_taxonomies_join_without_business_special_cases(self) -> None:
        cases = (
            (
                "星云产品的数据导出权限是什么",
                "确认星云产品对应的产品级别",
                "产品目录：星云产品属于企业版。",
                "数据导出权限：企业版允许导出业务数据。",
                "数据导出权限：基础版仅允许导出汇总数据。",
            ),
            (
                "供应商甲的风险处置措施是什么",
                "确认供应商甲对应的风险等级",
                "风险评估：供应商甲认定为高风险。",
                "风险处置措施：高风险供应商暂停准入并启动复核。",
                "风险处置措施：低风险供应商保持常规监测。",
            ),
            (
                "合同工的住宿标准是多少",
                "确认合同工对应的岗位等级",
                "用工分类：合同工属于L2类。",
                "住宿标准：L2类为300元/天。",
                "住宿标准：L3类为500元/天。",
            ),
        )

        for question, bridge_description, bridge, answer, wrong in cases:
            with self.subTest(question=question):
                requirements = _multi_hop_requirements(
                    question,
                    bridge_description,
                )
                bundle = assemble_evidence_bundle(
                    query=question,
                    answer_shape="multi_hop",
                    candidates=[
                        _candidate("bridge", content=bridge),
                        _candidate("answer", chunk_index=1, content=answer),
                        _candidate("wrong", chunk_index=2, content=wrong),
                    ],
                    requirements=requirements,
                    retrieval_queries=tuple(
                        item.description for item in requirements
                    ),
                    completeness="partial",
                )
                by_id = {item.chunk_id: item for item in bundle.items}
                self.assertEqual(by_id["bridge"].supports_requirement_ids, ("r2",))
                self.assertEqual(by_id["answer"].supports_requirement_ids, ("r1",))
                self.assertEqual(by_id["wrong"].supports_requirement_ids, ())
                self.assertEqual(bundle.missing_requirement_ids, ())
                self.assertEqual(bundle.state.completeness, "complete")

    def test_local_plan_requires_direct_named_entity_claim_for_attribute_answer(
        self,
    ) -> None:
        from core.rag_v2.query_plan import plan_query_locally

        question = "供应商甲的风险处置措施是什么"
        plan = plan_query_locally(question)
        candidates = [
            _candidate(
                "direct",
                content="供应商甲的风险处置措施为暂停准入并启动复核。",
                candidate_origins=["small_document_full"],
                full_document_chunk_count=4,
            ),
            _candidate(
                "generic-high",
                chunk_index=1,
                content="风险处置措施：高风险供应商暂停准入并启动复核。",
                candidate_origins=["small_document_full"],
                full_document_chunk_count=4,
            ),
            _candidate(
                "generic-low",
                chunk_index=2,
                content="风险处置措施：低风险供应商保持常规监测。",
                candidate_origins=["small_document_full"],
                full_document_chunk_count=4,
            ),
            _candidate(
                "supplier-b",
                chunk_index=3,
                content="供应商乙的风险处置措施为保持常规监测。",
                candidate_origins=["small_document_full"],
                full_document_chunk_count=4,
            ),
        ]

        bundle = assemble_evidence_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            candidates=candidates,
            requirements=plan.requirements,
            retrieval_queries=plan.retrieval_queries,
        )

        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual(bundle.answer_source_ids, ("direct",))
        self.assertNotIn("generic-high", bundle.answer_source_ids)
        self.assertNotIn("generic-low", bundle.answer_source_ids)
        self.assertNotIn("supplier-b", bundle.answer_source_ids)

    def test_direct_named_entity_guard_is_structural_and_not_a_topic_rule(self) -> None:
        """Named direct attributes need a same-claim subject, nothing more.

        This covers the guard's intended boundary: it isolates supplier A/B,
        leaves ordinary category augmentation alone, does not tighten a
        subject-free question, and cannot be bypassed by a document-root topic
        anchor.
        """

        named_question = "供应商甲的风险处置措施是什么"
        self.assertTrue(adjudicate_answer_claims(
            named_question,
            "供应商甲的风险处置措施为暂停准入并启动复核。",
        ))
        for content in (
            "风险处置措施：高风险供应商暂停准入并启动复核。",
            "供应商乙的风险处置措施为保持常规监测。",
        ):
            with self.subTest(content=content):
                self.assertFalse(adjudicate_answer_claims(
                    named_question,
                    content,
                ))
                self.assertFalse(adjudicate_answer_claims(
                    named_question,
                    content,
                    document_root_target_verified=True,
                ))

        # The constraint is tied to a syntactically stable named entity, not
        # to a risk/disposition vocabulary.  A subject-free question remains
        # eligible for a matching source claim.
        self.assertTrue(adjudicate_answer_claims(
            "风险处置措施是什么",
            "风险处置措施：高风险供应商暂停准入并启动复核。",
        ))
        # A declared optional classification edge owns its applicability
        # semantics; the direct named-entity guard must not interfere with it.
        self.assertTrue(adjudicate_answer_claims(
            "普通员工的餐饮补贴是多少",
            "普通员工的餐饮补贴为100元/天。",
            has_bridge_edge=True,
        ))

    def test_condition_requirement_question_uses_the_shared_surface_target(self) -> None:
        """Question grammar and evidence matching must share one target head.

        This is deliberately not a terminology alias for ``条件``.  The user
        text itself supplies the noun after ``满足``; the common surface
        parser reduces only the modal question shell, so a source authored as
        ``报销条件`` remains a direct, auditable match.
        """

        self.assertTrue(adjudicate_answer_claims(
            "报销需要满足什么条件",
            "报销条件：需要正规发票和费用明细。",
        ))
        self.assertFalse(adjudicate_answer_claims(
            "报销需要满足什么条件",
            "采购条件：需要完成供应商准入。",
        ))

    def test_bridge_extraction_rejects_negation_exclusion_and_cross_claims(self) -> None:
        cases = (
            ("确认普通员工对应的职级", "普通员工不属于D级。"),
            ("确认普通员工对应的职级", "除普通员工外，其他人员属于D级。"),
            ("确认普通员工对应的职级", "除专员、普通员工及助理外，其他人员属于D级。"),
            ("确认普通员工对应的职级", "D级适用于除普通员工外的人员。"),
            ("确认星云产品对应的产品级别", "星云产品属于非企业版。"),
            ("确认供应商甲对应的风险等级", "供应商甲并非高风险。"),
            (
                "确认普通员工对应的职级",
                "普通员工信息如下。高级经理对应A级。",
            ),
            (
                "确认普通员工对应的职级",
                "普通员工名单见附件。供应商甲认定为高风险。",
            ),
        )

        for description, content in cases:
            with self.subTest(content=content):
                self.assertEqual(extract_bridge_values(description, content), ())

    def test_bridge_subject_is_an_exact_entity_or_table_list_item(self) -> None:
        description = "确认普通员工对应的职级"
        for content in (
            "非普通员工属于D级。",
            "高级普通员工属于D级。",
            "普通员工家属属于D级。",
            "| 职级 | 适用人员 |\n| --- | --- |\n| D级 | 非普通员工 |",
            "| 职级 | 适用人员 |\n| --- | --- |\n| D级 | 普通员工家属 |",
        ):
            with self.subTest(content=content):
                self.assertEqual(extract_bridge_values(description, content), ())

        self.assertEqual(
            extract_bridge_values(
                description,
                "| 职级 | 适用人员 |\n| --- | --- |\n"
                "| D级 | 专员、普通员工、助理 |",
            ),
            ("D级",),
        )
        for claim, expected in (
            ("普通员工的餐补为100元", True),
            ("普通员工可以查看数据", True),
            ("非普通员工的餐补为100元", False),
            ("高级普通员工的餐补为100元", False),
            ("普通员工家属的餐补为100元", False),
        ):
            with self.subTest(claim=claim):
                self.assertEqual(
                    content_contains_positive_subject(claim, "普通员工"),
                    expected,
                )
        self.assertTrue(content_contains_positive_subject(
            "总经理在北京的住宿上限为1200元",
            "北京",
        ))

    def test_surface_normalization_preserves_condition_bound_answer_terms(self) -> None:
        """A question shell must not turn a direct condition clause into a bridge."""

        terms = answer_target_terms(
            "偏远地区出差有什么补贴",
            bridge_subjects=(),
        )
        self.assertIn("出差补贴", terms)
        self.assertTrue(content_matches_answer_target(
            "偏远地区出差有什么补贴",
            "偏远地区或艰苦地区出差，可申请额外补贴，标准另行审批。",
            bridge_subjects=(),
        ))

    def test_condition_coordination_is_a_positive_subject_boundary(self) -> None:
        for content in (
            "偏远地区或艰苦地区出差，可申请额外补贴。",
            "偏远地区及艰苦地区出差，可申请额外补贴。",
            "偏远地区和艰苦地区出差，可申请额外补贴。",
            "偏远地区与艰苦地区出差，可申请额外补贴。",
            "偏远地区以及艰苦地区出差，可申请额外补贴。",
        ):
            with self.subTest(content=content):
                self.assertTrue(content_contains_positive_subject(
                    content,
                    "偏远地区",
                ))
        for content in (
            "非偏远地区出差，可申请额外补贴。",
            "偏远地区以外出差，可申请额外补贴。",
            "偏远地区家属出差，可申请额外补贴。",
        ):
            with self.subTest(content=content):
                self.assertFalse(content_contains_positive_subject(
                    content,
                    "偏远地区",
                ))

    def test_uncanonicalized_bridge_never_falls_back_to_lexical_completion(
        self,
    ) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="供应商甲的风险处置措施",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="核对供应商甲的风险等级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="供应商甲",
                bridge_kind="classification",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="供应商甲的风险处置措施是什么",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "noise",
                    content="供应商甲风险等级申请流程已经发布。",
                ),
                _candidate(
                    "answer",
                    chunk_index=1,
                    content="风险处置措施：高风险供应商暂停准入。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=(
                "供应商甲的风险处置措施",
                "核对供应商甲的风险等级",
            ),
            completeness="complete",
        )

        self.assertTrue(all(
            "r2" not in item.supports_requirement_ids for item in bundle.items
        ))
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1", "r2"))
        self.assertNotEqual(bundle.state.completeness, "complete")

    def test_shorter_bridge_subject_cannot_erase_residual_query_context(self) -> None:
        """A taxonomy prefix is not the whole user entity/applicability scope."""

        family_requirements = _multi_hop_requirements(
            "普通员工家属的住宿标准是什么",
            "确认普通员工对应的职级",
        )
        family_expansion_queries = build_bridge_expansion_queries(
            family_requirements,
            [_candidate("family-query-bridge", content="普通员工对应D级。")],
        )
        self.assertEqual(len(family_expansion_queries), 1)
        self.assertIn("D级家属", family_expansion_queries[0])
        family_bundle = assemble_evidence_bundle(
            query="普通员工家属的住宿标准是什么",
            answer_shape="multi_hop",
            candidates=[
                _candidate("family-bridge", content="普通员工对应D级。"),
                _candidate(
                    "family-answer",
                    doc_id="family-answer-doc",
                    content="住宿标准：D级不超过450元/天。",
                    filename="公司住宿标准.docx",
                ),
                _candidate(
                    "employee-direct",
                    doc_id="employee-direct-doc",
                    content="普通员工的住宿标准为450元/天。",
                    filename="普通员工住宿标准.docx",
                ),
            ],
            requirements=family_requirements,
            retrieval_queries=tuple(
                item.description for item in family_requirements
            ),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in family_bundle.items}
        self.assertEqual(by_id["family-bridge"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["family-answer"].supports_requirement_ids, ())
        self.assertEqual(by_id["employee-direct"].supports_requirement_ids, ())
        self.assertEqual(family_bundle.missing_requirement_ids, ("r1",))
        self.assertEqual(family_bundle.state.completeness, "partial")

        # The same structural rule still permits a real activity context when
        # it is grounded by the candidate's document topic instead of guessed
        # away by the planner.
        travel_requirements = _multi_hop_requirements(
            "普通员工出差的住宿标准是什么",
            "确认普通员工对应的职级",
        )
        travel_bundle = assemble_evidence_bundle(
            query="普通员工出差的住宿标准是什么",
            answer_shape="multi_hop",
            candidates=[
                _candidate("travel-bridge", content="普通员工对应D级。"),
                _candidate(
                    "travel-answer",
                    doc_id="travel-answer-doc",
                    content="住宿标准：D级不超过450元/天。",
                    filename="公司出差管理标准.docx",
                ),
            ],
            requirements=travel_requirements,
            retrieval_queries=tuple(
                item.description for item in travel_requirements
            ),
            completeness="partial",
        )

        self.assertEqual(travel_bundle.missing_requirement_ids, ())
        self.assertEqual(travel_bundle.state.completeness, "complete")
        self.assertEqual(
            set(travel_bundle.answer_source_ids),
            {"travel-bridge", "travel-answer"},
        )

    def test_bridge_value_matching_uses_exact_positive_boundaries(self) -> None:
        cases = (
            ("R1仅可查看", "R1", True),
            ("R10仅可查看", "R1", False),
            ("D级100元/天", "D级", True),
            ("D级 100元/天", "D级", True),
            ("企业版允许导出", "企业版", True),
            ("非企业版允许导出", "企业版", False),
            ("高风险暂停准入", "高风险", True),
            ("非高风险暂停准入", "高风险", False),
        )
        for content, value, expected in cases:
            with self.subTest(content=content, value=value):
                self.assertEqual(
                    content_contains_bridge_value(content, value),
                    expected,
                )

    def test_target_normalization_preserves_business_action_names(self) -> None:
        cases = (
            ("申请权限是什么", "导出权限：允许导出"),
            ("查询权限是什么", "删除权限：允许删除"),
        )
        for question, unrelated in cases:
            with self.subTest(question=question):
                self.assertFalse(content_matches_answer_target(
                    question,
                    unrelated,
                    bridge_subjects=(),
                ))
                self.assertTrue(content_matches_answer_target(
                    question,
                    f"{question.removesuffix('是什么')}：已开启",
                    bridge_subjects=(),
                ))

        self.assertFalse(content_matches_answer_target(
            "普通员工的餐补金额是多少",
            "聚餐活动补录：D级为100元",
            bridge_subjects=("普通员工",),
        ))
        self.assertFalse(content_matches_answer_target(
            "普通员工的餐补金额是多少",
            "聚餐补录：D级为100元",
            bridge_subjects=("普通员工",),
        ))
        self.assertTrue(content_matches_answer_target(
            "普通员工的餐补金额是多少",
            "餐饮补贴：D级为100元",
            bridge_subjects=("普通员工",),
        ))

    def test_answer_target_bridge_and_result_must_share_one_claim(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的餐饮补贴是多少",
            "确认普通员工对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("bridge", content="普通员工对应D级。"),
                _candidate(
                    "same-claim",
                    chunk_index=1,
                    content="餐饮补贴：D级为100元/天。",
                ),
                _candidate(
                    "split-claim",
                    chunk_index=2,
                    content="餐饮补贴标准如下。D级为100元/天。",
                    filename="餐饮补贴标准.docx",
                ),
                _candidate(
                    "title-only",
                    chunk_index=3,
                    content="D级为100元/天。",
                    filename="餐饮补贴标准.docx",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["same-claim"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["split-claim"].supports_requirement_ids, ())
        self.assertEqual(by_id["title-only"].supports_requirement_ids, ())

    def test_resolved_answer_set_requires_all_bridges_in_one_claim(self) -> None:
        answer = AnswerRequirementV2(
            id="r1",
            description="总经理在北京的住宿标准是多少",
        )
        facts = (
            ResolvedBridgeFact(
                requirement_id="r2",
                subject="总经理",
                value="A级",
                source_chunk_id="manager-map",
                source_doc_id="doc-a",
                source_kb_id="kb-a",
            ),
            ResolvedBridgeFact(
                requirement_id="r3",
                subject="北京",
                value="一线城市",
                source_chunk_id="city-map",
                source_doc_id="doc-a",
                source_kb_id="kb-a",
            ),
        )
        same_sentence = _candidate(
            "answer",
            content="住宿标准：A级在一线城市不超过1200元/天。",
        )
        same_row = _candidate(
            "table-answer",
            content=(
                "## 住宿标准\n"
                "| 职级 | 城市类别 | 上限 |\n"
                "| --- | --- | --- |\n"
                "| A级 | 一线城市 | 1200元/天 |"
            ),
        )
        split_claims = _candidate(
            "split-answer",
            content=(
                "住宿标准：A级不超过1200元/天。"
                "住宿标准：一线城市不超过1200元/天。"
            ),
        )

        for candidate in (same_sentence, same_row):
            self.assertTrue(candidate_supports_resolved_answer_set(
                answer,
                candidate,
                facts,
                bridge_subjects=("总经理", "北京"),
            ))
        self.assertFalse(candidate_supports_resolved_answer_set(
            answer,
            split_claims,
            facts,
            bridge_subjects=("总经理", "北京"),
        ))

    def test_chunk_body_breadcrumb_is_table_semantic_context(self) -> None:
        """DOCX table chunks retain headings as body breadcrumbs, not metadata."""

        answer = AnswerRequirementV2(
            id="r1",
            description="负责人对应的住宿标准是多少",
            depends_on_requirement_ids=("r2",),
        )
        fact = ResolvedBridgeFact(
            requirement_id="r2",
            subject="负责人",
            value="P1级",
            source_chunk_id="role-map",
            source_doc_id="policy-doc",
            source_kb_id="kb-a",
        )
        candidate = _candidate(
            "lodging-table",
            doc_id="policy-doc",
            chunk_index=1,
            content=(
                "【差旅制度 › 住宿费用标准】\n"
                "| 职级 | 一线城市（元/天） | 二线城市（元/天） |\n"
                "| --- | --- | --- |\n"
                "| P1级 | ≤1200 | ≤800 |"
            ),
        )

        self.assertTrue(candidate_supports_resolved_answer_set(
            answer,
            candidate,
            (fact,),
            bridge_subjects=("负责人",),
        ))

    def test_resolved_table_claims_ignore_other_taxonomy_rows(self) -> None:
        """A matrix is not contradictory merely because other classes differ."""

        requirements = _multi_hop_requirements(
            "负责人对应的住宿标准是多少",
            "确认负责人对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="负责人对应的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("role-map", content="负责人对应P1级。"),
                _candidate(
                    "lodging-table",
                    chunk_index=1,
                    content=(
                        "【差旅制度 › 住宿费用标准】\n"
                        "| 职级 | 一线城市（元/天） |\n"
                        "| --- | --- |\n"
                        "| P1级 | ≤1200 |\n"
                        "| P2级 | ≤800 |"
                    ),
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["lodging-table"].supports_requirement_ids, ("r1",))
        self.assertNotIn("conflicting_active_answer_claims", bundle.state.reasons)

    def test_direct_self_contained_answer_does_not_require_optional_classification(self) -> None:
        """A direct policy sentence must not be blocked by an optional bridge."""

        requirements = _classification_augmentation_requirements(
            "普通员工的餐饮补贴是多少",
            "确认普通员工对应的职级",
            subject="普通员工",
        )
        direct = assemble_evidence_bundle(
            query="普通员工的餐饮补贴是多少",
            answer_shape="fact",
            candidates=[_candidate(
                "direct",
                content="普通员工的餐饮补贴为100元/天。",
            )],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )
        self.assertEqual(direct.missing_requirement_ids, ())
        self.assertEqual(direct.answer_source_ids, ("direct",))
        self.assertEqual(direct.state.completeness, "complete")
        self.assertEqual(
            requirements[0].augmentation_requirement_ids,
            ("r2",),
        )
        self.assertEqual(requirements[0].depends_on_requirement_ids, ())

    def test_bridge_subject_in_title_cannot_create_document_root_anchor(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的出差标准是什么",
            "确认总经理对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="总经理的出差标准是什么",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "mapping",
                    content="总经理对应A级。",
                    filename="总经理请假制度.md",
                    candidate_origins=["initial_retrieval"],
                ),
                _candidate(
                    "approval",
                    chunk_index=1,
                    content="## 审批\nA级审批上限为5天。",
                    filename="总经理请假制度.md",
                    candidate_origins=["small_document_full"],
                ),
                _candidate(
                    "leave",
                    chunk_index=2,
                    content="## 休假\nA级每年可休10天。",
                    filename="总经理请假制度.md",
                    candidate_origins=["small_document_full"],
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["mapping"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["approval"].supports_requirement_ids, ())
        self.assertEqual(by_id["leave"].supports_requirement_ids, ())
        self.assertIn("r1", bundle.missing_requirement_ids)

    def test_same_document_does_not_override_explicit_scope_conflicts(self) -> None:
        dimensions = (
            ({"scope_products": ("alpha",)}, {"product": "beta"}),
            ({"scope_versions": ("8.2",)}, {"version": "8.6"}),
            ({"scope_projects": ("project-a",)}, {"project": "project-b"}),
        )
        for fact_scope, candidate_scope in dimensions:
            fact = ResolvedBridgeFact(
                requirement_id="r2",
                subject="对象甲",
                value="L2类",
                source_chunk_id="map",
                source_doc_id="same-doc",
                source_kb_id="kb-a",
                **fact_scope,
            )
            candidate = _candidate(
                "answer",
                doc_id="same-doc",
                metadata=candidate_scope,
            )
            with self.subTest(fact_scope=fact_scope):
                self.assertFalse(
                    bridge_fact_matches_candidate_scope(fact, candidate)
                )

    def test_bridge_expansion_queries_keep_each_explicit_scope(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的住宿标准是多少",
            "确认总经理对应的职级",
        )
        queries = build_bridge_expansion_queries(
            requirements,
            (
                _candidate(
                    "map-82",
                    doc_id="doc-82",
                    content="总经理对应A级。",
                    metadata={"product": "alpha", "version": "8.2"},
                ),
                _candidate(
                    "map-86",
                    doc_id="doc-86",
                    content="总经理对应B级。",
                    metadata={"product": "alpha", "version": "8.6"},
                ),
            ),
        )

        self.assertEqual(len(queries), 2)
        self.assertTrue(any("A级" in query and "8.2" in query for query in queries))
        self.assertTrue(any("B级" in query and "8.6" in query for query in queries))

    def test_conflicting_same_document_bridge_values_fail_closed(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的住宿标准是多少",
            "确认普通员工对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("map-a", content="普通员工对应A级。"),
                _candidate(
                    "map-d",
                    content="普通员工对应D级。",
                ),
                _candidate(
                    "answer-a",
                    chunk_index=1,
                    content="住宿标准：A级为1200元/天。",
                ),
                _candidate(
                    "answer-d",
                    chunk_index=1,
                    content="住宿标准：D级为450元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["map-a"].role, "conflicting")
        self.assertEqual(by_id["map-d"].role, "conflicting")
        self.assertEqual(by_id["map-a"].supports_requirement_ids, ())
        self.assertEqual(by_id["map-d"].supports_requirement_ids, ())
        self.assertEqual(
            by_id["map-a"].metadata["bridge_resolution_statuses"]["r2"]["status"],
            "conflict",
        )
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(bundle.state.completeness, "unknown")

    def test_conflicting_bridge_cannot_reach_final_graph_or_generation(self) -> None:
        """A rejected mapping remains diagnostic-only through finalization."""

        requirements = _multi_hop_requirements(
            "普通员工的住宿标准是多少",
            "确认普通员工对应的职级",
        )
        candidates = [
            _candidate("map-a", content="普通员工对应A级。"),
            _candidate("map-d", content="普通员工对应D级。"),
            _candidate(
                "answer-a",
                chunk_index=1,
                content="住宿标准：A级为1200元/天。",
            ),
            _candidate(
                "answer-d",
                chunk_index=2,
                content="住宿标准：D级为450元/天。",
            ),
        ]
        bundle, task_graph, ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query="普通员工的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=candidates,
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        finalized = finalize_visible_evidence_bundle(
            bundle,
            requirements=_explicit_test_requirements(requirements),
            task_graph=task_graph,
            task_ledger=ledger,
        )

        self.assertEqual(bundle.context_item_ids, ())
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(finalized.context.item_ids, ())
        self.assertEqual(finalized.bundle.context_item_ids, ())
        self.assertEqual(finalized.bundle.answer_source_ids, ())
        self.assertFalse(finalized.generation_allowed)
        self.assertEqual(finalized.bundle.missing_requirement_ids, ("r1",))
        self.assertIsNotNone(finalized.bundle.coverage_graph)
        self.assertFalse(any(
            claim.requirement_id == "r2"
            for claim in finalized.bundle.coverage_graph.claims
        ))

    def test_bridge_conflict_does_not_erase_an_independent_direct_claim(self) -> None:
        """Conflict is edge-local; a same-source direct fact remains usable."""

        requirements = _classification_augmentation_requirements(
            "普通员工的餐饮补贴是多少",
            "确认普通员工对应的职级",
            subject="普通员工",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的餐饮补贴是多少",
            candidates=[
                _candidate(
                    "mixed-source",
                    content=(
                        "普通员工对应A级。普通员工对应D级。"
                        "普通员工的餐饮补贴为100元/天。"
                    ),
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ("r1",))
        self.assertIn(item.role, {"direct", "complement"})
        self.assertIn("bridge_conflicts", item.metadata)
        self.assertEqual(bundle.answer_source_ids, ("mixed-source",))
        self.assertEqual(bundle.missing_requirement_ids, ())

    def test_different_documents_keep_independent_complete_bridge_graphs(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的住宿标准是多少",
            "确认普通员工对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("map-a", content="普通员工对应A级。"),
                _candidate(
                    "answer-a",
                    chunk_index=1,
                    content="住宿标准：A级为1200元/天。",
                ),
                _candidate(
                    "map-d",
                    doc_id="doc-b",
                    content="普通员工对应D级。",
                ),
                _candidate(
                    "answer-d",
                    doc_id="doc-b",
                    chunk_index=1,
                    content="住宿标准：D级为450元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["map-a"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["answer-a"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["map-d"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["answer-d"].supports_requirement_ids, ("r1",))
        self.assertEqual(
            set(bundle.answer_source_ids),
            {"map-a", "answer-a", "map-d", "answer-d"},
        )
        self.assertEqual(bundle.missing_requirement_ids, ())

    def test_bridge_join_never_crosses_incompatible_document_version(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的住宿标准是多少",
            "确认总经理对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "map-82",
                    doc_id="map-doc-82",
                    content="总经理对应A级。",
                    metadata={"version": "8.2"},
                ),
                _candidate(
                    "answer-82",
                    doc_id="policy-doc-82",
                    content="住宿标准：A级为1200元/天。",
                    metadata={"version": "8.2"},
                ),
                _candidate(
                    "answer-86",
                    doc_id="policy-doc-86",
                    content="住宿标准：A级为1600元/天。",
                    metadata={"version": "8.6"},
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["answer-82"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["answer-86"].supports_requirement_ids, ())
        self.assertEqual(
            set(bundle.answer_source_ids),
            {"map-82", "answer-82"},
        )

    def test_answer_target_abbreviation_matches_only_same_semantic_anchor(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的餐补金额是多少",
            "确认普通员工对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的餐补金额是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("bridge", content="普通员工对应D级。"),
                _candidate(
                    "meal",
                    chunk_index=1,
                    content="餐饮补贴：D级为100元/天。",
                ),
                _candidate(
                    "lodging",
                    chunk_index=2,
                    content="住宿补贴：D级为450元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["meal"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ())
        self.assertEqual(set(bundle.answer_source_ids), {"bridge", "meal"})

    def test_broad_manager_travel_question_keeps_all_value_bearing_sections(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="总经理的出差标准是什么",
                depends_on_requirement_ids=("r2",),
                coverage_mode="collection",
            ),
            AnswerRequirementV2(
                id="r2",
                description=(
                    "确认总经理对应的适用分类、等级、类别或阶段"
                    "（用于确定出差标准）"
                ),
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="总经理",
                bridge_kind="classification",
            ),
        )
        filename = "公司出差管理标准.docx"
        full_document_chunk_count = 8
        candidates = [
            _candidate(
                "classification",
                content=(
                    "| 职级 | 适用人员 |\n"
                    "| --- | --- |\n"
                    "| A级 | 董事长、总经理、副总经理 |\n"
                    "| D级 | 普通员工、专员 |"
                ),
                filename=filename,
                candidate_origins=["initial_retrieval"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "flight",
                chunk_index=1,
                content="## 飞机\n| 职级 | 国内航班 |\n| --- | --- |\n| A级 | 头等舱或公务舱 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "train",
                chunk_index=2,
                content="## 火车\n| 职级 | 标准 |\n| --- | --- |\n| A级 | 高铁一等座、火车软卧 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "city",
                chunk_index=3,
                content="## 市内交通\n| 职级 | 标准 |\n| --- | --- |\n| A级 | 出租车、网约车、公务用车 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "lodging",
                chunk_index=4,
                content="## 住宿费用\n| 职级 | 一线城市 |\n| --- | --- |\n| A级 | 不超过1200元/天 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "meal",
                chunk_index=5,
                content="## 餐饮补贴\n| 职级 | 标准 |\n| --- | --- |\n| A级 | 200元/天 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "communication",
                chunk_index=6,
                content=(
                    "## 通讯补贴\n通讯补贴为50元/天，所有职级统一适用。"
                ),
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "leave",
                doc_id="leave-doc",
                content="请假超过5天由总经理审批。",
                filename="员工请假管理办法.docx",
            ),
            _candidate(
                "appendix",
                chunk_index=7,
                content="本标准未尽事宜由总经理办公会研究决定。",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
        ]
        bundle = assemble_evidence_bundle(
            query="总经理的出差标准是什么",
            answer_shape="multi_hop",
            candidates=candidates,
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        answer_ids = {
            "flight",
            "train",
            "city",
            "lodging",
            "meal",
            "communication",
        }
        self.assertTrue(all(
            by_id[chunk_id].supports_requirement_ids == ("r1",)
            for chunk_id in answer_ids
        ))
        self.assertEqual(by_id["classification"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["leave"].supports_requirement_ids, ())
        self.assertEqual(by_id["appendix"].supports_requirement_ids, ())
        self.assertEqual(set(bundle.answer_source_ids), answer_ids | {"classification"})
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_collection_requires_a_verified_full_snapshot(self) -> None:
        requirement = AnswerRequirementV2(
            id="r1",
            description="供应商管理要求是什么",
            depends_on_requirement_ids=(),
            coverage_mode="collection",
        )
        provisional, task_graph, ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query="供应商管理要求是什么",
            answer_shape="overview",
            candidates=[
                _candidate(
                    "single-clause",
                    content="供应商管理要求：必须每年复审一次。",
                    candidate_origins=["initial_retrieval"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=(requirement.description,),
            completeness="complete",
        )
        normalized_requirements = _explicit_test_requirements((requirement,))
        bundle = finalize_visible_evidence_bundle(
            provisional,
            requirements=normalized_requirements,
            task_graph=task_graph,
            task_ledger=ledger,
        ).bundle

        self.assertEqual(provisional.answer_source_ids, ("single-clause",))
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))
        self.assertEqual(bundle.state.completeness, "partial")
        self.assertIn("collection_snapshot_unproven", bundle.state.reasons)

    def test_collection_includes_fragment_is_not_exhaustive(self) -> None:
        cases = (
            (
                "公司出差标准是什么",
                "公司出差标准：交通包括飞机、高铁。",
            ),
            (
                "供应商管理要求是什么",
                "供应商管理要求包括年度复审、审计留痕。",
            ),
        )

        for query, content in cases:
            with self.subTest(query=query):
                requirement = AnswerRequirementV2(
                    id="r1",
                    description=query,
                    coverage_mode="collection",
                )
                provisional, task_graph, ledger = _ledgered_evidence_bundle(
                    return_execution_state=True,
                    query=query,
                    answer_shape="overview",
                    candidates=[
                        _candidate(
                            "partial-list",
                            content=content,
                            candidate_origins=["initial_retrieval"],
                        )
                    ],
                    requirements=(requirement,),
                    retrieval_queries=(query,),
                    completeness="complete",
                )
                normalized_requirements = _explicit_test_requirements((requirement,))
                bundle = finalize_visible_evidence_bundle(
                    provisional,
                    requirements=normalized_requirements,
                    task_graph=task_graph,
                    task_ledger=ledger,
                ).bundle

                self.assertEqual(provisional.answer_source_ids, ("partial-list",))
                self.assertEqual(bundle.answer_source_ids, ())
                self.assertEqual(bundle.missing_requirement_ids, ("r1",))
                self.assertEqual(bundle.state.completeness, "partial")
                self.assertIn(
                    "collection_snapshot_unproven",
                    bundle.state.reasons,
                )

    def test_collection_accepts_target_bound_exhaustive_enumeration(self) -> None:
        query = "供应商管理要求是什么"
        requirement = AnswerRequirementV2(
            id="r1",
            description=query,
            coverage_mode="collection",
            coverage_contract="structured_collection",
        )
        provisional, task_graph, ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query=query,
            answer_shape="overview",
            candidates=[
                _candidate(
                    "closed-list",
                    content=(
                        "供应商管理要求仅包括以下两项："
                        "年度复审、审计留痕。"
                    ),
                    candidate_origins=["initial_retrieval"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=(query,),
            completeness="partial",
        )
        bundle = finalize_visible_evidence_bundle(
            provisional,
            requirements=_explicit_test_requirements((requirement,)),
            task_graph=task_graph,
            task_ledger=ledger,
        ).bundle

        self.assertEqual(bundle.answer_source_ids, ("closed-list",))
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_collection_accepts_target_bound_complete_process(self) -> None:
        query = "采购申请流程是什么"
        requirement = AnswerRequirementV2(
            id="r1",
            description=query,
            coverage_mode="collection",
            coverage_contract="ordered_steps",
        )
        provisional, task_graph, ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query=query,
            answer_shape="overview",
            candidates=[
                _candidate(
                    "closed-process",
                    content=(
                        "采购申请流程如下：\n"
                        "1. 提交申请。\n"
                        "2. 负责人审批。\n"
                        "3. 系统归档。"
                    ),
                    candidate_origins=["initial_retrieval"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=(query,),
            completeness="partial",
        )
        bundle = finalize_visible_evidence_bundle(
            provisional,
            requirements=_explicit_test_requirements((requirement,)),
            task_graph=task_graph,
            task_ledger=ledger,
        ).bundle

        self.assertEqual(bundle.answer_source_ids, ("closed-process",))
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_ordered_process_ignores_incidental_scalar_claims(self) -> None:
        """A purpose clause or deadline cannot replace the declared steps."""

        query = "公司的请假流程是什么"
        requirement = AnswerRequirementV2(
            id="r1",
            description=query,
            coverage_mode="collection",
            coverage_contract="ordered_steps",
        )
        provisional, task_graph, ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query=query,
            answer_shape="process",
            candidates=[
                _candidate(
                    "leave-purpose",
                    content=(
                        "本办法用于规范公司请假管理流程，"
                        "自2026年1月1日起施行。"
                    ),
                    candidate_origins=["initial_retrieval"],
                ),
                _candidate(
                    "leave-process",
                    chunk_index=4,
                    content=(
                        "公司请假流程如下："
                        "（一）填写申请单；"
                        "（二）按权限逐级审批；"
                        "（三）审批后交人力资源部备案；"
                        "（四）假期结束后办理销假。"
                        "突发情况须在返岗后1个工作日内补办。"
                    ),
                    candidate_origins=["initial_retrieval"],
                ),
            ],
            requirements=(requirement,),
            retrieval_queries=(query,),
            rerank_succeeded=False,
            expansion_succeeded=True,
        )

        finalized = finalize_visible_evidence_bundle(
            provisional,
            requirements=_explicit_test_requirements((requirement,)),
            task_graph=task_graph,
            task_ledger=ledger,
        )
        bundle = finalized.bundle

        self.assertTrue(finalized.generation_allowed)
        self.assertEqual(bundle.answer_source_ids, ("leave-process",))
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")
        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["leave-purpose"].supports_requirement_ids, ())
        assertions = by_id["leave-process"].metadata["answer_claim_assertions"]
        self.assertEqual(assertions["r1"][0]["result_kind"], "ordered_steps")

    def test_collection_requires_every_part_of_target_bound_table(self) -> None:
        query = "系统支持的登录方式有哪些"
        requirement = AnswerRequirementV2(
            id="r1",
            description=query,
            coverage_mode="collection",
        )
        candidates = [
            _candidate(
                f"login-part-{index}",
                chunk_index=index,
                content=(
                    "| 系统支持的登录方式 | 说明 |\n"
                    "| --- | --- |\n"
                    f"| {method} | {description} |"
                ),
                filename="系统使用手册.md",
                candidate_origins=[
                    "initial_retrieval" if index == 0 else "same_section"
                ],
                section_path=["系统使用手册.md", "系统支持的登录方式"],
                table_id="login-methods",
                table_part_index=index,
                table_part_count=2,
            )
            for index, (method, description) in enumerate((
                ("密码登录", "使用账号密码"),
                ("单点登录", "使用企业身份源"),
            ))
        ]

        complete_provisional, complete_graph, complete_ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query=query,
            answer_shape="list",
            candidates=candidates,
            requirements=(requirement,),
            retrieval_queries=(query,),
            completeness="partial",
        )
        incomplete_provisional, incomplete_graph, incomplete_ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query=query,
            answer_shape="list",
            candidates=candidates[:1],
            requirements=(requirement,),
            retrieval_queries=(query,),
            completeness="complete",
        )
        complete_requirements = _explicit_test_requirements((requirement,))
        incomplete_requirements = _explicit_test_requirements((requirement,))
        complete = finalize_visible_evidence_bundle(
            complete_provisional,
            requirements=complete_requirements,
            task_graph=complete_graph,
            task_ledger=complete_ledger,
        ).bundle
        incomplete = finalize_visible_evidence_bundle(
            incomplete_provisional,
            requirements=incomplete_requirements,
            task_graph=incomplete_graph,
            task_ledger=incomplete_ledger,
        ).bundle

        self.assertEqual(
            set(complete.answer_source_ids),
            {"login-part-0", "login-part-1"},
        )
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(complete.state.completeness, "complete")
        self.assertEqual(incomplete.missing_requirement_ids, ("r1",))
        self.assertEqual(incomplete.state.completeness, "partial")

    def test_collection_downgrades_when_context_budget_drops_one_table_part(self) -> None:
        requirement = AnswerRequirementV2(
            id="r1",
            description="系统支持的登录方式有哪些",
            depends_on_requirement_ids=(),
            coverage_mode="collection",
        )
        candidates = [
            _candidate(
                "login-part-0",
                chunk_index=0,
                content=(
                    "| 系统支持的登录方式 | 说明 |\n"
                    "| --- | --- |\n"
                    "| 密码登录 | 使用账号密码 |"
                ),
                candidate_origins=["initial_retrieval"],
                table_id="login-methods",
                table_part_index=0,
                table_part_count=2,
            ),
            _candidate(
                "login-part-1",
                chunk_index=1,
                content=(
                    "| 系统支持的登录方式 | 说明 |\n"
                    "| --- | --- |\n"
                    "| 单点登录 | 使用企业身份源 |"
                ),
                candidate_origins=["same_section"],
                table_id="login-methods",
                table_part_index=1,
                table_part_count=2,
            ),
        ]
        provisional, task_graph, ledger = _ledgered_evidence_bundle(
            return_execution_state=True,
            query="系统支持的登录方式有哪些",
            answer_shape="list",
            candidates=candidates,
            requirements=(requirement,),
            retrieval_queries=(requirement.description,),
            completeness="complete",
            max_context_chunks=1,
        )
        normalized_requirements = _explicit_test_requirements((requirement,))
        bundle = finalize_visible_evidence_bundle(
            provisional,
            requirements=normalized_requirements,
            task_graph=task_graph,
            task_ledger=ledger,
            max_context_chunks=1,
        ).bundle

        self.assertEqual(len(provisional.answer_source_ids), 1)
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))
        self.assertEqual(bundle.state.completeness, "partial")
        self.assertIn("collection_context_incomplete", bundle.state.reasons)

    def test_required_coverage_precedes_higher_scored_background_under_budget(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="alpha target"),
            AnswerRequirementV2(id="r2", description="beta target"),
        )
        bundle = assemble_evidence_bundle(
            query="two targets",
            candidates=[
                _candidate("background", content="generic notes", score=100),
                _candidate(
                    "r1",
                    chunk_index=1,
                    content="alpha target: first mapped result",
                    score=0.1,
                    role="direct",
                    supports_requirement_ids=["r1"],
                ),
                _candidate(
                    "r2",
                    chunk_index=2,
                    content="beta target: second mapped result",
                    score=0.1,
                    role="direct",
                    supports_requirement_ids=["r2"],
                ),
            ],
            requirements=requirements,
            retrieval_queries=("alpha target", "beta target"),
            rerank_succeeded=True,
            completeness="complete",
            max_context_chunks=2,
        )

        self.assertEqual(bundle.context_item_ids, ("r1", "r2"))
        self.assertEqual(bundle.answer_source_ids, ("r1", "r2"))
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_invalid_query_indexes_cannot_manufacture_requirement_support(self) -> None:
        requirement = AnswerRequirementV2(id="r1", description="目标要求")
        bundle = assemble_evidence_bundle(
            query="目标要求",
            candidates=[
                _candidate(
                    "invalid-indexes",
                    content="完全无关正文",
                    expansion_query_indexes=[True, -1, 9, "bad"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=("目标要求",),
            completeness="complete",
        )

        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.items[0].role, "background")
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_unexecuted_reranker_cannot_mark_candidate_verified_or_direct(self) -> None:
        bundle = assemble_evidence_bundle(
            query="配置",
            candidates=[
                _candidate(
                    "legacy",
                    content="配置说明",
                    rerank_status="verified",
                    evidence_role="direct",
                )
            ],
            rerank_succeeded=None,
        )

        self.assertEqual(bundle.items[0].confidence, "retrieved")
        self.assertEqual(bundle.items[0].role, "background")

    def test_stale_support_annotations_are_ignored_without_verification(self) -> None:
        requirement = AnswerRequirementV2(id="r1", description="fresh target")
        bundle = assemble_evidence_bundle(
            query="fresh target",
            candidates=[
                _candidate(
                    "stale",
                    content="old unrelated context",
                    role="direct",
                    supports_requirement_ids=["r1"],
                    rerank_status="verified",
                )
            ],
            requirements=(requirement,),
            retrieval_queries=("fresh target",),
            rerank_succeeded=None,
            completeness="complete",
        )

        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.items[0].role, "background")
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_dependency_edge_is_coverage_critical_for_every_answer_shape(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="answer target",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="bridge target",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="target",
                bridge_kind="mapping",
            ),
        )
        values = dict(
            query="resolve target",
            candidates=[
                _candidate(
                    "answer",
                    content="answer target: mapped value",
                    expansion_query_indexes=[0],
                )
            ],
            requirements=requirements,
            retrieval_queries=("answer target", "bridge target"),
            completeness="complete",
        )

        multi_hop = assemble_evidence_bundle(answer_shape="multi_hop", **values)
        ordinary = assemble_evidence_bundle(answer_shape="multi_part", **values)

        self.assertEqual(multi_hop.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(multi_hop.state.completeness, "unknown")
        self.assertEqual(ordinary.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(ordinary.state.completeness, "unknown")

    def test_hard_mismatch_and_unauthorized_candidates_are_excluded(self) -> None:
        constraints = extract_query_constraints("云枢8.2.75消息接口怎么配置")
        candidates = [
            _candidate(
                "a-2",
                chunk_index=2,
                content="所属产品：云枢；产品版本：8.2.75。配置项B。",
                rerank_status="verified",
            ),
            _candidate(
                "b-0",
                doc_id="doc-b",
                content="所属产品：云枢；产品版本：7.0。旧配置项。",
                rerank_status="verified",
            ),
            _candidate(
                "a-0",
                chunk_index=0,
                content="所属产品：云枢；产品版本：8.2.75。",
                rerank_status="verified",
            ),
            _candidate("secret", doc_id="doc-secret", authorized=False),
        ]

        bundle = assemble_evidence_bundle(
            query="云枢8.2.75消息接口怎么配置",
            candidates=candidates,
            constraints=constraints,
            rerank_succeeded=True,
        )

        self.assertEqual([item.chunk_id for item in bundle.items], ["a-0", "a-2"])
        self.assertTrue(all(item.doc_id == "doc-a" for item in bundle.items))
        self.assertIn("hard_constraint_mismatch_excluded", bundle.state.reasons)
        self.assertIn("unauthorized_candidate_excluded", bundle.state.reasons)
        self.assertEqual(bundle.state.confidence, "verified")

    def test_explicit_product_version_excludes_unknown_scope(self) -> None:
        constraints = extract_query_constraints("云枢8.6登录配置")
        bundle = assemble_evidence_bundle(
            query="云枢8.6登录配置",
            candidates=[
                _candidate(
                    "exact",
                    content="所属产品：云枢；产品版本：8.6。登录配置A。",
                ),
                _candidate(
                    "unknown",
                    doc_id="doc-generic",
                    content="通用登录配置B。",
                ),
            ],
            constraints=constraints,
        )

        self.assertEqual([item.chunk_id for item in bundle.items], ["exact"])
        self.assertIn("hard_constraint_unknown_excluded", bundle.state.reasons)

    def test_items_are_grouped_by_document_and_sorted_by_chunk_index(self) -> None:
        bundle = assemble_evidence_bundle(
            query="配置标准",
            candidates=[
                _candidate("a-3", chunk_index=3),
                _candidate("b-2", doc_id="doc-b", chunk_index=2),
                _candidate("a-1", chunk_index=1),
                _candidate("b-0", doc_id="doc-b", chunk_index=0),
            ],
        )

        self.assertEqual(
            [(item.doc_id, item.chunk_index) for item in bundle.items],
            [("doc-a", 1), ("doc-a", 3), ("doc-b", 0), ("doc-b", 2)],
        )

    def test_rerank_failure_downgrades_but_does_not_erase_candidates(self) -> None:
        bundle = assemble_evidence_bundle(
            query="普通员工住宿标准",
            candidates=[
                _candidate(
                    "retrieved",
                    evidence_role="irrelevant",
                    rerank_status="unverified",
                )
            ],
            rerank_succeeded=False,
            completeness="complete",
        )

        self.assertEqual(bundle.state.availability, "degraded")
        self.assertEqual(bundle.state.confidence, "retrieved")
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual([item.chunk_id for item in bundle.context_items], ["retrieved"])
        self.assertEqual(bundle.answer_sources, ())
        self.assertIn("rerank_degraded", bundle.state.reasons)

    def test_expansion_degradation_is_independent_from_confidence_and_completeness(self) -> None:
        bundle = assemble_evidence_bundle(
            query="普通员工住宿标准",
            candidates=[
                _candidate("verified", rerank_status="verified")
            ],
            rerank_succeeded=True,
            expansion_succeeded=False,
            completeness="complete",
        )

        self.assertEqual(bundle.state.availability, "degraded")
        self.assertEqual(bundle.state.confidence, "verified")
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual(len(bundle.items), 1)
        self.assertIn("expansion_degraded", bundle.state.reasons)

    def test_missing_requirements_mark_partial_without_clearing_context(self) -> None:
        bundle = assemble_evidence_bundle(
            query="交通和住宿标准",
            candidates=[_candidate("住宿")],
            completeness="complete",
            missing_requirement_ids=("transport",),
        )

        self.assertEqual(bundle.state.completeness, "partial")
        self.assertEqual(bundle.missing_requirement_ids, ("transport",))
        self.assertEqual(bundle.context_item_ids, ("住宿",))

    def test_concrete_table_outranks_boilerplate_unless_explicitly_requested(self) -> None:
        candidates = [
            _candidate(
                "overview",
                chunk_index=0,
                content="一、总则\n为规范公司员工出差管理，制定本制度。",
                evidence_role="direct",
            ),
            _candidate(
                "table",
                chunk_index=1,
                content="| 职级 | 住宿标准 |\n| --- | --- |\n| D级 | 450元/天 |",
                evidence_role="related",
            ),
        ]

        concrete = assemble_evidence_bundle(
            query="普通员工住宿标准是多少",
            candidates=candidates,
            max_context_chunks=1,
        )
        overview = assemble_evidence_bundle(
            query="这份制度的总则是什么",
            candidates=candidates,
            answer_shape="overview",
            requirements=(
                AnswerRequirementV2(
                    id="r1",
                    description="这份制度的总则是什么",
                ),
            ),
            retrieval_queries=("这份制度的总则是什么",),
            max_context_chunks=1,
        )

        self.assertEqual(concrete.context_item_ids, ("table",))
        self.assertEqual(overview.context_item_ids, ("overview",))

    def test_overview_accepts_only_full_document_chunks_anchored_by_retrieval(self) -> None:
        bundle = assemble_evidence_bundle(
            query="请概述这份制度的主要内容",
            answer_shape="overview",
            candidates=[
                _candidate("anchor", content="公司出差管理标准")
            ],
            overview_candidates=[
                _candidate("full-1", chunk_index=1, content="一、总则：规范出差管理。"),
                _candidate(
                    "foreign",
                    doc_id="doc-foreign",
                    content="不属于已授权召回文档的全文。",
                ),
            ],
            requirements=(
                AnswerRequirementV2(
                    id="r1",
                    description="请概述这份制度的主要内容",
                ),
            ),
            retrieval_queries=("请概述这份制度的主要内容",),
            rerank_succeeded=True,
        )

        self.assertEqual(
            {item.chunk_id for item in bundle.items},
            {"anchor", "full-1"},
        )
        full = next(item for item in bundle.items if item.chunk_id == "full-1")
        self.assertEqual(full.confidence, "retrieved")
        self.assertIn("overview_full_document", full.origins)
        self.assertIn("unanchored_overview_candidate_excluded", bundle.state.reasons)

    def test_context_budget_never_truncates_a_source_claim(self) -> None:
        bundle = assemble_evidence_bundle(
            query="标准",
            candidates=[
                _candidate("first", content="1234567890", score=1.0),
                _candidate("second", chunk_index=1, content="abcdefghij", score=0.5),
            ],
            completeness="complete",
            max_context_chunks=1,
            max_context_chars=5,
        )

        # A partial prefix can omit the very predicate/value that made a
        # source admissible.  The evidence budget is therefore indivisible:
        # reject an oversized item instead of emitting an unsafe truncation.
        self.assertEqual(bundle.context_item_ids, ())
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.items[0].content, "1234567890")
        self.assertNotIn("context_truncated", bundle.items[0].metadata)
        self.assertIn("context_budget_limited", bundle.state.reasons)
        self.assertEqual(bundle.state.completeness, "unknown")

    def test_all_filtered_candidates_are_a_normal_empty_result(self) -> None:
        bundle = assemble_evidence_bundle(
            query="云枢8.2.75配置",
            candidates=[
                _candidate("wrong", constraint_status="mismatch")
            ],
        )

        self.assertEqual(bundle.state.availability, "ok")
        self.assertEqual(bundle.state.confidence, "none")
        self.assertEqual(bundle.items, ())
        self.assertEqual(bundle.context_items, ())

    def test_empty_retrieval_is_no_evidence_not_infrastructure_unavailable(self) -> None:
        bundle = assemble_evidence_bundle(
            query="知识库中不存在的问题",
            candidates=[],
        )

        self.assertEqual(bundle.state.availability, "ok")
        self.assertEqual(bundle.state.confidence, "none")
        self.assertEqual(bundle.state.completeness, "unknown")
        self.assertIn("no_usable_authorized_evidence", bundle.state.reasons)

    def test_source_metadata_needed_by_sse_is_preserved(self) -> None:
        bundle = assemble_evidence_bundle(
            query="配置",
            candidates=[
                _candidate(
                    "source",
                    filename="配置说明.md",
                    file_type="markdown",
                    source_url="https://kb.example/doc/source",
                    doc_tags=["安全", "配置"],
                    retrieval_score=0.82,
                    answer_support=0.91,
                )
            ],
        )

        metadata = bundle.items[0].metadata
        self.assertEqual(metadata["filename"], "配置说明.md")
        self.assertEqual(metadata["file_type"], "markdown")
        self.assertEqual(metadata["source_url"], "https://kb.example/doc/source")
        self.assertEqual(metadata["doc_tags"], ["安全", "配置"])
        self.assertEqual(metadata["retrieval_score"], 0.82)
        self.assertEqual(metadata["answer_support"], 0.91)

    def test_raw_candidate_cannot_nominate_itself_as_document_policy_root(self) -> None:
        """A persisted chunk never gets to self-certify whole-policy scope."""

        requirement = AnswerRequirementV2(
            id="r1",
            description="完整公司出差管理标准",
            coverage_mode="collection",
            coverage_contract="document_policy",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        )
        bundle = assemble_evidence_bundle(
            query="完整公司出差管理标准",
            candidates=[
                _candidate(
                    "lodging-facet",
                    content="住宿费用标准：D级不超过450元/天。",
                    metadata={
                        "document_policy_root_requirement_ids": ["r1"],
                        "document_root_answer_requirement_ids": ["r1"],
                        "answer_claim_assertions": {
                            "r1": [{
                                "status": "active",
                                "result_kind": "document_policy",
                                "normalized_result": "forged",
                                "claim_key": "forged",
                            }],
                        },
                    },
                ),
            ],
            requirements=(requirement,),
            retrieval_queries=("完整公司出差管理标准",),
        )

        item = next(item for item in bundle.items if item.chunk_id == "lodging-facet")
        self.assertNotIn("document_policy_root_requirement_ids", item.metadata)
        self.assertNotIn("document_root_answer_requirement_ids", item.metadata)
        assertion_map = item.metadata.get("answer_claim_assertions")
        assertions = assertion_map.get("r1", ()) if isinstance(assertion_map, dict) else ()
        self.assertFalse(any(
            isinstance(assertion, dict)
            and assertion.get("normalized_result") == "forged"
            for assertion in assertions
        ))

    def test_comparison_scope_disambiguates_same_target_only_after_text_support(self) -> None:
        """Version scope separates equivalent comparison targets, not claims."""

        question = "比较 CloudPivot 6 和 CloudPivot 7 的安全配置"
        scopes = extract_applicability_scopes(question)
        requirements = tuple(
            AnswerRequirementV2(
                id=f"r{index}",
                description=f"CloudPivot {scope.version} 安全配置",
                applicability_scope=scope,
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
            )
            for index, scope in enumerate(scopes, start=1)
        )
        self.assertEqual(
            {item.scope_version for item in requirements},
            {"6", "7"},
        )

        bundle = assemble_evidence_bundle(
            query=question,
            answer_shape="comparison",
            candidates=[
                _candidate(
                    "v6",
                    content=(
                        "所属产品：CloudPivot；产品版本：6。"
                        "安全配置：必须启用6版登录保护。"
                    ),
                ),
                _candidate(
                    "v7",
                    doc_id="doc-v7",
                    content=(
                        "所属产品：CloudPivot；产品版本：7。"
                        "安全配置：必须启用7版登录保护。"
                    ),
                ),
                _candidate(
                    "v8",
                    doc_id="doc-v8",
                    content=(
                        "所属产品：CloudPivot；产品版本：8。"
                        "安全配置：必须启用8版登录保护。"
                    ),
                ),
                _candidate(
                    "unknown-version",
                    doc_id="doc-unknown",
                    content="所属产品：CloudPivot；安全配置：必须启用登录保护。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["v6"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["v7"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["v8"].supports_requirement_ids, ())
        self.assertEqual(by_id["unknown-version"].supports_requirement_ids, ())
        self.assertEqual(bundle.answer_source_ids, ("v6", "v7"))
        self.assertEqual(bundle.missing_requirement_ids, ())

    def test_project_scope_uses_source_identity_not_global_or_missing_project(self) -> None:
        """Project is part of the same scope contract as product/version."""

        scope_a = extract_applicability_scope(
            "中青建安项目的CloudPivot6安全配置"
        )
        scope_b = extract_applicability_scope(
            "华东示范项目的CloudPivot6安全配置"
        )
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="CloudPivot 6 安全配置",
                applicability_scope=scope_a,
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
            ),
            AnswerRequirementV2(
                id="r2",
                description="CloudPivot 6 安全配置",
                applicability_scope=scope_b,
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
            ),
        )

        bundle = assemble_evidence_bundle(
            query="比较中青建安项目和华东示范项目的CloudPivot6安全配置",
            answer_shape="comparison",
            candidates=[
                _candidate(
                    "project-a",
                    content=(
                        "所属产品：CloudPivot；产品版本：6；"
                        "所属项目：中青建安；安全配置：必须启用项目A登录保护。"
                    ),
                ),
                _candidate(
                    "project-b",
                    doc_id="doc-project-b",
                    content=(
                        "所属产品：CloudPivot；产品版本：6；"
                        "所属项目：华东示范；安全配置：必须启用项目B登录保护。"
                    ),
                ),
                _candidate(
                    "project-unknown",
                    doc_id="doc-project-unknown",
                    content=(
                        "所属产品：CloudPivot；产品版本：6；"
                        "安全配置：必须启用登录保护。"
                    ),
                ),
                _candidate(
                    "global-clause",
                    doc_id="doc-global",
                    content=(
                        "所属产品：CloudPivot；产品版本：6；"
                        "适用项目：全局；安全配置：必须启用统一登录保护。"
                    ),
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["project-a"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["project-b"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["project-unknown"].supports_requirement_ids, ())
        self.assertEqual(by_id["global-clause"].supports_requirement_ids, ())
        self.assertEqual(bundle.answer_source_ids, ("project-a", "project-b"))


if __name__ == "__main__":
    unittest.main()
