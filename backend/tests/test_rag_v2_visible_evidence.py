import unittest
from dataclasses import replace
from itertools import product

from core.rag_v2.bridge_resolution import partition_bridge_facts, resolve_bridge_facts
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.evidence import (
    assemble_evidence_bundle,
    assemble_unverified_candidate_bundle_with_diagnostics,
    finalize_visible_evidence_bundle,
)
from core.rag_v2.task_execution import BridgeResolution, TaskExecutionLedger
from core.rag_v2.task_graph import compile_rag_execution_bundle


def _candidate(chunk_id: str, content: str, *, chunk_index: int = 0) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "travel-policy",
        "kb_id": "travel-kb",
        "chunk_index": chunk_index,
        "content": content,
    }


def _explicit_requirements(
    requirements: tuple[AnswerRequirementV2, ...],
) -> tuple[AnswerRequirementV2, ...]:
    """Make direct/bridge edges explicit before compiling a V2 task graph."""

    return tuple(
        requirement
        if requirement.role == "bridge"
        else replace(
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
        )
        for requirement in requirements
    )


def _ledgered_evidence_bundle(
    *,
    query: str,
    candidates: tuple[dict, ...],
    requirements: tuple[AnswerRequirementV2, ...],
    retrieval_queries: tuple[str, ...],
    answer_shape: str,
    max_context_chars: int = 16_000,
) -> tuple[
    object,
    tuple[AnswerRequirementV2, ...],
    object,
    TaskExecutionLedger,
]:
    """Build test evidence through the same task/ledger contract as runtime.

    These tests deliberately do not attach task ids or support ids to
    candidates.  Every source is observed by a request-local execution, each
    bridge obtains one terminal semantic resolution, and any second-hop path
    is recorded with its exact parent bridge facts.
    """

    normalized_requirements = _explicit_requirements(requirements)
    plan = QueryPlanV2(
        original_query=query,
        answer_shape=answer_shape,
        retrieval_queries=retrieval_queries,
        requirements=normalized_requirements,
        confidence=0.95,
        source="local",
    )
    execution_bundle = compile_rag_execution_bundle(plan)
    assert execution_bundle.uses_task_ledger
    assert execution_bundle.task_graph is not None
    task_graph = execution_bundle.task_graph
    ledger = TaskExecutionLedger(task_graph, run_id="visible-evidence-test")
    bridge_facts_by_task: dict[str, tuple] = {}

    for task in task_graph.tasks:
        if task.role != "bridge":
            continue
        execution_id = ledger.begin_execution(
            kind="test_bridge_query",
            query=task.query,
            task_ids=(task.task_id,),
        )
        observed = ledger.observe_candidates(
            candidates,
            execution_id=execution_id,
        )
        ledger.finish_execution(
            execution_id,
            status="succeeded",
            candidate_count=len(observed),
        )
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

    for task in task_graph.tasks:
        if task.role != "answer":
            continue
        execution_id = ledger.begin_execution(
            kind="test_answer_query",
            query=task.query,
            task_ids=(task.task_id,),
        )
        ledger.observe_candidates(candidates, execution_id=execution_id)
        ledger.finish_execution(
            execution_id,
            status="succeeded",
            candidate_count=len(candidates),
        )
        for mode in ("proof", "augmentation"):
            for path in task_graph.answer_bridge_paths(mode=mode):
                if path.answer_task_id != task.task_id:
                    continue
                parent_fact_sets = tuple(
                    bridge_facts_by_task.get(parent_task_id, ())
                    for parent_task_id in path.bridge_task_ids
                )
                if not parent_fact_sets or any(not values for values in parent_fact_sets):
                    continue
                for facts in product(*parent_fact_sets):
                    second_hop_id = ledger.begin_execution(
                        kind="test_bridge_second_hop",
                        query=task.query,
                        task_ids=(task.task_id,),
                        parent_task_ids=path.bridge_task_ids,
                        parent_chunk_ids=tuple(
                            fact.source_chunk_id for fact in facts
                        ),
                        route_kind="bridge_second_hop",
                        bridge_edge_mode=path.edge_mode,
                    )
                    ledger.observe_candidates(
                        candidates,
                        execution_id=second_hop_id,
                        parent_task_ids=path.bridge_task_ids,
                        parent_chunk_ids=tuple(
                            fact.source_chunk_id for fact in facts
                        ),
                    )
                    ledger.finish_execution(
                        second_hop_id,
                        status="succeeded",
                        candidate_count=len(candidates),
                    )

    bundle = assemble_evidence_bundle(
        query=query,
        candidates=candidates,
        requirements=normalized_requirements,
        retrieval_queries=retrieval_queries,
        task_graph=task_graph,
        task_ledger=ledger,
        answer_shape=answer_shape,
        max_context_chars=max_context_chars,
    )
    return bundle, normalized_requirements, task_graph, ledger


class VisibleEvidenceFinalizationTests(unittest.TestCase):
    def test_related_evidence_admission_is_bounded_and_labeled(self) -> None:
        result = assemble_unverified_candidate_bundle_with_diagnostics(
            candidates=[
                {
                    **_candidate(
                        "related-2",
                        "升级步骤二：执行迁移。",
                        chunk_index=1,
                    ),
                    "supports_requirement_ids": ["r1"],
                    "evidence_role": "related",
                },
                {
                    **_candidate(
                        "related-1",
                        "升级步骤一：先备份数据库。",
                        chunk_index=0,
                    ),
                    "supports_requirement_ids": ["r1"],
                    "evidence_role": "related",
                },
            ],
            allowed_requirement_ids=("r1",),
            admission_reason="related_evidence_admitted:semantic_coverage_incomplete",
        )

        self.assertIsNotNone(result.bundle)
        assert result.bundle is not None
        self.assertEqual(result.bundle.state.confidence, "retrieved")
        self.assertEqual(
            result.bundle.state.reasons,
            ("related_evidence_admitted:semantic_coverage_incomplete",),
        )
        self.assertEqual(
            result.bundle.answer_source_ids,
            ("related-1", "related-2"),
        )

    def test_bridge_only_evidence_never_reaches_the_generation_context(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的餐补是多少",
                depends_on_requirement_ids=("r2",),
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
        provisional, normalized_requirements, task_graph, ledger = _ledgered_evidence_bundle(
            query="普通员工的餐补是多少",
            candidates=(_candidate("mapping", "普通员工对应D级。"),),
            requirements=requirements,
            retrieval_queries=("普通员工的餐补是多少", "确认普通员工对应的职级"),
            answer_shape="multi_hop",
        )

        finalized = finalize_visible_evidence_bundle(
            provisional,
            requirements=normalized_requirements,
            task_graph=task_graph,
            task_ledger=ledger,
        )

        self.assertEqual(finalized.bundle.context_item_ids, ())
        self.assertEqual(finalized.bundle.answer_source_ids, ())
        self.assertEqual(finalized.context.item_ids, ())
        self.assertFalse(finalized.generation_allowed)
        self.assertEqual(finalized.bundle.missing_requirement_ids, ("r1",))

    def test_final_graph_context_and_sources_share_one_closed_direct_claim(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的餐补是多少",
            ),
        )
        provisional, normalized_requirements, task_graph, ledger = _ledgered_evidence_bundle(
            query="普通员工的餐补是多少",
            candidates=(_candidate("meal", "普通员工餐补标准为100元/天。"),),
            requirements=requirements,
            retrieval_queries=("普通员工的餐补是多少",),
            answer_shape="fact",
        )

        finalized = finalize_visible_evidence_bundle(
            provisional,
            requirements=normalized_requirements,
            task_graph=task_graph,
            task_ledger=ledger,
        )

        self.assertEqual(finalized.context.item_ids, ("meal",))
        self.assertEqual(finalized.bundle.context_item_ids, ("meal",))
        self.assertEqual(finalized.bundle.answer_source_ids, ("meal",))
        self.assertEqual(
            finalized.bundle.coverage_graph.visible_evidence_item_ids,
            finalized.context.item_ids,
        )
        self.assertEqual(finalized.assessment.completeness, "complete")
        self.assertEqual(finalized.answer_claim_item_ids, ("meal",))
        self.assertTrue(finalized.generation_allowed)

    def test_exact_header_budget_keeps_an_independent_answer_after_long_route_fails(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="住宿标准是多少"),
            AnswerRequirementV2(id="r2", description="餐补是多少"),
        )
        provisional, normalized_requirements, task_graph, ledger = _ledgered_evidence_bundle(
            query="住宿和餐补分别是多少",
            candidates=(
                _candidate(
                    "oversized-lodging",
                    "住宿标准为450元/天。" + "补充说明" * 100,
                ),
                _candidate("meal", "餐补标准为100元/天。", chunk_index=1),
            ),
            requirements=requirements,
            retrieval_queries=("住宿标准是多少", "餐补是多少"),
            answer_shape="multi_part",
            max_context_chars=160,
        )

        finalized = finalize_visible_evidence_bundle(
            provisional,
            requirements=normalized_requirements,
            task_graph=task_graph,
            task_ledger=ledger,
            max_context_chars=160,
        )

        self.assertEqual(finalized.context.item_ids, ("meal",))
        self.assertFalse(finalized.context.truncated)
        self.assertEqual(finalized.bundle.answer_source_ids, ("meal",))
        self.assertEqual(finalized.bundle.missing_requirement_ids, ("r1",))
        self.assertTrue(finalized.generation_allowed)


if __name__ == "__main__":
    unittest.main()
