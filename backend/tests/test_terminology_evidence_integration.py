"""Strict terminology proof integration at the evidence boundary."""

from __future__ import annotations

import unittest

from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.evidence import (
    assemble_evidence_bundle,
    finalize_visible_evidence_bundle,
)
from core.rag_v2.task_execution import TaskExecutionLedger
from core.rag_v2.task_graph import compile_rag_execution_bundle
from core.terminology_contracts import TerminologyBinding, TerminologyForm
from core.terminology_runtime import (
    RuntimeTerminologyBinding,
    TerminologyRuntimeResolution,
)


def _requirement() -> AnswerRequirementV2:
    return AnswerRequirementV2(
        id="r1",
        description="普通员工餐补额度是多少",
        coverage_contract="single_claim",
        depends_on_requirement_ids=(),
        augmentation_requirement_ids=(),
    )


def _resolution(*, source_mode: str) -> TerminologyRuntimeResolution:
    forms = (
        TerminologyForm(
            term="餐补",
            rule_id="meal_short",
            relation_strength=source_mode,
        ),
        TerminologyForm(
            term="餐饮补贴",
            rule_id="meal_full",
            relation_strength="strict_equivalent",
        ),
    )
    evidence_forms = (
        TerminologyForm(
            term="餐补",
            rule_id="meal_short",
            relation_strength="strict_equivalent",
        ),
        TerminologyForm(
            term="餐饮补贴",
            rule_id="meal_full",
            relation_strength="strict_equivalent",
        ),
    ) if source_mode == "strict_equivalent" else (
        forms[1],
    )
    binding = TerminologyBinding(
        requirement_id="r1",
        concept_id="meal_allowance",
        concept_key="meal_allowance",
        display_name="餐饮补贴",
        source_term="餐补",
        source_relation_strength=source_mode,
        query_forms=forms,
        evidence_forms=evidence_forms,
        scope_binding_ids=("binding_meal",),
    )
    return TerminologyRuntimeResolution(
        plan_fingerprint="a" * 64,
        scope_fingerprint="b" * 64,
        registry_revisions={"kb_a": 2},
        status="resolved",
        bindings=(RuntimeTerminologyBinding(
            binding=binding,
            kb_id="kb_a",
            document_id="doc_policy",
        ),),
        authorized_kb_ids=("kb_a",),
    )


def _candidate() -> dict:
    return {
        "id": "chunk_1",
        "kb_id": "kb_a",
        "doc_id": "doc_policy",
        "chunk_index": 0,
        "authorized": True,
        "content": "普通员工餐饮补贴额度为100元/天。",
        "metadata": {},
    }


def _assemble_finalized_v2_evidence(
    *,
    requirement: AnswerRequirementV2,
    terminology_resolution: TerminologyRuntimeResolution,
):
    """Exercise terminology proof through the production V2 final boundary.

    ``assemble_evidence_bundle`` intentionally returns a provisional candidate
    package: it has not yet seen the renderer-visible set and therefore cannot
    own a coverage graph.  A strict terminology form is still only retrieval
    and source support until the request-local task ledger has bound the
    candidate and the finalizer closes the exact visible route.  Tests must
    follow that handoff rather than asking the assembler to recreate legacy
    graph behaviour.
    """

    plan = QueryPlanV2(
        original_query=requirement.description,
        answer_shape="fact",
        retrieval_queries=(requirement.description,),
        requirements=(requirement,),
        confidence=0.95,
        source="local",
    )
    execution_bundle = compile_rag_execution_bundle(plan)
    if (
        not execution_bundle.uses_task_ledger
        or execution_bundle.task_graph is None
    ):
        raise AssertionError("terminology evidence fixture requires V2 ledger")
    task_graph = execution_bundle.task_graph
    ledger = TaskExecutionLedger(task_graph, run_id="terminology-evidence-test")
    answer_task = next(task for task in task_graph.tasks if task.role == "answer")
    execution_id = ledger.begin_execution(
        kind="terminology_test_answer_query",
        query=answer_task.query,
        task_ids=(answer_task.task_id,),
    )
    observed = ledger.observe_candidates(
        (_candidate(),),
        execution_id=execution_id,
    )
    ledger.finish_execution(
        execution_id,
        status="succeeded",
        candidate_count=len(observed),
    )
    provisional = assemble_evidence_bundle(
        query=requirement.description,
        candidates=observed,
        requirements=(requirement,),
        retrieval_queries=(requirement.description,),
        answer_shape="fact",
        task_graph=task_graph,
        task_ledger=ledger,
        terminology_resolution=terminology_resolution,
    )
    finalized = finalize_visible_evidence_bundle(
        provisional,
        requirements=(requirement,),
        task_graph=task_graph,
        task_ledger=ledger,
        terminology_resolution=terminology_resolution,
    )
    return provisional, finalized.bundle


class TerminologyEvidenceIntegrationTests(unittest.TestCase):
    def test_strict_equivalence_can_close_evidence_with_rule_provenance(self):
        requirement = _requirement()
        provisional, bundle = _assemble_finalized_v2_evidence(
            requirement=requirement,
            terminology_resolution=_resolution(
                source_mode="strict_equivalent"
            ),
        )

        self.assertIsNone(provisional.coverage_graph)
        item = bundle.items[0]
        self.assertIn("r1", item.supports_requirement_ids)
        self.assertEqual(
            item.metadata["claim_proof_kind"]["r1"],
            "terminology_strict",
        )
        self.assertIn(
            "binding_meal",
            item.metadata["strict_terminology_rule_ids"]["r1"],
        )
        self.assertIsNotNone(bundle.coverage_graph)
        self.assertEqual(bundle.answer_source_ids, ("chunk_1",))
        self.assertEqual(bundle.state.completeness, "complete")
        claims = bundle.coverage_graph.claims if bundle.coverage_graph else ()
        self.assertEqual(claims[0].proof_kind, "terminology_strict")
        self.assertIn("meal_full", claims[0].strict_terminology_rule_ids)

    def test_retrieval_only_never_promotes_an_alias_chunk_to_strict_evidence(self):
        requirement = _requirement()
        _provisional, bundle = _assemble_finalized_v2_evidence(
            requirement=requirement,
            terminology_resolution=_resolution(source_mode="retrieval_only"),
        )

        item = bundle.items[0]
        self.assertNotIn("r1", item.supports_requirement_ids)
        self.assertNotIn("claim_proof_kind", item.metadata)
        self.assertTrue(bundle.missing_requirement_ids)
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertNotEqual(bundle.state.completeness, "complete")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
