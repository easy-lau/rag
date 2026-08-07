import json
import unittest

from core.query_analysis_compiler import compile_query_analysis_plan
from core.query_analysis_contract import QUERY_ANALYSIS_SCHEMA_VERSION, parse_query_analysis
from core.query_analysis_validation import validate_query_analysis_for_execution
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.query_plan import plan_query_locally
from core.rag_v2.task_graph import compile_rag_execution_bundle


QUESTION = "普通员工的住宿标准、餐补和出差补贴这些分别是多少"


def _ref(source, span, *, occurrence=0):
    start = -1
    offset = 0
    for _ in range(occurrence + 1):
        start = source.index(span, offset)
        offset = start + len(span)
    return {
        "turn_key": "current",
        "start": start,
        "end": start + len(span),
        "span": span,
    }


def _analysis(question=QUESTION, *, targets=("住宿标准", "餐补", "出差补贴"), subject="普通员工", include_bridge=True):
    subject_ref = _ref(question, subject)
    bridge_candidates = [{
        "id": "b1",
        "subject_source_ref": subject_ref,
    }] if include_bridge else []
    payload = {
        "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
        "relation": "new",
        "self_contained": True,
        "context_turn_keys": [],
        "answer_candidates": [
            {
                "id": f"a{index}",
                "target_source_ref": _ref(question, target),
                "qualifier_source_refs": [subject_ref],
                "bridge_candidate_ids": ["b1"] if include_bridge else [],
            }
            for index, target in enumerate(targets, start=1)
        ],
        "bridge_candidates": bridge_candidates,
        "confidence": 0.95,
        "diagnostic": "原句中有多个并列目标和一个共享限定词。",
    }
    return parse_query_analysis(
        json.dumps(payload, ensure_ascii=False),
        current_question=question,
    )


def _generic_baseline(question):
    return QueryPlanV2(
        original_query=question,
        answer_shape="fact",
        retrieval_queries=(question,),
        requirements=(AnswerRequirementV2(
            id="r1",
            description=question,
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        ),),
        confidence=0.75,
        source="fallback",
        reason="route_authorized_single_fact_baseline",
    )


class QueryAnalysisV2CompilerTests(unittest.TestCase):
    def test_complete_source_frame_replaces_only_a_generic_baseline_and_keeps_anchor(self):
        baseline = _generic_baseline(QUESTION)
        analysis = _analysis()
        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline,
            current_question=QUESTION,
            deterministic_is_followup=False,
        )

        self.assertTrue(validation.accepted)
        self.assertTrue(validation.replacement_authorized)
        compiled = compile_query_analysis_plan(
            analysis,
            execution_validation=validation,
            baseline_plan=baseline,
            current_question=QUESTION,
            baseline_execution_bundle=compile_rag_execution_bundle(baseline),
        )

        self.assertEqual(compiled.compiler_decision, "generic_baseline_replaced")
        self.assertEqual(compiled.plan.answer_shape, "multi_part")
        self.assertNotEqual(compiled.applied_plan_fingerprint, compiled.baseline_fingerprint)
        answers = [item for item in compiled.plan.requirements if item.role == "answer"]
        bridges = [item for item in compiled.plan.requirements if item.role == "bridge"]
        self.assertEqual(len(answers), 3)
        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0].bridge_kind, "classification")
        self.assertTrue(all(
            answer.augmentation_requirement_ids == (bridges[0].id,)
            for answer in answers
        ))
        self.assertTrue(all(answer.depends_on_requirement_ids == () for answer in answers))
        self.assertEqual(
            compiled.task_graph.task_by_id["anchor_root"].query,
            QUESTION,
        )
        self.assertTrue(compiled.baseline_anchor_preserved)

    def test_partial_candidate_set_is_rejected_instead_of_dropping_a_sibling(self):
        baseline = _generic_baseline(QUESTION)
        analysis = _analysis(targets=("住宿标准", "餐补"))

        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline,
            current_question=QUESTION,
            deterministic_is_followup=False,
        )

        self.assertFalse(validation.accepted)
        self.assertEqual(validation.reason, "candidate_current_turn_coverage_incomplete")
        with self.assertRaisesRegex(ValueError, "accepted"):
            compile_query_analysis_plan(
                analysis,
                execution_validation=validation,
                baseline_plan=baseline,
                current_question=QUESTION,
            )

    def test_explicit_multi_answer_baseline_is_an_unreducible_floor(self):
        baseline = plan_query_locally(QUESTION)
        self.assertEqual(baseline.answer_shape, "multi_part")
        analysis = _analysis(targets=("住宿标准", "餐补"))

        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline,
            current_question=QUESTION,
            deterministic_is_followup=False,
        )

        self.assertFalse(validation.accepted)
        self.assertEqual(validation.reason, "candidate_current_turn_coverage_incomplete")

    def test_explicit_scope_is_copied_from_the_trusted_baseline_not_model_fields(self):
        question = "我使用的是产品甲8.6，普通员工的住宿标准和餐补分别是多少"
        baseline = _generic_baseline(question)
        analysis = _analysis(
            question,
            targets=("住宿标准", "餐补"),
        )
        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline,
            current_question=question,
            deterministic_is_followup=False,
        )
        self.assertTrue(validation.accepted)
        compiled = compile_query_analysis_plan(
            analysis,
            execution_validation=validation,
            baseline_plan=baseline,
            current_question=question,
        )

        for requirement in compiled.plan.requirements:
            self.assertEqual(
                (requirement.scope_product, requirement.scope_version),
                ("产品甲", "8.6"),
            )
            self.assertTrue(requirement.scope_explicit_version)
        self.assertEqual(compiled.task_graph.task_by_id["anchor_root"].query, question)

    def test_named_entity_candidate_cannot_create_a_classification_bridge(self):
        question = "供应商甲的风险处置措施是什么"
        baseline = _generic_baseline(question)
        analysis = _analysis(
            question,
            targets=("风险处置措施",),
            subject="供应商甲",
            include_bridge=False,
        )
        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline,
            current_question=question,
            deterministic_is_followup=False,
        )
        self.assertTrue(validation.accepted)
        self.assertFalse(validation.replacement_authorized)
        compiled = compile_query_analysis_plan(
            analysis,
            execution_validation=validation,
            baseline_plan=baseline,
            current_question=question,
        )
        self.assertEqual(compiled.compiler_decision, "baseline_preserved")
        self.assertFalse(any(item.role == "bridge" for item in compiled.plan.requirements))

    def test_explicit_mapping_proof_is_never_retyped_or_replaced_by_v2(self):
        question = "普通员工对应什么职级"
        baseline = plan_query_locally(question)
        self.assertEqual(baseline.answer_shape, "multi_hop")
        analysis = _analysis(
            question,
            targets=("职级",),
            include_bridge=False,
        )
        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline,
            current_question=question,
            deterministic_is_followup=False,
        )
        self.assertTrue(validation.accepted)
        compiled = compile_query_analysis_plan(
            analysis,
            execution_validation=validation,
            baseline_plan=baseline,
            current_question=question,
        )
        answer = next(item for item in compiled.plan.requirements if item.role == "answer")
        self.assertEqual(answer.depends_on_requirement_ids, ("r2",))
        self.assertEqual(answer.augmentation_requirement_ids, ())
        self.assertEqual(compiled.compiler_decision, "baseline_preserved")

    def test_current_clause_qualifier_normalizes_only_its_explicit_baseline_owner(self):
        """A second local clause may cite the first clause without new facts.

        ``报销`` is an exact current-turn span.  It closes the otherwise
        elliptical ``需要提供哪些凭证`` task, while the first explicit task and
        all its baseline evidence contracts remain unchanged.
        """

        question = "报销提交时限是多久？需要提供哪些凭证？"
        report = _ref(question, "报销")
        payload = {
            "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
            "relation": "new",
            "self_contained": True,
            "context_turn_keys": [],
            "answer_candidates": [
                {
                    "id": "a1",
                    "target_source_ref": _ref(question, "提交时限"),
                    "qualifier_source_refs": [report],
                    "bridge_candidate_ids": [],
                },
                {
                    "id": "a2",
                    "target_source_ref": _ref(question, "凭证"),
                    "qualifier_source_refs": [report],
                    "bridge_candidate_ids": [],
                },
            ],
            "bridge_candidates": [],
            "confidence": 0.96,
            "diagnostic": "第二个当前子句继承前一子句的字面业务对象。",
        }
        analysis = parse_query_analysis(
            json.dumps(payload, ensure_ascii=False),
            current_question=question,
        )
        baseline = plan_query_locally(question)
        baseline_by_id = {item.id: item for item in baseline.requirements}
        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline,
            current_question=question,
            deterministic_is_followup=False,
        )

        self.assertTrue(validation.accepted)
        self.assertFalse(validation.replacement_authorized)
        self.assertTrue(validation.canonicalization_authorized)
        compiled = compile_query_analysis_plan(
            analysis,
            execution_validation=validation,
            baseline_plan=baseline,
            current_question=question,
        )

        self.assertEqual(compiled.compiler_decision, "baseline_canonicalized")
        answers = {
            item.id: item
            for item in compiled.plan.requirements
            if item.role == "answer"
        }
        self.assertEqual(answers["r1"].description, "报销提交时限是多久")
        self.assertEqual(answers["r2"].description, "报销凭证")
        self.assertEqual(
            answers["r2"].coverage_mode,
            baseline_by_id["r2"].coverage_mode,
        )
        self.assertEqual(
            answers["r2"].coverage_contract,
            baseline_by_id["r2"].coverage_contract,
        )
        self.assertEqual(
            answers["r2"].depends_on_requirement_ids,
            baseline_by_id["r2"].depends_on_requirement_ids,
        )
        self.assertEqual(
            compiled.task_graph.task_by_id["anchor_root"].query,
            question,
        )
        self.assertIn("报销凭证", compiled.plan.retrieval_queries)


if __name__ == "__main__":
    unittest.main()
