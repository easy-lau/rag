import json
import unittest
from unittest.mock import patch

from core.query_analysis_execution import (
    build_execution_baseline,
    get_query_analysis_execution_service,
)
from core.query_contextual_ellipsis import (
    derive_contextual_ellipsis_analysis,
    derive_contextual_ellipsis_source_selection,
)
from core.rag_v2.query_plan import plan_query_locally


_PREVIOUS = "普通员工的出差标准是什么"


def _context(previous: str = _PREVIOUS):
    return ({
        "candidate_key": "t1",
        "user_input": previous,
        # Deliberately include a tempting, incompatible assistant statement:
        # the deterministic resolver must never read it as a source/fact.
        "assistant_answer": "普通员工对应 D级，餐补 99999 元。",
    },)


def _baseline(question: str, *, context=_context()):
    plan = plan_query_locally(question)
    return build_execution_baseline(
        plan=plan,
        local_surface_plan=plan,
        contextual_plan=plan,
        question=question,
        standalone_query=question,
        route_context=context,
        deterministic_is_followup=True,
    )


class ContextualEllipsisDerivationTests(unittest.TestCase):
    def test_lodging_followup_uses_only_current_target_and_previous_user_entity(self):
        result = derive_contextual_ellipsis_analysis(
            current_question="那住宿呢",
            route_context=_context(),
        )

        self.assertTrue(result.derived)
        self.assertEqual(result.reason, "previous_turn_unique_entity_qualifier")
        self.assertIsNotNone(result.analysis)
        candidate = result.analysis.answer_candidates[0]
        self.assertEqual(candidate.target_source_ref.turn_key, "current")
        self.assertEqual(candidate.target_source_ref.span, "住宿")
        self.assertEqual(candidate.qualifier_source_refs[0].turn_key, "t1")
        self.assertEqual(candidate.qualifier_source_refs[0].span, "普通员工")
        self.assertNotIn(
            "D级",
            json.dumps(result.analysis.to_dict(), ensure_ascii=False),
        )

    def test_meal_allowance_followup_has_the_same_source_contract(self):
        result = derive_contextual_ellipsis_analysis(
            current_question="那么餐补呢？",
            route_context=_context(),
        )

        self.assertTrue(result.derived)
        candidate = result.analysis.answer_candidates[0]
        self.assertEqual(candidate.target_source_ref.span, "餐补")
        self.assertEqual(candidate.qualifier_source_refs[0].span, "普通员工")

    def test_without_previous_user_turn_fails_closed(self):
        result = derive_contextual_ellipsis_analysis(
            current_question="那住宿呢",
            route_context=(),
        )

        self.assertFalse(result.derived)
        self.assertEqual(result.reason, "previous_user_turn_unavailable")

    def test_multiple_previous_entities_do_not_choose_one_subject(self):
        result = derive_contextual_ellipsis_analysis(
            current_question="那住宿呢",
            route_context=_context("普通员工和高级经理的出差标准分别是什么"),
        )

        self.assertFalse(result.derived)
        self.assertEqual(
            result.reason,
            "previous_turn_entity_not_unique_or_not_inheritable",
        )

    def test_current_explicit_scope_blocks_stale_subject_inheritance(self):
        result = derive_contextual_ellipsis_analysis(
            current_question="那云枢8.6的住宿呢",
            route_context=_context(),
        )

        self.assertFalse(result.derived)
        self.assertEqual(result.reason, "current_turn_has_explicit_qualifier_or_scope")

    def test_current_explicit_entity_blocks_stale_subject_inheritance(self):
        result = derive_contextual_ellipsis_analysis(
            current_question="那普通员工的住宿呢",
            route_context=_context("高级经理的出差标准是什么"),
        )

        self.assertFalse(result.derived)
        self.assertEqual(result.reason, "current_turn_has_explicit_qualifier_or_scope")

    def test_generic_reference_never_becomes_an_answer_target(self):
        result = derive_contextual_ellipsis_analysis(
            current_question="那这个呢",
            route_context=_context(),
        )

        self.assertFalse(result.derived)
        self.assertEqual(result.reason, "current_turn_not_supported_contextual_ellipsis")

    def test_previous_explicit_scope_is_not_silently_inherited(self):
        selection = derive_contextual_ellipsis_source_selection(
            current_question="那住宿呢",
            route_context=_context("普通员工在云枢8.6中的餐饮补贴是多少"),
        )

        self.assertFalse(selection.selected)
        self.assertEqual(selection.reason, "previous_turn_has_explicit_scope")

    def test_previous_condition_or_residual_context_is_not_silently_inherited(self):
        cases = (
            ("普通员工在上海出差的餐饮补贴是多少", "previous_turn_has_non_inheritable_qualifier"),
            ("普通员工按实际用餐次数的餐补是多少", "previous_turn_has_non_inheritable_qualifier"),
            ("普通员工国内出差的餐饮补贴是多少", "previous_turn_has_non_inheritable_qualifier"),
        )

        for previous, expected_reason in cases:
            with self.subTest(previous=previous):
                selection = derive_contextual_ellipsis_source_selection(
                    current_question="那住宿呢",
                    route_context=_context(previous),
                )
                self.assertFalse(selection.selected)
                self.assertEqual(selection.reason, expected_reason)


class ContextualEllipsisExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_deterministic_followup_compiles_and_emits_source_span_trace(self):
        traces = []
        with patch(
            "core.query_analysis_execution.trace_event",
            side_effect=lambda event, **payload: traces.append((event, payload)),
        ), patch(
            "core.query_analysis_execution.analyze_query",
        ) as analyze_query:
            result = await get_query_analysis_execution_service().run_deterministic_contextual_ellipsis(
                baseline=_baseline("那住宿呢"),
                trace_id="contextual-ellipsis",
                conversation_id="conversation-1",
                user_id="user-1",
            )

        self.assertTrue(result.applied)
        self.assertEqual(result.reason, "deterministic_contextual_ellipsis_applied")
        self.assertEqual(result.semantics.canonical_retrieval_query, "普通员工 住宿")
        answer = next(
            item
            for item in result.execution_bundle.plan.requirements
            if item.role == "answer"
        )
        self.assertEqual(answer.description, "普通员工 住宿")
        decision_trace = next(
            payload
            for event, payload in traces
            if event == "query.analysis.deterministic_contextual_ellipsis"
        )
        self.assertTrue(decision_trace["derived"])
        self.assertEqual(decision_trace["reason"], "previous_turn_unique_entity_qualifier")
        self.assertEqual(decision_trace["current_target_range"], [1, 3])
        self.assertEqual(decision_trace["historical_qualifier_range"], [0, 4])
        analyze_query.assert_not_awaited()

    async def test_non_match_retains_the_baseline_without_calling_a_model(self):
        baseline = _baseline("那这个呢")
        result = await get_query_analysis_execution_service().run_deterministic_contextual_ellipsis(
            baseline=baseline,
            trace_id="contextual-ellipsis-no-match",
            conversation_id="conversation-1",
            user_id="user-1",
        )

        self.assertEqual(result.decision, "skipped")
        self.assertEqual(result.reason, "current_turn_not_supported_contextual_ellipsis")
        self.assertIs(result.execution_bundle, baseline.execution_bundle)
        self.assertIsNone(result.semantics)


if __name__ == "__main__":
    unittest.main()
