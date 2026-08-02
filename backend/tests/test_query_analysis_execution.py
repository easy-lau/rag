import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.query_analysis_contract import QUERY_ANALYSIS_SCHEMA_VERSION, parse_query_analysis
from core.query_analysis_execution import (
    QUERY_EXECUTION_SCHEMA_VERSION,
    QUERY_EXECUTION_UNRESOLVED_ROLE,
    ShadowAnalysisDispatcher,
    _scope_fingerprint,
    build_execution_baseline,
    evaluate_query_execution_gate,
    get_query_analysis_execution_service,
)
from core.query_analyzer import QueryAnalysisRunResult
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.query_plan import plan_query_locally


QUESTION = "普通员工的住宿标准、餐补和出差补贴这些分别是多少"


def _ref(source, span):
    start = source.index(span)
    return {
        "turn_key": "current",
        "start": start,
        "end": start + len(span),
        "span": span,
    }


def _analysis(*, targets=("住宿标准", "餐补", "出差补贴")):
    subject = _ref(QUESTION, "普通员工")
    payload = {
        "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
        "relation": "new",
        "self_contained": True,
        "context_turn_keys": [],
        "answer_candidates": [
            {
                "id": f"a{index}",
                "target_source_ref": _ref(QUESTION, target),
                "qualifier_source_refs": [subject],
                "bridge_candidate_ids": ["b1"],
            }
            for index, target in enumerate(targets, start=1)
        ],
        "bridge_candidates": [{
            "id": "b1",
            "subject_source_ref": subject,
        }],
        "confidence": 0.95,
        "diagnostic": "三个并列目标共享一个人员限定词。",
    }
    return parse_query_analysis(
        json.dumps(payload, ensure_ascii=False),
        current_question=QUESTION,
    )


def _accepted_result(analysis):
    return QueryAnalysisRunResult(
        mode="active",
        analysis=analysis,
        model="test-model",
        latency_ms=12,
    )


def _generic_plan():
    return QueryPlanV2(
        original_query=QUESTION,
        answer_shape="fact",
        retrieval_queries=(QUESTION,),
        requirements=(AnswerRequirementV2(
            id="r1",
            description=QUESTION,
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        ),),
        confidence=0.75,
        source="fallback",
        reason="route_authorized_single_fact_baseline",
    )


class QueryExecutionGateTests(unittest.TestCase):
    def _baseline(self, plan):
        return build_execution_baseline(
            plan=plan,
            local_surface_plan=plan,
            contextual_plan=plan,
            question=plan.original_query,
            standalone_query=plan.original_query,
            route_context=(),
            deterministic_is_followup=False,
        )

    def test_ledgered_baseline_authorizes_execution(self):
        baseline = self._baseline(_generic_plan())
        gate = evaluate_query_execution_gate(baseline)

        self.assertTrue(baseline.is_runnable)
        self.assertTrue(gate.dispatch_authorized)
        self.assertEqual(
            gate.to_dict(),
            {
                "schema_version": QUERY_EXECUTION_SCHEMA_VERSION,
                "state": "ready",
                "dispatch_authorized": True,
                "decision_reason": "execution_baseline_runnable",
                "unresolved": [],
            },
        )

    def test_planner_clarification_maps_to_missing_execution_input(self):
        plan = plan_query_locally("该值取决于前一项")
        baseline = self._baseline(plan)
        gate = evaluate_query_execution_gate(baseline)

        self.assertTrue(plan.needs_clarification)
        self.assertEqual(gate.to_dict()["unresolved"], [{
            "role": QUERY_EXECUTION_UNRESOLVED_ROLE,
            "reason": "missing",
        }])


class QueryAnalysisExecutionServiceTests(unittest.IsolatedAsyncioTestCase):
    def _baseline(self):
        plan = _generic_plan()
        return build_execution_baseline(
            plan=plan,
            local_surface_plan=plan,
            contextual_plan=plan,
            question=QUESTION,
            standalone_query=QUESTION,
            route_context=(),
            deterministic_is_followup=False,
        )

    async def test_active_compiles_complete_candidates_and_keeps_original_anchor(self):
        baseline = self._baseline()
        traces = []
        with (
            patch(
                "core.query_analysis_execution.analyze_query",
                new=AsyncMock(return_value=_accepted_result(_analysis())),
            ),
            patch(
                "core.query_analysis_execution.trace_event",
                side_effect=lambda event, **payload: traces.append((event, payload)),
            ),
        ):
            result = await get_query_analysis_execution_service().run_active(
                baseline=baseline,
                trace_id="active-success",
                conversation_id="conversation-1",
                user_id="user-1",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        self.assertTrue(result.applied)
        self.assertEqual(result.reason, "generic_baseline_replaced")
        self.assertIsNotNone(result.semantics)
        self.assertEqual(result.semantics.request_kind, "finite_enumeration")
        self.assertEqual(result.execution_bundle.plan.answer_shape, "multi_part")
        self.assertEqual(
            result.execution_bundle.task_graph.task_by_id["anchor_root"].query,
            QUESTION,
        )
        compiled_trace = next(
            payload for event, payload in traces
            if event == "query.analysis.compiled"
        )
        self.assertEqual(
            compiled_trace["baseline_plan_fingerprint"],
            result.compilation.baseline_fingerprint,
        )
        self.assertEqual(
            compiled_trace["applied_plan_fingerprint"],
            result.compilation.applied_plan_fingerprint,
        )
        self.assertTrue(compiled_trace["baseline_anchor_preserved"])
        self.assertIn("resolved_turn_semantics", compiled_trace)

    async def test_timeout_or_incomplete_candidates_retain_exact_baseline_bundle(self):
        baseline = self._baseline()
        unavailable = QueryAnalysisRunResult(
            mode="active",
            analysis=None,
            model="test-model",
            latency_ms=15,
            fallback_reason="timeout",
        )
        with patch(
            "core.query_analysis_execution.analyze_query",
            new=AsyncMock(return_value=unavailable),
        ):
            timeout_result = await get_query_analysis_execution_service().run_active(
                baseline=baseline,
                trace_id="active-timeout",
                conversation_id="conversation-1",
                user_id="user-1",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )
        self.assertEqual(timeout_result.decision, "fallback")
        self.assertIs(timeout_result.execution_bundle, baseline.execution_bundle)

        with patch(
            "core.query_analysis_execution.analyze_query",
            new=AsyncMock(return_value=_accepted_result(_analysis(targets=("住宿标准", "餐补")))),
        ):
            rejected_result = await get_query_analysis_execution_service().run_active(
                baseline=baseline,
                trace_id="active-rejected",
                conversation_id="conversation-1",
                user_id="user-1",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )
        self.assertEqual(rejected_result.decision, "fallback")
        self.assertEqual(rejected_result.reason, "execution_validation_rejected")
        self.assertIs(rejected_result.execution_bundle, baseline.execution_bundle)
        self.assertEqual(
            rejected_result.validation.reason,
            "candidate_current_turn_coverage_incomplete",
        )

    async def test_explicit_proof_baseline_is_preserved_not_converted_to_augmentation(self):
        question = "普通员工对应什么职级"
        plan = plan_query_locally(question)
        baseline = build_execution_baseline(
            plan=plan,
            local_surface_plan=plan,
            contextual_plan=plan,
            question=question,
            standalone_query=question,
            route_context=(),
            deterministic_is_followup=False,
        )
        target = _ref(question, "职级")
        payload = {
            "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
            "relation": "new",
            "self_contained": True,
            "context_turn_keys": [],
            "answer_candidates": [{
                "id": "a1",
                "target_source_ref": target,
                "qualifier_source_refs": [_ref(question, "普通员工")],
                "bridge_candidate_ids": [],
            }],
            "bridge_candidates": [],
            "confidence": 0.95,
            "diagnostic": "一个明确关系目标。",
        }
        analysis = parse_query_analysis(json.dumps(payload, ensure_ascii=False), current_question=question)
        with patch(
            "core.query_analysis_execution.analyze_query",
            new=AsyncMock(return_value=_accepted_result(analysis)),
        ):
            result = await get_query_analysis_execution_service().run_active(
                baseline=baseline,
                trace_id="proof-preserved",
                conversation_id="conversation-1",
                user_id="user-1",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )
        answer = next(item for item in result.execution_bundle.plan.requirements if item.role == "answer")
        self.assertEqual(answer.depends_on_requirement_ids, ("r2",))
        self.assertEqual(answer.augmentation_requirement_ids, ())
        self.assertIs(result.execution_bundle, baseline.execution_bundle)

    async def test_nonrunnable_baseline_never_calls_model(self):
        plan = plan_query_locally("该值取决于前一项")
        baseline = build_execution_baseline(
            plan=plan,
            local_surface_plan=plan,
            contextual_plan=plan,
            question=plan.original_query,
            standalone_query=plan.original_query,
            route_context=(),
            deterministic_is_followup=False,
        )
        analyze = AsyncMock()
        with patch("core.query_analysis_execution.analyze_query", new=analyze):
            result = await get_query_analysis_execution_service().run_active(
                baseline=baseline,
                trace_id="nonrunnable",
                conversation_id="conversation-1",
                user_id="user-1",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )
        self.assertEqual(result.decision, "clarification")
        analyze.assert_not_awaited()

    async def test_shadow_observes_compilation_without_replacing_request_bundle(self):
        baseline = self._baseline()
        with patch(
            "core.query_analysis_execution.analyze_query",
            new=AsyncMock(return_value=_accepted_result(_analysis())),
        ):
            result = await get_query_analysis_execution_service().run_shadow(
                baseline=baseline,
                trace_id="shadow-success",
                conversation_id="conversation-1",
                user_id="user-1",
                timeout_seconds=0.5,
            )
        self.assertEqual(result.decision, "observed")
        self.assertIsNone(result.execution_bundle)
        self.assertEqual(result.compilation.compiler_decision, "generic_baseline_replaced")


class ScopeFingerprintTests(unittest.TestCase):
    def test_augmentation_edges_participate_in_scope_fingerprint(self):
        plain = QueryPlanV2(
            original_query="普通员工的餐补是多少",
            answer_shape="fact",
            retrieval_queries=("普通员工的餐补是多少",),
            requirements=(AnswerRequirementV2(
                id="r1",
                description="普通员工的餐补是多少",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
            ),),
            confidence=0.9,
            source="local",
            reason="test",
        )
        bridge = AnswerRequirementV2(
            id="r2",
            description="确认普通员工对应的适用分类。",
            role="bridge",
            importance="helpful",
            source="inferred",
            bridge_subject="普通员工",
            bridge_kind="classification",
        )
        augmented = QueryPlanV2(
            original_query=plain.original_query,
            answer_shape=plain.answer_shape,
            retrieval_queries=plain.retrieval_queries,
            requirements=(
                replace_requirement := AnswerRequirementV2(
                    id="r1",
                    description="普通员工的餐补是多少",
                    depends_on_requirement_ids=(),
                    augmentation_requirement_ids=("r2",),
                ),
                bridge,
            ),
            confidence=plain.confidence,
            source=plain.source,
            reason=plain.reason,
        )
        self.assertNotEqual(_scope_fingerprint(plain), _scope_fingerprint(augmented))
        self.assertEqual(replace_requirement.augmentation_requirement_ids, ("r2",))


class ShadowAnalysisDispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_try_submit_has_no_waiting_queue_and_releases_slot(self):
        dispatcher = ShadowAnalysisDispatcher()
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_job():
            started.set()
            await release.wait()

        self.assertTrue(dispatcher.try_submit(blocked_job, maximum=1, name="first-shadow"))
        await started.wait()
        self.assertFalse(dispatcher.try_submit(lambda: asyncio.sleep(0), maximum=1, name="second-shadow"))
        release.set()
        for _ in range(10):
            if dispatcher.inflight == 0:
                break
            await asyncio.sleep(0)
        self.assertEqual(dispatcher.inflight, 0)


if __name__ == "__main__":
    unittest.main()
