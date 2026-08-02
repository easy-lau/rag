"""Execution-boundary tests for ``query_understanding.v3``.

These tests deliberately use an unresolved local fallback.  A valid V3 span
selection must be able to compile a safe ledgered task graph; the old local
planner is a fallback floor, not a second semantic veto.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from core.query_analysis_execution import build_execution_baseline
from core.query_understanding_v3_analyzer import QueryUnderstandingV3RunResult
from core.query_understanding_v3_catalog import SourceSpanCatalog
from core.query_understanding_v3_contract import parse_query_understanding
from core.query_understanding_v3_execution import (
    QueryUnderstandingV3RevisionFence,
    build_query_understanding_v3_baseline,
    build_query_understanding_v3_fence_identity,
    get_query_understanding_v3_execution_service,
)
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2


def _fallback_plan(question: str, *, runnable: bool) -> QueryPlanV2:
    if runnable:
        return QueryPlanV2(
            original_query=question,
            answer_shape="fact",
            retrieval_queries=(question,),
            requirements=(
                AnswerRequirementV2(
                    id="r1",
                    description=question,
                    depends_on_requirement_ids=(),
                    augmentation_requirement_ids=(),
                ),
            ),
            confidence=0.5,
            source="fallback",
            reason="test_runnable_floor",
        )
    return QueryPlanV2(
        original_query=question,
        answer_shape="unknown",
        retrieval_queries=(),
        requirements=(),
        confidence=0.0,
        source="fallback",
        reason="test_unresolved_floor",
        needs_clarification=True,
        clarification_question="请补充必要限定条件。",
    )


def _execution_baseline(question: str, *, runnable: bool):
    plan = _fallback_plan(question, runnable=runnable)
    return build_execution_baseline(
        plan=plan,
        local_surface_plan=plan,
        contextual_plan=plan,
        question=question,
        standalone_query=question,
        route_context=(),
        deterministic_is_followup=False,
    )


def _contextual_execution_baseline(question: str):
    plan = _fallback_plan(question, runnable=True)
    return build_execution_baseline(
        plan=plan,
        local_surface_plan=plan,
        contextual_plan=plan,
        question=question,
        standalone_query=question,
        route_context=({
            "candidate_key": "t1",
            "user_input": "普通员工的餐饮补贴是多少？",
            # A V3 deterministic selection must never use answer text as its
            # source; this bait also proves no business fact enters planning.
            "assistant_answer": "普通员工对应 D级，餐补 99999 元。",
        },),
        deterministic_is_followup=True,
    )


def _history_execution_baseline(
    question: str,
    *,
    previous_user_input: str,
):
    plan = _fallback_plan(question, runnable=True)
    return build_execution_baseline(
        plan=plan,
        local_surface_plan=plan,
        contextual_plan=plan,
        question=question,
        standalone_query=question,
        route_context=({
            "candidate_key": "t1",
            "user_input": previous_user_input,
            "assistant_answer": "历史回答不能作为 V3 语义来源。",
        },),
        deterministic_is_followup=True,
    )


def _fence_identity(question: str):
    return build_query_understanding_v3_fence_identity(
        conversation_id="conversation-v3-fence",
        turn_id="turn-v3-fence",
        request_id="request-v3-fence",
        question=question,
        kb_ids=["00000000-0000-0000-0000-000000000001"],
        evidence_scope_filter=None,
        route_context=(),
        route_state_revision=0,
        task_contract={"schema_version": "test", "dispatch_authorized": True},
    )


def _span_id(catalog: SourceSpanCatalog, text: str) -> str:
    return next(
        item.span_id
        for item in catalog.current_entries
        if item.text == text
    )


def _travel_result(question: str) -> QueryUnderstandingV3RunResult:
    catalog = SourceSpanCatalog.build(current_question=question)
    employee = _span_id(catalog, "普通员工")
    payload = {
        "schema_version": "query_understanding.v3",
        "answer_candidates": [
            {
                "id": "a1",
                "target_span_id": _span_id(catalog, "住宿标准"),
                "qualifier_span_ids": [employee],
            },
            {
                "id": "a2",
                "target_span_id": _span_id(catalog, "餐补"),
                "qualifier_span_ids": [employee],
            },
            {
                "id": "a3",
                "target_span_id": _span_id(catalog, "出差补贴"),
                "qualifier_span_ids": [employee],
            },
        ],
    }
    analysis = parse_query_understanding(
        json.dumps(payload, ensure_ascii=False),
        catalog=catalog,
    )
    return QueryUnderstandingV3RunResult(
        mode="active",
        catalog=catalog,
        analysis=analysis,
        model="test-model",
        latency_ms=4,
    )


class QueryUnderstandingV3ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_followup_with_historical_scope_clarifies_before_model_or_retrieval(self) -> None:
        question = "那住宿呢"
        baseline = build_query_understanding_v3_baseline(
            fallback=_history_execution_baseline(
                question,
                previous_user_input="普通员工在云枢8.6中的餐饮补贴是多少",
            ),
        )
        model = AsyncMock(
            side_effect=AssertionError("unsafe contextual preflight must not call a model")
        )

        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=model,
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-scope-preflight",
                conversation_id="conversation-v3-scope-preflight",
                user_id="user-v3-scope-preflight",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        self.assertTrue(baseline.execution_preflight_blocked)
        self.assertEqual(result.decision, "clarification")
        self.assertEqual(result.reason, "contextual_ellipsis_history_not_inheritable")
        self.assertTrue(result.query_execution_gate.needs_clarification)
        self.assertTrue(result.selected_baseline.plan.needs_clarification)
        self.assertIn("适用范围", result.query_execution_gate.clarification_question)
        model.assert_not_awaited()

    async def test_model_history_scope_rejection_closes_runnable_current_baseline(self) -> None:
        """Non-strict model use of unsafe t1 cannot fall back to bare current text."""

        question = "餐补呢"
        baseline = build_query_understanding_v3_baseline(
            fallback=_history_execution_baseline(
                question,
                previous_user_input="普通员工在云枢8.6中的餐饮补贴是多少",
            ),
        )
        catalog = SourceSpanCatalog.build(
            current_question=question,
            route_context=baseline.fallback.route_context,
        )
        target = _span_id(catalog, "餐补")
        historical_entity = next(
            item.span_id
            for item in catalog.context_entries
            if item.source_key == "t1" and item.text == "普通员工"
        )
        analysis = parse_query_understanding(
            json.dumps({
                "schema_version": "query_understanding.v3",
                "answer_candidates": [{
                    "id": "a1",
                    "target_span_id": target,
                    "qualifier_span_ids": [historical_entity],
                }],
            }, ensure_ascii=False),
            catalog=catalog,
        )
        model_result = QueryUnderstandingV3RunResult(
            mode="active",
            catalog=catalog,
            analysis=analysis,
            model="test-model",
            latency_ms=1,
        )

        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=AsyncMock(return_value=model_result),
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-model-history-scope",
                conversation_id="conversation-v3-model-history-scope",
                user_id="user-v3-model-history-scope",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        self.assertFalse(baseline.execution_preflight_blocked)
        self.assertEqual(result.decision, "clarification")
        self.assertEqual(result.reason, "historical_context_requires_clarification")
        self.assertIsNotNone(result.validation)
        self.assertTrue(result.validation.requires_clarification)
        self.assertTrue(result.query_execution_gate.needs_clarification)
        self.assertTrue(result.selected_baseline.plan.needs_clarification)

    async def test_strict_contextual_ellipsis_survives_model_timeout_via_v3_compiler(self) -> None:
        question = "那住宿呢？"
        baseline = build_query_understanding_v3_baseline(
            fallback=_contextual_execution_baseline(question),
        )
        unavailable = QueryUnderstandingV3RunResult(
            mode="active",
            catalog=SourceSpanCatalog.build(current_question=question),
            analysis=None,
            model="test-model",
            latency_ms=500,
            fallback_reason="timeout",
        )
        model = AsyncMock(return_value=unavailable)

        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=model,
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-deterministic-ellipsis",
                conversation_id="conversation-v3-deterministic-ellipsis",
                user_id="user-v3-deterministic-ellipsis",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        self.assertTrue(result.applied)
        self.assertEqual(result.reason, "catalog_bound_candidate_compiled")
        self.assertEqual(result.producer, "deterministic_contextual")
        self.assertEqual(result.analysis_result.origin, "deterministic")
        self.assertEqual(result.context_selection.selected_context_turn_keys, ("t1",))
        self.assertEqual(result.selected_baseline.plan.original_query, question)
        self.assertEqual(result.selected_baseline.plan.retrieval_queries, (question,))
        self.assertEqual(
            result.execution_bundle.task_graph.task_by_id["anchor_root"].query,
            question,
        )
        answer = next(
            item
            for item in result.selected_baseline.plan.requirements
            if item.role == "answer"
        )
        self.assertEqual(answer.description, "普通员工 住宿")
        model.assert_not_awaited()

    async def test_strict_contextual_ellipsis_does_not_depend_on_model_capacity(self) -> None:
        question = "那住宿呢"
        baseline = build_query_understanding_v3_baseline(
            fallback=_contextual_execution_baseline(question),
        )
        model = AsyncMock(
            side_effect=AssertionError("strict local V3 path must not reserve model capacity")
        )

        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=model,
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-deterministic-capacity",
                conversation_id="conversation-v3-deterministic-capacity",
                user_id="user-v3-deterministic-capacity",
                timeout_seconds=0.5,
                maximum_inflight=0,
            )

        self.assertTrue(result.applied)
        self.assertEqual(result.producer, "deterministic_contextual")
        self.assertEqual(
            next(
                item.description
                for item in result.selected_baseline.plan.requirements
                if item.role == "answer"
            ),
            "普通员工 住宿",
        )
        model.assert_not_awaited()

    async def test_non_strict_followup_still_fails_closed_when_model_capacity_is_exhausted(self) -> None:
        question = "那这个呢"
        baseline = build_query_understanding_v3_baseline(
            fallback=_contextual_execution_baseline(question),
        )
        model = AsyncMock()

        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=model,
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-nonstrict-capacity",
                conversation_id="conversation-v3-nonstrict-capacity",
                user_id="user-v3-nonstrict-capacity",
                timeout_seconds=0.5,
                maximum_inflight=0,
            )

        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "active_capacity_exhausted")
        self.assertEqual(result.producer, "model")
        self.assertIs(result.selected_baseline, baseline.fallback)
        model.assert_not_awaited()

    async def test_valid_v3_candidate_replaces_unrunnable_local_floor(self) -> None:
        question = "普通员工的住宿标准、餐补和出差补贴分别是多少"
        baseline = build_query_understanding_v3_baseline(
            fallback=_execution_baseline(question, runnable=False),
        )
        result_from_model = _travel_result(question)

        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=AsyncMock(return_value=result_from_model),
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-apply",
                conversation_id="conversation-v3-apply",
                user_id="user-v3-apply",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        self.assertTrue(result.applied)
        self.assertTrue(result.query_execution_gate.dispatch_authorized)
        self.assertEqual(result.execution_bundle.mode, "ledgered")
        self.assertEqual(result.selected_baseline.plan.answer_shape, "multi_part")
        answers = [
            item
            for item in result.selected_baseline.plan.requirements
            if item.role == "answer"
        ]
        bridges = [
            item
            for item in result.selected_baseline.plan.requirements
            if item.role == "bridge"
        ]
        self.assertEqual(len(answers), 3)
        self.assertEqual(len(bridges), 1)
        self.assertTrue(all(item.augmentation_requirement_ids for item in answers))
        self.assertIsNotNone(result.context_selection)
        self.assertTrue(result.context_selection.self_contained)

    async def test_model_failure_keeps_ready_fallback_without_rewriting_it(self) -> None:
        question = "供应商甲的风险等级是什么"
        fallback = _execution_baseline(question, runnable=True)
        baseline = build_query_understanding_v3_baseline(fallback=fallback)
        unavailable = QueryUnderstandingV3RunResult(
            mode="active",
            catalog=SourceSpanCatalog.build(current_question=question),
            analysis=None,
            model="test-model",
            latency_ms=5,
            fallback_reason="timeout",
        )
        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=AsyncMock(return_value=unavailable),
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-fallback",
                conversation_id="conversation-v3-fallback",
                user_id="user-v3-fallback",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        self.assertFalse(result.applied)
        self.assertEqual(result.decision, "fallback")
        self.assertEqual(result.reason, "timeout")
        self.assertIs(result.selected_baseline, fallback)
        self.assertTrue(result.query_execution_gate.dispatch_authorized)

    async def test_hard_guard_cannot_be_bypassed_by_accepted_model_selection(self) -> None:
        question = "普通员工的住宿标准、餐补和出差补贴分别是多少"
        baseline = build_query_understanding_v3_baseline(
            fallback=_execution_baseline(question, runnable=False),
            hard_clarification_reason="route_scope_conflict",
        )
        result_from_model = _travel_result(question)
        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=AsyncMock(return_value=result_from_model),
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-hard-guard",
                conversation_id="conversation-v3-hard-guard",
                user_id="user-v3-hard-guard",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        self.assertEqual(result.decision, "clarification")
        self.assertFalse(result.query_execution_gate.dispatch_authorized)
        self.assertIsNotNone(result.validation)
        self.assertEqual(result.validation.reason, "hard_clarification_guard")

    async def test_revision_fence_rejects_late_adoption_after_retrieval_is_sealed(self) -> None:
        question = "供应商甲的风险等级是什么"
        fallback = _execution_baseline(question, runnable=True)
        baseline = build_query_understanding_v3_baseline(fallback=fallback)
        unavailable = QueryUnderstandingV3RunResult(
            mode="active",
            catalog=SourceSpanCatalog.build(current_question=question),
            analysis=None,
            model="test-model",
            latency_ms=1,
            fallback_reason="timeout",
        )
        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=AsyncMock(return_value=unavailable),
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-fence",
                conversation_id="conversation-v3-fence",
                user_id="user-v3-fence",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        fence = QueryUnderstandingV3RevisionFence(
            baseline_fingerprint=fallback.fingerprint,
            identity=_fence_identity(question),
        )
        self.assertTrue(fence.adopt(result, observed_identity=_fence_identity(question)))
        self.assertEqual(fence.revision, 1)
        self.assertEqual(fence.seal(), 1)
        self.assertFalse(fence.adopt(result, observed_identity=_fence_identity(question)))
        self.assertTrue(fence.sealed)
        self.assertEqual(fence.last_rejection_reason, "fence_sealed")

    async def test_revision_fence_rejects_changed_route_state_before_adoption(self) -> None:
        question = "供应商甲的风险等级是什么"
        fallback = _execution_baseline(question, runnable=True)
        baseline = build_query_understanding_v3_baseline(fallback=fallback)
        unavailable = QueryUnderstandingV3RunResult(
            mode="active",
            catalog=SourceSpanCatalog.build(current_question=question),
            analysis=None,
            model="test-model",
            latency_ms=1,
            fallback_reason="timeout",
        )
        with (
            patch(
                "core.query_understanding_v3_execution.analyze_query_understanding",
                new=AsyncMock(return_value=unavailable),
            ),
            patch("core.query_understanding_v3_execution.trace_event"),
        ):
            result = await get_query_understanding_v3_execution_service().run_active(
                baseline=baseline,
                trace_id="trace-v3-fence-state",
                conversation_id="conversation-v3-fence",
                user_id="user-v3-fence",
                timeout_seconds=0.5,
                maximum_inflight=1,
            )

        fence = QueryUnderstandingV3RevisionFence(
            baseline_fingerprint=fallback.fingerprint,
            identity=_fence_identity(question),
        )
        changed = build_query_understanding_v3_fence_identity(
            conversation_id="conversation-v3-fence",
            turn_id="turn-v3-fence",
            request_id="request-v3-fence",
            question=question,
            kb_ids=["00000000-0000-0000-0000-000000000001"],
            evidence_scope_filter=None,
            route_context=(),
            route_state_revision=1,
            task_contract={"schema_version": "test", "dispatch_authorized": True},
        )
        self.assertFalse(fence.adopt(result, observed_identity=changed))
        self.assertEqual(fence.last_rejection_reason, "request_identity_mismatch")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
