"""Execution-boundary regression tests for controlled terminology.

The terminology registry is optional recall infrastructure, so these tests
exercise the point where its scoped aliases enter the physical retrieval
pipeline.  They intentionally use the real task graph and fetch boundary
rather than only asserting pure-contract output.
"""

from __future__ import annotations

import time
import unittest
import uuid
from unittest.mock import AsyncMock, Mock, patch

from core.query_constraints import QueryConstraints
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.pipeline import (
    MAX_INITIAL_TASK_EXECUTIONS,
    _fetch_task_group,
    _retrieve_task_graph_initial_candidates,
)
from core.rag_v2.task_execution import (
    PhysicalRetrievalGroup,
    TaskExecutionLedger,
)
from core.rag_v2.task_graph import compile_retrieval_task_graph
from core.terminology_contracts import TerminologyBinding, TerminologyForm
from core.terminology_runtime import (
    RuntimeTerminologyBinding,
    TerminologyRuntimeResolution,
)


def _requirement(requirement_id: str, description: str) -> AnswerRequirementV2:
    return AnswerRequirementV2(
        id=requirement_id,
        description=description,
        coverage_contract="single_claim",
        depends_on_requirement_ids=(),
        augmentation_requirement_ids=(),
    )


def _plan(requirements: tuple[AnswerRequirementV2, ...]) -> QueryPlanV2:
    return QueryPlanV2(
        original_query="查询制度标准",
        answer_shape="multi_part" if len(requirements) > 1 else "fact",
        retrieval_queries=("查询制度标准",),
        requirements=requirements,
        confidence=0.9,
        source="local",
    )


def _resolution(*, kb_id: uuid.UUID, document_id: uuid.UUID | None = None):
    binding = TerminologyBinding(
        requirement_id="r1",
        concept_id="meal_allowance",
        concept_key="meal_allowance",
        display_name="餐饮补贴",
        source_term="餐补",
        source_relation_strength="strict_equivalent",
        query_forms=(
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
        ),
        evidence_forms=(
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
        ),
        scope_binding_ids=("binding_meal",),
    )
    return TerminologyRuntimeResolution(
        plan_fingerprint="a" * 64,
        scope_fingerprint="b" * 64,
        registry_revisions={str(kb_id): 1},
        status="resolved",
        bindings=(RuntimeTerminologyBinding(
            binding=binding,
            kb_id=str(kb_id),
            document_id=str(document_id) if document_id else None,
        ),),
        authorized_kb_ids=(str(kb_id),),
    )


def _candidate(*, kb_id: uuid.UUID, doc_id: uuid.UUID, chunk_id: str) -> dict:
    return {
        "id": chunk_id,
        "kb_id": kb_id,
        "doc_id": doc_id,
        "chunk_index": 0,
        "authorized": True,
        "content": "普通员工餐饮补贴为100元/天。",
        "metadata": {},
    }


class TerminologyPipelineScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_alias_fetch_intersects_request_kb_and_document_scope_before_and_after_io(self):
        kb_allowed = uuid.uuid4()
        kb_other = uuid.uuid4()
        document_allowed = uuid.uuid4()
        document_other = uuid.uuid4()
        group = PhysicalRetrievalGroup(
            group_id="alias_1",
            query="普通员工餐饮补贴是多少",
            task_ids=("answer_r1",),
            scope_product=None,
            scope_version=None,
            scope_explicit_version=False,
            terminology_variant_origin="terminology_alias",
            terminology_rule_ids=("binding_meal", "meal_full"),
            retrieval_kb_ids=(str(kb_allowed),),
            retrieval_document_ids=(str(document_allowed),),
        )
        scoped_search = AsyncMock(return_value=[
            _candidate(
                kb_id=kb_allowed,
                doc_id=document_allowed,
                chunk_id="allowed",
            ),
            _candidate(
                kb_id=kb_allowed,
                doc_id=document_other,
                chunk_id="wrong-document",
            ),
            _candidate(
                kb_id=kb_other,
                doc_id=document_other,
                chunk_id="wrong-kb",
            ),
        ])

        with patch("core.rag_v2.pipeline.search_within_documents", new=scoped_search):
            result = await _fetch_task_group(
                db=object(),
                session_factory=None,
                group=group,
                kb_ids=[kb_allowed, kb_other],
                scope_filter=None,
                scoped_doc_uuid_ids=[document_allowed, document_other],
                method="hybrid",
                trace_id="terminology-scope-test",
                deadline=time.perf_counter() + 5,
                stage_timeout_seconds=2,
                candidate_k=6,
                surface="test",
            )

        self.assertIsNone(result.error)
        self.assertEqual(
            scoped_search.await_args.kwargs["kb_ids"],
            [kb_allowed],
        )
        self.assertEqual(
            scoped_search.await_args.kwargs["doc_ids"],
            [document_allowed],
        )
        self.assertEqual([item["id"] for item in result.raw_candidates], ["allowed"])

    async def test_alias_empty_document_intersection_skips_io_instead_of_broadening(self):
        kb_allowed = uuid.uuid4()
        registry_document = uuid.uuid4()
        user_selected_document = uuid.uuid4()
        group = PhysicalRetrievalGroup(
            group_id="alias_1",
            query="普通员工餐饮补贴是多少",
            task_ids=("answer_r1",),
            scope_product=None,
            scope_version=None,
            scope_explicit_version=False,
            terminology_variant_origin="terminology_alias",
            terminology_rule_ids=("binding_meal", "meal_full"),
            retrieval_kb_ids=(str(kb_allowed),),
            retrieval_document_ids=(str(registry_document),),
        )
        scoped_search = AsyncMock()

        with patch("core.rag_v2.pipeline.search_within_documents", new=scoped_search):
            result = await _fetch_task_group(
                db=object(),
                session_factory=None,
                group=group,
                kb_ids=[kb_allowed],
                scope_filter=None,
                scoped_doc_uuid_ids=[user_selected_document],
                method="hybrid",
                trace_id="terminology-empty-scope-test",
                deadline=time.perf_counter() + 5,
                stage_timeout_seconds=2,
                candidate_k=6,
                surface="test",
            )

        scoped_search.assert_not_awaited()
        self.assertEqual(result.raw_candidates, ())
        self.assertTrue(result.diagnostics["scope_intersection_empty"])


class TerminologyPipelineBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_alias_budget_cannot_mark_executed_literals_skipped(self):
        """Budget accounting follows selected physical groups, not list order."""

        kb_id = uuid.uuid4()
        requirements = tuple(
            _requirement(f"r{index}", f"第{index}项餐补是多少")
            for index in range(1, MAX_INITIAL_TASK_EXECUTIONS + 1)
        )
        graph = compile_retrieval_task_graph(_plan(requirements))
        ledger = TaskExecutionLedger(graph, run_id="terminology-budget")
        search = AsyncMock(return_value=[])
        trace = Mock()

        with (
            patch("core.rag_v2.pipeline.hybrid_search", new=search),
            patch("core.rag_v2.pipeline.trace_event", new=trace),
        ):
            result = await _retrieve_task_graph_initial_candidates(
                db=object(),
                task_graph=graph,
                ledger=ledger,
                anchor_query="查询制度标准",
                kb_ids=[kb_id],
                scope_filter=None,
                scope_doc_ids=None,
                constraints=QueryConstraints(),
                method="hybrid",
                trace_id="terminology-budget-test",
                deadline=time.perf_counter() + 5,
                stage_timeout_seconds=2,
                candidate_k=6,
                task_read_session_factory=None,
                max_parallelism=1,
                terminology_resolution=_resolution(kb_id=kb_id),
                maximum_terminology_aliases=3,
            )

        # One anchor plus seven answer literals consume the eight-query budget.
        # The r1 alias and r8 literal are omitted, but all selected literal
        # task states remain cleanly successful.
        self.assertEqual(len(result.groups), MAX_INITIAL_TASK_EXECUTIONS)
        self.assertTrue(all(
            group.terminology_variant_origin == "original"
            for group in result.groups
        ))
        states = ledger.task_state_summary()
        for requirement_id in tuple(f"r{index}" for index in range(1, 8)):
            state = states[f"answer_{requirement_id}"]
            self.assertEqual(state["status"], "succeeded")
            self.assertEqual(state["budget_skipped"], 0)
        self.assertEqual(states["answer_r8"]["status"], "budget_skipped")
        self.assertEqual(states["answer_r8"]["budget_skipped"], 1)

        skipped_events = [
            call.kwargs
            for call in trace.call_args_list
            if call.args and call.args[0] == "retrieval.task_query_skipped"
        ]
        alias_skip = next(
            item for item in skipped_events
            if item["terminology_variant_origin"] == "terminology_alias"
        )
        self.assertFalse(alias_skip["task_budget_skip_recorded"])
        self.assertEqual(alias_skip["affected_task_ids"], [])

    async def test_degraded_registry_keeps_literal_baseline_retrieval(self):
        kb_id = uuid.uuid4()
        requirement = _requirement("r1", "普通员工餐补是多少")
        graph = compile_retrieval_task_graph(_plan((requirement,)))
        ledger = TaskExecutionLedger(graph, run_id="terminology-degraded")
        search = AsyncMock(return_value=[])
        degraded = TerminologyRuntimeResolution.degraded(
            plan_fingerprint="a" * 64,
            scope_fingerprint="b" * 64,
            authorized_kb_ids=(str(kb_id),),
        )

        with patch("core.rag_v2.pipeline.hybrid_search", new=search):
            result = await _retrieve_task_graph_initial_candidates(
                db=object(),
                task_graph=graph,
                ledger=ledger,
                anchor_query="查询制度标准",
                kb_ids=[kb_id],
                scope_filter=None,
                scope_doc_ids=None,
                constraints=QueryConstraints(),
                method="hybrid",
                trace_id="terminology-degraded-test",
                deadline=time.perf_counter() + 5,
                stage_timeout_seconds=2,
                candidate_k=6,
                task_read_session_factory=None,
                max_parallelism=1,
                terminology_resolution=degraded,
                maximum_terminology_aliases=3,
            )

        self.assertEqual(
            {group.terminology_variant_origin for group in result.groups},
            {"original"},
        )
        self.assertGreaterEqual(search.await_count, 2)
        self.assertIn(
            "普通员工餐补是多少",
            [call.args[1] for call in search.await_args_list],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
