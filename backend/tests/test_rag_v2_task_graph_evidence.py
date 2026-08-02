import unittest

from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.evidence import assemble_evidence_bundle
from core.rag_v2.task_execution import TaskExecutionLedger
from core.rag_v2.task_graph import compile_rag_execution_bundle


def _plan(requirements):
    return QueryPlanV2(
        original_query="普通员工的住宿和餐补分别是多少",
        answer_shape="multi_part",
        # Deliberately reversed/ambiguous legacy projection.
        retrieval_queries=("餐补", "住宿"),
        requirements=tuple(requirements),
        confidence=0.95,
        source="local",
    )


def _answer(requirement_id, description, *, dependencies=()):
    return AnswerRequirementV2(
        id=requirement_id,
        description=description,
        role="answer",
        importance="required",
        source="explicit",
        depends_on_requirement_ids=dependencies,
    )


def _candidate(chunk_id, content, *, task_ids=(), indexes=()):
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc-1",
        "kb_id": "kb-1",
        "content": content,
        "metadata": {
            "retrieval_task_ids": list(task_ids),
            "expansion_query_indexes": list(indexes),
        },
    }


def _ledgered_execution(requirements):
    """Create the only valid V2 test handoff: plan -> bundle -> ledger."""

    bundle = compile_rag_execution_bundle(_plan(requirements))
    if not bundle.uses_task_ledger or bundle.task_graph is None:
        raise AssertionError("fixture requires a ledgered execution bundle")
    return bundle.task_graph, TaskExecutionLedger(bundle.task_graph, run_id="current-run")


def _observe(ledger, *, task_ids, query, candidates):
    execution_id = ledger.begin_execution(
        kind="initial_task_query",
        query=query,
        task_ids=task_ids,
    )
    observed = ledger.observe_candidates(candidates, execution_id=execution_id)
    ledger.finish_execution(
        execution_id,
        status="succeeded",
        candidate_count=len(observed),
    )
    return observed


class TaskGraphEvidenceBindingTests(unittest.TestCase):
    def test_current_ledger_is_the_only_task_provenance_authority(self):
        requirements = (
            _answer("r1", "普通员工的住宿标准是多少"),
            _answer("r2", "普通员工的餐补是多少"),
        )
        graph, ledger = _ledgered_execution(requirements)
        candidates = _observe(
            ledger,
            query="普通员工的住宿标准是多少",
            task_ids=("answer_r1",),
            candidates=(
                {
                **_candidate(
                    "lodging",
                    "普通员工住宿标准为每晚450元。",
                    task_ids=("answer_r2",),
                ),
                "metadata": {
                    "retrieval_task_ids": ["answer_r2"],
                    "supports_requirement_ids": ["r2"],
                },
                },
            ),
        )

        bundle = assemble_evidence_bundle(
            query="普通员工住宿标准是多少",
            candidates=candidates,
            requirements=requirements,
            retrieval_queries=("餐补", "住宿"),
            task_graph=graph,
            task_ledger=ledger,
            answer_shape="multi_part",
        )

        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ("r1",))
        self.assertEqual(item.metadata["retrieval_task_ids"], ["answer_r1"])
        self.assertEqual(item.metadata["task_binding_status"], "bound")

    def test_unbound_but_visible_source_text_still_cannot_be_silently_dropped(self):
        requirements = (_answer("r1", "住宿标准是多少"),)
        graph, ledger = _ledgered_execution(requirements)

        bundle = assemble_evidence_bundle(
            query="住宿标准是多少",
            candidates=(_candidate("visible", "住宿标准为每晚450元。"),),
            requirements=requirements,
            task_graph=graph,
            task_ledger=ledger,
            answer_shape="fact",
        )

        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.items[0].metadata["task_binding_status"], "unbound_current_run")
        self.assertIn("unbound_current_task_provenance", bundle.state.reasons)

    def test_current_run_bindings_map_same_or_reordered_queries_without_position(self):
        requirements = (
            _answer("r1", "普通员工的住宿标准是多少"),
            _answer("r2", "普通员工的餐补是多少"),
        )
        graph, ledger = _ledgered_execution(requirements)
        lodging = _observe(
            ledger,
            task_ids=("answer_r1",),
            query="普通员工的住宿标准是多少",
            candidates=(
                _candidate(
                    "lodging",
                    "普通员工住宿标准为每晚450元。",
                ),
            ),
        )
        meal = _observe(
            ledger,
            task_ids=("answer_r2",),
            query="普通员工的餐补是多少",
            candidates=(
                _candidate(
                    "meal",
                    "普通员工餐补标准为每天100元。",
                ),
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的住宿和餐补分别是多少",
            requirements=requirements,
            retrieval_queries=("餐补", "住宿"),
            task_graph=graph,
            task_ledger=ledger,
            candidates=(*lodging, *meal),
            answer_shape="multi_part",
        )
        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["meal"].supports_requirement_ids, ("r2",))
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_bound_task_provenance_cannot_promote_an_irrelevant_chunk(self):
        requirements = (
            _answer("r1", "住宿标准是多少"),
            _answer("r2", "餐补是多少"),
        )
        graph, ledger = _ledgered_execution(requirements)
        candidates = _observe(
            ledger,
            task_ids=("answer_r1",),
            query="住宿标准是多少",
            candidates=(
                _candidate(
                    "legacy",
                    "出差结束后五个工作日内提交报销。",
                    indexes=(0,),
                ),
            ),
        )
        bundle = assemble_evidence_bundle(
            query="住宿和餐补分别是多少",
            requirements=requirements,
            retrieval_queries=("住宿标准是多少", "餐补是多少"),
            task_graph=graph,
            task_ledger=ledger,
            candidates=candidates,
            answer_shape="multi_part",
        )
        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ())
        self.assertEqual(item.metadata["task_binding_status"], "bound")
        self.assertNotIn("legacy_ambiguous_task_provenance", bundle.state.reasons)

    def test_anchor_provenance_never_substitutes_for_visible_claim(self):
        requirements = (_answer("r1", "住宿标准是多少"),)
        graph, ledger = _ledgered_execution(requirements)
        candidates = _observe(
            ledger,
            task_ids=("anchor_root",),
            query="住宿标准是多少",
            candidates=(
                _candidate("anchor-only", "公司出差管理标准总则。"),
            ),
        )
        bundle = assemble_evidence_bundle(
            query="住宿标准是多少",
            requirements=requirements,
            task_graph=graph,
            task_ledger=ledger,
            candidates=candidates,
            answer_shape="fact",
        )
        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))


if __name__ == "__main__":
    unittest.main()
