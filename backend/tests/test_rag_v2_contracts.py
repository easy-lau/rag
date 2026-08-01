import unittest

from core.rag_v2.contracts import (
    AnswerRequirementV2,
    EvidenceBundle,
    EvidenceItem,
    EvidenceState,
    QueryPlanV2,
)


def _item(
    chunk_id="c1",
    *,
    authorized=True,
    constraint_status="neutral",
):
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id="d1",
        kb_id="k1",
        content=f"evidence {chunk_id}",
        score=0.8,
        confidence="retrieved",
        constraint_status=constraint_status,
        authorized=authorized,
        origins=("hybrid",),
        metadata={"section_key": "s1"},
    )


class QueryPlanV2ContractTests(unittest.TestCase):
    def test_round_trip_and_narrow_fact_gate(self) -> None:
        plan = QueryPlanV2(
            original_query="Which value applies?",
            answer_shape="fact",
            retrieval_queries=("Which value applies?",),
            requirements=(
                AnswerRequirementV2(
                    id="r1",
                    description="Which value applies?",
                ),
            ),
            confidence=0.9,
            source="local",
            reason="explicit_scalar_lookup_signal",
        )

        self.assertTrue(plan.allows_narrow_fact_path)
        self.assertEqual(plan.to_dict()["answer_shape"], "fact")
        self.assertEqual(plan.to_dict()["requirements"][0]["id"], "r1")

    def test_fallback_fact_cannot_enable_narrow_path(self) -> None:
        plan = QueryPlanV2(
            original_query="ambiguous input",
            answer_shape="fact",
            retrieval_queries=("ambiguous input",),
            requirements=(
                AnswerRequirementV2(id="r1", description="ambiguous input"),
            ),
            confidence=1.0,
            source="fallback",
            reason="fallback",
        )

        self.assertFalse(plan.allows_narrow_fact_path)

    def test_rejects_ready_plan_without_requirements(self) -> None:
        with self.assertRaises(ValueError):
            QueryPlanV2(
                original_query="Give an overview",
                answer_shape="overview",
                retrieval_queries=("Give an overview",),
                requirements=(),
                confidence=0.9,
                source="local",
            )


class EvidenceContractTests(unittest.TestCase):
    def test_soft_degradation_keeps_authorized_context(self) -> None:
        state = EvidenceState(
            availability="degraded",
            confidence="retrieved",
            completeness="partial",
            reasons=("ranker_timeout",),
        )
        bundle = EvidenceBundle(
            state=state,
            items=(_item(),),
            context_item_ids=("c1",),
            answer_source_ids=("c1",),
            missing_requirement_ids=("r2",),
        )

        self.assertTrue(bundle.state.may_build_context)
        self.assertTrue(bundle.state.is_soft_degraded)
        self.assertEqual([item.chunk_id for item in bundle.context_items], ["c1"])

    def test_rejects_unauthorized_or_mismatched_context(self) -> None:
        state = EvidenceState(
            availability="ok",
            confidence="retrieved",
            completeness="partial",
        )
        with self.assertRaises(ValueError):
            EvidenceBundle(state=state, items=(_item(authorized=False),))
        with self.assertRaises(ValueError):
            EvidenceBundle(
                state=state,
                items=(_item(constraint_status="mismatch"),),
                context_item_ids=("c1",),
            )

    def test_unavailable_state_cannot_carry_context(self) -> None:
        state = EvidenceState(
            availability="unavailable",
            confidence="none",
            completeness="unknown",
            reasons=("retrieval_unavailable",),
        )
        with self.assertRaises(ValueError):
            EvidenceBundle(
                state=state,
                items=(_item(),),
                context_item_ids=("c1",),
            )


if __name__ == "__main__":
    unittest.main()
