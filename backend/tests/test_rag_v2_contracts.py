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
    role="background",
    supports_requirement_ids=(),
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
        role=role,
        supports_requirement_ids=supports_requirement_ids,
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
    def test_evidence_role_and_requirement_ids_are_first_class(self) -> None:
        item = _item(
            role="direct",
            supports_requirement_ids=("r1", "r1", "bridge_2"),
        )

        self.assertEqual(item.role, "direct")
        self.assertEqual(item.supports_requirement_ids, ("r1", "bridge_2"))
        self.assertEqual(item.to_dict()["role"], "direct")
        self.assertEqual(
            item.to_dict()["supports_requirement_ids"],
            ["r1", "bridge_2"],
        )

    def test_invalid_evidence_role_or_requirement_id_is_rejected(self) -> None:
        for values in (
            {"role": "related"},
            {"supports_requirement_ids": ("R1",)},
            {"supports_requirement_ids": "r1"},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _item(**values)

    def test_bundle_derives_positive_coverage_from_context_roles(self) -> None:
        direct = _item(
            "direct",
            role="direct",
            supports_requirement_ids=("r1",),
        )
        conflicting = _item(
            "conflict",
            role="conflicting",
            supports_requirement_ids=("r2",),
        )
        background = _item(
            "background",
            role="background",
            supports_requirement_ids=("r3",),
        )
        bundle = EvidenceBundle(
            state=EvidenceState("ok", "retrieved", "partial"),
            items=(direct, conflicting, background),
            context_item_ids=("direct", "conflict", "background"),
            answer_source_ids=("direct",),
            missing_requirement_ids=("r2", "r3"),
        )

        self.assertEqual(bundle.covered_requirement_ids, ("r1",))
        self.assertEqual(bundle.to_dict()["covered_requirement_ids"], ["r1"])

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
