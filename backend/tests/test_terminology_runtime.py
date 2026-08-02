"""Runtime safety tests for KB-scoped controlled terminology.

These tests deliberately exercise the pure request-local view before a
database-backed resolver is wired into the pipeline.  A registry alias is not
a global synonym: its KB, document and explicit applicability scope stay with
every retrieval variant, and only a reviewed strict equivalence may rewrite a
requirement for evidence adjudication.
"""

from __future__ import annotations

import unittest

from core.rag_v2.contracts import AnswerRequirementV2
from core.terminology_contracts import (
    TerminologyBinding,
    TerminologyForm,
)
from core.terminology_runtime import (
    RuntimeTerminologyBinding,
    TerminologyRuntimeResolution,
    build_runtime_terminology_resolution,
)


def _binding(
    *,
    requirement_id: str = "r1",
    concept_id: str = "meal_allowance",
    source_relation_strength: str = "strict_equivalent",
) -> TerminologyBinding:
    forms = (
        TerminologyForm(
            term="餐补",
            rule_id="term_meal_short",
            relation_strength=source_relation_strength,
        ),
        TerminologyForm(
            term="餐饮补贴",
            rule_id="term_meal_full",
            relation_strength="strict_equivalent",
        ),
    )
    evidence_forms = (
        TerminologyForm(
            term="餐补",
            rule_id="term_meal_short",
            relation_strength="strict_equivalent",
        ),
        TerminologyForm(
            term="餐饮补贴",
            rule_id="term_meal_full",
            relation_strength="strict_equivalent",
        ),
    ) if source_relation_strength == "strict_equivalent" else (
        TerminologyForm(
            term="餐饮补贴",
            rule_id="term_meal_full",
            relation_strength="strict_equivalent",
        ),
    )
    return TerminologyBinding(
        requirement_id=requirement_id,
        concept_id=concept_id,
        concept_key=concept_id,
        display_name="餐饮补贴",
        source_term="餐补",
        source_relation_strength=source_relation_strength,
        query_forms=forms,
        evidence_forms=evidence_forms,
        scope_binding_ids=("binding_meal",),
    )


def _requirement(*, product: str | None = None) -> AnswerRequirementV2:
    return AnswerRequirementV2(
        id="r1",
        description="普通员工的餐补是多少",
        scope_product=product,
        coverage_contract="single_claim",
        depends_on_requirement_ids=(),
        augmentation_requirement_ids=(),
    )


class TerminologyRuntimeResolutionTests(unittest.TestCase):
    def _resolution(
        self,
        *bindings: RuntimeTerminologyBinding,
    ) -> TerminologyRuntimeResolution:
        return TerminologyRuntimeResolution(
            plan_fingerprint="a" * 64,
            scope_fingerprint="b" * 64,
            registry_revisions={"kb_a": 3},
            status="resolved",
            bindings=bindings,
            authorized_kb_ids=("kb_a",),
        )

    def test_strict_alias_stays_scoped_and_can_supply_evidence_rewrite(self):
        resolution = self._resolution(RuntimeTerminologyBinding(
            binding=_binding(),
            kb_id="kb_a",
            document_id="doc_policy",
        ))
        requirement = _requirement()

        variants = resolution.retrieval_variants(
            requirement=requirement,
            maximum_aliases=3,
        )

        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].query, "普通员工的餐饮补贴是多少")
        self.assertEqual(variants[0].kb_ids, ("kb_a",))
        self.assertEqual(variants[0].document_ids, ("doc_policy",))
        self.assertEqual(
            resolution.evidence_rewrites(
                requirement=requirement,
                kb_id="kb_a",
                doc_id="doc_policy",
            )[0].description,
            "普通员工的餐饮补贴是多少",
        )
        self.assertEqual(
            resolution.evidence_rewrites(
                requirement=requirement,
                kb_id="kb_a",
                doc_id="another_doc",
            ),
            (),
        )

    def test_retrieval_only_alias_never_rewrites_evidence_target(self):
        resolution = self._resolution(RuntimeTerminologyBinding(
            binding=_binding(source_relation_strength="retrieval_only"),
            kb_id="kb_a",
            document_id=None,
        ))
        requirement = _requirement()

        self.assertEqual(len(resolution.retrieval_variants(
            requirement=requirement,
            maximum_aliases=3,
        )), 1)
        self.assertEqual(
            resolution.evidence_rewrites(
                requirement=requirement,
                kb_id="kb_a",
                doc_id="doc_policy",
            ),
            (),
        )
        self.assertIsNone(resolution.evidence_match(
            requirement=requirement,
            kb_id="kb_a",
            doc_id="doc_policy",
            content="D级餐饮补贴为100元/天。",
        ))

    def test_binding_with_unmatched_explicit_scope_does_not_expand(self):
        resolution = self._resolution(RuntimeTerminologyBinding(
            binding=_binding(),
            kb_id="kb_a",
            document_id=None,
            scope_product_key="钉钉",
        ))

        self.assertEqual(
            resolution.retrieval_variants(
                requirement=_requirement(product="云枢"),
                maximum_aliases=3,
            ),
            (),
        )

    def test_same_term_with_two_concepts_in_one_kb_fails_closed(self):
        resolution = self._resolution(
            RuntimeTerminologyBinding(
                binding=_binding(concept_id="meal_allowance"),
                kb_id="kb_a",
                document_id=None,
            ),
            RuntimeTerminologyBinding(
                binding=_binding(concept_id="meal_reimbursement"),
                kb_id="kb_a",
                document_id=None,
            ),
        )

        self.assertEqual(
            resolution.retrieval_variants(
                requirement=_requirement(),
                maximum_aliases=3,
            ),
            (),
        )
        self.assertIn("ambiguous_source_term", resolution.diagnostics)

    def test_unauthorized_binding_degrades_without_returning_aliases(self):
        resolution = build_runtime_terminology_resolution(
            plan_fingerprint="a" * 64,
            scope_fingerprint="b" * 64,
            authorized_kb_ids=("kb_a",),
            registry_revisions={"kb_a": 3, "kb_b": 4},
            bindings=(RuntimeTerminologyBinding(
                binding=_binding(),
                kb_id="kb_b",
            ),),
        )

        self.assertEqual(resolution.status, "degraded")
        self.assertEqual(
            resolution.retrieval_variants(
                requirement=_requirement(),
                maximum_aliases=3,
            ),
            (),
        )
        self.assertEqual(resolution.trace_summary()["binding_count"], 0)

    def test_unselected_registry_revision_is_not_retained_in_trace_state(self):
        resolution = build_runtime_terminology_resolution(
            plan_fingerprint="a" * 64,
            scope_fingerprint="b" * 64,
            authorized_kb_ids=("kb_a",),
            registry_revisions={"kb_a": 3, "kb_b": 99},
            bindings=(RuntimeTerminologyBinding(
                binding=_binding(),
                kb_id="kb_a",
            ),),
        )

        self.assertEqual(resolution.status, "resolved")
        self.assertEqual(resolution.trace_summary()["registry_revisions"], [3])

    def test_strict_evidence_match_retains_only_scoped_provenance(self):
        resolution = self._resolution(RuntimeTerminologyBinding(
            binding=_binding(),
            kb_id="kb_a",
            document_id="doc_policy",
        ))

        match = resolution.evidence_match(
            requirement=_requirement(),
            kb_id="kb_a",
            doc_id="doc_policy",
            content="D级餐饮补贴为100元/天。",
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.requirement.description, "普通员工的餐饮补贴是多少")
        self.assertIn("binding_meal", match.rule_ids)
        self.assertIsNone(resolution.evidence_match(
            requirement=_requirement(),
            kb_id="kb_a",
            doc_id="another_doc",
            content="D级餐饮补贴为100元/天。",
        ))

    def test_trace_summary_contains_no_business_terms_or_scope_ids(self):
        resolution = self._resolution(RuntimeTerminologyBinding(
            binding=_binding(),
            kb_id="kb_a",
            document_id="doc_policy",
        ))

        summary = resolution.trace_summary()

        self.assertNotIn("餐补", str(summary))
        self.assertNotIn("doc_policy", str(summary))
        self.assertNotIn("kb_a", str(summary))
        self.assertEqual(summary["registry_revisions"], [3])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
