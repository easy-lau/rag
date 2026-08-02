"""Response-status precedence tests for the finalized RAG v2 evidence artifact."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.evidence_ambiguity import EvidenceAmbiguityDecision
from core.rag_v2.pipeline import _apply_final_clarification_priority


class FinalClarificationPriorityTests(unittest.TestCase):
    @staticmethod
    def _finalized(
        *,
        completeness: str,
        missing_requirement_ids: tuple[str, ...] = (),
        generation_allowed: bool,
    ) -> SimpleNamespace:
        """Build the minimal immutable-finalization view used by the gate.

        This is deliberately a pure unit test.  Route closure and evidence
        graph construction are covered elsewhere; the invariant here is that
        no producer can make a terminal evidence state interactive merely by
        returning an ambiguity decision.
        """

        return SimpleNamespace(
            generation_allowed=generation_allowed,
            assessment=SimpleNamespace(
                completeness=completeness,
                missing_requirement_ids=missing_requirement_ids,
            ),
        )

    @staticmethod
    def _ambiguity() -> EvidenceAmbiguityDecision:
        return EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension="document",
            question="请在两份制度中选择一份。",
            reason="multiple_mutually_exclusive_assessed_scopes",
            relevant_document_count=2,
            allowed_doc_ids=("policy-a", "policy-b"),
        )

    def test_error_never_becomes_clarification(self) -> None:
        adjudication = _apply_final_clarification_priority(
            base_status="error",
            finalized=self._finalized(
                completeness="unknown",
                generation_allowed=False,
            ),
            ambiguity=self._ambiguity(),
        )

        self.assertEqual(adjudication.evidence_status, "error")
        self.assertFalse(adjudication.ambiguity.needs_clarification)
        self.assertEqual(adjudication.suppression_reason, "base_status_error")
        self.assertEqual(adjudication.ambiguity.question, "")
        self.assertEqual(adjudication.ambiguity.allowed_doc_ids, ())

    def test_partial_never_becomes_clarification(self) -> None:
        adjudication = _apply_final_clarification_priority(
            base_status="partial",
            finalized=self._finalized(
                completeness="partial",
                missing_requirement_ids=("r2",),
                generation_allowed=True,
            ),
            ambiguity=self._ambiguity(),
        )

        self.assertEqual(adjudication.evidence_status, "partial")
        self.assertFalse(adjudication.ambiguity.needs_clarification)
        self.assertEqual(adjudication.suppression_reason, "base_status_partial")

    def test_scope_mismatch_never_becomes_clarification(self) -> None:
        adjudication = _apply_final_clarification_priority(
            base_status="scope_mismatch",
            finalized=self._finalized(
                completeness="unknown",
                missing_requirement_ids=("r1",),
                generation_allowed=False,
            ),
            ambiguity=self._ambiguity(),
        )

        self.assertEqual(adjudication.evidence_status, "scope_mismatch")
        self.assertFalse(adjudication.ambiguity.needs_clarification)
        self.assertEqual(
            adjudication.suppression_reason,
            "base_status_scope_mismatch",
        )

    def test_other_non_hit_terminal_statuses_remain_authoritative(self) -> None:
        for status in ("no_hit", "insufficient_evidence", "unverified"):
            with self.subTest(status=status):
                adjudication = _apply_final_clarification_priority(
                    base_status=status,
                    finalized=self._finalized(
                        completeness="unknown",
                        missing_requirement_ids=("r1",),
                        generation_allowed=False,
                    ),
                    ambiguity=self._ambiguity(),
                )

                self.assertEqual(adjudication.evidence_status, status)
                self.assertFalse(adjudication.ambiguity.needs_clarification)
                self.assertEqual(
                    adjudication.suppression_reason,
                    f"base_status_{status}",
                )

    def test_hit_still_requires_complete_generation_safe_finalization(self) -> None:
        invalid_hit = _apply_final_clarification_priority(
            base_status="hit",
            finalized=self._finalized(
                completeness="partial",
                missing_requirement_ids=("r1",),
                generation_allowed=False,
            ),
            ambiguity=self._ambiguity(),
        )

        self.assertEqual(invalid_hit.evidence_status, "hit")
        self.assertFalse(invalid_hit.ambiguity.needs_clarification)
        self.assertEqual(
            invalid_hit.suppression_reason,
            "generation_not_allowed",
        )

    def test_complete_hit_with_closed_conflict_becomes_clarification(self) -> None:
        ambiguity = self._ambiguity()
        adjudication = _apply_final_clarification_priority(
            base_status="hit",
            finalized=self._finalized(
                completeness="complete",
                generation_allowed=True,
            ),
            ambiguity=ambiguity,
        )

        self.assertEqual(adjudication.evidence_status, "needs_clarification")
        self.assertIs(adjudication.ambiguity, ambiguity)
        self.assertIsNone(adjudication.suppression_reason)


if __name__ == "__main__":
    unittest.main()
