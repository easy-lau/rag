import unittest

from core.evidence_status import (
    ANSWER_SOURCE_REQUIRED_EVIDENCE_STATUSES,
    CANONICAL_EVIDENCE_STATUSES,
    NON_ANSWER_EVIDENCE_STATUSES,
    normalize_evidence_status,
)
from models.schemas import IntentEvidenceStatus


class EvidenceStatusContractTests(unittest.TestCase):
    def test_legacy_version_mismatch_normalizes_to_canonical_scope_mismatch(self):
        self.assertEqual(normalize_evidence_status("version_mismatch"), "scope_mismatch")
        self.assertIn("scope_mismatch", CANONICAL_EVIDENCE_STATUSES)
        self.assertIn("scope_mismatch", NON_ANSWER_EVIDENCE_STATUSES)
        self.assertNotIn("scope_mismatch", ANSWER_SOURCE_REQUIRED_EVIDENCE_STATUSES)

    def test_public_schema_accepts_canonical_and_legacy_read_statuses(self):
        # Literal aliases are represented at runtime as strings, but this
        # assertion guards accidental removal of either rolling-upgrade value.
        values = set(IntentEvidenceStatus.__args__)
        self.assertIn("scope_mismatch", values)
        self.assertIn("version_mismatch", values)


if __name__ == "__main__":
    unittest.main()
