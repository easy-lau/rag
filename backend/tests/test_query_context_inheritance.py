"""Shared safety contract for history-derived entity qualifiers."""

from __future__ import annotations

import unittest

from core.query_context_inheritance import (
    assess_historical_context_inheritability,
)


class HistoricalContextInheritabilityTests(unittest.TestCase):
    def test_plain_single_entity_is_the_only_inheritable_envelope(self) -> None:
        assessment = assess_historical_context_inheritability(
            source_key="t1",
            user_input="普通员工的餐饮补贴是多少",
        )

        self.assertTrue(assessment.inheritable)
        self.assertEqual(assessment.reason, "unique_entity_qualifier")
        self.assertIsNotNone(assessment.entity)
        self.assertEqual(assessment.entity.text, "普通员工")
        self.assertTrue(assessment.allows_range(start=0, end=4))
        self.assertFalse(assessment.allows_range(start=0, end=3))

    def test_scope_condition_and_multiple_entities_are_never_split_from_history(self) -> None:
        cases = (
            ("普通员工在云枢8.6中的餐饮补贴是多少", "explicit_scope"),
            ("普通员工在上海出差的餐饮补贴是多少", "non_inheritable_qualifier"),
            ("普通员工国内出差的餐饮补贴是多少", "non_inheritable_qualifier"),
            ("普通员工和高级经理的餐饮补贴分别是多少", "entity_not_unique_or_not_inheritable"),
        )

        for source, expected_reason in cases:
            with self.subTest(source=source):
                assessment = assess_historical_context_inheritability(
                    source_key="t1",
                    user_input=source,
                )
                self.assertFalse(assessment.inheritable)
                self.assertEqual(assessment.reason, expected_reason)
                self.assertIsNone(assessment.entity)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
