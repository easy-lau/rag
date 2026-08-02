"""Regression tests for the V3-native strict contextual producer."""

from __future__ import annotations

import unittest

from core.query_contextual_ellipsis import derive_contextual_ellipsis_source_selection
from core.query_understanding_v3_catalog import SourceSpanCatalog
from core.query_understanding_v3_deterministic import (
    bind_deterministic_v3_contextual_ellipsis,
)


_QUESTION = "那住宿呢"
_ROUTE_CONTEXT = ({
    "candidate_key": "t1",
    "user_input": "普通员工的餐饮补贴是多少",
    # The adapter must not inspect this tempting answer text.
    "assistant_answer": "普通员工对应D级，餐补99999元。",
},)


class DeterministicV3ContextualEllipsisTests(unittest.TestCase):
    def test_source_selection_binds_only_exact_catalog_entries(self) -> None:
        selection = derive_contextual_ellipsis_source_selection(
            current_question=_QUESTION,
            route_context=_ROUTE_CONTEXT,
        )
        catalog = SourceSpanCatalog.build(
            current_question=_QUESTION,
            route_context=_ROUTE_CONTEXT,
        )

        result = bind_deterministic_v3_contextual_ellipsis(
            catalog=catalog,
            source_selection=selection,
        )

        self.assertTrue(selection.selected)
        self.assertTrue(result.applied)
        candidate = result.understanding.answer_candidates[0]
        self.assertEqual(candidate.target_span.text, "住宿")
        self.assertEqual(candidate.qualifier_spans[0].text, "普通员工")
        self.assertEqual(candidate.qualifier_spans[0].source_key, "t1")

    def test_catalog_miss_fails_closed_without_nearby_span_substitution(self) -> None:
        selection = derive_contextual_ellipsis_source_selection(
            current_question=_QUESTION,
            route_context=_ROUTE_CONTEXT,
        )
        full_catalog = SourceSpanCatalog.build(
            current_question=_QUESTION,
            route_context=_ROUTE_CONTEXT,
        )
        # Simulate a future catalog refactor forgetting the exact target range.
        # The restricted catalog remains internally valid, so this proves the
        # adapter itself does not perform a text/offset-nearby fallback.
        restricted_catalog = SourceSpanCatalog(
            entries=tuple(
                item
                for item in full_catalog.entries
                if not (
                    item.source_key == "current"
                    and item.start == 1
                    and item.end == 3
                )
            ),
            _source_texts=full_catalog._source_texts,
        )

        result = bind_deterministic_v3_contextual_ellipsis(
            catalog=restricted_catalog,
            source_selection=selection,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.reason, "current_target_not_exposed_by_catalog")
        self.assertIsNone(result.understanding)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
