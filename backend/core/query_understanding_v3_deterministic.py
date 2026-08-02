"""Trusted local source selection for the narrow V3 ellipsis fast path.

This module does not add a second retrieval planner.  It adapts the shared
``那/那么 + 明确目标 + 呢`` grammar into a normal ``query_understanding.v3``
selection, then the ordinary V3 compiler and V2 task graph remain the only
execution path.  It is deliberately limited to an exact current target plus
one unique, re-authorised ``t1`` entity qualifier; anything less certain falls
through to the model-backed V3 selector or the normal fail-closed baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from core.query_contextual_ellipsis import (
    ContextualEllipsisSourceSelection,
    derive_contextual_ellipsis_source_selection,
)
from core.query_understanding_v3_catalog import SourceSpanCatalog
from core.query_understanding_v3_contract import (
    QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
    QueryUnderstandingV3,
    QueryUnderstandingV3ValidationError,
    parse_query_understanding,
)


@dataclass(frozen=True)
class DeterministicV3ContextualEllipsis:
    """A V3-compatible local selection or an exact non-selection reason."""

    understanding: QueryUnderstandingV3 | None
    reason: str
    source_selection: ContextualEllipsisSourceSelection

    @property
    def applied(self) -> bool:
        return self.understanding is not None

    def safe_summary(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "source_selection": self.source_selection.safe_summary(),
            "analysis": (
                self.understanding.safe_summary()
                if self.understanding is not None
                else None
            ),
        }


def derive_deterministic_v3_contextual_ellipsis(
    *,
    catalog: SourceSpanCatalog,
    current_question: str,
    route_context: Iterable[Mapping[str, Any]] | None,
) -> DeterministicV3ContextualEllipsis:
    """Derive then bind a proven ellipsis selection to this exact V3 catalog.

    This compatibility convenience is useful to isolated callers.  Production
    orchestration first derives the source selection, so it can avoid building
    a second catalog for ordinary non-ellipsis questions, then calls
    :func:`bind_deterministic_v3_contextual_ellipsis` with the same selection.
    """

    selection = derive_contextual_ellipsis_source_selection(
        current_question=current_question,
        route_context=route_context,
    )
    return bind_deterministic_v3_contextual_ellipsis(
        catalog=catalog,
        source_selection=selection,
    )


def bind_deterministic_v3_contextual_ellipsis(
    *,
    catalog: SourceSpanCatalog,
    source_selection: ContextualEllipsisSourceSelection,
) -> DeterministicV3ContextualEllipsis:
    """Bind one already-derived source selection to an exact V3 catalog.

    No source range is broadened or approximated: each range must already be
    exposed by the catalog as one exact entry.  The final call to the regular
    V3 parser preserves the same opaque-ID and overlap checks applied to a
    model response.
    """

    if not isinstance(catalog, SourceSpanCatalog):
        raise ValueError("catalog must be a SourceSpanCatalog")
    if not isinstance(source_selection, ContextualEllipsisSourceSelection):
        raise ValueError("source_selection must be a ContextualEllipsisSourceSelection")
    selection = source_selection
    if not selection.selected:
        return DeterministicV3ContextualEllipsis(
            understanding=None,
            reason=selection.reason,
            source_selection=selection,
        )
    current_target = selection.current_target
    historical_qualifier = selection.historical_qualifier
    if current_target is None or historical_qualifier is None:
        raise ValueError("selected contextual ellipsis is missing source spans")
    target_span = catalog.find_exact_span(
        source_key=current_target.source_key,
        start=current_target.start,
        end=current_target.end,
    )
    if target_span is None or target_span.source_kind != "current":
        return DeterministicV3ContextualEllipsis(
            understanding=None,
            reason="current_target_not_exposed_by_catalog",
            source_selection=selection,
        )
    qualifier_span = catalog.find_exact_span(
        source_key=historical_qualifier.source_key,
        start=historical_qualifier.start,
        end=historical_qualifier.end,
    )
    if qualifier_span is None or qualifier_span.source_kind != "route_context":
        return DeterministicV3ContextualEllipsis(
            understanding=None,
            reason="historical_qualifier_not_exposed_by_catalog",
            source_selection=selection,
        )
    payload: dict[str, Any] = {
        "schema_version": QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
        "answer_candidates": [{
            "id": "a1",
            "target_span_id": target_span.span_id,
            "qualifier_span_ids": [qualifier_span.span_id],
        }],
    }
    try:
        understanding = parse_query_understanding(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            catalog=catalog,
        )
    except QueryUnderstandingV3ValidationError:
        # The grammar only chooses source ranges.  If those ranges cannot
        # satisfy the public V3 protocol, a caller must retain its usual
        # model/baseline path rather than give this local adapter an exception.
        return DeterministicV3ContextualEllipsis(
            understanding=None,
            reason="deterministic_selection_rejected_by_v3_contract",
            source_selection=selection,
        )
    return DeterministicV3ContextualEllipsis(
        understanding=understanding,
        reason="previous_turn_unique_entity_qualifier",
        source_selection=selection,
    )


__all__ = [
    "DeterministicV3ContextualEllipsis",
    "bind_deterministic_v3_contextual_ellipsis",
    "derive_deterministic_v3_contextual_ellipsis",
]
