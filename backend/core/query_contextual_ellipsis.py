"""Deterministic, source-anchored resolution for one narrow follow-up form.

This module is intentionally not a conversational rewrite utility.  It does
not concatenate history into a new question, inspect assistant text, infer a
business fact, or choose evidence.  It first derives a reusable, exact
source-span selection; compatibility callers can then project that selection
into ``query_analysis.v2`` and the current retrieval contract.

The accepted form is deliberately small:

* current turn: ``那/那么 + explicit single target + 呢``;
* immediately preceding user turn (``t1``): exactly one inheritable entity
  qualifier; and
* no current-turn entity/condition/scope that would conflict with inheritance.

Everything else returns a stable non-success reason so callers retain their
existing baseline/model path instead of guessing a historical subject.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from core.query_analysis_contract import (
    QUERY_ANALYSIS_SCHEMA_VERSION,
    QueryAnalysis,
    QueryAnalysisSourceRef,
    parse_query_analysis,
)
from core.query_constraints import extract_query_constraints
from core.query_context_inheritance import (
    assess_historical_context_inheritability,
)
from core.query_surface_structure import (
    ContextualEllipsisTarget,
    parse_contextual_ellipsis_target,
    parse_query_surface_frame,
)

@dataclass(frozen=True)
class ContextualEllipsisSourceSpan:
    """One exact literal source range selected by trusted local grammar.

    This deliberately stays independent from both V2's source-ref protocol
    and the current source-selection contract. The grammar establishes only a source fact: which
    literal range can safely be inherited.  Each execution entry must still
    bind it to its own strict contract before it may affect retrieval.
    """

    source_key: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.source_key not in {"current", "t1"}:
            raise ValueError("unsupported contextual ellipsis source key")
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("contextual ellipsis source range is invalid")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("contextual ellipsis source span is empty")

    def to_query_analysis_source_ref(self) -> QueryAnalysisSourceRef:
        """Project the immutable range into the legacy contract on demand."""

        return QueryAnalysisSourceRef(
            turn_key=self.source_key,
            start=self.start,
            end=self.end,
            span=self.text,
        )


@dataclass(frozen=True)
class ContextualEllipsisSourceSelection:
    """A server-derived selection or a precise reason it cannot be derived."""

    reason: str
    current_target: ContextualEllipsisSourceSpan | None = None
    historical_qualifier: ContextualEllipsisSourceSpan | None = None

    @property
    def selected(self) -> bool:
        return self.current_target is not None and self.historical_qualifier is not None

    def safe_summary(self) -> dict[str, object]:
        return {
            "selected": self.selected,
            "reason": self.reason,
            "context_turn_key": (
                self.historical_qualifier.source_key
                if self.historical_qualifier is not None
                else None
            ),
            "current_target_range": (
                [self.current_target.start, self.current_target.end]
                if self.current_target is not None
                else None
            ),
            "historical_qualifier_range": (
                [self.historical_qualifier.start, self.historical_qualifier.end]
                if self.historical_qualifier is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ContextualEllipsisDerivation:
    """A deterministic candidate or a fail-closed reason for not deriving it."""

    analysis: QueryAnalysis | None
    reason: str
    current_target: ContextualEllipsisSourceSpan | None = None
    historical_qualifier: ContextualEllipsisSourceSpan | None = None

    @property
    def derived(self) -> bool:
        return self.analysis is not None

    def safe_summary(self) -> dict[str, object]:
        return {
            "derived": self.derived,
            "reason": self.reason,
            "context_turn_key": (
                self.historical_qualifier.source_key
                if self.historical_qualifier is not None
                else None
            ),
            "current_target_range": (
                [self.current_target.start, self.current_target.end]
                if self.current_target is not None
                else None
            ),
            "historical_qualifier_range": (
                [self.historical_qualifier.start, self.historical_qualifier.end]
                if self.historical_qualifier is not None
                else None
            ),
        }

    def content_summary(self) -> dict[str, object] | None:
        """Return source refs only for a caller guarded by content tracing."""

        if self.current_target is None or self.historical_qualifier is None:
            return None
        return {
            "current_target_source_ref": (
                self.current_target.to_query_analysis_source_ref().to_dict()
            ),
            "historical_qualifier_source_ref": (
                self.historical_qualifier.to_query_analysis_source_ref().to_dict()
            ),
        }


def _previous_user_input(
    values: Iterable[Mapping[str, Any]] | None,
) -> str | None:
    """Return only the re-authorized immediate prior *user* source ``t1``."""

    for raw in values or ():
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("candidate_key") or "").strip() != "t1":
            continue
        # Assistant text and stored sources deliberately never enter this
        # function.  The surface contract can cite only the prior user input.
        value = str(raw.get("user_input") or "")
        return value if value.strip() else None
    return None


def _previous_unique_entity_span(
    source: str,
) -> tuple[ContextualEllipsisSourceSpan | None, str]:
    """Return one entity only when the preceding turn has no other context.

    A short follow-up is allowed to inherit *one subject*, not an implicit
    bundle of subject plus city, date, scenario, product/version or another
    condition.  Dropping any of those dimensions would silently broaden the
    request (for example, ``普通员工在上海出差的餐补`` → ``普通员工住宿``).

    The shared historical-context policy is the sole syntactic authority here
    and in the semantic compiler. Keeping it centralized prevents a model
    selection from retaining an entity that this deterministic path has
    already rejected because the same prior turn carried scope or conditions.
    """

    assessment = assess_historical_context_inheritability(
        source_key="t1",
        user_input=source,
    )
    reason_by_policy = {
        "explicit_scope": "previous_turn_has_explicit_scope",
        "entity_not_unique_or_not_inheritable": (
            "previous_turn_entity_not_unique_or_not_inheritable"
        ),
        "non_inheritable_qualifier": "previous_turn_has_non_inheritable_qualifier",
        "entity_source_not_unique": "previous_turn_entity_source_not_unique",
        "source_unavailable": "previous_user_turn_unavailable",
    }
    if not assessment.inheritable or assessment.entity is None:
        return None, reason_by_policy.get(
            assessment.reason,
            "previous_turn_entity_not_unique_or_not_inheritable",
        )
    return (
        ContextualEllipsisSourceSpan(
            source_key="t1",
            start=assessment.entity.start,
            end=assessment.entity.end,
            text=assessment.entity.text,
        ),
        "previous_turn_unique_entity_qualifier",
    )


def _current_target_has_new_qualifier_or_scope(
    target: ContextualEllipsisTarget,
) -> bool:
    """Whether the new target itself supplies a competing applicability axis."""

    # Reuse the same grammar used by the local planner, with a minimal value
    # shell solely to expose a target phrase's syntactic qualifiers.  This is
    # not a history concatenation or a business-term expansion.
    frame = parse_query_surface_frame(f"{target.target}是多少")
    if frame is not None and frame.qualifiers:
        return True
    scope = extract_query_constraints(target.target)
    return bool(scope.has_scope_constraint)


def derive_contextual_ellipsis_source_selection(
    *,
    current_question: str,
    route_context: Iterable[Mapping[str, Any]] | None,
) -> ContextualEllipsisSourceSelection:
    """Derive a strict source selection without an LLM or execution plan.

    The result has no retrieval, KB, evidence, fact, scope or bridge meaning.
    It proves only one narrow source relationship. Adapters must each
    rebind it through their respective strict contracts before execution.
    """

    target = parse_contextual_ellipsis_target(current_question)
    if target is None:
        return ContextualEllipsisSourceSelection(
            reason="current_turn_not_supported_contextual_ellipsis",
        )
    current_target = ContextualEllipsisSourceSpan(
        source_key="current",
        start=target.start,
        end=target.end,
        text=target.target,
    )
    if _current_target_has_new_qualifier_or_scope(target):
        return ContextualEllipsisSourceSelection(
            reason="current_turn_has_explicit_qualifier_or_scope",
            current_target=current_target,
        )
    previous = _previous_user_input(route_context)
    if previous is None:
        return ContextualEllipsisSourceSelection(
            reason="previous_user_turn_unavailable",
            current_target=current_target,
        )
    historical_qualifier, historical_reason = _previous_unique_entity_span(previous)
    if historical_qualifier is None:
        return ContextualEllipsisSourceSelection(
            reason=historical_reason,
            current_target=current_target,
        )

    return ContextualEllipsisSourceSelection(
        reason=historical_reason,
        current_target=current_target,
        historical_qualifier=historical_qualifier,
    )


_CONTEXTUAL_ELLIPSIS_CLARIFICATION_REASONS = frozenset({
    "previous_user_turn_unavailable",
    "previous_turn_has_explicit_scope",
    "previous_turn_entity_not_unique_or_not_inheritable",
    "previous_turn_has_non_inheritable_qualifier",
    "previous_turn_entity_source_not_unique",
})


def contextual_ellipsis_requires_clarification(
    selection: ContextualEllipsisSourceSelection,
) -> bool:
    """Whether an explicit ``那/那么…呢`` reference has no safe antecedent.

    This is intentionally narrower than a general short-question detector.
    It only closes execution when the current turn *explicitly* asks to carry
    context and the immediate prior user source cannot be inherited without
    dropping an applicability dimension.  A self-contained current question
    still remains eligible for normal semantic analysis.
    """

    if not isinstance(selection, ContextualEllipsisSourceSelection):
        raise ValueError("selection must be a ContextualEllipsisSourceSelection")
    return bool(
        selection.current_target is not None
        and not selection.selected
        and selection.reason in _CONTEXTUAL_ELLIPSIS_CLARIFICATION_REASONS
    )


def contextual_ellipsis_clarification_question(
    selection: ContextualEllipsisSourceSelection,
) -> str:
    """Return the stable user-facing clarification for a blocked carry-over."""

    if not contextual_ellipsis_requires_clarification(selection):
        raise ValueError("selection does not require contextual clarification")
    return (
        "上一轮问题包含需要确认的对象、适用范围或条件。"
        "请在本轮明确要查询的对象，以及产品/版本、项目、地点或其他适用条件。"
    )


def derive_contextual_ellipsis_analysis(
    *,
    current_question: str,
    route_context: Iterable[Mapping[str, Any]] | None,
) -> ContextualEllipsisDerivation:
    """Project the shared source selection into the V2 compatibility contract."""

    # A caller may provide a one-shot iterable.  Snapshot the bounded route
    # candidates once so the source selection and compatibility projection
    # are guaranteed to inspect the exact same authorisation set.
    route_context_values = tuple(route_context or ())
    selection = derive_contextual_ellipsis_source_selection(
        current_question=current_question,
        route_context=route_context_values,
    )
    if not selection.selected:
        return ContextualEllipsisDerivation(
            analysis=None,
            reason=selection.reason,
            current_target=selection.current_target,
            historical_qualifier=selection.historical_qualifier,
        )
    current_target = selection.current_target
    historical_qualifier = selection.historical_qualifier
    if current_target is None or historical_qualifier is None:
        raise ValueError("selected contextual ellipsis is missing source spans")
    previous = _previous_user_input(route_context_values)
    if previous is None:
        # The selection could only have been built with t1, but retain a hard
        # boundary if a mutable caller supplied a one-shot iterator.
        return ContextualEllipsisDerivation(
            analysis=None,
            reason="previous_user_turn_unavailable",
            current_target=current_target,
            historical_qualifier=historical_qualifier,
        )
    payload = {
        "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
        "relation": "followup",
        "self_contained": False,
        "context_turn_keys": ["t1"],
        "answer_candidates": [{
            "id": "a1",
            "target_source_ref": current_target.to_query_analysis_source_ref().to_dict(),
            "qualifier_source_refs": [
                historical_qualifier.to_query_analysis_source_ref().to_dict()
            ],
            "bridge_candidate_ids": [],
        }],
        "bridge_candidates": [],
        # This is a fully deterministic grammar derivation, not a model
        # confidence estimate.  The trusted execution validator still owns
        # all plan, scope, coverage and evidence permissions.
        "confidence": 1.0,
        "diagnostic": "current explicit target bound to previous unique entity qualifier",
    }
    try:
        analysis = parse_query_analysis(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            current_question=current_question,
            context_user_inputs={"t1": previous},
        )
    except Exception:
        # This should be unreachable while the internal source construction is
        # correct, but failure must remain a baseline fallback rather than a
        # request error or an unvalidated handoff.
        return ContextualEllipsisDerivation(
            analysis=None,
            reason="deterministic_source_contract_rejected",
            current_target=current_target,
            historical_qualifier=historical_qualifier,
        )
    return ContextualEllipsisDerivation(
        analysis=analysis,
        reason=selection.reason,
        current_target=current_target,
        historical_qualifier=historical_qualifier,
    )


__all__ = [
    "ContextualEllipsisDerivation",
    "ContextualEllipsisSourceSelection",
    "ContextualEllipsisSourceSpan",
    "contextual_ellipsis_clarification_question",
    "contextual_ellipsis_requires_clarification",
    "derive_contextual_ellipsis_analysis",
    "derive_contextual_ellipsis_source_selection",
]
