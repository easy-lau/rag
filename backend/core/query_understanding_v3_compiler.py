"""Trusted compiler for catalog-bound ``query_understanding.v3`` candidates.

V3 deliberately moves the model boundary *before* query planning.  A model can
select only source spans issued by :mod:`query_understanding_v3_catalog`; it
cannot emit a knowledge-base choice, permission, scope, alias, fact, bridge
kind or graph edge.  This module owns all executable meaning:

* the current question remains the immutable retrieval anchor;
* explicit product/version partitions are re-derived from that question only;
* every requirement description is a whitespace join of catalog spans;
* an implicit bridge is admitted only when the existing trusted grammar proves
  it is a ``classification`` *augmentation*, never a proof dependency; and
* rejected candidates fall back atomically to the caller's deterministic plan.

The module is intentionally independent from ``chat.py``, the RAG pipeline and
the V2 query-analysis modules.  It emits the existing ``QueryPlanV2`` /
``RagExecutionBundle`` handoff so an integration layer can adopt it without a
second execution path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

from core.query_constraints import ApplicabilityScope, extract_applicability_scopes, extract_query_constraints
from core.query_context_inheritance import (
    HistoricalContextInheritability,
    assess_historical_context_inheritability,
)
from core.query_surface_structure import (
    is_exhaustive_configuration_request,
    answer_target_semantics,
    current_turn_candidate_targets_are_complete,
    has_current_turn_local_enumeration_antecedent,
    parse_distributive_enumeration,
    parse_query_surface_frame,
)
from core.query_understanding_v3_catalog import CatalogSpan, SourceSpanCatalog
from core.query_understanding_v3_contract import (
    QueryUnderstandingV3,
    QueryUnderstandingV3Candidate,
)
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.task_graph import RagExecutionBundle, compile_rag_execution_bundle


_MAX_REQUIREMENTS = 8
_NORMALIZE_RE = re.compile(r"\s+")
_NON_ANSWER_TARGET_RE = re.compile(
    r"^(?:什么|多少|哪些|如何|怎么|怎样|是否|能否|可以吗|吗|呢|这些|那些|"
    r"这个|那个|内容|信息|问题|标准|规定|政策|流程)$",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"(?:对比|比较|区别|差异|不同(?:点)?|异同|优劣|分别|各自|"
    r"(?:^|\s)(?:vs\.?|versus)(?:\s|$))",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[？?；;\n]")
_QUESTION_TAIL_RE = re.compile(
    r"(?:分别|各自|都)?(?:是|为)?(?:多少|什么|哪些|如何|怎么|怎样|"
    r"有何|有什么)\s*[？?。！!]*$",
    re.IGNORECASE,
)
_COORDINATION_RE = re.compile(r"(?:、|[,，]|以及|和|及|与|还有)")


CompilerDecision = Literal["compiled", "fallback"]


def _normalised(value: object) -> str:
    return _NORMALIZE_RE.sub("", str(value or "")).casefold()


def _normalised_text(value: object) -> str:
    return _NORMALIZE_RE.sub(" ", str(value or "")).strip()


def _scope_source_ranges(scope: ApplicabilityScope) -> tuple[tuple[int, int], ...]:
    """Return only trusted current-question scope ranges."""

    return tuple(
        (item.start, item.end)
        for item in scope.source_spans
        if item.origin == "current_query"
    )


def _scope_sort_key(scope: ApplicabilityScope) -> tuple[int, int, str]:
    ranges = _scope_source_ranges(scope)
    if not ranges:
        return (10**9, 10**9, scope.fingerprint)
    return (min(start for start, _ in ranges), max(end for _, end in ranges), scope.fingerprint)


def _deduplicated_scopes(values: tuple[ApplicabilityScope, ...]) -> tuple[ApplicabilityScope, ...]:
    result: list[ApplicabilityScope] = []
    seen: set[str] = set()
    for scope in values:
        if scope.fingerprint in seen:
            continue
        seen.add(scope.fingerprint)
        result.append(scope)
    return tuple(sorted(result, key=_scope_sort_key))


@dataclass(frozen=True)
class BaselineFloor:
    """Trusted non-model floor for one V3 compile attempt.

    ``fallback_plan`` is deliberately a *fallback*, not a semantic authority
    over accepted V3 candidates.  This matters when a legacy local planner is
    conservative or not runnable: a valid, fully source-bound V3 candidate
    may still produce a ledgered ``QueryPlanV2``.  Scope guard and partitions
    are re-derived from ``current_question`` in ``__post_init__``; callers
    cannot inject or widen them.
    """

    current_question: str
    fallback_plan: QueryPlanV2
    hard_clarification_reason: str | None = None
    scope_guard: ApplicabilityScope = field(init=False)
    scope_partitions: tuple[ApplicabilityScope, ...] = field(init=False)

    def __post_init__(self) -> None:
        question = str(self.current_question or "")
        if not question.strip():
            raise ValueError("baseline floor requires a non-empty current_question")
        if not isinstance(self.fallback_plan, QueryPlanV2):
            raise ValueError("baseline floor requires a QueryPlanV2 fallback_plan")
        hard_reason = self.hard_clarification_reason
        if hard_reason is not None:
            hard_reason = _normalised_text(hard_reason)
            if not hard_reason:
                raise ValueError("hard clarification reason must not be empty")
            if len(hard_reason) > 200:
                raise ValueError("hard clarification reason exceeds 200 characters")

        # No field supplied by a caller/model participates in this derivation.
        partitions = _deduplicated_scopes(tuple(extract_applicability_scopes(question)))
        object.__setattr__(self, "current_question", question)
        object.__setattr__(self, "hard_clarification_reason", hard_reason)
        object.__setattr__(self, "scope_guard", extract_query_constraints(question))
        object.__setattr__(self, "scope_partitions", partitions)

    def safe_summary(self) -> dict[str, object]:
        return {
            "has_hard_clarification_guard": self.hard_clarification_reason is not None,
            "explicit_scope_partition_count": len(self.scope_partitions),
            "scope_guard_has_constraint": self.scope_guard.has_scope_constraint,
            "fallback_answer_shape": self.fallback_plan.answer_shape,
            "fallback_is_runnable": not self.fallback_plan.needs_clarification
            and self.fallback_plan.answer_shape != "unknown",
        }


@dataclass(frozen=True)
class ScopeBinding:
    """Trusted association of one selected target with one-or-more scopes.

    Multiple scopes are allowed only for a source-visible comparison.  They are
    emitted as independent requirement instances, never folded into a single
    merged scope.
    """

    candidate_id: str
    scope_fingerprints: tuple[str, ...]


@dataclass(frozen=True)
class QueryUnderstandingV3ExecutionValidation:
    """Compiler-side validation outcome, safe to attach to an audit trace."""

    accepted: bool
    reason: str
    current_target_count: int
    candidate_target_count: int
    explicit_scope_partition_count: int
    projected_requirement_count: int = 0
    scope_bindings: tuple[ScopeBinding, ...] = ()
    # A rejected selection usually falls back to the current-turn baseline.
    # Historical-envelope violations are different: the model explicitly
    # declared the answer context-dependent, but the only selected history
    # cannot be carried without losing meaning.  That must close execution
    # through the normal clarification gate rather than retrieve a naked
    # phrase and silently change the user's scope.
    requires_clarification: bool = False

    def safe_summary(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "current_target_count": self.current_target_count,
            "candidate_target_count": self.candidate_target_count,
            "explicit_scope_partition_count": self.explicit_scope_partition_count,
            "projected_requirement_count": self.projected_requirement_count,
            "scope_binding_count": len(self.scope_bindings),
            "requires_clarification": self.requires_clarification,
        }


@dataclass(frozen=True)
class CompiledQueryUnderstanding:
    """One immutable V3 candidate-to-ledger handoff."""

    plan: QueryPlanV2
    execution_bundle: RagExecutionBundle
    validation: QueryUnderstandingV3ExecutionValidation
    compiler_decision: CompilerDecision
    description_span_ids: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, QueryPlanV2):
            raise ValueError("compiled V3 result requires a QueryPlanV2")
        if not isinstance(self.execution_bundle, RagExecutionBundle):
            raise ValueError("compiled V3 result requires a RagExecutionBundle")
        if self.execution_bundle.plan is not self.plan and self.execution_bundle.plan != self.plan:
            raise ValueError("compiled V3 execution bundle must carry the compiled plan")
        if not isinstance(self.validation, QueryUnderstandingV3ExecutionValidation):
            raise ValueError("compiled V3 result requires execution validation")
        if self.compiler_decision not in {"compiled", "fallback"}:
            raise ValueError("compiled V3 result has unsupported decision")
        normalized: dict[str, tuple[str, ...]] = {}
        for requirement_id, span_ids in dict(self.description_span_ids).items():
            key = str(requirement_id or "").strip()
            if not key:
                raise ValueError("description provenance requires a requirement id")
            if isinstance(span_ids, str) or not isinstance(span_ids, tuple):
                raise ValueError("description provenance requires tuple span ids")
            normalized[key] = tuple(str(item) for item in span_ids)
        object.__setattr__(self, "description_span_ids", MappingProxyType(normalized))

    @property
    def used_fallback(self) -> bool:
        return self.compiler_decision == "fallback"

    def safe_summary(self) -> dict[str, object]:
        return {
            "compiler_decision": self.compiler_decision,
            "used_fallback": self.used_fallback,
            "plan_schema_version": self.plan.schema_version,
            "answer_shape": self.plan.answer_shape,
            "requirement_count": len(self.plan.requirements),
            "description_provenance_count": len(self.description_span_ids),
            "validation": self.validation.safe_summary(),
            "execution_bundle": self.execution_bundle.safe_summary(),
        }


def _catalog_matches_floor(
    catalog: SourceSpanCatalog,
    floor: BaselineFloor,
) -> bool:
    try:
        return catalog.source_text_for("current") == floor.current_question
    except Exception:  # SourceSpanCatalog already has a strict public error type.
        return False


def _answer_like_target(span: CatalogSpan, *, question: str) -> bool:
    """Reject grammatical shells and qualifier-only target substitutions.

    This is not a business vocabulary check.  It only ensures a target is a
    source-visible answer head compatible with the trusted surface frame when
    that frame has one.  The full current question remains legal as the
    conservative direct target when no qualifier was selected.
    """

    text = _normalised_text(span.text).strip(" ，,。；;：:！？?!")
    if len(text) < 2 or _NON_ANSWER_TARGET_RE.fullmatch(text):
        return False
    frame = parse_query_surface_frame(question)
    if frame is None or not frame.answer_target:
        return True
    expected = _normalised(frame.answer_target)
    actual = _normalised(text)
    whole = _normalised(question)
    # A whole-turn target is a safe (if broad) fall-back selection.  Other
    # targets must be a literal part of the syntax-derived answer head, not a
    # qualifier such as ``普通员工`` substituted for ``餐补``.
    if actual == whole:
        return True
    return bool(actual and expected and (actual in expected or expected in actual))


def _current_scope_ranges(floor: BaselineFloor) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted({
            item
            for scope in floor.scope_partitions
            for item in _scope_source_ranges(scope)
        })
    )


def _candidate_current_qualifier_ranges(
    understanding: QueryUnderstandingV3,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (span.start, span.end)
        for candidate in understanding.answer_candidates
        for span in candidate.qualifier_spans
        if span.source_kind == "current"
    )


def _candidate_spans_are_safe(
    understanding: QueryUnderstandingV3,
    *,
    catalog: SourceSpanCatalog,
    floor: BaselineFloor,
) -> tuple[bool, str]:
    """Defence-in-depth after the strict parser binds span IDs to catalog."""

    targets: list[CatalogSpan] = []
    historical_source_assessments: dict[str, HistoricalContextInheritability] = {}
    for candidate in understanding.answer_candidates:
        try:
            target = catalog.resolve(candidate.target_span_id)
            qualifiers = tuple(catalog.resolve(value) for value in candidate.qualifier_span_ids)
        except Exception:
            return False, "candidate_span_not_in_catalog"
        if target != candidate.target_span or target.source_kind != "current":
            return False, "candidate_target_not_current_catalog_span"
        if not _answer_like_target(target, question=floor.current_question):
            return False, "candidate_target_not_surface_answer_like"
        if target in targets or any(target.overlaps(item) for item in targets):
            return False, "candidate_targets_overlap"
        targets.append(target)
        if tuple(candidate.qualifier_spans) != qualifiers:
            return False, "candidate_qualifier_catalog_mismatch"
        if any(target.overlaps(item) for item in qualifiers):
            return False, "candidate_target_qualifier_overlap"
        if len({item.span_id for item in qualifiers}) != len(qualifiers):
            return False, "candidate_qualifiers_duplicate"
        if understanding.self_contained and any(
            item.source_kind != "current" for item in qualifiers
        ):
            return False, "self_contained_candidate_uses_context"
        if any(
            item.source_kind == "route_context"
            and item.source_key not in catalog.authorised_context_keys
            for item in qualifiers
        ):
            return False, "candidate_context_not_authorised"
        context_sources = {
            item.source_key
            for item in qualifiers
            if item.source_kind == "route_context"
        }
        # One answer head may be enriched by one historical entity only.  A
        # model cannot establish a relationship between two prior turns merely
        # by selecting their words; accepting a mixed source set would make
        # source order an implicit, unverified semantic edge.
        if len(context_sources) > 1:
            return False, "candidate_uses_multiple_historical_sources"
        for qualifier in qualifiers:
            if qualifier.source_kind != "route_context":
                continue
            assessment = historical_source_assessments.get(qualifier.source_key)
            if assessment is None:
                assessment = assess_historical_context_inheritability(
                    source_key=qualifier.source_key,
                    user_input=catalog.source_text_for(qualifier.source_key),
                )
                historical_source_assessments[qualifier.source_key] = assessment
            if not assessment.inheritable:
                return False, (
                    "historical_context_not_inheritable_"
                    f"{assessment.reason}"
                )
            if not assessment.allows_range(
                start=qualifier.start,
                end=qualifier.end,
            ):
                return False, "historical_context_not_exact_entity_span"
    return True, "ok"


def _rejection_requires_contextual_clarification(reason: str) -> bool:
    """Whether a rejected V3 candidate proved unsafe history dependence.

    This is a backend policy classification, never model output.  The caller
    may only close execution for source-envelope failures; syntax/coverage
    validation failures continue to use the current-turn deterministic floor.
    """

    return bool(
        reason.startswith("historical_context_not_inheritable_")
        or reason in {
            "historical_context_not_exact_entity_span",
            "candidate_uses_multiple_historical_sources",
        }
    )


def _is_explicit_scope_comparison(question: str, floor: BaselineFloor) -> bool:
    return len(floor.scope_partitions) > 1 and bool(_COMPARISON_RE.search(question))


def _requires_multiple_current_target_coverage(
    question: str,
    floor: BaselineFloor,
) -> bool:
    """Whether current-turn grammar proves that one target would drop work.

    This is intentionally a structural guard, not a fallback-plan comparison.
    A valid explicit multi-version comparison is handled separately by trusted
    scope fan-out, so its shared answer head is not falsely required to appear
    once per version in the model candidate list.
    """

    if _is_explicit_scope_comparison(question, floor):
        return False
    enumeration = parse_distributive_enumeration(question)
    if enumeration is not None and len(enumeration.parts) >= 2:
        return True
    if has_current_turn_local_enumeration_antecedent(question):
        return True
    clauses = [
        item.strip()
        for item in _CLAUSE_BOUNDARY_RE.split(question)
        if item.strip()
    ]
    if len(clauses) >= 2:
        return True
    # Strong compact forms such as ``住宿标准和餐补是多少`` do not always
    # carry ``分别``.  A bounded interrogative tail plus a coordination marker
    # is enough to require full candidate coverage; no company terminology or
    # retrieval result enters this decision.
    tail_match = _QUESTION_TAIL_RE.search(question)
    if tail_match is None:
        return False
    body = question[:tail_match.start()]
    return bool(_COORDINATION_RE.search(body))


def _scope_covered_by_qualifier(
    scope: ApplicabilityScope,
    qualifier: CatalogSpan,
) -> bool:
    if qualifier.source_kind != "current":
        return False
    ranges = _scope_source_ranges(scope)
    return bool(ranges) and all(
        qualifier.start <= start and end <= qualifier.end
        for start, end in ranges
    )


def _same_clause(
    question: str,
    left: int,
    right: int,
) -> bool:
    start, end = sorted((left, right))
    return _CLAUSE_BOUNDARY_RE.search(question, start, end) is None


def _scopes_for_candidate(
    candidate: QueryUnderstandingV3Candidate,
    *,
    floor: BaselineFloor,
    sequential_scope_assignment: bool,
) -> tuple[ApplicabilityScope, ...] | None:
    """Associate a target with server-derived scope partitions only.

    A model has no scope field.  We can nevertheless use a literal qualifier
    that covers a source-authored scope span, or a target lying between two
    scope phrases in the same clause.  A common target after multiple scopes
    is expanded only for explicit comparison/distributive wording; otherwise
    the candidate is rejected instead of silently selecting/merging a version.
    """

    partitions = floor.scope_partitions
    if not partitions:
        return (floor.scope_guard,)
    if len(partitions) == 1:
        return partitions

    explicit_matches = tuple(
        scope
        for scope in partitions
        if any(_scope_covered_by_qualifier(scope, qualifier) for qualifier in candidate.qualifier_spans)
    )
    if len(explicit_matches) == 1:
        return explicit_matches
    if len(explicit_matches) > 1:
        # One answer requirement can never carry a combined product/version
        # boundary.  The only safe fan-out is a source-visible comparison.
        return explicit_matches if _is_explicit_scope_comparison(
            floor.current_question, floor
        ) else None

    target = candidate.target_span
    preceding: list[ApplicabilityScope] = []
    following: list[ApplicabilityScope] = []
    for scope in partitions:
        ranges = _scope_source_ranges(scope)
        if not ranges:
            continue
        scope_start = min(start for start, _ in ranges)
        scope_end = max(end for _, end in ranges)
        if scope_end <= target.start and _same_clause(
            floor.current_question, scope_end, target.start
        ):
            preceding.append(scope)
        if target.end <= scope_start and _same_clause(
            floor.current_question, target.end, scope_start
        ):
            following.append(scope)
    if not preceding:
        return partitions if _is_explicit_scope_comparison(
            floor.current_question, floor
        ) else None
    latest = max(preceding, key=_scope_sort_key)

    # A target physically between two explicit scope phrases has a trusted
    # local owner, even if the overall sentence is a comparison.  This avoids
    # cross-product expansion of ``V6 的安装要求和 V7 的升级要求``.
    if following:
        return (latest,)

    # If a target is after the final scope phrase, selecting the last scope
    # would silently lose another explicit version.  Require an exact source
    # qualifier or explicit comparison instead.
    final_scope_end = max(
        max(end for _, end in _scope_source_ranges(scope))
        for scope in partitions
        if _scope_source_ranges(scope)
    )
    if target.start >= final_scope_end:
        if _is_explicit_scope_comparison(floor.current_question, floor):
            # A single common answer head after all scopes (``比较 V6 和 V7
            # 的审批流程差异``) genuinely applies to every partition.  Once
            # another target is structurally located between scope phrases,
            # however, this is a sequential per-scope form and the terminal
            # target belongs to the final partition only.
            return (latest,) if sequential_scope_assignment else partitions
        return None
    return (latest,)


def _has_sequential_scope_assignment(
    understanding: QueryUnderstandingV3,
    *,
    floor: BaselineFloor,
) -> bool:
    """Prove that at least one selected target lies between scope phrases."""

    if len(floor.scope_partitions) < 2:
        return False
    for candidate in understanding.answer_candidates:
        target = candidate.target_span
        has_preceding = False
        has_following = False
        for scope in floor.scope_partitions:
            ranges = _scope_source_ranges(scope)
            if not ranges:
                continue
            scope_start = min(start for start, _ in ranges)
            scope_end = max(end for _, end in ranges)
            has_preceding = has_preceding or (
                scope_end <= target.start
                and _same_clause(floor.current_question, scope_end, target.start)
            )
            has_following = has_following or (
                target.end <= scope_start
                and _same_clause(floor.current_question, target.end, scope_start)
            )
        if has_preceding and has_following:
            return True
    return False


def _answer_description_span_ids(
    candidate: QueryUnderstandingV3Candidate,
) -> tuple[str, ...]:
    """Return the sole allowed provenance for a generated answer description."""

    values: list[str] = []
    for span in (*candidate.qualifier_spans, candidate.target_span):
        if span.span_id not in values:
            values.append(span.span_id)
    return tuple(values)


def _description_from_catalog(
    catalog: SourceSpanCatalog,
    span_ids: tuple[str, ...],
) -> str:
    if not span_ids:
        raise ValueError("catalog description requires at least one span")
    # A plain join is intentional: no relation wording, synonym, scope value
    # or business term is permitted to enter a description at this boundary.
    description = " ".join(
        _normalised_text(catalog.resolve(span_id).text)
        for span_id in span_ids
    ).strip()
    if not description:
        raise ValueError("catalog description resolved empty")
    return description


def _answer_coverage(
    *,
    question: str,
    candidate_count: int,
) -> tuple[str, str]:
    """Choose only a small trusted coverage contract from surface grammar."""

    if is_exhaustive_configuration_request(question):
        return "collection", "structured_collection"
    frame = parse_query_surface_frame(question)
    if candidate_count == 1 and frame is not None and frame.question_operator == "enumeration":
        return "collection", "structured_collection"
    return "single", "single_claim"


def _answer_shape(
    *,
    question: str,
    candidate_count: int,
    floor: BaselineFloor,
) -> str:
    if _is_explicit_scope_comparison(question, floor):
        return "comparison"
    if candidate_count > 1:
        return "multi_part"
    if is_exhaustive_configuration_request(question):
        return "list"
    frame = parse_query_surface_frame(question)
    if frame is not None and frame.question_operator == "enumeration":
        return "list"
    return "fact"


def _classification_subject(
    candidate: QueryUnderstandingV3Candidate,
    *,
    question: str,
) -> CatalogSpan | None:
    """Return a qualifier whose trusted grammar permits augmentation.

    This delegates the decision to the shared surface semantic contract.  The
    target can be a V3-selected current-turn span while the qualifier is a
    validated history span, so a compact synthetic question must not be used
    as a second parsing authority.
    """

    frame = parse_query_surface_frame(question)
    if frame is None or frame.question_operator == "relation":
        return None

    for qualifier in candidate.qualifier_spans:
        semantics = answer_target_semantics(
            question,
            answer_target=candidate.target_span.text,
            entity_qualifier=qualifier.text,
        )
        if semantics.classification_augmentation_allowed:
            return qualifier
    return None


def _projected_requirement_count(
    understanding: QueryUnderstandingV3,
    *,
    bindings: Mapping[str, tuple[ApplicabilityScope, ...]],
    question: str,
) -> int:
    """Count answer instances plus scope-local optional bridge instances."""

    answer_count = sum(len(bindings[candidate.id]) for candidate in understanding.answer_candidates)
    bridge_keys: set[tuple[str, str]] = set()
    for candidate in understanding.answer_candidates:
        subject = _classification_subject(candidate, question=question)
        if subject is None:
            continue
        for scope in bindings[candidate.id]:
            bridge_keys.add((subject.span_id, scope.fingerprint))
    return answer_count + len(bridge_keys)


def validate_query_understanding(
    *,
    catalog: SourceSpanCatalog,
    understanding: QueryUnderstandingV3,
    baseline_floor: BaselineFloor,
) -> QueryUnderstandingV3ExecutionValidation:
    """Validate one parsed V3 candidate before it can create retrieval work.

    This function accepts only the parser's strict contract and exact catalog;
    it does not inspect raw model JSON.  Rejection is atomic: callers must
    retain ``baseline_floor.fallback_plan`` and never adopt a subset of model
    targets.
    """

    if not isinstance(catalog, SourceSpanCatalog):
        raise ValueError("catalog must be a SourceSpanCatalog")
    if not isinstance(understanding, QueryUnderstandingV3):
        raise ValueError("understanding must be a QueryUnderstandingV3")
    if not isinstance(baseline_floor, BaselineFloor):
        raise ValueError("baseline_floor must be a BaselineFloor")

    candidate_count = len(understanding.answer_candidates)
    base = {
        "current_target_count": candidate_count,
        "candidate_target_count": candidate_count,
        "explicit_scope_partition_count": len(baseline_floor.scope_partitions),
    }

    def rejected(reason: str) -> QueryUnderstandingV3ExecutionValidation:
        return QueryUnderstandingV3ExecutionValidation(
            accepted=False,
            reason=reason,
            requires_clarification=(
                _rejection_requires_contextual_clarification(reason)
            ),
            **base,
        )

    if not _catalog_matches_floor(catalog, baseline_floor):
        return rejected("catalog_current_question_mismatch")
    if baseline_floor.hard_clarification_reason is not None:
        return rejected("hard_clarification_guard")
    spans_safe, span_reason = _candidate_spans_are_safe(
        understanding,
        catalog=catalog,
        floor=baseline_floor,
    )
    if not spans_safe:
        return rejected(span_reason)

    requires_multi_target_coverage = _requires_multiple_current_target_coverage(
        baseline_floor.current_question,
        baseline_floor,
    )
    if requires_multi_target_coverage and candidate_count < 2:
        return rejected("current_target_coverage_incomplete")
    if candidate_count > 1:
        target_ranges = tuple(
            (item.target_span.start, item.target_span.end)
            for item in understanding.answer_candidates
        )
        if not current_turn_candidate_targets_are_complete(
            baseline_floor.current_question,
            target_ranges=target_ranges,
            qualifier_ranges=_candidate_current_qualifier_ranges(understanding),
            trusted_ranges=_current_scope_ranges(baseline_floor),
        ):
            return rejected("current_target_coverage_incomplete")

    sequential_scope_assignment = _has_sequential_scope_assignment(
        understanding,
        floor=baseline_floor,
    )
    bindings: dict[str, tuple[ApplicabilityScope, ...]] = {}
    for candidate in understanding.answer_candidates:
        scopes = _scopes_for_candidate(
            candidate,
            floor=baseline_floor,
            sequential_scope_assignment=sequential_scope_assignment,
        )
        if not scopes:
            return rejected("multiple_explicit_scopes_unbound_target")
        bindings[candidate.id] = scopes

    projected_requirement_count = _projected_requirement_count(
        understanding,
        bindings=bindings,
        question=baseline_floor.current_question,
    )
    if projected_requirement_count > _MAX_REQUIREMENTS:
        return QueryUnderstandingV3ExecutionValidation(
            accepted=False,
            reason="compiled_requirement_capacity_exceeded",
            projected_requirement_count=projected_requirement_count,
            **base,
        )

    return QueryUnderstandingV3ExecutionValidation(
        accepted=True,
        reason="catalog_bound_candidate_compilable",
        projected_requirement_count=projected_requirement_count,
        scope_bindings=tuple(
            ScopeBinding(
                candidate_id=candidate.id,
                scope_fingerprints=tuple(
                    scope.fingerprint for scope in bindings[candidate.id]
                ),
            )
            for candidate in understanding.answer_candidates
        ),
        **base,
    )


def _bound_scopes(
    validation: QueryUnderstandingV3ExecutionValidation,
    *,
    candidate_id: str,
    floor: BaselineFloor,
) -> tuple[ApplicabilityScope, ...]:
    binding = next(
        (item for item in validation.scope_bindings if item.candidate_id == candidate_id),
        None,
    )
    if binding is None:
        raise ValueError("accepted validation is missing a candidate scope binding")
    by_fingerprint = {
        scope.fingerprint: scope
        for scope in (*floor.scope_partitions, floor.scope_guard)
    }
    scopes = tuple(
        by_fingerprint[fingerprint]
        for fingerprint in binding.scope_fingerprints
        if fingerprint in by_fingerprint
    )
    if len(scopes) != len(binding.scope_fingerprints):
        raise ValueError("accepted validation references an unknown scope partition")
    return scopes


def _next_requirement_id(prefix: str, used: set[str]) -> str:
    index = 1
    while f"{prefix}{index}" in used:
        index += 1
    value = f"{prefix}{index}"
    used.add(value)
    return value


def _compiled_plan(
    *,
    catalog: SourceSpanCatalog,
    understanding: QueryUnderstandingV3,
    floor: BaselineFloor,
    validation: QueryUnderstandingV3ExecutionValidation,
) -> tuple[QueryPlanV2, Mapping[str, tuple[str, ...]]]:
    if not validation.accepted:
        raise ValueError("only accepted V3 validation may be compiled")

    used_ids: set[str] = set()
    requirements: list[AnswerRequirementV2] = []
    provenance: dict[str, tuple[str, ...]] = {}
    # The bridge is scope-local because a category mapping can legitimately
    # differ between product/version partitions.  Its source/rendered
    # description remains exactly one catalog qualifier.
    bridge_by_key: dict[tuple[str, str], str] = {}
    answer_rows: list[tuple[QueryUnderstandingV3Candidate, ApplicabilityScope, str]] = []

    for candidate in understanding.answer_candidates:
        source_span_ids = _answer_description_span_ids(candidate)
        description = _description_from_catalog(catalog, source_span_ids)
        scopes = _bound_scopes(validation, candidate_id=candidate.id, floor=floor)
        for scope_index, scope in enumerate(scopes, start=1):
            candidate_suffix = "" if len(scopes) == 1 else f"_s{scope_index}"
            requirement_id = f"{candidate.id}{candidate_suffix}"
            if requirement_id in used_ids:
                requirement_id = _next_requirement_id("a", used_ids)
            else:
                used_ids.add(requirement_id)
            answer_rows.append((candidate, scope, requirement_id))
            provenance[requirement_id] = source_span_ids
            coverage_mode, coverage_contract = _answer_coverage(
                question=floor.current_question,
                candidate_count=len(understanding.answer_candidates),
            )
            requirements.append(AnswerRequirementV2(
                id=requirement_id,
                description=description,
                role="answer",
                importance="required",
                source="explicit",
                coverage_mode=coverage_mode,  # type: ignore[arg-type]
                coverage_contract=coverage_contract,  # type: ignore[arg-type]
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
                applicability_scope=scope,
            ))

    by_id = {item.id: item for item in requirements}
    for candidate, scope, answer_id in answer_rows:
        subject = _classification_subject(
            candidate,
            question=floor.current_question,
        )
        if subject is None:
            continue
        bridge_key = (subject.span_id, scope.fingerprint)
        bridge_id = bridge_by_key.get(bridge_key)
        if bridge_id is None:
            bridge_id = _next_requirement_id("b", used_ids)
            bridge_by_key[bridge_key] = bridge_id
            requirements.append(AnswerRequirementV2(
                id=bridge_id,
                description=_description_from_catalog(catalog, (subject.span_id,)),
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject=subject.text,
                bridge_kind="classification",
                applicability_scope=scope,
            ))
            provenance[bridge_id] = (subject.span_id,)
        current = by_id[answer_id]
        updated = AnswerRequirementV2(
            id=current.id,
            description=current.description,
            role=current.role,
            importance=current.importance,
            source=current.source,
            coverage_mode=current.coverage_mode,
            coverage_contract=current.coverage_contract,
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(bridge_id,),
            applicability_scope=current.applicability_scope,
        )
        requirements[requirements.index(current)] = updated
        by_id[answer_id] = updated

    if len(requirements) > _MAX_REQUIREMENTS:
        # Validation performs the same deterministic preflight.  Keep this
        # guard to prevent future refactors adding work after validation.
        raise ValueError("V3 compiler exceeded QueryPlanV2 requirement capacity")
    plan = QueryPlanV2(
        original_query=floor.current_question,
        answer_shape=_answer_shape(
            question=floor.current_question,
            candidate_count=len(understanding.answer_candidates),
            floor=floor,
        ),
        # The task graph owns executable task queries.  Keeping the immutable
        # original question as the only flat recall query both preserves the
        # anchor and avoids a second, truncation-prone retrieval authority.
        retrieval_queries=(floor.current_question,),
        requirements=tuple(requirements),
        confidence=0.9,
        source="model",
        reason="query_understanding_v3_catalog_compiled",
    )
    return plan, MappingProxyType(dict(provenance))


def compile_query_understanding(
    *,
    catalog: SourceSpanCatalog,
    understanding: QueryUnderstandingV3,
    baseline_floor: BaselineFloor,
) -> CompiledQueryUnderstanding:
    """Compile V3 candidates or atomically return the deterministic fallback.

    The fallback is used for malformed/partial/unsafe candidate semantics and
    hard guards.  A *valid* V3 candidate is compiled independently from a
    fallback planner's runnable state, which is the important V3 change over
    treating legacy query analysis as an unreducible authority.
    """

    validation = validate_query_understanding(
        catalog=catalog,
        understanding=understanding,
        baseline_floor=baseline_floor,
    )
    if not validation.accepted:
        fallback_bundle = compile_rag_execution_bundle(baseline_floor.fallback_plan)
        return CompiledQueryUnderstanding(
            plan=baseline_floor.fallback_plan,
            execution_bundle=fallback_bundle,
            validation=validation,
            compiler_decision="fallback",
            description_span_ids={},
        )

    plan, provenance = _compiled_plan(
        catalog=catalog,
        understanding=understanding,
        floor=baseline_floor,
        validation=validation,
    )
    bundle = compile_rag_execution_bundle(plan)
    if bundle.mode != "ledgered" or bundle.task_graph is None:
        raise ValueError("accepted V3 candidate did not produce a ledgered bundle")
    anchor = bundle.task_graph.task_by_id.get("anchor_root")
    if anchor is None or anchor.query != baseline_floor.current_question:
        raise ValueError("V3 compiler lost the immutable original-question anchor")
    return CompiledQueryUnderstanding(
        plan=plan,
        execution_bundle=bundle,
        validation=validation,
        compiler_decision="compiled",
        description_span_ids=provenance,
    )


__all__ = [
    "BaselineFloor",
    "CompiledQueryUnderstanding",
    "CompilerDecision",
    "QueryUnderstandingV3ExecutionValidation",
    "ScopeBinding",
    "compile_query_understanding",
    "validate_query_understanding",
]
