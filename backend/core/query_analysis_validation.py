"""Trusted validation of source-anchored ``query_analysis.v2`` candidates.

The model contract deliberately has no execution semantics.  This module is
the boundary between a syntactically valid candidate graph and the trusted
compiler: it proves that a multi-answer proposal is a complete current-turn
enumeration, preserves all route-authorised baseline work, and only permits a
candidate bridge to become a *possible* classification augmentation.  It can
never grant scope, coverage, proof edges, knowledge-base access or facts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterable, Literal

from core.query_analysis_contract import QueryAnalysis
from core.query_constraints import (
    extract_applicability_scopes,
    extract_query_constraints,
)
from core.query_surface_structure import (
    current_turn_candidate_targets_are_complete,
    is_entity_qualifier,
    is_stable_entity_qualifier,
)
from core.rag_v2.contracts import QueryPlanV2


CandidateCompilationMode = Literal[
    "replace_generic_baseline",
    "preserve_baseline",
    "rejected",
]

_NON_ANSWER_TARGET_RE = re.compile(
    r"^(?:什么|多少|哪些|如何|怎么|怎样|这些|那些|这个|那个|"
    r"标准|规定|政策|流程|内容|项目|问题|信息)$",
    re.IGNORECASE,
)
_VERSION_TOKEN_RE = re.compile(
    r"(?<![\d.])(?:\d{1,4}(?:\.\d{1,4})+|\d{4}\s*(?:版|版本))(?![\d.])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryAnalysisExecutionValidation:
    """Content-safe result bound to one analysis, baseline and source input."""

    accepted: bool
    reason: str
    mode: CandidateCompilationMode
    baseline_answer_count: int
    candidate_answer_count: int
    candidate_bridge_count: int
    baseline_fingerprint: str
    input_fingerprint: str
    analysis_fingerprint: str
    replacement_authorized: bool = False
    # A complete current-turn graph may refine lexical descriptions of an
    # already explicit multi-answer baseline (for example attach a literal
    # omitted subject from the preceding current clause).  It never changes
    # answer count, scope, coverage, dependencies or bridge semantics.
    canonicalization_authorized: bool = False
    augmentation_authorized: bool = False
    baseline_required_answer_ids: tuple[str, ...] = ()
    candidate_answer_ids: tuple[str, ...] = ()

    def safe_summary(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "mode": self.mode,
            "baseline_answer_count": self.baseline_answer_count,
            "candidate_answer_count": self.candidate_answer_count,
            "candidate_bridge_count": self.candidate_bridge_count,
            "baseline_fingerprint": self.baseline_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "analysis_fingerprint": self.analysis_fingerprint,
            "replacement_authorized": self.replacement_authorized,
            "canonicalization_authorized": self.canonicalization_authorized,
            "augmentation_authorized": self.augmentation_authorized,
            "baseline_required_answer_count": len(self.baseline_required_answer_ids),
        }


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_analysis_fingerprint(analysis: QueryAnalysis) -> str:
    """Fingerprint the immutable model candidate graph."""

    if not isinstance(analysis, QueryAnalysis):
        raise ValueError("analysis must be a QueryAnalysis")
    return _fingerprint(analysis.to_dict())


def query_plan_fingerprint(plan: QueryPlanV2) -> str:
    """Fingerprint every execution-semantic field of the trusted plan."""

    if not isinstance(plan, QueryPlanV2):
        raise ValueError("plan must be a QueryPlanV2")
    return _fingerprint(plan.to_dict())


def query_input_fingerprint(question: str) -> str:
    """Bind validation to the exact source text used for offset checks."""

    if not isinstance(question, str):
        raise ValueError("current_question must be a string")
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _normalized(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _answer_like_target(value: str) -> bool:
    target = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;：:！？?!")
    if len(target) < 2 or len(target) > 160:
        return False
    if _NON_ANSWER_TARGET_RE.fullmatch(target):
        return False
    return bool(re.search(r"[\u3400-\u9fffA-Za-z0-9]", target))


def _ranges_overlap(
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _candidate_ranges_are_well_formed(analysis: QueryAnalysis) -> bool:
    targets = tuple(
        (item.target_source_ref.start, item.target_source_ref.end)
        for item in analysis.answer_candidates
    )
    if any(
        _ranges_overlap(left, right)
        for index, left in enumerate(targets)
        for right in targets[index + 1:]
    ):
        return False
    for answer in analysis.answer_candidates:
        if not _answer_like_target(answer.target_source_ref.span):
            return False
        target_range = (
            answer.target_source_ref.start,
            answer.target_source_ref.end,
        )
        for qualifier in answer.qualifier_source_refs:
            if qualifier.turn_key != "current":
                continue
            qualifier_range = (qualifier.start, qualifier.end)
            if _ranges_overlap(target_range, qualifier_range):
                return False
    return True


def _scope_ranges(question: str) -> tuple[tuple[int, int], ...]:
    """Return source-authored route scope ranges without trusting the model."""

    scopes = extract_applicability_scopes(question)
    ranges = {
        (source.start, source.end)
        for scope in scopes
        for source in scope.source_spans
    }
    # The deterministic scope parser may retain a structural cue or field
    # label around the typed values (for example ``我使用的是 + 产品 + 版本``).
    # Candidate coverage must ignore that complete parser-owned span as
    # grammatical scope scaffolding; otherwise a valid multi-target analysis
    # is rejected merely because it did not pretend the cue was an answer
    # target.  Only an exact occurrence enclosing every typed source span is
    # trusted, so this cannot enlarge scope from model text or document data.
    for scope in scopes:
        matched_text = str(scope.matched_text or "")
        sources = tuple(scope.source_spans)
        if not matched_text or not sources:
            continue
        start = question.find(matched_text)
        while start >= 0:
            end = start + len(matched_text)
            if all(start <= source.start < source.end <= end for source in sources):
                ranges.add((start, end))
                break
            start = question.find(matched_text, start + 1)
    if ranges:
        return tuple(sorted(ranges))
    # Product-only legacy constraints do not appear in the multi-scope
    # enumerator.  Preserve its one deterministic source span without asking
    # the model to recover it.
    constraint = extract_query_constraints(question)
    return tuple(sorted({
        (source.start, source.end)
        for source in constraint.source_spans
    }))


def _has_multiple_explicit_version_tokens(question: str) -> bool:
    values = {
        re.sub(r"\s+", "", match.group(0)).casefold()
        for match in _VERSION_TOKEN_RE.finditer(question)
    }
    return len(values) > 1


def _baseline_answers(plan: QueryPlanV2):
    return tuple(item for item in plan.requirements if item.is_required_answer)


def _baseline_is_generic_direct(
    plan: QueryPlanV2,
    *,
    current_question: str,
) -> bool:
    """Whether the one baseline answer is only the original direct recall.

    A candidate may replace this generic requirement after full source-frame
    coverage, while ``anchor_root`` retains the original query.  Explicit
    answers or any existing bridge topology are never replaced here.
    """

    answers = _baseline_answers(plan)
    if len(answers) != 1:
        return False
    if any(item.role == "bridge" for item in plan.requirements):
        return False
    answer = answers[0]
    if _normalized(answer.description) == _normalized(current_question):
        return True
    return bool(
        plan.answer_shape == "unknown"
        or "route_authorized_single_fact_baseline" in str(plan.reason)
    )


def _candidate_matches_baseline_answer(candidate_target: str, description: str) -> bool:
    target = _normalized(candidate_target)
    rendered = _normalized(description)
    return bool(target and len(target) >= 2 and target in rendered)


def _preserves_explicit_multi_answer_floor(
    analysis: QueryAnalysis,
    plan: QueryPlanV2,
) -> bool:
    answers = _baseline_answers(plan)
    if len(answers) <= 1:
        return True
    if len(analysis.answer_candidates) != len(answers):
        return False
    unmatched = list(analysis.answer_candidates)
    for answer in answers:
        matches = [
            item
            for item in unmatched
            if _candidate_matches_baseline_answer(
                item.target_source_ref.span,
                answer.description,
            )
        ]
        if len(matches) != 1:
            return False
        unmatched.remove(matches[0])
    return not unmatched


def _bridge_policy_is_safe(
    analysis: QueryAnalysis,
    *,
    current_question: str,
) -> tuple[bool, bool, str | None]:
    """Validate only optional candidate-bridge eligibility.

    The return value never conveys a bridge kind or edge mode.  A later
    trusted compiler may independently choose an optional classification
    augmentation; explicit mapping/proof semantics remain owned by the
    baseline planner.
    """

    if not analysis.bridge_candidates:
        return True, False, None
    # Validate against the same shared grammatical entity classifier used by
    # the frame.  A leading explicit product/version can make the conservative
    # whole-sentence frame retain ``产品甲8.6普通员工`` as one qualifier, while
    # the source-anchored candidate correctly points at the literal subspan
    # ``普通员工``.  Requiring textual equality with that whole frame would
    # reject a safe scope-preserving case; the parser-owned classifier keeps
    # the rule structural without accepting arbitrary phrases.
    candidate_by_bridge = {
        item.id: [
            answer
            for answer in analysis.answer_candidates
            if item.id in answer.bridge_candidate_ids
        ]
        for item in analysis.bridge_candidates
    }
    augmentation_allowed = False
    for bridge in analysis.bridge_candidates:
        linked_answers = candidate_by_bridge[bridge.id]
        # A one-answer bridge cannot improve a multi-target decomposition and
        # is most often an attempt to manufacture a hidden dependency.  The
        # direct baseline remains sufficient for that case.
        if len(linked_answers) < 2:
            return False, False, "bridge_not_shared_by_multiple_answers"
        subject = bridge.subject_source_ref
        if subject.turn_key != "current":
            # Historical context may still support the preserved baseline,
            # but v2 does not create a new bridge from history alone.
            continue
        if not is_entity_qualifier(subject.span):
            return False, False, "bridge_subject_not_current_entity_qualifier"
        if is_stable_entity_qualifier(subject.span):
            return False, False, "named_entity_cannot_create_classification_augmentation"
        augmentation_allowed = True
    return True, augmentation_allowed, None


def validate_query_analysis_for_execution(
    analysis: QueryAnalysis,
    *,
    baseline_plan: QueryPlanV2,
    current_question: str,
    deterministic_is_followup: bool,
    allowed_context_turn_keys: Iterable[str] = (),
    minimum_confidence: float = 0.8,
) -> QueryAnalysisExecutionValidation:
    """Validate a model candidate graph before trusted compilation.

    ``baseline_plan`` is already route-contract merged.  This function never
    lets a model remove its explicit work, change scope/coverage, or turn an
    optional bridge into proof.  A rejected graph is an all-or-nothing
    baseline fallback rather than a partially adopted task list.
    """

    if not isinstance(analysis, QueryAnalysis):
        raise ValueError("analysis must be a QueryAnalysis")
    if not isinstance(baseline_plan, QueryPlanV2):
        raise ValueError("baseline_plan must be a QueryPlanV2")
    if not isinstance(current_question, str) or not current_question.strip():
        raise ValueError("current_question must be a non-empty string")

    answers = _baseline_answers(baseline_plan)
    base = {
        "baseline_answer_count": len(answers),
        "candidate_answer_count": len(analysis.answer_candidates),
        "candidate_bridge_count": len(analysis.bridge_candidates),
        "baseline_fingerprint": query_plan_fingerprint(baseline_plan),
        "input_fingerprint": query_input_fingerprint(current_question),
        "analysis_fingerprint": query_analysis_fingerprint(analysis),
        "baseline_required_answer_ids": tuple(item.id for item in answers),
        "candidate_answer_ids": tuple(item.id for item in analysis.answer_candidates),
    }

    def rejected(reason: str) -> QueryAnalysisExecutionValidation:
        return QueryAnalysisExecutionValidation(
            accepted=False,
            reason=reason,
            mode="rejected",
            **base,
        )

    allowed_context = {str(key).strip() for key in allowed_context_turn_keys}
    if not set(analysis.context_turn_keys).issubset(allowed_context):
        return rejected("context_key_not_allowed")
    # History ownership is decided by the source-anchored analysis contract,
    # not by the old regex follow-up flag.  The flag remains a diagnostic
    # baseline/fallback signal, but refusing a valid ``餐补呢`` -> ``t1:普通
    # 员工`` binding merely because a local rule did not recognize the short
    # ellipsis would bring back the original text-concatenation failure.
    # ``parse_query_analysis`` has already proved that every selected key is
    # referenced by an exact historical qualifier and that a self-contained
    # turn cannot cite history.  The model still cannot choose a KB, scope,
    # fact, coverage rule or bridge edge here.
    del deterministic_is_followup
    if baseline_plan.needs_clarification:
        return rejected("baseline_clarification_cannot_be_overridden")
    if analysis.confidence < max(0.0, min(float(minimum_confidence), 1.0)):
        return rejected("analysis_confidence_below_execution_floor")
    if not _candidate_ranges_are_well_formed(analysis):
        return rejected("candidate_source_ranges_not_answer_like")

    candidate_count = len(analysis.answer_candidates)
    if candidate_count > 1:
        target_ranges = tuple(
            (item.target_source_ref.start, item.target_source_ref.end)
            for item in analysis.answer_candidates
        )
        qualifier_ranges = tuple(
            (item.start, item.end)
            for answer in analysis.answer_candidates
            for item in answer.qualifier_source_refs
            if item.turn_key == "current"
        )
        if not current_turn_candidate_targets_are_complete(
            current_question,
            target_ranges=target_ranges,
            qualifier_ranges=qualifier_ranges,
            trusted_ranges=_scope_ranges(current_question),
        ):
            return rejected("candidate_current_turn_coverage_incomplete")
        if not _preserves_explicit_multi_answer_floor(analysis, baseline_plan):
            if len(answers) > 1:
                return rejected("explicit_baseline_answer_floor_not_preserved")
        if _has_multiple_explicit_version_tokens(current_question):
            # The deterministic planner now has a multi-scope comparison
            # compiler.  Only a baseline that already owns a separate
            # source-anchored scope for every explicit side may be preserved
            # or lexically normalized.  A generic one-answer baseline still
            # cannot be distributed by a model candidate.
            scoped_answers = [
                item
                for item in answers
                if item.applicability_scope is not None
                and item.applicability_scope.has_version_constraint
            ]
            distinct_scopes = {
                item.scope_fingerprint for item in scoped_answers
            }
            if len(scoped_answers) < 2 or len(distinct_scopes) < 2:
                return rejected("multiple_explicit_scope_values_require_baseline")

    bridge_valid, augmentation_authorized, bridge_error = _bridge_policy_is_safe(
        analysis,
        current_question=current_question,
    )
    if not bridge_valid:
        return rejected(bridge_error or "candidate_bridge_policy_rejected")

    # A single target only replaces a generic baseline when it is genuinely
    # contextual.  For a self-contained single question the deterministic
    # current-turn planner already owns the same target, so allowing a model
    # rewrite would create a second semantic authority without improving
    # recall.  Multi-target decomposition remains an authorized replacement.
    replacement_authorized = bool(
        (candidate_count > 1 or not analysis.self_contained)
        and _baseline_is_generic_direct(
            baseline_plan,
            current_question=current_question,
        )
    )
    if candidate_count > 1 and len(answers) == 1 and not replacement_authorized:
        return rejected("explicit_baseline_answer_cannot_be_replaced")

    # This is deliberately narrower than replacement.  The explicit answer
    # floor remains intact, but every source-anchored candidate has a unique
    # baseline owner and may carry a literal current-turn qualifier omitted by
    # that owner's short clause.  The compiler is allowed to normalize only
    # that lexical description; all evidence/coverage/bridge controls remain
    # baseline-owned.
    canonicalization_authorized = bool(
        candidate_count > 1
        and len(answers) > 1
        and candidate_count == len(answers)
        and _preserves_explicit_multi_answer_floor(analysis, baseline_plan)
    )

    if replacement_authorized:
        reason = "complete_current_turn_candidates_replace_generic_baseline"
        mode: CandidateCompilationMode = "replace_generic_baseline"
    elif canonicalization_authorized:
        reason = "complete_current_turn_candidates_may_normalize_explicit_baseline"
        mode = "preserve_baseline"
    elif augmentation_authorized:
        reason = "candidate_bridge_may_augment_preserved_baseline"
        mode = "preserve_baseline"
    else:
        reason = "candidate_observed_baseline_preserved"
        mode = "preserve_baseline"
    return QueryAnalysisExecutionValidation(
        accepted=True,
        reason=reason,
        mode=mode,
        replacement_authorized=replacement_authorized,
        canonicalization_authorized=canonicalization_authorized,
        augmentation_authorized=augmentation_authorized,
        **base,
    )


__all__ = [
    "CandidateCompilationMode",
    "QueryAnalysisExecutionValidation",
    "query_analysis_fingerprint",
    "query_input_fingerprint",
    "query_plan_fingerprint",
    "validate_query_analysis_for_execution",
]
