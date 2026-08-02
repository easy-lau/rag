"""Compile a validated ``query_analysis.v2`` candidate graph safely.

The model never emits a query plan.  It may identify literal targets in the
current user sentence; this module uses only trusted grammar, route-merged
baseline scope and typed bridge policy to turn a complete candidate set into a
constrained :class:`QueryPlanV2`.  The original direct question is always
retained as ``anchor_root`` recall in the resulting task graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from core.query_analysis_contract import QueryAnalysis, QueryAnalysisAnswerCandidate
from core.query_semantics import (
    ResolvedAnswerUnit,
    ResolvedTurnSemantics,
    resolve_turn_semantics,
)
from core.query_analysis_validation import (
    QueryAnalysisExecutionValidation,
    query_analysis_fingerprint,
    query_input_fingerprint,
    query_plan_fingerprint,
)
from core.query_constraints import ApplicabilityScope, extract_query_constraints
from core.query_surface_structure import (
    is_elliptical_current_clause_target,
    parse_query_surface_frame,
)
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.query_plan import (
    _answer_coverage_contract,
    _answer_coverage_mode,
    infer_implicit_bridge,
)
from core.rag_v2.task_graph import (
    RagExecutionBundle,
    RetrievalTaskGraph,
    compile_rag_execution_bundle,
)


CompilerDecision = Literal[
    "generic_baseline_replaced",
    "baseline_canonicalized",
    "baseline_canonicalized_augmented",
    "baseline_augmented",
    "baseline_preserved",
]


@dataclass(frozen=True)
class CompiledQueryAnalysisPlan:
    """Auditable result of trusted candidate compilation."""

    plan: QueryPlanV2
    task_graph: RetrievalTaskGraph
    execution_bundle: RagExecutionBundle
    compiler_decision: CompilerDecision
    baseline_fingerprint: str
    applied_plan_fingerprint: str
    analysis_fingerprint: str
    analysis_answer_candidate_ids: tuple[str, ...]
    analysis_bridge_candidate_ids: tuple[str, ...]
    baseline_anchor_preserved: bool
    semantics: ResolvedTurnSemantics

    def safe_summary(self) -> dict[str, object]:
        return {
            "compiler_decision": self.compiler_decision,
            "plan_schema_version": self.plan.schema_version,
            "plan_source": self.plan.source,
            "answer_shape": self.plan.answer_shape,
            "answer_requirement_count": sum(
                item.is_required_answer for item in self.plan.requirements
            ),
            "bridge_requirement_count": sum(
                item.role == "bridge" for item in self.plan.requirements
            ),
            "analysis_answer_candidate_count": len(self.analysis_answer_candidate_ids),
            "analysis_bridge_candidate_count": len(self.analysis_bridge_candidate_ids),
            "baseline_fingerprint": self.baseline_fingerprint,
            "applied_plan_fingerprint": self.applied_plan_fingerprint,
            "analysis_fingerprint": self.analysis_fingerprint,
            "baseline_anchor_preserved": self.baseline_anchor_preserved,
            "semantics": self.semantics.safe_summary(),
            "task_graph": self.task_graph.safe_summary(),
            "execution_bundle": self.execution_bundle.safe_summary(),
        }


def _normalized(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _unique_texts(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return tuple(result)


def _scope_from_baseline(
    baseline_plan: QueryPlanV2,
    *,
    current_question: str,
) -> ApplicabilityScope | None:
    """Return the one baseline-owned scope eligible for a new task.

    A model candidate has no authority to merge scopes.  When a deterministic
    baseline already contains several sides of a comparison, callers receive
    ``None`` and must preserve those requirements (or split work by their
    existing owner) instead of assigning one arbitrary product/version to all
    candidates.
    """

    values = {
        item.applicability_scope.fingerprint: item.applicability_scope
        for item in baseline_plan.requirements
        if item.is_required_answer and item.applicability_scope is not None
    }
    constrained = {
        fingerprint: scope
        for fingerprint, scope in values.items()
        if scope.has_scope_constraint
    }
    if len(constrained) == 1:
        return next(iter(constrained.values()))
    if len(constrained) > 1:
        return None
    # There is no route-owned constraint in the baseline.  The only allowed
    # fallback is a deterministic parse of the exact current question, never
    # a model field or an analysis paraphrase.
    return extract_query_constraints(current_question)


def _coverage_question(
    *,
    description: str,
    current_question: str,
) -> str:
    """Attach only trusted grammatical question form for coverage analysis."""

    frame = parse_query_surface_frame(current_question)
    source = str(current_question or "")
    if re.search(r"(?:多少|几(?:个|项|种|次|天|月|年)?)", source):
        return f"{description}是多少"
    if frame is not None and frame.question_operator == "enumeration":
        return f"{description}有哪些"
    if frame is not None and frame.question_operator == "value":
        return f"{description}是什么"
    return description


def _candidate_description(
    candidate: QueryAnalysisAnswerCandidate | ResolvedAnswerUnit,
    *,
    scope: ApplicabilityScope | None,
) -> str:
    """Build retrieval wording from literal candidates and trusted scope only."""

    qualifier_values = [item.span for item in candidate.qualifier_source_refs]
    prefix = _unique_texts([
        *(
            value
            for value in (
                scope.project if scope and scope.has_project_constraint else None,
                scope.product if scope else None,
                scope.version if scope else None,
            )
            if value
        ),
        *qualifier_values,
    ])
    target = re.sub(r"\s+", " ", candidate.target_source_ref.span).strip()
    if not prefix:
        return target
    return " ".join((*prefix, target))


def _candidate_requirement(
    candidate: QueryAnalysisAnswerCandidate | ResolvedAnswerUnit,
    *,
    requirement_id: str,
    current_question: str,
    scope: ApplicabilityScope,
    answer_shape: str,
) -> AnswerRequirementV2:
    description = _candidate_description(
        candidate,
        scope=scope,
    )
    coverage_source = _coverage_question(
        description=description,
        current_question=current_question,
    )
    coverage_mode = _answer_coverage_mode(
        coverage_source,
        answer_shape=answer_shape,
    )
    return AnswerRequirementV2(
        id=requirement_id,
        description=description,
        role="answer",
        importance="required",
        source="explicit",
        coverage_mode=coverage_mode,
        coverage_contract=_answer_coverage_contract(
            coverage_source,
            answer_shape=answer_shape,
            coverage_mode=coverage_mode,
        ),
        depends_on_requirement_ids=(),
        augmentation_requirement_ids=(),
        applicability_scope=scope,
    )


def _candidate_to_baseline_answer_ids(
    analysis: QueryAnalysis,
    baseline_plan: QueryPlanV2,
) -> dict[str, str]:
    """Map literal candidate targets to existing required answers only."""

    result: dict[str, str] = {}
    answers = tuple(
        item for item in baseline_plan.requirements if item.is_required_answer
    )
    for candidate in analysis.answer_candidates:
        target = _normalized(candidate.target_source_ref.span)
        matches = [
            item.id
            for item in answers
            if target and target in _normalized(item.description)
        ]
        if len(matches) == 1:
            result[candidate.id] = matches[0]
    return result


_COMPACT_CJK_TERM_RE = re.compile(r"^[\u3400-\u9fff]+$")


def _source_answer_head(value: object) -> str:
    """Return a grammar-normalized answer head from one exact source span.

    The parser may remove only question-shell scaffolding such as ``需要提供
    哪些``.  It does not consult documents or aliases, so the result remains a
    source-derived lexical rendering rather than a model-invented term.
    """

    source = re.sub(r"\s+", " ", str(value or "")).strip()
    frame = parse_query_surface_frame(source)
    if frame is not None and frame.answer_target:
        return frame.answer_target
    return source


def _compact_source_terms(values: list[str]) -> str:
    """Render source terms as one CJK noun phrase where that is unambiguous."""

    terms = _unique_texts(values)
    if terms and all(_COMPACT_CJK_TERM_RE.fullmatch(value) for value in terms):
        return "".join(terms)
    return " ".join(terms)


def _current_turn_qualifier_values(
    candidate: QueryAnalysisAnswerCandidate | ResolvedAnswerUnit,
) -> tuple[str, ...]:
    """Keep only literal current-turn qualifiers for baseline normalization.

    Explicit multi-answer baseline wording may be refined by a sibling clause
    from the *same* current input.  Historical bindings are executed through
    ``ResolvedTurnSemantics`` but never rewrite an already explicit baseline.
    """

    return _unique_texts([
        item.span
        for item in candidate.qualifier_source_refs
        if item.turn_key == "current"
    ])


def _canonicalized_baseline_description(
    *,
    candidate: ResolvedAnswerUnit,
    baseline_requirement: AnswerRequirementV2,
    current_question: str,
) -> str | None:
    """Use a literal missing current-turn qualifier to close one short clause.

    Example: the second clause in ``报销提交时限是多久？需要提供哪些凭证？``
    can cite ``报销`` from the first current clause.  Its original baseline
    description lacks that literal entirely; rendering ``报销凭证`` is thus a
    deterministic composition of exact user spans, not terminology lookup.
    If every candidate qualifier already appears in the baseline text, the
    original wording remains authoritative and no cosmetic rewrite occurs.
    """

    qualifiers = _current_turn_qualifier_values(candidate)
    if not qualifiers:
        return None
    current_qualifier_ranges = tuple(
        (item.start, item.end)
        for item in candidate.qualifier_source_refs
        if item.turn_key == "current"
    )
    if not is_elliptical_current_clause_target(
        current_question,
        target_range=(
            candidate.target_source_ref.start,
            candidate.target_source_ref.end,
        ),
        qualifier_ranges=current_qualifier_ranges,
    ):
        return None
    baseline_text = _normalized(baseline_requirement.description)
    if all(_normalized(value) in baseline_text for value in qualifiers):
        return None
    target = _source_answer_head(candidate.target_source_ref.span)
    if not target:
        return None
    rendered = _compact_source_terms([*qualifiers, target])
    return rendered or None


def _canonicalize_explicit_baseline(
    *,
    analysis: QueryAnalysis,
    semantics: ResolvedTurnSemantics,
    baseline_plan: QueryPlanV2,
    current_question: str,
) -> tuple[QueryPlanV2, bool]:
    """Refine lexical descriptions while retaining the full explicit plan.

    This is intentionally not a task replacement.  Requirement identifiers,
    answer count, coverage contract, scopes, bridge edges and dependency IDs
    are all copied from the deterministic baseline unchanged.  The original
    raw retrieval queries are retained as recall anchors alongside any newly
    composed current-turn source phrase.
    """

    candidate_ids = _candidate_to_baseline_answer_ids(analysis, baseline_plan)
    baseline_answers = tuple(
        item for item in baseline_plan.requirements if item.is_required_answer
    )
    if (
        len(candidate_ids) != len(semantics.answer_units)
        or set(candidate_ids.values()) != {item.id for item in baseline_answers}
    ):
        raise ValueError("canonical baseline normalization requires a full bijection")

    candidates_by_id = {item.id: item for item in semantics.answer_units}
    requirements: list[AnswerRequirementV2] = []
    changed = False
    for requirement in baseline_plan.requirements:
        candidate_id = next(
            (
                key for key, value in candidate_ids.items()
                if value == requirement.id
            ),
            None,
        )
        if candidate_id is None:
            requirements.append(requirement)
            continue
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError("canonical baseline candidate is unavailable")
        description = _canonicalized_baseline_description(
            candidate=candidate,
            baseline_requirement=requirement,
            current_question=current_question,
        )
        if description is None or _normalized(description) == _normalized(
            requirement.description
        ):
            requirements.append(requirement)
            continue
        requirements.append(replace(requirement, description=description))
        changed = True

    if not changed:
        return baseline_plan, False
    answer_descriptions = [
        item.description for item in requirements if item.role == "answer"
    ]
    return (
        replace(
            baseline_plan,
            retrieval_queries=_unique_texts([
                *baseline_plan.retrieval_queries,
                *answer_descriptions,
            ]),
            requirements=tuple(requirements),
            source="model",
            reason=(
                f"{baseline_plan.reason}; "
                "source_anchored_current_turn_descriptor_normalization"
            )[:500],
        ),
        True,
    )


def _next_id(prefix: str, used: set[str], index: int) -> str:
    value = f"{prefix}{index}"
    while value in used:
        index += 1
        value = f"{prefix}{index}"
    used.add(value)
    return value


def _attach_candidate_augmentations(
    *,
    analysis: QueryAnalysis,
    requirements: list[AnswerRequirementV2],
    candidate_answer_requirement_ids: dict[str, str],
) -> tuple[list[AnswerRequirementV2], bool]:
    """Create only trusted optional classification augmentation bridges.

    Validation has already proved that each candidate bridge is shared by
    source-anchored answers and that its subject is a non-named entity
    qualifier.  This function still asks the existing trusted bridge policy to
    infer a classification bridge from literal wording.  It never constructs
    a proof edge and skips an answer with any existing bridge semantics.
    """

    used_ids = {item.id for item in requirements}
    by_id = {item.id: item for item in requirements}
    answer_by_candidate = {
        candidate.id: candidate
        for candidate in analysis.answer_candidates
    }
    changed = False
    bridge_index = 1
    for bridge in analysis.bridge_candidates:
        target_answer_ids = [
            candidate_answer_requirement_ids[candidate.id]
            for candidate in analysis.answer_candidates
            if bridge.id in candidate.bridge_candidate_ids
            and candidate.id in candidate_answer_requirement_ids
        ]
        target_answer_ids = list(dict.fromkeys(target_answer_ids))
        if len(target_answer_ids) < 2:
            continue
        # A model may observe a shared qualifier but may not merge scoped
        # answer paths.  Partition by the existing baseline-owned canonical
        # scope before constructing/reusing any optional bridge.
        target_ids_by_scope: dict[str, list[str]] = {}
        for answer_id in target_answer_ids:
            target_ids_by_scope.setdefault(
                by_id[answer_id].scope_fingerprint,
                [],
            ).append(answer_id)
        for scoped_target_ids in target_ids_by_scope.values():
            if any(
                by_id[answer_id].proof_bridge_requirement_ids
                or by_id[answer_id].augmentation_bridge_requirement_ids
                for answer_id in scoped_target_ids
            ):
                continue
            scope = by_id[scoped_target_ids[0]].applicability_scope
            if scope is None:
                continue
            subject = bridge.subject_source_ref.span
            first_candidate = next(
                answer_by_candidate[candidate_id]
                for candidate_id in answer_by_candidate
                if candidate_answer_requirement_ids.get(candidate_id)
                in scoped_target_ids
                and bridge.id
                in answer_by_candidate[candidate_id].bridge_candidate_ids
            )
            inferred = infer_implicit_bridge(
                f"{subject}的{first_candidate.target_source_ref.span}是多少"
            )
            if inferred is None or inferred.kind != "classification":
                continue
            existing_bridge_ids = [
                item.id
                for item in requirements
                if item.role == "bridge"
                and item.bridge_kind == "classification"
                and _normalized(item.bridge_subject) == _normalized(inferred.subject)
                and item.scope_fingerprint == scope.fingerprint
            ]
            if existing_bridge_ids:
                bridge_id = existing_bridge_ids[0]
            else:
                bridge_id = _next_id("qa_b", used_ids, bridge_index)
                bridge_index += 1
                requirements.append(AnswerRequirementV2(
                    id=bridge_id,
                    description=inferred.description,
                    role="bridge",
                    importance="helpful",
                    source="inferred",
                    bridge_subject=inferred.subject,
                    bridge_kind="classification",
                    applicability_scope=scope,
                ))
                by_id[bridge_id] = requirements[-1]
            for answer_id in scoped_target_ids:
                current = by_id[answer_id]
                augmentation_ids = tuple(dict.fromkeys(
                    (*current.augmentation_bridge_requirement_ids, bridge_id)
                ))
                updated = replace(
                    current,
                    augmentation_requirement_ids=augmentation_ids,
                )
                by_id[answer_id] = updated
                requirements[requirements.index(current)] = updated
                changed = True
    return requirements, changed


def _bundle_for_plan(
    *,
    plan: QueryPlanV2,
    baseline_execution_bundle: RagExecutionBundle | None,
) -> RagExecutionBundle:
    if (
        baseline_execution_bundle is not None
        and baseline_execution_bundle.plan == plan
    ):
        return baseline_execution_bundle
    bundle = compile_rag_execution_bundle(plan)
    if bundle.mode != "ledgered" or bundle.task_graph is None:
        raise ValueError("trusted query-analysis compilation must produce a ledgered bundle")
    return bundle


def _compiled_plan(
    *,
    analysis: QueryAnalysis,
    semantics: ResolvedTurnSemantics,
    validation: QueryAnalysisExecutionValidation,
    baseline_plan: QueryPlanV2,
    current_question: str,
) -> tuple[QueryPlanV2, CompilerDecision]:
    scope = _scope_from_baseline(
        baseline_plan,
        current_question=current_question,
    )
    if validation.replacement_authorized:
        if scope is None:
            raise ValueError(
                "candidate replacement cannot merge multiple baseline scopes"
            )
        generic_answers = [
            item for item in baseline_plan.requirements if item.is_required_answer
        ]
        if len(generic_answers) != 1:
            raise ValueError("generic candidate replacement requires exactly one baseline answer")
        generic_id = generic_answers[0].id
        requirements = [
            item for item in baseline_plan.requirements if item.id != generic_id
        ]
        used_ids = {item.id for item in requirements}
        candidate_answer_ids: dict[str, str] = {}
        answer_index = 1
        for candidate in sorted(
            semantics.answer_units,
            key=lambda item: item.target_source_ref.start,
        ):
            requirement_id = _next_id("qa_a", used_ids, answer_index)
            answer_index += 1
            candidate_answer_ids[candidate.id] = requirement_id
            requirements.insert(
                len([item for item in requirements if item.role == "answer"]),
                _candidate_requirement(
                    candidate,
                    requirement_id=requirement_id,
                    current_question=current_question,
                    scope=scope,
                    answer_shape=semantics.answer_shape,
                ),
            )
        if validation.augmentation_authorized:
            requirements, _ = _attach_candidate_augmentations(
                analysis=analysis,
                requirements=requirements,
                candidate_answer_requirement_ids=candidate_answer_ids,
            )
        answer_descriptions = [
            item.description for item in requirements if item.role == "answer"
        ]
        bridge_descriptions = [
            item.description for item in requirements if item.role == "bridge"
        ]
        return (
            QueryPlanV2(
                original_query=baseline_plan.original_query,
                answer_shape=semantics.answer_shape,
                retrieval_queries=_unique_texts([
                    baseline_plan.original_query,
                    *semantics.canonical_retrieval_queries,
                    *answer_descriptions,
                    *bridge_descriptions,
                ]),
                requirements=tuple(requirements),
                confidence=max(0.0, min(float(baseline_plan.confidence), 0.95)),
                source="model",
                reason=(
                    f"{baseline_plan.reason}; "
                    "trusted_query_analysis_v2_complete_candidate_compilation"
                )[:500],
            ),
            "generic_baseline_replaced",
        )

    working_plan = baseline_plan
    canonicalized = False
    if validation.canonicalization_authorized:
        working_plan, canonicalized = _canonicalize_explicit_baseline(
            analysis=analysis,
            semantics=semantics,
            baseline_plan=baseline_plan,
            current_question=current_question,
        )

    if not validation.augmentation_authorized:
        return (
            working_plan,
            "baseline_canonicalized" if canonicalized else "baseline_preserved",
        )
    requirements = list(working_plan.requirements)
    candidate_answer_ids = _candidate_to_baseline_answer_ids(analysis, working_plan)
    requirements, changed = _attach_candidate_augmentations(
        analysis=analysis,
        requirements=requirements,
        candidate_answer_requirement_ids=candidate_answer_ids,
    )
    if not changed:
        return (
            working_plan,
            "baseline_canonicalized" if canonicalized else "baseline_preserved",
        )
    answer_descriptions = [
        item.description for item in requirements if item.role == "answer"
    ]
    bridge_descriptions = [
        item.description for item in requirements if item.role == "bridge"
    ]
    return (
        replace(
            working_plan,
            retrieval_queries=_unique_texts([
                *working_plan.retrieval_queries,
                *answer_descriptions,
                *bridge_descriptions,
            ]),
            requirements=tuple(requirements),
            source="model",
            reason=(
                f"{working_plan.reason}; "
                "trusted_query_analysis_v2_optional_augmentation"
            )[:500],
        ),
        "baseline_canonicalized_augmented"
        if canonicalized
        else "baseline_augmented",
    )


def compile_query_analysis_plan(
    analysis: QueryAnalysis,
    *,
    execution_validation: QueryAnalysisExecutionValidation,
    baseline_plan: QueryPlanV2,
    current_question: str,
    baseline_execution_bundle: RagExecutionBundle | None = None,
) -> CompiledQueryAnalysisPlan:
    """Compile an accepted candidate graph without trusting model semantics."""

    if not isinstance(analysis, QueryAnalysis):
        raise ValueError("analysis must be a QueryAnalysis")
    if not isinstance(execution_validation, QueryAnalysisExecutionValidation):
        raise ValueError("execution_validation must be a QueryAnalysisExecutionValidation")
    if not isinstance(baseline_plan, QueryPlanV2):
        raise ValueError("baseline_plan must be a QueryPlanV2")
    if not execution_validation.accepted:
        raise ValueError("only accepted query analysis can be compiled")
    if execution_validation.analysis_fingerprint != query_analysis_fingerprint(analysis):
        raise ValueError("execution validation does not belong to this analysis")
    if execution_validation.baseline_fingerprint != query_plan_fingerprint(baseline_plan):
        raise ValueError("execution validation does not belong to this baseline plan")
    if execution_validation.input_fingerprint != query_input_fingerprint(current_question):
        raise ValueError("execution validation does not belong to this source input")

    semantics = resolve_turn_semantics(
        analysis,
        current_question=current_question,
    )
    plan, decision = _compiled_plan(
        analysis=analysis,
        semantics=semantics,
        validation=execution_validation,
        baseline_plan=baseline_plan,
        current_question=current_question,
    )
    bundle = _bundle_for_plan(
        plan=plan,
        baseline_execution_bundle=baseline_execution_bundle,
    )
    graph = bundle.task_graph
    if graph is None:
        raise ValueError("ledgered query-analysis bundle is missing its task graph")
    anchor = graph.task_by_id.get("anchor_root")
    if anchor is None or anchor.query != baseline_plan.original_query:
        raise ValueError("compiled query-analysis plan lost the baseline anchor query")
    return CompiledQueryAnalysisPlan(
        plan=plan,
        task_graph=graph,
        execution_bundle=bundle,
        compiler_decision=decision,
        baseline_fingerprint=query_plan_fingerprint(baseline_plan),
        applied_plan_fingerprint=query_plan_fingerprint(plan),
        analysis_fingerprint=query_analysis_fingerprint(analysis),
        analysis_answer_candidate_ids=tuple(
            item.id for item in analysis.answer_candidates
        ),
        analysis_bridge_candidate_ids=tuple(
            item.id for item in analysis.bridge_candidates
        ),
        baseline_anchor_preserved=True,
        semantics=semantics,
    )


__all__ = [
    "CompilerDecision",
    "CompiledQueryAnalysisPlan",
    "compile_query_analysis_plan",
]
