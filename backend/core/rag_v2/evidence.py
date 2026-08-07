"""Deterministic evidence assembly for the incremental RAG v2 path.

This module deliberately does not call a model or a retriever.  It receives an
already-authorized candidate pool, removes only candidates that are known to be
unsafe for the requested scope, and builds a bounded :class:`EvidenceBundle`.
Soft rerank/expansion failures lower confidence without pretending that the
retrieval candidate pool disappeared.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Literal, Mapping, Sequence

from core.query_constraints import (
    QueryConstraints,
    evaluate_candidate_constraints,
    extract_document_constraint_identity,
)
from core.rag_v2.bridge_resolution import (
    adjudicate_answer_claims,
    answer_target_terms,
    bridge_dependency_ids_for_answer,
    bridge_requirement_ids_for_answer,
    bridge_subject_for_requirement,
    candidate_supports_resolved_answer_set,
    content_contains_positive_subject,
    content_matches_complete_answer_target,
    content_matches_answer_target,
    extract_bridge_subject,
    extract_bridge_values,
    partition_bridge_facts,
    resolve_bridge_facts,
)
from core.rag_v2.collection_proofs import (
    collection_target_matches,
    has_explicit_collection_closure,
    table_matches_collection_target,
)
from core.rag_v2.context import (
    EvidenceContext,
    build_evidence_context,
    evidence_context_char_cost,
    order_evidence_context_items,
)
from core.rag_v2.evidence_snapshots import (
    complete_document_keys as complete_snapshot_document_keys,
    complete_table_keys as complete_snapshot_table_keys,
    table_key as snapshot_table_key,
)
from core.rag_v2.contracts import (
    AnswerRequirementV2,
    BridgeClaimBinding,
    CLAIM_RESULT_KINDS,
    EvidenceClaim,
    EvidenceCoverageAssessment,
    EvidenceBundle,
    EvidenceItem,
    EvidenceState,
    validate_answer_requirement_graph,
)
from core.rag_v2.task_graph import AnswerBridgePath, RetrievalTaskGraph
from core.rag_v2.task_execution import TaskExecutionLedger
from core.terminology_runtime import TerminologyRuntimeResolution


EvidenceCompletenessValue = Literal["complete", "partial", "unknown"]

DEFAULT_CONTEXT_MAX_CHUNKS = 16
DEFAULT_CONTEXT_MAX_CHARS = 16_000
RELATED_EVIDENCE_REASON = "related_evidence_admitted"
LEGACY_UNVERIFIED_REASON = "unverified_candidates_after_adjudication_failure"


@dataclass(frozen=True)
class FinalizedVisibleEvidence:
    """The one immutable evidence artifact consumed by generation.

    It couples the exact rendered context with the graph and assessment that
    certified it.  Returning these values separately is how a renderer budget
    can accidentally create a second evidence truth after verification.
    """

    bundle: EvidenceBundle
    context: EvidenceContext
    assessment: EvidenceCoverageAssessment | None
    closed_answer_claim_ids: tuple[str, ...] = ()
    answer_claim_item_ids: tuple[str, ...] = ()
    route_item_ids: tuple[str, ...] = ()
    generation_allowed: bool = False
    unverified_generation_allowed: bool = False
    renderer_dropped_item_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, EvidenceBundle):
            raise ValueError("finalized evidence bundle must be an EvidenceBundle")
        if not isinstance(self.context, EvidenceContext):
            raise ValueError("finalized evidence context must be an EvidenceContext")
        if self.assessment is not None:
            if not isinstance(self.assessment, EvidenceCoverageAssessment):
                raise ValueError(
                    "finalized evidence assessment must be an EvidenceCoverageAssessment"
                )
            if self.bundle.coverage_assessment != self.assessment:
                raise ValueError(
                    "finalized evidence assessment must match its bundle"
                )
        elif self.bundle.coverage_assessment is not None:
            raise ValueError(
                "a bundle coverage assessment requires a finalized assessment"
            )
        if tuple(self.context.item_ids) != tuple(self.bundle.context_item_ids):
            raise ValueError(
                "finalized context item ids must match bundle context item ids"
            )
        if self.context.truncated:
            raise ValueError("finalized evidence context cannot be truncated")
        route_item_ids = tuple(self.route_item_ids)
        if route_item_ids != tuple(self.bundle.context_item_ids):
            raise ValueError(
                "finalized route item ids must match bundle context item ids"
            )
        source_ids = set(self.bundle.answer_source_ids)
        if not set(self.answer_claim_item_ids).issubset(source_ids):
            raise ValueError(
                "answer claim item ids must belong to final answer sources"
            )
        if self.unverified_generation_allowed:
            if (
                not self.generation_allowed
                or self.assessment is not None
                or self.answer_claim_item_ids
                or not self.bundle.answer_source_ids
                or self.bundle.state.confidence != "retrieved"
                or self.bundle.state.completeness != "unknown"
                or not any(
                    reason in {
                        RELATED_EVIDENCE_REASON,
                        LEGACY_UNVERIFIED_REASON,
                    }
                    or reason.startswith(f"{RELATED_EVIDENCE_REASON}:")
                    for reason in self.bundle.state.reasons
                )
            ):
                raise ValueError(
                    "unverified generation requires bounded retrieved candidates"
                )
        if (
            self.generation_allowed
            and not self.answer_claim_item_ids
            and not self.unverified_generation_allowed
        ):
            raise ValueError(
                "generation requires at least one closed answer claim item"
            )


@dataclass(frozen=True)
class UnverifiedCandidateBundleResult:
    """Content-free admission diagnostics plus the optional degraded bundle."""

    bundle: EvidenceBundle | None
    input_candidate_count: int
    converted_candidate_count: int
    selected_candidate_count: int
    exclusion_reason_counts: tuple[tuple[str, int], ...] = ()


_OVERVIEW_QUERY_RE = re.compile(
    r"(?:总则|适用范围|适用于谁|适用对象|术语(?:和|与)?定义|定义(?:是什么|为)?|"
    r"概述|概览|总体介绍|整体介绍|主要内容|全文|完整内容|overview|definition)",
    re.IGNORECASE,
)
_OVERVIEW_SECTION_RE = re.compile(
    r"^(?:\s*#{1,6}\s*)?(?:(?:第?[一二三四五六七八九十百0-9]+)[章节、.．]\s*)?"
    r"(?:总则|适用范围|适用对象|目的|背景|术语(?:和|与)?定义|定义)(?:\s|[：:。]|$)",
    re.IGNORECASE,
)
_MARKDOWN_TABLE_RE = re.compile(
    r"(?m)^\s*\|.+\|\s*$\n\s*\|\s*:?-{3,}",
)
_HTML_TABLE_RE = re.compile(r"<(?:table|thead|tbody|tr|td|th)\b", re.IGNORECASE)
_SPECIFIC_CLAUSE_RE = re.compile(
    r"(?:第[一二三四五六七八九十百0-9]+条|"
    r"\d+(?:\.\d+){1,4}\s+|"
    r"(?:不超过|不得|必须|应当|标准|上限|下限|步骤|流程|参数|配置项)|"
    r"\d+(?:\.\d+)?\s*(?:元|%|天|小时|分钟|个|次|公里|GB|MB))",
    re.IGNORECASE,
)
_ANSWER_BODY_EVIDENCE_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:%|元|天|小时|分钟|次|个|人|公里|gb|mb)?|"
    r"不超过|不少于|上限|下限|为|是|包括|采用|使用|执行|"
    r"必须|应当|应|须|需|不得|禁止|允许|支持|可以|仅限|"
    r"方法|步骤|参数|选项|命令|字段|开启|关闭|启用|禁用|"
    r"[：:=<>≤≥])",
    re.IGNORECASE,
)
_VALID_CONSTRAINT_STATUSES = {
    "exact",
    "compatible",
    "neutral",
    "unknown",
    "mismatch",
}
_VALID_EVIDENCE_ROLES = {
    "direct",
    "bridge",
    "complement",
    "background",
    "conflicting",
}
_COVERAGE_ROLES = {"direct", "bridge", "complement"}
_REQUIREMENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_COVERAGE_TEXT_RE = re.compile(r"[A-Za-z0-9_.+/-]{2,}|[\u3400-\u9fff]{2,}")
_GENERIC_COVERAGE_TERMS = {
    "什么",
    "多少",
    "查询",
    "确认",
    "标准",
    "信息",
    "内容",
    "对应",
    "需要",
    "如何",
    "怎么",
}
_COVERAGE_STOP_CHARS = frozenset("的是和与及为请查询需")
_COVERAGE_PART_SPLIT_RE = re.compile(
    r"(?:以及|并且|同时|的|和|与|及|、|[,，;；:/]|\s+)",
    re.IGNORECASE,
)


def _as_mapping(candidate: Mapping[str, Any] | EvidenceItem) -> dict[str, Any]:
    if isinstance(candidate, EvidenceItem):
        return candidate.to_dict()
    if not isinstance(candidate, Mapping):
        raise ValueError("evidence candidate must be a mapping or EvidenceItem")
    return dict(candidate)


def _string_id(value: Any) -> str:
    return str(value or "").strip()


def _chunk_index(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _origins(raw: Mapping[str, Any], *, extra: str | None = None) -> tuple[str, ...]:
    values: list[str] = []
    for field in ("origins", "candidate_origins"):
        raw_values = raw.get(field)
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        if isinstance(raw_values, (list, tuple, set)):
            values.extend(str(item or "").strip() for item in raw_values)
    for field in ("origin", "candidate_origin"):
        value = str(raw.get(field) or "").strip()
        if value:
            values.append(value)
    if extra:
        values.append(extra)
    return tuple(dict.fromkeys(value for value in values if value))[:12]


def _metadata(raw: Mapping[str, Any], *, section_kind: str) -> dict[str, Any]:
    source = raw.get("metadata")
    metadata = dict(source) if isinstance(source, Mapping) else {}
    for field in (
        "filename",
        "file_type",
        "source_url",
        "doc_tags",
        "retrieval_score",
        "evidence_role",
        "contribution_role",
        "rerank_status",
        "topic_relevance",
        "answer_support",
        "jointly_selected",
        "expansion_query_indexes",
        # V2 task-graph provenance.  Unlike query indexes, these identifiers
        # are stable logical owners and must survive every evidence stage.
        "retrieval_task_ids",
        "retrieval_task_id",
        "supports_requirement_ids",
        "role",
        # A small-document snapshot is admitted only after the retriever has
        # verified the database row count and character count.  Preserve that
        # proof through evidence assembly so exhaustive requirements can
        # distinguish a complete bounded document from an arbitrary snippet.
        "full_document_chunk_count",
        "full_document_char_count",
        # Structured parser identity is also required by downstream scope
        # slicing and collection diagnostics.  These fields are descriptive;
        # they never widen authorization or retrieval scope.
        "section_key",
        "section_path",
        "section_chunk_index",
        "heading",
        # Tables can span several chunks.  Keeping the parser's stable table
        # identity and part cardinality lets collection coverage prove that a
        # complete structured result is present instead of mistaking one
        # repeated-header fragment for the whole table.
        "table_id",
        "table_part_index",
        "table_part_count",
        "table_row_start",
        "table_row_end",
    ):
        value = raw.get(field)
        if value is not None and field not in metadata:
            metadata[field] = value
    # Document-root annotations are request-local proof products.  Retriever
    # input, a persisted chunk, or a previous request must never nominate its
    # own document as the governing policy for the current answer.  They are
    # re-created below only after a current-query root seed and title/topic
    # proof have both been verified.
    metadata.pop("document_root_answer_requirement_ids", None)
    metadata.pop("document_policy_root_requirement_ids", None)
    metadata["section_kind"] = section_kind
    return metadata


def _section_kind(content: str, metadata: Mapping[str, Any] | None) -> str:
    if isinstance(metadata, Mapping):
        declared = str(
            metadata.get("section_kind")
            or metadata.get("content_kind")
            or metadata.get("block_type")
            or ""
        ).strip().casefold()
        if declared in {"table", "specific", "overview", "other"}:
            return declared

    head = content[:300].strip()
    if _MARKDOWN_TABLE_RE.search(content) or _HTML_TABLE_RE.search(content):
        return "table"
    # Imported spreadsheet tables do not always retain a Markdown separator.
    pipe_rows = sum(1 for line in content.splitlines()[:20] if line.count("|") >= 2)
    if pipe_rows >= 2:
        return "table"
    if _OVERVIEW_SECTION_RE.search(head):
        return "overview"
    if _SPECIFIC_CLAUSE_RE.search(content):
        return "specific"
    return "other"


def _constraint_status(
    raw: Mapping[str, Any],
    constraints: QueryConstraints | None,
) -> str:
    declared = str(raw.get("constraint_status") or "").strip().casefold()
    if declared == "mismatch":
        return "mismatch"
    if constraints is not None:
        evaluated = evaluate_candidate_constraints(constraints, dict(raw)).status
        if evaluated in _VALID_CONSTRAINT_STATUSES:
            return evaluated
    if declared in _VALID_CONSTRAINT_STATUSES:
        return declared
    return "unknown"


def _requirement_constraints(
    requirement: AnswerRequirementV2,
) -> QueryConstraints | None:
    """Return the requirement's immutable applicability contract.

    Requirement scope is not a product/version display projection.  In
    particular, project boundaries and source provenance must survive into
    evidence mapping, otherwise two identical targets from different projects
    can be merged after retrieval had correctly kept them separate.
    """

    scope = requirement.applicability_scope
    if scope is None or not scope.has_scope_constraint:
        return None
    return scope


def _candidate_matches_requirement_scope(
    requirement: AnswerRequirementV2,
    item: EvidenceItem,
) -> bool:
    constraints = _requirement_constraints(requirement)
    if constraints is None:
        return True
    status = evaluate_candidate_constraints(constraints, item.to_dict()).status
    return status not in {"mismatch", "unknown"}


def _task_scope_disambiguated_answer_ids(
    *,
    requirements: Sequence[AnswerRequirementV2],
    item: EvidenceItem,
    task_graph: RetrievalTaskGraph | None,
    task_ids: Sequence[str],
    positive_target_ids: set[str],
) -> set[str]:
    """Return one safely scope-distinguished answer owner for a candidate.

    Multi-answer mapping normally distinguishes siblings from their visible
    target terms.  That is insufficient for a version/project comparison:
    ``ProductX 6 安全配置`` and ``ProductX 7 安全配置`` intentionally have
    the same target wording, while single-character versions are not general
    lexical evidence.  The immutable requirement scope can distinguish those
    siblings, but it remains an *admission identity*, not an answer claim.

    A scope-derived owner is therefore accepted only when all of these are
    true:

    * this current-run candidate was retrieved by that answer task;
    * its source identity evaluates exact/compatible for the requirement;
    * the result is not a project-global compatibility shortcut;
    * the candidate already has independent positive target support (text or
      a trusted typed claim) for that requirement; and
    * exactly one of multiple distinct required scopes matches.

    This makes scope a deterministic tie-breaker for genuinely equivalent
    answer targets.  It cannot promote an unscoped/unknown/global row, nor
    can it turn task lineage alone into evidence.
    """

    if task_graph is None or not task_ids or not positive_target_ids:
        return set()

    required_answers = tuple(
        requirement
        for requirement in requirements
        if requirement.is_required_answer
    )
    scoped_answers = tuple(
        requirement
        for requirement in required_answers
        if (
            requirement.applicability_scope is not None
            and requirement.applicability_scope.has_scope_constraint
        )
    )
    if len({requirement.scope_fingerprint for requirement in scoped_answers}) < 2:
        return set()

    direct_task_answer_ids = {
        requirement_id
        for task_id in task_ids
        for requirement_id in task_graph.task_by_id[task_id].target_requirement_ids
        if task_graph.task_by_id[task_id].role == "answer"
    }
    if not direct_task_answer_ids:
        return set()

    matched: set[str] = set()
    for requirement in scoped_answers:
        if (
            requirement.id not in direct_task_answer_ids
            or requirement.id not in positive_target_ids
        ):
            continue
        scope = requirement.applicability_scope
        if scope is None:  # guarded above; keeps static type narrowing local
            continue
        evaluation = evaluate_candidate_constraints(scope, item.to_dict())
        if (
            evaluation.status in {"exact", "compatible"}
            and evaluation.scope_applicability == "exact"
        ):
            matched.add(requirement.id)

    # An explicitly global clause and a multi-scope source are intentionally
    # not a unique comparison-side answer.  They still pass through ordinary
    # lexical/claim mapping when their text itself differentiates a target.
    return matched if len(matched) == 1 else set()


def _candidate_confidence(
    raw: Mapping[str, Any],
    *,
    rerank_succeeded: bool | None,
    force_retrieved: bool,
) -> Literal["verified", "retrieved"]:
    # V2 does not execute a model reranker.  ``None`` is therefore not a
    # successful verification signal, even when a legacy candidate happens to
    # carry stale ``verified`` metadata from another path.
    if force_retrieved or rerank_succeeded is not True:
        return "retrieved"
    declared = str(raw.get("confidence") or "").strip().casefold()
    rerank_status = str(raw.get("rerank_status") or "").strip().casefold()
    if declared == "verified" or rerank_status == "verified":
        return "verified"
    return "retrieved"


def _candidate_values(raw: Mapping[str, Any], field: str) -> list[Any]:
    values: list[Any] = []
    for source in (
        raw,
        raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else {},
    ):
        value = source.get(field)
        if isinstance(value, (list, tuple)):
            values.extend(value)
    return values


def _candidate_support_ids(
    raw: Mapping[str, Any],
    *,
    allowed_ids: set[str] | None = None,
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in _candidate_values(raw, "supports_requirement_ids"):
        if not isinstance(value, str):
            continue
        requirement_id = value.strip()
        if (
            not _REQUIREMENT_ID_RE.fullmatch(requirement_id)
            or requirement_id in seen
            or (allowed_ids is not None and requirement_id not in allowed_ids)
        ):
            continue
        seen.add(requirement_id)
        result.append(requirement_id)
        if len(result) >= 8:
            break
    return tuple(result)


def _candidate_task_ids(
    item: EvidenceItem,
    *,
    task_graph: RetrievalTaskGraph,
    task_ledger: TaskExecutionLedger,
) -> tuple[tuple[str, ...], bool]:
    """Read explicit task provenance without guessing from query position.

    The second return value reports malformed/unknown identifiers.  Candidate
    metadata is untrusted input, so malformed annotations are retained for
    diagnostics and fail closed rather than aborting the whole retrieval.
    """

    # Retrieval task ids inside a candidate are input observations, not proof.
    # The request-local ledger is the only authority that preserves the exact
    # physical execution and parent-child binding for this request.  Reading
    # persisted/retriever metadata here would make provenance depend on the
    # caller's entry point and can fabricate a bridge second hop.
    if task_ledger.task_graph != task_graph:
        raise ValueError("task ledger graph must match evidence task graph")
    lineage = task_ledger.lineage_for_candidate(item.to_dict())
    return (lineage.task_ids if lineage is not None else ()), False


def _candidate_role(raw: Mapping[str, Any]) -> str:
    metadata = raw.get("metadata")
    sources = (raw, metadata if isinstance(metadata, Mapping) else {})
    for source in sources:
        value = str(source.get("role") or "").strip().casefold()
        if value in _VALID_EVIDENCE_ROLES:
            return value
    for source in sources:
        contribution = str(
            source.get("contribution_role") or ""
        ).strip().casefold()
        if contribution == "standalone_answer":
            return "direct"
        if contribution in _VALID_EVIDENCE_ROLES:
            return contribution
        evidence_role = str(source.get("evidence_role") or "").strip().casefold()
        if evidence_role in _VALID_EVIDENCE_ROLES:
            return evidence_role
    return "background"


def _to_evidence_item(
    candidate: Mapping[str, Any] | EvidenceItem,
    *,
    constraints: QueryConstraints | None,
    rerank_succeeded: bool | None,
    extra_origin: str | None = None,
    force_retrieved: bool = False,
) -> tuple[EvidenceItem | None, str | None]:
    try:
        raw = _as_mapping(candidate)
        if raw.get("authorized", True) is not True:
            return None, "unauthorized_candidate_excluded"
        content = str(raw.get("content") or "").strip()
        doc_id = _string_id(raw.get("doc_id"))
        kb_id = _string_id(raw.get("kb_id"))
        index = _chunk_index(raw.get("chunk_index"))
        chunk_id = _string_id(raw.get("chunk_id") or raw.get("id"))
        if not chunk_id and kb_id and doc_id:
            chunk_id = f"{kb_id}:{doc_id}:{index}"
        if not content or not doc_id or not kb_id or not chunk_id:
            return None, "invalid_candidate_excluded"

        status = _constraint_status(raw, constraints)
        if status == "mismatch":
            return None, "hard_constraint_mismatch_excluded"
        # Explicit product/version wording is an applicability boundary, not
        # a ranking hint.  Candidates whose document identity cannot confirm
        # that boundary remain diagnostic-only and never enter generation.
        # Explicit multi-scope comparisons pass neutral constraints after a
        # source-anchored document allow-list has been resolved.
        if (
            constraints is not None
            and constraints.has_scope_constraint
            and status == "unknown"
        ):
            return None, "hard_constraint_unknown_excluded"

        section_kind = _section_kind(
            content,
            raw.get("metadata") if isinstance(raw.get("metadata"), Mapping) else None,
        )
        supports_requirement_ids = _candidate_support_ids(raw)
        role = _candidate_role(raw)
        # A role assertion without any requirement mapping is not usable
        # evidence classification.  Preserve the legacy annotation in metadata
        # and wait for deterministic mapping below before promoting the item.
        if role != "background" and not supports_requirement_ids:
            role = "background"
        metadata = _metadata(raw, section_kind=section_kind)
        metadata["supports_requirement_ids"] = list(supports_requirement_ids)
        metadata["evidence_role_v2"] = role
        return (
            EvidenceItem(
                chunk_id=chunk_id,
                doc_id=doc_id,
                kb_id=kb_id,
                content=content,
                chunk_index=index,
                score=_score(raw.get("score")),
                confidence=_candidate_confidence(
                    raw,
                    rerank_succeeded=rerank_succeeded,
                    force_retrieved=force_retrieved,
                ),
                constraint_status=status,
                authorized=True,
                origins=_origins(raw, extra=extra_origin),
                role=role,
                supports_requirement_ids=supports_requirement_ids,
                metadata=metadata,
            ),
            None,
        )
    except (TypeError, ValueError, AttributeError):
        return None, "invalid_candidate_excluded"


def assemble_unverified_candidate_bundle_with_diagnostics(
    *,
    candidates: Sequence[Mapping[str, Any] | EvidenceItem],
    allowed_requirement_ids: Sequence[str],
    constraints: QueryConstraints | None = None,
    max_context_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS,
    max_context_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    admission_reason: str = RELATED_EVIDENCE_REASON,
    role_mode: Literal["unverified", "direct"] = "unverified",
) -> UnverifiedCandidateBundleResult:
    """Build a bounded, scope-safe snapshot for related-evidence generation.

    This path deliberately does not run requirement mapping, claim creation or
    coverage reconciliation.  The caller must first bind every candidate to
    the current request and attach ``supports_requirement_ids``; this boundary
    then performs the same candidate validation, authorization and hard-scope
    admission used by normal evidence assembly.

    ``admission_reason`` is diagnostic only; it never grants authorization or
    scope. Requirement-aware round-robin selection prevents one comparison side from
    consuming the complete prompt budget.  The returned sources remain
    retrieved/unverified and all requirements stay missing: this artifact is a
    labelled generation aid, never an alternate proof result.
    """

    _validate_budget(max_context_chunks, max_context_chars)
    admission_reason = re.sub(r"[^a-z0-9_:.-]+", "_", str(admission_reason or "").casefold()).strip("_")
    if not admission_reason or len(admission_reason) > 120:
        raise ValueError("admission_reason is invalid")
    if isinstance(allowed_requirement_ids, (str, bytes)):
        raise ValueError("allowed_requirement_ids must be a sequence")
    allowed_ids = tuple(dict.fromkeys(
        str(value or "").strip()
        for value in allowed_requirement_ids
    ))
    if (
        len(allowed_ids) > 8
        or any(not _REQUIREMENT_ID_RE.fullmatch(value) for value in allowed_ids)
    ):
        raise ValueError("allowed_requirement_ids contains an invalid id")
    if not allowed_ids:
        return UnverifiedCandidateBundleResult(
            bundle=None,
            input_candidate_count=len(candidates),
            converted_candidate_count=0,
            selected_candidate_count=0,
            exclusion_reason_counts=(("no_allowed_requirement_ids", len(candidates)),),
        )
    allowed_id_set = set(allowed_ids)

    converted: list[EvidenceItem] = []
    seen_chunk_ids: set[str] = set()
    exclusion_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        item, reason = _to_evidence_item(
            candidate,
            constraints=constraints,
            rerank_succeeded=False,
            force_retrieved=True,
        )
        if item is None:
            exclusion_counts[reason or "candidate_conversion_failed"] += 1
            continue
        if item.chunk_id in seen_chunk_ids:
            exclusion_counts["duplicate_candidate_excluded"] += 1
            continue
        support_ids = tuple(
            requirement_id
            for requirement_id in item.supports_requirement_ids
            if requirement_id in allowed_id_set
        )
        if not support_ids:
            exclusion_counts["requirement_binding_excluded"] += 1
            continue
        seen_chunk_ids.add(item.chunk_id)
        direct_mode = role_mode == "direct"
        converted.append(replace(
            item,
            confidence="retrieved",
            role="direct" if direct_mode else "complement",
            contribution_kind=None,
            supports_requirement_ids=support_ids,
            metadata={
                **dict(item.metadata),
                "supports_requirement_ids": list(support_ids),
                "evidence_role_v2": "direct" if direct_mode else "complement",
                "evidence_role": "direct" if direct_mode else "unverified",
                # A server-side dominant-document auto-selection is a
                # deterministic scope decision, not a model verification.  Two
                # axes stay separate: the role axis keeps the answer evidence
                # direct (evidence counts, coverage and the answer policy agree),
                # while the verification axis stays unverified because the
                # reranker never confirmed it.  The deterministic basis is what
                # lets coverage/policy consumers treat the binding as positive
                # evidence without ever claiming model verification.
                "source_verification": "unverified",
                "verification_basis": (
                    "deterministic_candidate_scope_confirmed"
                    if direct_mode
                    else None
                ),
                "rerank_status": "unverified",
                "unverified_generation_fallback": True,
            },
        ))
    if not converted:
        return UnverifiedCandidateBundleResult(
            bundle=None,
            input_candidate_count=len(candidates),
            converted_candidate_count=0,
            selected_candidate_count=0,
            exclusion_reason_counts=tuple(sorted(exclusion_counts.items())),
        )

    queues = {
        requirement_id: [
            item
            for item in converted
            if requirement_id in item.supports_requirement_ids
        ]
        for requirement_id in allowed_ids
    }
    queue_indexes = {requirement_id: 0 for requirement_id in allowed_ids}
    selected: list[EvidenceItem] = []
    selected_ids: set[str] = set()
    budget_rejected_ids: set[str] = set()

    def try_add(item: EvidenceItem) -> bool:
        if item.chunk_id in selected_ids:
            return False
        prospective = [*selected, item]
        if (
            len(prospective) > max_context_chunks
            or evidence_context_char_cost(prospective) > max_context_chars
        ):
            budget_rejected_ids.add(item.chunk_id)
            return False
        selected.append(item)
        selected_ids.add(item.chunk_id)
        return True

    # Give each answer partition a chance before filling remaining capacity in
    # retrieval order.  Oversized candidates are skipped as complete blocks;
    # partial chunk text is never exposed to generation.
    while len(selected) < max_context_chunks:
        progressed = False
        exhausted = True
        for requirement_id in allowed_ids:
            queue = queues[requirement_id]
            index = queue_indexes[requirement_id]
            while index < len(queue) and queue[index].chunk_id in selected_ids:
                index += 1
            while index < len(queue):
                exhausted = False
                item = queue[index]
                index += 1
                if try_add(item):
                    progressed = True
                    break
            queue_indexes[requirement_id] = index
        if exhausted or not progressed:
            break

    for item in converted:
        if len(selected) >= max_context_chunks:
            break
        try_add(item)
    if budget_rejected_ids:
        exclusion_counts["context_budget_excluded"] += len(budget_rejected_ids)
    if not selected:
        return UnverifiedCandidateBundleResult(
            bundle=None,
            input_candidate_count=len(candidates),
            converted_candidate_count=len(converted),
            selected_candidate_count=0,
            exclusion_reason_counts=tuple(sorted(exclusion_counts.items())),
        )

    selected = list(order_evidence_context_items(selected))
    item_ids = tuple(item.chunk_id for item in selected)
    direct_mode = role_mode == "direct"
    if direct_mode:
        # The auto-confirmed dominant document was deterministically bound to
        # the requested answer targets.  Report those bindings as covered so
        # coverage counts and the answer policy agree; the reranker status
        # stays unverified and the state keeps completeness=unknown, so the
        # answer remains an honest labelled partial instead of a closed hit.
        covered: list[str] = []
        seen_covered: set[str] = set()
        for item in selected:
            for requirement_id in item.supports_requirement_ids:
                if requirement_id in seen_covered:
                    continue
                seen_covered.add(requirement_id)
                covered.append(requirement_id)
        missing_ids = tuple(
            requirement_id
            for requirement_id in allowed_ids
            if requirement_id not in seen_covered
        )
    else:
        covered = []
        missing_ids = allowed_ids
    bundle = EvidenceBundle(
        state=EvidenceState(
            availability="degraded",
            confidence="retrieved",
            completeness="unknown",
            reasons=(admission_reason,),
        ),
        items=tuple(selected),
        context_item_ids=item_ids,
        answer_source_ids=item_ids,
        # ``covered_requirement_ids`` is a derived property: it reads the
        # items' roles and source verification, so the direct-mode metadata
        # above is what makes coverage visible to the policy layer.
        missing_requirement_ids=missing_ids,
    )
    context = build_evidence_context(
        bundle,
        max_chunks=max_context_chunks,
        max_chars=max_context_chars,
    )
    if context.truncated or tuple(context.item_ids) != item_ids:
        exclusion_counts["rendered_context_mismatch"] += len(selected)
        bundle = None
    return UnverifiedCandidateBundleResult(
        bundle=bundle,
        input_candidate_count=len(candidates),
        converted_candidate_count=len(converted),
        selected_candidate_count=(len(selected) if bundle is not None else 0),
        exclusion_reason_counts=tuple(sorted(exclusion_counts.items())),
    )


def assemble_unverified_candidate_bundle(
    *,
    candidates: Sequence[Mapping[str, Any] | EvidenceItem],
    allowed_requirement_ids: Sequence[str],
    constraints: QueryConstraints | None = None,
    max_context_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS,
    max_context_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
) -> EvidenceBundle | None:
    """Compatibility wrapper returning only the bounded degraded bundle."""

    return assemble_unverified_candidate_bundle_with_diagnostics(
        candidates=candidates,
        allowed_requirement_ids=allowed_requirement_ids,
        constraints=constraints,
        max_context_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
    ).bundle


def _normalize_requirements(
    requirements: Sequence[AnswerRequirementV2],
) -> tuple[AnswerRequirementV2, ...]:
    if isinstance(requirements, (str, bytes)) or not isinstance(
        requirements,
        Sequence,
    ):
        raise ValueError("requirements must be a sequence")
    normalized = tuple(requirements)
    if len(normalized) > 8:
        raise ValueError("requirements has too many items")
    if any(not isinstance(item, AnswerRequirementV2) for item in normalized):
        raise ValueError("requirements must contain AnswerRequirementV2 values")
    ids = [item.id for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("requirements contains duplicate ids")
    validate_answer_requirement_graph(normalized)
    return normalized


def _normalize_retrieval_queries(
    retrieval_queries: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(retrieval_queries, (str, bytes)) or not isinstance(
        retrieval_queries,
        Sequence,
    ):
        raise ValueError("retrieval_queries must be a sequence")
    if len(retrieval_queries) > 8:
        raise ValueError("retrieval_queries has too many items")
    normalized: list[str] = []
    for value in retrieval_queries:
        if not isinstance(value, str):
            raise ValueError("retrieval_queries must contain strings")
        query = re.sub(r"\s+", " ", value).strip()
        if not query or len(query) > 1000:
            raise ValueError("retrieval query must contain 1 to 1000 characters")
        normalized.append(query)
    return tuple(normalized)


def _coverage_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for match in _COVERAGE_TEXT_RE.finditer(str(value or "").casefold()):
        token = match.group(0)
        terms.add(token)
        if re.fullmatch(r"[\u3400-\u9fff]+", token) and len(token) > 2:
            for index in range(len(token) - 1):
                pair = token[index:index + 2]
                if not (_COVERAGE_STOP_CHARS & set(pair)):
                    terms.add(pair)
    return terms - _GENERIC_COVERAGE_TERMS


def _text_support(
    description: str,
    content: str,
) -> tuple[bool, bool]:
    normalized_description = re.sub(r"\s+", " ", description).strip().casefold()
    normalized_content = re.sub(r"\s+", " ", content).strip().casefold()
    exact = bool(
        normalized_description
        and normalized_description in normalized_content
    )
    if exact:
        return True, True
    requirement_terms = _coverage_terms(normalized_description)
    if not requirement_terms:
        return False, False
    content_terms = _coverage_terms(normalized_content)
    # Repeated overlapping bigrams from one entity must not satisfy a compound
    # requirement by themselves.  For example, a chunk that only says
    # "普通岗位对应D级" cannot support "普通岗位的餐饮补贴金额" merely because
    # several bigrams from "普通岗位" overlap.  When the requirement contains
    # two or more meaningful parts, require evidence from at least two parts.
    part_terms = [
        terms
        for part in _COVERAGE_PART_SPLIT_RE.split(normalized_description)
        if (terms := _coverage_terms(part))
    ]
    if len(part_terms) >= 2:
        matched_parts = sum(bool(terms & content_terms) for terms in part_terms)
        if matched_parts < 2:
            return False, False
    overlap = requirement_terms & content_terms
    if len(requirement_terms) == 1:
        return bool(overlap), False
    if len(overlap) >= 2:
        return True, False
    # A stable identifier is often the only meaningful signal in an otherwise
    # generic requirement.  Chinese fragments still require two independent
    # overlaps so adjacent chunks cannot jointly manufacture coverage.
    ascii_identifier_hit = any(
        len(term) >= 4 and re.search(r"[0-9_.+/-]", term)
        for term in overlap
    )
    return ascii_identifier_hit, False


def _bridge_text_support(
    requirement: AnswerRequirementV2,
    content: str,
) -> tuple[bool, bool]:
    """Require a subject-anchored, source-resolved bridge value.

    Merely seeing the subject plus an arbitrary identifier is not a mapping.
    For example, ``A级人员出差需总经理审批`` contains both ``总经理`` and
    ``A级`` but does not establish that the manager belongs to A level.  The
    shared bridge resolver accepts only an explicit relation sentence or the
    exact table row containing the subject.
    """

    subject = bridge_subject_for_requirement(requirement)
    if subject is None:
        # A bridge is a relationship claim, never a bag-of-words topic.  Free
        # text that cannot be canonicalized must remain missing instead of
        # completing a multi-hop path through lexical overlap.
        return False, False
    values = extract_bridge_values(
        requirement.description,
        content,
        subject=subject,
        bridge_kind=requirement.bridge_kind,
    )
    if not values:
        return False, False
    normalized_content = re.sub(r"\s+", " ", str(content or "")).casefold()
    return True, subject.casefold() in normalized_content


def _item_topic_text(item: EvidenceItem) -> str:
    values = [item.content]
    for field in ("filename", "source", "heading", "title"):
        value = item.metadata.get(field)
        if str(value or "").strip():
            values.append(str(value).strip())
    return "\n".join(values)


def _declarative_answer_support(
    description: str,
    item: EvidenceItem,
    *,
    bridge_subjects: Sequence[str],
) -> bool:
    """Match the declarative answer target, never retrieval provenance.

    User questions and source clauses express the same target with different
    predicates (``值是多少`` vs ``值为...``).  A source filename or heading
    may establish the topic, but the retained body must still mention part of
    that target and contain a concrete value, relation, action or procedure.
    This keeps title-only hits out of answer sources while accepting grounded
    fact/process clauses that do not repeat the original interrogative.
    """

    if not content_matches_answer_target(
        description,
        _item_topic_text(item),
        bridge_subjects=bridge_subjects,
    ):
        return False
    target_terms = answer_target_terms(
        description,
        bridge_subjects=bridge_subjects,
    )
    content_terms = _coverage_terms(item.content)
    target_coverage_terms = {
        term
        for target in target_terms
        for term in _coverage_terms(target)
    }
    visible_target = bool(target_coverage_terms & content_terms) or any(
        content_matches_answer_target(
            target,
            item.content,
            bridge_subjects=(),
        )
        for target in target_terms
    )
    if not visible_target:
        return False
    return bool(
        adjudicate_answer_claims(
            description,
            item.content,
            bridge_subjects=bridge_subjects,
        )
    )


def _direct_subject_answer_support(
    description: str,
    item: EvidenceItem,
    *,
    bridge_subjects: Sequence[str],
) -> bool:
    """Whether one visible claim directly binds subject, target and result."""

    dependent_subjects = tuple(
        subject
        for subject in bridge_subjects
        if subject.casefold() in description.casefold()
    )
    if not dependent_subjects:
        return False
    target_terms = answer_target_terms(
        description,
        bridge_subjects=bridge_subjects,
    )
    for clause in re.split(r"[\n。；;]+", item.content):
        normalized = re.sub(r"\s+", "", clause).casefold()
        if not normalized or not all(
            content_contains_positive_subject(
                clause,
                subject,
                target_terms=target_terms,
            )
            for subject in dependent_subjects
        ):
            continue
        # When the planner extracts a taxonomy-bearing prefix from a longer
        # grammatical scope, the residual context remains evidence-critical.
        # Otherwise ``employee family lodging`` could be answered by a direct
        # ``employee lodging`` clause merely because both contain ``lodging``.
        if not content_matches_complete_answer_target(
            description,
            clause,
            bridge_subjects=dependent_subjects,
        ):
            continue
        if adjudicate_answer_claims(
            description,
            clause,
            bridge_subjects=dependent_subjects,
            required_subjects=dependent_subjects,
        ):
            return True
    return False


_DOCUMENT_ROOT_SEED_ORIGINS = frozenset({
    "initial_retrieval",
    "carryover_current_retrieval",
})
_DOCUMENT_ROOT_SIBLING_ORIGINS = frozenset({
    "small_document_full",
    "document_scoped",
    "adjacent",
    "same_section",
    "table_sibling",
})
_DOCUMENT_SECTION_LABEL_RE = re.compile(
    r"(?m)^\s*(?:第?[一二三四五六七八九十百0-9]+[章节、.．]|"
    r"\d+(?:\.\d+){0,4}[、.．])\s*([^\n：:]{1,160})",
)


def _document_root_text(item: EvidenceItem) -> str:
    values = [
        item.metadata.get("filename"),
        item.metadata.get("source"),
        item.metadata.get("title"),
    ]
    # A first/root chunk heading may be the only title retained by an older
    # ingester.  Section headings on later chunks remain facet-level signals
    # and cannot turn the whole document into that facet.
    if item.chunk_index == 0:
        values.append(item.metadata.get("heading"))
    return "\n".join(
        str(value).strip() for value in values if str(value or "").strip()
    )


def _document_section_label(item: EvidenceItem) -> str:
    for field in ("heading", "section_title", "section_name"):
        value = str(item.metadata.get(field) or "").strip()
        if value:
            return re.sub(r"\s+", " ", value).casefold()[:160]
    match = re.search(r"(?m)^\s*#{1,6}\s+([^\n#]{1,160})$", item.content)
    if match is None:
        # DOCX/plain-text imports often retain chapter prefixes but not
        # Markdown headings.  This is intentionally narrower than treating
        # every line as a section, so one arbitrary facet cannot authorize a
        # whole-document route.
        match = _DOCUMENT_SECTION_LABEL_RE.search(item.content)
    return (
        re.sub(r"\s+", " ", match.group(1)).strip().casefold()
        if match is not None
        else ""
    )


def _attach_document_root_topic_anchors(
    items: Sequence[EvidenceItem],
    *,
    requirements: Sequence[AnswerRequirementV2],
) -> list[EvidenceItem]:
    """Annotate bounded siblings of a current-query document root.

    This is a topic inheritance proof, never answer proof by itself.  The
    bridge resolver still requires the resolved value and a concrete result in
    the same row/sentence.  It exists for broad document-root questions such as
    ``出差标准`` where sections naturally say ``飞机``/``火车``/``住宿`` and do
    not repeat the document title.  A current-query seed and its same-document
    title/root identity may jointly establish the target: the title commonly
    carries ``出差管理标准`` while a later section names ``普通员工``.  A
    retrieved facet still cannot fan out by itself: source identity, a
    multi-section structure and bounded same-document expansion are all
    required before siblings inherit it.
    """

    answer_requirements = tuple(
        requirement for requirement in requirements if requirement.role == "answer"
    )
    if not answer_requirements or not items:
        return list(items)

    bridge_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }
    bridge_subjects_by_answer = {
        answer.id: tuple(dict.fromkeys(
            subject
            for dependency_id in bridge_dependency_ids_for_answer(
                answer,
                requirements,
            )
            for bridge in (bridge_by_id.get(dependency_id),)
            if bridge is not None
            if (subject := bridge_subject_for_requirement(bridge))
        ))
        for answer in answer_requirements
    }
    section_labels_by_document: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in items:
        if not set(item.origins) & (
            _DOCUMENT_ROOT_SEED_ORIGINS | _DOCUMENT_ROOT_SIBLING_ORIGINS
        ):
            continue
        if section_label := _document_section_label(item):
            section_labels_by_document[(item.kb_id, item.doc_id)].add(
                section_label
            )
    complete_snapshot_documents = frozenset(
        complete_snapshot_document_keys(
            items,
            require_visible=False,
        )
    )

    root_ids_by_document: dict[tuple[str, str], set[str]] = defaultdict(set)
    document_policy_root_ids_by_document: dict[
        tuple[str, str], set[str]
    ] = defaultdict(set)
    for item in items:
        is_primary_seed = bool(
            set(item.origins) & _DOCUMENT_ROOT_SEED_ORIGINS
        )
        if not is_primary_seed:
            continue
        root_text = _document_root_text(item)
        if not root_text:
            continue
        document_key = (item.kb_id, item.doc_id)
        for requirement in answer_requirements:
            # Ordinary fact routes still require a genuinely multi-section
            # source before a title/root seed may lend its topic to sibling
            # chunks.  A document-policy overview has a stronger, structural
            # proof available: the retriever supplied every indexed chunk of
            # one bounded source.  That complete snapshot remains sufficient
            # even when a DOCX importer did not preserve section headings.
            # This distinction is contract based; filenames identify the
            # retrieved source but never prove snapshot completeness.
            has_multi_section_structure = (
                len(section_labels_by_document.get(document_key, set())) >= 2
            )
            has_complete_policy_snapshot = (
                requirement.requires_document_policy_snapshot
                and document_key in complete_snapshot_documents
            )
            if not (
                has_multi_section_structure
                or has_complete_policy_snapshot
            ):
                continue
            # A governing policy name and its applicable subject frequently
            # live in different sections of the same source.  They can be
            # considered together only because this exact item is a current
            # retrieval seed and ``root_text`` is its immutable document
            # identity; text from another candidate is never borrowed.
            root_seed_evidence = "\n".join((root_text, item.content))
            if content_matches_answer_target(
                requirement.description,
                root_seed_evidence,
                bridge_subjects=bridge_subjects_by_answer[requirement.id],
            ):
                root_ids_by_document[document_key].add(requirement.id)
                if requirement.requires_document_policy_snapshot:
                    document_policy_root_ids_by_document[document_key].add(
                        requirement.id
                    )

    anchored: list[EvidenceItem] = []
    for item in items:
        root_ids = root_ids_by_document.get((item.kb_id, item.doc_id), set())
        is_current_query_seed = bool(
            set(item.origins) & _DOCUMENT_ROOT_SEED_ORIGINS
        )
        may_inherit = bool(
            set(item.origins) & (
                _DOCUMENT_ROOT_SEED_ORIGINS | _DOCUMENT_ROOT_SIBLING_ORIGINS
            )
        )
        metadata = dict(item.metadata)
        if root_ids and may_inherit:
            metadata["document_root_answer_requirement_ids"] = sorted(root_ids)
        else:
            metadata.pop("document_root_answer_requirement_ids", None)
        # Topic inheritance can span bounded same-document siblings, but a
        # whole-document policy root is stricter: only the current-query
        # retrieval seed whose root identity was verified may establish it.
        document_policy_root_ids = document_policy_root_ids_by_document.get(
            (item.kb_id, item.doc_id),
            set(),
        )
        if document_policy_root_ids and is_current_query_seed:
            metadata["document_policy_root_requirement_ids"] = sorted(
                document_policy_root_ids
            )
        else:
            metadata.pop("document_policy_root_requirement_ids", None)
        anchored.append(replace(item, metadata=metadata))
    return anchored


def _independent_answer_terms_match(
    answer_description: str,
    bridge_subjects: Sequence[str],
    content: str,
) -> bool:
    """Require target evidence beyond the entity words used by a bridge.

    Compact Chinese questions (``合同工住宿标准``) have no delimiter between
    qualifier and target.  Generic bigram overlap can therefore make the pure
    mapping clause ``合同工属于L2类`` look like answer support.  Remove every
    bridge-subject term before checking the answer target; a combined clause
    that also states ``住宿标准`` still passes, while bridge-only text cannot.
    """

    answer_terms = _coverage_terms(answer_description)
    bridge_subject_terms: set[str] = set()
    for subject in bridge_subjects:
        bridge_subject_terms.update(_coverage_terms(subject))
    target_terms = {
        term
        for term in answer_terms - bridge_subject_terms
        if term not in _GENERIC_COVERAGE_TERMS
    }
    if not target_terms:
        return False
    content_terms = _coverage_terms(content)
    return bool(target_terms & content_terms)


def _multi_answer_text_support_ids(
    *,
    requirements: Sequence[AnswerRequirementV2],
    content: str,
    lexical_ids: Sequence[str],
    exact_ids: set[str],
    scope_disambiguated_ids: set[str] | None = None,
) -> set[str]:
    """Return required answers that the visible chunk can distinguish.

    A global expansion may return the same chunk for several plan queries.  In
    that case every query index is merged onto the candidate, but those indexes
    only describe retrieval provenance and cannot mean that one chunk answers
    every sub-question.  Keep strong per-requirement lexical matches and, for
    terse value rows, retain only the requirement(s) with the strongest overlap
    on terms that distinguish them from the other required answers.

    Bridge subjects are removed before scoring multi-hop answer rows so a
    classification clause cannot masquerade as an answer merely because it
    repeats the employee/product/entity name.
    """

    required_answers = tuple(
        requirement
        for requirement in requirements
        if requirement.is_required_answer
    )
    if len(required_answers) <= 1:
        return {requirement.id for requirement in required_answers}

    required_answer_ids = {requirement.id for requirement in required_answers}
    lexical_answer_ids = set(lexical_ids) & required_answer_ids
    exact_answer_ids = set(exact_ids) & required_answer_ids
    # The caller has already verified that these ids have positive target
    # support, task-local current-run provenance, and one exact non-global
    # applicability scope.  Scope never enters this helper as a replacement
    # for text support; it only restores the identity signal that a generic
    # tokeniser intentionally omits for one-character versions/project keys.
    scope_owned_answer_ids = (
        set(scope_disambiguated_ids or ())
        & required_answer_ids
    )
    content_terms = _coverage_terms(content)
    if not content_terms:
        return exact_answer_ids | scope_owned_answer_ids

    terms_by_id = {
        requirement.id: _coverage_terms(requirement.description)
        for requirement in required_answers
    }
    bridge_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }
    scores: dict[str, int] = {}
    for requirement in required_answers:
        other_terms: set[str] = set()
        for other in required_answers:
            if other.id != requirement.id:
                other_terms.update(terms_by_id[other.id])
        discriminative_terms = terms_by_id[requirement.id] - other_terms
        if not discriminative_terms:
            continue
        dependent_bridge_subjects = tuple(
            subject
            for dependency_id in bridge_dependency_ids_for_answer(
                requirement,
                requirements,
            )
            for bridge in (bridge_by_id.get(dependency_id),)
            if bridge is not None
            if (subject := bridge_subject_for_requirement(bridge))
        )
        if dependent_bridge_subjects and not _independent_answer_terms_match(
            requirement.description,
            dependent_bridge_subjects,
            content,
        ):
            continue
        score = len(discriminative_terms & content_terms)
        if score:
            scores[requirement.id] = score

    if not scores:
        return exact_answer_ids | scope_owned_answer_ids
    strongest_score = max(scores.values())
    # Generic subject overlap can make the broad lexical matcher accept every
    # coordinated answer (for example ``普通员工制度`` against住宿/交通/餐补).
    # A non-exact lexical match is therefore retained only when the visible
    # chunk also contains a term that distinguishes that answer.
    discriminative_lexical_ids = lexical_answer_ids & set(scores)
    return exact_answer_ids | discriminative_lexical_ids | scope_owned_answer_ids | {
        requirement_id
        for requirement_id, score in scores.items()
        if score == strongest_score
    }


def _reconcile_multi_hop_links(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    task_ledger: TaskExecutionLedger | None = None,
) -> list[EvidenceItem]:
    """Promote and retain answers only through a complete bridge path.

    The previous implementation flattened every identifier in an entire
    mapping table.  A table containing A/B/C/D could therefore make any level
    look compatible, while a value clause that was loaded by full-document
    expansion could never become answer evidence.  Resolution is now row- and
    subject-anchored, and promotion/removal use the same fact contract.
    """

    requirement_by_id = {item.id: item for item in requirements}
    bridge_ids = {item.id for item in requirements if item.role == "bridge"}
    if not bridge_ids:
        return list(items)
    if task_ledger is not None:
        # The runtime scheduler is the only authority that can release a
        # bridge-dependent answer.  Re-parsing the mixed final candidate pool
        # here would let an anchor/sibling/full-document chunk impersonate a
        # bridge task after the scheduler had recorded no_fact/conflict/failed.
        # Source-local direct answer bypass remains below and is explicitly
        # marked, but it never silently invents a bridge value.
        facts = tuple(
            fact
            for resolution in task_ledger.bridge_resolutions()
            if resolution.status == "resolved"
            for fact in resolution.facts
        )
        conflicts = tuple(
            conflict
            for resolution in task_ledger.bridge_resolutions()
            if resolution.status == "conflict"
            for conflict in resolution.conflicts
        )
    else:
        all_facts = resolve_bridge_facts(
            requirements,
            items,
            supported_only=True,
        )
        facts, conflicts = partition_bridge_facts(all_facts)
    conflict_by_chunk: dict[str, dict[str, set[str]]] = {}
    for conflict in conflicts:
        for chunk_id in conflict.source_chunk_ids:
            values = conflict_by_chunk.setdefault(chunk_id, {}).setdefault(
                conflict.requirement_id,
                set(),
            )
            values.update(conflict.values)

    bridge_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }
    bridge_subjects_by_answer = {
        requirement.id: tuple(dict.fromkeys(
            subject
            for dependency_id in bridge_dependency_ids_for_answer(
                requirement,
                requirements,
            )
            for bridge in (bridge_by_id.get(dependency_id),)
            if bridge is not None
            if (subject := bridge_subject_for_requirement(bridge))
        ))
        for requirement in requirements
        if requirement.role == "answer"
    }
    answer_requirements = tuple(
        requirement for requirement in requirements if requirement.role == "answer"
    )
    bridge_dependencies_by_answer = {
        answer.id: bridge_dependency_ids_for_answer(answer, requirements)
        for answer in answer_requirements
    }
    bridge_requirement_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }

    promoted: list[EvidenceItem] = []
    for item in items:
        supports = set(item.supports_requirement_ids)
        # A proof edge is a hard applicability contract.  Even a source that
        # repeats the original subject cannot bypass it; otherwise a direct
        # lexical hit would silently turn a required relationship into an
        # optional one.  Augmentation direct routes are handled separately
        # below and deliberately do not enter this proof-only reconciler.
        direct_subject_answer_ids: set[str] = set()
        item_conflicts = conflict_by_chunk.get(item.chunk_id, {})
        supports -= set(item_conflicts)
        joined: list[dict[str, str]] = []
        for requirement in answer_requirements:
            dependency_ids = bridge_dependencies_by_answer[requirement.id]
            # Lexical/query-index support is provisional for a dependent
            # answer.  It becomes usable only after the exact resolved bridge
            # value joins this answer claim, or when the claim explicitly
            # binds the original subject, target and result and every bridge is
            # an inferred helpful dependency.  Never leave the provisional id
            # in place merely because a bridge could not be parsed.
            if dependency_ids:
                supports.discard(requirement.id)
            joined_for_answer: list[dict[str, str]] = []
            selected_facts_for_answer = []
            document_root_target_verified = requirement.id in {
                str(value)
                for value in item.metadata.get(
                    "document_root_answer_requirement_ids",
                    [],
                )
                if isinstance(value, str)
            }
            for dependency_id in dependency_ids:
                matched_facts = [
                    fact
                    for fact in facts
                    if fact.requirement_id == dependency_id
                    and candidate_supports_resolved_answer_set(
                        requirement,
                        item,
                        (fact,),
                        bridge_subjects=bridge_subjects_by_answer.get(
                            requirement.id,
                            (),
                        ),
                        document_root_target_verified=(
                            document_root_target_verified
                        ),
                    )
                ]
                matched_values = {
                    re.sub(r"\s+", "", fact.value).casefold()
                    for fact in matched_facts
                }
                # A bridge is valid only when this exact answer candidate
                # selects one value.  Different documents may form competing
                # complete graphs, but a clause matching two values is not a
                # safe path and must not inherit whichever fact happened to be
                # retrieved first.
                if len(matched_values) != 1:
                    joined_for_answer = []
                    break
                matched_fact = min(
                    matched_facts,
                    key=lambda fact: (
                        fact.source_doc_id != item.doc_id,
                        fact.source_kb_id.casefold(),
                        fact.source_doc_id.casefold(),
                        fact.source_chunk_id.casefold(),
                    ),
                )
                selected_facts_for_answer.append(matched_fact)
                joined_for_answer.append({
                    "answer_requirement_id": requirement.id,
                    "bridge_requirement_id": matched_fact.requirement_id,
                    "bridge_value": matched_fact.value,
                    "bridge_source_chunk_id": matched_fact.source_chunk_id,
                    "bridge_source_doc_id": matched_fact.source_doc_id,
                    "bridge_source_kb_id": matched_fact.source_kb_id,
                })
            if (
                dependency_ids
                and len(selected_facts_for_answer) == len(dependency_ids)
                and not candidate_supports_resolved_answer_set(
                    requirement,
                    item,
                    selected_facts_for_answer,
                    bridge_subjects=bridge_subjects_by_answer.get(
                        requirement.id,
                        (),
                    ),
                    document_root_target_verified=(
                        document_root_target_verified
                    ),
                )
            ):
                joined_for_answer = []
            if dependency_ids and len(joined_for_answer) == len(dependency_ids):
                supports.add(requirement.id)
                joined.extend(joined_for_answer)
            elif requirement.id in direct_subject_answer_ids:
                supports.add(requirement.id)
        ordered_supports = tuple(
            requirement.id
            for requirement in requirements
            if requirement.id in supports
        )
        metadata = dict(item.metadata)
        for field in (
            "resolved_bridge_joins",
            "bridge_linked_requirement_ids",
            "bridge_link_rejected_requirement_ids",
            "direct_subject_bridge_bypass_requirement_ids",
        ):
            metadata.pop(field, None)
        metadata["supports_requirement_ids"] = list(ordered_supports)
        if item_conflicts:
            metadata["bridge_conflicts"] = [
                {
                    "bridge_requirement_id": requirement_id,
                    "values": sorted(values, key=str.casefold),
                }
                for requirement_id, values in sorted(item_conflicts.items())
            ]
        if joined:
            metadata["resolved_bridge_joins"] = joined[:8]
        if direct_subject_answer_ids:
            metadata["direct_subject_bridge_bypass_requirement_ids"] = sorted(
                direct_subject_answer_ids
            )
        if not ordered_supports:
            role = "conflicting" if item_conflicts else "background"
        elif all(
            requirement_by_id[requirement_id].role == "bridge"
            for requirement_id in ordered_supports
        ):
            role = "bridge"
        elif item.role == "direct":
            role = "direct"
        else:
            role = "complement"
        metadata["evidence_role_v2"] = role
        promoted.append(replace(
            item,
            role=role,
            supports_requirement_ids=ordered_supports,
            metadata=metadata,
        ))

    reconciled: list[EvidenceItem] = []
    for item in promoted:
        answer_support_ids = {
            requirement_id
            for requirement_id in item.supports_requirement_ids
            if requirement_id in requirement_by_id
            and requirement_by_id[requirement_id].role == "answer"
        }
        if not answer_support_ids:
            reconciled.append(item)
            continue

        # A plan may contain several answer branches while only some of them
        # depend on a bridge (for example, ``普通员工餐补`` plus an unrelated
        # ``请假审批`` question).  The bridge closure rule is an edge-local
        # invariant, not a request-wide mode: applying it to every answer just
        # because *another* branch has a bridge erases independently grounded
        # evidence whenever that bridge is unavailable.  Restrict this second
        # reconciliation pass to the answer nodes that actually declare a
        # bridge dependency.  All answer nodes still pass the common active
        # claim adjudicator below, so this does not weaken evidence safety.
        bridge_bound_answer_ids = {
            requirement_id
            for requirement_id in answer_support_ids
            if bridge_dependencies_by_answer.get(requirement_id, ())
        }
        if not bridge_bound_answer_ids:
            reconciled.append(item)
            continue

        joined_ids = {
            str(value.get("answer_requirement_id") or "")
            for value in item.metadata.get("resolved_bridge_joins", [])
            if isinstance(value, Mapping)
        }
        direct_subject_ids = {
            str(value)
            for value in item.metadata.get(
                "direct_subject_bridge_bypass_requirement_ids",
                [],
            )
            if isinstance(value, str)
        }
        # For a bridge-bound answer, even an exact wording match, model
        # annotation or query provenance is not a relationship path.  It
        # survives only when it joins the resolved, same-scope bridge value.
        retained_bridge_bound_ids = bridge_bound_answer_ids & (
            joined_ids | direct_subject_ids
        )
        rejected_answer_ids = (
            bridge_bound_answer_ids - retained_bridge_bound_ids
        )
        if not rejected_answer_ids:
            metadata = dict(item.metadata)
            metadata["bridge_linked_requirement_ids"] = sorted(
                retained_bridge_bound_ids
            )
            reconciled.append(replace(item, metadata=metadata))
            continue

        supports = tuple(
            requirement_id
            for requirement_id in item.supports_requirement_ids
            if requirement_id not in rejected_answer_ids
        )
        if not supports:
            role = "background"
        elif all(
            requirement_by_id[requirement_id].role == "bridge"
            for requirement_id in supports
        ):
            role = "bridge"
        elif item.role == "direct":
            role = "direct"
        else:
            role = "complement"
        metadata = dict(item.metadata)
        metadata["supports_requirement_ids"] = list(supports)
        metadata["evidence_role_v2"] = role
        metadata["bridge_link_rejected_requirement_ids"] = sorted(
            rejected_answer_ids
        )
        reconciled.append(replace(
            item,
            role=role,
            supports_requirement_ids=supports,
            metadata=metadata,
        ))
    return reconciled


def _role_for_reconciled_supports(
    item: EvidenceItem,
    supports: tuple[str, ...],
    *,
    requirement_by_id: Mapping[str, AnswerRequirementV2],
) -> str:
    if not supports:
        return "background"
    if all(
        requirement_by_id[requirement_id].role == "bridge"
        for requirement_id in supports
    ):
        return "bridge"
    return "direct" if item.role == "direct" else "complement"


def _reconcile_augmentation_bridge_routes(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    task_graph: RetrievalTaskGraph | None,
    task_ledger: TaskExecutionLedger | None,
) -> list[EvidenceItem]:
    """Apply the two non-interchangeable augmentation evidence routes.

    A classification/condition augmentation is allowed to improve recall, not
    to relabel a class-level static row as a direct answer.  An answer can
    therefore survive only through one of these paths:

    * direct source claim: the original subject, target and result occur in
      the same visible source claim (and there is no proof dependency);
    * augmentation join: the row was returned by the exact current-run
      materialised execution and every resolved bridge fact is bound to that
      execution.

    The ledger verifier owns provenance.  The small metadata projection below
    is trace/UI information only; no later decision needs to trust retriever
    supplied metadata for this route.
    """

    augmentation_answers = tuple(
        requirement
        for requirement in requirements
        if requirement.role == "answer"
        and requirement.augmentation_bridge_requirement_ids
    )
    if not augmentation_answers:
        return list(items)
    requirement_by_id = {item.id: item for item in requirements}
    bridge_by_id = {
        item.id: item for item in requirements if item.role == "bridge"
    }
    paths_by_answer_id: dict[str, tuple[AnswerBridgePath, ...]] = {}
    if task_graph is not None and task_ledger is not None:
        for path in task_graph.answer_bridge_paths(mode="augmentation"):
            paths_by_answer_id.setdefault(path.answer_requirement_id, tuple())
            paths_by_answer_id[path.answer_requirement_id] = (
                *paths_by_answer_id[path.answer_requirement_id],
                path,
            )

    output: list[EvidenceItem] = []
    for item in items:
        supports = set(item.supports_requirement_ids)
        metadata = dict(item.metadata)
        existing_joins = [
            value
            for value in metadata.get("resolved_bridge_joins", [])
            if isinstance(value, Mapping)
            and str(value.get("edge_mode") or "proof").casefold() != "augmentation"
        ]
        direct_subject_ids = {
            str(value)
            for value in metadata.get("direct_subject_answer_requirement_ids", [])
            if isinstance(value, str)
        }
        route_joined_ids: set[str] = set()
        for answer in augmentation_answers:
            answer_id = answer.id
            # Remove a preliminary lexical/provenance match before rebuilding
            # it from one of the two trusted routes below.
            supports.discard(answer_id)
            all_bridge_ids = tuple(dict.fromkeys((
                *answer.proof_bridge_requirement_ids,
                *answer.augmentation_bridge_requirement_ids,
            )))
            bridge_subjects = tuple(dict.fromkeys(
                bridge_subject_for_requirement(bridge_by_id[bridge_id])
                for bridge_id in all_bridge_ids
                if bridge_id in bridge_by_id
                and bridge_subject_for_requirement(bridge_by_id[bridge_id])
            ))
            direct_allowed = (
                not answer.proof_bridge_requirement_ids
                and _direct_subject_answer_support(
                    answer.description,
                    item,
                    bridge_subjects=bridge_subjects,
                )
            )
            if direct_allowed:
                supports.add(answer_id)
                direct_subject_ids.add(answer_id)
                continue
            for path in paths_by_answer_id.get(answer_id, ()):
                fact_sets: list[tuple[Any, ...]] = []
                for bridge_task_id, bridge_requirement_id in zip(
                    path.bridge_task_ids,
                    path.bridge_requirement_ids,
                ):
                    resolution = task_ledger.bridge_resolution_for_task(
                        bridge_task_id
                    )
                    facts = tuple(
                        fact
                        for fact in (resolution.facts if resolution is not None else ())
                        if fact.requirement_id == bridge_requirement_id
                    )
                    if resolution is None or resolution.status != "resolved" or not facts:
                        fact_sets = []
                        break
                    fact_sets.append(facts)
                if not fact_sets:
                    continue
                for fact_set in product(*fact_sets):
                    selected_facts = tuple(fact_set)
                    if not candidate_supports_resolved_answer_set(
                        answer,
                        item,
                        selected_facts,
                        bridge_subjects=bridge_subjects,
                        bridge_requirement_ids=path.bridge_requirement_ids,
                        document_root_target_verified=(
                            answer_id in {
                                str(value)
                                for value in metadata.get(
                                    "document_root_answer_requirement_ids",
                                    [],
                                )
                                if isinstance(value, str)
                            }
                        ),
                    ):
                        continue
                    bindings = task_ledger.resolved_answer_bridge_bindings(
                        item.to_dict(),
                        path=path,
                        facts=selected_facts,
                    )
                    if len(bindings) != len(path.bridge_requirement_ids):
                        continue
                    supports.add(answer_id)
                    route_joined_ids.add(answer_id)
                    existing_joins.extend(
                        {
                            "answer_requirement_id": answer_id,
                            "bridge_requirement_id": binding.bridge_requirement_id,
                            "bridge_value": binding.bridge_value,
                            "bridge_source_chunk_id": binding.bridge_source_item_id,
                            "edge_mode": binding.edge_mode,
                            "bridge_execution_id": binding.bridge_execution_id,
                            "answer_execution_id": binding.answer_execution_id,
                        }
                        for binding in bindings
                    )
                    break
                if answer_id in route_joined_ids:
                    break

        ordered_supports = tuple(
            requirement.id
            for requirement in requirements
            if requirement.id in supports
        )
        metadata["supports_requirement_ids"] = list(ordered_supports)
        metadata["evidence_role_v2"] = _role_for_reconciled_supports(
            item,
            ordered_supports,
            requirement_by_id=requirement_by_id,
        )
        if existing_joins:
            metadata["resolved_bridge_joins"] = existing_joins[:16]
        else:
            metadata.pop("resolved_bridge_joins", None)
        if direct_subject_ids:
            metadata["direct_subject_answer_requirement_ids"] = sorted(
                direct_subject_ids
            )
        else:
            metadata.pop("direct_subject_answer_requirement_ids", None)
        if route_joined_ids:
            metadata["augmentation_bridge_linked_requirement_ids"] = sorted(
                route_joined_ids
            )
        else:
            metadata.pop("augmentation_bridge_linked_requirement_ids", None)
        output.append(replace(
            item,
            role=metadata["evidence_role_v2"],
            supports_requirement_ids=ordered_supports,
            metadata=metadata,
        ))
    return output


def _resolved_fact_sets_for_path(
    *,
    task_ledger: TaskExecutionLedger,
    path: AnswerBridgePath,
) -> tuple[tuple[Any, ...], ...]:
    """Return only complete, request-local fact combinations for one path.

    The scheduler resolves bridge tasks independently.  Evidence must retain
    that ownership rather than flattening all discovered facts by requirement
    id: two documents can contain different classifications, and one answer
    candidate is valid only for one exact combination of parent facts.
    """

    facts_by_task: list[tuple[Any, ...]] = []
    for bridge_task_id, bridge_requirement_id in zip(
        path.bridge_task_ids,
        path.bridge_requirement_ids,
    ):
        resolution = task_ledger.bridge_resolution_for_task(bridge_task_id)
        if resolution is None or resolution.status != "resolved":
            return ()
        matching = tuple(
            fact
            for fact in resolution.facts
            if fact.requirement_id == bridge_requirement_id
        )
        if not matching:
            return ()
        facts_by_task.append(matching)
    return tuple(tuple(values) for values in product(*facts_by_task))


def _verified_route_bindings_for_item(
    item: EvidenceItem,
    *,
    answer: AnswerRequirementV2,
    path: AnswerBridgePath,
    task_ledger: TaskExecutionLedger,
) -> tuple[BridgeClaimBinding, ...]:
    """Return one ledger-verified bridge route for an answer source.

    This is deliberately the single gateway from retrieval provenance into
    evidence support.  A merged candidate lineage, a persisted metadata row,
    or a static first-wave D-level rule is not enough: the candidate must be
    observed by an exact current-run second-hop/same-source execution whose
    parent task and source chunks equal the resolved bridge path.
    """

    for facts in _resolved_fact_sets_for_path(
        task_ledger=task_ledger,
        path=path,
    ):
        bridge_subjects = tuple(dict.fromkeys(
            fact.subject for fact in facts if str(fact.subject or "").strip()
        ))
        if not candidate_supports_resolved_answer_set(
            answer,
            item,
            facts,
            bridge_subjects=bridge_subjects,
            bridge_requirement_ids=path.bridge_requirement_ids,
            document_root_target_verified=(
                answer.id in {
                    str(value)
                    for value in item.metadata.get(
                        "document_root_answer_requirement_ids",
                        [],
                    )
                    if isinstance(value, str)
                }
            ),
        ):
            continue
        bindings = task_ledger.resolved_answer_bridge_bindings(
            item.to_dict(),
            path=path,
            facts=facts,
        )
        if len(bindings) == len(path.bridge_requirement_ids):
            return bindings
    return ()


def _project_ledgered_bridge_requirement_supports(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    task_graph: RetrievalTaskGraph,
    task_ledger: TaskExecutionLedger,
) -> list[EvidenceItem]:
    """Project one request's bridge semantic outcomes onto visible evidence.

    Candidate mapping is intentionally recall-oriented: a row may be returned
    by a bridge retrieval before the scheduler determines whether that row is
    the *one* applicable relationship.  ``BridgeResolution`` is the semantic
    authority for that determination.  Treating its result as an answer-route
    gate but leaving the original bridge row positively labelled creates two
    competing truths: provisional bridge rows can enter ``answer_source_ids``
    and ambiguity candidates even after the ledger has rejected them.

    This is the single projection boundary between the execution ledger and
    renderer-facing evidence:

    * only an exact, still-visible fact from a ``resolved`` bridge task can
      retain its bridge requirement support;
    * ``conflict``, ``no_fact``, ``failed``, ``budget_skipped`` and a missing
      terminal result remove that support from every candidate;
    * conflict source rows retain bounded diagnostics, but never a positive
      role unless they independently support a different requirement.

    It deliberately applies equally to proof and augmentation edges.  Those
    edges differ only in whether an answer may have a direct alternative; a
    bridge fact itself is never allowed to remain positively asserted after
    its own task has failed closed.
    """

    requirement_by_id = {requirement.id: requirement for requirement in requirements}
    bridge_requirement_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }
    if not bridge_requirement_by_id:
        return list(items)

    bridge_task_by_requirement_id = {
        task.target_requirement_ids[0]: task
        for task in task_graph.tasks
        if task.role == "bridge"
    }
    resolution_by_requirement_id = {
        requirement_id: task_ledger.bridge_resolution_for_task(task.task_id)
        for requirement_id, task in bridge_task_by_requirement_id.items()
    }

    conflict_values_by_requirement_and_chunk: dict[
        tuple[str, str], tuple[str, ...]
    ] = {}
    for requirement_id, resolution in resolution_by_requirement_id.items():
        if resolution is None or resolution.status != "conflict":
            continue
        values_by_chunk: dict[str, set[str]] = defaultdict(set)
        for conflict in resolution.conflicts:
            for chunk_id in conflict.source_chunk_ids:
                values_by_chunk[chunk_id].update(conflict.values)
        for chunk_id, values in values_by_chunk.items():
            conflict_values_by_requirement_and_chunk[(
                requirement_id,
                chunk_id,
            )] = tuple(sorted(values, key=str.casefold))

    projected: list[EvidenceItem] = []
    for item in items:
        supports = set(item.supports_requirement_ids)
        metadata = dict(item.metadata)
        # These annotations are request-local projection products.  Remove
        # stale/retriever-provided values before rebuilding them below.
        metadata.pop("bridge_resolution_statuses", None)
        metadata.pop("bridge_projection_rejected_requirement_ids", None)
        metadata.pop("bridge_conflicts", None)

        statuses: dict[str, dict[str, str]] = {}
        rejected_requirement_ids: set[str] = set()
        item_conflicts: list[dict[str, Any]] = []
        for requirement_id in tuple(supports):
            bridge_requirement = bridge_requirement_by_id.get(requirement_id)
            if bridge_requirement is None:
                continue
            task = bridge_task_by_requirement_id.get(requirement_id)
            resolution = resolution_by_requirement_id.get(requirement_id)
            status = resolution.status if resolution is not None else "missing"
            statuses[requirement_id] = {
                "bridge_task_id": task.task_id if task is not None else "missing",
                "status": status,
            }

            fact_is_current_and_visible = bool(
                resolution is not None
                and resolution.status == "resolved"
                and any(
                    _fact_still_matches_visible_source(
                        fact=fact,
                        bridge_requirement=bridge_requirement,
                        item=item,
                    )
                    for fact in resolution.facts
                    if fact.requirement_id == requirement_id
                    and fact.source_chunk_id == item.chunk_id
                )
            )
            if fact_is_current_and_visible:
                continue

            supports.discard(requirement_id)
            rejected_requirement_ids.add(requirement_id)
            conflict_values = conflict_values_by_requirement_and_chunk.get(
                (requirement_id, item.chunk_id),
            )
            if conflict_values:
                item_conflicts.append({
                    "bridge_requirement_id": requirement_id,
                    "values": list(conflict_values),
                })

        ordered_supports = tuple(
            requirement.id
            for requirement in requirements
            if requirement.id in supports
        )
        if statuses:
            metadata["bridge_resolution_statuses"] = statuses
        if rejected_requirement_ids:
            metadata["bridge_projection_rejected_requirement_ids"] = sorted(
                rejected_requirement_ids
            )
        if item_conflicts:
            metadata["bridge_conflicts"] = item_conflicts

        # A conflict is local to the rejected bridge claim.  If the same
        # source independently proves another answer requirement, preserve
        # that independent claim rather than allowing an unrelated bridge
        # conflict to erase it.
        if not ordered_supports:
            role = "conflicting" if item_conflicts else "background"
        else:
            role = _role_for_reconciled_supports(
                item,
                ordered_supports,
                requirement_by_id=requirement_by_id,
            )
        metadata["supports_requirement_ids"] = list(ordered_supports)
        metadata["evidence_role_v2"] = role
        projected.append(replace(
            item,
            role=role,
            supports_requirement_ids=ordered_supports,
            metadata=metadata,
        ))
    return projected


def _reconcile_ledgered_bridge_routes(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    task_graph: RetrievalTaskGraph,
    task_ledger: TaskExecutionLedger,
) -> list[EvidenceItem]:
    """Rebuild every bridge-dependent answer route from the execution ledger.

    ``_mapped_item`` is intentionally recall-friendly.  It may find an
    answer-shaped row during the literal first wave, but it cannot say whether
    a row for ``D级`` applies to ``普通员工``.  This reconciler is the semantic
    boundary: proof and augmentation routes are evaluated from the exact same
    typed graph/ledger pair, and only then projected to rendering metadata.

    The projection is diagnostic/UI data.  The coverage graph independently
    recomputes the same bindings from this ledger later, so no retriever
    metadata can become proof by surviving this function.
    """

    requirement_by_id = {requirement.id: requirement for requirement in requirements}
    # First project the ledger's one terminal bridge outcome into the
    # renderer-facing item set.  Answer-route reconciliation below must never
    # see a provisional mapping row that the bridge scheduler already rejected.
    items = _project_ledgered_bridge_requirement_supports(
        items,
        requirements=requirements,
        task_graph=task_graph,
        task_ledger=task_ledger,
    )
    bridge_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }
    paths_by_answer_id: dict[str, tuple[AnswerBridgePath, ...]] = {}
    for mode in ("proof", "augmentation"):
        for path in task_graph.answer_bridge_paths(mode=mode):
            paths_by_answer_id[path.answer_requirement_id] = (
                *paths_by_answer_id.get(path.answer_requirement_id, ()),
                path,
            )

    output: list[EvidenceItem] = []
    for item in items:
        supports = set(item.supports_requirement_ids)
        metadata = dict(item.metadata)
        for field in (
            "resolved_bridge_joins",
            "bridge_linked_requirement_ids",
            "bridge_link_rejected_requirement_ids",
            "augmentation_bridge_linked_requirement_ids",
            "direct_subject_answer_requirement_ids",
            "direct_subject_bridge_bypass_requirement_ids",
        ):
            metadata.pop(field, None)
        joins: list[dict[str, str]] = []
        direct_subject_answer_ids: set[str] = set()
        linked_answer_ids: set[str] = set()

        for answer_id, paths in paths_by_answer_id.items():
            answer = requirement_by_id[answer_id]
            # Any mapper/lexical support is provisional whenever an answer
            # declares a bridge edge.  It is restored only through a direct
            # original-subject claim (augmentation only) or a verified route.
            supports.discard(answer_id)
            proof_ids = answer.proof_bridge_requirement_ids
            all_bridge_ids = tuple(dict.fromkeys((
                *proof_ids,
                *answer.augmentation_bridge_requirement_ids,
            )))
            bridge_subjects = tuple(dict.fromkeys(
                bridge_subject_for_requirement(bridge_by_id[bridge_id])
                for bridge_id in all_bridge_ids
                if bridge_id in bridge_by_id
                and bridge_subject_for_requirement(bridge_by_id[bridge_id])
            ))

            # Optional augmentation may never obstruct a direct statement of
            # the original subject.  A proof edge deliberately has no such
            # bypass: declaring it means the relation itself is required.
            if (
                not proof_ids
                and _direct_subject_answer_support(
                    answer.description,
                    item,
                    bridge_subjects=bridge_subjects,
                )
            ):
                supports.add(answer_id)
                direct_subject_answer_ids.add(answer_id)

            for path in paths:
                bindings = _verified_route_bindings_for_item(
                    item,
                    answer=answer,
                    path=path,
                    task_ledger=task_ledger,
                )
                if not bindings:
                    continue
                supports.add(answer_id)
                linked_answer_ids.add(answer_id)
                joins.extend({
                    "answer_requirement_id": answer_id,
                    **binding.to_dict(),
                } for binding in bindings)

        ordered_supports = tuple(
            requirement.id
            for requirement in requirements
            if requirement.id in supports
        )
        metadata["supports_requirement_ids"] = list(ordered_supports)
        # Preserve the projection's conflict state when the item has no
        # surviving positive assertion.  A later route pass may not silently
        # downgrade it to generic background, because trace/search consumers
        # need to distinguish "not relevant" from "relevant but internally
        # contradictory" without allowing either to enter answer sources.
        metadata["evidence_role_v2"] = (
            "conflicting"
            if item.role == "conflicting" and not ordered_supports
            else _role_for_reconciled_supports(
                item,
                ordered_supports,
                requirement_by_id=requirement_by_id,
            )
        )
        if joins:
            metadata["resolved_bridge_joins"] = joins[:24]
        if direct_subject_answer_ids:
            metadata["direct_subject_answer_requirement_ids"] = sorted(
                direct_subject_answer_ids
            )
        if linked_answer_ids:
            metadata["bridge_linked_requirement_ids"] = sorted(linked_answer_ids)
        output.append(replace(
            item,
            role=metadata["evidence_role_v2"],
            supports_requirement_ids=ordered_supports,
            metadata=metadata,
        ))
    return output


def _reconcile_unledgered_proof_dependencies(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
) -> list[EvidenceItem]:
    """Fail closed for proof edges when no current-run ledger exists.

    A historical direct-call path previously re-parsed the mixed candidate
    pool and treated any D-level row as if it had been returned by a current
    bridge second hop.  That is not an alternate implementation of a proof
    edge; it is missing provenance.  Keep non-bridge/direct evidence for
    diagnostics, but remove only hard-proof answers until the caller migrates
    to the ledgered execution bundle.
    """

    requirement_by_id = {requirement.id: requirement for requirement in requirements}
    proof_answer_ids = {
        requirement.id
        for requirement in requirements
        if requirement.role == "answer"
        and requirement.proof_bridge_requirement_ids
    }
    if not proof_answer_ids:
        return list(items)
    output: list[EvidenceItem] = []
    for item in items:
        supports = tuple(
            requirement_id
            for requirement_id in item.supports_requirement_ids
            if requirement_id not in proof_answer_ids
        )
        metadata = dict(item.metadata)
        metadata["supports_requirement_ids"] = list(supports)
        metadata["evidence_role_v2"] = _role_for_reconciled_supports(
            item,
            supports,
            requirement_by_id=requirement_by_id,
        )
        if set(item.supports_requirement_ids) & proof_answer_ids:
            metadata["proof_route_provenance_missing"] = sorted(
                set(item.supports_requirement_ids) & proof_answer_ids
            )
        output.append(replace(
            item,
            role=metadata["evidence_role_v2"],
            supports_requirement_ids=supports,
            metadata=metadata,
        ))
    return output


def _is_document_policy_snapshot_member(
    item: EvidenceItem,
    *,
    requirement: AnswerRequirementV2,
    overview_requested: bool,
) -> bool:
    """Return whether a chunk may assert one member of a policy snapshot.

    An overview is not a looser fact lookup.  Its positive evidence is the
    *bounded source document* and therefore needs a distinct, typed assertion
    for every visible snapshot member.  The request-local root annotation is
    produced only by ``_attach_document_root_topic_anchors`` after the current
    query seed, title/topic match and same-document expansion boundary have
    all been verified.  Raw retriever metadata cannot supply that annotation:
    ``_metadata`` strips it before this stage.

    This helper intentionally does not decide that the snapshot is complete.
    ``evidence_snapshots.complete_document_keys`` and the final visible graph
    do that against the exact post-budget item set.  Keeping admission and
    closure separate makes an incomplete snapshot fail closed instead of
    turning a partial document into a fabricated policy answer.
    """

    if (
        not overview_requested
        or not requirement.requires_document_policy_snapshot
    ):
        return False
    rooted_requirement_ids = {
        str(value).strip()
        for value in item.metadata.get("document_root_answer_requirement_ids", ())
        if isinstance(value, str) and str(value).strip()
    }
    if requirement.id not in rooted_requirement_ids:
        return False
    try:
        expected_chunk_count = int(
            item.metadata.get("full_document_chunk_count")
        )
    except (TypeError, ValueError):
        return False
    return expected_chunk_count > 0 and 0 <= item.chunk_index < expected_chunk_count


def _reconcile_answer_claim_assertions(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    overview_requested: bool,
    terminology_resolution: TerminologyRuntimeResolution | None = None,
) -> list[EvidenceItem]:
    """Make active source claims the final boundary for answer support.

    Lexical alignment, retrieval provenance and verifier annotations can make a
    chunk a candidate for a requirement; none of them can turn a reference,
    revoked rule or tentative statement into an answer.  This reconciliation
    is shared by direct and bridge-linked answers and runs again after the
    renderer budget so stale metadata cannot survive truncation.
    """

    answer_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "answer"
    }
    bridge_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }
    output: list[EvidenceItem] = []
    for item in items:
        supports = set(item.supports_requirement_ids)
        metadata = dict(item.metadata)
        assertion_metadata: dict[str, list[dict[str, str]]] = {}
        rejected_answer_ids: list[str] = []
        joins = tuple(
            value
            for value in metadata.get("resolved_bridge_joins", [])
            if isinstance(value, Mapping)
        )
        for requirement_id in tuple(supports):
            answer = answer_by_id.get(requirement_id)
            if answer is None:
                continue
            # The user asked for the governing document, not a single scalar
            # hidden in one of its sections.  Once this member has passed the
            # rooted snapshot predicate, its semantic result must remain the
            # policy-member assertion even if its text also contains a price,
            # duration or other ordinary fact.  Otherwise unrelated sections
            # would be compared as competing answers to one document overview.
            if _is_document_policy_snapshot_member(
                item,
                requirement=answer,
                overview_requested=overview_requested,
            ):
                document_key = f"{item.kb_id}:{item.doc_id}"
                assertion_metadata[requirement_id] = [{
                    "status": "active",
                    "result_kind": "document_policy",
                    "normalized_result": (
                        f"document_policy_member:{document_key}:{item.chunk_index}"
                    ),
                    "claim_key": (
                        f"document_policy:{requirement_id}:{document_key}"
                    ),
                }]
                continue
            # Claim adjudication needs every bridge dimension that the
            # reconciler has attached to this exact source.  The old
            # proof-only lookup made an augmentation-resolved D-level row
            # indistinguishable from a direct "普通员工" row, so a static
            # first-wave candidate could bypass the second-hop provenance
            # gate.  Edge modes remain distinct for route validation; here
            # we only collect their canonical subjects/values for source
            # claim parsing.
            dependency_ids = tuple(dict.fromkeys((
                *bridge_requirement_ids_for_answer(
                    answer,
                    requirements,
                    mode="proof",
                ),
                *bridge_requirement_ids_for_answer(
                    answer,
                    requirements,
                    mode="augmentation",
                ),
            )))
            bridge_subjects = tuple(dict.fromkeys(
                subject
                for dependency_id in dependency_ids
                for bridge in (bridge_by_id.get(dependency_id),)
                if bridge is not None
                if (subject := bridge_subject_for_requirement(bridge))
            ))
            bridge_values = tuple(dict.fromkeys(
                str(join.get("bridge_value") or "").strip()
                for join in joins
                if str(join.get("answer_requirement_id") or "") == requirement_id
                and str(join.get("bridge_value") or "").strip()
            ))
            root_verified = requirement_id in {
                str(value)
                for value in metadata.get(
                    "document_root_answer_requirement_ids",
                    [],
                )
                if isinstance(value, str)
            }
            # Configuration blocks carry a stronger typed claim than a
            # generic collection marker.  Preserve the path/value assertion
            # so the evidence graph can compare conflicting assignments and
            # the renderer can explain exactly which setting was used.  This
            # runs before the generic collection-closure projection because
            # a YAML/properties block may be complete without numbered steps.
            config_assertions = adjudicate_answer_claims(
                answer.description,
                item.content,
                bridge_subjects=bridge_subjects,
                bridge_values=bridge_values,
                has_bridge_edge=bool(
                    answer.proof_bridge_requirement_ids
                    or answer.augmentation_bridge_requirement_ids
                ),
                document_root_target_verified=root_verified,
            )
            config_assertions = tuple(
                assertion
                for assertion in config_assertions
                if assertion.result_kind == "config_assignment"
                and assertion.status == "active"
            )
            if config_assertions:
                assertion_metadata[requirement_id] = [
                    {
                        "status": assertion.status,
                        "result_kind": assertion.result_kind,
                        "normalized_result": assertion.normalized_result,
                        "claim_key": assertion.claim_key,
                    }
                    for assertion in config_assertions[:12]
                ]
                continue
            # Collection contracts own their source structure.  When the
            # current source explicitly closes a list or ordered procedure,
            # emit that typed assertion before running the scalar/categorical
            # parser.  Otherwise incidental values such as "3天", "5天" or a
            # publication year can pre-empt the real procedure and leave the
            # final graph with unrelated scalar claims but no collection
            # closure.  This is contract-driven and applies to every
            # collection/process question; it does not depend on business
            # keywords or model availability.
            explicit_collection_closure = bool(
                answer.requires_collection_closure
                and has_explicit_collection_closure(
                    item,
                    requirement=answer,
                    requirements=requirements,
                )
            )
            if explicit_collection_closure:
                assertion_metadata[requirement_id] = [{
                    "status": "active",
                    "result_kind": answer.effective_coverage_contract,
                    "normalized_result": (
                        f"{answer.effective_coverage_contract}:{item.chunk_id}"
                    ),
                    "claim_key": (
                        f"{answer.effective_coverage_contract}:{requirement_id}"
                    ),
                }]
                continue

            if answer.effective_coverage_contract == "ordered_steps":
                # An ordered procedure is answerable only through an explicit
                # source-authored sequence closure.  A nearby purpose clause,
                # deadline or duration may be relevant background, but it is
                # not one member of an independently mergeable procedure.
                # Reject it here so one non-procedural scalar claim cannot join
                # the real process chunk and invalidate the single declared
                # sequence certificate in the final graph.
                supports.discard(requirement_id)
                rejected_answer_ids.append(requirement_id)
                continue

            assertions = adjudicate_answer_claims(
                answer.description,
                item.content,
                bridge_subjects=bridge_subjects,
                bridge_values=bridge_values,
                has_bridge_edge=bool(
                    answer.proof_bridge_requirement_ids
                    or answer.augmentation_bridge_requirement_ids
                ),
                document_root_target_verified=root_verified,
            )
            if not assertions and terminology_resolution is not None:
                terminology_match = terminology_resolution.evidence_match(
                    requirement=answer,
                    kb_id=item.kb_id,
                    doc_id=item.doc_id,
                    content=item.content,
                )
                if terminology_match is not None:
                    assertions = adjudicate_answer_claims(
                        terminology_match.requirement.description,
                        item.content,
                        bridge_subjects=bridge_subjects,
                        bridge_values=bridge_values,
                        has_bridge_edge=bool(
                            answer.proof_bridge_requirement_ids
                            or answer.augmentation_bridge_requirement_ids
                        ),
                        document_root_target_verified=root_verified,
                    )
                    if assertions:
                        # The mapper may have been bypassed by a bounded
                        # structural companion.  Preserve the strict proof
                        # provenance here as well, so every active rewritten
                        # claim is explicitly auditable at the graph boundary.
                        proof_kinds = dict(
                            metadata.get("claim_proof_kind", {})
                            if isinstance(metadata.get("claim_proof_kind"), Mapping)
                            else {}
                        )
                        proof_rules = dict(
                            metadata.get("strict_terminology_rule_ids", {})
                            if isinstance(
                                metadata.get("strict_terminology_rule_ids"),
                                Mapping,
                            )
                            else {}
                        )
                        proof_kinds[requirement_id] = "terminology_strict"
                        proof_rules[requirement_id] = list(
                            terminology_match.rule_ids
                        )
                        metadata["claim_proof_kind"] = proof_kinds
                        metadata["strict_terminology_rule_ids"] = proof_rules
            if (
                not assertions
                and requirement_id not in assertion_metadata
            ):
                supports.discard(requirement_id)
                rejected_answer_ids.append(requirement_id)
                continue
            if assertions:
                assertion_metadata[requirement_id] = [
                    {
                        "status": assertion.status,
                        "result_kind": assertion.result_kind,
                        "normalized_result": assertion.normalized_result,
                        "claim_key": assertion.claim_key,
                    }
                    for assertion in assertions[:12]
                ]

        ordered_supports = tuple(
            requirement.id
            for requirement in requirements
            if requirement.id in supports
        )
        metadata["supports_requirement_ids"] = list(ordered_supports)
        metadata["answer_claim_assertions"] = assertion_metadata
        if rejected_answer_ids:
            metadata["claim_rejected_requirement_ids"] = sorted(
                rejected_answer_ids
            )
        else:
            metadata.pop("claim_rejected_requirement_ids", None)
        # A bridge conflict invalidates that bridge claim, not every other
        # independently grounded assertion in the same source item.  The
        # ledger projection has already assigned ``conflicting`` when no
        # positive support survives; do not re-promote a metadata diagnostic
        # into a request-wide veto here.
        if item.role == "conflicting":
            role = "conflicting"
        elif not ordered_supports:
            role = "background"
        elif all(
            bridge_by_id.get(requirement_id) is not None
            for requirement_id in ordered_supports
        ):
            role = "bridge"
        elif item.role == "direct":
            role = "direct"
        else:
            role = "complement"
        metadata["evidence_role_v2"] = role
        output.append(replace(
            item,
            role=role,
            supports_requirement_ids=ordered_supports,
            metadata=metadata,
        ))
    return output


def _reconcile_same_source_answer_conflicts(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
) -> tuple[list[EvidenceItem], bool]:
    """Fail closed on contradictory scalar/category claims in one scope.

    Different documents remain independent answer graphs and are handled by
    the post-evidence document/scope clarification protocol.  Within one
    source document and one declared applicability slice, however, silently
    choosing between two active values is never safe.
    """

    answer_ids = {
        requirement.id
        for requirement in requirements
        if requirement.role == "answer"
        # Multiple categorical values are the expected result of an
        # exhaustive collection, not mutually exclusive scalar claims.
        # Collection completeness is adjudicated separately using source
        # closure/cardinality proofs below.
        and requirement.coverage_mode == "single"
    }
    grouped: dict[
        tuple[
            str,
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            str,
        ],
        list[tuple[str, str]],
    ] = defaultdict(list)
    for item in items:
        identity = extract_document_constraint_identity(item.to_dict())
        products = identity.canonical_products or identity.products
        assertion_map = item.metadata.get("answer_claim_assertions")
        if not isinstance(assertion_map, Mapping):
            continue
        for requirement_id, raw_assertions in assertion_map.items():
            if (
                requirement_id not in answer_ids
                or requirement_id not in item.supports_requirement_ids
                or not isinstance(raw_assertions, list)
            ):
                continue
            for raw in raw_assertions:
                if not isinstance(raw, Mapping):
                    continue
                result_kind = str(raw.get("result_kind") or "")
                normalized_result = str(raw.get("normalized_result") or "")
                claim_key = str(raw.get("claim_key") or "")
                if (
                    result_kind not in {"scalar", "categorical"}
                    or not normalized_result
                    or not claim_key
                ):
                    continue
                key = (
                    requirement_id,
                    item.kb_id,
                    item.doc_id,
                    tuple(sorted(products, key=str.casefold)),
                    tuple(sorted(identity.versions)),
                    tuple(sorted(identity.projects, key=str.casefold)),
                    claim_key,
                )
                grouped[key].append((item.chunk_id, normalized_result))

    conflicting_pairs: set[tuple[str, str]] = set()
    for key, claims in grouped.items():
        values = {value for _, value in claims}
        if len(values) <= 1:
            continue
        requirement_id = key[0]
        conflicting_pairs.update(
            (chunk_id, requirement_id) for chunk_id, _ in claims
        )
    if not conflicting_pairs:
        return list(items), False

    reconciled: list[EvidenceItem] = []
    for item in items:
        rejected_ids = {
            requirement_id
            for chunk_id, requirement_id in conflicting_pairs
            if chunk_id == item.chunk_id
        }
        if not rejected_ids:
            reconciled.append(item)
            continue
        supports = tuple(
            requirement_id
            for requirement_id in item.supports_requirement_ids
            if requirement_id not in rejected_ids
        )
        metadata = dict(item.metadata)
        metadata["supports_requirement_ids"] = list(supports)
        metadata["conflicting_answer_requirement_ids"] = sorted(rejected_ids)
        metadata["evidence_role_v2"] = "conflicting"
        reconciled.append(replace(
            item,
            role="conflicting",
            supports_requirement_ids=supports,
            metadata=metadata,
        ))
    return reconciled, True


def _query_requirement_alignment(
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    retrieval_queries: tuple[str, ...],
) -> dict[int, str]:
    if not requirements or not retrieval_queries:
        return {}
    if len(requirements) == len(retrieval_queries):
        return {
            index: requirements[index].id
            for index in range(len(retrieval_queries))
        }
    if len(requirements) == 1:
        return {index: requirements[0].id for index in range(len(retrieval_queries))}

    result: dict[int, str] = {}
    for index, query in enumerate(retrieval_queries):
        exact_ids = [
            item.id
            for item in requirements
            if re.sub(r"\s+", " ", item.description).strip().casefold()
            == query.casefold()
        ]
        if len(exact_ids) == 1:
            result[index] = exact_ids[0]
            continue
        scored: list[tuple[int, str]] = []
        query_terms = _coverage_terms(query)
        for item in requirements:
            supported, _ = _text_support(item.description, query)
            if not supported:
                continue
            overlap_count = len(query_terms & _coverage_terms(item.description))
            scored.append((overlap_count, item.id))
        scored.sort(reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            result[index] = scored[0][1]
    return result


def _candidate_query_indexes(
    item: EvidenceItem,
    *,
    query_count: int,
) -> tuple[int, ...]:
    raw = item.metadata.get("expansion_query_indexes")
    if not isinstance(raw, (list, tuple)):
        return ()
    indexes: list[int] = []
    seen: set[int] = set()
    for value in raw:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            index = value
        elif isinstance(value, str) and value.strip().isdigit():
            index = int(value.strip())
        else:
            continue
        if index < 0 or index >= query_count or index in seen:
            continue
        seen.add(index)
        indexes.append(index)
    return tuple(indexes)


def _mapped_item(
    item: EvidenceItem,
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    retrieval_queries: tuple[str, ...],
    query_alignment: Mapping[int, str],
    trust_candidate_annotations: bool,
    task_graph: RetrievalTaskGraph | None = None,
    task_ledger: TaskExecutionLedger | None = None,
    terminology_resolution: TerminologyRuntimeResolution | None = None,
    use_query_indexes: bool = True,
    use_structural_provenance: bool = True,
    overview_requested: bool = False,
) -> EvidenceItem:
    if not requirements:
        supports = (
            item.supports_requirement_ids
            if trust_candidate_annotations
            else ()
        )
        role = item.role if supports else "background"
        metadata = dict(item.metadata)
        metadata["supports_requirement_ids"] = list(supports)
        metadata["evidence_role_v2"] = role
        return replace(
            item,
            role=role,
            supports_requirement_ids=supports,
            metadata=metadata,
        )

    requirement_by_id = {requirement.id: requirement for requirement in requirements}
    allowed_ids = set(requirement_by_id)
    task_requirement_ids: frozenset[str] | None = None
    task_ids: tuple[str, ...] = ()
    task_binding_invalid = False
    if task_graph is not None:
        task_ids, task_binding_invalid = _candidate_task_ids(
            item,
            task_graph=task_graph,
            task_ledger=task_ledger,
        )
        task_requirement_ids = (
            task_graph.requirement_ids_reachable_from(task_ids)
            if task_ids
            else frozenset()
        ) & frozenset(allowed_ids)
    explicit_ids = (
        tuple(
            requirement_id
            for requirement_id in item.supports_requirement_ids
            if requirement_id in allowed_ids
        )
        if trust_candidate_annotations
        else ()
    )
    explicit_ids = tuple(
        requirement_id
        for requirement_id in explicit_ids
        if _candidate_matches_requirement_scope(
            requirement_by_id[requirement_id],
            item,
        )
    )
    if task_requirement_ids is not None:
        # A verifier annotation cannot widen a task's ownership.  It may only
        # refine a candidate already returned for that logical task.
        explicit_ids = tuple(
            requirement_id
            for requirement_id in explicit_ids
            if requirement_id in task_requirement_ids
        )
    aligned_ids: tuple[str, ...] = ()
    if task_graph is None and use_query_indexes:
        aligned_ids = tuple(dict.fromkeys(
            query_alignment[index]
            for index in _candidate_query_indexes(
                item,
                query_count=len(retrieval_queries),
            )
            if index in query_alignment
        ))
    structural_ids: tuple[str, ...] = ()
    required_answer_ids = tuple(
        requirement.id
        for requirement in requirements
        if requirement.is_required_answer
    )
    bridge_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }
    bridge_subjects_by_answer = {
        requirement.id: tuple(dict.fromkeys(
            subject
            for dependency_id in bridge_dependency_ids_for_answer(
                requirement,
                requirements,
            )
            for bridge in (bridge_by_id.get(dependency_id),)
            if bridge is not None
            if (subject := bridge_subject_for_requirement(bridge))
        ))
        for requirement in requirements
        if requirement.role == "answer"
    }
    if (
        use_structural_provenance
        and len(required_answer_ids) == 1
        and len(requirements) == 1
        and overview_requested
    ):
        current_query_seed = "initial_retrieval" in item.origins
        bounded_overview_chunk = bool(
            "overview_full_document" in item.origins
        )
        if current_query_seed or bounded_overview_chunk:
            # An explicit overview is the one structural query whose answer is
            # the bounded document itself.  Its authorized retrieval anchor and
            # full-document siblings may therefore support the overview without
            # each chunk repeating the query words.  Fact and multi-hop paths
            # never receive support from provenance.
            structural_ids = (required_answer_ids[0],)

    if task_requirement_ids is not None:
        # Query indexes are a legacy positional projection and are explicitly
        # ignored once a task graph is present.  Candidates without a valid
        # task owner cannot receive provenance-only/structural support; their
        # visible text is still evaluated independently by the claim checks.
        if task_binding_invalid or not task_ids:
            aligned_ids = ()
            structural_ids = ()
        else:
            structural_ids = tuple(
                requirement_id
                for requirement_id in structural_ids
                if requirement_id in task_requirement_ids
            )

    lexical_ids: list[str] = []
    exact_ids: set[str] = set()
    # A strict terminology equivalence is not a fuzzy lexical fallback.  This
    # map is populated only after the runtime registry has confirmed the
    # candidate's KB/document/scope and a reviewed strict form occurs in its
    # source text.  ``retrieval_only`` has no path into this map.
    strict_terminology_rule_ids: dict[str, tuple[str, ...]] = {}
    for requirement in requirements:
        if not _candidate_matches_requirement_scope(requirement, item):
            continue
        supported, exact = (
            _bridge_text_support(requirement, item.content)
            if requirement.role == "bridge"
            else _text_support(requirement.description, item.content)
        )
        if (
            not supported
            and requirement.role == "answer"
            and _declarative_answer_support(
                requirement.description,
                item,
                bridge_subjects=bridge_subjects_by_answer.get(
                    requirement.id,
                    (),
                ),
            )
        ):
            supported = True
        if (
            not supported
            and requirement.role == "answer"
            and terminology_resolution is not None
        ):
            terminology_match = terminology_resolution.evidence_match(
                requirement=requirement,
                kb_id=item.kb_id,
                doc_id=item.doc_id,
                content=item.content,
            )
            if terminology_match is not None:
                rewritten_description = terminology_match.requirement.description
                supported, exact = _text_support(
                    rewritten_description,
                    item.content,
                )
                if (
                    not supported
                    and _declarative_answer_support(
                        rewritten_description,
                        item,
                        bridge_subjects=bridge_subjects_by_answer.get(
                            requirement.id,
                            (),
                        ),
                    )
                ):
                    supported = True
                if supported:
                    strict_terminology_rule_ids[requirement.id] = (
                        terminology_match.rule_ids
                    )
        if not supported:
            continue
        lexical_ids.append(requirement.id)
        if exact:
            exact_ids.add(requirement.id)

    lexical_bridge_ids = {
        requirement_id
        for requirement_id in lexical_ids
        if requirement_by_id[requirement_id].role == "bridge"
    }
    if len(required_answer_ids) > 1:
        scope_disambiguated_ids = _task_scope_disambiguated_answer_ids(
            requirements=requirements,
            item=item,
            task_graph=task_graph,
            task_ids=task_ids,
            # ``explicit_ids`` is populated only when the caller has enabled
            # trusted typed annotations.  In the normal V2 path this is empty,
            # so source text remains the sole positive claim signal.
            positive_target_ids=set(lexical_ids) | set(explicit_ids),
        )
        visible_answer_ids = _multi_answer_text_support_ids(
            requirements=requirements,
            content=item.content,
            lexical_ids=lexical_ids,
            exact_ids=exact_ids,
            scope_disambiguated_ids=scope_disambiguated_ids,
        )
        # A merged expansion index is only provenance.  For coordinated
        # questions it may preserve a required-answer mapping only when the
        # retained text distinguishes that answer (or a trusted verifier
        # explicitly annotated the requirement).
        allowed_answer_ids = (
            visible_answer_ids
            | set(explicit_ids)
            | set(strict_terminology_rule_ids)
        )
        aligned_ids = tuple(
            requirement_id
            for requirement_id in aligned_ids
            if not requirement_by_id[requirement_id].is_required_answer
            or requirement_id in allowed_answer_ids
        )
        lexical_ids = [
            requirement_id
            for requirement_id in lexical_ids
            if not requirement_by_id[requirement_id].is_required_answer
            or requirement_id in allowed_answer_ids
        ]
        exact_ids &= set(lexical_ids)
    # Retrieval provenance explains why a chunk was returned; it cannot prove
    # an entity/classification relationship by itself.  A bridge requirement
    # therefore needs positive text support (or a trusted explicit annotation)
    # even when the same chunk appeared in the bridge sub-query.  This prevents
    # a value-only row such as ``D级为100元`` from being treated as proof that
    # ``普通员工对应D级``.
    aligned_ids = tuple(
        requirement_id
        for requirement_id in aligned_ids
        if requirement_by_id[requirement_id].role != "bridge"
        or requirement_id in lexical_ids
        or requirement_id in explicit_ids
    )
    if lexical_bridge_ids:
        lexical_ids = [
            requirement_id
            for requirement_id in lexical_ids
            if requirement_by_id[requirement_id].role == "bridge"
            or _independent_answer_terms_match(
                requirement_by_id[requirement_id].description,
                bridge_subjects_by_answer.get(requirement_id, ()),
                item.content,
            )
            or requirement_id in strict_terminology_rule_ids
        ]
        exact_ids = {
            requirement_id
            for requirement_id in exact_ids
            if requirement_id in lexical_ids
        }
        # Query provenance proves why a candidate was returned, not that it
        # answers every query.  If the visible text positively identifies this
        # chunk as bridge evidence, an answer-query index or current-query seed
        # cannot also turn it into answer support without independent lexical
        # (or trusted explicit) evidence.
        aligned_ids = tuple(
            requirement_id
            for requirement_id in aligned_ids
            if requirement_by_id[requirement_id].role == "bridge"
            or requirement_id in lexical_ids
            or requirement_id in strict_terminology_rule_ids
        )
        structural_ids = tuple(
            requirement_id
            for requirement_id in structural_ids
            if requirement_by_id[requirement_id].role == "bridge"
            or requirement_id in lexical_ids
        )

    # Retrieval/query provenance is a ranking and attribution signal, never
    # positive fact support.  With a task graph, ``aligned_ids`` is already
    # disabled above; lexical claim verification remains independent so a
    # bounded same-document sibling (which may not carry the original query's
    # task annotation) can still prove its own requirement.  Only trusted
    # candidate annotations are constrained by the explicit task lineage.
    candidate_ids = set(explicit_ids) | set(lexical_ids) | set(structural_ids)
    supports = tuple(
        requirement.id
        for requirement in requirements
        if requirement.id in candidate_ids
    )
    declared_role = (
        _candidate_role({"metadata": item.metadata})
        if trust_candidate_annotations
        else "background"
    )
    if not supports:
        role = "background"
    elif declared_role == "conflicting":
        role = "conflicting"
    elif declared_role == "bridge" or all(
        requirement_by_id[requirement_id].role == "bridge"
        for requirement_id in supports
    ):
        role = "bridge"
    elif declared_role == "direct" and (
        bool(set(explicit_ids) & set(supports))
        or bool(exact_ids & set(supports))
    ):
        role = "direct"
    elif exact_ids & set(supports):
        role = "direct"
    else:
        # Query-index and generic lexical overlap are deterministic relevance
        # signals, not semantic verification.  They can preserve a requirement
        # under budget without pretending a model proved direct entailment.
        role = "complement"

    metadata = dict(item.metadata)
    metadata["supports_requirement_ids"] = list(supports)
    metadata["evidence_role_v2"] = role
    if task_graph is not None:
        metadata["retrieval_task_ids"] = list(task_ids)
        metadata["task_direct_requirement_ids"] = sorted({
            requirement_id
            for task_id in task_ids
            for requirement_id in task_graph.task_by_id[task_id].target_requirement_ids
            if requirement_id in allowed_ids
        })
        metadata["task_lineage_requirement_ids"] = sorted(task_requirement_ids or ())
        metadata["task_binding_status"] = (
            "invalid" if task_binding_invalid else
            (
                "bound" if task_ids else
                ("unbound_current_run" if task_ledger is not None else "legacy_ambiguous")
            )
        )
        if task_ledger is not None:
            metadata["task_lineage_run_id"] = task_ledger.run_id
        if task_binding_invalid:
            metadata["task_binding_rejected"] = True
    metadata["exact_support_requirement_ids"] = [
        requirement.id
        for requirement in requirements
        if requirement.id in exact_ids
    ]
    metadata["explicit_support_requirement_ids"] = [
        requirement.id
        for requirement in requirements
        if requirement.id in explicit_ids
    ]
    if strict_terminology_rule_ids:
        metadata["claim_proof_kind"] = {
            requirement_id: "terminology_strict"
            for requirement_id in strict_terminology_rule_ids
        }
        metadata["strict_terminology_rule_ids"] = {
            requirement_id: list(rule_ids)
            for requirement_id, rule_ids in strict_terminology_rule_ids.items()
        }
    metadata["direct_subject_answer_requirement_ids"] = [
        requirement.id
        for requirement in requirements
        if requirement.role == "answer"
        and requirement.id in supports
        and _direct_subject_answer_support(
            requirement.description,
            item,
            bridge_subjects=bridge_subjects_by_answer.get(
                requirement.id,
                (),
            ),
        )
    ]
    provisional_ids = (
        (set(aligned_ids) | set(structural_ids))
        - set(explicit_ids)
        - set(lexical_ids)
    )
    metadata["provisional_support_requirement_ids"] = [
        requirement.id
        for requirement in requirements
        if requirement.id in provisional_ids
    ]
    return replace(
        item,
        role=role,
        supports_requirement_ids=supports,
        metadata=metadata,
    )


def _priority(
    item: EvidenceItem,
    *,
    overview_requested: bool,
    original_position: int,
) -> tuple[float, ...]:
    metadata = item.metadata
    # A current-query retrieval seed is the relevance anchor for every bounded
    # same-document expansion.  Section shape is useful only within the same
    # tier: an unrelated table/specific sibling must never evict that seed from
    # the generation context merely because it looks more concrete.
    retrieval_seed_priority = int("initial_retrieval" in item.origins)
    section_kind = str(metadata.get("section_kind") or "other")
    if overview_requested and section_kind == "overview":
        section_priority = 5
    elif section_kind == "table":
        section_priority = 4
    elif section_kind == "specific":
        section_priority = 3
    elif section_kind == "other":
        section_priority = 2
    else:
        # Generic definitions and applicability boilerplate should not crowd
        # out concrete clauses unless the user explicitly asks for them.
        section_priority = 1

    role = item.role
    contribution = str(metadata.get("contribution_role") or "").casefold()
    role_priority = 2 if role == "direct" else 0
    if metadata.get("jointly_selected") or contribution in {
        "bridge",
        "complement",
        "standalone_answer",
    }:
        role_priority = max(role_priority, 3)
    confidence_priority = 1 if item.confidence == "verified" else 0
    return (
        float(retrieval_seed_priority),
        float(section_priority),
        float(role_priority),
        float(confidence_priority),
        item.score if item.score is not None else -1.0,
        float(-original_position),
    )


def _validate_budget(max_context_chunks: int, max_context_chars: int) -> None:
    if (
        isinstance(max_context_chunks, bool)
        or not isinstance(max_context_chunks, int)
        or max_context_chunks <= 0
    ):
        raise ValueError("max_context_chunks must be a positive integer")
    if (
        isinstance(max_context_chars, bool)
        or not isinstance(max_context_chars, int)
        or max_context_chars <= 0
    ):
        raise ValueError("max_context_chars must be a positive integer")


def _ledgered_bridge_companion_ids(
    item: EvidenceItem,
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    task_graph: RetrievalTaskGraph | None,
    task_ledger: TaskExecutionLedger | None,
) -> tuple[str, ...]:
    """Return the minimum bridge-source set needed by an answer item.

    The renderer budget is part of semantic verification.  A D-level amount
    and the row which maps the original subject to D-level form one evidence
    unit; retaining only the amount must never leave a seemingly complete
    answer in the model context.  Recompute from the ledger instead of
    reading ``resolved_bridge_joins`` metadata, which is intentionally only a
    trace projection.
    """

    if task_graph is None or task_ledger is None:
        return ()
    requirement_by_id = {requirement.id: requirement for requirement in requirements}
    for requirement_id in item.supports_requirement_ids:
        answer = requirement_by_id.get(requirement_id)
        if answer is None or answer.role != "answer":
            continue
        paths = tuple(
            path
            for mode in ("proof", "augmentation")
            for path in task_graph.answer_bridge_paths(mode=mode)
            if path.answer_requirement_id == answer.id
        )
        if not paths:
            continue
        for path in paths:
            bindings = _verified_route_bindings_for_item(
                item,
                answer=answer,
                path=path,
                task_ledger=task_ledger,
            )
            if bindings:
                return tuple(dict.fromkeys(
                    binding.bridge_source_item_id for binding in bindings
                ))
    return ()


def _select_context_items(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    coverage_required_ids: frozenset[str],
    retrieval_queries: tuple[str, ...],
    query_alignment: Mapping[int, str],
    task_graph: RetrievalTaskGraph | None,
    task_ledger: TaskExecutionLedger | None,
    route_units: Mapping[str, tuple[frozenset[str], ...]] | None,
    overview_requested: bool,
    max_context_chunks: int,
    max_context_chars: int,
) -> tuple[tuple[EvidenceItem, ...], tuple[str, ...], bool]:
    ranked = sorted(
        enumerate(items),
        key=lambda pair: _priority(
            pair[1],
            overview_requested=overview_requested,
            original_position=pair[0],
        ),
        reverse=True,
    )
    priority_position = {
        item.chunk_id: position
        for position, (_, item) in enumerate(ranked)
    }
    pending = list(ranked)
    coverage_first: list[tuple[int, EvidenceItem]] = []
    uncovered = set(coverage_required_ids)
    while uncovered:
        choices = [
            pair
            for pair in pending
            if pair[1].role in _COVERAGE_ROLES
            and (set(pair[1].supports_requirement_ids) & uncovered)
        ]
        if not choices:
            break
        selected_pair = max(
            choices,
            key=lambda pair: (
                len(set(pair[1].supports_requirement_ids) & uncovered),
                -priority_position[pair[1].chunk_id],
            ),
        )
        coverage_first.append(selected_pair)
        pending.remove(selected_pair)
        uncovered -= set(selected_pair[1].supports_requirement_ids)
    # Diagnostic/background candidates stay in ``bundle.items`` for the
    # search panel, but they must not consume the generation budget.  The
    # renderer enforces the same positive-evidence boundary; applying it here
    # prevents dropped background rows from crowding out later answer clauses
    # and leaving otherwise available prompt capacity unused.
    positive_pending = (
        [
            pair
            for pair in pending
            if pair[1].role in _COVERAGE_ROLES
            and pair[1].supports_requirement_ids
        ]
        if requirements
        else pending
    )
    selection_order = [*coverage_first, *positive_pending]
    eligible_ids = {item.chunk_id for _, item in selection_order}

    item_by_id = {item.chunk_id: item for item in items}
    item_position = {
        item.chunk_id: position
        for position, item in enumerate(items)
    }
    renderable_item_ids = {
        item.chunk_id
        for item in items
        if item.role in _COVERAGE_ROLES and item.supports_requirement_ids
    }
    selected_ids: set[str] = set()
    selected_items: list[EvidenceItem] = []
    used_chars = 0
    budget_limited = False
    for _, item in selection_order:
        if item.chunk_id in selected_ids:
            continue
        if len(selected_ids) >= max_context_chunks or used_chars >= max_context_chars:
            budget_limited = True
            continue
        route_choices = tuple(
            unit
            for unit in (route_units or {}).get(item.chunk_id, ())
            if (
                item.chunk_id in unit
                and all(value in item_by_id for value in unit)
                # A structural companion hidden by the renderer cannot close a
                # route.  Reject it here instead of admitting a context the
                # final proof layer would have to discard later.
                and set(unit).issubset(renderable_item_ids)
            )
        )
        has_bridge_bound_support = any(
            requirement.role == "answer"
            and requirement.id in item.supports_requirement_ids
            and (
                requirement.proof_bridge_requirement_ids
                or requirement.augmentation_bridge_requirement_ids
            )
            for requirement in requirements
        )
        if has_bridge_bound_support and task_graph is not None and task_ledger is not None and not route_choices:
            # The source mapper may have retained a positive-looking row, but
            # without a complete graph route it cannot enter model context.
            budget_limited = True
            continue
        def route_admission_items(unit: frozenset[str]) -> tuple[EvidenceItem, ...]:
            return tuple(
                item_by_id[item_id]
                for item_id in sorted(
                    (candidate_id for candidate_id in unit if candidate_id not in selected_ids),
                    key=lambda candidate_id: item_position[candidate_id],
                )
            )

        selected_route = min(
            route_choices,
            key=lambda unit: (
                evidence_context_char_cost(
                    tuple((*selected_items, *route_admission_items(unit)))
                ) - used_chars,
                len(unit - selected_ids),
                tuple(sorted(unit)),
            ),
            default=frozenset({item.chunk_id}),
        )
        admission_items = route_admission_items(selected_route)
        candidate_items = tuple((*selected_items, *admission_items))
        candidate_chars = evidence_context_char_cost(candidate_items)
        if (
            len(candidate_items) > max_context_chunks
            or candidate_chars > max_context_chars
        ):
            # An evidence block is indivisible: a prefix can omit the exact
            # clause that justified its claim.  Skip this complete route and
            # continue looking for an independent route that genuinely fits.
            budget_limited = True
            continue
        for admission_item in admission_items:
            selected_ids.add(admission_item.chunk_id)
            selected_items.append(admission_item)
        used_chars = candidate_chars

    if len(selected_ids) < len(eligible_ids):
        budget_limited = True
    bounded_items = tuple(items)
    context_ids = tuple(
        item.chunk_id for item in bounded_items if item.chunk_id in selected_ids
    )
    return bounded_items, context_ids, budget_limited


def _group_and_sort(items: Sequence[EvidenceItem]) -> tuple[EvidenceItem, ...]:
    grouped: dict[tuple[str, str], list[EvidenceItem]] = defaultdict(list)
    first_position: dict[tuple[str, str], int] = {}
    for position, item in enumerate(items):
        key = (item.kb_id, item.doc_id)
        grouped[key].append(item)
        first_position.setdefault(key, position)

    ordered: list[EvidenceItem] = []
    for key in sorted(grouped, key=lambda value: first_position[value]):
        ordered.extend(
            sorted(
                grouped[key],
                key=lambda item: (item.chunk_index, item.chunk_id),
            )
        )
    return tuple(ordered)


def _has_active_answer_assertion(
    item: EvidenceItem,
    requirement_id: str,
) -> bool:
    """Read only the request-local assertion record written by this module."""

    raw = item.metadata.get("answer_claim_assertions")
    if not isinstance(raw, Mapping):
        return False
    values = raw.get(requirement_id)
    if isinstance(values, Mapping):
        values = (values,)
    if not isinstance(values, (list, tuple)):
        return False
    return any(
        isinstance(value, Mapping)
        and str(value.get("status") or "").strip().casefold() == "active"
        for value in values
    )


def _active_answer_claim_semantics(
    item: EvidenceItem,
    requirement_id: str,
) -> tuple[tuple[str, str, str], ...]:
    """Read semantic results emitted by this request's claim adjudicator.

    This helper is intentionally adjacent to the ledger-backed claim builder,
    rather than part of ``evidence_graph``.  The graph consumes only typed
    ``EvidenceClaim`` objects; it must never recover a proof or a value from
    renderer metadata.  ``_reconcile_answer_claim_assertions`` rewrites this
    field after task provenance, source content and final context selection
    have been checked, so malformed/incomplete rows fail closed here.
    """

    raw = item.metadata.get("answer_claim_assertions")
    if not isinstance(raw, Mapping):
        return ()
    values = raw.get(requirement_id)
    if isinstance(values, Mapping):
        values = (values,)
    if not isinstance(values, (list, tuple)):
        return ()
    semantics: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        if str(value.get("status") or "").strip().casefold() != "active":
            continue
        result_kind = str(value.get("result_kind") or "").strip().casefold()
        normalized_result = re.sub(
            r"\s+",
            " ",
            str(value.get("normalized_result") or ""),
        ).strip().casefold()
        claim_key = re.sub(
            r"\s+",
            " ",
            str(value.get("claim_key") or ""),
        ).strip().casefold()
        if (
            result_kind not in CLAIM_RESULT_KINDS
            or not normalized_result
            or not claim_key
        ):
            continue
        semantic = (result_kind, normalized_result, claim_key)
        if semantic in seen:
            continue
        seen.add(semantic)
        semantics.append(semantic)
    return tuple(semantics)


def _has_active_document_policy_snapshot_assertion(
    item: EvidenceItem,
    requirement_id: str,
) -> bool:
    """Whether the reconciler emitted a typed policy-snapshot member claim."""

    return any(
        result_kind == "document_policy"
        for result_kind, _normalized_result, _claim_key
        in _active_answer_claim_semantics(item, requirement_id)
    )


def _claim_proof_for_item(
    *,
    item: EvidenceItem,
    requirement: AnswerRequirementV2,
    terminology_resolution: TerminologyRuntimeResolution | None,
) -> tuple[str, tuple[str, ...]]:
    """Produce proof kind from the runtime registry, never candidate metadata."""

    if terminology_resolution is None:
        return "source_assertion", ()
    match = terminology_resolution.evidence_match(
        requirement=requirement,
        kb_id=item.kb_id,
        doc_id=item.doc_id,
        content=item.content,
    )
    if match is None:
        return "source_assertion", ()
    return "terminology_strict", tuple(match.rule_ids)


def _fact_still_matches_visible_source(
    *,
    fact: Any,
    bridge_requirement: AnswerRequirementV2,
    item: EvidenceItem,
) -> bool:
    """Re-parse a ledger fact against the exact post-budget source content."""

    if (
        item.chunk_id != fact.source_chunk_id
        or item.kb_id != fact.source_kb_id
        or item.doc_id != fact.source_doc_id
        or bridge_requirement.id not in item.supports_requirement_ids
    ):
        return False
    return any(
        candidate_fact.requirement_id == fact.requirement_id
        and candidate_fact.subject == fact.subject
        and candidate_fact.value == fact.value
        and candidate_fact.source_chunk_id == fact.source_chunk_id
        and candidate_fact.source_kb_id == fact.source_kb_id
        and candidate_fact.source_doc_id == fact.source_doc_id
        and candidate_fact.scope_products == fact.scope_products
        and candidate_fact.scope_versions == fact.scope_versions
        and candidate_fact.scope_projects == fact.scope_projects
        for candidate_fact in resolve_bridge_facts(
            (bridge_requirement,),
            (item,),
            supported_only=True,
        )
    )


def _build_ledgered_evidence_claims(
    evidence: EvidenceBundle,
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    task_graph: RetrievalTaskGraph,
    task_ledger: TaskExecutionLedger,
    terminology_resolution: TerminologyRuntimeResolution | None,
) -> tuple[EvidenceClaim, ...]:
    """Build controlled graph claims from source text and the execution ledger.

    This replaces the legacy ``resolved_bridge_joins`` metadata channel for
    every ledgered V2 request.  The graph receives explicit claims only after
    (1) the source remains valid after context truncation, and (2) the answer
    candidate is bound to the exact dynamic/same-source execution that used
    the matching bridge facts.  Metadata remains useful for trace rendering,
    but it cannot manufacture a proof at this boundary.
    """

    # Imported locally to keep the immutable evidence graph contract free of
    # a reverse import into the assembler.
    from core.rag_v2.evidence_graph import classify_claim_applicability

    requirement_by_id = {requirement.id: requirement for requirement in requirements}
    bridge_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.role == "bridge"
    }
    item_by_id = {item.chunk_id: item for item in evidence.items}
    claims: list[EvidenceClaim] = []
    claim_index = 0

    def append_claim(**kwargs: Any) -> None:
        nonlocal claim_index
        claim_index += 1
        claims.append(EvidenceClaim(
            id=f"ledger_claim_{claim_index}",
            **kwargs,
        ))

    def append_answer_claims(
        *,
        item: EvidenceItem,
        requirement: AnswerRequirementV2,
        proof_kind: str,
        strict_rule_ids: tuple[str, ...],
        applicability: str,
        bridge_bindings: tuple[BridgeClaimBinding, ...] = (),
    ) -> None:
        """Project each adjudicated semantic assertion into a typed claim.

        One chunk can contain multiple source assertions.  Keeping them
        separate is essential: collapsing them into one presentation item
        loses the normalized value needed to detect two closed but mutually
        exclusive answer routes later in the final graph.
        """

        for result_kind, normalized_result, claim_key in (
            _active_answer_claim_semantics(item, requirement.id)
        ):
            append_claim(
                requirement_id=requirement.id,
                evidence_item_id=item.chunk_id,
                document_key=(item.kb_id, item.doc_id),
                contribution_kind="answer_claim",
                applicability=applicability,
                proof_kind=proof_kind,
                strict_terminology_rule_ids=strict_rule_ids,
                result_kind=result_kind,
                normalized_result=normalized_result,
                claim_key=claim_key,
                bridge_bindings=bridge_bindings,
            )

    # A bridge fact is a source assertion in its own right.  Re-parse it from
    # the final evidence text, rather than accepting a source chunk merely
    # because a prior pre-budget execution resolved it.
    for resolution in task_ledger.bridge_resolutions():
        if resolution.status != "resolved":
            continue
        bridge_task = task_graph.task_by_id.get(resolution.bridge_task_id)
        if bridge_task is None:
            continue
        bridge_requirement_id = bridge_task.target_requirement_ids[0]
        bridge_requirement = bridge_by_id.get(bridge_requirement_id)
        if bridge_requirement is None:
            continue
        for fact in resolution.facts:
            item = item_by_id.get(fact.source_chunk_id)
            if item is None or not _fact_still_matches_visible_source(
                fact=fact,
                bridge_requirement=bridge_requirement,
                item=item,
            ):
                continue
            append_claim(
                requirement_id=bridge_requirement_id,
                evidence_item_id=item.chunk_id,
                document_key=(item.kb_id, item.doc_id),
                contribution_kind="bridge_fact",
                applicability="bridge_value",
            )

    paths_by_answer_id: dict[str, tuple[AnswerBridgePath, ...]] = {}
    for mode in ("proof", "augmentation"):
        for path in task_graph.answer_bridge_paths(mode=mode):
            paths_by_answer_id[path.answer_requirement_id] = (
                *paths_by_answer_id.get(path.answer_requirement_id, ()),
                path,
            )

    for item in evidence.items:
        for requirement in requirements:
            if (
                requirement.role != "answer"
                or requirement.id not in item.supports_requirement_ids
                or not _has_active_answer_assertion(item, requirement.id)
            ):
                continue
            proof_kind, strict_rule_ids = _claim_proof_for_item(
                item=item,
                requirement=requirement,
                terminology_resolution=terminology_resolution,
            )
            # ``document_policy`` is a source-snapshot assertion rather than
            # a scalar fact.  The reconciler above emits it only for a
            # current-query rooted full-document member; project it with the
            # explicit document-universal applicability instead of pretending
            # that every section independently states the user's full policy
            # target.  The coverage graph still requires the complete visible
            # snapshot and the separately verified document root.
            if _has_active_document_policy_snapshot_assertion(
                item,
                requirement.id,
            ):
                append_answer_claims(
                    item=item,
                    requirement=requirement,
                    applicability="document_universal",
                    proof_kind=proof_kind,
                    strict_rule_ids=strict_rule_ids,
                )
                continue
            paths = paths_by_answer_id.get(requirement.id, ())
            if not paths:
                applicability = classify_claim_applicability(item, requirement)
                if applicability is None:
                    continue
                append_answer_claims(
                    item=item,
                    requirement=requirement,
                    applicability=applicability,
                    proof_kind=proof_kind,
                    strict_rule_ids=strict_rule_ids,
                )
                continue

            # A bridge path is one possible applicability route, not a global
            # veto on a source that independently proves the original subject
            # or a document-universal rule.  Classify only from the controlled
            # source assertion state; never turn a D-level second-hop row into
            # a direct claim merely because it shares generic target words.
            independent_applicability = classify_claim_applicability(
                item,
                requirement,
            )
            if independent_applicability not in {None, "bridge_value"}:
                append_answer_claims(
                    item=item,
                    requirement=requirement,
                    applicability=independent_applicability,
                    proof_kind=proof_kind,
                    strict_rule_ids=strict_rule_ids,
                )

            seen_routes: set[tuple[tuple[str, str, str, str], ...]] = set()
            for path in paths:
                bindings = _verified_route_bindings_for_item(
                    item,
                    answer=requirement,
                    path=path,
                    task_ledger=task_ledger,
                )
                if (
                    len(bindings) != len(path.bridge_requirement_ids)
                    or any(
                        binding.bridge_execution_id is None
                        or binding.answer_execution_id is None
                        for binding in bindings
                    )
                ):
                    continue
                route_key = tuple(sorted(
                    (
                        binding.edge_mode,
                        binding.bridge_requirement_id,
                        binding.bridge_source_item_id,
                        binding.bridge_value,
                    )
                    for binding in bindings
                ))
                if route_key in seen_routes:
                    continue
                seen_routes.add(route_key)
                append_answer_claims(
                    item=item,
                    requirement=requirement,
                    applicability="bridge_value",
                    proof_kind=proof_kind,
                    strict_rule_ids=strict_rule_ids,
                    bridge_bindings=bindings,
                )
    return tuple(claims)


def _build_context_route_units(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    task_graph: RetrievalTaskGraph | None,
    task_ledger: TaskExecutionLedger | None,
    terminology_resolution: TerminologyRuntimeResolution | None,
) -> dict[str, tuple[frozenset[str], ...]]:
    """Compile structural, bridge and condition companions into route units.

    The unit map is derived from the same explicit claims that will later be
    assessed.  It is therefore not a second keyword heuristic: selecting an
    answer chunk admits the claim's structural companions, every bridge-source
    claim and its own conditions as one bounded unit.  If no complete unit fits
    the context budget, the answer claim is not selected.
    """

    if task_graph is None or task_ledger is None:
        return {}
    from core.rag_v2.evidence_graph import build_evidence_coverage_graph

    item_ids = tuple(item.chunk_id for item in items)
    full_bundle = EvidenceBundle(
        state=EvidenceState(
            availability="ok",
            confidence="retrieved",
            completeness="unknown",
        ),
        items=tuple(items),
        context_item_ids=item_ids,
    )
    claims = _build_ledgered_evidence_claims(
        full_bundle,
        requirements=requirements,
        task_graph=task_graph,
        task_ledger=task_ledger,
        terminology_resolution=terminology_resolution,
    )
    try:
        graph = build_evidence_coverage_graph(
            full_bundle,
            requirements,
            claims=claims,
        )
    except Exception:
        # A malformed structural unit must not widen retrieval.  The final
        # graph will report the same verification failure after assembly.
        return {}
    item_by_id = {item.chunk_id: item for item in items}
    groups_by_id = {group.id: group for group in graph.structural_groups}
    group_by_item_id: dict[str, Any] = {}
    for group in graph.structural_groups:
        for item_id in group.member_item_ids:
            group_by_item_id[item_id] = group

    # A document-policy claim is an all-or-nothing source snapshot.  Selecting
    # only the best-looking section first and discovering the missing pages in
    # the finalizer wastes budget and, more importantly, makes the planner and
    # verifier disagree about what one answer route means.  Compile the exact
    # verified source snapshot into every route unit up front.  The snapshot
    # helper checks the parser/retriever cardinality; the graph root check
    # separately ensures this is the document selected by the current query.
    complete_source_documents = set(complete_snapshot_document_keys(
        graph.evidence_items,
        require_visible=False,
    ))
    snapshot_members_by_requirement: dict[str, frozenset[str]] = {}
    for requirement in requirements:
        if (
            requirement.role != "answer"
            or not requirement.requires_document_policy_snapshot
        ):
            continue
        root_document_key = graph.document_root_keys.get(requirement.id)
        if root_document_key not in complete_source_documents:
            continue
        member_ids = frozenset(
            item.chunk_id
            for item in graph.evidence_items
            if (item.kb_id, item.doc_id) == root_document_key
            and item.metadata.get("full_document_chunk_count") is not None
        )
        if member_ids:
            snapshot_members_by_requirement[requirement.id] = member_ids

    def add_structural_closure(unit: set[str], item_id: str) -> None:
        group = group_by_item_id.get(item_id)
        if group is None:
            return
        unit.update(group.required_item_ids)
        # A condition can live in another section.  Keep its source item and
        # its own local companions, rather than treating a bare reference as
        # an independently sufficient condition.
        for condition_item_id in group.condition_item_ids:
            unit.add(condition_item_id)
            condition_item_group = group_by_item_id.get(condition_item_id)
            if condition_item_group is not None:
                unit.update(condition_item_group.required_item_ids)

    def claim_unit(claim: EvidenceClaim) -> frozenset[str]:
        unit = {claim.evidence_item_id}
        add_structural_closure(unit, claim.evidence_item_id)
        # Every member's local structural companions must travel with the
        # snapshot as well.  Otherwise a note/condition attached to a later
        # policy section could disappear while an earlier root paragraph made
        # the route look complete.
        if claim.result_kind == "document_policy":
            for member_id in snapshot_members_by_requirement.get(
                claim.requirement_id,
                (),
            ):
                unit.add(member_id)
                add_structural_closure(unit, member_id)
        if claim.condition_group_id is not None:
            condition_group = groups_by_id.get(claim.condition_group_id)
            if condition_group is not None:
                unit.update(condition_group.member_item_ids)
                unit.update(condition_group.required_item_ids)
        for binding in claim.bridge_bindings:
            unit.add(binding.bridge_source_item_id)
            add_structural_closure(unit, binding.bridge_source_item_id)
        return frozenset(
            value for value in unit if value in item_by_id
        )

    # Keep alternatives separate per answer requirement.  A candidate may be
    # valid through one of several source documents; the selector can choose
    # the smallest complete route instead of unioning all alternatives.
    by_item_requirement: dict[
        tuple[str, str], list[frozenset[str]]
    ] = defaultdict(list)
    for claim in graph.claims:
        if claim.contribution_kind != "answer_claim":
            continue
        by_item_requirement[(claim.evidence_item_id, claim.requirement_id)].append(
            claim_unit(claim)
        )

    units_by_item: dict[str, list[frozenset[str]]] = defaultdict(list)
    for item_id in item_by_id:
        requirement_choices = [
            tuple(dict.fromkeys(values))
            for (candidate_id, _requirement_id), values
            in by_item_requirement.items()
            if candidate_id == item_id
        ]
        if not requirement_choices:
            continue
        combinations: list[frozenset[str]] = [frozenset()]
        for choices in requirement_choices:
            next_combinations: list[frozenset[str]] = []
            for prefix in combinations:
                for choice in choices[:8]:
                    next_combinations.append(prefix | choice)
                    if len(next_combinations) >= 16:
                        break
                if len(next_combinations) >= 16:
                    break
            combinations = next_combinations
            if not combinations:
                break
        units_by_item[item_id].extend(
            sorted(
                set(combinations),
                key=lambda value: (len(value), tuple(sorted(value))),
            )[:16]
        )
    return {
        item_id: tuple(dict.fromkeys(units))
        for item_id, units in units_by_item.items()
        if units
    }


def reconcile_evidence_coverage_graph(
    evidence: EvidenceBundle,
    *,
    requirements: Sequence[AnswerRequirementV2],
    task_graph: RetrievalTaskGraph | None = None,
    task_ledger: TaskExecutionLedger | None = None,
    terminology_resolution: TerminologyRuntimeResolution | None = None,
) -> EvidenceBundle:
    """Attach and enforce the structural coverage graph for visible evidence.

    The ordinary mapper is deliberately recall-friendly: it decides which
    authorised chunks can enter a bounded prompt.  The coverage graph is the
    final proof boundary and sees that *exact* visible set.  For ordinary
    requirements it can only lower mapper confidence; for collection
    requirements it is the sole authority because it derives source closure
    from typed claims, rather than from renderer roles or support annotations.

    A malformed graph is treated as a semantic-verification failure rather
    than an infrastructure outage: keep the already authorised context for
    diagnostics, but downgrade completeness so it cannot produce a confident
    grounded answer.  This protects a user request from a future parser/graph
    integration regression without hiding the diagnostic reason.
    """

    normalized_requirements = _normalize_requirements(requirements)
    if not normalized_requirements:
        return evidence
    if task_ledger is not None and task_graph is None:
        raise ValueError(
            "coverage graph task_ledger requires task_graph"
        )
    if task_graph is not None and task_ledger is not None:
        if task_graph.requirements != normalized_requirements:
            raise ValueError("coverage graph task graph requirements do not match")
        if task_ledger.task_graph != task_graph:
            raise ValueError("coverage graph task ledger does not match task graph")

    try:
        # Import here to keep the low-level evidence item contracts acyclic
        # for migration/static tooling.  The graph consumes immutable items;
        # it never invokes retrievers, a model, or a database.
        from core.rag_v2.evidence_graph import (
            assess_evidence_coverage_graph,
            build_evidence_coverage_graph,
            derive_verified_collection_closures,
        )

        controlled_claims = (
            _build_ledgered_evidence_claims(
                evidence,
                requirements=normalized_requirements,
                task_graph=task_graph,
                task_ledger=task_ledger,
                terminology_resolution=terminology_resolution,
            )
            if task_graph is not None and task_ledger is not None
            else None
        )
        preliminary_graph = build_evidence_coverage_graph(
            evidence,
            normalized_requirements,
            claims=controlled_claims,
        )
        collection_closures = derive_verified_collection_closures(
            preliminary_graph,
        )
        graph = (
            build_evidence_coverage_graph(
                evidence,
                normalized_requirements,
                claims=controlled_claims,
                collection_closures=collection_closures,
            )
            if collection_closures
            else preliminary_graph
        )
        assessment = assess_evidence_coverage_graph(graph)
    except Exception as exc:
        # A failure at this final proof boundary must not retain a previous
        # ``complete`` claim.  We deliberately do not discard the context:
        # it remains valuable for trace diagnosis and can still be rendered as
        # a partial response when the caller's policy allows it.
        downgraded_completeness: EvidenceCompletenessValue = (
            "unknown"
            if not evidence.context_item_ids
            else "partial"
        )
        return replace(
            evidence,
            coverage_graph=None,
            coverage_assessment=None,
            state=EvidenceState(
                availability=evidence.state.availability,
                confidence=evidence.state.confidence,
                completeness=downgraded_completeness,
                reasons=tuple(dict.fromkeys((
                    *evidence.state.reasons,
                    "coverage_graph_assessment_failed",
                    type(exc).__name__,
                )))[:12],
            ),
        )

    collection_answer_ids = {
        requirement.id
        for requirement in normalized_requirements
        if requirement.role == "answer"
        and requirement.requires_collection_closure
    }
    # Legacy callers may have marked a collection missing with a renderer-side
    # heuristic.  Do not retain that parallel proof system: the graph's typed
    # assessment replaces only collection missing ids, while all non-collection
    # mapper failures remain a hard ceiling.
    base_non_collection_missing_ids = tuple(
        requirement_id
        for requirement_id in evidence.missing_requirement_ids
        if requirement_id not in collection_answer_ids
    )
    missing_ids = tuple(dict.fromkeys((
        *base_non_collection_missing_ids,
        *assessment.missing_requirement_ids,
    )))
    legacy_partial_was_collection_only = (
        evidence.state.completeness == "partial"
        and bool(evidence.missing_requirement_ids)
        and not base_non_collection_missing_ids
    )
    if evidence.state.completeness == "unknown":
        completeness: EvidenceCompletenessValue = "unknown"
    elif (
        evidence.state.completeness == "complete"
        or legacy_partial_was_collection_only
    ) and assessment.completeness == "complete" and not missing_ids:
        completeness = "complete"
    else:
        completeness = "partial"

    graph_reasons = tuple(
        f"coverage_graph:{reason}"
        for reason in assessment.reasons
    )
    collection_reasons: list[str] = []
    assessment_by_requirement_id = {
        value.requirement_id: value
        for value in assessment.requirement_assessments
    }
    closure_requirement_ids = {
        closure.requirement_id for closure in graph.collection_closures
    }
    for requirement_id in collection_answer_ids:
        requirement_assessment = assessment_by_requirement_id.get(requirement_id)
        if (
            requirement_assessment is None
            or requirement_assessment.completeness == "complete"
        ):
            continue
        collection_reasons.append(
            "collection_context_incomplete"
            if (
                requirement_id in closure_requirement_ids
                and requirement_assessment.missing_item_ids
            )
            else "collection_snapshot_unproven"
        )
    return replace(
        evidence,
        coverage_graph=graph,
        missing_requirement_ids=missing_ids,
        state=EvidenceState(
            availability=evidence.state.availability,
            confidence=evidence.state.confidence,
            completeness=completeness,
            reasons=tuple(dict.fromkeys((
                *evidence.state.reasons,
                *collection_reasons,
                *graph_reasons,
            )))[:12],
        ),
    )


def _model_visible_context_ids(
    evidence: EvidenceBundle,
    requested_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return exactly the item ids the renderer is permitted to expose.

    ``EvidenceBundle.context_item_ids`` historically meant both "selected by
    retrieval" and "shown to the model".  That is unsound: the renderer drops
    background/conflicting rows even when their ids remain in the bundle.  The
    final proof graph must therefore be built over this narrower, renderer-safe
    projection.
    """

    item_by_id = {item.chunk_id: item for item in evidence.items}
    visible: list[str] = []
    seen: set[str] = set()
    for raw_item_id in requested_ids:
        item_id = str(raw_item_id or "").strip()
        if not item_id or item_id in seen:
            continue
        item = item_by_id.get(item_id)
        if item is None:
            continue
        if (
            item.role not in _COVERAGE_ROLES
            or not item.supports_requirement_ids
            or item.constraint_status == "mismatch"
        ):
            continue
        seen.add(item_id)
        visible.append(item_id)
    return tuple(visible)


def _visible_context_basis(
    evidence: EvidenceBundle,
    *,
    context_item_ids: Sequence[str],
) -> EvidenceBundle:
    """Create an unfinalized bundle whose context is an exact visible set.

    The graph must never inherit an earlier graph or an earlier answer-source
    list after a renderer/route budget changes.  This helper intentionally
    clears both so the following graph evaluation is the only authority.
    """

    normalized_context_ids = tuple(context_item_ids)
    state = evidence.state
    if not normalized_context_ids and state.completeness == "complete":
        state = EvidenceState(
            availability=state.availability,
            confidence=state.confidence,
            completeness="partial" if evidence.items else "unknown",
            reasons=state.reasons,
        )
    return EvidenceBundle(
        state=state,
        items=evidence.items,
        context_item_ids=normalized_context_ids,
        # These ids are only a construction aid.  They satisfy the immutable
        # bundle contract but are discarded before the final public bundle is
        # produced from closed claims.
        answer_source_ids=normalized_context_ids,
        missing_requirement_ids=evidence.missing_requirement_ids,
    )


def _closed_claim_route_item_ids(
    graph: Any,
    assessment: Any,
) -> tuple[str, ...]:
    """Return the complete visible source route for every answer we may state.

    A source is eligible only when its requirement is complete under the graph
    contract.  The returned route contains the answer assertion, every local
    structural companion, conditions, and bridge source facts.  It is not a
    relevance ranking and intentionally has no lexical fallback.
    """

    claim_by_id = {claim.id: claim for claim in graph.claims}
    groups_by_id = {group.id: group for group in graph.structural_groups}
    group_by_item_id: dict[str, Any] = {}
    for group in graph.structural_groups:
        for item_id in group.member_item_ids:
            group_by_item_id[item_id] = group

    closed_claim_ids: set[str] = set()
    for requirement_assessment in assessment.requirement_assessments:
        if requirement_assessment.completeness == "complete":
            closed_claim_ids.update(requirement_assessment.supporting_claim_ids)

    route_ids: set[str] = set()
    visited_groups: set[str] = set()

    def add_item_with_structure(item_id: str) -> None:
        if item_id not in set(graph.visible_evidence_item_ids):
            return
        route_ids.add(item_id)
        group = group_by_item_id.get(item_id)
        if group is None or group.id in visited_groups:
            return
        visited_groups.add(group.id)
        for companion_id in group.required_item_ids:
            add_item_with_structure(companion_id)

    def add_condition_group(group_id: str | None) -> None:
        if not group_id:
            return
        group = groups_by_id.get(group_id)
        if group is None:
            return
        for item_id in group.member_item_ids:
            add_item_with_structure(item_id)

    for claim_id in closed_claim_ids:
        claim = claim_by_id.get(claim_id)
        if claim is None or claim.contribution_kind != "answer_claim":
            continue
        add_item_with_structure(claim.evidence_item_id)
        add_condition_group(claim.condition_group_id)
        for binding in claim.bridge_bindings:
            add_item_with_structure(binding.bridge_source_item_id)

    return tuple(
        item_id
        for item_id in graph.visible_evidence_item_ids
        if item_id in route_ids
    )


def _finalization_failure_bundle(
    evidence: EvidenceBundle,
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    reason: str,
) -> EvidenceBundle:
    """Fail closed when final proof construction cannot be trusted."""

    required_answer_ids = tuple(
        requirement.id
        for requirement in requirements
        if requirement.is_required_answer
    )
    state = EvidenceState(
        availability=evidence.state.availability,
        confidence=(
            "none"
            if evidence.state.availability == "unavailable"
            else evidence.state.confidence
        ),
        completeness="unknown" if not evidence.items else "partial",
        reasons=tuple(dict.fromkeys((
            *evidence.state.reasons,
            reason,
        )))[:12],
    )
    return EvidenceBundle(
        state=state,
        items=evidence.items,
        context_item_ids=(),
        answer_source_ids=(),
        missing_requirement_ids=required_answer_ids,
    )


def _finalization_failure(
    evidence: EvidenceBundle,
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    reason: str,
) -> FinalizedVisibleEvidence:
    bundle = _finalization_failure_bundle(
        evidence,
        requirements=requirements,
        reason=reason,
    )
    return FinalizedVisibleEvidence(
        bundle=bundle,
        context=EvidenceContext(text=""),
        assessment=None,
        route_item_ids=(),
        generation_allowed=False,
    )


def finalize_visible_evidence_bundle(
    evidence: EvidenceBundle,
    *,
    requirements: Sequence[AnswerRequirementV2],
    task_graph: RetrievalTaskGraph | None = None,
    task_ledger: TaskExecutionLedger | None = None,
    terminology_resolution: TerminologyRuntimeResolution | None = None,
    max_context_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS,
    max_context_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
) -> FinalizedVisibleEvidence:
    """Produce the sole generation-safe evidence bundle for one response.

    Retrieval selection, bridge resolution, and renderer budgeting can each
    remove an item.  This function is the mandatory final boundary after any
    such removal: it evaluates the exact model-visible set with the graph,
    keeps only fully closed answer routes, then evaluates that reduced set once
    more.  ``answer_source_ids`` and context therefore describe the same
    complete proof routes; a lone bridge row can never reach the model.
    """

    _validate_budget(max_context_chunks, max_context_chars)
    normalized_requirements = _normalize_requirements(requirements)
    if not normalized_requirements:
        context = build_evidence_context(
            evidence,
            max_chunks=max_context_chunks,
            max_chars=max_context_chars,
        )
        if context.truncated or tuple(context.item_ids) != evidence.context_item_ids:
            return _finalization_failure(
                evidence,
                requirements=(),
                reason="visible_evidence_renderer_budget_inconsistent",
            )
        return FinalizedVisibleEvidence(
            bundle=evidence,
            context=context,
            assessment=None,
            route_item_ids=tuple(context.item_ids),
            generation_allowed=False,
        )
    # Finalization is a V2 execution boundary, not a generic formatter.  A
    # non-empty answer contract must have the exact graph and request-local
    # ledger that produced the candidates; otherwise source provenance and
    # bridge closure are unknowable and generation must not continue.
    if task_graph is None or task_ledger is None:
        raise ValueError(
            "visible evidence finalization requires a ledgered task graph"
        )
    if task_graph.requirements != normalized_requirements:
        raise ValueError(
            "visible evidence finalization task graph requirements do not match"
        )
    if task_ledger.task_graph != task_graph:
        raise ValueError(
            "visible evidence finalization task ledger does not match task graph"
        )

    requested_ids = evidence.context_item_ids
    current = _visible_context_basis(
        evidence,
        context_item_ids=_model_visible_context_ids(evidence, requested_ids),
    )

    try:
        from core.rag_v2.evidence_graph import assess_evidence_coverage_graph

        # Route projection is monotonic: it can only remove evidence.  Two
        # passes are sufficient in normal operation (all claim companions are
        # inserted together); keep a third guarded pass so a future graph
        # extension fails closed instead of silently exposing a partial route.
        for _ in range(3):
            candidate_context = build_evidence_context(
                current,
                max_chunks=max_context_chunks,
                max_chars=max_context_chars,
            )
            if (
                candidate_context.truncated
                or tuple(candidate_context.item_ids) != current.context_item_ids
            ):
                return _finalization_failure(
                    evidence,
                    requirements=normalized_requirements,
                    reason="visible_evidence_renderer_budget_inconsistent",
                )
            evaluated = reconcile_evidence_coverage_graph(
                current,
                requirements=normalized_requirements,
                task_graph=task_graph,
                task_ledger=task_ledger,
                terminology_resolution=terminology_resolution,
            )
            graph = evaluated.coverage_graph
            if graph is None:
                return _finalization_failure(
                    evidence,
                    requirements=normalized_requirements,
                    reason="visible_evidence_graph_unavailable",
                )
            assessment = assess_evidence_coverage_graph(graph)
            route_ids = _closed_claim_route_item_ids(graph, assessment)
            if route_ids == current.context_item_ids:
                final_graph = graph
                final_assessment = assessment
                final_evaluated = evaluated
                break
            current = _visible_context_basis(
                evaluated,
                context_item_ids=route_ids,
            )
        else:
            return _finalization_failure(
                evidence,
                requirements=normalized_requirements,
                reason="visible_evidence_route_not_stable",
            )
    except Exception:
        return _finalization_failure(
            evidence,
            requirements=normalized_requirements,
            reason="visible_evidence_finalization_failed",
        )

    final_items_by_id = {item.chunk_id: item for item in final_evaluated.items}
    final_context_ids = tuple(final_graph.visible_evidence_item_ids)
    final_source_ids = tuple(
        item_id
        for item_id in final_context_ids
        if (
            final_items_by_id[item_id].role in _COVERAGE_ROLES
            and final_items_by_id[item_id].supports_requirement_ids
        )
    )
    if final_assessment.completeness == "complete" and final_source_ids:
        completeness: EvidenceCompletenessValue = "complete"
    elif final_evaluated.items:
        completeness = "partial"
    else:
        completeness = "unknown"
    if final_source_ids:
        confidence = (
            "verified"
            if all(
                final_items_by_id[item_id].confidence == "verified"
                for item_id in final_source_ids
            )
            else "retrieved"
        )
    else:
        confidence = final_evaluated.state.confidence
    state = EvidenceState(
        availability=final_evaluated.state.availability,
        confidence=confidence,
        completeness=completeness,
        reasons=tuple(dict.fromkeys((
            *final_evaluated.state.reasons,
            "visible_evidence_finalized",
            *(("unclosed_visible_evidence_pruned",)
              if final_context_ids != tuple(requested_ids)
              else ()),
        )))[:12],
    )
    bundle = EvidenceBundle(
        state=state,
        items=final_evaluated.items,
        context_item_ids=final_context_ids,
        answer_source_ids=final_source_ids,
        missing_requirement_ids=final_assessment.missing_requirement_ids,
        coverage_graph=final_graph,
        coverage_assessment=final_assessment,
    )
    context = build_evidence_context(
        bundle,
        max_chunks=max_context_chunks,
        max_chars=max_context_chars,
    )
    if (
        context.truncated
        or tuple(context.item_ids) != tuple(bundle.context_item_ids)
    ):
        return _finalization_failure(
            evidence,
            requirements=normalized_requirements,
            reason="visible_evidence_final_renderer_inconsistent",
        )

    claim_by_id = {claim.id: claim for claim in final_graph.claims}
    closed_answer_claim_ids = tuple(
        claim_id
        for requirement_assessment in final_assessment.requirement_assessments
        if requirement_assessment.completeness == "complete"
        for claim_id in requirement_assessment.supporting_claim_ids
        if (
            (claim := claim_by_id.get(claim_id)) is not None
            and claim.contribution_kind == "answer_claim"
        )
    )
    answer_claim_item_ids = tuple(dict.fromkeys(
        claim_by_id[claim_id].evidence_item_id
        for claim_id in closed_answer_claim_ids
        if claim_id in claim_by_id
    ))
    requirement_by_id = {
        requirement.id: requirement
        for requirement in normalized_requirements
    }
    generation_allowed = any(
        requirement_by_id[claim_by_id[claim_id].requirement_id].is_required_answer
        for claim_id in closed_answer_claim_ids
        if (
            claim_id in claim_by_id
            and claim_by_id[claim_id].requirement_id in requirement_by_id
        )
    )
    return FinalizedVisibleEvidence(
        bundle=bundle,
        context=context,
        assessment=final_assessment,
        closed_answer_claim_ids=closed_answer_claim_ids,
        answer_claim_item_ids=answer_claim_item_ids,
        route_item_ids=tuple(bundle.context_item_ids),
        generation_allowed=generation_allowed,
        renderer_dropped_item_ids=tuple(context.dropped_item_ids),
    )


def assemble_evidence_bundle(
    *,
    query: str,
    candidates: Sequence[Mapping[str, Any] | EvidenceItem],
    requirements: Sequence[AnswerRequirementV2] = (),
    retrieval_queries: Sequence[str] = (),
    task_graph: RetrievalTaskGraph | None = None,
    task_ledger: TaskExecutionLedger | None = None,
    terminology_resolution: TerminologyRuntimeResolution | None = None,
    constraints: QueryConstraints | None = None,
    overview_candidates: Sequence[Mapping[str, Any] | EvidenceItem] = (),
    answer_shape: str | None = None,
    rerank_succeeded: bool | None = None,
    trust_candidate_annotations: bool = False,
    expansion_succeeded: bool | None = None,
    retrieval_degraded: bool = False,
    completeness: EvidenceCompletenessValue = "unknown",
    missing_requirement_ids: Sequence[str] = (),
    max_context_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS,
    max_context_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
) -> EvidenceBundle:
    """Build a deterministic, scope-safe and budgeted evidence bundle.

    ``candidates`` is the authorization anchor.  Complete small-document
    ``overview_candidates`` may add sibling chunks only for a document already
    present in that authorized pool, and only for an explicit overview query.
    A failed soft dependency changes state/confidence but never empties the
    surviving candidate list by itself.
    """

    _validate_budget(max_context_chunks, max_context_chars)
    if completeness not in {"complete", "partial", "unknown"}:
        raise ValueError("unsupported evidence completeness")
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    if not isinstance(trust_candidate_annotations, bool):
        raise ValueError("trust_candidate_annotations must be a boolean")
    normalized_requirements = _normalize_requirements(requirements)
    normalized_task_graph = task_graph
    if normalized_task_graph is not None:
        if not isinstance(normalized_task_graph, RetrievalTaskGraph):
            raise ValueError("task_graph must be a RetrievalTaskGraph")
        if tuple(normalized_task_graph.requirements) != normalized_requirements:
            raise ValueError(
                "task_graph requirements must exactly match evidence requirements"
            )
        # Re-check at the evidence boundary.  A graph serialized/recreated by
        # an upstream layer must not silently downgrade to the legacy mapper.
        if not normalized_requirements:
            raise ValueError("task_graph requires non-empty requirements")
        if task_ledger is None:
            raise ValueError(
                "V2 evidence assembly requires a request-local task ledger"
            )
    if task_ledger is not None:
        if normalized_task_graph is None:
            raise ValueError("task_ledger requires task_graph")
        if not isinstance(task_ledger, TaskExecutionLedger):
            raise ValueError("task_ledger must be a TaskExecutionLedger")
        if task_ledger.task_graph != normalized_task_graph:
            raise ValueError("task_ledger graph must exactly match task_graph")
    if (
        terminology_resolution is not None
        and not isinstance(terminology_resolution, TerminologyRuntimeResolution)
    ):
        raise ValueError("terminology_resolution must be a TerminologyRuntimeResolution")
    # V2 generation is requirement-mapped by construction.  Keep the
    # requirement-less branch only for explicitly legacy diagnostic callers
    # that omit ``answer_shape``; a normal V2 shape without a requirement is a
    # protocol error rather than an invitation to treat all context as an
    # answer source.
    if not normalized_requirements and answer_shape is not None:
        raise ValueError("V2 evidence assembly requires non-empty requirements")
    bridge_requirements = tuple(
        item for item in normalized_requirements if item.role == "bridge"
    )
    if bridge_requirements:
        if normalized_task_graph is None:
            raise ValueError(
                "bridge evidence requires a ledgered task graph"
            )
        validate_answer_requirement_graph(
            normalized_requirements,
            require_explicit_answer_dependencies=True,
            require_referenced_bridges=True,
        )
    bridge_dependency_ids = frozenset(
        dependency_id
        for requirement in normalized_requirements
        if requirement.role == "answer"
        for dependency_id in (requirement.depends_on_requirement_ids or ())
    )
    bridge_resolution_required = bool(bridge_dependency_ids)
    if answer_shape == "multi_hop" and not bridge_resolution_required:
        raise ValueError(
            "multi_hop evidence assembly requires an answer-to-bridge dependency"
        )
    normalized_retrieval_queries = _normalize_retrieval_queries(
        retrieval_queries
    )
    query_alignment = (
        {}
        if normalized_task_graph is not None
        else _query_requirement_alignment(
            requirements=normalized_requirements,
            retrieval_queries=normalized_retrieval_queries,
        )
    )
    coverage_required_ids = frozenset(
        requirement.id
        for requirement in normalized_requirements
        if requirement.importance == "required"
        or requirement.id in bridge_dependency_ids
    )
    annotations_trusted = bool(
        rerank_succeeded is True or trust_candidate_annotations
    )

    overview_requested = bool(
        answer_shape == "overview" or _OVERVIEW_QUERY_RE.search(query)
    )
    reasons: list[str] = []
    converted: list[EvidenceItem] = []
    seen_chunk_ids: set[str] = set()

    for candidate in candidates:
        item, reason = _to_evidence_item(
            candidate,
            constraints=constraints,
            rerank_succeeded=rerank_succeeded,
        )
        if reason:
            reasons.append(reason)
        if item is None or item.chunk_id in seen_chunk_ids:
            continue
        # A completed rerank may safely exclude a candidate it explicitly
        # classified as irrelevant.  A failed rerank may not use those partial
        # labels to erase the authorized retrieval pool.
        if (
            rerank_succeeded is True
            and str(item.metadata.get("evidence_role") or "").casefold()
            == "irrelevant"
        ):
            reasons.append("verified_irrelevant_candidate_excluded")
            continue
        item = _mapped_item(
            item,
            requirements=normalized_requirements,
            retrieval_queries=normalized_retrieval_queries,
            query_alignment=query_alignment,
            trust_candidate_annotations=annotations_trusted,
            task_graph=normalized_task_graph,
            task_ledger=task_ledger,
            terminology_resolution=terminology_resolution,
            overview_requested=overview_requested,
        )
        if normalized_task_graph is not None and item.metadata.get(
            "task_binding_status"
        ) != "bound":
            reasons.append(
                (
                    "legacy_ambiguous_task_provenance"
                    if item.metadata.get("task_binding_status") == "legacy_ambiguous"
                    else (
                        "unbound_current_task_provenance"
                        if item.metadata.get("task_binding_status") == "unbound_current_run"
                        else "invalid_task_provenance"
                    )
                )
            )
        seen_chunk_ids.add(item.chunk_id)
        converted.append(item)

    authorized_documents = {(item.kb_id, item.doc_id) for item in converted}
    if overview_requested and authorized_documents:
        overview_added = False
        for candidate in overview_candidates:
            raw: dict[str, Any]
            try:
                raw = _as_mapping(candidate)
            except ValueError:
                reasons.append("invalid_candidate_excluded")
                continue
            document_key = (_string_id(raw.get("kb_id")), _string_id(raw.get("doc_id")))
            if document_key not in authorized_documents:
                reasons.append("unanchored_overview_candidate_excluded")
                continue
            item, reason = _to_evidence_item(
                raw,
                constraints=constraints,
                rerank_succeeded=rerank_succeeded,
                extra_origin="overview_full_document",
                force_retrieved=True,
            )
            if reason:
                reasons.append(reason)
            if item is None or item.chunk_id in seen_chunk_ids:
                continue
            item = _mapped_item(
                item,
                requirements=normalized_requirements,
                retrieval_queries=normalized_retrieval_queries,
                query_alignment=query_alignment,
                trust_candidate_annotations=annotations_trusted,
                task_graph=normalized_task_graph,
                task_ledger=task_ledger,
                terminology_resolution=terminology_resolution,
                overview_requested=overview_requested,
            )
            if normalized_task_graph is not None and item.metadata.get(
                "task_binding_status"
            ) != "bound":
                reasons.append(
                    (
                        "legacy_ambiguous_task_provenance"
                        if item.metadata.get("task_binding_status") == "legacy_ambiguous"
                        else (
                            "unbound_current_task_provenance"
                            if item.metadata.get("task_binding_status") == "unbound_current_run"
                            else "invalid_task_provenance"
                        )
                    )
                )
            seen_chunk_ids.add(item.chunk_id)
            converted.append(item)
            overview_added = True
        if overview_added:
            reasons.append("overview_full_document_added")

    converted = _attach_document_root_topic_anchors(
        converted,
        requirements=normalized_requirements,
    )

    # A task graph is accepted only with its request-local ledger above.  This
    # is intentionally not a capability flag: bridge route semantics must not
    # vary by whether a caller happened to provide provenance.
    ledgered_bridge_routes = normalized_task_graph is not None
    if ledgered_bridge_routes:
        # Production V2 uses a request-local execution ledger.  Do not mix it
        # with the historical content-only bridge reconciler: that older
        # path cannot prove that a D-level rule was actually returned by the
        # bridge-materialised answer query.
        converted = _reconcile_ledgered_bridge_routes(
            converted,
            requirements=normalized_requirements,
            task_graph=normalized_task_graph,
            task_ledger=task_ledger,
        )
    # Optional bridge augmentation has a different contract from a proof
    # dependency: it may improve retrieval precision, but only a source that
    # directly names the original subject or an execution-verified second hop
    # may support the answer.  Run this after proof reconciliation so a
    # mixed proof+augmentation path cannot accidentally retain provisional
    # lexical support from either mapper.
    converted = _reconcile_answer_claim_assertions(
        converted,
        requirements=normalized_requirements,
        overview_requested=overview_requested,
        terminology_resolution=terminology_resolution,
    )
    converted, answer_conflict = _reconcile_same_source_answer_conflicts(
        converted,
        requirements=normalized_requirements,
    )
    if answer_conflict:
        reasons.append("conflicting_active_answer_claims")

    ordered_items = _group_and_sort(converted)
    route_units = _build_context_route_units(
        ordered_items,
        requirements=normalized_requirements,
        task_graph=normalized_task_graph,
        task_ledger=task_ledger,
        terminology_resolution=terminology_resolution,
    )
    bounded_items, context_ids, budget_limited = _select_context_items(
        ordered_items,
        requirements=normalized_requirements,
        coverage_required_ids=coverage_required_ids,
        retrieval_queries=normalized_retrieval_queries,
        query_alignment=query_alignment,
        task_graph=normalized_task_graph,
        task_ledger=task_ledger,
        route_units=route_units,
        overview_requested=overview_requested,
        max_context_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
    )
    if budget_limited:
        reasons.append("context_budget_limited")
        if ledgered_bridge_routes:
            bounded_items = tuple(_reconcile_ledgered_bridge_routes(
                bounded_items,
                requirements=normalized_requirements,
                task_graph=normalized_task_graph,
                task_ledger=task_ledger,
            ))
            bounded_items = tuple(_reconcile_answer_claim_assertions(
                bounded_items,
                requirements=normalized_requirements,
                overview_requested=overview_requested,
                terminology_resolution=terminology_resolution,
            ))
            reasons.append("bridge_links_revalidated_after_budget")

    if normalized_requirements:
        selected_by_id = {item.chunk_id: item for item in bounded_items}
        covered_ids = {
            requirement_id
            for chunk_id in context_ids
            for item in (selected_by_id[chunk_id],)
            if item.role in _COVERAGE_ROLES
            for requirement_id in item.supports_requirement_ids
        }
        missing_ids = tuple(
            requirement.id
            for requirement in normalized_requirements
            if requirement.id in coverage_required_ids
            and requirement.id not in covered_ids
        )
    else:
        # Compatibility path for older callers that have not supplied typed
        # requirements yet.  Once requirements are present, per-chunk mapping
        # is authoritative and the legacy concatenated-corpus estimate is not.
        missing_ids = tuple(dict.fromkeys(
            str(value or "").strip()
            for value in missing_requirement_ids
            if str(value or "").strip()
        ))
    if normalized_requirements:
        # Typed context coverage is the authoritative completeness result.  A
        # pipeline estimate is only a retrieval-time ceiling and must neither
        # hold a complete evidence graph at ``partial`` nor promote a missing
        # path.  Coverage was computed after the exact chunk/character budget.
        effective_completeness: EvidenceCompletenessValue = (
            "complete" if context_ids and not missing_ids else "partial"
        )
    else:
        effective_completeness = completeness
        if missing_ids:
            effective_completeness = "partial"
    if (
        budget_limited
        and effective_completeness == "complete"
        and (not normalized_requirements or bool(missing_ids))
    ):
        # Legacy callers have no per-chunk coverage proof.  Typed callers may
        # retain completeness only when every required requirement is still
        # covered after the exact context budget has been applied.
        effective_completeness = "partial"

    if not bounded_items:
        soft_degraded = (
            retrieval_degraded
            or rerank_succeeded is False
            or expansion_succeeded is False
        )
        availability = "degraded" if soft_degraded else "ok"
        confidence = "none"
        effective_completeness = "unknown"
        if rerank_succeeded is False:
            reasons.append("rerank_degraded")
        if expansion_succeeded is False:
            reasons.append("expansion_degraded")
        if retrieval_degraded:
            reasons.append("retrieval_degraded")
        reasons.append("no_usable_authorized_evidence")
    else:
        soft_degraded = (
            retrieval_degraded
            or rerank_succeeded is False
            or expansion_succeeded is False
        )
        availability = "degraded" if soft_degraded else "ok"
        if rerank_succeeded is False:
            reasons.append("rerank_degraded")
        if expansion_succeeded is False:
            reasons.append("expansion_degraded")
        if retrieval_degraded:
            reasons.append("retrieval_degraded")
        selected_by_id = {item.chunk_id: item for item in bounded_items}
        selected = [selected_by_id[value] for value in context_ids]
        confidence = (
            "verified"
            if selected and all(item.confidence == "verified" for item in selected)
            else "retrieved"
        )
        if not context_ids:
            effective_completeness = "unknown"

    state = EvidenceState(
        availability=availability,
        confidence=confidence,
        completeness=effective_completeness,
        reasons=tuple(dict.fromkeys(reasons))[:12],
    )
    item_by_id = {item.chunk_id: item for item in bounded_items}
    answer_source_ids = (
        tuple(
            chunk_id
            for chunk_id in context_ids
            if item_by_id[chunk_id].role in _COVERAGE_ROLES
            and item_by_id[chunk_id].supports_requirement_ids
        )
        if normalized_requirements
        else tuple(
            chunk_id
            for chunk_id in context_ids
            if item_by_id[chunk_id].role in _COVERAGE_ROLES
            and item_by_id[chunk_id].supports_requirement_ids
        )
    )
    base_bundle = EvidenceBundle(
        state=state,
        items=bounded_items,
        context_item_ids=context_ids,
        # Background and conflicting chunks may provide bounded document
        # context, but cannot be advertised as positive answer support.
        answer_source_ids=answer_source_ids,
        missing_requirement_ids=missing_ids,
    )
    # Final graph closure and the exact serialized prompt budget are handled
    # together by ``finalize_visible_evidence_bundle`` in the runner.  Keeping
    # assembly provisional avoids publishing a second answer/source truth here.
    return base_bundle


__all__ = [
    "DEFAULT_CONTEXT_MAX_CHARS",
    "DEFAULT_CONTEXT_MAX_CHUNKS",
    "FinalizedVisibleEvidence",
    "UnverifiedCandidateBundleResult",
    "assemble_evidence_bundle",
    "assemble_unverified_candidate_bundle",
    "assemble_unverified_candidate_bundle_with_diagnostics",
    "finalize_visible_evidence_bundle",
    "reconcile_evidence_coverage_graph",
]
