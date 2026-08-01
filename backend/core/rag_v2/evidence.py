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
from dataclasses import replace
from typing import Any, Literal, Mapping, Sequence

from core.query_constraints import QueryConstraints, evaluate_candidate_constraints
from core.rag_v2.contracts import EvidenceBundle, EvidenceItem, EvidenceState


EvidenceCompletenessValue = Literal["complete", "partial", "unknown"]

DEFAULT_CONTEXT_MAX_CHUNKS = 16
DEFAULT_CONTEXT_MAX_CHARS = 16_000

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
_VALID_CONSTRAINT_STATUSES = {
    "exact",
    "compatible",
    "neutral",
    "unknown",
    "mismatch",
}


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
    ):
        value = raw.get(field)
        if value is not None and field not in metadata:
            metadata[field] = value
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


def _candidate_confidence(
    raw: Mapping[str, Any],
    *,
    rerank_succeeded: bool | None,
    force_retrieved: bool,
) -> Literal["verified", "retrieved"]:
    if force_retrieved or rerank_succeeded is False:
        return "retrieved"
    declared = str(raw.get("confidence") or "").strip().casefold()
    rerank_status = str(raw.get("rerank_status") or "").strip().casefold()
    if declared == "verified" or rerank_status == "verified":
        return "verified"
    return "retrieved"


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
                metadata=_metadata(raw, section_kind=section_kind),
            ),
            None,
        )
    except (TypeError, ValueError, AttributeError):
        return None, "invalid_candidate_excluded"


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

    role = str(metadata.get("evidence_role") or "").casefold()
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


def _select_context_items(
    items: Sequence[EvidenceItem],
    *,
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
    replacements: dict[str, EvidenceItem] = {}
    selected_ids: set[str] = set()
    used_chars = 0
    budget_limited = False
    for _, item in ranked:
        if len(selected_ids) >= max_context_chunks or used_chars >= max_context_chars:
            budget_limited = True
            break
        remaining = max_context_chars - used_chars
        content = item.content
        if len(content) > remaining:
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "context_truncated": True,
                    "original_content_chars": len(content),
                }
            )
            item = replace(item, content=content[:remaining], metadata=metadata)
            budget_limited = True
        replacements[item.chunk_id] = item
        selected_ids.add(item.chunk_id)
        used_chars += len(item.content)

    if len(selected_ids) < len(items):
        budget_limited = True
    bounded_items = tuple(replacements.get(item.chunk_id, item) for item in items)
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


def assemble_evidence_bundle(
    *,
    query: str,
    candidates: Sequence[Mapping[str, Any] | EvidenceItem],
    constraints: QueryConstraints | None = None,
    overview_candidates: Sequence[Mapping[str, Any] | EvidenceItem] = (),
    answer_shape: str | None = None,
    rerank_succeeded: bool | None = None,
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
            seen_chunk_ids.add(item.chunk_id)
            converted.append(item)
            overview_added = True
        if overview_added:
            reasons.append("overview_full_document_added")

    ordered_items = _group_and_sort(converted)
    bounded_items, context_ids, budget_limited = _select_context_items(
        ordered_items,
        overview_requested=overview_requested,
        max_context_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
    )
    if budget_limited:
        reasons.append("context_budget_limited")

    missing_ids = tuple(dict.fromkeys(
        str(value or "").strip() for value in missing_requirement_ids if str(value or "").strip()
    ))
    effective_completeness: EvidenceCompletenessValue = completeness
    if missing_ids:
        effective_completeness = "partial"
    if budget_limited and effective_completeness == "complete":
        # Without requirement-to-chunk coverage proof, discarded or truncated
        # content may contain a required fact.  A bounded context therefore
        # cannot continue to claim complete coverage merely because the wider
        # candidate pool was complete.
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
    return EvidenceBundle(
        state=state,
        items=bounded_items,
        context_item_ids=context_ids,
        # Retrieved evidence remains visible as a source with an explicit
        # confidence label; degradation must not masquerade as zero evidence.
        answer_source_ids=context_ids,
        missing_requirement_ids=missing_ids,
    )


__all__ = [
    "DEFAULT_CONTEXT_MAX_CHARS",
    "DEFAULT_CONTEXT_MAX_CHUNKS",
    "assemble_evidence_bundle",
]
