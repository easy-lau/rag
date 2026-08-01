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
from core.rag_v2.contracts import (
    AnswerRequirementV2,
    EvidenceBundle,
    EvidenceItem,
    EvidenceState,
)


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
_BRIDGE_RELATION_TARGET_RE = re.compile(
    r"(?:对应|属于|归属于|归属|映射到|映射为|认定为|划分为|列为|等同于)\s*"
    r"([^\s，,。；;|]{1,32})",
    re.IGNORECASE,
)
_BRIDGE_IDENTIFIER_RE = re.compile(
    r"(?:[A-Za-z]+(?:\d+(?:\.\d+)*)?(?:级|类|档|型|版|组|岗|层|序列)|"
    r"[A-Za-z]+\d+(?:\.\d+)*|"
    r"\d+(?:\.\d+)+(?:版|级|类|档|型|组|层)?)",
    re.IGNORECASE,
)
_STRUCTURED_VALUE_SPLIT_RE = re.compile(r"\s*(?:\||\t|->|=>|→|=)\s*")
_GENERIC_BRIDGE_VALUES = {
    "内容",
    "信息",
    "标准",
    "要求",
    "结果",
    "等级",
    "级别",
    "类型",
    "类别",
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
        "expansion_query_indexes",
        "supports_requirement_ids",
        "role",
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


def _bridge_link_values(content: str, *, excluded_text: str) -> set[str]:
    """Extract conservative join values from a bridge evidence chunk.

    A multi-hop answer is safe only when the mapping chunk and value chunk
    share the resolved intermediate value.  The value is intentionally derived
    from relation targets, stable mixed identifiers (for example ``D级`` or
    ``P2``), or explicit table/key-value cells.  Broad topical overlap such as
    "员工" or "标准" is never enough to join two chunks.
    """

    normalized = re.sub(r"\s+", " ", str(content or "")).strip().casefold()
    excluded = re.sub(r"\s+", " ", str(excluded_text or "")).strip().casefold()
    values: set[str] = set()

    for match in _BRIDGE_RELATION_TARGET_RE.finditer(normalized):
        value = match.group(1).strip(" ：:，,。；;|[]【】()（）")
        if (
            2 <= len(value) <= 32
            and value not in _GENERIC_BRIDGE_VALUES
            and value not in excluded
        ):
            values.add(value)

    for match in _BRIDGE_IDENTIFIER_RE.finditer(normalized):
        value = match.group(0).strip().casefold()
        if len(value) >= 2 and value not in excluded:
            values.add(value)

    for line in normalized.splitlines():
        if not _STRUCTURED_VALUE_SPLIT_RE.search(line):
            continue
        for cell in _STRUCTURED_VALUE_SPLIT_RE.split(line):
            value = cell.strip(" ：:，,。；;[]【】()（）")
            if (
                2 <= len(value) <= 24
                and not value.isdigit()
                and value not in _GENERIC_BRIDGE_VALUES
                and value not in excluded
            ):
                values.add(value)
    return values


def _has_bridge_link(
    bridge_content: str,
    answer_content: str,
    *,
    excluded_text: str,
) -> bool:
    answer = re.sub(r"\s+", " ", str(answer_content or "")).strip().casefold()
    if not answer:
        return False
    return any(
        value in answer
        for value in _bridge_link_values(
            bridge_content,
            excluded_text=excluded_text,
        )
    )


def _reconcile_multi_hop_links(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
) -> list[EvidenceItem]:
    """Drop provisional answer support that is not joined to bridge evidence."""

    requirement_by_id = {item.id: item for item in requirements}
    bridge_ids = {
        item.id for item in requirements if item.role == "bridge"
    }
    if not bridge_ids:
        return list(items)
    bridge_items = [
        item
        for item in items
        if item.role in _COVERAGE_ROLES
        and bool(set(item.supports_requirement_ids) & bridge_ids)
    ]
    # With no bridge evidence at all, retain independently retrieved answer
    # fragments as partial evidence; the missing bridge requirement prevents a
    # complete answer.  Once a bridge is present, however, provisional answer
    # support must join to its resolved value or it is unsafe to combine.
    if not bridge_items:
        return list(items)
    excluded_text = " ".join(item.description for item in requirements)
    reconciled: list[EvidenceItem] = []
    for item in items:
        raw_provisional = item.metadata.get(
            "provisional_support_requirement_ids"
        )
        provisional_ids = {
            str(value).strip()
            for value in raw_provisional
            if isinstance(value, str) and str(value).strip()
        } if isinstance(raw_provisional, (list, tuple)) else set()
        provisional_answer_ids = {
            requirement_id
            for requirement_id in provisional_ids
            if requirement_id in requirement_by_id
            and requirement_by_id[requirement_id].role == "answer"
        }
        if not provisional_answer_ids:
            reconciled.append(item)
            continue

        linked = any(
            bridge.chunk_id == item.chunk_id
            or _has_bridge_link(
                bridge.content,
                item.content,
                excluded_text=excluded_text,
            )
            for bridge in bridge_items
        )
        if linked:
            metadata = dict(item.metadata)
            metadata["bridge_linked_requirement_ids"] = sorted(
                provisional_answer_ids
            )
            reconciled.append(replace(item, metadata=metadata))
            continue

        supports = tuple(
            requirement_id
            for requirement_id in item.supports_requirement_ids
            if requirement_id not in provisional_answer_ids
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
            provisional_answer_ids
        )
        reconciled.append(replace(
            item,
            role=role,
            supports_requirement_ids=supports,
            metadata=metadata,
        ))
    return reconciled


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
    explicit_ids = (
        tuple(
            requirement_id
            for requirement_id in item.supports_requirement_ids
            if requirement_id in allowed_ids
        )
        if trust_candidate_annotations
        else ()
    )
    aligned_ids: tuple[str, ...] = ()
    if use_query_indexes:
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
    if use_structural_provenance and len(required_answer_ids) == 1:
        current_query_seed = "initial_retrieval" in item.origins
        bounded_overview_chunk = bool(
            len(requirements) == 1
            and overview_requested
            and "overview_full_document" in item.origins
        )
        if current_query_seed or bounded_overview_chunk:
            # A candidate that survived the current-query relevance gate is a
            # deterministic retrieval observation for a single answer target.
            # Likewise, an explicit overview may use every bounded chunk of the
            # already anchored small document.  These are complement signals,
            # never semantic verification or standalone direct entailment.
            structural_ids = (required_answer_ids[0],)

    lexical_ids: list[str] = []
    exact_ids: set[str] = set()
    for requirement in requirements:
        supported, exact = _text_support(requirement.description, item.content)
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
    if lexical_bridge_ids:
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
        )
        structural_ids = tuple(
            requirement_id
            for requirement_id in structural_ids
            if requirement_by_id[requirement_id].role == "bridge"
            or requirement_id in lexical_ids
        )

    candidate_ids = (
        set(explicit_ids)
        | set(aligned_ids)
        | set(structural_ids)
        | set(lexical_ids)
    )
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


def _select_context_items(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    coverage_required_ids: frozenset[str],
    retrieval_queries: tuple[str, ...],
    query_alignment: Mapping[int, str],
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
    selection_order = [*coverage_first, *pending]

    replacements: dict[str, EvidenceItem] = {}
    selected_ids: set[str] = set()
    used_chars = 0
    budget_limited = False
    for _, item in selection_order:
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
            # Annotation/index support may refer to content outside the retained
            # prefix.  After truncation, retain only support independently
            # visible in the actual generation text.
            item = _mapped_item(
                item,
                requirements=requirements,
                retrieval_queries=retrieval_queries,
                query_alignment=query_alignment,
                trust_candidate_annotations=False,
                use_query_indexes=False,
                use_structural_provenance=False,
                overview_requested=overview_requested,
            )
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
    requirements: Sequence[AnswerRequirementV2] = (),
    retrieval_queries: Sequence[str] = (),
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
    normalized_retrieval_queries = _normalize_retrieval_queries(
        retrieval_queries
    )
    query_alignment = _query_requirement_alignment(
        requirements=normalized_requirements,
        retrieval_queries=normalized_retrieval_queries,
    )
    coverage_required_ids = frozenset(
        requirement.id
        for requirement in normalized_requirements
        if requirement.importance == "required"
        or (answer_shape == "multi_hop" and requirement.role == "bridge")
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
            overview_requested=overview_requested,
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
                overview_requested=overview_requested,
            )
            seen_chunk_ids.add(item.chunk_id)
            converted.append(item)
            overview_added = True
        if overview_added:
            reasons.append("overview_full_document_added")

    if answer_shape == "multi_hop":
        converted = _reconcile_multi_hop_links(
            converted,
            requirements=normalized_requirements,
        )

    ordered_items = _group_and_sort(converted)
    bounded_items, context_ids, budget_limited = _select_context_items(
        ordered_items,
        requirements=normalized_requirements,
        coverage_required_ids=coverage_required_ids,
        retrieval_queries=normalized_retrieval_queries,
        query_alignment=query_alignment,
        overview_requested=overview_requested,
        max_context_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
    )
    if budget_limited:
        reasons.append("context_budget_limited")

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
    effective_completeness: EvidenceCompletenessValue = completeness
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
        else context_ids
    )
    return EvidenceBundle(
        state=state,
        items=bounded_items,
        context_item_ids=context_ids,
        # Background and conflicting chunks may provide bounded document
        # context, but cannot be advertised as positive answer support.
        answer_source_ids=answer_source_ids,
        missing_requirement_ids=missing_ids,
    )


__all__ = [
    "DEFAULT_CONTEXT_MAX_CHARS",
    "DEFAULT_CONTEXT_MAX_CHUNKS",
    "assemble_evidence_bundle",
]
