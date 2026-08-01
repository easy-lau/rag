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

from core.query_constraints import (
    QueryConstraints,
    evaluate_candidate_constraints,
    extract_document_constraint_identity,
)
from core.rag_v2.bridge_resolution import (
    adjudicate_answer_claims,
    answer_target_terms,
    bridge_dependency_ids_for_answer,
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
from core.rag_v2.contracts import (
    AnswerRequirementV2,
    EvidenceBundle,
    EvidenceItem,
    EvidenceState,
    validate_answer_requirement_graph,
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
    if not requirement.scope_product and not requirement.scope_version:
        return None
    return QueryConstraints(
        product=requirement.scope_product,
        version=requirement.scope_version,
        explicit_version=requirement.scope_explicit_version,
        matched_text=" ".join(
            value
            for value in (
                str(requirement.scope_product or ""),
                str(requirement.scope_version or ""),
            )
            if value
        ),
        extraction_reason="requirement_local_scope",
    )


def _candidate_matches_requirement_scope(
    requirement: AnswerRequirementV2,
    item: EvidenceItem,
) -> bool:
    constraints = _requirement_constraints(requirement)
    if constraints is None:
        return True
    status = evaluate_candidate_constraints(constraints, item.to_dict()).status
    return status not in {"mismatch", "unknown"}


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
    for clause in re.split(r"[\n。；;]+", item.content):
        normalized = re.sub(r"\s+", "", clause).casefold()
        if not normalized or not all(
            content_contains_positive_subject(clause, subject)
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
    not repeat the document title.  A retrieved facet cannot fan out to an
    entire document: only an admitted current-query seed whose source
    filename/title/root heading matches the answer target establishes the
    anchor, and only bounded same-document expansion siblings may inherit it.
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

    root_ids_by_document: dict[tuple[str, str], set[str]] = defaultdict(set)
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
        # Root inheritance is reserved for a genuinely multi-section document.
        # A narrow title-only result such as ``餐饮补贴标准`` + ``D级为100``
        # cannot use its filename to bypass the target-in-claim check.
        if len(section_labels_by_document.get(document_key, set())) < 2:
            continue
        for requirement in answer_requirements:
            if content_matches_answer_target(
                requirement.description,
                root_text,
                bridge_subjects=bridge_subjects_by_answer[requirement.id],
            ):
                root_ids_by_document[document_key].add(requirement.id)

    anchored: list[EvidenceItem] = []
    for item in items:
        root_ids = root_ids_by_document.get((item.kb_id, item.doc_id), set())
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
    content_terms = _coverage_terms(content)
    if not content_terms:
        return exact_answer_ids

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
        return exact_answer_ids
    strongest_score = max(scores.values())
    # Generic subject overlap can make the broad lexical matcher accept every
    # coordinated answer (for example ``普通员工制度`` against住宿/交通/餐补).
    # A non-exact lexical match is therefore retained only when the visible
    # chunk also contains a term that distinguishes that answer.
    discriminative_lexical_ids = lexical_answer_ids & set(scores)
    return exact_answer_ids | discriminative_lexical_ids | {
        requirement_id
        for requirement_id, score in scores.items()
        if score == strongest_score
    }


def _reconcile_multi_hop_links(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
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
        direct_subject_answer_ids = {
            str(value)
            for value in item.metadata.get(
                "direct_subject_answer_requirement_ids",
                [],
            )
            if isinstance(value, str)
            and all(
                bridge_requirement_by_id[dependency_id].importance == "helpful"
                and bridge_requirement_by_id[dependency_id].source == "inferred"
                for dependency_id in bridge_dependencies_by_answer.get(
                    str(value),
                    (),
                )
            )
        }
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
        # In a multi-hop plan even an exact wording match, model annotation or
        # query provenance is not a relationship path.  The answer survives
        # only when it joins the resolved, same-scope bridge value.
        retained_answer_ids = answer_support_ids & (
            joined_ids | direct_subject_ids
        )
        rejected_answer_ids = answer_support_ids - retained_answer_ids
        if not rejected_answer_ids:
            metadata = dict(item.metadata)
            metadata["bridge_linked_requirement_ids"] = sorted(
                retained_answer_ids
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


def _reconcile_answer_claim_assertions(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
    overview_requested: bool,
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
            dependency_ids = bridge_dependency_ids_for_answer(
                answer,
                requirements,
            )
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
            assertions = adjudicate_answer_claims(
                answer.description,
                item.content,
                bridge_subjects=bridge_subjects,
                bridge_values=bridge_values,
                document_root_target_verified=root_verified,
            )
            # An overview intentionally asks for the bounded document itself;
            # its structural support contract is separate from fact claims.
            if not assertions and not overview_requested:
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
        if item.role == "conflicting" or metadata.get("bridge_conflicts"):
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
    explicit_ids = tuple(
        requirement_id
        for requirement_id in explicit_ids
        if _candidate_matches_requirement_scope(
            requirement_by_id[requirement_id],
            item,
        )
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

    lexical_ids: list[str] = []
    exact_ids: set[str] = set()
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
        visible_answer_ids = _multi_answer_text_support_ids(
            requirements=requirements,
            content=item.content,
            lexical_ids=lexical_ids,
            exact_ids=exact_ids,
        )
        # A merged expansion index is only provenance.  For coordinated
        # questions it may preserve a required-answer mapping only when the
        # retained text distinguishes that answer (or a trusted verifier
        # explicitly annotated the requirement).
        allowed_answer_ids = visible_answer_ids | set(explicit_ids)
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
        )
        structural_ids = tuple(
            requirement_id
            for requirement_id in structural_ids
            if requirement_by_id[requirement_id].role == "bridge"
            or requirement_id in lexical_ids
        )

    # Retrieval/query provenance is a ranking hint, never positive fact
    # support.  ``structural_ids`` is populated only by the explicit bounded
    # overview exception above.  Multi-hop lexical answers are subsequently
    # required to traverse a resolved bridge fact as well.
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

    if len(selected_ids) < len(eligible_ids):
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


_COLLECTION_QUERY_PREFIX_RE = re.compile(
    r"^(?:(?:请问|请|麻烦|帮我(?:查|看|列|说明)?(?:一下)?)\s*|"
    r"(?:列出|列举|罗列|说明|概述|介绍)\s*|"
    r"(?:如何|怎么)(?:完成|执行|办理|操作|实现)?\s*|"
    r"(?:what\s+(?:is|are)\s+(?:the\s+)?|list\s+(?:all\s+)?(?:the\s+)?))",
    re.IGNORECASE,
)
_COLLECTION_QUERY_SUFFIX_RE = re.compile(
    r"(?:是什么|是哪些|(?:都)?有哪些(?:内容|项目|元素|成员|种类|方式|步骤|"
    r"要求|标准|规则|措施|选项)?|有(?:哪些|什么)(?:内容|项目|元素|成员|"
    r"种类|方式|步骤|要求|标准|规则|措施|选项)?|"
    r"(?:包括|包含)(?:哪些|什么)(?:内容|项目|元素|成员|种类|方式|步骤|"
    r"要求|标准|规则|措施|选项)?|分别是什么|(?:完整)?(?:清单|列表))$",
    re.IGNORECASE,
)
_COLLECTION_STRONG_CLOSURE_RE = re.compile(
    r"(?:仅(?:包括|包含|有|限于|支持|允许)|仅有|只有|"
    r"(?:共|一共|总计|合计)\s*(?:\d+|[一二三四五六七八九十百两]+)\s*"
    r"(?:项|条|个|种|类|步|阶段|部分|方面)|"
    r"全部(?:包括|包含|为|是|如下)|"
    r"所有(?:内容|项目|条目|成员|元素|项|条)?(?:如下|为|是)|"
    r"(?:由|是由)(?=.{1,200}(?:构成|组成))|"
    r"分为(?=.{1,200}(?:[、,，;；。]|$))|"
    r"(?:consists?\s+of|comprises?|only\s+(?:includes?|contains?)))",
    re.IGNORECASE,
)
_COLLECTION_INTRO_CLOSURE_RE = re.compile(
    # Match only the closure predicate.  Shape nouns immediately before it
    # (``process follows``, ``steps follow``) belong to the target subject and
    # must participate in target alignment.
    r"(?:如下(?:所示)?|以下(?:为|是)|依次为|分别为|"
    r"(?:the\s+)?following)",
    re.IGNORECASE,
)
_COLLECTION_NON_EXHAUSTIVE_RE = re.compile(
    r"(?:^|[^仅])(?:包括|包含)|例如|诸如|之一|部分|主要|至少|等等|\betc\.?\b|"
    r"(?:\.\.\.|……)",
    re.IGNORECASE,
)
_COLLECTION_COMPOUND_CONNECTOR_RE = re.compile(
    # A directly labelled workflow may enumerate steps with conjunctions or
    # with an authored sequence (``主管审批后由财务复核``).  Both are bounded
    # compound assertions; the surrounding target/closure checks still reject
    # a topical sentence or non-exhaustive wording such as ``主要``/``例如``.
    r"(?:并且|并|同时|以及|且|及|、|[,，;；]|"
    r"然后|随后|再由|最后(?:由)?|依次|后(?:再)?由?|\band\b)",
    re.IGNORECASE,
)
_COLLECTION_NUMBERED_ITEM_RE = re.compile(
    r"(?:^\s*|(?<=[：:；;。]))(?:步骤\s*)?"
    r"(?:\d+|[一二三四五六七八九十百]+)\s*[、.．):：]\s*\S+",
    re.MULTILINE,
)
_COLLECTION_BULLET_ITEM_RE = re.compile(r"(?m)^\s*[-*+]\s+\S+")
_COLLECTION_SEQUENCE_RE = re.compile(
    r"(?:首先|第一步).{0,240}(?:然后|其次|第二步).{0,240}(?:最后|最终)",
    re.IGNORECASE | re.DOTALL,
)


def _collection_target_description(description: str) -> str:
    """Remove list-question grammar while retaining the business target."""

    original = re.sub(r"\s+", " ", str(description or "")).strip()
    value = original.strip(" \t，,。；;：:！!？?")
    for _ in range(3):
        updated = _COLLECTION_QUERY_PREFIX_RE.sub("", value).strip()
        updated = _COLLECTION_QUERY_SUFFIX_RE.sub("", updated).strip()
        updated = updated.strip(" \t，,。；;：:！!？?")
        if updated == value:
            break
        value = updated
    return value if len(re.sub(r"\s+", "", value)) >= 2 else original


def _collection_bridge_subjects(
    requirement: AnswerRequirementV2,
    requirements: tuple[AnswerRequirementV2, ...],
) -> tuple[str, ...]:
    bridge_by_id = {
        value.id: value for value in requirements if value.role == "bridge"
    }
    return tuple(dict.fromkeys(
        subject
        for dependency_id in bridge_dependency_ids_for_answer(
            requirement,
            requirements,
        )
        for bridge in (bridge_by_id.get(dependency_id),)
        if bridge is not None
        if (subject := bridge_subject_for_requirement(bridge))
    ))


def _collection_target_matches(
    requirement: AnswerRequirementV2,
    content: str,
    *,
    requirements: tuple[AnswerRequirementV2, ...],
) -> bool:
    normalized_content = re.sub(r"\s+", " ", str(content or "")).strip()
    if not normalized_content:
        return False
    return content_matches_complete_answer_target(
        _collection_target_description(requirement.description),
        normalized_content,
        bridge_subjects=_collection_bridge_subjects(
            requirement,
            requirements,
        ),
    )


def _collection_declaration_units(content: str) -> tuple[str, ...]:
    units: list[str] = []
    for line in str(content or "").splitlines():
        normalized_line = re.sub(r"\s+", " ", line).strip(
            " \t#|【】。；;！？!?"
        )
        if not normalized_line:
            continue
        units.append(normalized_line)
        units.extend(
            value.strip()
            for value in re.split(r"[。；;！？!?]+", normalized_line)
            if value.strip()
        )
    return tuple(dict.fromkeys(units))


def _closure_subject(unit: str, marker_start: int) -> str:
    """Return the phrase directly governed by a closure predicate."""

    prefix = unit[:marker_start].strip(" \t，,。；;：:")
    for separator in ("：", ":"):
        if separator not in prefix:
            continue
        parent, local = prefix.rsplit(separator, 1)
        # ``broad target: sub-facet includes ...`` is a sub-collection, while
        # ``broad target: follows ...`` still binds the broad target.
        prefix = local.strip() or parent.strip()
        break
    return prefix


def _has_bounded_collection_structure(content: str, *, tail: str) -> bool:
    numbered_count = len(_COLLECTION_NUMBERED_ITEM_RE.findall(content))
    bullet_count = len(_COLLECTION_BULLET_ITEM_RE.findall(content))
    if numbered_count >= 2 or bullet_count >= 2:
        return True
    if _COLLECTION_SEQUENCE_RE.search(content):
        return True

    table_lines = [
        line.strip()
        for line in content.splitlines()
        if line.count("|") >= 2
    ]
    table_data_rows = [
        line
        for line in table_lines[1:]
        if not re.fullmatch(r"\|?[\s:|.-]+\|?", line)
    ]
    if table_data_rows:
        return True

    members = [
        value.strip()
        for value in re.split(r"(?:、|[,，;；]|\s+(?:以及|和|及|与)\s+)", tail)
        if value.strip(" \t：:")
    ]
    return len(members) >= 2


def _item_has_explicit_collection_closure(
    item: EvidenceItem,
    *,
    requirement: AnswerRequirementV2,
    requirements: tuple[AnswerRequirementV2, ...],
) -> bool:
    """Prove a collection from one source-authored, target-bound statement."""

    for unit in _collection_declaration_units(item.content):
        for match in _COLLECTION_STRONG_CLOSURE_RE.finditer(unit):
            subject = _closure_subject(unit, match.start())
            if _collection_target_matches(
                requirement,
                subject,
                requirements=requirements,
            ):
                return True
        for match in _COLLECTION_INTRO_CLOSURE_RE.finditer(unit):
            subject = _closure_subject(unit, match.start())
            if not _collection_target_matches(
                requirement,
                subject,
                requirements=requirements,
            ):
                continue
            if _has_bounded_collection_structure(
                item.content,
                tail=unit[match.end():],
            ):
                return True

    # A directly labelled compound rule is a closed answer assertion even
    # without list wording: both actions belong to one authored predicate.
    # Non-exhaustive language (``includes``, ``for example``, ``main``) keeps
    # the requirement partial.  This admits process/measure clauses across
    # domains while refusing a topical sub-list inherited from a filename.
    assertions_by_requirement = item.metadata.get("answer_claim_assertions", {})
    raw_assertions = (
        assertions_by_requirement.get(requirement.id, [])
        if isinstance(assertions_by_requirement, Mapping)
        else []
    )
    assertions = tuple(
        value for value in raw_assertions if isinstance(value, Mapping)
    )
    has_compound_rule_assertion = any(
        isinstance(value, Mapping)
        and str(value.get("result_kind") or "")
        in {"scalar", "categorical", "normative", "procedure"}
        for value in assertions
    )
    if not has_compound_rule_assertion:
        return False
    colon_units = tuple(
        unit
        for unit in _collection_declaration_units(item.content)
        if re.search(r"[：:]", unit) is not None
    )
    inherited_root_ids = {
        str(value)
        for value in item.metadata.get(
            "document_root_answer_requirement_ids",
            [],
        )
        if isinstance(value, str)
    }
    for unit in colon_units:
        separator_match = re.search(r"[：:]", unit)
        if separator_match is None:
            continue
        subject = unit[:separator_match.start()].strip()
        result = unit[separator_match.end():].strip()
        target_bound = _collection_target_matches(
            requirement,
            subject,
            requirements=requirements,
        ) or (
            requirement.id not in inherited_root_ids
            and len(assertions) == 1
            and len(colon_units) == 1
        )
        if (
            target_bound
            and _COLLECTION_COMPOUND_CONNECTOR_RE.search(result)
            and not _COLLECTION_NON_EXHAUSTIVE_RE.search(result)
        ):
            return True
    return False


def _complete_structured_tables(
    items: Sequence[EvidenceItem],
) -> tuple[tuple[EvidenceItem, ...], ...]:
    grouped: dict[tuple[str, str, str], list[EvidenceItem]] = defaultdict(list)
    for item in items:
        table_id = str(item.metadata.get("table_id") or "").strip()
        if table_id:
            grouped[(item.kb_id, item.doc_id, table_id)].append(item)

    complete: list[tuple[EvidenceItem, ...]] = []
    for table_items in grouped.values():
        expected_counts: set[int] = set()
        indexed_items: dict[int, EvidenceItem] = {}
        invalid = False
        for item in table_items:
            try:
                expected = int(item.metadata.get("table_part_count"))
                index = int(item.metadata.get("table_part_index"))
            except (TypeError, ValueError):
                invalid = True
                break
            if expected <= 0 or index < 0 or index >= expected:
                invalid = True
                break
            expected_counts.add(expected)
            if index in indexed_items:
                invalid = True
                break
            indexed_items[index] = item
        if invalid or len(expected_counts) != 1:
            continue
        expected = next(iter(expected_counts))
        if set(indexed_items) != set(range(expected)):
            continue
        complete.append(tuple(indexed_items[index] for index in range(expected)))
    return tuple(complete)


def _table_local_scope_text(items: Sequence[EvidenceItem]) -> str:
    values: list[str] = []
    for item in items:
        raw_path = item.metadata.get("section_path")
        if isinstance(raw_path, (list, tuple)):
            # The first path component is the document title.  It proves
            # retrieval topic, not that this particular table enumerates the
            # document-wide collection.
            values.extend(
                str(value).strip()
                for value in raw_path[1:]
                if str(value or "").strip()
            )
        for line in item.content.splitlines():
            normalized = line.strip()
            if (
                normalized.count("|") >= 2
                and not re.fullmatch(r"\|?[\s:|.-]+\|?", normalized)
            ):
                values.append(normalized.strip(" |"))
                break
    return "\n".join(dict.fromkeys(values))


def _positive_collection_expectations(
    items: Sequence[EvidenceItem],
    *,
    requirements: tuple[AnswerRequirementV2, ...],
) -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    """Return answer chunks required by each proven exhaustive snapshot.

    A lexical hit cannot prove that a collection is exhaustive.  Accepted
    proofs are source-structural: a verified full-document snapshot, every
    part of one parser-identified table whose local scope matches the target,
    or one target-bound source declaration with explicit closure semantics.
    This works for policies, product manuals, supplier rules and other domains
    without enumerating business sections.

    The returned chunk ids contain only active positive answer claims.  Bridge
    rows and background sections remain available, but do not become members
    of the collection merely because they belong to the same document.
    """

    collection_ids = {
        requirement.id
        for requirement in requirements
        if requirement.role == "answer"
        and requirement.coverage_mode == "collection"
    }
    if not collection_ids:
        return {}, frozenset()
    collection_by_id = {
        requirement.id: requirement
        for requirement in requirements
        if requirement.id in collection_ids
    }

    grouped: dict[tuple[str, str], list[EvidenceItem]] = defaultdict(list)
    for item in items:
        grouped[(item.kb_id, item.doc_id)].append(item)

    complete_documents: set[tuple[str, str]] = set()
    for document_key, document_items in grouped.items():
        has_verified_snapshot_origin = any(
            bool(
                set(item.origins)
                & {"small_document_full", "overview_full_document"}
            )
            for item in document_items
        )
        expected_counts: set[int] = set()
        for item in document_items:
            raw_expected = item.metadata.get("full_document_chunk_count")
            if isinstance(raw_expected, bool):
                continue
            try:
                expected = int(raw_expected)
            except (TypeError, ValueError):
                continue
            if expected > 0:
                expected_counts.add(expected)
        if (
            has_verified_snapshot_origin
            and
            len(expected_counts) == 1
            and len({item.chunk_id for item in document_items})
            == next(iter(expected_counts))
        ):
            complete_documents.add(document_key)

    expected_by_requirement: dict[str, set[str]] = defaultdict(set)
    proven_requirement_ids: set[str] = set()

    # A single source claim can define a closed collection only when the
    # closure predicate is bound to the requirement itself.  Mere plurality
    # or ``includes`` wording is not exhaustive: ``travel standard: transport
    # includes A and B`` still proves only one facet of the broader standard.
    for item in items:
        if item.role not in _COVERAGE_ROLES:
            continue
        for requirement_id in set(item.supports_requirement_ids) & collection_ids:
            requirement = collection_by_id[requirement_id]
            if _item_has_explicit_collection_closure(
                item,
                requirement=requirement,
                requirements=requirements,
            ):
                expected_by_requirement[requirement_id].add(item.chunk_id)
                proven_requirement_ids.add(requirement_id)

    # A parser table has its own stable cardinality.  Require every declared
    # part and a local section/header match; a document title is deliberately
    # excluded so one facet table cannot close a document-wide requirement.
    for table_items in _complete_structured_tables(items):
        local_scope = _table_local_scope_text(table_items)
        for requirement_id, requirement in collection_by_id.items():
            if not all(
                item.role in _COVERAGE_ROLES
                and requirement_id in item.supports_requirement_ids
                for item in table_items
            ):
                continue
            if not _collection_target_matches(
                requirement,
                local_scope,
                requirements=requirements,
            ):
                continue
            expected_by_requirement[requirement_id].update(
                item.chunk_id for item in table_items
            )
            proven_requirement_ids.add(requirement_id)

    for document_key in complete_documents:
        for item in grouped[document_key]:
            if item.role not in _COVERAGE_ROLES:
                continue
            supported_collection_ids = (
                set(item.supports_requirement_ids) & collection_ids
            )
            for requirement_id in supported_collection_ids:
                expected_by_requirement[requirement_id].add(item.chunk_id)
                proven_requirement_ids.add(requirement_id)

    return (
        {
            requirement_id: frozenset(chunk_ids)
            for requirement_id, chunk_ids in expected_by_requirement.items()
        },
        frozenset(proven_requirement_ids),
    )


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
    query_alignment = _query_requirement_alignment(
        requirements=normalized_requirements,
        retrieval_queries=normalized_retrieval_queries,
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

    converted = _attach_document_root_topic_anchors(
        converted,
        requirements=normalized_requirements,
    )

    if bridge_resolution_required:
        converted = _reconcile_multi_hop_links(
            converted,
            requirements=normalized_requirements,
        )
    converted = _reconcile_answer_claim_assertions(
        converted,
        requirements=normalized_requirements,
        overview_requested=overview_requested,
    )
    converted, answer_conflict = _reconcile_same_source_answer_conflicts(
        converted,
        requirements=normalized_requirements,
    )
    if answer_conflict:
        reasons.append("conflicting_active_answer_claims")

    (
        collection_expected_chunk_ids,
        proven_collection_requirement_ids,
    ) = _positive_collection_expectations(
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
        if bridge_resolution_required:
            # The renderer budget is a semantic boundary: a bridge value or
            # result that existed only beyond the retained prefix no longer
            # exists for generation.  Rebuild every join on the final visible
            # text, clearing stale join metadata in the reconciler.
            bounded_items = tuple(_reconcile_multi_hop_links(
                bounded_items,
                requirements=normalized_requirements,
            ))
            bounded_items = tuple(_reconcile_answer_claim_assertions(
                bounded_items,
                requirements=normalized_requirements,
                overview_requested=overview_requested,
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
        directly_satisfied_answer_ids = {
            str(value)
            for chunk_id in context_ids
            for item in (selected_by_id[chunk_id],)
            for field in (
                "direct_subject_answer_requirement_ids",
                "direct_subject_bridge_bypass_requirement_ids",
            )
            for value in item.metadata.get(field, [])
            if isinstance(value, str)
            and value in item.supports_requirement_ids
        }
        answer_by_id = {
            requirement.id: requirement
            for requirement in normalized_requirements
            if requirement.role == "answer"
        }
        waived_bridge_ids = {
            bridge.id
            for answer_id in directly_satisfied_answer_ids
            for answer in (answer_by_id.get(answer_id),)
            if answer is not None
            for bridge in normalized_requirements
            if bridge.role == "bridge"
            and bridge.importance == "helpful"
            and bridge.source == "inferred"
            and bridge.id in bridge_dependency_ids_for_answer(
                answer,
                normalized_requirements,
            )
        }
        if waived_bridge_ids:
            reasons.append("inferred_bridge_bypassed_by_direct_answer")
        missing_ids = tuple(
            requirement.id
            for requirement in normalized_requirements
            if requirement.id in coverage_required_ids
            and (
                requirement.id not in covered_ids
                or (
                    requirement.role == "answer"
                    and requirement.coverage_mode == "collection"
                    and (
                        requirement.id
                        not in proven_collection_requirement_ids
                        or not collection_expected_chunk_ids.get(
                            requirement.id,
                            frozenset(),
                        ).issubset({
                            chunk_id
                            for chunk_id in context_ids
                            if requirement.id
                            in selected_by_id[
                                chunk_id
                            ].supports_requirement_ids
                        })
                    )
                )
            )
            and requirement.id not in waived_bridge_ids
        )
        unproven_collection_ids = {
            requirement.id
            for requirement in normalized_requirements
            if requirement.role == "answer"
            and requirement.coverage_mode == "collection"
            and requirement.id not in proven_collection_requirement_ids
        }
        incomplete_collection_ids = {
            requirement_id
            for requirement_id, expected_chunk_ids
            in collection_expected_chunk_ids.items()
            if not expected_chunk_ids.issubset({
                chunk_id
                for chunk_id in context_ids
                if requirement_id
                in selected_by_id[chunk_id].supports_requirement_ids
            })
        }
        if unproven_collection_ids:
            reasons.append("collection_snapshot_unproven")
        if incomplete_collection_ids:
            reasons.append("collection_context_incomplete")
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
