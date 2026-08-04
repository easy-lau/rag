"""Shared retrieval helpers used by the active V2 evidence runner and search diagnostics."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Sequence

from core.evidence_ambiguity import EvidenceScopeChoice, ExplicitScopeComparisonPlan
from core.query_constraints import (
    QueryConstraints,
    candidate_section_key,
    evaluate_candidate_constraints,
    inherit_document_constraint_metadata,
)
from core.retriever import MAX_EVIDENCE_SCOPE_DOCUMENTS

RERANK_CANDIDATE_MIN = 12
RERANK_CANDIDATE_MULTIPLIER = 3
RERANK_CANDIDATE_MAX = 30
SIMPLE_RERANK_CANDIDATE_MIN = 8
SIMPLE_RERANK_CANDIDATE_MULTIPLIER = 2
SIMPLE_RERANK_CANDIDATE_MAX = 20

@dataclass(frozen=True)
class _EvidenceScopeSliceFilter:
    kb_id: uuid.UUID
    doc_id: uuid.UUID
    section_key: str | None
    chunk_ids: tuple[uuid.UUID, ...]
    is_anchor: bool

@dataclass(frozen=True)
class _EvidenceScopeChoiceFilter:
    key: str
    label: str
    products: tuple[str, ...]
    canonical_products: tuple[str, ...]
    versions: tuple[str, ...]
    projects: tuple[str, ...]
    filenames: tuple[str, ...]
    kb_ids: tuple[uuid.UUID, ...]
    doc_ids: tuple[uuid.UUID, ...]
    anchor_doc_ids: tuple[uuid.UUID, ...]
    companion_doc_ids: tuple[uuid.UUID, ...]
    scope_slices: tuple[_EvidenceScopeSliceFilter, ...] = ()

@dataclass(frozen=True)
class _EvidenceScopeFilter:
    mode: str
    kb_ids: tuple[uuid.UUID, ...]
    doc_ids: tuple[uuid.UUID, ...]
    choices: tuple[_EvidenceScopeChoiceFilter, ...]
    valid: bool = True
    invalid_reason: str | None = None

    @property
    def compare_all(self) -> bool:
        return self.mode == "compare_all"

    def label_by_document(self) -> dict[str, str]:
        labels: dict[str, list[str]] = {}
        allowed_doc_ids = {str(value) for value in self.doc_ids}
        for choice in self.choices:
            for doc_id in choice.doc_ids:
                doc_key = str(doc_id)
                if doc_key not in allowed_doc_ids:
                    continue
                values = labels.setdefault(doc_key, [])
                if choice.label not in values:
                    values.append(choice.label)
        return {
            doc_id: " / ".join(values)
            for doc_id, values in labels.items()
            if values
        }

def _bounded_text_values(value: object, *, limit: int = 20) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("scope text values must be a list")
    result: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError("scope text value must be a string")
        item = raw.strip()
        if not item or len(item) > 500:
            raise ValueError("scope text value is empty or too long")
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) > limit:
            raise ValueError("too many scope text values")
    return tuple(result)

def _bounded_uuid_values(value: object, *, limit: int) -> tuple[uuid.UUID, ...]:
    if not isinstance(value, list):
        raise ValueError("scope ids must be a list")
    result: list[uuid.UUID] = []
    seen: set[str] = set()
    for raw in value:
        try:
            item = uuid.UUID(str(raw))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("scope id must be a UUID") from exc
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if len(result) > limit:
            raise ValueError("too many scope ids")
    return tuple(result)

def _scope_slice_tokens(
    scope_slice: _EvidenceScopeSliceFilter,
) -> set[tuple[str, ...]]:
    if scope_slice.section_key:
        return {
            (
                str(scope_slice.kb_id),
                str(scope_slice.doc_id),
                "section",
                scope_slice.section_key,
            )
        }
    return {
        (
            str(scope_slice.kb_id),
            str(scope_slice.doc_id),
            "chunk",
            str(chunk_id),
        )
        for chunk_id in scope_slice.chunk_ids
    }

def _invalid_evidence_scope_filter(
    mode: str,
    reason: str,
) -> _EvidenceScopeFilter:
    return _EvidenceScopeFilter(
        mode=mode if mode in {"single", "compare_all"} else "invalid",
        kb_ids=(),
        doc_ids=(),
        choices=(),
        valid=False,
        invalid_reason=reason,
    )

def _normalize_evidence_scope_filter(
    value: object,
    *,
    authorized_kb_ids: Sequence[uuid.UUID | str],
) -> _EvidenceScopeFilter | None:
    """Validate a request-local clarification selection without granting access.

    The persisted pending state is only a set of user-selectable candidates.  The
    current request's already-authorized ``kb_ids`` remain the security boundary;
    selected document ids are additionally required to appear both at the top
    level and inside one of the supplied choices.  Any malformed shape fails
    closed to an empty scoped retrieval and never falls back to global search.
    """

    if value is None:
        return None
    if not isinstance(value, dict):
        return _invalid_evidence_scope_filter("invalid", "filter_not_object")
    mode = str(value.get("mode") or "").strip()
    if mode not in {"single", "compare_all"}:
        return _invalid_evidence_scope_filter(mode, "invalid_mode")

    try:
        requested_kb_ids = _bounded_uuid_values(value.get("kb_ids"), limit=100)
        requested_doc_ids = _bounded_uuid_values(
            value.get("doc_ids"),
            limit=MAX_EVIDENCE_SCOPE_DOCUMENTS,
        )
        raw_choices = value.get("choices")
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError("choices must be a non-empty list")
        if len(raw_choices) > 6:
            raise ValueError("too many choices")

        choices: list[_EvidenceScopeChoiceFilter] = []
        choice_keys: set[str] = set()
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                raise ValueError("choice must be an object")
            key = str(raw_choice.get("key") or "").strip()
            label = str(raw_choice.get("label") or "").strip()
            if (
                not re.fullmatch(r"c[1-9]\d*", key)
                or len(key) > 40
                or key in choice_keys
            ):
                raise ValueError("choice key is invalid")
            if not label or len(label) > 500:
                raise ValueError("choice label is invalid")
            choice_keys.add(key)
            choice_kb_ids = _bounded_uuid_values(
                raw_choice.get("kb_ids"),
                limit=100,
            )
            choice_doc_ids = _bounded_uuid_values(
                raw_choice.get("doc_ids"),
                limit=MAX_EVIDENCE_SCOPE_DOCUMENTS,
            )
            choice_anchor_doc_ids = _bounded_uuid_values(
                raw_choice.get("anchor_doc_ids"),
                limit=MAX_EVIDENCE_SCOPE_DOCUMENTS,
            )
            choice_companion_doc_ids = _bounded_uuid_values(
                raw_choice.get("companion_doc_ids"),
                limit=MAX_EVIDENCE_SCOPE_DOCUMENTS,
            )
            if (
                not choice_kb_ids
                or not choice_doc_ids
                or not choice_anchor_doc_ids
            ):
                raise ValueError("choice scope ids must be non-empty")
            choice_doc_keys = {str(item) for item in choice_doc_ids}
            choice_anchor_keys = {
                str(item) for item in choice_anchor_doc_ids
            }
            choice_companion_keys = {
                str(item) for item in choice_companion_doc_ids
            }
            raw_scope_slices = raw_choice.get("scope_slices")
            scope_slices: list[_EvidenceScopeSliceFilter] = []
            if raw_scope_slices not in (None, []):
                if (
                    not isinstance(raw_scope_slices, list)
                    or not raw_scope_slices
                    or len(raw_scope_slices) > 100
                ):
                    raise ValueError("choice scope slices are invalid")
                seen_slice_tokens: set[tuple[str, ...]] = set()
                for raw_slice in raw_scope_slices:
                    if not isinstance(raw_slice, dict):
                        raise ValueError("choice scope slice must be an object")
                    slice_kb_ids = _bounded_uuid_values(
                        [raw_slice.get("kb_id")],
                        limit=1,
                    )
                    slice_doc_ids = _bounded_uuid_values(
                        [raw_slice.get("doc_id")],
                        limit=1,
                    )
                    raw_section_key = raw_slice.get("section_key")
                    if raw_section_key is not None and not isinstance(
                        raw_section_key,
                        str,
                    ):
                        raise ValueError("choice section key is invalid")
                    section_key = (
                        raw_section_key.strip()
                        if isinstance(raw_section_key, str)
                        else None
                    )
                    if section_key is not None and (
                        not section_key or len(section_key) > 500
                    ):
                        raise ValueError("choice section key is invalid")
                    chunk_ids = _bounded_uuid_values(
                        raw_slice.get("chunk_ids", []),
                        limit=100,
                    )
                    is_anchor = raw_slice.get("is_anchor")
                    if (
                        not slice_kb_ids
                        or not slice_doc_ids
                        or not isinstance(is_anchor, bool)
                        or (section_key is None and not chunk_ids)
                    ):
                        raise ValueError("choice scope slice is incomplete")
                    scope_slice = _EvidenceScopeSliceFilter(
                        kb_id=slice_kb_ids[0],
                        doc_id=slice_doc_ids[0],
                        section_key=section_key,
                        chunk_ids=chunk_ids,
                        is_anchor=is_anchor,
                    )
                    tokens = _scope_slice_tokens(scope_slice)
                    if not tokens or tokens & seen_slice_tokens:
                        raise ValueError("choice scope slice is duplicated")
                    seen_slice_tokens.update(tokens)
                    scope_slices.append(scope_slice)
            if (
                choice_anchor_keys & choice_companion_keys
                or choice_doc_keys
                != choice_anchor_keys | choice_companion_keys
            ):
                raise ValueError("choice anchor/companion partition is invalid")
            if scope_slices:
                slice_kb_keys = {
                    str(value.kb_id) for value in scope_slices
                }
                slice_doc_keys = {
                    str(value.doc_id) for value in scope_slices
                }
                slice_anchor_doc_keys = {
                    str(value.doc_id)
                    for value in scope_slices
                    if value.is_anchor
                }
                if (
                    not slice_anchor_doc_keys
                    or not slice_kb_keys.issubset(
                        {str(value) for value in choice_kb_ids}
                    )
                    or slice_doc_keys != choice_doc_keys
                    or slice_anchor_doc_keys != choice_anchor_keys
                ):
                    raise ValueError("choice scope slice partition is invalid")
            choices.append(
                _EvidenceScopeChoiceFilter(
                    key=key,
                    label=label,
                    products=_bounded_text_values(
                        raw_choice.get("products"),
                    ),
                    canonical_products=_bounded_text_values(
                        raw_choice.get("canonical_products"),
                    ),
                    versions=_bounded_text_values(
                        raw_choice.get("versions"),
                    ),
                    projects=_bounded_text_values(
                        raw_choice.get("projects"),
                    ),
                    filenames=_bounded_text_values(
                        raw_choice.get("filenames"),
                    ),
                    kb_ids=choice_kb_ids,
                    doc_ids=choice_doc_ids,
                    anchor_doc_ids=choice_anchor_doc_ids,
                    companion_doc_ids=choice_companion_doc_ids,
                    scope_slices=tuple(scope_slices),
                )
            )
    except (TypeError, ValueError):
        return _invalid_evidence_scope_filter(mode, "malformed_filter")

    if (mode == "single" and len(choices) != 1) or (
        mode == "compare_all" and len(choices) < 2
    ):
        return _invalid_evidence_scope_filter(mode, "choice_count_mismatch")

    # Anchor documents prove one mutually exclusive choice.  A caller must not
    # be able to relabel a document shared by another included choice as an
    # anchor and thereby let one generic hit satisfy several scopes.
    for choice in choices:
        anchor_keys = {str(item) for item in choice.anchor_doc_ids}
        other_choice_doc_keys = {
            str(item)
            for other in choices
            if other.key != choice.key
            for item in other.doc_ids
        }
        if not choice.scope_slices and anchor_keys & other_choice_doc_keys:
            return _invalid_evidence_scope_filter(
                mode,
                "anchor_not_choice_exclusive",
            )
    for choice in choices:
        anchor_tokens = {
            token
            for scope_slice in choice.scope_slices
            if scope_slice.is_anchor
            for token in _scope_slice_tokens(scope_slice)
        }
        other_tokens = {
            token
            for other in choices
            if other.key != choice.key
            for scope_slice in other.scope_slices
            for token in _scope_slice_tokens(scope_slice)
        }
        if anchor_tokens & other_tokens:
            return _invalid_evidence_scope_filter(
                mode,
                "anchor_slice_not_choice_exclusive",
            )

    choice_kb_ids = {
        str(item)
        for choice in choices
        for item in choice.kb_ids
    }
    choice_doc_ids = {
        str(item)
        for choice in choices
        for item in choice.doc_ids
    }
    requested_kb_keys = {str(item) for item in requested_kb_ids}
    requested_doc_keys = {str(item) for item in requested_doc_ids}
    if (
        not requested_kb_keys
        or not requested_doc_keys
        or requested_kb_keys != choice_kb_ids
        or requested_doc_keys != choice_doc_ids
    ):
        return _invalid_evidence_scope_filter(mode, "scope_choice_mismatch")

    authorized_by_key: dict[str, uuid.UUID] = {}
    for raw in authorized_kb_ids:
        try:
            item = uuid.UUID(str(raw))
        except (TypeError, ValueError, AttributeError):
            continue
        authorized_by_key[str(item)] = item
    if not requested_kb_keys.issubset(authorized_by_key):
        return _invalid_evidence_scope_filter(mode, "kb_not_authorized")
    scoped_kb_ids = tuple(
        authorized_by_key[str(item)]
        for item in requested_kb_ids
    )

    return _EvidenceScopeFilter(
        mode=mode,
        kb_ids=scoped_kb_ids,
        doc_ids=requested_doc_ids,
        choices=tuple(choices),
    )

def _resolved_comparison_scope_filter(
    plan: ExplicitScopeComparisonPlan,
    *,
    authorized_kb_ids: Sequence[uuid.UUID | str],
) -> _EvidenceScopeFilter | None:
    """Turn a source-derived enumerated comparison into a fail-closed filter.

    This is intentionally limited to explicitly enumerated scopes.  Generic
    requests such as ``所有版本`` must first pass rerank relevance assessment;
    otherwise an unrelated raw-retrieval document could be promoted into a
    requested comparison scope merely because it declares another version.
    """

    if (
        not plan.matched
        or plan.reason != "explicit_enumerated_scopes"
        or len(plan.choices) < 2
    ):
        return None
    payload = {
        "mode": "compare_all",
        "kb_ids": list(dict.fromkeys(
            kb_id
            for choice in plan.choices
            for kb_id in choice.kb_ids
        )),
        "doc_ids": list(dict.fromkeys(
            doc_id
            for choice in plan.choices
            for doc_id in choice.doc_ids
        )),
        "choices": [
            {
                **choice.to_dict(),
                "products": list(choice.products),
                "canonical_products": list(choice.canonical_products),
                "versions": list(choice.versions),
                "projects": list(choice.projects),
                "filenames": list(choice.filenames),
                "kb_ids": list(choice.kb_ids),
                "doc_ids": list(choice.doc_ids),
                "anchor_doc_ids": list(choice.anchor_doc_ids),
                "companion_doc_ids": list(choice.companion_doc_ids),
                "scope_slices": [
                    {
                        **scope_slice.to_dict(),
                        "chunk_ids": list(scope_slice.chunk_ids),
                    }
                    for scope_slice in choice.scope_slices
                ],
            }
            for choice in plan.choices
        ],
    }
    normalized = _normalize_evidence_scope_filter(
        payload,
        authorized_kb_ids=authorized_kb_ids,
    )
    if normalized is None or not normalized.valid:
        return None
    if {str(value) for value in normalized.doc_ids} != set(plan.allowed_doc_ids):
        return None
    return normalized

def _scope_choice_labels_by_document(
    choices: Sequence[EvidenceScopeChoice],
) -> dict[str, str]:
    labels: dict[str, list[str]] = {}
    for choice in choices:
        for doc_id in choice.doc_ids:
            values = labels.setdefault(str(doc_id), [])
            if choice.label not in values:
                values.append(choice.label)
    return {
        doc_id: " / ".join(values)
        for doc_id, values in labels.items()
        if values
    }

def _scope_filter_queries(
    base_query: str,
    scope_filter: _EvidenceScopeFilter,
) -> tuple[str, list[str]]:
    """Build a standalone question plus bounded document-search queries."""

    original = str(base_query or "").strip()
    if not scope_filter.valid:
        return original, [original]
    if scope_filter.compare_all:
        # Rolling/custom callers may already have appended the display labels
        # for semantic routing.  Remove those exact untrusted display strings
        # from the Pipeline question so the first version cannot become a hard
        # query constraint; the second scoped-search query below still carries
        # every label for identity/header recall.
        comparison_original = original
        for choice in scope_filter.choices:
            comparison_original = comparison_original.replace(choice.label, "")
        comparison_original = re.sub(
            r"[；;、\s]+$",
            "",
            comparison_original,
        ).strip()
        standalone = (
            f"{comparison_original}\n用户已明确要求对所选全部适用范围分别对比回答。"
        )
        scope_terms = "；".join(choice.label for choice in scope_filter.choices)
        original_search_query = comparison_original
    else:
        scope_terms = scope_filter.choices[0].label
        standalone = f"{original}\n用户已明确选择适用范围：{scope_terms}"
        original_search_query = original
    return standalone, [
        original_search_query,
        f"{original_search_query}\n适用范围：{scope_terms}",
    ]

def _restrict_candidates_to_scope(
    candidates: Sequence[dict],
    scope_filter: _EvidenceScopeFilter | None,
) -> tuple[list[dict], int]:
    if scope_filter is None:
        return [dict(item) for item in candidates], 0
    if not scope_filter.valid:
        return [], len(candidates)
    selected = [
        dict(item)
        for item in candidates
        if any(
            _candidate_matches_scope_choice(item, choice)
            for choice in scope_filter.choices
        )
    ]
    return selected, len(candidates) - len(selected)

def _scope_candidate_identity(item: dict) -> str:
    identity = str(item.get("id") or "").strip()
    if identity:
        return identity
    return ":".join(
        str(item.get(field) or "")
        for field in ("kb_id", "doc_id", "chunk_index")
    )

def _candidate_matches_scope_choice(
    item: dict,
    choice: _EvidenceScopeChoiceFilter,
) -> bool:
    kb_id = str(item.get("kb_id") or "")
    doc_id = str(item.get("doc_id") or "")
    if (
        kb_id not in {str(value) for value in choice.kb_ids}
        or doc_id not in {str(value) for value in choice.doc_ids}
    ):
        return False
    if not choice.scope_slices:
        return True
    chunk_id = str(item.get("id") or item.get("chunk_id") or "").strip()
    section_key = candidate_section_key(item)
    return any(
        kb_id == str(scope_slice.kb_id)
        and doc_id == str(scope_slice.doc_id)
        and (
            (
                scope_slice.section_key is not None
                and section_key == scope_slice.section_key
            )
            or (
                chunk_id
                and chunk_id
                in {str(value) for value in scope_slice.chunk_ids}
            )
        )
        for scope_slice in choice.scope_slices
    )

def _candidate_proves_scope_choice(
    item: dict,
    choice: _EvidenceScopeChoiceFilter,
) -> bool:
    if not _candidate_matches_scope_choice(item, choice):
        return False
    if not choice.scope_slices:
        return str(item.get("doc_id") or "") in {
            str(value) for value in choice.anchor_doc_ids
        }
    kb_id = str(item.get("kb_id") or "")
    doc_id = str(item.get("doc_id") or "")
    chunk_id = str(item.get("id") or item.get("chunk_id") or "").strip()
    section_key = candidate_section_key(item)
    return any(
        scope_slice.is_anchor
        and kb_id == str(scope_slice.kb_id)
        and doc_id == str(scope_slice.doc_id)
        and (
            (
                scope_slice.section_key is not None
                and section_key == scope_slice.section_key
            )
            or (
                chunk_id
                and chunk_id
                in {str(value) for value in scope_slice.chunk_ids}
            )
        )
        for scope_slice in choice.scope_slices
    )

def _scope_anchor_coverage(
    candidates: Sequence[dict],
    scope_filter: _EvidenceScopeFilter,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether every selected choice has an anchor-document hit."""

    hit_ids: list[str] = []
    hit_id_set: set[str] = set()
    covered_choice_keys: set[str] = set()
    for choice in scope_filter.choices:
        for item in candidates:
            doc_id = str(item.get("doc_id") or "")
            if not _candidate_proves_scope_choice(item, choice):
                continue
            covered_choice_keys.add(choice.key)
            if doc_id not in hit_id_set:
                hit_id_set.add(doc_id)
                hit_ids.append(doc_id)
    return (
        len(covered_choice_keys) == len(scope_filter.choices),
        tuple(hit_ids),
    )
def rerank_candidate_limit(top_k: int, *, simple: bool = False) -> int:
    """候选池应大于最终 Top K，但必须受模型上下文和成本上限约束。"""

    normalized = max(1, min(int(top_k), 20))
    if simple:
        return min(
            SIMPLE_RERANK_CANDIDATE_MAX,
            max(
                normalized,
                SIMPLE_RERANK_CANDIDATE_MIN,
                normalized * SIMPLE_RERANK_CANDIDATE_MULTIPLIER,
            ),
        )
    return min(
        RERANK_CANDIDATE_MAX,
        max(RERANK_CANDIDATE_MIN, normalized * RERANK_CANDIDATE_MULTIPLIER),
    )

def annotate_deterministic_constraints(
    results: list[dict],
    constraints: QueryConstraints,
) -> list[dict]:
    """在没有可信 LLM 重排时也执行代码级产品/版本约束。

    这一步不把候选伪装成 direct（因为没有 topic/answer_support 分数），但会
    把明确冲突标为 related，确保旧版本资料不会进入“直接回答”上下文。
    """

    annotated: list[dict] = []
    for result in inherit_document_constraint_metadata(results):
        item = dict(result)
        evaluation = evaluate_candidate_constraints(constraints, item)
        item["constraint_status"] = evaluation.status
        item["constraint_reason"] = evaluation.reason
        item["query_has_constraint"] = constraints.has_scope_constraint
        item["query_has_product_constraint"] = constraints.has_product_constraint
        item["query_has_hard_constraint"] = constraints.has_hard_constraint
        item["query_has_version_constraint"] = constraints.has_version_constraint
        item["rerank_status"] = item.get("rerank_status") or "unverified"
        item["evidence_role"] = "related" if evaluation.status == "mismatch" else None
        annotated.append(item)
    return annotated
