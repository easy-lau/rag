"""Deterministic document admission for RAG v2 retrieval candidates.

The gate deliberately consumes retrieval signals only.  It does not infer a
business topic, use RRF ``score`` values as semantic confidence, or call an
external model.  Candidates are aggregated by document so one strong chunk can
anchor its document without allowing a weaker document to borrow that signal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.query_constraints import (
    canonical_product_name,
    extract_document_constraint_identity,
    extract_query_constraints,
)


MIN_VECTOR_SCORE = 0.78
MAX_DOC_VECTOR_GAP = 0.035
# PostgreSQL ``ts_rank`` is not a calibrated probability, but a tiny positive
# value is still distinguishable from a real lexical match.  Rank numbers and
# ``active_channels`` are ordering metadata and cannot establish relevance on
# their own.  The trigram floor mirrors the retriever's SQL candidate floor.
MIN_KEYWORD_SCORE = 1e-4
MIN_TRIGRAM_SCORE = 0.12
_FLOAT_COMPARISON_EPSILON = 1e-12
MIN_TOPIC_COVERAGE = 0.30
STRONG_VECTOR_TOPIC_BYPASS = 0.84
_TOPIC_STOP_TERMS = frozenset({
    "什么",
    "如何",
    "怎么",
    "怎样",
    "哪些",
    "哪个",
    "是否",
    "能否",
    "可否",
    "请问",
    "查询",
    "查一下",
    "标准",
    "制度",
    "政策",
    "规定",
    "办法",
    "规范",
    "要求",
    "信息",
    "内容",
    "相关",
})


@dataclass(frozen=True)
class DocumentRelevanceDecision:
    """Result of applying the deterministic document relevance gate."""

    admitted_doc_ids: tuple[str, ...]
    rejected_doc_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted_doc_ids": list(self.admitted_doc_ids),
            "rejected_doc_ids": list(self.rejected_doc_ids),
            "reason": self.reason,
        }


@dataclass
class _DocumentSignals:
    lexical_hit: bool = False
    best_vector_score: float | None = None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _number_at_least(value: object, minimum: float) -> bool:
    parsed = _finite_number(value)
    return parsed is not None and parsed >= minimum


def _has_lexical_hit(candidate: Mapping[str, Any]) -> bool:
    """Recognize a scored FTS/trigram observation.

    ``keyword_rank``/``trigram_rank`` and ``active_channels`` only describe
    ordering.  Treating them as evidence admitted arbitrary near-zero or
    adapter-synthesized hits and was the main path by which unrelated Chinese
    documents entered a grounded answer.
    """

    return bool(
        _number_at_least(candidate.get("keyword_score"), MIN_KEYWORD_SCORE)
        or _number_at_least(candidate.get("trigram_score"), MIN_TRIGRAM_SCORE)
    )


def _topic_terms(value: object) -> set[str]:
    """Extract bounded CJK n-grams/ASCII words for a query coverage check."""

    text = str(value or "").casefold()
    terms: set[str] = set()
    for match in re.finditer(r"[a-z0-9][a-z0-9_.+/-]{1,}|[\u3400-\u9fff]+", text):
        token = match.group(0)
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            for size in (2, 3):
                if len(token) >= size:
                    terms.update(
                        token[index:index + size]
                        for index in range(len(token) - size + 1)
                    )
        else:
            terms.add(token)
    return {
        term
        for term in terms
        if term not in _TOPIC_STOP_TERMS
        and not any(char in term for char in "的是否以及与和为请问查询")
    }


def _topic_coverage(query: object, candidate: Mapping[str, Any]) -> float | None:
    """Estimate lexical subject coverage without treating RRF as confidence."""

    query_terms = _topic_terms(query)
    if len(query_terms) <= 2:
        return None
    content = "\n".join(
        str(candidate.get(field) or "")
        for field in ("filename", "content", "heading", "title", "source")
    ).strip()
    if not content:
        return None
    candidate_terms = _topic_terms(content)
    if not candidate_terms:
        return 0.0
    return len(query_terms & candidate_terms) / max(len(query_terms), 1)


def _validated_threshold(value: object, *, field_name: str) -> float:
    parsed = _finite_number(value)
    if parsed is None or not 0 <= parsed <= 1:
        raise ValueError(f"{field_name} must be a finite number between 0 and 1")
    return parsed


def assess_document_relevance(
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_vector_score: float = MIN_VECTOR_SCORE,
    max_doc_vector_gap: float = MAX_DOC_VECTOR_GAP,
    query: str | None = None,
) -> DocumentRelevanceDecision:
    """Admit documents supported by lexical evidence or a strong vector score.

    A document is admitted when at least one candidate has a keyword/trigram
    retrieval hit, or when the document's best raw ``vector_score`` satisfies
    both the absolute minimum and the maximum gap from the best document in the
    current candidate pool.  Rank-fusion ``score`` and ``retrieval_score`` are
    intentionally ignored because they express ordering rather than calibrated
    semantic relevance.
    """

    minimum = _validated_threshold(
        min_vector_score,
        field_name="min_vector_score",
    )
    maximum_gap = _validated_threshold(
        max_doc_vector_gap,
        field_name="max_doc_vector_gap",
    )
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise ValueError("candidates must be a sequence of mappings")
    if not candidates:
        return DocumentRelevanceDecision((), (), "no_candidates")

    document_order: list[str] = []
    signals_by_document: dict[str, _DocumentSignals] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        doc_id = str(candidate.get("doc_id") or "").strip()
        if not doc_id:
            continue
        signals = signals_by_document.get(doc_id)
        if signals is None:
            signals = _DocumentSignals()
            signals_by_document[doc_id] = signals
            document_order.append(doc_id)

        signals.lexical_hit = signals.lexical_hit or _has_lexical_hit(candidate)
        vector_score = _finite_number(candidate.get("vector_score"))
        if vector_score is not None and (
            signals.best_vector_score is None
            or vector_score > signals.best_vector_score
        ):
            signals.best_vector_score = vector_score

    if not document_order:
        return DocumentRelevanceDecision((), (), "no_valid_documents")

    vector_scores = [
        signals.best_vector_score
        for signals in signals_by_document.values()
        if signals.best_vector_score is not None
    ]
    global_best_vector = max(vector_scores) if vector_scores else None

    admitted: list[str] = []
    rejected: list[str] = []
    admitted_by_lexical = False
    admitted_by_vector = False
    admitted_by_topic = False
    admitted_by_version = False
    query_constraints = extract_query_constraints(query or "") if query else None
    version_representatives: dict[str, tuple[str, float]] = {}
    if query_constraints is not None and query_constraints.product and not query_constraints.explicit_version:
        requested_product = canonical_product_name(query_constraints.product)
        for doc_id in document_order:
            doc_candidates = [
                candidate for candidate in candidates
                if isinstance(candidate, Mapping)
                and str(candidate.get("doc_id") or "").strip() == doc_id
            ]
            versions = {
                version.casefold()
                for item in doc_candidates
                for version in extract_document_constraint_identity(item).versions
                if version
                and requested_product in set(
                    extract_document_constraint_identity(item).canonical_products
                )
            }
            best = signals_by_document[doc_id].best_vector_score
            if best is None or best < minimum:
                continue
            for version in versions:
                current = version_representatives.get(version)
                if current is None or best > current[1]:
                    version_representatives[version] = (doc_id, best)
    for doc_id in document_order:
        signals = signals_by_document[doc_id]
        vector_admitted = False
        if (
            global_best_vector is not None
            and signals.best_vector_score is not None
            and signals.best_vector_score + _FLOAT_COMPARISON_EPSILON >= minimum
        ):
            gap = global_best_vector - signals.best_vector_score
            vector_admitted = gap <= maximum_gap + _FLOAT_COMPARISON_EPSILON

        topic_coverage_values = [
            _topic_coverage(query, candidate)
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and str(candidate.get("doc_id") or "").strip() == doc_id
        ]
        topic_coverage_values = [
            value for value in topic_coverage_values if value is not None
        ]
        topic_admitted = (
            query is None
            or not topic_coverage_values
            or max(topic_coverage_values) >= MIN_TOPIC_COVERAGE
            or (
                signals.best_vector_score is not None
                and signals.best_vector_score >= STRONG_VECTOR_TOPIC_BYPASS
            )
        )
        version_representative = any(
            representative_doc_id == doc_id
            for representative_doc_id, _ in version_representatives.values()
        )
        if (signals.lexical_hit or vector_admitted or version_representative) and topic_admitted:
            admitted.append(doc_id)
            admitted_by_lexical = admitted_by_lexical or signals.lexical_hit
            admitted_by_vector = admitted_by_vector or vector_admitted
            admitted_by_topic = admitted_by_topic or bool(topic_coverage_values)
            admitted_by_version = admitted_by_version or version_representative
        else:
            rejected.append(doc_id)

    if not admitted:
        reason = "no_document_met_lexical_or_vector_gate"
    elif admitted_by_lexical and admitted_by_vector:
        reason = "admitted_by_lexical_or_vector_evidence"
    elif admitted_by_version:
        reason = "admitted_by_each_explicit_version_representative"
    elif admitted_by_lexical and admitted_by_topic:
        reason = "admitted_by_lexical_and_topic_evidence"
    elif admitted_by_lexical:
        reason = "admitted_by_lexical_evidence"
    else:
        reason = "admitted_by_vector_score_and_global_gap"
    return DocumentRelevanceDecision(
        admitted_doc_ids=tuple(admitted),
        rejected_doc_ids=tuple(rejected),
        reason=reason,
    )


__all__ = [
    "DocumentRelevanceDecision",
    "MAX_DOC_VECTOR_GAP",
    "MIN_KEYWORD_SCORE",
    "MIN_TRIGRAM_SCORE",
    "MIN_VECTOR_SCORE",
    "MIN_TOPIC_COVERAGE",
    "assess_document_relevance",
]
