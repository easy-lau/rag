"""Deterministic admission between high-recall retrieval and answer evidence.

Retrieval is allowed to return noisy nearest neighbours.  Admission consumes
only calibrated retrieval signals and decides which authorized candidates may
reach semantic verification or answer generation.  RRF ranks are ordering
metadata and never count as relevance evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.evidence_contract import AdmissionStatus, CandidateRejection
from core.knowledge_records import STRUCTURED_RECORD_MIN_SCORE
from core.query_constraints import (
    canonical_product_name,
    extract_document_constraint_identity,
    extract_query_constraints,
)


MIN_VECTOR_SCORE = 0.78
MAX_DOC_VECTOR_GAP = 0.035
MIN_KEYWORD_SCORE = 1e-4
MIN_TRIGRAM_SCORE = 0.12
_FLOAT_COMPARISON_EPSILON = 1e-12


@dataclass(frozen=True)
class DocumentRelevanceDecision:
    admitted_doc_ids: tuple[str, ...]
    rejected_doc_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted_doc_ids": list(self.admitted_doc_ids),
            "rejected_doc_ids": list(self.rejected_doc_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CandidateAdmission:
    """One immutable high-recall-to-answer admission result."""

    status: AdmissionStatus
    candidates: tuple[dict[str, Any], ...]
    rejections: tuple[CandidateRejection, ...]
    reason: str
    document_decision: DocumentRelevanceDecision
    raw_candidate_count: int

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "admission_status": self.status,
            "admission_reason": self.reason,
            "raw_candidate_count": self.raw_candidate_count,
            "admitted_candidate_count": len(self.candidates),
            "rejected_candidate_count": len(self.rejections),
            "admitted_document_count": len(
                self.document_decision.admitted_doc_ids
            ),
            "rejected_document_count": len(
                self.document_decision.rejected_doc_ids
            ),
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
    return bool(
        _number_at_least(candidate.get("keyword_score"), MIN_KEYWORD_SCORE)
        or _number_at_least(candidate.get("trigram_score"), MIN_TRIGRAM_SCORE)
    )


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
    """Admit documents with real lexical evidence or calibrated vectors."""

    minimum = _validated_threshold(min_vector_score, field_name="min_vector_score")
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
    query_constraints = extract_query_constraints(query or "") if query else None
    version_representatives: dict[str, tuple[str, float]] = {}
    if (
        query_constraints is not None
        and query_constraints.product
        and not query_constraints.explicit_version
    ):
        requested_product = canonical_product_name(query_constraints.product)
        for doc_id in document_order:
            doc_candidates = [
                item
                for item in candidates
                if isinstance(item, Mapping)
                and str(item.get("doc_id") or "").strip() == doc_id
            ]
            versions = {
                version.casefold()
                for item in doc_candidates
                for version in extract_document_constraint_identity(item).versions
                if version
                and requested_product
                in set(
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

    admitted: list[str] = []
    rejected: list[str] = []
    admitted_by_lexical = False
    admitted_by_vector = False
    admitted_by_version = False
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
        version_representative = any(
            representative_doc_id == doc_id
            for representative_doc_id, _ in version_representatives.values()
        )
        if signals.lexical_hit or vector_admitted or version_representative:
            admitted.append(doc_id)
            admitted_by_lexical = admitted_by_lexical or signals.lexical_hit
            admitted_by_vector = admitted_by_vector or vector_admitted
            admitted_by_version = admitted_by_version or version_representative
        else:
            rejected.append(doc_id)

    if not admitted:
        reason = "no_document_met_lexical_or_vector_gate"
    elif admitted_by_lexical and admitted_by_vector:
        reason = "admitted_by_lexical_or_vector_evidence"
    elif admitted_by_version:
        reason = "admitted_by_each_explicit_version_representative"
    elif admitted_by_lexical:
        reason = "admitted_by_lexical_evidence"
    else:
        reason = "admitted_by_vector_score_and_global_gap"
    return DocumentRelevanceDecision(
        admitted_doc_ids=tuple(admitted),
        rejected_doc_ids=tuple(rejected),
        reason=reason,
    )


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    return str(
        candidate.get("record_id")
        or candidate.get("id")
        or candidate.get("chunk_id")
        or ""
    ).strip()


def admit_evidence_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    query: str,
) -> CandidateAdmission:
    """Admit answer candidates while retaining content-free rejections."""

    raw = [dict(item) for item in candidates if isinstance(item, Mapping)]
    if not raw:
        return CandidateAdmission(
            status="not_applied",
            candidates=(),
            rejections=(),
            reason="no_candidates",
            document_decision=DocumentRelevanceDecision(
                (),
                (),
                "no_candidates",
            ),
            raw_candidate_count=0,
        )
    chunk_candidates = [
        item
        for item in raw
        if str(item.get("source_kind") or "document_chunk")
        != "knowledge_record"
    ]
    document_decision = assess_document_relevance(
        chunk_candidates,
        query=query,
    )
    admitted_doc_ids = set(document_decision.admitted_doc_ids)
    admitted_record_ids = {
        _candidate_identity(item)
        for item in raw
        if str(item.get("source_kind") or "") == "knowledge_record"
        and _number_at_least(
            item.get("structured_score"),
            STRUCTURED_RECORD_MIN_SCORE,
        )
    }
    structured_admitted_doc_ids = {
        str(item.get("doc_id") or "").strip()
        for item in raw
        if _candidate_identity(item) in admitted_record_ids
    }
    admitted_doc_ids.update(structured_admitted_doc_ids)

    admitted: list[dict[str, Any]] = []
    rejections: list[CandidateRejection] = []
    for item in raw:
        candidate_id = _candidate_identity(item)
        doc_id = str(item.get("doc_id") or "").strip()
        source_kind = str(
            item.get("source_kind") or "document_chunk"
        ).strip()
        is_record = source_kind == "knowledge_record"
        accepted = (
            candidate_id in admitted_record_ids
            if is_record
            else doc_id in admitted_doc_ids
        )
        if accepted:
            item["admission_status"] = "admitted"
            item["admission_reason"] = (
                "structured_record_score"
                if is_record
                else (
                    "structured_record_document_anchor"
                    if doc_id in structured_admitted_doc_ids
                    and doc_id not in document_decision.admitted_doc_ids
                    else document_decision.reason
                )
            )
            admitted.append(item)
            continue
        rejections.append(CandidateRejection(
            candidate_id=candidate_id,
            doc_id=doc_id,
            source_kind=source_kind,
            reason=(
                "structured_record_below_threshold"
                if is_record
                else "document_relevance_gate"
            ),
        ))

    status: AdmissionStatus = "admitted" if admitted else "rejected"
    if admitted_record_ids and admitted:
        reason = "structured_or_document_evidence_admitted"
    elif admitted:
        reason = document_decision.reason
    else:
        reason = "no_candidate_met_admission_gate"
    return CandidateAdmission(
        status=status,
        candidates=tuple(admitted),
        rejections=tuple(rejections),
        reason=reason,
        document_decision=document_decision,
        raw_candidate_count=len(raw),
    )


__all__ = [
    "CandidateAdmission",
    "DocumentRelevanceDecision",
    "MAX_DOC_VECTOR_GAP",
    "MIN_KEYWORD_SCORE",
    "MIN_TRIGRAM_SCORE",
    "MIN_VECTOR_SCORE",
    "admit_evidence_candidates",
    "assess_document_relevance",
]
