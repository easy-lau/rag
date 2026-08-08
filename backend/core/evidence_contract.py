"""Shared evidence vocabulary and retrieval-to-answer contract.

Retrieval answers one question only: which authorized source candidates were
found?  Semantic verification may promote or annotate those candidates, but it
cannot rewrite a successful retrieval into ``no_hit``.  Chat, HTTP APIs and
offline regression can therefore consume the same evidence pack without
depending on a model-specific reranker response.  The same module retains the
wire vocabulary used inside semantic adjudication so coverage normalization,
retrieval and answer delivery cannot drift into separate protocols.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


EvidenceOutcome = Literal[
    "answered",
    "needs_clarification",
    "no_hit",
    "insufficient_evidence",
    "service_unavailable",
]
RetrievalStatus = Literal["hit", "no_hit", "service_unavailable"]
AdmissionStatus = Literal["admitted", "rejected", "not_applied"]
VerificationStatus = Literal[
    "verified",
    "ambiguous",
    "unverified",
    "not_requested",
]

CoverageStatus = Literal["complete", "partial", "insufficient"]
AdjudicationStatus = Literal["succeeded", "inconclusive", "failed"]

COVERAGE_STATUSES = frozenset({"complete", "partial", "insufficient"})
COVERAGE_STATUS_ALIASES: dict[str, CoverageStatus] = {
    "full": "complete",
    "fully_covered": "complete",
    "complete_coverage": "complete",
    "partially_covered": "partial",
    "partial_coverage": "partial",
    "not_covered": "insufficient",
    "insufficient_coverage": "insufficient",
}
INCONCLUSIVE_FAILURE_KINDS = frozenset({
    "empty_content",
    "timeout",
    "contract_validation",
})


def normalize_coverage_status(
    value: object,
) -> tuple[CoverageStatus, str | None, str]:
    """Normalize a model diagnostic without granting it evidence authority."""

    if not isinstance(value, str):
        return "insufficient", None, "missing_diagnostic"
    original = value.strip()[:80]
    normalized = original.casefold().replace("-", "_").replace(" ", "_")
    if normalized in COVERAGE_STATUSES:
        return normalized, original, "exact"  # type: ignore[return-value]
    alias = COVERAGE_STATUS_ALIASES.get(normalized)
    if alias is not None:
        return alias, original, "normalized_alias"
    return "insufficient", original or None, "unknown_diagnostic"


def coverage_status_protocol_text() -> str:
    return (
        "coverage_status 是诊断字段，只能使用 complete、partial 或 "
        "insufficient；full、fully_covered 等等价写法会由服务端归一化，"
        "最终覆盖状态仍由服务端根据候选绑定重新计算。"
    )


@dataclass(frozen=True)
class AdjudicationOutcome:
    """One semantic-adjudication call, independent from retrieval authority."""

    status: AdjudicationStatus
    reason: str | None = None
    elapsed_ms: int = 0
    payload: Any = None


@dataclass(frozen=True)
class RetrievalChannelReport:
    name: str
    status: Literal["succeeded", "failed"]
    candidate_count: int = 0
    elapsed_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "candidate_count": self.candidate_count,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


@dataclass(frozen=True)
class CandidateRejection:
    """Content-free reason why one authorized candidate cannot answer."""

    candidate_id: str
    doc_id: str
    source_kind: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "doc_id": self.doc_id,
            "source_kind": self.source_kind,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvidencePack:
    """One immutable result from the independent retrieval service."""

    original_query: str
    resolved_query: str
    retrieval_status: RetrievalStatus
    verification_status: VerificationStatus
    outcome: EvidenceOutcome
    admission_status: AdmissionStatus = "not_applied"
    candidates: tuple[dict[str, Any], ...] = ()
    admitted_candidates: tuple[dict[str, Any], ...] = ()
    admission_rejections: tuple[CandidateRejection, ...] = ()
    selected_evidence: tuple[dict[str, Any], ...] = ()
    ambiguity_candidates: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    channel_reports: tuple[RetrievalChannelReport, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "evidence_pack.v1"

    @property
    def answerable(self) -> bool:
        return self.outcome == "answered" and bool(self.selected_evidence)

    def to_trace_dict(self) -> dict[str, Any]:
        """Return a content-free operational summary for structured Trace."""

        return {
            **dict(self.trace),
            "schema_version": self.schema_version,
            "retrieval_status": self.retrieval_status,
            "admission_status": self.admission_status,
            "verification_status": self.verification_status,
            "outcome": self.outcome,
            "reason": self.reason,
            "candidate_count": len(self.candidates),
            "admitted_candidate_count": len(self.admitted_candidates),
            "rejected_candidate_count": len(self.admission_rejections),
            "selected_count": len(self.selected_evidence),
            "ambiguity_count": len(self.ambiguity_candidates),
            "channels": [item.to_dict() for item in self.channel_reports],
        }


def normalize_lookup_text(value: object) -> str:
    """Normalize compatibility CJK, whitespace and punctuation."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(
        r"[\s\-—_·,，。!！?？:：;；、/\\()（）\[\]【】《》〈〉'\"`]+",
        "",
        normalized,
    )


def has_exact_source_match(query: str, content: str) -> bool:
    normalized_query = normalize_lookup_text(query)
    normalized_content = normalize_lookup_text(content)
    return bool(normalized_query and normalized_query in normalized_content)


def extract_matching_text(
    query: str,
    content: str,
    *,
    maximum_chars: int = 1200,
) -> str:
    """Render a bounded exact line or, for semantic matches, bounded evidence."""

    source = str(content or "").strip()
    if not source:
        return ""
    for line in (line.strip() for line in source.splitlines() if line.strip()):
        if has_exact_source_match(query, line):
            return line[:maximum_chars]
    return source[:maximum_chars]


__all__ = [
    "AdmissionStatus",
    "AdjudicationOutcome",
    "AdjudicationStatus",
    "CandidateRejection",
    "COVERAGE_STATUS_ALIASES",
    "COVERAGE_STATUSES",
    "CoverageStatus",
    "EvidenceOutcome",
    "EvidencePack",
    "INCONCLUSIVE_FAILURE_KINDS",
    "RetrievalChannelReport",
    "RetrievalStatus",
    "VerificationStatus",
    "coverage_status_protocol_text",
    "extract_matching_text",
    "has_exact_source_match",
    "normalize_coverage_status",
    "normalize_lookup_text",
]
