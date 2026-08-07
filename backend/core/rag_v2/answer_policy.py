"""Policy contracts between retrieval, evidence assessment and presentation.

Retrieval visibility and answerability are deliberately separate axes.  An
authorized document can be a truthful retrieval candidate even when the final
evidence graph cannot yet certify a complete answer.  This module keeps that
distinction explicit and turns it into one bounded action without re-parsing
the user's question or granting any authority to a model.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from core.clarification import ClarificationContract
from core.query_constraints import evaluate_candidate_constraints
from core.rag_v2.contracts import (
    AnswerRequirementV2,
    EvidenceBundle,
    EvidenceItem,
    QueryPlanV2,
)
from core.rag_v2.evidence import FinalizedVisibleEvidence


RetrievalVisibility = Literal[
    "no_match",
    "authorized_candidates_found",
    "unauthorized_only",
]
AnswerabilityStatus = Literal[
    "answerable",
    "scope_unresolved",
    "evidence_incomplete",
    "provider_failed",
    "refused",
    "unavailable",
]
AnswerPolicyAction = Literal["answer", "clarify", "refuse", "unavailable"]
IntentStatus = Literal[
    "unknown",
    "lookup",
    "explain",
    "compare",
    "modify_guide",
    "troubleshoot",
]
QualityLevel = Literal["high", "medium", "low", "unknown"]


@dataclass(frozen=True)
class CandidateDocument:
    """One authorized public document candidate with all chunk ids retained."""

    kb_id: str
    doc_id: str
    filename: str
    chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.kb_id or not self.doc_id or not self.chunk_ids:
            raise ValueError("candidate document requires KB, document and chunks")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("candidate document chunk ids must be unique")

    def clarification_choice(self, *, index: int) -> dict[str, object] | None:
        """Return a private authorized choice, or none for non-resource ids.

        Production resource identities are UUIDs.  Requiring that shape here
        prevents diagnostic or legacy synthetic ids from becoming durable
        execution authority.
        """

        try:
            kb_id = str(uuid.UUID(self.kb_id))
            doc_id = str(uuid.UUID(self.doc_id))
        except (TypeError, ValueError, AttributeError):
            return None
        filename = self.filename.strip()[:500] or f"候选文档 {index}"
        label = f"使用《{filename}》继续"
        return {
            "key": f"c{index}",
            "label": label,
            "value": filename,
            "filenames": [filename],
            "kb_ids": [kb_id],
            "doc_ids": [doc_id],
            "anchor_doc_ids": [doc_id],
            "companion_doc_ids": [],
            "products": [],
            "canonical_products": [],
            "versions": [],
            "projects": [],
            "scope_slices": [],
        }


@dataclass(frozen=True)
class AuthorizedCandidateSet:
    """Authorized retrieval candidates grouped for UI without dropping chunks."""

    documents: tuple[CandidateDocument, ...] = ()
    unauthorized_match_exists: bool = False

    @property
    def retrieval_status(self) -> RetrievalVisibility:
        if self.documents:
            return "authorized_candidates_found"
        if self.unauthorized_match_exists:
            return "unauthorized_only"
        return "no_match"

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return sum(len(document.chunk_ids) for document in self.documents)

    def clarification_contract(self) -> ClarificationContract | None:
        choices = tuple(
            choice
            for index, document in enumerate(self.documents[:20], start=1)
            if (choice := document.clarification_choice(index=index)) is not None
        )
        if not choices:
            return None
        return ClarificationContract(
            adapter="evidence",
            dimension="candidate_document",
            reason_code="authorized_candidates_need_confirmation",
            selection_mode="choice",
            choices=choices,
        )


def manage_authorized_candidates(
    items: Sequence[EvidenceItem],
    *,
    requirements: Sequence[AnswerRequirementV2] = (),
    unauthorized_match_exists: bool = False,
) -> AuthorizedCandidateSet:
    """Create the one candidate set consumed by policy and presentation.

    The caller supplies the already ACL-filtered evidence pool.  This boundary
    still rejects an explicitly unauthorized/mismatched item and never retains
    its title, content or identifiers in the returned set.
    """

    required_answers = tuple(
        requirement
        for requirement in requirements
        if requirement.is_required_answer
    )
    answer_requirement_ids = {
        requirement.id for requirement in required_answers
    }
    def matches_answer_scope(item: EvidenceItem) -> bool:
        if not required_answers:
            return True
        for requirement in required_answers:
            scope = requirement.applicability_scope
            if scope is None or not scope.has_scope_constraint:
                return True
            if evaluate_candidate_constraints(
                scope,
                item.to_dict(),
            ).status in {"exact", "compatible"}:
                return True
        return False

    grouped: dict[tuple[str, str], dict[str, object]] = {}
    order: list[tuple[str, str]] = []
    unauthorized_seen = bool(unauthorized_match_exists)
    for item in items:
        if not isinstance(item, EvidenceItem):
            raise ValueError("candidate manager accepts EvidenceItem values only")
        if not item.authorized:
            unauthorized_seen = True
            continue
        if item.constraint_status == "mismatch":
            continue
        if not matches_answer_scope(item):
            continue
        support_ids = set(item.supports_requirement_ids)
        if support_ids and not support_ids.intersection(answer_requirement_ids):
            # A bridge/proof fragment is useful internal evidence, but choosing
            # it cannot answer the user's target.  It must not become a
            # user-facing document authority merely because it was retrieved.
            continue
        key = (item.kb_id, item.doc_id)
        if key not in grouped:
            order.append(key)
            grouped[key] = {
                "filename": str(item.metadata.get("filename") or "").strip(),
                "chunk_ids": [],
            }
        row = grouped[key]
        chunk_ids = row["chunk_ids"]
        assert isinstance(chunk_ids, list)
        if item.chunk_id not in chunk_ids:
            chunk_ids.append(item.chunk_id)
        if not row["filename"]:
            row["filename"] = str(item.metadata.get("filename") or "").strip()

    documents = tuple(
        CandidateDocument(
            kb_id=kb_id,
            doc_id=doc_id,
            filename=str(grouped[(kb_id, doc_id)]["filename"] or ""),
            chunk_ids=tuple(grouped[(kb_id, doc_id)]["chunk_ids"]),
        )
        for kb_id, doc_id in order
    )
    return AuthorizedCandidateSet(
        documents=documents,
        unauthorized_match_exists=unauthorized_seen and not documents,
    )


@dataclass(frozen=True)
class EvidenceQuality:
    coverage: QualityLevel
    coverage_ratio: float | None
    reliability: QualityLevel
    freshness: QualityLevel
    consistency: QualityLevel
    completeness: Literal["complete", "partial", "unknown"]
    missing_requirement_ids: tuple[str, ...] = ()
    # ``retrieval_coverage_ratio`` is the fraction of required answer targets
    # that have at least one authorized, request-bound candidate.  It is an
    # honest retrieval-level signal that survives when claim closure fails
    # (for example a deterministic evidence path without a model adjudicator).
    # ``coverage_ratio`` remains the closed-claim coverage; the answer policy
    # uses the stronger of the two so a bounded rules table is not reported as
    # zero coverage merely because closure was not certified.
    retrieval_coverage_ratio: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "coverage_ratio": self.coverage_ratio,
            "retrieval_coverage_ratio": self.retrieval_coverage_ratio,
            "reliability": self.reliability,
            "freshness": self.freshness,
            "consistency": self.consistency,
            "completeness": self.completeness,
            "missing_requirement_ids": list(self.missing_requirement_ids),
        }


def assess_evidence_quality(
    finalized: FinalizedVisibleEvidence,
    *,
    requirement_count: int,
    retrieval_coverage_ratio: float | None = None,
) -> EvidenceQuality:
    bundle = finalized.bundle
    assessment = finalized.assessment or bundle.coverage_assessment
    missing_ids = tuple(
        assessment.missing_requirement_ids
        if assessment is not None
        else bundle.missing_requirement_ids
    )
    if requirement_count > 0:
        covered_count = max(0, requirement_count - len(set(missing_ids)))
        ratio = min(1.0, max(0.0, covered_count / requirement_count))
        coverage: QualityLevel = (
            "high" if ratio == 1.0 else "medium" if ratio > 0.0 else "low"
        )
    else:
        ratio = None
        coverage = "unknown"
    if bundle.state.confidence == "verified":
        reliability: QualityLevel = "high"
    elif bundle.state.confidence == "retrieved":
        reliability = "medium"
    else:
        reliability = "low"
    conflicts = assessment.answer_conflicts if assessment is not None else ()
    consistency: QualityLevel = (
        "low" if conflicts else "high" if assessment is not None else "unknown"
    )
    return EvidenceQuality(
        coverage=coverage,
        coverage_ratio=(round(ratio, 4) if ratio is not None and math.isfinite(ratio) else None),
        reliability=reliability,
        # No source-age contract exists yet.  Reporting unknown is more honest
        # than turning an ingestion timestamp into business freshness.
        freshness="unknown",
        consistency=consistency,
        completeness=bundle.state.completeness,
        missing_requirement_ids=missing_ids,
        retrieval_coverage_ratio=(
            round(retrieval_coverage_ratio, 4)
            if retrieval_coverage_ratio is not None and math.isfinite(retrieval_coverage_ratio)
            else None
        ),
    )


def _intent_status(plan: QueryPlanV2) -> IntentStatus:
    return {
        "comparison": "compare",
        "overview": "explain",
        "process": "modify_guide",
        "unknown": "unknown",
    }.get(plan.answer_shape, "lookup")  # type: ignore[return-value]


@dataclass(frozen=True)
class AnswerPolicyDecision:
    action: AnswerPolicyAction
    retrieval_status: RetrievalVisibility
    answerability_status: AnswerabilityStatus
    intent_status: IntentStatus
    semantic_confidence: float
    evidence_quality: EvidenceQuality
    reason_code: str
    clarification_contract: ClarificationContract | None = None
    # True when the answer is grounded in authorized candidates whose coverage
    # is real but incomplete (the 50%-80% tier).  The UI may then label the
    # answer as uncertain instead of treating it as a verified hit.
    partial_answer: bool = False

    def to_dict(self, *, public: bool = True) -> dict[str, object]:
        # An unauthorized-only retrieval signal is useful for internal policy
        # and audit, but exposing it would reveal that inaccessible material
        # exists.  Public consumers see only the absence of authorized matches.
        retrieval_status: RetrievalVisibility = self.retrieval_status
        if public and retrieval_status == "unauthorized_only":
            retrieval_status = "no_match"
        return {
            "schema_version": "rag_answer_policy.v1",
            "action": self.action,
            "retrieval_status": retrieval_status,
            "answerability_status": self.answerability_status,
            "intent_status": self.intent_status,
            "semantic_confidence": round(self.semantic_confidence, 4),
            "evidence_quality": self.evidence_quality.to_dict(),
            "reason_code": self.reason_code,
            "partial_answer": self.partial_answer,
            "clarification": (
                self.clarification_contract.to_dict(public=public)
                if self.clarification_contract is not None
                else None
            ),
        }


def decide_answer_policy(
    *,
    finalized: FinalizedVisibleEvidence,
    candidates: AuthorizedCandidateSet,
    plan: QueryPlanV2,
    evidence_status: str,
    scope_clarification: ClarificationContract | None = None,
    retrieval_failed: bool = False,
    provider_failed: bool = False,
    candidate_scope_confirmed: bool = False,
    candidate_auto_confirmed: bool = False,
    retrieval_coverage_ratio: float | None = None,
) -> AnswerPolicyDecision:
    """Choose answer/confirmation/refusal from typed request-local facts.

    The decision is coverage-tiered instead of binary: a bounded rules table
    that retrieval bound to the requested answer targets is answerable even
    when the model adjudicator was skipped or failed.  Only genuinely weak
    retrieval coverage (<50% of required answer targets) retains the document
    confirmation flow.
    """

    quality = assess_evidence_quality(
        finalized,
        requirement_count=sum(item.is_required_answer for item in plan.requirements),
        retrieval_coverage_ratio=retrieval_coverage_ratio,
    )
    common = {
        "retrieval_status": candidates.retrieval_status,
        "intent_status": _intent_status(plan),
        "semantic_confidence": plan.confidence,
        "evidence_quality": quality,
    }
    if retrieval_failed or evidence_status == "error":
        return AnswerPolicyDecision(
            action="unavailable",
            answerability_status="unavailable",
            reason_code="retrieval_unavailable",
            **common,
        )
    if candidates.retrieval_status == "unauthorized_only":
        return AnswerPolicyDecision(
            action="refuse",
            answerability_status="refused",
            reason_code="no_authorized_material",
            **common,
        )
    if evidence_status == "scope_mismatch":
        return AnswerPolicyDecision(
            action="refuse",
            answerability_status="refused",
            reason_code="explicit_scope_mismatch",
            **common,
        )
    if scope_clarification is not None:
        return AnswerPolicyDecision(
            action="clarify",
            answerability_status="scope_unresolved",
            reason_code=scope_clarification.reason_code,
            clarification_contract=scope_clarification,
            **common,
        )
    if finalized.generation_allowed:
        unverified = bool(finalized.unverified_generation_allowed)
        if not (unverified and candidate_auto_confirmed and not candidate_scope_confirmed):
            partial_answer = bool(
                unverified
                and candidate_auto_confirmed
                and retrieval_coverage_ratio is not None
                and retrieval_coverage_ratio < 0.8
            )
            return AnswerPolicyDecision(
                action="answer",
                answerability_status="answerable",
                partial_answer=partial_answer,
                reason_code=(
                    "coverage_partial_answer"
                    if partial_answer
                    else "coverage_sufficient_answer"
                    if unverified and candidate_auto_confirmed
                    else "confirmed_candidate_partial_answer"
                    if unverified
                    else "evidence_answerable"
                ),
                **common,
            )
        # Defensive floor: an auto-confirmed scope must still clear the 50%
        # retrieval-coverage tier.  The pipeline already enforces this gate;
        # this branch prevents a future caller from answering with candidates
        # that were never bound to the requested answer targets.
        if retrieval_coverage_ratio is None or retrieval_coverage_ratio < 0.5:
            contract = candidates.clarification_contract()
            if contract is not None:
                return AnswerPolicyDecision(
                    action="clarify",
                    answerability_status="evidence_incomplete",
                    reason_code="authorized_candidates_need_confirmation",
                    clarification_contract=contract,
                    **common,
                )
    if candidates.documents and not candidate_scope_confirmed:
        contract = candidates.clarification_contract()
        if contract is not None:
            return AnswerPolicyDecision(
                action="clarify",
                answerability_status=(
                    "provider_failed" if provider_failed else "evidence_incomplete"
                ),
                reason_code=(
                    "provider_adjudication_failed_with_candidates"
                    if provider_failed
                    else "authorized_candidates_need_confirmation"
                ),
                clarification_contract=contract,
                **common,
            )
    return AnswerPolicyDecision(
        action="refuse",
        answerability_status=("provider_failed" if provider_failed else "evidence_incomplete"),
        reason_code=(
            "provider_adjudication_failed_without_safe_confirmation"
            if provider_failed
            else "no_answerable_evidence"
        ),
        **common,
    )


__all__ = [
    "AnswerPolicyDecision",
    "AuthorizedCandidateSet",
    "CandidateDocument",
    "EvidenceQuality",
    "assess_evidence_quality",
    "decide_answer_policy",
    "manage_authorized_candidates",
]
