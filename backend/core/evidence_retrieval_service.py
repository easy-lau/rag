"""Independent, retrieval-first evidence service.

The service owns authorized candidate discovery, cross-channel fusion, optional
semantic verification and trace metadata.  A verifier is an enhancement: its
timeout, provider error or inconclusive result never erases retrieved evidence.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.evidence_admission import CandidateAdmission, admit_evidence_candidates
from core.evidence_contract import EvidencePack, RetrievalChannelReport
from core.knowledge_records import (
    search_knowledge_records,
    search_knowledge_records_for_chunks,
)
from core.rag_trace import content_fields, trace_event
from core.reranker import joint_rerank_with_coverage
from core.retriever import RRF_K, hybrid_search


PIPELINE_VERSION = "evidence_retrieval.v1"
_MAX_SELECTED_EVIDENCE = 6
_MAX_EXPANSION_CHUNKS = 2
_MIN_SECOND_STAGE_RECORDS = 8
_MAX_SECOND_STAGE_RECORDS = 12

CandidateSearch = Callable[..., Awaitable[list[dict[str, Any]]]]
EvidenceVerifier = Callable[..., Awaitable[Any]]
CandidateAdmissionPolicy = Callable[..., CandidateAdmission]


def _safe_error(exc: BaseException) -> str:
    return type(exc).__name__


def _safe_float(value: object) -> float:
    """Normalize untrusted verifier scores at the service boundary."""

    try:
        number = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _trace_completed(pack: EvidencePack, *, trace_id: str) -> None:
    """Emit one completion event through the repository content policy."""

    trace_event(
        "evidence.retrieval.completed",
        trace_id=trace_id,
        **pack.to_trace_dict(),
        **content_fields("question", pack.original_query),
        **content_fields("standalone_query", pack.resolved_query),
    )


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    source_kind = str(candidate.get("source_kind") or "document_chunk")
    candidate_id = str(
        candidate.get("record_id")
        or candidate.get("id")
        or candidate.get("chunk_id")
        or ""
    ).strip()
    return f"{source_kind}:{candidate_id}" if candidate_id else ""


def _fuse_ranked_channels(
    channels: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Fuse independently ranked channels with RRF, never raw-score addition."""

    fused: dict[str, dict[str, Any]] = {}
    order = 0
    for channel, values in channels:
        seen_in_channel: set[str] = set()
        for rank, raw in enumerate(values, start=1):
            candidate = dict(raw)
            identity = _candidate_identity(candidate)
            if not identity or identity in seen_in_channel:
                continue
            seen_in_channel.add(identity)
            reciprocal = 1.0 / (RRF_K + rank)
            current = fused.get(identity)
            if current is None:
                order += 1
                candidate["retrieval_channels"] = [channel]
                candidate["channel_ranks"] = {channel: rank}
                candidate["fusion_score"] = reciprocal
                candidate["fusion_order"] = order
                fused[identity] = candidate
                continue
            current["fusion_score"] = float(current.get("fusion_score") or 0.0) + reciprocal
            current["retrieval_channels"] = [
                *list(current.get("retrieval_channels") or []),
                channel,
            ]
            current["channel_ranks"] = {
                **dict(current.get("channel_ranks") or {}),
                channel: rank,
            }

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item.get("fusion_score") or 0.0),
            int(item.get("fusion_order") or 0),
        ),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["fusion_rank"] = rank
    return ranked[: max(1, int(limit or 1))]


def _merge_candidate_sequences(
    *groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge candidate stages without losing their source identity or order."""

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for raw in group:
            identity = _candidate_identity(raw)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            output.append(dict(raw))
    return output


def _scope_ids(value: object, *keys: str) -> set[str] | None:
    if not isinstance(value, Mapping):
        return None
    raw: object = None
    for key in keys:
        raw = value.get(key)
        if raw is not None:
            break
    if not isinstance(raw, (list, tuple, set)):
        return None
    return {str(item).strip() for item in raw if str(item).strip()}


def _filter_scope(
    candidates: Sequence[Mapping[str, Any]],
    *,
    evidence_scope_filter: Mapping[str, Any] | None,
    active_task_scope: object | None,
) -> list[dict[str, Any]]:
    allowed_doc_ids = _scope_ids(evidence_scope_filter, "doc_ids", "document_ids")
    if allowed_doc_ids is None and active_task_scope is not None:
        allowed_doc_ids = {
            str(value).strip()
            for value in (getattr(active_task_scope, "doc_ids", ()) or ())
            if str(value).strip()
        }
    allowed_record_ids = _scope_ids(evidence_scope_filter, "record_ids")

    output = [dict(item) for item in candidates]
    if allowed_doc_ids is not None:
        output = [
            item
            for item in output
            if str(item.get("doc_id") or "").strip() in allowed_doc_ids
        ]
    if allowed_record_ids is not None:
        output = [
            item
            for item in output
            if str(item.get("record_id") or "").strip() in allowed_record_ids
        ]
    return output


def _answer_requirement(query: str) -> list[dict[str, str]]:
    normalized = str(query or "").strip()
    if not normalized:
        return []
    return [{
        "id": "user_answer",
        "description": normalized[:500],
        "importance": "required",
        "source": "explicit",
        "coverage_mode": "single",
        "coverage_contract": "single_claim",
    }]


def _candidate_is_complete(candidate: Mapping[str, Any]) -> bool:
    return (
        bool(candidate.get("jointly_selected"))
        and str(candidate.get("coverage_status") or "").casefold() == "complete"
        and str(candidate.get("evidence_role") or "").casefold() == "direct"
        and str(candidate.get("joint_rerank_status") or "").casefold()
        == "verified_joint"
    )


def _verified_selection(
    results: Sequence[Mapping[str, Any]],
    *,
    scope_mode: str,
) -> tuple[dict[str, Any], ...]:
    complete = tuple(dict(item) for item in results if _candidate_is_complete(item))
    if not complete:
        return ()
    if scope_mode == "single":
        return complete[:1]
    return complete[:_MAX_SELECTED_EVIDENCE]


def _fallback_selection(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Keep the strongest bounded retrieval evidence when verification cannot promote it."""

    return tuple(dict(item) for item in candidates[:_MAX_SELECTED_EVIDENCE])


def _expansion_chunk_ids(outcome: object) -> tuple[uuid.UUID, ...]:
    if str(getattr(outcome, "coverage_status", "") or "").casefold() == "complete":
        return ()
    selected_indexes = {
        int(value)
        for value in (getattr(outcome, "selected_candidate_indexes", ()) or ())
        if isinstance(value, int) and not isinstance(value, bool)
    }
    ranked: list[tuple[int, float, float, uuid.UUID]] = []
    for item in (getattr(outcome, "results", ()) or ()):
        if not isinstance(item, Mapping):
            continue
        candidate_index = item.get("rerank_candidate_index")
        selected = (
            isinstance(candidate_index, int)
            and not isinstance(candidate_index, bool)
            and candidate_index in selected_indexes
        )
        related = (
            str(item.get("joint_rerank_status") or "").casefold() == "verified"
            and str(item.get("evidence_role") or "").casefold()
            in {"direct", "related"}
            and str(item.get("contribution_role") or "").casefold() != "irrelevant"
        )
        if not selected and not related:
            continue
        raw_chunk_id = item.get("chunk_id") or (
            item.get("id")
            if str(item.get("source_kind") or "document_chunk") == "document_chunk"
            else None
        )
        try:
            chunk_id = uuid.UUID(str(raw_chunk_id))
        except (TypeError, ValueError, AttributeError):
            continue
        ranked.append((
            int(selected),
            _safe_float(item.get("answer_support")),
            _safe_float(item.get("topic_relevance")),
            chunk_id,
        ))
    ranked.sort(reverse=True)
    output: list[uuid.UUID] = []
    for _selected, _support, _topic, chunk_id in ranked:
        if chunk_id not in output:
            output.append(chunk_id)
        if len(output) >= _MAX_EXPANSION_CHUNKS:
            break
    return tuple(output)


def _ambiguity_candidates(
    outcome: object,
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    indexes = {
        int(value)
        for value in (getattr(outcome, "ambiguity_candidate_indexes", ()) or ())
        if isinstance(value, int) and not isinstance(value, bool)
    }
    if not indexes:
        return ()
    return tuple(
        dict(item)
        for item in results
        if isinstance(item.get("rerank_candidate_index"), int)
        and int(item["rerank_candidate_index"]) in indexes
    )


class EvidenceRetrievalService:
    """Orchestrate retrieval independently from chat answer generation."""

    def __init__(
        self,
        *,
        structured_search: CandidateSearch = search_knowledge_records,
        chunk_search: CandidateSearch = hybrid_search,
        scoped_record_search: CandidateSearch = search_knowledge_records_for_chunks,
        verifier: EvidenceVerifier = joint_rerank_with_coverage,
        admission_policy: CandidateAdmissionPolicy = admit_evidence_candidates,
    ) -> None:
        self._structured_search = structured_search
        self._chunk_search = chunk_search
        self._scoped_record_search = scoped_record_search
        self._verifier = verifier
        self._admission_policy = admission_policy

    async def retrieve(
        self,
        *,
        db: AsyncSession,
        original_query: str,
        resolved_query: str,
        kb_ids: Sequence[uuid.UUID],
        method: str,
        top_k: int,
        verify: bool,
        trace_id: str,
        evidence_scope_filter: Mapping[str, Any] | None = None,
        active_task_scope: object | None = None,
    ) -> EvidencePack:
        started = time.perf_counter()
        query = str(resolved_query or original_query or "").strip()
        scoped_kbs = list(dict.fromkeys(kb_ids))
        reports: list[RetrievalChannelReport] = []
        structured: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []

        trace_event(
            "evidence.retrieval.started",
            trace_id=trace_id,
            pipeline_version=PIPELINE_VERSION,
            kb_count=len(scoped_kbs),
            method=method,
            top_k=top_k,
            verification_requested=verify,
            **content_fields("query", query),
        )

        channel_started = time.perf_counter()
        try:
            structured = await self._structured_search(
                db,
                query,
                scoped_kbs,
                top_k=top_k,
            )
            reports.append(RetrievalChannelReport(
                name="knowledge_records",
                status="succeeded",
                candidate_count=len(structured),
                elapsed_ms=round((time.perf_counter() - channel_started) * 1000),
            ))
        except Exception as exc:
            reports.append(RetrievalChannelReport(
                name="knowledge_records",
                status="failed",
                elapsed_ms=round((time.perf_counter() - channel_started) * 1000),
                error=_safe_error(exc),
            ))

        channel_started = time.perf_counter()
        chunk_error: BaseException | None = None
        try:
            chunks = await self._chunk_search(
                db,
                query,
                scoped_kbs,
                top_k=top_k,
                method=method,
                trace_id=trace_id,
                surface="evidence_retrieval",
            )
            reports.append(RetrievalChannelReport(
                name="document_chunks",
                status="succeeded",
                candidate_count=len(chunks),
                elapsed_ms=round((time.perf_counter() - channel_started) * 1000),
            ))
        except Exception as exc:
            chunk_error = exc
            reports.append(RetrievalChannelReport(
                name="document_chunks",
                status="failed",
                elapsed_ms=round((time.perf_counter() - channel_started) * 1000),
                error=_safe_error(exc),
            ))

        fused = _fuse_ranked_channels(
            (("knowledge_records", structured), ("document_chunks", chunks)),
            limit=max(top_k * 2, top_k),
        )
        candidates = _filter_scope(
            fused,
            evidence_scope_filter=evidence_scope_filter,
            active_task_scope=active_task_scope,
        )
        common_trace = {
            "pipeline_version": PIPELINE_VERSION,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

        if not candidates:
            service_failed = chunk_error is not None
            pack = EvidencePack(
                original_query=str(original_query or "").strip(),
                resolved_query=query,
                retrieval_status=("service_unavailable" if service_failed else "no_hit"),
                verification_status=("unverified" if service_failed else "not_requested"),
                outcome=("service_unavailable" if service_failed else "no_hit"),
                reason=("primary_retrieval_channel_failed" if service_failed else "no_candidates"),
                channel_reports=tuple(reports),
                trace=common_trace,
            )
            _trace_completed(pack, trace_id=trace_id)
            return pack

        admission = self._admission_policy(candidates, query=query)
        admitted = list(admission.candidates)
        admission_trace = admission.to_trace_dict()
        trace_event(
            "evidence.admission.completed",
            trace_id=trace_id,
            pipeline_version=PIPELINE_VERSION,
            **admission_trace,
        )
        pack_trace = {**common_trace, **admission_trace}
        if not admitted:
            pack = EvidencePack(
                original_query=str(original_query or "").strip(),
                resolved_query=query,
                retrieval_status="hit",
                admission_status=admission.status,
                verification_status="not_requested",
                outcome="insufficient_evidence",
                candidates=tuple(candidates),
                admitted_candidates=(),
                admission_rejections=admission.rejections,
                reason=admission.reason,
                channel_reports=tuple(reports),
                trace=pack_trace,
            )
            _trace_completed(pack, trace_id=trace_id)
            return pack

        scope_mode = (
            str(evidence_scope_filter.get("mode") or "").strip().casefold()
            if isinstance(evidence_scope_filter, Mapping)
            else ""
        )
        if not verify:
            selected = _fallback_selection(admitted)
            pack = EvidencePack(
                original_query=str(original_query or "").strip(),
                resolved_query=query,
                retrieval_status="hit",
                admission_status=admission.status,
                verification_status="not_requested",
                outcome="answered",
                candidates=tuple(candidates),
                admitted_candidates=tuple(admitted),
                admission_rejections=admission.rejections,
                selected_evidence=selected,
                reason="deterministic_admission_evidence",
                channel_reports=tuple(reports),
                trace=pack_trace,
            )
            _trace_completed(pack, trace_id=trace_id)
            return pack

        verification_started = time.perf_counter()
        verification_error: str | None = None
        verification_outcome: object | None = None
        verification_results: list[dict[str, Any]] = []
        verification_stage_count = 0
        display_candidates = list(candidates)
        effective_admitted = list(admitted)
        try:
            verification_stage_count = 1
            verification_outcome = await self._verifier(
                query,
                list(admitted),
                _answer_requirement(query),
            )
            verification_results = [
                dict(item)
                for item in (getattr(verification_outcome, "results", ()) or ())
                if isinstance(item, Mapping)
            ]
            if bool(getattr(verification_outcome, "succeeded", False)):
                expansion_ids = _expansion_chunk_ids(verification_outcome)
                if expansion_ids:
                    expanded = await self._scoped_record_search(
                        db,
                        query,
                        expansion_ids,
                        top_k=min(
                            _MAX_SECOND_STAGE_RECORDS,
                            max(_MIN_SECOND_STAGE_RECORDS, top_k),
                        ),
                    )
                    allowed_records = _scope_ids(evidence_scope_filter, "record_ids")
                    if allowed_records:
                        expanded = [
                            item
                            for item in expanded
                            if str(item.get("record_id") or "") in allowed_records
                        ]
                    selected_chunks = [
                        item
                        for item in admitted
                        if str(item.get("chunk_id") or item.get("id") or "")
                        in {str(value) for value in expansion_ids}
                    ]
                    second_stage = _fuse_ranked_channels(
                        (("expanded_records", expanded), ("selected_chunks", selected_chunks)),
                        limit=_MAX_SECOND_STAGE_RECORDS,
                    )
                    if second_stage:
                        for item in second_stage:
                            item["admission_status"] = "admitted"
                            item["admission_reason"] = (
                                "scoped_document_expansion"
                            )
                        display_candidates = _merge_candidate_sequences(
                            candidates,
                            second_stage,
                        )
                        effective_admitted = _merge_candidate_sequences(
                            admitted,
                            second_stage,
                        )
                        verification_stage_count = 2
                        verification_outcome = await self._verifier(
                            query,
                            second_stage,
                            _answer_requirement(query),
                        )
                        verification_results = [
                            dict(item)
                            for item in (getattr(verification_outcome, "results", ()) or ())
                            if isinstance(item, Mapping)
                        ]
            if not bool(getattr(verification_outcome, "succeeded", False)):
                verification_error = str(
                    getattr(verification_outcome, "failure_kind", None)
                    or "semantic_verification_inconclusive"
                )[:100]
        except Exception as exc:
            verification_error = _safe_error(exc)

        verification_meta = {
            **pack_trace,
            "verification_stage_count": verification_stage_count,
            "verification_elapsed_ms": round(
                (time.perf_counter() - verification_started) * 1000
            ),
            "verification_error": verification_error,
            "verification_failure_kind": getattr(
                verification_outcome,
                "failure_kind",
                None,
            ),
            "verification_model": getattr(verification_outcome, "model", None),
            "verification_coverage_status": getattr(
                verification_outcome,
                "coverage_status",
                None,
            ),
        }

        decision_status = str(
            getattr(verification_outcome, "decision_status", "") or ""
        ).casefold()
        verification_succeeded = bool(
            getattr(verification_outcome, "succeeded", False)
        )
        if verification_succeeded and decision_status == "ambiguous":
            ambiguous = _ambiguity_candidates(verification_outcome, verification_results)
            if len(ambiguous) >= 2:
                pack = EvidencePack(
                    original_query=str(original_query or "").strip(),
                    resolved_query=query,
                    retrieval_status="hit",
                    admission_status=admission.status,
                    verification_status="ambiguous",
                    outcome="needs_clarification",
                    candidates=tuple(display_candidates),
                    admitted_candidates=tuple(effective_admitted),
                    admission_rejections=admission.rejections,
                    ambiguity_candidates=ambiguous,
                    reason="multiple_semantic_evidence_sets",
                    channel_reports=tuple(reports),
                    trace=verification_meta,
                )
                _trace_completed(pack, trace_id=trace_id)
                return pack

        verified = (
            _verified_selection(verification_results, scope_mode=scope_mode)
            if verification_succeeded
            else ()
        )
        if verified:
            pack = EvidencePack(
                original_query=str(original_query or "").strip(),
                resolved_query=query,
                retrieval_status="hit",
                admission_status=admission.status,
                verification_status="verified",
                outcome="answered",
                candidates=tuple(display_candidates),
                admitted_candidates=tuple(effective_admitted),
                admission_rejections=admission.rejections,
                selected_evidence=verified,
                reason="verified_evidence",
                channel_reports=tuple(reports),
                trace=verification_meta,
            )
            _trace_completed(pack, trace_id=trace_id)
            return pack

        # Verification is advisory.  Inconclusive, insufficient or unavailable
        # verification degrades confidence, never the underlying retrieval hit.
        pack = EvidencePack(
            original_query=str(original_query or "").strip(),
            resolved_query=query,
            retrieval_status="hit",
            admission_status=admission.status,
            verification_status="unverified",
            outcome="answered",
            candidates=tuple(display_candidates),
            admitted_candidates=tuple(effective_admitted),
            admission_rejections=admission.rejections,
            selected_evidence=_fallback_selection(admitted),
            reason=(
                "verification_unavailable"
                if verification_error
                else "verification_did_not_promote_evidence"
            ),
            channel_reports=tuple(reports),
            trace=verification_meta,
        )
        _trace_completed(pack, trace_id=trace_id)
        return pack


__all__ = [
    "EvidenceRetrievalService",
    "PIPELINE_VERSION",
    "_fuse_ranked_channels",
]
