"""Typed, side-effect-free contracts for the RAG v2 execution path.

These contracts deliberately separate evidence availability, confidence and
completeness.  A soft dependency failure may therefore mark a bundle as
``degraded`` without erasing otherwise authorized retrieval evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


QUERY_PLAN_V2_SCHEMA_VERSION = "query_plan.v2"

AnswerShape = Literal[
    "fact",
    "overview",
    "list",
    "process",
    "comparison",
    "judgement",
    "multi_hop",
    "multi_part",
    "unknown",
]
ANSWER_SHAPES = frozenset(
    {
        "fact",
        "overview",
        "list",
        "process",
        "comparison",
        "judgement",
        "multi_hop",
        "multi_part",
        "unknown",
    }
)

RequirementRole = Literal["answer", "bridge"]
RequirementImportance = Literal["required", "helpful"]
RequirementSource = Literal["explicit", "inferred"]
PlanSource = Literal["local", "model", "fallback"]

EvidenceAvailability = Literal["ok", "degraded", "unavailable"]
EvidenceConfidence = Literal["verified", "retrieved", "none"]
EvidenceCompleteness = Literal["complete", "partial", "unknown"]
EvidenceConstraintStatus = Literal[
    "exact",
    "compatible",
    "neutral",
    "unknown",
    "mismatch",
]
EvidenceRole = Literal[
    "direct",
    "bridge",
    "complement",
    "background",
    "conflicting",
]
EVIDENCE_ROLES = frozenset(
    {
        "direct",
        "bridge",
        "complement",
        "background",
        "conflicting",
    }
)

_REQUIREMENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_QUERY_CHARS = 1000
_MAX_REQUIREMENT_CHARS = 500
_MAX_REASON_CHARS = 500
_MAX_STATE_REASONS = 12


def _normalized_text(value: object, *, field_name: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    return normalized


def _normalized_unique_texts(
    values: object,
    *,
    field_name: str,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _normalized_text(
            raw,
            field_name=field_name,
            max_chars=max_chars,
        )
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) > max_items:
            raise ValueError(f"{field_name} has too many items")
    return tuple(normalized)


def _normalized_requirement_ids(
    values: object,
    *,
    field_name: str,
    max_items: int = 8,
) -> tuple[str, ...]:
    normalized = _normalized_unique_texts(
        values,
        field_name=field_name,
        max_items=max_items,
        max_chars=64,
    )
    if any(not _REQUIREMENT_ID_RE.fullmatch(value) for value in normalized):
        raise ValueError(
            f"{field_name} must contain stable lowercase requirement ids"
        )
    return normalized


@dataclass(frozen=True)
class AnswerRequirementV2:
    id: str
    description: str
    role: RequirementRole = "answer"
    importance: RequirementImportance = "required"
    source: RequirementSource = "explicit"

    def __post_init__(self) -> None:
        requirement_id = str(self.id or "").strip()
        if not _REQUIREMENT_ID_RE.fullmatch(requirement_id):
            raise ValueError("requirement id must be a stable lowercase identifier")
        if self.role not in {"answer", "bridge"}:
            raise ValueError("requirement role must be answer or bridge")
        if self.importance not in {"required", "helpful"}:
            raise ValueError("requirement importance must be required or helpful")
        if self.source not in {"explicit", "inferred"}:
            raise ValueError("requirement source must be explicit or inferred")
        description = _normalized_text(
            self.description,
            field_name="requirement description",
            max_chars=_MAX_REQUIREMENT_CHARS,
        )
        object.__setattr__(self, "id", requirement_id)
        object.__setattr__(self, "description", description)

    @property
    def is_required_answer(self) -> bool:
        return self.role == "answer" and self.importance == "required"

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "description": self.description,
            "role": self.role,
            "importance": self.importance,
            "source": self.source,
        }


@dataclass(frozen=True)
class QueryPlanV2:
    original_query: str
    answer_shape: AnswerShape
    retrieval_queries: tuple[str, ...] = ()
    requirements: tuple[AnswerRequirementV2, ...] = ()
    confidence: float = 0.0
    source: PlanSource = "fallback"
    reason: str = ""
    needs_clarification: bool = False
    clarification_question: str | None = None
    schema_version: str = QUERY_PLAN_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != QUERY_PLAN_V2_SCHEMA_VERSION:
            raise ValueError("unsupported QueryPlanV2 schema version")
        if self.answer_shape not in ANSWER_SHAPES:
            raise ValueError("unsupported answer shape")
        if self.source not in {"local", "model", "fallback"}:
            raise ValueError("plan source must be local, model or fallback")
        if not isinstance(self.needs_clarification, bool):
            raise ValueError("needs_clarification must be a boolean")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence,
            (int, float),
        ):
            raise ValueError("plan confidence must be numeric")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("plan confidence must be between 0 and 1")

        if not isinstance(self.original_query, str):
            raise ValueError("original_query must be a string")
        original_query = re.sub(r"\s+", " ", self.original_query).strip()
        if len(original_query) > _MAX_QUERY_CHARS:
            raise ValueError(f"original_query exceeds {_MAX_QUERY_CHARS} characters")

        retrieval_queries = _normalized_unique_texts(
            self.retrieval_queries,
            field_name="retrieval query",
            max_items=8,
            max_chars=_MAX_QUERY_CHARS,
        )
        requirements = tuple(self.requirements)
        if len(requirements) > 8:
            raise ValueError("query plan has too many requirements")
        if any(not isinstance(item, AnswerRequirementV2) for item in requirements):
            raise ValueError("requirements must contain AnswerRequirementV2 values")
        requirement_ids = [item.id for item in requirements]
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("query plan contains duplicate requirement ids")

        reason = re.sub(r"\s+", " ", str(self.reason or "")).strip()
        if len(reason) > _MAX_REASON_CHARS:
            raise ValueError(f"plan reason exceeds {_MAX_REASON_CHARS} characters")
        clarification = self.clarification_question
        if clarification is not None:
            clarification = _normalized_text(
                clarification,
                field_name="clarification question",
                max_chars=_MAX_REASON_CHARS,
            )
        if self.needs_clarification and not clarification:
            raise ValueError("a clarification question is required")
        if not self.needs_clarification and clarification:
            raise ValueError("clarification question requires needs_clarification")

        ready_shape = self.answer_shape != "unknown" and not self.needs_clarification
        if ready_shape and not original_query:
            raise ValueError("a ready query plan requires an original query")
        if ready_shape and not retrieval_queries:
            raise ValueError("a ready query plan requires a retrieval query")
        if ready_shape and not requirements:
            raise ValueError("a ready query plan requires an answer requirement")

        object.__setattr__(self, "original_query", original_query)
        object.__setattr__(self, "retrieval_queries", retrieval_queries)
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "clarification_question", clarification)

    @property
    def allows_narrow_fact_path(self) -> bool:
        """Only a positive, high-confidence fact plan may use the narrow path."""

        return bool(
            self.answer_shape == "fact"
            and self.confidence >= 0.8
            and self.source != "fallback"
            and not self.needs_clarification
            and self.retrieval_queries
            and any(item.is_required_answer for item in self.requirements)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "original_query": self.original_query,
            "answer_shape": self.answer_shape,
            "retrieval_queries": list(self.retrieval_queries),
            "requirements": [item.to_dict() for item in self.requirements],
            "confidence": self.confidence,
            "source": self.source,
            "reason": self.reason,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
        }


@dataclass(frozen=True)
class EvidenceState:
    availability: EvidenceAvailability
    confidence: EvidenceConfidence
    completeness: EvidenceCompleteness
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.availability not in {"ok", "degraded", "unavailable"}:
            raise ValueError("unsupported evidence availability")
        if self.confidence not in {"verified", "retrieved", "none"}:
            raise ValueError("unsupported evidence confidence")
        if self.completeness not in {"complete", "partial", "unknown"}:
            raise ValueError("unsupported evidence completeness")
        reasons = _normalized_unique_texts(
            self.reasons,
            field_name="evidence state reason",
            max_items=_MAX_STATE_REASONS,
            max_chars=_MAX_REASON_CHARS,
        )
        if self.availability == "unavailable" and self.confidence != "none":
            raise ValueError("unavailable evidence cannot be marked as usable")
        if self.confidence == "none" and self.completeness == "complete":
            raise ValueError("complete evidence requires usable confidence")
        object.__setattr__(self, "reasons", reasons)

    @property
    def may_build_context(self) -> bool:
        return self.availability != "unavailable" and self.confidence != "none"

    @property
    def is_soft_degraded(self) -> bool:
        return self.availability == "degraded" and self.confidence != "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability": self.availability,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    doc_id: str
    kb_id: str
    content: str
    chunk_index: int = 0
    score: float | None = None
    confidence: Literal["verified", "retrieved"] = "retrieved"
    constraint_status: EvidenceConstraintStatus = "unknown"
    authorized: bool = True
    origins: tuple[str, ...] = ()
    role: EvidenceRole = "background"
    supports_requirement_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        chunk_id = _normalized_text(
            self.chunk_id,
            field_name="chunk_id",
            max_chars=200,
        )
        doc_id = _normalized_text(self.doc_id, field_name="doc_id", max_chars=200)
        kb_id = _normalized_text(self.kb_id, field_name="kb_id", max_chars=200)
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("evidence content must not be empty")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise ValueError("chunk_index must be an integer")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ValueError("evidence score must be numeric")
            score = float(self.score)
            if not math.isfinite(score) or score < 0:
                raise ValueError("evidence score must be finite and non-negative")
        else:
            score = None
        if self.confidence not in {"verified", "retrieved"}:
            raise ValueError("evidence item confidence must be verified or retrieved")
        if self.constraint_status not in {
            "exact",
            "compatible",
            "neutral",
            "unknown",
            "mismatch",
        }:
            raise ValueError("unsupported evidence constraint status")
        if not isinstance(self.authorized, bool):
            raise ValueError("authorized must be a boolean")
        if self.role not in EVIDENCE_ROLES:
            raise ValueError("unsupported evidence role")
        origins = _normalized_unique_texts(
            self.origins,
            field_name="evidence origin",
            max_items=12,
            max_chars=100,
        )
        supports_requirement_ids = _normalized_requirement_ids(
            self.supports_requirement_ids,
            field_name="supported requirement id",
        )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("evidence metadata must be a mapping")

        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "doc_id", doc_id)
        object.__setattr__(self, "kb_id", kb_id)
        object.__setattr__(self, "content", self.content.strip())
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "origins", origins)
        object.__setattr__(
            self,
            "supports_requirement_ids",
            supports_requirement_ids,
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "kb_id": self.kb_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "score": self.score,
            "confidence": self.confidence,
            "constraint_status": self.constraint_status,
            "authorized": self.authorized,
            "origins": list(self.origins),
            "role": self.role,
            "supports_requirement_ids": list(self.supports_requirement_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvidenceBundle:
    state: EvidenceState
    items: tuple[EvidenceItem, ...] = ()
    context_item_ids: tuple[str, ...] = ()
    answer_source_ids: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, EvidenceState):
            raise ValueError("state must be an EvidenceState")
        items = tuple(self.items)
        if any(not isinstance(item, EvidenceItem) for item in items):
            raise ValueError("items must contain EvidenceItem values")
        if any(not item.authorized for item in items):
            raise ValueError("an EvidenceBundle must not contain unauthorized items")
        item_by_id = {item.chunk_id: item for item in items}
        if len(item_by_id) != len(items):
            raise ValueError("evidence bundle contains duplicate chunk ids")

        context_ids = _normalized_unique_texts(
            self.context_item_ids,
            field_name="context item id",
            max_items=100,
            max_chars=200,
        )
        source_ids = _normalized_unique_texts(
            self.answer_source_ids,
            field_name="answer source id",
            max_items=100,
            max_chars=200,
        )
        missing_ids = _normalized_requirement_ids(
            self.missing_requirement_ids,
            field_name="missing requirement id",
            max_items=8,
        )
        if any(value not in item_by_id for value in context_ids):
            raise ValueError("context item ids must reference bundle items")
        if any(value not in set(context_ids) for value in source_ids):
            raise ValueError("answer source ids must reference context items")
        for chunk_id in context_ids:
            if item_by_id[chunk_id].constraint_status == "mismatch":
                raise ValueError("constraint-mismatched evidence cannot enter context")
        if context_ids and not self.state.may_build_context:
            raise ValueError("the evidence state does not permit a context")
        if self.state.completeness == "complete" and not context_ids:
            raise ValueError("complete evidence requires at least one context item")
        if self.state.completeness == "complete" and missing_ids:
            raise ValueError("complete evidence cannot have missing requirements")

        object.__setattr__(self, "items", items)
        object.__setattr__(self, "context_item_ids", context_ids)
        object.__setattr__(self, "answer_source_ids", source_ids)
        object.__setattr__(self, "missing_requirement_ids", missing_ids)

    @property
    def context_items(self) -> tuple[EvidenceItem, ...]:
        item_by_id = {item.chunk_id: item for item in self.items}
        return tuple(item_by_id[value] for value in self.context_item_ids)

    @property
    def answer_sources(self) -> tuple[EvidenceItem, ...]:
        item_by_id = {item.chunk_id: item for item in self.items}
        return tuple(item_by_id[value] for value in self.answer_source_ids)

    @property
    def covered_requirement_ids(self) -> tuple[str, ...]:
        """Requirement ids supported by evidence admitted to generation.

        Background material and contradictory evidence remain visible for
        diagnostics, but neither can establish positive requirement coverage.
        """

        covered: list[str] = []
        seen: set[str] = set()
        for item in self.context_items:
            if item.role not in {"direct", "bridge", "complement"}:
                continue
            for requirement_id in item.supports_requirement_ids:
                if requirement_id in seen:
                    continue
                seen.add(requirement_id)
                covered.append(requirement_id)
        return tuple(covered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "context_item_ids": list(self.context_item_ids),
            "answer_source_ids": list(self.answer_source_ids),
            "covered_requirement_ids": list(self.covered_requirement_ids),
            "missing_requirement_ids": list(self.missing_requirement_ids),
        }
