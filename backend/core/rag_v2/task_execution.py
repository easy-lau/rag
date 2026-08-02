"""Trusted execution provenance for the RAG v2 retrieval task graph.

``RetrievalTaskGraph`` describes *what* logical evidence paths are needed.
This module records *how this request actually searched for them*.  It exists
because document/chunk metadata is persisted, may originate from a previous
turn, and must therefore never be trusted as current task provenance.

The ledger is deliberately request-local and sidecar-only:

* retriever output is sanitised before it enters the ledger;
* candidate-to-task bindings live in memory, keyed by a stable chunk identity;
* equal physical searches can serve multiple logical tasks without losing any
  of their task ids;
* every expansion records its parent task/chunk lineage;
* execution status is diagnostic only.  It never claims that a candidate's
  text proves an answer requirement.

Evidence assembly consumes this ledger explicitly.  That keeps source text,
scope checks and bridge joins as the only authorities for semantic support.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from core.rag_v2.bridge_resolution import (
    BridgeFactConflict,
    BridgeScopeAmbiguity,
    ResolvedBridgeFact,
    bridge_fact_matches_candidate_scope,
)
from core.rag_v2.contracts import AnswerRequirementV2, BridgeClaimBinding
from core.rag_v2.task_graph import (
    AnswerBridgePath,
    BridgeEdgeMode,
    RetrievalTask,
    RetrievalTaskGraph,
)
from core.query_constraints import ApplicabilityScope, ScopeCandidateRejection
from core.terminology_contracts import (
    TerminologySnapshot,
    TerminologyVariantOrigin,
)
from core.terminology_runtime import TerminologyRuntimeResolution


TASK_LINEAGE_SCHEMA_VERSION = "rag_task_lineage.v2"

# A retrieval HTTP/SQL call finishing is deliberately not the same thing as a
# bridge being usable.  The latter requires source-grounded semantic parsing,
# and is the only state that may release a bridge-dependent answer task.
TaskRunStatus = Literal[
    "pending",
    "attempted",
    "succeeded",
    "failed",
    "blocked_dependency",
    "budget_skipped",
]
BridgeSemanticStatus = Literal[
    "not_applicable",
    "pending",
    "resolved",
    "no_fact",
    "conflict",
    "failed",
    "budget_skipped",
]
BridgeMaterializationStatus = Literal[
    "not_applicable",
    "pending",
    "eligible",
    "blocked_scope_ambiguity",
]
ExecutionRouteKind = Literal[
    "static",
    "bridge_second_hop",
    "bridge_same_source_closure",
    "derived",
]
_EXECUTION_ROUTE_KINDS = frozenset({
    "static",
    "bridge_second_hop",
    "bridge_same_source_closure",
    "derived",
})
_BRIDGE_SEMANTIC_STATUSES = frozenset({
    "not_applicable",
    "pending",
    "resolved",
    "no_fact",
    "conflict",
    "failed",
    "budget_skipped",
})

# These fields are execution controls, not source facts.  A document can be
# returned by a retriever with arbitrary metadata, and a previous response can
# be reloaded as carryover; neither may claim a task id/support/role in this
# request.  Keep this list central so every ingestion/expansion path gets the
# same fail-closed cleanup.
_UNTRUSTED_TASK_FIELDS = frozenset({
    "retrieval_task_ids",
    "retrieval_task_id",
    "task_direct_requirement_ids",
    "task_lineage_requirement_ids",
    "task_binding_status",
    "task_binding_rejected",
    "rag_task_lineage",
    "_rag_task_lineage",
    "__rag_task_lineage",
    "expansion_query_indexes",
    "bridge_expansion_query_index",
    "supports_requirement_ids",
    "evidence_role",
    "evidence_role_v2",
    "contribution_role",
    "role",
    "jointly_selected",
    "resolved_bridge_joins",
    "bridge_linked_requirement_ids",
    "bridge_link_rejected_requirement_ids",
    "direct_subject_answer_requirement_ids",
    "direct_subject_bridge_bypass_requirement_ids",
    "document_root_answer_requirement_ids",
    "document_policy_root_requirement_ids",
    "answer_claim_assertions",
    "claim_applicability",
    "claim_proof_kind",
    "strict_terminology_rule_ids",
    "conflicting_answer_requirement_ids",
    "bridge_conflicts",
})


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized_query(value: object) -> str:
    return _text(value).casefold()


def _bounded_unique(values: Iterable[object], *, limit: int) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) >= limit:
            break
    return tuple(result)


def candidate_identity(candidate: Mapping[str, Any]) -> str:
    """Return a request-local stable identity for an authorized chunk.

    A chunk UUID is normally globally unique.  The fuller fallback protects
    fake/test retrievers and older adapters that return only document/index.
    Empty identities are intentionally not entered in the ledger: inventing a
    provenance key for arbitrary content would let unrelated rows merge.
    """

    if not isinstance(candidate, Mapping):
        return ""
    kb_id = _text(candidate.get("kb_id"))
    doc_id = _text(candidate.get("doc_id"))
    chunk_id = _text(candidate.get("chunk_id") or candidate.get("id"))
    if kb_id and doc_id and chunk_id:
        return f"chunk:{kb_id}:{doc_id}:{chunk_id}"
    if chunk_id:
        return f"chunk:{chunk_id}"
    if kb_id and doc_id:
        index = candidate.get("chunk_index")
        if isinstance(index, bool):
            return ""
        try:
            return f"position:{kb_id}:{doc_id}:{int(index)}"
        except (TypeError, ValueError):
            return ""
    return ""


def candidate_chunk_id(candidate: Mapping[str, Any]) -> str:
    """Return the display/reference chunk id without manufacturing one."""

    return _text(candidate.get("chunk_id") or candidate.get("id"))


def source_chunk_identity(
    *,
    kb_id: object,
    doc_id: object,
    chunk_id: object,
) -> str:
    """Return a collision-safe identity for a source fact.

    A display ``chunk_id`` is often globally unique in the production store,
    but adapters, imports and tests are allowed to reuse it in different
    documents.  Bridge provenance must therefore use the full immutable
    source identity; a bare id can remain a diagnostic/seed convenience only.
    """

    normalized_kb_id = _text(kb_id)
    normalized_doc_id = _text(doc_id)
    normalized_chunk_id = _text(chunk_id)
    if not (normalized_kb_id and normalized_doc_id and normalized_chunk_id):
        return ""
    return (
        f"source:{normalized_kb_id}:{normalized_doc_id}:{normalized_chunk_id}"
    )


def candidate_source_identity(candidate: Mapping[str, Any]) -> str:
    """Return the compound source identity for one candidate, if complete."""

    if not isinstance(candidate, Mapping):
        return ""
    return source_chunk_identity(
        kb_id=candidate.get("kb_id"),
        doc_id=candidate.get("doc_id"),
        chunk_id=candidate_chunk_id(candidate),
    )


def sanitize_untrusted_task_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a retriever row after removing untrusted execution annotations."""

    item = dict(candidate)
    for field in _UNTRUSTED_TASK_FIELDS:
        item.pop(field, None)
    metadata = item.get("metadata")
    if isinstance(metadata, Mapping):
        safe_metadata = dict(metadata)
        for field in _UNTRUSTED_TASK_FIELDS:
            safe_metadata.pop(field, None)
        item["metadata"] = safe_metadata
    elif metadata is not None:
        # Invalid metadata should not become an implicit task channel.  Keep
        # the candidate, but normalise it to an empty mapping for downstream
        # evidence handling.
        item["metadata"] = {}
    return item


@dataclass(frozen=True)
class PhysicalRetrievalGroup:
    """One deduplicated physical search and all logical tasks it serves."""

    group_id: str
    query: str
    task_ids: tuple[str, ...]
    scope_product: str | None
    scope_version: str | None
    scope_explicit_version: bool
    # Canonical task applicability.  Product/version scalar fields remain a
    # read-compatible projection only; the fingerprint below is the physical
    # retrieval identity so source-verified project boundaries cannot be
    # erased by query-string deduplication.
    applicability_scope: ApplicabilityScope | None = None
    # Query spelling is runtime provenance, not source evidence.  Keeping it
    # on the physical group lets trace/ledger records tell an original task
    # search from an approved terminology alias without contaminating task
    # ownership or candidate metadata.
    terminology_variant_origin: TerminologyVariantOrigin = "original"
    terminology_rule_ids: tuple[str, ...] = ()
    # An alias may narrow a physical search to one approved KB/document.  The
    # pipeline must still intersect these values with the request's
    # API-authorised scope; these fields can only narrow, never grant access.
    retrieval_kb_ids: tuple[str, ...] | None = None
    retrieval_document_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        group_id = _text(self.group_id)
        query = _text(self.query)
        task_ids = _bounded_unique(self.task_ids, limit=32)
        if not group_id or not query or not task_ids:
            raise ValueError("physical retrieval group is incomplete")
        legacy_product = _text(self.scope_product) or None
        legacy_version = _text(self.scope_version) or None
        legacy_explicit_version = bool(self.scope_explicit_version)
        if legacy_explicit_version and legacy_version is None:
            raise ValueError(
                "scope_explicit_version requires scope_version"
            )
        supplied_scope = self.applicability_scope
        if supplied_scope is not None and not isinstance(
            supplied_scope,
            ApplicabilityScope,
        ):
            raise ValueError("applicability_scope must be an ApplicabilityScope")
        if supplied_scope is None:
            canonical_scope = ApplicabilityScope(
                product=legacy_product,
                version=legacy_version,
                explicit_version=legacy_explicit_version,
                extraction_reason="legacy_physical_group_scope_projection",
            )
        else:
            canonical_scope = supplied_scope
            if (
                legacy_product is not None
                and legacy_product != canonical_scope.product
            ):
                raise ValueError("scope_product conflicts with applicability_scope")
            if (
                legacy_version is not None
                and legacy_version != canonical_scope.version
            ):
                raise ValueError("scope_version conflicts with applicability_scope")
            if (
                legacy_explicit_version
                and not canonical_scope.explicit_version
            ):
                raise ValueError(
                    "scope_explicit_version conflicts with applicability_scope"
                )
        origin = str(self.terminology_variant_origin or "").strip().casefold()
        if origin not in {"original", "terminology_alias"}:
            raise ValueError("physical retrieval group terminology origin is invalid")
        rule_ids = _bounded_unique(self.terminology_rule_ids, limit=24)
        if origin == "original" and rule_ids:
            raise ValueError("original physical group cannot carry terminology rules")
        if origin == "terminology_alias" and not rule_ids:
            raise ValueError("terminology alias physical group requires rules")
        retrieval_kb_ids = (
            None
            if self.retrieval_kb_ids is None
            else _bounded_unique(self.retrieval_kb_ids, limit=64)
        )
        retrieval_document_ids = (
            None
            if self.retrieval_document_ids is None
            else _bounded_unique(self.retrieval_document_ids, limit=128)
        )
        if retrieval_kb_ids is not None and not retrieval_kb_ids:
            raise ValueError("physical retrieval group kb scope is invalid")
        if retrieval_document_ids is not None and not retrieval_document_ids:
            raise ValueError("physical retrieval group document scope is invalid")
        if retrieval_document_ids is not None and retrieval_kb_ids is None:
            raise ValueError(
                "physical retrieval group document scope requires kb scope"
            )
        if origin == "terminology_alias" and retrieval_kb_ids is None:
            # Legacy ``TerminologySnapshot`` remains a compatibility-only
            # caller, but all new runtime aliases must preserve their scoped
            # registry binding instead of issuing a global synonym search.
            raise ValueError("terminology alias physical group requires kb scope")
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "applicability_scope", canonical_scope)
        object.__setattr__(self, "scope_product", canonical_scope.product)
        object.__setattr__(self, "scope_version", canonical_scope.version)
        object.__setattr__(
            self,
            "scope_explicit_version",
            canonical_scope.explicit_version,
        )
        object.__setattr__(self, "terminology_variant_origin", origin)
        object.__setattr__(self, "terminology_rule_ids", rule_ids)
        object.__setattr__(self, "retrieval_kb_ids", retrieval_kb_ids)
        object.__setattr__(self, "retrieval_document_ids", retrieval_document_ids)

    @property
    def role_order(self) -> int:
        # Kept in the group instead of parsing ids in the executor.  Bridge
        # mapping should be attempted before optional answer refinements, but
        # no semantic dependency is implied by this presentation priority.
        return 0

    @property
    def scope_project(self) -> str | None:
        return self.applicability_scope.project if self.applicability_scope else None

    @property
    def scope_project_source(self):
        return (
            self.applicability_scope.project_source
            if self.applicability_scope
            else None
        )

    @property
    def scope_fingerprint(self) -> str:
        return (
            self.applicability_scope.fingerprint
            if self.applicability_scope
            else ApplicabilityScope().fingerprint
        )


@dataclass(frozen=True)
class RetrievalExecutionStage:
    """One dependency-safe wave of static retrieval work.

    A graph task's *materialised bridge query* is not executable merely because
    its dependency node has a query.  The task's original, literal answer
    query is nevertheless safe to run after the anchor: a source may mention
    the user's subject directly, or may be universally applicable.  The
    bridge-resolved query is an additional evidence route, not a reason to
    suppress that direct route.
    """

    stage_id: str
    groups: tuple[PhysicalRetrievalGroup, ...]

    def __post_init__(self) -> None:
        stage_id = _text(self.stage_id)
        if not stage_id:
            raise ValueError("execution stage requires a stable stage_id")
        if not self.groups:
            raise ValueError("execution stage requires at least one group")
        if any(not isinstance(group, PhysicalRetrievalGroup) for group in self.groups):
            raise ValueError("execution stage groups must be PhysicalRetrievalGroup values")
        object.__setattr__(self, "stage_id", stage_id)


@dataclass(frozen=True)
class RetrievalExecutionSchedule:
    """Dependency-aware runtime schedule derived from an immutable task graph.

    ``static_stages`` always contains the literal, user-worded answer query.
    Proof and augmentation bridge parents are separately exposed: a proof
    edge constrains evidence applicability, while an augmentation edge may
    additionally receive a second, materialised query after a bridge fact is
    proven.  The dynamic query is never allowed to run before that fact, and
    its absence cannot invalidate a direct evidence path for an augmentation
    edge.  Final evidence applicability remains the authority that decides
    whether a direct or bridged claim is sufficient.
    """

    static_stages: tuple[RetrievalExecutionStage, ...]
    bridge_augmented_answer_task_ids: tuple[str, ...]
    bridge_proof_answer_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stages = tuple(self.static_stages)
        if not stages:
            raise ValueError("execution schedule requires an anchor stage")
        if any(not isinstance(stage, RetrievalExecutionStage) for stage in stages):
            raise ValueError("execution schedule stages must be RetrievalExecutionStage values")
        if len({stage.stage_id for stage in stages}) != len(stages):
            raise ValueError("execution schedule contains duplicate stage ids")
        task_ids = _bounded_unique(self.bridge_augmented_answer_task_ids, limit=32)
        proof_task_ids = _bounded_unique(self.bridge_proof_answer_task_ids, limit=32)
        object.__setattr__(self, "static_stages", stages)
        object.__setattr__(self, "bridge_augmented_answer_task_ids", task_ids)
        object.__setattr__(self, "bridge_proof_answer_task_ids", proof_task_ids)

    @property
    def bridge_bound_answer_task_ids(self) -> tuple[str, ...]:
        """All bridge-consuming answer tasks, preserving stable DAG order."""

        return tuple(dict.fromkeys(
            (*self.bridge_proof_answer_task_ids, *self.bridge_augmented_answer_task_ids)
        ))


@dataclass(frozen=True)
class TaskExecutionRecord:
    execution_id: str
    kind: str
    query: str
    task_ids: tuple[str, ...]
    parent_task_ids: tuple[str, ...] = ()
    parent_chunk_ids: tuple[str, ...] = ()
    terminology_variant_origin: TerminologyVariantOrigin = "original"
    terminology_rule_ids: tuple[str, ...] = ()
    status: str = "attempted"
    candidate_count: int = 0
    error_reason: str | None = None
    route_kind: ExecutionRouteKind = "static"
    bridge_edge_mode: BridgeEdgeMode | None = None


@dataclass(frozen=True)
class BridgeResolution:
    """The semantic result of one bridge task in this exact execution.

    The scheduler records this separately from retrieval records because a
    successful retrieval with zero/ambiguous mappings must never release an
    answer child.  Every resolved fact remains linked to the execution ids and
    chunks that actually produced it; callers cannot attach a fact found by an
    unrelated anchor/answer search after the fact.  A resolved bridge may
    still have ``scope_ambiguities``: its facts remain valid for an exact,
    source-local closure, but it must not choose one of those alternatives to
    materialise a broader second-hop query.
    """

    bridge_task_id: str
    status: BridgeSemanticStatus
    facts: tuple[ResolvedBridgeFact, ...] = ()
    conflicts: tuple[BridgeFactConflict, ...] = ()
    scope_ambiguities: tuple[BridgeScopeAmbiguity, ...] = ()
    source_execution_ids: tuple[str, ...] = ()
    source_chunk_ids: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        task_id = _text(self.bridge_task_id)
        if not task_id:
            raise ValueError("bridge resolution requires a bridge_task_id")
        if self.status not in _BRIDGE_SEMANTIC_STATUSES - {"not_applicable", "pending"}:
            raise ValueError("bridge resolution has an unsupported terminal status")
        facts = tuple(self.facts)
        conflicts = tuple(self.conflicts)
        scope_ambiguities = tuple(self.scope_ambiguities)
        if self.status == "resolved" and not facts:
            raise ValueError("resolved bridge requires at least one fact")
        if self.status == "conflict" and not conflicts:
            raise ValueError("conflicted bridge requires conflict evidence")
        if self.status == "resolved" and conflicts:
            raise ValueError("resolved bridge cannot retain conflict semantics")
        if self.status == "conflict" and (facts or scope_ambiguities):
            raise ValueError("conflicted bridge cannot retain alternate semantics")
        if self.status in {"no_fact", "failed", "budget_skipped"} and (
            facts or conflicts or scope_ambiguities
        ):
            raise ValueError("unresolved bridge cannot retain semantic facts")
        object.__setattr__(self, "bridge_task_id", task_id)
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "conflicts", conflicts)
        object.__setattr__(self, "scope_ambiguities", scope_ambiguities)
        object.__setattr__(self, "source_execution_ids", _bounded_unique(
            self.source_execution_ids,
            limit=32,
        ))
        object.__setattr__(self, "source_chunk_ids", _bounded_unique(
            self.source_chunk_ids,
            limit=64,
        ))
        object.__setattr__(self, "reason", _text(self.reason) or None)

    @property
    def materialization_status(self) -> BridgeMaterializationStatus:
        """Whether this bridge can release an unambiguous second-hop query.

        This deliberately differs from semantic ``status``.  A source-local
        answer closure can retain a fact from each explicit version, while a
        dynamic query would otherwise select a version before final evidence
        has proved a complete answer route.
        """

        if self.status != "resolved":
            return "not_applicable"
        return (
            "blocked_scope_ambiguity"
            if self.scope_ambiguities
            else "eligible"
        )


@dataclass(frozen=True)
class CandidateTaskLineage:
    """Trusted current-run lineage for a candidate chunk."""

    run_id: str
    task_ids: tuple[str, ...]
    execution_ids: tuple[str, ...]
    parent_task_ids: tuple[str, ...]
    parent_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidateExecutionBinding:
    """One candidate observation from one physical execution.

    ``CandidateTaskLineage`` deliberately remains an aggregate diagnostic
    view.  It cannot prove a semantic route because unions of two executions
    lose the pairing between a parent bridge and an answer query.  This record
    preserves that pairing and is the only input accepted by bridge-joined
    evidence verification.
    """

    execution_id: str
    task_ids: tuple[str, ...]
    parent_task_ids: tuple[str, ...]
    parent_chunk_ids: tuple[str, ...]


class TaskExecutionLedger:
    """Mutable request-local provenance ledger for one immutable task graph."""

    def __init__(
        self,
        task_graph: RetrievalTaskGraph,
        *,
        run_id: str | None = None,
    ) -> None:
        if not isinstance(task_graph, RetrievalTaskGraph):
            raise ValueError("task_graph must be a RetrievalTaskGraph")
        supplied_run_id = _text(run_id)
        self.task_graph = task_graph
        self.run_id = supplied_run_id or uuid.uuid4().hex
        self._task_ids = frozenset(task_graph.task_by_id)
        self._lineage_by_identity: dict[str, CandidateTaskLineage] = {}
        self._lineage_by_chunk_id: dict[str, CandidateTaskLineage] = {}
        self._lineage_by_source_identity: dict[str, CandidateTaskLineage] = {}
        self._bindings_by_identity: dict[
            str, tuple[CandidateExecutionBinding, ...]
        ] = {}
        self._bindings_by_chunk_id: dict[
            str, tuple[CandidateExecutionBinding, ...]
        ] = {}
        self._bindings_by_source_identity: dict[
            str, tuple[CandidateExecutionBinding, ...]
        ] = {}
        self._records: dict[str, TaskExecutionRecord] = {}
        self._bridge_resolutions: dict[str, BridgeResolution] = {}
        # Scope admission happens before candidate observation.  Rejections
        # therefore cannot be represented as task lineage (there is no
        # admitted candidate to bind), but they still need request-local,
        # content-free diagnostics so ``scope_mismatch`` is distinguishable
        # from a normal no-hit.  The key deliberately contains only the typed
        # identity/fingerprint contract, never source content or metadata.
        self._scope_rejections: dict[
            tuple[str, str, str, str, str, tuple[str, ...], str],
            ScopeCandidateRejection,
        ] = {}
        self._task_state: dict[str, dict[str, Any]] = {
            task_id: {
                "attempted": 0,
                "succeeded": 0,
                "failed": 0,
                "blocked_dependency": 0,
                "budget_skipped": 0,
                "candidate_count": 0,
                "last_error": None,
                "blocked_by_task_ids": (),
                "proof_bridge_parent_task_ids": (
                    task_graph.task_by_id[task_id].bridge_parent_task_ids(
                        mode="proof"
                    )
                    if task_graph.task_by_id[task_id].role == "answer"
                    else ()
                ),
                "augmentation_bridge_parent_task_ids": (
                    task_graph.task_by_id[task_id].bridge_parent_task_ids(
                        mode="augmentation"
                    )
                    if task_graph.task_by_id[task_id].role == "answer"
                    else ()
                ),
                "bridge_status": (
                    "pending"
                    if task_graph.task_by_id[task_id].role == "bridge"
                    else "not_applicable"
                ),
                "bridge_materialization_status": (
                    "pending"
                    if task_graph.task_by_id[task_id].role == "bridge"
                    else "not_applicable"
                ),
                # A direct answer claim and a bridge-materialised answer
                # claim are alternative evidence routes.  Keep the dynamic
                # route's terminal state separate from normal retrieval
                # success/failure so it cannot erase a usable direct path.
                "bridge_augmentation_status": "pending"
                if (
                    task_graph.task_by_id[task_id].role == "answer"
                    and task_graph.answer_bridge_parent_task_ids(
                        task_id,
                        mode="augmentation",
                    )
                )
                else "not_applicable",
                "bridge_augmentation_reason": None,
            }
            for task_id in self._task_ids
        }
        self._execution_sequence = 0

    def _validate_task_ids(self, task_ids: Iterable[object]) -> tuple[str, ...]:
        normalized = _bounded_unique(task_ids, limit=32)
        if not normalized:
            raise ValueError("execution must own at least one known task id")
        unknown = set(normalized) - self._task_ids
        if unknown:
            raise ValueError("execution contains unknown task ids")
        return normalized

    def _new_execution_id(self, kind: str) -> str:
        self._execution_sequence += 1
        normalized_kind = re.sub(r"[^a-z0-9_]+", "_", _text(kind).casefold())
        normalized_kind = normalized_kind.strip("_") or "retrieval"
        return f"{normalized_kind}_{self._execution_sequence}"

    def begin_execution(
        self,
        *,
        kind: str,
        query: str,
        task_ids: Iterable[object],
        parent_task_ids: Iterable[object] = (),
        parent_chunk_ids: Iterable[object] = (),
        terminology_variant_origin: TerminologyVariantOrigin = "original",
        terminology_rule_ids: Iterable[object] = (),
        route_kind: ExecutionRouteKind = "static",
        bridge_edge_mode: BridgeEdgeMode | None = None,
    ) -> str:
        """Record an attempted physical retrieval/expansion operation."""

        normalized_task_ids = self._validate_task_ids(task_ids)
        normalized_parent_task_ids = _bounded_unique(parent_task_ids, limit=32)
        if set(normalized_parent_task_ids) - self._task_ids:
            raise ValueError("execution contains unknown parent task ids")
        execution_id = self._new_execution_id(kind)
        variant_origin = str(terminology_variant_origin or "").strip().casefold()
        if variant_origin not in {"original", "terminology_alias"}:
            raise ValueError("execution terminology variant origin is invalid")
        normalized_rule_ids = _bounded_unique(terminology_rule_ids, limit=24)
        if variant_origin == "original" and normalized_rule_ids:
            raise ValueError("original execution cannot carry terminology rules")
        if variant_origin == "terminology_alias" and not normalized_rule_ids:
            raise ValueError("terminology alias execution requires rules")
        normalized_route_kind = str(route_kind or "").strip().casefold()
        if normalized_route_kind not in _EXECUTION_ROUTE_KINDS:
            raise ValueError("execution route kind is invalid")
        normalized_bridge_edge_mode = (
            None
            if bridge_edge_mode is None
            else str(bridge_edge_mode or "").strip().casefold()
        )
        if normalized_route_kind in {
            "bridge_second_hop",
            "bridge_same_source_closure",
        }:
            if normalized_bridge_edge_mode not in {"proof", "augmentation"}:
                raise ValueError("bridge execution route requires an edge mode")
        elif normalized_bridge_edge_mode is not None:
            raise ValueError("only bridge execution routes may carry an edge mode")
        record = TaskExecutionRecord(
            execution_id=execution_id,
            kind=_text(kind) or "retrieval",
            query=_text(query),
            task_ids=normalized_task_ids,
            parent_task_ids=normalized_parent_task_ids,
            parent_chunk_ids=_bounded_unique(parent_chunk_ids, limit=32),
            terminology_variant_origin=variant_origin,
            terminology_rule_ids=normalized_rule_ids,
            route_kind=normalized_route_kind,  # type: ignore[arg-type]
            bridge_edge_mode=normalized_bridge_edge_mode,  # type: ignore[arg-type]
        )
        self._records[execution_id] = record
        for task_id in normalized_task_ids:
            self._task_state[task_id]["attempted"] = int(
                self._task_state[task_id]["attempted"]
            ) + 1
        return execution_id

    def finish_execution(
        self,
        execution_id: str,
        *,
        status: str,
        candidate_count: int = 0,
        error_reason: str | None = None,
    ) -> None:
        """Finish a known operation and update every logical owner."""

        record = self._records.get(_text(execution_id))
        if record is None:
            raise ValueError("unknown execution id")
        if status not in {"succeeded", "failed", "budget_skipped"}:
            raise ValueError("unsupported execution status")
        if candidate_count < 0:
            raise ValueError("candidate_count cannot be negative")
        finished = TaskExecutionRecord(
            execution_id=record.execution_id,
            kind=record.kind,
            query=record.query,
            task_ids=record.task_ids,
            parent_task_ids=record.parent_task_ids,
            parent_chunk_ids=record.parent_chunk_ids,
            terminology_variant_origin=record.terminology_variant_origin,
            terminology_rule_ids=record.terminology_rule_ids,
            status=status,
            candidate_count=int(candidate_count),
            error_reason=_text(error_reason) or None,
            route_kind=record.route_kind,
            bridge_edge_mode=record.bridge_edge_mode,
        )
        self._records[execution_id] = finished
        for task_id in record.task_ids:
            state = self._task_state[task_id]
            state[status] = int(state[status]) + 1
            state["candidate_count"] = int(state["candidate_count"]) + int(
                candidate_count
            )
            if status == "failed":
                state["last_error"] = finished.error_reason or "execution_failed"

    def mark_tasks_budget_skipped(
        self,
        task_ids: Iterable[object],
        *,
        reason: str = "candidate_budget_exhausted",
    ) -> None:
        """Expose a bounded-execution omission instead of silently dropping it."""

        normalized_task_ids = self._validate_task_ids(task_ids)
        for task_id in normalized_task_ids:
            state = self._task_state[task_id]
            state["budget_skipped"] = int(state["budget_skipped"]) + 1
            state["last_error"] = _text(reason) or "candidate_budget_exhausted"

    def mark_tasks_blocked_by_dependency(
        self,
        task_ids: Iterable[object],
        *,
        blocked_by_task_ids: Iterable[object] = (),
        reason: str = "bridge_dependency_unresolved",
    ) -> None:
        """Record that an answer was never materialized without its bridge.

        This is not a retrieval error and must not be represented as one.  The
        distinction tells the evidence layer and trace reader that the system
        intentionally refused a broad, dependency-free fallback query.
        """

        normalized_task_ids = self._validate_task_ids(task_ids)
        normalized_blocked_by = _bounded_unique(blocked_by_task_ids, limit=32)
        if set(normalized_blocked_by) - self._task_ids:
            raise ValueError("blocked dependency contains unknown task ids")
        for task_id in normalized_task_ids:
            task = self.task_graph.task_by_id[task_id]
            if task.role != "answer":
                raise ValueError(
                    "only proof-bridge answer tasks can be blocked by dependency"
                )
            proof_bridge_parent_ids = self.task_graph.answer_bridge_parent_task_ids(
                task_id,
                mode="proof",
            )
            if not proof_bridge_parent_ids:
                raise ValueError(
                    "only proof-bridge answer tasks can be blocked by dependency"
                )
            effective_blocked_by = (
                normalized_blocked_by or proof_bridge_parent_ids
            )
            if any(
                dependency_id not in proof_bridge_parent_ids
                for dependency_id in effective_blocked_by
            ):
                raise ValueError(
                    "answer can be blocked only by one of its proof bridge parents"
                )
            state = self._task_state[task_id]
            state["blocked_dependency"] = int(state["blocked_dependency"]) + 1
            state["last_error"] = _text(reason) or "bridge_dependency_unresolved"
            state["blocked_by_task_ids"] = effective_blocked_by

    def unavailable_static_retrieval_dependencies(
        self,
        task_ids: Iterable[object],
    ) -> tuple[str, ...]:
        """Return unavailable upstream nodes for one static physical group.

        ``RetrievalExecutionSchedule`` has two intentionally different kinds
        of dependency:

        * an anchor dependency controls whether a later *static* retrieval is
          safe to dispatch; and
        * a bridge dependency controls only a materialised second-hop route.

        A literal answer query may run before its bridge fact is resolved, so
        treating every graph edge as a static scheduling prerequisite would
        silently remove valid direct evidence.  Conversely, continuing after
        an unavailable anchor used to issue bridge/answer calls after the
        request's retrieval root had already failed.

        This method therefore evaluates only declared anchor parents that are
        not co-owned by the same coalesced physical group.  A parent is ready
        when it has at least one successful execution, even when that
        execution returned zero candidates.  Empty retrieval is a semantic
        no-hit, not an infrastructure failure.
        """

        normalized_task_ids = self._validate_task_ids(task_ids)
        coowned_task_ids = set(normalized_task_ids)
        unavailable: list[str] = []
        for task_id in normalized_task_ids:
            task = self.task_graph.task_by_id[task_id]
            for dependency_id in task.dependency_task_ids:
                if dependency_id in coowned_task_ids:
                    # Exact physical-query coalescing is the execution of the
                    # root and its direct child together; waiting for the
                    # child here would create an artificial self-block.
                    continue
                dependency = self.task_graph.task_by_id[dependency_id]
                if dependency.role != "anchor":
                    # Bridge parents are semantic gates for Wave 2, handled
                    # by bridge resolution rather than static scheduling.
                    continue
                dependency_state = self._task_state[dependency_id]
                if int(dependency_state["succeeded"]) > 0:
                    continue
                unavailable.append(dependency_id)
        return _bounded_unique(unavailable, limit=32)

    def mark_tasks_blocked_by_static_dependency(
        self,
        task_ids: Iterable[object],
        *,
        blocked_by_task_ids: Iterable[object],
        reason: str = "upstream_static_dependency_unavailable",
    ) -> None:
        """Record a scheduler-level refusal to dispatch a static group.

        This is deliberately separate from :meth:`mark_tasks_blocked_by_dependency`:
        the latter records a proof-bridge semantic decision for an answer,
        whereas this method records an upstream retrieval-health boundary and
        may legitimately block bridge tasks as well.  Both are visible as
        ``blocked_dependency`` in the ledger, but their graph validation and
        reasons stay non-interchangeable.
        """

        normalized_task_ids = self._validate_task_ids(task_ids)
        normalized_blocked_by = _bounded_unique(blocked_by_task_ids, limit=32)
        if not normalized_blocked_by:
            raise ValueError("static dependency block requires an upstream task")
        if set(normalized_blocked_by) - self._task_ids:
            raise ValueError("blocked dependency contains unknown task ids")

        coowned_task_ids = set(normalized_task_ids)
        for task_id in normalized_task_ids:
            task = self.task_graph.task_by_id[task_id]
            if task.role == "anchor":
                raise ValueError("anchor task cannot be blocked by a static dependency")
            static_parent_ids = {
                dependency_id
                for dependency_id in task.dependency_task_ids
                if dependency_id not in coowned_task_ids
                and self.task_graph.task_by_id[dependency_id].role == "anchor"
            }
            if not static_parent_ids:
                raise ValueError("task has no external static dependency")
            if any(
                dependency_id not in static_parent_ids
                for dependency_id in normalized_blocked_by
            ):
                raise ValueError(
                    "task can be blocked only by one of its static anchor parents"
                )
            if any(
                int(self._task_state[dependency_id]["succeeded"]) > 0
                for dependency_id in normalized_blocked_by
            ):
                raise ValueError("cannot block a task behind a successful dependency")

        for task_id in normalized_task_ids:
            state = self._task_state[task_id]
            state["blocked_dependency"] = int(state["blocked_dependency"]) + 1
            state["last_error"] = (
                _text(reason) or "upstream_static_dependency_unavailable"
            )
            state["blocked_by_task_ids"] = _bounded_unique(
                (*state["blocked_by_task_ids"], *normalized_blocked_by),
                limit=32,
            )

    def record_answer_bridge_augmentation(
        self,
        task_ids: Iterable[object],
        *,
        status: str,
        reason: str | None = None,
    ) -> None:
        """Record one terminal outcome for an optional bridge-assisted path.

        A bridge second hop can be released only from source-grounded facts,
        but no-fact/conflict/timeout is not a failure of an independently
        retrieved direct claim.  This diagnostic state makes that distinction
        explicit and prevents request ordering from changing task status.
        """

        allowed_statuses = {
            "released",
            "direct_closed",
            "skipped_no_fact",
            "skipped_conflict",
            "skipped_scope_ambiguous",
            "skipped_failed",
            "skipped_budget",
            "skipped_not_materializable",
        }
        normalized_status = _text(status).casefold()
        if normalized_status not in allowed_statuses:
            raise ValueError("unsupported bridge augmentation status")
        normalized_task_ids = self._validate_task_ids(task_ids)
        for task_id in normalized_task_ids:
            task = self.task_graph.task_by_id[task_id]
            if task.role != "answer":
                raise ValueError(
                    "only bridge-augmented answer tasks can record augmentation"
                )
            augmentation_parent_ids = self.task_graph.answer_bridge_parent_task_ids(
                task_id,
                mode="augmentation",
            )
            if not augmentation_parent_ids:
                raise ValueError(
                    "only bridge-augmented answer tasks can record augmentation"
                )
            state = self._task_state[task_id]
            if state["bridge_augmentation_status"] != "pending":
                raise ValueError("bridge augmentation already has a terminal status")
            state["bridge_augmentation_status"] = normalized_status
            state["bridge_augmentation_reason"] = _text(reason) or None

    def record_bridge_resolution(
        self,
        resolution: BridgeResolution,
    ) -> None:
        """Commit the one terminal semantic status for a bridge task.

        This is intentionally a ledger operation, not a loose trace payload:
        a resolved bridge fact must originate from a candidate that the same
        bridge task retrieved in this request.  It prevents an anchor hit or a
        stale carryover row from impersonating a prerequisite.
        """

        if not isinstance(resolution, BridgeResolution):
            raise ValueError("bridge resolution must be a BridgeResolution")
        task = self.task_graph.task_by_id.get(resolution.bridge_task_id)
        if task is None or task.role != "bridge":
            raise ValueError("bridge resolution must reference a bridge task")
        target_requirement_id = task.target_requirement_ids[0]
        if any(
            fact.requirement_id != target_requirement_id
            for fact in resolution.facts
        ):
            raise ValueError("bridge fact does not belong to its bridge task")
        if any(
            conflict.requirement_id != target_requirement_id
            for conflict in resolution.conflicts
        ):
            raise ValueError("bridge conflict does not belong to its bridge task")
        if any(
            ambiguity.requirement_id != target_requirement_id
            for ambiguity in resolution.scope_ambiguities
        ):
            raise ValueError(
                "bridge scope ambiguity does not belong to its bridge task"
            )

        fact_chunk_ids = _bounded_unique(
            (fact.source_chunk_id for fact in resolution.facts),
            limit=64,
        )
        expected_chunk_ids = _bounded_unique(
            (*resolution.source_chunk_ids, *fact_chunk_ids),
            limit=64,
        )
        expected_execution_ids = _bounded_unique(
            resolution.source_execution_ids,
            limit=32,
        )
        for execution_id in expected_execution_ids:
            record = self._records.get(execution_id)
            if record is None or task.task_id not in record.task_ids:
                raise ValueError(
                    "bridge resolution references an execution outside its task"
                )
        for fact in resolution.facts:
            source_identity = source_chunk_identity(
                kb_id=fact.source_kb_id,
                doc_id=fact.source_doc_id,
                chunk_id=fact.source_chunk_id,
            )
            lineage = self._lineage_by_source_identity.get(source_identity)
            if lineage is None or task.task_id not in lineage.task_ids:
                raise ValueError(
                    "bridge fact is not bound to the current bridge task"
                )
            if expected_execution_ids and not (
                set(lineage.execution_ids) & set(expected_execution_ids)
            ):
                raise ValueError(
                    "bridge fact does not originate from the declared execution"
                )

        # A task has one semantic outcome per request.  Replacing a terminal
        # result would make downstream answer release timing-dependent.
        if task.task_id in self._bridge_resolutions:
            raise ValueError("bridge task already has a semantic resolution")
        normalized_resolution = BridgeResolution(
            bridge_task_id=resolution.bridge_task_id,
            status=resolution.status,
            facts=resolution.facts,
            conflicts=resolution.conflicts,
            scope_ambiguities=resolution.scope_ambiguities,
            source_execution_ids=expected_execution_ids,
            source_chunk_ids=expected_chunk_ids,
            reason=resolution.reason,
        )
        self._bridge_resolutions[task.task_id] = normalized_resolution
        state = self._task_state[task.task_id]
        state["bridge_status"] = normalized_resolution.status
        state["bridge_materialization_status"] = (
            normalized_resolution.materialization_status
        )
        if normalized_resolution.status != "resolved":
            state["last_error"] = (
                normalized_resolution.reason
                or f"bridge_{normalized_resolution.status}"
            )

    def bridge_resolution_for_task(
        self,
        task_id: object,
    ) -> BridgeResolution | None:
        """Return the immutable semantic state for one bridge task."""

        normalized_task_id = _text(task_id)
        return self._bridge_resolutions.get(normalized_task_id)

    def bridge_resolutions(self) -> tuple[BridgeResolution, ...]:
        """Return bridge outcomes in graph order for stable trace rendering."""

        return tuple(
            self._bridge_resolutions[task.task_id]
            for task in self.task_graph.tasks
            if task.role == "bridge" and task.task_id in self._bridge_resolutions
        )

    def _append_lineage(
        self,
        candidate: Mapping[str, Any],
        *,
        task_ids: Iterable[object],
        execution_id: str,
        parent_task_ids: Iterable[object] = (),
        parent_chunk_ids: Iterable[object] = (),
    ) -> None:
        identity = candidate_identity(candidate)
        if not identity:
            return
        normalized_task_ids = self._validate_task_ids(task_ids)
        record = self._records.get(execution_id)
        if record is None:
            raise ValueError("candidate lineage requires a known execution")
        if not set(normalized_task_ids).issubset(record.task_ids):
            raise ValueError("candidate lineage exceeds execution task ownership")
        normalized_parent_task_ids = _bounded_unique(parent_task_ids, limit=32)
        if set(normalized_parent_task_ids) - self._task_ids:
            raise ValueError("candidate lineage contains an unknown parent task")
        normalized_parent_chunk_ids = _bounded_unique(parent_chunk_ids, limit=64)
        existing = self._lineage_by_identity.get(identity)
        lineage = CandidateTaskLineage(
            run_id=self.run_id,
            task_ids=_bounded_unique(
                (*((existing.task_ids) if existing else ()), *normalized_task_ids),
                limit=32,
            ),
            execution_ids=_bounded_unique(
                (*((existing.execution_ids) if existing else ()), execution_id),
                limit=32,
            ),
            parent_task_ids=_bounded_unique(
                (*((existing.parent_task_ids) if existing else ()), *normalized_parent_task_ids),
                limit=32,
            ),
            parent_chunk_ids=_bounded_unique(
                (*((existing.parent_chunk_ids) if existing else ()), *normalized_parent_chunk_ids),
                limit=64,
            ),
        )
        self._lineage_by_identity[identity] = lineage
        binding = CandidateExecutionBinding(
            execution_id=execution_id,
            task_ids=normalized_task_ids,
            parent_task_ids=normalized_parent_task_ids,
            parent_chunk_ids=normalized_parent_chunk_ids,
        )
        existing_bindings = self._bindings_by_identity.get(identity, ())
        if binding not in existing_bindings:
            self._bindings_by_identity[identity] = (*existing_bindings, binding)
        raw_chunk_id = candidate_chunk_id(candidate)
        if raw_chunk_id:
            existing_by_chunk = self._lineage_by_chunk_id.get(raw_chunk_id)
            if existing_by_chunk is None:
                self._lineage_by_chunk_id[raw_chunk_id] = lineage
            else:
                self._lineage_by_chunk_id[raw_chunk_id] = CandidateTaskLineage(
                    run_id=self.run_id,
                    task_ids=_bounded_unique(
                        (*existing_by_chunk.task_ids, *lineage.task_ids),
                        limit=32,
                    ),
                    execution_ids=_bounded_unique(
                        (*existing_by_chunk.execution_ids, *lineage.execution_ids),
                        limit=32,
                    ),
                    parent_task_ids=_bounded_unique(
                        (*existing_by_chunk.parent_task_ids, *lineage.parent_task_ids),
                        limit=32,
                    ),
                    parent_chunk_ids=_bounded_unique(
                        (*existing_by_chunk.parent_chunk_ids, *lineage.parent_chunk_ids),
                        limit=64,
                    ),
                )
            existing_chunk_bindings = self._bindings_by_chunk_id.get(
                raw_chunk_id,
                (),
            )
            if binding not in existing_chunk_bindings:
                self._bindings_by_chunk_id[raw_chunk_id] = (
                    *existing_chunk_bindings,
                    binding,
                )
        source_identity = candidate_source_identity(candidate)
        if source_identity:
            existing_by_source = self._lineage_by_source_identity.get(
                source_identity
            )
            if existing_by_source is None:
                self._lineage_by_source_identity[source_identity] = lineage
            else:
                self._lineage_by_source_identity[source_identity] = (
                    CandidateTaskLineage(
                        run_id=self.run_id,
                        task_ids=_bounded_unique(
                            (*existing_by_source.task_ids, *lineage.task_ids),
                            limit=32,
                        ),
                        execution_ids=_bounded_unique(
                            (*existing_by_source.execution_ids, *lineage.execution_ids),
                            limit=32,
                        ),
                        parent_task_ids=_bounded_unique(
                            (*existing_by_source.parent_task_ids, *lineage.parent_task_ids),
                            limit=32,
                        ),
                        parent_chunk_ids=_bounded_unique(
                            (*existing_by_source.parent_chunk_ids, *lineage.parent_chunk_ids),
                            limit=64,
                        ),
                    )
                )
            existing_source_bindings = self._bindings_by_source_identity.get(
                source_identity,
                (),
            )
            if binding not in existing_source_bindings:
                self._bindings_by_source_identity[source_identity] = (
                    *existing_source_bindings,
                    binding,
                )

    def observe_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        execution_id: str,
        task_ids: Iterable[object] | None = None,
        parent_task_ids: Iterable[object] = (),
        parent_chunk_ids: Iterable[object] = (),
    ) -> list[dict[str, Any]]:
        """Sanitise and bind retriever output to an already-started execution."""

        record = self._records.get(_text(execution_id))
        if record is None:
            raise ValueError("candidate lineage requires a known execution")
        normalized_task_ids = (
            self._validate_task_ids(task_ids)
            if task_ids is not None
            else record.task_ids
        )
        safe: list[dict[str, Any]] = []
        for raw in candidates:
            if not isinstance(raw, Mapping):
                continue
            item = sanitize_untrusted_task_metadata(raw)
            self._append_lineage(
                item,
                task_ids=normalized_task_ids,
                execution_id=execution_id,
                parent_task_ids=parent_task_ids,
                parent_chunk_ids=parent_chunk_ids,
            )
            safe.append(item)
        return safe

    def inherit_by_document(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        source_candidates: Sequence[Mapping[str, Any]],
        kind: str,
    ) -> list[dict[str, Any]]:
        """Inherit current-run lineage from seeds in the same document.

        Full-document expansion does not know which seed selected a sibling.
        Inheriting the union of the document's current task ids is safe because
        it records recall provenance only; semantic evidence is re-evaluated
        later against the sibling's own text.
        """

        by_document: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
        for source in source_candidates:
            identity = candidate_identity(source)
            lineage = self._lineage_by_identity.get(identity)
            if lineage is None:
                continue
            key = (_text(source.get("kb_id")), _text(source.get("doc_id")))
            if not all(key):
                continue
            existing = by_document.get(key, ((), ()))
            by_document[key] = (
                _bounded_unique((*existing[0], *lineage.task_ids), limit=32),
                _bounded_unique(
                    (*existing[1], candidate_chunk_id(source)), limit=64
                ),
            )

        safe_candidates = [
            sanitize_untrusted_task_metadata(candidate)
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ]
        grouped: dict[tuple[str, ...], list[tuple[dict[str, Any], tuple[str, ...]]]] = {}
        for candidate in safe_candidates:
            source = by_document.get(
                (_text(candidate.get("kb_id")), _text(candidate.get("doc_id")))
            )
            if source is None or not source[0]:
                continue
            grouped.setdefault(source[0], []).append((candidate, source[1]))
        for task_ids, rows in grouped.items():
            execution_id = self.begin_execution(
                kind=kind,
                query="",
                task_ids=task_ids,
            )
            for candidate, parent_chunk_ids in rows:
                self._append_lineage(
                    candidate,
                    task_ids=task_ids,
                    execution_id=execution_id,
                    parent_chunk_ids=parent_chunk_ids,
                )
            self.finish_execution(
                execution_id,
                status="succeeded",
                candidate_count=len(rows),
            )
        return safe_candidates

    def inherit_by_seed(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        kind: str,
        seed_field: str = "expansion_seed_chunk_ids",
        source_candidates: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Inherit only from retriever-reported seed ids already in the ledger.

        ``source_candidates`` narrows the parent universe to an already
        admitted set chosen by the caller.  This is required by scoped
        expansion: a retriever-provided bare chunk id is only a proposed edge,
        so it must not inherit a same-named seed from another document or
        applicability branch.  Ambiguous bare ids fail closed when an explicit
        source set is supplied.
        """

        # The caller may supply UUIDs or strings; normalisation in
        # ``candidate_chunk_id`` keeps the current-run seed identity stable.
        if source_candidates is None:
            chunk_to_lineage = dict(self._lineage_by_chunk_id)
        else:
            chunk_to_lineage: dict[str, CandidateTaskLineage] = {}
            source_identity_by_chunk: dict[str, str] = {}
            ambiguous_chunk_ids: set[str] = set()
            for source in source_candidates:
                if not isinstance(source, Mapping):
                    continue
                seed_id = candidate_chunk_id(source)
                source_identity = candidate_source_identity(source)
                lineage = self._lineage_by_identity.get(candidate_identity(source))
                if not (seed_id and source_identity and lineage is not None):
                    continue
                existing_identity = source_identity_by_chunk.get(seed_id)
                if existing_identity is None:
                    source_identity_by_chunk[seed_id] = source_identity
                    chunk_to_lineage[seed_id] = lineage
                    continue
                if existing_identity != source_identity:
                    ambiguous_chunk_ids.add(seed_id)
            for seed_id in ambiguous_chunk_ids:
                chunk_to_lineage.pop(seed_id, None)

        safe_candidates = [
            sanitize_untrusted_task_metadata(candidate)
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ]
        grouped: dict[
            tuple[str, ...],
            list[tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]],
        ] = {}
        for candidate in safe_candidates:
            raw_seed_ids = candidate.get(seed_field)
            if isinstance(raw_seed_ids, str):
                raw_seed_ids = [raw_seed_ids]
            if not isinstance(raw_seed_ids, (list, tuple, set)):
                continue
            seed_ids = tuple(
                seed_id
                for seed_id in _bounded_unique(raw_seed_ids, limit=16)
                if seed_id in chunk_to_lineage
            )
            parent_lineages = [
                chunk_to_lineage.get(seed_id) for seed_id in seed_ids
            ]
            parent_lineages = [item for item in parent_lineages if item is not None]
            if not parent_lineages:
                continue
            task_ids = _bounded_unique(
                (
                    task_id
                    for lineage in parent_lineages
                    for task_id in lineage.task_ids
                ),
                limit=32,
            )
            parent_task_ids = _bounded_unique(
                (
                    task_id
                    for lineage in parent_lineages
                    for task_id in lineage.task_ids
                ),
                limit=32,
            )
            if task_ids:
                grouped.setdefault(task_ids, []).append(
                    (candidate, parent_task_ids, seed_ids)
                )
        for task_ids, rows in grouped.items():
            execution_id = self.begin_execution(
                kind=kind,
                query="",
                task_ids=task_ids,
            )
            for candidate, parent_task_ids, parent_chunk_ids in rows:
                self._append_lineage(
                    candidate,
                    task_ids=task_ids,
                    execution_id=execution_id,
                    parent_task_ids=parent_task_ids,
                    parent_chunk_ids=parent_chunk_ids,
                )
            self.finish_execution(
                execution_id,
                status="succeeded",
                candidate_count=len(rows),
            )
        return safe_candidates

    def lineage_for_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> CandidateTaskLineage | None:
        return self._lineage_by_identity.get(candidate_identity(candidate))

    def execution_bindings_for_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> tuple[CandidateExecutionBinding, ...]:
        """Return atomic current-run bindings, never an aggregate union."""

        return self._bindings_by_identity.get(candidate_identity(candidate), ())

    def _bridge_source_execution_binding(
        self,
        *,
        bridge_task_id: str,
        fact: ResolvedBridgeFact,
        resolution: BridgeResolution,
    ) -> CandidateExecutionBinding | None:
        """Find the exact bridge-task execution that produced one fact."""

        source_identity = source_chunk_identity(
            kb_id=fact.source_kb_id,
            doc_id=fact.source_doc_id,
            chunk_id=fact.source_chunk_id,
        )
        for binding in self._bindings_by_source_identity.get(source_identity, ()):
            if bridge_task_id not in binding.task_ids:
                continue
            if (
                resolution.source_execution_ids
                and binding.execution_id not in resolution.source_execution_ids
            ):
                continue
            return binding
        return None

    def resolved_answer_bridge_bindings(
        self,
        candidate: Mapping[str, Any],
        *,
        path: AnswerBridgePath,
        facts: Sequence[ResolvedBridgeFact],
    ) -> tuple[BridgeClaimBinding, ...]:
        """Verify one complete bridge-joined answer route for this request.

        The checker intentionally operates on an individual candidate
        observation.  It rejects metadata-only claims and candidate-lineage
        unions, which otherwise allow a static D-level row to borrow the
        parent chunks of an unrelated dynamic query.
        """

        if not isinstance(path, AnswerBridgePath):
            raise ValueError("bridge route verification requires an AnswerBridgePath")
        expected_answer_task = self.task_graph.task_by_id.get(path.answer_task_id)
        if (
            expected_answer_task is None
            or expected_answer_task.role != "answer"
            or expected_answer_task.target_requirement_ids != (
                path.answer_requirement_id,
            )
        ):
            raise ValueError("bridge route references an invalid answer task")
        expected_path_parent_ids = (
            self.task_graph.answer_bridge_parent_task_ids(
                path.answer_task_id,
                mode="proof",
            )
            + self.task_graph.answer_bridge_parent_task_ids(
                path.answer_task_id,
                mode="augmentation",
            )
            if path.edge_mode == "augmentation"
            else self.task_graph.answer_bridge_parent_task_ids(
                path.answer_task_id,
                mode="proof",
            )
        )
        if tuple(expected_path_parent_ids) != path.bridge_task_ids:
            raise ValueError("bridge route does not match the task graph")
        if tuple(
            self.task_graph.task_by_id[task_id].target_requirement_ids[0]
            for task_id in path.bridge_task_ids
        ) != path.bridge_requirement_ids:
            raise ValueError("bridge route requirement mapping is invalid")

        facts_by_requirement: dict[str, ResolvedBridgeFact] = {}
        for fact in facts:
            if fact.requirement_id not in path.bridge_requirement_ids:
                return ()
            if fact.requirement_id in facts_by_requirement:
                # One answer route must name one unambiguous fact per bridge
                # requirement.  Competing values require separate paths.
                return ()
            facts_by_requirement[fact.requirement_id] = fact
        if set(facts_by_requirement) != set(path.bridge_requirement_ids):
            return ()

        answer_bindings = self.execution_bindings_for_candidate(candidate)
        expected_parent_tasks = frozenset(path.bridge_task_ids)
        expected_parent_chunks = frozenset(
            fact.source_chunk_id for fact in facts_by_requirement.values()
        )
        accepted_answer_binding = next(
            (
                binding
                for binding in answer_bindings
                if path.answer_task_id in binding.task_ids
                and frozenset(binding.parent_task_ids) == expected_parent_tasks
                and frozenset(binding.parent_chunk_ids) == expected_parent_chunks
                and (
                    (record := self._records.get(binding.execution_id)) is not None
                    and record.route_kind in {
                        "bridge_second_hop",
                        "bridge_same_source_closure",
                    }
                    and record.bridge_edge_mode == path.edge_mode
                )
            ),
            None,
        )
        if accepted_answer_binding is None:
            return ()

        bindings: list[BridgeClaimBinding] = []
        for bridge_task_id, bridge_requirement_id in zip(
            path.bridge_task_ids,
            path.bridge_requirement_ids,
        ):
            fact = facts_by_requirement[bridge_requirement_id]
            resolution = self.bridge_resolution_for_task(bridge_task_id)
            if resolution is None or resolution.status != "resolved":
                return ()
            matching_facts = [
                resolved_fact
                for resolved_fact in resolution.facts
                if (
                    resolved_fact.requirement_id == fact.requirement_id
                    and resolved_fact.value == fact.value
                    and resolved_fact.source_chunk_id == fact.source_chunk_id
                    and resolved_fact.source_doc_id == fact.source_doc_id
                    and resolved_fact.source_kb_id == fact.source_kb_id
                )
            ]
            if len(matching_facts) != 1:
                return ()
            source_binding = self._bridge_source_execution_binding(
                bridge_task_id=bridge_task_id,
                fact=fact,
                resolution=resolution,
            )
            if source_binding is None:
                return ()
            if not bridge_fact_matches_candidate_scope(fact, candidate):
                return ()
            bindings.append(BridgeClaimBinding(
                bridge_requirement_id=bridge_requirement_id,
                bridge_source_item_id=fact.source_chunk_id,
                bridge_value=fact.value,
                edge_mode=path.bridge_edge_modes[bridge_task_id],
                bridge_execution_id=source_binding.execution_id,
                answer_execution_id=accepted_answer_binding.execution_id,
            ))
        return tuple(bindings)

    def task_ids_for_candidate(
        self,
        candidate: Mapping[str, Any],
    ) -> tuple[str, ...]:
        lineage = self.lineage_for_candidate(candidate)
        return lineage.task_ids if lineage is not None else ()

    def merge_candidate_pools(
        self,
        *pools: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge candidate rows while preserving sidecar lineage unions.

        The ledger has already merged task ids by identity when each pool was
        observed.  This function intentionally never reads or writes a task
        field in candidate metadata.
        """

        merged: dict[str, dict[str, Any]] = {}
        anonymous: list[dict[str, Any]] = []
        for pool in pools:
            for raw in pool:
                if not isinstance(raw, Mapping):
                    continue
                incoming = dict(raw)
                identity = candidate_identity(incoming)
                if not identity:
                    anonymous.append(incoming)
                    continue
                current = merged.get(identity)
                if current is None:
                    merged[identity] = incoming
                    continue
                combined = dict(current)
                for key, value in incoming.items():
                    if key in {"candidate_origins", "origins"}:
                        combined[key] = list(dict.fromkeys([
                            *(
                                current.get(key)
                                if isinstance(current.get(key), (list, tuple, set))
                                else [current.get(key)]
                            ),
                            *(
                                value
                                if isinstance(value, (list, tuple, set))
                                else [value]
                            ),
                        ]))
                    elif key == "metadata" and isinstance(value, Mapping):
                        metadata = dict(
                            current.get("metadata")
                            if isinstance(current.get("metadata"), Mapping)
                            else {}
                        )
                        for metadata_key, metadata_value in value.items():
                            if metadata_key == "expansion_seed_chunk_ids":
                                metadata[metadata_key] = list(dict.fromkeys([
                                    *(
                                        metadata.get(metadata_key)
                                        if isinstance(
                                            metadata.get(metadata_key),
                                            (list, tuple, set),
                                        )
                                        else [metadata.get(metadata_key)]
                                    ),
                                    *(
                                        metadata_value
                                        if isinstance(
                                            metadata_value,
                                            (list, tuple, set),
                                        )
                                        else [metadata_value]
                                    ),
                                ]))
                            else:
                                metadata.setdefault(metadata_key, metadata_value)
                        combined["metadata"] = metadata
                    elif combined.get(key) is None or key not in combined:
                        combined[key] = value
                merged[identity] = combined
        return [*merged.values(), *anonymous]

    def bounded_merge_groups(
        self,
        groups_with_candidates: Sequence[
            tuple[PhysicalRetrievalGroup, Sequence[Mapping[str, Any]]]
        ],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Reserve a candidate for every successful logical task group.

        A bounded context cannot silently discard a third/fourth sub-question.
        If its only candidate cannot be retained, record a task budget skip so
        final completeness is necessarily partial and traceable.
        """

        bounded_limit = max(1, int(limit))
        merged = self.merge_candidate_pools(
            *(candidates for _, candidates in groups_with_candidates)
        )
        by_identity = {
            candidate_identity(item): item
            for item in merged
            if candidate_identity(item)
        }
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for group, candidates in groups_with_candidates:
            first_identity = next(
                (
                    candidate_identity(candidate)
                    for candidate in candidates
                    if candidate_identity(candidate) in by_identity
                    and candidate_identity(candidate) not in selected_ids
                ),
                "",
            )
            if not first_identity:
                continue
            if len(selected) >= bounded_limit:
                self.mark_tasks_budget_skipped(group.task_ids)
                continue
            selected.append(by_identity[first_identity])
            selected_ids.add(first_identity)
        for candidate in merged:
            identity = candidate_identity(candidate)
            if not identity or identity in selected_ids:
                continue
            if len(selected) >= bounded_limit:
                break
            selected.append(candidate)
            selected_ids.add(identity)
        return selected

    def execution_records(self) -> tuple[TaskExecutionRecord, ...]:
        return tuple(self._records.values())

    def record_scope_rejections(
        self,
        rejections: Sequence[ScopeCandidateRejection],
    ) -> None:
        """Record content-free scope admission rejections for this request.

        Scope admission is a pre-lineage gate: rejected rows must not enter
        ``observe_candidates`` or any evidence/context pool.  Keeping these
        typed records in a dedicated sidecar preserves the reason a request
        produced ``scope_mismatch`` without letting a rejected candidate
        masquerade as current-run retrieval evidence.

        Repeated physical retrieval groups can report the same rejection.  A
        stable identity/fingerprint key makes recording idempotent while
        preserving first-observed order for trace rendering.
        """

        if isinstance(rejections, (str, bytes)) or not isinstance(
            rejections,
            Sequence,
        ):
            raise ValueError("scope rejections must be a sequence")
        for rejection in rejections:
            if not isinstance(rejection, ScopeCandidateRejection):
                raise ValueError(
                    "scope rejections must contain ScopeCandidateRejection"
                )
            key = (
                rejection.kb_id,
                rejection.doc_id,
                rejection.chunk_id,
                rejection.expected_scope_fingerprint,
                rejection.actual_identity_fingerprint,
                tuple(rejection.mismatch_dimensions),
                rejection.reason_code,
            )
            if key not in self._scope_rejections:
                self._scope_rejections[key] = rejection

    def scope_rejections(self) -> tuple[ScopeCandidateRejection, ...]:
        """Return immutable, insertion-stable scope diagnostics only.

        The returned dataclasses have no content, filename, arbitrary metadata
        or retriever score fields.  Callers must not turn them back into
        candidates; their sole role is status/trace attribution.
        """

        return tuple(self._scope_rejections.values())

    def scope_rejection_summary(self) -> dict[str, Any]:
        """Return a privacy-safe aggregate for traces and terminal status."""

        dimension_counts: dict[str, int] = {}
        reason_counts: dict[str, int] = {}
        candidate_keys: set[tuple[str, str, str]] = set()
        for rejection in self._scope_rejections.values():
            candidate_keys.add((
                rejection.kb_id,
                rejection.doc_id,
                rejection.chunk_id,
            ))
            reason_counts[rejection.reason_code] = (
                reason_counts.get(rejection.reason_code, 0) + 1
            )
            for dimension in rejection.mismatch_dimensions:
                dimension_counts[dimension] = (
                    dimension_counts.get(dimension, 0) + 1
                )
        return {
            "rejection_count": len(self._scope_rejections),
            "candidate_count": len(candidate_keys),
            "mismatch_dimension_counts": {
                key: dimension_counts[key]
                for key in sorted(dimension_counts)
            },
            "reason_counts": {
                key: reason_counts[key]
                for key in sorted(reason_counts)
            },
        }

    def task_state_summary(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for task_id in sorted(self._task_state):
            state = self._task_state[task_id]
            if int(state["succeeded"]) > 0:
                status = "succeeded"
            elif int(state["failed"]) > 0:
                status = "failed"
            elif int(state["blocked_dependency"]) > 0:
                status = "blocked_dependency"
            elif int(state["budget_skipped"]) > 0:
                status = "budget_skipped"
            elif int(state["attempted"]) > 0:
                status = "attempted"
            else:
                status = "pending"
            result[task_id] = {
                "status": status,
                "attempted": int(state["attempted"]),
                "succeeded": int(state["succeeded"]),
                "failed": int(state["failed"]),
                "blocked_dependency": int(state["blocked_dependency"]),
                "budget_skipped": int(state["budget_skipped"]),
                "candidate_count": int(state["candidate_count"]),
                "last_error": state["last_error"],
                "blocked_by_task_ids": list(state["blocked_by_task_ids"]),
                "proof_bridge_parent_task_ids": list(
                    state["proof_bridge_parent_task_ids"]
                ),
                "augmentation_bridge_parent_task_ids": list(
                    state["augmentation_bridge_parent_task_ids"]
                ),
                "bridge_status": state["bridge_status"],
                "bridge_materialization_status": state[
                    "bridge_materialization_status"
                ],
                "bridge_augmentation_status": state["bridge_augmentation_status"],
                "bridge_augmentation_reason": state["bridge_augmentation_reason"],
            }
        return result

    def safe_summary(self) -> dict[str, Any]:
        states = self.task_state_summary()
        return {
            "schema_version": TASK_LINEAGE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "task_count": len(states),
            "bound_candidate_count": len(self._lineage_by_identity),
            "execution_count": len(self._records),
            "scope_rejection_summary": self.scope_rejection_summary(),
            "bridge_resolutions": [
                {
                    "bridge_task_id": item.bridge_task_id,
                    "status": item.status,
                    "materialization_status": item.materialization_status,
                    "fact_count": len(item.facts),
                    "conflict_count": len(item.conflicts),
                    "scope_ambiguity_count": len(item.scope_ambiguities),
                    "source_execution_ids": list(item.source_execution_ids),
                    "source_chunk_ids": list(item.source_chunk_ids),
                    "reason": item.reason,
                }
                for item in self.bridge_resolutions()
            ],
            "task_states": states,
        }


def _build_physical_groups(
    tasks: Sequence[RetrievalTask],
    *,
    stage_id: str,
    anchor_query: str | None,
    requirements_by_id: Mapping[str, AnswerRequirementV2],
    terminology_runtime_resolution: TerminologyRuntimeResolution | None,
    terminology_snapshot: TerminologySnapshot | None,
    maximum_terminology_aliases: int,
) -> tuple[PhysicalRetrievalGroup, ...]:
    """Merge equivalent logical tasks inside one static scheduling stage.

    Cross-stage coalescing is handled explicitly by
    :func:`build_retrieval_execution_schedule`.  It is deliberately limited
    to the initial anchor/answer recall route, whose shared physical query
    carries no bridge proof.  Dynamic bridge materialisation remains in its
    own later wave with parent-fact bindings.
    """

    role_priority = {"anchor": 0, "bridge": 1, "answer": 2}
    ordered_tasks = sorted(
        tasks,
        key=lambda task: (
            role_priority[task.role],
            0 if task.required else 1,
            task.task_id,
        ),
    )
    grouped: dict[
        tuple[object, ...],
        list[
            tuple[
                RetrievalTask,
                str,
                str,
                tuple[str, ...],
                tuple[str, ...] | None,
                tuple[str, ...] | None,
            ]
        ],
    ] = {}
    for task in ordered_tasks:
        base_query = (
            anchor_query
            if task.role == "anchor" and anchor_query
            else task.query
        )
        # The literal task query is always present.  Registry aliases are
        # separate physical groups so a failed/ambiguous terminology read
        # cannot replace or consume the baseline retrieval route.
        variants: list[
            tuple[
                str,
                TerminologyVariantOrigin,
                tuple[str, ...],
                tuple[str, ...] | None,
                tuple[str, ...] | None,
            ]
        ] = [(base_query, "original", (), None, None)]
        # Terminology can only alter an answer node.  Bridge subjects and
        # bridge-resolved values remain source facts and must never be
        # synonym-expanded.
        if task.role == "answer" and task.target_requirement_ids:
            requirement = requirements_by_id.get(task.target_requirement_ids[0])
            if requirement is not None and terminology_runtime_resolution is not None:
                for variant in terminology_runtime_resolution.retrieval_variants(
                    requirement=requirement,
                    maximum_aliases=maximum_terminology_aliases,
                ):
                    variants.append((
                        variant.query,
                        "terminology_alias",
                        variant.rule_ids,
                        variant.kb_ids,
                        variant.document_ids,
                    ))
            elif terminology_snapshot is not None:
                # A legacy snapshot cannot carry KB/document scope, so it is
                # intentionally no longer allowed to create a global alias
                # query.  Keep its literal compatibility path only; callers
                # must migrate to TerminologyRuntimeResolution to opt in.
                _ = terminology_snapshot
        for (
            variant_query,
            variant_origin,
            variant_rule_ids,
            variant_kb_ids,
            variant_document_ids,
        ) in variants:
            normalized_query = _normalized_query(variant_query)
            if not normalized_query:
                continue
            key = (
                normalized_query,
                task.scope_fingerprint,
                variant_origin,
                variant_kb_ids,
                variant_document_ids,
            )
            grouped.setdefault(key, []).append((
                task,
                variant_query,
                variant_origin,
                variant_rule_ids,
                variant_kb_ids,
                variant_document_ids,
            ))
    result: list[PhysicalRetrievalGroup] = []
    for index, variants_in_group in enumerate(grouped.values(), start=1):
        (
            first_task,
            first_query,
            first_origin,
            _,
            first_kb_ids,
            first_document_ids,
        ) = variants_in_group[0]
        rule_ids = tuple(dict.fromkeys(
            rule_id
            for _task, _query, _origin, variant_rule_ids, _kb_ids, _doc_ids
            in variants_in_group
            for rule_id in variant_rule_ids
        ))
        result.append(PhysicalRetrievalGroup(
            group_id=f"{stage_id}_{index}",
            query=_text(first_query),
            task_ids=tuple(dict.fromkeys(
                task.task_id for task, *_ in variants_in_group
            )),
            scope_product=first_task.scope_product,
            scope_version=first_task.scope_version,
            scope_explicit_version=bool(first_task.scope_explicit_version),
            applicability_scope=first_task.applicability_scope,
            terminology_variant_origin=first_origin,
            terminology_rule_ids=rule_ids,
            retrieval_kb_ids=first_kb_ids,
            retrieval_document_ids=first_document_ids,
        ))
    return tuple(result)


def build_retrieval_execution_schedule(
    task_graph: RetrievalTaskGraph,
    *,
    anchor_query: str | None = None,
    terminology_runtime_resolution: TerminologyRuntimeResolution | None = None,
    terminology_snapshot: TerminologySnapshot | None = None,
    maximum_terminology_aliases: int = 0,
) -> RetrievalExecutionSchedule:
    """Compile a task graph into dependency-safe static retrieval stages.

    The graph shape deliberately permits only anchor -> bridge -> answer (or
    anchor -> answer) edges.  An answer's literal direct query is a recall
    route, not a proof of its bridge edge.  If it is exactly the same physical
    query as the anchor (including scope and terminology variant), both task
    identities share one static execution.  Only an explicitly declared
    bridge edge receives a later materialised route; proof still requires the
    dynamic binding and is never inferred from this coalescing.
    """

    if not isinstance(task_graph, RetrievalTaskGraph):
        raise ValueError("task_graph must be a RetrievalTaskGraph")
    if (
        terminology_runtime_resolution is not None
        and not isinstance(terminology_runtime_resolution, TerminologyRuntimeResolution)
    ):
        raise ValueError(
            "terminology_runtime_resolution must be a TerminologyRuntimeResolution"
        )
    if terminology_runtime_resolution is not None and terminology_snapshot is not None:
        raise ValueError(
            "runtime terminology resolution cannot be combined with legacy snapshot"
        )
    task_by_id = task_graph.task_by_id
    requirements_by_id = {item.id: item for item in task_graph.requirements}
    anchors = tuple(task for task in task_graph.tasks if task.role == "anchor")
    if len(anchors) != 1:
        raise ValueError("task graph execution requires exactly one anchor task")
    anchor = anchors[0]
    if anchor.dependency_task_ids:
        raise ValueError("anchor task cannot have dependencies")

    ready_after_anchor: list[RetrievalTask] = []
    bridge_augmented_answers: list[str] = []
    bridge_proof_answers: list[str] = []
    for task in task_graph.tasks:
        if task.task_id == anchor.task_id:
            continue
        dependencies = tuple(task.dependency_task_ids)
        dependency_roles = tuple(task_by_id[item].role for item in dependencies)
        if task.role == "bridge":
            if set(dependencies) != {anchor.task_id} or dependency_roles != ("anchor",):
                raise ValueError("bridge tasks must depend only on the anchor task")
            ready_after_anchor.append(task)
            continue
        if task.role != "answer":
            raise ValueError("unsupported task role in execution schedule")
        if any(role == "bridge" for role in dependency_roles):
            ready_after_anchor.append(task)
            if task_graph.answer_bridge_parent_task_ids(
                task.task_id,
                mode="proof",
            ):
                bridge_proof_answers.append(task.task_id)
            if task_graph.answer_bridge_parent_task_ids(
                task.task_id,
                mode="augmentation",
            ):
                bridge_augmented_answers.append(task.task_id)
            continue
        if set(dependencies) != {anchor.task_id} or dependency_roles != ("anchor",):
            raise ValueError("bridge-free answer tasks must depend only on anchor")
        ready_after_anchor.append(task)

    anchor_groups = list(_build_physical_groups(
        (anchor,),
        stage_id="anchor",
        anchor_query=anchor_query,
        requirements_by_id=requirements_by_id,
        terminology_runtime_resolution=terminology_runtime_resolution,
        terminology_snapshot=terminology_snapshot,
        maximum_terminology_aliases=maximum_terminology_aliases,
    ))
    after_anchor_groups = list(_build_physical_groups(
        ready_after_anchor,
        stage_id="after_anchor",
        anchor_query=anchor_query,
        requirements_by_id=requirements_by_id,
        terminology_runtime_resolution=terminology_runtime_resolution,
        terminology_snapshot=terminology_snapshot,
        maximum_terminology_aliases=maximum_terminology_aliases,
    )) if ready_after_anchor else []

    def physical_identity(group: PhysicalRetrievalGroup) -> tuple[object, ...]:
        return (
            _normalized_query(group.query),
            group.scope_fingerprint,
            group.terminology_variant_origin,
            group.terminology_rule_ids,
            group.retrieval_kb_ids,
            group.retrieval_document_ids,
        )

    # The anchor is a recall root.  Coalescing only exactly-equivalent static
    # groups saves duplicate retrieval while retaining every logical owner in
    # the ledger.  It cannot manufacture a bridge route because dynamic
    # bridge executions carry explicit parent task/chunk bindings.
    anchor_index = {
        physical_identity(group): index
        for index, group in enumerate(anchor_groups)
    }
    remaining_after_anchor_groups: list[PhysicalRetrievalGroup] = []
    for group in after_anchor_groups:
        index = anchor_index.get(physical_identity(group))
        if index is None:
            remaining_after_anchor_groups.append(group)
            continue
        anchor_group = anchor_groups[index]
        anchor_groups[index] = PhysicalRetrievalGroup(
            group_id=anchor_group.group_id,
            query=anchor_group.query,
            task_ids=tuple(dict.fromkeys(
                (*anchor_group.task_ids, *group.task_ids)
            )),
            scope_product=anchor_group.scope_product,
            scope_version=anchor_group.scope_version,
            scope_explicit_version=anchor_group.scope_explicit_version,
            applicability_scope=anchor_group.applicability_scope,
            terminology_variant_origin=anchor_group.terminology_variant_origin,
            terminology_rule_ids=anchor_group.terminology_rule_ids,
            retrieval_kb_ids=anchor_group.retrieval_kb_ids,
            retrieval_document_ids=anchor_group.retrieval_document_ids,
        )

    stages: list[RetrievalExecutionStage] = [
        RetrievalExecutionStage(
            stage_id="anchor",
            groups=tuple(anchor_groups),
        )
    ]
    if remaining_after_anchor_groups:
        stages.append(RetrievalExecutionStage(
            stage_id="after_anchor",
            groups=tuple(remaining_after_anchor_groups),
        ))
    return RetrievalExecutionSchedule(
        static_stages=tuple(stages),
        bridge_augmented_answer_task_ids=tuple(bridge_augmented_answers),
        bridge_proof_answer_task_ids=tuple(bridge_proof_answers),
    )


def build_initial_retrieval_groups(
    task_graph: RetrievalTaskGraph,
    *,
    anchor_query: str | None = None,
    terminology_runtime_resolution: TerminologyRuntimeResolution | None = None,
    terminology_snapshot: TerminologySnapshot | None = None,
    maximum_terminology_aliases: int = 0,
) -> tuple[PhysicalRetrievalGroup, ...]:
    """Return the graph's statically executable groups.

    This projection includes every static direct query.  Callers that need a
    bridge-materialised second hop must consume
    :func:`build_retrieval_execution_schedule` and use only its explicit
    augmentation task ids after bridge resolution.
    """

    schedule = build_retrieval_execution_schedule(
        task_graph,
        anchor_query=anchor_query,
        terminology_runtime_resolution=terminology_runtime_resolution,
        terminology_snapshot=terminology_snapshot,
        maximum_terminology_aliases=maximum_terminology_aliases,
    )
    return tuple(
        group
        for stage in schedule.static_stages
        for group in stage.groups
    )


__all__ = [
    "TASK_LINEAGE_SCHEMA_VERSION",
    "CandidateExecutionBinding",
    "CandidateTaskLineage",
    "BridgeResolution",
    "BridgeMaterializationStatus",
    "BridgeSemanticStatus",
    "PhysicalRetrievalGroup",
    "RetrievalExecutionSchedule",
    "RetrievalExecutionStage",
    "TaskExecutionLedger",
    "TaskExecutionRecord",
    "ExecutionRouteKind",
    "TaskRunStatus",
    "build_retrieval_execution_schedule",
    "build_initial_retrieval_groups",
    "candidate_chunk_id",
    "candidate_identity",
    "candidate_source_identity",
    "sanitize_untrusted_task_metadata",
    "source_chunk_identity",
]
