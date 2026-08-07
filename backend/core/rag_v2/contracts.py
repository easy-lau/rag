"""Typed, side-effect-free contracts for the RAG v2 execution path.

These contracts deliberately separate evidence availability, confidence and
completeness.  A soft dependency failure may therefore mark a bundle as
``degraded`` without erasing otherwise authorized retrieval evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from core.query_constraints import ApplicabilityScope, extract_query_constraints
from core.rag_v2.evidence_snapshots import (
    complete_document_keys as _derived_complete_document_keys,
    complete_table_keys as _derived_complete_table_keys,
    table_key as _snapshot_table_key,
)


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
RequirementCoverageMode = Literal["single", "collection"]
CoverageContractKind = Literal[
    "single_claim",
    "structured_collection",
    "ordered_steps",
    "document_policy",
]
BridgeRequirementKind = Literal["classification", "condition", "mapping"]
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
EvidenceContributionKind = Literal[
    "answer_claim",
    "bridge_fact",
    "qualifier",
    "companion",
    "background",
    "conflicting",
]
ClaimApplicability = Literal[
    "bridge_value",
    "direct_subject",
    "document_universal",
    "section_inherited",
    "condition_bound",
]
ClaimProofKind = Literal["source_assertion", "terminology_strict"]
ClaimResultKind = Literal[
    "scalar",
    "categorical",
    "normative",
    "procedure",
    "config_assignment",
    "structured_collection",
    "ordered_steps",
    "document_policy",
]
BridgeEdgeMode = Literal["proof", "augmentation"]
CollectionClosureSourceKind = Literal[
    "full_document_snapshot",
    "complete_table",
    "source_declaration",
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
BRIDGE_REQUIREMENT_KINDS = frozenset({
    "classification",
    "condition",
    "mapping",
})
COVERAGE_CONTRACT_KINDS = frozenset({
    "single_claim",
    "structured_collection",
    "ordered_steps",
    "document_policy",
})
EVIDENCE_CONTRIBUTION_KINDS = frozenset({
    "answer_claim",
    "bridge_fact",
    "qualifier",
    "companion",
    "background",
    "conflicting",
})
CLAIM_APPLICABILITY_KINDS = frozenset({
    "bridge_value",
    "direct_subject",
    "document_universal",
    "section_inherited",
    "condition_bound",
})
CLAIM_PROOF_KINDS = frozenset({"source_assertion", "terminology_strict"})
# A result is deliberately a narrower contract than an arbitrary source
# annotation.  It is produced by the current request's answer-claim
# adjudicator and carried into ``EvidenceClaim``.  Scalar/category values and
# normalized configuration assignments are safe to compare for a mutually
# exclusive answer decision; procedural and normative text remains an
# independently closed claim but is not string-compared as a contradiction.
CLAIM_RESULT_KINDS = frozenset({
    "scalar",
    "categorical",
    "normative",
    "procedure",
    "config_assignment",
    "structured_collection",
    "ordered_steps",
    "document_policy",
})
CONFLICT_COMPARABLE_CLAIM_RESULT_KINDS = frozenset({
    "scalar",
    "categorical",
    "config_assignment",
})
BRIDGE_EDGE_MODES = frozenset({"proof", "augmentation"})
COLLECTION_CLOSURE_SOURCE_KINDS = frozenset({
    "full_document_snapshot",
    "complete_table",
    "source_declaration",
})

_REQUIREMENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_QUERY_CHARS = 1000
_MAX_REQUIREMENT_CHARS = 500
_MAX_REASON_CHARS = 500
_MAX_STATE_REASONS = 12
_MAX_GRAPH_ITEMS = 200


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


def _normalized_bridge_edge_ids(
    values: object,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Normalise one declared bridge-edge set without silently weakening it.

    Requirement references carried by evidence are intentionally de-duplicated
    because they are annotations.  A dependency declaration is different: a
    duplicate is a compiler defect, and accepting it would hide an ambiguous
    execution graph.  Keep the stricter rule local to the plan contract.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _normalized_text(raw, field_name=field_name, max_chars=64)
        if not _REQUIREMENT_ID_RE.fullmatch(value):
            raise ValueError(
                f"{field_name} must contain stable lowercase requirement ids"
            )
        if value in seen:
            raise ValueError(f"{field_name} cannot contain duplicate ids")
        seen.add(value)
        normalized.append(value)
        if len(normalized) > 8:
            raise ValueError(f"{field_name} has too many items")
    return tuple(normalized)


@dataclass(frozen=True)
class AnswerRequirementV2:
    id: str
    description: str
    role: RequirementRole = "answer"
    importance: RequirementImportance = "required"
    source: RequirementSource = "explicit"
    # ``single`` means one active claim can satisfy the requirement.
    # ``collection`` means the user asked for an exhaustive set (for example
    # all applicable rules in a policy).  Evidence assembly then keeps every
    # discovered active claim in the bounded, authorized snapshot and refuses
    # to call the answer complete when the renderer drops one of them.
    coverage_mode: RequirementCoverageMode = "single"
    # ``coverage_mode`` is the legacy cardinality hint.  ``coverage_contract``
    # is the executable closure rule used by the evidence graph.  Leaving it
    # unset preserves old serialized plans: ``single`` maps to ``single_claim``
    # and ``collection`` maps to ``structured_collection``.  New executable
    # plans must choose exactly one of the four closed semantics below:
    #
    # * ``single_claim``: one grounded claim answers one bounded lookup;
    # * ``structured_collection``: a finite, unordered member set;
    # * ``ordered_steps``: a source-authored ordered procedure;
    # * ``document_policy``: the governing policy/document as a whole.
    #
    # Evidence must execute this declaration, never infer it again from words
    # like ``配置``/``流程`` or from a source title.
    coverage_contract: CoverageContractKind | None = None
    # Machine-readable bridge semantics.  ``description`` is presentation and
    # retrieval text; it must never be parsed as the only source of dependency
    # truth because route models are free to paraphrase it.
    bridge_subject: str | None = None
    # The relation family is independently typed from ``description``.  A
    # classification bridge, for example, permits a table resolver to look for
    # a taxonomy column plus an applicable-entity column without guessing that
    # a particular display sentence happens to mention "职级".  ``None`` is
    # retained only for legacy plans; all newly compiled executable plans must
    # carry one of the closed enum values below.
    bridge_kind: BridgeRequirementKind | None = None
    # Proof edges are a hard evidence contract: an answer using such an edge
    # cannot claim bridge-value applicability without the named bridge fact.
    # ``None`` is reserved for legacy/unbound requirements.  An empty tuple is
    # an explicit statement that this answer has no proof bridge dependency.
    # Keeping those states distinct prevents an independent answer in a
    # multi-part query from being silently attached to every bridge in a plan.
    depends_on_requirement_ids: tuple[str, ...] | None = None
    # Optional bridge augmentation is deliberately a separate edge family.
    # A resolved fact may release a more precise second-hop query, but absence
    # of that fact cannot block the answer's direct retrieval/evidence route.
    # ``None`` means an older/unbound compiler did not make an augmentation
    # decision; it must never be inferred at runtime.  ``()`` explicitly says
    # that no optional bridge augmentation exists for this answer.
    augmentation_requirement_ids: tuple[str, ...] | None = None
    # Applicability is requirement-local.  A multi-part turn may ask one
    # question about 8.2 and another about 8.6; compiling only one global query
    # constraint would necessarily contaminate a sibling requirement.  This is
    # the sole canonical representation.  It retains source spans for product,
    # version and project, so a later planner/model rewrite cannot manufacture
    # a project boundary or silently erase one from the current question.
    applicability_scope: ApplicabilityScope | None = None
    # The three scalar fields below are a read-compatible construction and
    # serialization projection for plans persisted before ``ApplicabilityScope``
    # existed.  New callers must pass ``applicability_scope``.  ``__post_init__``
    # always re-derives them from the canonical object and rejects divergence.
    scope_product: str | None = None
    scope_version: str | None = None
    scope_explicit_version: bool = False

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
        if self.coverage_mode not in {"single", "collection"}:
            raise ValueError("requirement coverage mode must be single or collection")
        if self.role == "bridge" and self.coverage_mode != "single":
            raise ValueError("bridge requirements must use single coverage")
        coverage_contract = self.coverage_contract
        if coverage_contract is not None:
            if not isinstance(coverage_contract, str):
                raise ValueError("coverage contract must be a string")
            coverage_contract = coverage_contract.strip().casefold()
            if coverage_contract not in COVERAGE_CONTRACT_KINDS:
                raise ValueError("coverage contract is not supported")
            if self.role != "answer":
                raise ValueError("only answer requirements can define coverage contracts")
            if (
                coverage_contract == "single_claim"
                and self.coverage_mode != "single"
            ):
                raise ValueError(
                    "single_claim coverage contract requires single coverage mode"
                )
            if (
                coverage_contract
                in {
                    "structured_collection",
                    "ordered_steps",
                    "document_policy",
                }
                and self.coverage_mode != "collection"
            ):
                raise ValueError(
                    "collection coverage contracts require collection coverage mode"
                )
        description = _normalized_text(
            self.description,
            field_name="requirement description",
            max_chars=_MAX_REQUIREMENT_CHARS,
        )
        bridge_subject = self.bridge_subject
        if bridge_subject is not None:
            bridge_subject = _normalized_text(
                bridge_subject,
                field_name="bridge subject",
                max_chars=_MAX_REQUIREMENT_CHARS,
            )
        bridge_kind = self.bridge_kind
        if bridge_kind is not None:
            if not isinstance(bridge_kind, str):
                raise ValueError("bridge kind must be a string")
            bridge_kind = bridge_kind.strip().casefold()
            if bridge_kind not in BRIDGE_REQUIREMENT_KINDS:
                raise ValueError("bridge kind is not supported")
        dependency_ids = (
            None
            if self.depends_on_requirement_ids is None
            else _normalized_bridge_edge_ids(
                self.depends_on_requirement_ids,
                field_name="proof bridge dependency ids",
            )
        )
        augmentation_ids = (
            None
            if self.augmentation_requirement_ids is None
            else _normalized_bridge_edge_ids(
                self.augmentation_requirement_ids,
                field_name="bridge augmentation ids",
            )
        )
        if self.role == "answer" and bridge_subject is not None:
            raise ValueError("answer requirements cannot define a bridge subject")
        if self.role == "answer" and bridge_kind is not None:
            raise ValueError("answer requirements cannot define a bridge kind")
        if self.role == "bridge" and dependency_ids is not None:
            raise ValueError(
                "bridge requirements cannot define dependencies (proof edge)"
            )
        if self.role == "bridge" and augmentation_ids is not None:
            raise ValueError("bridge requirements cannot define augmentation dependencies")
        if set(dependency_ids or ()) & set(augmentation_ids or ()):
            raise ValueError(
                "proof and augmentation bridge dependencies cannot overlap"
            )
        legacy_scope_product = (
            re.sub(r"\s+", " ", str(self.scope_product or "")).strip()
            or None
        )
        legacy_scope_version = (
            re.sub(r"\s+", " ", str(self.scope_version or "")).strip()
            or None
        )
        legacy_explicit_version = bool(self.scope_explicit_version)
        if legacy_explicit_version and legacy_scope_version is None:
            raise ValueError("scope_explicit_version requires scope_version")

        supplied_scope = self.applicability_scope
        if supplied_scope is not None and not isinstance(
            supplied_scope,
            ApplicabilityScope,
        ):
            raise ValueError("applicability_scope must be an ApplicabilityScope")

        if supplied_scope is None:
            if legacy_scope_product is None and legacy_scope_version is None:
                canonical_scope = extract_query_constraints(description)
            else:
                canonical_scope = ApplicabilityScope(
                    product=legacy_scope_product,
                    version=legacy_scope_version,
                    explicit_version=legacy_explicit_version,
                    extraction_reason="legacy_requirement_scope_projection",
                )
        else:
            canonical_scope = supplied_scope
            if (
                legacy_scope_product is not None
                and legacy_scope_product != canonical_scope.product
            ):
                raise ValueError(
                    "scope_product conflicts with applicability_scope"
                )
            if (
                legacy_scope_version is not None
                and legacy_scope_version != canonical_scope.version
            ):
                raise ValueError(
                    "scope_version conflicts with applicability_scope"
                )
            if (
                legacy_explicit_version
                and not canonical_scope.explicit_version
            ):
                raise ValueError(
                    "scope_explicit_version conflicts with applicability_scope"
                )
        object.__setattr__(self, "id", requirement_id)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "coverage_contract", coverage_contract)
        object.__setattr__(self, "bridge_subject", bridge_subject)
        object.__setattr__(self, "bridge_kind", bridge_kind)
        object.__setattr__(self, "depends_on_requirement_ids", dependency_ids)
        object.__setattr__(self, "augmentation_requirement_ids", augmentation_ids)
        object.__setattr__(self, "applicability_scope", canonical_scope)
        object.__setattr__(self, "scope_product", canonical_scope.product)
        object.__setattr__(self, "scope_version", canonical_scope.version)
        object.__setattr__(
            self,
            "scope_explicit_version",
            canonical_scope.explicit_version,
        )

    @property
    def is_required_answer(self) -> bool:
        return self.role == "answer" and self.importance == "required"

    @property
    def scope_project(self) -> str | None:
        """Compatibility projection of the canonical project boundary."""

        return self.applicability_scope.project if self.applicability_scope else None

    @property
    def scope_explicit_project(self) -> bool:
        return bool(
            self.applicability_scope
            and self.applicability_scope.explicit_project
        )

    @property
    def scope_project_source(self):
        """Return the source-authored project span, never model text."""

        return (
            self.applicability_scope.project_source
            if self.applicability_scope
            else None
        )

    @property
    def scope_fingerprint(self) -> str:
        """Stable identity used to keep task/group scopes non-interchangeable."""

        return (
            self.applicability_scope.fingerprint
            if self.applicability_scope
            else ApplicabilityScope().fingerprint
        )

    @property
    def proof_bridge_requirement_ids(self) -> tuple[str, ...]:
        """Return declared proof parents, never inferring a legacy edge."""

        return self.depends_on_requirement_ids or ()

    @property
    def augmentation_bridge_requirement_ids(self) -> tuple[str, ...]:
        """Return declared optional augmentation parents, never inferring one."""

        return self.augmentation_requirement_ids or ()

    @property
    def effective_coverage_contract(self) -> CoverageContractKind:
        """Return the proof rule without mutating legacy serialized plans."""

        if self.coverage_contract is not None:
            return self.coverage_contract
        return (
            "structured_collection"
            if self.coverage_mode == "collection"
            else "single_claim"
        )

    @property
    def requires_collection_closure(self) -> bool:
        """Whether this answer needs exhaustive/ordered closure proof.

        This is deliberately derived from the explicit executable contract,
        not from query wording or the legacy cardinality hint.  It gives every
        evidence stage one shared answer to "may a table/list close this?".
        """

        return self.effective_coverage_contract in {
            "structured_collection",
            "ordered_steps",
            "document_policy",
        }

    @property
    def requires_document_policy_snapshot(self) -> bool:
        """Whether completion requires the rooted whole-document snapshot."""

        return self.effective_coverage_contract == "document_policy"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "description": self.description,
            "role": self.role,
            "importance": self.importance,
            "source": self.source,
            "coverage_mode": self.coverage_mode,
        }
        if self.coverage_contract is not None:
            result["coverage_contract"] = self.coverage_contract
        if self.bridge_subject is not None:
            result["bridge_subject"] = self.bridge_subject
        if self.bridge_kind is not None:
            result["bridge_kind"] = self.bridge_kind
        if self.depends_on_requirement_ids is not None:
            result["depends_on_requirement_ids"] = list(
                self.depends_on_requirement_ids
            )
        if self.augmentation_requirement_ids is not None:
            result["augmentation_requirement_ids"] = list(
                self.augmentation_requirement_ids
            )
        if (
            self.applicability_scope is not None
            and self.applicability_scope.has_scope_constraint
        ):
            result["scope"] = self.applicability_scope.as_dict()
        return result


def validate_answer_requirement_graph(
    requirements: tuple[AnswerRequirementV2, ...],
    *,
    require_explicit_answer_dependencies: bool = False,
    require_referenced_bridges: bool = False,
) -> None:
    """Validate proof and optional answer-to-bridge edges without inference.

    This is intentionally a graph validator, not a natural-language fallback.
    Callers that compile executable plans must bind every answer explicitly and
    must remove dangling bridge hints before this boundary.
    """

    requirement_by_id = {item.id: item for item in requirements}
    referenced_bridge_ids: set[str] = set()
    scoped_answer_consumers: dict[str, list[AnswerRequirementV2]] = {}
    for requirement in requirements:
        if requirement.role == "answer":
            proof_ids = requirement.depends_on_requirement_ids
            augmentation_ids = requirement.augmentation_requirement_ids
            if proof_ids is None:
                if require_explicit_answer_dependencies:
                    raise ValueError(
                        "answer requirements must explicitly declare proof bridge dependencies"
                    )
                proof_ids = ()
            if set(proof_ids) & set(augmentation_ids or ()):
                # The dataclass rejects this too, but retain the graph-level
                # guard for defensive callers / future deserializers.
                raise ValueError(
                    "proof and augmentation bridge dependencies cannot overlap"
                )
            for edge_name, dependency_ids in (
                ("proof", proof_ids),
                ("augmentation", augmentation_ids or ()),
            ):
                for dependency_id in dependency_ids:
                    dependency = requirement_by_id.get(dependency_id)
                    if dependency is None:
                        raise ValueError(
                            f"query plan {edge_name} bridge dependency does not exist"
                        )
                    if dependency.role != "bridge":
                        raise ValueError(
                            f"answer requirements may {edge_name}-depend only on bridge requirements"
                        )
                    referenced_bridge_ids.add(dependency_id)
                    if (
                        requirement.applicability_scope is not None
                        and requirement.applicability_scope.has_scope_constraint
                    ):
                        scoped_answer_consumers.setdefault(
                            dependency_id,
                            [],
                        ).append(requirement)

    if require_referenced_bridges:
        bridge_ids = {
            requirement.id
            for requirement in requirements
            if requirement.role == "bridge"
        }
        if dangling := bridge_ids - referenced_bridge_ids:
            raise ValueError(
                "query plan contains unreferenced bridge requirements: "
                + ", ".join(sorted(dangling))
            )

        # A bridge fact is an execution prerequisite/augmentation for the
        # consuming answer, not a free-floating recall result.  A scoped
        # answer may therefore never consume an unscoped bridge, and one
        # bridge node cannot span mutually-exclusive versions/projects.  The
        # compiler is responsible for emitting separate bridge requirements
        # for separate scopes; this validator only refuses an unsafe graph.
        def scope_identity(scope: ApplicabilityScope) -> tuple[object, ...]:
            return (
                re.sub(r"\s+", "", str(scope.product or "")).casefold(),
                re.sub(r"\s+", "", str(scope.version or "")).casefold(),
                (
                    re.sub(r"\s+", "", str(scope.project or "")).casefold()
                    if scope.has_project_constraint
                    else ""
                ),
                bool(scope.has_product_constraint),
                bool(scope.has_version_constraint),
                bool(scope.has_project_constraint),
            )

        for bridge_id, consumers in scoped_answer_consumers.items():
            expected_scopes = {
                scope_identity(item.applicability_scope)
                for item in consumers
                if item.applicability_scope is not None
            }
            if len(expected_scopes) != 1:
                raise ValueError(
                    "one bridge requirement cannot serve answers with different applicability scopes"
                )
            bridge = requirement_by_id[bridge_id]
            bridge_scope = bridge.applicability_scope
            if (
                bridge_scope is None
                or not bridge_scope.has_scope_constraint
                or scope_identity(bridge_scope) not in expected_scopes
            ):
                raise ValueError(
                    "scoped answer bridge must carry the same applicability scope"
                )


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
        validate_answer_requirement_graph(requirements)
        for requirement in requirements:
            if requirement.role == "bridge" and not requirement.bridge_subject:
                raise ValueError(
                    "query plan bridge requirements require a canonical subject"
                )

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
        if ready_shape:
            validate_answer_requirement_graph(
                requirements,
                require_explicit_answer_dependencies=True,
                require_referenced_bridges=True,
            )
        if self.answer_shape == "multi_hop" and not any(
            item.role == "answer" and item.proof_bridge_requirement_ids
            for item in requirements
        ):
            raise ValueError(
                "multi_hop query plans require an answer-to-bridge dependency with proof semantics"
            )

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

    @property
    def has_bridge_dependencies(self) -> bool:
        """Whether evidence requires at least one answer-to-bridge proof edge."""

        return any(
            item.role == "answer" and bool(item.proof_bridge_requirement_ids)
            for item in self.requirements
        )

    @property
    def has_bridge_augmentations(self) -> bool:
        """Whether any answer declares a non-blocking bridge enhancement."""

        return any(
            item.role == "answer" and bool(item.augmentation_bridge_requirement_ids)
            for item in self.requirements
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
            "has_bridge_dependencies": self.has_bridge_dependencies,
            "has_bridge_augmentations": self.has_bridge_augmentations,
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
    # ``role`` remains the legacy renderer-facing classification.  This field
    # carries claim semantics for the evidence graph and deliberately does not
    # reinterpret an existing role.  ``None`` is the compatibility state: the
    # graph may derive only a conservative role-based default.
    contribution_kind: EvidenceContributionKind | None = None
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
        contribution_kind = self.contribution_kind
        if contribution_kind is not None:
            if not isinstance(contribution_kind, str):
                raise ValueError("evidence contribution kind must be a string")
            contribution_kind = contribution_kind.strip().casefold()
            if contribution_kind not in EVIDENCE_CONTRIBUTION_KINDS:
                raise ValueError("unsupported evidence contribution kind")
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
        object.__setattr__(self, "contribution_kind", contribution_kind)
        object.__setattr__(
            self,
            "supports_requirement_ids",
            supports_requirement_ids,
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
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
        if self.contribution_kind is not None:
            result["contribution_kind"] = self.contribution_kind
        return result


DocumentKey = tuple[str, str]


def _normalized_graph_item_ids(
    values: object,
    *,
    field_name: str,
    max_items: int = _MAX_GRAPH_ITEMS,
) -> tuple[str, ...]:
    return _normalized_unique_texts(
        values,
        field_name=field_name,
        max_items=max_items,
        max_chars=200,
    )


def _normalized_document_key(value: object, *, field_name: str) -> DocumentKey:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a (kb_id, doc_id) tuple")
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly kb_id and doc_id")
    return (
        _normalized_text(value[0], field_name=f"{field_name}.kb_id", max_chars=200),
        _normalized_text(value[1], field_name=f"{field_name}.doc_id", max_chars=200),
    )


def _normalized_document_keys(
    values: object,
    *,
    field_name: str,
    max_items: int = _MAX_GRAPH_ITEMS,
) -> tuple[DocumentKey, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{field_name} must be a sequence")
    result: list[DocumentKey] = []
    seen: set[DocumentKey] = set()
    for value in values:
        document_key = _normalized_document_key(value, field_name=field_name)
        if document_key in seen:
            continue
        seen.add(document_key)
        result.append(document_key)
        if len(result) > max_items:
            raise ValueError(f"{field_name} has too many items")
    return tuple(result)


@dataclass(frozen=True)
class BridgeClaimBinding:
    """A verified bridge fact consumed by one bridge-value answer claim.

    The binding intentionally names the exact source chunk.  A bare bridge
    requirement id is insufficient: it would let an answer claim join to an
    unrelated mapping found in a different document or a previous execution.
    """

    bridge_requirement_id: str
    bridge_source_item_id: str
    bridge_value: str
    # The edge is part of the proof contract.  A fact used to refine recall
    # (augmentation) is not interchangeable with a hard proof prerequisite.
    edge_mode: BridgeEdgeMode = "proof"
    # These ids are request-local execution provenance.  They remain optional
    # for standalone graph construction/backward-compatible diagnostic
    # callers, but production assembly always supplies both after the ledger
    # has verified the complete route.
    bridge_execution_id: str | None = None
    answer_execution_id: str | None = None

    def __post_init__(self) -> None:
        bridge_requirement_id = _normalized_requirement_ids(
            (self.bridge_requirement_id,),
            field_name="bridge claim binding requirement id",
            max_items=1,
        )[0]
        bridge_source_item_id = _normalized_text(
            self.bridge_source_item_id,
            field_name="bridge claim binding source item id",
            max_chars=200,
        )
        bridge_value = _normalized_text(
            self.bridge_value,
            field_name="bridge claim binding value",
            max_chars=_MAX_REQUIREMENT_CHARS,
        )
        edge_mode = str(self.edge_mode or "").strip().casefold()
        if edge_mode not in BRIDGE_EDGE_MODES:
            raise ValueError("bridge claim binding edge mode is not supported")
        bridge_execution_id = (
            None
            if self.bridge_execution_id is None
            else _normalized_text(
                self.bridge_execution_id,
                field_name="bridge claim binding bridge execution id",
                max_chars=200,
            )
        )
        answer_execution_id = (
            None
            if self.answer_execution_id is None
            else _normalized_text(
                self.answer_execution_id,
                field_name="bridge claim binding answer execution id",
                max_chars=200,
            )
        )
        object.__setattr__(self, "bridge_requirement_id", bridge_requirement_id)
        object.__setattr__(self, "bridge_source_item_id", bridge_source_item_id)
        object.__setattr__(self, "bridge_value", bridge_value)
        object.__setattr__(self, "edge_mode", edge_mode)
        object.__setattr__(self, "bridge_execution_id", bridge_execution_id)
        object.__setattr__(self, "answer_execution_id", answer_execution_id)

    def to_dict(self) -> dict[str, str]:
        result = {
            "bridge_requirement_id": self.bridge_requirement_id,
            "bridge_source_item_id": self.bridge_source_item_id,
            "bridge_value": self.bridge_value,
            "edge_mode": self.edge_mode,
        }
        if self.bridge_execution_id is not None:
            result["bridge_execution_id"] = self.bridge_execution_id
        if self.answer_execution_id is not None:
            result["answer_execution_id"] = self.answer_execution_id
        return result


@dataclass(frozen=True)
class EvidenceClaim:
    """One typed assertion made by one evidence item.

    This is deliberately separate from :class:`EvidenceItem.role`.  The role
    tells the current renderer how a chunk was selected; a claim records the
    narrower proposition that may count toward one requirement.  A terminology
    normalizer may supply ``terminology_strict`` only with stable registry rule
    ids.  Fuzzy lexical similarity is not a proof kind and cannot enter this
    contract.
    """

    id: str
    requirement_id: str
    evidence_item_id: str
    document_key: DocumentKey
    contribution_kind: EvidenceContributionKind
    applicability: ClaimApplicability | None = None
    proof_kind: ClaimProofKind = "source_assertion"
    strict_terminology_rule_ids: tuple[str, ...] = ()
    # Claim semantics are emitted by the request-local adjudicator, not
    # re-derived by coverage from chunk text or metadata.  The optional shape
    # preserves compatibility for hand-built structural graph tests while
    # production ledgered claims always carry all three fields together.
    result_kind: ClaimResultKind | None = None
    normalized_result: str | None = None
    claim_key: str | None = None
    structural_group_id: str | None = None
    bridge_bindings: tuple[BridgeClaimBinding, ...] = ()
    condition_group_id: str | None = None

    def __post_init__(self) -> None:
        claim_id = _normalized_text(self.id, field_name="evidence claim id", max_chars=300)
        requirement_id = _normalized_requirement_ids(
            (self.requirement_id,),
            field_name="evidence claim requirement id",
            max_items=1,
        )[0]
        evidence_item_id = _normalized_text(
            self.evidence_item_id,
            field_name="evidence claim item id",
            max_chars=200,
        )
        document_key = _normalized_document_key(
            self.document_key,
            field_name="evidence claim document key",
        )
        contribution_kind = str(self.contribution_kind or "").strip().casefold()
        if contribution_kind not in EVIDENCE_CONTRIBUTION_KINDS:
            raise ValueError("unsupported evidence claim contribution kind")
        applicability = self.applicability
        if applicability is not None:
            if not isinstance(applicability, str):
                raise ValueError("evidence claim applicability must be a string")
            applicability = applicability.strip().casefold()
            if applicability not in CLAIM_APPLICABILITY_KINDS:
                raise ValueError("unsupported evidence claim applicability")
        if contribution_kind == "answer_claim" and applicability is None:
            raise ValueError("answer claims require an applicability")
        if contribution_kind == "bridge_fact" and applicability not in {None, "bridge_value"}:
            raise ValueError("bridge facts can only use bridge_value applicability")
        proof_kind = str(self.proof_kind or "").strip().casefold()
        if proof_kind not in CLAIM_PROOF_KINDS:
            raise ValueError("unsupported evidence claim proof kind")
        strict_rule_ids = _normalized_unique_texts(
            self.strict_terminology_rule_ids,
            field_name="strict terminology rule id",
            max_items=24,
            max_chars=120,
        )
        if proof_kind == "terminology_strict" and not strict_rule_ids:
            raise ValueError(
                "terminology_strict evidence claims require terminology rule ids"
            )
        if proof_kind != "terminology_strict" and strict_rule_ids:
            raise ValueError(
                "only terminology_strict evidence claims may carry terminology rule ids"
            )
        result_kind = (
            None
            if self.result_kind is None
            else _normalized_text(
                self.result_kind,
                field_name="evidence claim result kind",
                max_chars=80,
            ).casefold()
        )
        normalized_result = (
            None
            if self.normalized_result is None
            else _normalized_text(
                self.normalized_result,
                field_name="evidence claim normalized result",
                max_chars=_MAX_REQUIREMENT_CHARS,
            ).casefold()
        )
        claim_key = (
            None
            if self.claim_key is None
            else _normalized_text(
                self.claim_key,
                field_name="evidence claim semantic key",
                max_chars=_MAX_REQUIREMENT_CHARS,
            ).casefold()
        )
        semantic_fields = (result_kind, normalized_result, claim_key)
        if any(value is not None for value in semantic_fields):
            if contribution_kind != "answer_claim":
                raise ValueError(
                    "only answer claims may carry semantic result fields"
                )
            if any(value is None for value in semantic_fields):
                raise ValueError(
                    "evidence claim semantic result fields must be supplied together"
                )
            if result_kind not in CLAIM_RESULT_KINDS:
                raise ValueError("unsupported evidence claim result kind")
        structural_group_id = (
            None
            if self.structural_group_id is None
            else _normalized_text(
                self.structural_group_id,
                field_name="evidence claim structural group id",
                max_chars=300,
            )
        )
        bridge_bindings = tuple(self.bridge_bindings)
        if any(not isinstance(value, BridgeClaimBinding) for value in bridge_bindings):
            raise ValueError("bridge_bindings must contain BridgeClaimBinding values")
        binding_keys = {
            (
                value.bridge_requirement_id,
                value.bridge_source_item_id,
                value.bridge_value,
                value.edge_mode,
                value.bridge_execution_id,
                value.answer_execution_id,
            )
            for value in bridge_bindings
        }
        if len(binding_keys) != len(bridge_bindings):
            raise ValueError("evidence claim contains duplicate bridge bindings")
        if applicability == "bridge_value" and contribution_kind == "answer_claim" and not bridge_bindings:
            raise ValueError("bridge-value answer claims require bridge bindings")
        if applicability != "bridge_value" and bridge_bindings:
            raise ValueError("only bridge-value claims may carry bridge bindings")
        condition_group_id = (
            None
            if self.condition_group_id is None
            else _normalized_text(
                self.condition_group_id,
                field_name="evidence claim condition group id",
                max_chars=300,
            )
        )
        if applicability != "condition_bound" and condition_group_id is not None:
            raise ValueError("only condition-bound claims may carry a condition group id")

        object.__setattr__(self, "id", claim_id)
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "evidence_item_id", evidence_item_id)
        object.__setattr__(self, "document_key", document_key)
        object.__setattr__(self, "contribution_kind", contribution_kind)
        object.__setattr__(self, "applicability", applicability)
        object.__setattr__(self, "proof_kind", proof_kind)
        object.__setattr__(self, "strict_terminology_rule_ids", strict_rule_ids)
        object.__setattr__(self, "result_kind", result_kind)
        object.__setattr__(self, "normalized_result", normalized_result)
        object.__setattr__(self, "claim_key", claim_key)
        object.__setattr__(self, "structural_group_id", structural_group_id)
        object.__setattr__(self, "bridge_bindings", bridge_bindings)
        object.__setattr__(self, "condition_group_id", condition_group_id)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "requirement_id": self.requirement_id,
            "evidence_item_id": self.evidence_item_id,
            "document_key": list(self.document_key),
            "contribution_kind": self.contribution_kind,
            "applicability": self.applicability,
            "proof_kind": self.proof_kind,
            "strict_terminology_rule_ids": list(self.strict_terminology_rule_ids),
            "bridge_bindings": [value.to_dict() for value in self.bridge_bindings],
        }
        if self.structural_group_id is not None:
            result["structural_group_id"] = self.structural_group_id
        if self.condition_group_id is not None:
            result["condition_group_id"] = self.condition_group_id
        if self.result_kind is not None:
            result["result_kind"] = self.result_kind
            result["normalized_result"] = self.normalized_result
            result["claim_key"] = self.claim_key
        return result


@dataclass(frozen=True)
class StructuralEvidenceGroup:
    """A structural unit whose qualifiers must travel with its primary claim."""

    id: str
    document_key: DocumentKey
    member_item_ids: tuple[str, ...]
    primary_item_ids: tuple[str, ...] = ()
    companion_item_ids: tuple[str, ...] = ()
    qualifier_item_ids: tuple[str, ...] = ()
    # Conditions may live in another section but only enter this tuple through
    # an explicit parser/evidence link.  They are intentionally not required to
    # be local ``member_item_ids``.
    condition_item_ids: tuple[str, ...] = ()
    section_key: str | None = None
    table_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        group_id = _normalized_text(self.id, field_name="structural group id", max_chars=300)
        document_key = _normalized_document_key(
            self.document_key,
            field_name="structural group document key",
        )
        member_item_ids = _normalized_graph_item_ids(
            self.member_item_ids,
            field_name="structural group member item id",
        )
        if not member_item_ids:
            raise ValueError("structural evidence groups require at least one member")
        primary_item_ids = _normalized_graph_item_ids(
            self.primary_item_ids,
            field_name="structural group primary item id",
        )
        companion_item_ids = _normalized_graph_item_ids(
            self.companion_item_ids,
            field_name="structural group companion item id",
        )
        qualifier_item_ids = _normalized_graph_item_ids(
            self.qualifier_item_ids,
            field_name="structural group qualifier item id",
        )
        condition_item_ids = _normalized_graph_item_ids(
            self.condition_item_ids,
            field_name="structural group condition item id",
        )
        member_ids = set(member_item_ids)
        for label, values in (
            ("primary", primary_item_ids),
            ("companion", companion_item_ids),
            ("qualifier", qualifier_item_ids),
        ):
            if not set(values).issubset(member_ids):
                raise ValueError(
                    f"structural group {label} item ids must be local members"
                )
        if set(primary_item_ids) & set(companion_item_ids):
            raise ValueError("structural group primary and companion items overlap")
        if set(primary_item_ids) & set(qualifier_item_ids):
            raise ValueError("structural group primary and qualifier items overlap")
        section_key = (
            None
            if self.section_key is None
            else _normalized_text(
                self.section_key,
                field_name="structural group section key",
                max_chars=300,
            )
        )
        table_keys = _normalized_unique_texts(
            self.table_keys,
            field_name="structural group table key",
            max_items=32,
            max_chars=300,
        )

        object.__setattr__(self, "id", group_id)
        object.__setattr__(self, "document_key", document_key)
        object.__setattr__(self, "member_item_ids", member_item_ids)
        object.__setattr__(self, "primary_item_ids", primary_item_ids)
        object.__setattr__(self, "companion_item_ids", companion_item_ids)
        object.__setattr__(self, "qualifier_item_ids", qualifier_item_ids)
        object.__setattr__(self, "condition_item_ids", condition_item_ids)
        object.__setattr__(self, "section_key", section_key)
        object.__setattr__(self, "table_keys", table_keys)

    @property
    def required_item_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            (*self.companion_item_ids, *self.qualifier_item_ids, *self.condition_item_ids)
        ))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "document_key": list(self.document_key),
            "member_item_ids": list(self.member_item_ids),
            "primary_item_ids": list(self.primary_item_ids),
            "companion_item_ids": list(self.companion_item_ids),
            "qualifier_item_ids": list(self.qualifier_item_ids),
            "condition_item_ids": list(self.condition_item_ids),
            "table_keys": list(self.table_keys),
        }
        if self.section_key is not None:
            result["section_key"] = self.section_key
        return result


@dataclass(frozen=True)
class VerifiedCollectionClosure:
    """A source-verified, requirement-scoped exhaustive collection proof.

    Collection completeness is a statement about the *question*, rather than
    about every byte of a source document.  A full-document expansion may
    contain an appendix that does not answer the requirement, while a parsed
    table or an authored ``only includes`` declaration may precisely bound the
    answer by itself.  This certificate records that upstream structural
    verification before the renderer budget discards non-answer material.

    It is deliberately not a free-form confidence hint.  The graph validates
    that every listed item is an actual positive answer source for the named
    collection requirement, and its assessor still requires all of those
    claims to be visible and structurally closed.  Thus it cannot promote a
    partial snapshot to complete merely by being present.
    """

    requirement_id: str
    claim_item_ids: tuple[str, ...]
    source_kind: CollectionClosureSourceKind
    source_document_key: DocumentKey
    source_table_key: str | None = None

    def __post_init__(self) -> None:
        requirement_id = _normalized_requirement_ids(
            (self.requirement_id,),
            field_name="collection closure requirement id",
            max_items=1,
        )[0]
        claim_item_ids = _normalized_graph_item_ids(
            self.claim_item_ids,
            field_name="collection closure claim item id",
        )
        if not claim_item_ids:
            raise ValueError("collection closure requires at least one claim item")
        source_kind = _normalized_text(
            self.source_kind,
            field_name="collection closure source kind",
            max_chars=64,
        ).casefold()
        if source_kind not in COLLECTION_CLOSURE_SOURCE_KINDS:
            raise ValueError("collection closure source kind is not supported")
        source_document_key = _normalized_document_key(
            self.source_document_key,
            field_name="collection closure source document key",
        )
        source_table_key = self.source_table_key
        if source_table_key is not None:
            source_table_key = _normalized_text(
                source_table_key,
                field_name="collection closure source table key",
                max_chars=300,
            )
        if source_kind == "complete_table":
            if source_table_key is None:
                raise ValueError("complete-table collection closure requires a table key")
        elif source_table_key is not None:
            raise ValueError("only a complete-table collection closure can carry a table key")
        if source_kind == "source_declaration" and len(claim_item_ids) != 1:
            raise ValueError(
                "a source-declaration collection closure requires exactly one claim"
            )
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "claim_item_ids", claim_item_ids)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_document_key", source_document_key)
        object.__setattr__(self, "source_table_key", source_table_key)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requirement_id": self.requirement_id,
            "claim_item_ids": list(self.claim_item_ids),
            "source_kind": self.source_kind,
            "source_document_key": list(self.source_document_key),
        }
        if self.source_table_key is not None:
            result["source_table_key"] = self.source_table_key
        return result


@dataclass(frozen=True)
class EvidenceCoverageGraph:
    """Immutable, context-bounded proof graph for typed RAG evidence."""

    requirements: tuple[AnswerRequirementV2, ...]
    evidence_item_ids: tuple[str, ...]
    visible_evidence_item_ids: tuple[str, ...]
    evidence_document_keys: Mapping[str, DocumentKey]
    # The graph carries the immutable source items it certifies.  Retaining
    # them avoids a second, caller-provided "complete snapshot" channel: all
    # document/table closure facts are recomputed from these exact items.
    evidence_items: tuple[EvidenceItem, ...]
    claims: tuple[EvidenceClaim, ...] = ()
    structural_groups: tuple[StructuralEvidenceGroup, ...] = ()
    document_root_keys: Mapping[str, DocumentKey] = field(default_factory=dict)
    complete_document_keys: tuple[DocumentKey, ...] = ()
    complete_table_keys: tuple[str, ...] = ()
    collection_closures: tuple[VerifiedCollectionClosure, ...] = ()
    schema_version: str = "evidence_coverage_graph.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evidence_coverage_graph.v1":
            raise ValueError("unsupported evidence coverage graph schema version")
        requirements = tuple(self.requirements)
        if any(not isinstance(value, AnswerRequirementV2) for value in requirements):
            raise ValueError("graph requirements must contain AnswerRequirementV2 values")
        requirement_ids = tuple(value.id for value in requirements)
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("evidence coverage graph contains duplicate requirements")
        validate_answer_requirement_graph(requirements)
        requirement_by_id = {value.id: value for value in requirements}
        evidence_item_ids = _normalized_graph_item_ids(
            self.evidence_item_ids,
            field_name="evidence graph item id",
        )
        if len(evidence_item_ids) != len(set(evidence_item_ids)):
            raise ValueError("evidence coverage graph contains duplicate item ids")
        visible_evidence_item_ids = _normalized_graph_item_ids(
            self.visible_evidence_item_ids,
            field_name="visible evidence graph item id",
        )
        item_ids = set(evidence_item_ids)
        if not set(visible_evidence_item_ids).issubset(item_ids):
            raise ValueError("visible evidence items must belong to the graph")
        evidence_items = tuple(self.evidence_items)
        if any(not isinstance(value, EvidenceItem) for value in evidence_items):
            raise ValueError("evidence graph items must contain EvidenceItem values")
        evidence_items_by_id = {item.chunk_id: item for item in evidence_items}
        if len(evidence_items_by_id) != len(evidence_items):
            raise ValueError("evidence graph items contain duplicate chunk ids")
        if set(evidence_items_by_id) != item_ids:
            raise ValueError("evidence graph items must exactly match item ids")
        if not isinstance(self.evidence_document_keys, Mapping):
            raise ValueError("evidence_document_keys must be a mapping")
        document_keys: dict[str, DocumentKey] = {}
        for item_id, document_key in self.evidence_document_keys.items():
            normalized_item_id = _normalized_text(
                item_id,
                field_name="evidence graph document item id",
                max_chars=200,
            )
            if normalized_item_id not in item_ids:
                raise ValueError("evidence graph document key references an unknown item")
            document_keys[normalized_item_id] = _normalized_document_key(
                document_key,
                field_name="evidence graph document key",
            )
        if set(document_keys) != item_ids:
            raise ValueError("every evidence graph item requires a document key")
        if any(
            document_keys[item_id] != (item.kb_id, item.doc_id)
            for item_id, item in evidence_items_by_id.items()
        ):
            raise ValueError("evidence graph document keys must match evidence items")

        claims = tuple(self.claims)
        if any(not isinstance(value, EvidenceClaim) for value in claims):
            raise ValueError("graph claims must contain EvidenceClaim values")
        if len({value.id for value in claims}) != len(claims):
            raise ValueError("evidence coverage graph contains duplicate claim ids")
        for claim in claims:
            requirement = requirement_by_id.get(claim.requirement_id)
            if requirement is None:
                raise ValueError("evidence claim references an unknown requirement")
            if claim.evidence_item_id not in item_ids:
                raise ValueError("evidence claim references an unknown evidence item")
            if claim.document_key != document_keys[claim.evidence_item_id]:
                raise ValueError("evidence claim document key must match its evidence item")
            if (
                claim.contribution_kind == "answer_claim"
                and requirement.role != "answer"
            ):
                raise ValueError("answer claims must target answer requirements")
            if (
                claim.contribution_kind == "bridge_fact"
                and requirement.role != "bridge"
            ):
                raise ValueError("bridge facts must target bridge requirements")
            proof_dependency_ids = set(
                requirement.proof_bridge_requirement_ids
            )
            augmentation_dependency_ids = set(
                requirement.augmentation_bridge_requirement_ids
            )
            if claim.applicability == "bridge_value" and claim.contribution_kind == "answer_claim":
                binding_ids_by_mode = {
                    mode: {
                        binding.bridge_requirement_id
                        for binding in claim.bridge_bindings
                        if binding.edge_mode == mode
                    }
                    for mode in BRIDGE_EDGE_MODES
                }
                # A proof route binds every hard prerequisite.  An
                # augmentation route may add its own complete optional path,
                # but it never replaces the proof set when both are declared.
                if (
                    binding_ids_by_mode["proof"] != proof_dependency_ids
                    or (
                        binding_ids_by_mode["augmentation"]
                        and binding_ids_by_mode["augmentation"]
                        != augmentation_dependency_ids
                    )
                    or (
                        not proof_dependency_ids
                        and not binding_ids_by_mode["augmentation"]
                    )
                ):
                    raise ValueError(
                        "bridge-value answer claims must bind the declared edge path"
                    )
            # A proof edge constrains a claim that *uses the resolved bridge
            # value*.  It does not make an independently applicable source
            # false: ``所有职级统一`` or a clause naming the original subject
            # can answer the same broad requirement without consuming a grade
            # mapping.  The controlled claim builder is responsible for
            # emitting those non-bridge applicability kinds only after it has
            # proved them from the exact source/ledger.  Rejecting them here
            # would incorrectly force a universal policy clause through an
            # irrelevant bridge and conflate two different logical routes.

        groups = tuple(self.structural_groups)
        if any(not isinstance(value, StructuralEvidenceGroup) for value in groups):
            raise ValueError("graph groups must contain StructuralEvidenceGroup values")
        if len({value.id for value in groups}) != len(groups):
            raise ValueError("evidence coverage graph contains duplicate structural groups")
        member_owner: dict[str, str] = {}
        for group in groups:
            group_item_ids = set(group.member_item_ids) | set(group.condition_item_ids)
            if not group_item_ids.issubset(item_ids):
                raise ValueError("structural group references an unknown evidence item")
            for item_id in group.member_item_ids:
                if document_keys[item_id] != group.document_key:
                    raise ValueError("local structural group members must share the group document")
                if item_id in member_owner:
                    raise ValueError("an evidence item can belong to only one local structural group")
                member_owner[item_id] = group.id
            for item_id in group.condition_item_ids:
                if document_keys[item_id] != group.document_key:
                    raise ValueError("condition items must share the target group document")

        group_ids = {value.id for value in groups}
        for claim in claims:
            if claim.structural_group_id is not None:
                if claim.structural_group_id not in group_ids:
                    raise ValueError("evidence claim references an unknown structural group")
                if claim.evidence_item_id not in {
                    item_id
                    for group in groups
                    if group.id == claim.structural_group_id
                    for item_id in group.member_item_ids
                }:
                    raise ValueError("evidence claim must belong to its structural group")
            if claim.condition_group_id is not None and claim.condition_group_id not in group_ids:
                raise ValueError("evidence claim references an unknown condition group")
            if claim.condition_group_id is not None:
                condition_group = next(
                    group
                    for group in groups
                    if group.id == claim.condition_group_id
                )
                if condition_group.document_key != claim.document_key:
                    raise ValueError(
                        "condition-bound claims cannot borrow conditions across documents"
                    )

        if not isinstance(self.document_root_keys, Mapping):
            raise ValueError("document_root_keys must be a mapping")
        answer_requirements_by_id = {
            value.id: value for value in requirements if value.role == "answer"
        }
        document_root_keys: dict[str, DocumentKey] = {}
        for requirement_id, document_key in self.document_root_keys.items():
            normalized_requirement_id = _normalized_requirement_ids(
                (requirement_id,),
                field_name="document root requirement id",
                max_items=1,
            )[0]
            requirement = answer_requirements_by_id.get(normalized_requirement_id)
            if requirement is None:
                raise ValueError("document roots can only target answer requirements")
            if not requirement.requires_document_policy_snapshot:
                raise ValueError(
                    "document roots can only target document-policy requirements"
                )
            document_root_keys[normalized_requirement_id] = _normalized_document_key(
                document_key,
                field_name="document root key",
            )
        requested_complete_document_keys = _normalized_document_keys(
            self.complete_document_keys,
            field_name="complete document key",
        )
        requested_complete_table_keys = _normalized_unique_texts(
            self.complete_table_keys,
            field_name="complete table key",
            max_items=64,
            max_chars=300,
        )
        complete_document_keys = _derived_complete_document_keys(
            evidence_items,
            visible_item_ids=visible_evidence_item_ids,
            require_visible=True,
        )
        complete_table_keys = _derived_complete_table_keys(
            evidence_items,
            visible_item_ids=visible_evidence_item_ids,
            require_visible=True,
        )
        source_complete_document_keys = _derived_complete_document_keys(
            evidence_items,
            require_visible=False,
        )
        source_complete_table_keys = _derived_complete_table_keys(
            evidence_items,
            require_visible=False,
        )
        if (
            requested_complete_document_keys
            and requested_complete_document_keys != complete_document_keys
        ):
            raise ValueError(
                "complete document keys must equal the graph-derived visible snapshot"
            )
        if (
            requested_complete_table_keys
            and requested_complete_table_keys != complete_table_keys
        ):
            raise ValueError(
                "complete table keys must equal the graph-derived visible snapshot"
            )
        collection_closures = tuple(self.collection_closures)
        if any(
            not isinstance(value, VerifiedCollectionClosure)
            for value in collection_closures
        ):
            raise ValueError(
                "collection closures must contain VerifiedCollectionClosure values"
            )
        closure_keys: set[tuple[str, tuple[str, ...], str, DocumentKey, str | None]] = set()
        answer_claim_item_ids_by_requirement: dict[str, set[str]] = {}
        for claim in claims:
            if claim.contribution_kind != "answer_claim":
                continue
            answer_claim_item_ids_by_requirement.setdefault(
                claim.requirement_id,
                set(),
            ).add(claim.evidence_item_id)
        for closure in collection_closures:
            requirement = requirement_by_id.get(closure.requirement_id)
            if (
                requirement is None
                or requirement.role != "answer"
                or not requirement.requires_collection_closure
            ):
                raise ValueError(
                    "collection closures can only target collection-contract answer requirements"
                )
            if not set(closure.claim_item_ids).issubset(item_ids):
                raise ValueError(
                    "collection closure claim items must belong to the evidence graph"
                )
            if any(
                document_keys[item_id] != closure.source_document_key
                for item_id in closure.claim_item_ids
            ):
                raise ValueError(
                    "collection closure claims must belong to its source document"
                )
            if not set(closure.claim_item_ids).issubset(
                answer_claim_item_ids_by_requirement.get(
                    closure.requirement_id,
                    set(),
                )
            ):
                raise ValueError(
                    "collection closure claims must be typed answer claims for its requirement"
                )
            if closure.source_kind == "full_document_snapshot":
                if closure.source_document_key not in source_complete_document_keys:
                    raise ValueError(
                        "full-document collection closures require a derived complete source snapshot"
                    )
            elif closure.source_kind == "complete_table":
                from core.rag_v2.collection_proofs import (
                    table_matches_collection_target,
                )

                source_table_items = tuple(
                    item
                    for item in evidence_items
                    if _snapshot_table_key(item) == closure.source_table_key
                )
                if (
                    closure.source_table_key not in source_complete_table_keys
                    or any(
                        _snapshot_table_key(evidence_items_by_id[item_id])
                        != closure.source_table_key
                        for item_id in closure.claim_item_ids
                    )
                    or not table_matches_collection_target(
                        source_table_items,
                        requirement=requirement,
                        requirements=requirements,
                    )
                ):
                    raise ValueError(
                        "table collection closures require a derived complete source table"
                    )
            else:  # source_declaration
                from core.rag_v2.collection_proofs import (
                    has_explicit_collection_closure,
                )

                declaration_item = evidence_items_by_id[closure.claim_item_ids[0]]
                if not has_explicit_collection_closure(
                    declaration_item,
                    requirement=requirement,
                    requirements=requirements,
                ):
                    raise ValueError(
                        "source-declaration collection closures require a target-bound source proof"
                    )
            contract = requirement.effective_coverage_contract
            if contract == "document_policy":
                root_document_key = document_root_keys.get(requirement.id)
                if (
                    closure.source_kind != "full_document_snapshot"
                    or root_document_key != closure.source_document_key
                ):
                    raise ValueError(
                        "document-policy collection closures require the rooted full-document snapshot"
                    )
            elif contract == "ordered_steps":
                if closure.source_kind != "source_declaration":
                    raise ValueError(
                        "ordered-step collection closures require a target-bound procedure declaration"
                    )
            elif contract == "structured_collection" and closure.source_kind not in {
                "complete_table",
                "source_declaration",
            }:
                raise ValueError(
                    "structured collection closures require a complete target table or source declaration"
                )
            key = (
                closure.requirement_id,
                closure.claim_item_ids,
                closure.source_kind,
                closure.source_document_key,
                closure.source_table_key,
            )
            if key in closure_keys:
                raise ValueError("evidence coverage graph contains duplicate collection closures")
            closure_keys.add(key)

        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "evidence_item_ids", evidence_item_ids)
        object.__setattr__(self, "visible_evidence_item_ids", visible_evidence_item_ids)
        object.__setattr__(self, "evidence_document_keys", MappingProxyType(document_keys))
        object.__setattr__(self, "evidence_items", evidence_items)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "structural_groups", groups)
        object.__setattr__(self, "document_root_keys", MappingProxyType(document_root_keys))
        object.__setattr__(self, "complete_document_keys", complete_document_keys)
        object.__setattr__(self, "complete_table_keys", complete_table_keys)
        object.__setattr__(self, "collection_closures", collection_closures)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requirement_ids": [value.id for value in self.requirements],
            "evidence_item_ids": list(self.evidence_item_ids),
            "visible_evidence_item_ids": list(self.visible_evidence_item_ids),
            "claims": [value.to_dict() for value in self.claims],
            "structural_groups": [value.to_dict() for value in self.structural_groups],
            "document_root_keys": {
                requirement_id: list(document_key)
                for requirement_id, document_key in self.document_root_keys.items()
            },
            "complete_document_keys": [list(value) for value in self.complete_document_keys],
            "complete_table_keys": list(self.complete_table_keys),
            "collection_closures": [
                value.to_dict() for value in self.collection_closures
            ],
        }


@dataclass(frozen=True)
class RequirementCoverageAssessment:
    requirement_id: str
    completeness: EvidenceCompleteness
    supporting_claim_ids: tuple[str, ...] = ()
    missing_item_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        requirement_id = _normalized_requirement_ids(
            (self.requirement_id,),
            field_name="coverage assessment requirement id",
            max_items=1,
        )[0]
        if self.completeness not in {"complete", "partial", "unknown"}:
            raise ValueError("unsupported requirement coverage completeness")
        supporting_claim_ids = _normalized_unique_texts(
            self.supporting_claim_ids,
            field_name="coverage supporting claim id",
            max_items=_MAX_GRAPH_ITEMS,
            max_chars=300,
        )
        missing_item_ids = _normalized_graph_item_ids(
            self.missing_item_ids,
            field_name="coverage missing item id",
        )
        reasons = _normalized_unique_texts(
            self.reasons,
            field_name="coverage reason",
            max_items=_MAX_STATE_REASONS,
            max_chars=_MAX_REASON_CHARS,
        )
        if self.completeness == "complete" and missing_item_ids:
            raise ValueError("complete requirement coverage cannot have missing items")
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "supporting_claim_ids", supporting_claim_ids)
        object.__setattr__(self, "missing_item_ids", missing_item_ids)
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "completeness": self.completeness,
            "supporting_claim_ids": list(self.supporting_claim_ids),
            "missing_item_ids": list(self.missing_item_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EvidenceAnswerConflict:
    """A set of closed answer claims that assert incompatible safe values.

    This is intentionally a graph result rather than a document-level
    heuristic.  ``claim_ids`` point back to the immutable final graph, where
    each alternative retains its exact bridge companions and structural
    closure.  A caller can therefore offer a scope clarification without
    rebuilding semantic routes from renderer metadata.
    """

    requirement_id: str
    claim_key: str
    result_kind: ClaimResultKind
    normalized_results: tuple[str, ...]
    claim_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        requirement_id = _normalized_requirement_ids(
            (self.requirement_id,),
            field_name="evidence answer conflict requirement id",
            max_items=1,
        )[0]
        claim_key = _normalized_text(
            self.claim_key,
            field_name="evidence answer conflict semantic key",
            max_chars=_MAX_REQUIREMENT_CHARS,
        ).casefold()
        result_kind = _normalized_text(
            self.result_kind,
            field_name="evidence answer conflict result kind",
            max_chars=80,
        ).casefold()
        if result_kind not in CONFLICT_COMPARABLE_CLAIM_RESULT_KINDS:
            raise ValueError(
                "evidence answer conflicts require a comparable result kind"
            )
        normalized_results = _normalized_unique_texts(
            self.normalized_results,
            field_name="evidence answer conflict normalized result",
            max_items=_MAX_GRAPH_ITEMS,
            max_chars=_MAX_REQUIREMENT_CHARS,
        )
        if len(normalized_results) < 2:
            raise ValueError(
                "evidence answer conflicts require at least two distinct results"
            )
        claim_ids = _normalized_unique_texts(
            self.claim_ids,
            field_name="evidence answer conflict claim id",
            max_items=_MAX_GRAPH_ITEMS,
            max_chars=300,
        )
        if len(claim_ids) < 2:
            raise ValueError(
                "evidence answer conflicts require at least two claim ids"
            )
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "claim_key", claim_key)
        object.__setattr__(self, "result_kind", result_kind)
        object.__setattr__(self, "normalized_results", normalized_results)
        object.__setattr__(self, "claim_ids", claim_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "claim_key": self.claim_key,
            "result_kind": self.result_kind,
            "normalized_results": list(self.normalized_results),
            "claim_ids": list(self.claim_ids),
        }


@dataclass(frozen=True)
class EvidenceCoverageAssessment:
    completeness: EvidenceCompleteness
    requirement_assessments: tuple[RequirementCoverageAssessment, ...]
    covered_requirement_ids: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    answer_conflicts: tuple[EvidenceAnswerConflict, ...] = ()

    def __post_init__(self) -> None:
        if self.completeness not in {"complete", "partial", "unknown"}:
            raise ValueError("unsupported evidence coverage completeness")
        requirement_assessments = tuple(self.requirement_assessments)
        if any(
            not isinstance(value, RequirementCoverageAssessment)
            for value in requirement_assessments
        ):
            raise ValueError(
                "requirement_assessments must contain RequirementCoverageAssessment values"
            )
        assessment_ids = [value.requirement_id for value in requirement_assessments]
        if len(assessment_ids) != len(set(assessment_ids)):
            raise ValueError("coverage assessment contains duplicate requirements")
        covered_requirement_ids = _normalized_requirement_ids(
            self.covered_requirement_ids,
            field_name="covered requirement id",
        )
        missing_requirement_ids = _normalized_requirement_ids(
            self.missing_requirement_ids,
            field_name="missing requirement id",
        )
        if set(covered_requirement_ids) & set(missing_requirement_ids):
            raise ValueError("covered and missing requirement ids cannot overlap")
        if self.completeness == "complete" and missing_requirement_ids:
            raise ValueError("complete coverage cannot have missing requirements")
        reasons = _normalized_unique_texts(
            self.reasons,
            field_name="evidence coverage reason",
            max_items=_MAX_STATE_REASONS,
            max_chars=_MAX_REASON_CHARS,
        )
        answer_conflicts = tuple(self.answer_conflicts)
        if any(
            not isinstance(value, EvidenceAnswerConflict)
            for value in answer_conflicts
        ):
            raise ValueError(
                "answer_conflicts must contain EvidenceAnswerConflict values"
            )
        assessment_id_set = set(assessment_ids)
        if any(
            value.requirement_id not in assessment_id_set
            for value in answer_conflicts
        ):
            raise ValueError(
                "evidence answer conflicts must target assessed requirements"
            )
        conflict_keys = {
            (
                value.requirement_id,
                value.claim_key,
                value.result_kind,
                value.normalized_results,
            )
            for value in answer_conflicts
        }
        if len(conflict_keys) != len(answer_conflicts):
            raise ValueError("evidence coverage contains duplicate answer conflicts")
        object.__setattr__(self, "requirement_assessments", requirement_assessments)
        object.__setattr__(self, "covered_requirement_ids", covered_requirement_ids)
        object.__setattr__(self, "missing_requirement_ids", missing_requirement_ids)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "answer_conflicts", answer_conflicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completeness": self.completeness,
            "requirement_assessments": [
                value.to_dict() for value in self.requirement_assessments
            ],
            "covered_requirement_ids": list(self.covered_requirement_ids),
            "missing_requirement_ids": list(self.missing_requirement_ids),
            "reasons": list(self.reasons),
            "answer_conflicts": [
                value.to_dict() for value in self.answer_conflicts
            ],
        }


@dataclass(frozen=True)
class EvidenceBundle:
    state: EvidenceState
    items: tuple[EvidenceItem, ...] = ()
    context_item_ids: tuple[str, ...] = ()
    answer_source_ids: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()
    # Optional because the graph is introduced independently of the current
    # pipeline.  Keeping it immutable and opt-in lets callers adopt stricter
    # coverage proof without changing legacy evidence assembly behavior.
    coverage_graph: EvidenceCoverageGraph | None = None
    # A graph is the source evidence topology; this is the result of evaluating
    # that exact topology against the exact context set.  Keeping the two
    # together prevents callers from later reconstructing "coverage" from
    # renderer roles or support labels after a budget change.
    coverage_assessment: EvidenceCoverageAssessment | None = None

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
        # Positive evidence is a two-part contract: its role says how the
        # chunk contributes, while the requirement ids say what it supports.
        # Allowing either half alone is how unrelated context leaks into
        # citations after a reranker/route failure.
        for item in items:
            if item.role in {"direct", "bridge", "complement"} and not item.supports_requirement_ids:
                raise ValueError(
                    "positive evidence roles require supports_requirement_ids"
                )
        for chunk_id in source_ids:
            source = item_by_id[chunk_id]
            if source.role not in {"direct", "bridge", "complement"}:
                raise ValueError(
                    "answer sources must have a positive evidence role"
                )
            if not source.supports_requirement_ids:
                raise ValueError(
                    "answer sources require supports_requirement_ids"
                )
        for chunk_id in context_ids:
            if item_by_id[chunk_id].constraint_status == "mismatch":
                raise ValueError("constraint-mismatched evidence cannot enter context")
        if context_ids and not self.state.may_build_context:
            raise ValueError("the evidence state does not permit a context")
        if self.state.completeness == "complete" and not context_ids:
            raise ValueError("complete evidence requires at least one context item")
        if self.state.completeness == "complete" and missing_ids:
            raise ValueError("complete evidence cannot have missing requirements")
        coverage_graph = self.coverage_graph
        coverage_assessment = self.coverage_assessment
        if coverage_graph is not None:
            if not isinstance(coverage_graph, EvidenceCoverageGraph):
                raise ValueError("coverage_graph must be an EvidenceCoverageGraph")
            if set(coverage_graph.evidence_item_ids) != set(item_by_id):
                raise ValueError("coverage graph items must match bundle items")
            if coverage_graph.evidence_items != items:
                raise ValueError("coverage graph source items must match bundle items")
            if coverage_graph.visible_evidence_item_ids != context_ids:
                raise ValueError(
                    "coverage graph visible items must match bundle context items"
                )
        if coverage_assessment is not None:
            if not isinstance(coverage_assessment, EvidenceCoverageAssessment):
                raise ValueError(
                    "coverage_assessment must be an EvidenceCoverageAssessment"
                )
            if coverage_graph is None:
                raise ValueError(
                    "coverage_assessment requires a coverage_graph"
                )
            graph_answer_ids = {
                requirement.id
                for requirement in coverage_graph.requirements
                if requirement.role == "answer"
            }
            assessment_ids = {
                assessment.requirement_id
                for assessment in coverage_assessment.requirement_assessments
            }
            if assessment_ids != graph_answer_ids:
                raise ValueError(
                    "coverage_assessment must assess every graph answer requirement"
                )
            if tuple(coverage_assessment.missing_requirement_ids) != missing_ids:
                raise ValueError(
                    "bundle missing requirements must match coverage_assessment"
                )

        object.__setattr__(self, "items", items)
        object.__setattr__(self, "context_item_ids", context_ids)
        object.__setattr__(self, "answer_source_ids", source_ids)
        object.__setattr__(self, "missing_requirement_ids", missing_ids)
        object.__setattr__(self, "coverage_graph", coverage_graph)
        object.__setattr__(self, "coverage_assessment", coverage_assessment)

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

        if self.coverage_assessment is not None:
            return self.coverage_assessment.covered_requirement_ids

        covered: list[str] = []
        seen: set[str] = set()
        for item in self.context_items:
            if item.role not in {"direct", "bridge", "complement"}:
                continue
            # Requirement bindings on an unverified fallback source partition
            # candidates by answer target; they are not proof that the target
            # is covered.  Coverage remains empty until normal adjudication and
            # graph closure succeed.  The one exception is a server-side
            # dominant-document auto-selection: that scope decision is
            # deterministic, so the binding counts as coverage even though the
            # reranker status stays unverified.
            verification = str(
                item.metadata.get("source_verification") or ""
            ).strip().casefold()
            verification_basis = str(
                item.metadata.get("verification_basis") or ""
            ).strip().casefold()
            if verification == "unverified" and verification_basis != (
                "deterministic_candidate_scope_confirmed"
            ):
                continue
            for requirement_id in item.supports_requirement_ids:
                if requirement_id in seen:
                    continue
                seen.add(requirement_id)
                covered.append(requirement_id)
        return tuple(covered)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "state": self.state.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "context_item_ids": list(self.context_item_ids),
            "answer_source_ids": list(self.answer_source_ids),
            "covered_requirement_ids": list(self.covered_requirement_ids),
            "missing_requirement_ids": list(self.missing_requirement_ids),
        }
        if self.coverage_graph is not None:
            result["coverage_graph"] = self.coverage_graph.to_dict()
        if self.coverage_assessment is not None:
            result["coverage_assessment"] = self.coverage_assessment.to_dict()
        return result
