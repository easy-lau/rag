"""Structural, immutable evidence-coverage graph for RAG v2.

The existing evidence assembler decides which authorized chunks may enter the
generation context.  This module answers the different question: whether those
*visible* chunks prove a requirement without dropping an attached note,
condition, bridge source, or document boundary along the way.

It deliberately consumes typed upstream facts instead of redoing lexical
matching.  In particular, a fuzzy synonym match can never manufacture an
``EvidenceClaim`` here.  A caller that used a terminology registry must pass a
``terminology_strict`` claim with its registry rule ids; otherwise the claim is
treated as a source assertion only.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from core.rag_v2.contracts import (
    CLAIM_APPLICABILITY_KINDS,
    EVIDENCE_CONTRIBUTION_KINDS,
    AnswerRequirementV2,
    BridgeClaimBinding,
    ClaimApplicability,
    DocumentKey,
    EvidenceBundle,
    EvidenceClaim,
    EvidenceAnswerConflict,
    EvidenceCoverageAssessment,
    EvidenceCoverageGraph,
    EvidenceItem,
    RequirementCoverageAssessment,
    StructuralEvidenceGroup,
    VerifiedCollectionClosure,
    validate_answer_requirement_graph,
)
from core.rag_v2.evidence_snapshots import table_key as _table_key
from core.rag_v2.evidence_snapshots import (
    complete_document_keys as _source_complete_document_keys,
    complete_table_keys as _source_complete_table_keys,
)
from core.rag_v2.collection_proofs import (
    has_explicit_collection_closure,
    table_matches_collection_target,
)


_NOTE_PREFIX_RE = re.compile(
    r"^\s*(?:注(?:意)?|说明|备注|提示)\s*[：:]",
    re.IGNORECASE,
)
_ROLE_TO_CONTRIBUTION = {
    "direct": "answer_claim",
    "bridge": "bridge_fact",
    "complement": "qualifier",
    "background": "background",
    "conflicting": "conflicting",
}


def _normalized_text(value: object, *, limit: int = 300) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized or len(normalized) > limit:
        return None
    return normalized


def _metadata_value(
    item: EvidenceItem,
    key: str,
    requirement_id: str | None = None,
) -> object | None:
    """Read only explicit, requirement-scoped metadata annotations.

    A mapping is interpreted as ``{requirement_id: annotation}``; a scalar is
    accepted for a chunk that makes one uniformly applicable assertion.  No
    content keyword is treated as an implicit annotation.
    """

    raw = item.metadata.get(key)
    if isinstance(raw, Mapping):
        if requirement_id is None:
            return None
        return raw.get(requirement_id)
    return raw


def _metadata_text(
    item: EvidenceItem,
    key: str,
    requirement_id: str | None = None,
    *,
    limit: int = 300,
) -> str | None:
    return _normalized_text(
        _metadata_value(item, key, requirement_id),
        limit=limit,
    )


def _metadata_ids(
    item: EvidenceItem,
    key: str,
    requirement_id: str | None = None,
    *,
    limit: int = 24,
) -> tuple[str, ...]:
    raw = _metadata_value(item, key, requirement_id)
    if isinstance(raw, str):
        values: Iterable[object] = (raw,)
    elif isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalized_text(value, limit=120)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) >= limit:
            break
    return tuple(result)


def _document_key(item: EvidenceItem) -> DocumentKey:
    return (item.kb_id, item.doc_id)


def _section_key(item: EvidenceItem) -> str | None:
    return _metadata_text(item, "section_key")


def _section_group_id(document_key: DocumentKey, section_key: str) -> str:
    return f"section:{document_key[0]}:{document_key[1]}:{section_key}"


def _chunk_group_id(item: EvidenceItem) -> str:
    return f"chunk:{item.chunk_id}"


def _effective_contribution_kind(item: EvidenceItem) -> str:
    """Use an explicit typed contribution first, then a conservative role map."""

    if item.contribution_kind is not None:
        return item.contribution_kind
    annotation = _metadata_text(item, "contribution_kind")
    if annotation in EVIDENCE_CONTRIBUTION_KINDS:
        return annotation
    return _ROLE_TO_CONTRIBUTION[item.role]


def _is_note(item: EvidenceItem) -> bool:
    return bool(_NOTE_PREFIX_RE.search(item.content))


def _bridge_bindings_for_item(
    item: EvidenceItem,
    requirement_id: str,
) -> tuple[BridgeClaimBinding, ...]:
    """Read exact upstream joins; malformed rows fail closed.

    ``resolved_bridge_joins`` is produced by the bridge resolver after it has
    already checked the subject/value/result relation.  It is the only legacy
    metadata accepted as a bridge proof.  We do not inspect words such as
    ``职级`` or a grade token again in this layer.
    """

    raw_joins = _metadata_value(item, "resolved_bridge_joins")
    if not isinstance(raw_joins, (list, tuple)):
        return ()
    bindings: list[BridgeClaimBinding] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_joins:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("answer_requirement_id") or "").strip() != requirement_id:
            continue
        try:
            binding = BridgeClaimBinding(
                bridge_requirement_id=str(raw.get("bridge_requirement_id") or ""),
                bridge_source_item_id=str(raw.get("bridge_source_chunk_id") or ""),
                bridge_value=str(raw.get("bridge_value") or ""),
            )
        except ValueError:
            continue
        key = (
            binding.bridge_requirement_id,
            binding.bridge_source_item_id,
            binding.bridge_value,
        )
        if key in seen:
            continue
        seen.add(key)
        bindings.append(binding)
    return tuple(bindings)


def _direct_subject_ids(item: EvidenceItem) -> frozenset[str]:
    values: set[str] = set()
    for key in (
        "direct_subject_answer_requirement_ids",
        "direct_subject_bridge_bypass_requirement_ids",
    ):
        values.update(_metadata_ids(item, key))
    return frozenset(values)


def _annotation_applicability(
    item: EvidenceItem,
    requirement_id: str,
) -> ClaimApplicability | None:
    value = _metadata_text(item, "claim_applicability", requirement_id)
    if value in CLAIM_APPLICABILITY_KINDS:
        return value  # type: ignore[return-value]
    return None


def _has_active_answer_assertion(
    item: EvidenceItem,
    requirement_id: str,
) -> bool:
    """Return whether the upstream evidence adjudicator closed one claim.

    ``EvidenceItem.role`` is a rendering role, not a complete proof model.  A
    resolved multi-hop amount may deliberately remain a ``complement`` in the
    prompt so its mapping/table companion is retained, while the trusted
    reconciler has already recorded an active assertion for the answer
    requirement.  The coverage graph must consume that typed result instead
    of incorrectly treating presentation order as semantic absence.

    The field is safe to consume here because task-ingestion strips it from
    untrusted retriever metadata and ``_reconcile_answer_claim_assertions``
    creates it only after source/scope/bridge adjudication.  Malformed or
    inactive entries deliberately fail closed.
    """

    raw = _metadata_value(item, "answer_claim_assertions", requirement_id)
    if raw is None:
        # The current reconciler writes a requirement-id keyed mapping.  Keep
        # this narrow fallback only for a direct mapping supplied by an older
        # in-memory caller; it cannot make one requirement's assertion apply
        # to another requirement.
        container = item.metadata.get("answer_claim_assertions")
        if isinstance(container, Mapping):
            raw = container.get(requirement_id)
    if isinstance(raw, Mapping):
        values: Iterable[object] = (raw,)
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        return False
    for value in values:
        if not isinstance(value, Mapping):
            continue
        status = str(value.get("status") or "").strip().casefold()
        if status == "active":
            return True
    return False


def classify_claim_applicability(
    item: EvidenceItem,
    requirement: AnswerRequirementV2,
) -> ClaimApplicability | None:
    """Classify a source assertion without lexical/fuzzy semantic inference.

    For dependent answers, only an exact bridge join or a typed direct/universal
    annotation may complete the claim.  This makes ``餐补`` vs ``餐饮补贴`` a
    terminology-registry concern rather than an accidental evidence join.

    This helper is deliberately *not* part of graph construction.  The
    request-local ledgered claim builder uses it before constructing an
    explicit :class:`EvidenceClaim`; ``build_evidence_coverage_graph`` never
    calls it or turns these metadata projections into proof on its own.
    """

    if not isinstance(item, EvidenceItem):
        raise ValueError("item must be an EvidenceItem")
    if not isinstance(requirement, AnswerRequirementV2):
        raise ValueError("requirement must be an AnswerRequirementV2")
    if requirement.role != "answer":
        return None
    if requirement.id not in item.supports_requirement_ids:
        return None
    if (
        _effective_contribution_kind(item) != "answer_claim"
        and not _has_active_answer_assertion(item, requirement.id)
    ):
        return None

    annotated = _annotation_applicability(item, requirement.id)
    if annotated is not None:
        if annotated == "bridge_value" and not _bridge_bindings_for_item(
            item,
            requirement.id,
        ):
            return None
        # A source clause can carry its own condition (for example, "偏远地区
        # 出差可申请额外补贴").  An external group is required only when the
        # parser/evidence stage explicitly says that a separate condition
        # section governs this claim.
        return annotated

    dependencies = requirement.depends_on_requirement_ids or ()
    if not dependencies:
        return "direct_subject"
    if _bridge_bindings_for_item(item, requirement.id):
        return "bridge_value"
    if requirement.id in _direct_subject_ids(item):
        return "direct_subject"
    # Dependent claims are not allowed to inherit an arbitrary nearby bridge.
    return None


def _build_structural_groups(
    items: Sequence[EvidenceItem],
    claims: Sequence[EvidenceClaim],
) -> tuple[tuple[StructuralEvidenceGroup, ...], dict[str, str]]:
    """Build local groups and reject ambiguous note/condition attachment.

    A parser section normally is the right atomic unit.  The exception is a
    section containing several independent tables: a bare ``注：`` is then not
    safely attributable to either table.  Those tables receive separate groups
    and the unbound note remains diagnostic-only until an upstream parser gives
    it an explicit group/table relation.
    """

    primary_item_ids = {
        claim.evidence_item_id
        for claim in claims
        if claim.contribution_kind == "answer_claim"
    }
    by_section: dict[tuple[DocumentKey, str], list[EvidenceItem]] = defaultdict(list)
    sectionless_items: list[EvidenceItem] = []
    for item in items:
        section_key = _section_key(item)
        if section_key is None:
            sectionless_items.append(item)
        else:
            by_section[(_document_key(item), section_key)].append(item)

    records: dict[str, dict[str, Any]] = {}
    item_group_ids: dict[str, str] = {}
    section_group_ids: dict[tuple[DocumentKey, str], set[str]] = defaultdict(set)

    def ensure_record(
        group_id: str,
        *,
        document_key: DocumentKey,
        section_key: str | None,
    ) -> dict[str, Any]:
        return records.setdefault(group_id, {
            "document_key": document_key,
            "section_key": section_key,
            "members": [],
            "primary": [],
            "companion": [],
            "qualifier": [],
            "condition": [],
            "table_keys": [],
        })

    for (document_key, section_key), section_items in by_section.items():
        table_keys = tuple(dict.fromkeys(
            table_key
            for item in section_items
            if (table_key := _table_key(item)) is not None
        ))
        if len(table_keys) <= 1:
            group_id = _section_group_id(document_key, section_key)
            ensure_record(
                group_id,
                document_key=document_key,
                section_key=section_key,
            )
            section_group_ids[(document_key, section_key)].add(group_id)
            for item in section_items:
                item_group_ids[item.chunk_id] = group_id
            continue
        for item in section_items:
            table_key = _table_key(item)
            if table_key is None:
                # No implicit cross-table attachment for an untyped note or
                # prose paragraph in a multi-table section.
                group_id = _chunk_group_id(item)
            else:
                group_id = f"table_group:{table_key}"
            ensure_record(
                group_id,
                document_key=document_key,
                section_key=section_key,
            )
            item_group_ids[item.chunk_id] = group_id
            section_group_ids[(document_key, section_key)].add(group_id)

    for item in sectionless_items:
        group_id = _chunk_group_id(item)
        ensure_record(
            group_id,
            document_key=_document_key(item),
            section_key=None,
        )
        item_group_ids[item.chunk_id] = group_id

    for item in items:
        group_id = item_group_ids[item.chunk_id]
        record = records[group_id]
        record["members"].append(item.chunk_id)
        table_key = _table_key(item)
        if table_key is not None:
            record["table_keys"].append(table_key)
        if item.chunk_id in primary_item_ids:
            record["primary"].append(item.chunk_id)

    # A ``注：`` only becomes a companion when a primary answer belongs to the
    # same unambiguous structural group.  It is never guessed across document
    # sections or across multiple tables in one section.
    for item in items:
        group_id = item_group_ids[item.chunk_id]
        record = records[group_id]
        contribution_kind = _effective_contribution_kind(item)
        # A typed active assertion is the primary proof even when the legacy
        # renderer labels the same chunk as a complement to keep a bridge
        # table nearby.  One item cannot simultaneously be the answer and its
        # own qualifier/companion; doing so makes structural closure depend on
        # presentation order rather than the proof graph.
        if item.chunk_id in primary_item_ids:
            continue
        if contribution_kind == "companion" or (
            contribution_kind == "background"
            and record["primary"]
            and _is_note(item)
        ):
            record["companion"].append(item.chunk_id)
        elif contribution_kind == "qualifier" and record["primary"]:
            record["qualifier"].append(item.chunk_id)

    by_document_section = {
        section: next(iter(group_ids))
        for section, group_ids in section_group_ids.items()
        if len(group_ids) == 1
    }
    for item in items:
        explicit_group_id = _metadata_text(item, "condition_for_group_id")
        if explicit_group_id is None:
            target_section_key = _metadata_text(item, "condition_for_section_key")
            target_group_id = (
                by_document_section.get((_document_key(item), target_section_key))
                if target_section_key is not None
                else None
            )
        else:
            target_group_id = explicit_group_id
        if target_group_id is None:
            continue
        target = records.get(target_group_id)
        # The explicit link is still constrained to the same document.  This
        # prevents a leave-policy condition from attaching to a travel-policy
        # group merely because it uses the same job title.
        if target is None or target["document_key"] != _document_key(item):
            continue
        target["condition"].append(item.chunk_id)

    groups: list[StructuralEvidenceGroup] = []
    for group_id, record in records.items():
        groups.append(StructuralEvidenceGroup(
            id=group_id,
            document_key=record["document_key"],
            member_item_ids=tuple(dict.fromkeys(record["members"])),
            primary_item_ids=tuple(dict.fromkeys(record["primary"])),
            companion_item_ids=tuple(dict.fromkeys(record["companion"])),
            qualifier_item_ids=tuple(dict.fromkeys(record["qualifier"])),
            condition_item_ids=tuple(dict.fromkeys(record["condition"])),
            section_key=record["section_key"],
            table_keys=tuple(dict.fromkeys(record["table_keys"])),
        ))
    return tuple(groups), item_group_ids


def _inferred_document_root_keys(
    items: Sequence[EvidenceItem],
    requirements: Sequence[AnswerRequirementV2],
) -> dict[str, DocumentKey]:
    """Return roots proven by the current evidence assembly stage only.

    A document-policy root is not a convenient title hint for a regular
    lookup.  It is a separate, requirement-scoped proof emitted only for a
    current-query root seed by ``evidence._attach_document_root_topic_anchors``.
    Keeping this field distinct from the wider topic-inheritance annotation
    prevents a transport/meal section from silently becoming the whole policy.
    """

    document_policy_ids = {
        requirement.id
        for requirement in requirements
        if requirement.role == "answer"
        and requirement.requires_document_policy_snapshot
    }
    candidates: dict[str, set[DocumentKey]] = defaultdict(set)
    for item in items:
        for requirement_id in _metadata_ids(
            item,
            "document_policy_root_requirement_ids",
        ):
            if requirement_id in document_policy_ids:
                candidates[requirement_id].add(_document_key(item))
    return {
        requirement_id: next(iter(document_keys))
        for requirement_id, document_keys in candidates.items()
        if len(document_keys) == 1
    }


def _normalize_explicit_roots(
    values: Mapping[str, DocumentKey] | None,
) -> dict[str, DocumentKey]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("document_root_keys must be a mapping")
    result: dict[str, DocumentKey] = {}
    for requirement_id, raw_key in values.items():
        normalized_requirement_id = _normalized_text(requirement_id, limit=64)
        if normalized_requirement_id is None:
            raise ValueError("document root requirement id must be a string")
        if not isinstance(raw_key, (tuple, list)) or len(raw_key) != 2:
            raise ValueError("document root key must be a (kb_id, doc_id) tuple")
        kb_id = _normalized_text(raw_key[0], limit=200)
        doc_id = _normalized_text(raw_key[1], limit=200)
        if kb_id is None or doc_id is None:
            raise ValueError("document root key contains an invalid id")
        result[normalized_requirement_id] = (kb_id, doc_id)
    return result


def derive_verified_collection_closures(
    graph: EvidenceCoverageGraph,
) -> tuple[VerifiedCollectionClosure, ...]:
    """Derive collection certificates from the graph's actual typed claims.

    This is deliberately a graph operation, not an evidence-mapper shortcut.
    The certificate claim ids therefore use exactly the same projection that
    the assessor consumes, eliminating role/support metadata drift.  Source
    structures are re-verified from immutable graph items before a certificate
    is emitted.
    """

    if not isinstance(graph, EvidenceCoverageGraph):
        raise ValueError("graph must be an EvidenceCoverageGraph")
    item_by_id = {item.chunk_id: item for item in graph.evidence_items}
    answer_claims_by_requirement: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in graph.claims:
        if claim.contribution_kind == "answer_claim":
            answer_claims_by_requirement[claim.requirement_id].append(claim)
    source_complete_documents = frozenset(_source_complete_document_keys(
        graph.evidence_items,
        require_visible=False,
    ))
    source_complete_tables = frozenset(_source_complete_table_keys(
        graph.evidence_items,
        require_visible=False,
    ))
    closures: list[VerifiedCollectionClosure] = []
    for requirement in graph.requirements:
        if (
            requirement.role != "answer"
            or not requirement.requires_collection_closure
        ):
            continue
        contract = requirement.effective_coverage_contract
        root_document_key = graph.document_root_keys.get(requirement.id)
        all_claims = tuple(answer_claims_by_requirement.get(requirement.id, ()))
        claims = (
            tuple(
                claim
                for claim in all_claims
                if claim.document_key == root_document_key
            )
            if contract == "document_policy" and root_document_key is not None
            else all_claims
        )
        if not claims:
            continue
        claim_item_ids = tuple(sorted({claim.evidence_item_id for claim in claims}))
        claim_documents = {claim.document_key for claim in claims}
        if len(claim_documents) != 1:
            continue
        source_document_key = next(iter(claim_documents))

        if contract == "document_policy":
            if (
                root_document_key == source_document_key
                and source_document_key in source_complete_documents
            ):
                closures.append(VerifiedCollectionClosure(
                    requirement_id=requirement.id,
                    claim_item_ids=claim_item_ids,
                    source_kind="full_document_snapshot",
                    source_document_key=source_document_key,
                ))
            continue

        # A finite, unordered collection may be closed by an authoritative
        # parsed table.  An ordered procedure cannot: row order alone does not
        # prove transitions or that no later step exists.  It needs the
        # target-bound sequence declaration below.
        if contract == "structured_collection":
            claim_table_keys = {
                _table_key(item_by_id[item_id])
                for item_id in claim_item_ids
            }
            if len(claim_table_keys) == 1:
                source_table_key = next(iter(claim_table_keys))
                table_items = tuple(
                    item
                    for item in graph.evidence_items
                    if _table_key(item) == source_table_key
                )
                if (
                    source_table_key is not None
                    and source_table_key in source_complete_tables
                    and table_matches_collection_target(
                        table_items,
                        requirement=requirement,
                        requirements=graph.requirements,
                    )
                ):
                    closures.append(VerifiedCollectionClosure(
                        requirement_id=requirement.id,
                        claim_item_ids=claim_item_ids,
                        source_kind="complete_table",
                        source_document_key=source_document_key,
                        source_table_key=source_table_key,
                    ))

        if len(claim_item_ids) == 1:
            source_item = item_by_id[claim_item_ids[0]]
            if has_explicit_collection_closure(
                source_item,
                requirement=requirement,
                requirements=graph.requirements,
            ):
                closures.append(VerifiedCollectionClosure(
                    requirement_id=requirement.id,
                    claim_item_ids=claim_item_ids,
                    source_kind="source_declaration",
                    source_document_key=source_document_key,
                ))
    return tuple(dict.fromkeys(closures))


def build_evidence_coverage_graph(
    evidence: EvidenceBundle,
    requirements: Sequence[AnswerRequirementV2],
    *,
    claims: Sequence[EvidenceClaim] | None = None,
    document_root_keys: Mapping[str, DocumentKey] | None = None,
    collection_closures: Sequence[VerifiedCollectionClosure] = (),
) -> EvidenceCoverageGraph:
    """Build an immutable graph over the evidence that generation can see.

    ``claims`` is the sole positive-proof input.  A graph validates and closes
    explicit, request-controlled claims; it does not derive a claim from an
    ``EvidenceItem`` role, support label, bridge projection, or arbitrary
    metadata.  Omitting claims is therefore a deliberately fail-closed empty
    proof graph, not a legacy compatibility path.
    """

    if not isinstance(evidence, EvidenceBundle):
        raise ValueError("evidence must be an EvidenceBundle")
    if isinstance(requirements, (str, bytes)) or not isinstance(requirements, Sequence):
        raise ValueError("requirements must be a sequence")
    normalized_requirements = tuple(requirements)
    if any(not isinstance(value, AnswerRequirementV2) for value in normalized_requirements):
        raise ValueError("requirements must contain AnswerRequirementV2 values")
    if len({value.id for value in normalized_requirements}) != len(normalized_requirements):
        raise ValueError("requirements contains duplicate ids")
    validate_answer_requirement_graph(normalized_requirements)
    normalized_collection_closures = tuple(collection_closures)
    if any(
        not isinstance(value, VerifiedCollectionClosure)
        for value in normalized_collection_closures
    ):
        raise ValueError(
            "collection_closures must contain VerifiedCollectionClosure values"
        )

    items = tuple(evidence.items)
    item_by_id = {item.chunk_id: item for item in items}
    visible_item_ids = frozenset(evidence.context_item_ids)
    graph_claims = () if claims is None else tuple(claims)
    if any(not isinstance(value, EvidenceClaim) for value in graph_claims):
        raise ValueError("claims must contain EvidenceClaim values")

    groups, item_group_ids = _build_structural_groups(items, graph_claims)
    group_ids = {group.id for group in groups}
    bound_claims: list[EvidenceClaim] = []
    for claim in graph_claims:
        item = item_by_id.get(claim.evidence_item_id)
        if item is None:
            raise ValueError("claim references an evidence item outside the bundle")
        expected_group_id = item_group_ids[claim.evidence_item_id]
        if claim.structural_group_id not in {None, expected_group_id}:
            raise ValueError("claim structural group does not match its source item")
        condition_group_id = claim.condition_group_id
        if (
            condition_group_id is not None
            and condition_group_id not in group_ids
        ):
            raise ValueError("claim condition group does not exist in this evidence graph")
        bound_claims.append(replace(
            claim,
            structural_group_id=expected_group_id,
        ))

    roots = _inferred_document_root_keys(items, normalized_requirements)
    roots.update(_normalize_explicit_roots(document_root_keys))
    return EvidenceCoverageGraph(
        requirements=normalized_requirements,
        evidence_item_ids=tuple(item_by_id),
        visible_evidence_item_ids=evidence.context_item_ids,
        evidence_document_keys={
            item.chunk_id: _document_key(item)
            for item in items
        },
        evidence_items=items,
        claims=tuple(bound_claims),
        structural_groups=groups,
        document_root_keys=roots,
        collection_closures=normalized_collection_closures,
    )


def _group_missing_item_ids(
    group: StructuralEvidenceGroup,
    visible_item_ids: frozenset[str],
) -> tuple[str, ...]:
    return tuple(
        item_id
        for item_id in group.required_item_ids
        if item_id not in visible_item_ids
    )


def _claim_closure(
    claim: EvidenceClaim,
    *,
    groups_by_id: Mapping[str, StructuralEvidenceGroup],
    visible_item_ids: frozenset[str],
    bridge_claims_by_source: Mapping[tuple[str, str], tuple[EvidenceClaim, ...]],
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Return whether one answer claim is visible and structurally closed."""

    missing_item_ids: list[str] = []
    reasons: list[str] = []
    if claim.evidence_item_id not in visible_item_ids:
        return False, (claim.evidence_item_id,), ("answer_claim_not_visible",)
    group = groups_by_id.get(claim.structural_group_id or "")
    if group is None:
        return False, (), ("answer_claim_group_missing",)
    missing_item_ids.extend(_group_missing_item_ids(group, visible_item_ids))
    if missing_item_ids:
        reasons.append("structural_companion_not_visible")

    if claim.applicability == "bridge_value":
        for binding in claim.bridge_bindings:
            source_claims = bridge_claims_by_source.get(
                (binding.bridge_requirement_id, binding.bridge_source_item_id),
                (),
            )
            if binding.bridge_source_item_id not in visible_item_ids or not source_claims:
                missing_item_ids.append(binding.bridge_source_item_id)
                reasons.append("bound_bridge_fact_not_visible")
                continue

            # A bridge fact is evidence too.  Requiring only the chunk that
            # names a resolved value used to let an attached note, qualifier,
            # or condition disappear while the answer still appeared closed.
            # Test the bridge source through the same local structural rule as
            # an answer claim; no bridge claim carries another bridge edge, so
            # this is deliberately non-recursive.
            source_is_closed = False
            source_missing_ids: list[str] = []
            for source_claim in source_claims:
                source_group = groups_by_id.get(
                    source_claim.structural_group_id or ""
                )
                if source_group is None:
                    source_missing_ids.append(binding.bridge_source_item_id)
                    continue
                missing_for_source = _group_missing_item_ids(
                    source_group,
                    visible_item_ids,
                )
                if not missing_for_source:
                    source_is_closed = True
                    break
                source_missing_ids.extend(missing_for_source)
            if not source_is_closed:
                missing_item_ids.extend(source_missing_ids or [
                    binding.bridge_source_item_id
                ])
                reasons.append("bound_bridge_fact_structurally_incomplete")
    elif claim.applicability == "condition_bound":
        condition_group = (
            groups_by_id.get(claim.condition_group_id)
            if claim.condition_group_id is not None
            else None
        )
        if claim.condition_group_id is not None and condition_group is None:
            reasons.append("bound_condition_group_missing")
        elif condition_group is not None:
            # A condition-bound claim uses the condition group as an atomic
            # condition source.  Require its local members instead of keyword
            # matching a different section that happens to mention a city/date.
            for item_id in condition_group.member_item_ids:
                if item_id not in visible_item_ids:
                    missing_item_ids.append(item_id)
            if any(item_id not in visible_item_ids for item_id in condition_group.member_item_ids):
                reasons.append("bound_condition_not_visible")

    missing = tuple(dict.fromkeys(missing_item_ids))
    return not missing and not reasons, missing, tuple(dict.fromkeys(reasons))


def _verified_closure_proven(
    closures: Sequence[VerifiedCollectionClosure],
    *,
    requirement: AnswerRequirementV2,
    root_document_key: DocumentKey | None,
    candidate_claims: Sequence[EvidenceClaim],
    closed_claims: Sequence[EvidenceClaim],
) -> bool:
    """Validate a source-verified collection certificate against live claims.

    A certificate is made before the renderer budget is applied.  It cannot
    paper over a dropped answer clause or a missing note/condition: it must
    cover exactly the currently typed answer claims and every claim still has
    to be visible and structurally closed.
    """

    if not closures or not candidate_claims:
        return False
    candidate_item_ids = {
        claim.evidence_item_id
        for claim in candidate_claims
    }
    closed_item_ids = {
        claim.evidence_item_id
        for claim in closed_claims
    }
    for closure in closures:
        if requirement.effective_coverage_contract == "document_policy" and (
            closure.source_kind != "full_document_snapshot"
            or closure.source_document_key != root_document_key
        ):
            continue
        closure_item_ids = set(closure.claim_item_ids)
        if (
            closure_item_ids == candidate_item_ids
            and closure_item_ids.issubset(closed_item_ids)
        ):
            return True
    return False


def _closed_answer_claim_conflicts(
    requirement: AnswerRequirementV2,
    closed_claims: Sequence[EvidenceClaim],
) -> tuple[EvidenceAnswerConflict, ...]:
    """Return only semantic conflicts proven by fully closed answer routes.

    A document/chunk count is not a contradiction: two independently sourced
    clauses can repeat the same amount, or legitimately cover different
    facets.  Conversely, a raw candidate's annotation cannot create a
    conflict.  Compare only the immutable semantic result carried by a typed
    answer claim *after* its bridge/condition/structural route has closed.

    ``claim_key`` groups the same requested facet (for example, lodging
    standard) while preserving unrelated requirements and table columns.  The
    source adjudicator produces these values; this graph deliberately does no
    text parsing or filename inference.
    """

    if requirement.role != "answer":
        return ()
    grouped: dict[tuple[str, str], list[EvidenceClaim]] = defaultdict(list)
    for claim in closed_claims:
        if (
            claim.contribution_kind != "answer_claim"
            or claim.result_kind not in {"scalar", "categorical"}
            or claim.claim_key is None
            or claim.normalized_result is None
        ):
            continue
        grouped[(claim.result_kind, claim.claim_key)].append(claim)

    conflicts: list[EvidenceAnswerConflict] = []
    for (result_kind, claim_key), claims in grouped.items():
        values = tuple(sorted({
            claim.normalized_result
            for claim in claims
            if claim.normalized_result is not None
        }))
        if len(values) < 2:
            continue
        # The conflict should identify every closed alternative.  A repeated
        # source for one value is diagnostic provenance, not a third choice.
        claim_ids = tuple(
            claim.id
            for claim in claims
            if claim.normalized_result in set(values)
        )
        conflicts.append(EvidenceAnswerConflict(
            requirement_id=requirement.id,
            claim_key=claim_key,
            result_kind=result_kind,
            normalized_results=values,
            claim_ids=claim_ids,
        ))
    return tuple(conflicts)


def assess_evidence_coverage_graph(
    graph: EvidenceCoverageGraph,
) -> EvidenceCoverageAssessment:
    """Assess each answer requirement against the graph's exact visible set."""

    if not isinstance(graph, EvidenceCoverageGraph):
        raise ValueError("graph must be an EvidenceCoverageGraph")
    visible_item_ids = frozenset(graph.visible_evidence_item_ids)
    groups_by_id = {group.id: group for group in graph.structural_groups}
    bridge_claims_by_source: dict[tuple[str, str], list[EvidenceClaim]] = defaultdict(list)
    for claim in graph.claims:
        if claim.contribution_kind != "bridge_fact":
            continue
        bridge_claims_by_source[(claim.requirement_id, claim.evidence_item_id)].append(claim)
    frozen_bridge_claims_by_source = {
        key: tuple(value)
        for key, value in bridge_claims_by_source.items()
    }
    answer_claims_by_requirement: dict[str, list[EvidenceClaim]] = defaultdict(list)
    for claim in graph.claims:
        if claim.contribution_kind == "answer_claim":
            answer_claims_by_requirement[claim.requirement_id].append(claim)

    assessments: list[RequirementCoverageAssessment] = []
    overall_reasons: list[str] = []
    covered_ids: list[str] = []
    missing_ids: list[str] = []
    answer_conflicts: list[EvidenceAnswerConflict] = []
    complete_document_keys = frozenset(graph.complete_document_keys)
    collection_closures_by_requirement: dict[
        str, list[VerifiedCollectionClosure]
    ] = defaultdict(list)
    for closure in graph.collection_closures:
        collection_closures_by_requirement[closure.requirement_id].append(closure)
    for requirement in graph.requirements:
        if requirement.role != "answer":
            continue
        all_claims = tuple(answer_claims_by_requirement.get(requirement.id, ()))
        candidate_claims = all_claims
        reasons: list[str] = []
        missing_item_ids: list[str] = []
        root_document_key = graph.document_root_keys.get(requirement.id)
        document_policy_root_missing = False
        document_policy_visible_snapshot_missing = False
        if requirement.effective_coverage_contract == "document_policy":
            if root_document_key is None:
                candidate_claims = ()
                document_policy_root_missing = True
            else:
                candidate_claims = tuple(
                    claim for claim in all_claims
                    if claim.document_key == root_document_key
                )
                if root_document_key not in complete_document_keys:
                    document_policy_visible_snapshot_missing = True
        if not candidate_claims:
            reasons.append("no_typed_answer_claim")
            if document_policy_root_missing:
                reasons.append("document_policy_root_unproven")
            completeness = "unknown" if not visible_item_ids else "partial"
            assessment = RequirementCoverageAssessment(
                requirement_id=requirement.id,
                completeness=completeness,
                missing_item_ids=tuple(dict.fromkeys(missing_item_ids)),
                reasons=tuple(dict.fromkeys(reasons)),
            )
            assessments.append(assessment)
            if requirement.is_required_answer:
                missing_ids.append(requirement.id)
                overall_reasons.append("required_answer_claim_missing")
            continue

        closed_claims: list[EvidenceClaim] = []
        for claim in candidate_claims:
            closed, claim_missing_item_ids, claim_reasons = _claim_closure(
                claim,
                groups_by_id=groups_by_id,
                visible_item_ids=visible_item_ids,
                bridge_claims_by_source=frozen_bridge_claims_by_source,
            )
            missing_item_ids.extend(claim_missing_item_ids)
            reasons.extend(claim_reasons)
            if closed:
                closed_claims.append(claim)

        contract = requirement.effective_coverage_contract
        verified_collection_closure = _verified_closure_proven(
            collection_closures_by_requirement.get(requirement.id, ()),
            requirement=requirement,
            root_document_key=root_document_key,
            candidate_claims=candidate_claims,
            closed_claims=closed_claims,
        )
        if contract == "single_claim":
            complete = bool(closed_claims)
        elif contract in {"structured_collection", "ordered_steps"}:
            complete = (
                len(closed_claims) == len(candidate_claims)
                and verified_collection_closure
            )
            if not complete:
                reasons.append(
                    "ordered_steps_closure_unproven"
                    if contract == "ordered_steps"
                    else "structured_collection_closure_unproven"
                )
        else:  # document_policy
            complete = (
                len(closed_claims) == len(candidate_claims)
                and bool(candidate_claims)
                and (
                    verified_collection_closure
                    or (
                        root_document_key is not None
                        and root_document_key in complete_document_keys
                    )
                )
            )
            if (
                not complete
                and not verified_collection_closure
                and document_policy_root_missing
            ):
                reasons.append("document_policy_root_unproven")
            elif (
                not complete
                and not verified_collection_closure
                and document_policy_visible_snapshot_missing
            ):
                reasons.append("document_policy_snapshot_incomplete")
            elif not complete:
                reasons.append("document_policy_member_incomplete")

        completeness = "complete" if complete else "partial"
        closed_conflicts = _closed_answer_claim_conflicts(
            requirement,
            closed_claims,
        )
        if closed_conflicts:
            reasons.append("mutually_exclusive_closed_answer_claims")
            answer_conflicts.extend(closed_conflicts)
        assessment = RequirementCoverageAssessment(
            requirement_id=requirement.id,
            completeness=completeness,
            supporting_claim_ids=tuple(claim.id for claim in closed_claims),
            missing_item_ids=tuple(dict.fromkeys(missing_item_ids)),
            reasons=tuple(dict.fromkeys(reasons)),
        )
        assessments.append(assessment)
        if complete:
            covered_ids.append(requirement.id)
        elif requirement.is_required_answer:
            missing_ids.append(requirement.id)
            overall_reasons.append("required_answer_coverage_incomplete")

    if not assessments:
        overall_completeness = "unknown"
        overall_reasons.append("no_answer_requirements")
    elif not missing_ids:
        overall_completeness = "complete"
    elif visible_item_ids:
        overall_completeness = "partial"
    else:
        overall_completeness = "unknown"
    return EvidenceCoverageAssessment(
        completeness=overall_completeness,
        requirement_assessments=tuple(assessments),
        covered_requirement_ids=tuple(covered_ids),
        missing_requirement_ids=tuple(missing_ids),
        reasons=tuple(dict.fromkeys(overall_reasons)),
        answer_conflicts=tuple(answer_conflicts),
    )


__all__ = [
    "assess_evidence_coverage_graph",
    "build_evidence_coverage_graph",
    "classify_claim_applicability",
]
