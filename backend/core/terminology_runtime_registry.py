"""Database adapter for the request-local terminology runtime resolution.

The pure contracts in :mod:`core.terminology_runtime` deliberately know
nothing about ORM rows or request permissions.  This adapter is the one place
where a live registry graph becomes a runtime resolution.  Its public API
accepts the *already API-authorised* ``retrieval_kb_ids`` and queries only that
set; it never learns KB scope from retrieval candidates, document metadata or
registry rows.

All database/read-consistency failures become a ``degraded`` resolution.  The
caller must then retain the normal original-query retrieval path -- aliases are
an optional recall enhancement, never a dependency for baseline search.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.rag_v2.contracts import AnswerRequirementV2
from core.read_sessions import ReadSessionFactory, isolated_read_session
from core.terminology_contracts import (
    TerminologyBinding,
    TerminologyForm,
    term_occurs,
)
from core.terminology_runtime import (
    RuntimeTerminologyBinding,
    TerminologyRuntimeResolution,
    build_runtime_terminology_resolution,
)
from models.db_models import (
    TerminologyConcept,
    TerminologyRegistryState,
)


_MAX_STABLE_READ_ATTEMPTS = 3


class RuntimeTerminologyReadError(RuntimeError):
    """No stable, authorised registry view was available for this request."""


def _uuid_values(values: Iterable[object]) -> tuple[uuid.UUID, ...]:
    """Return stable selected KB ids, rejecting an unparseable scope early."""

    result: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for raw in values:
        if isinstance(raw, uuid.UUID):
            value = raw
        else:
            try:
                value = uuid.UUID(str(raw))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError("retrieval kb id is invalid") from exc
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _fingerprint(payload: object) -> str:
    """Hash local structure without writing business text into a trace field."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def terminology_runtime_fingerprints(
    *,
    requirements: Sequence[AnswerRequirementV2],
    retrieval_kb_ids: Iterable[object],
    scoped_document_ids: Iterable[object] | None = None,
) -> tuple[str, str]:
    """Build stable plan/scope fingerprints for a terminology resolution.

    These hashes are only request-local cache/trace identities.  They are not
    a semantic authority and are intentionally not reversible in trace output.
    """

    kb_ids = tuple(sorted(str(value) for value in _uuid_values(retrieval_kb_ids)))
    documents = tuple(sorted(
        str(value).strip()
        for value in (scoped_document_ids or ())
        if str(value).strip()
    ))
    plan_payload = [
        {
            "id": item.id,
            "description": item.description,
            "role": item.role,
            "scope_product": item.scope_product,
            "scope_version": item.scope_version,
            "scope_explicit_version": item.scope_explicit_version,
        }
        for item in requirements
        if isinstance(item, AnswerRequirementV2) and item.role == "answer"
    ]
    return (
        _fingerprint(plan_payload),
        _fingerprint({"kb_ids": kb_ids, "document_ids": documents}),
    )


def _active_forms(concept: Any) -> tuple[TerminologyForm, ...]:
    forms: list[TerminologyForm] = []
    seen: set[str] = set()
    for term in tuple(getattr(concept, "terms", ()) or ()):
        if not bool(getattr(term, "is_active", False)):
            continue
        match_mode = str(getattr(term, "match_mode", "")).strip().casefold()
        if match_mode not in {"strict_equivalent", "retrieval_only"}:
            # Invalid persisted data must not become a runtime synonym.
            continue
        form = TerminologyForm(
            term=str(getattr(term, "term", "") or ""),
            rule_id=str(getattr(term, "id", "") or ""),
            relation_strength=match_mode,
        )
        key = form.term.casefold()
        if key in seen:
            continue
        seen.add(key)
        forms.append(form)
    return tuple(forms)


def _runtime_bindings_from_concepts(
    *,
    concepts: Iterable[Any],
    requirements: Sequence[AnswerRequirementV2],
    authorized_kb_ids: Iterable[object],
) -> tuple[RuntimeTerminologyBinding, ...]:
    """Compile only active, selected-KB registry rows into pure bindings.

    The defensive ownership checks repeat the SQL predicate.  A malformed row
    from an ORM relationship cannot be allowed to turn an unrelated KB into a
    retrieval source merely because it was eagerly loaded beside an authorised
    concept.
    """

    allowed = {str(value) for value in _uuid_values(authorized_kb_ids)}
    answer_requirements = tuple(
        item for item in requirements
        if isinstance(item, AnswerRequirementV2) and item.role == "answer"
    )
    result: list[RuntimeTerminologyBinding] = []
    seen: set[tuple[str, str, str, str | None, str | None, str | None, str | None]] = set()
    for concept in concepts:
        if not bool(getattr(concept, "is_active", False)):
            continue
        concept_kb_id = str(getattr(concept, "kb_id", "") or "")
        if concept_kb_id not in allowed:
            continue
        forms = _active_forms(concept)
        if not forms:
            continue
        evidence_forms = tuple(
            form for form in forms
            if form.relation_strength == "strict_equivalent"
        )
        # An active concept without a strict form violates the write-side
        # invariant.  Fail closed for this concept rather than fabricate proof.
        if not evidence_forms:
            continue
        for scope in tuple(getattr(concept, "scope_bindings", ()) or ()):
            if not bool(getattr(scope, "is_active", False)):
                continue
            scope_kb_id = str(getattr(scope, "kb_id", "") or "")
            if scope_kb_id != concept_kb_id or scope_kb_id not in allowed:
                continue
            scope_id = str(getattr(scope, "id", "") or "")
            if not scope_id:
                continue
            for requirement in answer_requirements:
                for source_form in forms:
                    if not term_occurs(requirement.description, source_form.term):
                        continue
                    # ``TerminologyBinding`` independently enforces that a
                    # retrieval-only source cannot quietly become evidence.
                    binding = TerminologyBinding(
                        requirement_id=requirement.id,
                        concept_id=str(getattr(concept, "id", "") or ""),
                        concept_key=str(getattr(concept, "code", "") or ""),
                        display_name=str(getattr(concept, "canonical_term", "") or ""),
                        source_term=source_form.term,
                        source_relation_strength=source_form.relation_strength,
                        query_forms=forms,
                        evidence_forms=evidence_forms,
                        scope_binding_ids=(scope_id,),
                    )
                    runtime = RuntimeTerminologyBinding(
                        binding=binding,
                        kb_id=scope_kb_id,
                        document_id=(
                            str(getattr(scope, "document_id"))
                            if getattr(scope, "document_id", None) is not None
                            else None
                        ),
                        scope_product_key=getattr(scope, "scope_product_key", None),
                        scope_version_key=getattr(scope, "scope_version_key", None),
                        scope_project_key=getattr(scope, "scope_project_key", None),
                    )
                    key = (
                        runtime.kb_id,
                        runtime.binding.requirement_id,
                        runtime.binding.concept_id,
                        runtime.document_id,
                        runtime.scope_product_key,
                        runtime.scope_version_key,
                        runtime.scope_project_key,
                    )
                    # A concept can have multiple source spellings occurring
                    # in one requirement.  Preserve each spelling; only an
                    # exact duplicate scope/source is redundant.
                    source_key = runtime.source_key
                    dedupe_key = (*key, source_key)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    result.append(runtime)
    return tuple(result)


async def _read_revisions(
    db: AsyncSession,
    *,
    retrieval_kb_ids: tuple[uuid.UUID, ...],
) -> dict[str, int]:
    rows = (await db.execute(
        select(TerminologyRegistryState.kb_id, TerminologyRegistryState.revision)
        .where(TerminologyRegistryState.kb_id.in_(retrieval_kb_ids))
    )).all()
    revisions = {str(kb_id): int(revision) for kb_id, revision in rows}
    expected = {str(value) for value in retrieval_kb_ids}
    if set(revisions) != expected:
        raise RuntimeTerminologyReadError("registry_state_missing")
    return revisions


async def _read_active_concepts(
    db: AsyncSession,
    *,
    retrieval_kb_ids: tuple[uuid.UUID, ...],
) -> list[TerminologyConcept]:
    return (await db.execute(
        select(TerminologyConcept)
        .options(
            selectinload(TerminologyConcept.terms),
            selectinload(TerminologyConcept.scope_bindings),
        )
        .where(
            TerminologyConcept.kb_id.in_(retrieval_kb_ids),
            TerminologyConcept.is_active.is_(True),
        )
        .order_by(TerminologyConcept.id.asc())
    )).scalars().unique().all()


async def _read_stable_authorized_registry(
    *,
    db: AsyncSession,
    retrieval_kb_ids: tuple[uuid.UUID, ...],
) -> tuple[dict[str, int], list[TerminologyConcept]]:
    """Read ``revisions -> graph -> revisions`` until all selected KBs agree."""

    for _ in range(_MAX_STABLE_READ_ATTEMPTS):
        before = await _read_revisions(db, retrieval_kb_ids=retrieval_kb_ids)
        concepts = await _read_active_concepts(db, retrieval_kb_ids=retrieval_kb_ids)
        after = await _read_revisions(db, retrieval_kb_ids=retrieval_kb_ids)
        if before == after:
            return after, concepts
    raise RuntimeTerminologyReadError("registry_view_changed_during_read")


async def load_terminology_runtime_resolution(
    *,
    db: AsyncSession,
    read_session_factory: ReadSessionFactory | None = None,
    requirements: Sequence[AnswerRequirementV2],
    retrieval_kb_ids: Iterable[object],
    scoped_document_ids: Iterable[object] | None = None,
) -> TerminologyRuntimeResolution:
    """Load one stable, selected-KB terminology resolution for a RAG request.

    ``retrieval_kb_ids`` must come from the API's already-authorised RAG
    request scope (or its server-validated clarification subset).  The adapter
    intentionally has no candidate parameter and never expands that scope.
    """

    try:
        selected = _uuid_values(retrieval_kb_ids)
        plan_fingerprint, scope_fingerprint = terminology_runtime_fingerprints(
            requirements=requirements,
            retrieval_kb_ids=selected,
            scoped_document_ids=scoped_document_ids,
        )
    except (TypeError, ValueError):
        # No valid scope means aliases must not run.  The caller's normal
        # request validation still decides whether baseline retrieval is legal.
        return TerminologyRuntimeResolution.degraded(
            plan_fingerprint="0" * 64,
            scope_fingerprint="0" * 64,
            reason="invalid_authorized_retrieval_scope",
        )

    if not selected:
        return TerminologyRuntimeResolution.empty(
            plan_fingerprint=plan_fingerprint,
            scope_fingerprint=scope_fingerprint,
            authorized_kb_ids=(),
            registry_revisions={},
            reason="no_authorized_retrieval_kbs",
        )
    try:
        # Registry lookup is an optional recall enhancement.  It must not run
        # in the request transaction: a missing table during a rolling release
        # would otherwise abort conversation persistence and every later read.
        async with isolated_read_session(
            request_db=db,
            session_factory=read_session_factory,
        ) as registry_db:
            revisions, concepts = await _read_stable_authorized_registry(
                db=registry_db,
                retrieval_kb_ids=selected,
            )
            bindings = _runtime_bindings_from_concepts(
                concepts=concepts,
                requirements=requirements,
                authorized_kb_ids=selected,
            )
        return build_runtime_terminology_resolution(
            plan_fingerprint=plan_fingerprint,
            scope_fingerprint=scope_fingerprint,
            authorized_kb_ids=selected,
            registry_revisions=revisions,
            bindings=bindings,
        )
    except Exception:
        # Registry availability must never abort or empty the original RAG
        # search.  Do not surface exception text because ORM/database errors
        # can contain raw identifiers or business values.
        return TerminologyRuntimeResolution.degraded(
            plan_fingerprint=plan_fingerprint,
            scope_fingerprint=scope_fingerprint,
            authorized_kb_ids=selected,
            reason="registry_runtime_read_failed",
        )


__all__ = [
    "RuntimeTerminologyReadError",
    "load_terminology_runtime_resolution",
    "terminology_runtime_fingerprints",
]
