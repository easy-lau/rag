"""Transport-only API for the KB-owned controlled terminology registry.

All registry mutations are delegated to ``TerminologyRegistryMutationService``.
Keeping endpoints free of direct model writes prevents a new route from
skipping the KB state lock, event payload, ordinary audit, or canonical-term
invariant.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.audit import AuditLogger, get_audit
from core.deps import require_kb_access
from core.permissions import TERMINOLOGY_MANAGE, TERMINOLOGY_READ
from core.terminology_contracts import normalize_term_key
from core.terminology_registry import (
    TerminologyRegistryMutationService,
    field_was_supplied as _field_was_supplied,
    merge_initial_terms as _merge_initial_terms,
    normalize_concept_code as _normalize_concept_code,
    normalize_optional_scope as _normalize_optional_scope,
    normalize_reviewed_term as _normalize_reviewed_term,
    scope_values as _scope_values,
)
from database import get_db
from models.db_models import User
from models.schemas import (
    TerminologyConceptCreate,
    TerminologyConceptUpdate,
    TerminologyMutationOut,
    TerminologyRegistryOut,
    TerminologyScopeBindingCreate,
    TerminologyScopeBindingUpdate,
    TerminologyTermCreate,
    TerminologyTermUpdate,
)


router = APIRouter(prefix="/knowledge/{kb_id}/terminology", tags=["terminology"])


def _service(
    *,
    db: AsyncSession,
    user: User,
    audit: AuditLogger | None,
    kb_id: uuid.UUID,
) -> TerminologyRegistryMutationService:
    return TerminologyRegistryMutationService(
        db=db,
        user=user,
        audit=audit,
        kb_id=kb_id,
    )


@router.get("", response_model=TerminologyRegistryOut)
async def get_terminology_registry(
    kb_id: uuid.UUID,
    include_inactive: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_kb_access(TERMINOLOGY_READ)),
):
    return await _service(db=db, user=user, audit=None, kb_id=kb_id).read_registry(
        include_inactive=include_inactive
    )


@router.post(
    "/concepts",
    response_model=TerminologyMutationOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_terminology_concept(
    kb_id: uuid.UUID,
    payload: TerminologyConceptCreate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(TERMINOLOGY_MANAGE)),
):
    return await _service(db=db, user=user, audit=audit, kb_id=kb_id).create_concept(payload)


@router.put("/concepts/{concept_id}", response_model=TerminologyMutationOut)
async def update_terminology_concept(
    kb_id: uuid.UUID,
    concept_id: uuid.UUID,
    payload: TerminologyConceptUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(TERMINOLOGY_MANAGE)),
):
    return await _service(db=db, user=user, audit=audit, kb_id=kb_id).update_concept(
        concept_id=concept_id,
        payload=payload,
    )


@router.post(
    "/concepts/{concept_id}/terms",
    response_model=TerminologyMutationOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_terminology_term(
    kb_id: uuid.UUID,
    concept_id: uuid.UUID,
    payload: TerminologyTermCreate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(TERMINOLOGY_MANAGE)),
):
    return await _service(db=db, user=user, audit=audit, kb_id=kb_id).create_term(
        concept_id=concept_id,
        payload=payload,
    )


@router.put(
    "/concepts/{concept_id}/terms/{term_id}",
    response_model=TerminologyMutationOut,
)
async def update_terminology_term(
    kb_id: uuid.UUID,
    concept_id: uuid.UUID,
    term_id: uuid.UUID,
    payload: TerminologyTermUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(TERMINOLOGY_MANAGE)),
):
    return await _service(db=db, user=user, audit=audit, kb_id=kb_id).update_term(
        concept_id=concept_id,
        term_id=term_id,
        payload=payload,
    )


@router.post(
    "/bindings",
    response_model=TerminologyMutationOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_terminology_scope_binding(
    kb_id: uuid.UUID,
    payload: TerminologyScopeBindingCreate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(TERMINOLOGY_MANAGE)),
):
    return await _service(db=db, user=user, audit=audit, kb_id=kb_id).create_binding(payload)


@router.put("/bindings/{binding_id}", response_model=TerminologyMutationOut)
async def update_terminology_scope_binding(
    kb_id: uuid.UUID,
    binding_id: uuid.UUID,
    payload: TerminologyScopeBindingUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(TERMINOLOGY_MANAGE)),
):
    return await _service(db=db, user=user, audit=audit, kb_id=kb_id).update_binding(
        binding_id=binding_id,
        payload=payload,
    )
