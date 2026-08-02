"""KB-owned controlled-terminology registry domain service.

This module is intentionally a management boundary, not a RAG runtime hook.
It owns every terminology write so a KB-local state lock, immutable event, and
ordinary operation audit always occur in one database transaction.  The API
layer only validates transport/auth dependencies and delegates here.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
import uuid
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime
from typing import Any, TypeVar

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.audit import AuditLogger
from core.deps import ensure_kb_access
from core.permissions import DOC_READ
from core.terminology_contracts import normalize_scope_key, normalize_term_key
from models.db_models import (
    Document,
    TerminologyConcept,
    TerminologyRegistryRevision,
    TerminologyRegistryState,
    TerminologyScopeBinding,
    TerminologyTerm,
    User,
    now_utc,
)
from models.schemas import (
    TerminologyConceptCreate,
    TerminologyConceptOut,
    TerminologyConceptUpdate,
    TerminologyMutationOut,
    TerminologyRegistryOut,
    TerminologyScopeBindingCreate,
    TerminologyScopeBindingOut,
    TerminologyScopeBindingUpdate,
    TerminologyTermCreate,
    TerminologyTermOut,
    TerminologyTermUpdate,
)


_CONCEPT_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SCOPE_FIELDS = (
    "scope_product_key",
    "scope_version_key",
    "scope_project_key",
)
_REGISTRY_EVENT_SCHEMA_VERSION = "terminology_event.v1"
_MAX_STABLE_READ_ATTEMPTS = 3
T = TypeVar("T")


class RegistryReadConsistencyError(RuntimeError):
    """The reader observed concurrent KB-local revisions until its budget ended."""


def field_was_supplied(payload: object, field: str) -> bool:
    fields_set = getattr(payload, "model_fields_set", None)
    if fields_set is None:  # Pydantic v1 compatibility for offline tools.
        fields_set = getattr(payload, "__fields_set__", set())
    return field in fields_set


def normalize_concept_code(value: object) -> str:
    code = str(value or "").strip().casefold()
    if not _CONCEPT_CODE_RE.fullmatch(code):
        raise HTTPException(
            status_code=422,
            detail="术语概念编码只能包含小写字母、数字、点、连字符或下划线，且以字母开头",
        )
    return code


def normalize_reviewed_term(value: object, *, field: str) -> tuple[str, str]:
    """Normalize presentation and identity forms without inventing aliases."""

    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail=f"{field}必须是文本")
    term = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()
    normalized = normalize_term_key(term)
    if not term or len(term) > 120 or len(normalized) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"{field}不能为空，且规范化后至少应包含两个字符",
        )
    return term, normalized


def normalize_optional_scope(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=422, detail=f"{field}如填写则不能为空")
    normalized = normalize_scope_key(value)
    if not normalized or len(normalized) > 160:
        raise HTTPException(status_code=422, detail=f"{field}格式无效")
    return normalized


def scope_values(payload: object, *, only_supplied: bool) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for field in _SCOPE_FIELDS:
        if only_supplied and not field_was_supplied(payload, field):
            continue
        values[field] = normalize_optional_scope(getattr(payload, field), field=field)
    return values


def merge_initial_terms(
    *,
    canonical_term: str,
    supplied: Iterable[TerminologyTermCreate],
) -> list[tuple[str, str, str, bool]]:
    """Create a canonical strict form and reject conflicting duplicate input."""

    canonical, canonical_key = normalize_reviewed_term(canonical_term, field="规范术语")
    terms: list[tuple[str, str, str, bool]] = [
        (canonical, canonical_key, "strict_equivalent", True)
    ]
    seen = {canonical_key: ("strict_equivalent", True)}
    for item in supplied:
        term, normalized = normalize_reviewed_term(item.term, field="术语")
        desired = (item.match_mode, item.is_active)
        previous = seen.get(normalized)
        if previous is not None:
            if previous != desired:
                raise HTTPException(
                    status_code=422,
                    detail="同一概念中不能提交关系或启用状态冲突的重复术语",
                )
            continue
        seen[normalized] = desired
        terms.append((term, normalized, item.match_mode, item.is_active))
    return terms


def assert_active_canonical_invariant(concept: TerminologyConcept) -> None:
    """Fail closed unless an active concept proves its canonical form strictly.

    All mutation entry points call this after state changes.  The function is
    deliberately independent from route code, so a later service/API addition
    cannot accidentally leave an active concept with a retrieval-only or
    disabled canonical term.
    """

    if not concept.is_active:
        return
    canonical_key = normalize_term_key(concept.canonical_term)
    canonical_terms = [
        term
        for term in tuple(concept.terms or ())
        if term.normalized_term == canonical_key
    ]
    if len(canonical_terms) != 1:
        raise HTTPException(
            status_code=422,
            detail="活跃术语概念必须保留唯一的规范术语严格等价定义",
        )
    canonical = canonical_terms[0]
    if canonical.match_mode != "strict_equivalent" or not canonical.is_active:
        raise HTTPException(
            status_code=422,
            detail="活跃术语概念的规范术语必须启用且为严格等价",
        )


def has_document_read_capability(user: User) -> bool:
    """Return whether a user may see or mutate document-scoped bindings."""

    return bool(
        getattr(user, "is_superadmin", False)
        or DOC_READ in set(getattr(user, "permissions", ()) or ())
    )


async def ensure_document_binding_authorized(
    *,
    user: User,
    db: AsyncSession,
    kb_id: uuid.UUID,
) -> None:
    """Require both document-read capability and the path KB data scope.

    A terminology manager can otherwise attach an alias to a document it is
    not allowed to inspect.  The path dependency already checks KB scope for
    normal requests; repeating the object-scope check here makes the dynamic
    body-dependent document rule explicit and testable.
    """

    if not has_document_read_capability(user):
        raise HTTPException(status_code=403, detail="文档范围术语绑定需要文档查看权限")
    await ensure_kb_access(user, kb_id, db)


async def read_stable_registry_view(
    *,
    read_revision: Callable[[], Awaitable[int]],
    read_graph: Callable[[], Awaitable[T]],
    max_attempts: int = _MAX_STABLE_READ_ATTEMPTS,
) -> tuple[int, T]:
    """Read ``state → graph → state`` and return only a stable KB view."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    for _ in range(max_attempts):
        before = await read_revision()
        graph = await read_graph()
        after = await read_revision()
        if before == after:
            return after, graph
    raise RegistryReadConsistencyError("registry revision changed while reading graph")


def _json_value(value: object) -> object:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _term_payload(term: TerminologyTerm) -> dict[str, object]:
    return {
        "id": _json_value(term.id),
        "concept_id": _json_value(term.concept_id),
        "kb_id": _json_value(term.kb_id),
        "term": term.term,
        "normalized_term": term.normalized_term,
        "match_mode": term.match_mode,
        "is_active": term.is_active,
        "created_by": _json_value(term.created_by),
        "updated_by": _json_value(term.updated_by),
        "created_at": _json_value(term.created_at),
        "updated_at": _json_value(term.updated_at),
    }


def _binding_payload(binding: TerminologyScopeBinding) -> dict[str, object]:
    return {
        "id": _json_value(binding.id),
        "concept_id": _json_value(binding.concept_id),
        "kb_id": _json_value(binding.kb_id),
        "document_id": _json_value(binding.document_id),
        "scope_product_key": binding.scope_product_key,
        "scope_version_key": binding.scope_version_key,
        "scope_project_key": binding.scope_project_key,
        "is_active": binding.is_active,
        "created_by": _json_value(binding.created_by),
        "updated_by": _json_value(binding.updated_by),
        "created_at": _json_value(binding.created_at),
        "updated_at": _json_value(binding.updated_at),
    }


def concept_graph_payload(concept: TerminologyConcept | None) -> dict[str, object]:
    """Return every persisted field needed to replay one concept mutation."""

    if concept is None:
        return {"concept": None, "terms": [], "bindings": []}
    terms = sorted(
        (_term_payload(term) for term in tuple(concept.terms or ())),
        key=lambda item: str(item["id"]),
    )
    bindings = sorted(
        (_binding_payload(binding) for binding in tuple(concept.scope_bindings or ())),
        key=lambda item: str(item["id"]),
    )
    return {
        "concept": {
            "id": _json_value(concept.id),
            "kb_id": _json_value(concept.kb_id),
            "code": concept.code,
            "canonical_term": concept.canonical_term,
            "description": concept.description,
            "is_active": concept.is_active,
            "created_by": _json_value(concept.created_by),
            "updated_by": _json_value(concept.updated_by),
            "created_at": _json_value(concept.created_at),
            "updated_at": _json_value(concept.updated_at),
        },
        "terms": terms,
        "bindings": bindings,
    }


def build_replayable_change_payload(
    *,
    action: str,
    kb_id: uuid.UUID,
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    """Deep-copy and JSON-normalize an immutable-in-storage event payload."""

    payload = {
        "schema_version": _REGISTRY_EVENT_SCHEMA_VERSION,
        "action": action,
        "kb_id": str(kb_id),
        "before": copy.deepcopy(before),
        "after": copy.deepcopy(after),
    }
    # JSONB persistence must never rely on an endpoint-owned serializer.  The
    # round trip validates all values and decouples the stored event from
    # mutable in-memory ORM dictionaries.
    try:
        return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError) as exc:  # pragma: no cover - programmer error
        raise ValueError("terminology registry event payload is not JSON-safe") from exc


def _concept_out(
    concept: TerminologyConcept,
    *,
    include_inactive: bool,
) -> TerminologyConceptOut:
    terms = [
        term
        for term in tuple(concept.terms or ())
        if include_inactive or term.is_active
    ]
    terms.sort(key=lambda item: (not item.is_active, item.normalized_term, str(item.id)))
    return TerminologyConceptOut(
        id=concept.id,
        kb_id=concept.kb_id,
        code=concept.code,
        canonical_term=concept.canonical_term,
        description=concept.description,
        is_active=concept.is_active,
        created_by=concept.created_by,
        updated_by=concept.updated_by,
        created_at=concept.created_at,
        updated_at=concept.updated_at,
        terms=[
            TerminologyTermOut(
                id=term.id,
                concept_id=term.concept_id,
                kb_id=term.kb_id,
                term=term.term,
                normalized_term=term.normalized_term,
                match_mode=term.match_mode,
                is_active=term.is_active,
                created_by=term.created_by,
                updated_by=term.updated_by,
                created_at=term.created_at,
                updated_at=term.updated_at,
            )
            for term in terms
        ],
    )


def _binding_out(
    binding: TerminologyScopeBinding,
    *,
    concept: TerminologyConceptOut | None,
) -> TerminologyScopeBindingOut:
    return TerminologyScopeBindingOut(
        id=binding.id,
        concept_id=binding.concept_id,
        kb_id=binding.kb_id,
        document_id=binding.document_id,
        scope_product_key=binding.scope_product_key,
        scope_version_key=binding.scope_version_key,
        scope_project_key=binding.scope_project_key,
        is_active=binding.is_active,
        created_by=binding.created_by,
        updated_by=binding.updated_by,
        created_at=binding.created_at,
        updated_at=binding.updated_at,
        concept=concept,
    )


class TerminologyRegistryMutationService:
    """The only writer for the terminology registry management API."""

    def __init__(
        self,
        *,
        db: AsyncSession,
        user: User,
        audit: AuditLogger | None,
        kb_id: uuid.UUID,
    ) -> None:
        self.db = db
        self.user = user
        self.audit = audit
        self.kb_id = kb_id

    async def _state(self, *, for_update: bool = False) -> TerminologyRegistryState:
        statement = (
            select(TerminologyRegistryState)
            .where(TerminologyRegistryState.kb_id == self.kb_id)
            # The second state read must not reuse an identity-map value after
            # another transaction commits a KB-local revision.
            .execution_options(populate_existing=True)
        )
        if for_update:
            statement = statement.with_for_update()
        state = (await self.db.execute(statement)).scalar_one_or_none()
        if state is None:
            raise HTTPException(
                status_code=503,
                detail="术语注册表尚未完成当前知识库的初始化，请确认数据库迁移已执行",
            )
        return state

    async def _concept(
        self,
        concept_id: uuid.UUID,
        *,
        for_update: bool = False,
    ) -> TerminologyConcept | None:
        statement = (
            select(TerminologyConcept)
            .options(
                selectinload(TerminologyConcept.terms),
                selectinload(TerminologyConcept.scope_bindings),
            )
            .where(
                TerminologyConcept.id == concept_id,
                TerminologyConcept.kb_id == self.kb_id,
            )
        )
        if for_update:
            statement = statement.with_for_update()
        return (await self.db.execute(statement)).scalar_one_or_none()

    async def _binding_concept(
        self,
        binding_id: uuid.UUID,
        *,
        for_update: bool = False,
    ) -> tuple[TerminologyConcept, TerminologyScopeBinding] | None:
        binding = (await self.db.execute(
            select(TerminologyScopeBinding.concept_id).where(
                TerminologyScopeBinding.id == binding_id,
                TerminologyScopeBinding.kb_id == self.kb_id,
            )
        )).scalar_one_or_none()
        if binding is None:
            return None
        concept = await self._concept(binding, for_update=for_update)
        if concept is None:  # Composite FK makes this impossible after migration.
            return None
        scoped_binding = next(
            (item for item in concept.scope_bindings if item.id == binding_id),
            None,
        )
        if scoped_binding is None:  # pragma: no cover - protects ORM loader drift
            return None
        return concept, scoped_binding

    async def _term_concept(
        self,
        concept_id: uuid.UUID,
        term_id: uuid.UUID,
        *,
        for_update: bool = False,
    ) -> tuple[TerminologyConcept, TerminologyTerm] | None:
        concept = await self._concept(concept_id, for_update=for_update)
        if concept is None:
            return None
        term = next((item for item in concept.terms if item.id == term_id), None)
        if term is None:
            return None
        return concept, term

    async def _validate_scoped_document(self, document_id: uuid.UUID | None) -> None:
        if document_id is None:
            return
        document = await self.db.get(Document, document_id)
        if document is None:
            raise HTTPException(status_code=400, detail="术语范围中的文档不存在")
        if document.kb_id != self.kb_id:
            raise HTTPException(status_code=400, detail="术语范围中的文档不属于当前知识库")

    async def _require_document_scope_permission(self) -> None:
        try:
            await ensure_document_binding_authorized(
                user=self.user,
                db=self.db,
                kb_id=self.kb_id,
            )
        except HTTPException:
            # The KB state row has already been locked by all write methods.
            # Release it even though no ORM mutation was made yet.
            await self.db.rollback()
            raise

    async def _authorize_document_binding(self, document_id: uuid.UUID | None) -> None:
        if document_id is None:
            return
        await self._require_document_scope_permission()
        try:
            await self._validate_scoped_document(document_id)
        except HTTPException:
            await self.db.rollback()
            raise

    async def _authorize_concept_mutation_scope(
        self,
        concept: TerminologyConcept,
    ) -> None:
        """Do not edit terms/concepts that apply to unreadable documents.

        A term update changes the semantics of every binding of that concept.
        Therefore a manager without ``doc:read`` cannot indirectly alter a
        document-scoped rule through the concept or one of its terms.
        """

        if any(binding.document_id is not None for binding in concept.scope_bindings):
            await self._require_document_scope_permission()

    async def _graph_rows(self) -> list[TerminologyConcept]:
        return (await self.db.execute(
            select(TerminologyConcept)
            .options(
                selectinload(TerminologyConcept.terms),
                selectinload(TerminologyConcept.scope_bindings),
            )
            .where(TerminologyConcept.kb_id == self.kb_id)
            .execution_options(populate_existing=True)
            .order_by(TerminologyConcept.created_at.asc(), TerminologyConcept.id.asc())
        )).scalars().unique().all()

    def _registry_from_graph(
        self,
        *,
        revision: int,
        concepts: list[TerminologyConcept],
        include_inactive: bool,
        include_document_scoped: bool = True,
    ) -> TerminologyRegistryOut:
        visible_concepts: list[TerminologyConcept] = []
        visible_bindings_by_concept: dict[uuid.UUID, list[TerminologyScopeBinding]] = {}
        for concept in concepts:
            if not include_inactive and not concept.is_active:
                continue
            all_bindings = tuple(concept.scope_bindings or ())
            visible_bindings = [
                binding
                for binding in all_bindings
                if (include_inactive or binding.is_active)
                and (include_document_scoped or binding.document_id is None)
            ]
            # A concept whose only scope is a document cannot be exposed to a
            # caller without doc:read, and an active concept without any
            # active applicable binding should not appear in the active view.
            if not visible_bindings and (
                (not include_inactive)
                or (not include_document_scoped and all_bindings)
            ):
                continue
            visible_concepts.append(concept)
            visible_bindings_by_concept[concept.id] = visible_bindings
        concept_outs = [
            _concept_out(concept, include_inactive=include_inactive)
            for concept in visible_concepts
        ]
        concept_by_id = {item.id: item for item in concept_outs}
        bindings: list[TerminologyScopeBindingOut] = []
        for concept in visible_concepts:
            for binding in visible_bindings_by_concept[concept.id]:
                bindings.append(_binding_out(binding, concept=concept_by_id[concept.id]))
        bindings.sort(key=lambda item: (str(item.created_at), str(item.id)))
        return TerminologyRegistryOut(
            kb_id=self.kb_id,
            registry_revision=revision,
            concepts=concept_outs,
            bindings=bindings,
        )

    async def read_registry(self, *, include_inactive: bool) -> TerminologyRegistryOut:
        async def read_revision() -> int:
            return int((await self._state()).revision)

        async def read_graph() -> list[TerminologyConcept]:
            return await self._graph_rows()

        try:
            revision, concepts = await read_stable_registry_view(
                read_revision=read_revision,
                read_graph=read_graph,
            )
        except RegistryReadConsistencyError as exc:
            raise HTTPException(
                status_code=503,
                detail="术语注册表正在并发更新，暂时无法取得一致视图，请稍后重试",
            ) from exc
        return self._registry_from_graph(
            revision=revision,
            concepts=concepts,
            include_inactive=include_inactive,
            include_document_scoped=has_document_read_capability(self.user),
        )

    async def _flush_or_conflict(self, *, detail: str) -> None:
        try:
            await self.db.flush()
        except IntegrityError as exc:
            await self.db.rollback()
            raise HTTPException(status_code=409, detail=detail) from exc

    async def _assert_invariant_or_rollback(
        self,
        concept: TerminologyConcept,
    ) -> None:
        """Do not leave an invalid pending ORM graph after a failed transition."""

        try:
            assert_active_canonical_invariant(concept)
        except HTTPException:
            await self.db.rollback()
            raise

    async def _commit_or_conflict(self, *, detail: str) -> None:
        try:
            await self.db.commit()
        except IntegrityError as exc:
            await self.db.rollback()
            raise HTTPException(status_code=409, detail=detail) from exc

    async def _no_op_result(
        self,
        *,
        state: TerminologyRegistryState,
        target_kind: str,
        target_id: uuid.UUID,
    ) -> TerminologyMutationOut:
        # The state row is locked, so the graph read is exactly this revision.
        registry = self._registry_from_graph(
            revision=state.revision,
            concepts=await self._graph_rows(),
            include_inactive=True,
            include_document_scoped=has_document_read_capability(self.user),
        )
        await self.db.rollback()  # release the lock; no business mutation happened
        return self._mutation_out(
            registry=registry,
            target_kind=target_kind,
            target_id=target_id,
        )

    def _mutation_out(
        self,
        *,
        registry: TerminologyRegistryOut,
        target_kind: str,
        target_id: uuid.UUID,
    ) -> TerminologyMutationOut:
        concept: TerminologyConceptOut | None = None
        term: TerminologyTermOut | None = None
        binding: TerminologyScopeBindingOut | None = None
        if target_kind == "concept":
            concept = next((item for item in registry.concepts if item.id == target_id), None)
        elif target_kind == "term":
            for candidate in registry.concepts:
                term = next((item for item in candidate.terms if item.id == target_id), None)
                if term is not None:
                    break
        elif target_kind == "binding":
            binding = next((item for item in registry.bindings if item.id == target_id), None)
        else:  # pragma: no cover - service-controlled enum
            raise ValueError("unsupported terminology mutation target")
        return TerminologyMutationOut(
            registry_revision=registry.registry_revision,
            registry=registry,
            concept=concept,
            term=term,
            binding=binding,
        )

    async def _record_and_commit(
        self,
        *,
        state: TerminologyRegistryState,
        action: str,
        target_kind: str,
        target_id: uuid.UUID,
        target_name: str,
        before: dict[str, object],
        after: dict[str, object],
        conflict_detail: str,
    ) -> TerminologyMutationOut:
        revision = int(state.revision) + 1
        state.revision = revision
        state.updated_by = self.user.id
        state.updated_at = now_utc()
        payload = build_replayable_change_payload(
            action=action,
            kb_id=self.kb_id,
            before=before,
            after=after,
        )
        self.db.add(TerminologyRegistryRevision(
            id=uuid.uuid4(),
            kb_id=self.kb_id,
            revision=revision,
            action=action,
            object_type=f"terminology_{target_kind}",
            object_id=str(target_id),
            change_payload=payload,
            created_by=self.user.id,
            created_at=now_utc(),
        ))
        # Ordinary operation audit and immutable registry event share the
        # transaction.  The audit has the full replay payload too, so a human
        # investigation does not depend on endpoint-written summaries.
        if self.audit is None:  # pragma: no cover - write endpoints always inject audit
            raise RuntimeError("terminology mutation requires an audit logger")
        self.audit.log(
            self.db,
            action,
            target_type=f"terminology_{target_kind}",
            target_id=target_id,
            target_name=target_name,
            detail={
                "kb_id": str(self.kb_id),
                "registry_revision": revision,
                "change_payload": payload,
            },
        )
        await self._flush_or_conflict(detail=conflict_detail)
        # This executes before commit while holding the KB state row lock, so
        # it is precisely the graph created by the returned revision.
        registry = self._registry_from_graph(
            revision=revision,
            concepts=await self._graph_rows(),
            include_inactive=True,
            include_document_scoped=has_document_read_capability(self.user),
        )
        result = self._mutation_out(
            registry=registry,
            target_kind=target_kind,
            target_id=target_id,
        )
        await self._commit_or_conflict(detail=conflict_detail)
        return result

    def _new_term(
        self,
        *,
        concept: TerminologyConcept,
        term: str,
        normalized_term: str,
        match_mode: str,
        is_active: bool,
        timestamp: datetime,
    ) -> TerminologyTerm:
        return TerminologyTerm(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=self.kb_id,
            term=term,
            normalized_term=normalized_term,
            match_mode=match_mode,
            is_active=is_active,
            created_by=self.user.id,
            updated_by=self.user.id,
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _ensure_canonical_for_active_concept(
        self,
        *,
        concept: TerminologyConcept,
        canonical_term: str,
        timestamp: datetime,
    ) -> bool:
        """Make an explicit canonical-term change preserve the active invariant."""

        term, normalized = normalize_reviewed_term(canonical_term, field="规范术语")
        existing = next(
            (item for item in concept.terms if item.normalized_term == normalized),
            None,
        )
        if existing is None:
            concept.terms.append(self._new_term(
                concept=concept,
                term=term,
                normalized_term=normalized,
                match_mode="strict_equivalent",
                is_active=True,
                timestamp=timestamp,
            ))
            return True
        if existing.match_mode != "strict_equivalent":
            raise HTTPException(
                status_code=422,
                detail="新规范术语当前被定义为仅检索匹配，请先将其改为严格等价后再设为规范术语",
            )
        if not existing.is_active:
            existing.is_active = True
            existing.updated_by = self.user.id
            existing.updated_at = timestamp
            return True
        return False

    async def create_concept(
        self,
        payload: TerminologyConceptCreate,
    ) -> TerminologyMutationOut:
        state = await self._state(for_update=True)
        code = normalize_concept_code(payload.code)
        if (await self.db.execute(
            select(TerminologyConcept.id).where(
                TerminologyConcept.kb_id == self.kb_id,
                TerminologyConcept.code == code,
            )
        )).scalar_one_or_none() is not None:
            await self.db.rollback()
            raise HTTPException(status_code=409, detail="术语概念编码已存在")
        await self._authorize_document_binding(payload.initial_binding.document_id)
        canonical_term, _ = normalize_reviewed_term(payload.canonical_term, field="规范术语")
        initial_terms = merge_initial_terms(
            canonical_term=canonical_term,
            supplied=payload.terms,
        )
        timestamp = now_utc()
        concept = TerminologyConcept(
            id=uuid.uuid4(),
            kb_id=self.kb_id,
            code=code,
            canonical_term=canonical_term,
            description=(payload.description or "").strip() or None,
            is_active=True,
            created_by=self.user.id,
            updated_by=self.user.id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        for term, normalized, match_mode, is_active in initial_terms:
            concept.terms.append(self._new_term(
                concept=concept,
                term=term,
                normalized_term=normalized,
                match_mode=match_mode,
                is_active=is_active,
                timestamp=timestamp,
            ))
        binding = TerminologyScopeBinding(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=self.kb_id,
            document_id=payload.initial_binding.document_id,
            **scope_values(payload.initial_binding, only_supplied=False),
            is_active=payload.initial_binding.is_active,
            created_by=self.user.id,
            updated_by=self.user.id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.scope_bindings.append(binding)
        self.db.add(concept)
        await self._assert_invariant_or_rollback(concept)
        await self._flush_or_conflict(detail="术语概念、术语或适用范围已存在")
        return await self._record_and_commit(
            state=state,
            action="terminology.concept.create",
            target_kind="concept",
            target_id=concept.id,
            target_name=concept.code,
            before=concept_graph_payload(None),
            after=concept_graph_payload(concept),
            conflict_detail="术语概念、术语或适用范围已存在",
        )

    async def update_concept(
        self,
        *,
        concept_id: uuid.UUID,
        payload: TerminologyConceptUpdate,
    ) -> TerminologyMutationOut:
        state = await self._state(for_update=True)
        concept = await self._concept(concept_id, for_update=True)
        if concept is None:
            await self.db.rollback()
            raise HTTPException(status_code=404, detail="当前知识库中不存在该术语概念")
        await self._authorize_concept_mutation_scope(concept)
        before = concept_graph_payload(concept)
        changed = False
        timestamp = now_utc()
        next_is_active = (
            payload.is_active
            if field_was_supplied(payload, "is_active")
            else concept.is_active
        )
        if field_was_supplied(payload, "canonical_term"):
            canonical, _ = normalize_reviewed_term(payload.canonical_term, field="规范术语")
            if canonical != concept.canonical_term:
                # A canonical-term update that activates the concept in this
                # same request is one logical transition, so it receives the
                # same invariant-preserving term creation as an active edit.
                if next_is_active:
                    self._ensure_canonical_for_active_concept(
                        concept=concept,
                        canonical_term=canonical,
                        timestamp=timestamp,
                    )
                concept.canonical_term = canonical
                changed = True
        if field_was_supplied(payload, "description"):
            description = (payload.description or "").strip() or None
            if description != concept.description:
                concept.description = description
                changed = True
        if next_is_active != concept.is_active:
            # Reactivation is intentionally fail-closed: it never silently
            # promotes an old retrieval-only alias into a strict equivalence.
            concept.is_active = next_is_active
            changed = True
        if not changed:
            return await self._no_op_result(
                state=state, target_kind="concept", target_id=concept.id
            )
        concept.updated_by = self.user.id
        concept.updated_at = timestamp
        await self._assert_invariant_or_rollback(concept)
        await self._flush_or_conflict(detail="术语概念更新与现有记录冲突")
        return await self._record_and_commit(
            state=state,
            action="terminology.concept.update",
            target_kind="concept",
            target_id=concept.id,
            target_name=concept.code,
            before=before,
            after=concept_graph_payload(concept),
            conflict_detail="术语概念更新与现有记录冲突",
        )

    async def create_term(
        self,
        *,
        concept_id: uuid.UUID,
        payload: TerminologyTermCreate,
    ) -> TerminologyMutationOut:
        state = await self._state(for_update=True)
        concept = await self._concept(concept_id, for_update=True)
        if concept is None:
            await self.db.rollback()
            raise HTTPException(status_code=404, detail="当前知识库中不存在该术语概念")
        await self._authorize_concept_mutation_scope(concept)
        before = concept_graph_payload(concept)
        term_text, normalized = normalize_reviewed_term(payload.term, field="术语")
        if any(item.normalized_term == normalized for item in concept.terms):
            await self.db.rollback()
            raise HTTPException(status_code=409, detail="该概念中已存在相同术语")
        timestamp = now_utc()
        term = self._new_term(
            concept=concept,
            term=term_text,
            normalized_term=normalized,
            match_mode=payload.match_mode,
            is_active=payload.is_active,
            timestamp=timestamp,
        )
        concept.terms.append(term)
        await self._assert_invariant_or_rollback(concept)
        await self._flush_or_conflict(detail="该概念中已存在相同术语")
        return await self._record_and_commit(
            state=state,
            action="terminology.term.create",
            target_kind="term",
            target_id=term.id,
            target_name=concept.code,
            before=before,
            after=concept_graph_payload(concept),
            conflict_detail="术语写入与现有记录冲突",
        )

    async def update_term(
        self,
        *,
        concept_id: uuid.UUID,
        term_id: uuid.UUID,
        payload: TerminologyTermUpdate,
    ) -> TerminologyMutationOut:
        state = await self._state(for_update=True)
        loaded = await self._term_concept(concept_id, term_id, for_update=True)
        if loaded is None:
            await self.db.rollback()
            raise HTTPException(status_code=404, detail="当前知识库中不存在该术语")
        concept, term = loaded
        await self._authorize_concept_mutation_scope(concept)
        before = concept_graph_payload(concept)
        next_term = term.term
        next_normalized = term.normalized_term
        if field_was_supplied(payload, "term"):
            next_term, next_normalized = normalize_reviewed_term(payload.term, field="术语")
            if next_normalized != term.normalized_term:
                if (
                    concept.is_active
                    and term.normalized_term == normalize_term_key(concept.canonical_term)
                ):
                    await self.db.rollback()
                    raise HTTPException(
                        status_code=422,
                        detail="活跃概念的规范术语必须通过概念接口修改，不能直接替换",
                    )
                if any(
                    item.id != term.id and item.normalized_term == next_normalized
                    for item in concept.terms
                ):
                    await self.db.rollback()
                    raise HTTPException(status_code=409, detail="该概念中已存在相同术语")
        next_match_mode = payload.match_mode if field_was_supplied(payload, "match_mode") else term.match_mode
        next_is_active = payload.is_active if field_was_supplied(payload, "is_active") else term.is_active
        if (
            concept.is_active
            and term.normalized_term == normalize_term_key(concept.canonical_term)
            and (next_match_mode != "strict_equivalent" or not next_is_active)
        ):
            await self.db.rollback()
            raise HTTPException(
                status_code=422,
                detail="活跃概念的规范术语必须保持启用且为严格等价",
            )
        changed = (
            next_term != term.term
            or next_normalized != term.normalized_term
            or next_match_mode != term.match_mode
            or next_is_active != term.is_active
        )
        if not changed:
            return await self._no_op_result(
                state=state, target_kind="term", target_id=term.id
            )
        timestamp = now_utc()
        term.term = next_term
        term.normalized_term = next_normalized
        term.match_mode = next_match_mode
        term.is_active = next_is_active
        term.updated_by = self.user.id
        term.updated_at = timestamp
        await self._assert_invariant_or_rollback(concept)
        await self._flush_or_conflict(detail="术语更新与现有记录冲突")
        return await self._record_and_commit(
            state=state,
            action="terminology.term.update",
            target_kind="term",
            target_id=term.id,
            target_name=concept.code,
            before=before,
            after=concept_graph_payload(concept),
            conflict_detail="术语更新与现有记录冲突",
        )

    async def create_binding(
        self,
        payload: TerminologyScopeBindingCreate,
    ) -> TerminologyMutationOut:
        state = await self._state(for_update=True)
        concept = await self._concept(payload.concept_id, for_update=True)
        if concept is None:
            await self.db.rollback()
            raise HTTPException(status_code=404, detail="当前知识库中不存在该术语概念")
        await self._authorize_concept_mutation_scope(concept)
        if payload.is_active and not concept.is_active:
            await self.db.rollback()
            raise HTTPException(status_code=422, detail="不能为停用的术语概念创建启用范围")
        await self._authorize_document_binding(payload.document_id)
        before = concept_graph_payload(concept)
        timestamp = now_utc()
        binding = TerminologyScopeBinding(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=self.kb_id,
            document_id=payload.document_id,
            **scope_values(payload, only_supplied=False),
            is_active=payload.is_active,
            created_by=self.user.id,
            updated_by=self.user.id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.scope_bindings.append(binding)
        await self._assert_invariant_or_rollback(concept)
        await self._flush_or_conflict(detail="该术语概念在当前适用范围内已存在绑定")
        return await self._record_and_commit(
            state=state,
            action="terminology.binding.create",
            target_kind="binding",
            target_id=binding.id,
            target_name=concept.code,
            before=before,
            after=concept_graph_payload(concept),
            conflict_detail="术语范围与现有绑定冲突",
        )

    async def update_binding(
        self,
        *,
        binding_id: uuid.UUID,
        payload: TerminologyScopeBindingUpdate,
    ) -> TerminologyMutationOut:
        state = await self._state(for_update=True)
        loaded = await self._binding_concept(binding_id, for_update=True)
        if loaded is None:
            await self.db.rollback()
            raise HTTPException(status_code=404, detail="当前知识库中不存在该术语范围")
        concept, binding = loaded
        await self._authorize_concept_mutation_scope(concept)
        next_document_id = (
            payload.document_id
            if field_was_supplied(payload, "document_id")
            else binding.document_id
        )
        if field_was_supplied(payload, "document_id"):
            await self._authorize_document_binding(next_document_id)
        before = concept_graph_payload(concept)
        changed = False
        if next_document_id != binding.document_id:
            binding.document_id = next_document_id
            changed = True
        for field, value in scope_values(payload, only_supplied=True).items():
            if value != getattr(binding, field):
                setattr(binding, field, value)
                changed = True
        next_active = payload.is_active if field_was_supplied(payload, "is_active") else binding.is_active
        if next_active != binding.is_active:
            if next_active and not concept.is_active:
                await self.db.rollback()
                raise HTTPException(status_code=422, detail="不能启用属于停用术语概念的范围")
            binding.is_active = next_active
            changed = True
        if not changed:
            return await self._no_op_result(
                state=state, target_kind="binding", target_id=binding.id
            )
        timestamp = now_utc()
        binding.updated_by = self.user.id
        binding.updated_at = timestamp
        await self._assert_invariant_or_rollback(concept)
        await self._flush_or_conflict(detail="术语范围与现有绑定冲突")
        return await self._record_and_commit(
            state=state,
            action="terminology.binding.update",
            target_kind="binding",
            target_id=binding.id,
            target_name=concept.code,
            before=before,
            after=concept_graph_payload(concept),
            conflict_detail="术语范围与现有绑定冲突",
        )
