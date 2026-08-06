"""Resolve version/project ambiguity from the caller's authorized document scope.

This module is intentionally independent from retrieval ranking.  A route that
names a product but omits a version must be checked against the documents the
caller is actually allowed to read before any vector/keyword candidate can
choose one version on the user's behalf.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.clarification import ClarificationContract
from core.evidence_ambiguity import query_requests_all_scopes
from core.query_constraints import (
    canonical_product_name,
    extract_document_constraint_identity,
    extract_query_constraints,
)
from core.rag_v2.contracts import QueryPlanV2
from models.db_models import Document, DocumentChunk


_MAX_SCOPE_DOCUMENTS = 2000
_MAX_SCOPE_CHOICES = 8
_VERSION_NUMBER_RE = re.compile(r"\d+(?:\.\d+){0,3}")


@dataclass(frozen=True)
class AuthorizedScopeChoice:
    """One server-owned selectable applicability value."""

    key: str
    product: str
    version: str
    kb_ids: tuple[str, ...]
    doc_ids: tuple[str, ...]

    @property
    def label(self) -> str:
        # Keep the version cue explicit.  Besides being clearer to users, this
        # lets the ordinary source-span parser recognize catalog-derived
        # product names that are not present in a static alias registry.
        return f"{self.product} 版本 {self.version}"

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.version,
            "products": [self.product],
            "versions": [self.version],
            "kb_ids": list(self.kb_ids),
            "doc_ids": list(self.doc_ids),
        }


@dataclass(frozen=True)
class AuthorizedScopeClarification:
    """Structured ambiguity facts derived from authorized scope only."""

    dimension: str
    choices: tuple[AuthorizedScopeChoice, ...]
    reason: str = "multiple_authorized_versions"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "authorized_scope_clarification.v1",
            "dimension": self.dimension,
            "reason": self.reason,
            "choices": [choice.to_dict() for choice in self.choices],
        }

    def to_contract(self) -> ClarificationContract:
        return ClarificationContract(
            adapter="semantic",
            dimension=self.dimension,
            reason_code=self.reason,
            selection_mode="choice" if self.choices else "refine",
            choices=tuple(choice.to_dict() for choice in self.choices),
        )


def _version_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"[.]", str(value or "").strip()):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.casefold()))
    return tuple(parts)


def _required_product_scopes(plan: QueryPlanV2) -> tuple[tuple[str, str], ...]:
    values: dict[str, tuple[str, str]] = {}
    for requirement in plan.requirements:
        if requirement.role != "answer" or requirement.importance != "required":
            continue
        scope = requirement.applicability_scope
        if scope is None or not scope.product or scope.has_version_constraint:
            continue
        product = str(scope.product).strip()
        canonical = canonical_product_name(product)
        if canonical:
            values.setdefault(canonical.casefold(), (product, canonical))
    return tuple(values.values())


def _has_required_version_scope(plan: QueryPlanV2) -> bool:
    return any(
        requirement.role == "answer"
        and requirement.importance == "required"
        and requirement.applicability_scope is not None
        and requirement.applicability_scope.has_version_constraint
        for requirement in plan.requirements
    )


def _catalog_product_scopes(
    query: str,
    identities: Sequence[Any],
) -> tuple[tuple[str, str], ...]:
    """Match exact source-declared product names in the current query.

    This is the extensibility boundary for products unknown to the static
    compatibility alias registry.  Values originate in authorized document
    metadata/content and are matched literally after cosmetic normalization;
    no fuzzy/model-generated identity can expand the request scope.
    """

    normalized_query = canonical_product_name(query)
    if not normalized_query:
        return ()
    matched: dict[str, tuple[str, str]] = {}
    for identity in identities:
        for raw_product in identity.products:
            product = str(raw_product or "").strip()
            canonical = canonical_product_name(product)
            if len(canonical) < 2 or canonical not in normalized_query:
                continue
            matched.setdefault(canonical.casefold(), (product, canonical))
    return tuple(matched.values())


def _candidate_from_catalog_row(row: Any) -> dict[str, object]:
    doc_id, kb_id, filename, tags, content, metadata = row
    return {
        "kb_id": kb_id,
        "doc_id": doc_id,
        "filename": filename,
        "doc_tags": list(tags or []),
        "metadata": dict(metadata or {}),
        "content": str(content or ""),
    }


async def resolve_authorized_scope_clarification(
    db: AsyncSession,
    *,
    plan: QueryPlanV2,
    query: str,
    kb_ids: Sequence[Any],
) -> AuthorizedScopeClarification | None:
    """Return a version picker when an authorized product has alternatives.

    This is deliberately a pre-retrieval gate.  It only runs for a required
    answer scope with an explicit product and no explicit version, and it does
    not inspect retrieval scores or ask a model to select documents.  Broad
    comparison/all-scope questions remain executable by design.
    """

    if not isinstance(plan, QueryPlanV2) or not kb_ids:
        return None
    if plan.needs_clarification or query_requests_all_scopes(query):
        return None
    # A whole-document overview is version-neutral unless the user supplies a
    # version.  Configuration/procedure/fact requests remain version-sensitive.
    if plan.answer_shape == "overview":
        return None
    # A source-authored explicit version is already a closed user choice.  It
    # must not be reopened merely because the selected catalog has siblings.
    if (
        extract_query_constraints(query).has_version_constraint
        or _has_required_version_scope(plan)
    ):
        return None
    scopes = _required_product_scopes(plan)

    authorized_kb_ids = tuple(dict.fromkeys(kb_ids))
    authorized_kb_keys = {str(value) for value in authorized_kb_ids}
    statement = (
        # Keep catalog disambiguation cheap: ORM entity loading would also
        # materialize Document.raw_content and DocumentChunk.embedding, neither
        # of which participates in applicability identity.
        select(
            Document.id,
            Document.kb_id,
            Document.filename,
            Document.tags,
            DocumentChunk.content,
            DocumentChunk.metadata_,
        )
        .outerjoin(
            DocumentChunk,
            and_(
                DocumentChunk.doc_id == Document.id,
                DocumentChunk.kb_id == Document.kb_id,
                DocumentChunk.chunk_index == 0,
            ),
        )
        .where(
            Document.kb_id.in_(authorized_kb_ids),
            Document.is_active.is_(True),
            Document.status == "ready",
        )
        .order_by(Document.id.asc())
        .limit(_MAX_SCOPE_DOCUMENTS)
    )
    rows = list((await db.execute(statement)).all())
    if not rows:
        return None

    catalog_entries: list[tuple[dict[str, object], Any]] = []
    for row in rows:
        candidate = _candidate_from_catalog_row(row)
        # The SQL predicate is the primary boundary; keep the same check in
        # the projection so a future query rewrite/test double cannot promote
        # a row outside the caller's selected and authorized KB set.
        if str(candidate["kb_id"]) not in authorized_kb_keys:
            continue
        identity = extract_document_constraint_identity(candidate)
        catalog_entries.append((candidate, identity))

    if not scopes:
        scopes = _catalog_product_scopes(
            query,
            [identity for _, identity in catalog_entries],
        )
    if not scopes:
        return None

    choices_by_version: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate, identity in catalog_entries:
        if not identity.versions:
            continue
        identity_products = {
            canonical_product_name(value).casefold()
            for value in identity.products
            if canonical_product_name(value)
        }
        for product, canonical in scopes:
            if canonical.casefold() not in identity_products:
                continue
            for version in identity.versions:
                normalized_version = str(version or "").strip()
                if not normalized_version or not _VERSION_NUMBER_RE.fullmatch(
                    normalized_version
                ):
                    continue
                choice_key = (
                    canonical.casefold(),
                    normalized_version.casefold(),
                )
                entry = choices_by_version.setdefault(
                    choice_key,
                    {
                        "product": product,
                        "version": normalized_version,
                        "kb_ids": set(),
                        "doc_ids": set(),
                    },
                )
                entry["kb_ids"].add(str(candidate["kb_id"]))
                entry["doc_ids"].add(str(candidate["doc_id"]))

    if len(choices_by_version) < 2:
        return None
    entries = sorted(
        choices_by_version.values(),
        key=lambda item: (
            canonical_product_name(str(item["product"])).casefold(),
            _version_sort_key(str(item["version"])),
        ),
    )
    if len(entries) > _MAX_SCOPE_CHOICES:
        return AuthorizedScopeClarification(
            dimension="product_version",
            choices=(),
            reason="too_many_authorized_versions",
        )
    choices = tuple(
        AuthorizedScopeChoice(
            key=f"scope{index}",
            product=str(item["product"]),
            version=str(item["version"]),
            kb_ids=tuple(sorted(item["kb_ids"])),
            doc_ids=tuple(sorted(item["doc_ids"])),
        )
        for index, item in enumerate(entries, start=1)
    )
    return AuthorizedScopeClarification(
        dimension="product_version",
        choices=choices,
    )


__all__ = [
    "AuthorizedScopeChoice",
    "AuthorizedScopeClarification",
    "resolve_authorized_scope_clarification",
]
