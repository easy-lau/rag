"""Safe, request-local runtime view of the controlled terminology registry.

The management registry deliberately stores *reviewed terminology*, not global
synonyms.  This module is the narrow runtime boundary that turns a previously
authorised, already-read registry graph into retrieval variants and strict
evidence rewrites.

It is intentionally independent from SQLAlchemy and the retriever.  The
database adapter must first select only the caller-authorised knowledge bases,
then build :class:`RuntimeTerminologyBinding` values.  Keeping that boundary
explicit prevents a convenient alias lookup from accidentally becoming a
second, unauthorised document search path.

Safety properties encoded here:

* every alias remains constrained to its KB, optional document, and explicit
  product/version/project applicability scope;
* ``retrieval_only`` can improve recall, but can never rewrite an evidence
  target or produce strict terminology provenance;
* the same source spelling mapped to different concepts in overlapping scope
  is rejected per source spelling (fail closed, not arbitrary first-match);
* a degraded registry read returns no variants.  Callers keep their original
  retrieval plan, so terminology infrastructure can never erase normal recall;
* trace summaries contain only counts, statuses, revisions and fingerprints --
  never business query text, terms, concept names, KB ids or document ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Literal, Mapping

from core.rag_v2.contracts import AnswerRequirementV2
from core.terminology_contracts import (
    TerminologyBinding,
    TerminologyForm,
    normalize_scope_key,
    normalize_term_key,
    replace_exact_term,
)


TERMINOLOGY_RUNTIME_SCHEMA_VERSION = "terminology_runtime.v1"
RuntimeResolutionStatus = Literal["resolved", "empty", "degraded"]

_RUNTIME_STATUSES = frozenset({"resolved", "empty", "degraded"})
_FINGERPRINT_RE = re.compile(r"^[a-f0-9]{32,128}$")
_SAFE_REASON_RE = re.compile(r"^[a-z][a-z0-9_.:-]{0,119}$")
_MAX_BINDINGS = 96
_MAX_ALIAS_VARIANTS = 12


def _identifier(value: object, *, field: str, max_chars: int = 200) -> str:
    """Normalise an opaque runtime identity without treating it as business text."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or len(text) > max_chars:
        raise ValueError(f"{field} is invalid")
    return text


def _optional_identifier(
    value: object,
    *,
    field: str,
    max_chars: int = 200,
) -> str | None:
    if value is None:
        return None
    return _identifier(value, field=field, max_chars=max_chars)


def _normalised_kb_ids(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _identifier(value, field="authorized kb id")
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _normalised_revisions(values: Mapping[object, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_kb_id, raw_revision in values.items():
        kb_id = _identifier(raw_kb_id, field="registry revision kb id")
        if isinstance(raw_revision, bool):
            raise ValueError("registry revision is invalid")
        try:
            revision = int(raw_revision)
        except (TypeError, ValueError) as exc:
            raise ValueError("registry revision is invalid") from exc
        if revision < 0:
            raise ValueError("registry revision is invalid")
        result[kb_id] = revision
    return result


def _normalised_reason(value: object | None) -> str | None:
    if value is None:
        return None
    reason = str(value).strip().casefold()
    if not _SAFE_REASON_RE.fullmatch(reason):
        raise ValueError("terminology runtime reason is invalid")
    return reason


def _scope_matches(
    binding_value: str | None,
    request_value: str | None,
) -> bool:
    """Return whether a binding selector is safely applicable to a request.

    A registry selector is an allow-list restriction.  When a binding is
    product/version/project specific and the request has no matching explicit
    selector, there is insufficient proof that it applies, so it is excluded.
    A global binding (``None``) remains applicable to every request scope.
    """

    if binding_value is None:
        return True
    return request_value is not None and binding_value == normalize_scope_key(request_value)


def _scope_overlaps(left: str | None, right: str | None) -> bool:
    """Return whether two applicability selectors can apply at once."""

    return left is None or right is None or left == right


def _bounded_alias_count(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, min(int(value), _MAX_ALIAS_VARIANTS))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class RuntimeTerminologyBinding:
    """One registry binding with its complete runtime applicability boundary.

    ``binding`` holds the reviewed concept/forms.  This wrapper supplies the
    row-level KB/document/scope ownership that must survive until retrieval and
    evidence selection; flattening it into a global synonym list is forbidden.
    """

    binding: TerminologyBinding
    kb_id: str
    document_id: str | None = None
    scope_product_key: str | None = None
    scope_version_key: str | None = None
    scope_project_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, TerminologyBinding):
            raise ValueError("runtime terminology binding must wrap TerminologyBinding")
        object.__setattr__(self, "kb_id", _identifier(self.kb_id, field="runtime kb id"))
        object.__setattr__(
            self,
            "document_id",
            _optional_identifier(self.document_id, field="runtime document id"),
        )
        for field in (
            "scope_product_key",
            "scope_version_key",
            "scope_project_key",
        ):
            raw = getattr(self, field)
            normalised = normalize_scope_key(raw)
            if raw is not None and normalised is None:
                raise ValueError(f"{field} is invalid")
            object.__setattr__(self, field, normalised)

    @property
    def source_key(self) -> str:
        return normalize_term_key(self.binding.source_term)

    @property
    def concept_id(self) -> str:
        return self.binding.concept_id

    @property
    def requirement_id(self) -> str:
        return self.binding.requirement_id

    def matches_requirement(
        self,
        requirement: AnswerRequirementV2,
        *,
        scope_project_key: str | None = None,
    ) -> bool:
        """Check requirement ownership and all non-document scope selectors."""

        if not isinstance(requirement, AnswerRequirementV2):
            return False
        if requirement.role != "answer" or requirement.id != self.binding.requirement_id:
            return False
        return (
            _scope_matches(self.scope_product_key, requirement.scope_product)
            and _scope_matches(self.scope_version_key, requirement.scope_version)
            and _scope_matches(self.scope_project_key, scope_project_key)
        )

    def matches_evidence_scope(
        self,
        *,
        kb_id: object,
        doc_id: object | None,
    ) -> bool:
        """Require evidence to remain in the binding's KB/document boundary."""

        if _identifier(kb_id, field="evidence kb id") != self.kb_id:
            return False
        if self.document_id is None:
            return True
        if doc_id is None:
            return False
        return _identifier(doc_id, field="evidence document id") == self.document_id

    def overlaps(self, other: "RuntimeTerminologyBinding") -> bool:
        """Return whether two bindings can govern the same source occurrence."""

        if self.kb_id != other.kb_id:
            return False
        documents_overlap = (
            self.document_id is None
            or other.document_id is None
            or self.document_id == other.document_id
        )
        return documents_overlap and all((
            _scope_overlaps(self.scope_product_key, other.scope_product_key),
            _scope_overlaps(self.scope_version_key, other.scope_version_key),
            _scope_overlaps(self.scope_project_key, other.scope_project_key),
        ))

    def safe_summary(self) -> dict[str, object]:
        """Return trace-safe structural information with no business contents."""

        return {
            "has_document_scope": self.document_id is not None,
            "has_product_scope": self.scope_product_key is not None,
            "has_version_scope": self.scope_version_key is not None,
            "has_project_scope": self.scope_project_key is not None,
            "strict_equivalent": self.binding.strict_equivalent,
            "query_form_count": len(self.binding.query_forms),
            "evidence_form_count": len(self.binding.evidence_forms),
        }


@dataclass(frozen=True)
class RuntimeTerminologyQueryVariant:
    """A terminology-expanded physical search with its non-negotiable scope."""

    query: str
    requirement_id: str
    kb_ids: tuple[str, ...]
    document_ids: tuple[str, ...] | None
    rule_ids: tuple[str, ...]
    relation_strength: Literal["strict_equivalent", "retrieval_only"]
    scope_product_key: str | None = None
    scope_version_key: str | None = None
    scope_project_key: str | None = None

    def __post_init__(self) -> None:
        query = _identifier(self.query, field="terminology variant query", max_chars=1000)
        requirement_id = _identifier(
            self.requirement_id,
            field="terminology variant requirement id",
            max_chars=64,
        )
        kb_ids = _normalised_kb_ids(self.kb_ids)
        if not kb_ids:
            raise ValueError("terminology variant requires an authorized kb id")
        if self.document_ids is None:
            document_ids = None
        else:
            document_ids = tuple(dict.fromkeys(
                _identifier(value, field="terminology variant document id")
                for value in self.document_ids
            ))
            if not document_ids:
                raise ValueError("terminology variant document scope is invalid")
        rule_ids = tuple(dict.fromkeys(
            _identifier(value, field="terminology variant rule id", max_chars=128)
            for value in self.rule_ids
        ))
        if not rule_ids:
            raise ValueError("terminology variant requires rule provenance")
        relation_strength = str(self.relation_strength or "").strip().casefold()
        if relation_strength not in {"strict_equivalent", "retrieval_only"}:
            raise ValueError("terminology variant relation strength is invalid")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "kb_ids", kb_ids)
        object.__setattr__(self, "document_ids", document_ids)
        object.__setattr__(self, "rule_ids", rule_ids)
        object.__setattr__(self, "relation_strength", relation_strength)
        for field in (
            "scope_product_key",
            "scope_version_key",
            "scope_project_key",
        ):
            raw = getattr(self, field)
            normalised = normalize_scope_key(raw)
            if raw is not None and normalised is None:
                raise ValueError(f"{field} is invalid")
            object.__setattr__(self, field, normalised)

    def safe_summary(self) -> dict[str, object]:
        return {
            "kb_count": len(self.kb_ids),
            "document_scoped": self.document_ids is not None,
            "rule_count": len(self.rule_ids),
            "relation_strength": self.relation_strength,
            "has_product_scope": self.scope_product_key is not None,
            "has_version_scope": self.scope_version_key is not None,
            "has_project_scope": self.scope_project_key is not None,
        }


@dataclass(frozen=True)
class RuntimeTerminologyEvidenceRewrite:
    """A strict-only requirement wording accepted for evidence adjudication."""

    requirement: AnswerRequirementV2
    rule_ids: tuple[str, ...]
    kb_id: str
    document_id: str | None
    concept_id: str
    source_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.requirement, AnswerRequirementV2):
            raise ValueError("terminology evidence rewrite requires an answer requirement")
        if self.requirement.role != "answer":
            raise ValueError("terminology evidence rewrite must target an answer requirement")
        rule_ids = tuple(dict.fromkeys(
            _identifier(value, field="terminology evidence rule id", max_chars=128)
            for value in self.rule_ids
        ))
        if not rule_ids:
            raise ValueError("terminology evidence rewrite requires rule provenance")
        object.__setattr__(self, "rule_ids", rule_ids)
        object.__setattr__(self, "kb_id", _identifier(self.kb_id, field="evidence rewrite kb id"))
        object.__setattr__(
            self,
            "document_id",
            _optional_identifier(self.document_id, field="evidence rewrite document id"),
        )
        object.__setattr__(
            self,
            "concept_id",
            _identifier(self.concept_id, field="evidence rewrite concept id", max_chars=128),
        )
        source_key = normalize_term_key(self.source_key)
        if len(source_key) < 2:
            raise ValueError("evidence rewrite source key is invalid")
        object.__setattr__(self, "source_key", source_key)

    @property
    def description(self) -> str:
        """Compatibility-facing rewritten text without exposing a second model."""

        return self.requirement.description

    @property
    def id(self) -> str:
        return self.requirement.id

    def safe_summary(self) -> dict[str, object]:
        return {
            "rule_count": len(self.rule_ids),
            "document_scoped": self.document_id is not None,
        }


@dataclass(frozen=True)
class RuntimeTerminologyEvidenceMatch:
    """A strict registry form actually found in an authorised source chunk."""

    rewrite: RuntimeTerminologyEvidenceRewrite
    matched_form: TerminologyForm

    def __post_init__(self) -> None:
        if not isinstance(self.rewrite, RuntimeTerminologyEvidenceRewrite):
            raise ValueError("terminology evidence match requires a strict rewrite")
        if not isinstance(self.matched_form, TerminologyForm):
            raise ValueError("terminology evidence match requires a terminology form")
        if self.matched_form.relation_strength != "strict_equivalent":
            raise ValueError("retrieval-only form cannot become evidence")

    @property
    def requirement(self) -> AnswerRequirementV2:
        return self.rewrite.requirement

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.rewrite.rule_ids, self.matched_form.rule_id)))

    def safe_summary(self) -> dict[str, object]:
        return {"rule_count": len(self.rule_ids)}


def _strict_form_rule_ids(
    binding: TerminologyBinding,
    *,
    target_form: TerminologyForm,
) -> tuple[str, ...]:
    """Return only rule ids actually needed for one strict rewrite."""

    source_key = normalize_term_key(binding.source_term)
    source_form = next(
        (
            form
            for form in binding.evidence_forms
            if normalize_term_key(form.term) == source_key
            and form.relation_strength == "strict_equivalent"
        ),
        None,
    )
    if source_form is None or target_form.relation_strength != "strict_equivalent":
        return ()
    return tuple(dict.fromkeys((
        *binding.scope_binding_ids,
        source_form.rule_id,
        target_form.rule_id,
    )))


def _binding_order_key(item: RuntimeTerminologyBinding) -> tuple[str, ...]:
    return (
        item.kb_id,
        item.document_id or "",
        item.scope_product_key or "",
        item.scope_version_key or "",
        item.scope_project_key or "",
        item.binding.requirement_id,
        item.binding.concept_id,
        item.source_key,
    )


def _ambiguous_source_keys(
    bindings: Iterable[RuntimeTerminologyBinding],
) -> frozenset[tuple[str, str]]:
    """Find only truly overlapping multi-concept source spellings.

    Two bindings for the same spelling in different KBs, documents, products
    or versions are not ambiguous if their scopes cannot co-apply.  A global
    binding and a narrower binding *do* overlap, so they fail closed until the
    registry owner makes the intended meaning explicit.
    """

    ordered = tuple(sorted(bindings, key=_binding_order_key))
    ambiguous: set[tuple[str, str]] = set()
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if left.kb_id != right.kb_id:
                break
            if left.source_key != right.source_key:
                continue
            if left.concept_id == right.concept_id:
                continue
            if left.overlaps(right):
                ambiguous.add((left.kb_id, left.source_key))
    return frozenset(ambiguous)


@dataclass(frozen=True)
class TerminologyRuntimeResolution:
    """Immutable terminology decision for exactly one authorised RAG plan.

    The resolution is deliberately not a drop-in ``TerminologySnapshot``.
    That older snapshot cannot preserve per-variant KB/document limits, so
    converting to it would silently widen scope.  The execution integration
    must consume :meth:`retrieval_variants` directly.
    """

    plan_fingerprint: str
    scope_fingerprint: str
    registry_revisions: Mapping[str, int]
    status: RuntimeResolutionStatus
    bindings: tuple[RuntimeTerminologyBinding, ...] = ()
    authorized_kb_ids: tuple[str, ...] = ()
    reason: str | None = None
    diagnostics: tuple[str, ...] = ()
    schema_version: str = TERMINOLOGY_RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TERMINOLOGY_RUNTIME_SCHEMA_VERSION:
            raise ValueError("unsupported terminology runtime schema version")
        plan_fingerprint = _identifier(
            self.plan_fingerprint,
            field="terminology runtime plan fingerprint",
            max_chars=128,
        )
        scope_fingerprint = _identifier(
            self.scope_fingerprint,
            field="terminology runtime scope fingerprint",
            max_chars=128,
        )
        if not _FINGERPRINT_RE.fullmatch(plan_fingerprint):
            raise ValueError("terminology runtime plan fingerprint is invalid")
        if not _FINGERPRINT_RE.fullmatch(scope_fingerprint):
            raise ValueError("terminology runtime scope fingerprint is invalid")
        status = str(self.status or "").strip().casefold()
        if status not in _RUNTIME_STATUSES:
            raise ValueError("terminology runtime status is invalid")
        revisions = _normalised_revisions(self.registry_revisions)
        authorized_kb_ids = _normalised_kb_ids(self.authorized_kb_ids)
        bindings = tuple(self.bindings)
        if len(bindings) > _MAX_BINDINGS:
            raise ValueError("terminology runtime has too many bindings")
        if any(not isinstance(item, RuntimeTerminologyBinding) for item in bindings):
            raise ValueError("terminology runtime bindings are invalid")
        if len({
            (
                item.kb_id,
                item.document_id,
                item.scope_product_key,
                item.scope_version_key,
                item.scope_project_key,
                item.binding.requirement_id,
                item.binding.concept_id,
                item.source_key,
            )
            for item in bindings
        }) != len(bindings):
            raise ValueError("terminology runtime contains duplicate bindings")
        if status == "resolved":
            if not authorized_kb_ids:
                raise ValueError("resolved terminology runtime requires authorized kb ids")
            if not bindings:
                raise ValueError("resolved terminology runtime requires bindings")
            if any(kb_id not in authorized_kb_ids for kb_id in revisions):
                raise ValueError("terminology runtime revision is outside authorized kb ids")
            if any(item.kb_id not in authorized_kb_ids for item in bindings):
                raise ValueError("terminology runtime binding is outside authorized kb ids")
            if any(item.kb_id not in revisions for item in bindings):
                raise ValueError("terminology runtime binding has no registry revision")
        elif bindings:
            raise ValueError("empty/degraded terminology runtime cannot carry bindings")
        reason = _normalised_reason(self.reason)
        diagnostics = tuple(dict.fromkeys(
            _normalised_reason(value) or "" for value in self.diagnostics
        ))
        if any(not value for value in diagnostics):
            raise ValueError("terminology runtime diagnostic is invalid")
        ambiguous = _ambiguous_source_keys(bindings)
        if ambiguous and "ambiguous_source_term" not in diagnostics:
            diagnostics = (*diagnostics, "ambiguous_source_term")
        object.__setattr__(self, "plan_fingerprint", plan_fingerprint)
        object.__setattr__(self, "scope_fingerprint", scope_fingerprint)
        object.__setattr__(self, "registry_revisions", dict(revisions))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "authorized_kb_ids", authorized_kb_ids)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "_ambiguous_source_keys", ambiguous)

    @classmethod
    def degraded(
        cls,
        *,
        plan_fingerprint: str,
        scope_fingerprint: str,
        reason: str = "registry_read_failed",
        authorized_kb_ids: Iterable[object] = (),
        registry_revisions: Mapping[object, object] | None = None,
    ) -> "TerminologyRuntimeResolution":
        """Return a safe no-augmentation decision after a registry failure."""

        return cls(
            plan_fingerprint=plan_fingerprint,
            scope_fingerprint=scope_fingerprint,
            registry_revisions=dict(registry_revisions or {}),
            status="degraded",
            authorized_kb_ids=_normalised_kb_ids(authorized_kb_ids),
            reason=reason,
        )

    @classmethod
    def empty(
        cls,
        *,
        plan_fingerprint: str,
        scope_fingerprint: str,
        authorized_kb_ids: Iterable[object],
        registry_revisions: Mapping[object, object],
        reason: str | None = None,
    ) -> "TerminologyRuntimeResolution":
        """Return a successfully read registry with no applicable bindings."""

        return cls(
            plan_fingerprint=plan_fingerprint,
            scope_fingerprint=scope_fingerprint,
            registry_revisions=dict(registry_revisions),
            status="empty",
            authorized_kb_ids=_normalised_kb_ids(authorized_kb_ids),
            reason=reason,
        )

    @property
    def degraded_or_empty(self) -> bool:
        return self.status != "resolved"

    @property
    def ambiguous_source_key_count(self) -> int:
        return len(self._ambiguous_source_keys)

    def _binding_is_usable(
        self,
        binding: RuntimeTerminologyBinding,
        requirement: AnswerRequirementV2,
        *,
        scope_project_key: str | None,
    ) -> bool:
        return (
            self.status == "resolved"
            and binding.kb_id in self.authorized_kb_ids
            and (binding.kb_id, binding.source_key) not in self._ambiguous_source_keys
            and binding.matches_requirement(
                requirement,
                scope_project_key=scope_project_key,
            )
        )

    def retrieval_variants(
        self,
        *,
        requirement: AnswerRequirementV2,
        maximum_aliases: int,
        scope_project_key: str | None = None,
    ) -> tuple[RuntimeTerminologyQueryVariant, ...]:
        """Build strictly scoped alias searches for one answer requirement.

        Original retrieval is intentionally absent from this return value: the
        regular task executor owns it.  Therefore an empty/degraded resolution
        means *no extra searches*, not ``no retrieval``.
        """

        if not isinstance(requirement, AnswerRequirementV2) or requirement.role != "answer":
            return ()
        limit = _bounded_alias_count(maximum_aliases)
        if limit == 0 or self.status != "resolved":
            return ()
        variants: list[RuntimeTerminologyQueryVariant] = []
        seen: set[tuple[object, ...]] = set()
        for item in sorted(self.bindings, key=_binding_order_key):
            if not self._binding_is_usable(
                item,
                requirement,
                scope_project_key=scope_project_key,
            ):
                continue
            for form in item.binding.query_alias_forms():
                rendered = replace_exact_term(
                    requirement.description,
                    item.binding.source_term,
                    form.term,
                )
                if not rendered:
                    continue
                rule_ids = tuple(dict.fromkeys((
                    *item.binding.scope_binding_ids,
                    form.rule_id,
                )))
                key = (
                    normalize_term_key(rendered),
                    item.kb_id,
                    item.document_id,
                    item.scope_product_key,
                    item.scope_version_key,
                    item.scope_project_key,
                    item.binding.requirement_id,
                    form.rule_id,
                )
                if key in seen:
                    continue
                seen.add(key)
                variants.append(RuntimeTerminologyQueryVariant(
                    query=rendered,
                    requirement_id=requirement.id,
                    kb_ids=(item.kb_id,),
                    document_ids=(item.document_id,) if item.document_id else None,
                    rule_ids=rule_ids,
                    relation_strength=form.relation_strength,
                    scope_product_key=item.scope_product_key,
                    scope_version_key=item.scope_version_key,
                    scope_project_key=item.scope_project_key,
                ))
                if len(variants) >= limit:
                    return tuple(variants)
        return tuple(variants)

    def evidence_rewrites(
        self,
        *,
        requirement: AnswerRequirementV2,
        kb_id: object,
        doc_id: object | None,
        scope_project_key: str | None = None,
    ) -> tuple[RuntimeTerminologyEvidenceRewrite, ...]:
        """Return strict-only alternate phrasings for one evidence location."""

        if not isinstance(requirement, AnswerRequirementV2) or requirement.role != "answer":
            return ()
        try:
            candidate_kb_id = _identifier(kb_id, field="evidence kb id")
        except ValueError:
            return ()
        rewrites: list[RuntimeTerminologyEvidenceRewrite] = []
        seen: set[tuple[str, tuple[str, ...], str, str | None]] = set()
        for item in sorted(self.bindings, key=_binding_order_key):
            if (
                not item.binding.strict_equivalent
                or not self._binding_is_usable(
                    item,
                    requirement,
                    scope_project_key=scope_project_key,
                )
                or item.kb_id != candidate_kb_id
            ):
                continue
            try:
                in_scope = item.matches_evidence_scope(kb_id=candidate_kb_id, doc_id=doc_id)
            except ValueError:
                in_scope = False
            if not in_scope:
                continue
            source_key = normalize_term_key(item.binding.source_term)
            for form in item.binding.evidence_forms:
                if (
                    form.relation_strength != "strict_equivalent"
                    or normalize_term_key(form.term) == source_key
                ):
                    continue
                rendered = replace_exact_term(
                    requirement.description,
                    item.binding.source_term,
                    form.term,
                )
                rule_ids = _strict_form_rule_ids(item.binding, target_form=form)
                if not rendered or not rule_ids:
                    continue
                key = (rendered, rule_ids, item.kb_id, item.document_id)
                if key in seen:
                    continue
                seen.add(key)
                rewrites.append(RuntimeTerminologyEvidenceRewrite(
                    requirement=replace(requirement, description=rendered),
                    rule_ids=rule_ids,
                    kb_id=item.kb_id,
                    document_id=item.document_id,
                    concept_id=item.binding.concept_id,
                    source_key=item.source_key,
                ))
        return tuple(rewrites)

    def evidence_match(
        self,
        *,
        requirement: AnswerRequirementV2,
        kb_id: object,
        doc_id: object | None,
        content: object,
        scope_project_key: str | None = None,
    ) -> RuntimeTerminologyEvidenceMatch | None:
        """Return a strict, source-present terminology proof for one chunk.

        A candidate containing only a retrieval-only alias is deliberately not
        enough.  The matched spelling must be a reviewed strict evidence form
        and the candidate must be within the exact binding scope.
        """

        for rewrite in self.evidence_rewrites(
            requirement=requirement,
            kb_id=kb_id,
            doc_id=doc_id,
            scope_project_key=scope_project_key,
        ):
            for item in self.bindings:
                if (
                    item.kb_id != rewrite.kb_id
                    or item.document_id != rewrite.document_id
                    or item.binding.requirement_id != requirement.id
                    or item.binding.concept_id != rewrite.concept_id
                    or item.source_key != rewrite.source_key
                    or not item.binding.strict_equivalent
                ):
                    continue
                form = item.binding.evidence_match(content)
                if form is not None and form.relation_strength == "strict_equivalent":
                    return RuntimeTerminologyEvidenceMatch(
                        rewrite=rewrite,
                        matched_form=form,
                    )
        return None

    def trace_summary(self) -> dict[str, object]:
        """Return a content-free payload suitable for ``rag.trace`` events."""

        revision_values = sorted(set(self.registry_revisions.values()))
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "plan_fingerprint": self.plan_fingerprint,
            "scope_fingerprint": self.scope_fingerprint,
            "authorized_kb_count": len(self.authorized_kb_ids),
            "registry_revision_count": len(self.registry_revisions),
            "registry_revisions": revision_values,
            "binding_count": len(self.bindings),
            "ambiguous_source_key_count": self.ambiguous_source_key_count,
            "diagnostic_count": len(self.diagnostics),
        }

    safe_summary = trace_summary


def build_runtime_terminology_resolution(
    *,
    plan_fingerprint: str,
    scope_fingerprint: str,
    authorized_kb_ids: Iterable[object],
    registry_revisions: Mapping[object, object],
    bindings: Iterable[RuntimeTerminologyBinding],
) -> TerminologyRuntimeResolution:
    """Build a resolution from an already-authorised registry read.

    This is the single fail-closed constructor intended for runtime adapters.
    Invalid/unauthorised rows result in ``degraded`` rather than a partial
    global alias list.  The caller then performs its baseline original-query
    retrieval as usual.
    """

    try:
        allowed = _normalised_kb_ids(authorized_kb_ids)
        # Registry state from an unselected KB is neither needed nor safe to
        # retain in this request-local view.  Dropping it here prevents a
        # trace summary from becoming a side channel for another KB's state.
        revisions = {
            kb_id: revision
            for kb_id, revision in _normalised_revisions(registry_revisions).items()
            if kb_id in allowed
        }
        runtime_bindings = tuple(bindings)
        if any(not isinstance(item, RuntimeTerminologyBinding) for item in runtime_bindings):
            raise ValueError("runtime terminology bindings are invalid")
        if any(item.kb_id not in allowed for item in runtime_bindings):
            raise ValueError("runtime terminology authorization boundary violation")
        if any(item.kb_id not in revisions for item in runtime_bindings):
            raise ValueError("runtime terminology revision is missing")
        if not runtime_bindings:
            return TerminologyRuntimeResolution.empty(
                plan_fingerprint=plan_fingerprint,
                scope_fingerprint=scope_fingerprint,
                authorized_kb_ids=allowed,
                registry_revisions=revisions,
                reason="no_applicable_binding",
            )
        return TerminologyRuntimeResolution(
            plan_fingerprint=plan_fingerprint,
            scope_fingerprint=scope_fingerprint,
            registry_revisions=revisions,
            status="resolved",
            bindings=runtime_bindings,
            authorized_kb_ids=allowed,
        )
    except (TypeError, ValueError):
        # Do not include exception text: it can contain a raw KB/document id
        # or database value.  The trace receives only this stable status code.
        return TerminologyRuntimeResolution.degraded(
            plan_fingerprint=plan_fingerprint,
            scope_fingerprint=scope_fingerprint,
            authorized_kb_ids=(),
            registry_revisions={},
            reason="registry_runtime_validation_failed",
        )


__all__ = [
    "TERMINOLOGY_RUNTIME_SCHEMA_VERSION",
    "RuntimeResolutionStatus",
    "RuntimeTerminologyBinding",
    "RuntimeTerminologyEvidenceMatch",
    "RuntimeTerminologyEvidenceRewrite",
    "RuntimeTerminologyQueryVariant",
    "TerminologyRuntimeResolution",
    "build_runtime_terminology_resolution",
]
