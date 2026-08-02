"""Immutable, request-local contracts for controlled terminology.

The query planner deliberately keeps the user's wording intact.  This module
is the separate, auditable boundary for a *previously approved* terminology
registry to contribute narrowly scoped retrieval variants and evidence forms.
It contains no database or model calls: once a snapshot is built, every
downstream RAG stage receives the same immutable vocabulary decision.

Two rules are intentionally encoded here rather than left to callers:

* only a binding that is already tied to one answer requirement can produce a
  variant; bridge subjects, product/version scopes and resolved bridge values
  are never rewritten;
* ``retrieval_only`` terms can widen recall but can never become an evidence
  form, so a weak alias cannot promote a chunk into an answer source.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal


TERMINOLOGY_SNAPSHOT_SCHEMA_VERSION = "terminology_snapshot.v1"

TerminologyRelationStrength = Literal["strict_equivalent", "retrieval_only"]
TerminologyResolutionStatus = Literal["resolved", "empty", "degraded"]
TerminologyVariantOrigin = Literal["original", "terminology_alias"]

_RELATION_STRENGTHS = frozenset({"strict_equivalent", "retrieval_only"})
_RESOLUTION_STATUSES = frozenset({"resolved", "empty", "degraded"})
_REQUIREMENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_RULE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_MAX_TERM_CHARS = 120
_MAX_FORMS = 12
_MAX_BINDINGS = 24


def _text(value: object, *, field: str, max_chars: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    if len(normalized) > max_chars:
        raise ValueError(f"{field} exceeds {max_chars} characters")
    return normalized


def _key(value: object) -> str:
    return normalize_term_key(value)


def normalize_term_key(value: object) -> str:
    """Return the registry-wide identity key for a reviewed term.

    The key is used for exact-form comparison and uniqueness only.  It does
    not assert that two different terms have the same business meaning; that
    assertion remains an explicit, reviewed registry ``match_mode``.
    """

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", "", normalized).casefold()


def normalize_scope_key(value: object) -> str | None:
    """Return the shared registry scope comparison key.

    Scope values are not free-text retrieval terms.  They need a stable,
    conservative equality key shared by write-time validation and request-time
    resolution, without pretending that distinct product/project names are
    synonyms.
    """

    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return normalized or None


def _unique_forms(
    values: Iterable["TerminologyForm"],
    *,
    field: str,
) -> tuple["TerminologyForm", ...]:
    forms: list[TerminologyForm] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, TerminologyForm):
            raise ValueError(f"{field} must contain TerminologyForm values")
        key = _key(value.term)
        if key in seen:
            continue
        seen.add(key)
        forms.append(value)
        if len(forms) > _MAX_FORMS:
            raise ValueError(f"{field} has too many forms")
    return tuple(forms)


def term_occurs(text: object, term: object) -> bool:
    """Match a registry term without heuristic CJK subsequences.

    Registry terms are human-reviewed strings.  A literal occurrence is
    therefore the only matching primitive here: ``餐补`` cannot be inferred
    from ``聚餐补录`` or another character subsequence.  ASCII terms retain
    token boundaries to avoid treating ``R1`` as ``R10``.
    """

    normalized_text = _key(text)
    normalized_term = _key(term)
    if len(normalized_term) < 2 or not normalized_text:
        return False
    escaped = re.escape(normalized_term)
    if re.fullmatch(r"[a-z0-9_.+/-]+", normalized_term):
        pattern = re.compile(
            rf"(?<![a-z0-9_.+/-]){escaped}(?![a-z0-9_.+/-])",
            re.IGNORECASE,
        )
        return pattern.search(normalized_text) is not None
    return escaped in normalized_text


def replace_exact_term(text: object, source_term: object, replacement: object) -> str | None:
    """Return one literal source-term replacement, or ``None`` when absent.

    This is deliberately a textual operation over an already-approved form.
    The caller is responsible for restricting it to an answer task; this
    helper does not attempt to understand business vocabulary.
    """

    source = _text(str(text or ""), field="query", max_chars=1000)
    original = _text(str(source_term or ""), field="source term", max_chars=_MAX_TERM_CHARS)
    target = _text(str(replacement or ""), field="replacement term", max_chars=_MAX_TERM_CHARS)
    if _key(original) == _key(target):
        return None
    if not term_occurs(source, original):
        return None
    # ``re.sub`` preserves all user wording around the approved term.  It is
    # case-insensitive for mixed ASCII terms and literal for CJK terms.
    pattern = re.compile(re.escape(original), re.IGNORECASE)
    replaced, count = pattern.subn(target, source, count=1)
    if count != 1:
        return None
    return re.sub(r"\s+", " ", replaced).strip()[:1000]


@dataclass(frozen=True)
class TerminologyForm:
    """One approved term form and the registry rule that introduced it."""

    term: str
    rule_id: str
    relation_strength: TerminologyRelationStrength

    def __post_init__(self) -> None:
        term = _text(self.term, field="terminology term", max_chars=_MAX_TERM_CHARS)
        if len(_key(term)) < 2:
            raise ValueError("terminology term must contain at least two characters")
        rule_id = _text(self.rule_id, field="terminology rule id", max_chars=128)
        if not _RULE_ID_RE.fullmatch(rule_id):
            raise ValueError("terminology rule id is invalid")
        relation = str(self.relation_strength or "").strip().casefold()
        if relation not in _RELATION_STRENGTHS:
            raise ValueError("terminology relation strength is unsupported")
        object.__setattr__(self, "term", term)
        object.__setattr__(self, "rule_id", rule_id)
        object.__setattr__(self, "relation_strength", relation)

    def safe_summary(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "relation_strength": self.relation_strength,
            "term_chars": len(self.term),
        }


@dataclass(frozen=True)
class TerminologyBinding:
    """A scope-approved concept attached to one answer requirement only."""

    requirement_id: str
    concept_id: str
    concept_key: str
    display_name: str
    source_term: str
    source_relation_strength: TerminologyRelationStrength
    query_forms: tuple[TerminologyForm, ...]
    evidence_forms: tuple[TerminologyForm, ...]
    scope_binding_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        requirement_id = _text(self.requirement_id, field="terminology requirement id", max_chars=64)
        if not _REQUIREMENT_ID_RE.fullmatch(requirement_id):
            raise ValueError("terminology binding requirement id is invalid")
        concept_id = _text(self.concept_id, field="terminology concept id", max_chars=128)
        concept_key = _text(self.concept_key, field="terminology concept key", max_chars=120)
        display_name = _text(self.display_name, field="terminology display name", max_chars=120)
        source_term = _text(self.source_term, field="terminology source term", max_chars=_MAX_TERM_CHARS)
        if len(_key(source_term)) < 2:
            raise ValueError("terminology source term must contain at least two characters")
        source_relation = str(self.source_relation_strength or "").strip().casefold()
        if source_relation not in _RELATION_STRENGTHS:
            raise ValueError("terminology source relation strength is unsupported")
        query_forms = _unique_forms(self.query_forms, field="terminology query forms")
        evidence_forms = _unique_forms(self.evidence_forms, field="terminology evidence forms")
        source_key = _key(source_term)
        if not any(_key(item.term) == source_key for item in query_forms):
            raise ValueError("terminology query forms must retain the source term")
        if source_relation == "strict_equivalent" and not any(
            _key(item.term) == source_key for item in evidence_forms
        ):
            raise ValueError("strict source terminology must retain itself as evidence")
        if any(
            item.relation_strength != "strict_equivalent"
            for item in evidence_forms
        ):
            raise ValueError("retrieval-only terminology cannot become evidence")
        binding_ids: list[str] = []
        seen_ids: set[str] = set()
        for raw in self.scope_binding_ids:
            item = _text(raw, field="terminology scope binding id", max_chars=128)
            if not _RULE_ID_RE.fullmatch(item):
                raise ValueError("terminology scope binding id is invalid")
            if item not in seen_ids:
                seen_ids.add(item)
                binding_ids.append(item)
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "concept_id", concept_id)
        object.__setattr__(self, "concept_key", concept_key)
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "source_term", source_term)
        object.__setattr__(self, "source_relation_strength", source_relation)
        object.__setattr__(self, "query_forms", query_forms)
        object.__setattr__(self, "evidence_forms", evidence_forms)
        object.__setattr__(self, "scope_binding_ids", tuple(binding_ids))

    @property
    def strict_equivalent(self) -> bool:
        return self.source_relation_strength == "strict_equivalent"

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            [*(form.rule_id for form in self.query_forms), *self.scope_binding_ids]
        ))

    def query_alias_forms(self) -> tuple[TerminologyForm, ...]:
        source_key = _key(self.source_term)
        return tuple(
            item
            for item in self.query_forms
            if _key(item.term) != source_key
        )

    def evidence_terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.term for item in self.evidence_forms))

    def evidence_match(self, content: object) -> TerminologyForm | None:
        """Return the first strict form explicitly present in source content."""

        for form in self.evidence_forms:
            if term_occurs(content, form.term):
                return form
        return None

    def safe_summary(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "concept_id": self.concept_id,
            "concept_key": self.concept_key,
            "source_relation_strength": self.source_relation_strength,
            "query_form_count": len(self.query_forms),
            "evidence_form_count": len(self.evidence_forms),
            "rule_ids": list(self.rule_ids),
            "scope_binding_count": len(self.scope_binding_ids),
        }


@dataclass(frozen=True)
class TerminologyQueryVariant:
    """One physical retrieval spelling for one logical answer task."""

    query: str
    origin: TerminologyVariantOrigin
    requirement_id: str | None = None
    rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        query = _text(self.query, field="terminology variant query", max_chars=1000)
        origin = str(self.origin or "").strip().casefold()
        if origin not in {"original", "terminology_alias"}:
            raise ValueError("terminology variant origin is unsupported")
        requirement_id = self.requirement_id
        if requirement_id is not None:
            requirement_id = _text(requirement_id, field="terminology variant requirement id", max_chars=64)
            if not _REQUIREMENT_ID_RE.fullmatch(requirement_id):
                raise ValueError("terminology variant requirement id is invalid")
        rule_ids: list[str] = []
        seen: set[str] = set()
        for raw in self.rule_ids:
            value = _text(raw, field="terminology variant rule id", max_chars=128)
            if not _RULE_ID_RE.fullmatch(value):
                raise ValueError("terminology variant rule id is invalid")
            if value not in seen:
                seen.add(value)
                rule_ids.append(value)
        if origin == "original" and rule_ids:
            raise ValueError("original terminology variant cannot carry rule ids")
        if origin == "terminology_alias" and (not requirement_id or not rule_ids):
            raise ValueError("terminology alias variant requires requirement and rule ids")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "rule_ids", tuple(rule_ids))

    def safe_summary(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "requirement_id": self.requirement_id,
            "rule_ids": list(self.rule_ids),
            "query_chars": len(self.query),
        }


@dataclass(frozen=True)
class TerminologySnapshot:
    """An immutable resolved registry view for one final execution plan."""

    plan_fingerprint: str
    scope_fingerprint: str
    registry_revision: int | None
    status: TerminologyResolutionStatus
    bindings: tuple[TerminologyBinding, ...] = ()
    reason: str | None = None
    schema_version: str = TERMINOLOGY_SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TERMINOLOGY_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported terminology snapshot schema version")
        plan_fingerprint = _text(self.plan_fingerprint, field="terminology plan fingerprint", max_chars=128)
        scope_fingerprint = _text(self.scope_fingerprint, field="terminology scope fingerprint", max_chars=128)
        if not re.fullmatch(r"[a-f0-9]{32,128}", plan_fingerprint):
            raise ValueError("terminology plan fingerprint is invalid")
        if not re.fullmatch(r"[a-f0-9]{32,128}", scope_fingerprint):
            raise ValueError("terminology scope fingerprint is invalid")
        status = str(self.status or "").strip().casefold()
        if status not in _RESOLUTION_STATUSES:
            raise ValueError("terminology snapshot status is unsupported")
        revision = self.registry_revision
        if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int) or revision < 0):
            raise ValueError("terminology registry revision is invalid")
        bindings = tuple(self.bindings)
        if len(bindings) > _MAX_BINDINGS:
            raise ValueError("terminology snapshot has too many bindings")
        if any(not isinstance(item, TerminologyBinding) for item in bindings):
            raise ValueError("terminology snapshot bindings are invalid")
        binding_keys = [
            (item.requirement_id, _key(item.source_term), item.concept_id)
            for item in bindings
        ]
        if len(set(binding_keys)) != len(binding_keys):
            raise ValueError("terminology snapshot contains duplicate bindings")
        # A resolver must turn a multi-concept term into clarification before
        # constructing the snapshot.  A single requirement can still bind
        # several source terms to the *same* concept (for example a user wrote
        # both the short and full form), which is safe.
        concepts_by_requirement: dict[str, set[str]] = {}
        for binding in bindings:
            concepts_by_requirement.setdefault(binding.requirement_id, set()).add(
                binding.concept_id
            )
        if any(len(value) > 1 for value in concepts_by_requirement.values()):
            raise ValueError("terminology snapshot contains ambiguous requirement concepts")
        reason = _text(self.reason, field="terminology snapshot reason", max_chars=300) if self.reason else None
        if status == "resolved" and revision is None:
            raise ValueError("resolved terminology snapshot requires a registry revision")
        if status == "resolved" and not bindings:
            raise ValueError("resolved terminology snapshot requires bindings")
        if status != "resolved" and bindings:
            raise ValueError("empty/degraded terminology snapshot cannot carry bindings")
        object.__setattr__(self, "plan_fingerprint", plan_fingerprint)
        object.__setattr__(self, "scope_fingerprint", scope_fingerprint)
        object.__setattr__(self, "registry_revision", revision)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "reason", reason)

    def bindings_for_requirement(self, requirement_id: str) -> tuple[TerminologyBinding, ...]:
        return tuple(
            item for item in self.bindings if item.requirement_id == requirement_id
        )

    def evidence_terms_for_requirement(self, requirement_id: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            term
            for binding in self.bindings_for_requirement(requirement_id)
            for term in binding.evidence_terms()
        ))

    def evidence_matches_for_requirement(
        self,
        requirement_id: str,
        content: object,
    ) -> tuple[TerminologyBinding, TerminologyForm] | None:
        for binding in self.bindings_for_requirement(requirement_id):
            if (form := binding.evidence_match(content)) is not None:
                return binding, form
        return None

    def query_variants(
        self,
        *,
        requirement_id: str,
        query: str,
        maximum_aliases: int,
    ) -> tuple[TerminologyQueryVariant, ...]:
        """Return original query plus bounded approved substitutions.

        Bindings only carry answer requirement IDs, so callers cannot use this
        API to alter a bridge task.  Duplicate rendered queries are collapsed
        while rule provenance remains attached to the first approved form.
        """

        original = TerminologyQueryVariant(query=query, origin="original")
        bounded_aliases = max(0, min(int(maximum_aliases), _MAX_FORMS - 1))
        variants: list[TerminologyQueryVariant] = [original]
        seen = {_key(query)}
        if bounded_aliases == 0:
            return tuple(variants)
        for binding in self.bindings_for_requirement(requirement_id):
            for form in binding.query_alias_forms():
                rendered = replace_exact_term(query, binding.source_term, form.term)
                if not rendered or _key(rendered) in seen:
                    continue
                seen.add(_key(rendered))
                variants.append(TerminologyQueryVariant(
                    query=rendered,
                    origin="terminology_alias",
                    requirement_id=requirement_id,
                    rule_ids=tuple(dict.fromkeys((
                        *binding.scope_binding_ids,
                        form.rule_id,
                    ))),
                ))
                if len(variants) - 1 >= bounded_aliases:
                    return tuple(variants)
        return tuple(variants)

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "registry_revision": self.registry_revision,
            "scope_fingerprint": self.scope_fingerprint,
            "binding_count": len(self.bindings),
            "requirement_ids": sorted({item.requirement_id for item in self.bindings}),
            "reason": self.reason,
        }


__all__ = [
    "TERMINOLOGY_SNAPSHOT_SCHEMA_VERSION",
    "TerminologyBinding",
    "TerminologyForm",
    "TerminologyQueryVariant",
    "TerminologyResolutionStatus",
    "TerminologySnapshot",
    "TerminologyVariantOrigin",
    "replace_exact_term",
    "normalize_scope_key",
    "normalize_term_key",
    "term_occurs",
]
