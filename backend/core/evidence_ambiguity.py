"""Deterministic post-retrieval ambiguity detection for scoped evidence.

The semantic router runs before retrieval and therefore cannot know whether the
authorized result set contains several mutually exclusive product/version
scopes.  This module evaluates already reranked candidates, groups them by
source-anchored document identity, and blocks generation when choosing one
scope would silently guess on the user's behalf.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping

from core.query_constraints import (
    QueryConstraints,
    candidate_section_key,
    canonical_product_name,
    evaluate_candidate_constraints,
    extract_document_constraint_identity,
    extract_query_constraints,
    inherit_document_constraint_metadata,
    query_span_is_negated,
)


# Keep this aligned with the Pipeline's verified evidence topic gate.  A second
# scope that is eligible to be shown or used by the Pipeline must not disappear
# merely because ambiguity detection applies a stricter, unrelated threshold.
AMBIGUITY_TOPIC_RELEVANCE_THRESHOLD = 0.30
MAX_AMBIGUITY_CHOICES = 6
MAX_CHOICE_TEXT_CHARS = 500

# A document-level ambiguity is intentionally a second, conservative layer
# after product/version/project scope detection.  It is driven by the
# retrieved documents' own titles/content and query-term distribution; it does
# not contain a business-topic allow-list.  The limits keep a broad query from
# turning a large knowledge base into an unusable choice list.
MAX_TOPIC_DOCUMENT_CHOICES = 6

AmbiguityDetectionMode = Literal["combined", "applicability_only"]
_AMBIGUITY_DETECTION_MODES = frozenset({"combined", "applicability_only"})
_DOCUMENT_ANSWER_ANCHOR_ROLES = frozenset({"direct", "standalone_answer"})

_ALL_SCOPES_REQUEST_RE = re.compile(
    r"(?:所有|全部|各个?|不同|多个)\s*(?:产品|版本|项目|范围)|"
    r"(?:(?:这|那)\s*)?(?:两个|两项|这些|上述|前述)\s*"
    r"(?:产品|版本|项目|范围).{0,8}(?:都要|都查|都看|都对比|全部对比)|"
    r"(?:产品|版本|项目|范围).{0,16}(?:都查|都看|都对比|全部对比)|"
    r"^(?:(?:两个|两项|这些|上述|前述)\s*)?"
    r"都(?:要|查|看(?:看)?|对比)(?:一下|吧|。|！|!)?$|"
    r"(?:分别|逐个|逐一)\s*(?:说明|回答|对比|比较|列出)|"
    r"(?:对比|比较).{0,20}(?:产品|版本|项目|范围)",
    re.IGNORECASE,
)
_TRAILING_PRODUCT_GENERATION_RE = re.compile(
    r"\s*\d{1,4}(?:\.\d{1,4}){0,3}\s*(?:全系)?\s*$",
    re.IGNORECASE,
)
_PROJECT_PLACEHOLDER_RE = re.compile(
    r"(?:非必填|选填|请填写|请填入|待填写|未填写|暂未填写|"
    r"出现问题的项目|项目名称占位|project\s*name)",
    re.IGNORECASE,
)
_PROJECT_PLACEHOLDER_EXACT_RE = re.compile(
    r"(?:n/?a|none|null|暂无|无|未知|-+)",
    re.IGNORECASE,
)
_COMPARISON_REQUEST_RE = re.compile(
    r"(?:对比|比较|区别|差异)|"
    r"(?<![A-Za-z0-9])vs\.?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_NAMED_MULTI_SCOPE_REQUEST_RE = re.compile(
    r"(?:都要|都需(?:要)?|均需(?:要)?|分别)",
    re.IGNORECASE,
)
_EXPLICIT_ALL_DIMENSION_RE = re.compile(
    r"(?P<prefix>所有|全部|各个?|不同|多个)\s*"
    r"(?P<dimension>产品|版本|项目|范围)|"
    r"(?:(?:这|那)\s*)?(?:两个|两项|这些|上述|前述)\s*"
    r"(?P<quantified_dimension>产品|版本|项目|范围).{0,8}"
    r"(?:都要|都查|都看|都对比|全部对比)|"
    r"(?P<dimension_first>产品|版本|项目|范围).{0,16}"
    r"(?:都查|都看|都对比|全部对比)",
    re.IGNORECASE,
)
_SHORT_ALL_SCOPES_RE = re.compile(
    r"^(?:(?:两个|两项|这些|上述|前述)\s*)?"
    r"都(?:要|查|看(?:看)?|对比)(?:一下|吧|。|！|!)?$",
    re.IGNORECASE,
)
_TOPIC_ALL_DOCUMENTS_RE = re.compile(
    r"(?:全部|所有|各项|各类|汇总|总览|完整(?:内容|资料)|整体(?:内容|情况)|"
    r"分别(?:说明|回答|列出|介绍)|都(?:要|查|看|了解|列出))",
    re.IGNORECASE,
)


def _positive_pattern_match(
    pattern: re.Pattern[str],
    text: str,
) -> re.Match[str] | None:
    source = str(text or "")
    return next((
        match
        for match in pattern.finditer(source)
        if not query_span_is_negated(source, match.start(), match.end())
    ), None)


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _version_key(
    value: str,
) -> tuple[int, tuple[int, ...] | tuple[str, ...]]:
    parts = str(value).split(".")
    if parts and all(part.isdigit() for part in parts):
        return (0, tuple(int(part) for part in parts))
    return (1, (str(value).casefold(),))


def _bounded_unique(
    values: Iterable[Any],
    *,
    limit: int = 20,
    max_chars: int | None = None,
) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if max_chars is not None:
            value = value[:max_chars].strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= limit:
            break
    return tuple(output)


def _meaningful_project(value: Any) -> str | None:
    project = str(value or "").strip()
    if not project or any(marker in project for marker in ("<", ">", "{{", "}}")):
        return None
    if _PROJECT_PLACEHOLDER_EXACT_RE.fullmatch(project) or _PROJECT_PLACEHOLDER_RE.search(project):
        return None
    return project


def query_requests_all_scopes(query: str) -> bool:
    """Whether the user explicitly asks to retain multiple applicability scopes.

    Short replies such as ``都对比`` are accepted, while a longer request such
    as ``账号锁定和密码策略都要配置`` is not treated as a version/product
    comparison unless it names an applicability dimension.
    """

    text = str(query or "").strip()
    return _positive_pattern_match(_ALL_SCOPES_REQUEST_RE, text) is not None


@dataclass(frozen=True)
class EvidenceScopeSlice:
    """One fail-closed applicability slice inside a physical document.

    New ingestions identify a slice by ``kb_id + doc_id + section_key``.  Old
    chunks without structural lineage fall back to the exact observed chunk
    ids.  At least one selector is therefore always present; a slice can never
    silently degrade to a whole-document grant.
    """

    kb_id: str
    doc_id: str
    section_key: str | None = None
    chunk_ids: tuple[str, ...] = ()
    is_anchor: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceScopeChoice:
    key: str
    label: str
    products: tuple[str, ...]
    canonical_products: tuple[str, ...]
    versions: tuple[str, ...]
    projects: tuple[str, ...]
    kb_ids: tuple[str, ...]
    doc_ids: tuple[str, ...]
    anchor_doc_ids: tuple[str, ...]
    companion_doc_ids: tuple[str, ...]
    filenames: tuple[str, ...]
    max_topic_relevance: float
    max_answer_support: float
    scope_slices: tuple[EvidenceScopeSlice, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentEvidenceAssessment:
    """Post-evidence assessment contract for one document chunk.

    Document choices are intentionally built from assessed answer evidence,
    not raw retrieval candidates.  ``assessment_valid`` must be asserted by
    the evidence adjudication stage; score presence alone is not verification.
    ``evidence_role`` is the contribution role (for example ``direct`` or
    ``bridge``), not the UI's direct/related display classification.
    """

    kb_id: str
    doc_id: str
    filename: str
    evidence_role: str
    supports_requirement_ids: tuple[str, ...]
    topic_relevance: float | None
    answer_support: float | None
    assessment_valid: bool
    # A standalone answer graph is anchored by the document that contains the
    # answer clause.  Cross-document mappings/prerequisites travel with that
    # anchor as companions so selecting a clarification option retains the
    # complete graph rather than only its final clause.
    companion_doc_ids: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    canonical_products: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()
    # Exact answer/bridge chunk lineage for same-document applicability slices.
    # Legacy callers may omit these fields and retain document-level behavior.
    chunk_ids: tuple[str, ...] = ()
    section_keys: tuple[str, ...] = ()
    companion_scope_slices: tuple[EvidenceScopeSlice, ...] = ()
    # Opaque semantic identity of one *closed answer claim*.  It is produced
    # only by the final evidence graph projection, never from a filename,
    # chunk rank or displayed text.  Keeping it here lets the ambiguity layer
    # distinguish two alternative answers inside one physical document from
    # ordinary complementary sections of that document.  It is intentionally
    # internal and is never exposed as a user-selectable scope.
    answer_route_key: str | None = None
    # Opaque identities of source structures that prove the route's answer
    # propositions are jointly presentable.  The final evidence graph emits
    # these only for a complete, parser-identified table; a same-document
    # section name or a retrieval neighbourhood cannot manufacture one.
    composable_answer_group_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceAmbiguityDecision:
    needs_clarification: bool
    dimension: str | None = None
    question: str = ""
    reason: str = ""
    choices: tuple[EvidenceScopeChoice, ...] = ()
    relevant_document_count: int = 0
    # Internal execution allow-list resolved from an explicit scope in the
    # current query.  An empty tuple means that ambiguity assessment did not
    # narrow the candidate document set; callers must not interpret it as an
    # instruction to discard every document.
    allowed_doc_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rag_evidence_clarification.v1",
            "needs_clarification": self.needs_clarification,
            "dimension": self.dimension,
            "question": self.question,
            "reason": self.reason,
            "choices": [choice.to_dict() for choice in self.choices],
            "relevant_document_count": self.relevant_document_count,
            "allowed_doc_ids": list(self.allowed_doc_ids),
        }


@dataclass(frozen=True)
class ExplicitScopeComparisonPlan:
    """A source-anchored scope plan for an explicit comparison request.

    ``matched`` is deliberately conservative.  Enumerated comparisons only
    match when at least two query aliases each resolve to exactly one mutually
    exclusive evidence group.  An alias shared by several groups is never used
    to guess which one the user intended.
    """

    matched: bool
    dimension: str | None = None
    choices: tuple[EvidenceScopeChoice, ...] = ()
    allowed_doc_ids: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "dimension": self.dimension,
            "choices": [choice.to_dict() for choice in self.choices],
            "allowed_doc_ids": list(self.allowed_doc_ids),
            "reason": self.reason,
        }


@dataclass
class _DocumentScope:
    kb_id: str
    doc_id: str
    filenames: set[str]
    products: set[str]
    canonical_products: set[str]
    versions: set[str]
    projects: set[str]
    max_topic_relevance: float
    max_answer_support: float
    slice_key: tuple[str, ...]
    section_key: str | None
    chunk_ids: set[str]


@dataclass
class _ScopeGroup:
    products: set[str]
    canonical_products: set[str]
    versions: set[str]
    projects: set[str]
    kb_ids: set[str]
    doc_ids: set[str]
    filenames: set[str]
    max_topic_relevance: float
    max_answer_support: float
    slices: dict[tuple[str, ...], _DocumentScope]


def _display_product(
    products: set[str],
    canonical_products: set[str],
    constraints: QueryConstraints,
) -> str:
    if constraints.product:
        query_product = str(constraints.product).strip()
        if canonical_product_name(query_product) in canonical_products:
            return query_product
    for product in sorted(products, key=lambda item: (len(item), item.casefold())):
        cleaned = _TRAILING_PRODUCT_GENERATION_RE.sub("", product).strip()
        if cleaned:
            return cleaned
    return sorted(canonical_products, key=str.casefold)[0] if canonical_products else ""


def _choice_label(
    group: _ScopeGroup,
    constraints: QueryConstraints,
) -> str:
    product = _display_product(
        group.products,
        group.canonical_products,
        constraints,
    )
    versions = sorted(group.versions, key=_version_key)
    scope = product
    if versions:
        scope = f"{scope} {' / '.join(versions)}".strip()
    projects = sorted(group.projects, key=str.casefold)
    if projects:
        scope = f"{scope}（{' / '.join(projects[:2])}）".strip()
    filenames = sorted(group.filenames, key=str.casefold)
    if filenames:
        document_label = f"《{filenames[0]}》"
        if len(filenames) > 1:
            document_label += f"等{len(filenames)}篇"
        return f"{scope} — {document_label}" if scope else document_label
    return scope or "未命名适用范围"


def _merge_document_into_group(group: _ScopeGroup, document: _DocumentScope) -> None:
    group.products.update(document.products)
    group.canonical_products.update(document.canonical_products)
    group.versions.update(document.versions)
    group.projects.update(document.projects)
    group.kb_ids.add(document.kb_id)
    group.doc_ids.add(document.doc_id)
    group.filenames.update(document.filenames)
    group.max_topic_relevance = max(
        group.max_topic_relevance,
        document.max_topic_relevance,
    )
    group.max_answer_support = max(
        group.max_answer_support,
        document.max_answer_support,
    )
    group.slices.setdefault(document.slice_key, document)


def _new_group(document: _DocumentScope) -> _ScopeGroup:
    return _ScopeGroup(
        products=set(document.products),
        canonical_products=set(document.canonical_products),
        versions=set(document.versions),
        projects=set(document.projects),
        kb_ids={document.kb_id},
        doc_ids={document.doc_id},
        filenames=set(document.filenames),
        max_topic_relevance=document.max_topic_relevance,
        max_answer_support=document.max_answer_support,
        slices={document.slice_key: document},
    )


def _merged_group(documents: list[_DocumentScope]) -> _ScopeGroup:
    group = _new_group(documents[0])
    for document in documents[1:]:
        _merge_document_into_group(group, document)
    return group


def _split_project_groups(
    documents: list[_DocumentScope],
) -> list[_ScopeGroup]:
    """Split one product/version cluster only on explicit project conflicts.

    Documents without a project are generic companions and therefore cannot
    create ambiguity by themselves.  When two explicit project clusters are
    mutually exclusive, the generic companions are attached to both choices so
    selecting either scope does not discard shared product/version material.
    A document declaring several projects bridges overlapping clusters in the
    same way that a multi-version compatibility document does.
    """

    if not documents:
        return []
    explicit = [document for document in documents if document.projects]
    generic = [document for document in documents if not document.projects]
    clusters: list[list[_DocumentScope]] = []
    cluster_projects: list[set[str]] = []
    for document in explicit:
        overlapping = [
            index
            for index, projects in enumerate(cluster_projects)
            if projects & document.projects
        ]
        if not overlapping:
            clusters.append([document])
            cluster_projects.append(set(document.projects))
            continue
        target = overlapping[0]
        clusters[target].append(document)
        cluster_projects[target].update(document.projects)
        for duplicate in reversed(overlapping[1:]):
            clusters[target].extend(clusters[duplicate])
            cluster_projects[target].update(cluster_projects[duplicate])
            del clusters[duplicate]
            del cluster_projects[duplicate]

    if len(clusters) < 2:
        return [_merged_group(documents)]
    return [
        _merged_group([*cluster, *generic])
        for cluster in clusters
    ]


def _document_scopes(
    candidates: list[dict[str, Any]],
    constraints: QueryConstraints,
) -> list[_DocumentScope]:
    enriched = inherit_document_constraint_metadata(candidates)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for candidate in enriched:
        kb_id = str(candidate.get("kb_id") or "").strip()
        doc_id = str(candidate.get("doc_id") or "").strip()
        if not kb_id or not doc_id:
            continue
        section_key = candidate_section_key(candidate)
        identity = extract_document_constraint_identity(candidate)
        raw_metadata = candidate.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
        unresolved_identity = metadata.get("ambiguous_document_identity")
        if (
            section_key is None
            and isinstance(unresolved_identity, Mapping)
            and bool(unresolved_identity)
        ):
            # Legacy chunks have no ingestion-owned section lineage.  When the
            # same physical document declares several mutually exclusive
            # identities, an unscoped chunk cannot be attributed to any one of
            # them safely.  Treating it as a generic companion would copy that
            # chunk into every clarification choice and leak a sibling version
            # back into generation after the user selects one.  Keep the chunk
            # in the broad retrieval snapshot, but never place it in an
            # executable scope choice; a selected legacy scope therefore fails
            # closed unless an exact identity-bearing chunk can answer it.
            continue
        if section_key is not None:
            local_key = ("section", section_key)
        elif (
            identity.canonical_products
            or identity.versions
            or identity.projects
        ):
            # Legacy chunks have no structural lineage.  Keep locally declared
            # identities separate and use the exact observed chunk ids later as
            # their request-local allow-list.
            local_key = (
                "local",
                "\x1e".join(identity.canonical_products),
                "\x1e".join(identity.versions),
                "\x1e".join(identity.projects),
            )
        else:
            local_key = ("generic",)
        grouped.setdefault((kb_id, doc_id, *local_key), []).append(candidate)

    documents: list[_DocumentScope] = []
    query_product = canonical_product_name(constraints.product or "")
    for slice_key, items in grouped.items():
        kb_id, doc_id = slice_key[:2]
        eligible: list[dict[str, Any]] = []
        for item in items:
            if str(item.get("evidence_role") or "") == "irrelevant":
                continue
            topic_relevance = _safe_score(item.get("topic_relevance"))
            rerank_status = str(item.get("rerank_status") or "").casefold()
            # When reranking is unavailable, required retrieval would otherwise
            # pass these candidates to generation as unverified context.  In
            # that degraded mode explicit, conflicting source scopes are safer
            # to clarify than to merge silently.  Verified candidates retain
            # the normal topic threshold so model-labelled noise cannot create
            # spurious choices.
            if (
                rerank_status == "verified"
                and topic_relevance < AMBIGUITY_TOPIC_RELEVANCE_THRESHOLD
            ):
                continue
            evaluation = evaluate_candidate_constraints(constraints, item)
            # Keep all source-declared versions together until the explicit
            # version scope filter below runs.  Otherwise a product-less query
            # such as ``2025版`` would discard the 2024/2025 alternatives before
            # it can produce an allow-list for the selected 2025 document.
            if (
                constraints.has_product_constraint
                and evaluation.status in {"mismatch", "unknown"}
            ):
                continue
            eligible.append(item)
        if not eligible:
            continue

        products: set[str] = set()
        canonical_products: set[str] = set()
        versions: set[str] = set()
        projects: set[str] = set()
        filenames: set[str] = set()
        chunk_ids: set[str] = set()
        for item in items:
            identity = extract_document_constraint_identity(item)
            products.update(identity.products)
            canonical_products.update(identity.canonical_products)
            versions.update(identity.versions)
            projects.update(
                project
                for raw_project in identity.projects
                if (project := _meaningful_project(raw_project)) is not None
            )
            filename = str(item.get("filename") or "").strip()
            if filename:
                filenames.add(filename)
            chunk_id = str(item.get("id") or item.get("chunk_id") or "").strip()
            if chunk_id:
                chunk_ids.add(chunk_id)
        if query_product and not canonical_products:
            canonical_products.add(query_product)
            products.add(str(constraints.product or "").strip())

        section_keys = {
            value
            for item in items
            if (value := candidate_section_key(item)) is not None
        }
        section_key = next(iter(section_keys)) if len(section_keys) == 1 else None
        # Without either ingestion lineage or an exact chunk id this physical
        # slice cannot be selected safely.  It may not create a clarification
        # option that later broadens to its whole document.
        if section_key is None and not chunk_ids:
            continue

        documents.append(
            _DocumentScope(
                kb_id=kb_id,
                doc_id=doc_id,
                filenames=filenames,
                products=products,
                canonical_products=canonical_products,
                versions=versions,
                projects=projects,
                max_topic_relevance=max(
                    _safe_score(item.get("topic_relevance")) for item in eligible
                ),
                max_answer_support=max(
                    _safe_score(item.get("answer_support")) for item in eligible
                ),
                slice_key=tuple(slice_key),
                section_key=section_key,
                chunk_ids=chunk_ids,
            )
        )
    return documents


def _filter_explicit_query_project(
    query: str,
    documents: list[_DocumentScope],
) -> list[_DocumentScope]:
    """Honor one source-declared project named verbatim by the user.

    Project names are not guessed from generic query words.  Only project
    values already extracted from eligible documents participate, and exactly
    one distinct value must occur in the query.  Unscoped companion documents
    remain available because they may contain product/version-wide facts.
    """

    query_text = str(query or "").casefold()
    declared_projects = {
        project
        for document in documents
        for project in document.projects
        if project
    }
    matches = {
        project
        for project in declared_projects
        if project.casefold() in query_text
    }
    if len(matches) != 1:
        return documents
    selected = next(iter(matches))
    return [
        document
        for document in documents
        if not document.projects or selected in document.projects
    ]


def _filter_explicit_query_version(
    query: str,
    documents: list[_DocumentScope],
    constraints: QueryConstraints,
) -> list[_DocumentScope]:
    """Honor one source-declared version named explicitly by the user.

    This also covers product-less policy versions such as ``2025版`` which the
    product/version constraint parser intentionally cannot bind to a product.
    Values come only from eligible source identities; arbitrary query numbers
    never become a scope by themselves.
    """

    query_text = str(query or "").casefold()
    declared_versions = {
        version
        for document in documents
        for version in document.versions
        if version
    }
    matches: set[str] = set()
    if constraints.explicit_version and constraints.version in declared_versions:
        matches.add(str(constraints.version))
    for version in declared_versions:
        escaped = re.escape(version.casefold())
        if re.search(
            rf"(?:版本|v)\s*{escaped}(?![\d.])|"
            rf"(?<![\d.]){escaped}\s*(?:版|版本|年度(?:制度)?)",
            query_text,
            re.IGNORECASE,
        ):
            matches.add(version)
    if len(matches) != 1:
        return documents
    selected = next(iter(matches))
    return [
        document
        for document in documents
        if not document.versions or selected in document.versions
    ]


def _scope_groups(
    documents: list[_DocumentScope],
    constraints: QueryConstraints,
) -> tuple[list[_ScopeGroup], str | None]:
    query_product = canonical_product_name(constraints.product or "")
    product_scoped: list[_DocumentScope] = []
    product_generic: list[_DocumentScope] = []
    for document in documents:
        canonical_products = set(document.canonical_products)
        if query_product:
            canonical_products = (
                {query_product} if query_product in canonical_products else set()
            )
        if canonical_products:
            product_scoped.append(document)
        else:
            # Versioned policies and project procedures do not always declare a
            # software product.  They still carry an applicability scope and
            # must participate instead of being silently dropped.
            product_generic.append(document)

    product_clusters: list[list[_DocumentScope]] = []
    cluster_products: list[set[str]] = []
    for document in product_scoped:
        overlapping = [
            index
            for index, products in enumerate(cluster_products)
            if products & document.canonical_products
        ]
        if not overlapping:
            product_clusters.append([document])
            cluster_products.append(set(document.canonical_products))
            continue
        target = overlapping[0]
        product_clusters[target].append(document)
        cluster_products[target].update(document.canonical_products)
        for duplicate in reversed(overlapping[1:]):
            product_clusters[target].extend(product_clusters[duplicate])
            cluster_products[target].update(cluster_products[duplicate])
            del product_clusters[duplicate]
            del cluster_products[duplicate]

    if not product_clusters:
        if product_generic:
            product_clusters.append(list(product_generic))
    elif product_generic:
        # Product-less documents are shared companions.  Attach them to each
        # mutually exclusive product cluster, just as generic project/version
        # documents are shared below, instead of inventing a third scope.
        product_clusters = [
            [*cluster, *product_generic]
            for cluster in product_clusters
        ]

    groups: list[_ScopeGroup] = []
    for product_documents in product_clusters:
        versioned = [document for document in product_documents if document.versions]
        unversioned = [document for document in product_documents if not document.versions]
        version_clusters: list[list[_DocumentScope]] = []
        cluster_versions: list[set[str]] = []
        for document in versioned:
            overlapping = [
                index
                for index, versions in enumerate(cluster_versions)
                if versions & document.versions
            ]
            if not overlapping:
                version_clusters.append([document])
                cluster_versions.append(set(document.versions))
                continue
            target = overlapping[0]
            version_clusters[target].append(document)
            cluster_versions[target].update(document.versions)
            for duplicate in reversed(overlapping[1:]):
                version_clusters[target].extend(version_clusters[duplicate])
                cluster_versions[target].update(cluster_versions[duplicate])
                del version_clusters[duplicate]
                del cluster_versions[duplicate]

        if not version_clusters and unversioned:
            version_clusters.append(list(unversioned))
        elif unversioned:
            # Generic companion documents do not form a competing version.
            # They are shared by every explicit version so a user's selection
            # does not discard common prerequisites or policy clauses.
            version_clusters = [
                [*cluster, *unversioned]
                for cluster in version_clusters
            ]
        for cluster in version_clusters:
            groups.extend(_split_project_groups(cluster))

    return groups, _scope_dimension(groups)


def _scope_dimension(groups: list[_ScopeGroup]) -> str | None:
    product_count = len({
        tuple(sorted(group.canonical_products, key=str.casefold))
        for group in groups
    })
    if product_count > 1:
        return "product_version"
    distinct_versions = {
        tuple(sorted(group.versions, key=_version_key))
        for group in groups
        if group.versions
    }
    if len(distinct_versions) > 1:
        return "version"
    if len(groups) > 1 and all(group.projects for group in groups):
        return "project"
    return None


def _ordered_scope_groups(groups: Iterable[_ScopeGroup]) -> list[_ScopeGroup]:
    return sorted(
        groups,
        key=lambda group: (
            tuple(sorted(group.canonical_products, key=str.casefold)),
            tuple(
                _version_key(value)
                for value in sorted(group.versions, key=_version_key)
            ),
            tuple(sorted(group.projects, key=str.casefold)),
            tuple(sorted(group.filenames, key=str.casefold)),
        ),
    )


def _scope_groups_to_choices(
    groups: Iterable[_ScopeGroup],
    constraints: QueryConstraints,
) -> tuple[EvidenceScopeChoice, ...]:
    """Convert mutually exclusive groups to the shared public choice shape."""

    ordered_groups = _ordered_scope_groups(groups)
    slice_scope_counts: dict[tuple[str, ...], int] = {}
    for group in ordered_groups:
        for slice_key in group.slices:
            slice_scope_counts[slice_key] = slice_scope_counts.get(slice_key, 0) + 1

    choices: list[EvidenceScopeChoice] = []
    for index, group in enumerate(ordered_groups, start=1):
        anchor_slice_keys = {
            slice_key
            for slice_key in group.slices
            if slice_scope_counts.get(slice_key, 0) == 1
        }
        if not anchor_slice_keys:
            # Distinct groups should own at least one source slice.  Retaining
            # this defensive fallback prevents a companion-only result from
            # being mistaken for proof that the selected scope was covered.
            anchor_slice_keys = set(group.slices)
        anchor_doc_ids = {
            group.slices[slice_key].doc_id
            for slice_key in anchor_slice_keys
        }
        companion_doc_ids = set(group.doc_ids) - anchor_doc_ids
        scope_slices = tuple(
            EvidenceScopeSlice(
                kb_id=value.kb_id,
                doc_id=value.doc_id,
                section_key=value.section_key,
                chunk_ids=tuple(sorted(value.chunk_ids)),
                is_anchor=slice_key in anchor_slice_keys,
            )
            for slice_key, value in sorted(
                group.slices.items(),
                key=lambda pair: pair[0],
            )
        )
        choices.append(
            EvidenceScopeChoice(
                key=f"c{index}",
                label=_choice_label(group, constraints)[:MAX_CHOICE_TEXT_CHARS],
                products=_bounded_unique(
                    sorted(group.products, key=str.casefold),
                    max_chars=MAX_CHOICE_TEXT_CHARS,
                ),
                canonical_products=_bounded_unique(
                    sorted(group.canonical_products, key=str.casefold),
                    max_chars=MAX_CHOICE_TEXT_CHARS,
                ),
                versions=_bounded_unique(
                    sorted(group.versions, key=_version_key),
                    max_chars=MAX_CHOICE_TEXT_CHARS,
                ),
                projects=_bounded_unique(
                    sorted(group.projects, key=str.casefold),
                    max_chars=MAX_CHOICE_TEXT_CHARS,
                ),
                # Candidate count is already bounded by retrieval.  Do not
                # independently truncate identifiers here: the allow-list and
                # anchor/companion partition must describe the same source set.
                kb_ids=tuple(sorted(group.kb_ids)),
                doc_ids=tuple(sorted(group.doc_ids)),
                anchor_doc_ids=tuple(sorted(anchor_doc_ids)),
                companion_doc_ids=tuple(sorted(companion_doc_ids)),
                filenames=_bounded_unique(
                    sorted(group.filenames, key=str.casefold),
                    max_chars=MAX_CHOICE_TEXT_CHARS,
                ),
                max_topic_relevance=round(group.max_topic_relevance, 4),
                max_answer_support=round(group.max_answer_support, 4),
                scope_slices=scope_slices,
            )
        )
    return tuple(choices)


# These are sentence-structure words rather than business vocabulary.  They
# prevent a broad request such as ``员工标准是什么`` from being made specific
# merely by the shared words ``标准``/``是什么``.  A subject such as ``出差`` or
# ``请假`` remains available to distinguish documents because it is not in this
# set and must be supported by the retrieved document text itself.
_TOPIC_QUERY_STOP_TERMS = frozenset({
    "什么",
    "如何",
    "怎么",
    "怎样",
    "哪些",
    "哪个",
    "哪一个",
    "是否",
    "能否",
    "可否",
    "可以",
    "请问",
    "查询",
    "查一下",
    "说明",
    "介绍",
    "标准",
    "制度",
    "政策",
    "规定",
    "办法",
    "规范",
    "要求",
    "信息",
    "内容",
    "资料",
    "相关",
    "一下",
})
_TOPIC_PRIMARY_ORIGIN_TOKENS = frozenset({
    "initial_retrieval",
    "current_retrieval",
    "carryover_current_retrieval",
})


def _topic_terms(value: Any) -> set[str]:
    """Return bounded, language-neutral title/content n-grams.

    The repository intentionally does not depend on a tokenizer package.  For
    CJK text, two- and three-character n-grams provide enough title overlap to
    distinguish independently named documents; ASCII words are kept as whole
    tokens.  Query stop terms are removed only after extraction and are never
    used as a business-topic list.
    """

    text = str(value or "").casefold()
    terms: set[str] = set()
    for match in re.finditer(r"[a-z0-9][a-z0-9_.+/-]{1,}|[\u3400-\u9fff]+", text):
        token = match.group(0)
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            if len(token) <= 4:
                terms.add(token)
            for size in (2, 3):
                if len(token) < size:
                    continue
                terms.update(token[index:index + size] for index in range(len(token) - size + 1))
        else:
            terms.add(token)
    return {
        term
        for term in terms
        if term
        and term not in _TOPIC_QUERY_STOP_TERMS
        and not _PROJECT_PLACEHOLDER_EXACT_RE.fullmatch(term)
    }


def _query_anchored_document_keys(
    query: str,
    candidates: Iterable[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Return documents whose retrieved text names a concrete query subject.

    Scope metadata answers *where* a rule applies, not *what* the user asked.
    In the degraded rerank path every candidate is intentionally retained for
    safe recovery, so a DingTalk version header may otherwise manufacture a
    version choice for an unrelated question such as an employee meal
    allowance.  This lexical guard is used only to decide whether an inferred
    product/version/project may create a clarification choice.  It never
    removes a candidate from retrieval or answer generation.

    Chinese two-character grams are useful for recall but too weak as a scope
    anchor (for example, a generic document mentioning ``员工``).  Require a
    three-character CJK n-gram or an ASCII/numeric token; an explicit product,
    version, or project constraint is handled separately and bypasses this
    guard.
    """

    query_terms = {
        term
        for term in _topic_terms(query)
        if len(term) >= 3 or bool(re.search(r"[a-z0-9]", term, re.IGNORECASE))
    }
    if not query_terms:
        # A product name may be an arbitrary source-declared label (for
        # example ``产品A``) that the global product parser deliberately does
        # not register.  Keep the identity check below available even when the
        # n-gram extractor has no sufficiently strong term.
        query_terms = set()

    anchored: set[tuple[str, str]] = set()
    query_text = str(query or "").casefold()
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("evidence_role") or "").strip().casefold() == "irrelevant":
            continue
        kb_id = str(raw.get("kb_id") or "").strip()
        doc_id = str(raw.get("doc_id") or "").strip()
        if not kb_id or not doc_id:
            continue
        metadata = raw.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        source_text = "\n".join(
            str(value or "")[:1800]
            for value in (
                raw.get("filename"),
                metadata.get("filename"),
                metadata.get("source"),
                raw.get("content"),
            )
        )
        identity = extract_document_constraint_identity(raw)
        identity_values = (
            *identity.products,
            *identity.projects,
        )
        identity_matches_query = any(
            len(str(value).strip()) >= 2
            and str(value).strip().casefold() in query_text
            for value in identity_values
        )
        if identity_matches_query or query_terms.intersection(_topic_terms(source_text)):
            anchored.add((kb_id, doc_id))
    return anchored


def _unverified_document_keys(
    candidates: Iterable[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Return document identities that have no verified rerank candidate."""

    statuses: dict[tuple[str, str], set[str]] = {}
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("evidence_role") or "").strip().casefold() == "irrelevant":
            continue
        kb_id = str(raw.get("kb_id") or "").strip()
        doc_id = str(raw.get("doc_id") or "").strip()
        if not kb_id or not doc_id:
            continue
        statuses.setdefault((kb_id, doc_id), set()).add(
            str(raw.get("rerank_status") or "").strip().casefold()
        )
    return {
        key
        for key, document_statuses in statuses.items()
        if "verified" not in document_statuses
    }


def _topic_primary_origins(candidate: dict[str, Any]) -> set[str]:
    values: list[Any] = []
    for field in ("candidate_origins", "origins"):
        raw = candidate.get(field)
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, (list, tuple, set)):
            values.extend(raw)
    for field in ("candidate_origin", "origin"):
        if candidate.get(field):
            values.append(candidate.get(field))
    return {
        str(value or "").strip().casefold()
        for value in values
        if str(value or "").strip()
    }


def _topic_document_rows(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate only current-query document anchors for topic ambiguity.

    Full-document/structural expansion chunks cannot introduce a new choice;
    at least one current-query seed must have retrieved the document.  Test and
    legacy adapters that omit origin metadata remain compatible and are treated
    as primary candidates.
    """

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for position, raw in enumerate(candidates):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("evidence_role") or "").strip().casefold() == "irrelevant":
            continue
        origins = _topic_primary_origins(raw)
        if origins and not origins.intersection(_TOPIC_PRIMARY_ORIGIN_TOKENS):
            continue
        kb_id = str(raw.get("kb_id") or "").strip()
        doc_id = str(raw.get("doc_id") or "").strip()
        if not kb_id or not doc_id:
            continue
        raw_metadata = raw.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        filename = str(
            raw.get("filename")
            or metadata.get("filename")
            or metadata.get("source")
            or ""
        ).strip()
        if not filename:
            continue
        key = (kb_id, doc_id)
        row = grouped.setdefault(
            key,
            {
                "kb_id": kb_id,
                "doc_id": doc_id,
                "filenames": set(),
                "text_parts": [],
                "positions": [],
                "scores": [],
                "topic_relevance": [],
                "answer_support": [],
            },
        )
        row["filenames"].add(filename)
        row["text_parts"].append(f"{filename}\n{str(raw.get('content') or '')[:1800]}")
        row["positions"].append(position)
        for field in ("score", "retrieval_score", "vector_score", "keyword_score", "trigram_score"):
            try:
                value = float(raw.get(field))
            except (TypeError, ValueError):
                continue
            if value == value and value >= 0:
                row["scores"].append(value)
        for field in ("topic_relevance", "answer_support"):
            try:
                value = float(raw.get(field))
            except (TypeError, ValueError):
                continue
            if value == value and value >= 0:
                row[field].append(value)
    rows = list(grouped.values())
    for row in rows:
        row["filename"] = sorted(row["filenames"], key=str.casefold)[0]
        row["text"] = "\n".join(row["text_parts"])
        row["position"] = min(row["positions"] or [0])
        row["best_score"] = max(row["scores"] or [0.0])
        row["max_topic_relevance"] = max(row["topic_relevance"] or [0.0])
        row["max_answer_support"] = max(row["answer_support"] or [0.0])
    return sorted(rows, key=lambda row: (row["position"], row["filename"].casefold()))


def _topic_label_key(filename: str) -> str:
    value = str(filename or "").casefold()
    value = re.sub(r"\.(?:docx?|pdf|xlsx?|pptx?|md|txt)$", "", value)
    value = re.sub(r"[（(]?副本[）)]?|[（(]?copy[）)]?", "", value)
    value = re.sub(r"[\s_—–-]+", "", value)
    return value


def _topic_groups_to_choices(rows: list[dict[str, Any]]) -> tuple[EvidenceScopeChoice, ...]:
    choices: list[EvidenceScopeChoice] = []
    for index, row in enumerate(rows, start=1):
        filename = str(row["filename"] or "未命名文档")[:MAX_CHOICE_TEXT_CHARS]
        choices.append(
            EvidenceScopeChoice(
                key=f"c{index}",
                label=f"《{filename}》",
                products=(),
                canonical_products=(),
                versions=(),
                projects=(),
                kb_ids=(str(row["kb_id"]),),
                doc_ids=(str(row["doc_id"]),),
                anchor_doc_ids=(str(row["doc_id"]),),
                companion_doc_ids=(),
                filenames=(filename,),
                max_topic_relevance=round(float(row["max_topic_relevance"]), 4),
                max_answer_support=round(float(row["max_answer_support"]), 4),
            )
        )
    return tuple(choices)


def _required_answer_requirement_ids(
    requirements: Iterable[Any] | None,
) -> tuple[str, ...]:
    """Return explicitly required answer requirement ids in source order.

    Post-evidence document ambiguity must never guess which planner
    requirements are answer-bearing.  Callers therefore need to provide the
    typed/mapping requirement contract, including ``id``, ``role`` and
    ``importance``.  Missing fields fail closed and cannot create choices.
    """

    result: list[str] = []
    seen: set[str] = set()
    for raw in requirements or ():
        if isinstance(raw, Mapping):
            requirement_id = raw.get("id")
            role = raw.get("role")
            importance = raw.get("importance")
        else:
            requirement_id = getattr(raw, "id", None)
            role = getattr(raw, "role", None)
            importance = getattr(raw, "importance", None)
        normalized_id = str(requirement_id or "").strip()
        if (
            not normalized_id
            or normalized_id in seen
            or str(role or "").strip().casefold() != "answer"
            or str(importance or "").strip().casefold() != "required"
        ):
            continue
        seen.add(normalized_id)
        result.append(normalized_id)
        if len(result) >= 8:
            break
    return tuple(result)


def _positive_assessment_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score <= 0:
        return None
    return min(score, 1.0)


def _eligible_document_assessment(
    assessment: DocumentEvidenceAssessment,
    *,
    required_answer_ids: set[str],
) -> tuple[set[str], float, float] | None:
    """Validate one post-evidence row as a document-choice anchor.

    Only a positively assessed standalone answer can introduce a document
    option.  Bridge, complement, background and display-only ``direct`` labels
    are not interchangeable: callers must pass the contribution role through
    ``evidence_role``.  This keeps supporting material attached to an answer
    without presenting it as a competing source.
    """

    if not isinstance(assessment, DocumentEvidenceAssessment):
        return None
    if assessment.assessment_valid is not True:
        return None
    if (
        str(assessment.evidence_role or "").strip().casefold()
        not in _DOCUMENT_ANSWER_ANCHOR_ROLES
    ):
        return None
    topic_relevance = _positive_assessment_score(assessment.topic_relevance)
    answer_support = _positive_assessment_score(assessment.answer_support)
    if topic_relevance is None or answer_support is None:
        return None
    supported_ids = {
        str(value or "").strip()
        for value in assessment.supports_requirement_ids or ()
        if str(value or "").strip() in required_answer_ids
    }
    if not supported_ids:
        return None
    return supported_ids, topic_relevance, answer_support


def _assessment_scope_slices(
    assessment: DocumentEvidenceAssessment,
) -> tuple[EvidenceScopeSlice, ...]:
    """Normalize one assessed answer graph into exact selectable slices."""

    kb_id = str(assessment.kb_id or "").strip()
    doc_id = str(assessment.doc_id or "").strip()
    chunk_ids = _bounded_unique(assessment.chunk_ids or (), limit=100)
    section_keys = _bounded_unique(assessment.section_keys or (), limit=20)
    anchors: list[EvidenceScopeSlice] = []
    if section_keys:
        anchors.extend(
            EvidenceScopeSlice(
                kb_id=kb_id,
                doc_id=doc_id,
                section_key=section_key,
                chunk_ids=chunk_ids,
                is_anchor=True,
            )
            for section_key in section_keys
        )
    elif chunk_ids:
        anchors.append(EvidenceScopeSlice(
            kb_id=kb_id,
            doc_id=doc_id,
            section_key=None,
            chunk_ids=chunk_ids,
            is_anchor=True,
        ))

    output: list[EvidenceScopeSlice] = []
    seen: set[tuple[str, str, str, tuple[str, ...], bool]] = set()
    for value in (*anchors, *assessment.companion_scope_slices):
        if not isinstance(value, EvidenceScopeSlice):
            continue
        identity = (
            str(value.kb_id),
            str(value.doc_id),
            str(value.section_key or ""),
            tuple(value.chunk_ids),
            bool(value.is_anchor),
        )
        if identity in seen:
            continue
        if not value.section_key and not value.chunk_ids:
            continue
        seen.add(identity)
        output.append(value)
    return tuple(output)


def _scope_slice_tokens(
    values: Iterable[EvidenceScopeSlice],
    *,
    anchors_only: bool = False,
) -> tuple[tuple[str, ...], ...]:
    tokens: set[tuple[str, ...]] = set()
    for value in values:
        if anchors_only and not value.is_anchor:
            continue
        if value.section_key:
            tokens.add((
                str(value.kb_id),
                str(value.doc_id),
                "section",
                str(value.section_key),
            ))
        else:
            tokens.update(
                (
                    str(value.kb_id),
                    str(value.doc_id),
                    "chunk",
                    str(chunk_id),
                )
                for chunk_id in value.chunk_ids
            )
    return tuple(sorted(tokens))


def _declared_applicability_identity(
    assessment: DocumentEvidenceAssessment,
) -> tuple[tuple[str, ...], ...]:
    """Return the source-declared applicability identity of an answer route.

    An exact chunk or section is a *lineage selector*, not proof that two
    clauses have different applicability.  Using it as the grouping identity
    made two unattributed clauses in the same legacy document look like two
    different documents, which then yielded duplicate, non-actionable
    clarification buttons.  Only values explicitly attached to the closed
    route (product, version or project) may divide answer alternatives.

    Section/chunk lineage remains in ``scope_slices`` for a choice that has a
    proven identity; it simply cannot manufacture such an identity here.
    """

    products = assessment.canonical_products or assessment.products
    dimensions = (
        (
            "product",
            tuple(sorted({
                str(value).strip().casefold()
                for value in products or ()
                if str(value).strip()
            })),
        ),
        (
            "version",
            tuple(sorted({
                str(value).strip()
                for value in assessment.versions or ()
                if str(value).strip()
            }, key=_version_key)),
        ),
        (
            "project",
            tuple(sorted({
                project
                for value in assessment.projects or ()
                if (project := _meaningful_project(value)) is not None
            }, key=str.casefold)),
        ),
    )
    return tuple(
        (name, *values)
        for name, values in dimensions
        if values
    )


def _answer_route_key(assessment: DocumentEvidenceAssessment) -> str | None:
    """Normalize the graph-projected answer identity for local comparison."""

    value = re.sub(r"\s+", " ", str(assessment.answer_route_key or "")).strip()
    return value[:600] or None


def _same_document_answer_scope_is_unresolved(
    rows: Iterable[Mapping[str, Any]],
    *,
    required_answer_ids: set[str],
) -> bool:
    """Whether one physical scope contains distinct closed answer routes.

    This is deliberately narrower than a generic same-document duplicate:
    complementary chapters and repeated evidence retain no route identity (or
    the same identity) and remain one answer source.  A positive result means
    the final graph proved more than one answer proposition for the same
    required answer, but the source did not prove an applicability dimension
    capable of separating them.  The only safe UX is a free-text refinement,
    never two copies of the same document choice.
    """

    for row in rows:
        route_keys_by_requirement = row.get("answer_route_keys_by_requirement")
        if not isinstance(route_keys_by_requirement, Mapping):
            continue
        composable_groups_by_requirement = row.get(
            "composable_answer_route_keys_by_requirement",
        )
        if not isinstance(composable_groups_by_requirement, Mapping):
            composable_groups_by_requirement = {}
        for requirement_id in required_answer_ids:
            values = route_keys_by_requirement.get(requirement_id)
            if not isinstance(values, (set, frozenset, tuple, list)):
                continue
            route_keys = set(values)
            if len(route_keys) <= 1:
                continue

            # A verified complete table is a single source-authored answer
            # structure. Its rows may differ by city, region, date or any
            # other table dimension, but the user must be able to read those
            # values together rather than being forced to choose a fabricated
            # scope. This exception is source-structural: rows merely sharing
            # a document, heading or retrieval rank do not qualify.
            route_groups = composable_groups_by_requirement.get(requirement_id)
            if isinstance(route_groups, Mapping) and any(
                route_keys.issubset(set(group_route_keys))
                for group_route_keys in route_groups.values()
                if isinstance(group_route_keys, (set, frozenset, tuple, list))
            ):
                continue
            return True
    return False


def _merge_interdependent_answer_graph_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse answer graphs which use another graph's anchor as evidence.

    Clarification choices must be independent alternatives.  If graph A uses
    graph B's answer anchor as a companion, advertising A and B as separate
    choices is both misleading and structurally invalid: B would be a shared
    document and an exclusive anchor at the same time.  Collapse every such
    connected component before choices are built.

    The representative prefers a complete graph that is not itself dependent
    on another member (a dependency sink).  All documents from the component
    remain in its bounded allow-list as companions, so graph normalization can
    neither invent access nor discard evidence needed by a selected answer.
    Rows are linked only inside the same knowledge base and when they support
    at least one common required answer.
    """

    if len(rows) < 2:
        return rows

    anchor_indexes: dict[tuple[str, str, tuple[tuple[str, ...], ...]], list[int]] = {}
    for index, row in enumerate(rows):
        anchor_indexes.setdefault(
            (
                str(row["kb_id"]),
                str(row["doc_id"]),
                tuple(row.get("applicability_scope_identity") or ()),
            ),
            [],
        ).append(index)

    adjacency = [set() for _ in rows]
    dependency_targets = [set() for _ in rows]

    # Multiple assessments of the same document are one answer anchor even
    # when their bridge sets differ.
    for indexes in anchor_indexes.values():
        for offset, left in enumerate(indexes):
            for right in indexes[offset + 1:]:
                adjacency[left].add(right)
                adjacency[right].add(left)

    for index, row in enumerate(rows):
        supported_ids = set(row["supported_required_answer_ids"])
        kb_id = str(row["kb_id"])
        for companion_doc_id in row["companion_doc_ids"]:
            matching_targets = [
                target
                for (target_kb, target_doc, _), indexes in anchor_indexes.items()
                if target_kb == kb_id and target_doc == str(companion_doc_id)
                for target in indexes
            ]
            for target in matching_targets:
                if target == index:
                    continue
                if not supported_ids.intersection(
                    rows[target]["supported_required_answer_ids"]
                ):
                    continue
                dependency_targets[index].add(target)
                adjacency[index].add(target)
                adjacency[target].add(index)

    components: list[set[int]] = []
    unseen = set(range(len(rows)))
    while unseen:
        start = min(unseen)
        component: set[int] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            unseen.discard(current)
            pending.extend(adjacency[current] - component)
        components.append(component)

    merged_rows: list[dict[str, Any]] = []
    for component in components:
        if len(component) == 1:
            merged_rows.append(rows[next(iter(component))])
            continue

        incoming_counts = {
            index: sum(
                index in dependency_targets[source]
                for source in component
            )
            for index in component
        }
        representative_index = min(
            component,
            key=lambda index: (
                len(dependency_targets[index].intersection(component)),
                -incoming_counts[index],
                int(rows[index]["position"]),
                str(rows[index]["filename"]).casefold(),
                str(rows[index]["doc_id"]),
            ),
        )
        representative = rows[representative_index]
        merged = dict(representative)
        component_doc_ids = {
            str(doc_id)
            for index in component
            for doc_id in (
                rows[index]["doc_id"],
                *rows[index]["companion_doc_ids"],
            )
            if str(doc_id)
        }
        representative_doc_id = str(representative["doc_id"])
        merged["companion_doc_ids"] = (
            component_doc_ids - {representative_doc_id}
        )
        for field in (
            "filenames",
            "products",
            "canonical_products",
            "versions",
            "projects",
            "supported_required_answer_ids",
            "chunk_ids",
            "section_keys",
        ):
            merged[field] = {
                value
                for index in component
                for value in rows[index][field]
            }
        merged_route_keys_by_requirement: dict[str, set[str]] = {}
        for index in component:
            raw_route_keys = rows[index].get(
                "answer_route_keys_by_requirement",
                {},
            )
            if not isinstance(raw_route_keys, Mapping):
                continue
            for requirement_id, values in raw_route_keys.items():
                if not isinstance(values, (set, frozenset, tuple, list)):
                    continue
                merged_route_keys_by_requirement.setdefault(
                    str(requirement_id),
                    set(),
                ).update(
                    str(value)
                    for value in values
                    if str(value)
                )
        merged["answer_route_keys_by_requirement"] = (
            merged_route_keys_by_requirement
        )
        # The composition certificate belongs to the closed source route, not
        # to the particular companion-document set through which that route
        # reached the final graph.  A route can legitimately have different
        # bridge/condition companions from its table sibling.  Preserve the
        # complete-table membership while collapsing those interdependent
        # rows; otherwise a later same-document check would see the route
        # identities but lose the proof that their values are jointly
        # presentable.
        merged_composable_groups: dict[str, dict[str, set[str]]] = {}
        for index in component:
            raw_groups = rows[index].get(
                "composable_answer_route_keys_by_requirement",
                {},
            )
            if not isinstance(raw_groups, Mapping):
                continue
            for requirement_id, groups in raw_groups.items():
                if not isinstance(groups, Mapping):
                    continue
                requirement_groups = merged_composable_groups.setdefault(
                    str(requirement_id),
                    {},
                )
                for group_id, values in groups.items():
                    if not isinstance(values, (set, frozenset, tuple, list)):
                        continue
                    requirement_groups.setdefault(str(group_id), set()).update(
                        str(value)
                        for value in values
                        if str(value)
                    )
        merged["composable_answer_route_keys_by_requirement"] = (
            merged_composable_groups
        )
        merged_scope_slices: dict[tuple[str, ...], EvidenceScopeSlice] = {}
        for index in component:
            for value in rows[index].get("scope_slices", ()):
                for token in _scope_slice_tokens((value,)):
                    merged_scope_slices[token] = value
        merged["scope_slices"] = tuple(merged_scope_slices.values())
        merged["applicability_scope_identity"] = tuple(sorted({
            scope_identity
            for index in component
            for scope_identity in (
                tuple(rows[index].get("applicability_scope_identity") or ()),
            )
        }))
        merged["position"] = min(
            int(rows[index]["position"]) for index in component
        )
        merged["max_topic_relevance"] = max(
            float(rows[index]["max_topic_relevance"])
            for index in component
        )
        merged["max_answer_support"] = max(
            float(rows[index]["max_answer_support"])
            for index in component
        )
        merged_rows.append(merged)

    merged_rows.sort(
        key=lambda row: (
            int(row["position"]),
            str(row["filename"]).casefold(),
            str(row["doc_id"]),
        )
    )
    return merged_rows


def _post_evidence_graph_choices(
    rows: list[dict[str, Any]],
) -> tuple[EvidenceScopeChoice, ...]:
    choices: list[EvidenceScopeChoice] = []
    label_counts: dict[str, int] = {}
    for row in rows:
        key = _topic_label_key(str(row.get("filename") or ""))
        label_counts[key] = label_counts.get(key, 0) + 1
    for index, row in enumerate(rows, start=1):
        anchor_doc_id = str(row["doc_id"])
        companion_doc_ids = tuple(sorted(
            {
                str(value).strip()
                for value in row["companion_doc_ids"]
                if str(value).strip() and str(value).strip() != anchor_doc_id
            }
        ))
        filename = str(row["filename"] or "未命名文档")[:MAX_CHOICE_TEXT_CHARS]
        label = f"《{filename}》"
        if label_counts.get(_topic_label_key(filename), 0) > 1:
            versions = sorted(row["versions"], key=_version_key)
            projects = sorted(row["projects"], key=str.casefold)
            qualifier = (
                f"版本 {' / '.join(versions[:2])}"
                if versions
                else (
                    f"项目 {' / '.join(projects[:2])}"
                    if projects
                    else f"文档ID {anchor_doc_id[:8]}"
                )
            )
            label = f"{label}（{qualifier}）"
        choices.append(EvidenceScopeChoice(
            key=f"c{index}",
            label=label[:MAX_CHOICE_TEXT_CHARS],
            products=_bounded_unique(
                sorted(row["products"], key=str.casefold),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            canonical_products=_bounded_unique(
                sorted(row["canonical_products"], key=str.casefold),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            versions=_bounded_unique(
                sorted(row["versions"], key=_version_key),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            projects=_bounded_unique(
                sorted(row["projects"], key=str.casefold),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            kb_ids=(str(row["kb_id"]),),
            doc_ids=(anchor_doc_id, *companion_doc_ids),
            anchor_doc_ids=(anchor_doc_id,),
            companion_doc_ids=companion_doc_ids,
            filenames=(filename,),
            max_topic_relevance=round(float(row["max_topic_relevance"]), 4),
            max_answer_support=round(float(row["max_answer_support"]), 4),
            scope_slices=tuple(row.get("scope_slices", ())),
        ))
    return tuple(choices)


def _post_evidence_scope_choices(
    *,
    query: str,
    rows: list[dict[str, Any]],
) -> tuple[str | None, tuple[EvidenceScopeChoice, ...]]:
    """Group complete answer graphs by their verified applicability scope.

    Raw retrieval candidates are deliberately absent here.  Every synthetic
    document below represents one graph that survived evidence adjudication
    and the final prompt budget; its scope is the union of its answer anchor
    and bridge companions.
    """

    constraints = extract_query_constraints(query)
    graph_rows: dict[str, dict[str, Any]] = {}
    documents: list[_DocumentScope] = []
    for index, row in enumerate(rows):
        graph_id = f"__assessed_graph_{index}"
        graph_rows[graph_id] = row
        documents.append(_DocumentScope(
            kb_id=str(row["kb_id"]),
            doc_id=graph_id,
            filenames={str(row["filename"])},
            products=set(row["products"]),
            canonical_products=set(row["canonical_products"]),
            versions=set(row["versions"]),
            projects=set(row["projects"]),
            max_topic_relevance=float(row["max_topic_relevance"]),
            max_answer_support=float(row["max_answer_support"]),
            slice_key=(str(row["kb_id"]), graph_id, "assessed_graph"),
            section_key=None,
            chunk_ids=set(),
        ))
    documents = _filter_explicit_query_project(
        query,
        _filter_explicit_query_version(query, documents, constraints),
    )
    groups, dimension = _scope_groups(documents, constraints)
    if dimension is None or len(groups) < 2:
        return None, ()

    # Generic graphs are copied into each scope by ``_scope_groups``.  That is
    # correct for ordinary supporting documents, but an answer graph cannot be
    # advertised as the exclusive anchor of two choices.  Fall back to a
    # document question rather than publishing an invalid allow-list.
    graph_occurrences: dict[str, int] = {}
    for group in groups:
        for graph_id in group.doc_ids:
            graph_occurrences[graph_id] = graph_occurrences.get(graph_id, 0) + 1
    if any(count != 1 for count in graph_occurrences.values()):
        return None, ()

    choices: list[EvidenceScopeChoice] = []
    for index, group in enumerate(_ordered_scope_groups(groups), start=1):
        member_rows = [graph_rows[graph_id] for graph_id in group.doc_ids]
        merged_scope_slices: dict[tuple[str, ...], EvidenceScopeSlice] = {}
        for row in member_rows:
            for value in row.get("scope_slices", ()):
                for token in _scope_slice_tokens((value,)):
                    merged_scope_slices[token] = value
        scope_slices = tuple(merged_scope_slices.values())
        if scope_slices:
            anchor_doc_ids = {
                str(value.doc_id) for value in scope_slices if value.is_anchor
            }
            all_doc_ids = {str(value.doc_id) for value in scope_slices}
        else:
            anchor_doc_ids = {
                str(row["doc_id"]) for row in member_rows if str(row["doc_id"])
            }
            all_doc_ids = set(anchor_doc_ids)
            all_doc_ids.update(
                str(doc_id)
                for row in member_rows
                for doc_id in row["companion_doc_ids"]
                if str(doc_id)
            )
        companion_doc_ids = all_doc_ids - anchor_doc_ids
        choices.append(EvidenceScopeChoice(
            key=f"c{index}",
            label=_choice_label(group, constraints)[:MAX_CHOICE_TEXT_CHARS],
            products=_bounded_unique(
                sorted(group.products, key=str.casefold),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            canonical_products=_bounded_unique(
                sorted(group.canonical_products, key=str.casefold),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            versions=_bounded_unique(
                sorted(group.versions, key=_version_key),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            projects=_bounded_unique(
                sorted(group.projects, key=str.casefold),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            kb_ids=tuple(sorted({str(row["kb_id"]) for row in member_rows})),
            doc_ids=tuple(sorted(all_doc_ids)),
            anchor_doc_ids=tuple(sorted(anchor_doc_ids)),
            companion_doc_ids=tuple(sorted(companion_doc_ids)),
            filenames=_bounded_unique(
                sorted(group.filenames, key=str.casefold),
                max_chars=MAX_CHOICE_TEXT_CHARS,
            ),
            max_topic_relevance=round(group.max_topic_relevance, 4),
            max_answer_support=round(group.max_answer_support, 4),
            scope_slices=scope_slices,
        ))
    return dimension, tuple(choices)


def detect_post_evidence_document_ambiguity(
    *,
    query: str,
    requirements: Iterable[Any] | None,
    assessments: Iterable[DocumentEvidenceAssessment],
) -> EvidenceAmbiguityDecision:
    """Detect document alternatives from adjudicated answer evidence only.

    This is the document/topic phase for a two-stage pipeline.  Applicability
    scopes are resolved before evidence assembly; this function runs after the
    evidence stage has assigned contribution roles and requirement support.

    Documents supporting different required answers are complementary and do
    not compete.  A choice is created only when at least two distinct document
    anchors independently support the *same* required answer.  Supporting
    bridge/complement/background rows can remain in the evidence bundle, but
    can never become user-facing options here.
    """

    text = str(query or "").strip()
    if not text:
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="empty_query",
        )
    if _positive_pattern_match(_TOPIC_ALL_DOCUMENTS_RE, text) is not None:
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="query_requests_all_documents",
        )
    if query_requests_all_scopes(text):
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="query_requests_all_scopes",
        )

    required_answer_ids = _required_answer_requirement_ids(requirements)
    if not required_answer_ids:
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="no_required_answer_requirements",
        )
    required_answer_id_set = set(required_answer_ids)

    grouped: dict[
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[tuple[str, ...], ...],
        ],
        dict[str, Any],
    ] = {}
    for position, assessment in enumerate(assessments):
        eligible = _eligible_document_assessment(
            assessment,
            required_answer_ids=required_answer_id_set,
        )
        if eligible is None:
            continue
        kb_id = str(assessment.kb_id or "").strip()
        doc_id = str(assessment.doc_id or "").strip()
        filename = str(assessment.filename or "").strip()
        if not kb_id or not doc_id or not filename:
            continue
        companion_doc_ids = tuple(sorted({
            str(value or "").strip()
            for value in assessment.companion_doc_ids or ()
            if str(value or "").strip()
            and str(value or "").strip() != doc_id
        }))
        scope_slices = _assessment_scope_slices(assessment)
        applicability_scope_identity = _declared_applicability_identity(
            assessment,
        )
        supported_ids, topic_relevance, answer_support = eligible
        row = grouped.setdefault(
            (kb_id, doc_id, companion_doc_ids, applicability_scope_identity),
            {
                "kb_id": kb_id,
                "doc_id": doc_id,
                "companion_doc_ids": set(companion_doc_ids),
                "filenames": set(),
                "products": set(),
                "canonical_products": set(),
                "versions": set(),
                "projects": set(),
                "position": position,
                "supported_required_answer_ids": set(),
                "max_topic_relevance": 0.0,
                "max_answer_support": 0.0,
                "chunk_ids": set(),
                "section_keys": set(),
                "scope_slices": (),
                # This is the semantic identity used to group answer
                # alternatives.  Exact chunk/section selectors remain in
                # scope_slices and are intentionally not a grouping key.
                "applicability_scope_identity": applicability_scope_identity,
                "answer_route_keys_by_requirement": {},
                # A complete source table is a composition certificate, not
                # an applicability scope, and must never become a user choice.
                "composable_answer_route_keys_by_requirement": {},
            },
        )
        row["filenames"].add(filename)
        row["products"].update(
            str(value).strip()
            for value in assessment.products or ()
            if str(value).strip()
        )
        row["canonical_products"].update(
            str(value).strip()
            for value in assessment.canonical_products or ()
            if str(value).strip()
        )
        row["versions"].update(
            str(value).strip()
            for value in assessment.versions or ()
            if str(value).strip()
        )
        row["projects"].update(
            project
            for value in assessment.projects or ()
            if (project := _meaningful_project(value)) is not None
        )
        row["position"] = min(int(row["position"]), position)
        row["supported_required_answer_ids"].update(supported_ids)
        row["max_topic_relevance"] = max(
            float(row["max_topic_relevance"]),
            topic_relevance,
        )
        row["max_answer_support"] = max(
            float(row["max_answer_support"]),
            answer_support,
        )
        row["chunk_ids"].update(assessment.chunk_ids or ())
        row["section_keys"].update(assessment.section_keys or ())
        answer_route_key = _answer_route_key(assessment)
        if answer_route_key is not None:
            for requirement_id in supported_ids:
                row["answer_route_keys_by_requirement"].setdefault(
                    requirement_id,
                    set(),
                ).add(answer_route_key)
                for group_id in _bounded_unique(
                    assessment.composable_answer_group_ids,
                    max_chars=300,
                ):
                    row[
                        "composable_answer_route_keys_by_requirement"
                    ].setdefault(requirement_id, {}).setdefault(
                        group_id,
                        set(),
                    ).add(answer_route_key)
        merged_scope_slices = {
            token: value
            for value in row.get("scope_slices", ())
            for token in _scope_slice_tokens((value,))
        }
        for value in scope_slices:
            for token in _scope_slice_tokens((value,)):
                merged_scope_slices[token] = value
        row["scope_slices"] = tuple(merged_scope_slices.values())

    rows = list(grouped.values())
    for row in rows:
        row["filename"] = sorted(row["filenames"], key=str.casefold)[0]
    rows.sort(key=lambda row: (row["position"], row["filename"].casefold()))
    rows = _merge_interdependent_answer_graph_rows(rows)
    if _same_document_answer_scope_is_unresolved(
        rows,
        required_answer_ids=required_answer_id_set,
    ):
        return EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension="scope",
            question=(
                "同一份资料中存在多个可能的规则，但现有证据无法可靠判断"
                "它们分别适用于什么范围。请补充产品、版本、项目、章节或"
                "制度范围后再查询。"
            ),
            reason="same_document_answer_scope_unresolved",
            relevant_document_count=len({
                (str(row["kb_id"]), str(row["doc_id"]))
                for row in rows
            }),
        )
    if len(rows) < 2:
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="single_assessed_answer_document",
            relevant_document_count=len(rows),
        )

    competing_groups: list[
        tuple[
            tuple[
                str,
                str,
                tuple[str, ...],
                tuple[tuple[str, ...], ...],
            ],
            ...,
        ]
    ] = []
    for requirement_id in required_answer_ids:
        group = tuple(
            (
                str(row["kb_id"]),
                str(row["doc_id"]),
                tuple(sorted(row["companion_doc_ids"])),
                tuple(row.get("applicability_scope_identity") or ()),
            )
            for row in rows
            if requirement_id in row["supported_required_answer_ids"]
        )
        if len(group) >= 2 and group not in competing_groups:
            competing_groups.append(group)
    if not competing_groups:
        # The assessed documents cover different required answers and are
        # therefore complementary parts of one response.
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="complementary_assessed_answer_documents",
            relevant_document_count=len(rows),
        )

    if len(competing_groups) > 1 and any(
        group != competing_groups[0] for group in competing_groups[1:]
    ):
        competing_document_count = len({
            key for group in competing_groups for key in group
        })
        return EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension="document",
            question=(
                "检索到多个问题部分分别存在不同的有效答案来源。"
                "请补充需要核对的具体问题部分或适用范围。"
            ),
            reason="multiple_assessed_document_ambiguity_groups",
            relevant_document_count=competing_document_count,
        )

    competing_keys = set(competing_groups[0])
    choice_rows = [
        row
        for row in rows
        if (
            str(row["kb_id"]),
            str(row["doc_id"]),
            tuple(sorted(row["companion_doc_ids"])),
            tuple(row.get("applicability_scope_identity") or ()),
        ) in competing_keys
    ]
    scope_dimension, scope_choices = _post_evidence_scope_choices(
        query=text,
        rows=choice_rows,
    )
    if scope_dimension is not None and len(scope_choices) >= 2:
        lines = [
            "检索到与当前问题相关、但适用范围不同的资料：",
            *(
                f"{index}. {choice.label}"
                for index, choice in enumerate(scope_choices, start=1)
            ),
            "请问需要查询哪一项？也可以回复“都对比”。",
        ]
        return EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension=scope_dimension,
            question="\n".join(lines),
            reason="multiple_mutually_exclusive_assessed_scopes",
            choices=scope_choices,
            relevant_document_count=len({
                doc_id for choice in scope_choices for doc_id in choice.doc_ids
            }),
        )
    if len(choice_rows) > MAX_TOPIC_DOCUMENT_CHOICES:
        return EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension="document",
            question=(
                "检索到超过 6 篇经过评估且能回答当前问题的资料。"
                "请补充具体主题、适用范围或文档名称。"
            ),
            reason="too_many_assessed_answer_documents",
            relevant_document_count=len(choice_rows),
        )

    choices = _post_evidence_graph_choices(choice_rows)
    question = (
        "检索到多篇均能支持当前问题、但答案来源不同的资料：\n"
        + "\n".join(
            f"{index}. {choice.label}"
            for index, choice in enumerate(choices, start=1)
        )
        + "\n请问需要查询哪一篇？也可以回复“都要”。"
    )
    return EvidenceAmbiguityDecision(
        needs_clarification=True,
        dimension="document",
        question=question,
        reason="multiple_assessed_answer_documents",
        choices=choices,
        relevant_document_count=len(choice_rows),
    )


def _requirement_descriptions(requirements: Iterable[Any] | None) -> tuple[str, ...]:
    """Read bounded requirement prose without depending on a planner class.

    The ambiguity module is also used by compatibility callers, so accepting
    mappings and simple objects keeps the signal generic while avoiding a
    dependency on ``rag_v2`` contracts.  Requirement text is only a negative
    document-picker signal: it may prove that several documents are needed
    together, but can never create or select an applicability scope.
    """

    descriptions: list[str] = []
    seen: set[str] = set()
    for raw in requirements or ():
        if isinstance(raw, str):
            value = raw
        elif isinstance(raw, Mapping):
            value = raw.get("description")
        else:
            value = getattr(raw, "description", None)
        description = re.sub(r"\s+", " ", str(value or "")).strip()
        key = description.casefold()
        if not description or key in seen:
            continue
        seen.add(key)
        descriptions.append(description[:500])
        if len(descriptions) >= 8:
            break
    return tuple(descriptions)


def _uniquely_anchored_document_indexes(
    values: Iterable[Any],
    *,
    rows: list[dict[str, Any]],
    document_terms: list[set[str]],
    allow_short_terms: bool = False,
) -> set[int]:
    """Return documents independently anchored by concrete request terms.

    A term occurring in one document is an anchor, not proof that the other
    documents are alternatives.  If different concrete terms anchor different
    documents, the documents are normally complementary parts of the requested
    answer (for example classification + amount, or risk explanation + config).
    Short CJK n-grams retain the existing filename-only safety rule.
    """

    anchored: set[int] = set()
    for value in values:
        for term in _topic_terms(value):
            matched = {
                index for index, terms in enumerate(document_terms) if term in terms
            }
            if len(matched) != 1:
                continue
            matched_index = next(iter(matched))
            if (
                len(term) >= 3
                or allow_short_terms
                or term in _topic_terms(rows[matched_index]["filename"])
            ):
                anchored.add(matched_index)
    return anchored


def _detect_topic_document_ambiguity(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    requirements: Iterable[Any] | None = None,
) -> EvidenceAmbiguityDecision | None:
    """Detect a broad query whose equally relevant hits are different docs.

    This gate is deliberately narrower than ``len(results) > 1``.  It needs
    distinct source labels, balanced retrieval support, and no query term that
    uniquely identifies one document.  Thus a normal multi-chunk answer or a
    query such as ``出差标准`` remains answerable, while ``员工标准是什么``
    asks the user to choose between independent policy documents.
    """

    text = str(query or "").strip()
    if (
        not text
        or _positive_pattern_match(_TOPIC_ALL_DOCUMENTS_RE, text) is not None
    ):
        return None
    # If source metadata already declares an applicability scope, the product/
    # version/project grouping above is the authoritative decision—even when a
    # compatibility document bridges those groups.  Topic labels must not
    # split such a source-anchored set a second time.
    scoped_documents = _document_scopes(candidates, QueryConstraints())
    if any(
        document.products or document.versions or document.projects
        for document in scoped_documents
    ):
        return None
    rows = _topic_document_rows(candidates)
    if len(rows) < 2:
        return None
    # Identical source labels do not give the user a meaningful choice.  Scope
    # metadata (handled by the caller) remains the correct disambiguator there.
    if len({_topic_label_key(row["filename"]) for row in rows}) < 2:
        return None
    if len(rows) > MAX_TOPIC_DOCUMENT_CHOICES:
        return EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension="document",
            question=(
                "检索到多篇可能相关的资料，但当前问题范围较宽。"
                "请补充具体主题或文档名称后再查询。"
            ),
            reason="too_many_mutually_relevant_documents",
            relevant_document_count=len(rows),
        )

    document_terms = [_topic_terms(row["text"]) for row in rows]
    query_anchors = _uniquely_anchored_document_indexes(
        (text,),
        rows=rows,
        document_terms=document_terms,
    )
    if query_anchors:
        # One concrete subject identifies a deterministic source.  Several
        # concrete subjects distributed across documents mean the request needs
        # those documents together; Chinese compact questions often omit an
        # explicit conjunction, so requiring ``和/与`` here turns normal
        # cross-document evidence chains into a false document choice.
        return None

    requirement_descriptions = _requirement_descriptions(requirements)
    requirement_anchors = _uniquely_anchored_document_indexes(
        requirement_descriptions,
        rows=rows,
        document_terms=document_terms,
        # Two or more explicit answer targets are already a strong structural
        # signal that independently matching documents are complementary.  In
        # that narrow negative-picker check, short Chinese nouns such as
        # ``时限`` and ``凭证`` are safe anchors: they can only suppress a
        # filename choice and can never select a product/version scope.
        allow_short_terms=len(requirement_descriptions) >= 2,
    )
    if len(requirement_anchors) >= 2:
        # The planner decomposed one answer across independently anchored
        # documents.  This signal can suppress only the filename/topic picker;
        # product/version/project conflicts are resolved before this function.
        return None

    # If there are raw retrieval scores, reject a long-tail candidate whose
    # support is less than roughly half of the best document.  RRF values are
    # ordering signals, not confidence; this is only a noise guard.
    scored = sorted(
        (float(row["best_score"]) for row in rows if row["best_score"] > 0),
        reverse=True,
    )
    if len(scored) >= 2 and scored[1] < scored[0] * 0.5:
        return None
    return EvidenceAmbiguityDecision(
        needs_clarification=True,
        dimension="document",
        question=(
            "检索到多篇可能相关的资料，但当前问题范围较宽：\n"
            + "\n".join(
                f"{index}. 《{row['filename'][:MAX_CHOICE_TEXT_CHARS]}》"
                for index, row in enumerate(rows, start=1)
            )
            + "\n请问需要查询哪一篇？也可以回复“都要”。"
        ),
        reason="multiple_mutually_relevant_documents",
        choices=_topic_groups_to_choices(rows),
        relevant_document_count=len(rows),
    )


def _version_prefixes(version: str) -> tuple[str, ...]:
    parts = str(version or "").strip().split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return (str(version).strip(),) if str(version).strip() else ()
    return tuple(".".join(parts[:index]) for index in range(1, len(parts) + 1))


def _source_product_aliases(group: _ScopeGroup) -> tuple[set[str], set[str]]:
    """Return exact source products and their generation-free family names."""

    products = {
        str(value).strip()
        for value in (*group.products, *group.canonical_products)
        if str(value).strip()
    }
    bases = {
        _TRAILING_PRODUCT_GENERATION_RE.sub("", product).strip()
        for product in products
    }
    bases.discard("")
    return products, bases


def _add_scope_alias(
    index: dict[str, dict[str, Any]],
    *,
    alias: str,
    dimension: str,
    group_index: int,
) -> None:
    normalized = re.sub(r"\s+", " ", str(alias or "").strip()).casefold()
    if not normalized:
        return
    entry = index.setdefault(
        normalized,
        {"alias": str(alias).strip(), "dimensions": set(), "groups": set()},
    )
    entry["dimensions"].add(dimension)
    entry["groups"].add(group_index)


def _scope_alias_index(groups: list[_ScopeGroup]) -> dict[str, dict[str, Any]]:
    """Build aliases exclusively from source-declared scope identities."""

    index: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(groups):
        products, product_bases = _source_product_aliases(group)
        for product in products | product_bases:
            _add_scope_alias(
                index,
                alias=product,
                dimension="product",
                group_index=group_index,
            )
        for project in group.projects:
            _add_scope_alias(
                index,
                alias=project,
                dimension="project",
                group_index=group_index,
            )
        for version in group.versions:
            for prefix in _version_prefixes(version):
                for alias in (
                    f"v{prefix}",
                    f"版本{prefix}",
                    f"{prefix}版",
                    f"{prefix}版本",
                ):
                    _add_scope_alias(
                        index,
                        alias=alias,
                        dimension="version",
                        group_index=group_index,
                    )
                for product in product_bases:
                    for alias in (
                        f"{product}{prefix}",
                        f"{product} {prefix}",
                        f"{product}v{prefix}",
                        f"{product} v{prefix}",
                        f"{product}版本{prefix}",
                        f"{product}{prefix}版",
                    ):
                        _add_scope_alias(
                            index,
                            alias=alias,
                            dimension="version",
                            group_index=group_index,
                        )
    return index


def _query_contains_scope_alias(query: str, alias: str) -> bool:
    escaped = re.escape(alias).replace(r"\ ", r"\s*")
    prefix = ""
    suffix = ""
    if alias and (alias[0].isascii() and alias[0].isalnum()):
        prefix = r"(?<![A-Za-z0-9_])"
    if alias and alias[0].isdigit():
        prefix = r"(?<![\d.])"
    if alias and (alias[-1].isascii() and alias[-1].isalpha()):
        suffix = r"(?![A-Za-z0-9_])"
    if alias and alias[-1].isdigit():
        suffix = r"(?![\d.])"
    return bool(re.search(f"{prefix}{escaped}{suffix}", query, re.IGNORECASE))


def _matched_scope_aliases(
    query: str,
    alias_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in alias_index.values()
        if _query_contains_scope_alias(query, str(entry["alias"]))
    ]


def _explicit_all_scope_dimension(query: str) -> tuple[bool, str | None]:
    text = str(query or "").strip()
    match = _positive_pattern_match(_EXPLICIT_ALL_DIMENSION_RE, text)
    if match is not None:
        raw_dimension = (
            match.group("dimension")
            or match.group("quantified_dimension")
            or match.group("dimension_first")
        )
        return True, {
            "产品": "product_version",
            "版本": "version",
            "项目": "project",
            "范围": None,
        }[raw_dimension]
    if _SHORT_ALL_SCOPES_RE.fullmatch(text):
        return True, None
    return False, None


def _groups_vary_on_dimension(
    groups: list[_ScopeGroup],
    dimension: str | None,
) -> bool:
    if len(groups) < 2:
        return False
    if dimension == "product_version":
        values = {
            tuple(sorted(group.canonical_products, key=str.casefold))
            for group in groups
            if group.canonical_products
        }
        return len(values) > 1
    if dimension == "version":
        values = {
            tuple(sorted(group.versions, key=_version_key))
            for group in groups
            if group.versions
        }
        return len(values) > 1
    if dimension == "project":
        values = {
            tuple(sorted(group.projects, key=str.casefold))
            for group in groups
            if group.projects
        }
        return len(values) > 1
    return _scope_dimension(groups) is not None


def _all_scope_group_indexes(
    *,
    matched_aliases: list[dict[str, Any]],
    group_count: int,
    target_dimension: str | None,
) -> set[int]:
    """Apply source-named context while retaining every requested scope."""

    selected = set(range(group_count))
    context_by_dimension: dict[str, set[int]] = {}
    for entry in matched_aliases:
        dimensions = set(entry["dimensions"])
        for dimension in dimensions:
            if dimension == target_dimension:
                continue
            context_by_dimension.setdefault(dimension, set()).update(entry["groups"])
    for group_indexes in context_by_dimension.values():
        selected.intersection_update(group_indexes)
    return selected


def resolve_explicit_scope_comparison(
    *,
    query: str,
    constraints: QueryConstraints,
    candidates: list[dict[str, Any]],
) -> ExplicitScopeComparisonPlan:
    """Resolve a first-turn comparison using only eligible source identities.

    Enumerated requests (for example, two named versions or projects) return a
    plan only when at least two aliases are unique across the candidate groups.
    Explicit all-scope requests retain every safely related group and never
    truncate the list to the clarification UI's normal six-choice limit.
    """

    text = str(query or "").strip()
    all_requested, all_dimension = _explicit_all_scope_dimension(text)
    enumerated_requested = bool(
        _positive_pattern_match(_COMPARISON_REQUEST_RE, text)
        or _positive_pattern_match(_NAMED_MULTI_SCOPE_REQUEST_RE, text)
    )
    if not candidates:
        return ExplicitScopeComparisonPlan(matched=False, reason="no_candidates")
    if not enumerated_requested and not all_requested:
        return ExplicitScopeComparisonPlan(
            matched=False,
            reason="not_explicit_scope_comparison",
        )

    # The generic query parser can bind the first mentioned version.  A
    # comparison must inspect every eligible source group before applying an
    # allow-list, otherwise the second explicitly named scope disappears.
    documents = _document_scopes(candidates, QueryConstraints())
    groups, dimension = _scope_groups(documents, QueryConstraints())
    if dimension is None or len(groups) < 2:
        return ExplicitScopeComparisonPlan(
            matched=False,
            reason="insufficient_mutually_exclusive_scopes",
        )

    ordered_groups = _ordered_scope_groups(groups)
    alias_index = _scope_alias_index(ordered_groups)
    matched_aliases = _matched_scope_aliases(text, alias_index)

    if enumerated_requested:
        enumerated_indexes = {
            next(iter(entry["groups"]))
            for entry in matched_aliases
            if len(entry["groups"]) == 1
        }
        if len(enumerated_indexes) >= 2:
            selected_groups = [
                group
                for index, group in enumerate(ordered_groups)
                if index in enumerated_indexes
            ]
            selected_dimension = _scope_dimension(selected_groups) or dimension
            if len(selected_groups) > MAX_AMBIGUITY_CHOICES:
                return ExplicitScopeComparisonPlan(
                    matched=False,
                    dimension=selected_dimension,
                    reason="too_many_explicit_scopes_for_complete_plan",
                )
            choices = _scope_groups_to_choices(selected_groups, constraints)
            return ExplicitScopeComparisonPlan(
                matched=True,
                dimension=selected_dimension,
                choices=choices,
                allowed_doc_ids=tuple(sorted({
                    doc_id
                    for group in selected_groups
                    for doc_id in group.doc_ids
                })),
                reason="explicit_enumerated_scopes",
            )

    if not all_requested:
        return ExplicitScopeComparisonPlan(
            matched=False,
            reason="enumerated_scope_aliases_not_unique",
        )

    selected_indexes = _all_scope_group_indexes(
        matched_aliases=matched_aliases,
        group_count=len(ordered_groups),
        target_dimension=all_dimension,
    )
    selected_groups = [
        group
        for index, group in enumerate(ordered_groups)
        if index in selected_indexes
    ]
    selected_dimension = all_dimension or _scope_dimension(selected_groups)
    if len(selected_groups) > MAX_AMBIGUITY_CHOICES:
        return ExplicitScopeComparisonPlan(
            matched=False,
            dimension=selected_dimension,
            reason="too_many_explicit_scopes_for_complete_plan",
        )
    if not _groups_vary_on_dimension(selected_groups, selected_dimension):
        return ExplicitScopeComparisonPlan(
            matched=False,
            reason="all_scope_context_not_safe",
        )
    choices = _scope_groups_to_choices(selected_groups, constraints)
    return ExplicitScopeComparisonPlan(
        matched=True,
        dimension=selected_dimension,
        choices=choices,
        allowed_doc_ids=tuple(sorted({
            doc_id
            for group in selected_groups
            for doc_id in group.doc_ids
        })),
        reason="explicit_all_scopes",
    )


def detect_evidence_scope_ambiguity(
    *,
    query: str,
    constraints: QueryConstraints,
    candidates: list[dict[str, Any]],
    requirements: Iterable[Any] | None = None,
    mode: AmbiguityDetectionMode = "combined",
) -> EvidenceAmbiguityDecision:
    """Return a clarification decision for mutually exclusive scopes.

    ``combined`` preserves the compatibility behavior for existing callers.
    Retrieval pre-gates must pass ``applicability_only``: that mode evaluates
    only source-anchored product/version/project scopes and can never emit a
    document/topic choice or use one as a retrieval rescue set.  New pipelines
    should run :func:`detect_post_evidence_document_ambiguity` after evidence
    adjudication for the document phase.
    """

    if mode not in _AMBIGUITY_DETECTION_MODES:
        raise ValueError(f"unsupported ambiguity detection mode: {mode}")

    if not candidates:
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="no_candidates",
        )
    comparison_plan = resolve_explicit_scope_comparison(
        query=query,
        constraints=constraints,
        candidates=candidates,
    )
    if comparison_plan.matched:
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            dimension=comparison_plan.dimension,
            reason=(
                "query_requests_all_scopes"
                if comparison_plan.reason == "explicit_all_scopes"
                else comparison_plan.reason
            ),
            choices=comparison_plan.choices,
            relevant_document_count=len(comparison_plan.allowed_doc_ids),
            allowed_doc_ids=comparison_plan.allowed_doc_ids,
        )
    if comparison_plan.reason == "too_many_explicit_scopes_for_complete_plan":
        return EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension=comparison_plan.dimension,
            question=(
                "检索到超过 6 个互不相同的适用范围，无法在一次选择中完整列出。"
                "请补充具体产品、版本或项目，以缩小查询范围。"
            ),
            reason="too_many_mutually_exclusive_scopes",
            choices=(),
            relevant_document_count=len({
                str(item.get("doc_id") or "")
                for item in candidates
                if str(item.get("doc_id") or "").strip()
            }),
        )
    explicit_all_requested, _ = _explicit_all_scope_dimension(query)
    if explicit_all_requested:
        # Preserve the established no-clarification behavior when the user
        # explicitly asks for all scopes but the eligible candidate set cannot
        # safely form two source-anchored groups.  No incomplete choice list is
        # exposed in that case.
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="query_requests_all_scopes",
        )

    # Establish the comparison baseline with the product constraint (if any),
    # but without the query's explicit version.  Version/project labels are
    # the dimensions this function narrows; applying the full constraint first
    # would remove every non-target document before we can publish the
    # request-local allow-list (notably for product-less ``2025版`` queries).
    baseline_constraints = (
        QueryConstraints(product=constraints.product)
        if constraints.product
        else QueryConstraints()
    )
    unfiltered_documents = _document_scopes(candidates, baseline_constraints)
    documents = _filter_explicit_query_project(
        query,
        _filter_explicit_query_version(
            query,
            unfiltered_documents,
            constraints,
        ),
    )
    unfiltered_doc_ids = {document.doc_id for document in unfiltered_documents}
    filtered_doc_ids = {document.doc_id for document in documents}
    # The two filters narrow only when the query names exactly one
    # source-declared version or project.  Preserve the default empty value for
    # ordinary questions so downstream execution never mistakes all retrieved
    # documents for an explicit user selection.
    allowed_doc_ids = (
        tuple(sorted(filtered_doc_ids))
        if filtered_doc_ids < unfiltered_doc_ids
        else ()
    )
    if not constraints.has_scope_constraint:
        # A source-declared scope must be grounded in the user's concrete
        # subject before it can force a product/version/project picker.  This
        # is especially important when reranking failed: unverified candidates
        # stay available to the pipeline, but unrelated version headers cannot
        # hijack a policy question into a false clarification.
        anchored_document_keys = _query_anchored_document_keys(query, candidates)
        unverified_document_keys = _unverified_document_keys(candidates)
        documents = [
            document
            for document in documents
            if (
                not (document.products or document.versions or document.projects)
                or (document.kb_id, document.doc_id) not in unverified_document_keys
                or (document.kb_id, document.doc_id) in anchored_document_keys
            )
        ]
    groups, dimension = _scope_groups(documents, constraints)
    # Scope metadata is the preferred disambiguator.  Only when no explicit
    # product/version/project constraint is present, and those dimensions do
    # not already explain the result set, inspect independent document topics.
    # This catches broad requests such as ``员工标准是什么`` without turning a
    # scoped multi-document answer into a filename picker.
    if (
        mode == "combined"
        and dimension is None
        and not constraints.has_scope_constraint
    ):
        topic_decision = _detect_topic_document_ambiguity(
            query=query,
            candidates=candidates,
            requirements=requirements,
        )
        if topic_decision is not None:
            return topic_decision
    if dimension is None or len(groups) < 2:
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="single_or_overlapping_scope",
            relevant_document_count=len(documents),
            allowed_doc_ids=allowed_doc_ids,
        )
    if len(groups) > MAX_AMBIGUITY_CHOICES:
        # Do not silently discard a real alternative. A broad generic question
        # is safer than presenting a truncated choice list as exhaustive.
        return EvidenceAmbiguityDecision(
            needs_clarification=True,
            dimension=dimension,
            question="检索到多个互不相同的适用范围，请补充具体产品和版本。",
            reason="too_many_mutually_exclusive_scopes",
            relevant_document_count=len(documents),
            allowed_doc_ids=allowed_doc_ids,
        )

    choices = _scope_groups_to_choices(groups, constraints)

    lines = [
        "检索到与当前问题相关、但适用范围不同的资料：",
        *(f"{index}. {choice.label}" for index, choice in enumerate(choices, start=1)),
        "请问需要查询哪一项？也可以回复“都对比”。",
    ]
    return EvidenceAmbiguityDecision(
        needs_clarification=True,
        dimension=dimension,
        question="\n".join(lines),
        reason="multiple_mutually_exclusive_relevant_scopes",
        choices=choices,
        relevant_document_count=len(documents),
        allowed_doc_ids=allowed_doc_ids,
    )
