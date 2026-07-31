"""Deterministic post-retrieval ambiguity detection for scoped evidence.

The semantic router runs before retrieval and therefore cannot know whether the
authorized result set contains several mutually exclusive product/version
scopes.  This module evaluates already reranked candidates, groups them by
source-anchored document identity, and blocks generation when choosing one
scope would silently guess on the user's behalf.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from core.query_constraints import (
    QueryConstraints,
    canonical_product_name,
    evaluate_candidate_constraints,
    extract_document_constraint_identity,
    inherit_document_constraint_metadata,
)


# Keep this aligned with the Pipeline's verified evidence topic gate.  A second
# scope that is eligible to be shown or used by the Pipeline must not disappear
# merely because ambiguity detection applies a stricter, unrelated threshold.
AMBIGUITY_TOPIC_RELEVANCE_THRESHOLD = 0.30
MAX_AMBIGUITY_CHOICES = 6
MAX_CHOICE_TEXT_CHARS = 500

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

    return bool(_ALL_SCOPES_REQUEST_RE.search(str(query or "").strip()))


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rag_evidence_clarification.v1",
            "needs_clarification": self.needs_clarification,
            "dimension": self.dimension,
            "question": self.question,
            "reason": self.reason,
            "choices": [choice.to_dict() for choice in self.choices],
            "relevant_document_count": self.relevant_document_count,
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
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in enriched:
        kb_id = str(candidate.get("kb_id") or "").strip()
        doc_id = str(candidate.get("doc_id") or "").strip()
        if not kb_id or not doc_id:
            continue
        grouped.setdefault((kb_id, doc_id), []).append(candidate)

    documents: list[_DocumentScope] = []
    query_product = canonical_product_name(constraints.product or "")
    for (kb_id, doc_id), items in grouped.items():
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
            if constraints.has_product_constraint and evaluation.status in {
                "mismatch",
                "unknown",
            }:
                continue
            eligible.append(item)
        if not eligible:
            continue

        products: set[str] = set()
        canonical_products: set[str] = set()
        versions: set[str] = set()
        projects: set[str] = set()
        filenames: set[str] = set()
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
        if query_product and not canonical_products:
            canonical_products.add(query_product)
            products.add(str(constraints.product or "").strip())

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
    document_scope_counts: dict[str, int] = {}
    for group in ordered_groups:
        for doc_id in group.doc_ids:
            document_scope_counts[doc_id] = document_scope_counts.get(doc_id, 0) + 1

    choices: list[EvidenceScopeChoice] = []
    for index, group in enumerate(ordered_groups, start=1):
        anchor_doc_ids = {
            doc_id
            for doc_id in group.doc_ids
            if document_scope_counts.get(doc_id, 0) == 1
        }
        companion_doc_ids = set(group.doc_ids) - anchor_doc_ids
        if not anchor_doc_ids:
            # Distinct groups should own at least one document.  Retaining this
            # defensive fallback prevents a companion-only result from being
            # mistaken for proof that the selected scope was covered.
            anchor_doc_ids = set(group.doc_ids)
            companion_doc_ids = set()
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
            )
        )
    return tuple(choices)


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
    match = _EXPLICIT_ALL_DIMENSION_RE.search(text)
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
        _COMPARISON_REQUEST_RE.search(text)
        or _NAMED_MULTI_SCOPE_REQUEST_RE.search(text)
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
) -> EvidenceAmbiguityDecision:
    """Return a clarification decision for mutually exclusive relevant scopes."""

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

    documents = _filter_explicit_query_project(
        query,
        _filter_explicit_query_version(
            query,
            _document_scopes(candidates, constraints),
            constraints,
        ),
    )
    groups, dimension = _scope_groups(documents, constraints)
    if dimension is None or len(groups) < 2:
        return EvidenceAmbiguityDecision(
            needs_clarification=False,
            reason="single_or_overlapping_scope",
            relevant_document_count=len(documents),
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
    )
