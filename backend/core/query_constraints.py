"""查询中的产品/版本硬约束提取与候选证据校验。

这里的规则只处理可以从原文中直接解释的显式约束，不猜测产品版本，也不让
LLM 的高语义相关度覆盖已经确定的版本冲突。规则有意保持保守：识别不到时
返回 ``unknown``，由上层将候选作为相近资料而不是直接证据。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Literal, Mapping, Sequence


ConstraintStatus = Literal["exact", "compatible", "unknown", "mismatch", "neutral"]
ScopeDimension = Literal["product", "version", "project"]


@dataclass(frozen=True)
class ScopeSourceSpan:
    """A strictly source-authored applicability value.

    The analyzer never emits an applicability value.  A project in particular
    is useful only when this contract can point back to the exact span in the
    current user question (or a trusted requirement derived from it).  Keeping
    the source separately from the display value makes a later model rewrite
    unable to manufacture a project boundary.
    """

    dimension: ScopeDimension
    start: int
    end: int
    span: str
    origin: Literal["current_query", "trusted_requirement"] = "current_query"

    def __post_init__(self) -> None:
        if self.dimension not in {"product", "version", "project"}:
            raise ValueError("unsupported scope source dimension")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise ValueError("scope source start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise ValueError("scope source end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("scope source range is invalid")
        span = re.sub(r"\s+", " ", str(self.span or "")).strip()
        if not span or len(span) > 200:
            raise ValueError("scope source span is invalid")
        if self.origin not in {"current_query", "trusted_requirement"}:
            raise ValueError("scope source origin is invalid")
        object.__setattr__(self, "span", span)

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "start": self.start,
            "end": self.end,
            "span": self.span,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class ApplicabilityScope:
    """Canonical applicability boundary for one answer requirement/task.

    Product, version and project are one conjunctive contract.  Values may be
    absent, but a project becomes a hard query boundary only when it has a
    source-verified span.  The historical ``QueryConstraints`` name remains a
    compatibility alias below so old call sites cannot accidentally retain a
    divergent product/version-only representation.
    """

    product: str | None = None
    version: str | None = None
    project: str | None = None
    explicit_version: bool = False
    explicit_project: bool = False
    product_source: ScopeSourceSpan | None = None
    version_source: ScopeSourceSpan | None = None
    project_source: ScopeSourceSpan | None = None
    # Retained for compatibility with existing traces and callers.  It is a
    # diagnostic summary only; source spans are the actual authority.
    matched_text: str | None = None
    extraction_reason: str = "未发现显式适用范围约束"

    def __post_init__(self) -> None:
        def normalized(value: object) -> str | None:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            return text or None

        product = normalized(self.product)
        version = normalized(self.version)
        project = normalized(self.project)
        product_source = self.product_source
        version_source = self.version_source
        project_source = self.project_source
        for expected, source in (
            ("product", product_source),
            ("version", version_source),
            ("project", project_source),
        ):
            if source is not None:
                if not isinstance(source, ScopeSourceSpan):
                    raise ValueError("scope source must be a ScopeSourceSpan")
                if source.dimension != expected:
                    raise ValueError("scope source dimension does not match value")
        if product is None and product_source is not None:
            raise ValueError("product source requires product value")
        if version is None and version_source is not None:
            raise ValueError("version source requires version value")
        if project is None and project_source is not None:
            raise ValueError("project source requires project value")
        explicit_version = bool(self.explicit_version or version is not None)
        # An unproven project string can exist in a legacy/caller-provided
        # object for diagnostics, but it is never a hard scope.  This is the
        # key model-safety invariant: only a trusted source span grants the
        # project dimension execution authority.
        explicit_project = bool(
            self.explicit_project and project is not None and project_source is not None
        )
        matched_text = normalized(self.matched_text)
        reason = re.sub(r"\s+", " ", str(self.extraction_reason or "")).strip()
        if len(reason) > 500:
            raise ValueError("scope extraction reason exceeds 500 characters")
        object.__setattr__(self, "product", product)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "project", project)
        object.__setattr__(self, "explicit_version", explicit_version)
        object.__setattr__(self, "explicit_project", explicit_project)
        object.__setattr__(self, "matched_text", matched_text)
        object.__setattr__(self, "extraction_reason", reason)

    @property
    def has_hard_constraint(self) -> bool:
        """Historical product+version hard-boundary projection."""

        return bool(self.product and self.version and self.explicit_version)

    @property
    def has_product_constraint(self) -> bool:
        return bool(self.product)

    @property
    def has_version_constraint(self) -> bool:
        return bool(self.version and self.explicit_version)

    @property
    def has_project_constraint(self) -> bool:
        return bool(self.project and self.explicit_project and self.project_source)

    @property
    def has_scope_constraint(self) -> bool:
        return bool(
            self.has_product_constraint
            or self.has_version_constraint
            or self.has_project_constraint
        )

    @property
    def source_spans(self) -> tuple[ScopeSourceSpan, ...]:
        return tuple(
            item
            for item in (
                self.product_source,
                self.version_source,
                self.project_source,
            )
            if item is not None
        )

    @property
    def terms(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (self.product, self.version, self.project)
            if value
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "product": self.product,
            "version": self.version,
            "project": self.project,
            "explicit_version": self.explicit_version,
            "explicit_project": self.explicit_project,
            "sources": [item.to_dict() for item in self.source_spans],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "version": self.version,
            "project": self.project,
            "explicit_version": self.explicit_version,
            "explicit_project": self.explicit_project,
            "has_scope_constraint": self.has_scope_constraint,
            "source_spans": [item.to_dict() for item in self.source_spans],
            "matched_text": self.matched_text,
            "extraction_reason": self.extraction_reason,
            "fingerprint": self.fingerprint,
        }


# Kept intentionally as an alias rather than a second dataclass.  Every
# existing ``QueryConstraints`` consumer now receives the canonical scope
# shape, including project/source provenance, without a compatibility fork.
QueryConstraints = ApplicabilityScope


# 版本号允许单段（产品 6/7、制度 2024）和多段（8.6/8.6.1），但所有
# 使用处都必须检查完整边界，避免把 8.6.1 的前缀误判成 8.6。
_VERSION_PATTERN = r"\d{1,4}(?:\.\d{1,4}){0,3}"
_VERSION_BOUNDARY = rf"(?![\d.])"
_QUERY_CUE_RE = re.compile(
    rf"(?:我是|我使用的是|我用的是|当前使用|正在使用|使用的是|用的是|使用|"
    rf"针对|关于|适用于|基于)\s*"
    rf"(?P<product>[A-Za-z\u3400-\u9fff][A-Za-z0-9_\-\u3400-\u9fff]{{0,30}}?)"
    rf"\s*(?:版本|[vV])?\s*(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}",
    re.IGNORECASE,
)
_QUERY_ADJACENT_RE = re.compile(
    rf"(?P<product>[A-Za-z][A-Za-z0-9_.\-]{{1,30}}|[\u3400-\u9fff]{{2,16}})"
    rf"\s*(?:版本\s*|[vV]\s*)?(?P<version>\d{{1,4}}(?:\.\d{{1,4}}){{1,3}}){_VERSION_BOUNDARY}",
    re.IGNORECASE,
)
_QUERY_VERSION_LABEL_RE = re.compile(
    # An explicit ``产品名 + 版本`` label is safe to parse across mixed
    # Chinese/ASCII names.  Unlike the looser adjacent-number grammar, the
    # literal version cue prevents ordinary quantities from becoming scopes.
    rf"(?P<product>[A-Za-z\u3400-\u9fff][A-Za-z0-9_.\-\u3400-\u9fff]{{1,30}})"
    rf"\s*版本\s*(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}",
    re.IGNORECASE,
)
_QUERY_STANDALONE_VERSION_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?:版本\s*|[vV]\s*)"
    rf"(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}|"
    rf"(?<![A-Za-z0-9_.])(?P<suffix_version>{_VERSION_PATTERN})\s*"
    rf"(?:版|版本|年度(?:制度)?)"
    rf"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_QUERY_BARE_VERSION_RE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}",
    re.IGNORECASE,
)
_LEADING_QUERY_WORDS = (
    "请问",
    "帮我",
    "如何解决",
    "怎么解决",
    "如何配置",
    "怎么配置",
    "解决",
    "配置",
    "关于",
    "针对",
)
_PRODUCT_METADATA_KEYS = {
    "product",
    "product_name",
    "productname",
    "产品",
    "产品名称",
    "所属产品",
}
_VERSION_METADATA_KEYS = {
    "version",
    "product_version",
    "productversion",
    "版本",
    "产品版本",
}
_PROJECT_METADATA_KEYS = {
    "project",
    "project_name",
    "projectname",
    "项目",
    "项目名称",
    "所属项目",
}
_COMPATIBLE_VERSION_KEYS = {
    "compatible_versions",
    "compatible_version",
    "supported_versions",
    "supported_version",
    "兼容版本",
    "适用版本",
    "支持版本",
}
_NORMALIZED_PRODUCT_METADATA_KEYS = {
    re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", item.casefold())
    for item in _PRODUCT_METADATA_KEYS
}
_NORMALIZED_VERSION_METADATA_KEYS = {
    re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", item.casefold())
    for item in _VERSION_METADATA_KEYS
}
_NORMALIZED_PROJECT_METADATA_KEYS = {
    re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", item.casefold())
    for item in _PROJECT_METADATA_KEYS
}
_NORMALIZED_COMPATIBLE_VERSION_KEYS = {
    re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", item.casefold())
    for item in _COMPATIBLE_VERSION_KEYS
}

# Product aliases are business data, not language grammar.  They belong to the
# scoped terminology registry or source-authored document metadata and must
# never be embedded in this request parser.  The parser below recognizes only
# explicit labels and generic ASCII identifier/version syntax.
_QUERY_ASCII_PRODUCT_VERSION_RE = re.compile(
    rf"(?P<product>[A-Za-z][A-Za-z0-9_\-]{{1,30}})"
    rf"\s*(?:版本\s*|[vV]\s*)?(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}",
    re.IGNORECASE,
)
_QUERY_PRODUCT_LABEL_ONLY_RE = re.compile(
    r"产品\s*(?:名称|名)?\s*[：:]\s*"
    r"(?P<product>[A-Za-z\u3400-\u9fff][A-Za-z0-9_.\-\u3400-\u9fff]{1,30})",
    re.IGNORECASE,
)
_QUERY_PRODUCT_AND_VERSION_LABEL_RE = re.compile(
    rf"产品\s*(?:名称|名)?\s*[：:]\s*"
    rf"(?P<product>[A-Za-z\u3400-\u9fff][A-Za-z0-9_.\-\u3400-\u9fff]{{1,30}}?)"
    rf"\s*[,，;；]\s*(?:产品)?版本\s*[：:]\s*"
    rf"(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}",
    re.IGNORECASE,
)
_VERSION_UNIT_WORDS = re.compile(
    r"^\s*(?:个|台|套|节点|实例|用户|条|项|次|分钟|小时|天|人|页|条记录)"
)
_DOCUMENT_PRODUCT_FIELD_RE = re.compile(
    r"(?:所属产品|产品名称|产品(?!版本))\s*[：:]\s*([^\n\r>,，;；]+)",
    re.IGNORECASE,
)
_DOCUMENT_VERSION_FIELD_RE = re.compile(
    r"产品版本\s*[：:]\s*",
    re.IGNORECASE,
)
_GENERIC_DOCUMENT_VERSION_FIELD_RE = re.compile(
    r"(?<!产品)版本\s*[：:]\s*",
    re.IGNORECASE,
)
_DOCUMENT_PROJECT_FIELD_RE = re.compile(
    r"(?:所属项目|项目名称|项目(?!版本))\s*[：:]\s*([^\n\r>,，;；]+)",
    re.IGNORECASE,
)
# These labels are intentionally stricter than the general identity parser
# above.  A product name in an operation description (for example, a target
# system named in one step) is useful for retrieval, but it does not prove
# that the document is partitioned into mutually exclusive applicability
# scopes.  Only an explicit scope header may participate in the final
# fail-closed clarification guard.
_DECLARED_APPLICABILITY_PRODUCT_FIELD_RE = re.compile(
    r"(?im)(?:^|[\n\r；;。]|>>)[ \t>*#-]*(?:适用产品|所属产品)"
    r"\s*[：:]\s*(?P<value>[^\n\r；;。>]+)",
)
_DECLARED_APPLICABILITY_VERSION_FIELD_RE = re.compile(
    r"(?im)(?:^|[\n\r；;。]|>>)[ \t>*#-]*(?:适用版本|所属版本|产品版本)"
    r"\s*[：:]\s*(?P<value>[^\n\r；;。>]+)",
)
_DECLARED_APPLICABILITY_PROJECT_FIELD_RE = re.compile(
    r"(?im)(?:^|[\n\r；;。]|>>)[ \t>*#-]*(?:适用项目|所属项目)"
    r"\s*[：:]\s*(?P<value>[^\n\r；;。>]+)",
)
# A version declaration is a small, typed grammar rather than an arbitrary
# stretch of prose after ``产品版本：``.  In particular, a sentence such as
# ``产品版本：6。住宿上限为 1200 元/天`` has exactly one declared version.  The
# old field regex captured the entire sentence and later extracted both ``6``
# and ``1200`` as versions, turning valid version-6 evidence into a false
# conflict.  Keep lists deliberately narrow: a list separator must be
# followed immediately by another declaration entry; normal prose does not
# qualify as a second version.
_DECLARED_VERSION_ENTRY_RE = re.compile(
    rf"""
    [ \t]*
    (?:
        (?:版本[ \t]*|[vV][ \t]*)
        |
        (?P<product_prefix>
            (?:[A-Za-z][A-Za-z0-9_.\- \t]{{0,60}}|[㐀-鿿][A-Za-z0-9_\-\u3400-鿿 \t]{{0,60}})
        )[ \t]*
        |
        [ \t]*
    )
    (?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}
    [ \t]*(?:版(?:本)?|全系|系列)?
    """,
    re.IGNORECASE | re.VERBOSE,
)
_DECLARED_VERSION_LIST_SEPARATOR_RE = re.compile(
    r"[ \t]*(?:[、,/，|]|(?:和|及|或)[ \t]*)[ \t]*",
    re.IGNORECASE,
)
# These are sentence/grammar words, not product identities.  The parser only
# permits a product prefix to preserve common forms like ``产品甲6全系`` and
# ``ProductX 8.2.75``; rejecting these markers prevents a malformed prose
# value such as ``当前版本为 6`` from gaining scope authority.
_NON_IDENTITY_VERSION_PREFIX_MARKERS = (
    "当前",
    "本次",
    "本系统",
    "系统",
    "组件",
    "浏览器",
    "数据库",
    "运行时",
    "使用",
    "采用",
    "升级",
    "支持",
    "兼容",
    "适用",
    "对应",
    "为",
    "是",
    "有",
    "含",
)
# File names and ingestion headings are often the only source-anchored
# applicability signal available for a chunk. They are useful for scope
# grouping only when an ASCII identifier is adjacent to a version or a number
# is explicitly marked as a version/edition. Arbitrary path/date numbers are
# never versions.
_SOURCE_IDENTITY_METADATA_KEYS = {
    "source",
    "heading",
    "title",
    "filename",
    "file_name",
    "name",
    "path",
}
_SOURCE_VERSION_RE = re.compile(
    rf"(?<![\d.])(?:版本\s*|[vV]\s*)"
    rf"(?P<prefix_version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}|"
    rf"(?<![\d.])(?P<suffix_version>{_VERSION_PATTERN})\s*"
    rf"(?:版|版本|年度(?:制度)?)"
    rf"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SCOPE_NEGATION_PREFIX_RE = re.compile(
    r"(?:不要|无需|无须|并非|不是|不(?:要|需|查|看|对比|比较|包含|包括)|"
    r"排除|除外)[^，,。；;！!？?]{0,24}$",
    re.IGNORECASE,
)
_SCOPE_NEGATION_SUFFIX_RE = re.compile(
    r"^[^，,。；;！!？?]{0,10}(?:不要|无需|无须|不查|不看|不对比|不比较)",
    re.IGNORECASE,
)
_SCOPE_EXCLUSIVE_FOCUS_RE = re.compile(
    r"(?:只|仅)(?:需要|需|要|查|看|查询|查看|保留|使用|针对)?"
    r"[^，,。；;！!？?]{0,10}$",
    re.IGNORECASE,
)


def query_span_is_negated(text: str, start: int, end: int) -> bool:
    """Return whether a scope phrase is governed by a local negation."""

    source = str(text or "")
    clause_start = max(
        source.rfind(marker, 0, start)
        for marker in ("，", ",", "。", "；", ";", "！", "!", "？", "?")
    ) + 1
    left = source[clause_start:start]
    right = source[end:end + 16]
    return bool(
        _SCOPE_NEGATION_PREFIX_RE.search(left)
        or _SCOPE_NEGATION_SUFFIX_RE.search(right)
    )


def _scope_match_has_exclusive_focus(text: str, start: int) -> bool:
    source = str(text or "")
    clause_start = max(
        source.rfind(marker, 0, start)
        for marker in ("，", ",", "。", "；", ";", "！", "!", "？", "?")
    ) + 1
    return bool(_SCOPE_EXCLUSIVE_FOCUS_RE.search(source[clause_start:start]))


def _alias_search_pattern(alias: str) -> re.Pattern[str]:
    """Build an alias matcher without treating CJK as an ASCII word char."""

    escaped = re.escape(alias)
    if re.fullmatch(r"[A-Za-z0-9_.+\- ]+", alias):
        escaped = escaped.replace(r"\ ", r"\s+")
        return re.compile(
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
    return re.compile(escaped, re.IGNORECASE)


@dataclass(frozen=True)
class ConstraintEvaluation:
    status: ConstraintStatus
    reason: str
    candidate_products: tuple[str, ...] = ()
    candidate_versions: tuple[str, ...] = ()
    candidate_projects: tuple[str, ...] = ()
    # A rejected candidate can explain *which dimensions* conflict without
    # carrying its body text into a trace, ledger or client payload.
    mismatch_dimensions: tuple[ScopeDimension, ...] = ()
    # ``global_compatible`` is deliberately distinct from exact scope match.
    # Generation can then say that a source is a global clause rather than
    # falsely presenting it as project-specific policy.
    scope_applicability: Literal[
        "exact",
        "global_compatible",
        "unscoped",
        "unknown",
    ] = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "candidate_products": list(self.candidate_products),
            "candidate_versions": list(self.candidate_versions),
            "candidate_projects": list(self.candidate_projects),
            "mismatch_dimensions": list(self.mismatch_dimensions),
            "scope_applicability": self.scope_applicability,
        }


@dataclass(frozen=True)
class ScopeCandidateRejection:
    """Content-free rejection handoff from scope admission to the ledger."""

    kb_id: str
    doc_id: str
    chunk_id: str
    expected_scope_fingerprint: str
    actual_identity_fingerprint: str
    mismatch_dimensions: tuple[ScopeDimension, ...]
    reason_code: str

    def __post_init__(self) -> None:
        for field_name in ("kb_id", "doc_id", "chunk_id"):
            value = re.sub(r"\s+", " ", str(getattr(self, field_name) or "")).strip()
            object.__setattr__(self, field_name, value)
        for field_name in (
            "expected_scope_fingerprint",
            "actual_identity_fingerprint",
        ):
            value = str(getattr(self, field_name) or "").strip().casefold()
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field_name} must be a sha256 fingerprint")
            object.__setattr__(self, field_name, value)
        dimensions = tuple(dict.fromkeys(self.mismatch_dimensions))
        if not dimensions or any(
            value not in {"product", "version", "project"}
            for value in dimensions
        ):
            raise ValueError("scope rejection requires mismatch dimensions")
        object.__setattr__(self, "mismatch_dimensions", dimensions)
        reason = re.sub(r"\s+", "_", str(self.reason_code or "")).strip("_").casefold()
        if not reason or len(reason) > 120:
            raise ValueError("scope rejection reason is invalid")
        object.__setattr__(self, "reason_code", reason)

    def to_dict(self) -> dict[str, Any]:
        # Intentionally omit candidate content, filename and arbitrary
        # metadata.  Identity/fingerprint diagnostics are sufficient to audit
        # the decision without leaking rejected business text.
        return {
            "kb_id": self.kb_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "expected_scope_fingerprint": self.expected_scope_fingerprint,
            "actual_identity_fingerprint": self.actual_identity_fingerprint,
            "mismatch_dimensions": list(self.mismatch_dimensions),
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class ScopeAdmission:
    """Typed candidate admission outcome with no rejected-body retention."""

    candidates: tuple[dict[str, Any], ...] = ()
    rejections: tuple[ScopeCandidateRejection, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(item, Mapping) for item in self.candidates):
            raise ValueError("scope admission candidates must be mappings")
        if any(
            not isinstance(item, ScopeCandidateRejection)
            for item in self.rejections
        ):
            raise ValueError("scope admission rejections are invalid")


@dataclass(frozen=True)
class DocumentConstraintIdentity:
    """Explicit, document-level applicability facts anchored in source text."""

    products: tuple[str, ...] = ()
    canonical_products: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentApplicabilityDeclaration:
    """Explicit source declarations eligible for document-scope refinement.

    ``DocumentConstraintIdentity`` intentionally has a broad remit: it also
    supports retrieval admission where source titles and ordinary product
    mentions are useful signals.  This narrower representation is only for
    deciding whether a fully covered answer must be withheld for an unknown
    applicability partition.

    Origins are bounded diagnostic categories in ``dimension:origin`` form;
    no source text or declared value is retained in trace diagnostics.
    """

    identity: DocumentConstraintIdentity = DocumentConstraintIdentity()
    origins: tuple[str, ...] = ()


def _canonical_project(value: object) -> str:
    """Normalize a project identity without fuzzy/substring matching.

    Project names are tenant-like applicability boundaries.  Matching a
    prefix (for example ``中青建安`` to ``中青建安二期``) would silently broaden a
    request, so this helper deliberately permits only a narrow cosmetic
    normalisation: punctuation/whitespace and one trailing ``项目``/``工程``.
    """

    normalized = _normalize_product(str(value or ""))
    for suffix in ("项目", "工程"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _normalize_version(value: str) -> str:
    raw = re.sub(r"^(?:版本|v)\s*", "", str(value).strip(), flags=re.IGNORECASE)
    parts = raw.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return raw.lower()
    return ".".join(str(int(part)) for part in parts)


def _normalize_product(value: str) -> str:
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", str(value).casefold())


def _canonical_product(value: str) -> str:
    normalized = _normalize_product(value)
    # 去掉明确的产品后缀，而不是允许任意一方作为另一方的 substring。
    for suffix in ("platform", "system", "平台", "系统"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _product_aliases(value: str) -> tuple[str, ...]:
    return (value,)


def _clean_product(value: str) -> str:
    cleaned = value.strip(" \t\r\n，。！？：:、/\\-_()（）[]【】")
    changed = True
    while changed:
        changed = False
        for prefix in _LEADING_QUERY_WORDS:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
                cleaned = cleaned[len(prefix):]
                changed = True
    return cleaned.strip()


def _query_product_is_identity(text: str, match: re.Match[str]) -> bool:
    """Return whether a parsed product token has source-level identity.

    A bare Chinese word adjacent to a number is not enough to establish an
    applicability boundary: verbs such as ``升级8.6`` and nouns such as
    ``部署8.6`` are ordinary query language, not product names.  Product
    ASCII identifiers remain valid; Chinese names require an explicit product
    label or a source-derived terminology binding outside this parser.
    """

    raw = match.groupdict().get("product") or ""
    product = _clean_product(raw)
    if not product:
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.\-]{1,30}", product):
        return True
    # Explicit source grammar, e.g. ``产品名称：某平台 8.6`` or
    # ``某平台产品版本 8.6``.  This is deliberately structural rather than a
    # list of question-specific verbs.
    start, end = match.span("product")
    prefix = text[max(0, start - 12):start]
    suffix = text[end:min(len(text), end + 12)]
    return bool(
        re.search(r"产品\s*(?:名称|名)?\s*[：:]?\s*$", prefix)
        or re.search(
            r"(?:我是|我使用的是|我用的是|当前使用|正在使用|"
            r"使用的是|用的是|使用|针对|关于|适用于|基于)\s*$",
            prefix,
        )
        or re.search(r"产品\s*(?:版本|v)?", suffix, re.IGNORECASE)
        or re.search(r"系统|平台|应用", suffix)
    )


def _valid_query_match(text: str, match: re.Match[str]) -> bool:
    """避免把“产品甲 8 个节点”这类数量误识别为版本。"""

    groups = match.groupdict()
    version = groups.get("version") or groups.get("suffix_version") or ""
    if "." not in version:
        tail = text[match.end():]
        if _VERSION_UNIT_WORDS.match(tail):
            return False
    return True


def _extract_product_version_constraint_legacy(query: str) -> ApplicabilityScope:
    """提取相邻的产品名和显式版本号。

    优先识别“我是/使用/针对 + 产品 + 版本”这类强提示，再识别
    ``产品甲8.6``、``ProductX v8.6`` 等紧邻写法；孤立的带点版本号可形成
    product-less 版本范围，普通动作词不会被当作产品。
    """

    text = str(query or "").strip()
    if not text:
        return ApplicabilityScope()

    match: re.Match[str] | None = None
    # Collect every positive match at the strongest available syntax tier.
    # A comparison/mixed query containing several distinct positive scopes has
    # no safe *global* constraint; requirement-level planning will preserve
    # each scope independently.  An explicit ``只看/仅查`` clause may narrow a
    # previously negated comparison to one final scope.
    for pattern in (
        _QUERY_PRODUCT_AND_VERSION_LABEL_RE,
        _QUERY_ASCII_PRODUCT_VERSION_RE,
        _QUERY_CUE_RE,
        _QUERY_STANDALONE_VERSION_RE,
        _QUERY_VERSION_LABEL_RE,
        _QUERY_ADJACENT_RE,
        _QUERY_BARE_VERSION_RE,
    ):
        matches = [
            candidate
            for candidate in pattern.finditer(text)
            if _valid_query_match(text, candidate)
            and not (
                pattern is _QUERY_ASCII_PRODUCT_VERSION_RE
                and re.fullmatch(
                    rf"[vV]\s*{_VERSION_PATTERN}",
                    candidate.group(0),
                )
            )
            and (
                "product" not in candidate.groupdict()
                or not candidate.group("product")
                or _query_product_is_identity(text, candidate)
            )
            and not query_span_is_negated(
                text,
                candidate.start(),
                candidate.end(),
            )
        ]
        if not matches:
            continue
        focused = [
            candidate
            for candidate in matches
            if _scope_match_has_exclusive_focus(text, candidate.start())
        ]
        if focused:
            match = max(focused, key=lambda candidate: candidate.start())
            break
        identities = {
            (
                _canonical_product(_clean_product(
                    candidate.groupdict().get("product") or ""
                )),
                _normalize_version(
                    candidate.groupdict().get("version")
                    or candidate.groupdict().get("suffix_version")
                    or ""
                ),
            )
            for candidate in matches
        }
        if len(identities) > 1:
            return ApplicabilityScope(
                extraction_reason=(
                    "查询包含多个独立的正向产品或版本范围，"
                    "不生成会污染兄弟需求的全局约束"
                ),
            )
        match = matches[0]
        if match is not None:
            break
    # Product-only identity cannot be inferred without scoped terminology or
    # an explicit product label.  Keep the query unbound and let authorized
    # catalog resolution propose candidates instead of guessing a product.
    if match is None:
        product_label = next(
            (
                candidate
                for candidate in _QUERY_PRODUCT_LABEL_ONLY_RE.finditer(text)
                if not query_span_is_negated(
                    text,
                    candidate.start(),
                    candidate.end(),
                )
            ),
            None,
        )
        if product_label is not None:
            matched_text = product_label.group(0)
            product = _clean_product(product_label.group("product"))
            return ApplicabilityScope(
                product=product,
                product_source=_scope_source(
                    dimension="product",
                    query=text,
                    start=product_label.start("product"),
                    end=product_label.end("product"),
                ),
                matched_text=matched_text,
                extraction_reason="由查询中的显式产品标签识别产品约束",
            )
        return ApplicabilityScope()

    raw_product = match.groupdict().get("product") or ""
    raw_version = (
        match.groupdict().get("version")
        or match.groupdict().get("suffix_version")
        or ""
    )
    product = _clean_product(raw_product) if raw_product else ""
    version = _normalize_version(raw_version)
    if not version:
        return QueryConstraints()

    matched_text = match.group(0).strip()
    if not product:
        return ApplicabilityScope(
            product=None,
            version=version,
            explicit_version=True,
            matched_text=matched_text,
            extraction_reason=(
                f"由查询片段“{matched_text}”识别出显式版本“{version}”，"
                "未指定产品"
            ),
        )
    return ApplicabilityScope(
        product=product,
        version=version,
        explicit_version=True,
        matched_text=matched_text,
        extraction_reason=(
            f"由查询片段“{matched_text}”识别出产品“{product}”和显式版本“{version}”"
        ),
    )


# Project detection is intentionally narrow.  A plain proper noun is not a
# project merely because a model says so; it must appear in an explicit source
# construction such as ``中青建安项目中`` / ``项目：中青建安``.  A possessive
# prefix such as ``某某的产品`` is only a semantic qualifier candidate; it is
# not authoritative project scope until authorized document metadata confirms
# it later in the evidence path.
_PROJECT_LABEL_PREFIX_RE = re.compile(
    r"(?:项目名称|所属项目|项目)\s*[：:]\s*"
    r"(?P<project>[A-Za-z0-9_\-\u3400-\u9fff]{2,48})",
    re.IGNORECASE,
)
_PROJECT_LABEL_SUFFIX_RE = re.compile(
    r"(?P<project>[A-Za-z0-9_\-\u3400-\u9fff]{2,48}?)"
    # A bare terminal ``项目`` is an ordinary noun (for example ``有哪些项目``),
    # not an applicability declaration.  Require a structural scope suffix.
    r"(?:项目|工程)(?:的|中|内|下|里|范围内)",
    re.IGNORECASE,
)
_PROJECT_GENERIC_VALUES = frozenset({
    "项目", "工程", "公司", "企业", "系统", "平台", "产品", "版本", "当前项目",
})


def _scope_source(
    *,
    dimension: ScopeDimension,
    query: str,
    start: int,
    end: int,
) -> ScopeSourceSpan | None:
    if start < 0 or end <= start or end > len(query):
        return None
    span = query[start:end]
    if not span.strip():
        return None
    return ScopeSourceSpan(
        dimension=dimension,
        start=start,
        end=end,
        span=span,
        origin="current_query",
    )


def _project_source_spans(query: str) -> tuple[ScopeSourceSpan, ...]:
    source = str(query or "")
    values: list[ScopeSourceSpan] = []
    seen: set[tuple[int, int]] = set()

    def add(match: re.Match[str]) -> None:
        start, end = match.span("project")
        value = source[start:end].strip()
        # The suffix grammar captures the name before ``项目/工程``.  Preserve
        # the literal source span; do not repair values with phrase blacklists.
        for suffix in ("项目", "工程"):
            if value.endswith(suffix) and len(value) > len(suffix):
                end -= len(suffix)
                value = source[start:end].strip()
                break
        if (
            not value
            or value.casefold() in _PROJECT_GENERIC_VALUES
            or (start, end) in seen
            or query_span_is_negated(source, start, end)
        ):
            return
        item = _scope_source(
            dimension="project",
            query=source,
            start=start,
            end=end,
        )
        if item is not None:
            seen.add((start, end))
            values.append(item)

    for pattern in (_PROJECT_LABEL_PREFIX_RE, _PROJECT_LABEL_SUFFIX_RE):
        for match in pattern.finditer(source):
            add(match)

    return tuple(values)


def _positive_product_version_scopes(query: str) -> tuple[ApplicabilityScope, ...]:
    """Return every independently source-anchored product/version scope.

    This does not decide which answer unit owns a scope; it simply preserves
    all literal alternatives so the planner can compile comparison units
    without a global-product/version shortcut.
    """

    source = str(query or "")
    if not source.strip():
        return ()
    # Generic ASCII identifiers and explicit labels are language structure;
    # product aliases are resolved by scoped terminology outside this parser.
    patterns = (
        _QUERY_PRODUCT_AND_VERSION_LABEL_RE,
        _QUERY_ASCII_PRODUCT_VERSION_RE,
        _QUERY_CUE_RE,
        _QUERY_VERSION_LABEL_RE,
        _QUERY_ADJACENT_RE,
        _QUERY_STANDALONE_VERSION_RE,
        _QUERY_BARE_VERSION_RE,
    )
    raw: list[tuple[re.Match[str], str | None, str]] = []
    seen_ranges: set[tuple[int, int, str | None, str]] = set()
    for pattern in patterns:
        for match in pattern.finditer(source):
            if (
                not _valid_query_match(source, match)
                or (
                    pattern is _QUERY_ASCII_PRODUCT_VERSION_RE
                    and re.fullmatch(
                        rf"[vV]\s*{_VERSION_PATTERN}",
                        match.group(0),
                    )
                )
                or query_span_is_negated(source, match.start(), match.end())
            ):
                continue
            groups = match.groupdict()
            raw_product = groups.get("product") or ""
            raw_version = groups.get("version") or groups.get("suffix_version") or ""
            version = _normalize_version(raw_version)
            if not version:
                continue
            product = _clean_product(raw_product) if raw_product else None
            if (
                product
                and not _query_product_is_identity(source, match)
            ):
                continue
            # ``版本8.6`` is a product-less edition constraint.  The generic
            # adjacent-product pattern can otherwise misread the literal
            # label ``版本`` as a product and prevent the following standalone
            # edition matcher from producing the correct source-anchored
            # scope.  This is a grammar boundary, not a product allow-list.
            if product and _normalize_product(product) in {"版本", "version"}:
                continue
            key = (match.start(), match.end(), product, version)
            if key in seen_ranges:
                continue
            # A generic full sentence capture must not duplicate a stronger
            # structural match occupying the same version token.
            if any(
                existing_version == version
                and (
                    existing_start <= match.start() < existing_end
                    or match.start() <= existing_start < match.end()
                )
                for (existing_start, existing_end, _existing_product, existing_version)
                in seen_ranges
            ):
                continue
            seen_ranges.add(key)
            raw.append((match, product, version))

    projects = _project_source_spans(source)
    scopes: list[ApplicabilityScope] = []
    for match, product, version in raw:
        product_source = (
            _scope_source(
                dimension="product",
                query=source,
                start=match.start("product"),
                end=match.end("product"),
            )
            if product and "product" in match.groupdict() and match.group("product")
            else None
        )
        version_group = "version" if match.groupdict().get("version") else "suffix_version"
        version_source = _scope_source(
            dimension="version",
            query=source,
            start=match.start(version_group),
            end=match.end(version_group),
        )
        # A labelled project preceding all product/version alternatives applies
        # to each scope; a possessive prefix applies to its immediately
        # following product.  No candidate/document identity is consulted.
        project_source = next(
            (
                item
                for item in reversed(projects)
                if item.end <= match.start()
                and len(source[item.end:match.start()].strip(" ，,：:；;的")) <= 8
            ),
            None,
        )
        scopes.append(ApplicabilityScope(
            product=product,
            version=version,
            project=(project_source.span if project_source is not None else None),
            explicit_version=True,
            explicit_project=project_source is not None,
            product_source=product_source,
            version_source=version_source,
            project_source=project_source,
            matched_text=match.group(0).strip(),
            extraction_reason="由当前问题的产品/显式版本原文跨度识别适用范围",
        ))
    if scopes:
        return tuple(scopes)
    # A project-only request still has a strict applicability boundary.
    if len(projects) == 1:
        project_source = projects[0]
        return (ApplicabilityScope(
            project=project_source.span,
            explicit_project=True,
            project_source=project_source,
            matched_text=project_source.span,
            extraction_reason="由当前问题的项目原文跨度识别适用范围",
        ),)
    return ()


def extract_applicability_scopes(query: str) -> tuple[ApplicabilityScope, ...]:
    """Return all source-anchored applicability alternatives in a query."""

    return _positive_product_version_scopes(str(query or ""))


def extract_applicability_scope(query: str) -> ApplicabilityScope:
    """Return one safe scope, never collapsing independent alternatives."""

    source = str(query or "")
    scopes = extract_applicability_scopes(source)
    if len(scopes) == 1:
        return scopes[0]
    if len(scopes) > 1:
        return ApplicabilityScope(
            extraction_reason=(
                "查询包含多个独立的正向适用范围，"
                "不生成会污染兄弟需求的全局约束"
            ),
        )

    legacy = _extract_product_version_constraint_legacy(source)
    if legacy.has_scope_constraint:
        # Product-only constraints have no version match to enumerate above.
        product_source = None
        if legacy.product:
            match = _alias_search_pattern(legacy.product).search(source)
            if match is not None:
                product_source = _scope_source(
                    dimension="product",
                    query=source,
                    start=match.start(),
                    end=match.end(),
                )
        projects = _project_source_spans(source)
        project_source = next(
            (
                item
                for item in reversed(projects)
                if product_source is not None
                and item.end <= product_source.start
                and len(source[item.end:product_source.start].strip(" ，,：:；;的")) <= 8
            ),
            None,
        )
        return ApplicabilityScope(
            product=legacy.product,
            version=legacy.version,
            project=(project_source.span if project_source is not None else None),
            explicit_version=legacy.explicit_version,
            explicit_project=project_source is not None,
            product_source=product_source,
            project_source=project_source,
            matched_text=legacy.matched_text,
            extraction_reason=legacy.extraction_reason,
        )
    return legacy


def extract_query_constraints(query: str) -> ApplicabilityScope:
    """Compatibility entry point returning the canonical scope contract."""

    return extract_applicability_scope(query)


def _flatten_metadata(metadata: Any) -> list[tuple[str, Any]]:
    flattened: list[tuple[str, Any]] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, (list, tuple, set)):
            for child_value in value:
                visit(child_value, key)
        else:
            flattened.append((key, value))

    visit(metadata)
    return flattened


def _versions_from_value(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (dict, list, tuple, set)):
        raw = json.dumps(value, ensure_ascii=False, default=str)
    else:
        raw = str(value)
    return {
        _normalize_version(item)
        for item in re.findall(
            rf"(?<![\d.])({_VERSION_PATTERN})(?![\d.])",
            raw,
        )
    }


def _is_declared_version_product_prefix(prefix: str) -> bool:
    """Return whether a field-local prefix can safely name a product.

    ``产品版本`` is authoritative only when the version token is directly
    declared. An identifier containing an ASCII product token (``产品A1.0``)
    is safe enough. A free-form Chinese
    phrase is intentionally *not* promoted to product identity: otherwise
    ``住宿标准1200`` after a comma can masquerade as another version-list item.
    This conservative choice may leave an unusual, unregistered all-Chinese
    product prefix as unknown, but never turns a business amount into an
    incompatible version boundary.
    """

    compact = re.sub(r"\s+", "", str(prefix or ""))
    if not compact:
        return False
    normalized = _normalize_product(compact)
    if not normalized:
        return False
    if any(marker in compact for marker in _NON_IDENTITY_VERSION_PREFIX_MARKERS):
        return False
    # Unregistered identifiers are allowed only when their token grammar is
    # explicit enough to be an identity (for example ``产品A``/``PlatformX``),
    # rather than ordinary Chinese prose following a list separator.
    return bool(re.search(r"[A-Za-z]", compact))


def _declared_versions_after_field_label(
    text: str,
    start: int,
) -> tuple[set[str], int]:
    """Parse the controlled version list immediately after a field label.

    The returned offset is the end of the last accepted list entry.  Callers
    can use that bounded slice to decide whether a generic ``版本：`` field
    explicitly names the queried product, without examining unrelated body
    text later in the sentence.
    """

    source = str(text or "")
    cursor = max(0, min(int(start), len(source)))
    declared: set[str] = set()
    end = cursor
    for _ in range(16):
        match = _DECLARED_VERSION_ENTRY_RE.match(source, cursor)
        if match is None:
            break
        prefix = match.group("product_prefix") or ""
        if prefix and not _is_declared_version_product_prefix(prefix):
            break
        version = match.group("version") or ""
        if not version:
            break
        declared.add(_normalize_version(version))
        end = match.end()

        separator = _DECLARED_VERSION_LIST_SEPARATOR_RE.match(source, end)
        if separator is None:
            break
        cursor = separator.end()
    return declared, end


def _declared_versions_from_document_fields(text: str) -> set[str]:
    """Extract only values governed by explicit ``产品版本：`` declarations."""

    source = str(text or "")
    versions: set[str] = set()
    for match in _DOCUMENT_VERSION_FIELD_RE.finditer(source):
        declared, _ = _declared_versions_after_field_label(source, match.end())
        versions.update(declared)
    return versions


def _source_identity_texts(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Return bounded, source-authored identity labels for one candidate.

    Retrieval adapters use slightly different names for the original source
    label.  Restricting this helper to the filename and a small allow-list of
    ingestion metadata fields prevents arbitrary metadata values (or chunk
    bodies) from becoming an applicability identity.
    """

    texts: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        texts.append(text[:1000])

    for field in ("filename", "source", "heading", "title", "name", "path"):
        add(candidate.get(field))
    metadata = candidate.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.pop("ambiguous_document_identity", None)
    for key, value in _flatten_metadata(metadata or {}):
        normalized_key = _normalize_product(key)
        if normalized_key in {
            _normalize_product(item)
            for item in _SOURCE_IDENTITY_METADATA_KEYS
        }:
            add(value)
    return tuple(texts)


def _source_identity_facts(
    texts: Sequence[str],
) -> tuple[set[str], set[str]]:
    """Extract conservative product/version facts from source labels.

    An ASCII product identifier followed by a version is a strong identity
    signal. Product-less policies may use an explicit ``版``/``版本``/``年度``
    marker, while bare numbers remain ignored.  The latter is important for
    filenames containing dates, ticket numbers, or section counters.
    """

    products: set[str] = set()
    versions: set[str] = set()
    product_version_pattern = _QUERY_ASCII_PRODUCT_VERSION_RE

    for text in texts:
        source = str(text or "")
        for match in product_version_pattern.finditer(source):
            version = match.group("version") or ""
            if "." not in version and _VERSION_UNIT_WORDS.match(
                source[match.end():]
            ):
                # ``产品甲 8 个节点`` is a quantity, not a product version.
                continue
            products.add(match.group("product").strip())
            versions.add(_normalize_version(version))

        for match in _SOURCE_VERSION_RE.finditer(source):
            version = match.group("prefix_version") or match.group("suffix_version")
            if version:
                versions.add(_normalize_version(version))
    return products, versions


def _declared_document_identity(
    candidate: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    """Extract explicit document identity fields from one chunk.

    In addition to structured fields in the body/metadata, source labels are
    parsed conservatively so a document whose header chunk was not retrieved
    can still participate in ambiguity and comparison decisions.
    """

    products: set[str] = set()
    versions: set[str] = set()
    projects: set[str] = set()
    raw_metadata = candidate.get("metadata") or {}
    if isinstance(raw_metadata, dict):
        # This marker records why a dimension was deliberately *not*
        # inherited.  Its alternatives are diagnostics, not assertions that
        # the current chunk belongs to every listed scope.
        identity_metadata = dict(raw_metadata)
        identity_metadata.pop("ambiguous_document_identity", None)
    else:
        identity_metadata = raw_metadata
    for key, value in _flatten_metadata(identity_metadata):
        normalized_key = _normalize_product(key)
        if (
            normalized_key in _NORMALIZED_PRODUCT_METADATA_KEYS
            and value is not None
            and str(value).strip()
        ):
            products.add(str(value).strip())
        if normalized_key in _NORMALIZED_VERSION_METADATA_KEYS:
            versions.update(_versions_from_value(value))
        if (
            normalized_key in _NORMALIZED_PROJECT_METADATA_KEYS
            and value is not None
            and str(value).strip()
        ):
            projects.add(str(value).strip())

    content = str(candidate.get("content") or "")
    for match in _DOCUMENT_PRODUCT_FIELD_RE.finditer(content):
        value = match.group(1).strip()
        if value:
            products.add(value)
    versions.update(_declared_versions_from_document_fields(content))
    for match in _DOCUMENT_PROJECT_FIELD_RE.finditer(content):
        value = match.group(1).strip()
        if value:
            projects.add(value)
    source_products, source_versions = _source_identity_facts(
        _source_identity_texts(candidate)
    )
    products.update(source_products)
    versions.update(source_versions)
    # A filename often carries only the product generation (``产品甲6配置``),
    # while structured metadata carries the full release (``6.0.1``).  Treat
    # the shorter numeric prefix as a display alias of the detailed identity;
    # retaining both would manufacture a second mutually-exclusive scope for
    # one document and make a comparison ask about a version the source never
    # declared as a separate release.
    detailed_versions = {
        value
        for value in versions
        if "." in value
    }
    if detailed_versions:
        versions = {
            value
            for value in versions
            if not any(
                value != other and other.startswith(f"{value}.")
                for other in detailed_versions
            )
        }
    return products, versions, projects


def extract_document_constraint_identity(
    candidate: dict[str, Any],
) -> DocumentConstraintIdentity:
    """Return source-anchored document identity without query-dependent guesses."""

    products, versions, projects = _declared_document_identity(candidate)
    canonical_products = {
        _canonical_product(product)
        for product in products
        if _canonical_product(product)
    }
    return DocumentConstraintIdentity(
        products=tuple(sorted(products, key=str.casefold)),
        canonical_products=tuple(sorted(canonical_products, key=str.casefold)),
        versions=tuple(sorted(versions)),
        projects=tuple(sorted(projects, key=str.casefold)),
    )


def _declared_scope_values(value: Any) -> set[str]:
    """Split a structured scope declaration without treating prose as scope.

    This helper is called only after a recognized metadata key or an explicit
    applicability header has established that ``value`` is a scope field.
    Separators describe one inclusive declaration partition (for example,
    "适用版本：2024、2025"), not separate document sections.
    """

    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        entries: set[str] = set()
        for item in value:
            entries.update(_declared_scope_values(item))
        return entries
    if isinstance(value, dict):
        return set()
    return {
        item.strip()
        for item in re.split(r"[、,，;/；|]", str(value))
        if item.strip()
    }


def extract_document_applicability_declaration(
    candidate: Mapping[str, Any],
) -> DocumentApplicabilityDeclaration:
    """Extract only explicit scope declarations from one source chunk.

    The result must not be substituted for
    :func:`extract_document_constraint_identity`: ordinary title/body product
    mentions remain valid retrieval evidence.  This API is deliberately for
    the much stricter question of whether several source-declared sections
    are mutually exclusive but cannot be bound to a closed answer route.
    """

    products: set[str] = set()
    versions: set[str] = set()
    projects: set[str] = set()
    origins: set[str] = set()

    raw_metadata = candidate.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
    # Inherited/ambiguous identities are retrieval aids, not source-authored
    # declarations for this particular chunk.
    metadata.pop("inherited_document_identity", None)
    metadata.pop("ambiguous_document_identity", None)
    for key, value in _flatten_metadata(metadata):
        normalized_key = _normalize_product(key)
        if normalized_key in _NORMALIZED_PRODUCT_METADATA_KEYS:
            declared = _declared_scope_values(value)
            if declared:
                products.update(declared)
                origins.add("product:metadata")
        elif normalized_key in _NORMALIZED_VERSION_METADATA_KEYS:
            declared = _versions_from_value(value)
            if declared:
                versions.update(declared)
                origins.add("version:metadata")
        elif normalized_key in _NORMALIZED_PROJECT_METADATA_KEYS:
            declared = _declared_scope_values(value)
            if declared:
                projects.update(declared)
                origins.add("project:metadata")

    content = str(candidate.get("content") or "")
    for match in _DECLARED_APPLICABILITY_PRODUCT_FIELD_RE.finditer(content):
        declared = _declared_scope_values(match.group("value"))
        if declared:
            products.update(declared)
            origins.add("product:explicit_scope_header")
    for match in _DECLARED_APPLICABILITY_VERSION_FIELD_RE.finditer(content):
        declared = _versions_from_value(match.group("value"))
        if declared:
            versions.update(declared)
            origins.add("version:explicit_scope_header")
    for match in _DECLARED_APPLICABILITY_PROJECT_FIELD_RE.finditer(content):
        declared = _declared_scope_values(match.group("value"))
        if declared:
            projects.update(declared)
            origins.add("project:explicit_scope_header")

    canonical_products = {
        _canonical_product(product)
        for product in products
        if _canonical_product(product)
    }
    return DocumentApplicabilityDeclaration(
        identity=DocumentConstraintIdentity(
            products=tuple(sorted(products, key=str.casefold)),
            canonical_products=tuple(sorted(canonical_products, key=str.casefold)),
            versions=tuple(sorted(versions)),
            projects=tuple(sorted(projects, key=str.casefold)),
        ),
        origins=tuple(sorted(origins)),
    )


def canonical_product_name(value: str) -> str:
    """Public canonicalization boundary shared by retrieval-scope consumers."""

    return _canonical_product(value)


def candidate_section_key(candidate: Mapping[str, Any]) -> str | None:
    """Return the ingestion-owned structural section identity for one chunk.

    ``section_key`` is deliberately read only from the candidate envelope or
    chunk metadata.  It is an opaque lineage identifier produced by ingestion,
    not a heading guessed from body text.  Keeping this boundary centralized
    lets applicability inheritance and later clarification filtering agree on
    the exact same section identity.
    """

    raw_metadata = candidate.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    value = candidate.get("section_key") or metadata.get("section_key")
    section_key = str(value or "").strip()
    if not section_key or len(section_key) > 500:
        return None
    return section_key


def inherit_document_constraint_metadata(
    candidates: Sequence[Mapping[str, Any]],
    *,
    identity_sources: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Copy only unambiguous document identity to otherwise unscoped chunks.

    Markdown and Word ingestion commonly put ``所属产品`` / ``产品版本`` in the
    first chunk only.  Constraint checks are nevertheless performed per chunk;
    without this propagation, the answer chunk from the same document becomes
    ``unknown`` and is excluded.

    A document can also contain several independently applicable slices (for
    example a 6.0 section and a 7.0 section).  Propagating the union of those
    values to every chunk destroys the slice boundary: an explicit 6.0 query
    can reject the real 6.0 chunk because that chunk now also appears to be
    7.0, while an unscoped query can join facts across both versions.  Local
    chunk identity is therefore authoritative.  A missing dimension is
    inherited only when that dimension has exactly one value across the
    document; otherwise it is marked ambiguous and deliberately left unset.

    Grouping includes both knowledge-base and document ids so unrelated
    documents cannot contaminate one another.  Existing derived inheritance
    markers are stripped before recomputation, making this function idempotent.

    ``identity_sources`` is a deliberately narrow, content-free provenance
    boundary for bounded expansion.  A small-document or structural-neighbor
    adapter may return a headless sibling while the already-admitted first-pass
    seed contains the document header.  In that case the source header may be
    used *only* to recompute document identity for a candidate in the same
    ``(kb_id, doc_id)`` group; it never supplies task lineage, relevance, an
    answer claim, or an applicability decision by itself.  Callers must pass
    only candidates that have already passed their request's authorization,
    scope, and relevance gates.  The returned sequence always contains only
    ``candidates`` in its original order, never the source rows.
    """

    target_rows = [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, Mapping)
    ]
    source_rows = [
        dict(candidate)
        for candidate in identity_sources
        if isinstance(candidate, Mapping)
    ]
    if not source_rows:
        return _inherit_document_constraint_metadata_rows(target_rows)
    enriched = _inherit_document_constraint_metadata_rows([
        *source_rows,
        *target_rows,
    ])
    return enriched[len(source_rows):]


def _inherit_document_constraint_metadata_rows(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Implementation for document-identity inheritance over one trusted pool.

    Keeping this private helper separate means the public wrapper can safely
    include already-admitted source rows during expansion without returning
    those rows or changing the long-standing no-source behaviour.
    """

    def local_identity(
        candidate: dict[str, Any],
    ) -> tuple[set[str], set[str], set[str]]:
        item = dict(candidate)
        raw_metadata = item.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
            metadata.pop("inherited_document_identity", None)
            metadata.pop("ambiguous_document_identity", None)
            item["metadata"] = metadata
        return _declared_document_identity(item)

    def strong_local_products(candidate: dict[str, Any]) -> set[str]:
        """Return explicit applicability fields, excluding source-title mentions."""

        products: set[str] = set()
        raw_metadata = candidate.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        metadata.pop("inherited_document_identity", None)
        metadata.pop("ambiguous_document_identity", None)
        for key, value in _flatten_metadata(metadata):
            if (
                _normalize_product(key) in _NORMALIZED_PRODUCT_METADATA_KEYS
                and value is not None
                and str(value).strip()
            ):
                products.add(str(value).strip())
        for match in _DOCUMENT_PRODUCT_FIELD_RE.finditer(
            str(candidate.get("content") or "")
        ):
            value = match.group(1).strip()
            if value:
                products.add(value)
        return products

    identities: dict[
        tuple[str, str],
        tuple[set[str], set[str], set[str]],
    ] = {}
    section_identities: dict[
        tuple[str, str, str],
        tuple[set[str], set[str], set[str]],
    ] = {}
    strong_document_products: dict[tuple[str, str], set[str]] = {}
    for candidate in candidates:
        doc_id = candidate.get("doc_id")
        if doc_id is None:
            continue
        key = (str(candidate.get("kb_id") or ""), str(doc_id))
        group_products, group_versions, group_projects = identities.setdefault(
            key,
            (set(), set(), set()),
        )
        products, versions, projects = local_identity(candidate)
        group_products.update(products)
        group_versions.update(versions)
        group_projects.update(projects)
        strong_document_products.setdefault(key, set()).update(
            strong_local_products(candidate)
        )
        section_key = candidate_section_key(candidate)
        if section_key is not None:
            (
                section_products,
                section_versions,
                section_projects,
            ) = section_identities.setdefault(
                (*key, section_key),
                (set(), set(), set()),
            )
            section_products.update(products)
            section_versions.update(versions)
            section_projects.update(projects)

    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        doc_id = item.get("doc_id")
        identity = (
            identities.get((str(item.get("kb_id") or ""), str(doc_id)))
            if doc_id is not None
            else None
        )
        section_key = candidate_section_key(item)
        section_identity = (
            section_identities.get(
                (
                    str(item.get("kb_id") or ""),
                    str(doc_id),
                    section_key,
                )
            )
            if doc_id is not None and section_key is not None
            else None
        )
        raw_metadata = item.get("metadata")
        metadata = (
            dict(raw_metadata)
            if isinstance(raw_metadata, dict)
            else ({"source_metadata": raw_metadata} if raw_metadata else {})
        )
        metadata.pop("inherited_document_identity", None)
        metadata.pop("ambiguous_document_identity", None)
        if identity is not None and (identity[0] or identity[1] or identity[2]):
            local_products, local_versions, local_projects = local_identity(item)
            strong_products = strong_document_products.get(
                (str(item.get("kb_id") or ""), str(doc_id)),
                set(),
            )
            strong_product_canonicals = {
                _canonical_product(value)
                for value in strong_products
                if _canonical_product(value)
            }
            authoritative_document_products = (
                strong_products
                if len(strong_product_canonicals) == 1
                else identity[0]
            )
            inherited: dict[str, Any] = {}
            ambiguous: dict[str, Any] = {}
            dimensions = (
                (
                    "product",
                    local_products,
                    section_identity[0] if section_identity is not None else set(),
                    authoritative_document_products,
                    lambda values: sorted(values, key=str.casefold),
                ),
                (
                    "version",
                    local_versions,
                    section_identity[1] if section_identity is not None else set(),
                    identity[1],
                    sorted,
                ),
                (
                    "project",
                    local_projects,
                    section_identity[2] if section_identity is not None else set(),
                    identity[2],
                    lambda values: sorted(values, key=str.casefold),
                ),
            )
            for (
                name,
                local_values,
                section_values,
                document_values,
                sorter,
            ) in dimensions:
                if local_values:
                    if (
                        name == "product"
                        and len(strong_product_canonicals) == 1
                        and not any(
                            _canonical_product(value)
                            in strong_product_canonicals
                            for value in local_values
                        )
                    ):
                        # A filename may name an integration target (for example
                        # ProductY) while an explicit ``所属产品`` header declares
                        # the document's actual applicability (ProductX).  Keep
                        # the title signal for recall, but also inherit the one
                        # authoritative document product so constraint checking
                        # does not misclassify the answer chunk.
                        inherited[name] = sorter(strong_products)
                    continue
                # A section-local declaration is more precise than a document
                # header.  This is what keeps a 2024 section and a 2025 section
                # in one physical document from inheriting each other's scope.
                if len(section_values) == 1:
                    inherited[name] = sorter(section_values)
                    continue
                if len(section_values) > 1:
                    ambiguous[name] = sorter(section_values)
                    continue
                if len(document_values) == 1:
                    inherited[name] = sorter(document_values)
                elif len(document_values) > 1:
                    ambiguous[name] = sorter(document_values)
            if inherited:
                metadata["inherited_document_identity"] = inherited
            if ambiguous:
                metadata["ambiguous_document_identity"] = ambiguous
        if metadata:
            item["metadata"] = metadata
        else:
            item.pop("metadata", None)
        enriched.append(item)
    return enriched


def _same_product(left: str, right: str) -> bool:
    normalized_left = _canonical_product(left)
    normalized_right = _canonical_product(right)
    if not normalized_left or not normalized_right:
        return False
    return normalized_left == normalized_right


def _version_regex_for_product(product: str) -> re.Pattern[str]:
    aliases = "|".join(re.escape(alias) for alias in _product_aliases(product))
    return re.compile(
        rf"(?P<product>{aliases})\s*(?:版本\s*|[vV]\s*)?"
        rf"(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}",
        re.IGNORECASE,
    )


def _collect_compatibility_versions(
    searchable: str,
    product: str,
) -> tuple[set[str], set[str], set[str]]:
    """返回正向兼容版本、明确不支持版本和语句中绑定的产品。

    正文兼容声明必须同时出现产品别名和版本。“组件支持8.6”这种
    裸版本语句不能将《产品甲7配置》升级为产品甲8.6的可用证据。
    """

    compatible: set[str] = set()
    unsupported: set[str] = set()
    referenced_products: set[str] = set()
    compatibility_text = re.sub(r"[ \t]+", "", searchable)
    aliases = "|".join(re.escape(alias) for alias in _product_aliases(product))

    action = r"(?:兼容|支持|适用(?:于)?|适配)"
    # 否定前缀必须与动作整体匹配，并且放在正向动作之前。否则正则会从
    # “不适用于”的第二个字符重新命中“适用于”，把明确否定反判成兼容。
    negative_action = (
        rf"(?:不代表|不再|并不|并非|尚未|尚不|从未|无法|不能|不可|未|不){action}"
    )
    cue = rf"(?:(?P<negative>{negative_action})|(?P<positive>{action}))"
    product_version = (
        rf"(?P<product>{aliases})\s*(?:版本\s*|[vV]\s*)?"
        rf"(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}"
    )
    patterns = (
        re.compile(rf"{cue}\s*{product_version}", re.IGNORECASE),
        # 同时覆盖“产品甲8.6不再兼容”这类产品版本在前的自然表达。
        re.compile(
            rf"{product_version}\s*[，,：:;；的]*\s*{cue}",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(compatibility_text):
            version = _normalize_version(match.group("version"))
            referenced_products.add(match.group("product"))
            if match.group("negative"):
                unsupported.add(version)
            else:
                compatible.add(version)
    return compatible, unsupported, referenced_products


def _evaluate_product_version_constraints(
    constraints: QueryConstraints,
    candidate: dict[str, Any],
) -> ConstraintEvaluation:
    """Evaluate only product/version applicability for one candidate.

    This is intentionally kept as a private compatibility implementation.
    The public evaluator below composes this result with the project boundary
    and is the only API that callers should use.  Keeping the old logic in a
    named leaf avoids two partially-overlapping product/version evaluators.
    """

    if not (
        constraints.has_product_constraint
        or constraints.has_version_constraint
    ):
        return ConstraintEvaluation(
            status="neutral",
            reason="查询没有可确定的产品或版本适用范围约束",
            scope_applicability="unscoped",
        )

    product = constraints.product or ""
    target_version = constraints.version or ""
    filename = str(candidate.get("filename") or "")
    tags = candidate.get("doc_tags", candidate.get("tags")) or []
    metadata = candidate.get("metadata") or {}
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        metadata.pop("ambiguous_document_identity", None)
    content = str(candidate.get("content") or "")
    metadata_items = _flatten_metadata(metadata)

    candidate_products: set[str] = set()
    # declared_versions 只收集“文档身份/适用范围”中的版本；正文普通提及的
    # 版本放到 mentioned_versions，不能凭一次提及覆盖明确的旧版本身份。
    declared_versions: set[str] = set()
    compatible_versions: set[str] = set()
    unsupported_versions: set[str] = set()
    mentioned_versions: set[str] = set()

    for key, value in metadata_items:
        normalized_key = _normalize_product(key)
        if normalized_key in _NORMALIZED_PRODUCT_METADATA_KEYS:
            if value is not None and str(value).strip():
                candidate_products.add(str(value).strip())
        if normalized_key in _NORMALIZED_VERSION_METADATA_KEYS:
            declared_versions.update(_versions_from_value(value))
        if normalized_key in _NORMALIZED_COMPATIBLE_VERSION_KEYS:
            compatible_versions.update(_versions_from_value(value))

    # 文件名、原始 source/heading 标签和标签字段通常是文档身份的强信号；
    # 正文只在明确字段/兼容语句中参与强判定，避免“本文比较产品甲8.6与产品甲6”
    # 把旧版本文档误判 exact。
    source_identity_texts = _source_identity_texts(candidate)
    source_products, source_versions = _source_identity_facts(
        source_identity_texts
    )
    candidate_products.update(source_products)
    declared_versions.update(source_versions)
    searchable_parts = [*source_identity_texts]
    identity_text_parts = [*source_identity_texts]
    if isinstance(tags, (list, tuple, set)):
        for tag in tags:
            tag_text = str(tag)
            searchable_parts.append(tag_text)
            identity_text_parts.append(tag_text)
            # 纯数字标签含义不明确；带小数点的 ``8.6`` 或显式 v/版本 标签可判作版本。
            if re.fullmatch(
                rf"\s*(?:(?:版本\s*|[vV]\s*){_VERSION_PATTERN}|"
                rf"\d{{1,3}}(?:\.\d{{1,3}}){{1,3}})\s*",
                tag_text,
                flags=re.IGNORECASE,
            ):
                declared_versions.update(_versions_from_value(tag_text))
    elif tags:
        searchable_parts.append(str(tags))
        identity_text_parts.append(str(tags))
    # 正文也参与校验，能覆盖“所属产品/产品版本”位于片段内的知识文档。
    searchable_parts.append(content)
    searchable = "\n".join(searchable_parts)
    identity_text = "\n".join(identity_text_parts)

    # 只把标题/文件名中相邻的产品版本视为文档身份版本。
    product_version_re = _version_regex_for_product(product)
    for alias in _product_aliases(product):
        if re.search(re.escape(alias), identity_text, flags=re.IGNORECASE):
            candidate_products.add(alias)
    for match in product_version_re.finditer(identity_text):
        version = _normalize_version(match.group("version"))
        declared_versions.add(version)
        candidate_products.add(match.group("product"))

    # metadata 未结构化时，仍识别常见的“所属产品：X / 产品版本：Y”写法。
    for match in _DOCUMENT_PRODUCT_FIELD_RE.finditer(searchable):
        value = match.group(1).strip()
        if value:
            candidate_products.add(value)
    declared_versions.update(_declared_versions_from_document_fields(searchable))
    # 兼容“版本：产品甲6”，但裸的“运行时版本：8.6”/“数据库版本：8.6”
    # 不能被当成目标产品版本。
    for match in _GENERIC_DOCUMENT_VERSION_FIELD_RE.finditer(searchable):
        versions_after_label, end = _declared_versions_after_field_label(
            searchable,
            match.end(),
        )
        declaration_text = searchable[match.end():end]
        if any(
            re.search(re.escape(alias), declaration_text, flags=re.IGNORECASE)
            for alias in _product_aliases(product)
        ):
            declared_versions.update(versions_after_label)

    # 普通正文提及只做观测，不参与 exact；兼容/不支持语句有明确语义，单独收集。
    for match in product_version_re.finditer(content):
        mentioned_versions.add(_normalize_version(match.group("version")))
    body_compatible, body_unsupported, compatibility_products = (
        _collect_compatibility_versions(searchable, product)
    )
    compatible_versions.update(body_compatible)
    unsupported_versions.update(body_unsupported)
    candidate_products.update(compatibility_products)

    sorted_products = tuple(sorted(candidate_products, key=str.casefold))
    sorted_versions = tuple(sorted(declared_versions | mentioned_versions))
    if not constraints.has_product_constraint:
        # A product-less explicit version (for example ``版本8.6`` or
        # ``2025版``) may still be enforced against source-declared versions.
        # Do not use incidental body mentions as identity evidence, and do not
        # infer a product from an unrelated candidate.
        if target_version in compatible_versions:
            return ConstraintEvaluation(
                status="compatible",
                reason=f"候选明确声明兼容或适用于版本“{target_version}”",
                candidate_products=sorted_products,
                candidate_versions=sorted_versions,
            )
        if target_version in declared_versions and declared_versions == {
            target_version
        }:
            return ConstraintEvaluation(
                status="exact",
                reason=f"候选明确包含目标版本“{target_version}”",
                candidate_products=sorted_products,
                candidate_versions=sorted_versions,
            )
        if target_version in declared_versions:
            return ConstraintEvaluation(
                status="mismatch",
                reason=(
                    f"候选同时声明目标版本“{target_version}”与其他版本"
                    f"“{'、'.join(sorted(declared_versions - {target_version}))}”，"
                    "无法作为精确版本依据"
                ),
                candidate_products=sorted_products,
                candidate_versions=sorted_versions,
            )
        if declared_versions:
            return ConstraintEvaluation(
                status="mismatch",
                reason=(
                    f"查询要求版本“{target_version}”，候选明确版本为"
                    f"“{'、'.join(sorted(declared_versions))}”，版本冲突"
                ),
                candidate_products=sorted_products,
                candidate_versions=sorted_versions,
            )
        return ConstraintEvaluation(
            status="unknown",
            reason=f"候选未标注明确版本，无法确认是否适用于“{target_version}”",
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    product_matches = any(_same_product(product, item) for item in candidate_products)
    if candidate_products and not product_matches:
        return ConstraintEvaluation(
            status="mismatch",
            reason=(
                f"查询产品为“{product}”，候选明确标注产品为"
                f"“{'、'.join(sorted_products)}”，产品冲突"
            ),
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    if not target_version:
        # 产品-only 查询不需要证明版本身份。知识库中的历史文档经常使用通用
        # 文件名，也没有结构化 metadata，但正文会明确写出产品名；在这种
        # 场景下，允许正文中的已知别名确认产品归属。版本查询仍只接受标题、
        # 标签、metadata、结构化字段或明确兼容声明，避免正文偶然提及某个
        # 版本就被误判为目标版本资料。
        if not product_matches and any(
            re.search(re.escape(alias), content, flags=re.IGNORECASE)
            for alias in _product_aliases(product)
        ):
            candidate_products.add(product)
            sorted_products = tuple(sorted(candidate_products, key=str.casefold))
            product_matches = True
        if product_matches:
            version_scope = (
                f"，文档声明版本为“{'、'.join(sorted(declared_versions))}”"
                if declared_versions
                else ""
            )
            return ConstraintEvaluation(
                status="neutral",
                reason=(
                    f"候选明确标注产品“{product}”{version_scope}，查询未指定版本"
                ),
                candidate_products=sorted_products,
                candidate_versions=sorted_versions,
            )
        return ConstraintEvaluation(
            status="unknown",
            reason=f"候选未标注可确认的产品“{product}”",
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    # 版本数字必须绑定到目标产品身份。只有 metadata.version=8.6
    # 或正文出现某个组件的8.6，不能证明这是产品甲8.6文档。
    if not product_matches:
        return ConstraintEvaluation(
            status="unknown",
            reason=f"候选未标注可确认的产品“{product}”",
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    if target_version in unsupported_versions:
        return ConstraintEvaluation(
            status="mismatch",
            reason=f"候选明确声明不支持或不适用于版本“{target_version}”",
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    if target_version in declared_versions and declared_versions == {target_version}:
        return ConstraintEvaluation(
            status="exact",
            reason=f"候选明确包含目标产品“{product}”的版本“{target_version}”",
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    if target_version in declared_versions:
        return ConstraintEvaluation(
            status="mismatch",
            reason=(
                f"候选同时声明目标版本“{target_version}”与其他产品版本"
                f"“{'、'.join(sorted(declared_versions - {target_version}))}”，无法作为精确版本依据"
            ),
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    if target_version in compatible_versions:
        return ConstraintEvaluation(
            status="compatible",
            reason=f"候选明确声明兼容或适用于版本“{target_version}”",
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    if declared_versions:
        return ConstraintEvaluation(
            status="mismatch",
            reason=(
                f"查询要求版本“{target_version}”，候选明确版本为"
                f"“{'、'.join(sorted(declared_versions))}”，版本冲突"
            ),
            candidate_products=sorted_products,
            candidate_versions=sorted_versions,
        )

    return ConstraintEvaluation(
        status="unknown",
        reason=f"候选未标注明确版本，无法确认是否适用于“{product} {target_version}”",
        candidate_products=sorted_products,
        candidate_versions=sorted_versions,
    )


# Only source-owned, explicit labels may make a project-less chunk usable for
# a project-scoped question.  A missing project label is *not* shorthand for
# global applicability: it is unknown and must fail closed.
_GLOBAL_SCOPE_METADATA_KEYS = frozenset({
    "scope_applicability",
    "applicability_scope",
    "scope",
    "project_scope",
    "project_applicability",
    "适用范围",
    "适用项目",
    "项目范围",
})
_NORMALIZED_GLOBAL_SCOPE_METADATA_KEYS = frozenset(
    _normalize_product(value) for value in _GLOBAL_SCOPE_METADATA_KEYS
)
_GLOBAL_SCOPE_VALUES = frozenset({
    "global",
    "globalcompatible",
    "allprojects",
    "allproject",
    "全局",
    "通用",
    "所有项目",
    "全项目",
    "公司统一",
})
_NORMALIZED_GLOBAL_SCOPE_VALUES = frozenset(
    _normalize_product(value) for value in _GLOBAL_SCOPE_VALUES
)
_GLOBAL_SCOPE_CONTENT_RE = re.compile(
    r"(?:适用范围|适用项目|项目范围|所属项目)\s*[：:]\s*"
    r"(?:全局|通用|所有项目|全项目|公司统一)\b",
    re.IGNORECASE,
)


def _candidate_declares_global_scope(candidate: Mapping[str, Any]) -> bool:
    """Whether a candidate explicitly declares a non-project-specific clause.

    This is intentionally stricter than normal relevance extraction.  It
    accepts a bounded metadata vocabulary or a labelled source sentence, but
    never infers global scope merely because the candidate has no project
    identity.
    """

    raw_metadata = candidate.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    for key, value in _flatten_metadata(metadata):
        if _normalize_product(key) not in _NORMALIZED_GLOBAL_SCOPE_METADATA_KEYS:
            continue
        if _normalize_product(value) in _NORMALIZED_GLOBAL_SCOPE_VALUES:
            return True
    return bool(_GLOBAL_SCOPE_CONTENT_RE.search(str(candidate.get("content") or "")))


def _identity_fingerprint(identity: DocumentConstraintIdentity) -> str:
    """Hash source identity facts for safe scope-rejection diagnostics."""

    payload = {
        "products": sorted(identity.canonical_products),
        "versions": sorted(identity.versions),
        "projects": sorted(
            {
                _canonical_project(value)
                for value in identity.projects
                if _canonical_project(value)
            }
        ),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _product_version_mismatch_dimensions(
    scope: ApplicabilityScope,
    evaluation: ConstraintEvaluation,
    identity: DocumentConstraintIdentity,
) -> tuple[ScopeDimension, ...]:
    """Produce deterministic, content-free mismatch dimensions.

    The historical evaluator returns human-readable reasons.  Re-parsing those
    reasons would turn UI wording into execution semantics, so this helper
    derives dimension codes only from the canonical scope and source identity.
    """

    dimensions: list[ScopeDimension] = []
    expected_product = _canonical_product(scope.product or "")
    candidate_products = set(identity.canonical_products)
    if (
        scope.has_product_constraint
        and candidate_products
        and expected_product not in candidate_products
    ):
        dimensions.append("product")

    candidate_versions = set(identity.versions) | set(evaluation.candidate_versions)
    if (
        scope.has_version_constraint
        and scope.version
        and candidate_versions
        and scope.version not in candidate_versions
    ):
        dimensions.append("version")

    # Explicit incompatibility such as “不支持产品甲8.6” can be a mismatch even
    # when the source identity lists the requested version.  It is still a
    # version dimension; do not expose the source sentence in diagnostics.
    if evaluation.status == "mismatch" and not dimensions:
        if scope.has_version_constraint:
            dimensions.append("version")
        elif scope.has_product_constraint:
            dimensions.append("product")
    return tuple(dict.fromkeys(dimensions))


def _project_scope_outcome(
    scope: ApplicabilityScope,
    candidate: Mapping[str, Any],
    identity: DocumentConstraintIdentity,
) -> Literal["not_requested", "exact", "global_compatible", "unknown", "mismatch"]:
    if not scope.has_project_constraint:
        return "not_requested"
    if _candidate_declares_global_scope(candidate):
        return "global_compatible"
    expected_project = _canonical_project(scope.project or "")
    candidate_projects = {
        _canonical_project(value)
        for value in identity.projects
        if _canonical_project(value)
    }
    if not candidate_projects:
        return "unknown"
    if expected_project in candidate_projects:
        return "exact"
    return "mismatch"


def evaluate_candidate_constraints(
    constraints: QueryConstraints,
    candidate: dict[str, Any],
) -> ConstraintEvaluation:
    """Evaluate the canonical product/version/project applicability contract.

    ``ApplicabilityScope`` has a conjunctive meaning: a candidate must satisfy
    every explicit dimension of the scope, except an explicitly-declared
    global clause may satisfy the project dimension.  Unknown project identity
    is deliberately not promoted to global applicability.
    """

    if not isinstance(constraints, ApplicabilityScope):
        raise TypeError("constraints must be an ApplicabilityScope")
    item = dict(candidate) if isinstance(candidate, Mapping) else {}
    identity = extract_document_constraint_identity(item)
    product_version = _evaluate_product_version_constraints(constraints, item)
    mismatch_dimensions = list(
        _product_version_mismatch_dimensions(
            constraints,
            product_version,
            identity,
        )
    )
    project_outcome = _project_scope_outcome(constraints, item, identity)

    if project_outcome == "mismatch":
        mismatch_dimensions.append("project")
    elif project_outcome == "unknown":
        # ``mismatch_dimensions`` doubles as the fail-closed dimension list
        # for rejection diagnostics.  The status remains ``unknown`` so
        # callers can distinguish absent identity from a contradictory one.
        mismatch_dimensions.append("project")

    mismatch_dimensions = list(dict.fromkeys(mismatch_dimensions))
    candidate_projects = tuple(sorted(identity.projects, key=str.casefold))

    # A product/version contradiction remains a contradiction even when the
    # candidate declares itself global.  Global only relaxes the project
    # dimension, never a product or version boundary.
    if product_version.status == "mismatch" or project_outcome == "mismatch":
        reason = product_version.reason
        if project_outcome == "mismatch":
            expected = scope_project = constraints.project or ""
            actual = "、".join(candidate_projects) or "未标注"
            project_reason = f"查询项目为“{expected}”，候选明确项目为“{actual}”，项目冲突"
            reason = (
                f"{reason}；{project_reason}"
                if product_version.status == "mismatch"
                else project_reason
            )
        return ConstraintEvaluation(
            status="mismatch",
            reason=reason,
            candidate_products=product_version.candidate_products,
            candidate_versions=product_version.candidate_versions,
            candidate_projects=candidate_projects,
            mismatch_dimensions=tuple(mismatch_dimensions),
            scope_applicability="unknown",
        )

    # With a project boundary, absence of an identity is a closed gate even
    # if the product/version part is otherwise exact.  Similarly, an unknown
    # product/version must not become admissible merely because the project
    # label matches.
    if product_version.status == "unknown" or project_outcome == "unknown":
        reason = product_version.reason
        if project_outcome == "unknown":
            project_reason = (
                f"查询限定项目“{constraints.project}”，候选未标注项目身份，"
                "无法确认适用范围"
            )
            reason = (
                f"{reason}；{project_reason}"
                if product_version.status == "unknown"
                else project_reason
            )
        return ConstraintEvaluation(
            status="unknown",
            reason=reason,
            candidate_products=product_version.candidate_products,
            candidate_versions=product_version.candidate_versions,
            candidate_projects=candidate_projects,
            mismatch_dimensions=tuple(mismatch_dimensions),
            scope_applicability="unknown",
        )

    if project_outcome == "global_compatible":
        return ConstraintEvaluation(
            status="compatible",
            reason=(
                f"{product_version.reason}；候选明确声明为全局条款，"
                f"可兼容项目“{constraints.project}”"
            ),
            candidate_products=product_version.candidate_products,
            candidate_versions=product_version.candidate_versions,
            candidate_projects=candidate_projects,
            mismatch_dimensions=(),
            scope_applicability="global_compatible",
        )

    if not constraints.has_scope_constraint:
        applicability: Literal["exact", "global_compatible", "unscoped", "unknown"] = "unscoped"
    else:
        applicability = "exact"
    return replace(
        product_version,
        candidate_projects=candidate_projects,
        mismatch_dimensions=tuple(mismatch_dimensions),
        scope_applicability=applicability,
    )


def _scope_rejection_reason_code(evaluation: ConstraintEvaluation) -> str:
    dimensions = tuple(dict.fromkeys(evaluation.mismatch_dimensions))
    suffix = "_".join(dimensions) if dimensions else "identity"
    if evaluation.status == "unknown":
        return f"scope_unknown_{suffix}"
    return f"scope_mismatch_{suffix}"


def _scope_rejection_for(
    candidate: Mapping[str, Any],
    *,
    scope: ApplicabilityScope,
    evaluation: ConstraintEvaluation,
    actual_identity_fingerprint: str,
) -> ScopeCandidateRejection:
    """Create a content-free rejection record for one failed scope."""

    dimensions = evaluation.mismatch_dimensions
    if not dimensions:
        # This branch is intentionally defensive.  A constrained candidate
        # must never be rejected without a dimension that can be inspected in
        # the trace/ledger, and scope admission cannot derive a body-dependent
        # fallback at this point.
        if scope.has_project_constraint:
            dimensions = ("project",)
        elif scope.has_version_constraint:
            dimensions = ("version",)
        else:
            dimensions = ("product",)
    return ScopeCandidateRejection(
        kb_id=str(candidate.get("kb_id") or "").strip(),
        doc_id=str(candidate.get("doc_id") or "").strip(),
        chunk_id=str(candidate.get("chunk_id") or candidate.get("id") or "").strip(),
        expected_scope_fingerprint=scope.fingerprint,
        actual_identity_fingerprint=actual_identity_fingerprint,
        mismatch_dimensions=dimensions,
        reason_code=_scope_rejection_reason_code(evaluation),
    )


def _admission_priority(
    evaluation: ConstraintEvaluation,
) -> tuple[int, int]:
    """Rank successful scope matches without using retrieval score/content."""

    status_rank = {"exact": 3, "compatible": 2, "neutral": 1}.get(
        evaluation.status,
        0,
    )
    applicability_rank = {
        "exact": 2,
        "global_compatible": 1,
        "unscoped": 0,
        "unknown": -1,
    }.get(evaluation.scope_applicability, -1)
    return status_rank, applicability_rank


def admit_candidates_for_scopes(
    candidates: Sequence[Mapping[str, Any]],
    scopes: Sequence[ApplicabilityScope],
    *,
    identity_sources: Sequence[Mapping[str, Any]] = (),
) -> ScopeAdmission:
    """Admit candidates against one or more canonical applicability scopes.

    The function is deliberately a pure boundary: it neither creates a
    clarification nor selects answer evidence.  A caller running a comparison
    may use a scope union for root recall, but answer tasks must later call it
    with their own single scope before evidence assembly.  A candidate is
    rejected only when it fails every constrained scope; then one safe record
    is emitted for each failed scope so a ledger can attribute the rejection
    without retaining rejected body text.
    """

    source_scopes = tuple(scopes)
    normalized_scopes = tuple(
        scope for scope in source_scopes if isinstance(scope, ApplicabilityScope)
    )
    if len(normalized_scopes) != len(source_scopes):
        raise TypeError("scopes must contain ApplicabilityScope values")
    constrained_scopes = tuple(
        scope for scope in normalized_scopes if scope.has_scope_constraint
    )
    enriched = inherit_document_constraint_metadata(
        [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)],
        identity_sources=identity_sources,
    )
    if not constrained_scopes:
        return ScopeAdmission(candidates=tuple(enriched))

    accepted: list[dict[str, Any]] = []
    rejections: list[ScopeCandidateRejection] = []
    seen_rejections: set[tuple[str, str, str, str, str, tuple[ScopeDimension, ...], str]] = set()
    for candidate in enriched:
        evaluations = tuple(
            (scope, evaluate_candidate_constraints(scope, candidate))
            for scope in constrained_scopes
        )
        matched = tuple(
            (scope, evaluation)
            for scope, evaluation in evaluations
            if evaluation.status in {"exact", "compatible", "neutral"}
        )
        if matched:
            scope, evaluation = max(
                matched,
                key=lambda item: (_admission_priority(item[1]), item[0].fingerprint),
            )
            item = dict(candidate)
            raw_metadata = item.get("metadata")
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            metadata["scope_applicability"] = evaluation.scope_applicability
            metadata["scope_fingerprint"] = scope.fingerprint
            metadata["scope_match_status"] = evaluation.status
            item["metadata"] = metadata
            item["constraint_status"] = evaluation.status
            accepted.append(item)
            continue

        identity_fingerprint = _identity_fingerprint(
            extract_document_constraint_identity(candidate)
        )
        for scope, evaluation in evaluations:
            rejection = _scope_rejection_for(
                candidate,
                scope=scope,
                evaluation=evaluation,
                actual_identity_fingerprint=identity_fingerprint,
            )
            key = (
                rejection.kb_id,
                rejection.doc_id,
                rejection.chunk_id,
                rejection.expected_scope_fingerprint,
                rejection.actual_identity_fingerprint,
                rejection.mismatch_dimensions,
                rejection.reason_code,
            )
            if key not in seen_rejections:
                seen_rejections.add(key)
                rejections.append(rejection)
    return ScopeAdmission(
        candidates=tuple(accepted),
        rejections=tuple(rejections),
    )
