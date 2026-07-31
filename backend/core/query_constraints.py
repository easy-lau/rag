"""查询中的产品/版本硬约束提取与候选证据校验。

这里的规则只处理可以从原文中直接解释的显式约束，不猜测产品版本，也不让
LLM 的高语义相关度覆盖已经确定的版本冲突。规则有意保持保守：识别不到时
返回 ``unknown``，由上层将候选作为相近资料而不是直接证据。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence


ConstraintStatus = Literal["exact", "compatible", "unknown", "mismatch", "neutral"]


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
    rf"(?P<product>[A-Za-z][A-Za-z0-9_.\-]{{1,30}}|[\u3400-\u9fff]{{2,16}})"
    rf"\s*版本\s*(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}",
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

# 企业产品别名集中维护在这里，供查询硬约束和意图路由共同使用。新增产品时
# 只扩展一个别名组，不在路由层继续堆叠产品关键词。英文别名使用 ASCII 边界
# 匹配，避免把 ``wecom`` 误命中为更长单词的一部分。
_PRODUCT_ALIAS_GROUPS = (
    frozenset({
        "云枢",
        "cloudpivot",
        "cloudpivotplatform",
        "cloudpivot platform",
        "cloudpivot平台",
    }),
    frozenset({"钉钉", "dingtalk"}),
    frozenset({"企业微信", "企微", "wecom", "workwechat", "wechatwork"}),
    frozenset({"泛微OA", "泛微", "weaveroa", "weaver oa"}),
)
_VERSION_UNIT_WORDS = re.compile(
    r"^\s*(?:个|台|套|节点|实例|用户|条|项|次|分钟|小时|天|人|页|条记录)"
)
_DOCUMENT_PRODUCT_FIELD_RE = re.compile(
    r"(?:所属产品|产品名称|产品(?!版本))\s*[：:]\s*([^\n\r>,，;；]+)",
    re.IGNORECASE,
)
_DOCUMENT_VERSION_FIELD_RE = re.compile(
    r"产品版本\s*[：:]\s*([^\n\r>,，;；]+)",
    re.IGNORECASE,
)
_DOCUMENT_PROJECT_FIELD_RE = re.compile(
    r"(?:所属项目|项目名称|项目(?!版本))\s*[：:]\s*([^\n\r>,，;；]+)",
    re.IGNORECASE,
)


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


def _known_enterprise_product_match(
    query: str,
) -> tuple[str, re.Match[str]] | None:
    text = str(query or "")
    aliases = sorted(
        (alias for group in _PRODUCT_ALIAS_GROUPS for alias in group),
        key=len,
        reverse=True,
    )
    matches: list[tuple[int, int, str, re.Match[str]]] = []
    for alias in aliases:
        match = _alias_search_pattern(alias).search(text)
        if match is not None:
            matches.append((match.start(), -len(match.group(0)), alias, match))
    if not matches:
        return None
    _, _, alias, match = min(matches, key=lambda item: (item[0], item[1]))
    return alias, match


def match_known_enterprise_product(query: str) -> str | None:
    """Return the explicit registered enterprise-product text in a query."""

    matched = _known_enterprise_product_match(query)
    return matched[1].group(0) if matched is not None else None


@dataclass(frozen=True)
class QueryConstraints:
    """从查询原文中提取出的可解释硬约束。"""

    product: str | None = None
    version: str | None = None
    explicit_version: bool = False
    matched_text: str | None = None
    extraction_reason: str = "未发现显式产品版本约束"

    @property
    def has_hard_constraint(self) -> bool:
        return bool(self.product and self.version and self.explicit_version)

    @property
    def has_product_constraint(self) -> bool:
        """查询至少明确指定了一个产品名。

        版本硬约束仍由 ``has_hard_constraint`` 表示；产品-only 查询也需要
        拦截其它产品的同主题资料，否则“云枢默认密码”可能被其它平台文档污染。
        """

        return bool(self.product)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstraintEvaluation:
    status: ConstraintStatus
    reason: str
    candidate_products: tuple[str, ...] = ()
    candidate_versions: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentConstraintIdentity:
    """Explicit, document-level applicability facts anchored in source text."""

    products: tuple[str, ...] = ()
    canonical_products: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    for aliases in _PRODUCT_ALIAS_GROUPS:
        normalized_aliases = {_normalize_product(alias) for alias in aliases}
        # 知识文档常把产品代际写进“所属产品”，例如“云枢8”或
        # “CloudPivot 8”。这里的尾部数字是产品版本，不是一个名为
        # “云枢8”的新产品；仅对注册过的产品别名和纯版本尾缀做该归一化，
        # 不允许任意 substring 命中。
        for alias in sorted(normalized_aliases, key=len, reverse=True):
            if not normalized.startswith(alias):
                continue
            suffix = normalized[len(alias):]
            if suffix and re.fullmatch(r"\d{1,12}(?:全系)?", suffix):
                return min(normalized_aliases, key=len)
        if normalized in normalized_aliases:
            return min(normalized_aliases, key=len)
    return normalized


def _product_aliases(value: str) -> tuple[str, ...]:
    normalized = _canonical_product(value)
    for aliases in _PRODUCT_ALIAS_GROUPS:
        normalized_aliases = {_canonical_product(alias) for alias in aliases}
        if normalized in normalized_aliases:
            return tuple(sorted(aliases, key=len, reverse=True))
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


def _valid_query_match(text: str, match: re.Match[str]) -> bool:
    """避免把“云枢 8 个节点”这类数量误识别为版本。"""

    version = match.groupdict().get("version") or ""
    if "." not in version:
        tail = text[match.end():]
        if _VERSION_UNIT_WORDS.match(tail):
            return False
    return True


def extract_query_constraints(query: str) -> QueryConstraints:
    """提取相邻的产品名和显式版本号。

    优先识别“我是/使用/针对 + 产品 + 版本”这类强提示，再识别
    ``云枢8.6``、``CloudPivot v8.6`` 等紧邻写法。孤立数字不会被当作版本。
    """

    text = str(query or "").strip()
    if not text:
        return QueryConstraints()

    match = None
    # 先处理系统已知别名，避免通用中文分组把“登录用户名枚举云枢8.6”整段
    # 当成产品名。别名词典是确定性边界，后续可由产品配置扩展。
    known_aliases = sorted(
        (alias for group in _PRODUCT_ALIAS_GROUPS for alias in group),
        key=len,
        reverse=True,
    )
    known_pattern = re.compile(
        rf"(?P<product>{'|'.join(re.escape(alias) for alias in known_aliases)})"
        rf"\s*(?:版本\s*|[vV]\s*)?(?P<version>{_VERSION_PATTERN}){_VERSION_BOUNDARY}",
        re.IGNORECASE,
    )
    # 每个正则都可能先命中一个数量表达式；继续寻找下一个合法版本片段。
    for pattern in (_QUERY_CUE_RE, known_pattern, _QUERY_VERSION_LABEL_RE, _QUERY_ADJACENT_RE):
        for candidate in pattern.finditer(text):
            if _valid_query_match(text, candidate):
                match = candidate
                break
        if match is not None:
            break
    # 产品-only 约束只对明确的产品别名启用，避免把“系统 6 个节点”之类普通词
    # 当成产品。版本查询仍优先走上面的相邻规则。
    if match is None:
        known_product = _known_enterprise_product_match(text)
        if known_product is not None:
            _, product_match = known_product
            matched_text = product_match.group(0)
            return QueryConstraints(
                product=matched_text,
                matched_text=matched_text,
                extraction_reason=f"由查询片段“{matched_text}”识别出产品约束，未指定显式版本",
            )
        return QueryConstraints()

    product = _clean_product(match.group("product"))
    version = _normalize_version(match.group("version"))
    if not product or not version:
        return QueryConstraints()

    matched_text = match.group(0).strip()
    return QueryConstraints(
        product=product,
        version=version,
        explicit_version=True,
        matched_text=matched_text,
        extraction_reason=(
            f"由查询片段“{matched_text}”识别出产品“{product}”和显式版本“{version}”"
        ),
    )


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


def _declared_document_identity(
    candidate: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    """Extract only explicit document identity fields from one chunk."""

    products: set[str] = set()
    versions: set[str] = set()
    projects: set[str] = set()
    for key, value in _flatten_metadata(candidate.get("metadata") or {}):
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
    for match in _DOCUMENT_VERSION_FIELD_RE.finditer(content):
        versions.update(_versions_from_value(match.group(1)))
    for match in _DOCUMENT_PROJECT_FIELD_RE.finditer(content):
        value = match.group(1).strip()
        if value:
            projects.add(value)
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


def canonical_product_name(value: str) -> str:
    """Public canonicalization boundary shared by retrieval-scope consumers."""

    return _canonical_product(value)


def inherit_document_constraint_metadata(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy explicit product/version identity to sibling chunks of a document.

    Markdown and Word ingestion commonly put ``所属产品`` / ``产品版本`` in the
    first chunk only.  Constraint checks are nevertheless performed per chunk;
    without this propagation, the answer chunk from the same document becomes
    ``unknown`` and is excluded.  Only explicit identity fields are inherited,
    and grouping includes both knowledge-base and document ids so unrelated
    documents cannot contaminate one another.
    """

    identities: dict[
        tuple[str, str],
        tuple[set[str], set[str], set[str]],
    ] = {}
    for candidate in candidates:
        doc_id = candidate.get("doc_id")
        if doc_id is None:
            continue
        key = (str(candidate.get("kb_id") or ""), str(doc_id))
        group_products, group_versions, group_projects = identities.setdefault(
            key,
            (set(), set(), set()),
        )
        products, versions, projects = _declared_document_identity(candidate)
        group_products.update(products)
        group_versions.update(versions)
        group_projects.update(projects)

    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        doc_id = item.get("doc_id")
        identity = (
            identities.get((str(item.get("kb_id") or ""), str(doc_id)))
            if doc_id is not None
            else None
        )
        if identity is not None and (identity[0] or identity[1] or identity[2]):
            raw_metadata = item.get("metadata")
            metadata = (
                dict(raw_metadata)
                if isinstance(raw_metadata, dict)
                else ({"source_metadata": raw_metadata} if raw_metadata else {})
            )
            inherited: dict[str, Any] = {}
            if identity[0]:
                inherited["product"] = sorted(identity[0], key=str.casefold)
            if identity[1]:
                inherited["version"] = sorted(identity[1])
            if identity[2]:
                inherited["project"] = sorted(identity[2], key=str.casefold)
            metadata["inherited_document_identity"] = inherited
            item["metadata"] = metadata
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
    裸版本语句不能将《云枢7配置》升级为云枢8.6的可用证据。
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
        # 同时覆盖“云枢8.6不再兼容”这类产品版本在前的自然表达。
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


def evaluate_candidate_constraints(
    constraints: QueryConstraints,
    candidate: dict[str, Any],
) -> ConstraintEvaluation:
    """使用候选的文件名、标签、metadata 和正文校验显式产品版本约束。"""

    if not constraints.has_product_constraint:
        return ConstraintEvaluation(
            status="neutral",
            reason="查询没有可确定的显式产品约束",
        )

    product = constraints.product or ""
    target_version = constraints.version or ""
    filename = str(candidate.get("filename") or "")
    tags = candidate.get("doc_tags", candidate.get("tags")) or []
    metadata = candidate.get("metadata") or {}
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

    # 文件名和标签通常是文档身份的强信号；正文只在明确字段/兼容语句中
    # 参与强判定，避免“本文比较云枢8.6与云枢6”把旧版本文档误判 exact。
    searchable_parts = [filename]
    identity_text_parts = [filename]
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
    for match in _DOCUMENT_VERSION_FIELD_RE.finditer(searchable):
        declared_versions.update(_versions_from_value(match.group(1)))
    # 兼容“版本：云枢6”，但裸的“Java版本：8.6”/“数据库版本：8.6”
    # 不能被当成目标产品版本。
    for value in re.findall(
        rf"(?<!产品)版本\s*[：:]\s*([^\n\r,，;；]+)",
        searchable,
        flags=re.IGNORECASE,
    ):
        if any(
            re.search(re.escape(alias), value, flags=re.IGNORECASE)
            for alias in _product_aliases(product)
        ):
            declared_versions.update(_versions_from_value(value))

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
    # 或正文出现某个组件的8.6，不能证明这是云枢8.6文档。
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
