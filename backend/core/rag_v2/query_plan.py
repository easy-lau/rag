"""Conservative, business-agnostic local planning for RAG v2.

The planner recognizes only strong structural signals in the user's wording.
When no signal is reliable, it returns ``unknown`` instead of guessing
``fact``.  This ensures an uncertain plan can never opt into a narrow
single-chunk evidence path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.query_constraints import extract_query_constraints
from core.rag_v2.contracts import (
    AnswerRequirementV2,
    QueryPlanV2,
    RequirementCoverageMode,
)


_MULTI_PART_RE = re.compile(
    r"(?:[?？].+[?？])|(?:[；;\n])|(?:^|\s)(?:"
    r"\d{1,2}(?:[)、]|\.(?!\d))|[一二三四五六七八九十]+[、.])",
    re.DOTALL,
)
_COMPARISON_RE = re.compile(
    r"(?:对比|比较|区别|差异|不同(?:点)?|异同|优劣|(?:^|\s)(?:vs\.?|versus)(?:\s|$))",
    re.IGNORECASE,
)
_PROCESS_RE = re.compile(
    r"(?:如何|怎么|怎样|步骤|流程|操作方法|办理方法|how\s+to|steps?|procedure)",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"(?:有哪些|包含(?:哪些|什么)|包括(?:哪些|什么)|列出|清单|哪几(?:个|项|种)?|"
    r"(?:(?:需要|需|应当|应|必须|须)\s*)?"
    r"(?:提供|提交|准备|填写|选择)?\s*哪些|"
    r"what\s+are|list\s+(?:the\s+)?)",
    re.IGNORECASE,
)
_JUDGEMENT_RE = re.compile(
    r"(?:是否|能否|可否|能不能|可不可以|是不是|有没有|"
    r"^(?:is|are|can|could|may|does|do|did|has|have)\b)",
    re.IGNORECASE,
)
_MULTI_HOP_RE = re.compile(
    r"(?:对应(?:的|关系|到)?|分别对应|根据.+(?:确定|判断|得到)|由.+(?:决定|确定)|"
    r"基于.+(?:确定|判断|得到)|取决于|映射(?:到|关系)?)",
    re.IGNORECASE,
)
_SCALAR_FACT_RE = re.compile(
    r"(?:多少|几(?:个|项|种|次|天|年|月|日)?|何时|哪里|谁|哪个|哪一个|"
    r"什么(?:时间|日期|金额|数量|级别|等级|状态|类型|版本|名称|值)|"
    r"when|where|who|which|how\s+many|how\s+much)",
    re.IGNORECASE,
)
_OVERVIEW_RE = re.compile(
    r"(?:介绍(?:一下)?|概述|概览|总体|整体|主要内容|说明(?:一下)?|"
    r"overview|introduction)|"
    r"(?:(?:制度|政策|规范|规定|办法|细则|流程|规则|标准|方案|手册|文档|"
    r"合同|说明书)\s*(?:是什么|有(?:哪些|什么)(?:内容)?|包含(?:哪些|什么)))|"
    r"(?:what\s+is\s+(?=[^?\n]{0,48}\b(?:policy|standard|procedure|process|"
    r"rule|guideline|manual|document)\b))",
    re.IGNORECASE,
)
_COORDINATED_GENERIC_ANSWER_RE = re.compile(
    r"(?:是什么|what\s+is)",
    re.IGNORECASE,
)
_COLLECTION_OPEN_RE = re.compile(
    r"(?:是什么|有哪些|有(?:哪些|什么)|包括(?:哪些|什么)|包含(?:哪些|什么)|"
    r"(?:(?:需要|需|应当|应|必须|须)\s*)?"
    r"(?:提供|提交|准备|列明|列出)\s*(?:哪些|什么)|"
    r"列出|清单|全部|所有|完整|主要内容|各项|分别|"
    r"what\s+(?:is|are)|list\s+(?:the\s+)?)",
    re.IGNORECASE,
)
_COLLECTION_POLICY_TARGET_RE = re.compile(
    r"(?:制度|政策|规范|规定|办法|细则|流程|规则|标准|要求|条件|资格|权限|待遇|"
    r"措施|策略|方案|处置)$",
    re.IGNORECASE,
)
_SINGLE_VALUE_TARGET_RE = re.compile(
    r"(?:金额|数额|额度|上限|下限|比例|天数|次数|时长|期限|数量|日期|时间|"
    r"地址|名称|数值|价格|频率|状态|等级|级别|版本)$",
    re.IGNORECASE,
)

# These are deliberately *shape* markers rather than business vocabulary.
# An implicit mapping question has an entity/condition on the left and a
# measurable/policy attribute on the right (for example ``合同工住宿标准`` or
# ``试用期年假天数``).  Keeping the vocabulary at the level of answer
# attributes makes this useful across HR, travel, security and configuration
# knowledge bases without teaching the planner any particular company rule.
_IMPLICIT_TARGET_SUFFIX_RE = re.compile(
    r"(?:标准|金额|天数|次数|额度|上限|下限|比例|时长|条件|要求|权限|角色|版本|"
    r"等级|级别|档位|类别|类型|待遇|补贴|补助|补|费用|价格|周期|规则|频率|数量|日期|"
    r"时间|地址|名称|数值|数额|期限|名额)$",
    re.IGNORECASE,
)
_IMPLICIT_MAPPING_TARGET_SUFFIX_RE = re.compile(
    r"(?:标准|金额|天数|次数|额度|上限|下限|比例|时长|条件|要求|权限|角色|版本|"
    r"等级|级别|档位|类别|类型|待遇|补贴|补助|补|费用|价格|周期|规则|频率|数量|日期|"
    r"时间|地址|名称|数值|数额|期限|名额|措施|策略|方案|处置)$",
    re.IGNORECASE,
)
# Coordinated questions need a wider notion of an already-complete answer
# target than bridge inference does.  For example, both ``风险处置措施`` and
# ``风险等级`` are complete siblings in ``...措施和风险等级分别是什么``;
# borrowing ``等级`` from the last item would corrupt the first question.
# This remains a grammatical target boundary only -- it does not itself cause
# a bridge to be inferred.
_COORDINATED_COMPLETE_TARGET_SUFFIX_RE = re.compile(
    r"(?:标准|金额|天数|次数|额度|上限|下限|比例|时长|条件|要求|权限|角色|版本|"
    r"等级|级别|档位|类别|类型|待遇|补贴|补助|补|费用|价格|周期|规则|频率|数量|日期|"
    r"时间|地址|名称|数值|数额|期限|名额|措施|策略|方案|处置)$",
    re.IGNORECASE,
)
# These targets are themselves identity/classification facts.  Asking for
# ``客户A的名称`` or ``供应商甲的风险等级`` does not imply an additional hidden
# mapping before the requested fact; the source claim answering the question
# is already the mapping.  This semantic class prevents stable entity IDs from
# turning every direct attribute lookup into a multi-hop query.
_IMPLICIT_DIRECT_ATTRIBUTE_TARGET_RE = re.compile(
    r"(?:名称|姓名|全称|简称|联系人(?:名称|姓名)?|联系方式|联系电话|电话|手机号?|"
    r"邮箱|邮件地址|地址|位置|坐标|网址|域名|IP地址|端口|编码|编号|代码|标识|"
    r"ID|账号|账户|统一社会信用代码|身份证号|职级|等级|级别|档位|类别|类型|"
    r"角色|版本|阶段|状态)$",
    re.IGNORECASE,
)
_IMPLICIT_QUALIFIER_SUFFIX_RE = re.compile(
    r"(?:员工|人员|用户|客户|供应商|合作方|承包商|经销商|代理商|租户|"
    r"组织|机构|岗位|经理|主管|总监|总裁|主任|负责人|顾问|"
    r"专家|工程师|设计师|分析师|会计师|律师|医师|教师|助理|代表|董事长|"
    r"组长|科长|处长|部长|员|工|岗|期|阶段|身份|对象|主体|申请人|"
    r"部门|地区|城市)$",
    re.IGNORECASE,
)
_IMPLICIT_ENTITY_IDENTIFIER_RE = re.compile(
    r"(?:[A-Za-z0-9_.+-]{1,16}|[甲乙丙丁戊己庚辛壬癸]{1,2})$",
    re.IGNORECASE,
)
_COMPACT_PREPOSITION_RE = re.compile(
    r"^(?:位于|针对|面向|在|于)(?P<body>[\u3400-\u9fffA-Za-z0-9_.+/-]{3,72})$",
    re.IGNORECASE,
)
_COMPACT_CONDITION_SUFFIX_RE = re.compile(
    r"(?:地区|城市|区域|省|市|县|区|州|国|阶段|期间|部门|组织|机构|"
    r"项目|产品|版本|岗位|角色|公司|企业)$",
    re.IGNORECASE,
)
_IMPLICIT_NESTED_CONTEXT_BLOCK_RE = re.compile(
    r"(?:文档|文件|手册|制度|政策|标准|规范|办法|产品|项目|版本|账号|账户|"
    r"角色|类型|类别|等级|级别|档位)$",
    re.IGNORECASE,
)
_IMPLICIT_POSSESSIVE_RE = re.compile(
    r"(?P<subject>[\u3400-\u9fffA-Za-z0-9_.+/-]{2,32}?)(?:的|所对应的|"
    r"适用的|享受的|对应的)(?P<target>[\u3400-\u9fffA-Za-z0-9_.+/-]{2,48})",
    re.IGNORECASE,
)
_IMPLICIT_RELATION_RE = re.compile(
    r"(?P<left>[\u3400-\u9fffA-Za-z0-9_.+/-]{2,32}?)(?:对应(?:的|到)?|"
    r"属于|归属于|归属|映射(?:到|为)?|由(?P<by>[\u3400-\u9fffA-Za-z0-9_.+/-]{2,32})"
    r"(?:决定|确定)|取决于)(?P<right>[\u3400-\u9fffA-Za-z0-9_.+/-]{1,48})",
    re.IGNORECASE,
)
_IMPLICIT_BY_RE = re.compile(
    r"(?P<left>[\u3400-\u9fffA-Za-z0-9_.+/-]{1,32})由"
    r"(?P<by>[\u3400-\u9fffA-Za-z0-9_.+/-]{2,32})(?:决定|确定)",
    re.IGNORECASE,
)
_IMPLICIT_QUERY_TAIL_RE = re.compile(
    r"(?:是(?:多少|什么)?|为多少|有多少|多少|几(?:个|项|种|次|天|年|月|日)?|"
    r"是什么|怎么(?:算|确定)?|如何|吗|呢|？|\?)$",
    re.IGNORECASE,
)
_IMPLICIT_GENERIC_SUBJECTS = frozenset(
    {
        "某项",
        "某个",
        "某种",
        "这个",
        "该",
        "该项",
        "该值",
        "最终",
        "结果",
        "制度",
        "标准",
        "内容",
        # Bare owner/scope type nouns do not identify an entity that must be
        # mapped to another class.  A concrete modifier or stable identifier
        # remains admissible (for example ``研发部门`` or ``组织A``).
        "公司",
        "企业",
        "集团",
        "组织",
        "机构",
        "部门",
        "单位",
        "团队",
        "地区",
        "城市",
        "区域",
        "项目",
        "产品",
        "系统",
        "平台",
        "用户",
        "客户",
        "供应商",
        "合作方",
        "承包商",
        "经销商",
        "代理商",
        "租户",
    }
)
_IMPLICIT_BROAD_TARGET_RE = re.compile(
    r"^(?:(?:管理|制度|政策|规范|办法|总体|整体)(?:标准|要求|规则|内容)|"
    r"标准|要求|规则|内容|制度|政策|规范|办法)$",
    re.IGNORECASE,
)
_COORDINATED_ENUMERATION_RE = re.compile(
    r"^(?P<body>.+?)(?:分别|各自)(?P<tail>[^；;\n]{1,48})$",
    re.IGNORECASE,
)
_COORDINATED_SEPARATOR_RE = re.compile(r"(?:、|[,，]|以及|和|及|与)")


@dataclass(frozen=True)
class ImplicitBridgePlan:
    """Machine-readable bridge inferred from the user's own syntax."""

    subject: str
    description: str
    retrieval_query: str


def _strip_query_tail(value: str) -> str:
    """Remove punctuation/question wording while preserving policy nouns."""

    result = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n，,。；;：:！!？?）)】]】")
    # A bounded loop handles forms such as ``标准是多少`` and ``天数是几天``
    # without attempting open-ended linguistic parsing.
    for _ in range(3):
        updated = _IMPLICIT_QUERY_TAIL_RE.sub("", result).strip(" \t\r\n，,。；;：:！!？?）)】")
        if updated == result:
            break
        result = updated
    return result


def _answer_coverage_mode(
    question: object,
    *,
    answer_shape: str | None = None,
) -> RequirementCoverageMode:
    """Classify whether one answer requirement denotes a bounded collection.

    This is driven by query shape and answer-target semantics, never by a
    company domain noun.  Explicit list/overview requests are exhaustive.
    For mapping questions, open ``what is`` wording is exhaustive only when
    the target is policy-like; scalar targets and scalar interrogatives stay
    single even if their source requires a bridge.
    """

    if answer_shape in {"list", "overview"}:
        return "collection"
    normalized = re.sub(r"\s+", " ", str(question or "")).strip()
    if not normalized:
        return "single"
    if _LIST_RE.search(normalized):
        return "collection"
    target = _strip_query_tail(normalized)
    if (
        _SCALAR_FACT_RE.search(normalized)
        or _SINGLE_VALUE_TARGET_RE.search(target)
    ):
        return "single"
    if (
        _COLLECTION_OPEN_RE.search(normalized)
        and _COLLECTION_POLICY_TARGET_RE.search(target)
    ):
        return "collection"
    return "single"


def _clean_implicit_subject(value: str) -> str:
    subject = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;：:！!？?（）()[]【】")
    subject = re.sub(r"^(?:请问|请|查询|确认|了解|帮我看一下|帮我查一下)\s*", "", subject)
    # Product/version/project applicability is an independent hard scope, not
    # part of the entity whose classification must be resolved.  Keeping a
    # leading scope such as ``云枢8.6`` inside ``云枢8.6普通员工`` makes the
    # evidence layer search for a nonexistent mapping subject and used to
    # discard the valid ``普通员工 -> D级`` bridge.  Remove only a source-text
    # constraint that is explicitly recognized at the beginning; unknown
    # prefixes remain untouched and can never be guessed away here.
    constraints = extract_query_constraints(subject)
    matched_scope = str(constraints.matched_text or "").strip()
    if matched_scope and subject.casefold().startswith(matched_scope.casefold()):
        subject = subject[len(matched_scope):].lstrip(" ：:，,。；;-_/()（）")
    return subject.strip()


def _suffix_after_explicit_scope(value: str) -> str | None:
    """Return only a self-contained suffix after an explicit version scope.

    A project or tenant label can precede the product/version applicability
    scope (for example ``中青建安的云枢8.2.75普通员工餐补标准``).  The planner
    cannot safely guess whether that unknown prefix is a project or the
    business subject.  It therefore never deletes the prefix directly: it
    exposes only the text after the source-recognized scope, and the caller may
    use it only when that suffix independently forms a valid bridge question.
    Otherwise normal parsing of the original sentence remains authoritative.
    """

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    constraints = extract_query_constraints(text)
    matched_scope = str(constraints.matched_text or "").strip()
    if not matched_scope:
        return None
    pattern = re.compile(
        re.escape(matched_scope).replace(r"\ ", r"\s+"),
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return None
    suffix = text[match.end():].lstrip(
        " \t\r\n：:，,。；;-_/()（）的"
    )
    if not suffix or suffix.casefold() == text.casefold():
        return None
    return suffix


def _is_implicit_subject(value: str) -> bool:
    subject = _clean_implicit_subject(value)
    if len(subject) < 2 or len(subject) > 32:
        return False
    if subject.casefold() in _IMPLICIT_GENERIC_SUBJECTS:
        return False
    if any(token in subject for token in ("多少", "什么", "如何", "怎么", "是否")):
        return False
    return True


def _implicit_qualifier_anchor(value: str) -> str | None:
    """Return the person/condition scope, allowing one trailing scene phrase.

    Natural questions sometimes put an activity between the qualifier and
    ``的`` (for example ``某类人员某活动的费用``).  The trailing phrase is not
    itself a taxonomy.  Select the longest structurally valid qualifier prefix
    while rejecting common document/object scope nouns; no concrete activity
    or policy topic is encoded here.
    """

    subject = _clean_implicit_subject(value)
    if not _is_implicit_subject(subject):
        return None

    def qualifier_with_optional_id(candidate: str) -> bool:
        if not _is_implicit_subject(candidate):
            return False
        if _IMPLICIT_QUALIFIER_SUFFIX_RE.search(candidate):
            return True
        for split_at in range(len(candidate) - 1, 1, -1):
            if (
                _IMPLICIT_ENTITY_IDENTIFIER_RE.fullmatch(candidate[split_at:])
                and _IMPLICIT_QUALIFIER_SUFFIX_RE.search(candidate[:split_at])
            ):
                return True
        return False

    # An explicit prepositional condition is a separate bridge candidate, not
    # part of the entity identifier on its left.
    for match in re.finditer(r"(?:在|于|位于|针对|面向|对|按)", subject):
        if match.start() < 2:
            continue
        prefix = subject[:match.start()]
        if qualifier_with_optional_id(prefix):
            return prefix
    if _IMPLICIT_QUALIFIER_SUFFIX_RE.search(subject):
        return subject
    # Stable ASCII identifiers are part of an entity (``客户A``, ``用户U01``),
    # not an activity suffix to be discarded.  Preserve the longest prefix
    # whose base is a structurally recognized qualifier.
    for split_at in range(len(subject) - 1, 1, -1):
        base = subject[:split_at]
        identifier = subject[split_at:]
        if (
            _IMPLICIT_ENTITY_IDENTIFIER_RE.fullmatch(identifier)
            and _IMPLICIT_QUALIFIER_SUFFIX_RE.search(base)
        ):
            return subject
    candidates: list[str] = []
    for split_at in range(2, len(subject)):
        qualifier = subject[:split_at]
        trailing_context = subject[split_at:]
        if (
            len(trailing_context) <= 6
            and not _IMPLICIT_NESTED_CONTEXT_BLOCK_RE.search(trailing_context)
            and _is_implicit_subject(qualifier)
            and _IMPLICIT_QUALIFIER_SUFFIX_RE.search(qualifier)
        ):
            candidates.append(qualifier)
    return max(candidates, key=len) if candidates else None


def _has_stable_entity_identifier(value: str) -> bool:
    """Whether *value* ends in a stable identifier after a named base.

    This is used only to distinguish coordinated entity scopes (``产品A、
    产品B的版本``) from coordinated answer targets.  It does not make the
    entity eligible for a bridge by itself.
    """

    candidate = _clean_implicit_subject(value)
    for split_at in range(len(candidate) - 1, 1, -1):
        if _IMPLICIT_ENTITY_IDENTIFIER_RE.fullmatch(candidate[split_at:]):
            return True
    return False


def _is_coordinated_scope_item(value: str) -> bool:
    """Recognize one explicit entity/scope in a coordinated subject list."""

    candidate = _clean_implicit_subject(value)
    if not candidate or "的" in candidate:
        return False
    qualifier = _implicit_qualifier_anchor(candidate)
    return qualifier == candidate or _has_stable_entity_identifier(candidate)


def _coordinated_target_is_complete(value: str) -> bool:
    """Check the local target, excluding any possessive scope on its left."""

    target = str(value or "").rsplit("的", 1)[-1].strip()
    return bool(_COORDINATED_COMPLETE_TARGET_SUFFIX_RE.search(target))


def _split_compact_prepositional_target(
    value: str,
) -> tuple[str, str] | None:
    """Split ``在北京住宿标准`` without guessing arbitrary word breaks.

    The possessive form (``在北京的住宿标准``) has an explicit delimiter and
    remains authoritative.  For compact wording we accept only a structurally
    named condition (entity suffix/stable identifier), or a two-CJK-character
    condition followed by a sufficiently complete policy target.  Everything
    else fails closed and leaves the primary bridge unchanged.
    """

    match = _COMPACT_PREPOSITION_RE.fullmatch(str(value or "").strip())
    if match is None:
        return None
    body = match.group("body")
    candidates: list[tuple[int, int, str, str]] = []
    for split_at in range(2, len(body) - 1):
        condition = _clean_implicit_subject(body[:split_at])
        target = _strip_query_tail(body[split_at:])
        if (
            not _is_implicit_subject(condition)
            or _IMPLICIT_NESTED_CONTEXT_BLOCK_RE.search(condition)
            or not _implicit_target_requires_bridge(target)
        ):
            continue
        structural = bool(
            _COMPACT_CONDITION_SUFFIX_RE.search(condition)
            or _implicit_qualifier_anchor(condition) == condition
            or _has_stable_entity_identifier(condition)
        )
        short_cjk = bool(
            len(condition) == 2
            and len(target) >= 4
            and re.fullmatch(r"[\u3400-\u9fff]{2}", condition)
        )
        if not structural and not short_cjk:
            continue
        candidates.append((
            2 if structural else 1,
            len(condition) if structural else -len(condition),
            condition,
            target,
        ))
    if not candidates:
        return None
    _, _, condition, target = max(candidates, key=lambda item: item[:2])
    return condition, target


def _implicit_target_requires_bridge(value: str) -> bool:
    """Whether a possessive target can denote a policy-derived outcome.

    The decision is about answer semantics, independent of any company domain:
    policy limits, entitlements and actions may depend on an intermediate
    source-authored class; direct identity/classification fields do not.
    """

    target = _strip_query_tail(value)
    return bool(
        target
        and _IMPLICIT_MAPPING_TARGET_SUFFIX_RE.search(target)
        and not _IMPLICIT_BROAD_TARGET_RE.search(target)
        and not _IMPLICIT_DIRECT_ATTRIBUTE_TARGET_RE.search(target)
    )


def infer_implicit_bridges(question: object) -> tuple[ImplicitBridgePlan, ...]:
    """Infer all independently stated classification conditions in one query.

    The first bridge is the existing entity/condition mapping.  Additional
    bridges are admitted only from explicit local prepositional conditions
    before the answer target (for example ``X在Y的额度``).  Direct clauses
    that state every original condition can still bypass these helpful edges;
    otherwise each mapping must be proved from source text.
    """

    primary = infer_implicit_bridge(question)
    if primary is None:
        return ()
    normalized = _normalize_query(question)
    possessive = _IMPLICIT_POSSESSIVE_RE.search(normalized)
    if possessive is None:
        compact = _strip_query_tail(normalized)
        primary_position = compact.casefold().find(primary.subject.casefold())
        if primary_position < 0:
            return (primary,)
        remainder = compact[
            primary_position + len(primary.subject):
        ]
        compact_condition = _split_compact_prepositional_target(remainder)
        if compact_condition is None:
            return (primary,)
        condition, target = compact_condition
        primary = ImplicitBridgePlan(
            subject=primary.subject,
            description=_implicit_bridge_description(primary.subject, target),
            retrieval_query=primary.retrieval_query,
        )
        return (
            primary,
            ImplicitBridgePlan(
                subject=condition,
                description=_implicit_bridge_description(condition, target),
                retrieval_query=(
                    f"{condition} 对应的适用分类 等级 类别 区域 档位 阶段"
                ),
            ),
        )
    raw_subject = _clean_implicit_subject(possessive.group("subject"))
    target = _strip_query_tail(possessive.group("target"))
    if not raw_subject or not target:
        return (primary,)
    primary_position = raw_subject.casefold().find(primary.subject.casefold())
    if primary_position < 0:
        return (primary,)
    remainder = raw_subject[primary_position + len(primary.subject):]
    additional: list[ImplicitBridgePlan] = []
    for match in re.finditer(
        r"(?:在|于|位于|针对|面向|对|按)"
        r"(?P<condition>[\u3400-\u9fffA-Za-z0-9_.+/-]{2,32}?)"
        r"(?=(?:在|于|位于|针对|面向|对|按)|$)",
        remainder,
        re.IGNORECASE,
    ):
        condition = _clean_implicit_subject(match.group("condition"))
        if (
            not _is_implicit_subject(condition)
            or condition.casefold() == primary.subject.casefold()
            or _IMPLICIT_NESTED_CONTEXT_BLOCK_RE.search(condition)
        ):
            continue
        additional.append(ImplicitBridgePlan(
            subject=condition,
            description=_implicit_bridge_description(condition, target),
            retrieval_query=(
                f"{condition} 对应的适用分类 等级 类别 区域 档位 阶段"
            ),
        ))
    return tuple(dict.fromkeys((primary, *additional)))[:4]


def _implicit_bridge_description(subject: str, target: str) -> str:
    """Build a bounded, domain-neutral bridge claim for evidence matching."""

    subject = _clean_implicit_subject(subject)
    target = _strip_query_tail(target)
    # Do not include a guessed concrete level/category.  The bridge is only
    # the relationship that evidence must establish; the value remains an
    # evidence-derived fact in the answer requirement.
    return (
        f"确认{subject}对应的适用分类、等级、类别、职级、角色、版本、"
        f"档位或阶段（用于确定{target}）"
    )[:500]


def infer_implicit_bridge(question: object) -> ImplicitBridgePlan | None:
    """Infer a safe intermediate mapping from query *shape* only.

    Returns a typed bridge plan when the wording exposes a
    concrete qualifier and an answer attribute.  It intentionally returns
    ``None`` for underspecified phrases instead of inventing a business
    taxonomy.  Callers must still prove the bridge with retrieved evidence.
    """

    normalized = _normalize_query(question)
    if not normalized:
        return None
    one_line = re.sub(r"\s+", " ", normalized).strip()
    if "\n" in normalized or _MULTI_PART_RE.search(normalized):
        return None

    # Prefer the suffix only when it can prove a complete structural mapping
    # by itself.  This separates a leading project/product/version scope from
    # the real entity (``...8.2.75普通员工``), while preserving the original
    # parse when the scope occurs inside the answer target
    # (``普通员工的云枢8.6配置权限``).
    scoped_suffix = _suffix_after_explicit_scope(one_line)
    if scoped_suffix is not None:
        scoped_bridge = infer_implicit_bridge(scoped_suffix)
        if scoped_bridge is not None:
            return scoped_bridge

    # Explicit relation wording is already a multi-hop signal.  Extract the
    # two sides when possible; otherwise use a deliberately generic relation
    # claim so the plan cannot be treated as a single verified fact.
    relation = _IMPLICIT_RELATION_RE.search(one_line)
    if relation:
        left = _clean_implicit_subject(relation.group("left"))
        right = _strip_query_tail(relation.group("right"))
        by = _clean_implicit_subject(relation.group("by") or "")
        if _is_implicit_subject(left) and right:
            description = (
                f"确认{left}与{right}之间的适用/决定关系"
            )[:500]
            return ImplicitBridgePlan(
                subject=left,
                description=description,
                retrieval_query=f"{left} {right} 对应关系",
            )
        # ``该值由前一项决定`` has a pronoun as the grammatical subject,
        # which is intentionally excluded above.  The deciding operand is
        # still a safe bridge anchor even when the answer object is implicit.
        if by and _is_implicit_subject(by):
            description = f"确认{by}与结果之间的决定关系"
            return ImplicitBridgePlan(
                subject=by,
                description=description,
                retrieval_query=f"{by} 对应结果的决定关系",
            )
    by_relation = _IMPLICIT_BY_RE.search(one_line)
    if by_relation:
        by = _clean_implicit_subject(by_relation.group("by"))
        left = _clean_implicit_subject(by_relation.group("left"))
        if _is_implicit_subject(by):
            description = f"确认{by}与{left or '结果'}之间的决定关系"
            return ImplicitBridgePlan(
                subject=by,
                description=description,
                retrieval_query=f"{by} {left or '结果'} 决定关系",
            )

    possessive = _IMPLICIT_POSSESSIVE_RE.search(one_line)
    if possessive:
        subject = _clean_implicit_subject(possessive.group("subject"))
        qualifier = _implicit_qualifier_anchor(subject)
        target = _strip_query_tail(possessive.group("target"))
        if (
            qualifier is not None
            and _implicit_target_requires_bridge(target)
        ):
            description = _implicit_bridge_description(qualifier, target)
            return ImplicitBridgePlan(
                subject=qualifier,
                description=description,
                retrieval_query=(
                    f"{qualifier} 对应的适用分类 等级 类别 职级 角色 版本 档位 阶段"
                ),
            )

    # Compact policy questions often omit ``的``.  Split only when the left
    # side has an entity/status suffix; this avoids treating ``交通费用标准``
    # or ``制度内容`` as an implicit relationship.
    compact = _strip_query_tail(one_line)
    if possessive is not None:
        return None
    compact_candidates: list[tuple[str, str]] = []
    for split_at in range(2, max(2, len(compact) - 1)):
        raw_subject = _clean_implicit_subject(compact[:split_at])
        subject = _implicit_qualifier_anchor(raw_subject)
        if subject is None:
            continue
        # A longer trial split may already contain a local condition, for
        # example ``客户A按地区``.  The entity anchor is still ``客户A`` and
        # the condition belongs to the answer target; never silently turn the
        # generic word ``地区`` into the entity being classified.
        target = (
            compact[len(subject):]
            if raw_subject != subject and compact.startswith(subject)
            else compact[split_at:]
        )
        if (
            _is_implicit_subject(subject)
            and len(target) >= 2
            and _implicit_target_requires_bridge(target)
        ):
            compact_candidates.append((subject, target))
    if compact_candidates:
        # Prefer the longest qualifier that still has an entity/status suffix;
        # this keeps modifiers such as ``普通`` with ``员工`` while the suffix
        # gate prevents a topic phrase such as ``交通费用`` from swallowing the
        # real target.
        subject, target = max(compact_candidates, key=lambda item: len(item[0]))
        description = _implicit_bridge_description(subject, target)
        return ImplicitBridgePlan(
            subject=subject,
            description=description,
            retrieval_query=(
                f"{subject} 对应的适用分类 等级 类别 职级 角色 版本 档位 阶段"
            ),
        )
    return None


def _normalize_query(question: object) -> str:
    if not isinstance(question, str):
        return ""
    normalized_lines = [
        re.sub(r"[^\S\r\n]+", " ", line).strip()
        for line in question.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    return "\n".join(line for line in normalized_lines if line).strip()[:1000]


def _split_multi_part_query(question: str) -> tuple[str, ...]:
    normalized = re.sub(
        r"(?:^|\s)(?:\d{1,2}(?:[)、]|\.(?!\d))|"
        r"[一二三四五六七八九十]+[、.])\s*",
        "\n",
        question,
    )
    values = [
        re.sub(r"\s+", " ", value).strip(" ?？；;。")
        for value in re.split(r"[?？；;\n]+", normalized)
    ]
    unique: list[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
        if len(unique) >= 8:
            break
    return tuple(unique) or (question,)


def _split_coordinated_enumeration(question: str) -> tuple[str, ...]:
    """Expand an explicit ``A、B 和 C 分别...`` question conservatively.

    The rule is driven by coordination syntax, not by policy vocabulary.  A
    suffix written once on the final item (``交通、住宿和餐补标准``) is shared
    with earlier bare items, and an explicit possessive prefix is retained on
    every generated query.  Without both a coordination separator and
    ``分别``/``各自``, this helper returns an empty tuple.
    """

    normalized = re.sub(r"\s+", " ", str(question or "")).strip()
    normalized = normalized.strip(" ；;。！？!?")
    match = _COORDINATED_ENUMERATION_RE.fullmatch(normalized)
    if match is None:
        return ()
    body = match.group("body").strip(" ，,。；;：:")
    tail = match.group("tail").strip(" ，,。；;：:！？!?")
    if not body or not tail or not _COORDINATED_SEPARATOR_RE.search(body):
        return ()
    if not any(pattern.search(tail) for pattern in (
        _PROCESS_RE,
        _LIST_RE,
        _JUDGEMENT_RE,
        _SCALAR_FACT_RE,
        _OVERVIEW_RE,
        _COORDINATED_GENERIC_ANSWER_RE,
    )):
        return ()

    raw_parts = [
        value.strip(" ，,。；;：:")
        for value in _COORDINATED_SEPARATOR_RE.split(body)
    ]
    if len(raw_parts) < 2 or any(not value for value in raw_parts):
        return ()

    stems = list(raw_parts)

    # ``供应商甲、供应商乙的风险处置措施`` coordinates subjects, not
    # answer-target fragments.  When the final item alone contains a
    # possessive target and every preceding item is an explicit named scope,
    # copy the whole target to those scopes.  This structural branch runs
    # before suffix sharing so ``措施`` can never be appended to an entity
    # name as if it were a missing unit.
    final_scope = ""
    final_target = ""
    if "的" in raw_parts[-1] and all("的" not in item for item in raw_parts[:-1]):
        final_scope, final_target = raw_parts[-1].rsplit("的", 1)
        final_scope = final_scope.strip()
        final_target = final_target.strip()
    coordinated_subjects = bool(
        final_scope
        and final_target
        and _is_coordinated_scope_item(final_scope)
        and all(_is_coordinated_scope_item(item) for item in raw_parts[:-1])
    )
    if coordinated_subjects:
        stems = [
            *(f"{value}的{final_target}" for value in raw_parts[:-1]),
            raw_parts[-1],
        ]
    else:
        # A trailing policy/measurement suffix commonly scopes a list of
        # answer targets (``交通、住宿和餐补标准``).  Borrow it only for a
        # sibling whose own local target is incomplete.  In particular,
        # ``风险处置措施`` must remain intact beside ``风险等级``.
        suffix_match = _IMPLICIT_TARGET_SUFFIX_RE.search(raw_parts[-1])
        shared_suffix = (
            suffix_match.group(0) if suffix_match is not None else ""
        )
        stems = [
            (
                value + shared_suffix
                if shared_suffix
                and not _coordinated_target_is_complete(value)
                else value
            )
            for value in raw_parts
        ]

    # ``普通员工的`` is grammatical scope for every coordinated target, not
    # merely the first one.  Retaining it also lets the bridge detector prove
    # the same classification dependency for every answer item.
    first_prefix = ""
    if "的" in stems[0]:
        prefix, first_target = stems[0].rsplit("的", 1)
        if _is_implicit_subject(prefix) and first_target:
            first_prefix = f"{prefix}的"
    if first_prefix:
        stems = [
            value
            if index == 0 or "的" in value or value.startswith(first_prefix)
            else f"{first_prefix}{value}"
            for index, value in enumerate(stems)
        ]

    queries = [f"{value}{tail}" for value in stems]

    # Compact forms may omit ``的`` (``普通员工交通、住宿标准分别是多少``).
    # If the first expanded item exposes one safe qualifier, carry only that
    # qualifier to still-unqualified siblings, then re-run bridge inference.
    if not first_prefix and queries:
        first_bridge = infer_implicit_bridge(queries[0])
        if first_bridge is not None:
            subject = first_bridge.subject
            if _is_implicit_subject(subject):
                for index in range(1, len(queries)):
                    if infer_implicit_bridge(queries[index]) is None:
                        queries[index] = f"{subject}{stems[index]}{tail}"

    unique: list[str] = []
    for value in queries:
        query = re.sub(r"\s+", " ", value).strip()
        if query and query not in unique:
            unique.append(query)
        if len(unique) >= 8:
            break
    return tuple(unique)


def _build_multi_answer_plan(
    question: str,
    answer_queries: tuple[str, ...],
    *,
    reason: str,
    answer_shape: str = "multi_part",
) -> QueryPlanV2:
    """Compile multiple answer units and their independent bridge edges.

    Dependency discovery is performed per sub-question before requirements are
    merged.  A bridge inferred for one answer is therefore never attached to a
    sibling merely because both occur in the same user turn.
    """

    BridgeKey = tuple[str, str, str, str]

    def bridge_key(
        inferred: ImplicitBridgePlan,
        query: str,
    ) -> BridgeKey:
        scope = extract_query_constraints(query)
        return (
            inferred.subject.casefold(),
            re.sub(r"\s+", " ", inferred.retrieval_query).strip().casefold(),
            str(scope.product or "").casefold(),
            str(scope.version or "").casefold(),
        )

    bounded_answers = list(answer_queries[:8])
    while bounded_answers:
        inferred_by_answer = [
            infer_implicit_bridges(query) for query in bounded_answers
        ]
        unique_bridges: dict[BridgeKey, ImplicitBridgePlan] = {}
        for query, inferred_set in zip(bounded_answers, inferred_by_answer):
            for inferred in inferred_set:
                unique_bridges.setdefault(bridge_key(inferred, query), inferred)
        if len(bounded_answers) + len(unique_bridges) <= 8:
            break
        bounded_answers.pop()

    if not bounded_answers:
        return _unknown_plan(question, reason="multi_answer_requirement_budget_exhausted")

    inferred_by_answer = [
        infer_implicit_bridges(query) for query in bounded_answers
    ]
    unique_bridges: dict[BridgeKey, ImplicitBridgePlan] = {}
    bridge_scope_by_key: dict[
        BridgeKey,
        tuple[str | None, str | None, bool],
    ] = {}
    bridge_keys_by_answer: list[tuple[BridgeKey, ...]] = []
    for query, inferred_set in zip(bounded_answers, inferred_by_answer):
        scope = extract_query_constraints(query)
        answer_keys: list[BridgeKey] = []
        for inferred in inferred_set:
            key = bridge_key(inferred, query)
            unique_bridges.setdefault(key, inferred)
            bridge_scope_by_key.setdefault(
                key,
                (scope.product, scope.version, scope.explicit_version),
            )
            answer_keys.append(key)
        bridge_keys_by_answer.append(tuple(dict.fromkeys(answer_keys)))

    bridge_id_by_key = {
        key: f"r{len(bounded_answers) + index}"
        for index, key in enumerate(unique_bridges, start=1)
    }
    requirements: list[AnswerRequirementV2] = [
        AnswerRequirementV2(
            id=f"r{index}",
            description=query,
            coverage_mode=_answer_coverage_mode(query),
            depends_on_requirement_ids=tuple(
                bridge_id_by_key[key] for key in keys
            ),
        )
        for index, (query, keys) in enumerate(
            zip(bounded_answers, bridge_keys_by_answer),
            start=1,
        )
    ]
    requirements.extend(
        AnswerRequirementV2(
            id=bridge_id_by_key[key],
            description=inferred.description,
            role="bridge",
            importance="helpful",
            source="inferred",
            bridge_subject=inferred.subject,
            scope_product=bridge_scope_by_key[key][0],
            scope_version=bridge_scope_by_key[key][1],
            scope_explicit_version=bridge_scope_by_key[key][2],
        )
        for key, inferred in unique_bridges.items()
    )
    retrieval_queries = [*bounded_answers]
    retrieval_queries.extend(
        re.sub(
            r"\s+",
            " ",
            " ".join(
                value
                for value in (
                    str(bridge_scope_by_key[key][0] or ""),
                    str(bridge_scope_by_key[key][1] or ""),
                    inferred.retrieval_query,
                )
                if value
            ),
        ).strip()
        for key, inferred in unique_bridges.items()
    )
    return _ready_plan(
        question,
        answer_shape=answer_shape,
        confidence=0.95,
        reason=(
            f"{reason}_with_mapping" if unique_bridges else reason
        ),
        retrieval_queries=tuple(retrieval_queries),
        requirements=tuple(requirements),
    )


def _unknown_plan(
    question: str,
    *,
    reason: str,
    needs_clarification: bool = False,
    clarification_question: str | None = None,
) -> QueryPlanV2:
    requirements = (
        (
            AnswerRequirementV2(
                id="r1",
                description=question,
                depends_on_requirement_ids=(),
            ),
        )
        if question
        else ()
    )
    return QueryPlanV2(
        original_query=question,
        answer_shape="unknown",
        retrieval_queries=((question,) if question else ()),
        requirements=requirements,
        confidence=0.0,
        source="fallback",
        reason=reason,
        needs_clarification=needs_clarification,
        clarification_question=(
            clarification_question
            or "请补充需要查询或了解的具体问题。"
            if needs_clarification
            else None
        ),
    )


def _ready_plan(
    question: str,
    *,
    answer_shape: str,
    confidence: float,
    reason: str,
    retrieval_queries: tuple[str, ...] | None = None,
    requirements: tuple[AnswerRequirementV2, ...] | None = None,
) -> QueryPlanV2:
    planned_queries = retrieval_queries or (question,)
    if requirements is None:
        requirement_descriptions = (
            planned_queries if answer_shape == "multi_part" else (question,)
        )
        planned_requirements = tuple(
            AnswerRequirementV2(
                id=f"r{index}",
                description=description,
                coverage_mode=_answer_coverage_mode(
                    description,
                    answer_shape=answer_shape,
                ),
                depends_on_requirement_ids=(),
            )
            for index, description in enumerate(
                requirement_descriptions,
                start=1,
            )
        )
    else:
        planned_requirements = tuple(requirements)
    return QueryPlanV2(
        original_query=question,
        answer_shape=answer_shape,
        retrieval_queries=planned_queries,
        requirements=planned_requirements,
        confidence=confidence,
        source="local",
        reason=reason,
    )


def plan_query_locally(question: object) -> QueryPlanV2:
    """Build a conservative local plan without adding domain assumptions.

    Classification order is intentional: explicit multi-part, comparison and
    relational signals take precedence over scalar wording.  For example, a
    question containing both a relationship and "how much" is ``multi_hop``,
    not a narrow ``fact`` lookup.
    """

    try:
        normalized = _normalize_query(question)
        if not normalized:
            return _unknown_plan(
                "",
                reason="empty_or_invalid_query",
                needs_clarification=True,
            )
        coordinated_queries = _split_coordinated_enumeration(normalized)
        if coordinated_queries:
            return _build_multi_answer_plan(
                normalized,
                coordinated_queries,
                reason="explicit_coordinated_enumeration",
            )
        if _MULTI_PART_RE.search(normalized):
            return _build_multi_answer_plan(
                normalized,
                _split_multi_part_query(normalized),
                reason="explicit_multi_part_structure",
            )
        if _COMPARISON_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="comparison",
                confidence=0.95,
                reason="explicit_comparison_signal",
            )
        implicit_bridges = infer_implicit_bridges(normalized)
        if implicit_bridges:
            local_scope = extract_query_constraints(normalized)
            bridge_ids = tuple(
                f"r{index}" for index in range(2, 2 + len(implicit_bridges))
            )
            requirements = (
                AnswerRequirementV2(
                    id="r1",
                    description=normalized,
                    coverage_mode=_answer_coverage_mode(normalized),
                    depends_on_requirement_ids=bridge_ids,
                ),
                *(
                    AnswerRequirementV2(
                        id=bridge_id,
                        description=implicit_bridge.description,
                        role="bridge",
                        importance="helpful",
                        source="inferred",
                        bridge_subject=implicit_bridge.subject,
                        scope_product=local_scope.product,
                        scope_version=local_scope.version,
                        scope_explicit_version=local_scope.explicit_version,
                    )
                    for bridge_id, implicit_bridge in zip(
                        bridge_ids,
                        implicit_bridges,
                    )
                ),
            )
            return _ready_plan(
                normalized,
                answer_shape="multi_hop",
                confidence=0.94,
                reason="implicit_mapping_dependency",
                retrieval_queries=(
                    normalized,
                    *(
                        re.sub(
                            r"\s+",
                            " ",
                            " ".join(
                                value
                                for value in (
                                    str(local_scope.product or ""),
                                    str(local_scope.version or ""),
                                    implicit_bridge.retrieval_query,
                                )
                                if value
                            ),
                        ).strip()
                        for implicit_bridge in implicit_bridges
                    ),
                ),
                requirements=requirements,
            )
        if _MULTI_HOP_RE.search(normalized):
            return _unknown_plan(
                normalized,
                reason="multi_hop_bridge_cannot_be_safely_decomposed",
                needs_clarification=True,
                clarification_question=(
                    "请补充要确认的对象、它与哪一分类或条件的对应关系，"
                    "以及最终需要查询的标准或数值。"
                ),
            )
        if _PROCESS_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="process",
                confidence=0.92,
                reason="explicit_process_signal",
            )
        if _LIST_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="list",
                confidence=0.92,
                reason="explicit_list_signal",
            )
        if _JUDGEMENT_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="judgement",
                confidence=0.92,
                reason="explicit_judgement_signal",
            )
        if _SCALAR_FACT_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="fact",
                confidence=0.9,
                reason="explicit_scalar_lookup_signal",
            )
        if _OVERVIEW_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="overview",
                confidence=0.86,
                reason="explicit_overview_signal",
            )
        return _unknown_plan(
            normalized,
            reason="no_reliable_local_answer_shape_signal",
        )
    except Exception:
        # Planning is advisory.  Any local implementation error must stay
        # conservative and can never authorize the narrow fact path.
        normalized = _normalize_query(question)
        return _unknown_plan(
            normalized,
            reason="local_planner_failed",
            needs_clarification=not bool(normalized),
        )
