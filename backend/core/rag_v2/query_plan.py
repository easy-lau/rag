"""Conservative, business-agnostic local planning for RAG v2.

The planner recognizes only strong structural signals in the user's wording.
When no signal is reliable, it returns ``unknown`` instead of guessing
``fact``.  This ensures an uncertain plan can never opt into a narrow
single-chunk evidence path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.query_constraints import (
    ApplicabilityScope,
    extract_applicability_scopes,
    extract_query_constraints,
)
from core.query_surface_structure import (
    answer_target_semantics,
    is_exhaustive_configuration_request,
    is_procedure_question,
    is_distributive_request_tail,
    normalize_coordination_body,
    parse_query_surface_frame,
    parse_distributive_enumeration,
    split_coordination_body,
)
from core.rag_v2.contracts import (
    AnswerRequirementV2,
    BridgeRequirementKind,
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
_CONDITIONAL_POLICY_DISPOSITION_RE = re.compile(
    r"(?:如何|怎么|怎样)\s*处理$",
    re.IGNORECASE,
)
_POLICY_DISPOSITION_TARGET_RE = re.compile(
    r"(?:标准|规定|政策|补贴|补助|费用|额度|上限|下限|期限|比例|待遇|规则|要求|条件)$",
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
# A request needs a whole-document proof only when its answer head is itself a
# governing policy artefact.  This is deliberately much narrower than the
# vocabulary that can appear in an answer.  For example, ``风险处置措施`` and
# ``系统权限`` are often concrete attributes of one named supplier/user; making
# either of them a document-level policy merely because it ends in ``措施`` or
# ``权限`` turns a directly supported answer into a false ``no document root``
# result.  Explicit list/process operators still use collection coverage below.
_DOCUMENT_POLICY_TARGET_RE = re.compile(
    r"(?:制度|政策|规范|规定|办法|细则|规则|标准|管理要求)$",
    re.IGNORECASE,
)
_SINGLE_VALUE_TARGET_RE = re.compile(
    r"(?:金额|数额|额度|上限|下限|比例|天数|次数|时长|期限|数量|日期|时间|"
    r"地址|名称|数值|价格|频率|状态|等级|级别|版本)$",
    re.IGNORECASE,
)
_ORDERED_PROCESS_CONTRACT_RE = re.compile(
    r"(?:流程|步骤|操作步骤|处理流程|按(?:以下)?顺序|"
    r"steps?|procedure|process)",
    re.IGNORECASE,
)

# These are deliberately *shape* markers rather than business vocabulary.
# An implicit mapping question has an entity/condition on the left and a
# measurable/policy attribute on the right (for example ``合同工住宿标准`` or
# ``试用期年假天数``).  Keeping the vocabulary at the level of answer
# attributes makes this useful across HR, travel, security and configuration
# knowledge bases without teaching the planner any particular company rule.
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
_COORDINATED_CLASS_LABEL_RE = re.compile(
    # Generic compact labels used in a list of subjects, for example
    # ``A级、B级、C级和D级的餐补``.  This only proves that the final explicit
    # possessive target can be copied to peer labels; it does not infer any
    # factual relationship or category value.
    r"^(?:[A-Za-z]|[甲乙丙丁戊己庚辛壬癸]|\d{1,3})\s*(?:级|类|档|组)$",
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
_IMPLICIT_COORDINATED_QUESTION_RE = re.compile(
    r"^(?P<body>.+?)(?P<tail>"
    r"(?:是|为)?(?:多少|什么)|"
    r"(?:如何|怎么|怎样)(?:配置|设置|处理|办理|计算|确定|执行)?"
    r")$",
    re.IGNORECASE,
)
_NON_SEMANTIC_QUERY_PLACEHOLDER_RE = re.compile(
    r"^(?:请)?回答(?:用户)?当前问题$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImplicitBridgePlan:
    """Machine-readable bridge inferred from the user's own syntax."""

    subject: str
    description: str
    retrieval_query: str
    # This is inferred only from the grammatical relation used to create the
    # bridge.  It never contains a business value and is preserved all the way
    # to source-table verification.
    kind: BridgeRequirementKind = "classification"


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

    normalized = re.sub(r"\s+", " ", str(question or "")).strip()
    if is_exhaustive_configuration_request(normalized):
        return "collection"
    if answer_shape in {"list", "overview"}:
        return "collection"
    if not normalized:
        return "single"
    # A process question asks for an ordered/composed procedure even when its
    # wording is the singular ``流程是什么``.  It therefore needs collection
    # closure, but it is not a full-document policy request.
    if is_procedure_question(normalized):
        return "collection"
    if _LIST_RE.search(normalized):
        return "collection"
    frame = parse_query_surface_frame(normalized)
    target = (
        frame.answer_target
        if frame is not None
        else _strip_query_tail(normalized)
    )
    # A syntactically named entity (``供应商甲`` / ``客户A`` / ``部门D01``)
    # makes ``X 的 Y 是什么`` a lookup of that entity's Y attribute.  It can
    # still be an explicit list above, but an open interrogative must not be
    # promoted to an exhaustive policy document just because Y resembles a
    # policy noun.
    if frame is not None and any(
        _has_stable_entity_identifier(item.text)
        for item in frame.entity_qualifiers
    ):
        return "single"
    # An open governing-policy head is not a scalar merely because Chinese
    # uses ``什么``.  For example, ``管理要求是什么`` asks for the applicable
    # policy as a whole, whereas ``金额是多少`` still asks for one value.
    if (
        _COLLECTION_OPEN_RE.search(normalized)
        and _DOCUMENT_POLICY_TARGET_RE.search(target)
    ):
        return "collection"
    if (
        _SCALAR_FACT_RE.search(normalized)
        or _SINGLE_VALUE_TARGET_RE.search(target)
    ):
        return "single"
    return "single"


def _answer_coverage_contract(
    question: object,
    *,
    answer_shape: str | None = None,
    coverage_mode: RequirementCoverageMode | None = None,
) -> str:
    """Choose a proof contract from question structure, not business values.

    A collection has two materially different completion meanings.  An
    explicit enumeration asks for list members and may be closed by a bounded
    structured collection.  An open question whose answer head is a governing
    policy/standard/requirement asks for the applicable policy as a whole; one
    sentence containing ``分为`` must not pretend to close that request.  The
    latter therefore requires a complete, authorised document-policy snapshot.
    """

    mode = coverage_mode or _answer_coverage_mode(
        question,
        answer_shape=answer_shape,
    )
    if mode == "single":
        return "single_claim"

    normalized = re.sub(r"\s+", " ", str(question or "")).strip()
    # An explicitly ordered process is not merely a finite set of members.
    # Keep generic operational how-to questions (for example ``如何配置VPN``)
    # as structured collections because a valid answer may be one declarative
    # operation rather than a multi-step sequence.  Both the local planner and
    # the V3 compiler call this helper, so timeout fallback and model
    # compilation retain identical closure semantics.
    if (
        answer_shape == "process"
        and _ORDERED_PROCESS_CONTRACT_RE.search(normalized)
    ):
        return "ordered_steps"

    frame = parse_query_surface_frame(question)
    # ``供应商管理要求有哪些`` still asks for the governing requirement set;
    # the presence of ``有哪些`` does not prove that a bounded list is the
    # complete source.  Check the policy-head contract before the generic
    # enumeration branch.  Finite lists such as ``登录方式有哪些`` do not pass
    # this predicate and remain structured collections.
    if _is_document_policy_request(normalized, frame=frame):
        return "document_policy"
    return "structured_collection"


def _is_document_policy_request(
    question: object,
    *,
    frame=None,
) -> bool:
    """Whether collection completion needs a governing-document snapshot.

    ``coverage_mode=collection`` only says that the answer has more than one
    component.  It must not be conflated with the much stronger claim that a
    whole policy document is required.  This predicate keeps that distinction
    in the planner, where sentence structure and target boundaries are still
    available, instead of making the evidence layer guess from source text.
    """

    normalized = re.sub(r"\s+", " ", str(question or "")).strip()
    if not normalized:
        return False
    parsed = frame or parse_query_surface_frame(normalized)
    if parsed is None:
        return False
    if parsed.question_operator == "process":
        return False
    if any(
        _has_stable_entity_identifier(item.text)
        for item in parsed.entity_qualifiers
    ):
        return False
    return bool(
        _COLLECTION_OPEN_RE.search(normalized)
        and _DOCUMENT_POLICY_TARGET_RE.search(parsed.answer_target)
    )


def _is_conditional_policy_disposition(
    question: object,
    *,
    frame=None,
) -> bool:
    """Recognise a conditional policy outcome instead of a process list.

    ``如何处理`` asks for an operational procedure in some contexts, but a
    conditional standard/limit normally has one applicable disposition.  The
    distinction stays in the shared grammatical planning layer: it neither
    reads documents nor encodes any company-specific business value.
    """

    normalized = re.sub(r"\s+", " ", str(question or "")).strip(" 。！？!?")
    if not _CONDITIONAL_POLICY_DISPOSITION_RE.search(normalized):
        return False
    disposition_head = _CONDITIONAL_POLICY_DISPOSITION_RE.sub("", normalized)
    parsed = frame or parse_query_surface_frame(normalized)
    if parsed is not None and parsed.condition_qualifiers:
        target = _strip_query_tail(parsed.answer_target)
        if _POLICY_DISPOSITION_TARGET_RE.search(target):
            return True
    # Conservative surface parsing intentionally leaves some compact forms
    # intact (for example ``超出报销标准如何处理``).  A visible policy head is
    # still enough to keep that one disposition on the direct single-claim
    # path rather than falsely requiring a complete procedure.
    return bool(_POLICY_DISPOSITION_TARGET_RE.search(disposition_head))


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
    if _COORDINATED_CLASS_LABEL_RE.fullmatch(candidate):
        return True
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


def _coordinated_answer_target(query: str, subject: str) -> str:
    """Return a literal answer target after one shared explicit subject."""

    normalized_query = re.sub(r"\s+", " ", str(query or "")).strip()
    normalized_subject = re.sub(r"\s+", " ", str(subject or "")).strip()
    if not normalized_query or not normalized_subject:
        return ""
    position = normalized_query.casefold().find(normalized_subject.casefold())
    if position < 0:
        return ""
    target = normalized_query[position + len(normalized_subject):]
    target = target.lstrip(" 的：:，,")
    return _strip_query_tail(target)


def _can_share_coordinated_bridge(query: str, bridge: ImplicitBridgePlan) -> bool:
    """Whether a proven sibling bridge is structurally relevant to this item.

    This is intentionally restricted to coordinated questions that already
    have an explicit common subject.  A bare item such as ``住宿`` should not
    be turned into an invented policy term, but it may depend on the same
    category mapping as its sibling ``餐补``.  Direct identity attributes
    (name, code, role, etc.) remain independent even when they share wording.
    """

    subject = bridge.subject
    if not subject or subject.casefold() not in str(query or "").casefold():
        return False
    target = _coordinated_answer_target(query, subject)
    if not target:
        return False
    return not answer_target_semantics(
        query,
        answer_target=target,
        entity_qualifier=subject,
    ).is_direct_attribute


def _with_shared_coordinated_bridges(
    answer_queries: list[str],
    inferred_by_answer: list[tuple[ImplicitBridgePlan, ...]],
) -> list[tuple[ImplicitBridgePlan, ...]]:
    """Propagate an already inferred subject bridge across safe siblings.

    The bridge itself is still inferred from an explicit sibling and remains a
    helpful lookup.  This helper only attaches that same dependency to sibling
    answer nodes; it never invents a subject, target, classification or
    retrieval term.  It fixes a graph inconsistency where ``普通员工的交通、
    住宿和餐补这些分别是多少`` used one category bridge for ``餐补`` but falsely
    treated the two sibling policy questions as independent facts.
    """

    bridge_pool: list[ImplicitBridgePlan] = []
    for values in inferred_by_answer:
        for bridge in values:
            if bridge not in bridge_pool:
                bridge_pool.append(bridge)
    if not bridge_pool:
        return inferred_by_answer
    expanded: list[tuple[ImplicitBridgePlan, ...]] = []
    for query, existing in zip(answer_queries, inferred_by_answer):
        values = list(existing)
        known = {
            (
                item.subject.casefold(),
                re.sub(r"\s+", " ", item.retrieval_query).strip().casefold(),
            )
            for item in values
        }
        for bridge in bridge_pool:
            key = (
                bridge.subject.casefold(),
                re.sub(r"\s+", " ", bridge.retrieval_query).strip().casefold(),
            )
            if key in known or not _can_share_coordinated_bridge(query, bridge):
                continue
            values.append(bridge)
            known.add(key)
        expanded.append(tuple(values))
    return expanded


def infer_implicit_bridges(question: object) -> tuple[ImplicitBridgePlan, ...]:
    """Return at most one bridge, preserving its surface semantics.

    A classification inferred from an ordinary noun phrase is only a possible
    retrieval enhancement.  It must therefore originate from an explicit
    ``entity`` qualifier in :func:`parse_query_surface_frame`; a condition
    (place, duration, phase, threshold or status) never becomes an invented
    classification axis merely because it appears before a policy noun.

    Explicit relation syntax is handled by :func:`infer_implicit_bridge` as a
    proof bridge.  There is deliberately no second pass that turns every
    prepositional condition into another category lookup.
    """

    bridge = infer_implicit_bridge(question)
    return (bridge,) if bridge is not None else ()


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
    # This phrase is an internal compatibility fallback used when a legacy
    # caller has no question text.  It has the surface shape of
    # ``回答用户 + 当前问题`` but is not a user-authored entity/attribute
    # request, so treating it as a mapping would manufacture a bridge task
    # and inflate retrieval for an otherwise single-fact contract.
    if _NON_SEMANTIC_QUERY_PLACEHOLDER_RE.fullmatch(one_line):
        return None

    # Prefer the suffix only when a product/version scope is explicit in the
    # source text and the suffix independently exposes the same grammar.  The
    # frame never treats that scope as the entity being classified.
    scoped_suffix = _suffix_after_explicit_scope(one_line)
    if scoped_suffix is not None:
        scoped_bridge = infer_implicit_bridge(scoped_suffix)
        if scoped_bridge is not None:
            return scoped_bridge

    frame = parse_query_surface_frame(one_line)

    # Explicit relation wording is a hard proof edge, not a helpful
    # classification guess.  Preserve the legacy ``由…决定`` form as well as
    # ``对应``; the frame supplies a normalized right-hand target where one is
    # syntactically present (for example ``对应什么职级`` -> ``职级``).
    relation = _IMPLICIT_RELATION_RE.search(one_line)
    if relation:
        left = _clean_implicit_subject(relation.group("left"))
        right = _strip_query_tail(relation.group("right"))
        if frame is not None and frame.question_operator == "relation":
            right = frame.answer_target or right
        by = _clean_implicit_subject(relation.group("by") or "")
        if _is_implicit_subject(left) and right:
            description = f"确认{left}与{right}之间的适用/决定关系"[:500]
            return ImplicitBridgePlan(
                subject=left,
                description=description,
                retrieval_query=f"{left} {right} 对应关系",
                kind="mapping",
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
                kind="condition",
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
                kind="condition",
            )

    if frame is None:
        return None
    # Only the explicit entity span may request a speculative classification
    # lookup.  Conditions and scopes retain their literal terms in the direct
    # answer query, so a directly applicable clause remains answerable even
    # when no mapping table exists.
    if frame.question_operator == "relation":
        return None
    target = _strip_query_tail(frame.answer_target)
    for qualifier_span in frame.entity_qualifiers:
        qualifier = _clean_implicit_subject(qualifier_span.text)
        if not _is_implicit_subject(qualifier):
            continue
        # A stable identifier makes this an attribute request about one named
        # entity (for example ``供应商甲`` / ``客户A`` / ``部门D01``), not a
        # population whose answer needs speculative class expansion.  Direct
        # evidence is sufficient unless explicit relation syntax above has
        # already created a typed proof edge.
        if _has_stable_entity_identifier(qualifier):
            continue
        if not answer_target_semantics(
            one_line,
            answer_target=target,
            entity_qualifier=qualifier,
        ).classification_augmentation_allowed:
            continue
        description = _implicit_bridge_description(qualifier, target)
        return ImplicitBridgePlan(
            subject=qualifier,
            description=description,
            retrieval_query=(
                f"{qualifier} 对应的适用分类 等级 类别 职级 角色 版本 档位 阶段"
            ),
            kind="classification",
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


def _expand_coordinated_body_tail(
    body: str,
    tail: str,
    *,
    require_complete_targets: bool,
    tail_is_validated_distributive_request: bool = False,
) -> tuple[str, ...]:
    """Expand coordinated answer units while preserving shared scope.

    Explicit distributive markers make every item independently answerable.
    Natural scalar questions often omit that marker (``A 和 B 是多少``), so
    the implicit caller additionally requires each expanded sibling to expose
    a complete target shape.  This avoids splitting arbitrary noun compounds
    merely because they contain a conjunction.
    """

    body = str(body or "").strip(" ，,。；;：:")
    tail = str(tail or "").strip(" ，,。；;：:！？!?")
    # ``还有`` can join a natural list, but it can also start an interrogative
    # phrase such as ``还有哪些``.  The shared surface parser normalizes only
    # the former, so a question word is never manufactured as an answer target.
    body = normalize_coordination_body(body)
    if not body or not tail:
        return ()
    if (
        tail_is_validated_distributive_request
        and not is_distributive_request_tail(tail)
    ):
        # Defensive invariant for future callers: only the shared surface
        # parser may certify an explicit distributive tail.
        return ()
    if not tail_is_validated_distributive_request and not (
        is_procedure_question(tail)
        or any(pattern.search(tail) for pattern in (
            _LIST_RE,
            _JUDGEMENT_RE,
            _SCALAR_FACT_RE,
            _OVERVIEW_RE,
            _COORDINATED_GENERIC_ANSWER_RE,
        ))
    ):
        return ()

    raw_parts = list(split_coordination_body(body))
    if not raw_parts:
        return ()
    if require_complete_targets and any(
        _IMPLICIT_BROAD_TARGET_RE.fullmatch(
            _strip_query_tail(value.rsplit("的", 1)[-1])
        )
        for value in raw_parts
    ):
        # ``制度和标准是什么`` may denote one compound overview rather than
        # two independently answerable values.  A bare shape noun cannot make
        # an omitted distributive marker safe merely through suffix sharing.
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
        # A final item such as ``餐补标准`` can grammatically be read either
        # as a shared head (``交通标准、住宿标准、餐补标准``) or as that item's
        # own compound.  The text alone cannot prove which.  Older code
        # guessed by appending a suffix and fabricated terms such as
        # ``住宿补贴``.  Preserve every literal item instead; the execution
        # DAG later carries the original full question as a shared retrieval
        # anchor without changing an answer target's source wording.
        stems = list(raw_parts)

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

    if require_complete_targets and not all(
        _coordinated_target_is_complete(value) for value in stems
    ):
        return ()

    queries = [f"{value}{tail}" for value in stems]

    # Compact forms may omit ``的`` (``普通员工交通、住宿标准分别是多少``).
    # If the first expanded item exposes one safe qualifier, carry only that
    # qualifier to still-unqualified siblings, then re-run bridge inference.
    # This copies an explicit subject boundary; it never synthesizes a target
    # suffix such as ``住宿补贴``.
    if not first_prefix and queries:
        first_bridge = infer_implicit_bridge(queries[0])
        subject = (
            first_bridge.subject
            if first_bridge is not None
            else _implicit_qualifier_anchor(stems[0])
        )
        if subject is not None and _is_implicit_subject(subject):
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


def _split_coordinated_enumeration(question: str) -> tuple[str, ...]:
    """Expand an explicit ``A、B 和 C 分别...`` question conservatively."""

    structure = parse_distributive_enumeration(question)
    if structure is None or structure.contains_historical_reference:
        return ()
    return _expand_coordinated_body_tail(
        structure.body,
        structure.tail,
        require_complete_targets=False,
        tail_is_validated_distributive_request=True,
    )


def _split_implicit_coordinated_targets(question: str) -> tuple[str, ...]:
    """Split safe ``A 和 B 是多少`` forms without requiring ``分别``.

    A split needs one bounded interrogative plus independently complete target
    shapes after suffix and possessive-scope propagation.  Ambiguous compounds
    remain a single requirement and therefore cannot silently change intent.
    """

    normalized = re.sub(r"\s+", " ", str(question or "")).strip()
    normalized = normalized.strip(" ；;。！？!?")
    match = _IMPLICIT_COORDINATED_QUESTION_RE.fullmatch(normalized)
    if match is None:
        return ()
    return _expand_coordinated_body_tail(
        match.group("body"),
        match.group("tail"),
        require_complete_targets=True,
    )


def _build_multi_answer_plan(
    question: str,
    answer_queries: tuple[str, ...],
    *,
    reason: str,
    answer_shape: str = "multi_part",
    share_coordinated_bridges: bool = False,
) -> QueryPlanV2:
    """Compile multiple answer units and their independent bridge edges.

    Dependency discovery is performed per sub-question before requirements are
    merged.  A bridge inferred for one answer is therefore never attached to a
    sibling merely because both occur in the same user turn.
    """

    # Scope fingerprint is part of the bridge identity.  Equal lexical bridge
    # wording from two projects/versions must remain separate execution
    # vertices; collapsing it would make a later successful bridge fact bleed
    # into its sibling answer.
    BridgeKey = tuple[str, str, str, str]

    def bridge_key(
        inferred: ImplicitBridgePlan,
        query: str,
    ) -> BridgeKey:
        scope = extract_query_constraints(query)
        return (
            inferred.subject.casefold(),
            re.sub(r"\s+", " ", inferred.retrieval_query).strip().casefold(),
            inferred.kind,
            scope.fingerprint,
        )

    bounded_answers = list(answer_queries[:8])
    while bounded_answers:
        inferred_by_answer = [
            infer_implicit_bridges(query) for query in bounded_answers
        ]
        if share_coordinated_bridges:
            inferred_by_answer = _with_shared_coordinated_bridges(
                bounded_answers,
                inferred_by_answer,
            )
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
    if share_coordinated_bridges:
        inferred_by_answer = _with_shared_coordinated_bridges(
            bounded_answers,
            inferred_by_answer,
        )
    unique_bridges: dict[BridgeKey, ImplicitBridgePlan] = {}
    bridge_scope_by_key: dict[BridgeKey, ApplicabilityScope] = {}
    bridge_keys_by_answer: list[tuple[BridgeKey, ...]] = []
    for query, inferred_set in zip(bounded_answers, inferred_by_answer):
        scope = extract_query_constraints(query)
        answer_keys: list[BridgeKey] = []
        for inferred in inferred_set:
            key = bridge_key(inferred, query)
            unique_bridges.setdefault(key, inferred)
            bridge_scope_by_key.setdefault(key, scope)
            answer_keys.append(key)
        bridge_keys_by_answer.append(tuple(dict.fromkeys(answer_keys)))

    bridge_id_by_key = {
        key: f"r{len(bounded_answers) + index}"
        for index, key in enumerate(unique_bridges, start=1)
    }
    answer_specs = [
        (query, keys, _answer_coverage_mode(query, answer_shape=answer_shape))
        for query, keys in zip(bounded_answers, bridge_keys_by_answer)
    ]
    requirements: list[AnswerRequirementV2] = [
        AnswerRequirementV2(
            id=f"r{index}",
            description=query,
            coverage_mode=coverage_mode,
            coverage_contract=_answer_coverage_contract(
                query,
                answer_shape=answer_shape,
                coverage_mode=coverage_mode,
            ),
            depends_on_requirement_ids=tuple(
                bridge_id_by_key[key]
                for key in keys
                if unique_bridges[key].kind != "classification"
            ),
            augmentation_requirement_ids=tuple(
                bridge_id_by_key[key]
                for key in keys
                if unique_bridges[key].kind == "classification"
            ),
            applicability_scope=extract_query_constraints(query),
        )
        for index, (query, keys, coverage_mode) in enumerate(
            answer_specs,
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
            bridge_kind=inferred.kind,
            applicability_scope=bridge_scope_by_key[key],
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
                    str(bridge_scope_by_key[key].project or "")
                    if bridge_scope_by_key[key].has_project_constraint
                    else "",
                    str(bridge_scope_by_key[key].product or ""),
                    str(bridge_scope_by_key[key].version or ""),
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
                coverage_contract="single_claim",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
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
                coverage_mode=(coverage_mode := _answer_coverage_mode(
                    description,
                    answer_shape=answer_shape,
                )),
                coverage_contract=_answer_coverage_contract(
                    description,
                    answer_shape=answer_shape,
                    coverage_mode=coverage_mode,
                ),
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
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


def _comparison_scope_display(scope: ApplicabilityScope) -> str:
    """Render only source-owned applicability terms for one comparison side."""

    values = (
        scope.project if scope.has_project_constraint else None,
        scope.product,
        scope.version,
    )
    return " ".join(
        value
        for value in values
        if isinstance(value, str) and value.strip()
    )


def _comparison_target_tail(
    question: str,
    scopes: tuple[ApplicabilityScope, ...],
) -> str | None:
    """Return the literal shared target after the final explicit scope.

    This is deliberately a rendering helper, not semantic decomposition.  If
    the sentence shape is unfamiliar we return ``None`` and retain the full
    source question on each independently scoped task.  The scope separation
    remains safe either way; only retrieval wording becomes less precise.
    """

    source = str(question or "")
    ends = [
        item.version_source.end
        for item in scopes
        if item.version_source is not None
    ]
    if not ends:
        return None
    tail = source[max(ends):]
    tail = re.sub(r"^\s*(?:的|中|下|内|里|分别|各自)\s*", "", tail)
    tail = tail.strip(" \t\r\n，,。；;：:！!？?")
    # The comparison operator itself is not an answer target.  Strip only a
    # bounded comparison shell; all remaining wording is copied verbatim.
    tail = re.sub(
        r"(?:有什么|有何|哪些)?(?:区别|差异|不同(?:点)?)$",
        "",
        tail,
    ).strip(" \t\r\n，,。；;：:！!？?")
    if not tail or len(tail) > 300:
        return None
    return tail


def _explicit_scope_comparison_plan(question: str) -> QueryPlanV2 | None:
    """Compile a deterministic one-scope-per-side comparison baseline.

    A model analysis can refine source wording later, but it must never be
    required to preserve an explicit product/version comparison.  Each side is
    therefore represented by a separate required answer requirement and its
    own canonical ``ApplicabilityScope``.  The anchor may recall their union;
    downstream task admission is still single-scope.
    """

    scopes = tuple(
        scope
        for scope in extract_applicability_scopes(question)
        if scope.has_version_constraint
    )
    unique_scopes: list[ApplicabilityScope] = []
    seen_fingerprints: set[str] = set()
    for scope in scopes:
        if scope.fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(scope.fingerprint)
        unique_scopes.append(scope)
    if len(unique_scopes) < 2:
        return None

    target_tail = _comparison_target_tail(question, tuple(unique_scopes))
    requirements: list[AnswerRequirementV2] = []
    descriptions: list[str] = []
    for index, scope in enumerate(unique_scopes[:8], start=1):
        scope_display = _comparison_scope_display(scope)
        # A full original question is an intentionally conservative fallback:
        # the local scope remains typed and prevents a third version entering
        # the answer, even when grammar cannot isolate a common target.
        description = (
            " ".join((scope_display, target_tail)).strip()
            if scope_display and target_tail
            else question
        )
        coverage_mode = _answer_coverage_mode(
            description,
            answer_shape="comparison",
        )
        requirement = AnswerRequirementV2(
            id=f"r{index}",
            description=description,
            role="answer",
            importance="required",
            source="explicit",
            coverage_mode=coverage_mode,
            coverage_contract=_answer_coverage_contract(
                description,
                answer_shape="comparison",
                coverage_mode=coverage_mode,
            ),
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
            applicability_scope=scope,
        )
        requirements.append(requirement)
        descriptions.append(description)
    return _ready_plan(
        question,
        answer_shape="comparison",
        confidence=0.97,
        reason="explicit_multi_scope_comparison",
        retrieval_queries=tuple(_unique for _unique in (question, *descriptions)),
        requirements=tuple(requirements),
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
                share_coordinated_bridges=True,
            )
        implicit_coordinated_queries = _split_implicit_coordinated_targets(
            normalized
        )
        if implicit_coordinated_queries:
            return _build_multi_answer_plan(
                normalized,
                implicit_coordinated_queries,
                reason="implicit_coordinated_answer_targets",
                share_coordinated_bridges=True,
            )
        if _MULTI_PART_RE.search(normalized):
            return _build_multi_answer_plan(
                normalized,
                _split_multi_part_query(normalized),
                reason="explicit_multi_part_structure",
            )
        if _COMPARISON_RE.search(normalized):
            scoped_comparison = _explicit_scope_comparison_plan(normalized)
            if scoped_comparison is not None:
                return scoped_comparison
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
            proof_bridge_ids = tuple(
                bridge_id
                for bridge_id, implicit_bridge in zip(
                    bridge_ids,
                    implicit_bridges,
                )
                if implicit_bridge.kind != "classification"
            )
            augmentation_bridge_ids = tuple(
                bridge_id
                for bridge_id, implicit_bridge in zip(
                    bridge_ids,
                    implicit_bridges,
                )
                if implicit_bridge.kind == "classification"
            )
            requirements = (
                AnswerRequirementV2(
                    id="r1",
                    description=normalized,
                    coverage_mode=(coverage_mode := _answer_coverage_mode(
                        normalized,
                    )),
                    coverage_contract=_answer_coverage_contract(
                        normalized,
                        coverage_mode=coverage_mode,
                    ),
                    depends_on_requirement_ids=proof_bridge_ids,
                    augmentation_requirement_ids=augmentation_bridge_ids,
                    applicability_scope=local_scope,
                ),
                *(
                    AnswerRequirementV2(
                        id=bridge_id,
                        description=implicit_bridge.description,
                        role="bridge",
                        importance="helpful",
                        source="inferred",
                        bridge_subject=implicit_bridge.subject,
                        bridge_kind=implicit_bridge.kind,
                        applicability_scope=local_scope,
                    )
                    for bridge_id, implicit_bridge in zip(
                        bridge_ids,
                        implicit_bridges,
                    )
                ),
            )
            # A surface-derived classification (for example ``普通员工``)
            # improves recall only.  The literal user question remains a
            # complete direct task even if no source ever proves that mapping.
            # Explicit ``对应`` / ``由…决定`` syntax is different: it requests
            # a relation and therefore keeps the proof-oriented multi-hop
            # answer shape.
            answer_shape = (
                "multi_hop"
                if proof_bridge_ids
                else "list"
                if (frame := parse_query_surface_frame(normalized)) is not None
                and frame.question_operator == "enumeration"
                else "fact"
            )
            return _ready_plan(
                normalized,
                answer_shape=answer_shape,
                confidence=0.94,
                reason=(
                    "explicit_relation_dependency"
                    if proof_bridge_ids
                    else "implicit_classification_augmentation"
                ),
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
        # A broad policy head can be identified from its open grammatical
        # operator and generic policy suffix even when no leading document
        # noun (``制度/政策``) is present.  Treat it as an overview so the
        # requirement receives ``document_policy`` coverage rather than
        # letting the unknown fallback silently downgrade it to one claim.
        surface_frame = parse_query_surface_frame(normalized)
        if (
            surface_frame is not None
            and _is_document_policy_request(
                normalized,
                frame=surface_frame,
            )
        ):
            return _ready_plan(
                normalized,
                answer_shape="overview",
                confidence=0.88,
                reason="surface_open_policy_signal",
            )
        if is_procedure_question(normalized, frame=surface_frame):
            return _ready_plan(
                normalized,
                answer_shape="process",
                confidence=0.92,
                reason="explicit_process_signal",
            )
        if is_exhaustive_configuration_request(normalized):
            return _ready_plan(
                normalized,
                answer_shape="list",
                confidence=0.9,
                reason="explicit_exhaustive_configuration_signal",
            )
        if _is_conditional_policy_disposition(
            normalized,
            frame=surface_frame,
        ):
            return _ready_plan(
                normalized,
                answer_shape="fact",
                confidence=0.9,
                reason="conditional_policy_disposition",
            )
        if (
            surface_frame is not None
            and surface_frame.question_operator == "value"
        ):
            # A bounded value question is a direct fact once governing-policy
            # heads and explicit process/list structures have been handled
            # above.  This is particularly important for named-entity
            # attributes: removing an unjustified classification augmentation
            # must not strand the direct lookup in ``unknown``.
            return _ready_plan(
                normalized,
                answer_shape="fact",
                confidence=0.88,
                reason="surface_value_lookup_signal",
            )
        if _LIST_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="list",
                confidence=0.92,
                reason="explicit_list_signal",
            )
        frame = parse_query_surface_frame(normalized)
        if frame is not None and frame.question_operator == "enumeration":
            return _ready_plan(
                normalized,
                answer_shape="list",
                confidence=0.9,
                reason="surface_enumeration_signal",
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
