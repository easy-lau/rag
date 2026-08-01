"""Conservative, business-agnostic local planning for RAG v2.

The planner recognizes only strong structural signals in the user's wording.
When no signal is reliable, it returns ``unknown`` instead of guessing
``fact``.  This ensures an uncertain plan can never opt into a narrow
single-chunk evidence path.
"""

from __future__ import annotations

import re

from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2


_MULTI_PART_RE = re.compile(
    r"(?:[?？].+[?？])|(?:[；;\n])|(?:^|\s)(?:\d{1,2}[.)、]|[一二三四五六七八九十]+[、.])",
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
    r"(?:是什么|介绍(?:一下)?|概述|概览|总体|整体|主要内容|说明(?:一下)?|"
    r"what\s+is|overview|introduction)",
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
_IMPLICIT_QUALIFIER_SUFFIX_RE = re.compile(
    r"(?:员工|人员|用户|客户|岗位|经理|主管|总监|总裁|主任|负责人|顾问|"
    r"专家|工程师|设计师|分析师|会计师|律师|医师|教师|助理|代表|董事长|"
    r"组长|科长|处长|部长|员|工|岗|期|阶段|身份|对象|主体|申请人|"
    r"部门|地区|城市)$",
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
    }
)
_IMPLICIT_BROAD_TARGET_RE = re.compile(
    r"^(?:管理|制度|政策|规范|办法|总体|整体)(?:标准|要求|规则|内容)$",
    re.IGNORECASE,
)
_COORDINATED_ENUMERATION_RE = re.compile(
    r"^(?P<body>.+?)(?:分别|各自)(?P<tail>[^；;\n]{1,48})$",
    re.IGNORECASE,
)
_COORDINATED_SEPARATOR_RE = re.compile(r"(?:、|[,，]|以及|和|及|与)")


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


def _clean_implicit_subject(value: str) -> str:
    subject = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;：:！!？?（）()[]【】")
    subject = re.sub(r"^(?:请问|请|查询|确认|了解|帮我看一下|帮我查一下)\s*", "", subject)
    return subject.strip()


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
    if _IMPLICIT_QUALIFIER_SUFFIX_RE.search(subject):
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


def _implicit_bridge_description(subject: str, target: str) -> str:
    """Build a bounded, domain-neutral bridge claim for evidence matching."""

    subject = _clean_implicit_subject(subject)
    target = _strip_query_tail(target)
    # Do not include a guessed concrete level/category.  The bridge is only
    # the relationship that evidence must establish; the value remains an
    # evidence-derived fact in the answer requirement.
    return f"确认{subject}对应的适用分类、等级、类别或阶段（用于确定{target}）"[:500]


def infer_implicit_bridge(question: object) -> tuple[str, str] | None:
    """Infer a safe intermediate mapping from query *shape* only.

    Returns ``(bridge_description, bridge_query)`` when the wording exposes a
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
            return description, f"{left} {right} 对应关系"
        # ``该值由前一项决定`` has a pronoun as the grammatical subject,
        # which is intentionally excluded above.  The deciding operand is
        # still a safe bridge anchor even when the answer object is implicit.
        if by and _is_implicit_subject(by):
            description = f"确认{by}与结果之间的决定关系"
            return description, f"{by} 对应结果的决定关系"
    by_relation = _IMPLICIT_BY_RE.search(one_line)
    if by_relation:
        by = _clean_implicit_subject(by_relation.group("by"))
        left = _clean_implicit_subject(by_relation.group("left"))
        if _is_implicit_subject(by):
            description = f"确认{by}与{left or '结果'}之间的决定关系"
            return description, f"{by} {left or '结果'} 决定关系"

    possessive = _IMPLICIT_POSSESSIVE_RE.search(one_line)
    if possessive:
        subject = _clean_implicit_subject(possessive.group("subject"))
        qualifier = _implicit_qualifier_anchor(subject)
        target = _strip_query_tail(possessive.group("target"))
        if (
            qualifier is not None
            and target
            and _IMPLICIT_TARGET_SUFFIX_RE.search(target)
            and not _IMPLICIT_BROAD_TARGET_RE.search(target)
        ):
            description = _implicit_bridge_description(qualifier, target)
            return description, f"{qualifier} 对应的适用分类 等级 类别 阶段"

    # Compact policy questions often omit ``的``.  Split only when the left
    # side has an entity/status suffix; this avoids treating ``交通费用标准``
    # or ``制度内容`` as an implicit relationship.
    compact = _strip_query_tail(one_line)
    if possessive is not None:
        return None
    compact_candidates: list[tuple[str, str]] = []
    for split_at in range(2, max(2, len(compact) - 1)):
        subject = _clean_implicit_subject(compact[:split_at])
        target = compact[split_at:]
        if (
            _is_implicit_subject(subject)
            and _IMPLICIT_QUALIFIER_SUFFIX_RE.search(subject)
            and len(target) >= 2
            and _IMPLICIT_TARGET_SUFFIX_RE.search(target)
            and not _IMPLICIT_BROAD_TARGET_RE.search(target)
        ):
            compact_candidates.append((subject, target))
    if compact_candidates:
        # Prefer the longest qualifier that still has an entity/status suffix;
        # this keeps modifiers such as ``普通`` with ``员工`` while the suffix
        # gate prevents a topic phrase such as ``交通费用`` from swallowing the
        # real target.
        subject, target = max(compact_candidates, key=lambda item: len(item[0]))
        description = _implicit_bridge_description(subject, target)
        return description, f"{subject} 对应的适用分类 等级 类别 阶段"
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
        r"(?:^|\s)(?:\d{1,2}[.)、]|[一二三四五六七八九十]+[、.])\s*",
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
    )):
        return ()

    raw_parts = [
        value.strip(" ，,。；;：:")
        for value in _COORDINATED_SEPARATOR_RE.split(body)
    ]
    if len(raw_parts) < 2 or any(not value for value in raw_parts):
        return ()

    # A trailing policy/measurement suffix commonly scopes the whole list.
    # Reuse it only for items that do not already carry a target suffix.
    suffix_match = _IMPLICIT_TARGET_SUFFIX_RE.search(raw_parts[-1])
    shared_suffix = suffix_match.group(0) if suffix_match is not None else ""
    stems = [
        (
            value + shared_suffix
            if shared_suffix
            and not _IMPLICIT_TARGET_SUFFIX_RE.search(value)
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
            bridge_query = first_bridge[1]
            subject = bridge_query.split(" 对应的适用分类", 1)[0].strip()
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
            answer_queries = list(coordinated_queries)
            bridge_by_query: dict[str, str] = {}
            for query in answer_queries:
                implicit_bridge = infer_implicit_bridge(query)
                if implicit_bridge is None:
                    continue
                bridge_description, bridge_query = implicit_bridge
                bridge_by_query.setdefault(bridge_query, bridge_description)

            # Query plans are bounded to eight requirements.  Reserve room for
            # every distinct inferred mapping so an eight-item list cannot
            # silently keep its answers while dropping the proof dependency.
            if bridge_by_query and len(answer_queries) + len(bridge_by_query) > 8:
                answer_queries = answer_queries[:max(1, 8 - len(bridge_by_query))]
                bridge_by_query = {}
                for query in answer_queries:
                    implicit_bridge = infer_implicit_bridge(query)
                    if implicit_bridge is not None:
                        bridge_by_query.setdefault(
                            implicit_bridge[1],
                            implicit_bridge[0],
                        )

            requirements: list[AnswerRequirementV2] = [
                AnswerRequirementV2(
                    id=f"r{index}",
                    description=query,
                )
                for index, query in enumerate(answer_queries, start=1)
            ]
            retrieval_queries = list(answer_queries)
            for bridge_query, bridge_description in bridge_by_query.items():
                if len(requirements) >= 8:
                    break
                requirements.append(AnswerRequirementV2(
                    id=f"r{len(requirements) + 1}",
                    description=bridge_description,
                    role="bridge",
                    importance="helpful",
                    source="inferred",
                ))
                retrieval_queries.append(bridge_query)
            has_bridge = any(item.role == "bridge" for item in requirements)
            return _ready_plan(
                normalized,
                # A coordinated list with an identity mapping is both
                # multi-answer and multi-hop.  ``multi_hop`` makes every
                # inferred bridge coverage-critical in the evidence layer.
                answer_shape="multi_hop" if has_bridge else "multi_part",
                confidence=0.95,
                reason=(
                    "explicit_coordinated_enumeration_with_mapping"
                    if has_bridge
                    else "explicit_coordinated_enumeration"
                ),
                retrieval_queries=tuple(retrieval_queries),
                requirements=tuple(requirements),
            )
        if _MULTI_PART_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="multi_part",
                confidence=0.95,
                reason="explicit_multi_part_structure",
                retrieval_queries=_split_multi_part_query(normalized),
            )
        if _COMPARISON_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="comparison",
                confidence=0.95,
                reason="explicit_comparison_signal",
            )
        implicit_bridge = infer_implicit_bridge(normalized)
        if implicit_bridge is not None:
            bridge_description, bridge_query = implicit_bridge
            requirements = (
                AnswerRequirementV2(id="r1", description=normalized),
                AnswerRequirementV2(
                    id="r2",
                    description=bridge_description,
                    role="bridge",
                    importance="helpful",
                    source="inferred",
                ),
            )
            return _ready_plan(
                normalized,
                answer_shape="multi_hop",
                confidence=0.94,
                reason="implicit_mapping_dependency",
                retrieval_queries=(normalized, bridge_query),
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
