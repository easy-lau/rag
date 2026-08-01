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


def _unknown_plan(
    question: str,
    *,
    reason: str,
    needs_clarification: bool = False,
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
            "请补充需要查询或了解的具体问题。"
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
) -> QueryPlanV2:
    return QueryPlanV2(
        original_query=question,
        answer_shape=answer_shape,
        retrieval_queries=retrieval_queries or (question,),
        requirements=(
            AnswerRequirementV2(
                id="r1",
                description=question,
            ),
        ),
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
        if _MULTI_HOP_RE.search(normalized):
            return _ready_plan(
                normalized,
                answer_shape="multi_hop",
                confidence=0.9,
                reason="explicit_relational_dependency",
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
