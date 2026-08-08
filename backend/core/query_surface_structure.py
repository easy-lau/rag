"""Shared, conservative parsing of the current user's surface structure.

This module deliberately answers only one question: does the *current
utterance itself* contain an explicit coordinated enumeration with a
distributive request?  It does not classify domains, infer facts, resolve
history, or decide retrieval.  Keeping that grammar in one place prevents the
conversation resolver and the local query planner from assigning incompatible
meanings to words such as ``这些`` / ``还有`` / ``分别``.

The parser is intentionally conservative.  When the grammar is incomplete or
could mean a historical reference, callers receive ``None`` or an enumeration
whose ``has_local_anaphora_antecedent`` is false.  They can then retain the
normal contextual-routing path rather than manufacture a new standalone task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


_DISTRIBUTIVE_ENUMERATION_RE = re.compile(
    r"^(?P<body>.+?)(?P<marker>分别|各自|逐项|各项)(?P<tail>[^；;\n]{1,64})$",
    re.IGNORECASE,
)
_COORDINATION_SEPARATOR_RE = re.compile(r"(?:、|[,，]|以及|和|及|与)")
_ADDITIVE_SEPARATOR_RE = re.compile(
    # ``还有`` is a coordinator in ``住宿、餐补还有出差补贴`` but a question
    # word in ``风险还有哪些``.  Treat only the former as a list delimiter.
    r"还有\s*(?!(?:哪些|什么|没有|吗|其他)(?:\s|$))"
)
_LOCAL_ANAPHORA_RE = re.compile(r"(?:这些|那些)\s*$")
_DISTRIBUTIVE_TAIL_RE = re.compile(
    r"\s*(?:"
    r"(?:是|为)?(?:多少|什么|哪些|如何|怎么|怎样|有何|有什么)|"
    r"(?:如何|怎么|怎样)\s*(?:配置|设置|处理|办理|计算|确定|执行|说明|回答|列出|确认)?|"
    r"(?:进行|予以|需要)?(?:说明|处理|配置|设置|回答|列出|确认)(?:一下|下)?"
    r")\s*[？?。.!！\s]*$",
    re.IGNORECASE,
)

_EXHAUSTIVE_CONFIGURATION_RE = re.compile(
    r"(?:完整|详细|具体|全部|所有)\s*(?:的)?\s*"
    r"(?:配置|参数|内容|步骤|方案|设置|清单|说明)"
    r"(?:是什么|有哪些|怎么配置|如何配置|是什么样的)?\s*[？?。.!！\s]*$",
    re.IGNORECASE,
)


def is_exhaustive_configuration_request(question: object) -> bool:
    """Recognize an open completeness/refinement request by grammar only."""

    normalized = str(question or "").strip()
    if _EXHAUSTIVE_CONFIGURATION_RE.fullmatch(normalized):
        return True
    # Canonical follow-up queries use an immutable root plus a refinement tail
    # separated by ordinary sentence punctuation or whitespace.  Inspect only
    # that terminal user-authored clause;
    # never classify an arbitrary answer body as exhaustive merely because it
    # contains the word “完整”.
    tail_match = re.search(
        r"(?:^|[；;，,。.!！?？\s])(?P<tail>(?:请给我|请提供|给我|提供|补充)?\s*"
        r"(?:完整|详细|具体|全部|所有)\s*(?:的)?\s*(?:配置|参数|内容|步骤|方案|设置|清单|说明)"
        r"(?:是什么|有哪些|怎么配置|如何配置|是什么样的)?\s*[？?。.!！\s]*)$",
        normalized,
        re.IGNORECASE,
    )
    return bool(tail_match and tail_match.group("tail").strip() != normalized)
# These are only reference forms that make an item non-local.  The full
# conversation reference classifier remains responsible for actual follow-up
# decisions; this is deliberately narrower and solely guards the current
# sentence's claimed antecedent.
_HISTORICAL_REFERENCE_IN_ITEM_RE = re.compile(
    r"(?:这些|那些|上述|上面(?:的)?|前面(?:的)?|刚才(?:的)?|上一(?:条|轮|个)(?:的)?|"
    r"其中|它们?)",
    re.IGNORECASE,
)
_LOCAL_ANAPHORA_REQUEST_RE = re.compile(
    r"^(?P<body>.+?)(?P<anaphora>这些|那些)"
    # A collective head is syntactically attached to the list in the current
    # turn (``住宿和餐补这些配置分别如何处理``), not an antecedent imported
    # from history.  Keep the set deliberately generic and bounded.
    r"(?P<collective_head>(?:配置|内容|项目|问题|参数|选项|事项|资料|信息)?)\s*"
    r"(?P<tail>(?:(?:分别|各自|逐项|各项)\s*)?"
    r"(?:"
    r"(?:是|为)?(?:多少|什么|哪些|如何|怎么|怎样|有何|有什么)|"
    r"(?:如何|怎么|怎样)\s*(?:配置|设置|处理|办理|计算|确定|执行|说明|回答|列出|确认)?|"
    r"(?:进行|予以|需要)?(?:说明|处理|配置|设置|回答|列出|确认)(?:一下|下)?"
    r"))\s*[？?。.!！\s]*$",
    re.IGNORECASE,
)
_CANDIDATE_COORDINATION_RE = re.compile(
    r"(?:、|[,，]|；|;|\n|以及|并且|还有|和|与|及|分别|各自|逐项|各项)",
    re.IGNORECASE,
)
_CANDIDATE_CLAUSE_SEPARATOR_RE = re.compile(r"(?:[?？；;\n])")
_CANDIDATE_SCAFFOLD_RE = re.compile(
    r"^(?:[\s、,，；;。！？?!（）()【】\[\]]|"
    r"的|和|与|及|以及|并且|还有|分别|各自|逐项|各项|这些|那些|"
    r"是|为|多少|多久|多长时间|什么|哪些|如何|怎么|怎样|有何|有什么|"
    r"需要|需|应当|应该|应|必须|须|要|"
    r"提供|提交|准备|上传|填写|选择|列出|列明|包含|包括|"
    r"请问|请|查询|帮我|一下|下|吗|呢)*$",
    re.IGNORECASE,
)
# A current-turn ellipsis may safely inherit literal wording from an earlier
# clause only when the later clause is visibly an action/question shell after
# its noun target is removed.  This is deliberately narrower than the general
# candidate-completeness check below: punctuation alone never authorises a
# model to paste arbitrary words from one clause onto another requirement.
_ELLIPTICAL_ACTION_TARGET_SHELL_RE = re.compile(
    r"^(?:(?:需要|需|应当|应该|应|必须|须|要)\s*)?"
    r"(?:(?:提供|提交|准备|上传|填写|选择|列出|列明|包含|包括)\s*)?"
    r"(?:什么|哪些|哪类|何种|几项)\s*$",
    re.IGNORECASE,
)

# A deliberately narrow conversational ellipsis form.  This parser does not
# resolve history; it only proves that the *current* turn contains a literal
# answer head after a discourse marker.  Keeping that proof in the shared
# surface layer prevents the execution service from growing a second set of
# ad-hoc "那……呢" regular expressions.
_CONTEXTUAL_ELLIPSIS_RE = re.compile(
    r"^\s*(?:那么|那麼|那)\s*"
    r"(?P<target>[^\s？?。！!；;，,]{2,80}?)\s*呢\s*[？?。！!]*\s*$",
    re.IGNORECASE,
)
_CONTEXTUAL_ELLIPSIS_GENERIC_TARGET_RE = re.compile(
    r"^(?:这个|那个|这些|那些|这项|那项|上述(?:内容|问题|项目|资料|信息)?|"
    r"上面(?:的)?|前面(?:的)?|刚才(?:的)?|上一(?:条|轮|个)(?:的)?|"
    r"什么|哪些|如何|怎么|怎样|怎么办|有没有|有吗|是否|可以吗|行吗)$",
    re.IGNORECASE,
)
_CONTEXTUAL_ELLIPSIS_NON_SINGLE_TARGET_RE = re.compile(
    r"(?:、|[,，]|以及|和|及|与|还有|分别|各自|逐项|各项)",
    re.IGNORECASE,
)


def normalize_current_question(value: object) -> str:
    """Normalize user text without deleting any meaningful source words."""

    return re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n；;。！？!?")


@dataclass(frozen=True)
class ContextualEllipsisTarget:
    """One explicit answer head in a narrowly-scoped follow-up utterance.

    The offsets always point into the original, unnormalised current user
    input.  That makes the object safe to turn into a
    ``QueryAnalysisSourceRef`` later without shifting source positions.
    """

    question: str
    target: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if not self.question or not self.target:
            raise ValueError("contextual ellipsis target requires source text")
        if not (0 <= self.start < self.end <= len(self.question)):
            raise ValueError("contextual ellipsis target offsets are invalid")
        if self.question[self.start:self.end] != self.target:
            raise ValueError("contextual ellipsis target must match source text")


def parse_contextual_ellipsis_target(
    question: object,
) -> ContextualEllipsisTarget | None:
    """Parse only ``那/那么 + explicit single target + 呢``.

    The result deliberately excludes pronouns, question shells and coordinated
    lists.  It has no history access and makes no claim that the target is a
    particular business term.  More ambiguous follow-ups remain available to
    the model-backed source-analysis path (or an explicit clarification),
    rather than inheriting a stale subject deterministically.
    """

    if not isinstance(question, str) or not question.strip():
        return None
    match = _CONTEXTUAL_ELLIPSIS_RE.fullmatch(question)
    if match is None:
        return None
    target = str(match.group("target") or "")
    if (
        not target
        or _CONTEXTUAL_ELLIPSIS_GENERIC_TARGET_RE.fullmatch(target) is not None
        or _CONTEXTUAL_ELLIPSIS_NON_SINGLE_TARGET_RE.search(target) is not None
    ):
        return None
    start, end = match.span("target")
    # The parser owns a source-offset contract.  Do not normalise the target
    # here: exact source text is required by query_analysis.v2 validation.
    return ContextualEllipsisTarget(
        question=question,
        target=target,
        start=start,
        end=end,
    )


def normalize_coordination_body(value: object) -> str:
    """Normalize the one safe additive coordinator into a regular separator."""

    body = str(value or "").strip(" ，,。；;：:")
    return _ADDITIVE_SEPARATOR_RE.sub("、", body)


def split_coordination_body(value: object) -> tuple[str, ...]:
    """Return non-empty coordinated units, or ``()`` for malformed input."""

    body = normalize_coordination_body(value)
    if not body or _COORDINATION_SEPARATOR_RE.search(body) is None:
        return ()
    parts = tuple(
        item.strip(" \t\r\n，,。；;：:")
        for item in _COORDINATION_SEPARATOR_RE.split(body)
    )
    if not 2 <= len(parts) <= 8 or any(not item for item in parts):
        return ()
    return parts


def is_distributive_request_tail(value: object) -> bool:
    """Whether a suffix is a complete request after ``分别/各项``."""

    return _DISTRIBUTIVE_TAIL_RE.fullmatch(str(value or "").strip()) is not None


def has_current_turn_local_enumeration_antecedent(question: object) -> bool:
    """Whether a demonstrative is resolved by an explicit list in this turn.

    The function is intentionally broader than task splitting: even without
    ``分别`` (for example ``住宿、餐补这些是多少``), an explicit preceding list
    proves that the demonstrative is not a request to inherit stale history.
    The planner may still keep such a question as one broad retrieval task
    when it cannot safely prove independent answer targets.
    """

    distributive = parse_distributive_enumeration(question)
    if (
        distributive is not None
        and distributive.has_local_anaphora_antecedent
    ):
        return True
    normalized = normalize_current_question(question)
    match = _LOCAL_ANAPHORA_REQUEST_RE.fullmatch(normalized)
    if match is None:
        return False
    parts = split_coordination_body(match.group("body"))
    return bool(
        parts
        and all(
            _HISTORICAL_REFERENCE_IN_ITEM_RE.search(part) is None
            for part in parts
        )
    )


@dataclass(frozen=True)
class DistributiveEnumeration:
    """A source-preserving, current-turn coordinated enumeration."""

    question: str
    body: str
    tail: str
    marker: str
    parts: tuple[str, ...]
    local_anaphora: str | None = None

    @property
    def contains_historical_reference(self) -> bool:
        """Whether any apparent item still needs a previous turn to resolve."""

        return any(
            _HISTORICAL_REFERENCE_IN_ITEM_RE.search(part) is not None
            for part in self.parts
        )

    @property
    def has_local_anaphora_antecedent(self) -> bool:
        """Whether ``这些/那些`` is safely resolved by siblings in this turn."""

        return bool(
            self.local_anaphora
            and len(self.parts) >= 2
            and not self.contains_historical_reference
        )


def parse_distributive_enumeration(question: object) -> DistributiveEnumeration | None:
    """Parse a complete ``A、B（还有）C 这些分别...`` current-turn form.

    A demonstrative is considered local only when it appears immediately after
    the enumerated body and before an explicit distributive marker.  This
    distinguishes ``住宿、餐补这些分别是多少`` from ``这些配置分别如何处理`` and
    from an incomplete fragment such as ``住宿和餐补这些分别``.
    """

    normalized = normalize_current_question(question)
    if not normalized:
        return None
    # Parse the current-turn anaphora form first.  A generic regular match
    # would otherwise leave ``这些配置`` inside the final list item and make
    # the conversation resolver and planner disagree again.  The collective
    # head is source text and may be safely copied to every explicit sibling.
    local_match = _LOCAL_ANAPHORA_REQUEST_RE.fullmatch(normalized)
    if local_match is not None:
        raw_tail = local_match.group("tail").strip(" ，,。；;：:！？!?")
        marker_match = re.match(r"(?P<marker>分别|各自|逐项|各项)\s*(?P<tail>.+)$", raw_tail)
        if marker_match is not None and is_distributive_request_tail(
            marker_match.group("tail")
        ):
            collective_head = local_match.group("collective_head").strip()
            base_parts = split_coordination_body(local_match.group("body"))
            if not base_parts:
                return None
            parts = tuple(
                f"{part}{collective_head}" if collective_head else part
                for part in base_parts
            )
            return DistributiveEnumeration(
                question=normalized,
                body="、".join(parts),
                tail=marker_match.group("tail").strip(),
                marker=marker_match.group("marker"),
                parts=parts,
                local_anaphora=local_match.group("anaphora"),
            )
    match = _DISTRIBUTIVE_ENUMERATION_RE.fullmatch(normalized)
    if match is None:
        return None
    tail = match.group("tail").strip(" ，,。；;：:！？!?")
    if not tail or not is_distributive_request_tail(tail):
        return None
    raw_body = match.group("body").strip(" ，,。；;：:")
    local_match = _LOCAL_ANAPHORA_RE.search(raw_body)
    local_anaphora = local_match.group(0).strip() if local_match else None
    body = raw_body[:local_match.start()].strip(" ，,。；;：:") if local_match else raw_body
    parts = split_coordination_body(body)
    if not parts:
        return None
    return DistributiveEnumeration(
        question=normalized,
        body=body,
        tail=tail,
        marker=match.group("marker"),
        parts=parts,
        local_anaphora=local_anaphora,
    )


# ---------------------------------------------------------------------------
# Current-turn query frame
# ---------------------------------------------------------------------------
#
# The enum parser above deliberately owns only coordinated enumerations.  The
# frame below is the other half of the same sentence-level boundary: it
# normalizes a single current question into its answer head and explicit
# syntactic qualifiers.  It is intentionally independent of retrieval,
# history, document names, tenant data and company vocabulary.  Callers may
# use the frame to decide *whether* an entity can be enriched, but never to
# assert a classification/value that is absent from source evidence.

QualifierKind = Literal["entity", "condition", "scope"]
QuestionOperator = Literal[
    "value",
    "enumeration",
    "relation",
    "comparison",
    "process",
    "judgement",
    "unknown",
]


# The answer head is intentionally classified from its grammatical role, not
# from any knowledge-base domain vocabulary.  A population question may need
# an optional ``population -> class`` lookup before a policy value can be
# found, whereas an identity/relationship question must be answered directly.
# Keeping this policy next to the surface parser gives planning and
# source-span compilation one authority for the decision.
_DIRECT_ANSWER_ATTRIBUTE_RE = re.compile(
    r"(?:名称|姓名|全称|简称|联系人(?:名称|姓名)?|联系方式|联系电话|电话|手机号?|"
    r"邮箱|邮件地址|地址|位置|坐标|网址|域名|IP地址|端口|编码|编号|代码|标识|"
    r"ID|账号|账户|统一社会信用代码|身份证号|职级|等级|级别|档位|类别|类型|"
    r"角色|版本|阶段|状态|负责人|主管|上级|归属|所属|成员)$",
    re.IGNORECASE,
)
_BROAD_ANSWER_TARGET_RE = re.compile(
    r"^(?:(?:管理|制度|政策|规范|办法|总体|整体)(?:标准|要求|规则|内容)|"
    r"标准|要求|规则|内容|制度|政策|规范|办法)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerTargetSemantics:
    """Trusted semantic boundary for one answer target.

    ``classification_augmentation_allowed`` is only permission to retrieve a
    possible population/class mapping.  It never asserts that such a mapping
    exists, never supplies a class value, and never turns the augmentation
    into a required proof edge.
    """

    target: str
    question_operator: QuestionOperator
    entity_qualifier: str | None
    is_direct_attribute: bool
    is_broad_policy_target: bool
    classification_augmentation_allowed: bool


_QUESTION_SHELL_RE = re.compile(
    r"^(?P<prefix>.+?)(?:有|具备|包含|包括|涉及|提供|支持)\s*"
    r"(?:什么|哪些|哪类|何种|几项)\s*(?P<noun>"
    r"[\u3400-\u9fffA-Za-z0-9_.+/-]{1,48})$",
    re.IGNORECASE,
)
# This is a sentence grammar rather than a company terminology rule:
# ``报销需要提供哪些凭证`` / ``申请需提交哪些材料``.  It separates the
# business/action head from the modal question shell so source-anchored
# semantics can later carry an explicit head from a sibling current clause.
_ACTION_QUESTION_SHELL_WITH_NOUN_RE = re.compile(
    r"^(?P<prefix>.+?)\s*"
    r"(?:(?:需要|需|应当|应该|应|必须|须|要)\s*)?"
    r"(?:提供|提交|准备|上传|填写|选择|列出|列明|包含|包括)\s*"
    r"(?:什么|哪些|哪类|何种|几项)\s*"
    r"(?P<noun>[\u3400-\u9fffA-Za-z0-9_.+/-]{1,48})$",
    re.IGNORECASE,
)
_ACTION_QUESTION_SHELL_WITHOUT_NOUN_RE = re.compile(
    r"^(?P<prefix>.+?)\s*"
    r"(?:(?:需要|需|应当|应该|应|必须|须|要)\s*)?"
    r"(?:提供|提交|准备|上传|填写|选择|列出|列明|包含|包括)\s*"
    r"(?:什么|哪些|哪类|何种|几项)$",
    re.IGNORECASE,
)
# ``X 需要满足什么条件`` is a question about the source-authored condition
# head ``X 条件``.  It is neither a free-form paraphrase nor a business
# vocabulary rule: the suffix explicitly supplies the noun that names the
# answer target, while the modal/predicate words are only question grammar.
# Keeping this in the shared surface parser makes retrieval planning and
# evidence adjudication use the same target boundary.  Without it, a valid
# source such as ``报销条件：……`` cannot support the user wording
# ``报销需要满足什么条件`` because the latter was treated as one opaque target.
_CONDITION_REQUIREMENT_QUESTION_RE = re.compile(
    r"^(?P<prefix>.+?)\s*"
    r"(?:需要|需|应当|应该|应|必须|须|要)\s*"
    r"(?:满足|符合|具备)\s*"
    r"(?:什么|哪些|哪类|何种)\s*"
    r"(?P<noun>条件|要求)$",
    re.IGNORECASE,
)
_QUESTION_TAIL_RE = re.compile(
    r"(?:是|为|有)?(?:多少|几(?:个|项|种|次|天|月|年)?|"
    r"多久|多长时间|何时|什么时候|哪里|什么|哪些|哪类|何种|几项|吗|呢|么)$",
    re.IGNORECASE,
)
_SHELL_ACTION_PREFIX_RE = re.compile(
    r"^(?:(?:需要|需|应当|应该|应|必须|须|要)\s*)?"
    r"(?:(?:提供|提交|准备|上传|填写|选择|列出|列明|包含|包括|采用|使用|配置|设置)\s*)?$",
    re.IGNORECASE,
)
_RELATION_RE = re.compile(
    r"^(?P<left>[\u3400-\u9fffA-Za-z0-9_.+/-]{2,64}?)"
    r"(?:所)?对应(?:的|到|为)?(?P<right>.+)$",
    re.IGNORECASE,
)
_COMPARISON_OPERATOR_RE = re.compile(
    r"(?:对比|比较|区别|差异|不同(?:点)?|异同|优劣|"
    r"(?:^|\s)(?:vs\.?|versus)(?:\s|$))",
    re.IGNORECASE,
)
# Interrogatives such as ``如何`` describe a question form, not an answer
# contract.  A grounded RAG pipeline must not turn every ``如何处理`` into an
# exhaustive procedure merely because an answer could contain steps.  Process
# closure is reserved for an explicit process noun or a concrete operational
# action.  The action list is grammatical/operational rather than a company
# terminology list, and is shared by the surface frame and local planner.
_EXPLICIT_PROCESS_NOUN_RE = re.compile(
    r"(?:流程|步骤|操作方法|办理方法|操作步骤|处理流程|"
    r"how\s+to|steps?|procedure)",
    re.IGNORECASE,
)
_PROCEDURAL_INTERROGATIVE_RE = re.compile(
    r"(?:如何|怎么|怎样)\s*(?:"
    r"完成|办理|申请|提交|配置|设置|安装|部署|注册|登录|开通|"
    r"填写|审批|报销|操作|执行|导入|同步|创建|修改|删除|撤销|"
    r"启用|停用|排查|修复|升级|迁移"
    r")|(?:想|需要|准备|打算)[^？?。.!！]{1,48}怎么办",
    re.IGNORECASE,
)
_JUDGEMENT_OPERATOR_RE = re.compile(
    r"(?:是否|能否|可否|能不能|可不可以|是不是|有没有|"
    r"^(?:is|are|can|could|may|does|do|did|has|have)\b)",
    re.IGNORECASE,
)
_ENUMERATION_OPERATOR_RE = re.compile(
    r"(?:有哪些|有(?:哪些|什么)|包含(?:哪些|什么)|包括(?:哪些|什么)|"
    r"具备(?:哪些|什么)|(?:提供|提交|准备|上传|填写|选择|列出|列明)(?:哪些|什么)|"
    r"支持(?:哪些|什么)|"
    r"哪类|何种|几项|列出|清单)",
    re.IGNORECASE,
)
_VALUE_OPERATOR_RE = re.compile(
    r"(?:多少|几(?:个|项|种|次|天|月|年)?|何时|哪里|谁|哪个|"
    r"什么(?:时间|日期|金额|数量|级别|等级|状态|类型|版本|名称|值)?|"
    r"when|where|who|which|how\s+many|how\s+much)",
    re.IGNORECASE,
)

# These are grammatical forms of an entity/population/party, not a registry of
# business values.  The words identify a syntactic subject capable of carrying
# a later source-proven classification.  A bare place, duration, state or
# threshold intentionally does not match this set.
_ENTITY_SURFACE_SUFFIX_RE = re.compile(
    r"(?:员工|人员|用户|客户|供应商|合作方|承包商|经销商|代理商|租户|"
    r"申请人|主体|对象|组织|机构|岗位|经理|主管|总监|总裁|主任|负责人|顾问|"
    r"专家|工程师|设计师|分析师|会计师|律师|医师|教师|助理|代表|董事长|"
    r"组长|科长|处长|部长|员|工|岗)$",
    re.IGNORECASE,
)
_ENTITY_IDENTIFIER_RE = re.compile(
    r"(?:[A-Za-z0-9_.+-]{1,16}|[甲乙丙丁戊己庚辛壬癸]{1,2})$",
    re.IGNORECASE,
)
_ENTITY_IDENTIFIER_BASE_RE = re.compile(
    r"(?:组织|机构|部门|单位|团队|公司|企业|集团)$",
    re.IGNORECASE,
)
_SCOPE_LABEL_RE = re.compile(
    r"^(?:[A-Za-z]|[甲乙丙丁戊己庚辛壬癸]|\d{1,4})\s*"
    r"(?:级|类|档|组|层|序列)$",
    re.IGNORECASE,
)
_VERSION_SCOPE_RE = re.compile(
    r"^(?:v(?:ersion)?\s*)?\d+(?:\.\d+){1,4}(?:版|版本)?$|"
    r"^\d{4}\s*版$|^[\u3400-\u9fffA-Za-z]{1,24}版(?:本)?$",
    re.IGNORECASE,
)
_SCOPE_NOUN_RE = re.compile(
    r"(?:角色|职级|等级|级别|类别|类型|版本|产品|项目|手册|文档)$",
    re.IGNORECASE,
)
_CONDITION_NOUN_RE = re.compile(
    r"(?:地区|城市|区域|省|市|县|区|州|国|期间|阶段|时段|周期|"
    r"状态|阈值|上限|下限|范围)$",
    re.IGNORECASE,
)
_LEADING_CONDITION_RE = re.compile(
    r"^(?P<condition>(?:连续)?(?:超过|不少于|不超过|大于|小于|高于|低于|"
    r"达到|未满|满)\s*\d+(?:\.\d+)?(?:天|日|周|月|年|小时|分钟|次|"
    r"个|项|%|％)?)"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
_LEADING_CONDITION_NOUN_RE = re.compile(
    r"^(?P<condition>[\u3400-\u9fffA-Za-z0-9_.+/-]{2,32}?"
    r"(?:地区|城市|区域|省|市|县|区|州|国|期间|阶段|时段|周期|状态|"
    r"阈值|上限|下限|范围))(?P<rest>.*)$",
    re.IGNORECASE,
)
_PREPOSITIONAL_CONDITION_PREFIX_RE = re.compile(
    r"^(?:在|于|针对|面向|按|对)\s*(?P<body>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QualifierSpan:
    """One explicit current-turn qualifier and its syntactic kind.

    Character positions refer to ``QuerySurfaceFrame.question`` after normal
    whitespace/punctuation normalization.  The spans are observational only;
    they never encode a resolved bridge value or a KB fact.
    """

    text: str
    kind: QualifierKind
    start: int
    end: int

    def __post_init__(self) -> None:
        text = re.sub(r"\s+", " ", str(self.text or "")).strip()
        if not text:
            raise ValueError("qualifier text must not be empty")
        if self.kind not in {"entity", "condition", "scope"}:
            raise ValueError("unsupported qualifier kind")
        if self.start < 0 or self.end < self.start:
            raise ValueError("qualifier span positions are invalid")
        object.__setattr__(self, "text", text)


@dataclass(frozen=True)
class QuerySurfaceFrame:
    """Pure sentence-level representation of the user's current question."""

    question: str
    answer_target: str
    context_terms: tuple[str, ...]
    qualifiers: tuple[QualifierSpan, ...]
    question_operator: QuestionOperator

    def __post_init__(self) -> None:
        question = normalize_current_question(self.question)
        answer_target = re.sub(r"\s+", " ", str(self.answer_target or "")).strip()
        if not answer_target:
            raise ValueError("query surface frame requires an answer target")
        if self.question_operator not in {
            "value",
            "enumeration",
            "relation",
            "comparison",
            "process",
            "judgement",
            "unknown",
        }:
            raise ValueError("unsupported question operator")
        context_terms = tuple(
            dict.fromkeys(
                term
                for value in self.context_terms
                if (term := re.sub(r"\s+", " ", str(value or "")).strip())
            )
        )[:16]
        qualifiers = tuple(self.qualifiers)
        if any(not isinstance(value, QualifierSpan) for value in qualifiers):
            raise ValueError("qualifiers must contain QualifierSpan values")
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "answer_target", answer_target)
        object.__setattr__(self, "context_terms", context_terms)
        object.__setattr__(self, "qualifiers", qualifiers)

    @property
    def entity_qualifiers(self) -> tuple[QualifierSpan, ...]:
        return tuple(item for item in self.qualifiers if item.kind == "entity")

    @property
    def condition_qualifiers(self) -> tuple[QualifierSpan, ...]:
        return tuple(item for item in self.qualifiers if item.kind == "condition")

    @property
    def scope_qualifiers(self) -> tuple[QualifierSpan, ...]:
        return tuple(item for item in self.qualifiers if item.kind == "scope")


def _clean_surface_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(
        " \t\r\n，,。；;：:！!？?（）()[]【】"
    )


def is_procedure_question(
    question: object,
    *,
    frame: QuerySurfaceFrame | None = None,
) -> bool:
    """Whether the current wording requests a procedure contract.

    ``如何`` / ``怎么`` / ``怎样`` alone are deliberately insufficient.  They
    also occur in conditional policy questions such as ``超出标准如何处理``;
    treating those as a process forces a collection-closure requirement and
    can discard a direct, fully supported rule.  ``frame`` is accepted so
    callers can pass their already parsed sentence boundary, but the decision
    itself remains purely source-grammatical and never inspects knowledge-base
    facts.
    """

    normalized = normalize_current_question(question)
    if not normalized:
        return False
    if _EXPLICIT_PROCESS_NOUN_RE.search(normalized):
        return True
    if _PROCEDURAL_INTERROGATIVE_RE.search(normalized):
        return True
    # English ``how to`` is itself the conventional imperative/procedure form;
    # it is covered by the explicit process expression above.  Deliberately do
    # not treat a bare Chinese ``如何处理`` as equivalent: it usually asks for
    # the applicable disposition of the immediately preceding policy target.
    return False


def _surface_operator(question: str) -> QuestionOperator:
    if _RELATION_RE.search(question):
        return "relation"
    if _COMPARISON_OPERATOR_RE.search(question):
        return "comparison"
    if is_procedure_question(question):
        return "process"
    if _JUDGEMENT_OPERATOR_RE.search(question):
        return "judgement"
    if _ENUMERATION_OPERATOR_RE.search(question):
        return "enumeration"
    if _VALUE_OPERATOR_RE.search(question):
        return "value"
    return "unknown"


def _classify_whole_qualifier(value: str) -> QualifierKind | None:
    """Classify a complete qualifier from syntax, without topic knowledge."""

    candidate = _clean_surface_text(value)
    if len(candidate) < 2:
        return None
    if (
        _SCOPE_LABEL_RE.fullmatch(candidate)
        or _VERSION_SCOPE_RE.fullmatch(candidate)
        or _SCOPE_NOUN_RE.search(candidate)
    ):
        return "scope"
    numeric_condition = _LEADING_CONDITION_RE.fullmatch(candidate)
    if (
        _CONDITION_NOUN_RE.search(candidate)
        or (
            numeric_condition is not None
            and not _clean_surface_text(numeric_condition.group("rest"))
        )
        or candidate.startswith(("在", "于", "当", "按", "针对", "面向"))
    ):
        return "condition"
    # A single-character role suffix must not absorb an attached condition:
    # ``总经理在岗`` is a manager plus an on-duty state, not one entity name.
    # The connector check is syntactic and remains independent of any domain
    # noun/value list.
    has_internal_condition_connector = bool(
        re.search(r"(?:在|于|按|对|向|给|以|将|把|超过|不少于|不超过)", candidate)
    )
    if (
        _ENTITY_SURFACE_SUFFIX_RE.search(candidate)
        and not has_internal_condition_connector
    ):
        return "entity"
    # A named entity can append a compact stable identifier to a generic
    # entity noun (``客户A`` / ``供应商甲``).  The base must still be one of the
    # syntactic entity forms; arbitrary word+letter strings remain unknown.
    for split_at in range(len(candidate) - 1, 1, -1):
        base, identifier = candidate[:split_at], candidate[split_at:]
        if (
            _ENTITY_IDENTIFIER_RE.fullmatch(identifier)
            and (
                _ENTITY_SURFACE_SUFFIX_RE.search(base)
                or _ENTITY_IDENTIFIER_BASE_RE.search(base)
            )
            and not re.search(r"(?:在|于|按|对|向|给|以|将|把|超过|不少于|不超过)", base)
        ):
            return "entity"
    return None


def is_entity_qualifier(value: object) -> bool:
    """Whether a literal source phrase is an entity/population qualifier.

    This exposes the parser's grammatical classification to trusted compiler
    code without making a separate RAG layer recreate entity suffix rules.
    It says nothing about a factual class or a knowledge-base mapping.
    """

    return _classify_whole_qualifier(str(value or "")) == "entity"


def is_stable_entity_qualifier(value: object) -> bool:
    """Whether an entity qualifier carries a stable instance identifier.

    ``供应商甲`` / ``客户A`` / ``部门D01`` are direct entity-attribute scopes,
    unlike role/population phrases such as ``普通员工``.  This syntactic fact
    is shared by bridge policy and never uses a company entity registry.
    """

    candidate = _clean_surface_text(value)
    if _classify_whole_qualifier(candidate) != "entity":
        return False
    for split_at in range(len(candidate) - 1, 1, -1):
        base, identifier = candidate[:split_at], candidate[split_at:]
        if (
            _ENTITY_IDENTIFIER_RE.fullmatch(identifier)
            and (
                _ENTITY_SURFACE_SUFFIX_RE.search(base)
                or _ENTITY_IDENTIFIER_BASE_RE.search(base)
            )
        ):
            return True
    return False


def answer_target_semantics(
    question: object,
    *,
    answer_target: object | None = None,
    entity_qualifier: object | None = None,
) -> AnswerTargetSemantics:
    """Describe whether one target may use class-mapping augmentation.

    ``answer_target`` and ``entity_qualifier`` may be supplied by a
    source-span compiler.  This is important for a contextual follow-up such
    as ``那住宿呢``: the target is in the current turn while ``普通员工`` is a
    validated history span, so reparsing concatenated history would be both
    unsafe and unnecessary.

    Bare outcome nouns are intentionally allowed.  The old suffix-only rule
    treated ``餐补`` as eligible but silently rejected ``住宿``.  Here the
    decision rests on the combination of a non-stable population qualifier,
    non-relationship question form, and a non-identity/non-broad answer head.
    """

    frame = parse_query_surface_frame(question)
    operator: QuestionOperator = (
        frame.question_operator if frame is not None else "unknown"
    )
    target = _clean_surface_text(
        answer_target if answer_target is not None
        else (frame.answer_target if frame is not None else "")
    )
    qualifier = _clean_surface_text(entity_qualifier)
    if not qualifier and frame is not None:
        qualifier = next((item.text for item in frame.entity_qualifiers), "")

    direct_attribute = bool(
        target and _DIRECT_ANSWER_ATTRIBUTE_RE.search(target)
    )
    broad_policy_target = bool(
        target and _BROAD_ANSWER_TARGET_RE.fullmatch(target)
    )
    population_qualifier = bool(
        qualifier
        and is_entity_qualifier(qualifier)
        and not is_stable_entity_qualifier(qualifier)
    )
    # ``unknown`` is intentionally included for source-bound ellipsis.  It
    # cannot create a bridge alone: a validated non-stable population and a
    # concrete answer target are still mandatory.
    outcome_operator = operator in {"value", "enumeration", "unknown"}
    allow_augmentation = bool(
        target
        and population_qualifier
        and outcome_operator
        and not direct_attribute
        and not broad_policy_target
    )
    return AnswerTargetSemantics(
        target=target,
        question_operator=operator,
        entity_qualifier=qualifier or None,
        is_direct_attribute=direct_attribute,
        is_broad_policy_target=broad_policy_target,
        classification_augmentation_allowed=allow_augmentation,
    )


def current_turn_candidate_targets_are_complete(
    question: object,
    *,
    target_ranges: tuple[tuple[int, int], ...],
    qualifier_ranges: tuple[tuple[int, int], ...] = (),
    trusted_ranges: tuple[tuple[int, int], ...] = (),
) -> bool:
    """Prove that source-anchored candidates cover all current-turn targets.

    Model candidates cannot replace a generic direct answer merely because
    they name two phrases.  This helper verifies that all remaining original
    text is grammatical scaffolding after exact target, qualifier and trusted
    route-scope ranges are removed.  It therefore rejects both a dropped
    sibling (leftover business words) and an invented extra target.  The
    implementation is intentionally generic: it recognises sentence-level
    coordination and explicitly separated question clauses, not company terms
    such as meals or travel.
    """

    source = str(question or "")
    if len(target_ranges) < 2 or not source or parse_query_surface_frame(source) is None:
        return False

    def valid_ranges(values: tuple[tuple[int, int], ...]) -> bool:
        return all(
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(source)
            for start, end in values
        )

    if not (
        valid_ranges(target_ranges)
        and valid_ranges(qualifier_ranges)
        and valid_ranges(trusted_ranges)
    ):
        return False
    ordered_targets = tuple(sorted(target_ranges))
    if len(set(ordered_targets)) != len(ordered_targets):
        return False
    if any(
        left_end > right_start
        for (_, left_end), (right_start, _) in zip(
            ordered_targets,
            ordered_targets[1:],
        )
    ):
        return False
    # Candidate targets must be syntactically coordinated *or* belong to
    # explicitly separated question clauses.  The latter supports a local
    # ellipsis such as ``A 的时限多久？需要哪些凭证？``: the second target may
    # cite the literal ``A`` from the first clause as a current-turn qualifier.
    # Adjacent arbitrary substrings remain invalid.
    between_targets = "".join(
        source[left_end:right_start]
        for (_, left_end), (right_start, _) in zip(
            ordered_targets,
            ordered_targets[1:],
        )
    )
    if not (
        _CANDIDATE_COORDINATION_RE.search(between_targets)
        or _CANDIDATE_CLAUSE_SEPARATOR_RE.search(between_targets)
    ):
        return False

    covered = [False] * len(source)
    for start, end in (*ordered_targets, *qualifier_ranges, *trusted_ranges):
        for index in range(start, end):
            covered[index] = True
    remainder = "".join(
        character
        for index, character in enumerate(source)
        if not covered[index]
    )
    return _CANDIDATE_SCAFFOLD_RE.fullmatch(remainder) is not None


def is_elliptical_current_clause_target(
    question: object,
    *,
    target_range: tuple[int, int],
    qualifier_ranges: tuple[tuple[int, int], ...],
) -> bool:
    """Whether a target is a noun omitted from an earlier current clause.

    The function proves only a source-surface fact.  It does not decide that a
    qualifier *means* the same business subject as the later noun; that remains
    the source-anchored analyzer's constrained candidate.  It does prove the
    narrow precondition for lexical normalisation:

    * the target is inside one explicitly separated current clause;
    * every supplied qualifier is a literal, non-overlapping span in an
      *earlier* current clause; and
    * removing the target leaves only a generic modal/action/interrogative
      shell such as ``需要提供哪些``.

    Thus ``报销提交时限是多久？需要提供哪些凭证？`` qualifies, while a
    same-clause qualifier or an arbitrary punctuated pair does not.  No
    company vocabulary, document text, aliases or historical turns enter this
    decision.
    """

    source = str(question or "")
    if not source:
        return False
    try:
        target_start, target_end = target_range
    except (TypeError, ValueError):
        return False
    if not (
        isinstance(target_start, int)
        and isinstance(target_end, int)
        and 0 <= target_start < target_end <= len(source)
    ):
        return False
    if not qualifier_ranges:
        return False

    clause_start = max(
        (index + 1 for index, value in enumerate(source[:target_start]) if value in "?？；;\n"),
        default=0,
    )
    clause_end = next(
        (
            index
            for index, value in enumerate(source[target_end:], start=target_end)
            if value in "?？；;\n"
        ),
        len(source),
    )
    if not (clause_start <= target_start and target_end <= clause_end):
        return False

    for value in qualifier_ranges:
        try:
            qualifier_start, qualifier_end = value
        except (TypeError, ValueError):
            return False
        if not (
            isinstance(qualifier_start, int)
            and isinstance(qualifier_end, int)
            and 0 <= qualifier_start < qualifier_end <= len(source)
        ):
            return False
        # An omitted-head reference must be a source span in an earlier
        # clause.  Same-clause qualifiers remain part of that requirement's
        # normal baseline wording and may not trigger cross-clause rewrite.
        if qualifier_end > clause_start:
            return False

    shell = (source[clause_start:target_start] + source[target_end:clause_end]).strip(
        " \t\r\n，,。！？!?：:"
    )
    return _ELLIPTICAL_ACTION_TARGET_SHELL_RE.fullmatch(shell) is not None


def _leading_qualifier(value: str) -> tuple[str, QualifierKind, str] | None:
    """Extract one syntactically explicit leading qualifier from *value*.

    The fallback is intentionally ``None`` rather than an arbitrary entity:
    an unmarked prefix is safer as part of the answer target.  This is what
    prevents place, duration, phase and threshold wording from being silently
    treated as a classification subject.
    """

    source = _clean_surface_text(value)
    if not source:
        return None

    if (kind := _classify_whole_qualifier(source)) is not None:
        return source, kind, ""

    condition_match = _LEADING_CONDITION_RE.match(source)
    if condition_match is not None:
        condition = _clean_surface_text(condition_match.group("condition"))
        rest = _clean_surface_text(condition_match.group("rest"))
        if condition:
            return condition, "condition", rest

    # Pick the longest valid entity prefix.  This preserves modifiers such as
    # ``普通`` in ``普通员工`` and compact identifiers in ``客户A`` while
    # retaining a following event/object phrase as answer context.
    candidates: list[tuple[str, str]] = []
    for split_at in range(2, len(source)):
        candidate = _clean_surface_text(source[:split_at])
        if _classify_whole_qualifier(candidate) != "entity":
            continue
        rest = _clean_surface_text(source[split_at:])
        if rest:
            candidates.append((candidate, rest))
    if candidates:
        qualifier, rest = max(candidates, key=lambda item: len(item[0]))
        return qualifier, "entity", rest

    condition_noun_match = _LEADING_CONDITION_NOUN_RE.match(source)
    if condition_noun_match is not None:
        condition = _clean_surface_text(condition_noun_match.group("condition"))
        rest = _clean_surface_text(condition_noun_match.group("rest"))
        if condition and rest:
            return condition, "condition", rest
    return None


def _following_prepositional_condition(value: str) -> tuple[str, str] | None:
    """Split an explicit prepositional condition after an entity/scope.

    A preposition itself is a grammatical boundary, so this helper never has
    to guess that a bare word is a city, project or stage.  It first preserves
    a structurally ended condition (``一线城市``), then accepts a bounded
    two-CJK-character phrase such as ``北京``.  The latter is not a location
    dictionary: it is merely the shortest non-empty noun phrase following an
    explicit condition preposition.
    """

    source = _clean_surface_text(value)
    match = _PREPOSITIONAL_CONDITION_PREFIX_RE.fullmatch(source)
    if match is None:
        return None
    body = _clean_surface_text(match.group("body"))
    if len(body) < 2:
        return None
    numeric = _LEADING_CONDITION_RE.match(body)
    if numeric is not None:
        condition = _clean_surface_text(numeric.group("condition"))
        rest = _clean_surface_text(numeric.group("rest"))
        if condition:
            return condition, rest
    named = _LEADING_CONDITION_NOUN_RE.match(body)
    if named is not None:
        condition = _clean_surface_text(named.group("condition"))
        rest = _clean_surface_text(named.group("rest"))
        if condition:
            return condition, rest
    if re.fullmatch(r"[\u3400-\u9fff]{2,}", body):
        return body, ""
    # A bare CJK condition immediately followed by a target has no lexical
    # separator.  The syntactic preposition allows a conservative two-character
    # condition boundary (``在北京住宿标准``); longer/unknown forms remain part
    # of the target rather than being over-segmented.
    if re.fullmatch(r"[\u3400-\u9fff]{4,}", body):
        return body[:2], _clean_surface_text(body[2:])
    return None


def _qualifier_sequence(
    source: str,
) -> tuple[tuple[tuple[str, QualifierKind], ...], str]:
    """Extract a leading qualifier plus any explicit following condition."""

    first = _leading_qualifier(source)
    if first is None:
        return (), _clean_surface_text(source)
    text, kind, residual = first
    qualifiers: list[tuple[str, QualifierKind]] = [(text, kind)]
    following = _following_prepositional_condition(residual)
    if following is not None:
        condition, residual = following
        qualifiers.append((condition, "condition"))
    return tuple(qualifiers), _clean_surface_text(residual)


def _frame_context_terms(
    qualifiers: tuple[tuple[str, QualifierKind], ...],
    residual: str,
    *,
    include_residual: bool = False,
) -> tuple[str, ...]:
    """Keep condition applicability and residual scene text explicit."""

    context: list[str] = []
    for text, kind in qualifiers:
        if kind == "condition":
            _append_context(context, text)
    if include_residual:
        _append_context(context, residual)
    return tuple(context)


def _frame_qualifier_spans(
    question: str,
    qualifiers: tuple[tuple[str, QualifierKind], ...],
) -> tuple[QualifierSpan, ...]:
    """Materialize deterministic spans without adding semantic facts."""

    return tuple(
        _make_qualifier_span(question, text, kind)
        for text, kind in qualifiers
    )


def _append_context(
    values: list[str],
    value: str,
) -> None:
    normalized = _clean_surface_text(value)
    if normalized and normalized not in values:
        values.append(normalized)


def _make_qualifier_span(
    question: str,
    text: str,
    kind: QualifierKind,
) -> QualifierSpan:
    start = question.casefold().find(text.casefold())
    if start < 0:
        start = 0
    return QualifierSpan(text=text, kind=kind, start=start, end=start + len(text))


def _strip_question_form(question: str) -> tuple[str, str | None]:
    """Return source body and optional noun retained after a question shell."""

    condition_requirement = _CONDITION_REQUIREMENT_QUESTION_RE.fullmatch(question)
    if condition_requirement is not None:
        return (
            _clean_surface_text(condition_requirement.group("prefix")),
            _clean_surface_text(condition_requirement.group("noun")),
        )
    action_with_noun = _ACTION_QUESTION_SHELL_WITH_NOUN_RE.fullmatch(question)
    if action_with_noun is not None:
        prefix = _clean_surface_text(action_with_noun.group("prefix"))
        if _SHELL_ACTION_PREFIX_RE.fullmatch(prefix):
            prefix = ""
        return prefix, _clean_surface_text(action_with_noun.group("noun"))
    action_without_noun = _ACTION_QUESTION_SHELL_WITHOUT_NOUN_RE.fullmatch(question)
    if action_without_noun is not None:
        prefix = _clean_surface_text(action_without_noun.group("prefix"))
        if _SHELL_ACTION_PREFIX_RE.fullmatch(prefix):
            prefix = ""
        return prefix, None
    shell = _QUESTION_SHELL_RE.fullmatch(question)
    if shell is not None:
        prefix = _clean_surface_text(shell.group("prefix"))
        if _SHELL_ACTION_PREFIX_RE.fullmatch(prefix):
            prefix = ""
        return prefix, _clean_surface_text(shell.group("noun"))
    return _clean_surface_text(_QUESTION_TAIL_RE.sub("", question)), None


def parse_query_surface_frame(question: object) -> QuerySurfaceFrame | None:
    """Parse one current-turn question into a conservative pure surface frame.

    ``None`` is returned for blank/non-string input.  A non-empty input always
    retains a literal answer target even when no qualifier can be recognized;
    that fail-closed behavior leaves later planning free to perform ordinary
    direct retrieval without manufacturing a bridge.
    """

    if not isinstance(question, str):
        return None
    normalized = normalize_current_question(question)
    if not normalized:
        return None
    operator = _surface_operator(normalized)

    relation = _RELATION_RE.fullmatch(normalized)
    if relation is not None:
        left = _clean_surface_text(relation.group("left"))
        right = _clean_surface_text(_QUESTION_TAIL_RE.sub("", relation.group("right")))
        # ``对应什么职级`` is a grammar shell just like ``有什么补贴``: the
        # interrogative introduces the requested noun rather than being part
        # of the target itself.
        right = re.sub(r"^(?:什么|哪些|哪类|何种)\s*", "", right).strip()
        qualifier_values, residual = _qualifier_sequence(left)
        qualifiers = _frame_qualifier_spans(normalized, qualifier_values)
        context_terms = _frame_context_terms(
            qualifier_values,
            residual,
            include_residual=True,
        )
        return QuerySurfaceFrame(
            question=normalized,
            answer_target=right or _clean_surface_text(relation.group("right")),
            context_terms=context_terms,
            qualifiers=qualifiers,
            question_operator="relation",
        )

    body, shell_noun = _strip_question_form(normalized)
    if not body and not shell_noun:
        return None

    # A possessive separator makes the target boundary explicit.  The text to
    # its left can itself be compound (``条件 + 场景``); peel only a syntactic
    # leading qualifier and retain the rest as context next to the answer
    # head.  This yields ``偏远地区出差的住宿标准`` -> target ``出差住宿标准``
    # rather than throwing away ``出差`` with the condition.
    if "的" in body:
        left, right = body.rsplit("的", 1)
        left = _clean_surface_text(left)
        right = _clean_surface_text(right)
        qualifier_values, residual = _qualifier_sequence(left)
        if qualifier_values:
            target = _clean_surface_text(f"{residual}{right}") or right
            return QuerySurfaceFrame(
                question=normalized,
                answer_target=target,
                context_terms=_frame_context_terms(
                    qualifier_values,
                    residual,
                    include_residual=bool(right),
                ),
                qualifiers=_frame_qualifier_spans(normalized, qualifier_values),
                question_operator=operator,
            )

    # No possessive separator is still common in Chinese compact questions.
    # When an interrogative shell supplied its final noun, split the leading
    # phrase first so a condition + event keeps the event as the answer head
    # (``偏远地区出差有什么补贴`` -> ``出差补贴``).
    source_prefix = body
    qualifier_values, residual = _qualifier_sequence(source_prefix)
    if qualifier_values:
        target = _clean_surface_text(f"{residual}{shell_noun or ''}")
        # A whole qualifier plus no shell noun may itself be a valid direct
        # target (for example an explicit scope label).  Do not erase it.
        if not target:
            target = shell_noun or source_prefix
        return QuerySurfaceFrame(
            question=normalized,
            answer_target=target,
            context_terms=_frame_context_terms(
                qualifier_values,
                residual,
                include_residual=bool(shell_noun),
            ),
            qualifiers=_frame_qualifier_spans(normalized, qualifier_values),
            question_operator=operator,
        )

    literal_target = _clean_surface_text(f"{body}{shell_noun or ''}")
    return QuerySurfaceFrame(
        question=normalized,
        answer_target=literal_target or normalized,
        context_terms=(),
        qualifiers=(),
        question_operator=operator,
    )
