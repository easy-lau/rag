"""Resolve bounded multi-hop joins without business-specific vocabulary.

The query planner describes an implicit bridge (for example ``employee ->
grade``) but deliberately does not guess the target value.  This module derives
that value only from source text, then uses it to connect answer clauses and to
build a bounded second-hop retrieval query.

The implementation is intentionally conservative:

* a bridge value must occur in the same sentence or table row as the subject;
* an answer clause must contain both the resolved value and an independent
  answer-target term;
* a bridge-only source cannot become answer evidence merely because the
  document title repeats the user's topic.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

from core.query_constraints import (
    QueryConstraints,
    evaluate_candidate_constraints,
    extract_document_constraint_identity,
    extract_query_constraints,
)
from core.rag_v2.contracts import AnswerRequirementV2, EvidenceItem


_BRIDGE_SUBJECT_RE = re.compile(
    r"^(?:确认|验证|判断|核对)?\s*(?P<subject>.+?)"
    r"(?:对应|属于|归属|映射|适用|决定)"
)
_BRIDGE_IDENTIFIER_RE = re.compile(
    r"(?:[A-Za-z]+(?:\d+(?:\.\d+)*)?(?:级|类|档|型|版|组|岗|层|序列|阶段|角色)|"
    r"[A-Za-z]+\d+(?:\.\d+)*(?:级|类|档|型|版|组|岗|层|序列|阶段|角色)?|"
    r"\d+(?:\.\d+)+(?:版|级|类|档|型|组|层|阶段)?)",
    re.IGNORECASE,
)
_FORWARD_RELATION_TEMPLATE = (
    r"(?P<subject>{subject})"
    # Never let an entity match through a suffix (``普通员工家属``) or
    # jump to a relation in a later sentence. Only structural separators may
    # occur between the exact subject and relation operator.
    r"(?P<link>[\s:：,，、()\uff08\uff09\[\]【】-]{{0,12}})"
    r"(?P<relation>(?:所)?对应(?:到|为)?|属于|归属于|归属|"
    r"映射(?:到|为)?|认定为|划分为|列为|等同于|适用)\s*"
    r"(?P<value>[A-Za-z0-9_.+/\u3400-\u9fff-]{{1,32}})"
)
_REVERSE_RELATION_TEMPLATE = (
    r"(?P<value>[A-Za-z0-9_.+/\u3400-\u9fff-]{{1,32}}?)\s*"
    r"(?P<relation>适用于|适用人员(?:包括|含|为)?|包括|包含|覆盖|"
    r"对应人员(?:包括|含|为)?)"
    r"(?P<link>[\s:：,，、()\uff08\uff09\[\]【】-]{{0,12}})"
    r"(?P<subject>{subject})"
)
_CORRECTED_RELATION_RE = re.compile(
    r"[,，](?:(?:但|但是|然而|不过)\s*)?"
    r"(?:现行|现在|当前|现|修订后|更正后|正式规定)[：:,，\s]*"
    r"(?:(?:所)?对应(?:到|为)?|属于|归属于|归属|映射(?:到|为)?|"
    r"认定为|划分为|列为|等同于|适用)\s*"
    r"(?P<value>[A-Za-z0-9_.+/\u3400-\u9fff-]{1,32})",
    re.IGNORECASE,
)
_STRUCTURED_SPLIT_RE = re.compile(r"\s*(?:\||\t|->|=>|→|=)\s*")
_SEGMENT_SPLIT_RE = re.compile(r"[\n。；;！!？?]+")
_TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9_.+/-]{2,}|[\u3400-\u9fff]{2,}")
_LEADING_TARGET_FILLER_RE = re.compile(
    r"^(?:请问|那么|那|呢|最终|具体|分别|各自)\s*",
    re.IGNORECASE,
)
_LEADING_TARGET_CONNECTOR_RE = re.compile(
    r"^(?:所对应的|对应的|适用的|享受的|的|用于)\s*",
    re.IGNORECASE,
)
_TRAILING_QUESTION_RE = re.compile(
    r"(?:如何(?:配置|设置|处理|操作|解决)|怎么(?:配置|设置|处理|操作|解决)|"
    r"怎样(?:配置|设置|处理|操作|解决)|是什么|是多少|为多少|是多久|要多久|多久|何时|什么时候|"
    r"在哪里|哪里|有哪些|哪几个|哪些|多少|什么|如何|怎么|怎样|呢|吗|么)"
    r"\s*[?？]*$",
    re.IGNORECASE,
)
_LEADING_INTERROGATIVE_ACTION_RE = re.compile(
    r"^(?:(?:需要|需|应当|应|必须|须)\s*)?"
    r"(?:提供|提交|准备|包含|包括|采用|使用|填写|选择|配置|设置)\s*"
    r"(?:哪些?|什么|多少|怎样的|何种)\s*",
    re.IGNORECASE,
)
_TARGET_CONDITION_QUESTION_RE = re.compile(
    r"(?:(?:需要|需|应当|应|必须|须)\s*)?"
    r"(?:满足|符合|具备|达到)\s*(?:哪些|什么|怎样的|何种)?\s*"
    r"(?P<target>条件|要求|资格)$",
    re.IGNORECASE,
)
_TARGET_ACTION_WORD_RE = re.compile(
    r"(?:查询|查看|确认|核验|提供|提交|准备|填写|选择|配置|设置|获取|取得)",
    re.IGNORECASE,
)
_ACTION_QUALIFIED_TARGET_SUFFIX_RE = re.compile(
    r"(?:时限|期限|时长|时间|日期|周期|次数|数量|金额|额度|上限|下限)$",
    re.IGNORECASE,
)
_LEADING_ANSWER_ACTION_RE = re.compile(
    r"^(?:(?:请问|请(?=(?:如何|怎么|怎样|查询|确认|验证|判断|核对|"
    r"回答|取得|获取|确定|设置|配置|解决|处理|查看|核验)))\s*)?"
    r"(?:(?:如何|怎么|怎样)\s*)?"
    r"(?:设置|配置|解决|处理|查看|查询|获取|确认|验证|判断|核对|"
    r"回答|取得|确定|核验)\s*"
    # After bridge-subject removal a real question verb is followed by a
    # grammatical connector (``查询 的餐补``). A capability name such as
    # ``查询权限`` has no connector and must retain the verb as target text.
    r"(?=(?:所对应的|对应的|适用的|享受的|的))",
    re.IGNORECASE,
)
_LEADING_POLICY_LOOKUP_ACTION_RE = re.compile(
    r"^(?:请问|请)?\s*(?:查询|查看|确认|了解|核验)\s*"
    r"(?=[A-Za-z0-9\u3400-\u9fff_.+/-]{2,}"
    r"(?:标准|规定|制度|政策|办法|规范|要求|流程|规则|方案|"
    r"金额|额度|上限|下限|时限|期限|数量|状态)$)",
    re.IGNORECASE,
)
_LEADING_GROUNDED_WRITING_RE = re.compile(
    r"^(?:请\s*)?(?:根据|依据|参考|结合)"
    r"[^。；;！？?]{1,48}?"
    r"(?:起草|撰写|编写|生成|写)\s*"
    r"(?:一份|一段|一个|一篇)?\s*",
    re.IGNORECASE,
)
_TARGET_SHAPE_SUFFIX_RE = re.compile(
    r"(?:金额|数额|数值|数量|个数|标准|规定|制度|政策|办法|规范|要求|"
    r"信息|内容|资料|结果)$",
    re.IGNORECASE,
)
_TARGET_CONTINUATION_RE = re.compile(
    r"^(?:金额|数额|数值|数量|个数|标准|规定|制度|政策|办法|规范|要求|"
    r"信息|内容|资料|结果|管理|指南|手册|说明|为|是|按|可|应|需|须|"
    r"不|上限|下限)",
    re.IGNORECASE,
)
_GENERIC_TARGET_TERMS = frozenset({
    "标准",
    "规定",
    "制度",
    "政策",
    "办法",
    "规范",
    "要求",
    "信息",
    "内容",
    "资料",
    "结果",
    "对应",
    "适用",
})
_GENERIC_BRIDGE_VALUES = frozenset({
    "人员",
    "员工",
    "对象",
    "主体",
    "职级",
    "分类",
    "等级",
    "级别",
    "类别",
    "类型",
    "阶段",
    "角色",
    "标准",
    "要求",
})
_MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_RESULT_SIGNAL_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:%|元|天|小时|分钟|次|个|人|公里|gb|mb)?|"
    r"不超过|不少于|上限|下限|必须|应当|应|须|需|不得|禁止|允许|支持|"
    r"可以|仅限|采用|执行|暂停|拒绝|通过|开启|关闭|为|是)",
    re.IGNORECASE,
)
_INACTIVE_ASSERTION_RE = re.compile(
    r"(?:已(?:经)?(?:废止|作废|失效|取消|删除|下线)|"
    r"(?:废止|作废|失效|无效|取消|删除|下线)(?:的|本|该)?(?:规定|条款|规则|版本)?|"
    r"不再(?:适用|执行|生效)|停止执行|尚未生效|未生效)",
    re.IGNORECASE,
)
_UNCERTAIN_ASSERTION_RE = re.compile(
    r"(?:待(?:确认|核实|验证|审批)|尚待(?:确认|核实)|"
    r"可能(?:有误|不准确|错误)?|疑似|或许|不确定|"
    r"说法不准确|信息有误|内容有误|暂定|草案|示例|假设)",
    re.IGNORECASE,
)
_STATUS_RESET_RE = re.compile(
    r"(?:但|但是|然而|不过|现行|现在|当前|修订后|更正后|正式规定)[：:,，\s]*$",
    re.IGNORECASE,
)
_REFERENCE_ONLY_RE = re.compile(
    r"(?:参见|详见|查阅|参考|见(?:第\s*[A-Za-z0-9一二三四五六七八九十百.]+\s*[章节条款]?|附件)|"
    r"按[^，,。；;]{0,30}(?:版|版本|文件|制度|规定|办法|附件|条款)"
    r"[^，,。；;]{0,8}执行|以[^，,。；;]{1,30}为准|"
    r"另行(?:规定|通知|说明|发布|制定))",
    re.IGNORECASE,
)
_SCALAR_RESULT_RE = re.compile(
    r"(?:[<>≤≥]=?\s*)?\d+(?:\.\d+)?\s*"
    r"(?:%|元|万元|天|小时|分钟|秒|次|个|人|公里|米|kg|g|gb|mb|tb|"
    r"年|月|日|位|席|台|套|份|条|级|档|类)|"
    r"(?:不超过|不少于|大于|小于|至多|至少|上限|下限)\s*\d+(?:\.\d+)?",
    re.IGNORECASE,
)
_BARE_VALUE_PREDICATE_RE = re.compile(
    r"(?:为|是|等于|共|合计|数量(?:为)?|金额(?:为)?|值(?:为)?)\s*"
    r"\d+(?:\.\d+)?(?:\s|$|[，,。；;])",
    re.IGNORECASE,
)
_NORMATIVE_RESULT_RE = re.compile(
    r"(?:必须|应当|应|须|需|不得|禁止|允许|支持|可以|仅限|可)"
    r"[A-Za-z0-9\u3400-\u9fff]{2,}",
    re.IGNORECASE,
)
_PROCEDURE_RESULT_RE = re.compile(
    r"(?:^|[；;，,:：])\s*(?:第?[一二三四五六七八九十0-9]+[步、.)）]|"
    r"先.+再|(?:[A-Za-z0-9\u3400-\u9fff]{0,12})?"
    r"(?:提交|填写|选择|点击|进入|打开|调用|执行命令|配置参数)|"
    r"[^，,。；;]{1,40}(?:后|然后|再|随后|最后|依次)"
    r"[^，,。；;]{1,40})",
    re.IGNORECASE,
)
_CATEGORICAL_PREDICATE_RE = re.compile(
    r"(?:为|是|等于|采用|使用|选择|设置为|配置为|结果为|包括|包含|[：:])\s*"
    r"(?P<value>[A-Za-z0-9_.+/\u3400-\u9fff-]{1,80})",
    re.IGNORECASE,
)
_NON_RESULT_TEXT_RE = re.compile(
    r"^(?:备注|说明|依据|来源|引用|附件|条款|索引|序号|编号|"
    r"标准|规定|制度|政策|办法|规范|要求|信息|内容|资料|结果|"
    r"待定|暂无|无)$",
    re.IGNORECASE,
)
_NON_RESULT_HEADER_RE = re.compile(
    r"(?:备注|说明|依据|来源|引用|附件|条款|索引|序号|编号|更新时间|状态)",
    re.IGNORECASE,
)
_NON_BRIDGE_VALUE_RE = re.compile(
    r"(?:标准|制度|办法|政策|规范|流程|金额|额度|补贴|补助|费用|"
    r"权限|措施|内容|资料|条款|附件|文档|文件|手册|指南)$",
    re.IGNORECASE,
)
_UNIVERSAL_APPLICABILITY_RE = re.compile(
    r"(?:(?:所有|全部|全体|各(?:个|类|级)?)\s*"
    r"(?:职级|等级|级别|类别|类型|档位|人员|员工|用户|客户|供应商|"
    r"对象|主体|地区|城市|项目|版本)\s*"
    r"(?:统一|相同|一致|均|都|适用)|"
    r"(?:统一|相同|一致)\s*(?:标准|规定|规则|要求|适用))",
    re.IGNORECASE,
)
_TAXONOMY_HEADER_TERMS = (
    "分类",
    "等级",
    "级别",
    "类别",
    "类型",
    "职级",
    "岗级",
    "角色",
    "版本",
    "档位",
    "阶段",
    "层级",
    "序列",
)
_NON_MAPPING_HEADER_RE = re.compile(
    r"(?:时长|日期|时间|天数|次数|金额|数量|额度|上限|下限|"
    r"审批|审核|流程|标准|补贴|费用|权限|措施|处置|交通|住宿)",
    re.IGNORECASE,
)
_NEGATIVE_PREFIX_RE = re.compile(
    r"(?:并非|不是|不为|不等于|不属于|不归属|不适用于?|"
    r"未归属|未划分为|未列为|排除|非)\s*$",
    re.IGNORECASE,
)
_NEGATED_RELATION_RE = re.compile(
    r"(?:不|未|无须|无需|并非|不是|非)\s*"
    r"(?:所)?(?:对应|属于|归属|映射|认定为|划分为|列为|"
    r"等同于|适用于?|包括|包含|覆盖)",
    re.IGNORECASE,
)
_NEGATED_VALUE_RE = re.compile(
    r"^(?:并非|不是|不为|不等于|不属于|不适用于?|"
    r"排除|除|非)",
    re.IGNORECASE,
)
_LIST_ITEM_SPLIT_RE = re.compile(
    r"\s*(?:、|,|，|/|／|；|;|以及|及|与|和)\s*"
)
_EXACT_CODE_SUFFIXES = "级类档型版组岗层序列阶段角色"


@dataclass(frozen=True)
class ResolvedBridgeFact:
    requirement_id: str
    subject: str
    value: str
    source_chunk_id: str
    source_doc_id: str
    source_kb_id: str
    scope_products: tuple[str, ...] = ()
    scope_versions: tuple[str, ...] = ()
    scope_projects: tuple[str, ...] = ()


@dataclass(frozen=True)
class BridgeFactConflict:
    """Different values asserted for one subject in one applicability scope."""

    requirement_id: str
    subject: str
    source_kb_id: str
    source_doc_id: str
    scope_products: tuple[str, ...]
    scope_versions: tuple[str, ...]
    scope_projects: tuple[str, ...]
    values: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class _ClaimUnit:
    """One independently assertable sentence or table row."""

    semantic_text: str
    result_text: str
    structured: bool = False
    section_heading: str = ""
    header_cells: tuple[str, ...] = ()
    row_cells: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClaimAssertion:
    status: str
    result_kind: str
    normalized_result: str = ""
    claim_key: str = ""

    @property
    def supports_answer(self) -> bool:
        return self.status == "active" and self.result_kind in {
            "scalar",
            "categorical",
            "normative",
            "procedure",
        }


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _compact_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _assertion_status(
    text: Any,
    *,
    start: int | None = None,
    end: int | None = None,
) -> str:
    """Classify whether a local claim is active enough to support an answer."""

    source = re.sub(r"\s+", " ", str(text or "")).strip()
    if start is not None and end is not None:
        left = source[max(0, start - 28):start]
        # A correction boundary makes the text after it authoritative for the
        # current match (``旧说法有误，现普通员工对应D级``).
        reset_matches = list(re.finditer(
            r"(?:但|但是|然而|不过|现行|现在|当前|修订后|更正后|正式规定)[：:,，\s]*",
            left,
            re.IGNORECASE,
        ))
        if reset_matches:
            left = left[reset_matches[-1].end():]
        right = source[end:min(len(source), end + 28)]
        boundary = re.search(
            r"(?:，|,)(?:(?:但|但是|然而|不过)\s*)?"
            r"(?:现行|现在|当前|现|修订后|更正后|正式规定)",
            right,
            re.IGNORECASE,
        )
        superseded = boundary is not None
        if boundary is not None:
            right = right[:boundary.start()]
        source = f"{left}{source[start:end]}{right}"
        if superseded:
            return "inactive"

    # Explicit negation of a status marker is not itself an invalidation.
    scrubbed = re.sub(
        r"(?:并未|没有|未曾|不是|并非)(?:被)?(?:废止|作废|失效|取消|删除|下线|草案|示例)",
        "",
        source,
        flags=re.IGNORECASE,
    )
    if _INACTIVE_ASSERTION_RE.search(scrubbed):
        return "inactive"
    if _UNCERTAIN_ASSERTION_RE.search(scrubbed):
        return "uncertain"
    return "active"


def _occurrence_is_excluded(text: str, start: int, end: int) -> bool:
    prefix = text[max(0, start - 16):start]
    suffix = text[end:end + 12]
    if _NEGATIVE_PREFIX_RE.search(prefix):
        return True
    if re.match(r"(?:除外|被排除|不适用)", suffix):
        return True
    # ``除X外`` / ``除...X之外`` is an exclusion, not a positive mention.
    exclusion_prefix = prefix.rsplit("除", 1)[-1] if "除" in prefix else None
    return bool(
        exclusion_prefix is not None
        and not re.search(r"[,，。；;]", exclusion_prefix)
        and re.match(r"[^,，。；;]{0,16}(?:外|之外|以外)", suffix)
    )


def _term_is_excluded(text: str, term: str) -> bool:
    """Return whether every occurrence of ``term`` is explicitly excluded.

    This is deliberately local: ``X属于A，但不属于B`` still provides the
    positive A fact.  Only negation immediately governing the occurrence, or an
    enclosing ``除...外`` construction, invalidates that occurrence.
    """

    normalized = _compact_text(text)
    needle = _compact_text(term)
    if not normalized or not needle:
        return True
    found = False
    for match in re.finditer(re.escape(needle), normalized, re.IGNORECASE):
        found = True
        if _occurrence_is_excluded(normalized, match.start(), match.end()):
            continue
        return False
    return found


def _has_positive_term_occurrence(text: Any, term: str) -> bool:
    normalized = _compact_text(text)
    needle = _compact_text(term)
    return bool(needle and needle in normalized and not _term_is_excluded(normalized, needle))


def _subject_item_matches(cell: str, subject: str) -> bool:
    """Match a table subject as an exact list item, never a substring."""

    normalized_subject = _compact_text(subject)
    if not normalized_subject:
        return False
    for raw_item in _LIST_ITEM_SPLIT_RE.split(str(cell or "")):
        item = _compact_text(raw_item).strip("：:|[]【】()（）<>")
        if item == normalized_subject and not _term_is_excluded(raw_item, subject):
            return True
    return False


def _subject_boundary_is_exact(
    claim: str,
    match: re.Match[str],
    *,
    reverse: bool,
) -> bool:
    """Reject a relation whose subject is embedded in a larger entity."""

    start, end = match.span("subject")
    if reverse:
        suffix = claim[end:end + 1]
        if suffix and re.match(r"[A-Za-z0-9_\u3400-\u9fff]", suffix):
            return False
        link = match.group("link")
        if _NEGATIVE_PREFIX_RE.search(link):
            return False
    else:
        prefix = claim[max(0, start - 1):start]
        if prefix and re.match(r"[A-Za-z0-9_\u3400-\u9fff]", prefix):
            structural_prefix = claim[max(0, start - 12):start]
            if not re.search(
                r"(?:现行|现在|当前|现|修订后|更正后|正式规定)$",
                structural_prefix,
                re.IGNORECASE,
            ):
                return False
    return not _term_is_excluded(claim, match.group("subject"))


def _relation_match_is_positive(
    claim: str,
    match: re.Match[str],
    *,
    reverse: bool,
) -> bool:
    if not _subject_boundary_is_exact(claim, match, reverse=reverse):
        return False
    relation_start = match.start("relation")
    relation_prefix = claim[max(match.start(), relation_start - 4):relation_start]
    if _NEGATED_RELATION_RE.search(
        relation_prefix + match.group("relation")
    ):
        return False
    value = match.group("value")
    if _NEGATED_VALUE_RE.search(_compact_text(value)):
        return False
    return not _term_is_excluded(claim, value)


def _candidate_content(candidate: Mapping[str, Any] | EvidenceItem) -> str:
    if isinstance(candidate, EvidenceItem):
        return candidate.content
    return str(candidate.get("content") or "")


def _candidate_id(candidate: Mapping[str, Any] | EvidenceItem) -> str:
    if isinstance(candidate, EvidenceItem):
        return candidate.chunk_id
    return str(candidate.get("chunk_id") or candidate.get("id") or "").strip()


def _candidate_mapping(
    candidate: Mapping[str, Any] | EvidenceItem,
) -> dict[str, Any]:
    return (
        candidate.to_dict()
        if isinstance(candidate, EvidenceItem)
        else dict(candidate)
    )


def _candidate_scope(
    candidate: Mapping[str, Any] | EvidenceItem,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    raw = _candidate_mapping(candidate)
    identity = extract_document_constraint_identity(raw)
    products = identity.canonical_products or identity.products
    return (
        str(raw.get("kb_id") or "").strip(),
        str(raw.get("doc_id") or "").strip(),
        tuple(sorted(
            {str(value).strip().casefold() for value in products if str(value).strip()}
        )),
        tuple(sorted(
            {str(value).strip().casefold() for value in identity.versions if str(value).strip()}
        )),
        tuple(sorted(
            {str(value).strip().casefold() for value in identity.projects if str(value).strip()}
        )),
    )


def _requirement_scope_allows_candidate(
    requirement: AnswerRequirementV2,
    candidate: Mapping[str, Any] | EvidenceItem,
) -> bool:
    if not requirement.scope_product and not requirement.scope_version:
        return True
    constraints = QueryConstraints(
        product=requirement.scope_product,
        version=requirement.scope_version,
        explicit_version=requirement.scope_explicit_version,
        matched_text=" ".join(
            value
            for value in (
                str(requirement.scope_product or ""),
                str(requirement.scope_version or ""),
            )
            if value
        ),
        extraction_reason="requirement_local_scope",
    )
    return evaluate_candidate_constraints(
        constraints,
        _candidate_mapping(candidate),
    ).status not in {"mismatch", "unknown"}


def extract_bridge_subject(description: Any) -> str | None:
    """Return the source entity named by a bridge requirement."""

    normalized = re.sub(r"\s+", " ", str(description or "")).strip()
    match = _BRIDGE_SUBJECT_RE.search(normalized)
    if match is None:
        return None
    subject = match.group("subject").strip(" ：:，,。；;（）()[]【】")
    # Product/version text is an applicability scope, not part of the entity
    # being classified.  ``云枢8.6普通员工`` must therefore resolve the
    # same bridge subject as the source clause ``普通员工对应D级``.  Only
    # remove a recognized explicit scope when a non-empty business subject
    # remains; a product that is itself the queried entity stays intact.
    constraints = extract_query_constraints(subject)
    matched_scope = str(constraints.matched_text or "").strip()
    if constraints.has_scope_constraint and matched_scope:
        reduced = re.sub(
            re.escape(matched_scope),
            " ",
            subject,
            count=1,
            flags=re.IGNORECASE,
        ).strip(" ：:，,。；;（）()[]【】-_")
        if len(reduced) >= 2:
            subject = reduced
    return subject if 2 <= len(subject) <= 64 else None


def bridge_subject_for_requirement(
    requirement: AnswerRequirementV2,
) -> str | None:
    """Return the canonical bridge subject carried by the query plan.

    Requirement descriptions are retrieval/presentation text and route models
    may paraphrase them.  New plans therefore store the subject explicitly;
    parsing the description remains only a compatibility fallback.
    """

    if requirement.role != "bridge":
        return None
    subject = str(requirement.bridge_subject or "").strip()
    return subject or extract_bridge_subject(requirement.description)


def bridge_dependency_ids_for_answer(
    answer: AnswerRequirementV2,
    requirements: Sequence[AnswerRequirementV2],
) -> tuple[str, ...]:
    """Resolve machine-readable answer-to-bridge edges, failing closed.

    Explicit plan edges are authoritative, including an explicit empty tuple.
    Older plans may infer an edge only from a positive, exact subject occurrence
    in the answer wording.  There is deliberately no "all bridges" fallback:
    that behavior attached independent siblings in mixed questions to an
    unrelated classification mapping.
    """

    bridge_requirements = tuple(
        item for item in requirements if item.role == "bridge"
    )
    if not bridge_requirements:
        return ()
    bridge_ids = {item.id for item in bridge_requirements}
    if answer.depends_on_requirement_ids is not None:
        return tuple(
            dependency_id
            for dependency_id in answer.depends_on_requirement_ids
            if dependency_id in bridge_ids
        )
    matched = tuple(
        bridge.id
        for bridge in bridge_requirements
        if (
            (subject := bridge_subject_for_requirement(bridge))
            and content_contains_positive_subject(answer.description, subject)
        )
    )
    return matched


def _clean_bridge_value(value: Any, *, subject: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(
        " ：:，,。；;|[]【】()（）<>"
    )
    if (
        not cleaned
        or len(cleaned) > 32
        or _NEGATED_VALUE_RE.search(_compact_text(cleaned))
        or cleaned.endswith(("除外", "之外", "以外"))
        or cleaned.casefold() == subject.casefold()
        or cleaned.casefold() in _GENERIC_BRIDGE_VALUES
        or _NON_BRIDGE_VALUE_RE.search(cleaned)
        or cleaned.isdigit()
        or _MARKDOWN_SEPARATOR_RE.fullmatch(cleaned)
    ):
        return None
    return cleaned


def _value_variants(value: str, *, subject: str) -> tuple[str, ...]:
    identifiers: list[str] = []
    for match in _BRIDGE_IDENTIFIER_RE.finditer(str(value or "")):
        identifier = _clean_bridge_value(match.group(0), subject=subject)
        if identifier is not None:
            identifiers.append(identifier)
    # A stable code is the canonical join key. Keeping both its surrounding
    # prose (``入职阶段P0``) and ``P0`` would manufacture a same-scope
    # conflict even though the source asserted only one mapping.
    if identifiers:
        return tuple(dict.fromkeys(identifiers))[:6]
    cleaned = _clean_bridge_value(value, subject=subject)
    return (cleaned,) if cleaned is not None else ()


def _structured_cells(line: str) -> tuple[str, ...]:
    if "|" not in line and "\t" not in line and not any(
        marker in line for marker in ("->", "=>", "→", "=")
    ):
        return ()
    return tuple(
        cell.strip()
        for cell in _STRUCTURED_SPLIT_RE.split(line.strip().strip("|"))
        if cell.strip()
    )


def _iter_claim_units(content: Any) -> tuple[_ClaimUnit, ...]:
    """Return bounded claims; never combine unrelated prose sentences.

    A Markdown table row inherits its local section heading and column header,
    because those are part of the row schema.  The ingestion format also
    persists a section breadcrumb as a standalone ``【document › section】``
    line immediately before a chunk; that line is part of the chunk body and
    is therefore equivalent to a heading.  External filename/title metadata
    remains intentionally absent: it may retrieve a chunk but cannot assert a
    row.
    """

    lines = str(content or "").splitlines()
    claims: list[_ClaimUnit] = []
    section_heading = ""
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        heading_match = re.match(r"^#{1,6}\s*(.+?)\s*#*$", line)
        if heading_match:
            section_heading = heading_match.group(1).strip()
            index += 1
            continue

        # The chunker emits breadcrumbs instead of repeating a Markdown
        # heading for tables extracted from DOCX/PDF.  Treat only a complete,
        # standalone breadcrumb as structural context.  This deliberately does
        # not consult candidate metadata or filenames, so a retrieved title
        # still cannot manufacture evidence for an unrelated table.
        breadcrumb_match = re.fullmatch(r"【\s*(.+?)\s*】", line)
        if breadcrumb_match:
            section_heading = breadcrumb_match.group(1).strip()
            index += 1
            continue

        cells = _structured_cells(line)
        if cells:
            block: list[tuple[str, tuple[str, ...]]] = []
            while index < len(lines):
                block_line = lines[index].strip()
                block_cells = _structured_cells(block_line)
                if not block_cells:
                    break
                block.append((block_line, block_cells))
                index += 1
            header_cells: tuple[str, ...] = ()
            data_start = 0
            if (
                len(block) >= 2
                and all(
                    _MARKDOWN_SEPARATOR_RE.fullmatch(cell)
                    for cell in block[1][1]
                )
            ):
                header_cells = block[0][1]
                data_start = 2
            for row_text, row_cells in block[data_start:]:
                if all(
                    _MARKDOWN_SEPARATOR_RE.fullmatch(cell)
                    for cell in row_cells
                ):
                    continue
                schema = " ".join(
                    value
                    for value in (section_heading, " ".join(header_cells))
                    if value
                )
                claims.append(_ClaimUnit(
                    semantic_text=" ".join(
                        value for value in (schema, row_text) if value
                    ),
                    result_text=row_text,
                    structured=True,
                    section_heading=section_heading,
                    header_cells=header_cells,
                    row_cells=row_cells,
                ))
            continue

        for segment in _SEGMENT_SPLIT_RE.split(line):
            if segment.strip():
                claims.append(_ClaimUnit(
                    semantic_text=" ".join(
                        value for value in (section_heading, segment.strip())
                        if value
                    ),
                    result_text=segment.strip(),
                    section_heading=section_heading,
                ))
        index += 1
    return tuple(claims)


def _bridge_descriptor_text(description: Any) -> str:
    """Return only the relation side, excluding the downstream answer target."""

    value = re.split(
        r"(?:[（(]\s*)?用于确定",
        str(description or ""),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return re.sub(r"\s+", "", value).casefold()


def _header_descriptor_score(header: str, description: Any) -> int:
    normalized_header = re.sub(r"\s+", "", str(header or "")).casefold()
    descriptor = _bridge_descriptor_text(description)
    if not normalized_header or not descriptor:
        return 0
    if normalized_header in descriptor:
        return len(normalized_header) * 4
    header_terms = {
        match.group(0)
        for match in _TEXT_TOKEN_RE.finditer(normalized_header)
        if match.group(0) not in _GENERIC_BRIDGE_VALUES
    }
    lexical_score = sum(len(term) for term in header_terms if term in descriptor)
    if lexical_score:
        return lexical_score * 2

    # The local planner deliberately asks for a domain-neutral taxonomy
    # (classification/grade/category/stage) rather than guessing that a source
    # calls the dimension ``职级`` or ``角色``.  When the description is
    # clearly that multi-alternative template, accept another taxonomy header.
    # Operational/result columns such as leave duration, approval level or an
    # amount are never a classification merely because the table has two cells.
    descriptor_taxonomy_count = sum(
        term in descriptor for term in _TAXONOMY_HEADER_TERMS
    )
    header_has_taxonomy = any(
        term in normalized_header for term in _TAXONOMY_HEADER_TERMS
    )
    if (
        descriptor_taxonomy_count >= 2
        and header_has_taxonomy
        and not _NON_MAPPING_HEADER_RE.search(normalized_header)
    ):
        return 1
    return 0


def extract_bridge_values(
    description: Any,
    content: Any,
    *,
    subject: str | None = None,
) -> tuple[str, ...]:
    """Extract subject-anchored relation values from one evidence chunk."""

    subject = str(subject or "").strip() or extract_bridge_subject(description)
    text = str(content or "")
    if subject is None or subject.casefold() not in text.casefold():
        return ()

    values: list[str] = []
    claim_units = _iter_claim_units(text)
    for claim in claim_units:
        if (
            not claim.structured
            or _assertion_status(claim.semantic_text) != "active"
        ):
            continue
        cells = claim.row_cells or _structured_cells(claim.result_text)
        header_cells = claim.header_cells
        if not _has_positive_term_occurrence(claim.result_text, subject):
            continue
        subject_indexes = {
            index
            for index, cell in enumerate(cells)
            if _subject_item_matches(cell, subject)
        }
        if not subject_indexes:
            continue
        other_indexes = [
            index for index in range(len(cells)) if index not in subject_indexes
        ]
        selected_indexes: list[int] = []
        if header_cells and len(header_cells) == len(cells):
            scored = [
                (_header_descriptor_score(header_cells[index], description), index)
                for index in other_indexes
            ]
            best_score = max((score for score, _ in scored), default=0)
            best_indexes = [index for score, index in scored if score == best_score]
            if best_score > 0 and len(best_indexes) == 1:
                selected_indexes = best_indexes
        # Header-bearing tables must prove that the other column describes the
        # requested taxonomy.  Without this gate a leave table such as
        # ``5天以上 | 总经理`` becomes the false fact ``总经理 -> 5天以上``.
        # Headerless key/value rows retain the conservative two-cell fallback.
        if (
            not selected_indexes
            and not header_cells
            and len(other_indexes) == 1
        ):
            selected_indexes = other_indexes
        # A row with several unrelated value columns is not a safe mapping
        # unless its header uniquely identifies the bridge dimension.
        for index in selected_indexes:
            raw_value = cells[index]
            if (
                _NEGATED_VALUE_RE.search(_compact_text(raw_value))
                or re.search(r"(?:不适用|排除|除外|之外|以外)", raw_value)
            ):
                continue
            values.extend(_value_variants(raw_value, subject=subject))

    escaped_subject = re.escape(subject)
    patterns = (
        (
            re.compile(
                _FORWARD_RELATION_TEMPLATE.format(subject=escaped_subject),
                re.IGNORECASE,
            ),
            False,
        ),
        (
            re.compile(
                _REVERSE_RELATION_TEMPLATE.format(subject=escaped_subject),
                re.IGNORECASE,
            ),
            True,
        ),
    )
    # Relation regexes are applied independently to each sentence/line. This
    # makes the claim boundary explicit instead of relying on ``.`` semantics,
    # which still cross Chinese punctuation.
    for claim_unit in claim_units:
        if claim_unit.structured:
            continue
        original_claim = claim_unit.result_text
        claim = re.sub(
            r"^(?:(?:但|但是|然而|不过)\s*)?"
            r"(?:现行|现在|当前|现|修订后|更正后|正式规定)?[：:,，\s]*",
            "",
            original_claim,
            flags=re.IGNORECASE,
        )
        for pattern, reverse in patterns:
            for match in pattern.finditer(claim):
                if (
                    _assertion_status(claim_unit.section_heading) != "active"
                    or _assertion_status(
                        claim,
                        start=match.start(),
                        end=match.end(),
                    ) != "active"
                ):
                    continue
                if not _relation_match_is_positive(
                    claim,
                    match,
                    reverse=reverse,
                ):
                    continue
                values.extend(
                    _value_variants(match.group("value"), subject=subject)
                )
        # Corrections often elide the already named subject:
        # ``X对应D级，但现在对应C级``.  The correction boundary is explicit,
        # so the right-hand relation inherits only that same local subject.
        # This is not a general cross-sentence coreference rule.
        if content_contains_positive_subject(original_claim, subject):
            for match in _CORRECTED_RELATION_RE.finditer(original_claim):
                if _assertion_status(
                    original_claim,
                    start=match.start("value"),
                    end=match.end("value"),
                ) != "active":
                    continue
                values.extend(
                    _value_variants(match.group("value"), subject=subject)
                )

    return tuple(dict.fromkeys(values))[:8]


def resolve_bridge_facts(
    requirements: Sequence[AnswerRequirementV2],
    candidates: Iterable[Mapping[str, Any] | EvidenceItem],
    *,
    supported_only: bool = False,
) -> tuple[ResolvedBridgeFact, ...]:
    """Resolve every source-grounded bridge fact in a bounded candidate set."""

    bridge_requirements = tuple(
        requirement for requirement in requirements if requirement.role == "bridge"
    )
    facts: list[ResolvedBridgeFact] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        chunk_id = _candidate_id(candidate)
        if not chunk_id:
            continue
        kb_id, doc_id, products, versions, projects = _candidate_scope(candidate)
        supported_ids = (
            set(candidate.supports_requirement_ids)
            if isinstance(candidate, EvidenceItem)
            else {
                str(value)
                for value in candidate.get("supports_requirement_ids", [])
                if isinstance(value, str)
            }
        )
        for requirement in bridge_requirements:
            if supported_only and requirement.id not in supported_ids:
                continue
            if not _requirement_scope_allows_candidate(requirement, candidate):
                continue
            subject = bridge_subject_for_requirement(requirement)
            if subject is None:
                continue
            for value in extract_bridge_values(
                requirement.description,
                _candidate_content(candidate),
                subject=subject,
            ):
                key = (requirement.id, value.casefold(), chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                facts.append(ResolvedBridgeFact(
                    requirement_id=requirement.id,
                    subject=subject,
                    value=value,
                    source_chunk_id=chunk_id,
                    source_doc_id=doc_id,
                    source_kb_id=kb_id,
                    scope_products=products,
                    scope_versions=versions,
                    scope_projects=projects,
                ))
    return tuple(facts)[:32]


def _normalized_fact_value(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _fact_scope_key(
    fact: ResolvedBridgeFact,
) -> tuple[
    str,
    str,
    str,
    str,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    return (
        fact.requirement_id,
        fact.subject.casefold(),
        fact.source_kb_id,
        fact.source_doc_id,
        fact.scope_products,
        fact.scope_versions,
        fact.scope_projects,
    )


def partition_bridge_facts(
    facts: Sequence[ResolvedBridgeFact],
) -> tuple[tuple[ResolvedBridgeFact, ...], tuple[BridgeFactConflict, ...]]:
    """Reject conflicting mappings asserted by the same source document.

    A knowledge base may legitimately contain two independently applicable
    documents (or two revisions that still need user disambiguation).  Treating
    their different values as one global conflict erased both complete answer
    graphs before the post-evidence ambiguity stage could compare them.  The
    integrity boundary is the source document plus its declared applicability
    scope: contradictions inside that boundary fail closed, while different
    documents remain candidate facts and must still prove a unique value on an
    answer path.
    """

    grouped: dict[
        tuple[
            str,
            str,
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ],
        list[ResolvedBridgeFact],
    ] = defaultdict(list)
    for fact in facts:
        grouped[_fact_scope_key(fact)].append(fact)

    accepted: list[ResolvedBridgeFact] = []
    conflicts: list[BridgeFactConflict] = []
    for key, values in grouped.items():
        normalized_values = {
            _normalized_fact_value(fact.value): fact.value for fact in values
        }
        if len(normalized_values) == 1:
            accepted.extend(values)
            continue
        (
            requirement_id,
            subject,
            kb_id,
            doc_id,
            products,
            versions,
            projects,
        ) = key
        conflicts.append(BridgeFactConflict(
            requirement_id=requirement_id,
            subject=subject,
            source_kb_id=kb_id,
            source_doc_id=doc_id,
            scope_products=products,
            scope_versions=versions,
            scope_projects=projects,
            values=tuple(sorted(normalized_values.values(), key=str.casefold)),
            source_chunk_ids=tuple(dict.fromkeys(
                fact.source_chunk_id for fact in values
            )),
        ))
    return tuple(accepted), tuple(conflicts)


def _scopes_overlap(left: Sequence[str], right: Sequence[str]) -> bool:
    return not left or not right or bool(set(left) & set(right))


def _scopes_explicitly_conflict(
    left: Sequence[str],
    right: Sequence[str],
) -> bool:
    return bool(left and right and not (set(left) & set(right)))


def bridge_fact_matches_candidate_scope(
    fact: ResolvedBridgeFact,
    candidate: Mapping[str, Any] | EvidenceItem,
) -> bool:
    """Allow same-document or non-conflicting cross-document evidence only."""

    kb_id, doc_id, products, versions, projects = _candidate_scope(candidate)
    if fact.source_kb_id and kb_id and fact.source_kb_id != kb_id:
        return False
    # A doc id is an origin identity, not permission to merge mutually
    # exclusive applicability scopes. A re-indexed/combined document can carry
    # several explicit product or version slices under the same id.
    if (
        _scopes_explicitly_conflict(fact.scope_products, products)
        or _scopes_explicitly_conflict(fact.scope_versions, versions)
        or _scopes_explicitly_conflict(fact.scope_projects, projects)
    ):
        return False
    if fact.source_doc_id and doc_id and fact.source_doc_id == doc_id:
        return True
    return bool(
        _scopes_overlap(fact.scope_products, products)
        and _scopes_overlap(fact.scope_versions, versions)
        and _scopes_overlap(fact.scope_projects, projects)
    )


def _target_text(answer_description: str, subjects: Sequence[str]) -> str:
    value = str(answer_description or "")
    clauses = [
        clause.strip()
        for clause in re.split(r"[。；;！？?]+", value)
        if clause.strip()
    ]
    if (
        len(clauses) >= 2
        and len(clauses[0]) <= 24
        and (
            re.match(r"^(?:那|那么|至于|关于)", clauses[0])
            or clauses[0].endswith("呢")
        )
    ):
        # Conversation resolution may append the prior standalone question to
        # a short current-turn facet (``那住宿呢。普通员工的出差标准是什么``).
        # The first clause is the requested answer head; the appended text is
        # retrieval context and must not make the claim require every old term.
        value = clauses[0]
    value = _LEADING_GROUNDED_WRITING_RE.sub("", value).strip()
    value = _LEADING_POLICY_LOOKUP_ACTION_RE.sub("", value).strip()
    constraints = extract_query_constraints(value)
    matched_scope = str(constraints.matched_text or "").strip()
    if matched_scope:
        value = re.sub(
            re.escape(matched_scope),
            " ",
            value,
            count=1,
            flags=re.IGNORECASE,
        )
    for subject in sorted(set(subjects), key=len, reverse=True):
        value = re.sub(re.escape(subject), " ", value, flags=re.IGNORECASE)
    # Subject removal can leave grammatical particles as isolated tokens. Only
    # remove those newly exposed boundaries; never delete the same characters
    # inside a business noun such as ``目的地`` or ``在途订单``.
    value = re.sub(
        r"(?<![A-Za-z0-9_\u3400-\u9fff])的(?=[\s\u3400-\u9fff])",
        " ",
        value,
    )
    value = re.sub(
        r"(?<![A-Za-z0-9_\u3400-\u9fff])(?:在|于|和|与|及)(?=\s)",
        " ",
        value,
    )
    value = value.strip(" \t：:,，。；;?？")
    value = _TARGET_CONDITION_QUESTION_RE.sub(
        lambda match: match.group("target"),
        value,
    )
    for _ in range(3):
        updated = _LEADING_TARGET_FILLER_RE.sub("", value).strip()
        updated = _LEADING_INTERROGATIVE_ACTION_RE.sub("", updated).strip()
        updated = _LEADING_ANSWER_ACTION_RE.sub("", updated).strip()
        updated = _LEADING_TARGET_CONNECTOR_RE.sub("", updated).strip()
        updated = _TRAILING_QUESTION_RE.sub("", updated).strip()
        if updated == value:
            break
        value = updated
    # Remove at most the outer answer-shape noun.  Repeated stripping turns a
    # meaningful compound such as ``目标制度标准`` into the vague word ``目标``
    # and makes unrelated claims collide under the same key.
    for _ in range(1):
        updated = _TARGET_SHAPE_SUFFIX_RE.sub("", value).strip()
        meaningful_updated = _LEADING_TARGET_FILLER_RE.sub(
            "",
            updated,
        ).strip()
        if (
            updated == value
            or len(re.sub(r"\s+", "", meaningful_updated)) < 2
        ):
            break
        value = updated
    return value


def answer_target_terms(
    answer_description: str,
    *,
    bridge_subjects: Sequence[str],
) -> tuple[str, ...]:
    """Return domain-neutral answer terms after removing bridge subjects."""

    terms: list[str] = []
    for match in _TEXT_TOKEN_RE.finditer(
        _target_text(answer_description, bridge_subjects).casefold()
    ):
        token = match.group(0)
        if not re.fullmatch(r"[\u3400-\u9fff]+", token) or 2 <= len(token) <= 24:
            terms.append(token)
        # Split a grammatical possessive only when both sides are substantial
        # phrases. This yields ``出差`` + ``住宿`` from ``出差的住宿`` without
        # globally deleting the same character inside ``目的地``.
        if re.fullmatch(r"[\u3400-\u9fff]{5,}", token) and "的" in token:
            for left, right in re.findall(
                r"([\u3400-\u9fff]{2,})的([\u3400-\u9fff]{2,})",
                token,
            ):
                terms.extend((left, right))
        if re.fullmatch(r"[\u3400-\u9fff]{4,}", token):
            without_action = _TARGET_ACTION_WORD_RE.sub("", token).strip()
            if (
                2 <= len(without_action) < len(token)
                and _ACTION_QUALIFIED_TARGET_SUFFIX_RE.search(token)
            ):
                terms.append(without_action)
    return tuple(dict.fromkeys(
        term
        for term in terms
        if term not in _GENERIC_TARGET_TERMS
    ))[:32]


def _answer_target_semantic_parts(
    answer_description: str,
    *,
    bridge_subjects: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Separate residual applicability context from the answer head.

    A bridge subject may be only the taxonomy-bearing prefix of the user's
    grammatical scope.  For example, the local planner can safely search an
    ``employee -> grade`` mapping for both ``employee travel lodging`` and
    ``employee family lodging``.  It must not, however, silently discard the
    residual ``travel``/``family`` qualifier when proving the final answer.

    Chinese possessives make that boundary observable after the bridge subject
    is removed: ``travel 的 lodging`` consists of an applicability context and
    an answer head.  Candidate-level proof must retain both, while the concrete
    result clause itself only needs to state the answer head.  This is a
    structural rule; it does not enumerate business entities or activities.
    """

    target_text = _target_text(answer_description, bridge_subjects).casefold()
    context_terms: list[str] = []
    head_terms: list[str] = []
    for match in _TEXT_TOKEN_RE.finditer(target_text):
        token = match.group(0)
        if re.fullmatch(r"[\u3400-\u9fff]{5,}", token) and "的" in token:
            parts = [
                part
                for raw_part in token.split("的")
                if (
                    part := re.sub(
                        r"^(?:在|于|对|向|给|按|以|将|把)",
                        "",
                        raw_part,
                    ).strip()
                )
                and len(part) >= 2
                and part not in _GENERIC_TARGET_TERMS
            ]
            if len(parts) >= 2:
                context_terms.extend(parts[:-1])
                head_terms.append(parts[-1])
                continue
        if (
            token not in _GENERIC_TARGET_TERMS
            and (
                not re.fullmatch(r"[\u3400-\u9fff]+", token)
                or 2 <= len(token) <= 24
            )
        ):
            head_terms.append(token)
    return (
        tuple(dict.fromkeys(context_terms))[:16],
        tuple(dict.fromkeys(head_terms))[:32],
    )


def content_matches_complete_answer_target(
    answer_description: str,
    content: Any,
    *,
    bridge_subjects: Sequence[str],
) -> bool:
    """Require every residual context plus the concrete answer head.

    ``content_matches_answer_target`` remains the intentionally broad topic
    matcher used by ordinary single-hop evidence.  A resolved multi-hop join
    needs a stronger contract: a shorter bridge subject cannot authorize an
    answer for a longer entity/context merely because the final noun matches.
    """

    context_terms, head_terms = _answer_target_semantic_parts(
        answer_description,
        bridge_subjects=bridge_subjects,
    )
    normalized_content = _normalized_text(content)
    return bool(
        head_terms
        and all(
            _target_anchor_matches(term, normalized_content)
            for term in context_terms
        )
        and any(
            _target_anchor_matches(term, normalized_content)
            for term in head_terms
        )
    )


def _ordered_near_match(anchor: str, content: str) -> bool:
    """Match compact CJK abbreviations with a bounded per-character gap.

    A previous global-subsequence check made ``餐补`` match arbitrary text such
    as ``聚餐活动补录``.  Allow at most one inserted CJK character between
    adjacent anchor characters; this retains common compact compounds while
    refusing an unbounded semantic leap.
    """

    if not re.fullmatch(r"[\u3400-\u9fff]{2,8}", anchor):
        return False
    # Preserve punctuation/space boundaries instead of concatenating the whole
    # document into one artificial CJK stream. Approximation is allowed only at
    # the beginning of a lexical run; an embedded sequence such as
    # ``聚餐...补录`` is not a safe abbreviation proof.
    for cjk_content in re.findall(r"[\u3400-\u9fff]+", content):
        if not cjk_content.startswith(anchor[0]):
            continue
        position = 0
        for expected in anchor[1:]:
            next_position = cjk_content.find(expected, position + 1)
            if next_position < 0 or next_position - position > 2:
                break
            position = next_position
        else:
            return True
    return False


def _exact_target_anchor_match(anchor: str, content: str) -> bool:
    if not re.fullmatch(r"[\u3400-\u9fff]{2,}", anchor):
        return anchor in content
    for match in re.finditer(re.escape(anchor), content):
        prefix = content[match.start() - 1:match.start()]
        suffix = content[match.end():]
        prefix_is_cjk = bool(prefix and re.match(r"[\u3400-\u9fff]", prefix))
        suffix_is_cjk = bool(suffix and re.match(r"[\u3400-\u9fff]", suffix))
        if not prefix_is_cjk or not suffix_is_cjk:
            return True
        if _TARGET_CONTINUATION_RE.match(suffix):
            return True
    return False


def _target_anchor_matches(anchor: str, content: str) -> bool:
    compact_anchor = re.sub(r"\s+", "", anchor).casefold()
    compact_content = re.sub(r"\s+", "", content).casefold()
    return bool(
        compact_anchor
        and (
            _exact_target_anchor_match(compact_anchor, compact_content)
            or _ordered_near_match(compact_anchor, compact_content)
        )
    )


def _target_claim_matches(
    target_terms: Sequence[str],
    content: str,
) -> bool:
    """Match a claim against a possibly paraphrased compact CJK target.

    Retrieval and reranker annotations are deliberately insufficient at this
    boundary, but requiring one exact, unsegmented Chinese compound is also
    brittle: ``采购申请单笔审批额度`` may be stated as ``单笔采购申请金额``.
    We first use the strict anchor matcher, then allow bounded CJK bigram
    coverage.  The fallback requires several ordered lexical facts from the
    same claim; one generic word such as ``审批`` or ``配置`` can never pass.
    """

    normalized_content = _normalized_text(content)
    if any(
        _target_anchor_matches(term, normalized_content)
        for term in target_terms
    ):
        return True

    content_bigrams = {
        run[index:index + 2]
        for run in re.findall(r"[\u3400-\u9fff]+", normalized_content)
        for index in range(len(run) - 1)
    }
    if not content_bigrams:
        return False
    for term in target_terms:
        compact_term = re.sub(r"\s+", "", str(term or ""))
        if not re.fullmatch(r"[\u3400-\u9fff]{4,24}", compact_term):
            continue
        target_bigrams = {
            compact_term[index:index + 2]
            for index in range(len(compact_term) - 1)
        }
        overlap = target_bigrams & content_bigrams
        if len(overlap) < 2:
            continue
        coverage = len(overlap) / len(target_bigrams)
        if coverage >= 0.5 or (len(overlap) >= 3 and coverage >= 0.35):
            return True
    return False


def content_contains_bridge_value(content: Any, value: str) -> bool:
    normalized_value = _compact_text(value)
    normalized_content = _compact_text(content)
    if len(normalized_value) < 2 or not normalized_content:
        return False
    starts_as_code = bool(re.match(r"[A-Za-z0-9]", normalized_value))
    ends_as_code = bool(re.search(r"[A-Za-z0-9]$", normalized_value))
    pattern_text = re.escape(normalized_value)
    if starts_as_code:
        pattern_text = r"(?<![A-Za-z0-9_.+-])" + pattern_text
    if ends_as_code:
        # R1 is distinct from R10 and R1级. A code followed by normal claim
        # prose (``R1仅可查看``) remains a valid occurrence.
        pattern_text += (
            r"(?![A-Za-z0-9_.+-]|[" + re.escape(_EXACT_CODE_SUFFIXES) + r"])"
        )
    pattern = re.compile(pattern_text, re.IGNORECASE)
    for match in pattern.finditer(normalized_content):
        if _occurrence_is_excluded(
            normalized_content,
            match.start(),
            match.end(),
        ):
            continue
        return True
    return False


def content_contains_positive_subject(content: Any, subject: str) -> bool:
    """Match a positive, non-embedded subject occurrence in one claim."""

    normalized = _compact_text(content)
    needle = _compact_text(subject)
    if len(needle) < 2 or not normalized:
        return False
    for match in re.finditer(re.escape(needle), normalized, re.IGNORECASE):
        if _occurrence_is_excluded(normalized, match.start(), match.end()):
            continue
        prefix = normalized[match.start() - 1:match.start()]
        if prefix and re.match(r"[A-Za-z0-9_\u3400-\u9fff]", prefix):
            structural_prefix = normalized[max(0, match.start() - 3):match.start()]
            if not re.search(r"(?:在|于|对|由|给|向|按|以|把|将)$", structural_prefix):
                continue
        suffix = normalized[match.end():]
        if not suffix:
            return True
        first = suffix[0]
        if not re.match(r"[A-Za-z0-9_\u3400-\u9fff]", first):
            return True
        # Chinese has no universal word boundary. Fail closed on arbitrary
        # entity suffixes, while permitting only grammatical/predicate starts.
        if re.match(
            r"(?:的|在|对应|属于|归属|映射|适用|享受|"
            r"可|应|需|须|为|是|有|无|不)",
            suffix,
        ):
            return True
    return False


def content_matches_answer_target(
    answer_description: str,
    content: Any,
    *,
    bridge_subjects: Sequence[str],
) -> bool:
    terms = answer_target_terms(
        answer_description,
        bridge_subjects=bridge_subjects,
    )
    normalized_content = _normalized_text(content)
    return bool(terms) and any(
        _target_anchor_matches(term, normalized_content) for term in terms
    )


def _claim_key(
    claim: _ClaimUnit,
    *,
    target_terms: Sequence[str],
    result_cells: Sequence[str] = (),
) -> str:
    """Build a stable local facet key without using filenames as semantics."""

    result_set = {_compact_text(value) for value in result_cells if value}
    row_labels: list[str] = []
    if claim.structured:
        for cell in claim.row_cells:
            normalized = _compact_text(cell)
            if not normalized or normalized in result_set:
                continue
            if any(_target_anchor_matches(term, normalized) for term in target_terms):
                continue
            if _NON_RESULT_TEXT_RE.fullmatch(normalized):
                continue
            row_labels.append(normalized)
    section = re.sub(
        r"^(?:第?[一二三四五六七八九十百0-9]+[章节、.．]\s*)+",
        "",
        _compact_text(claim.section_heading),
    )
    target = "/".join(_compact_text(term) for term in target_terms[:4])
    parts = [value for value in (target, section, *row_labels[:3]) if value]
    return "|".join(parts)[:240]


def _claim_assertion(
    claim: _ClaimUnit,
    *,
    target_terms: Sequence[str],
    bridge_values: Sequence[str] = (),
) -> ClaimAssertion:
    """Classify one source-authored claim as an answer assertion.

    Retrieval relevance is intentionally absent from this decision.  A claim
    must be active and carry an independently usable scalar, category,
    normative action or procedure.  References, stale rules and tentative
    statements remain context but can never complete an answer requirement.
    """

    source = re.sub(r"\s+", " ", claim.result_text).strip()
    structural_text = " ".join((claim.section_heading, *claim.header_cells))
    structural_status = _assertion_status(structural_text)
    if structural_status != "active":
        return ClaimAssertion(status=structural_status, result_kind="none")
    reference_only_match = _REFERENCE_ONLY_RE.search(source)
    actionable = _REFERENCE_ONLY_RE.sub(" ", source).strip(" ：:，,。；;")
    if not actionable:
        return ClaimAssertion(
            status="active",
            result_kind="reference_only" if reference_only_match else "none",
        )

    def positive_bridge_value(cell: str) -> bool:
        return any(
            content_contains_bridge_value(cell, value)
            for value in bridge_values
        )

    scalar_values = tuple(dict.fromkeys(
        _compact_text(match.group(0))
        for match in _SCALAR_RESULT_RE.finditer(actionable)
        if not _occurrence_is_excluded(
            actionable,
            match.start(),
            match.end(),
        )
        and _assertion_status(
            actionable,
            start=match.start(),
            end=match.end(),
        ) == "active"
    ))
    if not scalar_values:
        for match in _BARE_VALUE_PREDICATE_RE.finditer(actionable):
            if (
                not _occurrence_is_excluded(
                    actionable,
                    match.start(),
                    match.end(),
                )
                and _assertion_status(
                    actionable,
                    start=match.start(),
                    end=match.end(),
                ) == "active"
            ):
                scalar_values = (_compact_text(match.group(0)),)
                break
    if scalar_values:
        return ClaimAssertion(
            status="active",
            result_kind="scalar",
            normalized_result="|".join(scalar_values)[:240],
            claim_key=_claim_key(
                claim,
                target_terms=target_terms,
                result_cells=scalar_values,
            ),
        )

    normative = next((
        match
        for match in _NORMATIVE_RESULT_RE.finditer(actionable)
        if _assertion_status(
            actionable,
            start=match.start(),
            end=match.end(),
        ) == "active"
    ), None)
    if normative is not None:
        return ClaimAssertion(
            status="active",
            result_kind="normative",
            normalized_result=_compact_text(actionable)[:240],
            claim_key=_claim_key(claim, target_terms=target_terms),
        )
    procedure = _PROCEDURE_RESULT_RE.search(actionable)
    if (
        procedure is not None
        and _assertion_status(
            actionable,
            start=procedure.start(),
            end=procedure.end(),
        ) == "active"
    ):
        return ClaimAssertion(
            status="active",
            result_kind="procedure",
            normalized_result=_compact_text(actionable)[:240],
            claim_key=_claim_key(claim, target_terms=target_terms),
        )

    categorical_values: list[str] = []
    if claim.structured:
        cells = claim.row_cells or _structured_cells(claim.result_text)
        for cell in cells:
            normalized = _normalized_text(cell).strip("：:|[]【】()（）<>")
            if (
                not normalized
                or _MARKDOWN_SEPARATOR_RE.fullmatch(normalized)
                or _assertion_status(normalized) != "active"
                or _SCALAR_RESULT_RE.search(normalized)
                or positive_bridge_value(normalized)
                or any(
                    _target_anchor_matches(term, normalized)
                    for term in target_terms
                )
                or normalized in _GENERIC_BRIDGE_VALUES
                or _NON_RESULT_TEXT_RE.fullmatch(normalized)
            ):
                continue
            categorical_values.append(normalized)
    else:
        for predicate in _CATEGORICAL_PREDICATE_RE.finditer(actionable):
            if _assertion_status(
                actionable,
                start=predicate.start(),
                end=predicate.end(),
            ) != "active":
                continue
            value = predicate.group("value").strip(" ：:，,。；;")
            residual_value = value
            for bridge_value in sorted(bridge_values, key=len, reverse=True):
                residual_value = re.sub(
                    re.escape(bridge_value),
                    "",
                    residual_value,
                    flags=re.IGNORECASE,
                )
            residual_value = re.sub(
                r"^(?:的)?(?:标准|规定|制度|政策|办法|规范|要求)?$",
                "",
                residual_value,
                flags=re.IGNORECASE,
            ).strip()
            if residual_value in {
                "将",
                "把",
                "由",
                "按",
                "以",
                "对",
                "向",
                "给",
                "为",
                "是",
            }:
                continue
            # A bare taxonomy code is a join key, not a final answer, unless
            # the requested target itself asks for that taxonomy dimension.
            taxonomy_requested = any(
                any(term in target for term in _TAXONOMY_HEADER_TERMS)
                for target in target_terms
            )
            bare_identifier = bool(
                _BRIDGE_IDENTIFIER_RE.fullmatch(residual_value)
            )
            if (
                residual_value
                and not _NON_RESULT_TEXT_RE.fullmatch(residual_value)
                and (taxonomy_requested or not bare_identifier)
            ):
                categorical_values.append(residual_value)
                break

    normalized_categories = tuple(dict.fromkeys(
        _compact_text(value) for value in categorical_values if _compact_text(value)
    ))
    if normalized_categories:
        return ClaimAssertion(
            status="active",
            result_kind="categorical",
            normalized_result="|".join(normalized_categories)[:240],
            claim_key=_claim_key(
                claim,
                target_terms=target_terms,
                result_cells=categorical_values,
            ),
        )
    final_status = _assertion_status(actionable)
    return ClaimAssertion(
        status=final_status,
        result_kind="reference_only" if reference_only_match else "none",
    )


def adjudicate_answer_claims(
    answer_description: str,
    content: Any,
    *,
    bridge_subjects: Sequence[str] = (),
    bridge_values: Sequence[str] = (),
    required_subjects: Sequence[str] = (),
    document_root_target_verified: bool = False,
) -> tuple[ClaimAssertion, ...]:
    """Return active, concrete claims that can support one answer target."""

    target_terms = answer_target_terms(
        answer_description,
        bridge_subjects=bridge_subjects,
    )
    if not target_terms:
        return ()
    assertions: list[ClaimAssertion] = []
    for claim in _iter_claim_units(content):
        if required_subjects and not all(
            content_contains_positive_subject(claim.semantic_text, subject)
            for subject in required_subjects
        ):
            continue
        # Once a dependent answer has resolved its bridge (for example
        # ``person -> grade``), the assertion ledger must evaluate only the
        # row/clause applicable to that resolved value.  Without this boundary
        # every row in a classification matrix is recorded as an answer claim,
        # and the later conflict detector mistakes normal per-class values for
        # contradictory policy.  A genuinely universal rule remains eligible
        # because it explicitly declares that the bridge dimension does not
        # differentiate the result.
        if bridge_values and not all(
            content_contains_bridge_value(claim.result_text, value)
            for value in bridge_values
        ) and not _UNIVERSAL_APPLICABILITY_RE.search(claim.semantic_text):
            continue
        if (
            not document_root_target_verified
            and not _target_claim_matches(
                target_terms,
                _normalized_text(claim.semantic_text),
            )
        ):
            continue
        assertion = _claim_assertion(
            claim,
            target_terms=target_terms,
            bridge_values=bridge_values,
        )
        if assertion.supports_answer:
            assertions.append(assertion)
    return tuple(assertions)


def _candidate_topic_text(
    candidate: Mapping[str, Any] | EvidenceItem,
) -> str:
    raw = _candidate_mapping(candidate)
    metadata = raw.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    values = [
        raw.get("content"),
        raw.get("filename"),
        raw.get("source"),
        raw.get("heading"),
        raw.get("title"),
        metadata.get("filename"),
        metadata.get("source"),
        metadata.get("heading"),
        metadata.get("title"),
    ]
    return "\n".join(
        str(value).strip() for value in values if str(value or "").strip()
    )


def _claim_has_concrete_result_set(
    claim: _ClaimUnit,
    *,
    bridge_values: Sequence[str],
    target_terms: Sequence[str],
    allow_universal_bridge: bool = False,
) -> bool:
    """Require a result independent of the join key and target label."""

    values_present = bool(bridge_values) and all(
        content_contains_bridge_value(claim.result_text, value)
        for value in bridge_values
    )
    universal = bool(
        allow_universal_bridge
        and _UNIVERSAL_APPLICABILITY_RE.search(claim.semantic_text)
    )
    if not values_present and not universal:
        return False
    assertion = _claim_assertion(
        claim,
        target_terms=target_terms,
        bridge_values=bridge_values,
    )
    if not assertion.supports_answer:
        return False
    if claim.structured:
        cells = _structured_cells(claim.result_text)
        for cell in cells:
            if any(
                content_contains_bridge_value(cell, value)
                for value in bridge_values
            ):
                continue
            normalized = _normalized_text(cell).strip("：:|[]【】()（）<>")
            if not normalized or _MARKDOWN_SEPARATOR_RE.fullmatch(normalized):
                continue
            if any(
                _target_anchor_matches(term, normalized)
                for term in target_terms
            ):
                continue
            if normalized in _GENERIC_BRIDGE_VALUES or any(
                normalized == term for term in _TAXONOMY_HEADER_TERMS
            ):
                continue
            # A table cell itself may be a categorical result (seat class,
            # operation mode, disposition), so it need not contain a number.
            return True
        return False

    residual = _compact_text(claim.result_text)
    for bridge_value in sorted(bridge_values, key=len, reverse=True):
        residual = re.sub(
            re.escape(_compact_text(bridge_value)),
            "",
            residual,
            flags=re.IGNORECASE,
        )
    if _RESULT_SIGNAL_RE.search(residual):
        return True
    # Categorical prose after an explicit predicate separator is also a result,
    # but a label plus bridge value alone (``餐补：D级``) is not.
    separator = re.search(r"[：:=]", claim.result_text)
    if separator:
        tail = claim.result_text[separator.end():]
        for bridge_value in sorted(bridge_values, key=len, reverse=True):
            tail = re.sub(
                re.escape(bridge_value),
                "",
                tail,
                flags=re.IGNORECASE,
            )
        tail = re.sub(r"[^A-Za-z0-9\u3400-\u9fff]+", "", tail)
        if len(tail) >= 2 and not any(
            _target_anchor_matches(term, tail) for term in target_terms
        ):
            return True
    return False


def _claim_supports_resolved_answer_set(
    *,
    answer_description: str,
    claim: _ClaimUnit,
    bridge_values: Sequence[str],
    bridge_subjects: Sequence[str],
    document_root_target_verified: bool = False,
    allow_universal_bridge: bool = False,
) -> bool:
    target_terms = answer_target_terms(
        answer_description,
        bridge_subjects=bridge_subjects,
    )
    values_present = bool(bridge_values) and all(
        content_contains_bridge_value(claim.result_text, value)
        for value in bridge_values
    )
    universal = bool(
        allow_universal_bridge
        and _UNIVERSAL_APPLICABILITY_RE.search(claim.semantic_text)
    )
    return bool(
        target_terms
        and bridge_values
        and (values_present or universal)
        and (
            document_root_target_verified
            or any(
                _target_anchor_matches(
                    term,
                    _normalized_text(claim.semantic_text),
                )
                for term in target_terms
            )
        )
        and _claim_has_concrete_result_set(
            claim,
            bridge_values=bridge_values,
            target_terms=target_terms,
            allow_universal_bridge=allow_universal_bridge,
        )
    )


def _bridge_source_contains_independent_answer(
    *,
    answer_description: str,
    content: str,
    fact: ResolvedBridgeFact,
    bridge_subjects: Sequence[str],
) -> bool:
    """Reject a mapping-only chunk whose document title repeats the topic."""

    return any(
        _claim_supports_resolved_answer_set(
            answer_description=answer_description,
            claim=claim,
            bridge_values=(fact.value,),
            bridge_subjects=bridge_subjects,
        )
        for claim in _iter_claim_units(content)
    )


def candidate_supports_resolved_answer(
    answer_requirement: AnswerRequirementV2,
    candidate: Mapping[str, Any] | EvidenceItem,
    fact: ResolvedBridgeFact,
    *,
    bridge_subjects: Sequence[str],
) -> bool:
    return candidate_supports_resolved_answer_set(
        answer_requirement,
        candidate,
        (fact,),
        bridge_subjects=bridge_subjects,
    )


def candidate_supports_resolved_answer_set(
    answer_requirement: AnswerRequirementV2,
    candidate: Mapping[str, Any] | EvidenceItem,
    facts: Sequence[ResolvedBridgeFact],
    *,
    bridge_subjects: Sequence[str],
    document_root_target_verified: bool = False,
) -> bool:
    """Require every bridge dependency in one answer-bearing claim.

    Evaluating facts independently and OR-ing their candidate matches can join
    two different answers in the same chunk.  This set-level boundary verifies
    all resolved values, the answer target and one concrete result together.
    """

    if not facts:
        return False
    if not _requirement_scope_allows_candidate(answer_requirement, candidate):
        return False
    if not _bridge_facts_are_scope_compatible(facts):
        return False
    content = _candidate_content(candidate)
    if answer_requirement.depends_on_requirement_ids is not None:
        dependency_ids = set(answer_requirement.depends_on_requirement_ids)
        if not dependency_ids or not all(
            fact.requirement_id in dependency_ids for fact in facts
        ):
            return False
    elif not all(
        fact.subject.casefold() in answer_requirement.description.casefold()
        for fact in facts
    ):
        return False
    if not all(
        bridge_fact_matches_candidate_scope(fact, candidate) for fact in facts
    ):
        return False
    allow_universal_bridge = len(facts) == 1
    if not all(
        content_contains_bridge_value(content, fact.value) for fact in facts
    ) and not (
        allow_universal_bridge
        and _UNIVERSAL_APPLICABILITY_RE.search(content)
    ):
        return False
    if not content_matches_complete_answer_target(
        answer_requirement.description,
        _candidate_topic_text(candidate),
        bridge_subjects=bridge_subjects,
    ):
        return False
    if not any(
        _claim_supports_resolved_answer_set(
            answer_description=answer_requirement.description,
            claim=claim,
            bridge_values=tuple(fact.value for fact in facts),
            bridge_subjects=bridge_subjects,
            document_root_target_verified=document_root_target_verified,
            allow_universal_bridge=allow_universal_bridge,
        )
        for claim in _iter_claim_units(content)
    ):
        # Filename/title may retrieve this candidate, but only body-level claim
        # units can establish target + join key + concrete result.
        return False
    if any(_candidate_id(candidate) == fact.source_chunk_id for fact in facts):
        # A mapping source may also contain a real answer, but only the same
        # claim-set proof above can make it dual-role. A pure mapping sentence
        # has no independent target/result and has already failed.
        return all(
            _bridge_source_contains_independent_answer(
                answer_description=answer_requirement.description,
                content=content,
                fact=fact,
                bridge_subjects=bridge_subjects,
            )
            for fact in facts
            if _candidate_id(candidate) == fact.source_chunk_id
        )
    return True


def _resolved_query(answer_description: str, fact: ResolvedBridgeFact) -> str:
    replaced = re.sub(
        re.escape(fact.subject),
        fact.value,
        answer_description,
        flags=re.IGNORECASE,
    )
    if replaced == answer_description:
        replaced = f"{fact.value} {answer_description}"
    scope_terms = tuple(dict.fromkeys(
        value
        for values in (
            fact.scope_products,
            fact.scope_versions,
            fact.scope_projects,
        )
        for value in values
        if value
    ))
    if scope_terms:
        replaced = f"{replaced} {' '.join(scope_terms)}"
    return re.sub(r"\s+", " ", replaced).strip()[:500]


def _bridge_facts_are_scope_compatible(
    facts: Sequence[ResolvedBridgeFact],
) -> bool:
    for index, left in enumerate(facts):
        for right in facts[index + 1:]:
            if (
                left.source_kb_id
                and right.source_kb_id
                and left.source_kb_id != right.source_kb_id
            ):
                return False
            if (
                _scopes_explicitly_conflict(
                    left.scope_products,
                    right.scope_products,
                )
                or _scopes_explicitly_conflict(
                    left.scope_versions,
                    right.scope_versions,
                )
                or _scopes_explicitly_conflict(
                    left.scope_projects,
                    right.scope_projects,
                )
            ):
                return False
    return True


def _resolved_query_set(
    answer_description: str,
    facts: Sequence[ResolvedBridgeFact],
) -> str:
    replaced = str(answer_description or "")
    for fact in sorted(facts, key=lambda item: len(item.subject), reverse=True):
        updated = re.sub(
            re.escape(fact.subject),
            fact.value,
            replaced,
            flags=re.IGNORECASE,
        )
        if updated == replaced and fact.value.casefold() not in replaced.casefold():
            replaced = f"{fact.value} {replaced}"
        else:
            replaced = updated
    scope_terms = tuple(dict.fromkeys(
        value
        for fact in facts
        for values in (
            fact.scope_products,
            fact.scope_versions,
            fact.scope_projects,
        )
        for value in values
        if value
    ))
    if scope_terms:
        replaced = f"{replaced} {' '.join(scope_terms)}"
    return re.sub(r"\s+", " ", replaced).strip()[:500]


def build_bridge_expansion_queries(
    requirements: Sequence[AnswerRequirementV2],
    candidates: Sequence[Mapping[str, Any] | EvidenceItem],
) -> tuple[str, ...]:
    """Build only still-missing, source-resolved second-hop queries."""

    facts, _ = partition_bridge_facts(resolve_bridge_facts(requirements, candidates))
    if not facts:
        return ()
    queries: list[str] = []
    answer_requirements = tuple(
        requirement for requirement in requirements if requirement.role == "answer"
    )
    for answer in answer_requirements:
        dependency_ids = set(bridge_dependency_ids_for_answer(
            answer,
            requirements,
        ))
        if not dependency_ids:
            continue
        facts_by_dependency: list[tuple[ResolvedBridgeFact, ...]] = []
        for dependency_id in dependency_ids:
            unique_scoped_facts: dict[
                tuple[
                    str,
                    tuple[str, ...],
                    tuple[str, ...],
                    tuple[str, ...],
                ],
                ResolvedBridgeFact,
            ] = {}
            for fact in facts:
                if fact.requirement_id != dependency_id:
                    continue
                unique_scoped_facts.setdefault(
                    (
                        _normalized_fact_value(fact.value),
                        fact.scope_products,
                        fact.scope_versions,
                        fact.scope_projects,
                    ),
                    fact,
                )
            if not unique_scoped_facts:
                facts_by_dependency = []
                break
            facts_by_dependency.append(tuple(unique_scoped_facts.values()))
        if not facts_by_dependency:
            continue

        # A multi-bridge answer is resolved as one graph.  Per-fact checks can
        # be satisfied by two incompatible half-answers and suppress the query
        # that combines every resolved value.  Generate only bounded,
        # scope-compatible Cartesian combinations and test each complete set.
        for fact_set in product(*facts_by_dependency):
            if not _bridge_facts_are_scope_compatible(fact_set):
                continue
            bridge_subjects = tuple(dict.fromkeys(
                fact.subject for fact in fact_set
            ))
            if any(
                candidate_supports_resolved_answer_set(
                    answer,
                    candidate,
                    fact_set,
                    bridge_subjects=bridge_subjects,
                )
                for candidate in candidates
            ):
                continue
            query = _resolved_query_set(answer.description, fact_set)
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= 4:
                return tuple(queries)
    return tuple(queries)


__all__ = [
    "BridgeFactConflict",
    "ClaimAssertion",
    "ResolvedBridgeFact",
    "adjudicate_answer_claims",
    "answer_target_terms",
    "bridge_dependency_ids_for_answer",
    "bridge_fact_matches_candidate_scope",
    "bridge_subject_for_requirement",
    "build_bridge_expansion_queries",
    "candidate_supports_resolved_answer",
    "candidate_supports_resolved_answer_set",
    "content_contains_positive_subject",
    "content_matches_complete_answer_target",
    "content_matches_answer_target",
    "extract_bridge_subject",
    "extract_bridge_values",
    "partition_bridge_facts",
    "resolve_bridge_facts",
]
