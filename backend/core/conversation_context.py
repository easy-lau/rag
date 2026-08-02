"""Conversation-aware query preparation for multi-turn RAG.

The intent router and retriever need a self-contained query, while the chat UI
must keep the user's original wording.  This module bridges those two needs:
it detects explicit references to the previous turn, builds a deterministic
standalone query, and reloads prior evidence from the database instead of
trusting the JSON snapshot stored on a message.
"""

from __future__ import annotations

import re
import uuid
import logging
from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.query_constraints import extract_query_constraints
from core.query_surface_structure import (
    has_current_turn_local_enumeration_antecedent,
    parse_distributive_enumeration,
)
from core.query_semantics import ResolvedTurnSemantics
from core.read_sessions import ReadSessionFactory, isolated_read_session
from models.db_models import Document, DocumentChunk, Message


logger = logging.getLogger(__name__)


HISTORY_MESSAGE_LIMIT = 6
HISTORY_TOTAL_CHARS = 6000
HISTORY_MESSAGE_CHARS = 2000
CARRYOVER_SOURCE_LIMIT = 5

# 该提示既是用户可见文案，也是下一轮识别“上一轮正在补槽位”的稳定标记。
# 保持在上下文模块中，避免 API 层和解析层各自复制后发生漂移。
UNRESOLVED_REFERENCE_MESSAGE = (
    "我还无法确定你说的内容具体指什么。请补充对应的配置项、方案或原文，"
    "或者在相关回答后继续追问。"
)

_EXPLICIT_NEW_TOPIC_RE = re.compile(
    r"^\s*(?:换(?:个|一个)(?:问题|话题)|另一个问题|新问题|"
    r"不说这个|先不谈这个|跳过这个)"
    r"(?:[，,、：:。.!！?？\s]*)",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    r"(?:"
    r"这些|那些|上述|上面(?:的)?|前面(?:的)?|刚才(?:的)?|上一(?:条|轮|个)(?:的)?|"
    r"这个(?:配置|参数|功能|方案|问题|版本|系统|文档|内容|结果|回答)|"
    r"这个(?=\s*(?:怎么|如何|为什么|有(?:什么|何)|能否|能不能|可以|是否|"
    r"会不会|会|要|该|呢|[，。！？,.!?]))|"
    r"(?<!应)该(?:配置|参数|功能|方案|问题|版本|系统|文档|内容|结果)|"
    r"它们?|其中|继续(?:说|讲|分析|回答)?|再详细(?:说|讲)?"
    r")",
    re.IGNORECASE,
)
_UNRESOLVED_REFERENCE_RE = re.compile(
    r"(?:"
    r"这些|那些|上述|上面(?:的)?|前面(?:的)?|刚才(?:的)?|上一(?:条|轮|个)(?:的)?|"
    r"这个(?:配置|参数|方案|版本|文档|内容|结果|回答)|"
    r"这个(?=\s*(?:怎么|如何|为什么|有(?:什么|何)|能否|能不能|可以|是否|"
    r"会不会|会|要|该|呢|[，。！？,.!?]))|"
    r"(?<!应)该(?:配置|参数|方案|版本|文档|内容|结果)|"
    r"它们?|其中"
    r")",
    re.IGNORECASE,
)
_ELLIPTICAL_ENTITY_FOLLOWUP_RE = re.compile(
    # 裸“如果”通常会引出完整条件问题；只有“如果是 Redis 呢”这类
    # 明确省略被比较对象的表达才继承上一轮。
    r"^\s*(?:"
    r"(?:那(?:么)?|如果是|改成|换成)\s*(?=\S).{1,48}?|"
    r"(?:云\s*枢|cloudpivot(?:\s*platform)?)\s*"
    r"\d{1,3}(?:\.\d{1,3}){0,3}"
    r")\s*(?:呢|怎么样|如何|可以吗)[？?。.!！\s]*$",
    re.IGNORECASE,
)
_SHORT_FOLLOWUP_RE = re.compile(
    r"^\s*(?:那(?:么)?|所以|然后)?\s*(?:"
    r"为什么|有什么影响|有何影响|怎么处理|如何处理|具体呢|然后呢|还有呢|"
    r"能详细说说吗|可以展开说说吗"
    r")[？?。.!！\s]*$",
    re.IGNORECASE,
)
_MISSING_ACTION_OBJECT_RE = re.compile(
    # “云枢中如何配置 / 那在云枢里怎么设置 / 那要怎么处理”给出了
    # 操作方式，却省略了配置或处理的对象。只有整句在动作处结束才命中；
    # “云枢中如何配置登录用户名枚举”因动作后存在宾语，不会继承旧主题。
    r"^\s*(?:那(?:么)?|然后|所以)?\s*"
    r"(?:"
    r"(?:在|用)\s*(?P<prep_scope>[A-Za-z0-9_.+\-\u3400-\u9fff]"
    r"(?:[A-Za-z0-9_.+\-\u3400-\u9fff \t]{0,38}?[A-Za-z0-9_.+\-\u3400-\u9fff])?)"
    r"\s*(?:里面|中|里|内|上)\s*|"
    # 无介词分支禁止把“对象在产品中”整段吞成 scope；否则“登录用户名枚举在云枢中如何配置”
    # 会被错误地当成缺少宾语的追问。
    r"(?![^中里内上]{0,40}在)(?P<located_scope>[A-Za-z0-9_.+\-\u3400-\u9fff]"
    r"(?:[A-Za-z0-9_.+\-\u3400-\u9fff \t]{0,38}?[A-Za-z0-9_.+\-\u3400-\u9fff])?)"
    r"\s*(?:里面|中|里|内|上)\s*|"
    # 无“中/里/内/上”的写法只接受一个明确的系统名。这里不能使用任意文本，
    # 否则“云枢默认密码怎么配置”会把“云枢默认密码”整体吞成 scope，误判为
    # 缺少配置对象。带宾语的完整问题必须继续作为 standalone question。
    r"(?P<bare_scope>(?![^\n]{0,40}在)(?:"
    r"(?:云\s*枢|cloudpivot(?:\s*platform)?)(?:\s*\d+(?:\.\d+)*)?|"
    r"[A-Za-z][A-Za-z0-9_.+\-]{1,30}|"
    r"[\u3400-\u9fff]{1,12}(?:系统|平台|产品|服务)"
    r"))"
    r"(?=\s*(?:具体|该)?\s*(?:要|应该)?\s*(?:如何|怎么|怎样))"
    r")?"
    r"\s*(?:具体|该)?\s*(?:要|应该)?\s*(?:如何|怎么|怎样)\s*(?:进行)?\s*"
    r"(?:配置|设置|处理|解决|修改|开启|关闭|调整|操作|配)"
    r"(?:一下)?(?:呢)?[？?。.!！\s]*$",
    re.IGNORECASE,
)
_VERSION_ONLY_FOLLOWUP_RE = re.compile(
    r"^\s*(?:那(?:么)?|如果是)?\s*"
    r"(?P<version>\d{1,3}(?:\.\d{1,3}){0,3})(?![\d.])"
    r"\s*(?:版本)?(?:呢|怎么样|如何|可以吗)?[？?。.!！\s]*$",
    re.IGNORECASE,
)
_TECHNICAL_TERM_RE = re.compile(
    r"\b(?:"
    r"[A-Za-z][A-Za-z0-9]*(?:[._&-][A-Za-z0-9]+)+|"
    r"[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*|"
    r"[A-Z][A-Z0-9]{1,15}"
    r")\b"
)
_PLAIN_LATIN_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_.+\-]{1,30}"
    r"(?![A-Za-z0-9_])"
)
_CJK_SCOPE_ENTITY_RE = re.compile(
    r"[\u3400-\u9fff]{1,12}(?:系统|平台|产品|项目|服务)"
)
_COMMON_LATIN_WORDS = {
    "how", "what", "why", "where", "when", "which", "can", "could",
    "should", "please", "help", "the", "this", "that",
}
_KNOWN_LATIN_SCOPE_KEYS = {
    "cloudpivot", "cloudpivotplatform", "redis", "python", "postgresql",
    "postgres", "mysql", "oracle", "sqlserver", "mongodb", "elasticsearch",
    "kafka", "nginx", "docker", "kubernetes", "k8s", "spring", "java",
}
_REFERENCE_WITH_POSTFIX_RE = re.compile(
    r"(?:这个|该|上述|上面(?:的)?|前面(?:的)?)(?:配置|参数|功能|方案|问题|"
    r"版本|系统|文档|内容|结果|回答)?[^：:\n]{0,24}[：:]\s*"
    r"(?P<object>\S[^\n]{1,300})",
    re.IGNORECASE,
)
_GENERIC_POSTFIX_RE = re.compile(
    r"^(?:怎么(?:办|改|处理|配置)?|如何(?:处理|配置|解决)?|为什么|"
    r"有什么影响|具体呢|继续)$",
    re.IGNORECASE,
)
_QUESTION_LIKE_RE = re.compile(
    r"(?:怎么|如何|怎样|为什么|是什么|什么是|怎么办|哪里|在哪|是否|能否|可以吗|"
    r"有何|有什么|请问|介绍|解释|说明|告诉我)",
    re.IGNORECASE,
)
_ACTIONABLE_PREVIOUS_TOPIC_RE = re.compile(
    r"(?:"
    r"配置|设置|参数|开关|登录|用户名|账号|密码|认证|权限|安全|漏洞|枚举|"
    r"错误|异常|报错|超时|接口|地址|流程|审批|字段|表单|组织|用户|功能|"
    r"部署|安装|升级|迁移|集成|对接|通知|数据源|数据库|缓存|持久化|高可用|"
    r"机器人|回调|Webhook|Token|API|SQL|SSO|OAuth|OIDC|SAML|LDAP"
    r")",
    re.IGNORECASE,
)
_CLARIFICATION_CANCEL_RE = re.compile(
    r"^\s*(?:"
    r"算了|不用了?|不需要|暂时不需要|先不用|不了|取消|谢谢|多谢|"
    r"换(?:个|一个)(?:问题|话题|方向)|换个方向|先这样"
    r")(?:[，,、。.!！\s]*(?:谢谢|多谢))?[。.!！\s]*$",
    re.IGNORECASE,
)
_TOPIC_LEADING_FILLER_RE = re.compile(
    r"^\s*(?:(?:请问|我想问(?:一下)?|想请问(?:一下)?|麻烦问一下|"
    r"请(?:帮我)?(?:解释|介绍|说明)(?:一下)?|"
    r"(?:解释|介绍|说明)(?:一下)?|告诉我)\s*[，,：:]?\s*)+",
    re.IGNORECASE,
)
_TOPIC_LEADING_QUESTION_RE = re.compile(
    r"^\s*(?:什么是|为什么(?:要|需要)?|如何理解|怎么理解|"
    r"(?:怎么|如何|怎样)(?:配置|设置|处理|解决))\s*",
    re.IGNORECASE,
)
_TOPIC_TRAILING_QUESTION_RE = re.compile(
    r"\s*(?:"
    r"指(?:的)?(?:是)?什么|"
    r"是(?:什么|啥)(?:意思)?|(?:什么|啥)意思|"
    r"有(?:什么|何)(?:作用|影响|危害|风险|区别|用途)|"
    r"(?:怎么|如何|怎样)(?:配置|设置|处理|解决)|"
    r"怎么回事|怎么办|为什么"
    r")\s*$",
    re.IGNORECASE,
)


def _scope_key(value: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9\u3400-\u9fff]+",
        "",
        str(value or "").casefold(),
    )
    if re.fullmatch(r"(?:云枢|cloudpivot(?:platform)?)(?:\d+(?:\.\d+)*)?", normalized):
        return "cloudpivot"
    return normalized


def _is_known_product_scope(value: str) -> bool:
    return _scope_key(value) == "cloudpivot"


def _previous_scope_keys(question: str | None) -> set[str]:
    """Extract only explicit product/system entities for topic-change guards."""

    text = str(question or "")
    if not text:
        return set()
    keys: set[str] = set()
    constraints = extract_query_constraints(text)
    if constraints.product:
        keys.add(_scope_key(constraints.product))
    for entity in _PLAIN_LATIN_ENTITY_RE.findall(text):
        key = _scope_key(entity)
        if key in _KNOWN_LATIN_SCOPE_KEYS:
            keys.add(key)
    keys.update(
        key
        for key in (_scope_key(entity) for entity in _CJK_SCOPE_ENTITY_RE.findall(text))
        if key
    )
    return keys


def _missing_action_scope(question: str) -> str | None:
    match = _MISSING_ACTION_OBJECT_RE.fullmatch(question.strip())
    if match is None:
        return None
    raw = (
        match.group("prep_scope")
        or match.group("located_scope")
        or match.group("bare_scope")
    )
    return re.sub(r"[ \t]+", " ", raw).strip() if raw else None


def _is_explicit_scope_change(
    question: str,
    previous_user_question: str | None,
) -> bool:
    """Reject inheritance when both turns name different explicit systems."""

    current_scope = _missing_action_scope(question)
    if not current_scope or previous_user_question is None:
        return False
    previous_scopes = _previous_scope_keys(previous_user_question)
    if previous_scopes:
        return _scope_key(current_scope) not in previous_scopes
    # 当前产品词典能明确识别“云枢”时，允许它补齐上一轮的通用主题；
    # Redis/Python 等未知新实体没有上一轮同实体依据，保守视为新话题。
    return not _is_known_product_scope(current_scope)


def _has_explicit_postfix_object(question: str) -> bool:
    """Detect anaphora followed by a concrete object supplied in this turn."""

    match = _REFERENCE_WITH_POSTFIX_RE.search(question)
    if match is None:
        return False
    postfix = match.group("object").strip(" \t\r\n，。！？；,.!?;")
    return bool(
        len(postfix) >= 2
        and not _GENERIC_POSTFIX_RE.fullmatch(postfix)
    )


def _previous_topic_supports_action_followup(question: str | None) -> bool:
    """Require a plausible configurable object before filling an omitted one.

    Product-scoped phrases such as ``云枢中如何配置`` are ambiguous even when
    a conversation exists.  They may inherit ``登录用户名枚举`` or ``默认密码``,
    but must not turn an unrelated greeting, weather question or general
    science topic into a retrieval query merely because it was the last turn.
    """

    text = str(question or "").strip()
    if not text:
        return True
    if _previous_scope_keys(text):
        return True
    return bool(
        _ACTIONABLE_PREVIOUS_TOPIC_RE.search(text)
        or _TECHNICAL_TERM_RE.search(text)
    )


def _unresolved_reason_without_history(question: str) -> str | None:
    """Classify a query fragment that is not allowed to inherit any history."""

    normalized = (question or "").strip()
    if not normalized:
        return None
    unresolved = _UNRESOLVED_REFERENCE_RE.search(normalized)
    if unresolved:
        return f"unresolved_reference:{unresolved.group(0)}"
    if _ELLIPTICAL_ENTITY_FOLLOWUP_RE.fullmatch(normalized):
        return "unresolved_reference:elliptical_entity"
    if _MISSING_ACTION_OBJECT_RE.fullmatch(normalized):
        return "unresolved_reference:missing_action_object"
    if len(normalized) <= 32 and _SHORT_FOLLOWUP_RE.fullmatch(normalized):
        return "unresolved_reference:short_elliptical_question"
    return None


def _has_local_anaphora_antecedent(
    question: str,
    match: re.Match[str],
) -> bool:
    """Whether ``这些/那些`` resolves to an explicit list in this sentence.

    The test is intentionally syntactic and domain-neutral.  It does not try
    to infer what a list item means; it only recognizes a sequence of at least
    two non-empty units before the demonstrative plus a distributive question
    tail.  That distinction prevents ``这些配置有什么影响`` from losing its
    genuine historical reference while keeping a self-contained question such
    as ``住宿、餐补还有出差补贴这些分别是多少`` out of stale context rewriting.
    """

    marker = str(match.group(0) or "").strip()
    if marker not in {"这些", "那些"}:
        return False
    # The context resolver and the query planner must share this boundary.
    # Do not reconstruct it with an independent regular expression here: that
    # was the root cause of one module treating a current enumeration as a
    # history-follow-up while the other split it into three questions.
    structure = parse_distributive_enumeration(question)
    if (
        structure is not None
        and structure.local_anaphora == marker
        and structure.has_local_anaphora_antecedent
    ):
        return True
    return has_current_turn_local_enumeration_antecedent(question)


@dataclass(frozen=True)
class ConversationContext:
    """Prepared context passed from the API layer into the RAG pipeline."""

    is_followup: bool
    followup_reason: str
    standalone_query: str
    history_messages: tuple[dict[str, str], ...]
    carryover_sources: tuple[dict[str, Any], ...]
    previous_user_question: str | None = None
    unresolved_reference: bool = False
    # ``t1`` is the newest completed/partial user turn, ``t2`` the one before
    # it, and so on.  These request-local keys are the only historical
    # identities exposed to the semantic router; database/message/chunk ids
    # never enter the model contract.
    route_turn_candidates: tuple["RouteTurnCandidate", ...] = ()
    relation: str = "new"
    query_resolution_mode: str = "current"
    context_turn_keys: tuple[str, ...] = ()
    pending_route_state: dict[str, Any] | None = None


V2ExecutionContextMode = Literal[
    "current_turn_baseline",
    "resolved_turn_semantics",
    "v3_catalog_context",
]


@dataclass(frozen=True)
class V2ExecutionContext:
    """The only conversation data V2 is allowed to receive for execution.

    ``ConversationContext`` deliberately keeps a candidate pool while routing
    and source-anchored analysis are in progress.  That pool may contain
    legacy route-selected history, but it is *not* executable semantic state.
    Passing the same object to V2 used to let that candidate history reach the
    answer model whenever analysis timed out or was rejected.

    This narrower hand-off makes the boundary explicit: the initial execution
    context is always current-turn-only; history and carry-over evidence may
    appear only after ``ResolvedTurnSemantics`` has been applied and verified.
    """

    mode: V2ExecutionContextMode
    retrieval_query: str
    conversation_history: tuple[dict[str, str], ...]
    carryover_sources: tuple[dict[str, Any], ...]
    is_followup: bool
    followup_reason: str
    context_turn_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        query = str(self.retrieval_query or "").strip()
        if not query:
            raise ValueError("V2 execution context requires a retrieval query")
        if self.mode == "current_turn_baseline":
            if (
                self.conversation_history
                or self.carryover_sources
                or self.is_followup
                or self.context_turn_keys
            ):
                raise ValueError(
                    "current-turn V2 execution context cannot contain history"
                )
        elif self.mode not in {"resolved_turn_semantics", "v3_catalog_context"}:
            raise ValueError("unsupported V2 execution context mode")
        object.__setattr__(self, "retrieval_query", query)

    @property
    def semantic_context_applied(self) -> bool:
        return self.mode != "current_turn_baseline"

    def safe_summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "semantic_context_applied": self.semantic_context_applied,
            "is_followup": self.is_followup,
            "context_turn_count": len(self.context_turn_keys),
            "history_message_count": len(self.conversation_history),
            "carryover_source_count": len(self.carryover_sources),
        }


def build_current_turn_v2_execution_context(
    *,
    retrieval_query: str,
) -> V2ExecutionContext:
    """Build the fail-closed V2 baseline before semantic analysis succeeds.

    The caller supplies the exact current-turn (or a server-authorized pending
    evidence-selection) retrieval query.  There is intentionally no
    ``ConversationContext`` argument: a route candidate context must not be
    copied into the execution baseline by accident.
    """

    return V2ExecutionContext(
        mode="current_turn_baseline",
        retrieval_query=retrieval_query,
        conversation_history=(),
        carryover_sources=(),
        is_followup=False,
        followup_reason="v2_current_turn_baseline",
        context_turn_keys=(),
    )


def build_v3_catalog_candidate_context(
    *,
    context: ConversationContext,
    current_question: str,
) -> ConversationContext:
    """Clear provisional history before V3 selects a catalog-bound span.

    ``prepare_conversation_context`` retains a legacy heuristic projection for
    the explicit compatibility entry.  V3 must never treat that projection as
    execution state: route candidates are only a bounded, authorised source
    catalog and the literal current question remains the immutable anchor
    until ``apply_v3_catalog_context_selection`` has validated a selection and
    reloaded its evidence under the current request scope.
    """

    if not isinstance(context, ConversationContext):
        raise ValueError("context must be a ConversationContext")
    query = str(current_question or "").strip()
    if not query:
        raise ValueError("V3 catalog candidate context requires current question")
    return replace(
        context,
        is_followup=False,
        followup_reason="v3_catalog_candidate_pool",
        standalone_query=query,
        history_messages=(),
        carryover_sources=(),
        unresolved_reference=False,
        relation="new",
        query_resolution_mode="v3_catalog_candidate_pool",
        context_turn_keys=(),
    )


def build_resolved_v2_execution_context(
    *,
    context: ConversationContext,
    semantics: ResolvedTurnSemantics,
) -> V2ExecutionContext:
    """Project a successfully applied semantic contract into V2 inputs.

    ``apply_resolved_turn_semantics`` reloads selected sources under current
    RBAC/KB scope.  This function performs a second structural equality check
    before handing the result to V2, preventing a stale route context from
    being paired with a newly compiled semantic plan.
    """

    if not isinstance(context, ConversationContext):
        raise ValueError("context must be a ConversationContext")
    if not isinstance(semantics, ResolvedTurnSemantics):
        raise ValueError("semantics must be a ResolvedTurnSemantics")
    if context.standalone_query != semantics.canonical_retrieval_query:
        raise ValueError("resolved V2 context query does not match semantics")
    if context.context_turn_keys != semantics.selected_context_turn_keys:
        raise ValueError("resolved V2 context keys do not match semantics")
    expected_followup = not semantics.self_contained
    if context.is_followup != expected_followup:
        raise ValueError("resolved V2 context follow-up state does not match semantics")
    return V2ExecutionContext(
        mode="resolved_turn_semantics",
        retrieval_query=context.standalone_query,
        conversation_history=tuple(context.history_messages),
        carryover_sources=tuple(context.carryover_sources),
        is_followup=context.is_followup,
        followup_reason=context.followup_reason,
        context_turn_keys=context.context_turn_keys,
    )


def build_v3_catalog_v2_execution_context(
    *,
    context: ConversationContext,
    current_question: str,
) -> V2ExecutionContext:
    """Project a trusted V3 catalog selection into V2 execution inputs.

    The V3 compiler already carries historical qualifier text in its answer
    task descriptions.  The retrieval anchor must nevertheless remain the
    literal current question, rather than the old history-concatenated
    standalone query.  Selected history is available only as bounded dialogue
    context and freshly reloaded carry-over evidence.
    """

    if not isinstance(context, ConversationContext):
        raise ValueError("context must be a ConversationContext")
    query = str(current_question or "").strip()
    if not query:
        raise ValueError("V3 catalog execution requires the current question")
    if context.standalone_query != query:
        raise ValueError("V3 catalog context must retain the current question")
    expected_followup = bool(context.context_turn_keys)
    if context.is_followup != expected_followup:
        raise ValueError("V3 catalog context follow-up state does not match keys")
    return V2ExecutionContext(
        mode="v3_catalog_context",
        retrieval_query=query,
        conversation_history=tuple(context.history_messages),
        carryover_sources=tuple(context.carryover_sources),
        is_followup=context.is_followup,
        followup_reason=context.followup_reason,
        context_turn_keys=context.context_turn_keys,
    )


@dataclass(frozen=True)
class RouteTurnCandidate:
    """A request-local historical turn candidate for semantic routing.

    ``raw_sources`` remains application-only.  ``to_prompt_dict`` intentionally
    emits only bounded dialogue text and a source-count flag, so the model can
    choose ``t1``/``t2`` without learning persistent resource identities.
    """

    candidate_key: str
    user_question: str
    assistant_answer: str | None
    raw_sources: tuple[dict[str, Any], ...] = ()

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "user_input": self.user_question[:1200],
            "assistant_answer": (self.assistant_answer or "")[:1200],
            "reusable_source_count": sum(
                1 for source in self.raw_sources if _source_is_reusable(source)
            ),
        }


def detect_followup(
    question: str,
    *,
    has_previous_turn: bool,
    previous_user_question: str | None = None,
) -> tuple[bool, str]:
    """Conservatively detect follow-ups that cannot stand alone.

    We deliberately avoid classifying every short question as a follow-up.
    Explicit anaphora (``这些配置``/``上述内容``) and a small set of complete
    elliptical questions are safe deterministic signals; ambiguous cases can
    still go through normal retrieval rather than inheriting stale context.
    """

    normalized = (question or "").strip()
    if not normalized:
        return False, "no_previous_turn"
    new_topic = _EXPLICIT_NEW_TOPIC_RE.match(normalized)
    if new_topic is not None:
        # “换个问题”明确切断旧主题，但后半句仍可能自身缺少宾语。此时应澄清，
        # 不能因为用户声明了新话题就拿残缺问题直接检索。
        remainder = normalized[new_topic.end():].strip()
        unresolved = _unresolved_reason_without_history(remainder)
        if unresolved is not None:
            return False, f"{unresolved}:explicit_new_topic"
        return False, "explicit_new_topic"
    if _has_explicit_postfix_object(normalized):
        return False, "explicit_postfix_object"
    reference_matches = tuple(_REFERENCE_RE.finditer(normalized))
    if (
        len(reference_matches) == 1
        and _has_local_anaphora_antecedent(normalized, reference_matches[0])
    ):
        return False, f"local_anaphora_antecedent:{reference_matches[0].group(0)}"
    if not has_previous_turn:
        unresolved = _unresolved_reason_without_history(normalized)
        if unresolved is not None:
            return False, unresolved
        return False, "no_previous_turn"
    match = reference_matches[0] if reference_matches else None
    if match:
        return True, f"anaphora:{match.group(0)}"
    if _ELLIPTICAL_ENTITY_FOLLOWUP_RE.fullmatch(normalized):
        return True, "elliptical_entity"
    if _MISSING_ACTION_OBJECT_RE.fullmatch(normalized):
        if not _previous_topic_supports_action_followup(previous_user_question):
            return False, "unresolved_reference:missing_action_object"
        if _is_explicit_scope_change(normalized, previous_user_question):
            return (
                False,
                "unresolved_reference:missing_action_object:explicit_new_scope",
            )
        return True, "missing_action_object"
    if len(normalized) <= 32 and _SHORT_FOLLOWUP_RE.fullmatch(normalized):
        return True, "short_elliptical_question"
    return False, "standalone_question"


def _bounded_history(messages: Iterable[Message]) -> tuple[dict[str, str], ...]:
    """Keep the newest useful dialogue within a predictable prompt budget."""

    prepared: list[dict[str, str]] = []
    remaining = HISTORY_TOTAL_CHARS
    for message in reversed(list(messages)):
        if message.role not in {"user", "assistant"}:
            continue
        content = (message.content or "").strip()
        if not content or remaining <= 0:
            continue
        bounded = content[: min(HISTORY_MESSAGE_CHARS, remaining)]
        prepared.append({"role": message.role, "content": bounded})
        remaining -= len(bounded)
    prepared.reverse()
    return tuple(prepared)


def _source_is_reusable(source: dict[str, Any]) -> bool:
    """Keep prior displayed evidence unless it was explicitly irrelevant.

    ``answer_support`` is query-dependent: a version-mismatched source can have
    zero support for “is this valid on 8.6?” and still become useful when the
    user explicitly asks what the cited legacy setting does.  Every retained
    chunk is therefore reloaded and reranked for the new standalone query.
    """

    if source.get("evidence_role") == "irrelevant":
        return False
    return bool(source.get("id") or source.get("chunk_id"))


def _route_turn_candidates(
    messages: Iterable[Message],
    *,
    limit: int = 3,
) -> tuple[RouteTurnCandidate, ...]:
    """Build newest-first, request-local turn candidates.

    A failed or interrupted stream may leave a user message without an
    assistant reply; it is still a valid topic candidate.  An assistant is
    paired only with the user immediately preceding it, never with an older
    user across another user boundary.
    """

    ordered = list(messages)
    turns: list[tuple[Message, Message | None]] = []
    index = 0
    while index < len(ordered):
        message = ordered[index]
        if message.role != "user":
            index += 1
            continue
        assistant: Message | None = None
        cursor = index + 1
        while cursor < len(ordered) and ordered[cursor].role != "user":
            if assistant is None and ordered[cursor].role == "assistant":
                assistant = ordered[cursor]
            cursor += 1
        turns.append((message, assistant))
        index = cursor

    candidates: list[RouteTurnCandidate] = []
    for position, (user_message, assistant_message) in enumerate(
        reversed(turns[-max(1, limit):]),
        start=1,
    ):
        raw_sources = tuple(
            source
            for source in (assistant_message.sources or [])
            if isinstance(source, dict)
        ) if assistant_message is not None else ()
        candidates.append(
            RouteTurnCandidate(
                candidate_key=f"t{position}",
                user_question=(user_message.content or "").strip(),
                assistant_answer=(
                    (assistant_message.content or "").strip()
                    if assistant_message is not None
                    else None
                ),
                raw_sources=raw_sources,
            )
        )
    return tuple(candidates)


def route_context_payloads(
    context: ConversationContext,
) -> tuple[dict[str, Any], ...]:
    """Return the bounded, identity-free candidate catalogue for the router."""

    return tuple(candidate.to_prompt_dict() for candidate in context.route_turn_candidates)


def build_route_context_payloads(
    messages: Iterable[dict[str, Any]],
    *,
    limit: int = 3,
) -> tuple[dict[str, Any], ...]:
    """Build a synthetic candidate catalogue for the admin routing sandbox.

    This helper never queries conversations, messages, knowledge bases or
    documents.  The ``intent:read`` test endpoint can therefore exercise
    multi-turn semantics without becoming a data-scope bypass.
    """

    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().casefold()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            normalized.append({"role": role, "content": content})

    turns: list[tuple[str, str]] = []
    index = 0
    while index < len(normalized):
        item = normalized[index]
        if item["role"] != "user":
            index += 1
            continue
        assistant = ""
        cursor = index + 1
        while cursor < len(normalized) and normalized[cursor]["role"] != "user":
            if not assistant and normalized[cursor]["role"] == "assistant":
                assistant = normalized[cursor]["content"]
            cursor += 1
        turns.append((item["content"], assistant))
        index = cursor

    return tuple(
        {
            "candidate_key": f"t{position}",
            "user_input": user_text[:1200],
            "assistant_answer": assistant_text[:1200],
            "reusable_source_count": 0,
        }
        for position, (user_text, assistant_text) in enumerate(
            reversed(turns[-max(1, limit):]),
            start=1,
        )
    )


def _technical_terms(*texts: str, limit: int = 8) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in _TECHNICAL_TERM_RE.findall(text or ""):
            normalized = match.casefold()
            if normalized in seen or len(match) > 100:
                continue
            seen.add(normalized)
            terms.append(match)
            if len(terms) >= limit:
                return terms
    return terms


def _previous_topic_object(question: str) -> str:
    """Extract the object of a previous question without diagnostic prose."""

    original = question.strip(" \t\r\n，。！？；：,.!?;:")
    # 只清理问句首尾的语气和问法，不在主题中间做全局替换。
    # 例如“请问什么是登录用户名枚举？”与“登录用户名枚举指的是什么”
    # 都应得到同一个稳定宾语“登录用户名枚举”。
    topic = _TOPIC_LEADING_FILLER_RE.sub("", original)
    topic = _TOPIC_LEADING_QUESTION_RE.sub("", topic)
    topic = topic.strip(" \t\r\n，。！？；：,.!?;:")
    topic = _TOPIC_TRAILING_QUESTION_RE.sub("", topic)
    topic = topic.strip(" \t\r\n，。！？；：,.!?;:")
    return topic or original


def _clean_action_question(question: str) -> str:
    """Remove conversational fillers while preserving product/version scope."""

    without_topic_marker = _EXPLICIT_NEW_TOPIC_RE.sub(
        "",
        question.strip(),
        count=1,
    )
    cleaned = re.sub(
        r"^\s*(?:那(?:么)?|然后|所以)\s*",
        "",
        without_topic_marker,
        flags=re.IGNORECASE,
    ).rstrip(" \t\r\n，。！？；：,.!?;:")
    # The object is appended immediately afterwards; terminal particles would
    # otherwise produce ``怎么设置呢登录用户名枚举`` or ``配置一下默认密码``.
    return re.sub(r"(?:一下呢|一下|呢)$", "", cleaned).rstrip()


def _merge_missing_action_object(action_question: str, object_text: str) -> str:
    """Fill a missing configuration object into a natural standalone query."""

    action = _clean_action_question(action_question)
    topic = _previous_topic_object(object_text)
    current_scope = _missing_action_scope(action)
    if current_scope:
        # “云枢中如何配置” + “云枢登录用户名枚举是什么” should not
        # become “云枢中如何配置云枢登录用户名枚举”.
        if _is_known_product_scope(current_scope):
            scope_pattern = (
                r"(?:云\s*枢|cloudpivot(?:\s*platform)?)"
                r"(?:\s*\d+(?:\.\d+)*)?"
            )
        else:
            scope_pattern = r"\s*".join(
                re.escape(part)
                for part in re.split(r"\s+", current_scope)
                if part
            )
        topic = re.sub(
            rf"^\s*(?:在|用)?\s*{scope_pattern}\s*(?:中|里|内|上|的)?\s*",
            "",
            topic,
            flags=re.IGNORECASE,
        ).strip()
    if not topic:
        topic = _previous_topic_object(object_text)
    return f"{action}{topic}"[:8000]


def _is_clarification_slot_answer(question: str) -> bool:
    """Accept a short entity/value after our deterministic clarification."""

    text = (question or "").strip()
    if (
        not text
        or len(text) > 120
        or "\n" in text
        or _CLARIFICATION_CANCEL_RE.fullmatch(text)
    ):
        return False
    if _EXPLICIT_NEW_TOPIC_RE.search(text) or _QUESTION_LIKE_RE.search(text):
        return False
    return not bool(re.search(r"[？?]", text))


def build_standalone_query(
    question: str,
    *,
    previous_user_question: str | None,
    previous_assistant_answer: str | None,
    carryover_sources: Iterable[dict[str, Any]] = (),
    followup_reason: str | None = None,
) -> str:
    """Turn an anaphoric follow-up into a retrieval-friendly query.

    Previous assistant text only contributes technical identifiers to query
    resolution.  It is never injected as knowledge evidence; factual grounding
    continues to come from revalidated document chunks.
    """

    previous = (previous_user_question or "").strip()
    if not previous:
        return question.strip()
    source_text = "\n".join(
        f"{source.get('filename') or ''}\n{source.get('content') or ''}"
        for source in carryover_sources
    )
    terms = _technical_terms(previous_assistant_answer or "", source_text)
    # 技术标识可以帮助词面召回，但不要把“上一轮提到的关键配置项”这类
    # 诊断元话语送进 embedding/FTS；它们会稀释真实主题或制造 AND 条件。
    key_items = f" {' '.join(terms)}" if terms else ""
    current_text = question.strip()[:8000]
    # “那8.6呢 / 那云枢7呢”继承上一轮主题，但必须把旧版本替换掉；同时不把
    # “原始追问”这类调试字段送进召回。
    version_only = _VERSION_ONLY_FOLLOWUP_RE.fullmatch(question.strip())
    previous_constraints = extract_query_constraints(previous)
    current_constraints = extract_query_constraints(current_text)
    target_product = None
    target_version = None
    if version_only and previous_constraints.product:
        target_product = previous_constraints.product
        target_version = version_only.group("version")
    elif (
        _ELLIPTICAL_ENTITY_FOLLOWUP_RE.fullmatch(question.strip())
        and current_constraints.has_hard_constraint
    ):
        target_product = current_constraints.product
        target_version = current_constraints.version
    if target_product and target_version:
        previous_topic = previous
        if previous_constraints.matched_text:
            previous_topic = re.sub(
                re.escape(previous_constraints.matched_text),
                "",
                previous_topic,
                count=1,
                flags=re.IGNORECASE,
            )
        previous_topic = previous_topic.strip(" \t\r\n，。！？；：,.!?;:")
        return f"{target_product}{target_version} {previous_topic}{key_items}".strip()[:8000]
    if followup_reason == "missing_action_object":
        # 检索问题中不能出现“需要继承的上一轮主题”这类调试元话语；它会
        # 稀释向量并让全文检索产生无意义的 AND 条件。上下文原因只写 Trace。
        return _merge_missing_action_object(current_text, previous)
    return (
        # Current text must come first: deterministic constraint extraction
        # selects the first explicit product/version.  Natural sentences keep
        # vector/FTS inputs free from “用于消解指代” style diagnostic terms.
        f"{current_text}。{previous[:600]}{key_items}"
    )


async def _reload_carryover_sources(
    db: AsyncSession,
    raw_sources: Iterable[dict[str, Any]],
    kb_ids: Iterable[uuid.UUID],
    *,
    read_session_factory: ReadSessionFactory | None = None,
) -> tuple[dict[str, Any], ...]:
    """Reload prior chunks under the current KB scope and document state.

    Carry-over is a bounded recall enhancement, not part of turn/message
    persistence.  In production it therefore runs in an owned read session:
    a missing optional table or a stale document row cannot abort the request
    transaction that owns the current conversation turn.  Scalar snapshots
    are projected before the owned session rolls back, so no detached ORM
    object can escape this boundary.
    """

    allowed_kb_ids = tuple(dict.fromkeys(kb_ids))
    if not allowed_kb_ids:
        return ()

    snapshots: dict[uuid.UUID, dict[str, Any]] = {}
    ordered_ids: list[uuid.UUID] = []
    for source in raw_sources:
        if not isinstance(source, dict) or not _source_is_reusable(source):
            continue
        try:
            chunk_id = uuid.UUID(str(source.get("id") or source.get("chunk_id")))
        except (TypeError, ValueError, AttributeError):
            continue
        if chunk_id in snapshots:
            continue
        snapshots[chunk_id] = source
        ordered_ids.append(chunk_id)
        if len(ordered_ids) >= CARRYOVER_SOURCE_LIMIT:
            break
    if not ordered_ids:
        return ()

    statement = (
        select(DocumentChunk, Document)
        .join(
            Document,
            (Document.id == DocumentChunk.doc_id)
            & (Document.kb_id == DocumentChunk.kb_id),
        )
        .where(
            DocumentChunk.id.in_(ordered_ids),
            DocumentChunk.kb_id.in_(allowed_kb_ids),
            Document.is_active.is_(True),
            Document.status == "ready",
        )
    )
    valid: dict[uuid.UUID, dict[str, Any]] = {}
    try:
        async with isolated_read_session(
            request_db=db,
            session_factory=read_session_factory,
        ) as read_db:
            rows = (await read_db.execute(statement)).all()
            for chunk, document in rows:
                snapshot = snapshots.get(chunk.id, {})
                valid[chunk.id] = {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "kb_id": chunk.kb_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": dict(chunk.metadata_ or {}),
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "source_url": document.source_url,
                    "doc_tags": list(document.tags or []),
                    # Old ranking scores are intentionally not reused for a new query.
                    "retrieval_score": 0.0,
                    "score": 0.0,
                    "candidate_origin": "carryover_previous_turn",
                    "carryover_previous_role": snapshot.get("evidence_role"),
                    "carryover_previous_support": snapshot.get("answer_support"),
                }
    except Exception as exc:
        logger.warning(
            "[conversation context] carry-over source reload degraded error=%s",
            type(exc).__name__,
        )
        return ()
    return tuple(valid[chunk_id] for chunk_id in ordered_ids if chunk_id in valid)


async def prepare_conversation_context(
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    question: str,
    kb_ids: Iterable[uuid.UUID],
    pending_route_state: dict[str, Any] | None = None,
    read_session_factory: ReadSessionFactory | None = None,
) -> ConversationContext:
    """Load recent dialogue, resolve follow-up references and validate sources."""

    rows = (
        await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(HISTORY_MESSAGE_LIMIT)
        )
    ).scalars().all()
    messages = list(reversed(rows))
    route_turn_candidates = _route_turn_candidates(messages)
    # 上下文必须以最近一条 user 为锚点，而不是以最近一条
    # assistant 反向寻找 user。流式失败、用户中止或回答尚未落库时，
    # 最新 user 后可能没有 assistant；此时仍要继承该 user 的主题，
    # 且不能误用更早回答的 sources。
    previous_user_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index].role == "user"
        ),
        None,
    )
    previous_user = (
        messages[previous_user_index]
        if previous_user_index is not None
        else None
    )
    previous_assistant = (
        next(
            (
                message
                for message in messages[previous_user_index + 1:]
                if message.role == "assistant"
            ),
            None,
        )
        if previous_user_index is not None
        else None
    )
    has_previous_turn = previous_user is not None
    clarification_slot_answer = False
    if has_previous_turn and previous_assistant is not None and previous_user is not None:
        previous_unresolved = detect_followup(
            previous_user.content,
            has_previous_turn=False,
        )[1]
        clarification_slot_answer = bool(
            previous_assistant.content == UNRESOLVED_REFERENCE_MESSAGE
            and previous_unresolved.startswith(
                "unresolved_reference:missing_action_object"
            )
            and _is_clarification_slot_answer(question)
        )
    if clarification_slot_answer:
        is_followup, reason = True, "clarification_answer:missing_action_object"
    else:
        is_followup, reason = detect_followup(
            question,
            has_previous_turn=has_previous_turn,
            previous_user_question=(
                previous_user.content if previous_user is not None else None
            ),
        )

    carryover_sources: tuple[dict[str, Any], ...] = ()
    if is_followup and previous_assistant is not None:
        carryover_sources = await _reload_carryover_sources(
            db,
            previous_assistant.sources or [],
            kb_ids,
            read_session_factory=read_session_factory,
        )

    if clarification_slot_answer and previous_user is not None:
        standalone_query = _merge_missing_action_object(
            previous_user.content,
            question,
        )
    elif is_followup:
        standalone_query = build_standalone_query(
            question,
            previous_user_question=previous_user.content if previous_user else None,
            previous_assistant_answer=(
                previous_assistant.content if previous_assistant else None
            ),
            carryover_sources=carryover_sources,
            followup_reason=reason,
        )
    else:
        standalone_query = question.strip()
    return ConversationContext(
        is_followup=is_followup,
        followup_reason=reason,
        standalone_query=standalone_query,
        history_messages=_bounded_history(messages),
        carryover_sources=carryover_sources,
        previous_user_question=previous_user.content if previous_user else None,
        unresolved_reference=reason.startswith("unresolved_reference:"),
        route_turn_candidates=route_turn_candidates,
        relation="followup" if is_followup else "new",
        query_resolution_mode=("contextualize" if is_followup else "current"),
        context_turn_keys=((route_turn_candidates[0].candidate_key,) if is_followup and route_turn_candidates else ()),
        pending_route_state=pending_route_state,
    )


async def resolve_routed_conversation_context(
    db: AsyncSession,
    *,
    context: ConversationContext,
    question: str,
    kb_ids: Iterable[uuid.UUID],
    route_decision: Any,
    read_session_factory: ReadSessionFactory | None = None,
) -> ConversationContext:
    """Apply a validated semantic route to the local conversation context.

    The router is allowed to say that a complete sentence is a refinement of
    an earlier turn without forcing a query rewrite.  Historical evidence is
    reloaded only after the compiler has selected request-local ``t*`` keys;
    this is the key distinction missing from the legacy ``is_followup`` flag.
    """

    def read(name: str, default: Any = None) -> Any:
        if isinstance(route_decision, dict):
            return route_decision.get(name, default)
        return getattr(route_decision, name, default)

    relation = str(read("relation", "new") or "new")
    readiness = str(read("readiness", "ready") or "ready")
    query_resolution = read("query_resolution", {}) or {}
    if isinstance(query_resolution, dict):
        mode_value = query_resolution.get("mode", "current")
        requested_keys = query_resolution.get("context_turn_keys", ()) or ()
    else:
        # Production passes the validated ``RouteQueryResolution`` dataclass,
        # while compatibility callers may still pass a plain dictionary.
        mode_value = getattr(query_resolution, "mode", "current")
        requested_keys = getattr(query_resolution, "context_turn_keys", ()) or ()
    mode = str(mode_value or "current")
    if not isinstance(requested_keys, (list, tuple)):
        requested_keys = ()
    requested_keys = tuple(str(key).strip() for key in requested_keys if str(key).strip())
    if relation == "new":
        requested_keys = ()
    candidate_map = {item.candidate_key: item for item in context.route_turn_candidates}
    selected = tuple(candidate_map[key] for key in requested_keys if key in candidate_map)

    # A malformed/unknown key must never silently widen history.  The contract
    # compiler normally rejects it; this defensive branch makes direct callers
    # safe as well.
    unknown_key = any(key not in candidate_map for key in requested_keys)
    if unknown_key:
        selected = ()
        mode = "clarify"
        readiness = "needs_clarification"

    is_contextual = relation != "new" and mode == "contextualize" and bool(selected)
    is_followup = relation != "new" and bool(selected or mode == "current")
    carryover_sources: tuple[dict[str, Any], ...] = ()
    if selected:
        raw_sources: list[dict[str, Any]] = []
        for candidate in selected:
            raw_sources.extend(candidate.raw_sources)
        carryover_sources = await _reload_carryover_sources(
            db,
            raw_sources,
            kb_ids,
            read_session_factory=read_session_factory,
        )

    previous_candidate = selected[0] if selected else None
    pending_slot_answer = bool(
        is_contextual
        and context.pending_route_state
        and previous_candidate is not None
        and _MISSING_ACTION_OBJECT_RE.fullmatch(previous_candidate.user_question.strip())
        and _is_clarification_slot_answer(question)
    )
    if pending_slot_answer:
        standalone_query = _merge_missing_action_object(
            previous_candidate.user_question,
            question,
        )
    elif is_contextual:
        standalone_query = build_standalone_query(
            question,
            previous_user_question=(
                previous_candidate.user_question if previous_candidate else None
            ),
            previous_assistant_answer=(
                previous_candidate.assistant_answer if previous_candidate else None
            ),
            carryover_sources=carryover_sources,
            followup_reason=(
                "missing_action_object"
                if _MISSING_ACTION_OBJECT_RE.fullmatch(question.strip())
                else None
            ),
        )
    else:
        standalone_query = question.strip()

    history_messages: tuple[dict[str, str], ...] = ()
    if is_followup and selected:
        # Keep the selected turns only.  This avoids sending unrelated older
        # dialogue to the answer model while retaining the semantic relation.
        prepared: list[dict[str, str]] = []
        for candidate in reversed(selected):
            if candidate.user_question:
                prepared.append({"role": "user", "content": candidate.user_question[:HISTORY_MESSAGE_CHARS]})
            if candidate.assistant_answer:
                prepared.append({"role": "assistant", "content": candidate.assistant_answer[:HISTORY_MESSAGE_CHARS]})
        history_messages = _bounded_history(
            tuple(
                _SyntheticMessage(role=item["role"], content=item["content"])
                for item in prepared
            )
        )

    unresolved = readiness == "needs_clarification" or mode == "clarify"
    reason = (
        "route_clarification"
        if unresolved
        else ("route_contextualized" if is_contextual else ("route_followup_current" if relation != "new" else "standalone_question"))
    )
    return replace(
        context,
        is_followup=is_followup,
        followup_reason=reason,
        standalone_query=standalone_query,
        history_messages=history_messages,
        carryover_sources=carryover_sources,
        previous_user_question=(
            previous_candidate.user_question if previous_candidate else context.previous_user_question
        ),
        unresolved_reference=unresolved,
        relation=relation,
        query_resolution_mode=mode,
        context_turn_keys=tuple(candidate.candidate_key for candidate in selected),
    )


async def apply_resolved_turn_semantics(
    db: AsyncSession,
    *,
    context: ConversationContext,
    semantics: ResolvedTurnSemantics,
    kb_ids: Iterable[uuid.UUID],
    read_session_factory: ReadSessionFactory | None = None,
) -> ConversationContext:
    """Apply the one validated semantic context selection to a conversation.

    This is deliberately separate from :func:`resolve_routed_conversation_context`.
    The route contract classifies a request and applies permission policy;
    source-anchored turn semantics select literal historical qualifiers.  The
    old implementation joined prior text into ``standalone_query`` and made a
    later planner guess again.  Here selected history is retained as typed
    source references and only the canonical retrieval rendering is updated.

    Prior evidence is always reloaded under the *current* KB scope.  A model
    supplied turn key can therefore never replay a stale document snapshot or
    widen access to a previous source.
    """

    if not isinstance(semantics, ResolvedTurnSemantics):
        raise ValueError("semantics must be a ResolvedTurnSemantics")
    candidate_by_key = {
        item.candidate_key: item for item in context.route_turn_candidates
    }
    selected: list[RouteTurnCandidate] = []
    for key in semantics.selected_context_turn_keys:
        candidate = candidate_by_key.get(key)
        if candidate is None:
            # The strict model parser normally prevents this.  Fail closed for
            # direct callers too: silently substituting t1 would be a stale
            # context leak.
            raise ValueError("resolved semantics references unavailable context turn")
        selected.append(candidate)

    raw_sources: list[dict[str, Any]] = []
    for candidate in selected:
        raw_sources.extend(candidate.raw_sources)
    carryover_sources = await _reload_carryover_sources(
        db,
        raw_sources,
        kb_ids,
        read_session_factory=read_session_factory,
    ) if selected else ()

    history_messages: tuple[dict[str, str], ...] = ()
    if selected:
        prepared: list[_SyntheticMessage] = []
        for candidate in reversed(selected):
            if candidate.user_question:
                prepared.append(_SyntheticMessage(
                    role="user",
                    content=candidate.user_question[:HISTORY_MESSAGE_CHARS],
                ))
            if candidate.assistant_answer:
                prepared.append(_SyntheticMessage(
                    role="assistant",
                    content=candidate.assistant_answer[:HISTORY_MESSAGE_CHARS],
                ))
        history_messages = _bounded_history(prepared)

    is_contextual = not semantics.self_contained
    previous_candidate = selected[0] if selected else None
    return replace(
        context,
        is_followup=is_contextual,
        followup_reason=(
            "resolved_turn_semantics_contextual"
            if is_contextual
            else "resolved_turn_semantics_current"
        ),
        # This is a terminal retrieval rendering only.  No planner/analyzer
        # may consume it as a replacement source sentence.
        standalone_query=semantics.canonical_retrieval_query,
        history_messages=history_messages,
        carryover_sources=carryover_sources,
        previous_user_question=(
            previous_candidate.user_question
            if previous_candidate is not None
            else context.previous_user_question
        ),
        unresolved_reference=False,
        relation=semantics.relation,
        query_resolution_mode=("contextualize" if is_contextual else "current"),
        context_turn_keys=semantics.selected_context_turn_keys,
    )


async def apply_v3_catalog_context_selection(
    db: AsyncSession,
    *,
    context: ConversationContext,
    current_question: str,
    selected_context_turn_keys: Iterable[str],
    kb_ids: Iterable[uuid.UUID],
    read_session_factory: ReadSessionFactory | None = None,
) -> ConversationContext:
    """Apply only V3 catalog-selected history under current authorisation.

    A V3 model cannot name a database message, document or source.  It can
    only select a ``tN`` span from the route-authorised catalog.  This adapter
    re-resolves those keys against the request-local candidates, reloads any
    reusable evidence inside an owned read transaction, and retains the raw
    *current* question as the retrieval anchor.  It deliberately does not
    concatenate historical user text into a new query or fabricate a
    ``ResolvedTurnSemantics`` object from V3 data.
    """

    if not isinstance(context, ConversationContext):
        raise ValueError("context must be a ConversationContext")
    query = str(current_question or "").strip()
    if not query:
        raise ValueError("V3 catalog context selection requires current question")
    requested = tuple(str(key or "").strip() for key in selected_context_turn_keys)
    if len(requested) > 3 or len(set(requested)) != len(requested):
        raise ValueError("V3 catalog context keys are invalid")
    candidate_by_key = {
        item.candidate_key: item for item in context.route_turn_candidates
    }
    selected: list[RouteTurnCandidate] = []
    for key in requested:
        candidate = candidate_by_key.get(key)
        if candidate is None:
            raise ValueError("V3 catalog references unavailable context turn")
        selected.append(candidate)

    raw_sources: list[dict[str, Any]] = []
    for candidate in selected:
        raw_sources.extend(candidate.raw_sources)
    carryover_sources = await _reload_carryover_sources(
        db,
        raw_sources,
        kb_ids,
        read_session_factory=read_session_factory,
    ) if selected else ()

    history_messages: tuple[dict[str, str], ...] = ()
    if selected:
        prepared: list[_SyntheticMessage] = []
        for candidate in reversed(selected):
            if candidate.user_question:
                prepared.append(_SyntheticMessage(
                    role="user",
                    content=candidate.user_question[:HISTORY_MESSAGE_CHARS],
                ))
            if candidate.assistant_answer:
                prepared.append(_SyntheticMessage(
                    role="assistant",
                    content=candidate.assistant_answer[:HISTORY_MESSAGE_CHARS],
                ))
        history_messages = _bounded_history(prepared)

    return replace(
        context,
        is_followup=bool(selected),
        followup_reason=(
            "query_understanding_v3_contextual"
            if selected
            else "query_understanding_v3_current"
        ),
        standalone_query=query,
        history_messages=history_messages,
        carryover_sources=carryover_sources,
        previous_user_question=(
            selected[0].user_question if selected else context.previous_user_question
        ),
        unresolved_reference=False,
        relation="followup" if selected else "new",
        query_resolution_mode="v3_catalog",
        context_turn_keys=tuple(candidate.candidate_key for candidate in selected),
    )


@dataclass(frozen=True)
class _SyntheticMessage:
    role: str
    content: str
