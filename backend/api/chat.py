import asyncio
import uuid
import logging
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models.db_models import (
    ChatTurn,
    Conversation,
    Document,
    DocumentChunk,
    IntentRouteLog,
    Message,
    User,
    now_utc,
)
from core.chat_turns import (
    MAX_PERSIST_ATTEMPTS,
    RECOVERABLE_TURN_STATUSES,
    TurnRequestConflict,
    assert_turn_request_matches,
    build_turn_request_context,
    commit_with_retry,
    find_turn_for_user,
    message_turn_metadata,
    normalize_request_id,
    normalize_turn_id,
    reclaim_stale_turn,
    renew_turn_lease,
    reserve_turn,
    transition_turn,
    turn_duration_ms,
    turn_lease_expired,
)
from models.schemas import ChatRequest, ConversationOut, ConversationRenameRequest, MessageOut
from core.rag_pipeline import run_rag_stream
from core.rag_v2.pipeline import (
    _plan_with_contract_requirements,
    run_rag_v2_stream,
)
from core.rag_v2.query_plan import plan_query_locally
from core.direct_response import run_direct_response_stream
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import CHAT_USE
from core.intent_router import (
    build_verified_evidence_scope_result,
    classify_intent_result,
)
from core.conversation_context import (
    UNRESOLVED_REFERENCE_MESSAGE,
    prepare_conversation_context,
    resolve_routed_conversation_context,
    route_context_payloads,
)
from core.query_route_compiler import (
    RagTaskContract,
    rag_task_contract_gate_reason,
)
from core.query_route_contract import RouteClarification, RouteUnresolvedSlot
from core.rag_trace import content_fields, log_exception_safely, trace_event
from config import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)
_NON_ANSWER_SOURCE_STATUSES = {
    "no_hit",
    "skipped",
    "error",
    "needs_clarification",
    "version_mismatch",
}
_ANSWER_SOURCE_REQUIRED_STATUSES = {"hit", "partial", "unverified"}
_SUPPORTED_EVIDENCE_STATUSES = (
    _NON_ANSWER_SOURCE_STATUSES | _ANSWER_SOURCE_REQUIRED_STATUSES
)
_EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE = (
    "回答依据校验失败，无法可靠生成知识库答案。请稍后重试。"
)
_SUCCESSFUL_EVIDENCE_SCOPE_STATUSES = {
    "hit",
    "partial",
    "version_mismatch",
    "no_hit",
    "unverified",
}
_EVIDENCE_PENDING_SCHEMA = "rag_pending_clarification.v2"
_EVIDENCE_PENDING_KIND = "evidence_scope"
_EVIDENCE_EVENT_SCHEMA = "rag_evidence_clarification.v1"
_EVIDENCE_ACK_SCHEMA = "rag_evidence_clarification_ack.v1"
_EVIDENCE_DIMENSIONS = {
    "version",
    "product",
    "product_version",
    "project",
    "document",
}
_EVIDENCE_SELECTION_MODES = {"choice", "refine"}
_CHOICE_TEXT_FIELDS = (
    "products",
    "canonical_products",
    "versions",
    "projects",
    "filenames",
)
_CHOICE_UUID_FIELDS = ("kb_ids", "doc_ids")
_PUBLIC_CHOICE_FIELDS = (
    "key",
    "label",
    "products",
    "versions",
    "projects",
    "filenames",
)
_ALL_EVIDENCE_SCOPES_RE = re.compile(
    r"^(?:全部|全都|都要|都查|都看|都对比|全部对比|全部都要|分别对比|"
    r"所有版本|全部版本|两个都要|两个都看|两个都对比)(?:一下|吧|。|！|!)?$",
    re.IGNORECASE,
)
_CANCEL_EVIDENCE_SCOPE_RE = re.compile(
    r"^(?:取消|算了|不用了|不查了|先不查|停止|退出)(?:吧|。|！|!)?$",
    re.IGNORECASE,
)
_EXPLICIT_NEW_QUESTION_RE = re.compile(
    r"[?？]|(?:重新问|新问题|另外|还有个问题|请问|帮我|告诉我|查询|查一下|"
    r"怎么|如何|为什么|为何|什么|哪些|哪个|哪里|是否|能否|可以吗|标准|规定|"
    r"配置|解决|介绍|说明|分析|流程|方法|要求)",
    re.IGNORECASE,
)
_SUBSTANTIVE_NEW_QUESTION_RE = re.compile(
    r"[?？]|(?:重新问|新问题|另外|还有个问题|怎么|如何|为什么|为何|什么|"
    r"哪些|哪个|哪里|是否|能否|可以吗|标准|规定|配置|解决|介绍|说明|"
    r"分析|流程|方法|要求)",
    re.IGNORECASE,
)
_INVALID_EVIDENCE_SELECTION_RE = re.compile(
    r"^(?:随便(?:一个)?|都行|都可以|不知道|不清楚|你看|你决定|任选|任意|"
    r"选一个|哪个好|哪个都行|这个|那个|上面|下面|版本呢|高版本|低版本|"
    r"新版本|旧版本|最新版|最新版本|任意版本|某个版本|默认)(?:吧|。|！|!)?$",
    re.IGNORECASE,
)
_EVIDENCE_REFINEMENT_HINT_RE = re.compile(
    r"(?:\d{2,4}(?:\.\d+){0,4}\s*(?:版|版本)?)|"
    r"(?:产品|版本|项目|范围|系统|平台|环境|地区|部门|职级)",
    re.IGNORECASE,
)


def _select_rag_pipeline_version(
    *,
    configured_version: Literal["v1", "v2"],
    task_contract: object,
    evidence_scope_filter: dict | None,
    evidence_scope_refinement_active: bool,
    is_followup: bool,
    carryover_sources: tuple[dict, ...] | list[dict],
    selected_kb_count: int | None = None,
) -> tuple[Literal["v1", "v2", "direct", "reject"], str]:
    """Choose exactly one runner; V2 never silently falls back to legacy RAG.

    ``v1`` remains an explicit deployment rollback switch.  Once a deployment
    selects V2, however, a missing/invalid contract is a protocol failure, not
    permission to execute the old heuristic pipeline.  Retrieval-dependent
    QA/writing use V2, while verified non-retrieval contracts use the isolated
    direct-response runner.
    """

    if configured_version != "v2":
        return "v1", "configured_v1"
    if not isinstance(task_contract, RagTaskContract):
        return "reject", "missing_or_invalid_task_contract"
    if not task_contract.dispatch_authorized:
        return "reject", "dispatch_not_authorized"
    contract_gate_reason = rag_task_contract_gate_reason(
        task_contract,
        selected_kb_count=selected_kb_count,
    )
    if contract_gate_reason is not None:
        return "reject", f"invalid_task_contract:{contract_gate_reason}"
    if not task_contract.need_retrieval:
        if (
            task_contract.retrieval_policy == "skip"
            and task_contract.response_mode
            in {"general_chat", "writing", "platform_help"}
        ):
            return "direct", f"verified_{task_contract.response_mode}"
        return "reject", "invalid_direct_task_contract"
    if (
        task_contract.retrieval_policy != "required"
        or task_contract.response_mode not in {"grounded_qa", "writing"}
    ):
        return "reject", "invalid_retrieval_task_contract"
    # V2 preserves the existing tag semantics as a soft ordering boost.  Tags
    # neither widen authorization nor bypass the deterministic relevance gate,
    # so they no longer require falling back to the slow model-rerank pipeline.
    # A non-null filter is rebuilt server-side from a validated pending choice
    # and carries an authorized KB/document allow-list.
    evidence_scope_selection = evidence_scope_filter is not None
    if evidence_scope_selection:
        return "v2", "eligible_evidence_scope_selection"
    if evidence_scope_refinement_active:
        return "v2", "eligible_evidence_scope_refinement"
    if is_followup or carryover_sources:
        return "v2", "eligible_grounded_followup"
    if task_contract.response_mode == "writing":
        return "v2", "eligible_knowledge_writing"
    return "v2", "eligible_grounded_qa"


@dataclass(frozen=True)
class _EvidenceScopeReply:
    action: Literal[
        "single",
        "compare_all",
        "refine",
        "repeat",
        "cancel",
        "new_question",
    ]
    choices: tuple[dict, ...] = ()


def _evidence_scope_reply_display_text(
    question: str,
    reply: _EvidenceScopeReply | None,
) -> str:
    """Turn an accepted internal choice token into readable chat history."""

    original = str(question or "").strip()
    if reply is None or reply.action not in {"single", "compare_all"}:
        return original
    labels = [
        str(choice.get("label") or "").strip()
        for choice in reply.choices
        if isinstance(choice, dict) and str(choice.get("label") or "").strip()
    ]
    if not labels:
        return original
    if reply.action == "single":
        return f"选择：{labels[0]}"
    return "选择：都对比（" + "；".join(labels) + "）"


def _future_expiry(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > datetime.now(timezone.utc) else None


def _parsed_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return parsed


def _strict_string_list(
    value: object,
    *,
    max_items: int,
    uuid_values: bool = False,
    require_non_empty: bool = False,
    max_chars: int = 500,
) -> tuple[str, ...] | None:
    if not isinstance(value, list) or len(value) > max_items:
        return None
    if require_non_empty and not value:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            return None
        text = item.strip()
        if not text or len(text) > max_chars:
            return None
        if uuid_values:
            try:
                text = str(uuid.UUID(text))
            except (TypeError, ValueError, AttributeError):
                return None
        key = text.casefold()
        if key in seen:
            return None
        seen.add(key)
        normalized.append(text)
    return tuple(normalized)


def _validated_evidence_choices(
    value: object,
    *,
    selected_kb_ids: tuple[str, ...],
) -> tuple[dict, ...] | None:
    if not isinstance(value, list) or not (2 <= len(value) <= 6):
        return None
    prevalidated_doc_ids: list[tuple[str, ...]] = []
    doc_occurrences: dict[str, int] = {}
    for raw_choice in value:
        if not isinstance(raw_choice, dict):
            return None
        doc_ids = _strict_string_list(
            raw_choice.get("doc_ids"),
            max_items=100,
            uuid_values=True,
            require_non_empty=True,
        )
        if doc_ids is None:
            return None
        prevalidated_doc_ids.append(doc_ids)
        for doc_id in doc_ids:
            doc_occurrences[doc_id] = doc_occurrences.get(doc_id, 0) + 1
    selected_kb_set = set(selected_kb_ids)
    choices: list[dict] = []
    seen_keys: set[str] = set()
    all_doc_ids: set[str] = set()
    for choice_index, raw_choice in enumerate(value):
        key = raw_choice.get("key")
        label = raw_choice.get("label")
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"c[1-9]\d*", key.strip(), re.IGNORECASE)
            or len(key.strip()) > 40
            or not isinstance(label, str)
            or not label.strip()
            or len(label.strip()) > 500
        ):
            return None
        normalized_key = key.strip().casefold()
        if normalized_key in seen_keys:
            return None
        seen_keys.add(normalized_key)

        normalized: dict[str, object] = {
            "key": normalized_key,
            "label": label.strip(),
        }
        for field in _CHOICE_TEXT_FIELDS:
            items = _strict_string_list(raw_choice.get(field), max_items=20)
            if items is None:
                return None
            normalized[field] = list(items)
        for field in _CHOICE_UUID_FIELDS:
            items = (
                prevalidated_doc_ids[choice_index]
                if field == "doc_ids"
                else _strict_string_list(
                    raw_choice.get(field),
                    max_items=100,
                    uuid_values=True,
                    require_non_empty=True,
                )
            )
            if items is None:
                return None
            normalized[field] = list(items)
        raw_anchor_doc_ids = raw_choice.get("anchor_doc_ids")
        raw_companion_doc_ids = raw_choice.get("companion_doc_ids")
        if raw_anchor_doc_ids is None and raw_companion_doc_ids is None:
            # Rolling v2 compatibility: a document appearing in one choice is
            # its exclusive anchor; a cross-choice document is a companion.
            anchor_doc_ids = tuple(
                doc_id
                for doc_id in prevalidated_doc_ids[choice_index]
                if doc_occurrences.get(doc_id) == 1
            )
            companion_doc_ids = tuple(
                doc_id
                for doc_id in prevalidated_doc_ids[choice_index]
                if doc_occurrences.get(doc_id, 0) > 1
            )
        elif raw_anchor_doc_ids is None or raw_companion_doc_ids is None:
            return None
        else:
            anchor_doc_ids = _strict_string_list(
                raw_anchor_doc_ids,
                max_items=100,
                uuid_values=True,
                require_non_empty=True,
            )
            companion_doc_ids = _strict_string_list(
                raw_companion_doc_ids,
                max_items=100,
                uuid_values=True,
            )
            if anchor_doc_ids is None or companion_doc_ids is None:
                return None
        doc_id_set = set(prevalidated_doc_ids[choice_index])
        anchor_set = set(anchor_doc_ids)
        companion_set = set(companion_doc_ids)
        if (
            not anchor_set
            or anchor_set & companion_set
            or anchor_set | companion_set != doc_id_set
            or any(doc_occurrences.get(doc_id) != 1 for doc_id in anchor_set)
            or any(doc_occurrences.get(doc_id, 0) < 2 for doc_id in companion_set)
        ):
            return None
        normalized["anchor_doc_ids"] = list(anchor_doc_ids)
        normalized["companion_doc_ids"] = list(companion_doc_ids)
        if not set(normalized["kb_ids"]).issubset(selected_kb_set):
            return None
        all_doc_ids.update(normalized["doc_ids"])
        if len(all_doc_ids) > 30:
            return None
        choices.append(normalized)
    return tuple(choices)


def _validated_evidence_pending_state(value: object) -> dict | None:
    """Validate v2 state as data only; it never grants route or retrieval access."""

    if not isinstance(value, dict):
        return None
    if (
        value.get("schema_version") != _EVIDENCE_PENDING_SCHEMA
        or value.get("kind") != _EVIDENCE_PENDING_KIND
        or value.get("dispatch_authorized") is not False
    ):
        return None
    try:
        state_id = str(uuid.UUID(str(value.get("state_id") or "")))
        base_user_message_id = str(
            uuid.UUID(str(value.get("base_user_message_id") or ""))
        )
        clarification_message_id = str(
            uuid.UUID(str(value.get("clarification_message_id") or ""))
        )
    except (TypeError, ValueError, AttributeError):
        return None
    original_query = value.get("original_query")
    clarification_message = value.get("clarification_message")
    dimension = value.get("dimension")
    selection_mode = value.get("selection_mode")
    if selection_mode is None and isinstance(value.get("choices"), list) and value["choices"]:
        # Compatibility for bounded v2 states created before selection_mode
        # was introduced. Broad states must always declare refine explicitly.
        selection_mode = "choice"
    created_at = _parsed_datetime(value.get("created_at"))
    expires_at = _future_expiry(value.get("expires_at"))
    if (
        not isinstance(original_query, str)
        or not original_query.strip()
        or len(original_query.strip()) > 12000
        or not isinstance(clarification_message, str)
        or not clarification_message.strip()
        or len(clarification_message.strip()) > 12000
        or dimension not in _EVIDENCE_DIMENSIONS
        or selection_mode not in _EVIDENCE_SELECTION_MODES
        or created_at is None
        or expires_at is None
    ):
        return None
    now = datetime.now(timezone.utc)
    if (
        created_at > now + timedelta(minutes=5)
        or expires_at <= created_at
        or expires_at - created_at > timedelta(hours=24, minutes=5)
    ):
        return None
    selected_kb_ids = _strict_string_list(
        value.get("selected_kb_ids_snapshot"),
        max_items=100,
        uuid_values=True,
        require_non_empty=True,
    )
    if selected_kb_ids is None:
        return None
    if selection_mode == "choice":
        choices = _validated_evidence_choices(
            value.get("choices"),
            selected_kb_ids=selected_kb_ids,
        )
        if choices is None:
            return None
    else:
        if value.get("choices") != []:
            return None
        choices = ()
    return {
        "schema_version": _EVIDENCE_PENDING_SCHEMA,
        "kind": _EVIDENCE_PENDING_KIND,
        "state_id": state_id,
        "base_user_message_id": base_user_message_id,
        "clarification_message_id": clarification_message_id,
        "original_query": original_query.strip(),
        "dimension": dimension,
        "selection_mode": selection_mode,
        "choices": [dict(choice) for choice in choices],
        "clarification_message": clarification_message.strip(),
        "selected_kb_ids_snapshot": list(selected_kb_ids),
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "dispatch_authorized": False,
    }


def _active_pending_route_state(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") == _EVIDENCE_PENDING_SCHEMA:
        return _validated_evidence_pending_state(value)
    if value.get("schema_version") != "rag_pending_clarification.v1":
        return None
    if not str(value.get("state_id") or "").strip():
        return None
    # A pending state is never an execution grant.  Missing or truthy values
    # are treated as malformed rather than being interpreted permissively.
    if value.get("dispatch_authorized") is not False:
        return None
    return value if _future_expiry(value.get("expires_at")) is not None else None


def _normalized_choice_text(value: object) -> str:
    return re.sub(
        r"[\s\-—_·,，。!！?？:：;；、《》〈〉()（）\[\]【】'\"]+",
        "",
        str(value or ""),
    ).casefold()


def _evidence_choice_aliases(choice: dict) -> set[str]:
    aliases = {
        _normalized_choice_text(choice.get("key")),
        _normalized_choice_text(choice.get("label")),
    }
    for field in _CHOICE_TEXT_FIELDS:
        aliases.update(
            _normalized_choice_text(item)
            for item in choice.get(field, [])
            if isinstance(item, str)
        )
    aliases.discard("")
    return aliases


def _is_pure_evidence_scope_selection(text: str, choice: dict) -> bool:
    """Distinguish a short choice from a new question naming that choice."""

    normalized = _normalized_choice_text(text)
    aliases = _evidence_choice_aliases(choice)
    if not normalized or not aliases:
        return False
    if normalized in aliases:
        return True

    residual = normalized
    for alias in sorted(aliases, key=len, reverse=True):
        residual = residual.replace(alias, "")
    selection_fillers = re.compile(
        r"(?:选择|查询|看看|使用|按照|确认|确定|想要|需要|"
        r"我|请|想|要|选|就|查|看|用|按|这个|那个|该|这篇|那篇|"
        r"版本|版|产品|项目|范围|的|吧|呀|啊|呢|哦|哈)",
        re.IGNORECASE,
    )
    return not selection_fillers.sub("", residual)


def _parse_evidence_scope_reply(
    question: str,
    pending_state: dict,
) -> _EvidenceScopeReply:
    """Resolve a bounded reply without treating the pending JSON as authority."""

    text = str(question or "").strip()
    choices = tuple(
        dict(choice)
        for choice in pending_state.get("choices", [])
        if isinstance(choice, dict)
    )
    # Pending clarifications are often answered conversationally (for example
    # ``2吧`` / ``c2。`` / ``第二个吧``). Strip only trailing particles and
    # punctuation for bounded index recognition.
    index_text = re.sub(r"(?:吧|呀|啊|呢|哦|哈|[。！!])+$", "", text).strip()
    if _CANCEL_EVIDENCE_SCOPE_RE.fullmatch(text):
        return _EvidenceScopeReply("cancel")
    if pending_state.get("selection_mode") == "refine":
        if (
            _INVALID_EVIDENCE_SELECTION_RE.fullmatch(text)
            or re.fullmatch(r"(?:c\s*)?\d+", index_text, re.IGNORECASE)
            or re.fullmatch(
                r"第\s*(?:[1-9]\d*|[一二三四五六七八九十])\s*(?:项|个|版本)?",
                index_text,
            )
        ):
            return _EvidenceScopeReply("repeat")
        if _ALL_EVIDENCE_SCOPES_RE.fullmatch(text):
            return _EvidenceScopeReply("refine")
        if _SUBSTANTIVE_NEW_QUESTION_RE.search(text):
            return _EvidenceScopeReply("new_question")
        if (
            _EXPLICIT_NEW_QUESTION_RE.search(text) or len(text) > 32
        ) and not _EVIDENCE_REFINEMENT_HINT_RE.search(text):
            return _EvidenceScopeReply("new_question")
        return _EvidenceScopeReply("refine")
    if _ALL_EVIDENCE_SCOPES_RE.fullmatch(text):
        return _EvidenceScopeReply("compare_all", choices)

    index_match = re.fullmatch(
        r"(?:选(?:择)?\s*)?(?:c\s*)?([1-9]\d*)(?:\s*(?:项|个|版本))?",
        index_text,
        re.IGNORECASE,
    ) or re.fullmatch(
        r"第\s*([1-9]\d*)\s*(?:项|个|版本)?",
        index_text,
    )
    chinese_index_match = re.fullmatch(
        r"第\s*([一二三四五六七八九十])\s*(?:项|个|版本)?",
        index_text,
    )
    selected_index: int | None = None
    if index_match:
        selected_index = int(index_match.group(1))
    elif chinese_index_match:
        selected_index = {
            "一": 1,
            "二": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }[chinese_index_match.group(1)]
    if selected_index is not None:
        index = selected_index - 1
        if 0 <= index < len(choices):
            return _EvidenceScopeReply("single", (choices[index],))
        return _EvidenceScopeReply("repeat")

    normalized_reply = _normalized_choice_text(text)
    matched: list[dict] = []
    for choice in choices:
        aliases = _evidence_choice_aliases(choice)
        if any(
            normalized_reply == alias
            or (
                len(alias) >= 3
                and alias in normalized_reply
                and (
                    alias == _normalized_choice_text(choice.get("label"))
                    or alias in {
                        _normalized_choice_text(item)
                        for item in choice.get("versions", [])
                        if isinstance(item, str)
                    }
                )
            )
            for alias in aliases
        ):
            matched.append(choice)

    if len(matched) == 1:
        if _is_pure_evidence_scope_selection(text, matched[0]):
            return _EvidenceScopeReply("single", (matched[0],))
        # A substantive question that happens to name one candidate scope is a
        # fresh request, not a lossy answer to the previous clarification.  It
        # must be routed and retrieved from the complete authorized KB scope.
        return _EvidenceScopeReply("new_question")
    if len(matched) > 1 and re.search(r"对比|比较|区别|差异|分别", text):
        # An explicit subset comparison is not the same as ``都对比``.  Keep
        # only the uniquely named choices; otherwise an unmentioned third
        # product/version/project would silently enter the document filter.
        return _EvidenceScopeReply("compare_all", tuple(matched))
    if len(matched) > 1:
        return _EvidenceScopeReply("repeat")
    if (
        _INVALID_EVIDENCE_SELECTION_RE.fullmatch(text)
        or re.fullmatch(r"(?:c\s*)?\d+", text, re.IGNORECASE)
        or re.fullmatch(r"\d+(?:\.\d+){0,4}(?:\s*版本)?", text)
    ):
        return _EvidenceScopeReply("repeat")
    if _EXPLICIT_NEW_QUESTION_RE.search(text) or len(text) > 32:
        return _EvidenceScopeReply("new_question")
    return _EvidenceScopeReply("new_question")


def _evidence_scope_filter(
    reply: _EvidenceScopeReply,
    *,
    current_kb_ids: list[uuid.UUID],
) -> dict | None:
    """Rebuild a request-local filter and intersect it with current KB scope."""

    if reply.action not in {"single", "compare_all"} or not reply.choices:
        return None
    current = {str(value) for value in current_kb_ids}
    choices: list[dict] = []
    kb_ids: set[str] = set()
    doc_ids: set[str] = set()
    for raw_choice in reply.choices:
        choice_kb_ids = [str(value) for value in raw_choice.get("kb_ids", [])]
        # Choices only carry aggregate KB/doc lists, not a doc->KB mapping.  If
        # any original KB is absent from the current request, partial filtering
        # could retain document ids from a revoked KB.  Keep the pending state
        # and ask again instead of silently narrowing an advertised choice.
        if not choice_kb_ids or not set(choice_kb_ids).issubset(current):
            return None
        choice = {
            "key": raw_choice["key"],
            "label": raw_choice["label"],
            **{
                field: list(raw_choice.get(field, []))
                for field in _CHOICE_TEXT_FIELDS
            },
            "kb_ids": choice_kb_ids,
            "doc_ids": list(raw_choice.get("doc_ids", [])),
            "anchor_doc_ids": list(raw_choice.get("anchor_doc_ids", [])),
            "companion_doc_ids": list(raw_choice.get("companion_doc_ids", [])),
        }
        choices.append(choice)
        kb_ids.update(choice_kb_ids)
        doc_ids.update(choice["doc_ids"])
    if not kb_ids or not doc_ids:
        return None
    return {
        "mode": "compare_all" if reply.action == "compare_all" else "single",
        "kb_ids": sorted(kb_ids),
        "doc_ids": sorted(doc_ids),
        "choices": choices,
    }


def _scoped_evidence_query(original_query: str, scope_filter: dict) -> str:
    labels = [
        str(choice.get("label") or "").strip()
        for choice in scope_filter.get("choices", [])
        if str(choice.get("label") or "").strip()
    ]
    if scope_filter.get("mode") == "compare_all":
        instruction = "请分别对比以下适用范围后回答：" + "；".join(labels)
    else:
        instruction = "本次只查询以下适用范围：" + "；".join(labels)
    return f"{original_query.strip()}\n{instruction}".strip()


def _refined_evidence_query(original_query: str, refinement: str) -> str:
    return (
        f"{original_query.strip()}。"
        f"用户补充的适用范围：{str(refinement or '').strip()}"
    ).strip()


def _evidence_event_pending_state(
    payload: object,
    *,
    original_query: str,
    selected_kb_ids: list[uuid.UUID],
    base_user_message_id: uuid.UUID,
    clarification_message_id: uuid.UUID,
) -> dict | None:
    """Convert a bounded-choice or broad-refinement event into pending data."""

    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema_version") != _EVIDENCE_EVENT_SCHEMA
        or payload.get("needs_clarification") is not True
        or payload.get("dimension") not in _EVIDENCE_DIMENSIONS
        or not isinstance(payload.get("question"), str)
        or not payload["question"].strip()
    ):
        return None
    selected_snapshot = [str(value) for value in selected_kb_ids]
    raw_choices = payload.get("choices")
    if raw_choices == []:
        selection_mode = "refine"
        choices: tuple[dict, ...] = ()
    else:
        selection_mode = "choice"
        validated_choices = _validated_evidence_choices(
            raw_choices,
            selected_kb_ids=tuple(selected_snapshot),
        )
        if validated_choices is None:
            return None
        choices = validated_choices
    if not selected_snapshot:
        return None
    created_at = now_utc()
    state = {
        "schema_version": _EVIDENCE_PENDING_SCHEMA,
        "kind": _EVIDENCE_PENDING_KIND,
        "state_id": str(uuid.uuid4()),
        "base_user_message_id": str(base_user_message_id),
        "clarification_message_id": str(clarification_message_id),
        "original_query": original_query.strip(),
        "dimension": payload["dimension"],
        "selection_mode": selection_mode,
        "choices": list(choices),
        "clarification_message": payload["question"].strip(),
        "selected_kb_ids_snapshot": selected_snapshot,
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(hours=24)).isoformat(),
        "dispatch_authorized": False,
    }
    return _validated_evidence_pending_state(state)


def _evidence_clarification_ack(
    *,
    conversation_id: uuid.UUID,
    pending_state: dict,
    route_state_revision: int,
) -> dict:
    """Acknowledge only a clarification state that is already durable."""

    return {
        "type": "evidence_clarification_ack",
        "schema_version": _EVIDENCE_ACK_SCHEMA,
        "persisted": True,
        "pending_state_id": pending_state["state_id"],
        "clarification_message_id": pending_state["clarification_message_id"],
        "route_state_revision": route_state_revision,
        "conversation_id": str(conversation_id),
    }


async def _route_clarification_response(
    *,
    db: AsyncSession,
    conv: Conversation,
    user: User,
    question: str,
    clarification_message: str,
    decision_reason: str,
    trace_id: str,
    selected_kb_ids: list[uuid.UUID],
    task_contract: RagTaskContract | None,
    emit_clarification_event: bool = True,
    turn: ChatTurn | None = None,
) -> StreamingResponse:
    """Persist a versioned, non-executable clarification and stream it."""

    input_route_state_revision = (
        _stored_pending_request_identity(turn)[0]
        if turn is not None
        else int(getattr(conv, "route_state_revision", 0) or 0)
    )
    if turn is not None and turn.status == "accepted":
        transition_turn(turn, "generating", trace_id=trace_id)
    user_metadata = (
        message_turn_metadata(turn, status="generating") if turn is not None else {}
    )
    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="user",
        content=question,
        **user_metadata,
    )
    if turn is not None:
        turn.user_message_id = user_msg.id
        transition_turn(
            turn,
            "generated",
            trace_id=trace_id,
            evidence_status="skipped",
            retrieval_executed=False,
            answer_content=clarification_message,
            answer_sources=[],
            search_snapshot={
                "schema_version": "rag_search_snapshot.v1",
                "candidates": [],
                "answer_sources": [],
                "counters": {
                    "retrieval_executed": False,
                    "evidence_status": "skipped",
                    "displayed_result_count": 0,
                    "answer_source_count": 0,
                    "hit_count": 0,
                    "trace_id": trace_id,
                },
            },
            tokens=0,
        )
        transition_turn(turn, "completed", assistant_message_id=uuid.uuid4())
        assistant_id = turn.assistant_message_id
        assistant_metadata = message_turn_metadata(turn, status="completed")
    else:
        assistant_id = uuid.uuid4()
        assistant_metadata = {}
    assistant_msg = Message(
        id=assistant_id,
        conversation_id=conv.id,
        role="assistant",
        content=clarification_message,
        sources=[],
        **assistant_metadata,
    )
    if turn is not None:
        for key, value in assistant_metadata.items():
            setattr(user_msg, key, value)
    created_at = now_utc()
    unresolved = (
        task_contract.clarification.unresolved if task_contract is not None else ()
    )
    conv.pending_route_state = {
        "schema_version": "rag_pending_clarification.v1",
        "state_id": str(uuid.uuid4()),
        "base_user_message_id": str(user_msg.id),
        "clarification_message_id": str(assistant_msg.id),
        "intent_code": (task_contract.intent_code if task_contract is not None else "other"),
        "unresolved": [
            {
                "role": item.role,
                "reason": item.reason,
                "candidate_count": len(item.candidate_keys),
            }
            for item in unresolved
        ],
        "selected_kb_ids_snapshot": [str(value) for value in selected_kb_ids],
        "created_at": created_at.isoformat(),
        "expires_at": (created_at + timedelta(hours=24)).isoformat(),
        # Execution authorization is deliberately not persisted.  A reply must
        # re-enter the full semantic router and compiler.
        "dispatch_authorized": False,
    }
    conv.route_state_revision = int(getattr(conv, "route_state_revision", 0) or 0) + 1
    db.add_all([user_msg, assistant_msg])
    final_pending_route_state = json.loads(
        json.dumps(conv.pending_route_state, ensure_ascii=False)
    )
    final_route_state_revision = int(conv.route_state_revision)
    if turn is not None:
        turn_values = {
            "status": turn.status,
            "trace_id": turn.trace_id,
            "evidence_status": turn.evidence_status,
            "retrieval_executed": turn.retrieval_executed,
            "error_code": turn.error_code,
            "answer_content": turn.answer_content,
            "answer_sources": list(turn.answer_sources or []),
            "search_snapshot": turn.search_snapshot,
            "tokens": turn.tokens,
            "user_message_id": turn.user_message_id,
            "assistant_message_id": turn.assistant_message_id,
            "generated_at": turn.generated_at,
            "completed_at": turn.completed_at,
            "updated_at": turn.updated_at,
            "lease_owner": None,
            "lease_expires_at": None,
        }
        message_specs = (
            {
                "id": user_msg.id,
                "conversation_id": conv.id,
                "role": "user",
                "content": question,
                "sources": None,
                "tokens": None,
                **assistant_metadata,
            },
            {
                "id": assistant_msg.id,
                "conversation_id": conv.id,
                "role": "assistant",
                "content": clarification_message,
                "sources": [],
                "tokens": None,
                **assistant_metadata,
            },
        )
        conversation_id = conv.id
        turn_id = turn.id

        async def reapply(session: AsyncSession):
            nonlocal conv, turn
            persisted_conv, persisted_turn = await _reapply_immediate_response(
                session,
                conversation_id=conversation_id,
                final_pending_route_state=final_pending_route_state,
                final_route_state_revision=final_route_state_revision,
                input_route_state_revision=input_route_state_revision,
                messages=message_specs,
                turn_id=turn_id,
                turn_values=turn_values,
            )
            conv = persisted_conv
            if persisted_turn is not None:
                turn = persisted_turn

        await commit_with_retry(db, reapply=reapply)
    else:
        await commit_with_retry(db)

    if emit_clarification_event:
        trace_event(
            "intent.clarification_created",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            decision_reason=decision_reason,
            selected_kb_count=len(selected_kb_ids),
            route_state_revision=conv.route_state_revision,
            unresolved_count=len(unresolved),
            unresolved_roles=[item.role for item in unresolved],
            unresolved_reasons=[item.reason for item in unresolved],
            **content_fields("clarification", clarification_message),
        )
    trace_event(
        "chat.response",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        evidence_status="skipped",
        retrieval_executed=False,
        decision_reason=decision_reason,
        displayed_result_count=0,
        answer_source_count=0,
        context_evidence_count=0,
        hit_count=0,
        direct_evidence_count=0,
        related_reference_count=0,
        sources=[],
        **content_fields("answer", clarification_message),
    )

    async def generate_clarification():
        events: list[dict] = [
            {
                "type": "conversation_started",
                "conversation_id": str(conv.id),
                **(
                    {"turn_id": str(turn.id), "request_id": turn.request_id}
                    if turn is not None
                    else {}
                ),
            },
        ]
        if turn is not None:
            events.append(_turn_state_event(turn))
        if task_contract is not None:
            # The clarification branch does not enter ``run_rag_stream``, so
            # publish the same contract-authoritative intent state here.  Do
            # not include the model's raw route/rationale; the deterministic
            # contract projection is sufficient for the client to show that
            # clarification is required and dispatch is forbidden.
            events.append(
                {
                    "type": "intent",
                    "decision": {
                        "intent_code": task_contract.intent_code,
                        "intent_name": task_contract.intent_name,
                        "action": task_contract.action,
                        "confidence": task_contract.confidence,
                        "source": task_contract.source,
                        "relation": task_contract.relation,
                        "readiness": task_contract.readiness,
                        "response_mode": task_contract.response_mode,
                        "retrieval_policy": task_contract.retrieval_policy,
                        "need_retrieval": task_contract.need_retrieval,
                        "dispatch_authorized": task_contract.dispatch_authorized,
                        "decision_reason": task_contract.decision_reason,
                        "task_contract": task_contract.to_dict(),
                    },
                }
            )
        events.extend(
            [
                {"type": "search_step", "step": "analyze", "status": "done"},
            {
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "total": 0,
                "displayed_result_count": 0,
                "answer_source_count": 0,
                "context_evidence_count": 0,
                "hit_count": 0,
                "retrieval_executed": False,
                "evidence_status": "skipped",
                "decision_reason": decision_reason,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
                "trace_id": trace_id,
                "is_followup": False,
                "carryover_source_count": 0,
            },
            {"type": "text_delta", "content": clarification_message},
            {"type": "done", "conversation_id": str(conv.id)},
            ]
        )
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_clarification(),
        media_type="text/event-stream",
        headers=_turn_response_headers(
            conversation_id=conv.id, trace_id=trace_id, turn=turn
        ),
    )


async def _evidence_pending_direct_response(
    *,
    db: AsyncSession,
    conv: Conversation,
    user: User,
    question: str,
    pending_state: dict,
    trace_id: str,
    action: Literal["repeat", "cancel"],
    repeat_reason: str | None = None,
    turn: ChatTurn | None = None,
) -> StreamingResponse:
    """Repeat selectable evidence scopes or cancel them without model dispatch."""

    input_route_state_revision = (
        _stored_pending_request_identity(turn)[0]
        if turn is not None
        else int(getattr(conv, "route_state_revision", 0) or 0)
    )

    if action == "cancel":
        answer = "已取消上一次资料范围选择。你可以继续提出新问题。"
        evidence_status = "skipped"
        decision_reason = "evidence_scope_selection_cancelled"
    else:
        if repeat_reason == "scope_unavailable":
            # The old labels/filenames belong to the previous KB selection and
            # must not be replayed after that request scope changes.
            answer = (
                "上一次候选资料与当前选择的知识库范围不一致。"
                "请恢复原知识库范围后重新选择，或直接提出一个新问题。"
            )
        elif repeat_reason == "route_contract_unavailable":
            answer = (
                "已识别你补充的资料范围，但本次路由状态无法安全执行。"
                "请重试刚才的选择；原范围选择已为你保留。"
            )
        else:
            answer = (
                "没有识别到有效选项，请按编号、版本或完整名称选择。\n"
                + str(pending_state["clarification_message"])
            )
        evidence_status = "needs_clarification"
        decision_reason = "evidence_scope_selection_invalid"

    if turn is not None and turn.status == "accepted":
        transition_turn(turn, "generating", trace_id=trace_id)
    user_metadata = (
        message_turn_metadata(turn, status="generating") if turn is not None else {}
    )
    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="user",
        content=question,
        **user_metadata,
    )
    if turn is not None:
        turn.user_message_id = user_msg.id
        transition_turn(
            turn,
            "generated",
            trace_id=trace_id,
            evidence_status=evidence_status,
            retrieval_executed=False,
            answer_content=answer,
            answer_sources=[],
            search_snapshot={
                "schema_version": "rag_search_snapshot.v1",
                "candidates": [],
                "answer_sources": [],
                "counters": {
                    "retrieval_executed": False,
                    "evidence_status": evidence_status,
                    "displayed_result_count": 0,
                    "answer_source_count": 0,
                    "hit_count": 0,
                    "trace_id": trace_id,
                },
            },
            tokens=0,
        )
        transition_turn(turn, "completed", assistant_message_id=uuid.uuid4())
        assistant_id = turn.assistant_message_id
        assistant_metadata = message_turn_metadata(turn, status="completed")
    else:
        assistant_id = uuid.uuid4()
        assistant_metadata = {}
    assistant_msg = Message(
        id=assistant_id,
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        sources=[],
        **assistant_metadata,
    )
    if turn is not None:
        for key, value in assistant_metadata.items():
            setattr(user_msg, key, value)
    if action == "cancel":
        current = getattr(conv, "pending_route_state", None)
        if isinstance(current, dict) and current.get("state_id") == pending_state.get("state_id"):
            conv.pending_route_state = None
            conv.route_state_revision = int(
                getattr(conv, "route_state_revision", 0) or 0
            ) + 1
    db.add_all([user_msg, assistant_msg])
    final_pending_route_state = (
        json.loads(json.dumps(conv.pending_route_state, ensure_ascii=False))
        if isinstance(getattr(conv, "pending_route_state", None), dict)
        else None
    )
    final_route_state_revision = int(
        getattr(conv, "route_state_revision", 0) or 0
    )
    if turn is not None:
        turn_values = {
            "status": turn.status,
            "trace_id": turn.trace_id,
            "evidence_status": turn.evidence_status,
            "retrieval_executed": turn.retrieval_executed,
            "error_code": turn.error_code,
            "answer_content": turn.answer_content,
            "answer_sources": list(turn.answer_sources or []),
            "search_snapshot": turn.search_snapshot,
            "tokens": turn.tokens,
            "user_message_id": turn.user_message_id,
            "assistant_message_id": turn.assistant_message_id,
            "generated_at": turn.generated_at,
            "completed_at": turn.completed_at,
            "updated_at": turn.updated_at,
            "lease_owner": None,
            "lease_expires_at": None,
        }
        message_specs = (
            {
                "id": user_msg.id,
                "conversation_id": conv.id,
                "role": "user",
                "content": question,
                "sources": None,
                "tokens": None,
                **assistant_metadata,
            },
            {
                "id": assistant_msg.id,
                "conversation_id": conv.id,
                "role": "assistant",
                "content": answer,
                "sources": [],
                "tokens": None,
                **assistant_metadata,
            },
        )
        conversation_id = conv.id
        turn_id = turn.id

        async def reapply(session: AsyncSession):
            nonlocal conv, turn
            persisted_conv, persisted_turn = await _reapply_immediate_response(
                session,
                conversation_id=conversation_id,
                final_pending_route_state=final_pending_route_state,
                final_route_state_revision=final_route_state_revision,
                input_route_state_revision=input_route_state_revision,
                messages=message_specs,
                turn_id=turn_id,
                turn_values=turn_values,
            )
            conv = persisted_conv
            if persisted_turn is not None:
                turn = persisted_turn

        await commit_with_retry(db, reapply=reapply)
    else:
        await commit_with_retry(db)

    if action == "cancel":
        trace_event(
            "evidence.clarification_resolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=pending_state["state_id"],
            resolution="cancelled",
            route_state_revision=conv.route_state_revision,
        )
    else:
        trace_event(
            "evidence.clarification_repeated",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=pending_state["state_id"],
            dimension=pending_state["dimension"],
            choice_count=len(pending_state["choices"]),
            reason=repeat_reason or "invalid_selection",
            route_state_revision=conv.route_state_revision,
        )
    trace_event(
        "chat.response",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        evidence_status=evidence_status,
        retrieval_executed=False,
        decision_reason=decision_reason,
        displayed_result_count=0,
        answer_source_count=0,
        context_evidence_count=0,
        hit_count=0,
        direct_evidence_count=0,
        related_reference_count=0,
        sources=[],
        **content_fields("answer", answer),
    )

    async def generate_direct():
        events: list[dict] = [
            {
                "type": "conversation_started",
                "conversation_id": str(conv.id),
                **(
                    {"turn_id": str(turn.id), "request_id": turn.request_id}
                    if turn is not None
                    else {}
                ),
            },
            {"type": "search_step", "step": "analyze", "status": "done"},
            {
                "type": "search_results",
                "results": [],
                "answer_sources": [],
                "total": 0,
                "displayed_result_count": 0,
                "answer_source_count": 0,
                "context_evidence_count": 0,
                "hit_count": 0,
                "retrieval_executed": False,
                "evidence_status": evidence_status,
                "decision_reason": decision_reason,
                "direct_evidence_count": 0,
                "related_reference_count": 0,
                "trace_id": trace_id,
                "is_followup": True,
                "carryover_source_count": 0,
            },
        ]
        if turn is not None:
            events.insert(1, _turn_state_event(turn))
        if action == "repeat" and repeat_reason != "scope_unavailable":
            events.append(
                {
                    "type": "evidence_clarification",
                    "schema_version": _EVIDENCE_EVENT_SCHEMA,
                    "needs_clarification": True,
                    "dimension": pending_state["dimension"],
                    "question": pending_state["clarification_message"],
                    "reason": "selection_not_recognized",
                    "choices": pending_state["choices"],
                }
            )
            # This branch commits the repeated assistant message before the
            # stream starts, and ``pending_state`` was validated as active by
            # the caller.  The client may therefore enable the choices only
            # after receiving this durable-state acknowledgement.
            events.append(
                _evidence_clarification_ack(
                    conversation_id=conv.id,
                    pending_state=pending_state,
                    route_state_revision=int(
                        getattr(conv, "route_state_revision", 0) or 0
                    ),
                )
            )
        events.extend(
            [
                {"type": "text_delta", "content": answer},
                {"type": "done", "conversation_id": str(conv.id)},
            ]
        )
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate_direct(),
        media_type="text/event-stream",
        headers=_turn_response_headers(
            conversation_id=conv.id, trace_id=trace_id, turn=turn
        ),
    )


def _parse_sse_payload(chunk: str) -> dict | None:
    if not chunk.startswith("data: "):
        return None
    try:
        payload = json.loads(chunk.removeprefix("data: ").strip())
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _bounded_source_identity_snapshot(value: object) -> dict | None:
    """Keep identity/ranking fields only; never persist producer content."""

    if not isinstance(value, dict):
        return None
    identity = _source_snapshot_identity(value)
    if identity is None:
        return None
    kb_id, doc_id, chunk_id = identity
    item: dict[str, object] = {
        "id": str(chunk_id),
        "chunk_id": str(chunk_id),
        "doc_id": str(doc_id),
        "kb_id": str(kb_id),
    }
    for key in (
        "filename",
        "file_type",
        "chunk_index",
        "score",
        "retrieval_score",
        "vector_score",
        "keyword_score",
        "trigram_score",
        "answer_support",
        "topic_relevance",
        "constraint_status",
        "constraint_reason",
        "evidence_role",
        "rerank_status",
        "rerank_reason",
        "active_channels",
    ):
        value_item = value.get(key)
        if value_item is None:
            continue
        if isinstance(value_item, (str, int, float, bool, list, dict)):
            item[key] = value_item
    return item


def _bounded_search_snapshot(payload: object) -> dict:
    """Build a bounded, content-free final search state for history/replay."""

    data = payload if isinstance(payload, dict) else {}
    candidates: list[dict] = []
    raw_results = data.get("results")
    if isinstance(raw_results, list):
        for raw in raw_results:
            item = _bounded_source_identity_snapshot(raw)
            if item is not None:
                candidates.append(item)
            if len(candidates) >= 20:
                break
    answer_sources: list[dict] = []
    raw_answer_sources = data.get("answer_sources")
    if isinstance(raw_answer_sources, list):
        for raw in raw_answer_sources:
            item = _bounded_source_identity_snapshot(raw)
            if item is not None:
                answer_sources.append(item)
            if len(answer_sources) >= 20:
                break
    counters = {}
    for key in (
        "total",
        "displayed_result_count",
        "answer_source_count",
        "context_evidence_count",
        "hit_count",
        "direct_evidence_count",
        "related_reference_count",
        "retrieval_executed",
        "evidence_status",
        "coverage_status",
        "trace_id",
    ):
        if key in data:
            counters[key] = data.get(key)
    return {
        "schema_version": "rag_search_snapshot.v1",
        "candidates": candidates,
        "answer_sources": answer_sources,
        "counters": counters,
    }


def _clarification_locked_search_results(payload: dict, *, trace_id: str) -> dict:
    """Keep every search event fail-closed after an evidence clarification.

    A custom or rolling producer may accidentally emit another ``search_results``
    after the clarification gate.  The later event may still be useful as a
    related-results panel, but it can no longer restore generation authority,
    answer sources, or direct-hit counters.
    """

    raw_results = payload.get("results")
    results: list[dict] = []
    if isinstance(raw_results, list):
        for raw in raw_results:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if item.get("evidence_role") == "direct":
                item["evidence_role"] = "related"
                item["score"] = 0.0
                item["pipeline_override_reason"] = (
                    "证据范围待用户选择，不得作为当前回答依据"
                )
            results.append(item)
    related_count = sum(
        item.get("evidence_role") == "related" for item in results
    )
    displayed_count = payload.get(
        "displayed_result_count",
        payload.get("total", len(results)),
    )
    if (
        isinstance(displayed_count, bool)
        or not isinstance(displayed_count, int)
        or displayed_count < 0
    ):
        displayed_count = len(results)
    return {
        **payload,
        "results": results,
        "answer_sources": [],
        "displayed_result_count": displayed_count,
        "answer_source_count": 0,
        "context_evidence_count": 0,
        "hit_count": 0,
        "direct_evidence_count": 0,
        "related_reference_count": related_count,
        "evidence_scope_anchor_hit": False,
        "evidence_scope_anchor_doc_ids": [],
        "evidence_status": "needs_clarification",
        "decision_reason": "evidence_scope_ambiguous",
        "trace_id": payload.get("trace_id") or trace_id,
    }


def _source_snapshot_identity(
    source: object,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID] | None:
    """读取历史来源的知识库/文档/片段标识。

    缺少任一标识时按不可披露处理。只验证 doc_id 会让文档重新分块
    后已删除的旧 content 仍从 Message.sources JSON 中返回。
    """

    if not isinstance(source, dict):
        return None
    try:
        kb_id = uuid.UUID(str(source.get("kb_id")))
        doc_id = uuid.UUID(str(source.get("doc_id")))
        chunk_id = uuid.UUID(str(source.get("id") or source.get("chunk_id")))
        # A producer may include both aliases.  They must describe the same
        # chunk; otherwise accepting the first one would make the source
        # identity ambiguous and could pair trusted metadata with another
        # chunk's content.
        raw_id = source.get("id")
        raw_chunk_id = source.get("chunk_id")
        if raw_id is not None and raw_chunk_id is not None:
            if uuid.UUID(str(raw_id)) != uuid.UUID(str(raw_chunk_id)):
                return None
    except (TypeError, ValueError, AttributeError):
        return None
    return kb_id, doc_id, chunk_id


async def _validate_stream_answer_sources(
    db: AsyncSession,
    *,
    raw_sources: object,
    raw_results: object,
    selected_kb_ids: list[uuid.UUID],
) -> tuple[list[dict], set[tuple[uuid.UUID, uuid.UUID]], str | None]:
    """Validate and refresh producer-provided answer evidence.

    ``search_results`` is an internal SSE boundary, but it is still possible
    for a rolling/custom producer to emit stale or forged source snapshots.
    Before those snapshots are persisted (or used to resolve a pending scope),
    require a complete ``kb/doc/chunk`` identity, membership in the current
    request's KB set and a live ``DocumentChunk`` joined to an active/ready
    document.  The returned source body is reloaded from the database so old
    or producer-supplied content cannot cross the boundary.

    An empty source list is a valid no-evidence result and returns ``None`` as
    the validation error.  Any non-empty list is fail-closed when the DB
    adapter is unavailable or an identity is missing.
    """

    if not isinstance(raw_sources, list):
        return [], set(), "answer_sources_not_a_list"
    if not isinstance(raw_results, list):
        raw_results = []

    allowed_kb_ids: set[uuid.UUID] = set()
    for raw_kb_id in selected_kb_ids:
        parsed_kb_id = (
            raw_kb_id
            if isinstance(raw_kb_id, uuid.UUID)
            else _parse_uuid(raw_kb_id)
        )
        if parsed_kb_id is not None:
            allowed_kb_ids.add(parsed_kb_id)

    # Empty answer evidence is expected for no-hit/error/clarification states.
    # We still do not need a database round trip in that case; the caller will
    # independently force the scope anchor state to false.
    if not raw_sources:
        return [], set(), None
    if not allowed_kb_ids:
        return [], set(), "selected_kb_ids_empty"

    parsed_sources: list[tuple[dict, tuple[uuid.UUID, uuid.UUID, uuid.UUID]]] = []
    seen_source_identities: set[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = set()
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            return [], set(), "answer_source_not_an_object"
        identity = _source_snapshot_identity(raw_source)
        if identity is None:
            return [], set(), "answer_source_identity_invalid"
        kb_id, _doc_id, _chunk_id = identity
        if kb_id not in allowed_kb_ids:
            return [], set(), "answer_source_kb_forbidden"
        if identity in seen_source_identities:
            return [], set(), "answer_source_duplicate"
        seen_source_identities.add(identity)
        role = raw_source.get("evidence_role")
        if role is not None and str(role).strip().casefold() not in {
            "direct",
            "related",
        }:
            return [], set(), "answer_source_role_invalid"
        parsed_sources.append((dict(raw_source), identity))

    # A source claimed as generation context must also be present in the
    # producer's displayed result snapshot.  This catches a common rolling
    # upgrade failure where answer_sources is populated from a previous pass.
    result_identities: set[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = set()
    for raw_result in raw_results:
        if not isinstance(raw_result, dict):
            continue
        identity = _source_snapshot_identity(raw_result)
        if identity is not None:
            result_identities.add(identity)
    if any(identity not in result_identities for _, identity in parsed_sources):
        return [], set(), "answer_source_not_in_results"

    chunk_ids = {identity[2] for _, identity in parsed_sources}
    try:
        statement = (
            select(DocumentChunk, Document)
            .join(
                Document,
                (Document.id == DocumentChunk.doc_id)
                & (Document.kb_id == DocumentChunk.kb_id),
            )
            .where(
                DocumentChunk.id.in_(chunk_ids),
                DocumentChunk.kb_id.in_(allowed_kb_ids),
                Document.is_active.is_(True),
                Document.status == "ready",
            )
        )
        result = await db.execute(statement)
        rows = result.all() if hasattr(result, "all") else []
    except Exception as exc:
        # Do not expose producer content or clear a pending scope when the
        # authorization refresh itself is unavailable.  The caller records a
        # compact reason and persists an empty source list instead.
        logger.warning(
            "[chat/evidence source validation] refresh failed error=%s",
            type(exc).__name__,
        )
        return [], set(), f"source_refresh_failed:{type(exc).__name__}"

    current: dict[tuple[uuid.UUID, uuid.UUID, uuid.UUID], tuple[DocumentChunk, Document]] = {}
    for row in rows or ():
        try:
            chunk, document = row
            if (
                document is None
                or document.is_active is not True
                or str(document.status or "").strip().casefold() != "ready"
                or document.id != chunk.doc_id
                or document.kb_id != chunk.kb_id
            ):
                continue
            identity = (chunk.kb_id, chunk.doc_id, chunk.id)
        except (TypeError, ValueError, AttributeError):
            continue
        current[identity] = (chunk, document)

    if any(identity not in current for _, identity in parsed_sources):
        return [], set(), "answer_source_not_current"

    refreshed: list[dict] = []
    answer_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for source, identity in parsed_sources:
        chunk, document = current[identity]
        refreshed.append(
            {
                **source,
                "id": str(chunk.id),
                "chunk_id": str(chunk.id),
                "doc_id": str(chunk.doc_id),
                "kb_id": str(chunk.kb_id),
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata_ or {},
                "filename": document.filename,
                "file_type": document.file_type,
                "source_url": document.source_url,
                "image_url": document.image_url,
                "doc_tags": document.tags or [],
            }
        )
        answer_pairs.add((chunk.kb_id, chunk.doc_id))
    return refreshed, answer_pairs, None


def _parse_uuid(value: object) -> uuid.UUID | None:
    """Parse one UUID without allowing malformed values into a SQL filter."""

    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def _scope_anchor_coverage_from_sources(
    scope_filter: dict | None,
    answer_pairs: set[tuple[uuid.UUID, uuid.UUID]],
) -> tuple[bool | None, list[str]]:
    """Recompute selected-scope anchors from refreshed answer sources.

    The boolean and document list emitted by a producer are advisory only.
    Pending state can be resolved only when every server-derived choice has at
    least one anchor document in the evidence set that was actually accepted
    by ``_validate_stream_answer_sources``.
    """

    if not isinstance(scope_filter, dict):
        return None, []
    anchor_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for raw_choice in scope_filter.get("choices", []):
        if not isinstance(raw_choice, dict):
            continue
        kb_ids = {
            parsed
            for value in raw_choice.get("kb_ids", [])
            if (parsed := _parse_uuid(value)) is not None
        }
        anchor_doc_ids = {
            str(value).strip()
            for value in raw_choice.get("anchor_doc_ids", [])
            if str(value).strip()
        }
        for kb_id in kb_ids:
            for raw_doc_id in anchor_doc_ids:
                doc_id = _parse_uuid(raw_doc_id)
                if doc_id is not None:
                    anchor_pairs.add((kb_id, doc_id))
    if not anchor_pairs:
        return False, []
    covered = anchor_pairs & answer_pairs
    covered_doc_ids = sorted({str(doc_id) for _, doc_id in covered})
    return covered == anchor_pairs, covered_doc_ids


def _scope_document_pairs(
    scope_filter: dict | None,
) -> set[tuple[uuid.UUID, uuid.UUID]]:
    """Expand a server-derived scope choice into exact KB/document pairs."""

    if not isinstance(scope_filter, dict):
        return set()
    pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
    for raw_choice in scope_filter.get("choices", []):
        if not isinstance(raw_choice, dict):
            continue
        kb_ids = {
            parsed
            for value in raw_choice.get("kb_ids", [])
            if (parsed := _parse_uuid(value)) is not None
        }
        doc_ids = {
            parsed
            for value in raw_choice.get("doc_ids", [])
            if (parsed := _parse_uuid(value)) is not None
        }
        pairs.update((kb_id, doc_id) for kb_id in kb_ids for doc_id in doc_ids)
    return pairs


def _source_snapshot_is_answer_evidence(source: object) -> bool:
    """Reject legacy broad-candidate snapshots that were never answer evidence.

    Older releases persisted every displayed retrieval candidate in
    ``Message.sources``.  Their shared ``evidence_status`` marker lets history
    reads fail closed for requests that had no generation context, while
    snapshots created before the marker remain backward compatible.
    """

    if not isinstance(source, dict):
        return False
    status = str(source.get("evidence_status") or "").strip().casefold()
    return status not in _NON_ANSWER_SOURCE_STATUSES


async def _historical_evidence_clarification(
    pending_route_state: object,
    *,
    route_state_revision: object,
    assistant_message_ids: set[str],
    accessible_set: set[uuid.UUID] | None,
    db: AsyncSession,
) -> dict | None:
    """Build a public picker only from the currently valid server state.

    Pending-state UUID allow-lists are routing internals, not client
    authorization.  Historical restoration therefore rechecks the user's
    current KB scope and every referenced document before exposing a reduced
    choice projection.
    """

    state = _active_pending_route_state(pending_route_state)
    if (
        state is None
        or state.get("schema_version") != _EVIDENCE_PENDING_SCHEMA
        or state.get("clarification_message_id") not in assistant_message_ids
    ):
        return None
    try:
        revision = int(route_state_revision or 0)
        selected_kb_ids = {
            uuid.UUID(value) for value in state["selected_kb_ids_snapshot"]
        }
    except (TypeError, ValueError, AttributeError):
        return None
    if revision < 0:
        return None
    if accessible_set is not None and not selected_kb_ids.issubset(accessible_set):
        return None

    choices = state["choices"]
    referenced_doc_ids = {
        uuid.UUID(value)
        for choice in choices
        for value in choice["doc_ids"]
    }
    if referenced_doc_ids:
        statement = select(Document.id, Document.kb_id).where(
            Document.id.in_(referenced_doc_ids),
            Document.kb_id.in_(selected_kb_ids),
            Document.is_active.is_(True),
            Document.status == "ready",
        )
        if accessible_set is not None:
            statement = statement.where(Document.kb_id.in_(accessible_set))
        current_doc_kbs = {
            doc_id: kb_id
            for doc_id, kb_id in (await db.execute(statement)).all()
        }
        if set(current_doc_kbs) != referenced_doc_ids:
            return None
        for choice in choices:
            choice_kb_ids = {uuid.UUID(value) for value in choice["kb_ids"]}
            if any(
                current_doc_kbs[uuid.UUID(value)] not in choice_kb_ids
                for value in choice["doc_ids"]
            ):
                return None

    public_choices = []
    for choice in choices:
        public_choice = {}
        for field in _PUBLIC_CHOICE_FIELDS:
            value = choice[field]
            public_choice[field] = list(value) if isinstance(value, list) else value
        public_choices.append(public_choice)
    return {
        "schema_version": _EVIDENCE_EVENT_SCHEMA,
        "needs_clarification": True,
        "dimension": state["dimension"],
        "question": state["clarification_message"],
        "choices": public_choices,
        "acknowledged": True,
        "persisted": True,
        "pending_state_id": state["state_id"],
        "clarification_message_id": state["clarification_message_id"],
        "route_state_revision": revision,
    }


async def _messages_with_current_source_scope(
    rows: list[Message],
    *,
    user: User,
    db: AsyncSession,
    pending_route_state: object = None,
    route_state_revision: object = 0,
) -> list[MessageOut]:
    """按当前角色范围和文档状态过滤历史 ``sources`` 快照。

    assistant 正文属于用户自己的既有会话记录；额外展开的原始检索片段则必须
    每次按当前 RBAC 重新授权，防止角色范围被撤销或文档停用后仍从 JSONB 快照
    读取 ``content`` / ``source_url``。
    """

    accessible = await get_accessible_kb_ids(user, db)
    accessible_set = set(accessible) if accessible is not None else None
    historical_clarification = await _historical_evidence_clarification(
        pending_route_state,
        route_state_revision=route_state_revision,
        assistant_message_ids={
            str(row.id) for row in rows if row.role == "assistant"
        },
        accessible_set=accessible_set,
        db=db,
    )
    referenced_sources: set[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = set()
    for row in rows:
        for source in (row.sources if isinstance(row.sources, list) else ()):
            if not _source_snapshot_is_answer_evidence(source):
                continue
            identity = _source_snapshot_identity(source)
            if identity is None:
                continue
            kb_id, _, _ = identity
            if accessible_set is not None and kb_id not in accessible_set:
                continue
            referenced_sources.add(identity)
        raw_snapshot = getattr(row, "search_snapshot", None)
        if isinstance(raw_snapshot, dict):
            for collection_key in ("candidates", "answer_sources"):
                collection = raw_snapshot.get(collection_key)
                if not isinstance(collection, list):
                    continue
                for source in collection[:20]:
                    identity = _source_snapshot_identity(source)
                    if identity is None:
                        continue
                    kb_id, _, _ = identity
                    if accessible_set is not None and kb_id not in accessible_set:
                        continue
                    referenced_sources.add(identity)

    current_sources: dict[tuple[uuid.UUID, uuid.UUID, uuid.UUID], dict] = {}
    if referenced_sources:
        chunk_ids = {chunk_id for _, _, chunk_id in referenced_sources}
        statement = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.doc_id)
            .where(
                DocumentChunk.id.in_(chunk_ids),
                Document.kb_id == DocumentChunk.kb_id,
                Document.is_active.is_(True),
                Document.status == "ready",
            )
        )
        if accessible_set is not None:
            statement = statement.where(DocumentChunk.kb_id.in_(accessible_set))
        for chunk, document in (await db.execute(statement)).all():
            identity = (chunk.kb_id, chunk.doc_id, chunk.id)
            if identity not in referenced_sources:
                continue
            # 排名、证据角色与分数保留当轮快照；可披露的文档内容和
            # 元数据始终从当前有效 chunk 重载，避免返回已删除或已更新的旧片段。
            current_sources[identity] = {
                "id": str(chunk.id),
                "chunk_id": str(chunk.id),
                "doc_id": str(chunk.doc_id),
                "kb_id": str(chunk.kb_id),
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata_ or {},
                "filename": document.filename,
                "file_type": document.file_type,
                "source_url": document.source_url,
                "image_url": document.image_url,
                "doc_tags": document.tags or [],
            }

    output: list[MessageOut] = []
    for row in rows:
        visible_sources = []
        if isinstance(row.sources, list):
            for source in row.sources:
                if not _source_snapshot_is_answer_evidence(source):
                    continue
                identity = _source_snapshot_identity(source)
                current = current_sources.get(identity) if identity else None
                if current is not None:
                    visible_sources.append({**dict(source), **current})
        visible_search_snapshot = None
        raw_search_snapshot = getattr(row, "search_snapshot", None)
        if isinstance(raw_search_snapshot, dict):
            counters = raw_search_snapshot.get("counters")
            safe_counters = {}
            if isinstance(counters, dict):
                for key in (
                    "total",
                    "displayed_result_count",
                    "answer_source_count",
                    "context_evidence_count",
                    "hit_count",
                    "direct_evidence_count",
                    "related_reference_count",
                    "retrieval_executed",
                    "evidence_status",
                    "coverage_status",
                    "trace_id",
                ):
                    value = counters.get(key)
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        safe_counters[key] = value

            def refreshed_snapshot_items(key: str) -> list[dict]:
                refreshed_items: list[dict] = []
                raw_items = raw_search_snapshot.get(key)
                if not isinstance(raw_items, list):
                    return refreshed_items
                for raw_item in raw_items[:20]:
                    safe_identity = _bounded_source_identity_snapshot(raw_item)
                    identity = _source_snapshot_identity(safe_identity)
                    current = current_sources.get(identity) if identity else None
                    if safe_identity is not None and current is not None:
                        # ``safe_identity`` contains only bounded ranking and
                        # identity fields; every body/URL/name comes from the
                        # current authorized active+ready document row.
                        refreshed_items.append({**safe_identity, **current})
                return refreshed_items

            visible_search_snapshot = {
                "schema_version": "rag_search_snapshot.v1",
                "candidates": refreshed_snapshot_items("candidates"),
                "answer_sources": refreshed_snapshot_items("answer_sources"),
                # Keep outcome/counters even when current authorization removes
                # every candidate so the UI can still distinguish no-hit from
                # a historical hit whose evidence is no longer accessible.
                "counters": safe_counters,
            }
        # 历史脏数据可能把 sources 存成 dict；不先让 Pydantic 验证 ORM
        # 对象，否则一条异常消息会让整个会话返回 500。
        serialized = MessageOut(
            id=row.id,
            conversation_id=row.conversation_id,
            role=row.role,
            content=row.content,
            sources=visible_sources if isinstance(row.sources, list) else None,
            clarification=(
                historical_clarification
                if historical_clarification is not None
                and historical_clarification["clarification_message_id"]
                == str(row.id)
                else None
            ),
            tokens=row.tokens,
            turn_id=getattr(row, "turn_id", None),
            request_id=getattr(row, "request_id", None),
            status=getattr(row, "turn_status", None),
            turn_status=getattr(row, "turn_status", None),
            trace_id=getattr(row, "trace_id", None),
            evidence_status=getattr(row, "evidence_status", None),
            retrieval_executed=getattr(row, "retrieval_executed", None),
            error_code=getattr(row, "error_code", None),
            delivery_status=getattr(row, "delivery_status", None),
            persistence_status=getattr(row, "persistence_status", None),
            duration_ms=getattr(row, "duration_ms", None),
            search_snapshot=visible_search_snapshot,
            created_at=row.created_at,
        )
        output.append(serialized)
    return output


def _public_stream_error_message(exc: BaseException) -> str:
    """返回可安全展示给前端的生成错误。

    详细异常已按 trace_id 写入服务端日志；不把上游 URL、响应体或
    请求信息通过 SSE 直接暴露给终端用户。
    """

    error_name = type(exc).__name__.casefold()
    if "timeout" in error_name or isinstance(exc, TimeoutError):
        return "模型服务响应超时，请稍后重试"
    return "回答生成失败，请稍后重试"


def _durable_turn_supported(session: object) -> bool:
    """Whether the injected session is a real enough session for turn ledger.

    A handful of legacy unit tests use tiny ``add/commit`` doubles that predate
    the turn protocol.  Production ``AsyncSession`` always exposes ``execute``
    and ``flush``; retaining this capability check keeps those tests focused on
    SSE parsing while the real path remains fully durable.
    """

    return callable(getattr(session, "execute", None)) and callable(
        getattr(session, "flush", None)
    )


def _pending_route_identity(conv: Conversation) -> tuple[int, str | None]:
    revision = max(0, int(getattr(conv, "route_state_revision", 0) or 0))
    pending = _active_pending_route_state(
        getattr(conv, "pending_route_state", None)
    )
    return revision, (str(pending["state_id"]) if pending is not None else None)


def _request_context_for_payload(
    payload: ChatRequest,
    *,
    conversation_id: uuid.UUID,
    pending_route_revision: int,
    pending_state_id: str | None,
) -> dict:
    return build_turn_request_context(
        question=payload.question,
        conversation_id=conversation_id,
        knowledge_base_ids=payload.knowledge_base_ids,
        search_config=payload.search_config.model_dump(),
        pending_route_revision=pending_route_revision,
        pending_state_id=pending_state_id,
    )


def _stored_pending_request_identity(turn: ChatTurn) -> tuple[int, str | None]:
    context = getattr(turn, "request_context", None)
    pending = context.get("pending_route") if isinstance(context, dict) else None
    if not isinstance(pending, dict):
        return 0, None
    try:
        revision = max(0, int(pending.get("revision") or 0))
    except (TypeError, ValueError):
        revision = 0
    state_id = str(pending.get("state_id") or "").strip() or None
    return revision, state_id


def _stored_resume_identity(turn: ChatTurn) -> tuple[int, str | None] | None:
    context = getattr(turn, "resume_context", None)
    if not isinstance(context, dict):
        return None
    try:
        revision = max(0, int(context.get("revision") or 0))
    except (TypeError, ValueError):
        return None
    state_id = str(context.get("state_id") or "").strip() or None
    return revision, state_id


async def _reapply_immediate_response(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    final_pending_route_state: dict | None,
    final_route_state_revision: int,
    input_route_state_revision: int,
    messages: tuple[dict, ...],
    turn_id: uuid.UUID | None,
    turn_values: dict | None,
) -> tuple[Conversation, ChatTurn | None]:
    """Reconstruct an immediate clarification/direct-response transaction.

    This callback runs only after a failed commit and rollback.  Every ORM row
    is loaded again; repeated invocation is an idempotent upsert rather than an
    empty commit of detached objects.
    """

    persisted_turn: ChatTurn | None = None
    if turn_id is not None:
        persisted_turn = await session.get(
            ChatTurn,
            turn_id,
            with_for_update=True,
        )
        if not isinstance(persisted_turn, ChatTurn):
            raise RuntimeError("chat turn 不存在，无法重试保存即时回答")
    persisted_conv = await session.get(
        Conversation,
        conversation_id,
        with_for_update=True,
    )
    if not isinstance(persisted_conv, Conversation):
        raise RuntimeError("会话不存在，无法重试保存即时回答")
    if persisted_turn is not None:
        if persisted_turn.status != "completed":
            persisted_revision = int(
                getattr(persisted_conv, "route_state_revision", 0) or 0
            )
            if persisted_revision != input_route_state_revision:
                raise RuntimeError("会话范围状态已变化，拒绝覆盖重试")

    # A commit acknowledgement can fail after PostgreSQL already committed.
    # In that case the completed turn is authoritative and no state is replayed.
    already_completed = bool(
        persisted_turn is not None and persisted_turn.status == "completed"
    )
    if not already_completed:
        persisted_conv.pending_route_state = final_pending_route_state
        persisted_conv.route_state_revision = final_route_state_revision
        if persisted_turn is not None and turn_values is not None:
            for key, value in turn_values.items():
                setattr(persisted_turn, key, value)

    for spec in messages:
        message_id = spec["id"]
        persisted_message = await session.get(Message, message_id)
        if not isinstance(persisted_message, Message):
            persisted_message = Message(**spec)
            session.add(persisted_message)
        elif not already_completed:
            for key, value in spec.items():
                if key != "id":
                    setattr(persisted_message, key, value)
    return persisted_conv, persisted_turn


def _turn_state_event(turn: ChatTurn, *, replayed: bool = False) -> dict:
    persistence_status = (
        "completed"
        if turn.status == "completed"
        else (
            "failed"
            if turn.status in {"persist_failed", "failed"}
            else turn.status
        )
    )
    return {
        "type": "turn_state",
        "turn_id": str(turn.id),
        "request_id": turn.request_id,
        "status": turn.status,
        "trace_id": turn.trace_id,
        "evidence_status": turn.evidence_status,
        "retrieval_executed": turn.retrieval_executed,
        "error_code": turn.error_code,
        "duration_ms": turn_duration_ms(turn),
        "persistence_status": persistence_status,
        "same_request_recoverable": bool(
            turn.status in RECOVERABLE_TURN_STATUSES
            and turn.answer_content is not None
        ),
        "replayed": replayed,
    }


def _turn_response_headers(
    *,
    conversation_id: uuid.UUID,
    trace_id: str | None,
    turn: ChatTurn | None,
) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
        "X-Conversation-ID": str(conversation_id),
    }
    if trace_id:
        headers["X-RAG-Trace-ID"] = str(trace_id)
    if turn is not None:
        headers["X-RAG-Request-ID"] = str(turn.request_id)
        headers["X-RAG-Turn-ID"] = str(turn.id)
    return headers


def _turn_replay_response(
    *,
    conv: Conversation,
    turn: ChatTurn,
    status_code: int = 200,
    replayed: bool = True,
) -> StreamingResponse:
    """Replay a durable result without invoking routing/retrieval/model code."""

    async def stream():
        yield (
            "data: "
            + json.dumps(
                {
                    "type": "conversation_started",
                    "conversation_id": str(conv.id),
                    "turn_id": str(turn.id),
                    "request_id": turn.request_id,
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )
        yield "data: " + json.dumps(
            _turn_state_event(turn, replayed=replayed), ensure_ascii=False
        ) + "\n\n"
        if turn.status in {"failed", "cancelled", "persist_failed"}:
            same_request_recoverable = bool(
                turn.status == "persist_failed"
                and turn.answer_content is not None
            )
            if same_request_recoverable:
                message = (
                    "回答已生成但仍未完成保存，请稍后使用相同 request_id 重试"
                )
            elif turn.status == "cancelled":
                message = "该请求已取消，请重新发送并使用新的 request_id"
            else:
                message = (
                    "该请求未能完成，且没有可恢复的已生成回答。"
                    "请重新发送问题并使用新的 request_id。"
                )
            yield "data: " + json.dumps(
                {
                    "type": "error",
                    "message": message,
                    "turn_id": str(turn.id),
                    "request_id": turn.request_id,
                    "status": turn.status,
                    "error_code": turn.error_code,
                    "persistence_status": (
                        "failed" if turn.status != "cancelled" else "cancelled"
                    ),
                    "same_request_recoverable": same_request_recoverable,
                    "retry_with_new_request_id": not same_request_recoverable,
                    "replayed": replayed,
                },
                ensure_ascii=False,
            ) + "\n\n"
        if turn.status not in {"failed", "cancelled"} and turn.answer_content:
            sources = (
                list(turn.answer_sources or [])
                if isinstance(turn.answer_sources, list)
                else []
            )
            snapshot = (
                turn.search_snapshot
                if isinstance(turn.search_snapshot, dict)
                else {}
            )
            candidates = snapshot.get("candidates")
            if not isinstance(candidates, list):
                candidates = sources
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "search_results",
                        "results": candidates,
                        "answer_sources": sources,
                        "total": len(candidates),
                        "displayed_result_count": len(candidates),
                        "answer_source_count": len(sources),
                        "context_evidence_count": len(sources),
                        "hit_count": sum(
                            str(item.get("evidence_role") or "").casefold() == "direct"
                            for item in sources
                            if isinstance(item, dict)
                        ),
                        "retrieval_executed": turn.retrieval_executed,
                        "evidence_status": turn.evidence_status,
                        "trace_id": turn.trace_id,
                        "replayed": replayed,
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps(
                    {"type": "text_delta", "content": turn.answer_content},
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        if turn.tokens is not None:
            yield "data: " + json.dumps(
                {"type": "usage", "total_tokens": turn.tokens}, ensure_ascii=False
            ) + "\n\n"
        yield "data: " + json.dumps(
            {
                "type": "done",
                "conversation_id": str(conv.id),
                "turn_id": str(turn.id),
                "request_id": turn.request_id,
                "status": turn.status,
                "replayed": replayed,
            },
            ensure_ascii=False,
        ) + "\n\n"

    return StreamingResponse(
        stream(),
        status_code=status_code,
        media_type="text/event-stream",
        headers=_turn_response_headers(
            conversation_id=conv.id, trace_id=turn.trace_id, turn=turn
        ),
    )


async def _recover_turn_answer(
    *,
    conv: Conversation,
    turn: ChatTurn,
) -> ChatTurn:
    """Finish a generated/persist-failed turn in a fresh retryable session."""

    from database import AsyncSessionLocal

    last_error: BaseException | None = None
    for attempt in range(MAX_PERSIST_ATTEMPTS):
        try:
            async with AsyncSessionLocal() as save_db:
                persisted = await save_db.get(ChatTurn, turn.id)
                if not isinstance(persisted, ChatTurn):
                    # Lightweight test doubles may not model the new table;
                    # use the already-loaded object and keep the operation
                    # idempotent for them.
                    persisted = turn
                if persisted.status == "completed":
                    return persisted
                if persisted.status not in RECOVERABLE_TURN_STATUSES:
                    return persisted
                if persisted.answer_content is None:
                    raise RuntimeError("turn 缺少可恢复的已生成回答")
                assistant_id = persisted.assistant_message_id or uuid.uuid4()
                persisted.assistant_message_id = assistant_id
                transition_turn(
                    persisted,
                    "completed",
                    assistant_message_id=assistant_id,
                )
                persisted.error_code = None
                existing = await save_db.get(Message, assistant_id)
                metadata = message_turn_metadata(persisted, status="completed")
                if persisted.user_message_id is not None:
                    persisted_user = await save_db.get(
                        Message, persisted.user_message_id
                    )
                    if isinstance(persisted_user, Message):
                        for key, value in metadata.items():
                            setattr(persisted_user, key, value)
                if isinstance(existing, Message):
                    existing.content = persisted.answer_content
                    existing.sources = list(persisted.answer_sources or [])
                    existing.tokens = persisted.tokens
                    for key, value in metadata.items():
                        setattr(existing, key, value)
                else:
                    save_db.add(
                        Message(
                            id=assistant_id,
                            conversation_id=conv.id,
                            role="assistant",
                            content=persisted.answer_content,
                            sources=list(persisted.answer_sources or []),
                            tokens=persisted.tokens,
                            **metadata,
                        )
                    )
                await save_db.commit()
                turn.status = persisted.status
                turn.assistant_message_id = persisted.assistant_message_id
                turn.completed_at = persisted.completed_at
                turn.updated_at = persisted.updated_at
                turn.error_code = persisted.error_code
                return turn
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_PERSIST_ATTEMPTS:
                await asyncio.sleep(0)
    assert last_error is not None
    raise last_error


async def _mark_turn_persist_failed(
    *,
    turn: ChatTurn,
    trace_id: str,
    error_code: str = "assistant_persistence_failed",
) -> bool:
    """Mark a staged payload recoverable and return whether disk proves it.

    An in-memory ``generated`` object is not recovery evidence: the process may
    have failed before that payload reached PostgreSQL.  Retry guidance is
    emitted only when this function commits a row whose answer payload exists.
    """

    from database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as save_db:
            persisted = await save_db.get(ChatTurn, turn.id)
            if isinstance(persisted, ChatTurn):
                if persisted.status != "completed":
                    # ``generated`` and ``persist_failed`` are the only states
                    # that can retain an answer for recovery.
                    if (
                        persisted.status == "generated"
                        and persisted.answer_content is not None
                    ):
                        transition_turn(
                            persisted,
                            "persist_failed",
                            trace_id=trace_id,
                            error_code=error_code,
                        )
                    elif (
                        persisted.status == "persist_failed"
                        and persisted.answer_content is not None
                    ):
                        persisted.error_code = error_code
                        persisted.trace_id = trace_id
                    elif persisted.status in {
                        "accepted",
                        "generating",
                        "generated",
                        "persist_failed",
                    }:
                        # The generated payload itself could not be staged, so
                        # there is nothing a same-id retry can safely recover.
                        transition_turn(
                            persisted,
                            "failed",
                            trace_id=trace_id,
                            error_code="generated_payload_not_persisted",
                        )
                await save_db.commit()
                turn.status = persisted.status
                turn.error_code = persisted.error_code
                turn.trace_id = persisted.trace_id
                turn.answer_content = persisted.answer_content
                turn.generated_at = persisted.generated_at
                return bool(
                    persisted.status in RECOVERABLE_TURN_STATUSES
                    and persisted.answer_content is not None
                )
            else:
                # A missing row (including a lightweight test double) cannot
                # prove that the generated payload survived the process.
                turn.status = "failed"
                turn.error_code = "generated_payload_not_persisted"
                turn.trace_id = trace_id
                return False
    except Exception as exc:
        log_exception_safely(
            logger,
            "[chat/turn persist-failed marker] turn=%s trace=%s",
            turn.id,
            trace_id,
            exc=exc,
        )
    return False


async def _mark_turn_terminal(
    *,
    turn: ChatTurn | None,
    status: Literal["failed", "cancelled"],
    trace_id: str,
    error_code: str,
    evidence_status: str | None = None,
    retrieval_executed: bool | None = None,
) -> None:
    if turn is None:
        return
    from database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as save_db:
            persisted = await save_db.get(ChatTurn, turn.id)
            if not isinstance(persisted, ChatTurn):
                persisted = turn
            if persisted.status in {"accepted", "generating"}:
                transition_turn(
                    persisted,
                    status,
                    trace_id=trace_id,
                    evidence_status=evidence_status,
                    retrieval_executed=retrieval_executed,
                    error_code=error_code,
                )
                await save_db.commit()
            turn.status = persisted.status
            turn.error_code = persisted.error_code
            turn.evidence_status = persisted.evidence_status
            turn.retrieval_executed = persisted.retrieval_executed
    except Exception as exc:
        log_exception_safely(
            logger,
            "[chat/turn terminal marker] turn=%s trace=%s status=%s",
            turn.id,
            trace_id,
            status,
            exc=exc,
        )


async def _stage_generated_turn(
    *,
    turn: ChatTurn,
    conv: Conversation,
    answer: str,
    sources: list,
    tokens: int | None,
    evidence_status: str | None,
    retrieval_executed: bool | None,
    trace_id: str,
    search_snapshot: dict | None = None,
) -> ChatTurn:
    """Durably stage the answer before the assistant message transaction."""

    from database import AsyncSessionLocal

    last_error: BaseException | None = None
    for attempt in range(MAX_PERSIST_ATTEMPTS):
        try:
            async with AsyncSessionLocal() as save_db:
                persisted = await save_db.get(ChatTurn, turn.id)
                if not isinstance(persisted, ChatTurn):
                    persisted = turn
                if persisted.status == "completed":
                    return persisted
                if persisted.status not in {"accepted", "generating", "generated", "persist_failed"}:
                    return persisted
                if persisted.status in {"accepted", "generating"}:
                    transition_turn(persisted, "generated")
                elif persisted.status == "persist_failed":
                    transition_turn(persisted, "generated")
                # ``generated`` -> ``generated`` is intentionally avoided by
                # transition_turn; update its payload directly on retries.
                persisted.answer_content = answer
                persisted.answer_sources = list(sources or [])
                persisted.search_snapshot = search_snapshot
                persisted.tokens = tokens
                persisted.evidence_status = evidence_status
                persisted.retrieval_executed = retrieval_executed
                persisted.trace_id = trace_id
                persisted.error_code = None
                persisted.assistant_message_id = (
                    persisted.assistant_message_id or uuid.uuid4()
                )
                persisted.persistence_attempts = int(
                    getattr(persisted, "persistence_attempts", 0) or 0
                ) + 1
                await save_db.commit()
                turn.status = persisted.status
                turn.answer_content = persisted.answer_content
                turn.answer_sources = persisted.answer_sources
                turn.search_snapshot = persisted.search_snapshot
                turn.tokens = persisted.tokens
                turn.evidence_status = persisted.evidence_status
                turn.retrieval_executed = persisted.retrieval_executed
                turn.trace_id = persisted.trace_id
                turn.assistant_message_id = persisted.assistant_message_id
                turn.generated_at = persisted.generated_at
                turn.updated_at = persisted.updated_at
                return turn
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_PERSIST_ATTEMPTS:
                await asyncio.sleep(0)
    assert last_error is not None
    raise last_error


async def _existing_turn_response(
    *,
    conv: Conversation,
    turn: ChatTurn,
    user: User,
    db: AsyncSession,
) -> StreamingResponse:
    """Resolve duplicate request ids without dispatching retrieval again."""

    if turn.status in RECOVERABLE_TURN_STATUSES:
        try:
            turn = await _recover_turn_answer(conv=conv, turn=turn)
        except Exception as exc:
            await _mark_turn_persist_failed(
                turn=turn,
                trace_id=turn.trace_id or uuid.uuid4().hex,
            )
            trace_event(
                "chat.persistence_error",
                trace_id=turn.trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                operation="recover_turn_answer",
                error=exc,
            )
    # A retry may arrive after the user's KB scope changed or a document was
    # disabled/re-chunked.  Reuse the history authorization boundary so the
    # durable identity snapshot is reloaded from current active+ready rows;
    # never replay stale turn JSON directly.
    replay_row = Message(
        id=turn.assistant_message_id or uuid.uuid4(),
        conversation_id=conv.id,
        role="assistant",
        content=turn.answer_content or "",
        sources=list(turn.answer_sources or []),
        tokens=turn.tokens,
        turn_id=turn.id,
        request_id=turn.request_id,
        turn_status=turn.status,
        trace_id=turn.trace_id,
        evidence_status=turn.evidence_status,
        retrieval_executed=turn.retrieval_executed,
        error_code=turn.error_code,
        delivery_status=("delivered" if turn.status == "completed" else "pending"),
        persistence_status=(
            "completed" if turn.status == "completed" else turn.status
        ),
        duration_ms=turn_duration_ms(turn),
        search_snapshot=turn.search_snapshot,
        created_at=turn.completed_at or turn.updated_at or turn.created_at or now_utc(),
    )
    visible = await _messages_with_current_source_scope(
        [replay_row],
        user=user,
        db=db,
    )
    if visible:
        turn.answer_sources = list(visible[0].sources or [])
        turn.search_snapshot = visible[0].search_snapshot
    status_code = 200
    if turn.status in {"accepted", "generating"}:
        status_code = 202
    return _turn_replay_response(
        conv=conv,
        turn=turn,
        status_code=status_code,
        replayed=True,
    )


@router.post("/send")
async def send_message(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    trace_id = uuid.uuid4().hex
    request_search_config = payload.search_config.model_dump()
    try:
        request_id = normalize_request_id(payload.request_id)
        requested_turn_id = normalize_turn_id(payload.turn_id) if payload.turn_id else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # 流式开始前校验请求的知识库都在可访问范围内（accessible 为 None 表示全部）
    accessible = await get_accessible_kb_ids(user, db)
    if accessible is not None and not set(payload.knowledge_base_ids).issubset(set(accessible)):
        raise HTTPException(status_code=403, detail="无权访问部分知识库")

    durable_turn_enabled = _durable_turn_supported(db)
    durable_turn: ChatTurn | None = None
    conv: Conversation | None = None
    if durable_turn_enabled:
        existing_user_turn = await find_turn_for_user(db, user.id, request_id)
        if existing_user_turn is not None:
            if (
                requested_turn_id is not None
                and requested_turn_id != existing_user_turn.id
            ):
                raise HTTPException(
                    status_code=409,
                    detail="同一 request_id 对应的 turn_id 不一致",
                )
            if (
                payload.conversation_id is not None
                and payload.conversation_id != existing_user_turn.conversation_id
            ):
                raise HTTPException(
                    status_code=409,
                    detail="同一 request_id 已属于其他会话",
                )
            existing_conv = await db.get(
                Conversation, existing_user_turn.conversation_id
            )
            if not existing_conv or (
                not user.is_superadmin and existing_conv.user_id != user.id
            ):
                raise HTTPException(status_code=404, detail="会话不存在")
            stored_revision, stored_state_id = (
                _stored_pending_request_identity(existing_user_turn)
            )
            retry_revision = (
                payload.pending_route_revision
                if payload.pending_route_revision is not None
                else stored_revision
            )
            retry_state_id = (
                payload.pending_state_id
                if payload.pending_state_id is not None
                else stored_state_id
            )
            retry_context = _request_context_for_payload(
                payload,
                conversation_id=existing_user_turn.conversation_id,
                pending_route_revision=retry_revision,
                pending_state_id=retry_state_id,
            )
            try:
                assert_turn_request_matches(existing_user_turn, retry_context)
            except TurnRequestConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            if turn_lease_expired(existing_user_turn):
                # Serialize stale recovery.  Only one retry may replace the
                # expired lease; all others observe its renewed deadline.
                locked_turn = await find_turn_for_user(
                    db,
                    user.id,
                    request_id,
                    for_update=True,
                )
                if not isinstance(locked_turn, ChatTurn):
                    raise HTTPException(status_code=409, detail="请求状态已变化，请重试")
                if locked_turn.status in {"accepted", "generating"}:
                    refresh = getattr(db, "refresh", None)
                    if callable(refresh):
                        await refresh(
                            existing_conv,
                            attribute_names=[
                                "route_state_revision",
                                "pending_route_state",
                            ],
                        )
                    current_revision, current_state_id = _pending_route_identity(
                        existing_conv
                    )
                    stored_revision, stored_state_id = (
                        _stored_pending_request_identity(locked_turn)
                    )
                    allowed_resume_identities = {
                        (stored_revision, stored_state_id)
                    }
                    stored_resume_identity = _stored_resume_identity(locked_turn)
                    if stored_resume_identity is not None:
                        allowed_resume_identities.add(stored_resume_identity)
                    if (
                        current_revision,
                        current_state_id,
                    ) not in allowed_resume_identities:
                        transition_turn(
                            locked_turn,
                            "failed",
                            trace_id=trace_id,
                            error_code="stale_request_context_changed",
                        )
                        await db.commit()
                        raise HTTPException(
                            status_code=409,
                            detail="待处理的澄清上下文已变化，请使用新的 request_id 重新发送",
                        )
                    if reclaim_stale_turn(locked_turn, owner=trace_id):
                        await db.commit()
                        durable_turn = locked_turn
                        conv = existing_conv
                        trace_event(
                            "chat.turn_reclaimed",
                            trace_id=trace_id,
                            conversation_id=conv.id,
                            user_id=user.id,
                            turn_id=durable_turn.id,
                            request_id=durable_turn.request_id,
                            execution_attempts=durable_turn.execution_attempts,
                        )
                    else:
                        await db.commit()
                        return await _existing_turn_response(
                            conv=existing_conv,
                            turn=locked_turn,
                            user=user,
                            db=db,
                        )
                else:
                    await db.commit()
                    return await _existing_turn_response(
                        conv=existing_conv,
                        turn=locked_turn,
                        user=user,
                        db=db,
                    )
            else:
                return await _existing_turn_response(
                    conv=existing_conv,
                    turn=existing_user_turn,
                    user=user,
                    db=db,
                )

    # 获取或创建会话。新会话先 flush 取得 id，但不提前提交；若后续路由/校验失败，
    # 请求结束时整个未提交事务会回滚，避免留下空白会话。
    if conv is None and payload.conversation_id:
        conv = await db.get(Conversation, payload.conversation_id)
        # 复用已有会话时校验归属：非超管不可操作他人会话
        if conv and not user.is_superadmin and conv.user_id != user.id:
            raise HTTPException(status_code=404, detail="会话不存在")
    elif conv is None:
        conv = None

    if not conv:
        conv = Conversation(title=payload.question[:50], user_id=user.id)
        db.add(conv)
        await db.flush()

    # Reserve the idempotency ledger before semantic routing.  A duplicate
    # request therefore never re-enters retrieval/model code, including when
    # the first attempt stopped after generation but before assistant-message
    # persistence.  Legacy test doubles are intentionally left on the old path;
    # real AsyncSession instances always support both methods below.
    durable_turn_created = False
    if durable_turn_enabled and durable_turn is None:
        pending_route_revision, pending_state_id = _pending_route_identity(conv)
        requested_pending_revision = (
            payload.pending_route_revision
            if payload.pending_route_revision is not None
            else pending_route_revision
        )
        requested_pending_state_id = (
            payload.pending_state_id
            if payload.pending_state_id is not None
            else pending_state_id
        )
        if (requested_pending_revision, requested_pending_state_id) != (
            pending_route_revision,
            pending_state_id,
        ):
            raise HTTPException(
                status_code=409,
                detail="待处理的澄清上下文已变化，请刷新会话后重试",
            )
        try:
            durable_turn, durable_turn_created = await reserve_turn(
                db,
                conversation_id=conv.id,
                user_id=user.id,
                request_id=request_id,
                turn_id=requested_turn_id,
                question=payload.question,
                trace_id=trace_id,
                knowledge_base_ids=payload.knowledge_base_ids,
                search_config=request_search_config,
                pending_route_revision=pending_route_revision,
                pending_state_id=pending_state_id,
            )
        except TurnRequestConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not durable_turn_created:
            # The existing row is authoritative.  Do not mutate its trace or
            # pending state with this duplicate request.
            if durable_turn.conversation_id != conv.id:
                winner_conv = await db.get(
                    Conversation, durable_turn.conversation_id
                )
                if not winner_conv or (
                    not user.is_superadmin and winner_conv.user_id != user.id
                ):
                    raise HTTPException(status_code=404, detail="会话不存在")
                conv = winner_conv
            return await _existing_turn_response(
                conv=conv,
                turn=durable_turn,
                user=user,
                db=db,
            )
        try:
            await db.commit()
        except Exception as exc:
            try:
                await db.rollback()
            except Exception:
                pass
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "请求已接收但暂时无法保存，请使用相同 request_id 重试",
                    "error_code": "turn_reservation_persistence_uncertain",
                    "same_request_recoverable": True,
                    "retry_with_new_request_id": False,
                },
            ) from exc

    stored_pending_state = getattr(conv, "pending_route_state", None)
    pending_route_state = _active_pending_route_state(stored_pending_state)
    expired_pending_state = bool(stored_pending_state and pending_route_state is None)
    if stored_pending_state and pending_route_state is None:
        conv.pending_route_state = None
        conv.route_state_revision = int(getattr(conv, "route_state_revision", 0) or 0) + 1

    evidence_pending_state = (
        pending_route_state
        if pending_route_state
        and pending_route_state.get("schema_version") == _EVIDENCE_PENDING_SCHEMA
        else None
    )
    evidence_reply = (
        _parse_evidence_scope_reply(payload.question, evidence_pending_state)
        if evidence_pending_state is not None
        else None
    )
    evidence_filter = (
        _evidence_scope_filter(
            evidence_reply,
            current_kb_ids=payload.knowledge_base_ids,
        )
        if evidence_reply is not None
        and evidence_reply.action in {"single", "compare_all"}
        else None
    )
    evidence_repeat_reason = None
    if evidence_pending_state is not None and evidence_reply is not None:
        pending_kb_snapshot = set(
            evidence_pending_state.get("selected_kb_ids_snapshot", [])
        )
        current_kb_snapshot = {str(value) for value in payload.knowledge_base_ids}
        if (
            pending_kb_snapshot != current_kb_snapshot
            and evidence_reply.action
            in {"single", "compare_all", "refine", "repeat"}
        ):
            if evidence_reply.action != "repeat":
                evidence_reply = _EvidenceScopeReply("repeat")
                evidence_filter = None
            evidence_repeat_reason = "scope_unavailable"
    if (
        evidence_reply is not None
        and evidence_reply.action in {"single", "compare_all"}
        and evidence_filter is None
    ):
        evidence_reply = _EvidenceScopeReply("repeat")
        evidence_repeat_reason = "scope_unavailable"

    cleared_evidence_state_id: str | None = None
    if evidence_reply is not None and evidence_reply.action == "new_question":
        cleared_evidence_state_id = str(evidence_pending_state["state_id"])
        conv.pending_route_state = None
        conv.route_state_revision = int(
            getattr(conv, "route_state_revision", 0) or 0
        ) + 1
        pending_route_state = None
        evidence_pending_state = None

    evidence_refinement_active = bool(
        evidence_pending_state is not None
        and evidence_reply is not None
        and evidence_reply.action == "refine"
    )
    refined_evidence_query = (
        _refined_evidence_query(
            str(evidence_pending_state["original_query"]),
            payload.question,
        )
        if evidence_refinement_active and evidence_pending_state is not None
        else None
    )
    evidence_routing_query = (
        _scoped_evidence_query(
            str(evidence_pending_state["original_query"]),
            evidence_filter,
        )
        if evidence_pending_state is not None and evidence_filter is not None
        else refined_evidence_query
    )
    pipeline_base_query = (
        str(evidence_pending_state["original_query"]).strip()
        if evidence_pending_state is not None and evidence_filter is not None
        else refined_evidence_query
    )
    evidence_pending_execution = bool(
        evidence_pending_state is not None
        and (evidence_filter is not None or evidence_refinement_active)
    )
    user_message_content = _evidence_scope_reply_display_text(
        payload.question,
        evidence_reply if evidence_pending_execution else None,
    )

    # 在保存本轮用户消息之前读取已有对话。带“这些配置/上述内容”等指代的追问
    # 会得到独立检索问题；上一轮来源只作为候选 id，随后按当前知识库范围和文档
    # 状态重新加载，不能直接信任消息 JSON 快照。
    conversation_context = await prepare_conversation_context(
        db,
        conversation_id=conv.id,
        question=payload.question,
        kb_ids=payload.knowledge_base_ids,
        pending_route_state=pending_route_state,
    )
    if pipeline_base_query is not None:
        conversation_context = replace(
            conversation_context,
            standalone_query=pipeline_base_query,
        )

    # 在任何路由/检索之前记录请求和多轮上下文，保证调用链的第一阶段始终
    # 是“接收请求”。此前这里等意图模型返回后才写 chat.request，导致模型
    # 事件排在请求之前；路由校验失败时也会留下没有起点的 running 记录。
    settings = get_settings()
    trace_include_content = settings.rag_trace_include_content
    search_config = payload.search_config.model_dump()
    trace_event(
        "chat.request",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        selected_kb_ids=payload.knowledge_base_ids,
        search_config=search_config,
        intent=None,
        decision_reason=(
            "pending_evidence_scope_selection"
            if evidence_pending_execution
            else (
                "unresolved_reference"
                if conversation_context.unresolved_reference
                else "pending_intent_routing"
            )
        ),
        is_followup=conversation_context.is_followup,
        followup_reason=conversation_context.followup_reason,
        history_message_count=len(conversation_context.history_messages),
        carryover_source_count=len(conversation_context.carryover_sources),
        **content_fields("question", payload.question),
        **content_fields(
            "standalone_query",
            conversation_context.standalone_query,
        ),
    )
    if expired_pending_state:
        trace_event(
            "intent.clarification_expired",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            route_state_revision=conv.route_state_revision,
        )
    trace_event(
        "conversation.context_resolved",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        is_followup=conversation_context.is_followup,
        followup_reason=conversation_context.followup_reason,
        unresolved_reference=conversation_context.unresolved_reference,
        history_message_count=len(conversation_context.history_messages),
        carryover_source_count=len(conversation_context.carryover_sources),
        **content_fields(
            "standalone_query",
            conversation_context.standalone_query,
        ),
    )

    if evidence_pending_state is not None and evidence_reply is not None:
        if evidence_reply.action == "cancel":
            return await _evidence_pending_direct_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                pending_state=evidence_pending_state,
                trace_id=trace_id,
                action="cancel",
                turn=durable_turn,
            )
        if evidence_reply.action == "repeat":
            return await _evidence_pending_direct_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                pending_state=evidence_pending_state,
                trace_id=trace_id,
                action="repeat",
                repeat_reason=evidence_repeat_reason,
                turn=durable_turn,
            )

    route_candidates = route_context_payloads(conversation_context)
    trace_event(
        "conversation.context_candidates",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        candidate_count=len(route_candidates),
        reusable_source_count=sum(
            max(0, int(item.get("reusable_source_count") or 0))
            for item in route_candidates
            if isinstance(item, dict)
        ),
        has_pending_clarification=bool(pending_route_state),
    )
    if (
        conversation_context.unresolved_reference
        and not route_candidates
        and not evidence_pending_execution
    ):
        trace_event(
            "conversation.reference_unresolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            reason=conversation_context.followup_reason,
            selected_kb_count=len(payload.knowledge_base_ids),
            **content_fields("question", payload.question),
        )
        response = await _route_clarification_response(
            db=db,
            conv=conv,
            user=user,
            question=payload.question,
            clarification_message=UNRESOLVED_REFERENCE_MESSAGE,
            decision_reason="unresolved_reference",
            trace_id=trace_id,
            selected_kb_ids=payload.knowledge_base_ids,
            task_contract=None,
            emit_clarification_event=False,
            turn=durable_turn,
        )
        if cleared_evidence_state_id is not None:
            trace_event(
                "evidence.clarification_resolved",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                pending_state_id=cleared_evidence_state_id,
                resolution="new_question",
                route_state_revision=conv.route_state_revision,
            )
        return response

    # 普通请求由语义合同判断是否需要改写。服务端已验证的证据选择/补充范围
    # 不包含新的路由语义，直接构造 continuation 合同，避免为“2”之类的短回复
    # 再等待一次外部意图模型。
    routing_question = evidence_routing_query or (
        payload.question if route_candidates else conversation_context.standalone_query
    )
    if evidence_pending_execution:
        try:
            routing_result = build_verified_evidence_scope_result(
                db,
                pipeline_base_query or routing_question,
                user=user,
                selected_kb_ids=payload.knowledge_base_ids,
                conversation_id=conv.id,
                record_log=True,
                trace_id=trace_id,
                refined=evidence_refinement_active,
            )
            trace_event(
                "evidence.route_contract_built",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                pending_state_id=evidence_pending_state["state_id"],
                resolution=(
                    "refined"
                    if evidence_refinement_active
                    else (
                        "compare_all"
                        if evidence_filter.get("mode") == "compare_all"
                        else "selected"
                    )
                ),
                selected_kb_count=len(set(payload.knowledge_base_ids)),
                decision_reason=routing_result.decision.decision_reason,
            )
        except Exception as exc:
            trace_event(
                "evidence.route_contract_failed",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                pending_state_id=evidence_pending_state["state_id"],
                resolution=(
                    "refined"
                    if evidence_refinement_active
                    else (
                        "compare_all"
                        if evidence_filter.get("mode") == "compare_all"
                        else "selected"
                    )
                ),
                error=exc,
            )
            log_exception_safely(
                logger,
                "[chat/evidence pending route error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=exc,
            )
            return await _evidence_pending_direct_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                pending_state=evidence_pending_state,
                trace_id=trace_id,
                action="repeat",
                repeat_reason="route_contract_unavailable",
                turn=durable_turn,
            )
    else:
        try:
            routing_result = await classify_intent_result(
                db,
                routing_question,
                user=user,
                selected_kb_ids=payload.knowledge_base_ids,
                conversation_id=conv.id,
                record_log=True,
                trace_id=trace_id,
                route_context=route_candidates,
                has_pending_clarification=bool(pending_route_state),
                fallback_relation=conversation_context.relation,
                fallback_query_mode=conversation_context.query_resolution_mode,
                fallback_unresolved=conversation_context.unresolved_reference,
            )
        except Exception as exc:
            # 路由配置/数据库故障不应让调用链停留在 running；接口仍按原语义
            # 抛出异常，但保留安全的阶段错误供后台排查。
            trace_event(
                "chat.error",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="intent_routing",
                error=exc,
            )
            log_exception_safely(
                logger,
                "[chat/intent routing error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=exc,
            )
            await _mark_turn_terminal(
                turn=durable_turn,
                status="failed",
                trace_id=trace_id,
                error_code="intent_routing_failed",
                evidence_status="error",
                retrieval_executed=False,
            )
            raise
    decision = routing_result.decision
    route_decision = getattr(routing_result, "route_decision", None)
    task_contract = getattr(routing_result, "task_contract", None)
    if route_decision is not None and task_contract is not None:
        conversation_context = await resolve_routed_conversation_context(
            db,
            context=conversation_context,
            question=routing_question,
            kb_ids=payload.knowledge_base_ids,
            route_decision=route_decision,
        )
        if pipeline_base_query is not None:
            conversation_context = replace(
                conversation_context,
                standalone_query=pipeline_base_query,
            )
        trace_event(
            "conversation.context_resolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            relation=(route_decision.relation if route_decision is not None else "legacy"),
            readiness=task_contract.readiness,
            query_mode=task_contract.query_mode,
            context_turn_count=len(task_contract.context_turn_keys),
            is_followup=conversation_context.is_followup,
            followup_reason=conversation_context.followup_reason,
            unresolved_reference=conversation_context.unresolved_reference,
            history_message_count=len(conversation_context.history_messages),
            carryover_source_count=len(conversation_context.carryover_sources),
            **content_fields("standalone_query", conversation_context.standalone_query),
        )
        trace_event(
            "intent.routing_decision",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            intent=decision.to_dict(),
            route_schema_version=route_decision.schema_version,
            contract_schema_version=task_contract.schema_version,
            relation=(route_decision.relation if route_decision is not None else "legacy"),
            readiness=task_contract.readiness,
            evidence_scope=route_decision.evidence_scope,
            query_mode=task_contract.query_mode,
            context_turn_count=len(task_contract.context_turn_keys),
            requirement_count=len(task_contract.requirements),
            dispatch_authorized=task_contract.dispatch_authorized,
            selected_kb_count=len(payload.knowledge_base_ids),
            decision_reason=decision.decision_reason,
        )
        if not task_contract.dispatch_authorized:
            clarification_message = (
                task_contract.clarification.question.strip()
                or UNRESOLVED_REFERENCE_MESSAGE
            )
            response = await _route_clarification_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                clarification_message=clarification_message,
                decision_reason=task_contract.decision_reason,
                trace_id=trace_id,
                selected_kb_ids=payload.knowledge_base_ids,
                task_contract=task_contract,
                turn=durable_turn,
            )
            if cleared_evidence_state_id is not None:
                trace_event(
                    "evidence.clarification_resolved",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pending_state_id=cleared_evidence_state_id,
                    resolution="new_question",
                    route_state_revision=conv.route_state_revision,
                )
            return response
        intent_payload = {
            **decision.to_dict(),
            "route_decision": route_decision.to_dict(),
            "task_contract": task_contract.to_dict(),
            "diagnostics": getattr(routing_result, "diagnostics", {}),
        }
    else:
        # Rolling-upgrade compatibility for an old classifier implementation.
        # New chat requests in this release always receive the v1 fields.
        trace_event(
            "intent.routing_decision",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            intent=decision.to_dict(),
            selected_kb_count=len(payload.knowledge_base_ids),
            decision_reason=decision.decision_reason,
        )
        if decision.need_retrieval and not payload.knowledge_base_ids:
            trace_event(
                "chat.error",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="request_validation",
                error=ValueError("该问题需要查询知识库，请至少选择一个知识库"),
                evidence_status="error",
            )
            await _mark_turn_terminal(
                turn=durable_turn,
                status="failed",
                trace_id=trace_id,
                error_code="knowledge_base_required",
                evidence_status="error",
                retrieval_executed=False,
            )
            raise HTTPException(
                status_code=400,
                detail="该问题需要查询知识库，请至少选择一个知识库",
            )
        intent_payload = decision.to_dict()

    pipeline_version, pipeline_reason = _select_rag_pipeline_version(
        configured_version=settings.rag_pipeline_version,
        task_contract=task_contract,
        evidence_scope_filter=evidence_filter,
        evidence_scope_refinement_active=evidence_refinement_active,
        is_followup=conversation_context.is_followup,
        carryover_sources=conversation_context.carryover_sources,
        selected_kb_count=len(set(payload.knowledge_base_ids)),
    )
    if pipeline_version == "reject":
        trace_event(
            "chat.error",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            stage="runner_selection",
            error=ValueError(pipeline_reason),
            contract_present=isinstance(task_contract, RagTaskContract),
            evidence_status="error",
        )
        await _mark_turn_terminal(
            turn=durable_turn,
            status="failed",
            trace_id=trace_id,
            error_code="runner_contract_rejected",
            evidence_status="error",
            retrieval_executed=False,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "请求执行合同校验失败，请重新发送",
                "error_code": "runner_contract_rejected",
                "same_request_recoverable": False,
                "retry_with_new_request_id": True,
            },
        )
    rag_stream_runner = {
        "v1": run_rag_stream,
        "v2": run_rag_v2_stream,
        "direct": run_direct_response_stream,
    }[pipeline_version]
    trace_event(
        "chat.pipeline_selected",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        version=pipeline_version,
        reason=pipeline_reason,
    )

    if pipeline_version == "v2" and isinstance(task_contract, RagTaskContract):
        # The semantic router can be ready while the deterministic local plan
        # still cannot identify a safe relationship/bridge query.  Promote
        # that uncertainty into the durable route-clarification path before
        # saving a generating user turn or opening any retrieval/model call.
        execution_plan = _plan_with_contract_requirements(
            plan_query_locally(conversation_context.standalone_query),
            task_contract,
        )
        if execution_plan.needs_clarification:
            clarification_question = (
                execution_plan.clarification_question
                or "请补充需要查询或了解的具体问题。"
            )
            blocked_contract = replace(
                task_contract,
                readiness="needs_clarification",
                dispatch_authorized=False,
                decision_reason="query_plan_requires_clarification",
                clarification=RouteClarification(
                    question=clarification_question,
                    unresolved=(
                        RouteUnresolvedSlot(
                            role="query_plan",
                            reason="missing",
                        ),
                    ),
                ),
            )
            trace_event(
                "query.plan",
                trace_id=trace_id,
                pipeline_version="v2",
                execution_surface="api_clarification_gate",
                plan=(
                    execution_plan.to_dict()
                    if settings.rag_trace_include_content
                    else {
                        "schema_version": execution_plan.schema_version,
                        "answer_shape": execution_plan.answer_shape,
                        "query_count": len(execution_plan.retrieval_queries),
                        "requirement_count": len(execution_plan.requirements),
                        "confidence": execution_plan.confidence,
                        "source": execution_plan.source,
                        "needs_clarification": True,
                    }
                ),
                **content_fields(
                    "query",
                    conversation_context.standalone_query,
                ),
            )
            response = await _route_clarification_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                clarification_message=clarification_question,
                decision_reason="query_plan_requires_clarification",
                trace_id=trace_id,
                selected_kb_ids=payload.knowledge_base_ids,
                task_contract=blocked_contract,
                turn=durable_turn,
            )
            if cleared_evidence_state_id is not None:
                trace_event(
                    "evidence.clarification_resolved",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pending_state_id=cleared_evidence_state_id,
                    resolution="new_question",
                    route_state_revision=conv.route_state_revision,
                )
            return response

    # 保存用户消息并把 durable turn 推进到 generating。用户消息和状态在
    # RAG 流启动前一起提交，因此断线/进程退出后仍能判断该 request_id 已被
    # 接受，不会误触发第二次检索。
    if durable_turn is not None and durable_turn.status == "accepted":
        transition_turn(durable_turn, "generating", trace_id=trace_id)
        renew_turn_lease(durable_turn, owner=trace_id)
    user_metadata = (
        message_turn_metadata(durable_turn, status="generating")
        if durable_turn is not None
        else {}
    )
    user_msg = None
    if durable_turn is not None and durable_turn.user_message_id is not None:
        loaded_user_msg = await db.get(Message, durable_turn.user_message_id)
        if isinstance(loaded_user_msg, Message):
            user_msg = loaded_user_msg
            user_msg.content = user_message_content
            for key, value in user_metadata.items():
                setattr(user_msg, key, value)
    if user_msg is None:
        user_msg = Message(
            id=(
                durable_turn.user_message_id
                if durable_turn is not None
                and durable_turn.user_message_id is not None
                else uuid.uuid4()
            ),
            conversation_id=conv.id,
            role="user",
            content=user_message_content,
            **user_metadata,
        )
        db.add(user_msg)
    if durable_turn is not None:
        durable_turn.user_message_id = user_msg.id
    current_pending = getattr(conv, "pending_route_state", None)
    if (
        isinstance(current_pending, dict)
        and current_pending.get("schema_version") == "rag_pending_clarification.v1"
    ):
        previous_state_id = current_pending.get("state_id")
        conv.pending_route_state = None
        conv.route_state_revision = int(getattr(conv, "route_state_revision", 0) or 0) + 1
        trace_event(
            "intent.clarification_resolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=previous_state_id,
            route_state_revision=conv.route_state_revision,
            relation=(route_decision.relation if route_decision is not None else "legacy"),
        )
    if durable_turn is not None:
        resume_revision, resume_state_id = _pending_route_identity(conv)
        durable_turn.resume_context = {
            "revision": resume_revision,
            "state_id": resume_state_id,
        }
    await db.commit()
    if cleared_evidence_state_id is not None:
        trace_event(
            "evidence.clarification_resolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=cleared_evidence_state_id,
            resolution="new_question",
            route_state_revision=conv.route_state_revision,
        )

    expected_route_state_revision = int(
        getattr(conv, "route_state_revision", 0) or 0
    )

    async def generate():
        nonlocal durable_turn
        full_response = []
        sources = []
        tokens = None
        retrieval_executed = None
        evidence_status = None
        displayed_result_count = None
        context_evidence_count = None
        hit_count = None
        direct_evidence_count = None
        related_reference_count = None
        evidence_scope_anchor_hit = None
        evidence_scope_anchor_doc_ids: list[str] = []
        # This is deliberately independent from the producer's boolean
        # ``evidence_scope_anchor_hit``.  It is recomputed from DB-refreshed
        # answer sources below and is the only value allowed to resolve a
        # pending evidence scope.
        evidence_source_validation_ok: bool | None = None
        evidence_source_validation_error: str | None = None
        evidence_answer_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()
        evidence_source_validation_locked = False
        evidence_source_validation_failure_emitted = False
        pending_done_chunk = None
        evidence_clarification_payload = None
        evidence_clarification_locked = False
        search_snapshot: dict | None = None
        # 会话和用户消息已提交。先告知前端会话 ID，用户在首条回答完成前停止时也能继续该会话。
        yield "data: " + json.dumps(
            {
                "type": "conversation_started",
                "conversation_id": str(conv.id),
                **(
                    {
                        "turn_id": str(durable_turn.id),
                        "request_id": durable_turn.request_id,
                    }
                    if durable_turn is not None
                    else {}
                ),
            },
            ensure_ascii=False,
        ) + "\n\n"
        if durable_turn is not None:
            yield "data: " + json.dumps(
                _turn_state_event(durable_turn), ensure_ascii=False
            ) + "\n\n"
        try:
            rag_stream = rag_stream_runner(
                question=pipeline_base_query or payload.question,
                kb_ids=payload.knowledge_base_ids,
                search_config=search_config,
                conversation_id=str(conv.id),
                db=db,
                intent=intent_payload,
                task_contract=task_contract,
                trace_id=trace_id,
                standalone_query=conversation_context.standalone_query,
                # 独立新问题不把旧轮正文发送给外部模型，避免无关历史污染回答并
                # 遵循最小披露；只有本地规则确认是追问时才提供有界历史帮助消解。
                conversation_history=(
                    list(conversation_context.history_messages)
                    if conversation_context.is_followup
                    else []
                ),
                carryover_sources=list(conversation_context.carryover_sources),
                is_followup=conversation_context.is_followup,
                followup_reason=conversation_context.followup_reason,
                evidence_scope_filter=evidence_filter,
            )
            async for chunk in rag_stream:
                data = _parse_sse_payload(chunk)
                event_type = data.get("type") if data else None
                source_validation_delta: str | None = None
                # done 必须等 AI 消息持久化成功后再发给前端；其它事件保持实时流式。
                if event_type == "done":
                    pending_done_chunk = chunk
                    continue
                if event_type == "text_delta":
                    if (
                        evidence_clarification_locked
                        or evidence_source_validation_locked
                    ):
                        # Chat emits the trusted clarification question exactly
                        # once when the gate event arrives.  A failed source
                        # validation similarly emits one deterministic message.
                        # Any later model/custom-producer delta is suppressed.
                        continue
                    full_response.append(str(data.get("content") or ""))
                elif event_type == "evidence_clarification":
                    if evidence_clarification_payload is not None:
                        continue
                    evidence_clarification_payload = data
                    evidence_clarification_locked = True
                    # The clarification event is the final generation gate.
                    # Fail closed even if a rolling/custom producer emitted a
                    # contradictory hit status or answer_sources beforehand.
                    evidence_status = "needs_clarification"
                    sources = []
                    context_evidence_count = 0
                    hit_count = 0
                    direct_evidence_count = 0
                    full_response.clear()
                    # Forward the structured gate, then synthesize one exact
                    # clarification delta. The Pipeline's following delta is
                    # ignored by the locked text branch above.
                    yield chunk
                    clarification_question = str(data.get("question") or "").strip()
                    if clarification_question:
                        full_response.append(clarification_question)
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "text_delta",
                                    "content": clarification_question,
                                },
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        )
                    continue
                elif event_type == "search_results":
                    if evidence_clarification_locked:
                        previous_retrieval_executed = retrieval_executed
                        data = _clarification_locked_search_results(
                            data,
                            trace_id=trace_id,
                        )
                        if previous_retrieval_executed is not None:
                            data["retrieval_executed"] = previous_retrieval_executed
                        chunk = (
                            "data: "
                            + json.dumps(data, ensure_ascii=False)
                            + "\n\n"
                        )
                    retrieval_executed = data.get("retrieval_executed")
                    raw_anchor_hit = data.get("evidence_scope_anchor_hit")
                    evidence_scope_anchor_hit = (
                        raw_anchor_hit if isinstance(raw_anchor_hit, bool) else None
                    )
                    raw_anchor_doc_ids = data.get("evidence_scope_anchor_doc_ids")
                    evidence_scope_anchor_doc_ids = (
                        [str(value) for value in raw_anchor_doc_ids]
                        if isinstance(raw_anchor_doc_ids, list)
                        else []
                    )
                    evidence_status = data.get("evidence_status")
                    normalized_evidence_status = str(
                        evidence_status or ""
                    ).strip().casefold()
                    direct_evidence_count = data.get("direct_evidence_count")
                    if (
                        isinstance(direct_evidence_count, bool)
                        or not isinstance(direct_evidence_count, int)
                        or direct_evidence_count < 0
                    ):
                        direct_evidence_count = 0
                    related_reference_count = data.get("related_reference_count")
                    display_results = data.get("results")
                    if not isinstance(display_results, list):
                        display_results = []
                    raw_answer_sources = data.get("answer_sources")
                    has_answer_source_list = isinstance(raw_answer_sources, list)
                    evidence_answer_pairs = set()
                    answer_source_required = (
                        normalized_evidence_status
                        in _ANSWER_SOURCE_REQUIRED_STATUSES
                    )
                    evidence_status_invalid = (
                        normalized_evidence_status
                        not in _SUPPORTED_EVIDENCE_STATUSES
                    )
                    non_answer_source_status = (
                        normalized_evidence_status in _NON_ANSWER_SOURCE_STATUSES
                    )
                    if non_answer_source_status:
                        # 状态是服务端最终证据门控：即使异常/旧版/自定义流生产者
                        # 同时错误携带了 answer_sources 和非零 direct 数，也不得
                        # 把这些正文保存成历史回答依据。持久化层必须 fail closed，
                        # 不能只依赖正常 Pipeline 或前端隐藏。
                        direct_evidence_count = 0
                    if evidence_status_invalid:
                        raw_answer_sources = []
                        evidence_source_validation_ok = False
                        evidence_source_validation_error = (
                            "evidence_status_invalid"
                        )
                    elif not has_answer_source_list:
                        # Fail closed for rolling upgrades and custom stream
                        # producers: a missing answer_sources list cannot be
                        # treated as the generation context.
                        raw_answer_sources = []
                        evidence_source_validation_ok = False
                        evidence_source_validation_error = (
                            "answer_sources_not_a_list"
                        )
                    elif non_answer_source_status:
                        # Non-answer states never persist producer snapshots,
                        # even if an old producer accidentally included them.
                        if raw_answer_sources:
                            evidence_source_validation_ok = False
                            evidence_source_validation_error = (
                                "non_answer_status_with_sources"
                            )
                        else:
                            evidence_source_validation_ok = True
                            evidence_source_validation_error = None
                        raw_answer_sources = []
                    else:
                        (
                            answer_source_items,
                            evidence_answer_pairs,
                            evidence_source_validation_error,
                        ) = await _validate_stream_answer_sources(
                            db,
                            raw_sources=raw_answer_sources,
                            raw_results=display_results,
                            selected_kb_ids=payload.knowledge_base_ids,
                        )
                        if (
                            evidence_filter is not None
                            and evidence_source_validation_error is None
                        ):
                            allowed_scope_pairs = _scope_document_pairs(
                                evidence_filter
                            )
                            if not evidence_answer_pairs.issubset(
                                allowed_scope_pairs
                            ):
                                evidence_source_validation_error = (
                                    "answer_source_scope_forbidden"
                                )
                        evidence_source_validation_ok = (
                            evidence_source_validation_error is None
                        )
                        # If refresh/identity validation fails, do not retain
                        # producer-provided body or metadata.  The validation
                        # lock below replaces the model stream with one
                        # deterministic retry message and keeps pending scope.
                        raw_answer_sources = answer_source_items
                        if evidence_source_validation_error is not None:
                            raw_answer_sources = []
                            answer_source_items = []
                            evidence_answer_pairs = set()
                            if normalized_evidence_status not in {
                                "error",
                                "no_hit",
                                "skipped",
                                "needs_clarification",
                            }:
                                evidence_status = "error"
                                normalized_evidence_status = "error"
                    recomputed_anchor_hit, recomputed_anchor_doc_ids = (
                        _scope_anchor_coverage_from_sources(
                            evidence_filter,
                            evidence_answer_pairs,
                        )
                    )
                    if evidence_filter is not None:
                        # Ignore producer-advertised anchor fields entirely.
                        evidence_scope_anchor_hit = recomputed_anchor_hit
                        evidence_scope_anchor_doc_ids = recomputed_anchor_doc_ids
                    else:
                        evidence_scope_anchor_hit = False
                        evidence_scope_anchor_doc_ids = []
                    displayed_result_count = data.get(
                        "displayed_result_count",
                        data.get("total", len(display_results)),
                    )
                    if (
                        isinstance(displayed_result_count, bool)
                        or not isinstance(displayed_result_count, int)
                        or displayed_result_count < 0
                    ):
                        displayed_result_count = len(display_results)
                    answer_source_items = [
                        source
                        for source in (raw_answer_sources or [])[:20]
                        if isinstance(source, dict)
                    ]
                    if evidence_source_validation_error is not None:
                        # Once producer evidence fails identity/authorization
                        # validation, its broad panel snapshot is untrusted too.
                        # Do not expose stale or cross-scope content through the
                        # UI while merely clearing the persisted citation list.
                        display_results = []
                        data["results"] = []
                        data["total"] = 0
                        data["displayed_result_count"] = 0
                        data["related_reference_count"] = 0
                        displayed_result_count = 0
                        related_reference_count = 0
                    source_validation_failed = bool(
                        evidence_status_invalid
                        or evidence_source_validation_error is not None
                        or (answer_source_required and not answer_source_items)
                    )
                    if (
                        not evidence_clarification_locked
                        and (
                            source_validation_failed
                            or evidence_source_validation_locked
                        )
                    ):
                        if evidence_source_validation_error is None:
                            evidence_source_validation_error = (
                                "required_answer_sources_empty"
                            )
                        display_results = []
                        data["results"] = []
                        data["total"] = 0
                        data["displayed_result_count"] = 0
                        data["related_reference_count"] = 0
                        displayed_result_count = 0
                        related_reference_count = 0
                        evidence_source_validation_ok = False
                        evidence_source_validation_locked = True
                        evidence_answer_pairs = set()
                        evidence_scope_anchor_hit = False
                        evidence_scope_anchor_doc_ids = []
                        evidence_status = "error"
                        normalized_evidence_status = "error"
                        answer_source_items = []
                        raw_answer_sources = []
                        sources = []
                        context_evidence_count = 0
                        hit_count = 0
                        direct_evidence_count = 0
                        full_response.clear()
                        full_response.append(
                            _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE
                        )
                        if not evidence_source_validation_failure_emitted:
                            source_validation_delta = (
                                "data: "
                                + json.dumps(
                                    {
                                        "type": "text_delta",
                                        "content": (
                                            _EVIDENCE_SOURCE_VALIDATION_FAILURE_MESSAGE
                                        ),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )
                            evidence_source_validation_failure_emitted = True
                    # Counts are derived from the validated context snapshot,
                    # never copied from a producer that may have counted broad
                    # candidates or stale sources.
                    direct_evidence_count = sum(
                        str(source.get("evidence_role") or "")
                        .strip()
                        .casefold()
                        == "direct"
                        for source in answer_source_items
                    )
                    data["answer_sources"] = answer_source_items
                    data["answer_source_count"] = len(answer_source_items)
                    data["context_evidence_count"] = len(answer_source_items)
                    data["direct_evidence_count"] = direct_evidence_count
                    data["hit_count"] = direct_evidence_count
                    data["evidence_scope_anchor_hit"] = (
                        evidence_scope_anchor_hit
                    )
                    data["evidence_scope_anchor_doc_ids"] = list(
                        evidence_scope_anchor_doc_ids
                    )
                    data["evidence_status"] = evidence_status
                    # Keep only a bounded identity/ranking snapshot for durable
                    # retries/history.  Message.sources remains the current
                    # validated citation payload and is re-authorized on read.
                    search_snapshot = _bounded_search_snapshot(data)
                    chunk = (
                        "data: "
                        + json.dumps(data, ensure_ascii=False)
                        + "\n\n"
                    )
                    context_evidence_count = len(answer_source_items)
                    hit_count = direct_evidence_count
                    source_meta = {
                        "trace_id": data.get("trace_id") or trace_id,
                        "retrieval_executed": bool(retrieval_executed),
                        "evidence_status": evidence_status,
                        "displayed_result_count": displayed_result_count or 0,
                        "context_evidence_count": context_evidence_count,
                        "hit_count": hit_count,
                        "direct_evidence_count": direct_evidence_count or 0,
                        "related_reference_count": related_reference_count or 0,
                        "is_followup": bool(data.get("is_followup")),
                        "carryover_source_count": data.get("carryover_source_count") or 0,
                    }
                    # 右侧检索面板继续消费 ``results``，但历史回答与引用只保存
                    # Pipeline 实际送入 generation.context 的 answer_sources。
                    # 两者不能再共用一份宽候选列表。
                    sources = [
                        {**source, **source_meta}
                        for source in answer_source_items
                    ]
                elif event_type == "usage":
                    tokens = data.get("total_tokens")
                yield chunk
                if source_validation_delta is not None:
                    yield source_validation_delta
                if (
                    event_type == "search_results"
                    and evidence_source_validation_locked
                ):
                    # ``search_results`` is emitted before final generation in
                    # both pipelines.  Once its source identity/authorization
                    # contract fails, pulling one more item would let the
                    # producer open an LLM stream whose output is guaranteed
                    # to be discarded.  Stop at the gate, close the suspended
                    # async generator, persist the deterministic failure
                    # message below, and emit ``done`` only after that commit.
                    trace_event(
                        "generation.skipped",
                        trace_id=trace_id,
                        pipeline_version=pipeline_version,
                        reason="answer_source_validation_failed",
                        evidence_status=evidence_status,
                        validation_error=evidence_source_validation_error,
                    )
                    close_stream = getattr(rag_stream, "aclose", None)
                    if callable(close_stream):
                        try:
                            await close_stream()
                        except Exception as close_exc:
                            # The answer is already fail-closed.  Cleanup is
                            # best-effort and must not replace it with a generic
                            # stream error or delay persistence/retry guidance.
                            trace_event(
                                "chat.stream_close_error",
                                trace_id=trace_id,
                                conversation_id=conv.id,
                                stage="answer_source_validation",
                                error=close_exc,
                            )
                    break
        except asyncio.CancelledError:
            # 浏览器停止生成、断开连接或服务关闭都会取消流协程。同步入队即可
            # 立即把调用链标成 interrupted；不要在已取消任务里继续等待数据库。
            trace_event(
                "chat.cancelled",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="streaming",
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                **content_fields("partial_answer", "".join(full_response)),
            )
            try:
                await asyncio.shield(
                    _mark_turn_terminal(
                        turn=durable_turn,
                        status="cancelled",
                        trace_id=trace_id,
                        error_code="stream_cancelled",
                        evidence_status=evidence_status,
                        retrieval_executed=retrieval_executed,
                    )
                )
            except Exception:
                pass
            raise
        except Exception as e:
            log_exception_safely(
                logger,
                "[chat/stream error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=e,
            )
            if retrieval_executed is None:
                retrieval_executed = bool(decision.need_retrieval)
            if evidence_status is None:
                evidence_status = "error" if decision.need_retrieval else "skipped"
            if hit_count is None:
                hit_count = 0
            if context_evidence_count is None:
                context_evidence_count = 0
            if displayed_result_count is None:
                displayed_result_count = 0
            trace_event(
                "chat.error",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                error=e,
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count,
                answer_source_count=context_evidence_count,
                context_evidence_count=context_evidence_count,
                hit_count=hit_count,
                direct_evidence_count=direct_evidence_count or 0,
                related_reference_count=related_reference_count or 0,
                **content_fields("partial_answer", "".join(full_response)),
            )
            await _mark_turn_terminal(
                turn=durable_turn,
                status="failed",
                trace_id=trace_id,
                error_code="stream_failed",
                evidence_status=evidence_status,
                retrieval_executed=retrieval_executed,
            )
            from database import AsyncSessionLocal
            if routing_result.route_log_id is not None:
                try:
                    async with AsyncSessionLocal() as save_db:
                        route_log = await save_db.get(IntentRouteLog, routing_result.route_log_id)
                        if route_log is not None:
                            route_log.retrieval_executed = retrieval_executed
                            route_log.evidence_status = evidence_status
                            route_log.hit_count = hit_count
                            await save_db.commit()
                except Exception as persistence_exc:
                    # 路由统计是 best-effort；失败不能覆盖真正的模型/检索错误，也不能
                    # 阻断随后发给前端的安全 error + done 事件。
                    log_exception_safely(
                        logger,
                        "[chat/error route-log persistence] trace=%s conv=%s",
                        trace_id,
                        conv.id,
                        exc=persistence_exc,
                    )
                    trace_event(
                        "chat.persistence_error",
                        trace_id=trace_id,
                        conversation_id=conv.id,
                        operation="update_route_log_after_stream_error",
                        error=persistence_exc,
                    )
            yield f"data: {json.dumps({'type': 'error', 'message': _public_stream_error_message(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"
            return

        # 保存 AI 回复
        answer = "".join(full_response)
        from database import AsyncSessionLocal
        created_pending_state = None
        resolved_pending_state_id = None
        persisted_route_state_revision = expected_route_state_revision
        # Stage the generated payload first, then complete the transcript in a
        # retryable transaction.  Any later pending-route CAS or statistics
        # failure can no longer roll the assistant answer back.
        if durable_turn is not None:
            try:
                staged_sources = [
                    item
                    for source in sources[:20]
                    if (item := _bounded_source_identity_snapshot(source)) is not None
                ]
                durable_turn = await _stage_generated_turn(
                    turn=durable_turn,
                    conv=conv,
                    answer=answer,
                    sources=staged_sources,
                    tokens=tokens,
                    evidence_status=evidence_status,
                    retrieval_executed=retrieval_executed,
                    trace_id=trace_id,
                    search_snapshot=search_snapshot,
                )
                durable_turn = await _recover_turn_answer(
                    conv=conv,
                    turn=durable_turn,
                )
            except asyncio.CancelledError:
                await _mark_turn_terminal(
                    turn=durable_turn,
                    status="cancelled",
                    trace_id=trace_id,
                    error_code="response_persistence_cancelled",
                    evidence_status=evidence_status,
                    retrieval_executed=retrieval_executed,
                )
                raise
            except Exception as exc:
                payload_recoverable = await _mark_turn_persist_failed(
                    turn=durable_turn,
                    trace_id=trace_id,
                )
                trace_event(
                    "chat.persistence_error",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    operation="stage_or_complete_turn",
                    error=exc,
                    **content_fields("answer", answer),
                )
                yield "data: " + json.dumps(
                    {
                        "type": "error",
                        "message": (
                            "回答已生成但暂时无法完成保存，请使用相同 request_id 重试"
                            if payload_recoverable
                            else (
                                "回答已生成，但恢复数据未能保存。"
                                "请重新发送问题并使用新的 request_id。"
                            )
                        ),
                        "request_id": durable_turn.request_id,
                        "turn_id": str(durable_turn.id),
                        "status": durable_turn.status,
                        "same_request_recoverable": payload_recoverable,
                        "persistence_status": "failed",
                        "retry_with_new_request_id": not payload_recoverable,
                    },
                    ensure_ascii=False,
                ) + "\n\n"
                yield "data: " + json.dumps(
                    {
                        "type": "done",
                        "conversation_id": str(conv.id),
                        "request_id": durable_turn.request_id,
                        "turn_id": str(durable_turn.id),
                        "status": durable_turn.status,
                        "persistence_status": "failed",
                        "same_request_recoverable": payload_recoverable,
                        "retry_with_new_request_id": not payload_recoverable,
                    },
                    ensure_ascii=False,
                ) + "\n\n"
                return
        try:
            async with AsyncSessionLocal() as save_db:
                ai_msg = None
                if durable_turn is not None and durable_turn.assistant_message_id:
                    loaded_ai_msg = await save_db.get(
                        Message, durable_turn.assistant_message_id
                    )
                    if isinstance(loaded_ai_msg, Message):
                        ai_msg = loaded_ai_msg
                if ai_msg is None:
                    ai_msg = Message(
                        id=(
                            durable_turn.assistant_message_id
                            if durable_turn is not None
                            and durable_turn.assistant_message_id is not None
                            else uuid.uuid4()
                        ),
                        conversation_id=conv.id,
                        role="assistant",
                        content=answer,
                        sources=sources,
                        tokens=tokens,
                        **(
                            message_turn_metadata(
                                durable_turn, status="completed"
                            )
                            if durable_turn is not None
                            else {}
                        ),
                    )
                    save_db.add(ai_msg)

                new_pending_state = None
                if evidence_clarification_payload is not None:
                    new_pending_state = _evidence_event_pending_state(
                        evidence_clarification_payload,
                        original_query=conversation_context.standalone_query,
                        selected_kb_ids=payload.knowledge_base_ids,
                        base_user_message_id=user_msg.id,
                        clarification_message_id=ai_msg.id,
                    )
                    raw_choices = evidence_clarification_payload.get("choices")
                    if raw_choices and new_pending_state is None:
                        raise ValueError("Pipeline 返回了无效的证据范围澄清选项")

                normalized_final_evidence_status = str(
                    evidence_status or ""
                ).strip().casefold()
                selected_scope_completed = bool(
                    (evidence_filter is not None or evidence_refinement_active)
                    and evidence_pending_state is not None
                    and pending_done_chunk is not None
                    and retrieval_executed is True
                    and normalized_final_evidence_status
                    in _SUCCESSFUL_EVIDENCE_SCOPE_STATUSES
                    and evidence_source_validation_ok is True
                    and not evidence_source_validation_locked
                    and (
                        evidence_refinement_active
                        or evidence_scope_anchor_hit is True
                    )
                )
                # A fresh valid clarification replaces the old state.  Without
                # one, technical/skip outcomes are persisted as assistant
                # feedback but leave the previous choice pending for retry.
                should_update_route_state = bool(
                    new_pending_state is not None or selected_scope_completed
                )
                if should_update_route_state:
                    persisted_conv = await save_db.get(Conversation, conv.id)
                    if persisted_conv is None:
                        raise RuntimeError("会话不存在，无法保存证据范围状态")
                    persisted_revision = int(
                        getattr(persisted_conv, "route_state_revision", 0) or 0
                    )
                    if persisted_revision != expected_route_state_revision:
                        raise RuntimeError("会话范围状态已被其他请求更新")
                    persisted_pending = _active_pending_route_state(
                        getattr(persisted_conv, "pending_route_state", None)
                    )
                    expected_pending_id = (
                        str(evidence_pending_state.get("state_id"))
                        if evidence_pending_state is not None
                        else None
                    )
                    persisted_pending_id = (
                        str(persisted_pending.get("state_id"))
                        if persisted_pending is not None
                        else None
                    )
                    if persisted_pending_id != expected_pending_id:
                        raise RuntimeError("待选择的证据范围已发生变化")

                    if expected_pending_id is not None:
                        resolved_pending_state_id = expected_pending_id
                    if new_pending_state is not None:
                        persisted_conv.pending_route_state = new_pending_state
                        created_pending_state = new_pending_state
                    else:
                        persisted_conv.pending_route_state = None
                    persisted_conv.route_state_revision = persisted_revision + 1
                    persisted_route_state_revision = persisted_conv.route_state_revision

                if (
                    durable_turn is None
                    and routing_result.route_log_id is not None
                ):
                    route_log = await save_db.get(IntentRouteLog, routing_result.route_log_id)
                    if route_log is not None:
                        route_log.retrieval_executed = (
                            bool(retrieval_executed)
                            if retrieval_executed is not None
                            else bool(decision.need_retrieval)
                        )
                        route_log.evidence_status = evidence_status or (
                            "no_hit" if decision.need_retrieval else "skipped"
                        )
                        route_log.hit_count = int(hit_count or 0)
                await save_db.commit()
            if resolved_pending_state_id is not None:
                trace_event(
                    "evidence.clarification_resolved",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pending_state_id=resolved_pending_state_id,
                    resolution=(
                        "compare_all"
                        if evidence_filter and evidence_filter.get("mode") == "compare_all"
                        else (
                            "refined"
                            if evidence_refinement_active
                            else "selected"
                        )
                    ),
                    selected_choice_keys=[
                        choice.get("key")
                        for choice in (evidence_filter or {}).get("choices", [])
                    ],
                    evidence_scope_anchor_hit=evidence_scope_anchor_hit,
                    evidence_scope_anchor_doc_ids=evidence_scope_anchor_doc_ids,
                    route_state_revision=persisted_route_state_revision,
                )
            if created_pending_state is not None:
                trace_event(
                    "evidence.clarification_created",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    user_id=user.id,
                    pending_state_id=created_pending_state["state_id"],
                    dimension=created_pending_state["dimension"],
                    choice_count=len(created_pending_state["choices"]),
                    selected_kb_count=len(
                        created_pending_state["selected_kb_ids_snapshot"]
                    ),
                    route_state_revision=persisted_route_state_revision,
                    **content_fields(
                        "original_query",
                        created_pending_state["original_query"],
                    ),
                    **content_fields(
                        "clarification",
                        created_pending_state["clarification_message"],
                    ),
                )
            # 只有回答和路由统计真正提交成功后才把 Trace 标成 success，避免
            # “调用链成功但历史消息不存在”的竞态。
            trace_event(
                "chat.response",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                evidence_status=evidence_status,
                retrieval_executed=retrieval_executed,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                direct_evidence_count=direct_evidence_count,
                related_reference_count=related_reference_count,
                evidence_source_validation_ok=evidence_source_validation_ok,
                evidence_source_validation_error=evidence_source_validation_error,
                evidence_source_validation_locked=evidence_source_validation_locked,
                tokens=tokens,
                sources=[
                    {
                        "doc_id": source.get("doc_id"),
                        "chunk_id": source.get("id"),
                        "evidence_role": source.get("evidence_role"),
                        "constraint_status": source.get("constraint_status"),
                        "retrieval_score": source.get("retrieval_score"),
                        "effective_score": source.get("score"),
                        "answer_support": source.get("answer_support"),
                        **content_fields(
                            "filename",
                            str(source.get("filename") or ""),
                        ),
                    }
                    for source in sources
                ],
                **content_fields("answer", answer),
            )
        except asyncio.CancelledError:
            trace_event(
                "chat.cancelled",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="response_persistence",
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                **content_fields("partial_answer", answer),
            )
            await _mark_turn_terminal(
                turn=durable_turn,
                status="cancelled",
                trace_id=trace_id,
                error_code="response_persistence_cancelled",
                evidence_status=evidence_status,
                retrieval_executed=retrieval_executed,
            )
            raise
        except Exception as exc:
            log_exception_safely(
                logger,
                "[chat/persistence error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=exc,
            )
            if durable_turn is not None and durable_turn.status == "completed":
                # The transcript is already durable.  This exception can only
                # come from the independent pending-route/statistics phase.
                # Never emit an ACK for its rolled-back pending state.
                created_pending_state = None
                resolved_pending_state_id = None
                trace_event(
                    "chat.persistence_error",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    operation="post_answer_state_update",
                    answer_persisted=True,
                    error=exc,
                )
            else:
                trace_event(
                    "chat.persistence_error",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    error=exc,
                    **content_fields("answer", answer),
                )
                yield f"data: {json.dumps({'type': 'error', 'message': '回答已生成，但保存失败，请重试'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"
                return
        if durable_turn is not None and routing_result.route_log_id is not None:
            try:
                async with AsyncSessionLocal() as stats_db:
                    route_log = await stats_db.get(
                        IntentRouteLog, routing_result.route_log_id
                    )
                    if isinstance(route_log, IntentRouteLog):
                        route_log.retrieval_executed = (
                            bool(retrieval_executed)
                            if retrieval_executed is not None
                            else bool(decision.need_retrieval)
                        )
                        route_log.evidence_status = evidence_status or (
                            "no_hit" if decision.need_retrieval else "skipped"
                        )
                        route_log.hit_count = int(hit_count or 0)
                        await stats_db.commit()
            except Exception as stats_exc:
                log_exception_safely(
                    logger,
                    "[chat/route statistics best-effort] trace=%s conv=%s",
                    trace_id,
                    conv.id,
                    exc=stats_exc,
                )
                trace_event(
                    "chat.persistence_error",
                    trace_id=trace_id,
                    conversation_id=conv.id,
                    operation="update_route_statistics",
                    answer_persisted=True,
                    error=stats_exc,
                )
        if created_pending_state is not None:
            # ``evidence_clarification`` is streamed as soon as the pipeline
            # closes generation, but choices must remain disabled until the
            # assistant message and pending state have committed together.
            yield (
                "data: "
                + json.dumps(
                    _evidence_clarification_ack(
                        conversation_id=conv.id,
                        pending_state=created_pending_state,
                        route_state_revision=persisted_route_state_revision,
                    ),
                    ensure_ascii=False,
                )
                + "\n\n"
            )
        if durable_turn is not None:
            yield "data: " + json.dumps(
                _turn_state_event(durable_turn), ensure_ascii=False
            ) + "\n\n"
        if pending_done_chunk is not None:
            yield pending_done_chunk
        else:
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers=_turn_response_headers(
            conversation_id=conv.id,
            trace_id=trace_id,
            turn=durable_turn,
        ),
    )


@router.get("/history", response_model=list[ConversationOut])
async def get_history(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    offset = (page - 1) * page_size
    stmt = select(Conversation).order_by(Conversation.created_at.desc())
    # 非超管只看自己的会话；超管可见全部
    if not user.is_superadmin:
        stmt = stmt.where(Conversation.user_id == user.id)
    rows = (await db.execute(
        stmt.offset(offset).limit(page_size)
    )).scalars().all()
    return rows


@router.get("/{conv_id}/messages", response_model=list[MessageOut])
async def get_messages(
    conv_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")
    rows = (await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
    )).scalars().all()
    return await _messages_with_current_source_scope(
        rows,
        user=user,
        db=db,
        pending_route_state=getattr(conv, "pending_route_state", None),
        route_state_revision=getattr(conv, "route_state_revision", 0),
    )


@router.patch("/{conv_id}", response_model=ConversationOut)
async def rename_conversation(
    conv_id: uuid.UUID,
    payload: ConversationRenameRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    """更新会话标题；沿用读取/删除时的归属校验。"""
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="会话标题不能为空")

    conv.title = title
    await db.commit()
    await db.refresh(conv)
    return conv


@router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")
    await db.delete(conv)
    await db.commit()
    return {"message": "删除成功"}
