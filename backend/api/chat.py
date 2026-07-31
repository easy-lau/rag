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
    Conversation,
    Document,
    DocumentChunk,
    IntentRouteLog,
    Message,
    User,
    now_utc,
)
from models.schemas import ChatRequest, ConversationOut, ConversationRenameRequest, MessageOut
from core.rag_pipeline import run_rag_stream
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import CHAT_USE
from core.intent_router import classify_intent_result
from core.conversation_context import (
    UNRESOLVED_REFERENCE_MESSAGE,
    prepare_conversation_context,
    resolve_routed_conversation_context,
    route_context_payloads,
)
from core.query_route_compiler import (
    CompiledAnswerRequirement,
    RagTaskContract,
    TaskContractDispatchError,
    require_rag_task_contract_dispatchable,
)
from core.query_route_contract import RouteClarification, RouteRequirement
from core.rag_trace import content_fields, log_exception_safely, trace_event
from config import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)
_NON_ANSWER_SOURCE_STATUSES = {
    "no_hit",
    "skipped",
    "error",
    "needs_clarification",
}
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
_EVIDENCE_DIMENSIONS = {"version", "product", "product_version", "project"}
_EVIDENCE_SELECTION_MODES = {"choice", "refine"}
_CHOICE_TEXT_FIELDS = (
    "products",
    "canonical_products",
    "versions",
    "projects",
    "filenames",
)
_CHOICE_UUID_FIELDS = ("kb_ids", "doc_ids")
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
        f"{original_query.strip()}\n"
        f"用户补充的适用范围：{str(refinement or '').strip()}"
    ).strip()


def _recover_evidence_pending_contract(
    route_decision: object,
    task_contract: RagTaskContract,
    *,
    pending_state: dict,
    selected_kb_count: int,
    refined: bool,
) -> tuple[object, RagTaskContract] | None:
    """Recover a blocked second-pass route from a server-validated v2 state."""

    original_query = str(pending_state.get("original_query") or "").strip()
    if not original_query or selected_kb_count <= 0:
        return None
    empty_clarification = RouteClarification(question="", unresolved=())
    route_requirement = RouteRequirement(
        role="answer",
        origin="user_text",
        description=original_query,
    )
    contract_requirement = CompiledAnswerRequirement(
        id="r1",
        role="answer",
        origin="user_text",
        description=original_query,
        importance="required",
        source="explicit",
    )
    decision_reason = (
        "evidence_scope_refined" if refined else "evidence_scope_selected"
    )
    try:
        recovered_route = replace(
            route_decision,
            readiness="ready",
            relation="continuation",
            evidence_scope="enterprise_kb",
            query_resolution=replace(
                route_decision.query_resolution,
                mode="current",
                context_turn_keys=(),
            ),
            requirements=(route_requirement,),
            clarification=empty_clarification,
            confidence=1.0,
        )
        recovered_contract = replace(
            task_contract,
            readiness="ready",
            action="retrieve",
            confidence=1.0,
            source="evidence_pending_rule",
            relation="continuation",
            evidence_scope="enterprise_kb",
            query_mode="current",
            context_turn_keys=(),
            response_mode="grounded_qa",
            retrieval_policy="required",
            need_retrieval=True,
            dispatch_authorized=True,
            decision_reason=decision_reason,
            selected_kb_count=selected_kb_count,
            requirements=(contract_requirement,),
            clarification=empty_clarification,
        )
        require_rag_task_contract_dispatchable(
            recovered_contract,
            selected_kb_count=selected_kb_count,
            available_turn_keys=(),
        )
    except (AttributeError, TypeError, TaskContractDispatchError):
        return None
    return recovered_route, recovered_contract


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
) -> StreamingResponse:
    """Persist a versioned, non-executable clarification and stream it."""

    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="user",
        content=question,
    )
    assistant_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="assistant",
        content=clarification_message,
        sources=[],
    )
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
    await db.commit()

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
            {"type": "conversation_started", "conversation_id": str(conv.id)},
        ]
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Conversation-ID": str(conv.id),
            "X-RAG-Trace-ID": trace_id,
        },
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
) -> StreamingResponse:
    """Repeat selectable evidence scopes or cancel them without model dispatch."""

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

    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="user",
        content=question,
    )
    assistant_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="assistant",
        content=answer,
        sources=[],
    )
    if action == "cancel":
        current = getattr(conv, "pending_route_state", None)
        if isinstance(current, dict) and current.get("state_id") == pending_state.get("state_id"):
            conv.pending_route_state = None
            conv.route_state_revision = int(
                getattr(conv, "route_state_revision", 0) or 0
            ) + 1
    db.add_all([user_msg, assistant_msg])
    await db.commit()

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
            {"type": "conversation_started", "conversation_id": str(conv.id)},
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Conversation-ID": str(conv.id),
            "X-RAG-Trace-ID": trace_id,
        },
    )


def _parse_sse_payload(chunk: str) -> dict | None:
    if not chunk.startswith("data: "):
        return None
    try:
        payload = json.loads(chunk.removeprefix("data: ").strip())
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


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
    except (TypeError, ValueError, AttributeError):
        return None
    return kb_id, doc_id, chunk_id


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


async def _messages_with_current_source_scope(
    rows: list[Message],
    *,
    user: User,
    db: AsyncSession,
) -> list[MessageOut]:
    """按当前角色范围和文档状态过滤历史 ``sources`` 快照。

    assistant 正文属于用户自己的既有会话记录；额外展开的原始检索片段则必须
    每次按当前 RBAC 重新授权，防止角色范围被撤销或文档停用后仍从 JSONB 快照
    读取 ``content`` / ``source_url``。
    """

    accessible = await get_accessible_kb_ids(user, db)
    accessible_set = set(accessible) if accessible is not None else None
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
        # 历史脏数据可能把 sources 存成 dict；不先让 Pydantic 验证 ORM
        # 对象，否则一条异常消息会让整个会话返回 500。
        serialized = MessageOut(
            id=row.id,
            conversation_id=row.conversation_id,
            role=row.role,
            content=row.content,
            sources=visible_sources if isinstance(row.sources, list) else None,
            tokens=row.tokens,
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


@router.post("/send")
async def send_message(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    trace_id = uuid.uuid4().hex
    # 流式开始前校验请求的知识库都在可访问范围内（accessible 为 None 表示全部）
    accessible = await get_accessible_kb_ids(user, db)
    if accessible is not None and not set(payload.knowledge_base_ids).issubset(set(accessible)):
        raise HTTPException(status_code=403, detail="无权访问部分知识库")

    # 获取或创建会话。新会话先 flush 取得 id，但不提前提交；若后续路由/校验失败，
    # 请求结束时整个未提交事务会回滚，避免留下空白会话。
    if payload.conversation_id:
        conv = await db.get(Conversation, payload.conversation_id)
        # 复用已有会话时校验归属：非超管不可操作他人会话
        if conv and not user.is_superadmin and conv.user_id != user.id:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        conv = None

    if not conv:
        conv = Conversation(title=payload.question[:50], user_id=user.id)
        db.add(conv)
        await db.flush()

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
    trace_include_content = get_settings().rag_trace_include_content
    search_config = payload.search_config.model_dump()
    trace_search_config = dict(search_config)
    if not trace_include_content:
        trace_search_config["tags"] = []
    trace_event(
        "chat.request",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        selected_kb_ids=payload.knowledge_base_ids,
        search_config=trace_search_config,
        selected_tag_count=len(search_config.get("tags") or []),
        intent=None,
        decision_reason=(
            "unresolved_reference"
            if conversation_context.unresolved_reference
            else (
                "pending_evidence_scope_selection"
                if evidence_filter is not None or evidence_refinement_active
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
    if conversation_context.unresolved_reference and not route_candidates:
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

    # 模型接收原始当前输入和 request-local t1/t2/t3 候选。查询是否需要改写由
    # 语义合同独立表达，不能再由旧 is_followup 正则提前决定。
    routing_question = evidence_routing_query or (
        payload.question if route_candidates else conversation_context.standalone_query
    )
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
        raise
    decision = routing_result.decision
    route_decision = getattr(routing_result, "route_decision", None)
    task_contract = getattr(routing_result, "task_contract", None)
    evidence_pending_execution = bool(
        evidence_pending_state is not None
        and (evidence_filter is not None or evidence_refinement_active)
    )
    if (
        evidence_pending_execution
        and route_decision is not None
        and isinstance(task_contract, RagTaskContract)
        and not task_contract.dispatch_authorized
    ):
        recovered = _recover_evidence_pending_contract(
            route_decision,
            task_contract,
            pending_state=evidence_pending_state,
            selected_kb_count=len(set(payload.knowledge_base_ids)),
            refined=evidence_refinement_active,
        )
        if recovered is None:
            return await _evidence_pending_direct_response(
                db=db,
                conv=conv,
                user=user,
                question=payload.question,
                pending_state=evidence_pending_state,
                trace_id=trace_id,
                action="repeat",
                repeat_reason="route_contract_unavailable",
            )
        route_decision, task_contract = recovered
        trace_event(
            "evidence.route_contract_recovered",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            pending_state_id=evidence_pending_state["state_id"],
            resolution=(
                "refined" if evidence_refinement_active else "selected"
            ),
            selected_kb_count=len(set(payload.knowledge_base_ids)),
            decision_reason=task_contract.decision_reason,
        )
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
            raise HTTPException(
                status_code=400,
                detail="该问题需要查询知识库，请至少选择一个知识库",
            )
        intent_payload = decision.to_dict()

    # 保存用户消息
    user_msg = Message(
        id=uuid.uuid4(),
        conversation_id=conv.id,
        role="user",
        content=payload.question,
    )
    db.add(user_msg)
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
        pending_done_chunk = None
        evidence_clarification_payload = None
        evidence_clarification_locked = False
        # 会话和用户消息已提交。先告知前端会话 ID，用户在首条回答完成前停止时也能继续该会话。
        yield f"data: {json.dumps({'type': 'conversation_started', 'conversation_id': str(conv.id)})}\n\n"
        try:
            async for chunk in run_rag_stream(
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
            ):
                data = _parse_sse_payload(chunk)
                event_type = data.get("type") if data else None
                # done 必须等 AI 消息持久化成功后再发给前端；其它事件保持实时流式。
                if event_type == "done":
                    pending_done_chunk = chunk
                    continue
                if event_type == "text_delta":
                    if evidence_clarification_locked:
                        # Chat emits the trusted clarification question exactly
                        # once when the gate event arrives.  Any later model or
                        # custom-producer delta must not append an answer.
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
                    non_answer_source_status = (
                        normalized_evidence_status in _NON_ANSWER_SOURCE_STATUSES
                    )
                    if non_answer_source_status:
                        # 状态是服务端最终证据门控：即使异常/旧版/自定义流生产者
                        # 同时错误携带了 answer_sources 和非零 direct 数，也不得
                        # 把这些正文保存成历史回答依据。持久化层必须 fail closed，
                        # 不能只依赖正常 Pipeline 或前端隐藏。
                        direct_evidence_count = 0
                    if not has_answer_source_list or non_answer_source_status:
                        # Fail closed for rolling upgrades, custom stream
                        # producers, and all non-answer evidence states: broad
                        # candidates must never be persisted as answer sources.
                        raw_answer_sources = []
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
                        for source in raw_answer_sources[:20]
                        if isinstance(source, dict)
                    ]
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
        try:
            async with AsyncSessionLocal() as save_db:
                ai_msg = Message(
                    id=uuid.uuid4(),
                    conversation_id=conv.id,
                    role="assistant",
                    content=answer,
                    sources=sources,
                    tokens=tokens,
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

                if routing_result.route_log_id is not None:
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
            raise
        except Exception as exc:
            log_exception_safely(
                logger,
                "[chat/persistence error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=exc,
            )
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
        if pending_done_chunk is not None:
            yield pending_done_chunk
        else:
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            # 首条 SSE 数据到达前，前端也可从响应头立即绑定会话，降低刚开始就停止时丢失会话 ID 的概率。
            "X-Conversation-ID": str(conv.id),
            # 便于开发阶段把浏览器请求与结构化 rag.trace 日志精确关联。
            "X-RAG-Trace-ID": trace_id,
        },
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
    return await _messages_with_current_source_scope(rows, user=user, db=db)


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
