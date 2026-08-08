"""Unified clarification contract, durable state and reply resolution.

Clarification is one conversation capability regardless of whether ambiguity
is discovered before retrieval or after evidence adjudication.  Producers emit
structured facts; this module owns persistence validation and user selection
semantics.  Execution adapters remain separate because semantic recompilation
and evidence authorization have different security boundaries.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Sequence


CLARIFICATION_CONTRACT_SCHEMA = "rag_clarification_contract.v2"
CLARIFICATION_STATE_SCHEMA = "rag_clarification_state.v1"
CLARIFICATION_EVENT_SCHEMA = "rag_clarification_state.v1"

ClarificationAdapter = Literal["semantic", "evidence"]
ClarificationSelectionMode = Literal["choice", "refine"]
ClarificationSelectionPolicy = Literal["single", "single_or_all"]
ClarificationAction = Literal[
    "single",
    "all",
    "refine",
    "cancel",
    "new_question",
    "repeat",
]
ClarificationCommandAction = Literal[
    "select",
    "select_all",
    "refine",
    "cancel",
    "new_question",
]

_MAX_CHOICES = 20
_MAX_TEXT = 12000
_MAX_REPLY = 1200
_MAX_METADATA_ITEMS = 20
_MAX_RESOURCE_IDS = 200
_MAX_SCOPE_SLICES = 100
_DIMENSION_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CHOICE_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,119}")
_UUID_FIELDS = (
    "kb_ids",
    "doc_ids",
    "record_ids",
    "anchor_doc_ids",
    "companion_doc_ids",
)
_TEXT_LIST_FIELDS = (
    "products",
    "canonical_products",
    "versions",
    "projects",
    "filenames",
)

_CANCEL_RE = re.compile(
    r"^(?:取消|算了|不用了?|不需要|暂时不需要|先不用|不了|停止|退出)"
    r"(?:吧|。|！|!)?$",
    re.IGNORECASE,
)
_ALL_RE = re.compile(
    r"^(?:全部|全都|都要|都查|都看|都比较|都对比|全部比较|全部对比|"
    r"全部都要|分别比较|分别对比|所有(?:产品|版本|项目|范围|文档)|"
    r"全部(?:产品|版本|项目|范围|文档))(?:一下|吧|。|！|!)?$",
    re.IGNORECASE,
)
_NEW_QUESTION_RE = re.compile(
    r"[?？]|(?:换(?:个|一个)(?:问题|话题)|另一个问题|新问题|重新问|"
    r"请问|帮我|告诉我|怎么|如何|为什么|为何|什么|哪些|哪个|哪里|"
    r"是否|能否|可以吗|介绍|说明|分析|标准|规定|配置|流程|方法|要求)",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(r"(?:对比|比较|区别|差异|分别)", re.IGNORECASE)
_SELECTION_FILLERS_RE = re.compile(
    r"(?:选择|查询|看看|使用|按照|确认|确定|改成|改为|换成|换为|想要|需要|"
    r"我|请|想|要|选|就|查|看|用|按|换|改|这个|那个|该|这项|那项|"
    r"这篇|那篇|版本|版|产品|项目|范围|文档|项|个|的|吧|呀|啊|呢|哦|哈)",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(
    r"第\s*(?P<value>[1-9]\d*|[一二三四五六七八九十两]+)\s*"
    r"(?:项|个|篇|份|条|版本|文档)?",
    re.IGNORECASE,
)
_NUMERIC_SELECTION_RE = re.compile(
    r"(?:选(?:择)?\s*)?(?:c\s*)?(?P<value>[1-9]\d*)"
    r"(?:\s*(?:项|个|篇|份|条|版本|文档))?"
    r"(?:吧|呀|啊|呢|哦|。|！|!)?",
    re.IGNORECASE,
)


def _bounded_text(value: object, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _strict_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > limit:
        return None
    return text


def _normalized_text(value: object) -> str:
    return re.sub(
        r"[\s\-—_·,，。!！?？:：;；、《》〈〉()（）\[\]【】'\"]+",
        "",
        str(value or ""),
    ).casefold()


def _unique_texts(value: object, *, limit: int = _MAX_METADATA_ITEMS) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) > limit:
        return None
    output: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = _strict_text(raw, 500)
        if item is None:
            return None
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _uuid_list(value: object) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_RESOURCE_IDS:
        return None
    output: list[str] = []
    seen: set[str] = set()
    for raw in value:
        try:
            item = str(uuid.UUID(str(raw)))
        except (TypeError, ValueError, AttributeError):
            return None
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def _scope_slices(value: object) -> list[dict[str, Any]] | None:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_SCOPE_SLICES:
        return None
    output: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            return None
        kb_ids = _uuid_list([raw.get("kb_id")])
        doc_ids = _uuid_list([raw.get("doc_id")])
        chunk_ids = _uuid_list(raw.get("chunk_ids"))
        if kb_ids is None or doc_ids is None or chunk_ids is None:
            return None
        section_key = _bounded_text(raw.get("section_key"), 500) or None
        if not section_key and not chunk_ids:
            return None
        output.append({
            "kb_id": kb_ids[0],
            "doc_id": doc_ids[0],
            "section_key": section_key,
            "chunk_ids": chunk_ids,
            "is_anchor": raw.get("is_anchor") is not False,
        })
    return output


def normalize_choice(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    key = _strict_text(value.get("key"), 120)
    label = _strict_text(value.get("label"), 500)
    raw_choice_value = value.get("value")
    choice_value = (
        label
        if raw_choice_value in (None, "")
        else _strict_text(raw_choice_value, 500)
    )
    if (
        key is None
        or label is None
        or choice_value is None
        or not _CHOICE_KEY_RE.fullmatch(key)
    ):
        return None
    output: dict[str, Any] = {
        "key": key,
        "label": label,
        "value": choice_value,
    }
    for field in _TEXT_LIST_FIELDS:
        items = _unique_texts(value.get(field))
        if items is None:
            return None
        output[field] = items
    for field in _UUID_FIELDS:
        items = _uuid_list(value.get(field))
        if items is None:
            return None
        output[field] = items
    slices = _scope_slices(value.get("scope_slices"))
    if slices is None:
        return None
    output["scope_slices"] = slices
    return output


@dataclass(frozen=True)
class ClarificationContract:
    """Structured ambiguity facts; never a rendered business answer."""

    adapter: ClarificationAdapter
    dimension: str
    reason_code: str
    selection_mode: ClarificationSelectionMode
    selection_policy: ClarificationSelectionPolicy = "single"
    choices: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.adapter not in {"semantic", "evidence"}:
            raise ValueError("clarification adapter is invalid")
        if not _DIMENSION_RE.fullmatch(self.dimension):
            raise ValueError("clarification dimension is invalid")
        if not _DIMENSION_RE.fullmatch(self.reason_code):
            raise ValueError("clarification reason_code is invalid")
        if self.selection_mode not in {"choice", "refine"}:
            raise ValueError("clarification selection_mode is invalid")
        if self.selection_policy not in {"single", "single_or_all"}:
            raise ValueError("clarification selection_policy is invalid")
        if self.selection_mode == "refine" and self.selection_policy != "single":
            raise ValueError("refine clarification cannot allow multi-selection")
        normalized: list[dict[str, Any]] = []
        for raw in self.choices:
            choice = normalize_choice(raw)
            if choice is None:
                raise ValueError("clarification choice is invalid")
            normalized.append(choice)
        if len(normalized) > _MAX_CHOICES:
            raise ValueError("clarification has too many choices")
        if self.selection_mode == "choice" and not normalized:
            raise ValueError("choice clarification requires choices")
        if self.selection_mode == "refine" and normalized:
            raise ValueError("refine clarification cannot expose choices")
        if len({item["key"] for item in normalized}) != len(normalized):
            raise ValueError("clarification choice keys must be unique")
        object.__setattr__(self, "choices", tuple(normalized))

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        choices: list[dict[str, Any]] = []
        for raw in self.choices:
            choice = dict(raw)
            if public:
                for field in (*_UUID_FIELDS, "scope_slices"):
                    choice.pop(field, None)
            choices.append(choice)
        return {
            "schema_version": CLARIFICATION_CONTRACT_SCHEMA,
            "needs_clarification": True,
            "adapter": self.adapter,
            "dimension": self.dimension,
            "reason_code": self.reason_code,
            "selection_mode": self.selection_mode,
            "selection_policy": self.selection_policy,
            "choices": choices,
            "allowed_actions": [
                "select",
                *(
                    ["select_all"]
                    if self.selection_policy == "single_or_all" and len(choices) > 1
                    else []
                ),
                "cancel",
                "new_question",
            ] if self.selection_mode == "choice" else [
                "refine",
                "cancel",
                "new_question",
            ],
        }


def contract_from_dict(value: object) -> ClarificationContract | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("schema_version") not in {
        CLARIFICATION_CONTRACT_SCHEMA,
        CLARIFICATION_EVENT_SCHEMA,
    }:
        return None
    choices = value.get("choices")
    if not isinstance(choices, list):
        return None
    try:
        return ClarificationContract(
            adapter=str(value.get("adapter") or ""),  # type: ignore[arg-type]
            dimension=str(value.get("dimension") or ""),
            reason_code=str(value.get("reason_code") or ""),
            selection_mode=str(value.get("selection_mode") or ""),  # type: ignore[arg-type]
            selection_policy=str(value.get("selection_policy") or "single"),  # type: ignore[arg-type]
            choices=tuple(choices),
        )
    except (TypeError, ValueError):
        return None


def build_clarification_state(
    *,
    contract: ClarificationContract,
    original_query: str,
    selected_kb_ids: Sequence[object],
    base_user_message_id: object,
    clarification_message_id: object,
    prior_answers: Sequence[object] = (),
    ttl: timedelta = timedelta(hours=24),
) -> dict[str, Any]:
    query = _bounded_text(original_query, _MAX_TEXT)
    if not query:
        raise ValueError("clarification original_query is required")
    kb_ids = _uuid_list(list(selected_kb_ids))
    if kb_ids is None or (contract.adapter == "evidence" and not kb_ids):
        raise ValueError("clarification selected KB snapshot is invalid")
    if any(
        not set(choice.get("kb_ids", [])).issubset(set(kb_ids))
        for choice in contract.choices
    ):
        raise ValueError("clarification choice is outside selected KB snapshot")
    try:
        user_message_id = str(uuid.UUID(str(base_user_message_id)))
        assistant_message_id = str(uuid.UUID(str(clarification_message_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("clarification message identity is invalid") from exc
    answers = [
        answer
        for raw in prior_answers[-6:]
        if (answer := _bounded_text(raw, _MAX_REPLY))
    ]
    now = datetime.now(timezone.utc)
    return {
        "schema_version": CLARIFICATION_STATE_SCHEMA,
        "state_id": str(uuid.uuid4()),
        "contract": contract.to_dict(),
        "original_query": query,
        "prior_answers": answers,
        "selected_kb_ids_snapshot": kb_ids,
        "base_user_message_id": user_message_id,
        "clarification_message_id": assistant_message_id,
        "created_at": now.isoformat(),
        "expires_at": (now + ttl).isoformat(),
        "dispatch_authorized": False,
    }


def _future_datetime(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed if parsed > datetime.now(timezone.utc) else None


def validate_clarification_state(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if (
        value.get("schema_version") != CLARIFICATION_STATE_SCHEMA
        or value.get("dispatch_authorized") is not False
    ):
        return None
    try:
        state_id = str(uuid.UUID(str(value.get("state_id"))))
        base_user_message_id = str(uuid.UUID(str(value.get("base_user_message_id"))))
        clarification_message_id = str(
            uuid.UUID(str(value.get("clarification_message_id")))
        )
    except (TypeError, ValueError, AttributeError):
        return None
    contract = contract_from_dict(value.get("contract"))
    original_query = _bounded_text(value.get("original_query"), _MAX_TEXT)
    expires_at = _future_datetime(value.get("expires_at"))
    try:
        created_at = datetime.fromisoformat(str(value.get("created_at")))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    if (
        contract is None
        or not original_query
        or expires_at is None
        or created_at > datetime.now(timezone.utc) + timedelta(minutes=5)
        or expires_at <= created_at
        or expires_at - created_at > timedelta(hours=24, minutes=5)
    ):
        return None
    selected_kb_ids = _uuid_list(value.get("selected_kb_ids_snapshot"))
    prior_answers = _unique_texts(value.get("prior_answers"), limit=6)
    if (
        selected_kb_ids is None
        or prior_answers is None
        or (contract.adapter == "evidence" and not selected_kb_ids)
    ):
        return None
    selected_kb_set = set(selected_kb_ids)
    if any(
        not set(choice.get("kb_ids", [])).issubset(selected_kb_set)
        for choice in contract.choices
    ):
        return None
    return {
        "schema_version": CLARIFICATION_STATE_SCHEMA,
        "state_id": state_id,
        "contract": contract.to_dict(),
        "original_query": original_query,
        "prior_answers": prior_answers,
        "selected_kb_ids_snapshot": selected_kb_ids,
        "base_user_message_id": base_user_message_id,
        "clarification_message_id": clarification_message_id,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "dispatch_authorized": False,
    }


def public_clarification_event(
    state: Mapping[str, Any],
    *,
    route_state_revision: int,
    conversation_id: object,
    persisted: bool,
) -> dict[str, Any]:
    normalized = validate_clarification_state(state)
    if normalized is None:
        raise ValueError("cannot publish invalid clarification state")
    contract = contract_from_dict(normalized["contract"])
    assert contract is not None
    return {
        **contract.to_dict(public=True),
        "type": "clarification_state",
        "schema_version": CLARIFICATION_EVENT_SCHEMA,
        "status": "active" if persisted else "proposed",
        "persisted": persisted,
        "pending_state_id": normalized["state_id"] if persisted else None,
        "clarification_message_id": (
            normalized["clarification_message_id"] if persisted else None
        ),
        "route_state_revision": route_state_revision if persisted else None,
        "conversation_id": str(conversation_id),
        "selected_kb_ids_snapshot": list(normalized["selected_kb_ids_snapshot"]),
    }


def proposed_clarification_event(
    contract: ClarificationContract,
    *,
    include_private: bool = False,
) -> dict[str, Any]:
    return {
        **contract.to_dict(public=not include_private),
        "type": "clarification_state",
        "schema_version": CLARIFICATION_EVENT_SCHEMA,
        "status": "proposed",
        "persisted": False,
        "pending_state_id": None,
        "clarification_message_id": None,
        "route_state_revision": None,
    }


def _small_chinese_number(value: str) -> int | None:
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    text = str(value or "").strip()
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        ones = digits.get(right, 0) if right else 0
        number = tens * 10 + ones
        return number if 1 <= number <= 99 else None
    return digits.get(text)


def _ordinal_index(text: str) -> tuple[int | None, str]:
    match = _ORDINAL_RE.search(text)
    if match is not None:
        raw = match.group("value")
        value = int(raw) if raw.isdigit() else _small_chinese_number(raw)
        residual = text[: match.start()] + text[match.end() :]
        return value, residual
    match = _NUMERIC_SELECTION_RE.fullmatch(text.strip())
    if match is not None:
        return int(match.group("value")), ""
    return None, text


def _choice_aliases(choice: Mapping[str, Any]) -> set[str]:
    aliases = {
        _normalized_text(choice.get("key")),
        _normalized_text(choice.get("label")),
        _normalized_text(choice.get("value")),
        _normalized_text(choice.get("label")).replace("版本", ""),
    }
    for field in _TEXT_LIST_FIELDS:
        aliases.update(
            _normalized_text(item)
            for item in choice.get(field, [])
            if isinstance(item, str)
        )
    aliases.discard("")
    return aliases


@dataclass(frozen=True)
class ClarificationResolution:
    action: ClarificationAction
    choices: tuple[dict[str, Any], ...] = ()
    answer: str = ""


def resolve_clarification_reply(
    question: str,
    state: Mapping[str, Any],
    *,
    command: Mapping[str, Any] | None = None,
) -> ClarificationResolution:
    normalized_state = validate_clarification_state(state)
    text = _bounded_text(question, _MAX_REPLY)
    if normalized_state is None or not text:
        return ClarificationResolution("repeat")
    contract = contract_from_dict(normalized_state["contract"])
    assert contract is not None
    if command is not None:
        action = str(command.get("action") or "").strip().casefold()
        raw_keys = command.get("choice_keys")
        if not isinstance(raw_keys, (list, tuple)):
            return ClarificationResolution("repeat")
        keys = [str(value or "").strip() for value in raw_keys]
        if any(not _CHOICE_KEY_RE.fullmatch(value) for value in keys):
            return ClarificationResolution("repeat")
        if len(set(keys)) != len(keys):
            return ClarificationResolution("repeat")
        choices_by_key = {str(choice["key"]): choice for choice in contract.choices}
        selected = tuple(choices_by_key[key] for key in keys if key in choices_by_key)
        if len(selected) != len(keys):
            return ClarificationResolution("repeat")
        if action == "select" and len(selected) == 1:
            return ClarificationResolution(
                "single",
                choices=selected,
                answer=str(selected[0]["label"]),
            )
        if (
            action == "select_all"
            and contract.selection_policy == "single_or_all"
            and len(selected) > 1
        ):
            return ClarificationResolution("all", choices=selected, answer=text)
        if action == "refine" and contract.selection_mode == "refine" and not keys:
            return ClarificationResolution("refine", answer=text)
        if action == "cancel" and not keys:
            return ClarificationResolution("cancel")
        if action == "new_question" and not keys:
            return ClarificationResolution("new_question")
        return ClarificationResolution("repeat")
    if _CANCEL_RE.fullmatch(text):
        return ClarificationResolution("cancel")
    if contract.selection_mode == "refine":
        if _NEW_QUESTION_RE.search(text):
            return ClarificationResolution("new_question")
        return ClarificationResolution("refine", answer=text)
    choices = contract.choices
    if _ALL_RE.fullmatch(text):
        if contract.selection_policy != "single_or_all":
            return ClarificationResolution("repeat")
        return ClarificationResolution("all", choices=choices, answer=text)

    normalized_reply = _normalized_text(text)
    choice_aliases = [(_choice_aliases(choice), choice) for choice in choices]
    alias_counts: dict[str, int] = {}
    for aliases, _choice in choice_aliases:
        for alias in aliases:
            alias_counts[alias] = alias_counts.get(alias, 0) + 1
    exact_matches = [
        choice
        for aliases, choice in choice_aliases
        if normalized_reply in aliases
    ]
    if len(exact_matches) == 1:
        return ClarificationResolution(
            "single",
            choices=(exact_matches[0],),
            answer=str(exact_matches[0]["label"]),
        )
    if len(exact_matches) > 1:
        return ClarificationResolution("repeat")

    ordinal, residual = _ordinal_index(text)
    if ordinal is not None:
        if _SELECTION_FILLERS_RE.sub("", _normalized_text(residual)):
            return ClarificationResolution("new_question")
        index = ordinal - 1
        if 0 <= index < len(choices):
            choice = choices[index]
            return ClarificationResolution(
                "single",
                choices=(choice,),
                answer=str(choice["label"]),
            )
        return ClarificationResolution("repeat")

    matched: list[dict[str, Any]] = []
    comparison_requested = bool(_COMPARE_RE.search(text))
    for aliases, choice in choice_aliases:
        if comparison_requested and any(
            len(alias) >= 2
            and alias_counts.get(alias) == 1
            and alias in normalized_reply
            for alias in aliases
        ):
            matched.append(choice)
            continue
        residual_reply = normalized_reply
        matched_alias = False
        for alias in sorted(aliases, key=len, reverse=True):
            if len(alias) < 2 or alias not in residual_reply:
                continue
            residual_reply = residual_reply.replace(alias, "")
            matched_alias = True
        if matched_alias and not _SELECTION_FILLERS_RE.sub("", residual_reply):
            matched.append(choice)
    if len(matched) == 1:
        return ClarificationResolution(
            "single",
            choices=(matched[0],),
            answer=str(matched[0]["label"]),
        )
    if (
        len(matched) > 1
        and comparison_requested
        and contract.selection_policy == "single_or_all"
    ):
        return ClarificationResolution(
            "all",
            choices=tuple(matched),
            answer=text,
        )
    if matched:
        return ClarificationResolution("repeat")
    if _NEW_QUESTION_RE.search(text) or len(text) > 32:
        return ClarificationResolution("new_question")
    return ClarificationResolution("repeat")


__all__ = [
    "CLARIFICATION_CONTRACT_SCHEMA",
    "CLARIFICATION_EVENT_SCHEMA",
    "CLARIFICATION_STATE_SCHEMA",
    "ClarificationContract",
    "ClarificationCommandAction",
    "ClarificationResolution",
    "build_clarification_state",
    "contract_from_dict",
    "normalize_choice",
    "proposed_clarification_event",
    "public_clarification_event",
    "resolve_clarification_reply",
    "validate_clarification_state",
]
