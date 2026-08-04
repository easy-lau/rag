"""Strict semantic contract for RAG query routing.

The route model is allowed to describe semantic intent and select request-local
conversation candidates.  It is deliberately not allowed to choose database
identities, permissions, retrieval switches, or executable response modes.

This module has no database or application-service dependencies so the same
schema and parser can be reused by the model workflow, test endpoint, and
offline evaluator without creating import cycles.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping


ROUTE_DECISION_SCHEMA_VERSION = "rag_route_decision.v1"
ROUTE_DECISION_SCHEMA_NAME = "rag_route_decision_v1"

MAX_CONTEXT_TURN_KEYS = 3
MAX_REQUIREMENTS = 6
MAX_UNRESOLVED_SLOTS = 6
MAX_REQUIREMENT_DESCRIPTION_CHARS = 240
MAX_CLARIFICATION_QUESTION_CHARS = 500
MAX_RATIONALE_CHARS = 500
MAX_UNRESOLVED_ROLE_CHARS = 64

Readiness = Literal["ready", "needs_clarification"]
Relation = Literal["new", "followup", "correction", "continuation"]
EvidenceScope = Literal[
    "enterprise_kb",
    "current_input",
    "platform_self",
    "general_world",
    "mixed",
]
QueryResolutionMode = Literal["current", "contextualize"]
RequirementRole = Literal["answer", "bridge"]
RequirementOrigin = Literal["user_text", "semantically_entailed"]
UnresolvedReason = Literal["missing", "ambiguous", "unavailable"]

VALID_READINESS = {"ready", "needs_clarification"}
VALID_RELATIONS = {"new", "followup", "correction", "continuation"}
VALID_EVIDENCE_SCOPES = {
    "enterprise_kb",
    "current_input",
    "platform_self",
    "general_world",
    "mixed",
}
VALID_QUERY_MODES = {"current", "contextualize"}
VALID_REQUIREMENT_ROLES = {"answer", "bridge"}
VALID_REQUIREMENT_ORIGINS = {"user_text", "semantically_entailed"}
VALID_UNRESOLVED_REASONS = {"missing", "ambiguous", "unavailable"}

_CANDIDATE_KEY_RE = re.compile(r"^t[1-9][0-9]{0,2}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "readiness",
    "intent_code",
    "relation",
    "evidence_scope",
    "query_resolution",
    "requirements",
    "clarification",
    "confidence",
    "rationale",
}
_QUERY_RESOLUTION_KEYS = {"mode", "context_turn_keys"}
_REQUIREMENT_KEYS = {"role", "origin", "description"}
_CLARIFICATION_KEYS = {"question", "unresolved"}
_UNRESOLVED_KEYS = {"role", "reason", "candidate_keys"}


class RouteDecisionValidationError(ValueError):
    """Raised when model output is not an exact ``rag_route_decision.v1``."""


@dataclass(frozen=True)
class RouteQueryResolution:
    mode: QueryResolutionMode
    context_turn_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "context_turn_keys": list(self.context_turn_keys),
        }


@dataclass(frozen=True)
class RouteRequirement:
    role: RequirementRole
    origin: RequirementOrigin
    description: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "origin": self.origin,
            "description": self.description,
        }


@dataclass(frozen=True)
class RouteUnresolvedSlot:
    role: str
    reason: UnresolvedReason
    candidate_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "reason": self.reason,
            "candidate_keys": list(self.candidate_keys),
        }


@dataclass(frozen=True)
class RouteClarification:
    question: str
    unresolved: tuple[RouteUnresolvedSlot, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "unresolved": [item.to_dict() for item in self.unresolved],
        }


@dataclass(frozen=True)
class RagRouteDecision:
    schema_version: Literal["rag_route_decision.v1"]
    readiness: Readiness
    intent_code: str
    relation: Relation
    evidence_scope: EvidenceScope
    query_resolution: RouteQueryResolution
    requirements: tuple[RouteRequirement, ...]
    clarification: RouteClarification
    confidence: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "readiness": self.readiness,
            "intent_code": self.intent_code,
            "relation": self.relation,
            "evidence_scope": self.evidence_scope,
            "query_resolution": self.query_resolution.to_dict(),
            "requirements": [item.to_dict() for item in self.requirements],
            "clarification": self.clarification.to_dict(),
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def _normalized_choice_list(
    values: Iterable[str],
    *,
    field: str,
    candidate_keys: bool = False,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise RouteDecisionValidationError(f"{field} 必须只包含字符串")
        value = raw.strip()
        if not value:
            raise RouteDecisionValidationError(f"{field} 不能包含空值")
        if candidate_keys and not _CANDIDATE_KEY_RE.fullmatch(value):
            raise RouteDecisionValidationError(
                f"{field} 包含非法候选键: {value}"
            )
        if value in seen:
            raise RouteDecisionValidationError(f"{field} 包含重复值: {value}")
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def normalize_intent_codes(values: Iterable[str]) -> tuple[str, ...]:
    """Return a unique, non-empty intent-code whitelist for one request."""

    codes = _normalized_choice_list(values, field="allowed_intent_codes")
    if not codes:
        raise RouteDecisionValidationError("allowed_intent_codes 不能为空")
    for code in codes:
        if not _IDENTIFIER_RE.fullmatch(code):
            raise RouteDecisionValidationError(f"非法 intent_code: {code}")
    return codes


def normalize_turn_candidate_keys(values: Iterable[str]) -> tuple[str, ...]:
    """Validate request-local conversation candidate keys.

    Candidate keys are intentionally small opaque handles.  Database UUIDs and
    arbitrary model-created identifiers never pass this boundary.
    """

    return _normalized_choice_list(
        values,
        field="available_turn_keys",
        candidate_keys=True,
    )


def _candidate_array_schema(available_turn_keys: tuple[str, ...]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "maxItems": min(MAX_CONTEXT_TURN_KEYS, len(available_turn_keys)),
        "items": {"type": "string"},
    }
    # OpenAI-compatible strict-schema providers do not consistently accept
    # ``uniqueItems``.  The local parser below already rejects duplicates, so
    # omitting this unsupported annotation preserves the exact same trust
    # boundary without forcing a slow fallback-model request.
    if available_turn_keys:
        schema["items"]["enum"] = list(available_turn_keys)
    else:
        # ``enum: []`` is invalid JSON Schema. maxItems=0 makes the empty array
        # the only valid value while keeping the schema accepted by providers.
        schema["items"]["pattern"] = _CANDIDATE_KEY_RE.pattern
    return schema


def build_rag_route_decision_schema(
    *,
    allowed_intent_codes: Iterable[str],
    available_turn_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the request-specific strict JSON Schema sent to the route model."""

    intent_codes = normalize_intent_codes(allowed_intent_codes)
    turn_keys = normalize_turn_candidate_keys(available_turn_keys)
    candidate_array = _candidate_array_schema(turn_keys)

    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_TOP_LEVEL_KEYS),
        "properties": {
            "schema_version": {
                "type": "string",
                "const": ROUTE_DECISION_SCHEMA_VERSION,
            },
            "readiness": {
                "type": "string",
                "enum": sorted(VALID_READINESS),
            },
            "intent_code": {
                "type": "string",
                "enum": list(intent_codes),
            },
            "relation": {
                "type": "string",
                "enum": sorted(VALID_RELATIONS),
            },
            "evidence_scope": {
                "type": "string",
                "enum": sorted(VALID_EVIDENCE_SCOPES),
            },
            "query_resolution": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_QUERY_RESOLUTION_KEYS),
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": sorted(VALID_QUERY_MODES),
                    },
                    "context_turn_keys": candidate_array,
                },
            },
            "requirements": {
                "type": "array",
                "maxItems": MAX_REQUIREMENTS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(_REQUIREMENT_KEYS),
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": sorted(VALID_REQUIREMENT_ROLES),
                        },
                        "origin": {
                            "type": "string",
                            "enum": sorted(VALID_REQUIREMENT_ORIGINS),
                        },
                        "description": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_REQUIREMENT_DESCRIPTION_CHARS,
                        },
                    },
                },
            },
            "clarification": {
                "type": "object",
                "additionalProperties": False,
                "required": list(_CLARIFICATION_KEYS),
                "properties": {
                    "question": {
                        "type": "string",
                        "maxLength": MAX_CLARIFICATION_QUESTION_CHARS,
                    },
                    "unresolved": {
                        "type": "array",
                        "maxItems": MAX_UNRESOLVED_SLOTS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(_UNRESOLVED_KEYS),
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": MAX_UNRESOLVED_ROLE_CHARS,
                                    "pattern": _IDENTIFIER_RE.pattern,
                                },
                                "reason": {
                                    "type": "string",
                                    "enum": sorted(VALID_UNRESOLVED_REASONS),
                                },
                                "candidate_keys": candidate_array,
                            },
                        },
                    },
                },
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "rationale": {
                "type": "string",
                "maxLength": MAX_RATIONALE_CHARS,
            },
        },
    }


def build_rag_route_response_format(
    *,
    allowed_intent_codes: Iterable[str],
    available_turn_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Return an OpenAI-compatible strict ``response_format`` envelope."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": ROUTE_DECISION_SCHEMA_NAME,
            "strict": True,
            "schema": build_rag_route_decision_schema(
                allowed_intent_codes=allowed_intent_codes,
                available_turn_keys=available_turn_keys,
            ),
        },
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RouteDecisionValidationError(f"JSON 包含重复字段: {key}")
        result[key] = value
    return result


def _load_payload(value: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise RouteDecisionValidationError("路由模型响应为空")
        try:
            loaded = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except RouteDecisionValidationError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RouteDecisionValidationError("路由模型响应不是合法 JSON") from exc
        if not isinstance(loaded, dict):
            raise RouteDecisionValidationError("路由模型响应必须是 JSON 对象")
        return loaded
    if isinstance(value, Mapping):
        return value
    raise RouteDecisionValidationError("路由决定必须是 JSON 字符串或对象")


def _exact_object(
    value: Any,
    expected_keys: set[str],
    field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RouteDecisionValidationError(f"{field} 必须是对象")
    actual = set(value.keys())
    if actual != expected_keys:
        missing = sorted(expected_keys - actual)
        extra = sorted(actual - expected_keys)
        raise RouteDecisionValidationError(
            f"{field} 字段不完整或包含额外字段: missing={missing}, extra={extra}"
        )
    return value


def _string(
    value: Any,
    field: str,
    *,
    min_length: int = 0,
    max_length: int,
) -> str:
    if not isinstance(value, str):
        raise RouteDecisionValidationError(f"{field} 必须是字符串")
    if len(value) < min_length or len(value) > max_length:
        raise RouteDecisionValidationError(
            f"{field} 长度必须位于 {min_length}~{max_length}"
        )
    return value


def _array(value: Any, field: str, *, max_items: int) -> list[Any]:
    if not isinstance(value, list):
        raise RouteDecisionValidationError(f"{field} 必须是数组")
    if len(value) > max_items:
        raise RouteDecisionValidationError(
            f"{field} 最多允许 {max_items} 项"
        )
    return value


def _candidate_keys(
    value: Any,
    *,
    field: str,
    available_turn_keys: set[str],
) -> tuple[str, ...]:
    items = _array(value, field, max_items=MAX_CONTEXT_TURN_KEYS)
    keys = _normalized_choice_list(items, field=field, candidate_keys=True)
    unknown = [key for key in keys if key not in available_turn_keys]
    if unknown:
        raise RouteDecisionValidationError(
            f"{field} 引用了本次请求不存在的候选键: {unknown}"
        )
    return keys


def parse_rag_route_decision(
    value: str | Mapping[str, Any],
    *,
    allowed_intent_codes: Iterable[str],
    available_turn_keys: Iterable[str] = (),
) -> RagRouteDecision:
    """Parse and validate one exact semantic route decision.

    Unlike the legacy parser this function does not extract JSON from Markdown
    or ignore unknown fields.  A provider response must be the complete object.
    """

    allowed_codes = set(normalize_intent_codes(allowed_intent_codes))
    available_keys = set(normalize_turn_candidate_keys(available_turn_keys))
    payload = _exact_object(_load_payload(value), _TOP_LEVEL_KEYS, "route_decision")

    schema_version = payload["schema_version"]
    if schema_version != ROUTE_DECISION_SCHEMA_VERSION:
        raise RouteDecisionValidationError(
            f"不支持的 schema_version: {schema_version}"
        )

    readiness = payload["readiness"]
    if readiness not in VALID_READINESS:
        raise RouteDecisionValidationError(f"无效 readiness: {readiness}")
    intent_code = payload["intent_code"]
    if not isinstance(intent_code, str) or intent_code not in allowed_codes:
        raise RouteDecisionValidationError(f"intent_code 不在允许列表中: {intent_code}")
    relation = payload["relation"]
    if relation not in VALID_RELATIONS:
        raise RouteDecisionValidationError(f"无效 relation: {relation}")
    evidence_scope = payload["evidence_scope"]
    if evidence_scope not in VALID_EVIDENCE_SCOPES:
        raise RouteDecisionValidationError(
            f"无效 evidence_scope: {evidence_scope}"
        )

    query_payload = _exact_object(
        payload["query_resolution"],
        _QUERY_RESOLUTION_KEYS,
        "query_resolution",
    )
    query_mode = query_payload["mode"]
    if query_mode not in VALID_QUERY_MODES:
        raise RouteDecisionValidationError(f"无效 query_resolution.mode: {query_mode}")
    context_turn_keys = _candidate_keys(
        query_payload["context_turn_keys"],
        field="query_resolution.context_turn_keys",
        available_turn_keys=available_keys,
    )

    requirements: list[RouteRequirement] = []
    for index, raw_item in enumerate(
        _array(payload["requirements"], "requirements", max_items=MAX_REQUIREMENTS)
    ):
        item = _exact_object(raw_item, _REQUIREMENT_KEYS, f"requirements[{index}]")
        role = item["role"]
        if role not in VALID_REQUIREMENT_ROLES:
            raise RouteDecisionValidationError(
                f"requirements[{index}].role 无效: {role}"
            )
        origin = item["origin"]
        if origin not in VALID_REQUIREMENT_ORIGINS:
            raise RouteDecisionValidationError(
                f"requirements[{index}].origin 无效: {origin}"
            )
        description = _string(
            item["description"],
            f"requirements[{index}].description",
            min_length=1,
            max_length=MAX_REQUIREMENT_DESCRIPTION_CHARS,
        )
        if not description.strip():
            raise RouteDecisionValidationError(
                f"requirements[{index}].description 不能为空白"
            )
        requirements.append(
            RouteRequirement(
                role=role,
                origin=origin,
                description=description,
            )
        )

    clarification_payload = _exact_object(
        payload["clarification"],
        _CLARIFICATION_KEYS,
        "clarification",
    )
    clarification_question = _string(
        clarification_payload["question"],
        "clarification.question",
        max_length=MAX_CLARIFICATION_QUESTION_CHARS,
    )
    unresolved: list[RouteUnresolvedSlot] = []
    for index, raw_item in enumerate(
        _array(
            clarification_payload["unresolved"],
            "clarification.unresolved",
            max_items=MAX_UNRESOLVED_SLOTS,
        )
    ):
        item = _exact_object(
            raw_item,
            _UNRESOLVED_KEYS,
            f"clarification.unresolved[{index}]",
        )
        role = _string(
            item["role"],
            f"clarification.unresolved[{index}].role",
            min_length=1,
            max_length=MAX_UNRESOLVED_ROLE_CHARS,
        )
        if not _IDENTIFIER_RE.fullmatch(role):
            raise RouteDecisionValidationError(
                f"clarification.unresolved[{index}].role 必须是稳定标识符"
            )
        reason = item["reason"]
        if reason not in VALID_UNRESOLVED_REASONS:
            raise RouteDecisionValidationError(
                f"clarification.unresolved[{index}].reason 无效: {reason}"
            )
        candidate_keys = _candidate_keys(
            item["candidate_keys"],
            field=f"clarification.unresolved[{index}].candidate_keys",
            available_turn_keys=available_keys,
        )
        # A context-turn ambiguity must compare at least two historical
        # candidates.  Object/goal or product/version ambiguity can be a
        # perfectly valid unresolved slot even when only one prior turn
        # provides the topic anchor; rejecting it here discards useful model
        # semantics and forces an unrelated fallback query.
        if (
            reason == "ambiguous"
            and role == "context_turn"
            and len(candidate_keys) < 2
        ):
            raise RouteDecisionValidationError(
                "ambiguous unresolved 至少需要两个 candidate_keys"
            )
        # A missing slot may still point at the bounded history candidates
        # that establish why the slot is unresolved (for example, a prior
        # turn says "普通员工" but never confirms that it describes the user).
        # Candidate keys remain request-local and are validated above.  An
        # unavailable slot, however, has no usable context to bind.
        if reason == "unavailable" and candidate_keys:
            raise RouteDecisionValidationError(
                f"{reason} unresolved 不得携带 candidate_keys"
            )
        unresolved.append(
            RouteUnresolvedSlot(
                role=role,
                reason=reason,
                candidate_keys=candidate_keys,
            )
        )

    confidence_value = payload["confidence"]
    if isinstance(confidence_value, bool) or not isinstance(
        confidence_value, (int, float)
    ):
        raise RouteDecisionValidationError("confidence 必须是数字")
    confidence = float(confidence_value)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise RouteDecisionValidationError("confidence 必须位于 0~1")
    rationale = _string(
        payload["rationale"],
        "rationale",
        max_length=MAX_RATIONALE_CHARS,
    )

    if readiness == "ready":
        if clarification_question != "" or unresolved:
            raise RouteDecisionValidationError(
                "ready 路由的 clarification 必须完全为空"
            )
        if not any(item.role == "answer" for item in requirements):
            raise RouteDecisionValidationError(
                "ready 路由必须包含至少一个 answer requirement"
            )
    elif not clarification_question.strip() or not unresolved:
        raise RouteDecisionValidationError(
            "needs_clarification 必须包含问题和至少一个 unresolved"
        )

    if relation == "new" and context_turn_keys:
        raise RouteDecisionValidationError("new 路由不得绑定历史 turn candidate")
    if query_mode == "contextualize" and not context_turn_keys:
        raise RouteDecisionValidationError(
            "contextualize 必须绑定至少一个历史 turn candidate"
        )

    return RagRouteDecision(
        schema_version=ROUTE_DECISION_SCHEMA_VERSION,
        readiness=readiness,
        intent_code=intent_code,
        relation=relation,
        evidence_scope=evidence_scope,
        query_resolution=RouteQueryResolution(
            mode=query_mode,
            context_turn_keys=context_turn_keys,
        ),
        requirements=tuple(requirements),
        clarification=RouteClarification(
            question=clarification_question,
            unresolved=tuple(unresolved),
        ),
        confidence=confidence,
        rationale=rationale,
    )


__all__ = [
    "MAX_CONTEXT_TURN_KEYS",
    "MAX_REQUIREMENTS",
    "ROUTE_DECISION_SCHEMA_NAME",
    "ROUTE_DECISION_SCHEMA_VERSION",
    "RagRouteDecision",
    "RouteClarification",
    "RouteDecisionValidationError",
    "RouteQueryResolution",
    "RouteRequirement",
    "RouteUnresolvedSlot",
    "build_rag_route_decision_schema",
    "build_rag_route_response_format",
    "normalize_intent_codes",
    "normalize_turn_candidate_keys",
    "parse_rag_route_decision",
]
