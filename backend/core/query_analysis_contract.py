"""Strict source-anchored candidate contract for model-assisted query analysis.

``query_analysis.v2`` is deliberately a *candidate graph*, not an executable
query plan.  A model may point to literal answer targets and qualifiers already
present in the current user turn (or route-authorised historical user turns),
and may connect a candidate bridge to one of those qualifiers.  It cannot
provide retrieval wording, aliases, scope, coverage, factual values, knowledge
base/document identifiers, permissions, bridge kinds, or DAG edge modes.

The trusted backend compiler is the only component allowed to turn an accepted
candidate graph into retrieval tasks.  Every source reference uses a zero-based
Unicode-code-point half-open range ``[start, end)`` and repeats the exact span;
the parser checks both the offsets and the literal text before accepting it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping


QUERY_ANALYSIS_SCHEMA_VERSION = "query_analysis.v2"
QUERY_ANALYSIS_SCHEMA_NAME = "query_analysis_v2"

MAX_CONTEXT_TURN_KEYS = 3
MAX_ANSWER_CANDIDATES = 8
MAX_BRIDGE_CANDIDATES = 8
MAX_QUALIFIER_REFS = 8
MAX_BRIDGE_REFS = 8
MAX_SOURCE_SPAN_CHARS = 160
MAX_DIAGNOSTIC_CHARS = 300

Relation = Literal["new", "followup", "correction", "continuation"]

VALID_RELATIONS = frozenset({"new", "followup", "correction", "continuation"})

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ANSWER_ID_RE = re.compile(r"^a[1-9][0-9]{0,2}$")
_BRIDGE_ID_RE = re.compile(r"^b[1-9][0-9]{0,2}$")
_TURN_KEY_RE = re.compile(r"^t[1-9][0-9]{0,2}$")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "relation",
    "self_contained",
    "context_turn_keys",
    "answer_candidates",
    "bridge_candidates",
    "confidence",
    "diagnostic",
}
_SOURCE_REF_KEYS = {"turn_key", "start", "end", "span"}
_ANSWER_CANDIDATE_KEYS = {
    "id",
    "target_source_ref",
    "qualifier_source_refs",
    "bridge_candidate_ids",
}
_BRIDGE_CANDIDATE_KEYS = {"id", "subject_source_ref"}


class QueryAnalysisValidationError(ValueError):
    """Raised when a model response violates ``query_analysis.v2``."""


def _expect_exact_keys(value: object, *, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueryAnalysisValidationError(f"{field} 必须是对象")
    actual = set(value)
    if actual == keys:
        return value
    missing = sorted(keys - actual)
    extra = sorted(actual - keys)
    details: list[str] = []
    if missing:
        details.append("缺少 " + ",".join(missing))
    if extra:
        details.append("包含未允许字段 " + ",".join(extra))
    raise QueryAnalysisValidationError(
        f"{field} 字段不精确: {'；'.join(details) or '未知原因'}"
    )


def _parse_json_object(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise QueryAnalysisValidationError("模型输出为空或不是 JSON 字符串")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise QueryAnalysisValidationError(f"JSON 包含重复字段: {key}")
            output[key] = value
        return output

    def reject_non_finite(value: str) -> None:
        raise QueryAnalysisValidationError(f"JSON 不允许非有限数字: {value}")

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except QueryAnalysisValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QueryAnalysisValidationError("模型输出不是严格 JSON 对象") from exc
    if not isinstance(parsed, dict):
        raise QueryAnalysisValidationError("模型输出顶层必须是对象")
    return parsed


def _source_text(value: object, *, field: str, max_chars: int) -> str:
    """Keep source bytes/code points intact so offsets remain meaningful."""

    if not isinstance(value, str):
        raise QueryAnalysisValidationError(f"{field} 必须是字符串")
    if not value.strip():
        raise QueryAnalysisValidationError(f"{field} 不能为空")
    if len(value) > max_chars:
        raise QueryAnalysisValidationError(f"{field} 超过最大长度")
    return value


def _normalized_text(value: object, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise QueryAnalysisValidationError(f"{field} 必须是字符串")
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise QueryAnalysisValidationError(f"{field} 不能为空")
    if len(normalized) > max_chars:
        raise QueryAnalysisValidationError(f"{field} 超过最大长度")
    return normalized


def _parse_identifier(
    value: object,
    *,
    field: str,
    pattern: re.Pattern[str],
) -> str:
    identifier = _normalized_text(value, field=field, max_chars=64)
    if not pattern.fullmatch(identifier):
        raise QueryAnalysisValidationError(f"{field} 包含非法标识")
    return identifier


def _parse_id_list(
    value: object,
    *,
    field: str,
    allowed_ids: set[str],
    max_items: int,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list):
        raise QueryAnalysisValidationError(f"{field} 必须是数组")
    if len(value) > max_items:
        raise QueryAnalysisValidationError(f"{field} 超过上限")
    result: list[str] = []
    for index, raw in enumerate(value):
        identifier = _parse_identifier(
            raw,
            field=f"{field}[{index}]",
            pattern=_IDENTIFIER_RE,
        )
        if identifier in result:
            raise QueryAnalysisValidationError(f"{field} 包含重复标识")
        if identifier not in allowed_ids:
            raise QueryAnalysisValidationError(f"{field} 引用了不存在的标识: {identifier}")
        result.append(identifier)
    return tuple(result)


@dataclass(frozen=True)
class QueryAnalysisSourceRef:
    """Exact source reference using zero-based Unicode-code-point offsets."""

    turn_key: str
    start: int
    end: int
    span: str

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_key": self.turn_key,
            "start": self.start,
            "end": self.end,
            "span": self.span,
        }

    @property
    def identity(self) -> tuple[str, int, int, str]:
        return (self.turn_key, self.start, self.end, self.span)


@dataclass(frozen=True)
class QueryAnalysisBridgeCandidate:
    """A non-executable candidate bridge around one source qualifier."""

    id: str
    subject_source_ref: QueryAnalysisSourceRef

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "subject_source_ref": self.subject_source_ref.to_dict(),
        }


@dataclass(frozen=True)
class QueryAnalysisAnswerCandidate:
    """A proposed answer target plus literal qualifier/bridge references."""

    id: str
    target_source_ref: QueryAnalysisSourceRef
    qualifier_source_refs: tuple[QueryAnalysisSourceRef, ...]
    bridge_candidate_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "target_source_ref": self.target_source_ref.to_dict(),
            "qualifier_source_refs": [
                item.to_dict() for item in self.qualifier_source_refs
            ],
            "bridge_candidate_ids": list(self.bridge_candidate_ids),
        }


@dataclass(frozen=True)
class QueryAnalysis:
    """Validated model candidate graph with no execution authority."""

    schema_version: Literal["query_analysis.v2"]
    relation: Relation
    self_contained: bool
    context_turn_keys: tuple[str, ...]
    answer_candidates: tuple[QueryAnalysisAnswerCandidate, ...]
    bridge_candidates: tuple[QueryAnalysisBridgeCandidate, ...]
    confidence: float
    diagnostic: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relation": self.relation,
            "self_contained": self.self_contained,
            "context_turn_keys": list(self.context_turn_keys),
            "answer_candidates": [
                item.to_dict() for item in self.answer_candidates
            ],
            "bridge_candidates": [
                item.to_dict() for item in self.bridge_candidates
            ],
            "confidence": self.confidence,
            "diagnostic": self.diagnostic,
        }

    def safe_summary(self) -> dict[str, Any]:
        """Content-free diagnostics safe for production traces."""

        return {
            "schema_version": self.schema_version,
            "relation": self.relation,
            "self_contained": self.self_contained,
            "context_turn_count": len(self.context_turn_keys),
            "answer_candidate_count": len(self.answer_candidates),
            "bridge_candidate_count": len(self.bridge_candidates),
            "confidence": self.confidence,
        }


def _parse_source_ref(
    value: object,
    *,
    field: str,
    available_source_texts: Mapping[str, str],
) -> QueryAnalysisSourceRef:
    raw = _expect_exact_keys(value, keys=_SOURCE_REF_KEYS, field=field)
    turn_key = raw["turn_key"]
    if not isinstance(turn_key, str):
        raise QueryAnalysisValidationError(f"{field}.turn_key 必须是字符串")
    turn_key = turn_key.strip()
    if turn_key != "current" and not _TURN_KEY_RE.fullmatch(turn_key):
        raise QueryAnalysisValidationError(f"{field}.turn_key 非法")
    source_text = available_source_texts.get(turn_key)
    if source_text is None:
        raise QueryAnalysisValidationError(f"{field}.turn_key 不在允许范围")

    start = raw["start"]
    end = raw["end"]
    if isinstance(start, bool) or not isinstance(start, int):
        raise QueryAnalysisValidationError(f"{field}.start 必须是整数")
    if isinstance(end, bool) or not isinstance(end, int):
        raise QueryAnalysisValidationError(f"{field}.end 必须是整数")
    if start < 0 or end <= start or end > len(source_text):
        raise QueryAnalysisValidationError(f"{field} 偏移范围非法")
    span = raw["span"]
    if not isinstance(span, str):
        raise QueryAnalysisValidationError(f"{field}.span 必须是字符串")
    if not span.strip():
        raise QueryAnalysisValidationError(f"{field}.span 不能为空")
    if len(span) > MAX_SOURCE_SPAN_CHARS:
        raise QueryAnalysisValidationError(f"{field}.span 超过最大长度")
    if source_text[start:end] != span:
        raise QueryAnalysisValidationError(f"{field} 的 start/end/span 与来源原文不精确一致")
    return QueryAnalysisSourceRef(
        turn_key=turn_key,
        start=start,
        end=end,
        span=span,
    )


def _parse_source_ref_list(
    value: object,
    *,
    field: str,
    available_source_texts: Mapping[str, str],
    max_items: int,
) -> tuple[QueryAnalysisSourceRef, ...]:
    if not isinstance(value, list):
        raise QueryAnalysisValidationError(f"{field} 必须是数组")
    if len(value) > max_items:
        raise QueryAnalysisValidationError(f"{field} 超过上限")
    result: list[QueryAnalysisSourceRef] = []
    seen: set[tuple[str, int, int, str]] = set()
    for index, raw in enumerate(value):
        source = _parse_source_ref(
            raw,
            field=f"{field}[{index}]",
            available_source_texts=available_source_texts,
        )
        if source.identity in seen:
            raise QueryAnalysisValidationError(f"{field} 包含重复来源片段")
        seen.add(source.identity)
        result.append(source)
    return tuple(result)


def _parse_bridge_candidates(
    value: object,
    *,
    available_source_texts: Mapping[str, str],
) -> tuple[QueryAnalysisBridgeCandidate, ...]:
    if not isinstance(value, list):
        raise QueryAnalysisValidationError("bridge_candidates 必须是数组")
    if len(value) > MAX_BRIDGE_CANDIDATES:
        raise QueryAnalysisValidationError("bridge_candidates 超过上限")
    result: list[QueryAnalysisBridgeCandidate] = []
    seen_ids: set[str] = set()
    seen_subjects: set[tuple[str, int, int, str]] = set()
    for index, raw in enumerate(value):
        item = _expect_exact_keys(
            raw,
            keys=_BRIDGE_CANDIDATE_KEYS,
            field=f"bridge_candidates[{index}]",
        )
        identifier = _parse_identifier(
            item["id"],
            field=f"bridge_candidates[{index}].id",
            pattern=_BRIDGE_ID_RE,
        )
        if identifier in seen_ids:
            raise QueryAnalysisValidationError("bridge_candidates 包含重复 id")
        seen_ids.add(identifier)
        subject = _parse_source_ref(
            item["subject_source_ref"],
            field=f"bridge_candidates[{index}].subject_source_ref",
            available_source_texts=available_source_texts,
        )
        if subject.identity in seen_subjects:
            raise QueryAnalysisValidationError("bridge_candidates 不能重复使用同一主体")
        seen_subjects.add(subject.identity)
        result.append(QueryAnalysisBridgeCandidate(
            id=identifier,
            subject_source_ref=subject,
        ))
    return tuple(result)


def _parse_answer_candidates(
    value: object,
    *,
    available_source_texts: Mapping[str, str],
    bridge_ids: set[str],
) -> tuple[QueryAnalysisAnswerCandidate, ...]:
    if not isinstance(value, list):
        raise QueryAnalysisValidationError("answer_candidates 必须是数组")
    if not value:
        raise QueryAnalysisValidationError("answer_candidates 必须至少包含一个目标")
    if len(value) > MAX_ANSWER_CANDIDATES:
        raise QueryAnalysisValidationError("answer_candidates 超过上限")
    result: list[QueryAnalysisAnswerCandidate] = []
    seen_ids: set[str] = set()
    seen_targets: set[tuple[str, int, int, str]] = set()
    for index, raw in enumerate(value):
        item = _expect_exact_keys(
            raw,
            keys=_ANSWER_CANDIDATE_KEYS,
            field=f"answer_candidates[{index}]",
        )
        identifier = _parse_identifier(
            item["id"],
            field=f"answer_candidates[{index}].id",
            pattern=_ANSWER_ID_RE,
        )
        if identifier in seen_ids:
            raise QueryAnalysisValidationError("answer_candidates 包含重复 id")
        seen_ids.add(identifier)
        target = _parse_source_ref(
            item["target_source_ref"],
            field=f"answer_candidates[{index}].target_source_ref",
            available_source_texts=available_source_texts,
        )
        if target.turn_key != "current":
            raise QueryAnalysisValidationError("答案目标只能来自当前输入")
        if target.identity in seen_targets:
            raise QueryAnalysisValidationError("answer_candidates 不能重复使用同一目标来源片段")
        seen_targets.add(target.identity)
        qualifiers = _parse_source_ref_list(
            item["qualifier_source_refs"],
            field=f"answer_candidates[{index}].qualifier_source_refs",
            available_source_texts=available_source_texts,
            max_items=MAX_QUALIFIER_REFS,
        )
        candidate_bridge_ids = _parse_id_list(
            item["bridge_candidate_ids"],
            field=f"answer_candidates[{index}].bridge_candidate_ids",
            allowed_ids=bridge_ids,
            max_items=MAX_BRIDGE_REFS,
        )
        result.append(QueryAnalysisAnswerCandidate(
            id=identifier,
            target_source_ref=target,
            qualifier_source_refs=qualifiers,
            bridge_candidate_ids=candidate_bridge_ids,
        ))
    return tuple(result)


def _parse_context_turn_keys(
    value: object,
    *,
    available_turn_keys: tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list):
        raise QueryAnalysisValidationError("context_turn_keys 必须是数组")
    if len(value) > MAX_CONTEXT_TURN_KEYS:
        raise QueryAnalysisValidationError("context_turn_keys 超过上限")
    allowed = set(available_turn_keys)
    result: list[str] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str):
            raise QueryAnalysisValidationError("context_turn_keys 只能包含字符串")
        key = raw.strip()
        if not _TURN_KEY_RE.fullmatch(key):
            raise QueryAnalysisValidationError(f"context_turn_keys[{index}] 非法")
        if key not in allowed:
            raise QueryAnalysisValidationError(f"context_turn_keys 包含不可用键: {key}")
        if key in result:
            raise QueryAnalysisValidationError("context_turn_keys 包含重复键")
        result.append(key)
    return tuple(result)


def _validate_graph_and_history_binding(analysis: QueryAnalysis) -> None:
    if len(analysis.answer_candidates) + len(analysis.bridge_candidates) > MAX_ANSWER_CANDIDATES:
        raise QueryAnalysisValidationError("候选答案与 bridge 节点总数超过执行上限")

    bridge_by_id = {item.id: item for item in analysis.bridge_candidates}
    referenced_bridge_ids: set[str] = set()
    history_turn_keys: set[str] = set()
    for answer in analysis.answer_candidates:
        qualifier_identities = {
            item.identity for item in answer.qualifier_source_refs
        }
        history_turn_keys.update(
            item.turn_key
            for item in answer.qualifier_source_refs
            if item.turn_key != "current"
        )
        for bridge_id in answer.bridge_candidate_ids:
            bridge = bridge_by_id[bridge_id]
            referenced_bridge_ids.add(bridge_id)
            if bridge.subject_source_ref.identity not in qualifier_identities:
                raise QueryAnalysisValidationError(
                    "answer candidate 引用的 bridge 的主体必须出现在该答案的 qualifier_source_refs"
                )

    for bridge in analysis.bridge_candidates:
        if bridge.id not in referenced_bridge_ids:
            raise QueryAnalysisValidationError(
                "分析图包含未被答案引用的 bridge candidate: " + bridge.id
            )
        if bridge.subject_source_ref.turn_key != "current":
            history_turn_keys.add(bridge.subject_source_ref.turn_key)

    context_keys = set(analysis.context_turn_keys)
    if analysis.self_contained:
        if context_keys or history_turn_keys:
            raise QueryAnalysisValidationError("自足问题不得绑定或引用历史来源")
        return
    if analysis.relation == "new":
        raise QueryAnalysisValidationError("非自足问题不能标记为 new")
    if not context_keys:
        raise QueryAnalysisValidationError("非自足问题必须绑定历史上下文")
    if not history_turn_keys:
        raise QueryAnalysisValidationError("非自足问题必须引用历史限定词或主体")
    if context_keys != history_turn_keys:
        raise QueryAnalysisValidationError(
            "context_turn_keys 必须精确覆盖被引用的历史来源"
        )


def _source_ref_schema(available_turn_keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["turn_key", "start", "end", "span"],
        "properties": {
            "turn_key": {"type": "string", "enum": ["current", *available_turn_keys]},
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 1},
            "span": {"type": "string", "minLength": 1, "maxLength": MAX_SOURCE_SPAN_CHARS},
        },
    }


def build_query_analysis_schema(
    *,
    available_turn_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the exact request-local JSON schema for ``query_analysis.v2``."""

    keys = tuple(str(value).strip() for value in available_turn_keys)
    if len(keys) > MAX_CONTEXT_TURN_KEYS or len(set(keys)) != len(keys):
        raise QueryAnalysisValidationError("available_turn_keys 非法")
    if any(not _TURN_KEY_RE.fullmatch(value) for value in keys):
        raise QueryAnalysisValidationError("available_turn_keys 包含非法键")
    source_ref = _source_ref_schema(keys)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_TOP_LEVEL_KEYS),
        "properties": {
            "schema_version": {"type": "string", "const": QUERY_ANALYSIS_SCHEMA_VERSION},
            "relation": {"type": "string", "enum": sorted(VALID_RELATIONS)},
            "self_contained": {"type": "boolean"},
            "context_turn_keys": {
                "type": "array",
                "maxItems": min(MAX_CONTEXT_TURN_KEYS, len(keys)),
                "items": {"type": "string", "enum": list(keys)},
            },
            "answer_candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_ANSWER_CANDIDATES,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_ANSWER_CANDIDATE_KEYS),
                    "properties": {
                        "id": {"type": "string", "pattern": _ANSWER_ID_RE.pattern},
                        "target_source_ref": source_ref,
                        "qualifier_source_refs": {
                            "type": "array",
                            "maxItems": MAX_QUALIFIER_REFS,
                            "items": source_ref,
                        },
                        "bridge_candidate_ids": {
                            "type": "array",
                            "maxItems": MAX_BRIDGE_REFS,
                            "items": {"type": "string", "pattern": _BRIDGE_ID_RE.pattern},
                        },
                    },
                },
            },
            "bridge_candidates": {
                "type": "array",
                "maxItems": MAX_BRIDGE_CANDIDATES,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_BRIDGE_CANDIDATE_KEYS),
                    "properties": {
                        "id": {"type": "string", "pattern": _BRIDGE_ID_RE.pattern},
                        "subject_source_ref": source_ref,
                    },
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "diagnostic": {"type": "string", "maxLength": MAX_DIAGNOSTIC_CHARS},
        },
    }


def build_query_analysis_response_format(
    *,
    available_turn_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Return the strict OpenAI-compatible envelope for ``query_analysis.v2``."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": QUERY_ANALYSIS_SCHEMA_NAME,
            "strict": True,
            "schema": build_query_analysis_schema(
                available_turn_keys=available_turn_keys,
            ),
        },
    }


def parse_query_analysis(
    raw: object,
    *,
    current_question: str,
    context_user_inputs: Mapping[str, str] | None = None,
) -> QueryAnalysis:
    """Parse one all-or-nothing ``query_analysis.v2`` candidate graph.

    ``current_question`` and each request-local historical user input are kept
    byte/code-point exact during parsing.  Offsets therefore describe precisely
    the text the model was permitted to inspect; assistant answers, persistent
    IDs and any retrieval/evidence state are intentionally absent.
    """

    source_texts: dict[str, str] = {
        "current": _source_text(
            current_question,
            field="current_question",
            max_chars=8000,
        )
    }
    for key, value in (context_user_inputs or {}).items():
        normalized_key = str(key).strip()
        if not _TURN_KEY_RE.fullmatch(normalized_key):
            raise QueryAnalysisValidationError("context_user_inputs 包含非法键")
        if normalized_key in source_texts:
            raise QueryAnalysisValidationError("context_user_inputs 包含重复键")
        source_texts[normalized_key] = _source_text(
            value,
            field=f"context_user_inputs[{normalized_key}]",
            max_chars=2000,
        )
    available_turn_keys = tuple(
        key for key in source_texts if key != "current"
    )
    if len(available_turn_keys) > MAX_CONTEXT_TURN_KEYS:
        raise QueryAnalysisValidationError("context_user_inputs 超过上限")

    payload = _expect_exact_keys(
        _parse_json_object(raw),
        keys=_TOP_LEVEL_KEYS,
        field="query_analysis",
    )
    if payload["schema_version"] != QUERY_ANALYSIS_SCHEMA_VERSION:
        raise QueryAnalysisValidationError("schema_version 不受支持")
    relation = _normalized_text(payload["relation"], field="relation", max_chars=32)
    if relation not in VALID_RELATIONS:
        raise QueryAnalysisValidationError("relation 不在允许枚举中")
    self_contained = payload["self_contained"]
    if not isinstance(self_contained, bool):
        raise QueryAnalysisValidationError("self_contained 必须是布尔值")
    context_turn_keys = _parse_context_turn_keys(
        payload["context_turn_keys"],
        available_turn_keys=available_turn_keys,
    )
    bridge_candidates = _parse_bridge_candidates(
        payload["bridge_candidates"],
        available_source_texts=source_texts,
    )
    answer_candidates = _parse_answer_candidates(
        payload["answer_candidates"],
        available_source_texts=source_texts,
        bridge_ids={item.id for item in bridge_candidates},
    )
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise QueryAnalysisValidationError("confidence 必须是数字")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise QueryAnalysisValidationError("confidence 必须位于 0~1")
    diagnostic = _normalized_text(
        payload["diagnostic"],
        field="diagnostic",
        max_chars=MAX_DIAGNOSTIC_CHARS,
    )
    analysis = QueryAnalysis(
        schema_version=QUERY_ANALYSIS_SCHEMA_VERSION,
        relation=relation,  # type: ignore[arg-type]
        self_contained=self_contained,
        context_turn_keys=context_turn_keys,
        answer_candidates=answer_candidates,
        bridge_candidates=bridge_candidates,
        confidence=confidence,
        diagnostic=diagnostic,
    )
    _validate_graph_and_history_binding(analysis)
    return analysis


__all__ = [
    "MAX_ANSWER_CANDIDATES",
    "MAX_BRIDGE_CANDIDATES",
    "QUERY_ANALYSIS_SCHEMA_NAME",
    "QUERY_ANALYSIS_SCHEMA_VERSION",
    "QueryAnalysis",
    "QueryAnalysisAnswerCandidate",
    "QueryAnalysisBridgeCandidate",
    "QueryAnalysisSourceRef",
    "QueryAnalysisValidationError",
    "build_query_analysis_response_format",
    "build_query_analysis_schema",
    "parse_query_analysis",
]
