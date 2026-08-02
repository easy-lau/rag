"""Strict, catalog-bound model contract for ``query_understanding.v3``.

Unlike ``query_analysis.v2``, the model never submits source offsets or raw
text.  It can choose only ``span_id`` values issued by a request-local
:class:`~core.query_understanding_v3_catalog.SourceSpanCatalog`.  The parser
resolves every selected id back to a server-verified literal source range and
rejects unknown, duplicate, overlapping or unauthorised selections before a
compiler ever sees them.

This is still a candidate description, not a retrieval plan.  It contains no
knowledge-base/document ids, facts, scope decision, bridge edge, synonym or
execution information.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from core.query_understanding_v3_catalog import (
    CatalogSpan,
    SourceSpanCatalog,
    SourceSpanCatalogError,
)


QUERY_UNDERSTANDING_V3_SCHEMA_VERSION = "query_understanding.v3"
QUERY_UNDERSTANDING_V3_SCHEMA_NAME = "query_understanding_v3"

MAX_ANSWER_CANDIDATES = 8
MAX_QUALIFIER_SPAN_IDS = 8

_ANSWER_ID_RE = re.compile(r"^a[1-9][0-9]{0,2}$")
_SPAN_ID_RE = re.compile(r"^s_(?:current|t[1-9][0-9]{0,2})_[0-9]{3}$")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "answer_candidates",
}
_ANSWER_CANDIDATE_KEYS = {"id", "target_span_id", "qualifier_span_ids"}


class QueryUnderstandingV3ValidationError(ValueError):
    """Raised when a V3 model response breaks the bounded catalog contract."""


def _expect_exact_keys(value: object, *, keys: set[str], field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QueryUnderstandingV3ValidationError(f"{field} 必须是对象")
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
    raise QueryUnderstandingV3ValidationError(
        f"{field} 字段不精确: {'；'.join(details) or '未知原因'}"
    )


def _parse_json_object(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise QueryUnderstandingV3ValidationError("模型输出为空或不是 JSON 字符串")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise QueryUnderstandingV3ValidationError(
                    f"JSON 包含重复字段: {key}"
                )
            output[key] = value
        return output

    def reject_non_finite(value: str) -> None:
        raise QueryUnderstandingV3ValidationError(f"JSON 不允许非有限数字: {value}")

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except QueryUnderstandingV3ValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise QueryUnderstandingV3ValidationError("模型输出不是严格 JSON 对象") from exc
    if not isinstance(parsed, dict):
        raise QueryUnderstandingV3ValidationError("模型输出顶层必须是对象")
    return parsed


def _parse_answer_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _ANSWER_ID_RE.fullmatch(value):
        raise QueryUnderstandingV3ValidationError(f"{field} 包含非法标识")
    return value


def _resolve_span(
    value: object,
    *,
    field: str,
    catalog: SourceSpanCatalog,
) -> CatalogSpan:
    if not isinstance(value, str) or not _SPAN_ID_RE.fullmatch(value):
        raise QueryUnderstandingV3ValidationError(f"{field} span_id 非法")
    try:
        return catalog.resolve(value)
    except SourceSpanCatalogError as exc:
        raise QueryUnderstandingV3ValidationError(f"{field} span_id 不在当前 catalog") from exc


def _parse_qualifier_spans(
    value: object,
    *,
    field: str,
    catalog: SourceSpanCatalog,
) -> tuple[CatalogSpan, ...]:
    if isinstance(value, str) or not isinstance(value, list):
        raise QueryUnderstandingV3ValidationError(f"{field} 必须是数组")
    if len(value) > MAX_QUALIFIER_SPAN_IDS:
        raise QueryUnderstandingV3ValidationError(f"{field} 超过上限")
    result: list[CatalogSpan] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        span = _resolve_span(raw, field=f"{field}[{index}]", catalog=catalog)
        if span.span_id in seen:
            raise QueryUnderstandingV3ValidationError(f"{field} 包含重复 span_id")
        if any(span.overlaps(existing) for existing in result):
            raise QueryUnderstandingV3ValidationError(f"{field} 包含重叠来源片段")
        seen.add(span.span_id)
        result.append(span)
    return tuple(result)


@dataclass(frozen=True)
class QueryUnderstandingV3Candidate:
    """One model-selected target and literal qualifiers, all catalog-bound."""

    id: str
    target_span_id: str
    qualifier_span_ids: tuple[str, ...]
    target_span: CatalogSpan
    qualifier_spans: tuple[CatalogSpan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "target_span_id": self.target_span_id,
            "qualifier_span_ids": list(self.qualifier_span_ids),
        }


@dataclass(frozen=True)
class QueryUnderstandingV3:
    """Validated V3 candidate graph with no retrieval or evidence authority.

    The model has no relation, self-contained or confidence fields.  Those
    values are derived solely from the server-issued catalog entries it
    selected: a historical qualifier means the current request depends on a
    route-authorised previous user turn; otherwise it is self-contained.
    """

    schema_version: Literal["query_understanding.v3"]
    answer_candidates: tuple[QueryUnderstandingV3Candidate, ...]

    @property
    def referenced_context_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for candidate in self.answer_candidates:
            for span in candidate.qualifier_spans:
                if span.source_kind == "route_context" and span.source_key not in keys:
                    keys.append(span.source_key)
        return tuple(keys)

    @property
    def self_contained(self) -> bool:
        """Whether the selected spans require no authorised history."""

        return not self.referenced_context_keys

    @property
    def relation(self) -> Literal["new", "followup"]:
        """Server-derived relation for diagnostics; never model supplied."""

        return "new" if self.self_contained else "followup"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "answer_candidates": [item.to_dict() for item in self.answer_candidates],
        }

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "relation": self.relation,
            "self_contained": self.self_contained,
            "answer_candidate_count": len(self.answer_candidates),
            "referenced_context_turn_count": len(self.referenced_context_keys),
        }


def _parse_answer_candidates(
    value: object,
    *,
    catalog: SourceSpanCatalog,
) -> tuple[QueryUnderstandingV3Candidate, ...]:
    if isinstance(value, str) or not isinstance(value, list):
        raise QueryUnderstandingV3ValidationError("answer_candidates 必须是数组")
    if not value:
        raise QueryUnderstandingV3ValidationError("answer_candidates 必须至少包含一个目标")
    if len(value) > MAX_ANSWER_CANDIDATES:
        raise QueryUnderstandingV3ValidationError("answer_candidates 超过上限")
    result: list[QueryUnderstandingV3Candidate] = []
    seen_ids: set[str] = set()
    selected_targets: list[CatalogSpan] = []
    for index, raw in enumerate(value):
        item = _expect_exact_keys(
            raw,
            keys=_ANSWER_CANDIDATE_KEYS,
            field=f"answer_candidates[{index}]",
        )
        identifier = _parse_answer_id(item["id"], field=f"answer_candidates[{index}].id")
        if identifier in seen_ids:
            raise QueryUnderstandingV3ValidationError("answer_candidates 包含重复 id")
        seen_ids.add(identifier)
        target = _resolve_span(
            item["target_span_id"],
            field=f"answer_candidates[{index}].target_span_id",
            catalog=catalog,
        )
        if target.source_kind != "current":
            raise QueryUnderstandingV3ValidationError("答案目标只能来自当前输入")
        if any(target.overlaps(existing) for existing in selected_targets):
            raise QueryUnderstandingV3ValidationError(
                "answer_candidates 不能重复或重叠使用答案目标"
            )
        qualifiers = _parse_qualifier_spans(
            item["qualifier_span_ids"],
            field=f"answer_candidates[{index}].qualifier_span_ids",
            catalog=catalog,
        )
        if target.span_id in {item.span_id for item in qualifiers}:
            raise QueryUnderstandingV3ValidationError(
                "answer candidate 不能将答案目标重复作为 qualifier"
            )
        if any(target.overlaps(item) for item in qualifiers):
            raise QueryUnderstandingV3ValidationError(
                "answer candidate 的目标与 qualifier 来源片段重叠"
            )
        selected_targets.append(target)
        result.append(QueryUnderstandingV3Candidate(
            id=identifier,
            target_span_id=target.span_id,
            qualifier_span_ids=tuple(item.span_id for item in qualifiers),
            target_span=target,
            qualifier_spans=qualifiers,
        ))
    return tuple(result)


def build_query_understanding_schema(*, catalog: SourceSpanCatalog) -> dict[str, Any]:
    """Build the request-local strict JSON schema for V3 model selection."""

    if not isinstance(catalog, SourceSpanCatalog):
        raise QueryUnderstandingV3ValidationError("catalog 必须是 SourceSpanCatalog")
    target_span_ids = list(catalog.current_span_ids)
    all_span_ids = list(catalog.all_span_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_TOP_LEVEL_KEYS),
        "properties": {
            "schema_version": {"type": "string", "const": QUERY_UNDERSTANDING_V3_SCHEMA_VERSION},
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
                        "target_span_id": {"type": "string", "enum": target_span_ids},
                        "qualifier_span_ids": {
                            "type": "array",
                            "maxItems": MAX_QUALIFIER_SPAN_IDS,
                            "items": {"type": "string", "enum": all_span_ids},
                        },
                    },
                },
            },
        },
    }


def build_query_understanding_response_format(
    *,
    catalog: SourceSpanCatalog,
) -> dict[str, Any]:
    """Return the strict OpenAI-compatible response-format envelope for V3."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": QUERY_UNDERSTANDING_V3_SCHEMA_NAME,
            "strict": True,
            "schema": build_query_understanding_schema(catalog=catalog),
        },
    }


def parse_query_understanding(
    raw: object,
    *,
    catalog: SourceSpanCatalog,
) -> QueryUnderstandingV3:
    """Parse a model response using only spans issued by ``catalog``.

    ``catalog`` is mandatory, rather than raw current/history strings.  This
    makes it structurally impossible for a parser caller to accidentally
    authorise a model-selected source outside the route decision.
    """

    if not isinstance(catalog, SourceSpanCatalog):
        raise QueryUnderstandingV3ValidationError("catalog 必须是 SourceSpanCatalog")
    payload = _expect_exact_keys(
        _parse_json_object(raw),
        keys=_TOP_LEVEL_KEYS,
        field="query_understanding",
    )
    if payload["schema_version"] != QUERY_UNDERSTANDING_V3_SCHEMA_VERSION:
        raise QueryUnderstandingV3ValidationError("schema_version 不受支持")
    candidates = _parse_answer_candidates(payload["answer_candidates"], catalog=catalog)
    analysis = QueryUnderstandingV3(
        schema_version=QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
        answer_candidates=candidates,
    )
    return analysis


__all__ = [
    "MAX_ANSWER_CANDIDATES",
    "MAX_QUALIFIER_SPAN_IDS",
    "QUERY_UNDERSTANDING_V3_SCHEMA_NAME",
    "QUERY_UNDERSTANDING_V3_SCHEMA_VERSION",
    "QueryUnderstandingV3",
    "QueryUnderstandingV3Candidate",
    "QueryUnderstandingV3ValidationError",
    "build_query_understanding_response_format",
    "build_query_understanding_schema",
    "parse_query_understanding",
]
