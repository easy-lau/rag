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
from core.query_semantics import (
    KnowledgeRequestSemantics,
    content_knowledge_request,
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
    "knowledge_request",
}
_LEGACY_TOP_LEVEL_KEYS = {"schema_version", "answer_candidates"}
_ANSWER_CANDIDATE_KEYS = {"id", "target_span_id", "qualifier_span_ids"}
_KNOWLEDGE_REQUEST_KEYS = {
    "resource",
    "operation",
    "filter_span_ids",
    "group_by",
    "status_filter",
    "result_handles",
    "answer_form",
}
_LEGACY_KNOWLEDGE_REQUEST_KEYS = _KNOWLEDGE_REQUEST_KEYS - {
    "result_handles",
    "answer_form",
}
_RESULT_HANDLE_KNOWLEDGE_REQUEST_KEYS = _KNOWLEDGE_REQUEST_KEYS - {"answer_form"}


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
    knowledge_request: KnowledgeRequestSemantics

    @property
    def referenced_context_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for candidate in self.answer_candidates:
            for span in candidate.qualifier_spans:
                if span.source_kind == "route_context" and span.source_key not in keys:
                    keys.append(span.source_key)
        for source_key in _knowledge_request_context_keys(self.knowledge_request):
            if source_key not in keys:
                keys.append(source_key)
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
            "knowledge_request": {
                "resource": self.knowledge_request.resource,
                "operation": self.knowledge_request.operation,
                "filter_span_ids": list(self.knowledge_request.filter_span_ids),
                "group_by": self.knowledge_request.group_by,
                "status_filter": self.knowledge_request.status_filter,
                "result_handles": list(self.knowledge_request.result_handles),
                "answer_form": self.knowledge_request.answer_form,
            },
        }

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "relation": self.relation,
            "self_contained": self.self_contained,
            "answer_candidate_count": len(self.answer_candidates),
            "referenced_context_turn_count": len(self.referenced_context_keys),
            "knowledge_request": self.knowledge_request.safe_summary(),
        }


def _knowledge_request_context_keys(
    request: KnowledgeRequestSemantics,
) -> tuple[str, ...]:
    keys: list[str] = []
    for span_id in request.filter_span_ids:
        match = re.fullmatch(r"s_(t[1-9][0-9]{0,2})_[0-9]{3}", span_id)
        if match is not None and match.group(1) not in keys:
            keys.append(match.group(1))
    for handle in request.result_handles:
        match = re.fullmatch(r"r_(t[1-9][0-9]{0,2})_[0-9]{3}", handle)
        if match is not None and match.group(1) not in keys:
            keys.append(match.group(1))
    return tuple(keys)


def _parse_knowledge_request(
    value: object,
    *,
    catalog: SourceSpanCatalog,
) -> KnowledgeRequestSemantics:
    if isinstance(value, dict) and set(value) == _LEGACY_KNOWLEDGE_REQUEST_KEYS:
        value = {**value, "result_handles": [], "answer_form": "fact"}
    elif isinstance(value, dict) and set(value) == _RESULT_HANDLE_KNOWLEDGE_REQUEST_KEYS:
        # Read compatibility for responses produced immediately before answer
        # form became part of the shared semantic contract.  New schemas and
        # prompts always require it; an old response retains the conservative
        # single-fact behavior.
        value = {**value, "answer_form": "fact"}
    item = _expect_exact_keys(
        value, keys=_KNOWLEDGE_REQUEST_KEYS, field="knowledge_request"
    )
    filter_spans = _parse_qualifier_spans(
        item["filter_span_ids"],
        field="knowledge_request.filter_span_ids",
        catalog=catalog,
    )
    for field in (
        "resource",
        "operation",
        "group_by",
        "status_filter",
        "answer_form",
    ):
        if not isinstance(item[field], str):
            raise QueryUnderstandingV3ValidationError(
                f"knowledge_request.{field} 必须是字符串"
            )
    raw_handles = item["result_handles"]
    if not isinstance(raw_handles, list):
        raise QueryUnderstandingV3ValidationError(
            "knowledge_request.result_handles 必须是数组"
        )
    result_references = []
    for index, handle in enumerate(raw_handles):
        try:
            reference = catalog.resolve_result(handle)
        except SourceSpanCatalogError as exc:
            raise QueryUnderstandingV3ValidationError(
                f"knowledge_request.result_handles[{index}] 非法"
            ) from exc
        if reference in result_references:
            raise QueryUnderstandingV3ValidationError(
                "knowledge_request.result_handles 包含重复项"
            )
        result_references.append(reference)
    try:
        return KnowledgeRequestSemantics(
            resource=item["resource"],
            operation=item["operation"],
            filter_span_ids=tuple(span.span_id for span in filter_spans),
            filter_terms=tuple(span.text for span in filter_spans),
            group_by=item["group_by"],
            status_filter=item["status_filter"],
            result_handles=tuple(item.handle for item in result_references),
            result_labels=tuple(item.label for item in result_references),
            answer_form=item["answer_form"],
        )
    except ValueError as exc:
        raise QueryUnderstandingV3ValidationError(
            f"knowledge_request 组合非法: {exc}"
        ) from exc


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
        # ``id`` is only a server-local candidate handle and carries no model
        # authority.  Some json_object-only providers omit it despite the
        # prompt.  Assigning the deterministic ordinal is therefore safe.  A
        # target repeated verbatim as its own qualifier is equally redundant;
        # removing only that exact id cannot add a source, historical context,
        # scope or execution permission.  All other shape/source violations
        # remain strict failures below.
        if isinstance(raw, dict):
            raw = dict(raw)
            if "id" not in raw:
                raw["id"] = f"a{index + 1}"
            target_span_id = raw.get("target_span_id")
            qualifier_span_ids = raw.get("qualifier_span_ids")
            if isinstance(qualifier_span_ids, list):
                raw["qualifier_span_ids"] = [
                    item for item in qualifier_span_ids
                    if item != target_span_id
                ]
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
    result_handles = list(catalog.all_result_handles)
    resources = ["document_content", "document_catalog"]
    operations = ["answer", "count", "list", "group"]
    if result_handles:
        resources.append("document_result")
        operations.extend(["read", "summarize", "compare"])
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
            "knowledge_request": {
                "type": "object",
                "additionalProperties": False,
                "required": sorted(_KNOWLEDGE_REQUEST_KEYS),
                "properties": {
                    "resource": {
                        "type": "string",
                        "enum": resources,
                    },
                    "operation": {
                        "type": "string",
                        "enum": operations,
                    },
                    "filter_span_ids": {
                        "type": "array",
                        "maxItems": MAX_QUALIFIER_SPAN_IDS,
                        "items": {"type": "string", "enum": all_span_ids},
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["none", "knowledge_base", "status", "file_type"],
                    },
                    "status_filter": {
                        "type": "string",
                        "enum": [
                            "any",
                            "ready",
                            "processing",
                            "failed",
                            "inactive",
                            "not_ready",
                        ],
                    },
                    "result_handles": {
                        "type": "array",
                        "maxItems": 4,
                        "items": (
                            {"type": "string", "enum": result_handles}
                            if result_handles
                            else {"type": "string", "pattern": "a^"}
                        ),
                    },
                    "answer_form": {
                        "type": "string",
                        "enum": [
                            "fact",
                            "enumeration",
                            "procedure",
                            "overview",
                            "comparison",
                            "judgement",
                        ],
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
    raw_payload = _parse_json_object(raw)
    # Read compatibility for in-flight/recorded V3 responses produced before
    # the capability field existed.  New response schemas require the field;
    # an old response can only retain the conservative content-answer path.
    if set(raw_payload) == _LEGACY_TOP_LEVEL_KEYS:
        raw_payload = dict(raw_payload)
        legacy = content_knowledge_request()
        raw_payload["knowledge_request"] = {
            "resource": legacy.resource,
            "operation": legacy.operation,
            "filter_span_ids": [],
            "group_by": legacy.group_by,
            "status_filter": legacy.status_filter,
            "result_handles": [],
            "answer_form": legacy.answer_form,
        }
    payload = _expect_exact_keys(
        raw_payload,
        keys=_TOP_LEVEL_KEYS,
        field="query_understanding",
    )
    if payload["schema_version"] != QUERY_UNDERSTANDING_V3_SCHEMA_VERSION:
        raise QueryUnderstandingV3ValidationError("schema_version 不受支持")
    candidates = _parse_answer_candidates(payload["answer_candidates"], catalog=catalog)
    knowledge_request = _parse_knowledge_request(
        payload["knowledge_request"],
        catalog=catalog,
    )
    analysis = QueryUnderstandingV3(
        schema_version=QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
        answer_candidates=candidates,
        knowledge_request=knowledge_request,
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
