"""Bounded model adapter for ``query_understanding.v3``.

The adapter intentionally has no chat/pipeline side effects.  It builds a
request-local :class:`SourceSpanCatalog`, gives the model only its reduced
catalog view, and accepts an answer only after strict catalog-id parsing.  A
future V3 execution service can consume the result or retain its deterministic
baseline when ``analysis`` is absent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from config import get_settings
from core.openai_client import get_client
from core.query_understanding_v3_catalog import (
    SourceSpanCatalog,
    SourceSpanCatalogError,
    build_source_span_catalog,
)
from core.query_understanding_v3_contract import (
    QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
    QueryUnderstandingV3,
    QueryUnderstandingV3ValidationError,
    build_query_understanding_response_format,
    parse_query_understanding,
)
from core.rag_trace import content_fields, exception_log_text, trace_event
from core.structured_output import create_structured_completion


logger = logging.getLogger(__name__)

QUERY_UNDERSTANDING_V3_PROMPT_VERSION = "2026-08-02.query-understanding-v3"
QUERY_UNDERSTANDING_V3_MAX_TOKENS = 1400
QueryUnderstandingMode = Literal["off", "shadow", "active"]
QueryUnderstandingOrigin = Literal["model", "deterministic"]


@dataclass(frozen=True)
class QueryUnderstandingV3RunResult:
    """The bounded result of one V3 model attempt.

    ``catalog`` remains available for an accepted response and for a caller's
    deterministic fallback, but no model-derived field is considered trusted
    unless ``analysis`` is non-``None``.
    """

    mode: QueryUnderstandingMode
    catalog: SourceSpanCatalog | None
    analysis: QueryUnderstandingV3 | None
    model: str
    latency_ms: int
    fallback_reason: str | None = None
    strict_schema_used: bool = False
    json_object_fallback_used: bool = False
    structured_output_mode: str | None = None
    # The execution service may construct the same strict V3 contract from a
    # narrowly proven local source grammar before invoking a model.  Keeping
    # the origin explicit prevents traces from presenting that local fast path
    # as an LLM response.
    origin: QueryUnderstandingOrigin = "model"

    @property
    def accepted(self) -> bool:
        return self.analysis is not None

    def safe_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "origin": self.origin,
            "accepted": self.accepted,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "fallback_reason": self.fallback_reason,
            "strict_schema_used": self.strict_schema_used,
            "json_object_fallback_used": self.json_object_fallback_used,
            "structured_output_mode": self.structured_output_mode,
            "catalog": self.catalog.safe_summary() if self.catalog else None,
            "analysis": self.analysis.safe_summary() if self.analysis else None,
        }


def _normalised_mode(value: object) -> QueryUnderstandingMode:
    mode = str(value or "shadow").strip().casefold()
    if mode not in {"off", "shadow", "active"}:
        return "shadow"
    return mode  # type: ignore[return-value]


def _system_prompt() -> str:
    return (
        f"提示词版本：{QUERY_UNDERSTANDING_V3_PROMPT_VERSION}。输出 schema_version 必须是 "
        f"{QUERY_UNDERSTANDING_V3_SCHEMA_VERSION}。\n"
        "你是企业 RAG 的问题结构分析器，不回答问题、不判断事实、不选择知识库、文档、"
        "权限、范围、覆盖规则、检索词或执行方式。输入只包含服务器签发的 source span catalog。\n"
        "每个 answer candidate 只能填写 catalog 中已给出的 span_id：target_span_id 必须选择"
        "source=current 的一个 span；qualifier_span_ids 也只能选择 catalog 中的 span_id。"
        "绝不能输出或复述原文、span/text、start/end/offset、turn key、会话 ID、KB/文档 ID、"
        "产品/版本范围、同义词、金额、职级、日期或其他事实。\n"
        "不得输出 scope、scope_span_ids、bridge、bridge_probe、bridge edge、依赖边、"
        "coverage/retrieval 字段。桥接关系、范围和执行关系完全由后端从已验证 span 决定。\n"
        "同一句内已有的限定词应使用 current span_id；仅当前输入确实缺少限定词且必须依赖"
        "既有上下文时，才选择 catalog 中的 route_context span_id。后端会依据你实际选择的"
        "span 来源推导是否追问和其它关系。输出中不得包含任何自由文本字段。"
    )


def _user_payload(*, catalog: SourceSpanCatalog) -> dict[str, object]:
    """The model sees only the reduced catalog, never raw request/context fields."""

    return {"span_catalog": catalog.model_payload()}


def _validation_error_code(exc: BaseException) -> str:
    text = str(exc)
    if "span_id" in text or "catalog" in text:
        return "catalog_span_invalid"
    if "当前输入" in text:
        return "target_source_invalid"
    if "历史" in text or "自足" in text:
        return "context_binding_invalid"
    if "字段不精确" in text or "重复字段" in text:
        return "schema_shape_invalid"
    if "JSON" in text:
        return "invalid_json"
    if "重叠" in text or "重复" in text:
        return "selection_invalid"
    return "validation_rejected"


async def analyze_query_understanding(
    *,
    question: str,
    route_context: Iterable[Mapping[str, Any]] | None = None,
    trace_id: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    mode: QueryUnderstandingMode | None = None,
    timeout_seconds: float | None = None,
) -> QueryUnderstandingV3RunResult:
    """Run one strict V3 catalog-selection request under an absolute timeout.

    The function does not raise model/JSON failures.  On any such failure it
    returns ``analysis=None`` with a stable fallback reason so a caller can
    retain a deterministic query plan.  It deliberately does not call chat,
    V2 compilation, retrieval or evidence code.
    """

    settings = get_settings()
    selected_mode = _normalised_mode(
        mode if mode is not None else getattr(settings, "rag_query_analyzer_mode", "shadow")
    )
    started = time.perf_counter()
    model = str(
        getattr(settings, "intent_model", "")
        or getattr(settings, "chat_model", "")
        or ""
    ).strip()
    configured_timeout_seconds = max(
        0.1,
        float(getattr(settings, "rag_query_analyzer_timeout_seconds", 5.0)),
    )
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool):
            raise ValueError("timeout_seconds must be numeric")
        configured_timeout_seconds = max(0.1, float(timeout_seconds))

    catalog: SourceSpanCatalog | None = None
    try:
        catalog = build_source_span_catalog(
            current_question=question,
            route_context=route_context,
        )
    except SourceSpanCatalogError as exc:
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        trace_event(
            "query.understanding.v3.fallback",
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            schema_version=QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
            mode=selected_mode,
            model=model,
            reason="catalog_invalid",
            latency_ms=latency_ms,
            catalog_error_code="invalid_source_catalog",
        )
        return QueryUnderstandingV3RunResult(
            mode=selected_mode,
            catalog=None,
            analysis=None,
            model=model,
            latency_ms=latency_ms,
            fallback_reason="catalog_invalid",
        )

    common_trace = {
        "trace_id": trace_id,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "schema_version": QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
        "prompt_version": QUERY_UNDERSTANDING_V3_PROMPT_VERSION,
        "mode": selected_mode,
        "timeout_seconds": configured_timeout_seconds,
        "catalog_summary": catalog.safe_summary(),
    }
    trace_event(
        "query.understanding.v3.requested",
        **common_trace,
        model=model,
        **content_fields(
            "query_understanding_v3_catalog",
            json.dumps(catalog.model_payload(), ensure_ascii=False, separators=(",", ":")),
        ),
    )
    if selected_mode == "off":
        result = QueryUnderstandingV3RunResult(
            mode=selected_mode,
            catalog=catalog,
            analysis=None,
            model=model,
            latency_ms=0,
            fallback_reason="analyzer_disabled",
        )
        trace_event(
            "query.understanding.v3.fallback",
            **common_trace,
            model=model,
            reason=result.fallback_reason,
            latency_ms=0,
        )
        return result
    if not model:
        result = QueryUnderstandingV3RunResult(
            mode=selected_mode,
            catalog=catalog,
            analysis=None,
            model="",
            latency_ms=0,
            fallback_reason="model_not_configured",
        )
        trace_event(
            "query.understanding.v3.fallback",
            **common_trace,
            model="",
            reason=result.fallback_reason,
            latency_ms=0,
        )
        return result

    strict_schema_used = True
    json_object_fallback_used = False
    structured_output_mode = "json_schema"
    raw_content = ""
    try:
        client = get_client()
        if hasattr(client, "with_options"):
            client = client.with_options(max_retries=0)
        request = {
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        _user_payload(catalog=catalog),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": QUERY_UNDERSTANDING_V3_MAX_TOKENS,
            "timeout": configured_timeout_seconds,
        }
        response_format = build_query_understanding_response_format(catalog=catalog)

        structured = await create_structured_completion(
            client,
            request=request,
            strict_response_format=response_format,
            timeout_seconds=configured_timeout_seconds,
            provider_identity=getattr(settings, "llm_base_url", ""),
            model=model,
        )
        response = structured.response
        structured_output_mode = structured.mode
        json_object_fallback_used = structured.mode != "json_schema"
        choices = list(getattr(response, "choices", None) or [])
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None) if choice is not None else None
        raw_content = str(getattr(message, "content", None) or "")
        finish_reason = str(getattr(choice, "finish_reason", None) or "")
        usage = getattr(response, "usage", None)
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        trace_event(
            "query.understanding.v3.completed",
            **common_trace,
            model=model,
            latency_ms=latency_ms,
            strict_schema_used=strict_schema_used,
            json_object_fallback_used=json_object_fallback_used,
            structured_output_mode=structured_output_mode,
            finish_reason=finish_reason,
            choice_count=len(choices),
            response_model=getattr(response, "model", None),
            response_id=getattr(response, "id", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            **content_fields("query_understanding_v3_raw_response", raw_content),
        )
        if finish_reason.casefold() == "length":
            raise QueryUnderstandingV3ValidationError("finish_reason_length")
        analysis = parse_query_understanding(raw_content, catalog=catalog)
        validated_json = json.dumps(
            analysis.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        trace_event(
            "query.understanding.v3.validated",
            **common_trace,
            model=model,
            accepted=True,
            latency_ms=latency_ms,
            strict_schema_used=strict_schema_used,
            json_object_fallback_used=json_object_fallback_used,
            structured_output_mode=structured_output_mode,
            analysis_summary=analysis.safe_summary(),
            **content_fields("query_understanding_v3_validated", validated_json),
        )
        return QueryUnderstandingV3RunResult(
            mode=selected_mode,
            catalog=catalog,
            analysis=analysis,
            model=model,
            latency_ms=latency_ms,
            strict_schema_used=strict_schema_used,
            json_object_fallback_used=json_object_fallback_used,
        )
    except asyncio.TimeoutError:
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        reason = "timeout"
    except QueryUnderstandingV3ValidationError as exc:
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        reason = _validation_error_code(exc)
    except Exception as exc:
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        reason = "model_error"
        logger.warning(
            "[问题理解V3] 模型调用失败 model=%s latency=%dms error=%s",
            model,
            latency_ms,
            exception_log_text(exc),
        )
    trace_event(
        "query.understanding.v3.validated",
        **common_trace,
        model=model,
        accepted=False,
        latency_ms=latency_ms,
        strict_schema_used=strict_schema_used,
        json_object_fallback_used=json_object_fallback_used,
        structured_output_mode=structured_output_mode,
        rejection_reason=reason,
        **content_fields("query_understanding_v3_raw_response", raw_content),
    )
    trace_event(
        "query.understanding.v3.fallback",
        **common_trace,
        model=model,
        reason=reason,
        latency_ms=latency_ms,
    )
    return QueryUnderstandingV3RunResult(
        mode=selected_mode,
        catalog=catalog,
        analysis=None,
        model=model,
        latency_ms=latency_ms,
        fallback_reason=reason,
        strict_schema_used=strict_schema_used,
        json_object_fallback_used=json_object_fallback_used,
    )


__all__ = [
    "QUERY_UNDERSTANDING_V3_MAX_TOKENS",
    "QUERY_UNDERSTANDING_V3_PROMPT_VERSION",
    "QueryUnderstandingMode",
    "QueryUnderstandingOrigin",
    "QueryUnderstandingV3RunResult",
    "analyze_query_understanding",
]
