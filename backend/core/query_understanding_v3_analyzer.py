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
import re
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

QUERY_UNDERSTANDING_V3_PROMPT_VERSION = "2026-08-05.query-understanding-v3-answer-form"
QUERY_UNDERSTANDING_V3_MAX_TOKENS = 1400
QueryUnderstandingMode = Literal["off", "shadow", "active"]
QueryUnderstandingOrigin = Literal["model", "deterministic"]

_RESULT_ORDINAL_RE = re.compile(
    r"第(?P<ordinal>[0-9一二三四五六七八九十两]+)(?:个|篇|份|条)?"
    r"(?:文章|文档|资料|文件)?",
    re.IGNORECASE,
)
_RESULT_LAST_RE = re.compile(
    r"(?:最后|末尾)(?:一个|一篇|一份|一条)?(?:文章|文档|资料|文件)?",
    re.IGNORECASE,
)
_RESULT_PREFIX_RE = re.compile(
    r"前(?P<count>[0-9一二三四五六七八九十两]+)(?:个|篇|份|条)?",
    re.IGNORECASE,
)
_RESULT_COMPARE_RE = re.compile(r"(?:比较|对比|区别|差异|异同)", re.IGNORECASE)
_RESULT_SUMMARIZE_RE = re.compile(r"(?:总结|概括|摘要|归纳)", re.IGNORECASE)
_RESULT_READ_RE = re.compile(
    r"(?:看|查看|打开|阅读|正文|内容|详情)", re.IGNORECASE
)


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
    thinking_disabled: bool = False
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
            "thinking_disabled": self.thinking_disabled,
            "catalog": self.catalog.safe_summary() if self.catalog else None,
            "analysis": self.analysis.safe_summary() if self.analysis else None,
        }


def _normalised_mode(value: object) -> QueryUnderstandingMode:
    mode = str(value or "shadow").strip().casefold()
    if mode not in {"off", "shadow", "active"}:
        return "shadow"
    return mode  # type: ignore[return-value]


def _small_ordinal(value: str) -> int | None:
    text = str(value or "").strip()
    if text.isdigit():
        number = int(text)
        return number if 1 <= number <= 20 else None
    digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = digits.get(left, 1) if left else 1
        units = digits.get(right, 0) if right else 0
        number = tens * 10 + units
        return number if 1 <= number <= 20 else None
    return digits.get(text)


def _deterministic_result_reference_analysis(
    *,
    question: str,
    catalog: SourceSpanCatalog,
) -> QueryUnderstandingV3 | None:
    """Bind unambiguous ordinal references to the newest trusted result set."""

    if not catalog.result_references:
        return None
    newest_source = catalog.result_references[0].source_key
    results = [
        item for item in catalog.result_references
        if item.source_key == newest_source
    ]
    if not results:
        return None

    selected = []
    operation = "read"
    prefix = _RESULT_PREFIX_RE.search(question)
    ordinal = _RESULT_ORDINAL_RE.search(question)
    if prefix is not None:
        count = _small_ordinal(prefix.group("count"))
        if count is None or count < 2 or count > len(results):
            return None
        selected = results[:count]
        operation = "compare"
    elif _RESULT_LAST_RE.search(question):
        selected = [results[-1]]
    elif ordinal is not None:
        index = _small_ordinal(ordinal.group("ordinal"))
        if index is None or index > len(results):
            return None
        selected = [results[index - 1]]
    else:
        return None

    if len(selected) == 1 and _RESULT_SUMMARIZE_RE.search(question):
        operation = "summarize"
    elif len(selected) >= 2 or _RESULT_COMPARE_RE.search(question):
        if len(selected) < 2:
            return None
        operation = "compare"
    elif not _RESULT_READ_RE.search(question):
        # An ordinal alone can mean selection rather than content reading.
        # Let the model clarify that ambiguous action instead of granting one.
        return None

    target = catalog.current_entries[0]
    raw = {
        "schema_version": QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
        "answer_candidates": [{
            "id": "a1",
            "target_span_id": target.span_id,
            "qualifier_span_ids": [],
        }],
        "knowledge_request": {
            "resource": "document_result",
            "operation": operation,
            "filter_span_ids": [],
            "group_by": "none",
            "status_filter": "any",
            "result_handles": [item.handle for item in selected],
            "answer_form": (
                "comparison" if operation == "compare" else "overview"
            ),
        },
    }
    return parse_query_understanding(
        json.dumps(raw, ensure_ascii=False),
        catalog=catalog,
    )


def _system_prompt() -> str:
    return (
        f"提示词版本：{QUERY_UNDERSTANDING_V3_PROMPT_VERSION}。输出 schema_version 必须是 "
        f"{QUERY_UNDERSTANDING_V3_SCHEMA_VERSION}。\n"
        "你是企业 RAG 的问题结构分析器，不回答问题、不判断事实、不选择知识库、文档、"
        "权限、范围、覆盖规则、检索词或执行方式。输入只包含服务器签发的 source span catalog。\n"
        "每个 answer candidate 只能填写 catalog 中已给出的 span_id：target_span_id 必须选择"
        "source=current 的一个 span；qualifier_span_ids 也只能选择 catalog 中的 span_id。"
        "每个 candidate 必须且只能包含 id、target_span_id、qualifier_span_ids 三个字段；"
        "id 按 a1、a2 顺序填写。若当前问题本身已经完整，qualifier_span_ids 必须是空数组；"
        "绝不能把 target_span_id 再重复放入 qualifier_span_ids。"
        "绝不能输出或复述原文、span/text、start/end/offset、turn key、会话 ID、KB/文档 ID、"
        "产品/版本范围、同义词、金额、职级、日期或其他事实。\n"
        "不得输出 scope、scope_span_ids、bridge、bridge_probe、bridge edge、依赖边、"
        "coverage/retrieval 字段。桥接关系、范围和执行关系完全由后端从已验证 span 决定。\n"
        "顶层 knowledge_request 只声明知识能力类型：普通知识正文问答必须使用"
        "resource=document_content、operation=answer、filter_span_ids=[]、group_by=none、"
        "status_filter=any。只有用户明确询问知识库中文档/文章/资料本身的数量、名称、"
        "状态、类型或分组统计时，才使用 resource=document_catalog；数量用 count，列举"
        "名称或状态用 list，明确要求按知识库/状态/文件类型分别统计时用 group，并选择"
        "对应 group_by。filter_span_ids 只能选择 catalog 中表示主题筛选条件的原文 span，"
        "不得把‘文章/文档/知识库/多少/哪些’等资源名或问句外壳当作筛选条件。"
        "status_filter 只表达用户明确要求的文档状态；未明确时必须为 any。"
        "knowledge_request 不能包含 SQL、ID、事实、结果或自由文本。\n"
        "knowledge_request.answer_form 只描述用户期望的回答结构，不选择检索或证据："
        "单点事实用 fact；询问有哪些、列举项目用 enumeration；询问怎么办、如何操作、"
        "步骤或流程用 procedure；要求整体介绍、总结制度或完整内容用 overview；"
        "比较差异用 comparison；要求判断是否成立用 judgement。"
        "不得因为某个业务主题固定选择一种形态，必须依据当前问句的表达目标选择。\n"
        "输入中的 results 是服务器根据上一轮已验证展示结果生成的临时句柄目录。"
        "当用户说第一篇、第二个、最后一篇、这篇、刚才那篇、前两篇或比较这些结果时，"
        "必须使用 resource=document_result，并且 result_handles 只能选择 results 中的 handle；"
        "查看正文使用 operation=read，概括一篇使用 summarize，比较至少两篇使用 compare。"
        "其他 resource 的 result_handles 必须为空数组。句柄只是引用，不得根据 label 猜测事实，"
        "也不得输出 label、文档 ID 或知识库 ID。\n"
        "同一句内已有的限定词应使用 current span_id；仅当前输入确实缺少限定词且必须依赖"
        "既有上下文时，才选择 catalog 中的 route_context span_id。后端会依据你实际选择的"
        "span 来源推导是否追问和其它关系。输出中不得包含任何自由文本字段。"
        "只输出一个合法 JSON object，不要输出 Markdown、解释或 JSON 之外的文字。"
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
    deterministic_result = _deterministic_result_reference_analysis(
        question=question,
        catalog=catalog,
    )
    if deterministic_result is not None and selected_mode != "off":
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        trace_event(
            "query.understanding.v3.validated",
            **common_trace,
            model="server_deterministic_result_reference",
            accepted=True,
            latency_ms=latency_ms,
            origin="deterministic",
            analysis_summary=deterministic_result.safe_summary(),
        )
        return QueryUnderstandingV3RunResult(
            mode=selected_mode,
            catalog=catalog,
            analysis=deterministic_result,
            model="server_deterministic_result_reference",
            latency_ms=latency_ms,
            origin="deterministic",
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
    thinking_disabled = False
    structured_output_attempted_modes: tuple[str, ...] = ()
    model_error_type: str | None = None
    model_error_status_code: int | None = None
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
        structured_output_attempted_modes = structured.attempted_modes
        json_object_fallback_used = structured.mode != "json_schema"
        thinking_disabled = structured.thinking_disabled
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
            thinking_disabled=thinking_disabled,
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
            thinking_disabled=thinking_disabled,
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
            structured_output_mode=structured_output_mode,
            thinking_disabled=thinking_disabled,
        )
    except asyncio.TimeoutError as exc:
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        reason = "timeout"
        structured_output_attempted_modes = tuple(
            str(item)
            for item in getattr(exc, "structured_output_attempted_modes", ())
        )
    except QueryUnderstandingV3ValidationError as exc:
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        reason = _validation_error_code(exc)
    except Exception as exc:
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        reason = "model_error"
        model_error_type = type(exc).__name__
        raw_status_code = getattr(exc, "status_code", None)
        try:
            model_error_status_code = (
                int(raw_status_code) if raw_status_code is not None else None
            )
        except (TypeError, ValueError):
            model_error_status_code = None
        structured_output_attempted_modes = tuple(
            str(item)
            for item in getattr(exc, "structured_output_attempted_modes", ())
        )
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
        thinking_disabled=thinking_disabled,
        structured_output_attempted_modes=list(structured_output_attempted_modes),
        model_error_type=model_error_type,
        model_error_status_code=model_error_status_code,
        rejection_reason=reason,
        **content_fields("query_understanding_v3_raw_response", raw_content),
    )
    trace_event(
        "query.understanding.v3.fallback",
        **common_trace,
        model=model,
        reason=reason,
        latency_ms=latency_ms,
        structured_output_attempted_modes=list(structured_output_attempted_modes),
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
        thinking_disabled=thinking_disabled,
    )


__all__ = [
    "QUERY_UNDERSTANDING_V3_MAX_TOKENS",
    "QUERY_UNDERSTANDING_V3_PROMPT_VERSION",
    "QueryUnderstandingMode",
    "QueryUnderstandingOrigin",
    "QueryUnderstandingV3RunResult",
    "analyze_query_understanding",
]
