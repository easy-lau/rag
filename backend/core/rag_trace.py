"""Machine-readable RAG algorithm tracing.

The normal application logger remains concise.  This module emits one JSON
object per event under the ``rag.trace`` logger so development logs can later
be parsed into evaluation samples without scraping human-oriented messages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from config import get_settings


logger = logging.getLogger("rag.trace")
# v2 separates displayed retrieval candidates from evidence actually injected
# into generation.  v1 remains readable in persisted/exported historical rows;
# consumers must group metrics by this field instead of mixing both semantics.
TRACE_SCHEMA_VERSION = 2

_SENSITIVE_KEY_PARTS = (
    "apikey",
    "password",
    "passwd",
    "defaultpwd",
    "clientsecret",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "jwttoken",
    "bearertoken",
    "authorization",
    "proxyauthorization",
    "setcookie",
)
_SENSITIVE_EXACT_KEYS = {
    "secret",
    "pwd",
    "cookie",
    "token",
    "accesstoken",
    "refreshtoken",
    "idtoken",
    "jwttoken",
    "bearertoken",
}
_SAFE_TOKEN_METRIC_KEYS = {
    "tokens",
    "prompttokens",
    "completiontokens",
    "totaltokens",
    "maxtokens",
}
# SDK/数据库异常不只包含 HTTP URL，也可能回显带账号密码的 PostgreSQL、
# Redis、AMQP 等连接串。统一识别 RFC 风格 scheme，随后仅保留 scheme、
# host、port 与 path，去掉 userinfo、query 和 fragment。
_URL_RE = re.compile(
    r"[a-z][a-z0-9+.-]*://[^\s\"'<>]+",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|password|passwd|pwd|defaultpwd|secret|"
    r"client[_-]?secret|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"authorization|cookie)\b[\"']?\s*[:=]\s*)"
    r"(?:[\"'])?[^\s,;\]}]+(?:[\"'])?"
)
_BUSINESS_CONTENT_KEYS = {
    "answer",
    "body",
    "candidatecontent",
    "constraintoverridereason",
    "constraintreason",
    "content",
    "context",
    "extractionreason",
    "filename",
    "inputpreview",
    "intentrawresponse",
    "matchedtext",
    "message",
    "metadata",
    "outputpreview",
    "partialanswer",
    "pipelineoverridereason",
    "prompt",
    "query",
    "question",
    "rawcontent",
    "requestbody",
    "responsebody",
    "rerankreason",
    "selectedtags",
    "standalonequery",
    "tags",
}


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _is_sensitive_key(value: Any) -> bool:
    key = _normalized_key(value)
    if key in _SAFE_TOKEN_METRIC_KEYS:
        return False
    return key in _SENSITIVE_EXACT_KEYS or any(
        part in key for part in _SENSITIVE_KEY_PARTS
    )


def _redact_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, "", ""))
    except (TypeError, ValueError):
        return "[REDACTED_URL]"


def redact_sensitive_text(value: str) -> str:
    """Remove credentials while retaining enough error text for diagnosis."""

    text = _BEARER_RE.sub("Bearer [REDACTED]", str(value))
    text = _CREDENTIAL_ASSIGNMENT_RE.sub(r"\1[REDACTED]", text)
    return _URL_RE.sub(_redact_url, text)


def _has_trace_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def trace_contains_business_content(value: Any) -> bool:
    """Detect nested fields that may carry user, model, or document text.

    The detector intentionally follows field semantics instead of treating every
    string as content. Model names, stage codes and IDs therefore remain safe
    diagnostics, while nested ``results[].candidate_content`` and metadata are
    still recognized even when the request event was dropped before persistence.
    """

    if isinstance(value, dict):
        for key, item in value.items():
            if (
                _normalized_key(key) in _BUSINESS_CONTENT_KEYS
                and _has_trace_value(item)
            ):
                return True
            if trace_contains_business_content(item):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(trace_contains_business_content(item) for item in value)
    return False


def exception_log_text(exc: BaseException) -> str:
    """返回适合普通应用日志的异常摘要。

    开发环境保留完整异常，便于定位兼容接口返回；生产环境只保留异常类型，
    避免 SDK 异常把供应商 URL、响应体、请求头或内部标识写入容器日志。
    """

    if get_settings().rag_trace_include_content:
        return f"{type(exc).__name__}: {redact_sensitive_text(str(exc))}"
    return type(exc).__name__


def log_exception_safely(
    target_logger: logging.Logger,
    message: str,
    *args: Any,
    exc: BaseException,
) -> None:
    """开发环境记录 traceback，生产环境仅记录异常类型。"""

    if get_settings().rag_trace_include_content:
        target_logger.exception(
            f"{message} error=%s",
            *args,
            exception_log_text(exc),
        )
    else:
        target_logger.error(
            f"{message} error=%s",
            *args,
            type(exc).__name__,
        )


def json_safe(
    value: Any,
    *,
    include_exception_message: bool = True,
    redact_sensitive: bool = False,
) -> Any:
    """Recursively convert values to deterministic JSON-compatible data."""

    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            output_key = str(key)
            if redact_sensitive and _is_sensitive_key(output_key):
                result[output_key] = "[REDACTED]"
                continue
            result[output_key] = json_safe(
                item,
                include_exception_message=include_exception_message,
                redact_sensitive=redact_sensitive,
            )
        return result
    if isinstance(value, set):
        return [
            json_safe(
                item,
                include_exception_message=include_exception_message,
                redact_sensitive=redact_sensitive,
            )
            for item in sorted(value, key=str)
        ]
    if isinstance(value, (list, tuple)):
        return [
            json_safe(
                item,
                include_exception_message=include_exception_message,
                redact_sensitive=redact_sensitive,
            )
            for item in value
        ]
    if isinstance(value, BaseException):
        error = {"type": type(value).__name__}
        if include_exception_message:
            message = str(value)
            error["message"] = (
                redact_sensitive_text(message)
                if redact_sensitive
                else message
            )
        return error
    if redact_sensitive and isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def trace_query_constraints(value: Any) -> dict[str, Any]:
    """生成可写入结构化日志的查询约束。

    生产环境保留产品/版本的结构化结果，但移除会重复用户原文片段的
    ``matched_text`` 和 ``extraction_reason``。
    """

    if hasattr(value, "as_dict"):
        value = value.as_dict()
    raw = dict(value or {}) if isinstance(value, dict) else {}
    if not get_settings().rag_trace_include_content:
        raw.pop("matched_text", None)
        raw.pop("extraction_reason", None)
    return json_safe(raw, redact_sensitive=True)


def content_fields(name: str, content: str | None) -> dict[str, Any]:
    """Return safe trace fields for user/model content.

    Content is included only when development tracing explicitly allows it.
    Length and digest are always available for correlation and regression
    analysis without retaining the original business text.
    """

    text = redact_sensitive_text(content or "")
    fields: dict[str, Any] = {
        f"{name}_chars": len(text),
        f"{name}_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    settings = get_settings()
    if settings.rag_trace_include_content:
        limit = settings.rag_trace_content_max_chars
        fields[name] = text if len(text) <= limit else f"{text[:limit]}\n...[truncated {len(text) - limit} chars]"
    return fields


def trace_event(event: str, /, **payload: Any) -> None:
    """Emit a single structured trace event when tracing is enabled."""

    settings = get_settings()
    if not settings.rag_trace_enabled:
        return
    record = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "app_version": getattr(settings, "app_version", "dev"),
        "app_revision": getattr(settings, "app_revision", ""),
        "event": event,
        # Conservative per-event capture policy marker. Every event carries it,
        # so a dropped request event cannot make an exported development trace
        # falsely claim that no business content may be present.
        "content_capture_enabled": settings.rag_trace_include_content,
        **payload,
    }
    try:
        safe_record = json_safe(
            record,
            include_exception_message=settings.rag_trace_include_content,
            redact_sensitive=True,
        )
        safe_record["contains_business_content"] = trace_contains_business_content(
            safe_record
        )
        logger.info(
            "%s",
            json.dumps(
                safe_record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
        # Lazy import keeps the tracing formatter usable in lightweight tests
        # and avoids making observability a startup dependency.
        from core.rag_trace_store import enqueue_trace_record

        enqueue_trace_record(safe_record)
    except Exception as exc:
        # 可观测代码必须 best-effort；序列化或日志 handler 故障不能打断问答。
        log_exception_safely(
            logger,
            "rag trace event serialization failed event=%s",
            event,
            exc=exc,
        )
