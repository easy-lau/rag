"""Provider-neutral structured-output capability negotiation.

Semantic model adapters require validated JSON, but OpenAI-compatible
providers do not expose one uniform ``response_format`` capability.  This
module owns the negotiation so callers do not grow provider/model-specific
retry branches:

``json_schema`` -> ``json_object`` -> plain response parsed as JSON.

Only an explicit 400/422 rejection of the current response-format parameter
permits a downgrade.  All attempts share one absolute deadline and confirmed
capabilities are cached per endpoint/model for the process lifetime.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping


StructuredOutputMode = Literal["json_schema", "json_object", "plain_json"]


@dataclass(frozen=True)
class StructuredOutputResult:
    response: Any
    mode: StructuredOutputMode
    attempted_modes: tuple[StructuredOutputMode, ...]


_CAPABILITY_CACHE: dict[tuple[str, str], StructuredOutputMode] = {}
_MODE_ORDER: tuple[StructuredOutputMode, ...] = (
    "json_schema",
    "json_object",
    "plain_json",
)


def clear_structured_output_capability_cache() -> None:
    """Clear process-local capability observations (primarily for tests)."""

    _CAPABILITY_CACHE.clear()


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def response_format_is_unsupported(
    exc: BaseException,
    *,
    mode: StructuredOutputMode,
) -> bool:
    """Return true only for an explicit client-side capability rejection."""

    if mode == "plain_json" or _status_code(exc) not in {400, 422}:
        return False
    body = getattr(exc, "body", None)
    try:
        body_text = json.dumps(body, ensure_ascii=False, default=str) if body else ""
    except (TypeError, ValueError):
        body_text = str(body or "")
    detail = f"{exc} {body_text}".casefold()
    parameter_markers = (
        "response_format",
        "json_schema",
        "json schema",
        "json_object",
        "json mode",
        "json模式",
    )
    rejection_markers = (
        "unsupported",
        "not supported",
        "does not support",
        "unavailable",
        "not available",
        "unknown parameter",
        "unrecognized",
        "unexpected",
        "extra inputs are not permitted",
        "invalid parameter",
        "invalid value",
        "不支持",
        "不可用",
        "未知参数",
    )
    return (
        any(marker in detail for marker in parameter_markers)
        and any(marker in detail for marker in rejection_markers)
    )


def _cache_key(*, provider_identity: object, model: object) -> tuple[str, str]:
    return (
        str(provider_identity or "default").strip().casefold(),
        str(model or "").strip().casefold(),
    )


async def create_structured_completion(
    client: Any,
    *,
    request: Mapping[str, Any],
    strict_response_format: Mapping[str, Any],
    timeout_seconds: float,
    provider_identity: object,
    model: object,
) -> StructuredOutputResult:
    """Create one completion using the strongest supported JSON transport.

    Validation of the returned JSON remains the caller's responsibility.  A
    plain transport is only a wire-level compatibility fallback; it does not
    weaken the caller's schema or source-contract parser.
    """

    deadline = time.perf_counter() + max(0.1, float(timeout_seconds))
    key = _cache_key(provider_identity=provider_identity, model=model)
    initial_mode = _CAPABILITY_CACHE.get(key, "json_schema")
    initial_index = _MODE_ORDER.index(initial_mode)
    attempted: list[StructuredOutputMode] = []
    base_request = dict(request)

    for mode in _MODE_ORDER[initial_index:]:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.05:
            raise TimeoutError("structured_output_deadline_exhausted")
        call_request = dict(base_request)
        call_request["timeout"] = remaining
        if mode == "json_schema":
            call_request["response_format"] = dict(strict_response_format)
        elif mode == "json_object":
            call_request["response_format"] = {"type": "json_object"}
        attempted.append(mode)
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**call_request),
                timeout=remaining,
            )
        except Exception as exc:
            if not response_format_is_unsupported(exc, mode=mode):
                raise
            next_index = _MODE_ORDER.index(mode) + 1
            if next_index >= len(_MODE_ORDER):
                raise
            _CAPABILITY_CACHE[key] = _MODE_ORDER[next_index]
            continue
        _CAPABILITY_CACHE[key] = mode
        return StructuredOutputResult(
            response=response,
            mode=mode,
            attempted_modes=tuple(attempted),
        )
    raise RuntimeError("structured output negotiation produced no attempt")


__all__ = [
    "StructuredOutputMode",
    "StructuredOutputResult",
    "clear_structured_output_capability_cache",
    "create_structured_completion",
    "response_format_is_unsupported",
]
