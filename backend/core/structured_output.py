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
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping


StructuredOutputMode = Literal["json_schema", "json_object", "plain_json"]


@dataclass(frozen=True)
class StructuredOutputResult:
    response: Any
    mode: StructuredOutputMode
    attempted_modes: tuple[StructuredOutputMode, ...]
    thinking_disabled: bool = False


_CAPABILITY_CACHE: dict[tuple[str, str], StructuredOutputMode] = {}
_THINKING_CONTROL_CACHE: dict[tuple[str, str], bool] = {}
_MODE_ORDER: tuple[StructuredOutputMode, ...] = (
    "json_schema",
    "json_object",
    "plain_json",
)

# Several OpenAI-compatible gateways reject ``response_format=json_object``
# unless the prompt itself explicitly asks for JSON.  Keep this instruction in
# the shared transport adapter: callers should not need provider-specific
# prompt branches merely to use a weaker wire-format mode.
_JSON_ONLY_INSTRUCTION = (
    "Return exactly one valid JSON object. Do not include Markdown, prose, or "
    "any text outside the JSON object."
)


def _configured_initial_mode(*, provider_identity: object, model: object) -> StructuredOutputMode | None:
    """Read the optional administrator-selected mode without coupling imports."""

    try:
        from config import get_settings

        configured = str(getattr(get_settings(), "llm_structured_output_mode", "auto") or "auto")
    except Exception:
        configured = "auto"
    if configured in _MODE_ORDER:
        return configured  # explicit mode applies to the configured LLM service
    return None


def _preferred_initial_mode(
    preferred_mode: object,
    *,
    provider_identity: object,
    model: object,
) -> StructuredOutputMode | None:
    """Resolve an optional role-specific mode before the global default."""

    normalized = str(preferred_mode or "auto").strip().casefold()
    if normalized in _MODE_ORDER:
        return normalized  # type: ignore[return-value]
    if normalized != "auto":
        raise ValueError("structured_output_mode 无效")
    return _configured_initial_mode(
        provider_identity=provider_identity,
        model=model,
    )


def _configured_disable_thinking() -> bool:
    """Read the administrator-declared transport capability."""

    try:
        from config import get_settings

        return bool(getattr(get_settings(), "llm_disable_thinking", False))
    except Exception:
        return False


def clear_structured_output_capability_cache() -> None:
    """Clear process-local capability observations (primarily for tests)."""

    _CAPABILITY_CACHE.clear()
    _THINKING_CONTROL_CACHE.clear()


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        if value is not None:
            return int(value)
    except (TypeError, ValueError):
        pass
    # AxonHub and some OpenAI-compatible wrappers omit ``status_code`` but put
    # the HTTP code in the rendered error, e.g. ``Error code: 400 - ...``.
    # Recover only the explicit 400/422 values used for format negotiation.
    detail = str(exc)
    match = re.search(r"\b(?:error\s+code|status[_ ]?code)\s*[:=]\s*(400|422)\b", detail, re.I)
    if match:
        return int(match.group(1))
    # The AxonHub gateway emits ``Request failed: Bad Request, error: ...``
    # without exposing a numeric status on the client exception.  This is the
    # standard HTTP 400 phrase; the caller still additionally requires an
    # explicit response_format rejection before allowing a downgrade.
    if "bad request" in detail.casefold():
        return 400
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


def _thinking_control_is_unsupported(exc: BaseException) -> bool:
    """Detect an explicit gateway rejection of a declared thinking control."""

    if _status_code(exc) not in {400, 422}:
        return False
    body = getattr(exc, "body", None)
    try:
        body_text = json.dumps(body, ensure_ascii=False, default=str) if body else ""
    except (TypeError, ValueError):
        body_text = str(body or "")
    detail = f"{exc} {body_text}".casefold()
    rejection_markers = (
        "unsupported",
        "not supported",
        "does not support",
        "unavailable",
        "unknown parameter",
        "unrecognized",
        "unexpected",
        "extra inputs are not permitted",
        "invalid parameter",
        "不支持",
        "不可用",
        "未知参数",
    )
    return "thinking" in detail and any(
        marker in detail for marker in rejection_markers
    )


def _with_disabled_thinking(request: Mapping[str, Any]) -> dict[str, Any]:
    """Merge a declared non-reasoning control without dropping caller extras."""

    updated = dict(request)
    raw_extra_body = request.get("extra_body")
    extra_body = (
        dict(raw_extra_body)
        if isinstance(raw_extra_body, Mapping)
        else {}
    )
    extra_body["thinking"] = {"type": "disabled"}
    updated["extra_body"] = extra_body
    return updated


def _without_thinking_control(request: Mapping[str, Any]) -> dict[str, Any]:
    """Return a request with only the optional thinking control removed."""

    updated = dict(request)
    raw_extra_body = request.get("extra_body")
    if not isinstance(raw_extra_body, Mapping):
        return updated
    extra_body = dict(raw_extra_body)
    extra_body.pop("thinking", None)
    if extra_body:
        updated["extra_body"] = extra_body
    else:
        updated.pop("extra_body", None)
    return updated


async def create_stream_completion(
    client: Any,
    *,
    request: Mapping[str, Any],
    provider_identity: object,
    model: object,
    disable_thinking: bool | None = None,
) -> tuple[Any, bool]:
    """Open a streaming completion with declared thinking negotiation.

    ``thinking.type=disabled`` is sent only when the caller/configuration has
    declared that capability.  No model name enables transport behaviour.  If
    the gateway rejects the optional parameter, retry once without it before
    any text can be emitted, then cache the observed endpoint/model capability.
    """

    key = _cache_key(provider_identity=provider_identity, model=model)
    thinking_enabled = bool(
        (_configured_disable_thinking() if disable_thinking is None else disable_thinking)
        and _THINKING_CONTROL_CACHE.get(key, True)
    )
    call_request = (
        _with_disabled_thinking(request) if thinking_enabled else dict(request)
    )
    try:
        stream = await client.chat.completions.create(**call_request)
    except Exception as exc:
        if not (thinking_enabled and _thinking_control_is_unsupported(exc)):
            raise
        _THINKING_CONTROL_CACHE[key] = False
        stream = await client.chat.completions.create(
            **_without_thinking_control(call_request)
        )
        return stream, False
    if thinking_enabled:
        _THINKING_CONTROL_CACHE[key] = True
    return stream, thinking_enabled


def _cache_key(*, provider_identity: object, model: object) -> tuple[str, str]:
    return (
        str(provider_identity or "default").strip().casefold(),
        str(model or "").strip().casefold(),
    )


def _with_json_only_instruction(request: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a request and append a trusted JSON-only system instruction.

    The instruction is appended only for ``json_object`` and plain-JSON
    transports.  It is independent from caller-provided business content and
    preserves every existing message unchanged.
    """

    updated = dict(request)
    raw_messages = request.get("messages")
    if not isinstance(raw_messages, (list, tuple)):
        return updated
    messages: list[Any] = [
        dict(message) if isinstance(message, Mapping) else message
        for message in raw_messages
    ]
    messages.append({"role": "system", "content": _JSON_ONLY_INSTRUCTION})
    updated["messages"] = messages
    return updated


async def create_structured_completion(
    client: Any,
    *,
    request: Mapping[str, Any],
    strict_response_format: Mapping[str, Any],
    timeout_seconds: float,
    provider_identity: object,
    model: object,
    disable_thinking: bool | None = None,
    structured_output_mode: object = "auto",
) -> StructuredOutputResult:
    """Create one completion using the strongest supported JSON transport.

    Validation of the returned JSON remains the caller's responsibility.  A
    plain transport is only a wire-level compatibility fallback; it does not
    weaken the caller's schema or source-contract parser.
    """

    total_timeout = max(0.1, float(timeout_seconds))
    deadline = time.perf_counter() + total_timeout
    key = _cache_key(provider_identity=provider_identity, model=model)
    configured_mode = _preferred_initial_mode(
        structured_output_mode,
        provider_identity=provider_identity,
        model=model,
    )
    cached_mode = _CAPABILITY_CACHE.get(key)
    capability_verified = configured_mode is not None or cached_mode is not None
    initial_mode = configured_mode or cached_mode or "json_schema"
    initial_index = _MODE_ORDER.index(initial_mode)
    attempted: list[StructuredOutputMode] = []
    base_request = dict(request)
    thinking_control_enabled = bool(
        (_configured_disable_thinking() if disable_thinking is None else disable_thinking)
        and _THINKING_CONTROL_CACHE.get(key, True)
    )

    for mode in _MODE_ORDER[initial_index:]:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.005:
                raise TimeoutError("structured_output_deadline_exhausted")
            mode_index = _MODE_ORDER.index(mode)
            has_compatibility_mode = mode_index + 1 < len(_MODE_ORDER)
            attempt_timeout = remaining
            if not capability_verified and has_compatibility_mode:
                # An unprobed transport must not consume the complete request
                # deadline.  Reserve a bounded tail for the next compatible
                # mode; administrator-selected and process-cached modes retain
                # the full deadline because their capability is already known.
                reserve = min(5.0, max(0.01, total_timeout * 0.35))
                attempt_timeout = max(0.005, remaining - reserve)
            call_request = dict(base_request)
            if thinking_control_enabled:
                call_request = _with_disabled_thinking(call_request)
            if mode != "json_schema":
                call_request = _with_json_only_instruction(call_request)
            call_request["timeout"] = attempt_timeout
            if mode == "json_schema":
                call_request["response_format"] = dict(strict_response_format)
            elif mode == "json_object":
                call_request["response_format"] = {"type": "json_object"}
            if not attempted or attempted[-1] != mode:
                attempted.append(mode)
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**call_request),
                    timeout=attempt_timeout,
                )
            except asyncio.TimeoutError as exc:
                # Some gateways keep an unsupported json_schema request open
                # until their upstream timeout instead of returning 400. A strict
                # absolute deadline would otherwise make compatibility fallback
                # unreachable. Only the strongest mode gets a fresh retry budget.
                if mode == "json_schema":
                    # A timeout may be transient, so do not persist a capability
                    # downgrade.  The reserved part of the original absolute
                    # deadline still permits a compatibility attempt now.
                    try:
                        setattr(exc, "structured_output_attempted_modes", tuple(attempted))
                    except (AttributeError, TypeError):
                        pass
                    break
                try:
                    setattr(exc, "structured_output_attempted_modes", tuple(attempted))
                except (AttributeError, TypeError):
                    pass
                raise
            except Exception as exc:
                if (
                    thinking_control_enabled
                    and _thinking_control_is_unsupported(exc)
                ):
                    # The current proxy/model cannot forward the declared
                    # optional control. Retry the same response-format mode
                    # without it and remember the capability for this process.
                    thinking_control_enabled = False
                    _THINKING_CONTROL_CACHE[key] = False
                    continue
                if not response_format_is_unsupported(exc, mode=mode):
                    # Preserve attempt diagnostics for callers' safe traces without
                    # changing the original exception type or error handling.
                    try:
                        setattr(exc, "structured_output_attempted_modes", tuple(attempted))
                    except (AttributeError, TypeError):
                        pass
                    raise
                next_index = _MODE_ORDER.index(mode) + 1
                if next_index >= len(_MODE_ORDER):
                    raise
                _CAPABILITY_CACHE[key] = _MODE_ORDER[next_index]
                break
            _CAPABILITY_CACHE[key] = mode
            if thinking_control_enabled:
                _THINKING_CONTROL_CACHE[key] = True
            return StructuredOutputResult(
                response=response,
                mode=mode,
                attempted_modes=tuple(attempted),
                thinking_disabled=thinking_control_enabled,
            )
    raise RuntimeError("structured output negotiation produced no attempt")


__all__ = [
    "StructuredOutputMode",
    "StructuredOutputResult",
    "clear_structured_output_capability_cache",
    "create_stream_completion",
    "create_structured_completion",
    "response_format_is_unsupported",
]
