"""Shared wire vocabulary for model-produced evidence decisions.

The model may use an unambiguous spelling variant when the provider only
supports ``json_object`` and cannot enforce the generated JSON Schema.  This
module owns the vocabulary and normalization so prompts, parsers and traces do
not grow separate per-provider aliases.  Normalization never grants scope or
evidence authority; the backend recomputes the final coverage decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


CoverageStatus = Literal["complete", "partial", "insufficient"]

COVERAGE_STATUSES = frozenset({"complete", "partial", "insufficient"})

# Only one-to-one wire spellings are accepted.  They carry no execution
# authority and are retained as diagnostics after canonicalization.
COVERAGE_STATUS_ALIASES: dict[str, CoverageStatus] = {
    "full": "complete",
    "fully_covered": "complete",
    "complete_coverage": "complete",
    "partially_covered": "partial",
    "partial_coverage": "partial",
    "not_covered": "insufficient",
    "insufficient_coverage": "insufficient",
}


def normalize_coverage_status(
    value: object,
) -> tuple[CoverageStatus, str | None, str]:
    """Normalize a model diagnostic without allowing it to reject evidence.

    Returns ``(canonical, original, resolution)``.  Unknown or missing values
    become the conservative ``insufficient`` diagnostic; the actual final
    coverage is recomputed from verified candidate bindings elsewhere.
    """

    if not isinstance(value, str):
        return "insufficient", None, "missing_diagnostic"
    original = value.strip()[:80]
    normalized = (
        original.casefold().replace("-", "_").replace(" ", "_")
    )
    if normalized in COVERAGE_STATUSES:
        return normalized, original, "exact"  # type: ignore[return-value]
    alias = COVERAGE_STATUS_ALIASES.get(normalized)
    if alias is not None:
        return alias, original, "normalized_alias"
    return "insufficient", original or None, "unknown_diagnostic"


def coverage_status_protocol_text() -> str:
    """Return the same enum contract for providers without JSON Schema."""

    return (
        "coverage_status 是诊断字段，只能使用 complete、partial 或 "
        "insufficient；full、fully_covered 等等价写法会由服务端归一化，"
        "最终覆盖状态仍由服务端根据候选绑定重新计算。"
    )


__all__ = [
    "AdjudicationOutcome",
    "AdjudicationStatus",
    "INCONCLUSIVE_FAILURE_KINDS",
    "COVERAGE_STATUS_ALIASES",
    "COVERAGE_STATUSES",
    "CoverageStatus",
    "coverage_status_protocol_text",
    "normalize_coverage_status",
]


AdjudicationStatus = Literal["succeeded", "inconclusive", "failed"]

# Failure kinds that mean "the model produced no usable conclusion" (empty
# content, output rejected by the contract parser, or deadline timeout).
# They are deliberately distinct from provider infrastructure failures: they
# never open the circuit breaker and keep the deterministic candidate scope
# eligible for server-side auto-selection.
INCONCLUSIVE_FAILURE_KINDS = frozenset({
    "empty_content",
    "timeout",
    "contract_validation",
})


@dataclass(frozen=True)
class AdjudicationOutcome:
    """裁决结果契约：区分模型无结论与供应商基础设施故障。

    ``succeeded`` 携带重排后的候选/证据选择；``inconclusive`` 表示模型没有
    给出可用结论（空内容、契约拒绝或超时），不是供应商故障；``failed`` 表示
    协议/连接级基础设施失败，保持 fail-closed（熔断、保留候选供澄清）。
    该契约只描述一次裁决调用，不决定证据角色——是否回答由确定性候选范围
    auto-confirm 与回答策略共同决定。
    """

    status: AdjudicationStatus
    reason: str | None = None
    elapsed_ms: int = 0
    payload: Any = None
