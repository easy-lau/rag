"""Shared wire vocabulary for model-produced evidence decisions.

The model may use an unambiguous spelling variant when the provider only
supports ``json_object`` and cannot enforce the generated JSON Schema.  This
module owns the vocabulary and normalization so prompts, parsers and traces do
not grow separate per-provider aliases.  Normalization never grants scope or
evidence authority; the backend recomputes the final coverage decision.
"""

from __future__ import annotations

from typing import Literal


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
    "COVERAGE_STATUS_ALIASES",
    "COVERAGE_STATUSES",
    "CoverageStatus",
    "coverage_status_protocol_text",
    "normalize_coverage_status",
]
