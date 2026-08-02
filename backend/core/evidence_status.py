"""Canonical evidence-status protocol shared by RAG producers and consumers.

Status values cross the streaming API, durable chat history, trace records and
both V1/V2 runners.  Keeping their ownership here prevents one layer from
treating a terminal non-answer result as a successful answer merely because a
rolling producer still uses an older spelling.

``version_mismatch`` is accepted only as a legacy *input/read* alias.  New
responses must emit ``scope_mismatch`` because applicability is not limited to
product versions: product, version and project are one hard boundary.
"""

from __future__ import annotations

from typing import Literal


CanonicalEvidenceStatus = Literal[
    "skipped",
    "hit",
    "partial",
    "scope_mismatch",
    "no_hit",
    "insufficient_evidence",
    "unverified",
    "needs_clarification",
    "error",
]

CANONICAL_EVIDENCE_STATUSES = frozenset({
    "skipped",
    "hit",
    "partial",
    "scope_mismatch",
    "no_hit",
    "insufficient_evidence",
    "unverified",
    "needs_clarification",
    "error",
})

# Retain only an explicit rolling-upgrade alias.  Unknown future values remain
# invalid/fail-closed at the chat persistence boundary.
LEGACY_EVIDENCE_STATUS_ALIASES = {
    "version_mismatch": "scope_mismatch",
}

ACCEPTED_EVIDENCE_STATUSES = frozenset({
    *CANONICAL_EVIDENCE_STATUSES,
    *LEGACY_EVIDENCE_STATUS_ALIASES,
})

NON_ANSWER_EVIDENCE_STATUSES = frozenset({
    "skipped",
    "scope_mismatch",
    "no_hit",
    "insufficient_evidence",
    "needs_clarification",
    "error",
    # Historical rows/source snapshots can still carry the old spelling.
    "version_mismatch",
})

ANSWER_SOURCE_REQUIRED_EVIDENCE_STATUSES = frozenset({
    "hit",
    "partial",
    "unverified",
})

# These terminal states complete a retrieval attempt and may safely resolve a
# pending server-side scope selection.  They do not all permit answer sources.
SUCCESSFUL_EVIDENCE_SCOPE_STATUSES = frozenset({
    "hit",
    "partial",
    "scope_mismatch",
    "no_hit",
    "unverified",
})


def normalize_evidence_status(value: object) -> str:
    """Normalize a legacy spelling; validity is checked by the caller/helper."""

    status = str(value or "").strip().casefold()
    return LEGACY_EVIDENCE_STATUS_ALIASES.get(status, status)


def canonical_evidence_status(value: object) -> str | None:
    """Return a supported canonical status, or ``None`` for an invalid input.

    Producers use this at their write boundary and consumers use it before
    trusting an external/rolling stream.  It intentionally accepts the one
    legacy spelling on input but never returns it, so a successful read/write
    round trip converges to ``scope_mismatch``.
    """

    normalized = normalize_evidence_status(value)
    return normalized if normalized in CANONICAL_EVIDENCE_STATUSES else None


def is_accepted_evidence_status(value: object) -> bool:
    """Whether a producer/status row is accepted during a rolling upgrade."""

    return canonical_evidence_status(value) is not None


__all__ = [
    "ACCEPTED_EVIDENCE_STATUSES",
    "ANSWER_SOURCE_REQUIRED_EVIDENCE_STATUSES",
    "CANONICAL_EVIDENCE_STATUSES",
    "CanonicalEvidenceStatus",
    "canonical_evidence_status",
    "LEGACY_EVIDENCE_STATUS_ALIASES",
    "NON_ANSWER_EVIDENCE_STATUSES",
    "SUCCESSFUL_EVIDENCE_SCOPE_STATUSES",
    "is_accepted_evidence_status",
    "normalize_evidence_status",
]
