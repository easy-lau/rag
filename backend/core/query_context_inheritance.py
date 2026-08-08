"""Trusted eligibility checks for historical user-source inheritance.

An earlier user turn is not a bag of independently reusable words.  Its
subject, product/version/project boundary and operational conditions form one
applicability envelope. A later semantic selection may therefore inherit
only a *single, exact entity span* from a source whose envelope is otherwise
empty.  This module is deliberately shared by the deterministic ellipsis path
and the model-backed compiler so one path cannot keep an entity after the
other path has rejected the source's scope or condition.

It does not select evidence, create retrieval queries or inspect assistant
text.  It merely proves whether a literal historical user span is safe to use
as a bare entity qualifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.query_constraints import extract_query_constraints
from core.query_surface_structure import parse_query_surface_frame


_TURN_KEY_RE = re.compile(r"^t[1-9][0-9]{0,2}$")
_ENTITY_COORDINATION_RE = re.compile(
    r"(?:、|[,，]|以及|和|及|与|并且|还有)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HistoricalEntitySpan:
    """One exact, source-verified historical entity range."""

    source_key: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        source_key = str(self.source_key or "").strip()
        if not _TURN_KEY_RE.fullmatch(source_key):
            raise ValueError("historical entity source key is invalid")
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("historical entity source range is invalid")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("historical entity source text is empty")
        object.__setattr__(self, "source_key", source_key)


@dataclass(frozen=True)
class HistoricalContextInheritability:
    """Whether one historical user input may supply a bare entity qualifier.

    ``reason`` is a stable policy code rather than a model judgement.  The
    optional ``entity`` appears only after the full historical source passes
    the envelope test, and is always an exact literal substring of that input.
    """

    source_key: str
    reason: str
    entity: HistoricalEntitySpan | None = None

    def __post_init__(self) -> None:
        source_key = str(self.source_key or "").strip()
        if not _TURN_KEY_RE.fullmatch(source_key):
            raise ValueError("historical context source key is invalid")
        reason = str(self.reason or "").strip()
        if not reason:
            raise ValueError("historical context inheritability requires a reason")
        if self.entity is not None and self.entity.source_key != source_key:
            raise ValueError("historical entity must belong to assessed source")
        object.__setattr__(self, "source_key", source_key)
        object.__setattr__(self, "reason", reason)

    @property
    def inheritable(self) -> bool:
        return self.entity is not None

    def allows_range(self, *, start: object, end: object) -> bool:
        """Whether a catalog qualifier is exactly the proved entity span."""

        if self.entity is None:
            return False
        return bool(start == self.entity.start and end == self.entity.end)

    def safe_summary(self) -> dict[str, object]:
        return {
            "source_key": self.source_key,
            "inheritable": self.inheritable,
            "reason": self.reason,
            "entity_range": (
                [self.entity.start, self.entity.end]
                if self.entity is not None
                else None
            ),
        }


def assess_historical_context_inheritability(
    *,
    source_key: str,
    user_input: str,
) -> HistoricalContextInheritability:
    """Assess a prior user turn as one indivisible applicability envelope.

    The accepted form has exactly one entity qualifier and no other scope,
    condition or residual context.  Rejecting a richer turn is intentional:
    selecting only its entity would otherwise silently discard city, date,
    product/version/project or business conditions from the user's request.
    """

    normalized_key = str(source_key or "").strip()
    if not _TURN_KEY_RE.fullmatch(normalized_key):
        raise ValueError("historical context source key is invalid")
    source = str(user_input or "")
    if not source.strip():
        return HistoricalContextInheritability(
            source_key=normalized_key,
            reason="source_unavailable",
        )

    scope = extract_query_constraints(source)
    if scope.has_scope_constraint:
        return HistoricalContextInheritability(
            source_key=normalized_key,
            reason="explicit_scope",
        )

    frame = parse_query_surface_frame(source)
    if frame is None or len(frame.entity_qualifiers) != 1:
        return HistoricalContextInheritability(
            source_key=normalized_key,
            reason="entity_not_unique_or_not_inheritable",
        )
    if (
        len(frame.qualifiers) != 1
        or frame.qualifiers[0] != frame.entity_qualifiers[0]
        or bool(frame.context_terms)
    ):
        return HistoricalContextInheritability(
            source_key=normalized_key,
            reason="non_inheritable_qualifier",
        )

    qualifier = frame.entity_qualifiers[0]
    if _ENTITY_COORDINATION_RE.search(qualifier.text) is not None:
        return HistoricalContextInheritability(
            source_key=normalized_key,
            reason="entity_not_unique_or_not_inheritable",
        )
    start = source.find(qualifier.text)
    if start < 0 or source.find(qualifier.text, start + len(qualifier.text)) >= 0:
        return HistoricalContextInheritability(
            source_key=normalized_key,
            reason="entity_source_not_unique",
        )
    return HistoricalContextInheritability(
        source_key=normalized_key,
        reason="unique_entity_qualifier",
        entity=HistoricalEntitySpan(
            source_key=normalized_key,
            start=start,
            end=start + len(qualifier.text),
            text=qualifier.text,
        ),
    )


__all__ = [
    "HistoricalContextInheritability",
    "HistoricalEntitySpan",
    "assess_historical_context_inheritability",
]
