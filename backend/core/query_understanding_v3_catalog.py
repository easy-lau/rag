"""Trusted source-span catalog for ``query_understanding.v3``.

The catalog is the only object which knows source text and offsets.  A model
gets a deliberately reduced view (``span_id``, source class and literal text)
and can later refer only to those opaque identifiers.  It cannot mint a new
source span, change an offset, or smuggle a historical assistant response into
the query-understanding contract.

This module is intentionally independent of the V2 query-analysis contract.
It is a bounded input adapter, not a semantic planner: candidate fragments are
deterministic slices of current/route-authorised *user* text only.  Any
meaning such as scope, bridge kind, coverage or retrieval wording remains a
trusted compiler concern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from core.query_surface_structure import (
    parse_contextual_ellipsis_target,
    parse_query_surface_frame,
)


SOURCE_SPAN_CATALOG_SCHEMA_VERSION = "source_span_catalog.v1"
MAX_CURRENT_QUESTION_CHARS = 8000
MAX_ROUTE_CONTEXT_TURNS = 3
MAX_ROUTE_CONTEXT_CHARS = 2000
MAX_CATALOG_SPANS_PER_SOURCE = 96
MAX_RESULT_REFERENCES_PER_TURN = 20

CatalogSourceKind = Literal["current", "route_context"]

_TURN_KEY_RE = re.compile(r"^t[1-9][0-9]{0,2}$")
_SPAN_ID_RE = re.compile(r"^s_(?:current|t[1-9][0-9]{0,2})_[0-9]{3}$")
_RESULT_HANDLE_RE = re.compile(r"^r_(t[1-9][0-9]{0,2})_[0-9]{3}$")
_FRAGMENT_SEPARATOR_RE = re.compile(
    r"(?:以及|还有|并且|或者|并|和|与|及|或|的|[、，,；;。！？?])"
)
_TRIM_BOUNDARY_RE = re.compile(r"[\s\u3000,，、;；:：。！？?!]+")
_LEADING_DISCOURSE_RE = re.compile(
    r"^(?:请问|我想问(?:一下)?|想问(?:一下)?|麻烦(?:问)?一下|那(?:么)?|比较|对比|"
    r"请(?:帮我)?(?:说明|解释|介绍)(?:一下)?)+[\s，,:：]*"
)
_QUESTION_SUFFIX_RE = re.compile(
    r"(?:这些|这些项目|这些内容)?(?:分别|各自|都)?(?:是|有)?(?:多少|什么|哪些|怎样|如何|怎么(?:办|处理)?|"
    r"是否|能否|可以吗|吗|呢)+$"
)
_COMPARISON_SUFFIX_RE = re.compile(r"(?:的)?(?:差异|区别|不同|异同|对比)$")
_KNOWLEDGE_CATALOG_SUBJECT_RE = re.compile(
    r"(?:关于|有关(?!于)|针对|围绕)\s*"
    r"(?P<subject>[^，,。；;！？?]{1,96}?)\s*"
    r"(?:相关)?的\s*(?:知识库|文档|文章|资料|文件)",
    re.IGNORECASE,
)
_KNOWLEDGE_CATALOG_RELATED_SUBJECT_RE = re.compile(
    r"(?P<subject>[^，,。；;！？?]{1,96}?)\s*"
    r"(?:相关|有关)(?:的)?\s*(?:文档|文章|资料|文件)",
    re.IGNORECASE,
)


class SourceSpanCatalogError(ValueError):
    """Raised when a caller tries to build an unverifiable source catalog."""


@dataclass(frozen=True)
class CatalogSpan:
    """One exact, server-issued source span.

    ``source_key`` stays server-side.  The model view intentionally exposes
    only the coarse source class, so a model cannot fabricate a historical
    turn binding or discover persistent conversation identifiers.
    """

    span_id: str
    source_kind: CatalogSourceKind
    source_key: str
    start: int
    end: int
    text: str

    def model_dict(self) -> dict[str, str]:
        return {
            "span_id": self.span_id,
            "source": self.source_kind,
            "text": self.text,
        }

    @property
    def identity(self) -> tuple[str, int, int]:
        return (self.source_key, self.start, self.end)

    def overlaps(self, other: "CatalogSpan") -> bool:
        """Return whether two spans overlap in the same trusted source."""

        return (
            self.source_key == other.source_key
            and self.start < other.end
            and other.start < self.end
        )


@dataclass(frozen=True)
class CatalogResultReference:
    """One server-issued handle for a previously displayed trusted result.

    The model receives the handle, ordinal and display label, but never a
    database, knowledge-base or document id.  ``source_key`` remains local so
    a selected handle can be rebound to the exact authorised turn.
    """

    handle: str
    source_key: str
    ordinal: int
    resource: Literal["document"]
    label: str
    status: str | None = None

    def model_dict(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "ordinal": self.ordinal,
            "resource": self.resource,
            "label": self.label,
            "status": self.status,
        }


def _source_text(
    value: object,
    *,
    field: str,
    max_chars: int,
) -> str:
    if not isinstance(value, str):
        raise SourceSpanCatalogError(f"{field} 必须是字符串")
    if not value.strip():
        raise SourceSpanCatalogError(f"{field} 不能为空")
    if len(value) > max_chars:
        raise SourceSpanCatalogError(f"{field} 超过最大长度")
    return value


def _trim_range(source: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim only source characters; never normalise/rewrite a fragment."""

    while start < end and source[start].isspace():
        start += 1
    while start < end and source[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return (start, end)


def _semantic_piece_range(source: str, start: int, end: int) -> tuple[int, int] | None:
    """Produce a bounded literal phrase from a separator-delimited slice.

    This deliberately does not infer aliases or facts.  It only removes
    conversational glue at the outer boundary, retaining an exact source
    range that can be verified by :class:`SourceSpanCatalog`.
    """

    trimmed = _trim_range(source, start, end)
    if trimmed is None:
        return None
    start, end = trimmed
    text = source[start:end]
    leading = _LEADING_DISCOURSE_RE.match(text)
    if leading is not None:
        start += leading.end()
        trimmed = _trim_range(source, start, end)
        if trimmed is None:
            return None
        start, end = trimmed
        text = source[start:end]
    suffix = _QUESTION_SUFFIX_RE.search(text)
    if suffix is not None and suffix.start() > 0:
        end = start + suffix.start()
        trimmed = _trim_range(source, start, end)
        if trimmed is None:
            return None
        start, end = trimmed
    if not source[start:end].strip():
        return None
    return (start, end)


def _nested_terminal_target_range(
    source: str,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    """Expose a literal head alongside a comparison-result phrase.

    For ``审批流程差异`` the full phrase is useful to describe the request,
    while ``审批流程`` is the actual source head a deterministic comparison
    compiler can validate.  This is source slicing, not a semantic rewrite:
    both catalog entries are exact source ranges and the compiler still decides
    whether a comparison is warranted.
    """

    suffix = _COMPARISON_SUFFIX_RE.search(source[start:end])
    if suffix is None or suffix.start() <= 0:
        return None
    trimmed = _trim_range(source, start, start + suffix.start())
    return trimmed


def _fragment_ranges(source: str) -> tuple[tuple[int, int], ...]:
    """Return deterministic literal candidate fragments for one user turn.

    A whole-turn entry comes first as a safe fallback.  Subsequent entries are
    separator-delimited leaves (``普通员工`` / ``住宿标准`` / ``餐补`` in a
    typical policy question).  The extraction is intentionally shallow and
    bounded; it never consults KB content or an assistant answer.
    """

    whole = _trim_range(source, 0, len(source))
    if whole is None:  # guarded by _source_text, kept for totality
        return ()
    ranges: list[tuple[int, int]] = [whole]
    cursor = whole[0]
    for match in _FRAGMENT_SEPARATOR_RE.finditer(source, whole[0], whole[1]):
        piece = _semantic_piece_range(source, cursor, match.start())
        if piece is not None and piece not in ranges:
            ranges.append(piece)
        cursor = match.end()
    tail = _semantic_piece_range(source, cursor, whole[1])
    if tail is not None and tail not in ranges:
        ranges.append(tail)
        nested = _nested_terminal_target_range(source, *tail)
        if nested is not None and nested not in ranges:
            ranges.append(nested)
    return tuple(ranges[:MAX_CATALOG_SPANS_PER_SOURCE])


def _literal_ranges(source: str, normalized_text: str) -> tuple[tuple[int, int], ...]:
    """Find exact source ranges for one surface-parser literal.

    ``parse_query_surface_frame`` intentionally normalises whitespace for
    grammar analysis, whereas catalog spans must always point at the original
    user string.  This adapter accepts only the parser's literal text and
    reconstructs an exact source range by permitting whitespace variation
    between its words.  It never invents a synonym, value or business term.
    """

    normalized = " ".join(str(normalized_text or "").split())
    if not normalized:
        return ()
    parts = normalized.split(" ")
    pattern = r"\s+".join(re.escape(part) for part in parts)
    return tuple(
        (match.start(), match.end())
        for match in re.finditer(pattern, source, flags=re.IGNORECASE)
    )


def _surface_frame_ranges(source: str) -> tuple[tuple[int, int], ...]:
    """Add generic grammatical target/qualifier spans to the catalog.

    Separator splitting alone cannot expose compact relation forms such as
    ``普通员工对应什么职级``.  The shared source-surface parser already knows
    their literal answer head and qualifier without inspecting any knowledge
    base, so cataloguing those exact ranges gives the model enough choices
    while keeping all offsets and source text server-owned.
    """

    frame = parse_query_surface_frame(source)
    if frame is None:
        return ()
    values = [frame.answer_target, *(item.text for item in frame.qualifiers)]
    result: list[tuple[int, int]] = []
    for value in values:
        for item in _literal_ranges(source, value):
            if item not in result:
                result.append(item)
    return tuple(result)


def _knowledge_catalog_subject_ranges(source: str) -> tuple[tuple[int, int], ...]:
    """Expose exact subject literals from generic catalog-reference grammar.

    ``关于 X 的文章`` and ``X 相关文档`` identify ``X`` as a metadata filter,
    independent of any business vocabulary.  Publishing the literal range
    lets V3 select the filter without copying or rewriting user text.
    """

    result: list[tuple[int, int]] = []
    for pattern in (
        _KNOWLEDGE_CATALOG_SUBJECT_RE,
        _KNOWLEDGE_CATALOG_RELATED_SUBJECT_RE,
    ):
        for match in pattern.finditer(source):
            span = _trim_range(source, match.start("subject"), match.end("subject"))
            if span is not None and span not in result:
                result.append(span)
    return tuple(result)


def _normalise_route_context(
    values: Iterable[Mapping[str, Any]] | None,
) -> tuple[tuple[str, str], ...]:
    """Validate the explicitly route-authorised user fragments.

    Passing ``route_context`` to :func:`build_source_span_catalog` is the
    explicit authorisation boundary.  We accept no assistant text, DB ids,
    source docs or arbitrary conversation keys here; callers must reduce their
    route decision to bounded ``tN`` + ``user_input`` entries first.
    """

    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values or ()):
        if not isinstance(raw, Mapping):
            raise SourceSpanCatalogError(f"route_context[{index}] 必须是对象")
        key = raw.get("candidate_key")
        if not isinstance(key, str) or not _TURN_KEY_RE.fullmatch(key.strip()):
            raise SourceSpanCatalogError(f"route_context[{index}] 候选键非法")
        key = key.strip()
        if key in seen:
            raise SourceSpanCatalogError("route_context 包含重复候选键")
        if set(raw) - {
            "candidate_key",
            "user_input",
            "assistant_answer",
            "result_items",
        }:
            raise SourceSpanCatalogError(
                f"route_context[{index}] 包含未允许字段"
            )
        text = _source_text(
            raw.get("user_input"),
            field=f"route_context[{index}].user_input",
            max_chars=MAX_ROUTE_CONTEXT_CHARS,
        )
        seen.add(key)
        result.append((key, text))
        if len(result) > MAX_ROUTE_CONTEXT_TURNS:
            raise SourceSpanCatalogError("route_context 超过上限")
    return tuple(result)


def _normalise_result_references(
    values: Iterable[Mapping[str, Any]] | None,
) -> tuple[CatalogResultReference, ...]:
    """Validate identity-free result handles supplied by the conversation layer."""

    result: list[CatalogResultReference] = []
    seen_handles: set[str] = set()
    for raw_turn in values or ():
        if not isinstance(raw_turn, Mapping):
            continue
        source_key = str(raw_turn.get("candidate_key") or "").strip()
        if not _TURN_KEY_RE.fullmatch(source_key):
            continue
        raw_items = raw_turn.get("result_items")
        if raw_items is None:
            continue
        if not isinstance(raw_items, (list, tuple)):
            raise SourceSpanCatalogError("route_context.result_items 必须是数组")
        for expected_ordinal, raw_item in enumerate(
            raw_items[:MAX_RESULT_REFERENCES_PER_TURN],
            start=1,
        ):
            if not isinstance(raw_item, Mapping):
                raise SourceSpanCatalogError("result item 必须是对象")
            if set(raw_item) - {"handle", "ordinal", "resource", "label", "status"}:
                raise SourceSpanCatalogError("result item 包含未允许字段")
            handle = str(raw_item.get("handle") or "").strip()
            match = _RESULT_HANDLE_RE.fullmatch(handle)
            if match is None or match.group(1) != source_key:
                raise SourceSpanCatalogError("result handle 非法")
            ordinal = raw_item.get("ordinal")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal != expected_ordinal
            ):
                raise SourceSpanCatalogError("result ordinal 非法")
            if str(raw_item.get("resource") or "").strip() != "document":
                raise SourceSpanCatalogError("result resource 非法")
            label = str(raw_item.get("label") or "").strip()
            if not label or len(label) > 255:
                raise SourceSpanCatalogError("result label 非法")
            status_value = raw_item.get("status")
            status = str(status_value).strip()[:32] if status_value is not None else None
            if handle in seen_handles:
                raise SourceSpanCatalogError("result handle 重复")
            seen_handles.add(handle)
            result.append(CatalogResultReference(
                handle=handle,
                source_key=source_key,
                ordinal=ordinal,
                resource="document",
                label=label,
                status=status or None,
            ))
    return tuple(result)


@dataclass(frozen=True)
class SourceSpanCatalog:
    """Immutable, request-scoped catalog of exact user-source fragments."""

    entries: tuple[CatalogSpan, ...]
    _source_texts: tuple[tuple[str, str], ...]
    result_references: tuple[CatalogResultReference, ...] = ()

    def __post_init__(self) -> None:
        sources = dict(self._source_texts)
        if "current" not in sources:
            raise SourceSpanCatalogError("catalog 缺少当前输入来源")
        if len(sources) != len(self._source_texts):
            raise SourceSpanCatalogError("catalog 来源键重复")
        for source_key, source_text in self._source_texts:
            if source_key != "current" and not _TURN_KEY_RE.fullmatch(source_key):
                raise SourceSpanCatalogError("catalog 来源键非法")
            _source_text(
                source_text,
                field=f"catalog 来源[{source_key}]",
                max_chars=(
                    MAX_CURRENT_QUESTION_CHARS
                    if source_key == "current"
                    else MAX_ROUTE_CONTEXT_CHARS
                ),
            )
        seen_ids: set[str] = set()
        seen_identities: set[tuple[str, int, int]] = set()
        for entry in self.entries:
            if not _SPAN_ID_RE.fullmatch(entry.span_id):
                raise SourceSpanCatalogError("catalog span_id 非法")
            if not entry.span_id.startswith(f"s_{entry.source_key}_"):
                raise SourceSpanCatalogError("catalog span_id 与来源不一致")
            if entry.span_id in seen_ids:
                raise SourceSpanCatalogError("catalog span_id 重复")
            seen_ids.add(entry.span_id)
            if entry.source_key not in sources:
                raise SourceSpanCatalogError("catalog span 来源不可验证")
            expected_kind: CatalogSourceKind = (
                "current" if entry.source_key == "current" else "route_context"
            )
            if entry.source_kind != expected_kind:
                raise SourceSpanCatalogError("catalog span 来源类型不一致")
            text = sources[entry.source_key]
            if (
                isinstance(entry.start, bool)
                or isinstance(entry.end, bool)
                or not isinstance(entry.start, int)
                or not isinstance(entry.end, int)
                or entry.start < 0
                or entry.end <= entry.start
                or entry.end > len(text)
            ):
                raise SourceSpanCatalogError("catalog span 范围非法")
            if not entry.text.strip() or text[entry.start:entry.end] != entry.text:
                raise SourceSpanCatalogError("catalog span 原文不可验证")
            if entry.identity in seen_identities:
                raise SourceSpanCatalogError("catalog span 范围重复")
            seen_identities.add(entry.identity)
        if not self.entries:
            raise SourceSpanCatalogError("catalog 至少包含一个 span")
        result_handles: set[str] = set()
        for reference in self.result_references:
            if (
                not _RESULT_HANDLE_RE.fullmatch(reference.handle)
                or reference.source_key not in sources
                or reference.source_key == "current"
                or not reference.handle.startswith(f"r_{reference.source_key}_")
            ):
                raise SourceSpanCatalogError("catalog result reference 非法")
            if reference.handle in result_handles:
                raise SourceSpanCatalogError("catalog result handle 重复")
            result_handles.add(reference.handle)

    @classmethod
    def build(
        cls,
        *,
        current_question: str,
        route_context: Iterable[Mapping[str, Any]] | None = None,
    ) -> "SourceSpanCatalog":
        current = _source_text(
            current_question,
            field="current_question",
            max_chars=MAX_CURRENT_QUESTION_CHARS,
        )
        contexts = _normalise_route_context(route_context)
        result_references = _normalise_result_references(route_context)
        sources = (("current", current), *contexts)
        entries: list[CatalogSpan] = []
        for source_key, source_text in sources:
            source_kind: CatalogSourceKind = (
                "current" if source_key == "current" else "route_context"
            )
            ranges = list(_fragment_ranges(source_text))
            for item in _surface_frame_ranges(source_text):
                if item not in ranges:
                    ranges.append(item)
            for item in _knowledge_catalog_subject_ranges(source_text):
                if item not in ranges:
                    ranges.append(item)
            # The strict contextual producer binds this literal target by
            # exact offsets.  Fragment extraction currently exposes it for
            # common forms, but that is an implementation detail rather than a
            # contract.  Publish it explicitly so a future generic catalog
            # refactor cannot make a source-proven follow-up fall back to a
            # fuzzy nearby span.
            if source_key == "current":
                contextual_target = parse_contextual_ellipsis_target(source_text)
                if contextual_target is not None:
                    target_range = (
                        contextual_target.start,
                        contextual_target.end,
                    )
                    if target_range not in ranges:
                        ranges.append(target_range)
            for ordinal, (start, end) in enumerate(
                ranges[:MAX_CATALOG_SPANS_PER_SOURCE],
                start=1,
            ):
                entries.append(CatalogSpan(
                    span_id=f"s_{source_key}_{ordinal:03d}",
                    source_kind=source_kind,
                    source_key=source_key,
                    start=start,
                    end=end,
                    text=source_text[start:end],
                ))
        return cls(
            entries=tuple(entries),
            _source_texts=tuple(sources),
            result_references=result_references,
        )

    @property
    def current_entries(self) -> tuple[CatalogSpan, ...]:
        return tuple(item for item in self.entries if item.source_key == "current")

    @property
    def context_entries(self) -> tuple[CatalogSpan, ...]:
        return tuple(item for item in self.entries if item.source_kind == "route_context")

    @property
    def current_span_ids(self) -> tuple[str, ...]:
        return tuple(item.span_id for item in self.current_entries)

    @property
    def all_span_ids(self) -> tuple[str, ...]:
        return tuple(item.span_id for item in self.entries)

    @property
    def all_result_handles(self) -> tuple[str, ...]:
        return tuple(item.handle for item in self.result_references)

    @property
    def authorised_context_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self._source_texts if key != "current")

    def source_text_for(self, source_key: str) -> str:
        for key, value in self._source_texts:
            if key == source_key:
                return value
        raise SourceSpanCatalogError("catalog 来源键不可用")

    def resolve(self, span_id: object) -> CatalogSpan:
        if not isinstance(span_id, str) or not _SPAN_ID_RE.fullmatch(span_id):
            raise SourceSpanCatalogError("span_id 非法")
        for entry in self.entries:
            if entry.span_id == span_id:
                return entry
        raise SourceSpanCatalogError("span_id 不在当前 catalog")

    def resolve_result(self, handle: object) -> CatalogResultReference:
        if not isinstance(handle, str) or not _RESULT_HANDLE_RE.fullmatch(handle):
            raise SourceSpanCatalogError("result handle 非法")
        for reference in self.result_references:
            if reference.handle == handle:
                return reference
        raise SourceSpanCatalogError("result handle 不在当前 catalog")

    def find_exact_span(
        self,
        *,
        source_key: object,
        start: object,
        end: object,
    ) -> CatalogSpan | None:
        """Return the unique catalog entry for one server-verified range.

        Deterministic source grammar never invents a span ID.  It can only
        ask whether its exact, already verified range was exposed by this
        request-local catalog.  ``None`` is fail-closed: callers must retain
        the ordinary V3/model or baseline path rather than approximate a
        nearby fragment.
        """

        if (
            not isinstance(source_key, str)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            return None
        matches = tuple(
            entry
            for entry in self.entries
            if (
                entry.source_key == source_key
                and entry.start == start
                and entry.end == end
            )
        )
        return matches[0] if len(matches) == 1 else None

    def model_payload(self) -> dict[str, object]:
        """Return the exact bounded model view; offsets and source keys stay local."""

        return {
            "schema_version": SOURCE_SPAN_CATALOG_SCHEMA_VERSION,
            "spans": [entry.model_dict() for entry in self.entries],
            "results": [item.model_dict() for item in self.result_references],
        }

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_SPAN_CATALOG_SCHEMA_VERSION,
            "span_count": len(self.entries),
            "current_span_count": len(self.current_entries),
            "route_context_span_count": len(self.context_entries),
            "authorised_context_turn_count": len(self.authorised_context_keys),
            "result_reference_count": len(self.result_references),
        }


def build_source_span_catalog(
    *,
    current_question: str,
    route_context: Iterable[Mapping[str, Any]] | None = None,
) -> SourceSpanCatalog:
    """Build a request-local catalog from current plus authorised route context."""

    return SourceSpanCatalog.build(
        current_question=current_question,
        route_context=route_context,
    )


__all__ = [
    "CatalogSpan",
    "CatalogResultReference",
    "CatalogSourceKind",
    "MAX_CATALOG_SPANS_PER_SOURCE",
    "MAX_RESULT_REFERENCES_PER_TURN",
    "SOURCE_SPAN_CATALOG_SCHEMA_VERSION",
    "SourceSpanCatalog",
    "SourceSpanCatalogError",
    "build_source_span_catalog",
]
