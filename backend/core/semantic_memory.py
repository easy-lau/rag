"""Durable, authorization-neutral resolved-entity memory for grounded turns.

Follow-up detection covers *reference* reuse (``这些配置``/``上述内容``) but
not *entity* reuse.  A later turn such as ``普通员工可以乘坐头等舱吗`` is a
correctly self-contained question that still mentions an entity resolved by an
earlier grounded turn.  This module persists the facts a grounded answer
actually used, extracted from markdown table rows in the answer evidence, so
the next turn can bind a repeated mention back to its already-authorized
sources without hardcoding any business vocabulary.

Extraction is a pure document-format rule (markdown tables are a structural
convention, not business knowledge).  A fact is only recorded when the cell
text actually occurs in the current question, so the memory stays bounded to
what the user asked about and never stores unrelated document content.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

SEMANTIC_MEMORY_SCHEMA_VERSION = "rag_semantic_memory.v1"
MAX_FACTS = 60
MAX_FACT_SOURCES = 40
MAX_MENTION_CHARS = 32
MIN_MENTION_CHARS = 2
_MAX_TABLE_ROWS = 40
_MAX_CELL_CHARS = 120
_MENTION_SPLIT_RE = re.compile(
    r"(?:<br\s*/?>|、|，|,|；|;|和|与|或|及|\s)+"
)
_TABLE_ROW_RE = re.compile(r"^\s*\|(?P<cells>.+)\|\s*$")
_HEADER_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
_DIGITS_ONLY_RE = re.compile(r"^\d[\d.,%]*$")


def _parse_table_rows(content: str) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Parse the first markdown table in ``content``.

    Returns ``(headers, data_rows)`` where headers may be empty when no
    separator row exists.  Only structural table syntax is recognised; the
    caller filters mentions against the current question, so no business
    vocabulary is required here.
    """

    if not isinstance(content, str) or not content.strip():
        return (), ()
    rows: list[tuple[str, ...]] = []
    for raw_line in content.splitlines():
        match = _TABLE_ROW_RE.match(raw_line)
        if match is None:
            continue
        cells = tuple(
            cell.strip()[: _MAX_CELL_CHARS]
            for cell in match.group("cells").split("|")
        )
        cells = tuple(cell for cell in cells if cell)
        if len(cells) < 2:
            continue
        rows.append(cells)
        if len(rows) >= _MAX_TABLE_ROWS:
            break
    if len(rows) < 2:
        return (), ()
    headers: tuple[str, ...] = ()
    data_rows: list[tuple[str, ...]] = []
    separator_index = -1
    for index, row in enumerate(rows):
        if all(_HEADER_SEPARATOR_CELL_RE.fullmatch(cell) for cell in row):
            separator_index = index
            break
    if separator_index > 0:
        headers = rows[separator_index - 1]
        data_rows = rows[separator_index + 1:]
    else:
        data_rows = rows[1:]
    if not data_rows:
        return (), ()
    return headers, tuple(data_rows)


def _mention_candidates(cell: str) -> tuple[str, ...]:
    """Split one table cell into candidate mention terms.

    A cell such as ``普通员工、专员`` or ``公务舱（航程>3小时）<br>经济舱（航程≤3小时）``
    may hold several values; each is an independent mention candidate.  Splits
    on structural separators only, never on business vocabulary.
    """

    if not isinstance(cell, str):
        return ()
    terms: list[str] = []
    seen: set[str] = set()
    for raw in _MENTION_SPLIT_RE.split(cell):
        term = raw.strip().strip("（）()")
        if not term or not (MIN_MENTION_CHARS <= len(term) <= MAX_MENTION_CHARS):
            continue
        if _DIGITS_ONLY_RE.fullmatch(term):
            continue
        normalized = term.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(term)
    return tuple(terms)


def _bound_mention(question: str, cell: str) -> tuple[str, ...]:
    """Return mention terms from ``cell`` that literally occur in the question."""

    normalized_question = (question or "").strip().casefold()
    if not normalized_question:
        return ()
    matched: list[str] = []
    for term in _mention_candidates(cell):
        if term.casefold() in normalized_question:
            matched.append(term)
    return tuple(dict.fromkeys(matched))


@dataclass(frozen=True)
class ResolvedFact:
    """One extracted entity binding with its exact evidence identity."""

    mention: str
    attribute: str
    value: str
    kb_id: uuid.UUID
    doc_id: uuid.UUID
    chunk_id: uuid.UUID
    filename: str
    source_turn_id: uuid.UUID
    trace_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.mention, str) or not self.mention.strip():
            raise ValueError("resolved fact requires a mention")
        if not isinstance(self.attribute, str) or not self.attribute.strip():
            raise ValueError("resolved fact requires an attribute")
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("resolved fact requires a value")
        for name, value in (
            ("kb_id", self.kb_id),
            ("doc_id", self.doc_id),
            ("chunk_id", self.chunk_id),
            ("source_turn_id", self.source_turn_id),
        ):
            if not isinstance(value, uuid.UUID):
                raise ValueError(f"resolved fact {name} must be a UUID")
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("resolved fact requires a filename")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention": self.mention,
            "attribute": self.attribute,
            "value": self.value,
            "kb_id": str(self.kb_id),
            "doc_id": str(self.doc_id),
            "chunk_id": str(self.chunk_id),
            "filename": self.filename,
            "source_turn_id": str(self.source_turn_id),
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class ResolvedEntityMemory:
    """Entity facts plus the full answer-source identity of the turn.

    ``source_chunk_ids`` retains the whole grounded source set (not only the
    fact rows), so a later turn can re-bind the complete answer evidence under
    the current authorization scope.
    """

    facts: tuple[ResolvedFact, ...]
    source_chunk_ids: tuple[uuid.UUID, ...]
    schema_version: str = SEMANTIC_MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_MEMORY_SCHEMA_VERSION:
            raise ValueError("unsupported semantic memory schema")
        if not isinstance(self.facts, tuple):
            raise ValueError("semantic memory facts must be a tuple")
        if not self.facts:
            raise ValueError("semantic memory requires facts")
        if len(self.facts) > MAX_FACTS:
            raise ValueError("semantic memory fact count exceeds the limit")
        if not isinstance(self.source_chunk_ids, tuple):
            raise ValueError("semantic memory source ids must be a tuple")
        if len(self.source_chunk_ids) > MAX_FACT_SOURCES:
            raise ValueError("semantic memory source count exceeds the limit")
        if not self.source_chunk_ids or len(self.source_chunk_ids) != len(
            set(self.source_chunk_ids)
        ):
            raise ValueError("semantic memory requires unique source chunks")
        object.__setattr__(self, "facts", tuple(self.facts))
        object.__setattr__(self, "source_chunk_ids", tuple(self.source_chunk_ids))

    @property
    def mentions(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(fact.mention for fact in self.facts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "facts": [fact.to_dict() for fact in self.facts],
            "source_chunk_ids": [str(item) for item in self.source_chunk_ids],
        }

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fact_count": len(self.facts),
            "mention_count": len(self.mentions),
            "source_chunk_count": len(self.source_chunk_ids),
        }


def parse_resolved_entity_memory(value: object) -> ResolvedEntityMemory | None:
    """Strictly parse persisted JSON; malformed shapes fail closed."""

    if not isinstance(value, Mapping):
        return None
    if set(value) != {
        "schema_version",
        "facts",
        "source_chunk_ids",
    }:
        return None
    if value.get("schema_version") != SEMANTIC_MEMORY_SCHEMA_VERSION:
        return None
    raw_facts = value.get("facts")
    raw_sources = value.get("source_chunk_ids")
    if not isinstance(raw_facts, list) or not isinstance(raw_sources, list):
        return None
    if len(raw_facts) > MAX_FACTS or len(raw_sources) > MAX_FACT_SOURCES:
        return None
    try:
        facts: list[ResolvedFact] = []
        for raw in raw_facts:
            if not isinstance(raw, Mapping) or set(raw) != {
                "mention",
                "attribute",
                "value",
                "kb_id",
                "doc_id",
                "chunk_id",
                "filename",
                "source_turn_id",
                "trace_id",
            }:
                return None
            facts.append(
                ResolvedFact(
                    mention=str(raw["mention"]),
                    attribute=str(raw["attribute"]),
                    value=str(raw["value"]),
                    kb_id=uuid.UUID(str(raw["kb_id"])),
                    doc_id=uuid.UUID(str(raw["doc_id"])),
                    chunk_id=uuid.UUID(str(raw["chunk_id"])),
                    filename=str(raw["filename"]),
                    source_turn_id=uuid.UUID(str(raw["source_turn_id"])),
                    trace_id=str(raw["trace_id"]),
                )
            )
        source_chunk_ids = tuple(
            uuid.UUID(str(raw_id)) for raw_id in raw_sources
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    try:
        return ResolvedEntityMemory(
            facts=tuple(facts),
            source_chunk_ids=source_chunk_ids,
        )
    except ValueError:
        return None


def has_entity_reuse(question: object, memory: ResolvedEntityMemory) -> bool:
    """Return whether the current question re-mentions a resolved entity."""

    if not isinstance(memory, ResolvedEntityMemory):
        return False
    normalized = (question or "").strip().casefold()
    if not normalized:
        return False
    return any(fact.mention.casefold() in normalized for fact in memory.facts)


def extract_resolved_entity_memory(
    *,
    sources: Iterable[Mapping[str, Any]],
    question: object,
    source_turn_id: uuid.UUID,
    trace_id: str,
) -> ResolvedEntityMemory | None:
    """Extract bounded entity facts from a grounded answer's evidence.

    Only markdown table rows whose cells literally occur in ``question`` are
    recorded.  A row binds its leading category cell to each remaining column
    (``D级 → 国内航班 = 经济舱``), and any mention appearing in a non-leading
    column binds back to the category (``普通员工 → 职级 = D级``).
    """

    normalized_question = (question or "").strip()
    if not normalized_question:
        return None
    if not isinstance(source_turn_id, uuid.UUID):
        raise ValueError("source turn id must be a UUID")
    source_chunk_ids: list[uuid.UUID] = []
    seen_chunks: set[uuid.UUID] = set()
    facts: list[ResolvedFact] = []
    seen_facts: set[tuple[str, str, str, str]] = set()

    def add_fact(
        mention: str,
        attribute: str,
        value: str,
        kb_id: uuid.UUID,
        doc_id: uuid.UUID,
        chunk_id: uuid.UUID,
        filename: str,
    ) -> None:
        key = (
            mention.casefold(),
            attribute.casefold(),
            value.casefold(),
            str(chunk_id),
        )
        if key in seen_facts or len(facts) >= MAX_FACTS:
            return
        seen_facts.add(key)
        facts.append(
            ResolvedFact(
                mention=mention,
                attribute=attribute,
                value=value,
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_id=chunk_id,
                filename=filename,
                source_turn_id=source_turn_id,
                trace_id=trace_id,
            )
        )

    for raw_source in sources:
        if not isinstance(raw_source, Mapping):
            continue
        try:
            kb_id = uuid.UUID(str(raw_source.get("kb_id") or ""))
            doc_id = uuid.UUID(str(raw_source.get("doc_id") or ""))
            chunk_id = uuid.UUID(
                str(raw_source.get("id") or raw_source.get("chunk_id") or "")
            )
        except (TypeError, ValueError, AttributeError):
            continue
        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            source_chunk_ids.append(chunk_id)
            if len(source_chunk_ids) >= MAX_FACT_SOURCES:
                break
        content = raw_source.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        filename = str(raw_source.get("filename") or "").strip()
        headers, data_rows = _parse_table_rows(content)
        if not data_rows:
            continue
        # Two question shapes are deliberately distinguished.  When the user
        # names a row category (``D级可以坐头等舱吗``), only that category's
        # column bindings are extracted.  When the user names a member value
        # instead (``普通员工出差交通``), the member binds back to the row
        # category (``普通员工 → 职级 = D级``).  Mixing both would turn shared
        # member values such as ``经济舱`` into noisy reverse bindings.
        category_matched = any(
            bool(_bound_mention(normalized_question, row[0]))
            for row in data_rows
        )
        for cells in data_rows:
            if not cells:
                continue
            category = cells[0]
            if category_matched:
                mentions = _bound_mention(normalized_question, category)
                if not mentions:
                    continue
                for column_index in range(1, len(cells)):
                    if column_index >= len(headers):
                        continue
                    attribute = headers[column_index]
                    if not attribute:
                        continue
                    for mention in mentions:
                        add_fact(
                            mention,
                            attribute,
                            cells[column_index],
                            kb_id,
                            doc_id,
                            chunk_id,
                            filename,
                        )
                continue
            if not headers:
                continue
            for index in range(1, len(cells)):
                cell = cells[index]
                if len(cell) > _MAX_CELL_CHARS:
                    continue
                mentions = _bound_mention(normalized_question, cell)
                if not mentions:
                    continue
                # Member-value mention: bind back to the row category under
                # the leading header.
                for mention in mentions:
                    add_fact(
                        mention,
                        headers[0],
                        category,
                        kb_id,
                        doc_id,
                        chunk_id,
                        filename,
                    )
    if not facts:
        return None
    return ResolvedEntityMemory(
        facts=tuple(facts),
        source_chunk_ids=tuple(source_chunk_ids),
    )


__all__ = [
    "MAX_FACTS",
    "MAX_FACT_SOURCES",
    "ResolvedEntityMemory",
    "ResolvedFact",
    "SEMANTIC_MEMORY_SCHEMA_VERSION",
    "extract_resolved_entity_memory",
    "has_entity_reuse",
    "parse_resolved_entity_memory",
]
