"""Deterministic snapshot proofs over immutable evidence items.

These helpers deliberately inspect only parser/retrieval structural metadata:
trusted full-document expansion origins plus an exact chunk index/cardinality,
or an exact table part index/cardinality.  They never accept a caller supplied
"complete" flag, score, or document title as a proof of exhaustiveness.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


DocumentKey = tuple[str, str]


def document_key(item: Any) -> DocumentKey:
    return (str(item.kb_id), str(item.doc_id))


def table_key(item: Any) -> str | None:
    metadata = getattr(item, "metadata", {})
    table_id = str(metadata.get("table_id") or "").strip()
    if not table_id:
        return None
    kb_id, doc_id = document_key(item)
    return f"table:{kb_id}:{doc_id}:{table_id}"


def complete_document_keys(
    items: Sequence[Any],
    *,
    visible_item_ids: Iterable[str] = (),
    require_visible: bool,
) -> tuple[DocumentKey, ...]:
    """Return verified complete source documents in stable first-seen order."""

    visible = frozenset(str(value) for value in visible_item_ids)
    expanded_documents = {
        document_key(item)
        for item in items
        if "small_document_full" in tuple(getattr(item, "origins", ()) or ())
        or "overview_full_document" in tuple(getattr(item, "origins", ()) or ())
    }
    by_document: dict[DocumentKey, list[Any]] = defaultdict(list)
    first_seen: list[DocumentKey] = []
    for item in items:
        key = document_key(item)
        metadata = getattr(item, "metadata", {})
        if key not in expanded_documents or metadata.get("full_document_chunk_count") is None:
            continue
        if key not in by_document:
            first_seen.append(key)
        by_document[key].append(item)

    complete: list[DocumentKey] = []
    for key in first_seen:
        snapshot_items = by_document[key]
        expected_counts: set[int] = set()
        indexes: set[int] = set()
        valid = True
        for item in snapshot_items:
            metadata = getattr(item, "metadata", {})
            try:
                expected = int(metadata.get("full_document_chunk_count"))
                index = int(getattr(item, "chunk_index"))
            except (TypeError, ValueError):
                valid = False
                break
            if expected <= 0 or index < 0 or index >= expected:
                valid = False
                break
            expected_counts.add(expected)
            indexes.add(index)
        expected = next(iter(expected_counts), 0)
        if (
            not valid
            or len(expected_counts) != 1
            or len(snapshot_items) != expected
            or indexes != set(range(expected))
            or (
                require_visible
                and any(str(item.chunk_id) not in visible for item in snapshot_items)
            )
        ):
            continue
        complete.append(key)
    return tuple(complete)


def complete_table_keys(
    items: Sequence[Any],
    *,
    visible_item_ids: Iterable[str] = (),
    require_visible: bool,
) -> tuple[str, ...]:
    """Return parser-identified tables whose declared parts are all present."""

    visible = frozenset(str(value) for value in visible_item_ids)
    by_table: dict[str, list[Any]] = defaultdict(list)
    first_seen: list[str] = []
    for item in items:
        key = table_key(item)
        if key is None:
            continue
        if key not in by_table:
            first_seen.append(key)
        by_table[key].append(item)

    complete: list[str] = []
    for key in first_seen:
        table_items = by_table[key]
        expected_counts: set[int] = set()
        indexes: set[int] = set()
        valid = True
        for item in table_items:
            metadata = getattr(item, "metadata", {})
            try:
                expected = int(metadata.get("table_part_count"))
                index = int(metadata.get("table_part_index"))
            except (TypeError, ValueError):
                valid = False
                break
            if expected <= 0 or index < 0 or index >= expected:
                valid = False
                break
            expected_counts.add(expected)
            indexes.add(index)
        expected = next(iter(expected_counts), 0)
        if (
            not valid
            or len(expected_counts) != 1
            or len(table_items) != expected
            or indexes != set(range(expected))
            or (
                require_visible
                and any(str(item.chunk_id) not in visible for item in table_items)
            )
        ):
            continue
        complete.append(key)
    return tuple(complete)


__all__ = [
    "DocumentKey",
    "complete_document_keys",
    "complete_table_keys",
    "document_key",
    "table_key",
]
