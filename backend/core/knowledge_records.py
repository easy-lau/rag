"""Generic structured-record extraction and lookup.

The extractor operates on Markdown structure rather than business keywords.
Every record remains bound to its original document chunk, so ACL, lifecycle,
version and citation checks continue to use the established source boundary.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Mapping, Sequence

from sqlalchemy import UUID as SA_UUID
from sqlalchemy import bindparam, delete, select, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import Document, DocumentChunk, KnowledgeRecord


_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
STRUCTURED_RECORD_MIN_SCORE = 0.18


def _table_cells(line: str) -> list[str] | None:
    stripped = str(line or "").strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if len(cells) < 2 or all(not cell for cell in cells):
        return None
    if all(_TABLE_SEPARATOR_RE.fullmatch(cell.replace(" ", "")) for cell in cells):
        return None
    return cells


def extract_knowledge_records(content: str) -> list[dict[str, Any]]:
    """Extract data rows from Markdown tables without domain assumptions."""

    rows: list[list[str]] = []
    for line in str(content or "").splitlines():
        cells = _table_cells(line)
        if cells is not None:
            rows.append(cells)
    if len(rows) < 2:
        return []
    headers = rows[0]
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for cells in rows[1:]:
        if len(cells) != len(headers) or tuple(cells) in seen:
            continue
        seen.add(tuple(cells))
        if all(not cell for cell in cells):
            continue
        subject = cells[0]
        object_value = " | ".join(cells[1:])
        if not subject or not object_value:
            continue
        output.append({
            "record_type": "table_row",
            "subject": subject,
            "predicate": headers[1] if len(headers) == 2 else "table_row",
            "object_value": object_value,
            "search_text": " ".join(cells),
            "metadata": {
                "headers": headers,
                "cells": cells,
            },
        })
    return output


async def search_knowledge_records(
    db: AsyncSession,
    query: str,
    kb_ids: Sequence[uuid.UUID],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search ready/active structured records and return chunk-bound candidates."""

    normalized_query = str(query or "").strip()
    scoped_kbs = list(dict.fromkeys(kb_ids))
    if not normalized_query or not scoped_kbs or top_k <= 0:
        return []
    sql = text(
        """
        SELECT
            kr.id,
            kr.id AS record_id,
            kr.chunk_id,
            kr.doc_id,
            kr.kb_id,
            d.filename,
            d.file_type,
            d.source_url,
            d.tags AS doc_tags,
            dc.chunk_index,
            dc.metadata,
            kr.search_text AS content,
            kr.subject,
            kr.predicate,
            kr.object_value,
            kr.record_type,
            kr.metadata AS record_metadata,
            'knowledge_record' AS source_kind,
            GREATEST(
                word_similarity(:query, kr.search_text),
                word_similarity(:query, kr.subject)
            ) AS structured_score,
            GREATEST(
                word_similarity(:query, kr.search_text),
                word_similarity(:query, kr.subject)
            ) AS retrieval_score,
            GREATEST(
                word_similarity(:query, kr.search_text),
                word_similarity(:query, kr.subject)
            ) AS score
        FROM knowledge_records kr
        JOIN documents d
          ON d.id = kr.doc_id
         AND d.kb_id = kr.kb_id
        JOIN document_chunks dc
          ON dc.id = kr.chunk_id
         AND dc.doc_id = kr.doc_id
         AND dc.kb_id = kr.kb_id
        WHERE kr.kb_id = ANY(:kb_ids)
          AND d.is_active = TRUE
          AND d.status = 'ready'
          AND (
              to_tsvector('simple', kr.search_text)
                  @@ plainto_tsquery('simple', :query)
              OR word_similarity(:query, kr.search_text) >= :min_score
              OR word_similarity(:query, kr.subject) >= :min_score
          )
        ORDER BY structured_score DESC, kr.doc_id, kr.chunk_id, kr.id
        LIMIT :top_k
        """
    ).bindparams(bindparam("kb_ids", type_=ARRAY(SA_UUID())))
    rows = (
        await db.execute(
            sql,
            {
                "query": normalized_query,
                "kb_ids": scoped_kbs,
                "min_score": STRUCTURED_RECORD_MIN_SCORE,
                "top_k": min(top_k, 50),
            },
        )
    ).mappings().all()
    return [dict(row) for row in rows]


async def search_knowledge_records_for_chunks(
    db: AsyncSession,
    query: str,
    chunk_ids: Sequence[uuid.UUID],
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Retrieve the strongest records inside already-selected chunks.

    This is the second stage of retrieval-first execution.  Document retrieval
    and semantic adjudication must select the chunk before this function is
    called, so a fact lookup never expands every row from every initially
    retrieved table into one model request.
    """

    normalized_query = str(query or "").strip()
    scoped_chunks = list(dict.fromkeys(chunk_ids))
    if not normalized_query or not scoped_chunks or top_k <= 0:
        return []
    sql = text(
        """
        SELECT
            kr.id,
            kr.id AS record_id,
            kr.chunk_id,
            kr.doc_id,
            kr.kb_id,
            d.filename,
            d.file_type,
            d.source_url,
            d.tags AS doc_tags,
            dc.chunk_index,
            dc.metadata,
            kr.search_text AS content,
            kr.subject,
            kr.predicate,
            kr.object_value,
            kr.record_type,
            kr.metadata AS record_metadata,
            'knowledge_record' AS source_kind,
            GREATEST(
                word_similarity(:query, kr.search_text),
                word_similarity(:query, kr.subject)
            ) AS structured_score,
            GREATEST(
                word_similarity(:query, kr.search_text),
                word_similarity(:query, kr.subject)
            ) AS retrieval_score,
            GREATEST(
                word_similarity(:query, kr.search_text),
                word_similarity(:query, kr.subject)
            ) AS score
        FROM knowledge_records kr
        JOIN documents d
          ON d.id = kr.doc_id
         AND d.kb_id = kr.kb_id
        JOIN document_chunks dc
          ON dc.id = kr.chunk_id
         AND dc.doc_id = kr.doc_id
         AND dc.kb_id = kr.kb_id
        WHERE kr.chunk_id = ANY(:chunk_ids)
          AND d.is_active = TRUE
          AND d.status = 'ready'
        ORDER BY structured_score DESC, kr.doc_id, kr.chunk_id, kr.id
        LIMIT :top_k
        """
    ).bindparams(bindparam("chunk_ids", type_=ARRAY(SA_UUID())))
    rows = (
        await db.execute(
            sql,
            {
                "query": normalized_query,
                "chunk_ids": scoped_chunks,
                "top_k": min(top_k, 20),
            },
        )
    ).mappings().all()
    return [dict(row) for row in rows]


async def rebuild_knowledge_records(
    db: AsyncSession,
    *,
    kb_ids: Sequence[uuid.UUID] | None = None,
) -> int:
    """Rebuild structured records for existing ready/active chunks."""

    statement = (
        select(DocumentChunk)
        .join(
            Document,
            (Document.id == DocumentChunk.doc_id)
            & (Document.kb_id == DocumentChunk.kb_id),
        )
        .where(Document.is_active.is_(True), Document.status == "ready")
        .order_by(DocumentChunk.doc_id, DocumentChunk.chunk_index, DocumentChunk.id)
    )
    scoped_kbs = list(dict.fromkeys(kb_ids or ()))
    if scoped_kbs:
        statement = statement.where(DocumentChunk.kb_id.in_(scoped_kbs))
        await db.execute(delete(KnowledgeRecord).where(KnowledgeRecord.kb_id.in_(scoped_kbs)))
    else:
        await db.execute(delete(KnowledgeRecord))
    chunks = (await db.execute(statement)).scalars().all()
    records: list[KnowledgeRecord] = []
    for chunk in chunks:
        for record in extract_knowledge_records(chunk.content):
            records.append(KnowledgeRecord(
                kb_id=chunk.kb_id,
                doc_id=chunk.doc_id,
                chunk_id=chunk.id,
                record_type=str(record["record_type"]),
                subject=str(record["subject"]),
                predicate=(
                    str(record["predicate"])
                    if record.get("predicate") is not None
                    else None
                ),
                object_value=str(record["object_value"]),
                search_text=str(record["search_text"]),
                metadata_=dict(record.get("metadata") or {}),
            ))
    db.add_all(records)
    await db.flush()
    return len(records)
