"""Durable, authorization-neutral state for grounded conversation tasks.

The state remembers *what* a successfully grounded turn was about and which
source identities closed it.  It never remembers an execution grant.  Every
continuation resolves those identities again under the current request's KB
selection and the current document/chunk rows before the state can influence
retrieval.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.read_sessions import ReadSessionFactory, isolated_read_session
from models.db_models import Document, DocumentChunk


ACTIVE_TASK_STATE_SCHEMA_VERSION = "rag_active_task.v1"
ACTIVE_TASK_STATE_STATUS = "active"
MAX_ACTIVE_TASK_QUERY_CHARS = 8000
MAX_ACTIVE_TASK_SOURCES = 20
MAX_ACTIVE_TASK_DOCUMENTS = 12
MAX_ACTIVE_TASK_KBS = 12
_ANSWER_SHAPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
_TRACE_ID_RE = re.compile(r"^[a-fA-F0-9]{16,64}$")


def _uuid_values(
    values: object,
    *,
    field: str,
    limit: int,
    required: bool = True,
) -> tuple[uuid.UUID, ...]:
    if not isinstance(values, list) or len(values) > limit:
        raise ValueError(f"{field} must be a bounded list")
    result: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for raw in values:
        try:
            item = uuid.UUID(str(raw))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"{field} contains an invalid UUID") from exc
        if item in seen:
            raise ValueError(f"{field} contains duplicate UUIDs")
        seen.add(item)
        result.append(item)
    if required and not result:
        raise ValueError(f"{field} must not be empty")
    return tuple(result)


def _bounded_text(value: object, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > limit:
        raise ValueError(f"{field} is empty or too long")
    return normalized


@dataclass(frozen=True)
class ActiveTaskState:
    state_id: uuid.UUID
    revision: int
    root_query: str
    answer_shape: str
    selected_kb_ids: tuple[uuid.UUID, ...]
    selected_doc_ids: tuple[uuid.UUID, ...]
    selected_chunk_ids: tuple[uuid.UUID, ...]
    source_turn_id: uuid.UUID
    trace_id: str
    created_at: datetime
    status: str = ACTIVE_TASK_STATE_STATUS
    schema_version: str = ACTIVE_TASK_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ACTIVE_TASK_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported active-task schema")
        if self.status != ACTIVE_TASK_STATE_STATUS:
            raise ValueError("active-task state is not active")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("active-task revision must be positive")
        root_query = _bounded_text(
            self.root_query,
            field="root_query",
            limit=MAX_ACTIVE_TASK_QUERY_CHARS,
        )
        answer_shape = _bounded_text(
            self.answer_shape,
            field="answer_shape",
            limit=48,
        ).casefold()
        if not _ANSWER_SHAPE_RE.fullmatch(answer_shape):
            raise ValueError("active-task answer_shape is invalid")
        trace_id = _bounded_text(self.trace_id, field="trace_id", limit=64)
        if not _TRACE_ID_RE.fullmatch(trace_id):
            raise ValueError("active-task trace_id is invalid")
        created_at = self.created_at
        if not isinstance(created_at, datetime) or created_at.tzinfo is None:
            raise ValueError("active-task created_at must be timezone-aware")
        if created_at > datetime.now(timezone.utc):
            raise ValueError("active-task created_at is in the future")
        if not self.selected_kb_ids or len(self.selected_kb_ids) > MAX_ACTIVE_TASK_KBS:
            raise ValueError("active-task KB scope is invalid")
        if not self.selected_doc_ids or len(self.selected_doc_ids) > MAX_ACTIVE_TASK_DOCUMENTS:
            raise ValueError("active-task document scope is invalid")
        if not self.selected_chunk_ids or len(self.selected_chunk_ids) > MAX_ACTIVE_TASK_SOURCES:
            raise ValueError("active-task source scope is invalid")
        for values, field in (
            (self.selected_kb_ids, "selected_kb_ids"),
            (self.selected_doc_ids, "selected_doc_ids"),
            (self.selected_chunk_ids, "selected_chunk_ids"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"active-task {field} contains duplicates")
        object.__setattr__(self, "root_query", root_query)
        object.__setattr__(self, "answer_shape", answer_shape)
        object.__setattr__(self, "trace_id", trace_id.casefold())
        object.__setattr__(self, "created_at", created_at.astimezone(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_id": str(self.state_id),
            "revision": self.revision,
            "root_query": self.root_query,
            "answer_shape": self.answer_shape,
            "selected_kb_ids": [str(item) for item in self.selected_kb_ids],
            "selected_doc_ids": [str(item) for item in self.selected_doc_ids],
            "selected_chunk_ids": [str(item) for item in self.selected_chunk_ids],
            "source_turn_id": str(self.source_turn_id),
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_id": str(self.state_id),
            "revision": self.revision,
            "answer_shape": self.answer_shape,
            "kb_count": len(self.selected_kb_ids),
            "document_count": len(self.selected_doc_ids),
            "source_count": len(self.selected_chunk_ids),
            "status": self.status,
        }


def parse_active_task_state(value: object) -> ActiveTaskState | None:
    """Strictly parse persisted JSON; malformed/stale shapes fail closed."""

    if not isinstance(value, Mapping):
        return None
    expected_fields = {
        "schema_version",
        "state_id",
        "revision",
        "root_query",
        "answer_shape",
        "selected_kb_ids",
        "selected_doc_ids",
        "selected_chunk_ids",
        "source_turn_id",
        "trace_id",
        "created_at",
        "status",
    }
    if set(value) != expected_fields:
        return None
    try:
        revision = value["revision"]
        if isinstance(revision, bool):
            return None
        created_at = datetime.fromisoformat(str(value["created_at"]))
        return ActiveTaskState(
            schema_version=str(value["schema_version"]),
            state_id=uuid.UUID(str(value["state_id"])),
            revision=int(revision),
            root_query=str(value["root_query"]),
            answer_shape=str(value["answer_shape"]),
            selected_kb_ids=_uuid_values(
                value["selected_kb_ids"],
                field="selected_kb_ids",
                limit=MAX_ACTIVE_TASK_KBS,
            ),
            selected_doc_ids=_uuid_values(
                value["selected_doc_ids"],
                field="selected_doc_ids",
                limit=MAX_ACTIVE_TASK_DOCUMENTS,
            ),
            selected_chunk_ids=_uuid_values(
                value["selected_chunk_ids"],
                field="selected_chunk_ids",
                limit=MAX_ACTIVE_TASK_SOURCES,
            ),
            source_turn_id=uuid.UUID(str(value["source_turn_id"])),
            trace_id=str(value["trace_id"]),
            created_at=created_at,
            status=str(value["status"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def build_active_task_state(
    *,
    root_query: str,
    answer_shape: str,
    sources: Iterable[Mapping[str, Any]],
    source_turn_id: uuid.UUID,
    trace_id: str,
    previous_revision: int = 0,
) -> ActiveTaskState:
    """Create state only from already validated answer-source identities."""

    kb_ids: list[uuid.UUID] = []
    doc_ids: list[uuid.UUID] = []
    chunk_ids: list[uuid.UUID] = []
    seen_kbs: set[uuid.UUID] = set()
    seen_docs: set[uuid.UUID] = set()
    seen_chunks: set[uuid.UUID] = set()
    for raw in sources:
        if not isinstance(raw, Mapping):
            continue
        try:
            kb_id = uuid.UUID(str(raw.get("kb_id")))
            doc_id = uuid.UUID(str(raw.get("doc_id")))
            chunk_id = uuid.UUID(str(raw.get("id") or raw.get("chunk_id")))
        except (TypeError, ValueError, AttributeError):
            continue
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        chunk_ids.append(chunk_id)
        if kb_id not in seen_kbs:
            seen_kbs.add(kb_id)
            kb_ids.append(kb_id)
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            doc_ids.append(doc_id)
        if len(chunk_ids) >= MAX_ACTIVE_TASK_SOURCES:
            break
    return ActiveTaskState(
        state_id=uuid.uuid4(),
        revision=max(0, int(previous_revision)) + 1,
        root_query=root_query,
        answer_shape=answer_shape,
        selected_kb_ids=tuple(kb_ids),
        selected_doc_ids=tuple(doc_ids),
        selected_chunk_ids=tuple(chunk_ids),
        source_turn_id=source_turn_id,
        trace_id=trace_id,
        created_at=datetime.now(timezone.utc),
    )


@dataclass(frozen=True)
class ResolvedActiveTask:
    state: ActiveTaskState
    sources: tuple[dict[str, Any], ...]
    kb_ids: tuple[uuid.UUID, ...]
    doc_ids: tuple[uuid.UUID, ...]

    def __post_init__(self) -> None:
        if not self.sources or not self.kb_ids or not self.doc_ids:
            raise ValueError("resolved active task requires authorized sources")

    def safe_summary(self) -> dict[str, Any]:
        return {
            **self.state.safe_summary(),
            "resolved_kb_count": len(self.kb_ids),
            "resolved_document_count": len(self.doc_ids),
            "resolved_source_count": len(self.sources),
        }


async def resolve_active_task_state(
    db: AsyncSession,
    *,
    value: object,
    selected_kb_ids: Iterable[uuid.UUID],
    read_session_factory: ReadSessionFactory | None = None,
) -> ResolvedActiveTask | None:
    """Re-authorize a persisted task against current KB/document/chunk state."""

    state = parse_active_task_state(value)
    if state is None:
        return None
    selected = tuple(dict.fromkeys(selected_kb_ids))
    allowed_kbs = tuple(item for item in state.selected_kb_ids if item in selected)
    if not allowed_kbs:
        return None
    statement = (
        select(DocumentChunk, Document)
        .join(
            Document,
            (Document.id == DocumentChunk.doc_id)
            & (Document.kb_id == DocumentChunk.kb_id),
        )
        .where(
            DocumentChunk.id.in_(state.selected_chunk_ids),
            DocumentChunk.doc_id.in_(state.selected_doc_ids),
            DocumentChunk.kb_id.in_(allowed_kbs),
            Document.is_active.is_(True),
            Document.status == "ready",
        )
    )
    try:
        async with isolated_read_session(
            request_db=db,
            session_factory=read_session_factory,
        ) as read_db:
            rows = (await read_db.execute(statement)).all()
            # ORM instances are owned by this short-lived session.  Project
            # every scalar while the transaction is still open: the owned
            # rollback intentionally expires ORM state, so no mapped object may
            # cross this boundary.
            resolved_rows = tuple(
                {
                    "id": chunk.id,
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "kb_id": chunk.kb_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "metadata": dict(chunk.metadata_ or {}),
                    "filename": document.filename,
                    "file_type": document.file_type,
                    "source_url": document.source_url,
                    "doc_tags": list(document.tags or []),
                }
                for chunk, document in rows
            )
    except Exception:
        return None
    by_chunk: dict[uuid.UUID, dict[str, Any]] = {}
    resolved_docs: list[uuid.UUID] = []
    resolved_kbs: list[uuid.UUID] = []
    for row in resolved_rows:
        chunk_id = row["id"]
        doc_id = row["doc_id"]
        kb_id = row["kb_id"]
        if doc_id not in state.selected_doc_ids or kb_id not in allowed_kbs:
            continue
        by_chunk[chunk_id] = {
            **row,
            "retrieval_score": 0.0,
            "score": 0.0,
            "candidate_origin": "active_task_state",
        }
        if doc_id not in resolved_docs:
            resolved_docs.append(doc_id)
        if kb_id not in resolved_kbs:
            resolved_kbs.append(kb_id)
    sources = tuple(
        by_chunk[item]
        for item in state.selected_chunk_ids
        if item in by_chunk
    )
    if not sources:
        return None
    return ResolvedActiveTask(
        state=state,
        sources=sources,
        kb_ids=tuple(resolved_kbs),
        doc_ids=tuple(resolved_docs),
    )


__all__ = [
    "ACTIVE_TASK_STATE_SCHEMA_VERSION",
    "ActiveTaskState",
    "ResolvedActiveTask",
    "build_active_task_state",
    "parse_active_task_state",
    "resolve_active_task_state",
]
