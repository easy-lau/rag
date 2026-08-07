"""Durable, authorization-neutral memory of numbered result lists.

A catalog answer (for example ``我现在知识库里面有多少文章``) presents an
ordered list of documents.  A later turn such as ``我想看第四个`` or
``第四个不是《钉钉》吗`` refers back to that exact list.  This module persists
the numbered list the user actually saw (index -> document identity), so the
next ordinal reference resolves directly against it instead of re-running
retrieval or re-interpreting the number through a model.

The memory never grants authorization: every resolution re-checks the item
against the current request KB selection and the current document state, the
same way ``active_task_state`` re-authorizes chunk identities.
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
from core.result_reference import (
    ResultReferenceSurface,
    is_result_list_reference,
    parse_result_reference_surface,
)
from models.db_models import Document


RESULT_REFERENCE_MEMORY_SCHEMA_VERSION = "rag_result_reference_memory.v1"
MAX_RESULT_MEMORY_ITEMS = 20
MAX_RESULT_MEMORY_QUERY_CHARS = 8000
MAX_RESULT_MEMORY_LABEL_CHARS = 120
MAX_RESULT_MEMORY_FILENAME_CHARS = 300
_TRACE_ID_RE = re.compile(r"^[a-fA-F0-9]{16,64}$")

# 语言结构规则，不是业务知识：识别“第 N 个不是《X》吗 / 你刚才说错了 /
# 应该是第 N 个”这类对前面列表结果的纠正。是否真的引用列表仍由序数
# 解析器判定；这里只决定回答是否需要先确认/道歉。
_CORRECTION_RE = re.compile(
    r"(?:"
    r"(?:第[0-9一二三四五六七八九十两]+(?:个|篇|份|条)?)[^。！？!?]{0,24}?"
    r"(?:不是|难道不是)|"
    r"不是[^。！？!?]{0,20}(?:第[0-9一二三四五六七八九十两]+)(?:个|篇|份|条)?|"
    r"(?:你|您|系统)(?:刚才|刚刚)?(?:说|答|看|返回|给|发|标)"
    r"(?:错|错了|反了|成别的)|"
    r"(?:你|您|系统)?(?:搞错|弄错|记错|看错)|"
    r"应该是|才是|才对|纠正|更正|修正"
    r")",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"《([^》]{1,60})》")
_RESULT_MEMORY_STATE_KEYS = {
    "schema_version",
    "state_id",
    "revision",
    "root_query",
    "list_label",
    "source_turn_id",
    "trace_id",
    "created_at",
    "items",
}
_RESULT_MEMORY_ITEM_KEYS = {"index", "kb_id", "doc_id", "filename", "status"}


def _bounded_text(value: object, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > limit:
        raise ValueError(f"{field} is empty or too long")
    return normalized


def _uuid_value(value: object, *, field: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field} is not a UUID") from exc


@dataclass(frozen=True)
class ResultReferenceMemoryItem:
    index: int
    kb_id: uuid.UUID
    doc_id: uuid.UUID
    filename: str
    status: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ValueError("result memory item index must be an integer")
        if self.index < 1:
            raise ValueError("result memory item index must be positive")
        filename = _bounded_text(
            self.filename,
            field="filename",
            limit=MAX_RESULT_MEMORY_FILENAME_CHARS,
        )
        status = str(self.status or "").strip()[:32] or None
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kb_id": str(self.kb_id),
            "doc_id": str(self.doc_id),
            "filename": self.filename,
            "status": self.status,
        }


@dataclass(frozen=True)
class ResultReferenceMemory:
    state_id: uuid.UUID
    revision: int
    root_query: str
    list_label: str
    source_turn_id: uuid.UUID
    trace_id: str
    created_at: datetime
    items: tuple[ResultReferenceMemoryItem, ...]

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("result memory revision must be positive")
        root_query = _bounded_text(
            self.root_query,
            field="root_query",
            limit=MAX_RESULT_MEMORY_QUERY_CHARS,
        )
        list_label = _bounded_text(
            self.list_label,
            field="list_label",
            limit=MAX_RESULT_MEMORY_LABEL_CHARS,
        )
        trace_id = _bounded_text(self.trace_id, field="trace_id", limit=64)
        if not _TRACE_ID_RE.fullmatch(trace_id):
            raise ValueError("result memory trace_id is invalid")
        created_at = self.created_at
        if not isinstance(created_at, datetime) or created_at.tzinfo is None:
            raise ValueError("result memory created_at must be timezone-aware")
        if created_at > datetime.now(timezone.utc):
            raise ValueError("result memory created_at is in the future")
        items = tuple(self.items)
        if not items or len(items) > MAX_RESULT_MEMORY_ITEMS:
            raise ValueError("result memory item count is invalid")
        if tuple(item.index for item in items) != tuple(
            range(1, len(items) + 1)
        ):
            raise ValueError("result memory item indexes must be sequential")
        identities = [(item.kb_id, item.doc_id) for item in items]
        if len(identities) != len(set(identities)):
            raise ValueError("result memory contains duplicate documents")
        object.__setattr__(self, "root_query", root_query)
        object.__setattr__(self, "list_label", list_label)
        object.__setattr__(self, "trace_id", trace_id.casefold())
        object.__setattr__(self, "created_at", created_at.astimezone(timezone.utc))
        object.__setattr__(self, "items", items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_REFERENCE_MEMORY_SCHEMA_VERSION,
            "state_id": str(self.state_id),
            "revision": self.revision,
            "root_query": self.root_query,
            "list_label": self.list_label,
            "source_turn_id": str(self.source_turn_id),
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat(),
            "items": [item.to_dict() for item in self.items],
        }

    def safe_summary(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_REFERENCE_MEMORY_SCHEMA_VERSION,
            "state_id": str(self.state_id),
            "revision": self.revision,
            "root_query": self.root_query,
            "list_label": self.list_label,
            "item_count": len(self.items),
            "first_item": self.items[0].filename,
            "last_item": self.items[-1].filename,
        }

    def item_for_surface(self, surface: ResultReferenceSurface) -> ResultReferenceMemoryItem | None:
        """Return one item for an ordinal/last reference, or ``None``."""

        if surface.kind == "last":
            return self.items[-1]
        if surface.kind != "ordinal" or surface.value is None:
            return None
        if surface.value < 1 or surface.value > len(self.items):
            return None
        return self.items[surface.value - 1]


def parse_result_reference_memory(value: object) -> ResultReferenceMemory | None:
    """Strictly parse persisted JSON; malformed or stale shapes fail closed."""

    if not isinstance(value, Mapping):
        return None
    if str(value.get("schema_version") or "") != RESULT_REFERENCE_MEMORY_SCHEMA_VERSION:
        return None
    if set(value) != _RESULT_MEMORY_STATE_KEYS:
        return None
    raw_items = value.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        return None
    items: list[ResultReferenceMemoryItem] = []
    try:
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, Mapping):
                return None
            if set(raw_item) != _RESULT_MEMORY_ITEM_KEYS:
                return None
            if raw_item.get("index") != index:
                return None
            items.append(ResultReferenceMemoryItem(
                index=int(raw_item["index"]),
                kb_id=_uuid_value(raw_item["kb_id"], field="items.kb_id"),
                doc_id=_uuid_value(raw_item["doc_id"], field="items.doc_id"),
                filename=str(raw_item["filename"]),
                status=(
                    str(raw_item["status"]).strip()[:32]
                    if raw_item.get("status") is not None
                    else None
                ),
            ))
        created_at = datetime.fromisoformat(str(value["created_at"]))
        return ResultReferenceMemory(
            state_id=_uuid_value(value["state_id"], field="state_id"),
            revision=int(value["revision"]),
            root_query=str(value["root_query"]),
            list_label=str(value["list_label"]),
            source_turn_id=_uuid_value(
                value["source_turn_id"],
                field="source_turn_id",
            ),
            trace_id=str(value["trace_id"]),
            created_at=created_at,
            items=tuple(items),
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def build_result_reference_memory(
    *,
    root_query: str,
    list_label: str,
    items: Iterable[Mapping[str, Any]],
    source_turn_id: uuid.UUID,
    trace_id: str,
    previous_revision: int = 0,
) -> ResultReferenceMemory:
    """Create memory only from an ordered, already authorized result list."""

    normalized: list[ResultReferenceMemoryItem] = []
    for index, raw in enumerate(items, start=1):
        if not isinstance(raw, Mapping):
            continue
        try:
            kb_id = uuid.UUID(str(raw.get("kb_id")))
            doc_id = uuid.UUID(str(raw.get("doc_id") or raw.get("id")))
        except (TypeError, ValueError, AttributeError):
            continue
        filename = str(raw.get("filename") or "").strip()
        if not filename or not kb_id or not doc_id:
            continue
        normalized.append(ResultReferenceMemoryItem(
            index=index,
            kb_id=kb_id,
            doc_id=doc_id,
            filename=filename[:MAX_RESULT_MEMORY_FILENAME_CHARS],
            status=(
                str(raw.get("status") or "").strip()[:32]
                if raw.get("status") is not None
                else None
            ),
        ))
        if len(normalized) >= MAX_RESULT_MEMORY_ITEMS:
            break
    if not normalized:
        raise ValueError("result reference memory requires at least one item")
    return ResultReferenceMemory(
        state_id=uuid.uuid4(),
        revision=max(0, int(previous_revision)) + 1,
        root_query=_bounded_text(
            root_query,
            field="root_query",
            limit=MAX_RESULT_MEMORY_QUERY_CHARS,
        ),
        list_label=_bounded_text(
            list_label,
            field="list_label",
            limit=MAX_RESULT_MEMORY_LABEL_CHARS,
        ),
        source_turn_id=source_turn_id,
        trace_id=str(trace_id or ""),
        created_at=datetime.now(timezone.utc),
        items=tuple(normalized),
    )


def is_reference_correction(question: object) -> bool:
    """Whether the question corrects a previously presented result list."""

    text = str(question or "").strip()
    if not text:
        return False
    return bool(_CORRECTION_RE.search(text))


def _surface_label(surface: ResultReferenceSurface, item: ResultReferenceMemoryItem) -> str:
    """Return the label used in acknowledgements.

    Echo the user's own surface span (``第四个``) so the correction reads
    naturally; fall back to a normalized label for ``last`` references.
    """

    if surface.kind == "ordinal" and surface.span:
        return surface.span
    if surface.kind == "last":
        return "最后一个"
    return f"第 {item.index} 个"


def build_reference_correction_acknowledgement(
    *,
    surface: ResultReferenceSurface,
    item: ResultReferenceMemoryItem,
    question: str,
) -> str | None:
    """Build a deterministic correction lead-in, or ``None`` for plain reads."""

    if not is_reference_correction(question):
        return None
    label = _surface_label(surface, item)
    expected = _TITLE_RE.search(str(question or ""))
    expected_label = expected.group(1).strip() if expected is not None else None
    if expected_label and expected_label != item.filename:
        return (
            f"你说得对，按前面列出的目录，{label}是《{item.filename}》，"
            f"不是《{expected_label}》。以下是《{item.filename}》的正文："
        )
    return f"是的，按前面列出的目录，{label}就是《{item.filename}》。以下是正文："


@dataclass(frozen=True)
class ResolvedResultReference:
    memory: ResultReferenceMemory
    item: ResultReferenceMemoryItem
    source: dict[str, Any]
    surface: ResultReferenceSurface
    correction: bool
    acknowledgement: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory, ResultReferenceMemory):
            raise ValueError("resolved reference requires memory")
        if not isinstance(self.item, ResultReferenceMemoryItem):
            raise ValueError("resolved reference requires an item")
        if not isinstance(self.source, dict) or not self.source:
            raise ValueError("resolved reference requires a source")
        if not isinstance(self.surface, ResultReferenceSurface):
            raise ValueError("resolved reference requires a surface")
        if not isinstance(self.correction, bool):
            raise ValueError("resolved reference correction must be a boolean")
        object.__setattr__(self, "acknowledgement", str(self.acknowledgement) if self.acknowledgement else None)

    def safe_summary(self) -> dict[str, Any]:
        return {
            "memory_revision": self.memory.revision,
            "list_label": self.memory.list_label,
            "index": self.item.index,
            "filename": self.item.filename,
            "kb_id": str(self.item.kb_id),
            "doc_id": str(self.item.doc_id),
            "kind": self.surface.kind,
            "correction": self.correction,
            "acknowledgement": bool(self.acknowledgement),
        }


async def resolve_result_reference_memory(
    db: AsyncSession,
    *,
    value: object,
    question: str,
    selected_kb_ids: Iterable[uuid.UUID],
    read_session_factory: ReadSessionFactory | None = None,
) -> ResolvedResultReference | None:
    """Resolve one ordinal/reference-correction question against persisted memory.

    The ordinal is language structure; the document identity comes from the
    persisted list.  The selected document is then re-authorized against the
    current KB scope and must still be active and ready before it can be read.
    """

    memory = parse_result_reference_memory(value)
    if memory is None:
        return None
    if not is_result_list_reference(question):
        return None
    surface = parse_result_reference_surface(question)
    if surface is None or surface.kind not in {"ordinal", "last"}:
        return None
    item = memory.item_for_surface(surface)
    if item is None:
        return None
    selected = {str(kb_id) for kb_id in selected_kb_ids}
    if str(item.kb_id) not in selected:
        return None
    statement = (
        select(Document)
        .where(
            Document.id == item.doc_id,
            Document.kb_id == item.kb_id,
            Document.is_active.is_(True),
            Document.status == "ready",
        )
    )
    try:
        async with isolated_read_session(
            request_db=db,
            session_factory=read_session_factory,
        ) as read_db:
            document = (await read_db.execute(statement)).scalar_one_or_none()
            if (
                document is None
                or not document.is_active
                or str(document.status or "").strip() != "ready"
            ):
                return None
            filename = str(document.filename or "").strip() or item.filename
            source = {
                "source_kind": "document_result_reference",
                "id": str(document.id),
                "doc_id": str(document.id),
                "kb_id": str(document.kb_id),
                "filename": filename,
                "file_type": document.file_type,
                "doc_tags": list(document.tags or []),
                "evidence_role": "direct",
                "evidence_contribution_role": "result_reference",
                "status": str(document.status or "").strip().casefold(),
                "content": f"文档名称：{filename}；状态：已就绪",
            }
    except Exception:
        return None
    correction = is_reference_correction(question)
    return ResolvedResultReference(
        memory=memory,
        item=item,
        source=source,
        surface=surface,
        correction=correction,
        acknowledgement=build_reference_correction_acknowledgement(
            surface=surface,
            item=item,
            question=question,
        ),
    )


__all__ = [
    "RESULT_REFERENCE_MEMORY_SCHEMA_VERSION",
    "ResolvedResultReference",
    "ResultReferenceMemory",
    "ResultReferenceMemoryItem",
    "build_result_reference_memory",
    "build_reference_correction_acknowledgement",
    "is_reference_correction",
    "parse_result_reference_memory",
    "resolve_result_reference_memory",
]
