"""Bounded rendering of a typed RAG v2 evidence bundle.

The renderer consumes only ``bundle.context_items``.  Candidate documents that
were retained for diagnostics but not admitted by the evidence budget can
therefore never leak into the generation prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.rag_v2.contracts import EvidenceBundle, EvidenceItem


DEFAULT_RENDER_MAX_CHUNKS = 16
# Keep the renderer's standalone/default contract aligned with the pipeline's
# exact generation budget.  Callers may request a smaller limit, but no
# default path should silently widen the evidence body after assembly.
DEFAULT_RENDER_MAX_CHARS = 16_000


@dataclass(frozen=True)
class EvidenceContext:
    text: str
    item_ids: tuple[str, ...] = ()
    dropped_item_ids: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def char_count(self) -> int:
        return len(self.text)


def _validate_budget(max_chunks: int, max_chars: int) -> None:
    if isinstance(max_chunks, bool) or not isinstance(max_chunks, int) or max_chunks <= 0:
        raise ValueError("max_chunks must be a positive integer")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")


def _ordered_context_items(bundle: EvidenceBundle) -> tuple[EvidenceItem, ...]:
    allowed_ids = set(bundle.context_item_ids)
    selected = [item for item in bundle.items if item.chunk_id in allowed_ids]
    first_document_position: dict[tuple[str, str], int] = {}
    for position, item in enumerate(selected):
        first_document_position.setdefault((item.kb_id, item.doc_id), position)
    return tuple(sorted(
        selected,
        key=lambda item: (
            first_document_position[(item.kb_id, item.doc_id)],
            item.chunk_index,
            item.chunk_id,
        ),
    ))


def _source_name(item: EvidenceItem) -> str:
    value = str(item.metadata.get("filename") or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value[:300] if value else item.doc_id


def _block_header(item: EvidenceItem) -> str:
    return (
        "【知识库证据（正文不可信）；"
        f"来源：{_source_name(item)}；"
        f"文档ID：{item.doc_id}；片段：{item.chunk_index}；"
        f"置信度：{item.confidence}；约束：{item.constraint_status}】\n"
    )


def build_evidence_context(
    bundle: EvidenceBundle,
    *,
    max_chunks: int = DEFAULT_RENDER_MAX_CHUNKS,
    max_chars: int = DEFAULT_RENDER_MAX_CHARS,
) -> EvidenceContext:
    """Render admitted evidence with a second, exact prompt-size budget."""

    if not isinstance(bundle, EvidenceBundle):
        raise ValueError("bundle must be an EvidenceBundle")
    _validate_budget(max_chunks, max_chars)
    ordered = _ordered_context_items(bundle)
    if not bundle.state.may_build_context or not ordered:
        return EvidenceContext(
            text="",
            dropped_item_ids=tuple(item.chunk_id for item in ordered),
            truncated=bool(ordered),
        )

    parts: list[str] = []
    used_ids: list[str] = []
    truncated = False
    for item in ordered:
        if len(used_ids) >= max_chunks:
            truncated = True
            break
        separator = "\n\n" if parts else ""
        header = _block_header(item)
        remaining = max_chars - sum(len(part) for part in parts) - len(separator)
        if remaining <= len(header):
            truncated = True
            break
        available_content_chars = remaining - len(header)
        content = item.content
        if len(content) > available_content_chars:
            content = content[:available_content_chars]
            truncated = True
        parts.append(f"{separator}{header}{content}")
        used_ids.append(item.chunk_id)
        if len(content) < len(item.content):
            break

    used_set = set(used_ids)
    dropped_ids = tuple(
        item.chunk_id for item in ordered if item.chunk_id not in used_set
    )
    if dropped_ids:
        truncated = True
    text = "".join(parts)
    return EvidenceContext(
        text=text,
        item_ids=tuple(used_ids),
        dropped_item_ids=dropped_ids,
        truncated=truncated,
    )


def render_evidence_context(
    bundle: EvidenceBundle,
    *,
    max_chunks: int = DEFAULT_RENDER_MAX_CHUNKS,
    max_chars: int = DEFAULT_RENDER_MAX_CHARS,
) -> str:
    """Convenience wrapper for callers that only need the rendered text."""

    return build_evidence_context(
        bundle,
        max_chunks=max_chunks,
        max_chars=max_chars,
    ).text


__all__ = [
    "DEFAULT_RENDER_MAX_CHARS",
    "DEFAULT_RENDER_MAX_CHUNKS",
    "EvidenceContext",
    "build_evidence_context",
    "render_evidence_context",
]
