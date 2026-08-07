"""Execute a V3-selected reference to previously displayed document results."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from core.llm_stream import stream_with_retry_before_first_delta
from core.openai_client import get_client
from core.query_semantics import KnowledgeRequestSemantics
from core.rag_trace import content_fields, json_safe, trace_event
from core.structured_output import create_stream_completion
from core.document_content import render_document_chunks
from models.db_models import Document, DocumentChunk, KnowledgeBase


KNOWLEDGE_RESULT_RUNNER_VERSION = "knowledge_result.v1"
_MAX_SOURCE_CHUNKS = 20
_MAX_CONTEXT_CHARS = 30_000


def _sse(payload: Mapping[str, Any]) -> str:
    return "data: " + json.dumps(
        json_safe(dict(payload)), ensure_ascii=False, allow_nan=False
    ) + "\n\n"


def _step(step: str, status: str) -> str:
    return _sse({"type": "search_step", "step": step, "status": status})


def _process_event() -> str:
    return _sse({
        "type": "search_process",
        "schema_version": "search_process.v1",
        "execution_path": "result_reference",
        "steps": [
            {"key": "analyze", "label": "问题分析"},
            {"key": "retrieve", "label": "结果读取"},
            {"key": "generate", "label": "生成"},
        ],
    })


def _parse_uuid(value: object) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("result source identity is invalid") from exc


def _chunk_source(chunk: DocumentChunk, document: Document) -> dict[str, Any]:
    return {
        "source_kind": "document_chunk",
        "id": str(chunk.id),
        "chunk_id": str(chunk.id),
        "doc_id": str(chunk.doc_id),
        "kb_id": str(chunk.kb_id),
        "filename": document.filename,
        "file_type": document.file_type,
        "content": chunk.content,
        "chunk_index": chunk.chunk_index,
        "metadata": dict(chunk.metadata_ or {}),
        "doc_tags": list(document.tags or []),
        "evidence_role": "direct",
        "evidence_contribution_role": "result_reference",
        "answer_support": 1.0,
        "relevance": 1.0,
    }


def _bounded_document_text(
    ordered_documents: list[Document],
    chunks_by_document: dict[uuid.UUID, list[DocumentChunk]],
) -> tuple[str, bool]:
    sections: list[str] = []
    remaining = _MAX_CONTEXT_CHARS
    truncated = False
    for document in ordered_documents:
        rendered, section_truncated = render_document_chunks(
            document.filename,
            chunks_by_document.get(document.id, []),
            max_chars=remaining,
        )
        if rendered:
            sections.append(rendered)
            remaining -= len(rendered)
        truncated = truncated or section_truncated
        if remaining <= 0:
            truncated = True
            break
    return "\n\n---\n\n".join(sections), truncated


async def run_knowledge_result_stream(
    question: str,
    kb_ids: list[uuid.UUID],
    search_config: dict,
    conversation_id: str,
    db: AsyncSession,
    intent: dict | None = None,
    trace_id: str | None = None,
    standalone_query: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    carryover_sources: list[dict] | None = None,
    is_followup: bool = False,
    followup_reason: str | None = None,
    task_contract: object | None = None,
    evidence_scope_filter: dict | None = None,
    knowledge_request: KnowledgeRequestSemantics | None = None,
    result_sources: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    acknowledgement: str | None = None,
) -> AsyncGenerator[str, None]:
    """Read or synthesize only the documents selected by server-issued handles.

    ``acknowledgement`` is a server-produced lead-in for reference-correction
    turns (for example ``第四个不是《钉钉》吗``).  It is prepended verbatim to
    the deterministic read answer so the user sees the confirmation before the
    document body; it never alters document selection or retrieval.
    """

    del (
        search_config,
        conversation_history,
        carryover_sources,
        is_followup,
        followup_reason,
        task_contract,
        evidence_scope_filter,
    )
    if not isinstance(knowledge_request, KnowledgeRequestSemantics):
        raise ValueError("result runner requires a knowledge request")
    if not knowledge_request.is_result_operation:
        raise ValueError("result runner received a non-result request")
    bound_sources = list(result_sources or ())
    if len(bound_sources) != len(knowledge_request.result_handles):
        raise ValueError("result handle binding is incomplete")

    authorized_kb_ids = set(kb_ids)
    ordered_pairs: list[tuple[uuid.UUID, uuid.UUID]] = []
    for source in bound_sources:
        kb_id = _parse_uuid(source.get("kb_id"))
        doc_id = _parse_uuid(source.get("doc_id"))
        if kb_id not in authorized_kb_ids:
            raise ValueError("result document is outside the authorized KB scope")
        ordered_pairs.append((kb_id, doc_id))

    trace_id = trace_id or uuid.uuid4().hex
    started = time.perf_counter()
    user_question = (standalone_query or question).strip() or question.strip()
    yield _process_event()
    yield _step("analyze", "active")
    if intent:
        yield _sse({"type": "intent", "decision": intent})
    yield _step("analyze", "done")
    yield _step("retrieve", "active")

    document_ids = [doc_id for _, doc_id in ordered_pairs]
    document_rows = list((await db.execute(
        select(Document, KnowledgeBase)
        .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
        .where(
            Document.id.in_(document_ids),
            Document.kb_id.in_(authorized_kb_ids),
        )
    )).all())
    document_by_id = {
        document.id: (document, knowledge_base)
        for document, knowledge_base in document_rows
        if document is not None and knowledge_base is not None
    }
    if any(
        doc_id not in document_by_id
        or document_by_id[doc_id][0].kb_id != kb_id
        for kb_id, doc_id in ordered_pairs
    ):
        raise ValueError("result document is no longer available in the authorized scope")
    ordered_documents = [document_by_id[doc_id][0] for _, doc_id in ordered_pairs]
    readable_documents = [
        item for item in ordered_documents
        if item.is_active is True and str(item.status or "").strip().casefold() == "ready"
    ]
    if len(readable_documents) != len(ordered_documents):
        names = "、".join(f"《{item.filename}》" for item in ordered_documents)
        answer = f"{names}当前不是已就绪且启用的文档，暂时无法读取正文。"
        yield _step("retrieve", "done")
        yield _sse({
            "type": "search_results",
            "trace_id": trace_id,
            "results": [],
            "answer_sources": [],
            "total": 0,
            "displayed_result_count": 0,
            "context_evidence_count": 0,
            "answer_source_count": 0,
            "hit_count": 0,
            "direct_evidence_count": 0,
            "related_reference_count": 0,
            "retrieval_executed": True,
            "evidence_status": "no_hit",
            "coverage_status": "insufficient",
            "decision_reason": "referenced_document_not_readable",
            "pipeline_version": KNOWLEDGE_RESULT_RUNNER_VERSION,
            "answer_provenance": "knowledge_base",
        })
        yield _step("generate", "active")
        yield _sse({"type": "text_delta", "content": answer})
        yield _step("generate", "done")
        yield _sse({"type": "done", "conversation_id": conversation_id})
        return

    chunk_rows = list((await db.execute(
        select(DocumentChunk)
        .where(
            DocumentChunk.doc_id.in_(document_ids),
            DocumentChunk.kb_id.in_(authorized_kb_ids),
        )
        .order_by(DocumentChunk.doc_id.asc(), DocumentChunk.chunk_index.asc())
    )).scalars().all())
    chunks_by_document: dict[uuid.UUID, list[DocumentChunk]] = {}
    for chunk in chunk_rows:
        chunks_by_document.setdefault(chunk.doc_id, []).append(chunk)
    per_document_limit = max(1, _MAX_SOURCE_CHUNKS // len(ordered_documents))
    selected_chunks = [
        (chunk, document)
        for document in ordered_documents
        for chunk in chunks_by_document.get(document.id, [])[:per_document_limit]
    ][:_MAX_SOURCE_CHUNKS]
    answer_sources = [
        _chunk_source(chunk, document)
        for chunk, document in selected_chunks
    ]
    if (
        not answer_sources
        or any(not chunks_by_document.get(document.id) for document in ordered_documents)
    ):
        raise ValueError("referenced document has no current readable chunks")
    selected_chunks_by_document: dict[uuid.UUID, list[DocumentChunk]] = {}
    for chunk, document in selected_chunks:
        selected_chunks_by_document.setdefault(document.id, []).append(chunk)
    document_text, truncated = _bounded_document_text(
        ordered_documents, selected_chunks_by_document
    )
    truncated = truncated or len(selected_chunks) < len(chunk_rows)
    retrieval_ms = max(0, round((time.perf_counter() - started) * 1000))
    trace_event(
        "knowledge.result_reference.resolved",
        trace_id=trace_id,
        runner_version=KNOWLEDGE_RESULT_RUNNER_VERSION,
        operation=knowledge_request.operation,
        result_count=len(ordered_documents),
        source_chunk_count=len(answer_sources),
        context_chars=len(document_text),
        context_truncated=truncated,
        elapsed_ms=retrieval_ms,
        correction_acknowledged=bool(acknowledgement),
        **content_fields("question", user_question),
    )
    yield _step("retrieve", "done")
    yield _sse({
        "type": "search_results",
        "trace_id": trace_id,
        "results": answer_sources,
        "answer_sources": answer_sources,
        "total": len(answer_sources),
        "displayed_result_count": len(answer_sources),
        "context_evidence_count": len(answer_sources),
        "answer_source_count": len(answer_sources),
        "hit_count": len(answer_sources),
        "direct_evidence_count": len(answer_sources),
        "related_reference_count": 0,
        "retrieval_executed": True,
        "evidence_status": "hit",
        "coverage_status": "complete" if not truncated else "partial",
        "decision_reason": "authorized_result_reference",
        "pipeline_version": KNOWLEDGE_RESULT_RUNNER_VERSION,
        "answer_provenance": "knowledge_base",
    })

    yield _step("generate", "active")
    generation_started = time.perf_counter()
    if knowledge_request.operation == "read":
        answer = document_text
        if acknowledgement:
            answer = f"{acknowledgement}\n\n{answer}"
        if truncated:
            answer += "\n\n> 文档较长，当前展示了前 30000 个字符。"
        yield _sse({"type": "text_delta", "content": answer})
        first_token_ms = 0
        chunk_count = 1
        max_chunk_gap_ms = 0
        model = None
        finish_reason = "deterministic"
        usage = None
        answer_chars = len(answer)
    else:
        settings = get_settings()
        model = str(settings.chat_model or "").strip()
        action = "总结" if knowledge_request.operation == "summarize" else "比较"
        messages = [
            {
                "role": "system",
                "content": (
                    f"仅依据给出的文档正文完成{action}。不得补充正文之外的事实；"
                    "信息不足时明确指出。使用自然、简洁的中文回答。"
                ),
            },
            {"role": "user", "content": f"用户问题：{user_question}\n\n文档正文：\n{document_text}"},
        ]
        client = get_client().with_options(max_retries=0)
        thinking_disabled = False

        async def open_stream():
            nonlocal thinking_disabled
            stream, thinking_disabled = await create_stream_completion(
                client,
                request={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": settings.max_tokens,
                    "stream": True,
                    "timeout": float(settings.llm_request_timeout_seconds),
                },
                provider_identity=getattr(settings, "llm_base_url", ""),
                model=model,
            )
            return stream

        stream = stream_with_retry_before_first_delta(
            open_stream,
            model=model,
            prompt_chars=sum(len(item["content"]) for item in messages),
            timeout_seconds=float(settings.llm_request_timeout_seconds),
            max_attempts=settings.llm_max_attempts,
            retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        )
        first_delta_at: float | None = None
        previous_delta_at: float | None = None
        chunk_count = 0
        max_chunk_gap_ms = 0
        usage = None
        finish_reason = None
        answer_chars = 0
        try:
            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = str(getattr(getattr(choice, "delta", None), "content", "") or "")
                if not delta:
                    continue
                now = time.perf_counter()
                if first_delta_at is None:
                    first_delta_at = now
                if previous_delta_at is not None:
                    max_chunk_gap_ms = max(
                        max_chunk_gap_ms,
                        round((now - previous_delta_at) * 1000),
                    )
                previous_delta_at = now
                chunk_count += 1
                answer_chars += len(delta)
                yield _sse({"type": "text_delta", "content": delta})
        finally:
            await stream.aclose()
        first_token_ms = (
            round((first_delta_at - generation_started) * 1000)
            if first_delta_at is not None
            else None
        )

    yield _step("generate", "done")
    if usage is not None:
        yield _sse({
            "type": "usage",
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        })
    trace_event(
        "generation.completed",
        trace_id=trace_id,
        runner_version=KNOWLEDGE_RESULT_RUNNER_VERSION,
        model=model,
        answer_provenance="knowledge_base_result_reference",
        operation=knowledge_request.operation,
        answer_chars=answer_chars,
        prompt_tokens=(getattr(usage, "prompt_tokens", None) if usage else None),
        completion_tokens=(getattr(usage, "completion_tokens", None) if usage else None),
        total_tokens=(getattr(usage, "total_tokens", None) if usage else None),
        thinking_disabled=(
            thinking_disabled if knowledge_request.operation != "read" else False
        ),
        first_token_ms=first_token_ms,
        chunk_count=chunk_count,
        max_chunk_gap_ms=max_chunk_gap_ms,
        finish_reason=finish_reason,
        generation_ms=max(0, round((time.perf_counter() - generation_started) * 1000)),
        total_ms=max(0, round((time.perf_counter() - started) * 1000)),
    )
    yield _sse({"type": "done", "conversation_id": conversation_id})


__all__ = ["KNOWLEDGE_RESULT_RUNNER_VERSION", "run_knowledge_result_stream"]
