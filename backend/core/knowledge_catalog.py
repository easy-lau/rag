"""Authorized document-catalog capability for conversational knowledge tools.

This is not a keyword shortcut around RAG.  It executes one typed
``KnowledgeRequestSemantics`` produced by the shared semantic authority.  The
caller supplies the already-authorized request KB scope; the model cannot add
KB/document ids or SQL.  Catalog operations use document metadata only and
never invoke embeddings, vector retrieval, reranking or a generation model.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from sqlalchemy import Text, and_, case, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.query_semantics import KnowledgeRequestSemantics
from core.rag_trace import content_fields, json_safe, trace_event
from models.db_models import Document, KnowledgeBase


KNOWLEDGE_CATALOG_RUNNER_VERSION = "knowledge_catalog.v1"
_MAX_DISPLAY_DOCUMENTS = 20
_FILTER_PUNCTUATION_RE = re.compile(r"[\s\u3000,，、;；:：。！？?!'\"`]+")
_STATUS_LABELS = {
    "ready": "已就绪",
    "processing": "处理中",
    "failed": "处理失败",
    "inactive": "已停用",
}


def _sse(payload: Mapping[str, Any]) -> str:
    return (
        "data: "
        + json.dumps(json_safe(dict(payload)), ensure_ascii=False, allow_nan=False)
        + "\n\n"
    )


def _process_event() -> str:
    return _sse({
        "type": "search_process",
        "schema_version": "search_process.v1",
        "execution_path": "catalog",
        "steps": [
            {"key": "analyze", "label": "问题分析"},
            {"key": "retrieve", "label": "目录查询"},
            {"key": "generate", "label": "生成"},
        ],
    })


def _step_event(step: str, status: str) -> str:
    return _sse({"type": "search_step", "step": step, "status": status})


def _catalog_filter_pattern(value: object) -> str:
    """Compile one exact source literal into a bounded ordered match.

    Ordered-character matching allows metadata such as ``产品甲6配置`` to match
    the user literal ``产品甲配置`` without inventing synonyms or consulting a
    vector model.  The input is request-catalog text, not raw SQL.
    """

    compact = _FILTER_PUNCTUATION_RE.sub("", str(value or ""))[:96]
    if not compact:
        raise ValueError("catalog filter term is empty")
    return ".*".join(re.escape(character) for character in compact)


def _status_condition(request: KnowledgeRequestSemantics):
    if request.status_filter == "inactive":
        return Document.is_active.is_(False)
    active = Document.is_active.is_(True)
    if request.status_filter == "any":
        return None
    if request.status_filter == "not_ready":
        return and_(active, Document.status != "ready")
    return and_(active, Document.status == request.status_filter)


def _catalog_conditions(
    *,
    request: KnowledgeRequestSemantics,
    kb_ids: list[uuid.UUID],
) -> list[Any]:
    conditions: list[Any] = [
        Document.kb_id.in_(kb_ids),
    ]
    status_condition = _status_condition(request)
    if status_condition is not None:
        conditions.append(status_condition)
    searchable_fields = (
        func.coalesce(Document.filename, ""),
        func.coalesce(cast(Document.tags, Text), ""),
        func.coalesce(KnowledgeBase.name, ""),
        func.coalesce(KnowledgeBase.description, ""),
    )
    for term in request.filter_terms:
        pattern = _catalog_filter_pattern(term)
        conditions.append(or_(*(
            field.op("~*")(pattern)
            for field in searchable_fields
        )))
    return conditions


def _record(row: Any) -> dict[str, Any]:
    document, knowledge_base = row
    status = (
        str(document.status or "").strip().casefold()
        if document.is_active is True
        else "inactive"
    )
    return {
        "source_kind": "document_metadata",
        "id": str(document.id),
        "doc_id": str(document.id),
        "kb_id": str(document.kb_id),
        "filename": document.filename,
        "file_type": document.file_type,
        "status": status,
        "status_label": _STATUS_LABELS.get(status, status or "未知"),
        "is_active": bool(document.is_active),
        "doc_tags": list(document.tags or []),
        "knowledge_base_name": knowledge_base.name,
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "updated_at": document.updated_at.isoformat() if document.updated_at else None,
        "evidence_role": "direct",
        "evidence_contribution_role": "metadata",
        "content": (
            f"文档名称：{document.filename}；知识库：{knowledge_base.name}；"
            f"状态：{_STATUS_LABELS.get(status, status or '未知')}；"
            f"文件类型：{document.file_type or '未知'}"
        ),
    }


def _filter_label(request: KnowledgeRequestSemantics) -> str:
    return "、".join(request.filter_terms)


def _render_answer(
    *,
    request: KnowledgeRequestSemantics,
    total: int,
    records: list[dict[str, Any]],
    groups: list[tuple[str, int]],
    selected_kb_count: int,
) -> str:
    subject = _filter_label(request)
    scope = f"你当前选择且有权限访问的 {selected_kb_count} 个知识库"
    match_note = (
        f"按文档名称、标签和知识库名称匹配“{subject}”"
        if subject
        else "按当前授权目录"
    )
    if total == 0:
        return f"在{scope}中，{match_note}没有找到符合条件的文章。"

    if request.operation == "group":
        group_label = {
            "knowledge_base": "知识库",
            "status": "状态",
            "file_type": "文件类型",
        }[request.group_by]
        display_groups = [
            (
                _STATUS_LABELS.get(label, label or "未标注")
                if request.group_by == "status"
                else (label or "未标注"),
                count,
            )
            for label, count in groups
        ]
        lines = [f"- {label}：{count} 篇" for label, count in display_groups]
        return (
            f"在{scope}中，{match_note}共找到 {total} 篇文章。"
            f"按{group_label}统计如下：\n\n" + "\n".join(lines)
        )

    names = [
        f"《{item['filename']}》（{item['knowledge_base_name']}，{item['status_label']}）"
        for item in records
    ]
    if request.operation == "count":
        answer = f"在{scope}中，{match_note}共有 {total} 篇文章。"
    else:
        answer = f"在{scope}中，{match_note}共找到 {total} 篇文章。"
    if names:
        answer += "\n\n" + "\n".join(
            f"{index}. {name}" for index, name in enumerate(names, start=1)
        )
        if total > len(names):
            answer += f"\n\n当前先展示前 {len(names)} 篇。"
    return answer


async def run_knowledge_catalog_stream(
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
) -> AsyncGenerator[str, None]:
    """Execute one authorized metadata capability and stream an exact answer."""

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
        raise ValueError("catalog runner requires a knowledge request")
    if not knowledge_request.is_catalog_operation:
        raise ValueError("catalog runner received a content request")
    authorized_kb_ids = list(dict.fromkeys(kb_ids))
    if not authorized_kb_ids:
        raise ValueError("catalog runner requires an authorized KB scope")

    trace_id = trace_id or uuid.uuid4().hex
    started = time.perf_counter()
    user_question = (standalone_query or question).strip() or question.strip()

    yield _process_event()
    yield _step_event("analyze", "active")
    if intent:
        yield _sse({"type": "intent", "decision": intent})
    trace_event(
        "knowledge.capability.selected",
        trace_id=trace_id,
        runner_version=KNOWLEDGE_CATALOG_RUNNER_VERSION,
        capability=knowledge_request.safe_summary(),
        selected_kb_count=len(authorized_kb_ids),
        **content_fields("question", user_question),
    )
    yield _step_event("analyze", "done")
    yield _step_event("retrieve", "active")

    conditions = _catalog_conditions(
        request=knowledge_request,
        kb_ids=authorized_kb_ids,
    )
    count_statement = (
        select(func.count())
        .select_from(Document)
        .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
        .where(*conditions)
    )
    total = int((await db.execute(count_statement)).scalar_one())
    rows_statement = (
        select(Document, KnowledgeBase)
        .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
        .where(*conditions)
        .order_by(KnowledgeBase.name.asc(), Document.filename.asc(), Document.id.asc())
        .limit(_MAX_DISPLAY_DOCUMENTS)
    )
    rows = list((await db.execute(rows_statement)).all())
    records = [_record(row) for row in rows]

    groups: list[tuple[str, int]] = []
    if knowledge_request.operation == "group":
        if knowledge_request.group_by == "knowledge_base":
            group_value = KnowledgeBase.name
        elif knowledge_request.group_by == "file_type":
            group_value = func.coalesce(Document.file_type, "未标注")
        else:
            group_value = case(
                (Document.is_active.is_(False), "inactive"),
                else_=func.coalesce(Document.status, "未知"),
            )
        group_statement = (
            select(group_value, func.count())
            .select_from(Document)
            .join(KnowledgeBase, KnowledgeBase.id == Document.kb_id)
            .where(*conditions)
            .group_by(group_value)
            .order_by(func.count().desc(), group_value.asc())
        )
        groups = [(str(label or "未标注"), int(count)) for label, count in (await db.execute(group_statement)).all()]

    elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
    evidence_status = "hit" if total else "no_hit"
    trace_event(
        "knowledge.capability.executed",
        trace_id=trace_id,
        runner_version=KNOWLEDGE_CATALOG_RUNNER_VERSION,
        capability=knowledge_request.safe_summary(),
        result_count=total,
        displayed_result_count=len(records),
        group_count=len(groups),
        elapsed_ms=elapsed_ms,
    )
    trace_event(
        "retrieval.completed",
        trace_id=trace_id,
        runner_version=KNOWLEDGE_CATALOG_RUNNER_VERSION,
        executed=True,
        method="document_metadata",
        candidate_count=total,
        elapsed_ms=elapsed_ms,
    )
    yield _step_event("retrieve", "done")
    trace_event(
        "rerank.completed",
        trace_id=trace_id,
        runner_version=KNOWLEDGE_CATALOG_RUNNER_VERSION,
        requested=False,
        executed=False,
        succeeded=None,
        reason="authoritative_metadata_result",
        elapsed_ms=0,
    )
    yield _sse({
        "type": "search_results",
        "trace_id": trace_id,
        "results": records,
        "answer_sources": records if total else [],
        "total": total,
        "displayed_result_count": len(records),
        "context_evidence_count": len(records),
        "answer_source_count": len(records),
        "hit_count": len(records),
        "direct_evidence_count": len(records),
        "related_reference_count": 0,
        "retrieval_executed": True,
        "evidence_status": evidence_status,
        "coverage_status": "complete" if total else "insufficient",
        "decision_reason": "authorized_document_catalog_query",
        "pipeline_version": KNOWLEDGE_CATALOG_RUNNER_VERSION,
        "capability": knowledge_request.safe_summary(),
    })

    yield _step_event("generate", "active")
    answer = _render_answer(
        request=knowledge_request,
        total=total,
        records=records,
        groups=groups,
        selected_kb_count=len(authorized_kb_ids),
    )
    yield _sse({"type": "text_delta", "content": answer})
    trace_event(
        "generation.context",
        trace_id=trace_id,
        runner_version=KNOWLEDGE_CATALOG_RUNNER_VERSION,
        evidence_status=evidence_status,
        answer_provenance="authoritative_metadata",
        model=None,
        context_source_count=len(records),
    )
    trace_event(
        "generation.completed",
        trace_id=trace_id,
        runner_version=KNOWLEDGE_CATALOG_RUNNER_VERSION,
        model=None,
        answer_provenance="authoritative_metadata",
        answer_chars=len(answer),
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        finish_reason="deterministic",
        total_ms=max(0, round((time.perf_counter() - started) * 1000)),
    )
    yield _step_event("generate", "done")
    yield _sse({"type": "done", "conversation_id": conversation_id})


__all__ = [
    "KNOWLEDGE_CATALOG_RUNNER_VERSION",
    "run_knowledge_catalog_stream",
]
