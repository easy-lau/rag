import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from models.db_models import User
from models.schemas import SearchRequest, SearchResponse, SearchResultItem
from core.retriever import hybrid_search
from core.reranker import rerank_with_status
from core.query_constraints import extract_query_constraints
from core.rag_shared import (
    annotate_deterministic_constraints,
    rerank_candidate_limit,
)
from core.rag_trace import (
    content_fields,
    log_exception_safely,
    trace_event,
    trace_query_constraints,
)
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import SEARCH_USE
from config import get_settings

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


@router.post("/test", response_model=SearchResponse)
async def search_test(
    payload: SearchRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(SEARCH_USE)),
):
    trace_id = uuid.uuid4().hex
    response.headers["X-RAG-Trace-ID"] = trace_id
    started_at = time.perf_counter()
    # 校验请求的知识库都在可访问范围内（accessible 为 None 表示全部）
    accessible = await get_accessible_kb_ids(user, db)
    if accessible is not None and not set(payload.knowledge_base_ids).issubset(set(accessible)):
        raise HTTPException(status_code=403, detail="无权访问部分知识库")

    s = get_settings()
    trace_include_content = s.rag_trace_include_content
    candidate_k = rerank_candidate_limit(payload.top_k) if payload.rerank else payload.top_k
    constraints = extract_query_constraints(payload.query)
    trace_event(
        "search_test.request",
        trace_id=trace_id,
        user_id=user.id,
        selected_kb_ids=payload.knowledge_base_ids,
        method=payload.method,
        top_k=payload.top_k,
        candidate_k=candidate_k,
        rerank=payload.rerank,
        query_constraints=trace_query_constraints(constraints),
        **content_fields("query", payload.query),
    )

    retrieval_ms = 0
    rerank_ms = 0
    rerank_succeeded: bool | None = None
    rerank_error = None
    rerank_status = "skipped"
    rerank_attempted = False
    current_stage = "retrieval"
    try:
        stage_started = time.perf_counter()
        results = await hybrid_search(
            db=db,
            query=payload.query,
            kb_ids=payload.knowledge_base_ids,
            top_k=candidate_k,
            method=payload.method,
            trace_id=trace_id,
            surface="search_test",
        )
        retrieval_ms = round((time.perf_counter() - stage_started) * 1000)
        trace_event(
            "retrieval.completed",
            trace_id=trace_id,
            surface="search_test",
            method=payload.method,
            succeeded=True,
            candidate_count=len(results),
            active_channels=[
                channel
                for channel in ("vector", "keyword", "trigram")
                if any(channel in (item.get("active_channels") or []) for item in results)
            ],
            channel_candidate_counts={
                channel: sum(
                    channel in (item.get("active_channels") or []) for item in results
                )
                for channel in ("vector", "keyword", "trigram")
            },
            elapsed_ms=retrieval_ms,
        )
        if s.rag_trace_include_candidate_details:
            for rank, result in enumerate(results, start=1):
                candidate_payload = {
                    "trace_id": trace_id,
                    "surface": "search_test",
                    "rank": rank,
                    "chunk_id": result.get("id"),
                    "doc_id": result.get("doc_id"),
                    "kb_id": result.get("kb_id"),
                    "chunk_index": result.get("chunk_index"),
                    "vector_score": result.get("vector_score"),
                    "vector_rank": result.get("vector_rank"),
                    "keyword_score": result.get("keyword_score"),
                    "keyword_rank": result.get("keyword_rank"),
                    "trigram_score": result.get("trigram_score"),
                    "trigram_rank": result.get("trigram_rank"),
                    "retrieval_score": result.get(
                        "retrieval_score",
                        result.get("score"),
                    ),
                    "active_channels": result.get("active_channels"),
                    **content_fields(
                        "filename",
                        str(result.get("filename") or ""),
                    ),
                    **content_fields(
                        "candidate_content",
                        str(result.get("content") or ""),
                    ),
                }
                if trace_include_content:
                    candidate_payload.update(
                        file_type=result.get("file_type"),
                        tags=result.get("doc_tags") or [],
                        metadata=result.get("metadata") or {},
                    )
                trace_event(
                    "retrieval.candidate",
                    **candidate_payload,
                )

        current_stage = "rerank"
        if payload.rerank and results:
            rerank_attempted = True
            stage_started = time.perf_counter()
            outcome = await rerank_with_status(payload.query, results)
            rerank_ms = round((time.perf_counter() - stage_started) * 1000)
            results = outcome.results
            constraints = outcome.constraints or constraints
            rerank_succeeded = outcome.succeeded
            rerank_error = outcome.error
            rerank_status = "verified" if outcome.succeeded else "unverified"
        else:
            results = annotate_deterministic_constraints(results, constraints)
            rerank_status = "skipped"
        trace_event(
            "rerank.completed",
            trace_id=trace_id,
            surface="search_test",
            requested=payload.rerank,
            attempted=rerank_attempted,
            succeeded=rerank_succeeded,
            candidate_count=len(results),
            elapsed_ms=rerank_ms,
            error=(
                rerank_error
                if trace_include_content
                else ((rerank_error or "").partition(":")[0] or None)
            ),
        )
    except Exception as exc:
        log_exception_safely(
            logger,
            "[检索测试] trace=%s 执行失败",
            trace_id,
            exc=exc,
        )
        if current_stage == "retrieval":
            retrieval_ms = round((time.perf_counter() - stage_started) * 1000)
            trace_event(
                "retrieval.completed",
                trace_id=trace_id,
                surface="search_test",
                method=payload.method,
                succeeded=False,
                candidate_count=0,
                elapsed_ms=retrieval_ms,
                error=exc,
            )
        else:
            rerank_ms = round((time.perf_counter() - stage_started) * 1000)
            trace_event(
                "rerank.completed",
                trace_id=trace_id,
                surface="search_test",
                requested=payload.rerank,
                attempted=rerank_attempted,
                succeeded=False,
                candidate_count=len(results),
                elapsed_ms=rerank_ms,
                error=exc,
            )
        trace_event(
            "search_test.error",
            trace_id=trace_id,
            user_id=user.id,
            method=payload.method,
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            total_ms=round((time.perf_counter() - started_at) * 1000),
            error=exc,
        )
        raise

    # 检索测试与真实问答保持“扩大召回→重排→Top K”。
    results = results[:payload.top_k]

    items = [
        SearchResultItem(
            id=r["id"],
            content=r["content"],
            filename=r["filename"],
            file_type=r.get("file_type"),
            score=round(float(r.get("score", 0)), 4),
            chunk_index=r.get("chunk_index", 0),
            metadata=r.get("metadata"),
            tags=r.get("doc_tags") or [],
            kb_id=r.get("kb_id"),
            doc_id=r.get("doc_id"),
            retrieval_score=r.get("retrieval_score"),
            vector_score=r.get("vector_score"),
            vector_rank=r.get("vector_rank"),
            keyword_score=r.get("keyword_score"),
            keyword_rank=r.get("keyword_rank"),
            trigram_score=r.get("trigram_score"),
            trigram_rank=r.get("trigram_rank"),
            active_channels=r.get("active_channels") or [],
            rerank_status=r.get("rerank_status"),
            topic_relevance=r.get("topic_relevance"),
            answer_support=r.get("answer_support"),
            constraint_status=r.get("constraint_status"),
            evidence_role=r.get("evidence_role"),
            rerank_reason=r.get("rerank_reason"),
            constraint_reason=r.get("constraint_reason"),
            ranking_factors=r.get("ranking_factors"),
        )
        for r in results
    ]

    direct_count = sum(item.evidence_role == "direct" for item in items)
    related_count = sum(item.evidence_role == "related" for item in items)
    constraint_statuses = {item.constraint_status for item in items if item.constraint_status}
    if direct_count:
        evidence_status = "partial" if related_count else "hit"
    elif (
        related_count
        and constraints.has_scope_constraint
        and constraint_statuses == {"mismatch"}
    ):
        # Applicability is one product/version/project boundary.  The old
        # spelling is retained only for reads of historical rows, never for a
        # newly generated search response.
        evidence_status = "scope_mismatch"
    elif related_count:
        evidence_status = "partial"
    elif items:
        evidence_status = "unverified" if rerank_succeeded is not True else "no_hit"
    else:
        evidence_status = "no_hit"

    def trace_result(item: SearchResultItem) -> dict:
        raw = item.model_dump(exclude={"content", "metadata", "tags"})
        if s.rag_trace_include_content:
            raw["filename"] = item.filename
            raw["metadata"] = item.metadata
            raw["tags"] = item.tags
        else:
            raw.pop("filename", None)
            raw.pop("rerank_reason", None)
            raw.pop("constraint_reason", None)
            raw.update(content_fields("filename", item.filename))
        raw.update(content_fields("candidate_content", item.content))
        return raw

    trace_event(
        "search_test.completed",
        trace_id=trace_id,
        user_id=user.id,
        method=payload.method,
        top_k=payload.top_k,
        rerank=payload.rerank,
        rerank_status=rerank_status,
        rerank_succeeded=rerank_succeeded,
        rerank_error=(
            rerank_error
            if trace_include_content
            else ((rerank_error or "").partition(":")[0] or None)
        ),
        query_constraints=trace_query_constraints(constraints),
        direct_evidence_count=direct_count,
        related_reference_count=related_count,
        displayed_result_count=len(items),
        hit_count=direct_count,
        evidence_status=evidence_status,
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        total_ms=round((time.perf_counter() - started_at) * 1000),
        results=[trace_result(item) for item in items],
    )

    return SearchResponse(
        results=items,
        total=len(items),
        search_meta={
            "method": payload.method,
            "rerank": payload.rerank,
            "top_k": payload.top_k,
            "embedding_model": s.embedding_model,
            "trace_id": trace_id,
            "candidate_k": candidate_k,
            "rerank_status": rerank_status,
            "rerank_succeeded": rerank_succeeded,
            # 客户端只需要知道失败类别；完整供应商异常仅写入开发追踪日志。
            "rerank_error": ((rerank_error or "").partition(":")[0] or None),
            "query_constraints": constraints.as_dict(),
            "direct_evidence_count": direct_count,
            "related_reference_count": related_count,
            "evidence_status": evidence_status,
            "displayed_candidate_count": len(items),
            "hit_count": direct_count,
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
        },
    )
