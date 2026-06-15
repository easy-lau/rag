from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from models.db_models import User
from models.schemas import SearchRequest, SearchResponse, SearchResultItem
from core.retriever import hybrid_search
from core.reranker import rerank
from core.rag_pipeline import apply_tag_boost
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import SEARCH_USE
from config import get_settings
import uuid

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/test", response_model=SearchResponse)
async def search_test(
    payload: SearchRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(SEARCH_USE)),
):
    # 校验请求的知识库都在可访问范围内（accessible 为 None 表示全部）
    accessible = await get_accessible_kb_ids(user, db)
    if accessible is not None and not set(payload.knowledge_base_ids).issubset(set(accessible)):
        raise HTTPException(status_code=403, detail="无权访问部分知识库")

    s = get_settings()
    results = await hybrid_search(
        db=db,
        query=payload.query,
        kb_ids=payload.knowledge_base_ids,
        top_k=payload.top_k,
        method=payload.method,
    )

    if payload.rerank and results:
        results = await rerank(payload.query, results)

    # 标签软加权：命中所选标签的文档上浮排序（与问答检索一致）
    results = apply_tag_boost(results, payload.tags)

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
        )
        for r in results
    ]

    return SearchResponse(
        results=items,
        total=len(items),
        search_meta={
            "method": payload.method,
            "rerank": payload.rerank,
            "top_k": payload.top_k,
            "embedding_model": s.embedding_model,
        },
    )
