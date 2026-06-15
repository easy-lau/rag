import uuid
from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import UUID as SA_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from core.embeddings import embed_text


async def hybrid_search(
    db: AsyncSession,
    query: str,
    kb_ids: list[uuid.UUID],
    top_k: int = 5,
    method: str = "hybrid",
) -> list[dict]:
    if not kb_ids:
        return []

    if method == "vector":
        return await _vector_search(db, query, kb_ids, top_k)
    elif method == "keyword":
        return await _keyword_search(db, query, kb_ids, top_k)
    else:
        return await _hybrid_rrf(db, query, kb_ids, top_k)


async def _vector_search(db: AsyncSession, query: str, kb_ids: list[uuid.UUID], top_k: int) -> list[dict]:
    embedding = await embed_text(query)
    sql = text("""
        SELECT
            dc.id, dc.content, dc.chunk_index, dc.metadata,
            dc.kb_id, dc.doc_id, d.filename, d.file_type, d.source_url, d.tags AS doc_tags,
            1 - (dc.embedding <=> CAST(:emb AS vector)) AS score
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.doc_id
        WHERE dc.kb_id = ANY(:kb_ids)
          AND dc.embedding IS NOT NULL
          AND d.is_active = TRUE
        ORDER BY dc.embedding <=> CAST(:emb AS vector)
        LIMIT :top_k
    """).bindparams(bindparam("kb_ids", type_=ARRAY(SA_UUID())))
    rows = (await db.execute(sql, {
        "emb": str(embedding),
        "kb_ids": kb_ids,
        "top_k": top_k,
    })).mappings().all()
    return [dict(r) for r in rows]


async def _keyword_search(db: AsyncSession, query: str, kb_ids: list[uuid.UUID], top_k: int) -> list[dict]:
    sql = text("""
        SELECT
            dc.id, dc.content, dc.chunk_index, dc.metadata,
            dc.kb_id, dc.doc_id, d.filename, d.file_type, d.source_url, d.tags AS doc_tags,
            ts_rank(to_tsvector('simple', dc.content), plainto_tsquery('simple', :query)) AS score
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.doc_id
        WHERE dc.kb_id = ANY(:kb_ids)
          AND to_tsvector('simple', dc.content) @@ plainto_tsquery('simple', :query)
          AND d.is_active = TRUE
        ORDER BY score DESC
        LIMIT :top_k
    """).bindparams(bindparam("kb_ids", type_=ARRAY(SA_UUID())))
    rows = (await db.execute(sql, {
        "query": query,
        "kb_ids": kb_ids,
        "top_k": top_k,
    })).mappings().all()
    return [dict(r) for r in rows]


async def _hybrid_rrf(db: AsyncSession, query: str, kb_ids: list[uuid.UUID], top_k: int) -> list[dict]:
    embedding = await embed_text(query)
    sql = text("""
        WITH vector_r AS (
            SELECT dc.id,
                   ROW_NUMBER() OVER (ORDER BY dc.embedding <=> CAST(:emb AS vector)) AS rn
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.doc_id
            WHERE dc.kb_id = ANY(:kb_ids)
              AND dc.embedding IS NOT NULL
              AND d.is_active = TRUE
            LIMIT 20
        ),
        keyword_r AS (
            SELECT dc.id,
                   ROW_NUMBER() OVER (
                       ORDER BY ts_rank(to_tsvector('simple', dc.content),
                                        plainto_tsquery('simple', :query)) DESC
                   ) AS rn
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.doc_id
            WHERE dc.kb_id = ANY(:kb_ids)
              AND to_tsvector('simple', dc.content) @@ plainto_tsquery('simple', :query)
              AND d.is_active = TRUE
            LIMIT 20
        ),
        rrf AS (
            SELECT
                COALESCE(v.id, k.id) AS id,
                COALESCE(1.0/(60+v.rn), 0) + COALESCE(1.0/(60+k.rn), 0) AS rrf_score
            FROM vector_r v FULL JOIN keyword_r k USING(id)
        )
        SELECT
            dc.id, dc.content, dc.chunk_index, dc.metadata,
            dc.kb_id, dc.doc_id, d.filename, d.file_type, d.source_url, d.tags AS doc_tags,
            rrf.rrf_score AS score
        FROM rrf
        JOIN document_chunks dc ON dc.id = rrf.id
        JOIN documents d ON d.id = dc.doc_id
        ORDER BY rrf_score DESC
        LIMIT :top_k
    """).bindparams(bindparam("kb_ids", type_=ARRAY(SA_UUID())))
    rows = (await db.execute(sql, {
        "emb": str(embedding),
        "query": query,
        "kb_ids": kb_ids,
        "top_k": top_k,
    })).mappings().all()
    return [dict(r) for r in rows]
