import re
import uuid
import logging

from sqlalchemy import Text as SA_Text
from sqlalchemy import bindparam, text
from sqlalchemy import UUID as SA_UUID
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession

from core.embeddings import embed_text
from core.rag_trace import exception_log_text, trace_event


logger = logging.getLogger(__name__)


# RRF 的常用平滑常数。它让不同召回通道按“名次”而非原始分数尺度参与融合，
# 避免余弦相似度、ts_rank 和 pg_trgm 分数不可直接相加的问题。
RRF_K = 60
TRIGRAM_MIN_SCORE = 0.12
_MIN_CANDIDATE_POOL = 40
_MAX_CANDIDATE_POOL = 240
# 模糊问题只取每篇文档一个片段时，“问题描述：无”可能挤掉同文档真正的
# “解决方案”。允许少量片段进入重排，再由证据模型选择，兼顾召回与成本。
PER_DOCUMENT_RERANK_CHUNKS = 3

_TRIGRAM_TOKEN_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:+/\-]*|[\u3400-\u9fff]{2,}",
    re.IGNORECASE,
)
_QUERY_FILLER_PREFIXES = (
    "请帮我",
    "麻烦帮我",
    "我想知道",
    "我想了解",
    "我想问",
    "请问",
    "如何解决",
    "怎么解决",
    "解决",
    "如何",
    "怎么",
    "怎样",
    "我是",
    "我用的是",
    "当前是",
)
_QUERY_FILLER_SUFFIXES = (
    "应该怎么配置",
    "要怎么配置",
    "怎么配置",
    "如何配置",
    "要配置什么",
    "配置什么",
    "怎么办",
    "是什么",
    "可以吗",
    "吗",
)
_GENERIC_TRIGRAM_TERMS = {
    "请问",
    "帮我",
    "麻烦",
    "如何",
    "怎么",
    "怎样",
    "什么",
    "配置",
    "问题",
    "解决",
    "现在",
    "当前",
    "要配置什么",
    "配置什么",
    "怎么配置",
    "如何配置",
    "应该怎么配置",
}


def _candidate_pool_size(top_k: int) -> int:
    """为按文档去重预留足够候选，同时限制全表评分后的返回规模。"""

    return min(max(top_k * 8, _MIN_CANDIDATE_POOL), _MAX_CANDIDATE_POOL)


def _strip_query_fillers(term: str) -> str:
    """去掉问句外壳，保留产品名、版本号、配置项等可检索实体。"""

    value = term.strip(" \t\r\n，。！？；：,.!?;:()（）[]【】\"'")
    changed = True
    while changed and value:
        changed = False
        for prefix in _QUERY_FILLER_PREFIXES:
            if value.startswith(prefix) and len(value) > len(prefix) + 1:
                value = value[len(prefix):].strip()
                changed = True
                break
        for suffix in _QUERY_FILLER_SUFFIXES:
            if value.endswith(suffix) and len(value) > len(suffix) + 1:
                value = value[:-len(suffix)].strip()
                changed = True
                break
    return value


def _build_trigram_terms(query: str) -> list[str]:
    """生成适合 ``pg_trgm.word_similarity`` 的紧凑查询词。

    ``similarity(长问句, 长片段)`` 会被双方大量无关字符稀释。这里同时保留完整
    问句和去掉“请问/怎么配置”等外壳后的短语，让中文配置项、产品名和版本号都
    有机会进入词面候选；SQL 中再按词长加权，避免仅命中两字通用词就排到最前。
    """

    normalized = re.sub(r"\s+", " ", query or "").strip()
    if not normalized:
        return []

    base_candidates: list[str] = [normalized[:96]]
    base_candidates.extend(part for part in normalized.split(" ") if part)
    base_candidates.extend(
        match.group(0) for match in _TRIGRAM_TOKEN_RE.finditer(normalized)
    )

    # 同时加入去掉问句外壳后的版本；例如“我是云枢8.6”会得到“云枢8.6”，
    # “解决登录用户名枚举”会得到“登录用户名枚举”。
    stripped_candidates = [_strip_query_fillers(term) for term in base_candidates]
    # 完整问句用于召回整句近似；去外壳实体优先于其余原句片段，确保特别长的问句
    # 达到词项上限时，产品、版本和主题仍不会被尾部的口语成分挤掉。
    candidates = [normalized[:96], *stripped_candidates, *base_candidates]

    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        term = re.sub(r"\s+", " ", candidate).strip()[:96]
        if len(term) < 2 or term.lower() in _GENERIC_TRIGRAM_TERMS:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(term)
        if len(result) >= 16:
            break
    return result


def _normalize_result_scores(row: dict) -> dict:
    """统一检索结果的可观测字段，并保持旧 ``score`` 字段兼容。"""

    item = dict(row)
    retrieval_score = item.get("retrieval_score")
    if retrieval_score is None:
        retrieval_score = item.get("score")
    item["retrieval_score"] = retrieval_score
    item["score"] = retrieval_score
    for field in (
        "vector_score",
        "vector_rank",
        "keyword_score",
        "keyword_rank",
        "trigram_score",
        "trigram_rank",
    ):
        item.setdefault(field, None)
    item["active_channels"] = [
        channel
        for channel, rank_field in (
            ("vector", "vector_rank"),
            ("keyword", "keyword_rank"),
            ("trigram", "trigram_rank"),
        )
        if item.get(rank_field) is not None
    ]
    return item


def _results(rows) -> list[dict]:
    return [_normalize_result_scores(dict(row)) for row in rows]


async def hybrid_search(
    db: AsyncSession,
    query: str,
    kb_ids: list[uuid.UUID],
    top_k: int = 5,
    method: str = "hybrid",
    *,
    trace_id: str | None = None,
    surface: str = "chat",
) -> list[dict]:
    if not kb_ids or top_k <= 0 or not query.strip():
        return []

    if method == "vector":
        return await _vector_search(db, query, kb_ids, top_k)
    if method == "keyword":
        # 中文长问句在 PostgreSQL simple FTS 中通常是单个 token；关键词模式
        # 同时使用 FTS + pg_trgm，但明确不调用向量模型。
        return await _hybrid_rrf(db, query, kb_ids, top_k, include_vector=False)
    return await _hybrid_rrf(
        db,
        query,
        kb_ids,
        top_k,
        include_vector=True,
        trace_id=trace_id,
        surface=surface,
    )


async def _vector_search(
    db: AsyncSession,
    query: str,
    kb_ids: list[uuid.UUID],
    top_k: int,
) -> list[dict]:
    embedding = await embed_text(query)
    sql = text("""
        WITH vector_ann AS MATERIALIZED (
            -- 现有 2560 维索引是 halfvec HNSW。这一层只使用与索引
            -- 完全相同的 ORDER BY 表达式扩大召回，不加次级排序以免
            -- PostgreSQL 退化为全表排序。
            SELECT dc.id
            FROM document_chunks dc
            JOIN documents d
              ON d.id = dc.doc_id
             AND d.kb_id = dc.kb_id
            WHERE dc.kb_id = ANY(:kb_ids)
              AND dc.embedding IS NOT NULL
              AND d.is_active = TRUE
              AND d.status = 'ready'
            ORDER BY
                dc.embedding::halfvec(2560)
                    <=> CAST(:emb AS halfvec(2560))
            LIMIT :candidate_pool
        ),
        vector_candidates AS (
            SELECT
                dc.id, dc.content, dc.chunk_index, dc.metadata,
                dc.kb_id, dc.doc_id, d.filename, d.file_type, d.source_url,
                d.tags AS doc_tags,
                1 - (dc.embedding <=> CAST(:emb AS vector)) AS vector_score
            FROM vector_ann ann
            JOIN document_chunks dc ON dc.id = ann.id
            JOIN documents d
              ON d.id = dc.doc_id
             AND d.kb_id = dc.kb_id
            WHERE d.is_active = TRUE
              AND d.status = 'ready'
            -- HNSW 用 halfvec 快速召回，再在小候选池上用原始
            -- vector 距离精排，最大限度保留 2560 维精度。
            ORDER BY
                vector_score DESC,
                dc.doc_id ASC,
                dc.chunk_index ASC,
                dc.id ASC
        ),
        document_best AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY doc_id
                       ORDER BY vector_score DESC, chunk_index ASC, id ASC
                   ) AS document_chunk_rank
            FROM vector_candidates
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       ORDER BY vector_score DESC, doc_id ASC, id ASC
                   ) AS vector_rank
            FROM document_best
            WHERE document_chunk_rank = 1
        )
        SELECT
            id, content, chunk_index, metadata,
            kb_id, doc_id, filename, file_type, source_url, doc_tags,
            vector_score,
            vector_rank,
            NULL::double precision AS keyword_score,
            NULL::bigint AS keyword_rank,
            NULL::double precision AS trigram_score,
            NULL::bigint AS trigram_rank,
            vector_score AS retrieval_score,
            vector_score AS score
        FROM ranked
        ORDER BY vector_score DESC, doc_id ASC, chunk_index ASC, id ASC
        LIMIT :top_k
    """).bindparams(bindparam("kb_ids", type_=ARRAY(SA_UUID())))
    rows = (await db.execute(sql, {
        "emb": str(embedding),
        "kb_ids": kb_ids,
        "candidate_pool": _candidate_pool_size(top_k),
        "top_k": top_k,
    })).mappings().all()
    return _results(rows)


async def _keyword_search(
    db: AsyncSession,
    query: str,
    kb_ids: list[uuid.UUID],
    top_k: int,
) -> list[dict]:
    sql = text("""
        WITH keyword_candidates AS (
            SELECT
                dc.id, dc.content, dc.chunk_index, dc.metadata,
                dc.kb_id, dc.doc_id, d.filename, d.file_type, d.source_url,
                d.tags AS doc_tags,
                ts_rank(
                    to_tsvector('simple', dc.content),
                    plainto_tsquery('simple', :query)
                ) AS keyword_score
            FROM document_chunks dc
            JOIN documents d
              ON d.id = dc.doc_id
             AND d.kb_id = dc.kb_id
            WHERE dc.kb_id = ANY(:kb_ids)
              AND to_tsvector('simple', dc.content)
                  @@ plainto_tsquery('simple', :query)
              AND d.is_active = TRUE
              AND d.status = 'ready'
            ORDER BY keyword_score DESC, dc.doc_id ASC, dc.chunk_index ASC, dc.id ASC
            LIMIT :candidate_pool
        ),
        document_best AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY doc_id
                       ORDER BY keyword_score DESC, chunk_index ASC, id ASC
                   ) AS document_chunk_rank
            FROM keyword_candidates
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       ORDER BY keyword_score DESC, doc_id ASC, id ASC
                   ) AS keyword_rank
            FROM document_best
            WHERE document_chunk_rank = 1
        )
        SELECT
            id, content, chunk_index, metadata,
            kb_id, doc_id, filename, file_type, source_url, doc_tags,
            NULL::double precision AS vector_score,
            NULL::bigint AS vector_rank,
            keyword_score,
            keyword_rank,
            NULL::double precision AS trigram_score,
            NULL::bigint AS trigram_rank,
            keyword_score AS retrieval_score,
            keyword_score AS score
        FROM ranked
        ORDER BY keyword_score DESC, doc_id ASC, chunk_index ASC, id ASC
        LIMIT :top_k
    """).bindparams(bindparam("kb_ids", type_=ARRAY(SA_UUID())))
    rows = (await db.execute(sql, {
        "query": query,
        "kb_ids": kb_ids,
        "candidate_pool": _candidate_pool_size(top_k),
        "top_k": top_k,
    })).mappings().all()
    return _results(rows)


async def _hybrid_rrf(
    db: AsyncSession,
    query: str,
    kb_ids: list[uuid.UUID],
    top_k: int,
    *,
    include_vector: bool = True,
    trace_id: str | None = None,
    surface: str = "chat",
) -> list[dict]:
    embedding = None
    vector_enabled = include_vector
    if include_vector:
        try:
            embedding = await embed_text(query)
        except Exception as exc:
            # 向量服务故障时保留 FTS/pg_trgm 词面通道，避免整个混合检索不可用。
            vector_enabled = False
            logger.warning(
                "[检索降级] 向量通道不可用，继续执行关键词通道 error=%s",
                exception_log_text(exc),
            )
            if trace_id:
                trace_event(
                    "retrieval.channel_error",
                    trace_id=trace_id,
                    surface=surface,
                    channel="vector",
                    fallback_channels=["keyword", "trigram"],
                    error=exc,
                )
    trigram_terms = _build_trigram_terms(query)
    sql = text("""
        WITH vector_ann AS MATERIALIZED (
            SELECT dc.id
            FROM document_chunks dc
            JOIN documents d
              ON d.id = dc.doc_id
             AND d.kb_id = dc.kb_id
            WHERE dc.kb_id = ANY(:kb_ids)
              AND :vector_enabled
              AND dc.embedding IS NOT NULL
              AND d.is_active = TRUE
              AND d.status = 'ready'
            ORDER BY
                dc.embedding::halfvec(2560)
                    <=> CAST(:emb AS halfvec(2560))
            LIMIT :candidate_pool
        ),
        vector_candidates AS (
            SELECT
                dc.id, dc.doc_id, dc.chunk_index,
                1 - (dc.embedding <=> CAST(:emb AS vector)) AS vector_score
            FROM vector_ann ann
            JOIN document_chunks dc ON dc.id = ann.id
            ORDER BY
                vector_score DESC,
                dc.doc_id ASC,
                dc.chunk_index ASC,
                dc.id ASC
        ),
        vector_document_best AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY doc_id
                       ORDER BY vector_score DESC, chunk_index ASC, id ASC
                   ) AS document_chunk_rank
            FROM vector_candidates
        ),
        vector_r AS (
            SELECT id, doc_id, chunk_index, vector_score,
                   ROW_NUMBER() OVER (
                       ORDER BY vector_score DESC, doc_id ASC, id ASC
                   ) AS vector_rank
            FROM vector_document_best
            WHERE document_chunk_rank <= :per_document_chunks
        ),
        keyword_candidates AS (
            SELECT
                dc.id, dc.doc_id, dc.chunk_index,
                ts_rank(
                    to_tsvector('simple', dc.content),
                    plainto_tsquery('simple', :query)
                ) AS keyword_score
            FROM document_chunks dc
            JOIN documents d
              ON d.id = dc.doc_id
             AND d.kb_id = dc.kb_id
            WHERE dc.kb_id = ANY(:kb_ids)
              AND to_tsvector('simple', dc.content)
                  @@ plainto_tsquery('simple', :query)
              AND d.is_active = TRUE
              AND d.status = 'ready'
            ORDER BY keyword_score DESC, dc.doc_id ASC, dc.chunk_index ASC, dc.id ASC
            LIMIT :candidate_pool
        ),
        keyword_document_best AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY doc_id
                       ORDER BY keyword_score DESC, chunk_index ASC, id ASC
                   ) AS document_chunk_rank
            FROM keyword_candidates
        ),
        keyword_r AS (
            SELECT id, doc_id, chunk_index, keyword_score,
                   ROW_NUMBER() OVER (
                       ORDER BY keyword_score DESC, doc_id ASC, id ASC
                   ) AS keyword_rank
            FROM keyword_document_best
            WHERE document_chunk_rank <= :per_document_chunks
        ),
        trigram_scored AS (
            SELECT
                dc.id, dc.doc_id, dc.chunk_index,
                lexical.trigram_score
            FROM document_chunks dc
            JOIN documents d
              ON d.id = dc.doc_id
             AND d.kb_id = dc.kb_id
            CROSS JOIN LATERAL (
                SELECT MAX(
                    word_similarity(
                        LOWER(term),
                        LOWER(COALESCE(d.filename, '') || E'\\n' || dc.content)
                    ) * (
                        0.5 + 0.5 * LEAST(char_length(term), 12)::double precision / 12.0
                    )
                ) AS trigram_score
                FROM unnest(:trigram_terms) AS query_terms(term)
            ) lexical
            WHERE dc.kb_id = ANY(:kb_ids)
              AND d.is_active = TRUE
              AND d.status = 'ready'
        ),
        trigram_candidates AS (
            SELECT *
            FROM trigram_scored
            WHERE trigram_score >= :trigram_min_score
            ORDER BY trigram_score DESC, doc_id ASC, chunk_index ASC, id ASC
            LIMIT :candidate_pool
        ),
        trigram_document_best AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY doc_id
                       ORDER BY trigram_score DESC, chunk_index ASC, id ASC
                   ) AS document_chunk_rank
            FROM trigram_candidates
        ),
        trigram_r AS (
            SELECT id, doc_id, chunk_index, trigram_score,
                   ROW_NUMBER() OVER (
                       ORDER BY trigram_score DESC, doc_id ASC, id ASC
                   ) AS trigram_rank
            FROM trigram_document_best
            WHERE document_chunk_rank <= :per_document_chunks
        ),
        all_chunks AS (
            SELECT id, doc_id FROM vector_r
            UNION
            SELECT id, doc_id FROM keyword_r
            UNION
            SELECT id, doc_id FROM trigram_r
        ),
        fused AS (
            SELECT
                chunks.id AS representative_id,
                chunks.doc_id,
                v.vector_score,
                v.vector_rank,
                k.keyword_score,
                k.keyword_rank,
                t.trigram_score,
                t.trigram_rank,
                COALESCE(1.0 / (:rrf_k + v.vector_rank), 0.0)
                    + COALESCE(1.0 / (:rrf_k + k.keyword_rank), 0.0)
                    + COALESCE(1.0 / (:rrf_k + t.trigram_rank), 0.0)
                    AS retrieval_score,
                LEAST(
                    COALESCE(v.vector_rank, 2147483647),
                    COALESCE(k.keyword_rank, 2147483647),
                    COALESCE(t.trigram_rank, 2147483647)
                ) AS best_rank
            FROM all_chunks chunks
            LEFT JOIN vector_r v ON v.id = chunks.id
            LEFT JOIN keyword_r k ON k.id = chunks.id
            LEFT JOIN trigram_r t ON t.id = chunks.id
        ),
        fused_diverse AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY doc_id
                       ORDER BY retrieval_score DESC, best_rank ASC, representative_id ASC
                   ) AS fused_document_chunk_rank
            FROM fused
        )
        SELECT
            dc.id, dc.content, dc.chunk_index, dc.metadata,
            dc.kb_id, dc.doc_id, d.filename, d.file_type, d.source_url,
            d.tags AS doc_tags,
            fused_diverse.vector_score,
            fused_diverse.vector_rank,
            fused_diverse.keyword_score,
            fused_diverse.keyword_rank,
            fused_diverse.trigram_score,
            fused_diverse.trigram_rank,
            fused_diverse.retrieval_score,
            fused_diverse.retrieval_score AS score
        FROM fused_diverse
        JOIN document_chunks dc ON dc.id = fused_diverse.representative_id
        JOIN documents d
          ON d.id = dc.doc_id
         AND d.kb_id = dc.kb_id
        WHERE fused_diverse.fused_document_chunk_rank <= :per_document_chunks
          AND d.is_active = TRUE
          AND d.status = 'ready'
        ORDER BY
            fused_diverse.retrieval_score DESC,
            fused_diverse.best_rank ASC,
            fused_diverse.doc_id ASC,
            dc.chunk_index ASC,
            dc.id ASC
        LIMIT :top_k
    """).bindparams(
        bindparam("kb_ids", type_=ARRAY(SA_UUID())),
        bindparam("trigram_terms", type_=ARRAY(SA_Text())),
    )
    rows = (await db.execute(sql, {
        "emb": str(embedding) if embedding is not None else None,
        "vector_enabled": vector_enabled,
        "query": query,
        "trigram_terms": trigram_terms,
        "trigram_min_score": TRIGRAM_MIN_SCORE,
        "kb_ids": kb_ids,
        "candidate_pool": _candidate_pool_size(top_k),
        "per_document_chunks": PER_DOCUMENT_RERANK_CHUNKS,
        "rrf_k": RRF_K,
        "top_k": top_k,
    })).mappings().all()
    return _results(rows)
