import re
import uuid
import logging
import time

from sqlalchemy import Text as SA_Text
from sqlalchemy import bindparam, text
from sqlalchemy import UUID as SA_UUID
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
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
# 制度/表格问题经常需要“适用对象/等级映射 + 标准明细”多个片段共同闭合。
# 只保留 3 个片段会把同一文档中的关键表格行挤出重排池；扩大到 6 个，仍由
# 后续证据图和上下文预算负责最终筛选。
PER_DOCUMENT_RERANK_CHUNKS = 6

# 文档内二次检索只服务于首轮已经定位到的少量文档。这里的上限与全局召回分开，
# 避免通过扩大全局 Top K 来弥补跨片段问题，也避免异常计划扫描大量文档。
MAX_SCOPED_DOCUMENTS = 3
# 检索后澄清选择可能包含多个互斥范围。默认的证据扩展仍只允许 3 篇文档，
# 只有显式的范围选择重检索才可以提高该上限；总候选数、单文档候选数以及
# 大文档精确扫描 guard 仍沿用下方现有硬预算，不能随文档数线性膨胀。
MAX_EVIDENCE_SCOPE_DOCUMENTS = 30
MAX_SCOPED_QUERIES = 2
MAX_SCOPED_RESULTS = 12
MAX_SCOPED_RESULTS_PER_DOCUMENT = 4
MAX_SCOPED_QUERY_CHARS = 1000
MAX_STRUCTURAL_SEEDS = 4
MAX_STRUCTURAL_RESULTS = 12
# 精确向量距离与 word_similarity 都需要遍历目标文档片段。目标集合超过此硬阈值
# 时只保留可使用 GIN FTS 索引的词面通道和后续结构扩展，避免一次问题拖垮数据库。
MAX_SCOPED_EXACT_TOTAL_CHUNKS = 1200
MAX_SCOPED_EXACT_CHUNKS_PER_DOCUMENT = 600

# 小文档可以把全部片段加入“待联合重排”的候选池，但不能直接进入生成上下文。
# 阈值使用数据库中实际片段数和正文字符数校验，不信任可能滞后的 documents.chunk_count。
MAX_SMALL_DOCUMENT_CHUNKS = 30
MAX_SMALL_DOCUMENT_CHARS = 16_000

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

    # 同时加入去掉问句外壳后的版本；例如“我是产品8.6”会得到“产品8.6”，
    # “解决缓存失效问题”会得到“缓存失效问题”。
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
    diagnostics: dict | None = None,
) -> list[dict]:
    if diagnostics is not None:
        diagnostics.update(
            requested_method=method,
            vector_channel_failed=False,
            vector_error_type=None,
        )
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
        diagnostics=diagnostics,
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
    diagnostics: dict | None = None,
) -> list[dict]:
    embedding = None
    vector_enabled = include_vector
    if include_vector:
        try:
            embedding = await embed_text(query)
        except Exception as exc:
            # 向量服务故障时保留 FTS/pg_trgm 词面通道，避免整个混合检索不可用。
            vector_enabled = False
            if diagnostics is not None:
                diagnostics["vector_channel_failed"] = True
                diagnostics["vector_error_type"] = type(exc).__name__
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


def _bounded_unique(values: list, limit: int) -> list:
    """按输入顺序去重并施加硬上限，兼容 UUID 与字符串等可比较值。"""

    bounded: list = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        identity = str(value)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        bounded.append(value)
        if len(bounded) >= limit:
            break
    return bounded


def _bounded_scoped_queries(queries: list[str]) -> list[str]:
    bounded: list[str] = []
    seen: set[str] = set()
    for query in queries:
        normalized = re.sub(r"\s+", " ", str(query or "")).strip()
        if not normalized:
            continue
        normalized = normalized[:MAX_SCOPED_QUERY_CHARS]
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        bounded.append(normalized)
        if len(bounded) >= MAX_SCOPED_QUERIES:
            break
    return bounded


def _candidate_identity(item: dict) -> str:
    chunk_id = item.get("id")
    if chunk_id:
        return f"id:{chunk_id}"
    return f"position:{item.get('doc_id')}:{item.get('chunk_index')}"


def _merge_string_list(*values) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            text_value = str(item or "").strip()
            if not text_value or text_value in seen:
                continue
            seen.add(text_value)
            merged.append(text_value)
    return merged


def _safe_numeric(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


async def _execute_in_savepoint(db: AsyncSession, statement, params: dict):
    """把单条可降级 SQL 隔离在 SAVEPOINT 中。

    PostgreSQL 任意 statement 失败都会把当前事务标记为 aborted。真实 AsyncSession
    使用 ``begin_nested`` 回滚到保存点；极简测试 fake 没有该方法时直接执行，避免
    让测试替身被迫完整模拟 SQLAlchemy 事务对象。
    """

    begin_nested = getattr(db, "begin_nested", None)
    if not callable(begin_nested):
        return await db.execute(statement, params)
    async with begin_nested():
        return await db.execute(statement, params)


async def _scoped_document_chunk_stats(
    db: AsyncSession,
    *,
    kb_ids: list[uuid.UUID],
    doc_ids: list[uuid.UUID],
) -> tuple[int, int]:
    """使用 documents 上的计数做廉价预检，不读取片段正文或向量。"""

    sql = text("""
        SELECT
            COALESCE(SUM(GREATEST(COALESCE(d.chunk_count, 0), 0)), 0)
                AS total_chunk_count,
            COALESCE(MAX(GREATEST(COALESCE(d.chunk_count, 0), 0)), 0)
                AS max_document_chunk_count
        FROM documents d
        WHERE d.id = ANY(:doc_ids)
          AND d.kb_id = ANY(:kb_ids)
          AND d.is_active = TRUE
          AND d.status = 'ready'
    """).bindparams(
        bindparam("kb_ids", type_=ARRAY(SA_UUID())),
        bindparam("doc_ids", type_=ARRAY(SA_UUID())),
    )
    result = await _execute_in_savepoint(db, sql, {
        "kb_ids": kb_ids,
        "doc_ids": doc_ids,
    })
    rows = result.mappings().all()
    row = dict(rows[0]) if rows else {}
    try:
        total = max(0, int(row.get("total_chunk_count") or 0))
    except (TypeError, ValueError):
        total = 0
    try:
        per_document = max(0, int(row.get("max_document_chunk_count") or 0))
    except (TypeError, ValueError):
        per_document = 0
    return total, per_document


async def fetch_small_document_candidates(
    db: AsyncSession,
    *,
    kb_ids: list[uuid.UUID],
    doc_ids: list[uuid.UUID],
    max_chunks: int = MAX_SMALL_DOCUMENT_CHUNKS,
    max_chars: int = MAX_SMALL_DOCUMENT_CHARS,
    trace_id: str | None = None,
) -> list[dict]:
    """加载目标小文档的全部片段，供后续联合重排使用。

    ``doc_ids`` 必须来自调用方已经定位的目标文档，且查询始终同时受当前授权
    ``kb_ids``、文档 ``ready/active`` 状态约束。片段数和字符数使用同一数据库
    快照中的实际 chunks 统计；单篇或多篇合计超过调用预算时整篇跳过，绝不返回
    半篇文档。返回值只是候选，不代表这些片段可以直接进入生成上下文。

    调用方可以收紧 ``max_chunks``/``max_chars``，但不能突破模块的 30 片段、
    16000 字符硬上限。最多检查 ``MAX_SCOPED_DOCUMENTS`` 篇目标文档。
    """

    started_at = time.perf_counter()
    scoped_kb_ids = _bounded_unique(kb_ids, 100)
    scoped_doc_ids = _bounded_unique(doc_ids, MAX_SCOPED_DOCUMENTS)
    if not scoped_kb_ids or not scoped_doc_ids:
        return []

    bounded_chunks = max(1, min(int(max_chunks), MAX_SMALL_DOCUMENT_CHUNKS))
    bounded_chars = max(1, min(int(max_chars), MAX_SMALL_DOCUMENT_CHARS))
    # 每篇文档最多返回 bounded_chunks 条；文档数本身也有硬上限，因此 SQL
    # 返回规模仍然有界。Python 再按 doc_ids 输入顺序执行跨文档整体预算，确保
    # 不会因为 LIMIT 截断而把半篇文档误当作“完整全文”。
    row_limit = bounded_chunks * len(scoped_doc_ids)
    sql = text("""
        WITH requested_documents AS MATERIALIZED (
            SELECT requested.doc_id, requested.doc_order
            FROM unnest(:doc_ids) WITH ORDINALITY
                AS requested(doc_id, doc_order)
        ),
        eligible_documents AS MATERIALIZED (
            SELECT
                d.id, d.kb_id, d.filename, d.file_type, d.source_url,
                d.tags AS doc_tags, requested.doc_order,
                COUNT(probed.id)::integer AS actual_chunk_count,
                COALESCE(
                    SUM(char_length(COALESCE(probed.content, ''))),
                    0
                )::bigint
                    AS actual_char_count
            FROM requested_documents requested
            JOIN documents d ON d.id = requested.doc_id
            JOIN LATERAL (
                SELECT dc.id, dc.content
                FROM document_chunks dc
                WHERE dc.doc_id = d.id
                  AND dc.kb_id = d.kb_id
                ORDER BY dc.chunk_index, dc.id
                LIMIT :probe_chunk_limit
            ) probed ON TRUE
            WHERE d.id = ANY(:doc_ids)
              AND d.kb_id = ANY(:kb_ids)
              AND d.is_active = TRUE
              AND d.status = 'ready'
            GROUP BY
                d.id, d.kb_id, d.filename, d.file_type, d.source_url,
                d.tags, requested.doc_order
            HAVING COUNT(probed.id) BETWEEN 1 AND :max_chunks
               AND COALESCE(
                    SUM(char_length(COALESCE(probed.content, ''))),
                    0
               ) <= :max_chars
        )
        SELECT
            dc.id, dc.content, dc.chunk_index, dc.metadata,
            dc.kb_id, dc.doc_id,
            eligible.filename, eligible.file_type, eligible.source_url,
            eligible.doc_tags,
            NULL::double precision AS vector_score,
            NULL::bigint AS vector_rank,
            NULL::double precision AS keyword_score,
            NULL::bigint AS keyword_rank,
            NULL::double precision AS trigram_score,
            NULL::bigint AS trigram_rank,
            NULL::double precision AS retrieval_score,
            NULL::double precision AS score,
            eligible.doc_order,
            eligible.actual_chunk_count,
            eligible.actual_char_count
        FROM eligible_documents eligible
        JOIN document_chunks dc
          ON dc.doc_id = eligible.id
         AND dc.kb_id = eligible.kb_id
        WHERE dc.doc_id = ANY(:doc_ids)
          AND dc.kb_id = ANY(:kb_ids)
        ORDER BY eligible.doc_order, dc.chunk_index, dc.id
        LIMIT :row_limit
    """).bindparams(
        bindparam("kb_ids", type_=ARRAY(SA_UUID())),
        bindparam("doc_ids", type_=ARRAY(SA_UUID())),
    )
    result = await _execute_in_savepoint(db, sql, {
        "kb_ids": scoped_kb_ids,
        "doc_ids": scoped_doc_ids,
        "max_chunks": bounded_chunks,
        "max_chars": bounded_chars,
        "probe_chunk_limit": bounded_chunks + 1,
        "row_limit": row_limit,
    })
    rows = [dict(row) for row in result.mappings().all()]

    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    allowed_doc_ids = {str(value) for value in scoped_doc_ids}
    allowed_kb_ids = {str(value) for value in scoped_kb_ids}
    for row in rows:
        doc_key = str(row.get("doc_id") or "")
        kb_key = str(row.get("kb_id") or "")
        if doc_key not in allowed_doc_ids or kb_key not in allowed_kb_ids:
            continue
        if doc_key not in grouped:
            grouped[doc_key] = []
            order.append(doc_key)
        grouped[doc_key].append(row)

    selected: list[dict] = []
    loaded_doc_count = 0
    skipped_by_budget = 0
    used_chunks = 0
    used_chars = 0
    for doc_key in order:
        document_rows = grouped[doc_key]
        first = document_rows[0]
        try:
            expected_chunks = int(first.get("actual_chunk_count") or 0)
            expected_chars = int(first.get("actual_char_count") or 0)
        except (TypeError, ValueError):
            continue
        actual_chars = sum(len(str(row.get("content") or "")) for row in document_rows)
        # 防止测试替身、异常驱动返回或并发脏数据造成半篇候选。正常 PostgreSQL
        # 单语句快照下这里应始终相等。
        if (
            expected_chunks != len(document_rows)
            or expected_chars != actual_chars
            or expected_chunks < 1
            or expected_chunks > bounded_chunks
            or expected_chars > bounded_chars
        ):
            logger.warning(
                "[小文档全文候选] 完整性校验失败 doc=%s expected_chunks=%s "
                "actual_chunks=%s expected_chars=%s actual_chars=%s",
                doc_key,
                expected_chunks,
                len(document_rows),
                expected_chars,
                actual_chars,
            )
            continue
        if (
            used_chunks + expected_chunks > bounded_chunks
            or used_chars + expected_chars > bounded_chars
        ):
            skipped_by_budget += 1
            continue

        for row in document_rows:
            item = _normalize_result_scores(row)
            item.pop("doc_order", None)
            item.pop("actual_chunk_count", None)
            item.pop("actual_char_count", None)
            item["candidate_origin"] = "small_document_full"
            item["candidate_origins"] = ["small_document_full"]
            item["full_document_chunk_count"] = expected_chunks
            item["full_document_char_count"] = expected_chars
            selected.append(item)
        used_chunks += expected_chunks
        used_chars += expected_chars
        loaded_doc_count += 1

    if trace_id:
        trace_event(
            "retrieval.small_document_candidates_completed",
            trace_id=trace_id,
            requested_document_count=len(scoped_doc_ids),
            eligible_document_count=len(grouped),
            loaded_document_count=loaded_doc_count,
            skipped_by_budget_document_count=skipped_by_budget,
            candidate_count=len(selected),
            candidate_chars=used_chars,
            max_chunks=bounded_chunks,
            max_chars=bounded_chars,
            elapsed_ms=round((time.perf_counter() - started_at) * 1000),
        )
    return selected


def _merge_scoped_query_candidates(
    query_results: list[tuple[int, list[dict]]],
    *,
    per_document_limit: int,
    total_limit: int,
) -> list[dict]:
    """合并至多两条文档内查询结果，同一 chunk 不重复占用重排预算。"""

    merged: dict[str, dict] = {}
    for query_index, results in query_results:
        for result in results:
            item = dict(result)
            identity = _candidate_identity(item)
            item["candidate_origin"] = "document_scoped"
            item["candidate_origins"] = _merge_string_list(
                item.get("candidate_origins"),
                item.get("candidate_origin"),
            )
            item["expansion_query_indexes"] = [query_index]
            item["document_scoped_score"] = item.get("retrieval_score")

            current = merged.get(identity)
            if current is None:
                merged[identity] = item
                continue

            current["candidate_origins"] = _merge_string_list(
                current.get("candidate_origins"),
                item.get("candidate_origins"),
            )
            current["expansion_query_indexes"] = sorted({
                *current.get("expansion_query_indexes", []),
                query_index,
            })
            current["active_channels"] = _merge_string_list(
                current.get("active_channels"),
                item.get("active_channels"),
            )

            # 多查询命中只保留各通道的最佳观测值；不会把相邻/章节来源伪装成
            # 更高的语义分数。query 命中次数仅用于确定性平局排序。
            for score_field in (
                "retrieval_score",
                "score",
                "document_scoped_score",
                "vector_score",
                "keyword_score",
                "trigram_score",
            ):
                existing = _safe_numeric(current.get(score_field))
                incoming = _safe_numeric(item.get(score_field))
                if incoming is not None and (existing is None or incoming > existing):
                    current[score_field] = item.get(score_field)
            for rank_field in ("vector_rank", "keyword_rank", "trigram_rank"):
                existing = _safe_numeric(current.get(rank_field))
                incoming = _safe_numeric(item.get(rank_field))
                if incoming is not None and (existing is None or incoming < existing):
                    current[rank_field] = item.get(rank_field)

    ordered = sorted(
        merged.values(),
        key=lambda item: (
            -(_safe_numeric(item.get("document_scoped_score")) or 0.0),
            -len(item.get("expansion_query_indexes") or []),
            str(item.get("doc_id") or ""),
            int(item.get("chunk_index") or 0),
            str(item.get("id") or ""),
        ),
    )
    doc_counts: dict[str, int] = {}
    selected: list[dict] = []
    for item in ordered:
        doc_key = str(item.get("doc_id") or "")
        if doc_counts.get(doc_key, 0) >= per_document_limit:
            continue
        doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
        selected.append(item)
        if len(selected) >= total_limit:
            break
    return selected


async def _search_within_documents_once(
    db: AsyncSession,
    *,
    query: str,
    kb_ids: list[uuid.UUID],
    doc_ids: list[uuid.UUID],
    method: str,
    per_document_limit: int,
    total_limit: int,
    allow_expensive_channels: bool,
    trace_id: str | None,
    surface: str,
) -> tuple[list[dict], bool]:
    """在已经定位的文档小集合上做精确混合检索。"""

    vector_requested = allow_expensive_channels and method in {"hybrid", "vector"}
    keyword_enabled = method in {"hybrid", "keyword"}
    trigram_enabled = allow_expensive_channels and keyword_enabled
    if not allow_expensive_channels:
        # 大文档 guard 下即使用户选择向量模式，也只在首轮定位文档中使用可走
        # GIN 的 FTS；结构扩展随后仍可补充种子附近内容。
        keyword_enabled = True
        trigram_enabled = False
    vector_enabled = vector_requested
    vector_fallback = False
    embedding = None
    if vector_requested:
        try:
            embedding = await embed_text(query)
        except Exception as exc:
            # 二次检索的目标是补齐证据。即使用户选择了向量模式，向量服务故障
            # 时也保留限定文档内的 FTS/trigram，且绝不退回全局扫描。
            vector_enabled = False
            keyword_enabled = True
            trigram_enabled = True
            vector_fallback = True
            logger.warning(
                "[文档内检索降级] 向量通道不可用，限定文档内继续执行词面通道 error=%s",
                exception_log_text(exc),
            )
            if trace_id:
                trace_event(
                    "retrieval.channel_error",
                    trace_id=trace_id,
                    surface=surface,
                    stage="document_scoped",
                    channel="vector",
                    fallback_channels=["keyword", "trigram"],
                    error=exc,
                )

    trigram_terms = _build_trigram_terms(query) if trigram_enabled else []
    channel_limit = min(MAX_SCOPED_RESULTS * 4, max(16, total_limit * 4))
    sql = text("""
        WITH vector_candidates AS MATERIALIZED (
            SELECT
                dc.id, dc.doc_id, dc.chunk_index,
                1 - (dc.embedding <=> CAST(:emb AS vector)) AS vector_score
            FROM document_chunks dc
            JOIN documents d
              ON d.id = dc.doc_id
             AND d.kb_id = dc.kb_id
            WHERE :vector_enabled
              AND dc.doc_id = ANY(:doc_ids)
              AND dc.kb_id = ANY(:kb_ids)
              AND dc.embedding IS NOT NULL
              AND d.is_active = TRUE
              AND d.status = 'ready'
            ORDER BY vector_score DESC, dc.doc_id, dc.chunk_index, dc.id
            LIMIT :channel_limit
        ),
        vector_r AS (
            SELECT id, vector_score,
                   ROW_NUMBER() OVER (
                       ORDER BY vector_score DESC, doc_id ASC, chunk_index ASC, id ASC
                   ) AS vector_rank
            FROM vector_candidates
        ),
        keyword_candidates AS MATERIALIZED (
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
            WHERE :keyword_enabled
              AND dc.doc_id = ANY(:doc_ids)
              AND dc.kb_id = ANY(:kb_ids)
              AND to_tsvector('simple', dc.content)
                  @@ plainto_tsquery('simple', :query)
              AND d.is_active = TRUE
              AND d.status = 'ready'
            ORDER BY keyword_score DESC, dc.doc_id, dc.chunk_index, dc.id
            LIMIT :channel_limit
        ),
        keyword_r AS (
            SELECT id, keyword_score,
                   ROW_NUMBER() OVER (
                       ORDER BY keyword_score DESC, doc_id ASC, chunk_index ASC, id ASC
                   ) AS keyword_rank
            FROM keyword_candidates
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
            WHERE :trigram_enabled
              AND dc.doc_id = ANY(:doc_ids)
              AND dc.kb_id = ANY(:kb_ids)
              AND d.is_active = TRUE
              AND d.status = 'ready'
        ),
        trigram_candidates AS MATERIALIZED (
            SELECT *
            FROM trigram_scored
            WHERE trigram_score >= :trigram_min_score
            ORDER BY trigram_score DESC, doc_id, chunk_index, id
            LIMIT :channel_limit
        ),
        trigram_r AS (
            SELECT id, trigram_score,
                   ROW_NUMBER() OVER (
                       ORDER BY trigram_score DESC, doc_id ASC, chunk_index ASC, id ASC
                   ) AS trigram_rank
            FROM trigram_candidates
        ),
        all_chunks AS (
            SELECT id FROM vector_r
            UNION
            SELECT id FROM keyword_r
            UNION
            SELECT id FROM trigram_r
        ),
        fused AS (
            SELECT
                chunks.id,
                v.vector_score,
                v.vector_rank,
                k.keyword_score,
                k.keyword_rank,
                t.trigram_score,
                t.trigram_rank,
                COALESCE(1.0 / (:rrf_k + v.vector_rank), 0.0)
                    + COALESCE(1.0 / (:rrf_k + k.keyword_rank), 0.0)
                    + COALESCE(1.0 / (:rrf_k + t.trigram_rank), 0.0)
                    AS retrieval_score
            FROM all_chunks chunks
            LEFT JOIN vector_r v ON v.id = chunks.id
            LEFT JOIN keyword_r k ON k.id = chunks.id
            LEFT JOIN trigram_r t ON t.id = chunks.id
        ),
        document_diverse AS (
            SELECT
                fused.*,
                dc.doc_id,
                dc.chunk_index,
                ROW_NUMBER() OVER (
                    PARTITION BY dc.doc_id
                    ORDER BY fused.retrieval_score DESC,
                             dc.chunk_index ASC,
                             fused.id ASC
                ) AS document_chunk_rank
            FROM fused
            JOIN document_chunks dc ON dc.id = fused.id
        )
        SELECT
            dc.id, dc.content, dc.chunk_index, dc.metadata,
            dc.kb_id, dc.doc_id, d.filename, d.file_type,
            d.source_url, d.tags AS doc_tags,
            diverse.vector_score, diverse.vector_rank,
            diverse.keyword_score, diverse.keyword_rank,
            diverse.trigram_score, diverse.trigram_rank,
            diverse.retrieval_score,
            diverse.retrieval_score AS score
        FROM document_diverse diverse
        JOIN document_chunks dc ON dc.id = diverse.id
        JOIN documents d
          ON d.id = dc.doc_id
         AND d.kb_id = dc.kb_id
        WHERE diverse.document_chunk_rank <= :per_document_limit
          AND dc.doc_id = ANY(:doc_ids)
          AND dc.kb_id = ANY(:kb_ids)
          AND d.is_active = TRUE
          AND d.status = 'ready'
        ORDER BY diverse.retrieval_score DESC,
                 dc.doc_id ASC,
                 dc.chunk_index ASC,
                 dc.id ASC
        LIMIT :total_limit
    """).bindparams(
        bindparam("kb_ids", type_=ARRAY(SA_UUID())),
        bindparam("doc_ids", type_=ARRAY(SA_UUID())),
        bindparam("trigram_terms", type_=ARRAY(SA_Text())),
    )
    result = await _execute_in_savepoint(db, sql, {
        "query": query,
        "emb": str(embedding) if embedding is not None else None,
        "vector_enabled": vector_enabled,
        "keyword_enabled": keyword_enabled,
        "trigram_enabled": trigram_enabled and bool(trigram_terms),
        "trigram_terms": trigram_terms,
        "trigram_min_score": TRIGRAM_MIN_SCORE,
        "kb_ids": kb_ids,
        "doc_ids": doc_ids,
        "rrf_k": RRF_K,
        "channel_limit": channel_limit,
        "per_document_limit": per_document_limit,
        "total_limit": total_limit,
    })
    rows = result.mappings().all()
    return _results(rows), vector_fallback


async def search_within_documents(
    db: AsyncSession,
    *,
    queries: list[str],
    kb_ids: list[uuid.UUID],
    doc_ids: list[uuid.UUID],
    method: str = "hybrid",
    per_document_limit: int = MAX_SCOPED_RESULTS_PER_DOCUMENT,
    total_limit: int = 8,
    max_document_count: int = MAX_SCOPED_DOCUMENTS,
    trace_id: str | None = None,
    surface: str = "chat",
) -> list[dict]:
    """在首轮定位文档内执行至多两条精确混合查询。

    与全局 ``hybrid_search`` 不同，本函数不会扩大知识库 Top K，也不会把整篇
    文档返回给生成模型。候选始终同时受 ``doc_ids``、当前授权 ``kb_ids``、文档
    ready/active 状态以及每文档/总候选预算约束。
    """

    started_at = time.perf_counter()
    scoped_queries = _bounded_scoped_queries(queries)
    scoped_kb_ids = _bounded_unique(kb_ids, 100)
    bounded_document_count = max(
        1,
        min(int(max_document_count), MAX_EVIDENCE_SCOPE_DOCUMENTS),
    )
    scoped_doc_ids = _bounded_unique(doc_ids, bounded_document_count)
    if not scoped_queries or not scoped_kb_ids or not scoped_doc_ids:
        return []

    normalized_method = method if method in {"hybrid", "vector", "keyword"} else "hybrid"
    bounded_per_document = max(
        1,
        min(int(per_document_limit), MAX_SCOPED_RESULTS_PER_DOCUMENT),
    )
    bounded_total = max(1, min(int(total_limit), MAX_SCOPED_RESULTS))
    scoped_chunk_count: int | None = None
    max_document_chunk_count: int | None = None
    scan_guard_reason: str | None = None
    try:
        scoped_chunk_count, max_document_chunk_count = await _scoped_document_chunk_stats(
            db,
            kb_ids=scoped_kb_ids,
            doc_ids=scoped_doc_ids,
        )
        if scoped_chunk_count > MAX_SCOPED_EXACT_TOTAL_CHUNKS:
            scan_guard_reason = "total_chunk_limit"
        elif max_document_chunk_count > MAX_SCOPED_EXACT_CHUNKS_PER_DOCUMENT:
            scan_guard_reason = "per_document_chunk_limit"
    except Exception as exc:
        # 无法确认规模时采用保守 guard，仍允许索引化 FTS 和结构扩展继续工作。
        scan_guard_reason = "chunk_stats_unavailable"
        logger.warning(
            "[文档内检索保护] 无法读取目标文档规模，跳过精确向量/trigram error=%s",
            exception_log_text(exc),
        )

    scan_guard_triggered = scan_guard_reason is not None
    if scan_guard_triggered:
        logger.info(
            "[文档内检索保护] 已启用 reason=%s documents=%d total_chunks=%s "
            "max_document_chunks=%s；仅执行索引化 FTS + 结构扩展",
            scan_guard_reason,
            len(scoped_doc_ids),
            scoped_chunk_count,
            max_document_chunk_count,
        )

    query_results: list[tuple[int, list[dict]]] = []
    vector_fallback_count = 0
    query_errors: list[Exception] = []
    successful_query_count = 0
    for query_index, query in enumerate(scoped_queries):
        try:
            results, vector_fallback = await _search_within_documents_once(
                db,
                query=query,
                kb_ids=scoped_kb_ids,
                doc_ids=scoped_doc_ids,
                method=normalized_method,
                per_document_limit=bounded_per_document,
                total_limit=bounded_total,
                allow_expensive_channels=not scan_guard_triggered,
                trace_id=trace_id,
                surface=surface,
            )
        except Exception as exc:
            query_errors.append(exc)
            logger.warning(
                "[文档内检索] 子查询失败 query_index=%d/%d，继续其它子查询 error=%s",
                query_index + 1,
                len(scoped_queries),
                exception_log_text(exc),
            )
            continue
        successful_query_count += 1
        vector_fallback_count += int(vector_fallback)
        query_results.append((query_index, results))

    if successful_query_count == 0 and query_errors:
        if trace_id:
            trace_event(
                "retrieval.document_scoped_completed",
                trace_id=trace_id,
                succeeded=False,
                query_count=len(scoped_queries),
                successful_query_count=0,
                failed_query_count=len(query_errors),
                scoped_document_count=len(scoped_doc_ids),
                candidate_count=0,
                vector_fallback_count=vector_fallback_count,
                scan_guard_triggered=scan_guard_triggered,
                scan_guard_reason=scan_guard_reason,
                scoped_chunk_count=scoped_chunk_count,
                max_document_chunk_count=max_document_chunk_count,
                elapsed_ms=round((time.perf_counter() - started_at) * 1000),
                error=query_errors[-1],
            )
        raise query_errors[-1]

    merged = _merge_scoped_query_candidates(
        query_results,
        per_document_limit=bounded_per_document,
        total_limit=bounded_total,
    )
    if trace_id:
        trace_event(
            "retrieval.document_scoped_completed",
            trace_id=trace_id,
            succeeded=True,
            query_count=len(scoped_queries),
            successful_query_count=successful_query_count,
            failed_query_count=len(query_errors),
            scoped_document_count=len(scoped_doc_ids),
            candidate_count=len(merged),
            vector_fallback_count=vector_fallback_count,
            scan_guard_triggered=scan_guard_triggered,
            scan_guard_reason=scan_guard_reason,
            scoped_chunk_count=scoped_chunk_count,
            max_document_chunk_count=max_document_chunk_count,
            channel_candidate_counts={
                channel: sum(
                    channel in (item.get("active_channels") or []) for item in merged
                )
                for channel in ("vector", "keyword", "trigram")
            },
            elapsed_ms=round((time.perf_counter() - started_at) * 1000),
        )
    return merged


def _seed_specs(seed_candidates: list[dict]) -> tuple[list[dict], list]:
    specs: list[dict] = []
    doc_ids: list = []
    seen: set[str] = set()
    for candidate in seed_candidates:
        try:
            chunk_id = uuid.UUID(str(candidate.get("id")))
            doc_id = uuid.UUID(str(candidate.get("doc_id")))
            chunk_index = int(candidate.get("chunk_index"))
        except (TypeError, ValueError, AttributeError):
            continue
        identity = str(chunk_id)
        if identity in seen:
            continue
        seen.add(identity)
        metadata = candidate.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        section_key = str(metadata.get("section_key") or "").strip() or None
        heading = str(metadata.get("heading") or "").strip() or None
        table_id = str(metadata.get("table_id") or "").strip() or None
        table_part_index = metadata.get("table_part_index")
        try:
            table_part_index = (
                int(table_part_index) if table_part_index is not None else None
            )
        except (TypeError, ValueError):
            table_part_index = None
        block_type = str(
            metadata.get("block_type") or metadata.get("type") or ""
        ).strip().lower() or None
        specs.append({
            "seed_order": len(specs),
            "seed_chunk_id": str(chunk_id),
            "doc_id": str(doc_id),
            "chunk_index": chunk_index,
            "section_key": section_key,
            "heading": heading,
            "table_id": table_id,
            "table_part_index": table_part_index,
            "block_type": block_type,
        })
        doc_ids.append(doc_id)
        if len(specs) >= MAX_STRUCTURAL_SEEDS:
            break
    return specs, _bounded_unique(doc_ids, MAX_SCOPED_DOCUMENTS)


def _merge_structural_rows(rows: list[dict], total_limit: int) -> list[dict]:
    merged: dict[str, dict] = {}
    origin_priority = {"table_sibling": 0, "same_section": 1, "adjacent": 2}
    for raw in rows:
        item = _normalize_result_scores(dict(raw))
        origin = str(item.pop("structural_origin", "") or "").strip()
        seed_chunk_id = str(item.pop("seed_chunk_id", "") or "").strip()
        distance = item.pop("structure_distance", None)
        if not origin:
            continue
        item["candidate_origin"] = origin
        item["candidate_origins"] = [origin]
        item["expansion_seed_chunk_ids"] = [seed_chunk_id] if seed_chunk_id else []
        item["expansion_sources"] = [{
            "origin": origin,
            "seed_chunk_id": seed_chunk_id or None,
            "distance": distance,
        }]
        if origin == "adjacent":
            item["neighbor_distance"] = distance
        identity = _candidate_identity(item)
        current = merged.get(identity)
        if current is None:
            merged[identity] = item
            continue
        current["candidate_origins"] = _merge_string_list(
            current.get("candidate_origins"), origin
        )
        current["expansion_seed_chunk_ids"] = _merge_string_list(
            current.get("expansion_seed_chunk_ids"), seed_chunk_id
        )
        source_key = {
            (source.get("origin"), source.get("seed_chunk_id"), source.get("distance"))
            for source in current.get("expansion_sources", [])
            if isinstance(source, dict)
        }
        new_source = (origin, seed_chunk_id or None, distance)
        if new_source not in source_key:
            current.setdefault("expansion_sources", []).append({
                "origin": origin,
                "seed_chunk_id": seed_chunk_id or None,
                "distance": distance,
            })

    ordered = sorted(
        merged.values(),
        key=lambda item: (
            min(
                origin_priority.get(origin, 99)
                for origin in (item.get("candidate_origins") or [""])
            ),
            str(item.get("doc_id") or ""),
            int(item.get("chunk_index") or 0),
            str(item.get("id") or ""),
        ),
    )
    return ordered[:total_limit]


async def fetch_structural_neighbors(
    db: AsyncSession,
    *,
    kb_ids: list[uuid.UUID],
    seed_candidates: list[dict],
    neighbor_radius: int = 1,
    same_section_limit: int = 2,
    table_sibling_radius: int = 1,
    total_limit: int = 8,
    trace_id: str | None = None,
) -> list[dict]:
    """有界补充种子片段的相邻、同章节与同表格分片。

    邻接和表格分片半径硬限制为 1；同章节按与种子 ``chunk_index`` 的距离取前
    两条。新版 metadata 优先使用 section_key/table_id，旧文档则回退到
    heading/type，不会为缺失结构信息而加载整篇文档。
    """

    started_at = time.perf_counter()
    scoped_kb_ids = _bounded_unique(kb_ids, 100)
    specs, scoped_doc_ids = _seed_specs(seed_candidates)
    if not specs or not scoped_kb_ids or not scoped_doc_ids:
        return []

    neighbors_enabled = int(neighbor_radius) > 0
    tables_enabled = int(table_sibling_radius) > 0
    bounded_section_limit = max(0, min(int(same_section_limit), 2))
    bounded_total = max(1, min(int(total_limit), MAX_STRUCTURAL_RESULTS))
    # 同一 chunk 可能同时属于相邻、同章节和表格分片，SQL 为保留全部来源允许
    # 最多返回最终预算的三倍，Python 再按 chunk id 合并并执行最终硬上限。
    sql_row_limit = min(MAX_STRUCTURAL_SEEDS * 6, bounded_total * 3)

    sql = text("""
        WITH seeds AS MATERIALIZED (
            SELECT *
            FROM jsonb_to_recordset(:seed_specs) AS seed(
                seed_order integer,
                seed_chunk_id uuid,
                doc_id uuid,
                chunk_index integer,
                section_key text,
                heading text,
                table_id text,
                table_part_index integer,
                block_type text
            )
        ),
        eligible_documents AS MATERIALIZED (
            SELECT
                d.id, d.kb_id, d.filename, d.file_type, d.source_url,
                d.tags AS doc_tags
            FROM documents d
            WHERE d.id = ANY(:doc_ids)
              AND d.kb_id = ANY(:kb_ids)
              AND d.is_active = TRUE
              AND d.status = 'ready'
        ),
        adjacent_matches AS (
            SELECT
                candidate.id, seed.seed_chunk_id,
                'adjacent'::text AS structural_origin,
                candidate.structure_distance,
                seed.seed_order
            FROM seeds seed
            CROSS JOIN LATERAL (
                SELECT
                    dc.id,
                    ABS(dc.chunk_index - seed.chunk_index) AS structure_distance
                FROM document_chunks dc
                JOIN eligible_documents d
                  ON d.id = dc.doc_id
                 AND d.kb_id = dc.kb_id
                WHERE :neighbors_enabled
                  AND dc.doc_id = seed.doc_id
                  AND dc.doc_id = ANY(:doc_ids)
                  AND dc.kb_id = ANY(:kb_ids)
                  AND dc.id <> seed.seed_chunk_id
                  AND dc.chunk_index IN (
                      seed.chunk_index - 1,
                      seed.chunk_index + 1
                  )
                ORDER BY dc.chunk_index, dc.id
                LIMIT 2
            ) candidate
        ),
        section_key_matches AS (
            SELECT
                candidate.id, seed.seed_chunk_id,
                'same_section'::text AS structural_origin,
                candidate.structure_distance,
                seed.seed_order
            FROM seeds seed
            CROSS JOIN LATERAL (
                SELECT
                    nearby.id,
                    ABS(nearby.chunk_index - seed.chunk_index)
                        AS structure_distance
                FROM (
                    (
                        SELECT dc.id, dc.chunk_index
                        FROM document_chunks dc
                        JOIN eligible_documents d
                          ON d.id = dc.doc_id
                         AND d.kb_id = dc.kb_id
                        WHERE :same_section_enabled
                          AND seed.section_key IS NOT NULL
                          AND dc.doc_id = seed.doc_id
                          AND dc.doc_id = ANY(:doc_ids)
                          AND dc.kb_id = ANY(:kb_ids)
                          AND dc.id <> seed.seed_chunk_id
                          AND dc.metadata ? 'section_key'
                          AND dc.metadata->>'section_key' = seed.section_key
                          AND dc.chunk_index < seed.chunk_index
                        ORDER BY dc.chunk_index DESC, dc.id
                        LIMIT :same_section_limit
                    )
                    UNION ALL
                    (
                        SELECT dc.id, dc.chunk_index
                        FROM document_chunks dc
                        JOIN eligible_documents d
                          ON d.id = dc.doc_id
                         AND d.kb_id = dc.kb_id
                        WHERE :same_section_enabled
                          AND seed.section_key IS NOT NULL
                          AND dc.doc_id = seed.doc_id
                          AND dc.doc_id = ANY(:doc_ids)
                          AND dc.kb_id = ANY(:kb_ids)
                          AND dc.id <> seed.seed_chunk_id
                          AND dc.metadata ? 'section_key'
                          AND dc.metadata->>'section_key' = seed.section_key
                          AND dc.chunk_index > seed.chunk_index
                        ORDER BY dc.chunk_index ASC, dc.id
                        LIMIT :same_section_limit
                    )
                ) nearby
                ORDER BY structure_distance, nearby.chunk_index, nearby.id
                LIMIT :same_section_limit
            ) candidate
        ),
        heading_section_matches AS (
            SELECT
                candidate.id, seed.seed_chunk_id,
                'same_section'::text AS structural_origin,
                candidate.structure_distance,
                seed.seed_order
            FROM seeds seed
            CROSS JOIN LATERAL (
                SELECT
                    nearby.id,
                    ABS(nearby.chunk_index - seed.chunk_index)
                        AS structure_distance
                FROM (
                    (
                        SELECT dc.id, dc.chunk_index
                        FROM document_chunks dc
                        JOIN eligible_documents d
                          ON d.id = dc.doc_id
                         AND d.kb_id = dc.kb_id
                        WHERE :same_section_enabled
                          AND seed.section_key IS NULL
                          AND seed.heading IS NOT NULL
                          AND dc.doc_id = seed.doc_id
                          AND dc.doc_id = ANY(:doc_ids)
                          AND dc.kb_id = ANY(:kb_ids)
                          AND dc.id <> seed.seed_chunk_id
                          AND dc.metadata ? 'heading'
                          AND dc.metadata->>'heading' = seed.heading
                          AND dc.chunk_index < seed.chunk_index
                        ORDER BY dc.chunk_index DESC, dc.id
                        LIMIT :same_section_limit
                    )
                    UNION ALL
                    (
                        SELECT dc.id, dc.chunk_index
                        FROM document_chunks dc
                        JOIN eligible_documents d
                          ON d.id = dc.doc_id
                         AND d.kb_id = dc.kb_id
                        WHERE :same_section_enabled
                          AND seed.section_key IS NULL
                          AND seed.heading IS NOT NULL
                          AND dc.doc_id = seed.doc_id
                          AND dc.doc_id = ANY(:doc_ids)
                          AND dc.kb_id = ANY(:kb_ids)
                          AND dc.id <> seed.seed_chunk_id
                          AND dc.metadata ? 'heading'
                          AND dc.metadata->>'heading' = seed.heading
                          AND dc.chunk_index > seed.chunk_index
                        ORDER BY dc.chunk_index ASC, dc.id
                        LIMIT :same_section_limit
                    )
                ) nearby
                ORDER BY structure_distance, nearby.chunk_index, nearby.id
                LIMIT :same_section_limit
            ) candidate
        ),
        section_matches AS (
            SELECT * FROM section_key_matches
            UNION ALL
            SELECT * FROM heading_section_matches
        ),
        table_id_matches AS (
            SELECT
                candidate.id, seed.seed_chunk_id,
                'table_sibling'::text AS structural_origin,
                candidate.structure_distance,
                seed.seed_order
            FROM seeds seed
            CROSS JOIN LATERAL (
                SELECT
                    dc.id,
                    ABS(
                        CASE
                            WHEN COALESCE(dc.metadata->>'table_part_index', '')
                                 ~ '^[0-9]{1,9}$'
                            THEN (dc.metadata->>'table_part_index')::integer
                            ELSE NULL
                        END - seed.table_part_index
                    ) AS structure_distance
                FROM document_chunks dc
                JOIN eligible_documents d
                  ON d.id = dc.doc_id
                 AND d.kb_id = dc.kb_id
                WHERE :tables_enabled
                  AND seed.block_type = 'table'
                  AND seed.table_id IS NOT NULL
                  AND seed.table_part_index IS NOT NULL
                  AND dc.doc_id = seed.doc_id
                  AND dc.doc_id = ANY(:doc_ids)
                  AND dc.kb_id = ANY(:kb_ids)
                  AND dc.id <> seed.seed_chunk_id
                  AND dc.metadata ? 'table_id'
                  AND dc.metadata ? 'table_part_index'
                  AND dc.metadata->>'table_id' = seed.table_id
                  AND COALESCE(
                      dc.metadata->>'block_type',
                      dc.metadata->>'type'
                  ) = 'table'
                  AND CASE
                      WHEN COALESCE(dc.metadata->>'table_part_index', '')
                           ~ '^[0-9]{1,9}$'
                      THEN (dc.metadata->>'table_part_index')::integer
                      ELSE NULL
                  END IN (
                      seed.table_part_index - 1,
                      seed.table_part_index + 1
                  )
                ORDER BY structure_distance, dc.chunk_index, dc.id
                LIMIT 2
            ) candidate
        ),
        legacy_table_matches AS (
            SELECT
                candidate.id, seed.seed_chunk_id,
                'table_sibling'::text AS structural_origin,
                candidate.structure_distance,
                seed.seed_order
            FROM seeds seed
            CROSS JOIN LATERAL (
                SELECT
                    dc.id,
                    ABS(dc.chunk_index - seed.chunk_index) AS structure_distance
                FROM document_chunks dc
                JOIN eligible_documents d
                  ON d.id = dc.doc_id
                 AND d.kb_id = dc.kb_id
                WHERE :tables_enabled
                  AND seed.block_type = 'table'
                  AND seed.table_id IS NULL
                  AND seed.heading IS NOT NULL
                  AND dc.doc_id = seed.doc_id
                  AND dc.doc_id = ANY(:doc_ids)
                  AND dc.kb_id = ANY(:kb_ids)
                  AND dc.id <> seed.seed_chunk_id
                  AND dc.metadata ? 'heading'
                  AND dc.metadata->>'heading' = seed.heading
                  AND COALESCE(
                      dc.metadata->>'block_type',
                      dc.metadata->>'type'
                  ) = 'table'
                  AND dc.chunk_index IN (
                      seed.chunk_index - 1,
                      seed.chunk_index + 1
                  )
                ORDER BY dc.chunk_index, dc.id
                LIMIT 2
            ) candidate
        ),
        table_matches AS (
            SELECT * FROM table_id_matches
            UNION ALL
            SELECT * FROM legacy_table_matches
        ),
        structural_matches AS (
            SELECT * FROM table_matches
            UNION ALL
            SELECT * FROM section_matches
            UNION ALL
            SELECT * FROM adjacent_matches
        )
        SELECT
            dc.id, dc.content, dc.chunk_index, dc.metadata,
            dc.kb_id, dc.doc_id, d.filename, d.file_type, d.source_url,
            d.doc_tags,
            NULL::double precision AS vector_score,
            NULL::bigint AS vector_rank,
            NULL::double precision AS keyword_score,
            NULL::bigint AS keyword_rank,
            NULL::double precision AS trigram_score,
            NULL::bigint AS trigram_rank,
            NULL::double precision AS retrieval_score,
            NULL::double precision AS score,
            matches.structural_origin,
            matches.seed_chunk_id,
            matches.structure_distance
        FROM structural_matches matches
        JOIN document_chunks dc ON dc.id = matches.id
        JOIN eligible_documents d
          ON d.id = dc.doc_id
         AND d.kb_id = dc.kb_id
        WHERE dc.doc_id = ANY(:doc_ids)
          AND dc.kb_id = ANY(:kb_ids)
        ORDER BY
            CASE matches.structural_origin
                WHEN 'table_sibling' THEN 1
                WHEN 'same_section' THEN 2
                ELSE 3
            END,
            matches.seed_order,
            matches.structure_distance,
            dc.doc_id,
            dc.chunk_index,
            dc.id
        LIMIT :row_limit
    """).bindparams(
        bindparam("seed_specs", type_=JSONB),
        bindparam("kb_ids", type_=ARRAY(SA_UUID())),
        bindparam("doc_ids", type_=ARRAY(SA_UUID())),
    )
    result = await _execute_in_savepoint(db, sql, {
        "seed_specs": specs,
        "kb_ids": scoped_kb_ids,
        "doc_ids": scoped_doc_ids,
        "neighbors_enabled": neighbors_enabled,
        "same_section_enabled": bounded_section_limit > 0,
        "same_section_limit": bounded_section_limit,
        "tables_enabled": tables_enabled,
        "row_limit": sql_row_limit,
    })
    rows = result.mappings().all()
    results = _merge_structural_rows(rows, bounded_total)
    if trace_id:
        trace_event(
            "retrieval.structure_expanded",
            trace_id=trace_id,
            seed_chunk_count=len(specs),
            scoped_document_count=len(scoped_doc_ids),
            candidate_count=len(results),
            counts_by_origin={
                origin: sum(
                    origin in (item.get("candidate_origins") or []) for item in results
                )
                for origin in ("adjacent", "same_section", "table_sibling")
            },
            elapsed_ms=round((time.perf_counter() - started_at) * 1000),
        )
    return results
