import time
import logging
from core.openai_client import get_embedding_client
from config import get_settings

logger = logging.getLogger(__name__)


async def embed_text(text: str) -> list[float]:
    s = get_settings()
    t0 = time.perf_counter()
    resp = await get_embedding_client().embeddings.create(
        model=s.embedding_model,
        input=text,
    )
    logger.info(
        "[向量化] 模型=%s 输入=%d字符 维度=%d 耗时=%.0fms",
        s.embedding_model, len(text), len(resp.data[0].embedding),
        (time.perf_counter() - t0) * 1000,
    )
    return resp.data[0].embedding


# 单次嵌入请求的最大条数：一篇大文档可能有上千个 chunk，整篇塞一次会超过
# 供应商单请求的条数/Token 上限导致整篇失败，故分批发送（保持输入顺序）。
_EMBED_BATCH_SIZE = 64


async def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    client = get_embedding_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i:i + _EMBED_BATCH_SIZE]
        resp = await client.embeddings.create(model=s.embedding_model, input=batch)
        # 同一批内按 index 还原顺序，再按批次顺序拼接 → 与输入 texts 顺序严格一致
        out.extend(item.embedding for item in sorted(resp.data, key=lambda x: x.index))
    return out
