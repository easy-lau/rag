import json
import logging
import math
from dataclasses import dataclass

from core.openai_client import get_client
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankOutcome:
    """重排结果及其可信状态。

    ``succeeded`` 只有在模型返回了完整、唯一且合法的全部分数时才为 True。
    调用失败或响应不完整时保留原始排序，避免把 RRF 等低量纲原始分数误当成
    0~1 相关度分数并在后续阈值过滤中全部清空。
    """

    results: list[dict]
    succeeded: bool
    error: str | None = None


def _parse_complete_scores(raw: str, result_count: int) -> dict[int, float]:
    data = json.loads(raw)
    items = data.get("scores", data.get("results"))
    if not isinstance(items, list) or len(items) != result_count:
        raise ValueError("重排分数未覆盖全部候选")

    scores: dict[int, float] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("重排分数项格式无效")
        index = item.get("index")
        score = item.get("score")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("重排索引必须为整数")
        if index in scores:
            raise ValueError("重排索引重复")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("重排分数必须为数字")
        numeric_score = float(score)
        if not math.isfinite(numeric_score) or not 0 <= numeric_score <= 1:
            raise ValueError("重排分数必须位于 0~1")
        scores[index] = numeric_score

    if set(scores) != set(range(1, result_count + 1)):
        raise ValueError("重排索引未完整覆盖全部候选")
    return scores


async def rerank_with_status(query: str, results: list[dict]) -> RerankOutcome:
    """LLM-based reranking with an explicit success signal."""
    if not results:
        return RerankOutcome(results=[], succeeded=True)

    s = get_settings()
    client = get_client()

    snippets = "\n\n".join(
        f"[{i+1}] {r['content'][:300]}" for i, r in enumerate(results)
    )
    prompt = (
        f"用户查询：{query}\n\n"
        f"下面有 {len(results)} 段文本。请评估每一段与该查询的相关度，分值 0.0~1.0"
        "（完全不相关为 0.0，高度相关为 1.0）。"
        '只返回 JSON，格式为 {"scores": [{"index": 1, "score": 0.0}, ...]}，'
        "index 从 1 开始且必须覆盖全部段落。\n\n"
        f"{snippets}"
    )

    try:
        resp = await client.chat.completions.create(
            model=s.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        scores = _parse_complete_scores(raw, len(results))
        ranked = []
        for i, result in enumerate(results):
            item = dict(result)
            item["score"] = scores[i + 1]
            ranked.append(item)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return RerankOutcome(results=ranked, succeeded=True)
    except Exception as e:
        # 重排失败不致命：保留检索原始排序继续走后续流程
        logger.warning("[重排] 调用失败，保留检索原始排序: %s: %s", type(e).__name__, e)
        return RerankOutcome(
            results=[dict(result) for result in results],
            succeeded=False,
            error=f"{type(e).__name__}: {e}",
        )


async def rerank(query: str, results: list[dict]) -> list[dict]:
    """兼容旧调用方的重排接口；需要可信状态时使用 ``rerank_with_status``。"""

    return (await rerank_with_status(query, results)).results
