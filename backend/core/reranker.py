from core.openai_client import get_client
from config import get_settings


async def rerank(query: str, results: list[dict]) -> list[dict]:
    """LLM-based reranking: score each chunk's relevance to the query."""
    if not results:
        return results

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
        import json
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        scores = {item["index"]: item["score"] for item in data.get("scores", data.get("results", []))}
        for i, r in enumerate(results):
            r["score"] = scores.get(i + 1, r.get("score", 0.5))
        results.sort(key=lambda x: x["score"], reverse=True)
    except Exception:
        pass

    return results
