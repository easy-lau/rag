import json
import uuid
import asyncio
from decimal import Decimal
from typing import AsyncGenerator
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from core.retriever import hybrid_search
from core.reranker import rerank
from core.openai_client import get_client
from config import get_settings


def _step_event(step: str, status: str) -> str:
    return f"data: {json.dumps({'type': 'search_step', 'step': step, 'status': status})}\n\n"


def _results_event(results: list[dict]) -> str:
    serializable = []
    for r in results:
        item = {
            k: str(v) if isinstance(v, uuid.UUID)
            else float(v) if isinstance(v, Decimal)
            else v
            for k, v in dict(r).items()
        }
        serializable.append(item)
    return f"data: {json.dumps({'type': 'search_results', 'results': serializable, 'total': len(serializable)})}\n\n"


def _delta_event(content: str) -> str:
    return f"data: {json.dumps({'type': 'text_delta', 'content': content})}\n\n"


def _done_event(conv_id: str) -> str:
    return f"data: {json.dumps({'type': 'done', 'conversation_id': conv_id})}\n\n"


def _usage_event(prompt: int, completion: int, total: int) -> str:
    return f"data: {json.dumps({'type': 'usage', 'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': total})}\n\n"


# 命中文档若小于该字符数，则整篇注入上下文，保证跨段落/跨表格的信息完整
WHOLE_DOC_MAX_CHARS = 6000
WHOLE_DOC_TOTAL_BUDGET = 12000

# 重排相关度低于该值的结果视为不相关，不纳入上下文与来源（仅在开启重排时生效）。
# 实测重排器区分度很好：真正相关 0.75~1.0，无关内容 ≤0.05，0.3 正好落在两者之间的空档。
RELEVANCE_THRESHOLD = 0.3

# 标签软加权：命中用户所选标签的文档，排序分上浮该比例（0.5 = 上浮 50%）。
# 软加权——只影响排序先后，不排除未命中文档，因此不会把库里相关内容直接漏掉。
TAG_BOOST = 0.5


def apply_tag_boost(results: list[dict], selected_tags: list[str]) -> list[dict]:
    """用户手动勾选的标签做软加权：命中标签的文档片段排序分上浮，未命中的保持原样。
    关键：只用上浮后的分数『重新排序』，不改写每条结果的语义相关度分 score，
    因此 _select_relevant 的相关度阈值仍作用于真实语义分，标签不会让不相关内容蒙混过关。"""
    if not selected_tags or not results:
        return results
    wanted = set(selected_tags)

    def sort_key(r: dict) -> float:
        base = float(r.get("score") or 0)
        matched = bool(wanted & set(r.get("doc_tags") or []))
        # 仅对正分上浮：负分（已属不相关）上浮只会更糟，保持原值即可
        return base * (1 + TAG_BOOST) if (matched and base > 0) else base

    return sorted(results, key=sort_key, reverse=True)


def _select_relevant(results: list[dict], top_k: int, reranked: bool) -> list[dict]:
    """重排后按相关度过滤，剔除明显不相关的文档，避免它们被当作来源或污染上下文。
    全部低于阈值 → 返回空：说明知识库中没有相关内容（此时由上层改用『未找到』提示词，
    明确告知用户而非用模型自身知识编造，也不会展示一堆不相关的来源）。"""
    limit = max(top_k, 8)
    if not results:
        return []
    if reranked:
        relevant = [r for r in results if float(r.get("score") or 0) >= RELEVANCE_THRESHOLD]
        return relevant[:limit]
    return results[:limit]


async def _fetch_doc_text(db: AsyncSession, doc_id) -> str:
    rows = (await db.execute(
        sa_text("SELECT content FROM document_chunks WHERE doc_id = :d ORDER BY chunk_index"),
        {"d": doc_id},
    )).scalars().all()
    return "\n\n".join(r for r in rows if r)


async def _build_context(db: AsyncSession, results: list[dict]) -> str:
    """构建给 LLM 的上下文：命中的小文档整篇注入（保证跨段/跨表的完整信息），其余用命中片段。"""
    doc_ids = []
    for r in results:
        did = r.get("doc_id")
        if did and did not in doc_ids:
            doc_ids.append(did)

    whole, used = {}, 0
    for did in doc_ids:
        full = await _fetch_doc_text(db, did)
        if full and len(full) <= WHOLE_DOC_MAX_CHARS and used + len(full) <= WHOLE_DOC_TOTAL_BUDGET:
            whole[did] = full
            used += len(full)

    parts, seen, idx = [], set(), 1
    for r in results:
        did = r.get("doc_id")
        if did in whole:
            if did in seen:
                continue
            seen.add(did)
            parts.append(f"《{r.get('filename', '')}》（完整内容）：\n{whole[did]}")
        else:
            parts.append(f"[片段{idx}] 来源：{r.get('filename', '')}\n{r.get('content', '')}")
            idx += 1
    return "\n\n".join(parts)


async def _needs_retrieval(question: str) -> bool:
    """轻量意图判断：这条输入是否需要查知识库才能回答。
    闲聊/问候/寒暄/与资料无关的请求 → False；涉及业务/制度/流程/文档内容 → True。
    出错时保守返回 True（宁可多检索，也不漏答真问题）。"""
    s = get_settings()
    try:
        resp = await get_client().chat.completions.create(
            model=s.chat_model,
            messages=[{
                "role": "user",
                "content": (
                    "判断下面这句用户输入是否需要查询企业知识库/文档资料才能回答。\n"
                    "- 闲聊、问候、寒暄、感谢、自我介绍、与资料无关的常识或写作请求 → 不需要\n"
                    "- 涉及具体业务、制度、流程、数据、文档内容的提问 → 需要\n"
                    '只返回 JSON：{"need_retrieval": true} 或 {"need_retrieval": false}。\n\n'
                    f"用户输入：{question}"
                ),
            }],
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return bool(data.get("need_retrieval", True))
    except Exception:
        return True


async def run_rag_stream(
    question: str,
    kb_ids: list[uuid.UUID],
    search_config: dict,
    conversation_id: str,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    s = get_settings()

    # Step 1: 问题分析（顺带判断是否真的需要查知识库，闲聊/问候直接跳过检索）
    yield _step_event("analyze", "active")
    need_retrieval = bool(kb_ids) and await _needs_retrieval(question)
    yield _step_event("analyze", "done")

    # Step 2: 查询扩展
    yield _step_event("expand", "active")
    await asyncio.sleep(0.1)
    yield _step_event("expand", "done")

    top_k = search_config.get("top_k", s.top_k)
    candidate_k = max(20, top_k * 4)

    # Step 3: 检索（扩大召回，给重排留足候选）；无需检索时跳过，results 保持为空
    yield _step_event("retrieve", "active")
    results = await hybrid_search(
        db=db,
        query=question,
        kb_ids=kb_ids,
        top_k=candidate_k,
        method=search_config.get("method", "hybrid"),
    ) if need_retrieval else []
    yield _step_event("retrieve", "done")

    # Step 4: 重排（大候选池上重排后，按相关度过滤+截断，剔除不相关文档）
    reranked = False
    if search_config.get("rerank", s.rerank_enabled) and results:
        yield _step_event("rerank", "active")
        results = await rerank(question, results)
        reranked = True
        yield _step_event("rerank", "done")
    else:
        yield _step_event("rerank", "done")

    # 标签软加权：命中用户所选标签的文档上浮排序（不改语义分、不排除未命中）
    results = apply_tag_boost(results, search_config.get("tags") or [])

    results = _select_relevant(results, top_k, reranked)
    yield _results_event(results)

    # Step 5: LLM 生成
    yield _step_event("generate", "active")

    context = await _build_context(db, results)
    if context:
        # 有相关内容：依据文档作答
        system_prompt = (
            "你是一个专业的知识库问答助手。根据以下检索到的文档内容回答用户问题。"
            "回答要准确、详细、条理清晰。"
            "如果用户用职位或人员名称提问（如『普通员工』），而文档用分级/分类表述（如『D级』），"
            "请先对照文档中的分类对应关系（如职级分类表）确定其类别，再据此查找对应标准后回答。"
            "不要在正文中插入来源编号或引用标记（如 [1]、[2]），来源会在回答下方单独展示给用户。"
            "你只能依据下面检索到的文档作答；如果文档中没有回答该问题所需的信息，"
            "必须明确告诉用户『知识库中未找到相关内容』，禁止使用你自己的知识或经验编造答案。\n\n"
            f"检索到的文档：\n{context}"
        )
    elif need_retrieval:
        # 检索过但知识库里没有相关内容：如实告知，禁止用模型自身知识编造（也不会展示无关来源）
        system_prompt = (
            "你是一个企业知识库问答助手。本次在知识库中没有检索到与用户问题相关的内容。"
            "请明确告诉用户『知识库中未找到相关内容』，可简要建议其补充资料或换种问法，"
            "但禁止使用你自己的知识或经验编造答案。"
        )
    else:
        # 闲聊/问候等无需查库的输入：按通用助手回答
        system_prompt = "你是一个专业的助手，请尽力回答用户问题。"

    create_kwargs = dict(
        model=s.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        stream=True,
    )
    try:
        # 请求流式响应附带 token 用量统计
        stream = await get_client().chat.completions.create(**create_kwargs, stream_options={"include_usage": True})
    except Exception:
        # 个别 OpenAI 兼容服务不支持 stream_options，降级为不带用量统计
        stream = await get_client().chat.completions.create(**create_kwargs)

    usage = None
    async for chunk in stream:
        # 末尾的用量统计块 choices 为空，需先取 usage、再判空取增量
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield _delta_event(delta)

    # 生成完成，标记最后一步为完成（否则前端步骤条会一直停在蓝色转圈）
    yield _step_event("generate", "done")
    if usage:
        yield _usage_event(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
    yield _done_event(conversation_id)
