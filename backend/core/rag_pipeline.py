import json
import time
import uuid
import asyncio
import logging
from decimal import Decimal
from typing import AsyncGenerator
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from core.retriever import hybrid_search
from core.reranker import rerank
from core.openai_client import get_client
from core.llm_stream import stream_with_retry_before_first_delta
from config import get_settings

logger = logging.getLogger(__name__)


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


def _intent_event(intent: dict) -> str:
    """向前端公开已校验过的智能路由决策，不暴露分类提示词或原始问题。"""
    return f"data: {json.dumps({'type': 'intent', 'decision': intent})}\n\n"


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
    t0 = time.perf_counter()
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
        need = bool(data.get("need_retrieval", True))
        logger.info(
            "[意图判断] 模型=%s 结果=%s 耗时=%.0fms",
            s.chat_model, "需要检索" if need else "无需检索（闲聊/问候）",
            (time.perf_counter() - t0) * 1000,
        )
        return need
    except Exception as e:
        logger.warning(
            "[意图判断] 调用失败，保守按需要检索处理: %s: %s（耗时=%.0fms）",
            type(e).__name__, e, (time.perf_counter() - t0) * 1000,
        )
        return True


async def run_rag_stream(
    question: str,
    kb_ids: list[uuid.UUID],
    search_config: dict,
    conversation_id: str,
    db: AsyncSession,
    intent: dict | None = None,
) -> AsyncGenerator[str, None]:
    s = get_settings()
    t_total = time.perf_counter()
    logger.info(
        "[提问] conv=%s 知识库数=%d 问题=%.80s",
        conversation_id, len(kb_ids), question,
    )

    # Step 1: 问题分析。正常聊天接口会先完成可配置的智能路由；保留旧判断作为
    # 兼容兜底，以便其他调用方仍可直接使用本函数。
    yield _step_event("analyze", "active")
    route_action = (intent or {}).get("action", "retrieve")
    if intent:
        yield _intent_event(intent)
        need_retrieval = route_action == "retrieve"
        logger.info(
            "[智能路由] conv=%s intent=%s action=%s source=%s confidence=%s",
            conversation_id,
            intent.get("intent_code"),
            route_action,
            intent.get("source"),
            intent.get("confidence"),
        )
    else:
        need_retrieval = bool(kb_ids) and await _needs_retrieval(question)
        if not kb_ids:
            logger.info("[意图判断] 未选择知识库，跳过检索")
    yield _step_event("analyze", "done")

    # Step 2: 查询扩展
    yield _step_event("expand", "active")
    await asyncio.sleep(0.1)
    yield _step_event("expand", "done")

    top_k = search_config.get("top_k", s.top_k)
    candidate_k = max(20, top_k * 4)

    # Step 3: 检索（扩大召回，给重排留足候选）；无需检索时跳过，results 保持为空
    yield _step_event("retrieve", "active")
    method = search_config.get("method", "hybrid")
    if need_retrieval:
        t0 = time.perf_counter()
        results = await hybrid_search(
            db=db,
            query=question,
            kb_ids=kb_ids,
            top_k=candidate_k,
            method=method,
        )
        logger.info(
            "[检索] 方式=%s 候选上限=%d 召回=%d条 耗时=%.0fms",
            method, candidate_k, len(results), (time.perf_counter() - t0) * 1000,
        )
    else:
        results = []
        logger.info("[检索] 跳过（无需查库）")
    yield _step_event("retrieve", "done")

    # Step 4: 重排（大候选池上重排后，按相关度过滤+截断，剔除不相关文档）
    reranked = False
    if search_config.get("rerank", s.rerank_enabled) and results:
        yield _step_event("rerank", "active")
        t0 = time.perf_counter()
        results = await rerank(question, results)
        reranked = True
        scores = [float(r.get("score") or 0) for r in results]
        logger.info(
            "[重排] %d条 分数区间=%.2f~%.2f 耗时=%.0fms",
            len(results), min(scores), max(scores), (time.perf_counter() - t0) * 1000,
        )
        yield _step_event("rerank", "done")
    else:
        if results:
            logger.info("[重排] 已关闭，跳过")
        yield _step_event("rerank", "done")

    # 标签软加权：命中用户所选标签的文档上浮排序（不改语义分、不排除未命中）
    selected_tags = search_config.get("tags") or []
    results = apply_tag_boost(results, selected_tags)
    if selected_tags:
        logger.info("[标签加权] 所选标签=%s", selected_tags)

    before_filter = len(results)
    results = _select_relevant(results, top_k, reranked)
    if need_retrieval:
        logger.info(
            "[相关度过滤] 阈值=%s 过滤前=%d条 保留=%d条 命中文档=%s",
            RELEVANCE_THRESHOLD if reranked else "未启用（未重排）",
            before_filter, len(results),
            sorted({r.get("filename") or "" for r in results}) or "无",
        )
    yield _results_event(results)

    # Step 5: LLM 生成
    yield _step_event("generate", "active")

    context = await _build_context(db, results)
    if need_retrieval:
        logger.info(
            "[上下文] 长度=%d字符 模式=%s",
            len(context),
            "有相关内容" if context else "知识库无相关内容（按未找到回答）",
        )
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
    elif route_action == "writing":
        # 写作类请求明确不带知识库上下文，避免检索片段干扰改写、总结等输出。
        system_prompt = (
            "你是一位专业的中文写作助手。根据用户目标完成润色、改写、总结、起草等任务。"
            "除非用户明确要求，否则不要编造事实；输出应直接可用、结构清晰。"
        )
    elif route_action == "system_help":
        system_prompt = (
            "你是该企业 RAG 检索系统的使用助手。请简洁说明如何选择知识库、提问、"
            "查看检索结果、管理文档和在权限不足时该如何处理。不要虚构不存在的功能。"
        )
    else:
        # 闲聊/问候等无需查库的输入：按通用助手回答。
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
    # 不依赖 stream_options：部分 OpenAI 兼容服务不支持它，且旧逻辑会把超时误判为
    # 参数兼容问题而重复发起整轮请求。流式响应开始后不重试，避免重复输出给用户。
    client = get_client().with_options(max_retries=0)

    async def open_stream():
        return await client.chat.completions.create(
            **create_kwargs,
            timeout=s.llm_request_timeout_seconds,
        )

    usage = None
    t_gen = time.perf_counter()
    answer_chars = 0
    prompt_chars = len(system_prompt) + len(question)
    async for chunk in stream_with_retry_before_first_delta(
        open_stream,
        model=s.chat_model,
        prompt_chars=prompt_chars,
        timeout_seconds=s.llm_request_timeout_seconds,
        max_attempts=s.llm_max_attempts,
        retry_base_delay_seconds=s.llm_retry_base_delay_seconds,
    ):
        # 末尾的用量统计块 choices 为空，需先取 usage、再判空取增量
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                answer_chars += len(delta)
                yield _delta_event(delta)

    # 生成完成，标记最后一步为完成（否则前端步骤条会一直停在蓝色转圈）
    yield _step_event("generate", "done")
    if usage:
        yield _usage_event(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        logger.info(
            "[生成] 模型=%s 回答=%d字符 tokens(输入/输出/合计)=%d/%d/%d 生成耗时=%.1fs 全程耗时=%.1fs",
            s.chat_model, answer_chars,
            usage.prompt_tokens, usage.completion_tokens, usage.total_tokens,
            time.perf_counter() - t_gen, time.perf_counter() - t_total,
        )
    else:
        logger.info(
            "[生成] 模型=%s 回答=%d字符（服务未返回token用量） 生成耗时=%.1fs 全程耗时=%.1fs",
            s.chat_model, answer_chars,
            time.perf_counter() - t_gen, time.perf_counter() - t_total,
        )
    yield _done_event(conversation_id)
