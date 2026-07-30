import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models.db_models import Conversation, IntentRouteLog, Message, User
from models.schemas import ChatRequest, ConversationOut, ConversationRenameRequest, MessageOut
from core.rag_pipeline import run_rag_stream
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import CHAT_USE
from core.intent_router import classify_intent_result

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/send")
async def send_message(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    # 流式开始前校验请求的知识库都在可访问范围内（accessible 为 None 表示全部）
    accessible = await get_accessible_kb_ids(user, db)
    if accessible is not None and not set(payload.knowledge_base_ids).issubset(set(accessible)):
        raise HTTPException(status_code=403, detail="无权访问部分知识库")

    # 获取或创建会话。新会话先 flush 取得 id，但不提前提交；若后续路由/校验失败，
    # 请求结束时整个未提交事务会回滚，避免留下空白会话。
    if payload.conversation_id:
        conv = await db.get(Conversation, payload.conversation_id)
        # 复用已有会话时校验归属：非超管不可操作他人会话
        if conv and not user.is_superadmin and conv.user_id != user.id:
            raise HTTPException(status_code=404, detail="会话不存在")
    else:
        conv = None

    if not conv:
        conv = Conversation(title=payload.question[:50], user_id=user.id)
        db.add(conv)
        await db.flush()

    # 智能路由只输出受白名单约束的动作；知识库授权仍以上方接口校验为准。
    routing_result = await classify_intent_result(
        db,
        payload.question,
        user=user,
        selected_kb_ids=payload.knowledge_base_ids,
        conversation_id=conv.id,
        record_log=True,
    )
    decision = routing_result.decision
    if decision.need_retrieval and not payload.knowledge_base_ids:
        raise HTTPException(status_code=400, detail="该问题需要查询知识库，请至少选择一个知识库")

    # 保存用户消息
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=payload.question,
    )
    db.add(user_msg)
    await db.commit()

    search_config = payload.search_config.model_dump()

    async def generate():
        import json as _json
        full_response = []
        sources = []
        tokens = None
        retrieval_executed = None
        evidence_status = None
        hit_count = None
        # 会话和用户消息已提交。先告知前端会话 ID，用户在首条回答完成前停止时也能继续该会话。
        yield f"data: {_json.dumps({'type': 'conversation_started', 'conversation_id': str(conv.id)})}\n\n"
        try:
            async for chunk in run_rag_stream(
                question=payload.question,
                kb_ids=payload.knowledge_base_ids,
                search_config=search_config,
                conversation_id=str(conv.id),
                db=db,
                intent=decision.to_dict(),
            ):
                yield chunk
                if '"type": "text_delta"' in chunk:
                    try:
                        data = _json.loads(chunk.removeprefix("data: ").strip())
                        full_response.append(data.get("content", ""))
                    except Exception:
                        pass
                if '"type": "search_results"' in chunk:
                    try:
                        data = _json.loads(chunk.removeprefix("data: ").strip())
                        sources = data.get("results", [])[:5]
                        retrieval_executed = data.get("retrieval_executed")
                        evidence_status = data.get("evidence_status")
                        hit_count = data.get("total", len(data.get("results", [])))
                    except Exception:
                        pass
                if '"type": "usage"' in chunk:
                    try:
                        data = _json.loads(chunk.removeprefix("data: ").strip())
                        tokens = data.get("total_tokens")
                    except Exception:
                        pass
        except Exception as e:
            print(f"[chat/stream error] {type(e).__name__}: {e}")
            if retrieval_executed is None:
                retrieval_executed = bool(decision.need_retrieval)
            if evidence_status is None:
                evidence_status = "error" if decision.need_retrieval else "skipped"
            if hit_count is None:
                hit_count = 0
            from database import AsyncSessionLocal
            async with AsyncSessionLocal() as save_db:
                if routing_result.route_log_id is not None:
                    route_log = await save_db.get(IntentRouteLog, routing_result.route_log_id)
                    if route_log is not None:
                        route_log.retrieval_executed = retrieval_executed
                        route_log.evidence_status = evidence_status
                        route_log.hit_count = hit_count
                        await save_db.commit()
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {_json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"
            return

        # 保存 AI 回复
        from database import AsyncSessionLocal
        async with AsyncSessionLocal() as save_db:
            ai_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content="".join(full_response),
                sources=sources,
                tokens=tokens,
            )
            save_db.add(ai_msg)
            if routing_result.route_log_id is not None:
                route_log = await save_db.get(IntentRouteLog, routing_result.route_log_id)
                if route_log is not None:
                    route_log.retrieval_executed = (
                        bool(retrieval_executed)
                        if retrieval_executed is not None
                        else bool(decision.need_retrieval)
                    )
                    route_log.evidence_status = evidence_status or (
                        "no_hit" if decision.need_retrieval else "skipped"
                    )
                    route_log.hit_count = int(hit_count or 0)
            await save_db.commit()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            # 首条 SSE 数据到达前，前端也可从响应头立即绑定会话，降低刚开始就停止时丢失会话 ID 的概率。
            "X-Conversation-ID": str(conv.id),
        },
    )


@router.get("/history", response_model=list[ConversationOut])
async def get_history(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    offset = (page - 1) * page_size
    stmt = select(Conversation).order_by(Conversation.created_at.desc())
    # 非超管只看自己的会话；超管可见全部
    if not user.is_superadmin:
        stmt = stmt.where(Conversation.user_id == user.id)
    rows = (await db.execute(
        stmt.offset(offset).limit(page_size)
    )).scalars().all()
    return rows


@router.get("/{conv_id}/messages", response_model=list[MessageOut])
async def get_messages(
    conv_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")
    rows = (await db.execute(
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
    )).scalars().all()
    return rows


@router.patch("/{conv_id}", response_model=ConversationOut)
async def rename_conversation(
    conv_id: uuid.UUID,
    payload: ConversationRenameRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    """更新会话标题；沿用读取/删除时的归属校验。"""
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")

    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="会话标题不能为空")

    conv.title = title
    await db.commit()
    await db.refresh(conv)
    return conv


@router.delete("/{conv_id}")
async def delete_conversation(
    conv_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    conv = await db.get(Conversation, conv_id)
    if not conv or (not user.is_superadmin and conv.user_id != user.id):
        raise HTTPException(status_code=404, detail="会话不存在")
    await db.delete(conv)
    await db.commit()
    return {"message": "删除成功"}
