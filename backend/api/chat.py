import asyncio
import uuid
import logging
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models.db_models import (
    Conversation,
    Document,
    DocumentChunk,
    IntentRouteLog,
    Message,
    User,
)
from models.schemas import ChatRequest, ConversationOut, ConversationRenameRequest, MessageOut
from core.rag_pipeline import run_rag_stream
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import CHAT_USE
from core.intent_router import classify_intent_result
from core.conversation_context import (
    UNRESOLVED_REFERENCE_MESSAGE,
    prepare_conversation_context,
)
from core.rag_trace import content_fields, log_exception_safely, trace_event
from config import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)
_NON_ANSWER_SOURCE_STATUSES = {"no_hit", "skipped", "error"}


def _parse_sse_payload(chunk: str) -> dict | None:
    if not chunk.startswith("data: "):
        return None
    try:
        payload = json.loads(chunk.removeprefix("data: ").strip())
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _source_snapshot_identity(
    source: object,
) -> tuple[uuid.UUID, uuid.UUID, uuid.UUID] | None:
    """读取历史来源的知识库/文档/片段标识。

    缺少任一标识时按不可披露处理。只验证 doc_id 会让文档重新分块
    后已删除的旧 content 仍从 Message.sources JSON 中返回。
    """

    if not isinstance(source, dict):
        return None
    try:
        kb_id = uuid.UUID(str(source.get("kb_id")))
        doc_id = uuid.UUID(str(source.get("doc_id")))
        chunk_id = uuid.UUID(str(source.get("id") or source.get("chunk_id")))
    except (TypeError, ValueError, AttributeError):
        return None
    return kb_id, doc_id, chunk_id


def _source_snapshot_is_answer_evidence(source: object) -> bool:
    """Reject legacy broad-candidate snapshots that were never answer evidence.

    Older releases persisted every displayed retrieval candidate in
    ``Message.sources``.  Their shared ``evidence_status`` marker lets history
    reads fail closed for requests that had no generation context, while
    snapshots created before the marker remain backward compatible.
    """

    if not isinstance(source, dict):
        return False
    status = str(source.get("evidence_status") or "").strip().casefold()
    return status not in _NON_ANSWER_SOURCE_STATUSES


async def _messages_with_current_source_scope(
    rows: list[Message],
    *,
    user: User,
    db: AsyncSession,
) -> list[MessageOut]:
    """按当前角色范围和文档状态过滤历史 ``sources`` 快照。

    assistant 正文属于用户自己的既有会话记录；额外展开的原始检索片段则必须
    每次按当前 RBAC 重新授权，防止角色范围被撤销或文档停用后仍从 JSONB 快照
    读取 ``content`` / ``source_url``。
    """

    accessible = await get_accessible_kb_ids(user, db)
    accessible_set = set(accessible) if accessible is not None else None
    referenced_sources: set[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = set()
    for row in rows:
        for source in (row.sources if isinstance(row.sources, list) else ()):
            if not _source_snapshot_is_answer_evidence(source):
                continue
            identity = _source_snapshot_identity(source)
            if identity is None:
                continue
            kb_id, _, _ = identity
            if accessible_set is not None and kb_id not in accessible_set:
                continue
            referenced_sources.add(identity)

    current_sources: dict[tuple[uuid.UUID, uuid.UUID, uuid.UUID], dict] = {}
    if referenced_sources:
        chunk_ids = {chunk_id for _, _, chunk_id in referenced_sources}
        statement = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.doc_id)
            .where(
                DocumentChunk.id.in_(chunk_ids),
                Document.kb_id == DocumentChunk.kb_id,
                Document.is_active.is_(True),
                Document.status == "ready",
            )
        )
        if accessible_set is not None:
            statement = statement.where(DocumentChunk.kb_id.in_(accessible_set))
        for chunk, document in (await db.execute(statement)).all():
            identity = (chunk.kb_id, chunk.doc_id, chunk.id)
            if identity not in referenced_sources:
                continue
            # 排名、证据角色与分数保留当轮快照；可披露的文档内容和
            # 元数据始终从当前有效 chunk 重载，避免返回已删除或已更新的旧片段。
            current_sources[identity] = {
                "id": str(chunk.id),
                "chunk_id": str(chunk.id),
                "doc_id": str(chunk.doc_id),
                "kb_id": str(chunk.kb_id),
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata_ or {},
                "filename": document.filename,
                "file_type": document.file_type,
                "source_url": document.source_url,
                "image_url": document.image_url,
                "doc_tags": document.tags or [],
            }

    output: list[MessageOut] = []
    for row in rows:
        visible_sources = []
        if isinstance(row.sources, list):
            for source in row.sources:
                if not _source_snapshot_is_answer_evidence(source):
                    continue
                identity = _source_snapshot_identity(source)
                current = current_sources.get(identity) if identity else None
                if current is not None:
                    visible_sources.append({**dict(source), **current})
        # 历史脏数据可能把 sources 存成 dict；不先让 Pydantic 验证 ORM
        # 对象，否则一条异常消息会让整个会话返回 500。
        serialized = MessageOut(
            id=row.id,
            conversation_id=row.conversation_id,
            role=row.role,
            content=row.content,
            sources=visible_sources if isinstance(row.sources, list) else None,
            tokens=row.tokens,
            created_at=row.created_at,
        )
        output.append(serialized)
    return output


def _public_stream_error_message(exc: BaseException) -> str:
    """返回可安全展示给前端的生成错误。

    详细异常已按 trace_id 写入服务端日志；不把上游 URL、响应体或
    请求信息通过 SSE 直接暴露给终端用户。
    """

    error_name = type(exc).__name__.casefold()
    if "timeout" in error_name or isinstance(exc, TimeoutError):
        return "模型服务响应超时，请稍后重试"
    return "回答生成失败，请稍后重试"


@router.post("/send")
async def send_message(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission(CHAT_USE)),
):
    trace_id = uuid.uuid4().hex
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

    # 在保存本轮用户消息之前读取已有对话。带“这些配置/上述内容”等指代的追问
    # 会得到独立检索问题；上一轮来源只作为候选 id，随后按当前知识库范围和文档
    # 状态重新加载，不能直接信任消息 JSON 快照。
    conversation_context = await prepare_conversation_context(
        db,
        conversation_id=conv.id,
        question=payload.question,
        kb_ids=payload.knowledge_base_ids,
    )

    # 在任何路由/检索之前记录请求和多轮上下文，保证调用链的第一阶段始终
    # 是“接收请求”。此前这里等意图模型返回后才写 chat.request，导致模型
    # 事件排在请求之前；路由校验失败时也会留下没有起点的 running 记录。
    trace_include_content = get_settings().rag_trace_include_content
    search_config = payload.search_config.model_dump()
    trace_search_config = dict(search_config)
    if not trace_include_content:
        trace_search_config["tags"] = []
    trace_event(
        "chat.request",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        selected_kb_ids=payload.knowledge_base_ids,
        search_config=trace_search_config,
        selected_tag_count=len(search_config.get("tags") or []),
        intent=None,
        decision_reason=(
            "unresolved_reference"
            if conversation_context.unresolved_reference
            else "pending_intent_routing"
        ),
        is_followup=conversation_context.is_followup,
        followup_reason=conversation_context.followup_reason,
        history_message_count=len(conversation_context.history_messages),
        carryover_source_count=len(conversation_context.carryover_sources),
        **content_fields("question", payload.question),
        **content_fields(
            "standalone_query",
            conversation_context.standalone_query,
        ),
    )
    trace_event(
        "conversation.context_resolved",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        is_followup=conversation_context.is_followup,
        followup_reason=conversation_context.followup_reason,
        unresolved_reference=conversation_context.unresolved_reference,
        history_message_count=len(conversation_context.history_messages),
        carryover_source_count=len(conversation_context.carryover_sources),
        **content_fields(
            "standalone_query",
            conversation_context.standalone_query,
        ),
    )

    # 新会话中的“这些配置/上述方案/有什么影响”缺少可消解对象。此时全库
    # 向量检索只会制造高相似假命中，直接返回确定性澄清提示，不调用意图模型、
    # 检索器或回答模型。
    if conversation_context.unresolved_reference:
        user_msg = Message(
            conversation_id=conv.id,
            role="user",
            content=payload.question,
        )
        assistant_msg = Message(
            conversation_id=conv.id,
            role="assistant",
            content=UNRESOLVED_REFERENCE_MESSAGE,
            sources=[],
        )
        db.add_all([user_msg, assistant_msg])
        await db.commit()
        trace_event(
            "conversation.reference_unresolved",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            reason=conversation_context.followup_reason,
            selected_kb_count=len(payload.knowledge_base_ids),
            **content_fields("question", payload.question),
        )
        trace_event(
            "chat.response",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            evidence_status="skipped",
            retrieval_executed=False,
            decision_reason="unresolved_reference",
            displayed_result_count=0,
            answer_source_count=0,
            context_evidence_count=0,
            hit_count=0,
            direct_evidence_count=0,
            related_reference_count=0,
            sources=[],
            **content_fields("answer", UNRESOLVED_REFERENCE_MESSAGE),
        )

        async def generate_clarification():
            events = (
                {
                    "type": "conversation_started",
                    "conversation_id": str(conv.id),
                },
                {"type": "search_step", "step": "analyze", "status": "done"},
                {
                    "type": "search_results",
                    "results": [],
                    "answer_sources": [],
                    "total": 0,
                    "displayed_result_count": 0,
                    "answer_source_count": 0,
                    "context_evidence_count": 0,
                    "hit_count": 0,
                    "retrieval_executed": False,
                    "evidence_status": "skipped",
                    "decision_reason": "unresolved_reference",
                    "direct_evidence_count": 0,
                    "related_reference_count": 0,
                    "trace_id": trace_id,
                    "is_followup": False,
                    "carryover_source_count": 0,
                },
                {
                    "type": "text_delta",
                    "content": UNRESOLVED_REFERENCE_MESSAGE,
                },
                {"type": "done", "conversation_id": str(conv.id)},
            )
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_clarification(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
                "X-Conversation-ID": str(conv.id),
                "X-RAG-Trace-ID": trace_id,
            },
        )

    # 智能路由使用已消解指代的独立问题；知识库授权仍以上方接口校验为准。
    try:
        routing_result = await classify_intent_result(
            db,
            conversation_context.standalone_query,
            user=user,
            selected_kb_ids=payload.knowledge_base_ids,
            conversation_id=conv.id,
            record_log=True,
            trace_id=trace_id,
        )
    except Exception as exc:
        # 路由配置/数据库故障不应让调用链停留在 running；接口仍按原语义
        # 抛出异常，但保留安全的阶段错误供后台排查。
        trace_event(
            "chat.error",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            stage="intent_routing",
            error=exc,
        )
        log_exception_safely(
            logger,
            "[chat/intent routing error] trace=%s conv=%s",
            trace_id,
            conv.id,
            exc=exc,
        )
        raise
    decision = routing_result.decision
    trace_event(
        "intent.routing_decision",
        trace_id=trace_id,
        conversation_id=conv.id,
        user_id=user.id,
        intent=decision.to_dict(),
        selected_kb_count=len(payload.knowledge_base_ids),
        decision_reason=decision.decision_reason,
    )
    if decision.need_retrieval and not payload.knowledge_base_ids:
        trace_event(
            "chat.error",
            trace_id=trace_id,
            conversation_id=conv.id,
            user_id=user.id,
            stage="request_validation",
            error=ValueError("该问题需要查询知识库，请至少选择一个知识库"),
            evidence_status="error",
        )
        raise HTTPException(status_code=400, detail="该问题需要查询知识库，请至少选择一个知识库")

    # 保存用户消息
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=payload.question,
    )
    db.add(user_msg)
    await db.commit()


    async def generate():
        full_response = []
        sources = []
        tokens = None
        retrieval_executed = None
        evidence_status = None
        displayed_result_count = None
        context_evidence_count = None
        hit_count = None
        direct_evidence_count = None
        related_reference_count = None
        pending_done_chunk = None
        # 会话和用户消息已提交。先告知前端会话 ID，用户在首条回答完成前停止时也能继续该会话。
        yield f"data: {json.dumps({'type': 'conversation_started', 'conversation_id': str(conv.id)})}\n\n"
        try:
            async for chunk in run_rag_stream(
                question=payload.question,
                kb_ids=payload.knowledge_base_ids,
                search_config=search_config,
                conversation_id=str(conv.id),
                db=db,
                intent=decision.to_dict(),
                trace_id=trace_id,
                standalone_query=conversation_context.standalone_query,
                # 独立新问题不把旧轮正文发送给外部模型，避免无关历史污染回答并
                # 遵循最小披露；只有本地规则确认是追问时才提供有界历史帮助消解。
                conversation_history=(
                    list(conversation_context.history_messages)
                    if conversation_context.is_followup
                    else []
                ),
                carryover_sources=list(conversation_context.carryover_sources),
                is_followup=conversation_context.is_followup,
                followup_reason=conversation_context.followup_reason,
            ):
                data = _parse_sse_payload(chunk)
                event_type = data.get("type") if data else None
                # done 必须等 AI 消息持久化成功后再发给前端；其它事件保持实时流式。
                if event_type == "done":
                    pending_done_chunk = chunk
                    continue
                if event_type == "text_delta":
                    full_response.append(str(data.get("content") or ""))
                elif event_type == "search_results":
                    retrieval_executed = data.get("retrieval_executed")
                    evidence_status = data.get("evidence_status")
                    normalized_evidence_status = str(
                        evidence_status or ""
                    ).strip().casefold()
                    direct_evidence_count = data.get("direct_evidence_count")
                    if (
                        isinstance(direct_evidence_count, bool)
                        or not isinstance(direct_evidence_count, int)
                        or direct_evidence_count < 0
                    ):
                        direct_evidence_count = 0
                    related_reference_count = data.get("related_reference_count")
                    display_results = data.get("results")
                    if not isinstance(display_results, list):
                        display_results = []
                    raw_answer_sources = data.get("answer_sources")
                    if not isinstance(raw_answer_sources, list):
                        # Fail closed for rolling upgrades or custom stream
                        # producers: broad retrieval candidates must never be
                        # persisted as answer evidence merely because the new
                        # field is absent.
                        raw_answer_sources = []
                    if normalized_evidence_status in _NON_ANSWER_SOURCE_STATUSES:
                        # 状态是服务端最终证据门控：即使异常/旧版/自定义流生产者
                        # 同时错误携带了 answer_sources 和非零 direct 数，也不得
                        # 把这些正文保存成历史回答依据。持久化层必须 fail closed，
                        # 不能只依赖正常 Pipeline 或前端隐藏。
                        raw_answer_sources = []
                        direct_evidence_count = 0
                    displayed_result_count = data.get(
                        "displayed_result_count",
                        data.get("total", len(display_results)),
                    )
                    if (
                        isinstance(displayed_result_count, bool)
                        or not isinstance(displayed_result_count, int)
                        or displayed_result_count < 0
                    ):
                        displayed_result_count = len(display_results)
                    answer_source_items = [
                        source
                        for source in raw_answer_sources[:20]
                        if isinstance(source, dict)
                    ]
                    context_evidence_count = len(answer_source_items)
                    hit_count = direct_evidence_count
                    source_meta = {
                        "trace_id": data.get("trace_id") or trace_id,
                        "retrieval_executed": bool(retrieval_executed),
                        "evidence_status": evidence_status,
                        "displayed_result_count": displayed_result_count or 0,
                        "context_evidence_count": context_evidence_count,
                        "hit_count": hit_count,
                        "direct_evidence_count": direct_evidence_count or 0,
                        "related_reference_count": related_reference_count or 0,
                        "is_followup": bool(data.get("is_followup")),
                        "carryover_source_count": data.get("carryover_source_count") or 0,
                    }
                    # 右侧检索面板继续消费 ``results``，但历史回答与引用只保存
                    # Pipeline 实际送入 generation.context 的 answer_sources。
                    # 两者不能再共用一份宽候选列表。
                    sources = [
                        {**source, **source_meta}
                        for source in answer_source_items
                    ]
                elif event_type == "usage":
                    tokens = data.get("total_tokens")
                yield chunk
        except asyncio.CancelledError:
            # 浏览器停止生成、断开连接或服务关闭都会取消流协程。同步入队即可
            # 立即把调用链标成 interrupted；不要在已取消任务里继续等待数据库。
            trace_event(
                "chat.cancelled",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="streaming",
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                **content_fields("partial_answer", "".join(full_response)),
            )
            raise
        except Exception as e:
            log_exception_safely(
                logger,
                "[chat/stream error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=e,
            )
            if retrieval_executed is None:
                retrieval_executed = bool(decision.need_retrieval)
            if evidence_status is None:
                evidence_status = "error" if decision.need_retrieval else "skipped"
            if hit_count is None:
                hit_count = 0
            if context_evidence_count is None:
                context_evidence_count = 0
            if displayed_result_count is None:
                displayed_result_count = 0
            trace_event(
                "chat.error",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                error=e,
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count,
                answer_source_count=context_evidence_count,
                context_evidence_count=context_evidence_count,
                hit_count=hit_count,
                direct_evidence_count=direct_evidence_count or 0,
                related_reference_count=related_reference_count or 0,
                **content_fields("partial_answer", "".join(full_response)),
            )
            from database import AsyncSessionLocal
            if routing_result.route_log_id is not None:
                try:
                    async with AsyncSessionLocal() as save_db:
                        route_log = await save_db.get(IntentRouteLog, routing_result.route_log_id)
                        if route_log is not None:
                            route_log.retrieval_executed = retrieval_executed
                            route_log.evidence_status = evidence_status
                            route_log.hit_count = hit_count
                            await save_db.commit()
                except Exception as persistence_exc:
                    # 路由统计是 best-effort；失败不能覆盖真正的模型/检索错误，也不能
                    # 阻断随后发给前端的安全 error + done 事件。
                    log_exception_safely(
                        logger,
                        "[chat/error route-log persistence] trace=%s conv=%s",
                        trace_id,
                        conv.id,
                        exc=persistence_exc,
                    )
                    trace_event(
                        "chat.persistence_error",
                        trace_id=trace_id,
                        conversation_id=conv.id,
                        operation="update_route_log_after_stream_error",
                        error=persistence_exc,
                    )
            yield f"data: {json.dumps({'type': 'error', 'message': _public_stream_error_message(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"
            return

        # 保存 AI 回复
        answer = "".join(full_response)
        from database import AsyncSessionLocal
        try:
            async with AsyncSessionLocal() as save_db:
                ai_msg = Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=answer,
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
            # 只有回答和路由统计真正提交成功后才把 Trace 标成 success，避免
            # “调用链成功但历史消息不存在”的竞态。
            trace_event(
                "chat.response",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                evidence_status=evidence_status,
                retrieval_executed=retrieval_executed,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                direct_evidence_count=direct_evidence_count,
                related_reference_count=related_reference_count,
                tokens=tokens,
                sources=[
                    {
                        "doc_id": source.get("doc_id"),
                        "chunk_id": source.get("id"),
                        "evidence_role": source.get("evidence_role"),
                        "constraint_status": source.get("constraint_status"),
                        "retrieval_score": source.get("retrieval_score"),
                        "effective_score": source.get("score"),
                        "answer_support": source.get("answer_support"),
                        **content_fields(
                            "filename",
                            str(source.get("filename") or ""),
                        ),
                    }
                    for source in sources
                ],
                **content_fields("answer", answer),
            )
        except asyncio.CancelledError:
            trace_event(
                "chat.cancelled",
                trace_id=trace_id,
                conversation_id=conv.id,
                user_id=user.id,
                stage="response_persistence",
                retrieval_executed=retrieval_executed,
                evidence_status=evidence_status,
                displayed_result_count=displayed_result_count or 0,
                answer_source_count=context_evidence_count or 0,
                context_evidence_count=context_evidence_count or 0,
                hit_count=hit_count or 0,
                **content_fields("partial_answer", answer),
            )
            raise
        except Exception as exc:
            log_exception_safely(
                logger,
                "[chat/persistence error] trace=%s conv=%s",
                trace_id,
                conv.id,
                exc=exc,
            )
            trace_event(
                "chat.persistence_error",
                trace_id=trace_id,
                conversation_id=conv.id,
                error=exc,
                **content_fields("answer", answer),
            )
            yield f"data: {json.dumps({'type': 'error', 'message': '回答已生成，但保存失败，请重试'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"
            return
        if pending_done_chunk is not None:
            yield pending_done_chunk
        else:
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conv.id)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            # 首条 SSE 数据到达前，前端也可从响应头立即绑定会话，降低刚开始就停止时丢失会话 ID 的概率。
            "X-Conversation-ID": str(conv.id),
            # 便于开发阶段把浏览器请求与结构化 rag.trace 日志精确关联。
            "X-RAG-Trace-ID": trace_id,
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
    return await _messages_with_current_source_scope(rows, user=user, db=db)


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
