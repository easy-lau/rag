"""智能路由后台配置、分类、测试和决策日志接口。"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.audit import AuditLogger, get_audit
from core.deps import require_permission
from core.intent_router import (
    classify_intent_result,
    ensure_intent_routing_defaults,
    get_intent_router_config,
    list_intent_categories,
)
from core.permissions import INTENT_MANAGE, INTENT_READ
from database import get_db
from models.db_models import IntentCategory, IntentRouteLog, IntentRouterConfig, User, now_utc
from models.schemas import (
    IntentCategoryCreate,
    IntentCategoryOut,
    IntentCategoryUpdate,
    IntentDecisionOut,
    IntentRouteFeedbackUpdate,
    IntentRouteLogOut,
    IntentRouteLogPage,
    IntentRouteTestRequest,
    IntentRouteTestResponse,
    IntentRouterConfigOut,
    IntentRouterConfigUpdate,
)

router = APIRouter(prefix="/intent-routing", tags=["intent-routing"])


def _config_out(config: IntentRouterConfig) -> IntentRouterConfigOut:
    return IntentRouterConfigOut(
        enabled=config.enabled,
        mode=config.mode,
        confidence_threshold=config.confidence_threshold,
        fallback_intent_code=config.fallback_intent_code,
        allow_general_chat=config.allow_general_chat,
    )


def _category_out(category: IntentCategory) -> IntentCategoryOut:
    return IntentCategoryOut(
        id=category.id,
        code=category.code,
        name=category.name,
        description=category.description or "",
        examples=list(category.examples or []),
        action=category.action,
        enabled=category.enabled,
        priority=category.priority,
    )


async def _ensure_defaults_and_commit(db: AsyncSession) -> None:
    """使首次访问空表时自动生成可用的配置和内置分类。"""

    if await ensure_intent_routing_defaults(db):
        await db.commit()


async def _require_retrieve_fallback(db: AsyncSession, code: str) -> IntentCategory:
    """fallback 必须是启用中的 retrieve 分类，保证未知问题永远安全检索。"""

    category = (await db.execute(
        select(IntentCategory).where(
            IntentCategory.code == code,
            IntentCategory.enabled.is_(True),
            IntentCategory.action == "retrieve",
        )
    )).scalar_one_or_none()
    if category is None:
        raise HTTPException(
            status_code=400,
            detail="兜底意图必须是一个已启用且动作为 retrieve 的分类",
        )
    return category


@router.get("/config", response_model=IntentRouterConfigOut)
async def get_config(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(INTENT_READ)),
):
    await _ensure_defaults_and_commit(db)
    return _config_out(await get_intent_router_config(db))


@router.put("/config", response_model=IntentRouterConfigOut)
async def update_config(
    payload: IntentRouterConfigUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(INTENT_MANAGE)),
):
    await _ensure_defaults_and_commit(db)
    config = await get_intent_router_config(db)
    updates = payload.model_dump(exclude_none=True)

    fallback_code = updates.get("fallback_intent_code", config.fallback_intent_code)
    await _require_retrieve_fallback(db, fallback_code)

    changed: list[str] = []
    for key, value in updates.items():
        if getattr(config, key) != value:
            setattr(config, key, value)
            changed.append(key)

    if changed:
        audit.log(
            db,
            "intent_router.config.update",
            target_type="intent_router_config",
            target_id=config.id,
            detail={"changed": sorted(changed)},
        )
    await db.commit()
    await db.refresh(config)
    return _config_out(config)


@router.get("/categories", response_model=list[IntentCategoryOut])
async def get_categories(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(INTENT_READ)),
):
    await _ensure_defaults_and_commit(db)
    return [_category_out(row) for row in await list_intent_categories(db)]


@router.post("/categories", response_model=IntentCategoryOut, status_code=status.HTTP_201_CREATED)
async def create_category(
    payload: IntentCategoryCreate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(INTENT_MANAGE)),
):
    await _ensure_defaults_and_commit(db)
    code = payload.code.strip()
    exists = (await db.execute(
        select(IntentCategory.id).where(IntentCategory.code == code)
    )).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(status_code=400, detail="意图 code 已存在")

    category = IntentCategory(
        code=code,
        name=payload.name.strip(),
        description=payload.description.strip(),
        examples=[item.strip() for item in payload.examples if item.strip()],
        action=payload.action,
        enabled=payload.enabled,
        priority=payload.priority,
    )
    db.add(category)
    await db.flush()
    audit.log(
        db,
        "intent_router.category.create",
        target_type="intent_category",
        target_id=category.id,
        target_name=category.name,
        detail={"code": category.code, "action": category.action},
    )
    await db.commit()
    await db.refresh(category)
    return _category_out(category)


@router.put("/categories/{category_id}", response_model=IntentCategoryOut)
async def update_category(
    category_id: uuid.UUID,
    payload: IntentCategoryUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(INTENT_MANAGE)),
):
    await _ensure_defaults_and_commit(db)
    category = await db.get(IntentCategory, category_id)
    if category is None:
        raise HTTPException(status_code=404, detail="意图分类不存在")
    config = await get_intent_router_config(db)
    updates = payload.model_dump(exclude_none=True)

    # 当前兜底分类不允许被关闭或改成非 retrieve；先改 config 到其它 retrieve 分类即可调整。
    if category.code == config.fallback_intent_code:
        next_enabled = updates.get("enabled", category.enabled)
        next_action = updates.get("action", category.action)
        if not next_enabled or next_action != "retrieve":
            raise HTTPException(
                status_code=400,
                detail="当前兜底意图必须保持启用且动作为 retrieve，请先修改路由配置的兜底意图",
            )

    changed: list[str] = []
    for key, value in updates.items():
        if key == "name":
            value = value.strip()
        elif key == "description":
            value = value.strip()
        elif key == "examples":
            value = [item.strip() for item in value if item.strip()]
        if getattr(category, key) != value:
            setattr(category, key, value)
            changed.append(key)

    if changed:
        audit.log(
            db,
            "intent_router.category.update",
            target_type="intent_category",
            target_id=category.id,
            target_name=category.name,
            detail={"code": category.code, "changed": sorted(changed)},
        )
    await db.commit()
    await db.refresh(category)
    return _category_out(category)


@router.delete("/categories/{category_id}")
async def delete_category(
    category_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(INTENT_MANAGE)),
):
    await _ensure_defaults_and_commit(db)
    category = await db.get(IntentCategory, category_id)
    if category is None:
        raise HTTPException(status_code=404, detail="意图分类不存在")
    config = await get_intent_router_config(db)
    if category.code == config.fallback_intent_code:
        raise HTTPException(status_code=400, detail="当前兜底意图不可删除，请先修改路由配置")

    audit.log(
        db,
        "intent_router.category.delete",
        target_type="intent_category",
        target_id=category.id,
        target_name=category.name,
        detail={"code": category.code},
    )
    await db.delete(category)
    await db.commit()
    return {"message": "删除成功"}


@router.post("/test", response_model=IntentRouteTestResponse)
async def test_route(
    payload: IntentRouteTestRequest,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(INTENT_READ)),
):
    """只测试决策，不写运行日志，也不访问用户未授权的知识库内容。"""

    await _ensure_defaults_and_commit(db)
    result = await classify_intent_result(
        db,
        payload.question,
        selected_kb_ids=payload.knowledge_base_ids,
        record_log=False,
    )
    return IntentRouteTestResponse(
        decision=IntentDecisionOut(**result.decision.to_dict()),
        latency_ms=result.latency_ms,
    )


@router.get("/logs", response_model=IntentRouteLogPage)
async def get_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    intent_code: str | None = Query(None, max_length=64),
    feedback: str | None = Query(None, pattern="^(correct|incorrect)$"),
    user_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(INTENT_READ)),
):
    conds = []
    if intent_code:
        conds.append(IntentRouteLog.intent_code == intent_code)
    if feedback:
        conds.append(IntentRouteLog.feedback == feedback)
    if user_id:
        conds.append(IntentRouteLog.user_id == user_id)

    count_stmt = select(func.count()).select_from(IntentRouteLog)
    rows_stmt = select(IntentRouteLog)
    for cond in conds:
        count_stmt = count_stmt.where(cond)
        rows_stmt = rows_stmt.where(cond)
    total = (await db.execute(count_stmt)).scalar_one()
    rows = (await db.execute(
        rows_stmt
        .order_by(IntentRouteLog.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )).scalars().all()
    return IntentRouteLogPage(
        items=[IntentRouteLogOut.model_validate(row) for row in rows],
        total=total,
    )


@router.post("/logs/{log_id}/feedback", response_model=IntentRouteLogOut)
async def update_log_feedback(
    log_id: uuid.UUID,
    payload: IntentRouteFeedbackUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(INTENT_MANAGE)),
):
    log = await db.get(IntentRouteLog, log_id)
    if log is None:
        raise HTTPException(status_code=404, detail="路由日志不存在")
    log.feedback = payload.feedback
    log.feedback_at = now_utc()
    audit.log(
        db,
        "intent_router.log.feedback",
        target_type="intent_route_log",
        target_id=log.id,
        detail={"feedback": payload.feedback},
    )
    await db.commit()
    await db.refresh(log)
    return IntentRouteLogOut.model_validate(log)
