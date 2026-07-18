"""操作日志路由：分页查询操作审计日志，支持按动作前缀 / 操作人筛选，需要 log:read 权限。"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import require_permission
from core.permissions import LOG_READ
from database import get_db
from models.db_models import OperationLog, User
from models.schemas import OperationLogOut, OperationLogPage

router = APIRouter(prefix="/operation-logs", tags=["operation-logs"])


@router.get("", response_model=OperationLogPage)
async def list_operation_logs(
    page: int = 1,
    page_size: int = 20,
    action: str | None = Query(None, description="按动作码前缀过滤，如 user / doc / role"),
    username: str | None = Query(None, description="按操作人用户名过滤（模糊）"),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(LOG_READ)),
):
    """按 created_at 倒序分页返回操作日志，可选按动作前缀 / 操作人过滤。"""
    conds = []
    if action:
        conds.append(OperationLog.action.like(f"{action}%"))
    if username:
        conds.append(OperationLog.username.ilike(f"%{username}%"))

    count_q = select(func.count()).select_from(OperationLog)
    rows_q = select(OperationLog)
    for c in conds:
        count_q = count_q.where(c)
        rows_q = rows_q.where(c)

    total = (await db.execute(count_q)).scalar_one()
    offset = (page - 1) * page_size
    rows = (await db.execute(
        rows_q.order_by(OperationLog.created_at.desc()).offset(offset).limit(page_size)
    )).scalars().all()
    return OperationLogPage(items=[OperationLogOut.model_validate(r) for r in rows], total=total)
