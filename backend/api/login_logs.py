"""登录日志路由：分页查询登录日志，需要 log:read 权限。"""

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import require_permission
from core.permissions import LOG_READ
from database import get_db
from models.db_models import LoginLog, User
from models.schemas import LoginLogOut, LoginLogPage

router = APIRouter(prefix="/login-logs", tags=["login-logs"])


@router.get("", response_model=LoginLogPage)
async def list_login_logs(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(LOG_READ)),
):
    """按 created_at 倒序分页返回登录日志，并附带总数用于分页控件。"""
    offset = (page - 1) * page_size
    total = (await db.execute(
        select(func.count()).select_from(LoginLog)
    )).scalar_one()
    rows = (await db.execute(
        select(LoginLog)
        .order_by(LoginLog.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )).scalars().all()
    return LoginLogPage(items=[LoginLogOut.model_validate(r) for r in rows], total=total)
