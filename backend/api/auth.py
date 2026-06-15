"""认证路由：登录、查询当前用户、修改密码。"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.deps import _load_permissions, get_current_user
from core.security import create_access_token, hash_password, verify_password
from database import get_db
from models.db_models import LoginLog, User
from models.schemas import LoginRequest, MeOut, TokenOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str | None:
    """解析客户端 IP：优先取 X-Forwarded-For 首段，否则回退 request.client.host。"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def _build_me(user: User, permissions: list[str]) -> MeOut:
    """根据用户与权限列表构造 MeOut；menus 取以 'menu:' 开头的权限项。"""
    menus = [p for p in permissions if p.startswith("menu:")]
    return MeOut(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        is_superadmin=user.is_superadmin,
        role_name=user.role.name if user.role else None,
        permissions=permissions,
        menus=menus,
    )


@router.post("/login", response_model=TokenOut)
async def login(payload: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """用户名/密码登录，校验通过后签发 JWT 并返回当前用户信息；各分支均记录登录日志。"""
    ip = _client_ip(request)
    user_agent = request.headers.get("user-agent")
    result = await db.execute(
        select(User).options(selectinload(User.role)).where(User.username == payload.username)
    )
    user = result.scalar_one_or_none()

    # 用户不存在：写失败日志（user_id 空），再抛 401
    if user is None:
        db.add(LoginLog(user_id=None, username=payload.username, success=False, ip=ip, user_agent=user_agent))
        await db.commit()
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    # 密码错误：写失败日志，再抛 401
    if not verify_password(payload.password, user.password_hash):
        db.add(LoginLog(user_id=user.id, username=user.username, success=False, ip=ip, user_agent=user_agent))
        await db.commit()
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    # 用户被禁用：写失败日志，再抛 403
    if not user.is_active:
        db.add(LoginLog(user_id=user.id, username=user.username, success=False, ip=ip, user_agent=user_agent))
        await db.commit()
        raise HTTPException(status_code=403, detail="用户已禁用")

    # 登录成功：更新最后登录时间并写成功日志（一起 commit）
    user.last_login_at = datetime.now(timezone.utc)
    db.add(LoginLog(user_id=user.id, username=user.username, success=True, ip=ip, user_agent=user_agent))
    await db.commit()

    permissions = await _load_permissions(user, db)
    token = create_access_token(subject=str(user.id))
    return TokenOut(access_token=token, user=_build_me(user, permissions))


@router.get("/me", response_model=MeOut)
async def get_me(user: User = Depends(get_current_user)):
    """返回当前登录用户信息（权限已由 get_current_user 计算挂载）。"""
    return _build_me(user, getattr(user, "permissions", []))


class ChangePasswordIn(BaseModel):
    old_password: str
    new_password: str


@router.post("/change-password")
async def change_password(
    body: ChangePasswordIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """校验旧密码后更新为新密码。"""
    if not verify_password(body.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="原密码错误")
    user.password_hash = hash_password(body.new_password)
    await db.commit()
    return {"message": "密码修改成功"}
