"""认证路由：登录、查询当前用户、修改密码。"""

from datetime import datetime, timezone
import math
from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from starlette.concurrency import run_in_threadpool

from core.audit import AuditLogger, client_ip, get_audit
from core.deps import _load_permissions, get_current_user
from core.login_security import (
    LockedLoginSource,
    clear_login_pair_failures,
    lock_login_source,
    record_login_attempt,
    register_login_failure,
)
from core.permissions import KB_SCOPE_ALL, KB_SCOPE_NONE
from core.security import create_access_token, hash_password, verify_password
from database import get_db
from models.db_models import User
from models.schemas import LoginRequest, MeOut, TokenOut

router = APIRouter(prefix="/auth", tags=["auth"])

# 固定 dummy hash 让“不存在的用户”也执行一次同成本 bcrypt，降低用户名枚举风险。
_DUMMY_PASSWORD_HASH = "$2b$12$lrBrXsiG6Anzu7hVqV7kJ.XAOq6/cKbW6qMvgwvpIcBvMa72qxqmO"
_GENERIC_LOGIN_ERROR = "用户名或密码错误"


def _raise_rate_limited(retry_after: int) -> None:
    seconds = max(1, math.ceil(retry_after))
    raise HTTPException(
        status_code=429,
        detail="登录尝试过于频繁，请稍后再试",
        headers={"Retry-After": str(seconds), "Cache-Control": "no-store"},
    )


async def _record_failure_and_raise(
    db: AsyncSession,
    *,
    user: User | None,
    username: str,
    fail_reason: str,
    ip: str | None,
    user_agent: str | None,
    now: datetime,
    source: LockedLoginSource,
) -> NoReturn:
    """原子登记来源失败与审计，然后返回通用 401 或来源级 429。"""
    throttle = await register_login_failure(
        db,
        source=source,
        username=username,
        ip=ip,
        now=now,
    )
    audit_reason = (
        f"{fail_reason}，来源临时受限"
        if throttle.retry_after_seconds
        else fail_reason
    )
    await record_login_attempt(
        db,
        user=user,
        username=username,
        success=False,
        fail_reason=audit_reason,
        ip=ip,
        user_agent=user_agent,
        now=now,
    )
    await db.commit()
    if throttle.retry_after_seconds:
        _raise_rate_limited(throttle.retry_after_seconds)
    raise HTTPException(status_code=401, detail=_GENERIC_LOGIN_ERROR)


def _build_me(user: User, permissions: list[str]) -> MeOut:
    """根据用户与权限列表构造 MeOut；menus 取以 'menu:' 开头的权限项。"""
    menus = [p for p in permissions if p.startswith("menu:")]
    return MeOut(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        is_superadmin=user.is_superadmin,
        role_name=user.role.name if user.role else None,
        kb_scope=(
            KB_SCOPE_ALL
            if user.is_superadmin
            else getattr(user.role, "scope_mode", KB_SCOPE_NONE) if user.role else KB_SCOPE_NONE
        ),
        permissions=permissions,
        menus=menus,
    )


@router.post("/login", response_model=TokenOut)
async def login(payload: LoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    """用户名/密码登录；限制攻击来源，但不因外部失败全局锁定账号。"""
    now = datetime.now(timezone.utc)
    username = payload.username.strip()
    ip = client_ip(request)
    user_agent = request.headers.get("user-agent")

    source = await lock_login_source(
        db,
        username=username,
        ip=ip,
        now=now,
    )
    if source.throttle.retry_after_seconds:
        # 受限请求不再执行 bcrypt 或查询用户，但仍以 60 秒窗口聚合留痕。
        await record_login_attempt(
            db,
            user=None,
            username=username,
            success=False,
            fail_reason="来源已限流",
            ip=ip,
            user_agent=user_agent,
            now=now,
        )
        await db.commit()
        _raise_rate_limited(source.throttle.retry_after_seconds)

    # 首次查询不预加载角色，避免用户存在与否产生额外查询时序差异；只有密码校验
    # 成功后才加载登录响应所需的角色数据。
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()

    # 用户不存在仍执行固定 bcrypt；对外响应与密码错误完全一致。
    if user is None:
        await run_in_threadpool(verify_password, payload.password, _DUMMY_PASSWORD_HASH)
        await _record_failure_and_raise(
            db,
            user=None,
            username=username,
            fail_reason="用户不存在",
            ip=ip,
            user_agent=user_agent,
            now=now,
            source=source,
        )

    # 已禁用账号也执行真实密码校验，但无论密码是否正确都返回同一通用错误。
    if not user.is_active:
        await run_in_threadpool(verify_password, payload.password, user.password_hash)
        await _record_failure_and_raise(
            db,
            user=user,
            username=user.username,
            fail_reason="账号已禁用",
            ip=ip,
            user_agent=user_agent,
            now=now,
            source=source,
        )

    password_matches = await run_in_threadpool(
        verify_password,
        payload.password,
        user.password_hash,
    )
    if not password_matches:
        await _record_failure_and_raise(
            db,
            user=user,
            username=user.username,
            fail_reason="密码错误",
            ip=ip,
            user_agent=user_agent,
            now=now,
            source=source,
        )

    role_result = await db.execute(
        select(User).options(selectinload(User.role)).where(User.id == user.id)
    )
    user = role_result.scalar_one()

    # 登录成功只清理当前来源 + 用户名组合；IP 扫描和账号异常统计不能被掩盖。
    await clear_login_pair_failures(db, username=user.username, ip=ip)
    user.last_login_at = now
    await record_login_attempt(
        db,
        user=user,
        username=user.username,
        success=True,
        fail_reason=None,
        ip=ip,
        user_agent=user_agent,
        now=now,
    )
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
    audit: AuditLogger = Depends(get_audit),
):
    """校验旧密码后更新为新密码。"""
    if not verify_password(body.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="原密码错误")
    user.password_hash = hash_password(body.new_password)
    audit.log(db, "auth.change_password", target_type="user", target_id=user.id, target_name=user.username)
    await db.commit()
    return {"message": "密码修改成功"}
