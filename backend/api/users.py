"""用户管理路由：CRUD + 启用/禁用，全部需要 user:manage 权限。"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.audit import AuditLogger, get_audit
from core.deps import require_permission
from core.permissions import USER_MANAGE
from core.security import hash_password
from database import get_db
from models.db_models import Role, User
from models.schemas import UserCreate, UserOut, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


def _to_out(user: User) -> UserOut:
    """构造带 role_name 的 UserOut。"""
    return UserOut(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        is_active=user.is_active,
        is_superadmin=user.is_superadmin,
        role_id=user.role_id,
        role_name=user.role.name if user.role else None,
        created_at=user.created_at,
    )


async def _active_superadmin_count(db: AsyncSession) -> int:
    """统计当前 active 的超管数量，用于防锁死护栏。"""
    return (await db.execute(
        select(func.count()).select_from(User)
        .where(User.is_superadmin.is_(True), User.is_active.is_(True))
    )).scalar_one()


@router.get("", response_model=list[UserOut])
async def list_users(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(USER_MANAGE)),
):
    rows = (await db.execute(
        select(User).options(selectinload(User.role)).order_by(User.created_at.desc())
    )).scalars().all()
    return [_to_out(u) for u in rows]


@router.post("", response_model=UserOut)
async def create_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(USER_MANAGE)),
):
    exists = (await db.execute(
        select(User).where(User.username == payload.username)
    )).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")

    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
        role_id=payload.role_id,
        is_active=payload.is_active,
    )
    db.add(user)
    await db.flush()  # 取得 user.id 以记审计
    audit.log(db, "user.create", target_type="user", target_id=user.id, target_name=user.username,
              detail={"role_id": str(payload.role_id) if payload.role_id else None, "is_active": payload.is_active})
    await db.commit()
    await db.refresh(user, attribute_names=["role"])
    return _to_out(user)


@router.put("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    current: User = Depends(require_permission(USER_MANAGE)),
):
    user = (await db.execute(
        select(User).options(selectinload(User.role)).where(User.id == user_id)
    )).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 防锁死：禁用最后一个 active 超管被拒绝
    if (
        payload.is_active is False
        and user.is_superadmin
        and user.is_active
        and await _active_superadmin_count(db) <= 1
    ):
        raise HTTPException(status_code=400, detail="不能禁用最后一个超级管理员")

    # 防自我锁死：当前用户不能撤销自己的 user:manage 权限来源（改角色/禁用自己）
    if user.id == current.id and not current.is_superadmin:
        if payload.is_active is False:
            raise HTTPException(status_code=400, detail="不能禁用自己")
        if payload.role_id is not None and payload.role_id != user.role_id:
            raise HTTPException(status_code=400, detail="不能修改自己的角色以避免锁死")

    changes: dict = {}  # 记录每个字段的前后值，便于审计追溯"怎么改的"
    if payload.display_name is not None and payload.display_name != user.display_name:
        changes["display_name"] = {"from": user.display_name, "to": payload.display_name}
        user.display_name = payload.display_name
    if payload.role_id is not None and payload.role_id != user.role_id:
        old_role = user.role.name if user.role else None              # 旧角色名（关系已 eager load）
        new_role = await db.get(Role, payload.role_id) if payload.role_id else None
        changes["role"] = {"from": old_role, "to": (new_role.name if new_role else None)}
        user.role_id = payload.role_id
    if payload.is_active is not None and payload.is_active != user.is_active:
        changes["is_active"] = {"from": user.is_active, "to": payload.is_active}
        user.is_active = payload.is_active
    if payload.password is not None and payload.password.strip():
        user.password_hash = hash_password(payload.password)
        changes["password"] = "已重置"  # 只记动作，绝不记密码值

    if changes:  # 空保存（无任何实际改动）不记审计
        audit.log(db, "user.update", target_type="user", target_id=user.id,
                  target_name=user.username, detail={"changes": changes})
    await db.commit()
    await db.refresh(user, attribute_names=["role"])
    return _to_out(user)


@router.delete("/{user_id}")
async def delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    current: User = Depends(require_permission(USER_MANAGE)),
):
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.id == current.id:
        raise HTTPException(status_code=400, detail="不能删除自己")
    # 防锁死：删除最后一个 active 超管被拒绝
    if user.is_superadmin and user.is_active and await _active_superadmin_count(db) <= 1:
        raise HTTPException(status_code=400, detail="不能删除最后一个超级管理员")
    audit.log(db, "user.delete", target_type="user", target_id=user.id, target_name=user.username)
    await db.delete(user)
    await db.commit()
    return {"message": "删除成功"}
