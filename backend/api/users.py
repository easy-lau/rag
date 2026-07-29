"""用户管理路由：CRUD + 启用/禁用，全部需要 user:manage 权限。"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.audit import AuditLogger, get_audit
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import (
    SUPERADMIN_ROLE_CODE,
    USER_MANAGE,
    capability_closure,
    non_superadmin_delegation_error,
)
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


def _role_capabilities(role: Role | None) -> set[str]:
    if role is None:
        return set()
    return capability_closure(
        (p.permission_key for p in role.permissions),
        scope_mode=role.scope_mode,
    )


def _role_kb_ids(role: Role | None) -> set[uuid.UUID]:
    if role is None:
        return set()
    return {grant.kb_id for grant in role.knowledge_bases}


async def _ensure_role_scope_within_actor(role: Role | None, current: User, db: AsyncSession) -> None:
    if current.is_superadmin or role is None:
        return
    accessible = await get_accessible_kb_ids(current, db)
    if accessible is None:
        return
    if not _role_kb_ids(role).issubset(set(accessible)):
        raise HTTPException(status_code=403, detail="不能分配超出当前账号知识库范围的角色")


async def _load_role_for_assignment(
    role_id: uuid.UUID | None,
    current: User,
    db: AsyncSession,
) -> Role | None:
    """加载并校验可分配角色，防止 user:manage 借角色分配完成提权。"""
    if role_id is None:
        return None
    role = (await db.execute(
        select(Role)
        .options(selectinload(Role.permissions), selectinload(Role.knowledge_bases))
        .where(Role.id == role_id)
    )).scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=400, detail="所选角色不存在")
    is_superadmin_role = role.code == SUPERADMIN_ROLE_CODE or (await db.execute(
        select(User.id)
        .where(User.is_superadmin.is_(True), User.role_id == role.id)
        .limit(1)
    )).scalar_one_or_none() is not None
    if is_superadmin_role:
        raise HTTPException(status_code=403, detail="超级管理员角色不可分配给普通账号")
    if not role.is_assignable:
        raise HTTPException(status_code=403, detail="该系统角色不可分配")
    if not current.is_superadmin:
        reason = non_superadmin_delegation_error(
            getattr(current, "permissions", []),
            _role_capabilities(role),
            role.scope_mode,
            target_is_assignable=role.is_assignable,
        )
        if reason:
            raise HTTPException(status_code=403, detail=reason)
    await _ensure_role_scope_within_actor(role, current, db)
    return role


async def _ensure_target_user_manageable(current: User, target: User, db: AsyncSession) -> None:
    """非超管不能重置或删除高权限账号，避免通过接管账号间接提权。"""
    if current.is_superadmin:
        return
    if target.is_superadmin:
        raise HTTPException(status_code=403, detail="只有超级管理员可以管理超级管理员账号")
    role = target.role
    reason = non_superadmin_delegation_error(
        getattr(current, "permissions", []),
        _role_capabilities(role),
        getattr(role, "scope_mode", "none") if role else "none",
        target_is_assignable=getattr(role, "is_assignable", True) if role else True,
    )
    if reason:
        raise HTTPException(status_code=403, detail="不能管理权限级别高于当前账号的用户")
    await _ensure_role_scope_within_actor(role, current, db)


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
    current: User = Depends(require_permission(USER_MANAGE)),
):
    exists = (await db.execute(
        select(User).where(User.username == payload.username)
    )).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=400, detail="用户名已存在")

    role = await _load_role_for_assignment(payload.role_id, current, db)
    user = User(
        username=payload.username,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
        role_id=role.id if role else None,
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
        select(User)
        .options(
            selectinload(User.role).selectinload(Role.permissions),
            selectinload(User.role).selectinload(Role.knowledge_bases),
        )
        .where(User.id == user_id)
    )).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    await _ensure_target_user_manageable(current, user, db)

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
        if "role_id" in payload.model_fields_set and payload.role_id != user.role_id:
            raise HTTPException(status_code=400, detail="不能修改自己的角色以避免锁死")

    if user.is_superadmin and "role_id" in payload.model_fields_set and payload.role_id != user.role_id:
        raise HTTPException(status_code=400, detail="超级管理员的系统角色不可调整")

    changes: dict = {}  # 记录每个字段的前后值，便于审计追溯"怎么改的"
    if "display_name" in payload.model_fields_set and payload.display_name != user.display_name:
        changes["display_name"] = {"from": user.display_name, "to": payload.display_name}
        user.display_name = payload.display_name
    if "role_id" in payload.model_fields_set and payload.role_id != user.role_id:
        old_role = user.role.name if user.role else None              # 旧角色名（关系已 eager load）
        new_role = await _load_role_for_assignment(payload.role_id, current, db)
        changes["role"] = {"from": old_role, "to": (new_role.name if new_role else None)}
        user.role_id = new_role.id if new_role else None
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
    user = (await db.execute(
        select(User)
        .options(
            selectinload(User.role).selectinload(Role.permissions),
            selectinload(User.role).selectinload(Role.knowledge_bases),
        )
        .where(User.id == user_id)
    )).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.id == current.id:
        raise HTTPException(status_code=400, detail="不能删除自己")
    await _ensure_target_user_manageable(current, user, db)
    # 防锁死：删除最后一个 active 超管被拒绝
    if user.is_superadmin and user.is_active and await _active_superadmin_count(db) <= 1:
        raise HTTPException(status_code=400, detail="不能删除最后一个超级管理员")
    audit.log(db, "user.delete", target_type="user", target_id=user.id, target_name=user.username)
    await db.delete(user)
    await db.commit()
    return {"message": "删除成功"}
