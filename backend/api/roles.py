"""角色管理路由：CRUD + 权限/知识库分配 + 权限目录，全部需要 role:manage 权限。"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.deps import require_permission
from core.permissions import ALL_PERMISSIONS, MENUS, ROLE_MANAGE
from database import get_db
from models.db_models import Role, RoleKnowledgeBase, RolePermission, User
from models.schemas import RoleCreate, RoleOut, RoleUpdate

router = APIRouter(prefix="/roles", tags=["roles"])


def _to_out(role: Role) -> RoleOut:
    """构造带 permissions[] / kb_ids[] 的 RoleOut（依赖已 eager load 关系）。"""
    return RoleOut(
        id=role.id,
        name=role.name,
        description=role.description,
        is_system=role.is_system,
        permissions=[p.permission_key for p in role.permissions],
        kb_ids=[k.kb_id for k in role.knowledge_bases],
        created_at=role.created_at,
    )


@router.get("/permission-catalog")
async def permission_catalog(_: User = Depends(require_permission(ROLE_MANAGE))):
    """返回全部权限 key 与菜单清单，供前端分配矩阵渲染。"""
    return {"permissions": ALL_PERMISSIONS, "menus": MENUS}


@router.get("", response_model=list[RoleOut])
async def list_roles(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(ROLE_MANAGE)),
):
    rows = (await db.execute(
        select(Role)
        .options(selectinload(Role.permissions), selectinload(Role.knowledge_bases))
        .order_by(Role.created_at.desc())
    )).scalars().all()
    return [_to_out(r) for r in rows]


@router.post("", response_model=RoleOut)
async def create_role(
    payload: RoleCreate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(ROLE_MANAGE)),
):
    exists = (await db.execute(
        select(Role).where(Role.name == payload.name)
    )).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=400, detail="角色名已存在")

    role = Role(name=payload.name, description=payload.description)
    for key in set(payload.permissions):
        role.permissions.append(RolePermission(permission_key=key))
    for kb_id in set(payload.kb_ids):
        role.knowledge_bases.append(RoleKnowledgeBase(kb_id=kb_id))
    db.add(role)
    await db.commit()
    await db.refresh(role, attribute_names=["permissions", "knowledge_bases"])
    return _to_out(role)


@router.put("/{role_id}", response_model=RoleOut)
async def update_role(
    role_id: uuid.UUID,
    payload: RoleUpdate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(ROLE_MANAGE)),
):
    role = (await db.execute(
        select(Role)
        .options(selectinload(Role.permissions), selectinload(Role.knowledge_bases))
        .where(Role.id == role_id)
    )).scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=404, detail="角色不存在")

    if payload.name is not None:
        role.name = payload.name
    if payload.description is not None:
        role.description = payload.description

    # 覆盖式重写权限：先 clear 再 flush，把删除落库后再插入，
    # 否则同一次 flush 中新行 INSERT 可能早于旧行 DELETE，撞唯一约束。
    if payload.permissions is not None:
        role.permissions.clear()
        await db.flush()
        for key in set(payload.permissions):
            role.permissions.append(RolePermission(permission_key=key))
    # 覆盖式重写知识库授权（同上，避免 uq_role_kb 冲突）
    if payload.kb_ids is not None:
        role.knowledge_bases.clear()
        await db.flush()
        for kb_id in set(payload.kb_ids):
            role.knowledge_bases.append(RoleKnowledgeBase(kb_id=kb_id))

    await db.commit()
    await db.refresh(role, attribute_names=["permissions", "knowledge_bases"])
    return _to_out(role)


@router.delete("/{role_id}")
async def delete_role(
    role_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(ROLE_MANAGE)),
):
    role = await db.get(Role, role_id)
    if role is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    if role.is_system:
        raise HTTPException(status_code=400, detail="内建角色不可删除")

    # 有用户在用时拒绝，提示先改派用户角色
    in_use = (await db.execute(
        select(func.count()).select_from(User).where(User.role_id == role_id)
    )).scalar_one()
    if in_use > 0:
        raise HTTPException(
            status_code=400,
            detail=f"该角色下还有 {in_use} 个用户，请先改派这些用户的角色后再删除",
        )

    await db.delete(role)
    await db.commit()
    return {"message": "删除成功"}
