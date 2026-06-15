"""鉴权依赖：当前用户解析、权限校验、可访问知识库范围。"""

import uuid

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.permissions import ALL_PERMISSIONS, KB_ACCESS_ALL
from core.security import decode_access_token
from database import get_db
from models.db_models import RoleKnowledgeBase, RolePermission, User

bearer_scheme = HTTPBearer(auto_error=False)


async def _load_permissions(user: User, db: AsyncSession) -> list[str]:
    """计算用户权限列表：超管拥有全部权限，否则取其角色的 role_permissions。"""
    if user.is_superadmin:
        return list(ALL_PERMISSIONS)
    if user.role_id is None:
        return []
    result = await db.execute(
        select(RolePermission.permission_key).where(RolePermission.role_id == user.role_id)
    )
    return [row[0] for row in result.all()]


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """解析 Authorization: Bearer，加载 user 并校验有效性。

    权限列表计算好后挂在 user.permissions 属性上供后续依赖消费。
    """
    if creds is None:
        raise HTTPException(status_code=401, detail="未认证")

    payload = decode_access_token(creds.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="令牌无效或已过期")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="令牌无效")

    try:
        user_id = uuid.UUID(sub)
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="令牌无效")

    result = await db.execute(
        select(User).options(selectinload(User.role)).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="用户已禁用")

    # 计算并挂载权限列表，供 require_permission 等依赖直接消费
    user.permissions = await _load_permissions(user, db)
    return user


def require_permission(key: str):
    """依赖工厂：校验当前用户是否拥有指定权限 key；超管直接放行。"""

    async def checker(user: User = Depends(get_current_user)) -> User:
        if user.is_superadmin:
            return user
        if key not in getattr(user, "permissions", []):
            raise HTTPException(status_code=403, detail="无权限")
        return user

    return checker


async def get_accessible_kb_ids(user: User, db: AsyncSession) -> list[uuid.UUID] | None:
    """返回用户可访问的知识库 id 列表。

    超管或拥有 kb:access_all 权限时返回 None，代表"全部知识库"。
    否则查 role_knowledge_bases 返回被授权的 kb_id 列表。
    """
    permissions = getattr(user, "permissions", None)
    if permissions is None:
        permissions = await _load_permissions(user, db)
    if user.is_superadmin or KB_ACCESS_ALL in permissions:
        return None
    if user.role_id is None:
        return []
    result = await db.execute(
        select(RoleKnowledgeBase.kb_id).where(RoleKnowledgeBase.role_id == user.role_id)
    )
    return [row[0] for row in result.all()]


async def ensure_kb_access(user: User, kb_id: uuid.UUID, db: AsyncSession) -> None:
    """校验用户对指定知识库有访问权；无权抛 403。

    get_accessible_kb_ids 返回 None 表示"全部知识库"（超管 / kb:access_all），直接放行。
    """
    accessible = await get_accessible_kb_ids(user, db)
    if accessible is not None and kb_id not in accessible:
        raise HTTPException(status_code=403, detail="无权访问该知识库")


def require_kb_access(permission_key: str):
    """依赖工厂：先校验操作权限，再校验对路径参数 kb_id 的访问权。

    FastAPI 会把路由路径中的 {kb_id} 自动注入本依赖，因此接口签名无需改动函数体。
    """

    async def checker(
        kb_id: uuid.UUID,
        user: User = Depends(require_permission(permission_key)),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        await ensure_kb_access(user, kb_id, db)
        return user

    return checker
