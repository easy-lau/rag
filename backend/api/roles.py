"""角色管理：能力、知识库范围和委派边界的唯一写入入口。"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.audit import AuditLogger, get_audit
from core.deps import get_accessible_kb_ids, require_permission
from core.permissions import (
    ASSIGNABLE_CAPABILITIES,
    KB_SCOPE_ALL,
    KB_SCOPE_NONE,
    KB_SCOPE_SELECTED,
    MENUS,
    ROLE_MANAGE,
    SUPERADMIN_ROLE_CODE,
    USER_MANAGE,
    capability_catalog_payload,
    capability_closure,
    non_superadmin_delegation_error,
    normalize_assignable_capabilities,
    normalize_scope_mode,
    validate_capability_scope,
)
from database import get_db
from models.db_models import KnowledgeBase, Role, RoleKnowledgeBase, RolePermission, User
from models.schemas import RoleCreate, RoleOut, RoleUpdate

router = APIRouter(prefix="/roles", tags=["roles"])

_ROLE_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _field_was_supplied(payload: object, field: str) -> bool:
    """Work with Pydantic v2 and retain a small compatibility fallback."""
    fields_set = getattr(payload, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(payload, "__fields_set__", set())
    return field in fields_set


def _role_scope(role: Role) -> str:
    # A deployment that starts the app between code and migration rollout may
    # still hold a legacy role object.  The dependency layer keeps the same
    # fallback; after 0019 this always resolves to the stored scope_mode.
    return getattr(role, "scope_mode", None) or KB_SCOPE_NONE


def _role_capabilities(role: Role) -> set[str]:
    return capability_closure(
        (p.permission_key for p in role.permissions),
        scope_mode=_role_scope(role),
    )


def _actor_capabilities(actor: User) -> set[str]:
    return set(getattr(actor, "permissions", [])) & set(ASSIGNABLE_CAPABILITIES)


def _is_reserved_superadmin_role(role: Role, superadmin_role_ids: set[uuid.UUID] | None = None) -> bool:
    """Identify the non-delegable superadmin definition without using its name."""
    return (
        getattr(role, "code", None) == SUPERADMIN_ROLE_CODE
        or (superadmin_role_ids is not None and role.id in superadmin_role_ids)
    )


def _validate_superadmin_role_update(
    role: Role,
    *,
    desired_code: str | None,
    desired_assignable: bool,
    linked_to_superadmin: bool,
) -> None:
    """Keep the built-in superadmin role identity and assignment lock immutable."""
    is_superadmin_role = _is_reserved_superadmin_role(
        role,
        {role.id} if linked_to_superadmin else set(),
    )
    if desired_code == SUPERADMIN_ROLE_CODE and not is_superadmin_role:
        raise HTTPException(status_code=400, detail="superadmin 是系统保留角色编码")
    if not is_superadmin_role:
        return
    if role.code == SUPERADMIN_ROLE_CODE and desired_code != SUPERADMIN_ROLE_CODE:
        raise HTTPException(status_code=400, detail="超级管理员角色编码不可修改")
    if desired_assignable:
        raise HTTPException(status_code=400, detail="超级管理员角色不可分配给普通账号")


def _to_out(role: Role) -> RoleOut:
    """Construct RoleOut from eager-loaded relations without exposing menu grants."""
    capabilities = _role_capabilities(role)
    return RoleOut(
        id=role.id,
        name=role.name,
        description=role.description,
        code=getattr(role, "code", None),
        is_system=role.is_system,
        is_assignable=getattr(role, "is_assignable", True),
        scope_mode=_role_scope(role),
        permissions=[key for key in ASSIGNABLE_CAPABILITIES if key in capabilities],
        kb_ids=[k.kb_id for k in role.knowledge_bases],
        created_at=role.created_at,
    )


async def _ensure_kb_ids_exist(db: AsyncSession, kb_ids: Iterable[uuid.UUID]) -> None:
    ids = set(kb_ids)
    if not ids:
        return
    found = set((await db.execute(
        select(KnowledgeBase.id).where(KnowledgeBase.id.in_(ids))
    )).scalars().all())
    missing = ids - found
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"存在不存在的知识库：{', '.join(str(item) for item in sorted(missing, key=str))}",
        )


def _normalize_code(raw: str | None) -> str | None:
    code = raw.strip().lower() if raw else None
    if code and not _ROLE_CODE_PATTERN.fullmatch(code):
        raise HTTPException(status_code=400, detail="角色编码只能包含小写字母、数字和下划线，且以字母开头")
    return code


async def _ensure_code_available(
    db: AsyncSession,
    code: str | None,
    *,
    except_role_id: uuid.UUID | None = None,
) -> None:
    if not code:
        return
    stmt = select(Role.id).where(Role.code == code)
    if except_role_id is not None:
        stmt = stmt.where(Role.id != except_role_id)
    if (await db.execute(stmt)).scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="角色编码已存在")


def _assert_non_superadmin_may_manage_existing(actor: User, role: Role) -> None:
    if actor.is_superadmin:
        return
    if actor.role_id == role.id:
        raise HTTPException(status_code=403, detail="不能编辑或删除自己的角色")
    if role.is_system:
        raise HTTPException(status_code=403, detail="只有超级管理员可以管理系统角色")
    error = non_superadmin_delegation_error(
        _actor_capabilities(actor),
        _role_capabilities(role),
        _role_scope(role),
        target_is_assignable=getattr(role, "is_assignable", True),
    )
    if error:
        raise HTTPException(status_code=403, detail=error)


def _assert_non_superadmin_may_delegate(
    actor: User,
    capabilities: Iterable[str],
    scope_mode: str,
    *,
    is_assignable: bool,
) -> None:
    if actor.is_superadmin:
        return
    error = non_superadmin_delegation_error(
        _actor_capabilities(actor), capabilities, scope_mode, target_is_assignable=is_assignable
    )
    if error:
        raise HTTPException(status_code=403, detail=error)


def _kb_scope_is_within_actor_access(
    scope_mode: str,
    kb_ids: Iterable[uuid.UUID],
    accessible_ids: list[uuid.UUID] | None,
) -> bool:
    """Whether a selected scope contains only KBs the actor can access.

    ``None`` represents all accessible KBs.  All-scope roles are rejected for
    non-superadmins by the delegation rule before this helper is reached.
    """
    if scope_mode != KB_SCOPE_SELECTED:
        return True
    return accessible_ids is None or set(kb_ids).issubset(set(accessible_ids))


async def _ensure_actor_can_delegate_kb_scope(
    actor: User,
    scope_mode: str,
    kb_ids: Iterable[uuid.UUID],
    db: AsyncSession,
) -> None:
    if actor.is_superadmin:
        return
    accessible_ids = await get_accessible_kb_ids(actor, db)
    if not _kb_scope_is_within_actor_access(scope_mode, kb_ids, accessible_ids):
        allowed = set(accessible_ids or [])
        outside = sorted(set(kb_ids) - allowed, key=str)
        raise HTTPException(
            status_code=403,
            detail="不能授予自己无权访问的知识库：" + ", ".join(str(item) for item in outside),
        )


def _resolve_update_scope(role: Role, payload: RoleUpdate) -> tuple[str, set[uuid.UUID], bool]:
    """Resolve compatibility payloads without silently downgrading all scope.

    Legacy clients only submit ``kb_ids``.  When such a client edits an
    all-scope role, retaining all scope is safer than interpreting its empty
    list as a request to revoke global access.  New clients send scope_mode
    explicitly and receive strict mutual-exclusion validation.
    """
    old_scope = _role_scope(role)
    old_ids = {item.kb_id for item in role.knowledge_bases}
    # ``null`` from a compatibility client is treated like omission.  In
    # particular it must not turn an existing all-scope role into none.
    scope_supplied = _field_was_supplied(payload, "scope_mode") and payload.scope_mode is not None
    ids_supplied = payload.kb_ids is not None
    if not scope_supplied:
        if not ids_supplied:
            return old_scope, old_ids, False
        if old_scope == KB_SCOPE_ALL:
            return old_scope, old_ids, False
        mode, ids = normalize_scope_mode(None, payload.kb_ids or [])
        return mode, ids, True

    # Explicit selected with omitted IDs means "keep the current selection";
    # explicit all/none never retain selected IDs.
    requested_scope = payload.scope_mode
    normalized_requested_scope = (requested_scope or "").strip().lower()
    requested_ids = payload.kb_ids if ids_supplied else (old_ids if normalized_requested_scope == "selected" else [])
    mode, ids = normalize_scope_mode(requested_scope, requested_ids)
    return mode, ids, mode != old_scope or ids != old_ids


@router.get("/permission-catalog")
async def permission_catalog(_: User = Depends(require_permission(ROLE_MANAGE))):
    """返回可分配能力及后端维护的目录/模板元数据。

    ``permissions`` 保持 string[] 的旧合同；菜单是 capabilities 的派生展示，
    不能保存到 role_permissions。
    """
    catalog = capability_catalog_payload()
    return {
        "permissions": list(ASSIGNABLE_CAPABILITIES),
        "menus": [dict(menu) for menu in MENUS],
        "catalog": catalog,
        "templates": catalog["templates"],
        "scope_modes": catalog["scope_modes"],
    }


@router.get("/assignable", response_model=list[RoleOut])
async def list_assignable_roles(
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(require_permission(USER_MANAGE)),
):
    """返回当前操作者可安全分配给用户的角色。"""
    rows = (await db.execute(
        select(Role)
        .options(selectinload(Role.permissions), selectinload(Role.knowledge_bases))
        .order_by(Role.created_at.desc())
    )).scalars().all()
    superadmin_role_ids = set((await db.execute(
        select(User.role_id).where(User.is_superadmin.is_(True), User.role_id.is_not(None))
    )).scalars().all())
    if actor.is_superadmin:
        return [
            _to_out(role)
            for role in rows
            if getattr(role, "is_assignable", True)
            and not _is_reserved_superadmin_role(role, superadmin_role_ids)
        ]

    accessible_ids = await get_accessible_kb_ids(actor, db)
    allowed: list[RoleOut] = []
    for role in rows:
        if (
            not getattr(role, "is_assignable", True)
            or _is_reserved_superadmin_role(role, superadmin_role_ids)
            or actor.role_id == role.id
        ):
            continue
        try:
            # “可以分配给用户”不等于“可以编辑角色定义”。例如内建普通用户角色
            # 可以被用户管理员分配，但仍只能由超级管理员修改。
            _assert_non_superadmin_may_delegate(
                actor,
                _role_capabilities(role),
                _role_scope(role),
                is_assignable=True,
            )
        except HTTPException:
            continue
        if not _kb_scope_is_within_actor_access(
            _role_scope(role),
            (item.kb_id for item in role.knowledge_bases),
            accessible_ids,
        ):
            continue
        allowed.append(_to_out(role))
    return allowed


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
    return [_to_out(role) for role in rows]


@router.post("", response_model=RoleOut)
async def create_role(
    payload: RoleCreate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    actor: User = Depends(require_permission(ROLE_MANAGE)),
):
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="角色名不能为空")
    if (await db.execute(select(Role.id).where(Role.name == name))).scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="角色名已存在")

    try:
        capabilities = normalize_assignable_capabilities(payload.permissions)
        # Older clients send only kb_ids.  Inspect the fields actually present
        # rather than the schema default so selected scope can be inferred.
        requested_scope = payload.scope_mode if _field_was_supplied(payload, "scope_mode") else None
        scope_mode, kb_ids = normalize_scope_mode(requested_scope, payload.kb_ids)
        validate_capability_scope(capabilities, scope_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _ensure_kb_ids_exist(db, kb_ids)

    code = _normalize_code(payload.code)
    is_assignable = payload.is_assignable
    if code == SUPERADMIN_ROLE_CODE:
        raise HTTPException(status_code=400, detail="superadmin 是系统保留角色编码")
    if not actor.is_superadmin:
        # Generated clients often echo schema defaults.  Explicit null/true is
        # equivalent to the only values a non-superadmin may create anyway.
        if code is not None or is_assignable is not True:
            raise HTTPException(status_code=403, detail="只有超级管理员可以设置角色编码或可分配状态")
        _assert_non_superadmin_may_delegate(actor, capabilities, scope_mode, is_assignable=True)
        await _ensure_actor_can_delegate_kb_scope(actor, scope_mode, kb_ids, db)
    await _ensure_code_available(db, code)

    role = Role(
        name=name,
        description=payload.description,
        code=code,
        scope_mode=scope_mode,
        is_assignable=is_assignable,
    )
    for key in capabilities:
        role.permissions.append(RolePermission(permission_key=key))
    for kb_id in kb_ids:
        role.knowledge_bases.append(RoleKnowledgeBase(kb_id=kb_id))
    db.add(role)
    await db.flush()
    audit.log(
        db,
        "role.create",
        target_type="role",
        target_id=role.id,
        target_name=role.name,
        detail={"code": code, "permissions": sorted(capabilities), "scope_mode": scope_mode, "kb_ids": len(kb_ids)},
    )
    await db.commit()
    await db.refresh(role, attribute_names=["permissions", "knowledge_bases"])
    return _to_out(role)


@router.put("/{role_id}", response_model=RoleOut)
async def update_role(
    role_id: uuid.UUID,
    payload: RoleUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    actor: User = Depends(require_permission(ROLE_MANAGE)),
):
    role = (await db.execute(
        select(Role)
        .options(selectinload(Role.permissions), selectinload(Role.knowledge_bases))
        .where(Role.id == role_id)
    )).scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    _assert_non_superadmin_may_manage_existing(actor, role)

    old_capabilities = _role_capabilities(role)
    old_scope = _role_scope(role)
    old_kbs = {item.kb_id for item in role.knowledge_bases}
    if payload.permissions is None:
        capabilities = old_capabilities
    else:
        try:
            capabilities = normalize_assignable_capabilities(payload.permissions)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        scope_mode, kb_ids, scope_changed = _resolve_update_scope(role, payload)
        validate_capability_scope(capabilities, scope_mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _ensure_kb_ids_exist(db, kb_ids)

    desired_assignable = role.is_assignable
    desired_code = role.code
    if _field_was_supplied(payload, "code"):
        desired_code = _normalize_code(payload.code)
    if _field_was_supplied(payload, "is_assignable"):
        if payload.is_assignable is None:
            raise HTTPException(status_code=400, detail="is_assignable 只能是 true 或 false")
        desired_assignable = payload.is_assignable
    linked_to_superadmin = (await db.execute(
        select(User.id)
        .where(User.is_superadmin.is_(True), User.role_id == role.id)
        .limit(1)
    )).scalar_one_or_none() is not None
    _validate_superadmin_role_update(
        role,
        desired_code=desired_code,
        desired_assignable=desired_assignable,
        linked_to_superadmin=linked_to_superadmin,
    )
    if not actor.is_superadmin:
        if desired_code != role.code or desired_assignable != role.is_assignable:
            raise HTTPException(status_code=403, detail="只有超级管理员可以设置角色编码或可分配状态")
        _assert_non_superadmin_may_delegate(actor, capabilities, scope_mode, is_assignable=True)
        await _ensure_actor_can_delegate_kb_scope(actor, scope_mode, kb_ids, db)
    await _ensure_code_available(db, desired_code, except_role_id=role.id)

    changes: dict[str, object] = {}
    if payload.name is not None:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="角色名不能为空")
        if name != role.name:
            if (await db.execute(select(Role.id).where(Role.name == name, Role.id != role.id))).scalar_one_or_none() is not None:
                raise HTTPException(status_code=400, detail="角色名已存在")
            changes["name"] = {"from": role.name, "to": name}
            role.name = name
    if _field_was_supplied(payload, "description") and payload.description != role.description:
        changes["description"] = {"from": role.description, "to": payload.description}
        role.description = payload.description
    if desired_code != role.code:
        changes["code"] = {"from": role.code, "to": desired_code}
        role.code = desired_code
    if desired_assignable != role.is_assignable:
        changes["is_assignable"] = {"from": role.is_assignable, "to": desired_assignable}
        role.is_assignable = desired_assignable
    if capabilities != old_capabilities:
        changes["permissions"] = {
            "added": sorted(capabilities - old_capabilities),
            "removed": sorted(old_capabilities - capabilities),
        }
        role.permissions.clear()
        await db.flush()
        for key in capabilities:
            role.permissions.append(RolePermission(permission_key=key))
    if scope_changed:
        changes["scope_mode"] = {"from": old_scope, "to": scope_mode}
        role.scope_mode = scope_mode
    if kb_ids != old_kbs:
        name_map = dict((await db.execute(
            select(KnowledgeBase.id, KnowledgeBase.name).where(KnowledgeBase.id.in_(kb_ids | old_kbs))
        )).all()) if (kb_ids or old_kbs) else {}
        changes["kb_ids"] = {
            "added": [name_map.get(item, str(item)) for item in sorted(kb_ids - old_kbs, key=str)],
            "removed": [name_map.get(item, str(item)) for item in sorted(old_kbs - kb_ids, key=str)],
        }
        role.knowledge_bases.clear()
        await db.flush()
        for kb_id in kb_ids:
            role.knowledge_bases.append(RoleKnowledgeBase(kb_id=kb_id))

    if changes:
        audit.log(db, "role.update", target_type="role", target_id=role.id, target_name=role.name, detail={"changes": changes})
    await db.commit()
    await db.refresh(role, attribute_names=["permissions", "knowledge_bases"])
    return _to_out(role)


@router.delete("/{role_id}")
async def delete_role(
    role_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    actor: User = Depends(require_permission(ROLE_MANAGE)),
):
    role = (await db.execute(
        select(Role).options(selectinload(Role.permissions), selectinload(Role.knowledge_bases)).where(Role.id == role_id)
    )).scalar_one_or_none()
    if role is None:
        raise HTTPException(status_code=404, detail="角色不存在")
    if role.is_system:
        raise HTTPException(status_code=400, detail="系统角色不可删除")
    _assert_non_superadmin_may_manage_existing(actor, role)
    # Deleting a role is also a management operation on its delegated data
    # scope.  Capability checks alone are insufficient: otherwise a scoped
    # role administrator could delete a low-risk role belonging to another KB.
    await _ensure_actor_can_delegate_kb_scope(
        actor,
        _role_scope(role),
        (item.kb_id for item in role.knowledge_bases),
        db,
    )

    in_use = (await db.execute(
        select(func.count()).select_from(User).where(User.role_id == role_id)
    )).scalar_one()
    if in_use > 0:
        raise HTTPException(status_code=400, detail=f"该角色下还有 {in_use} 个用户，请先改派这些用户的角色后再删除")

    audit.log(db, "role.delete", target_type="role", target_id=role.id, target_name=role.name)
    await db.delete(role)
    await db.commit()
    return {"message": "删除成功"}
