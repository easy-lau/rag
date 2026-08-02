import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text, bindparam
from sqlalchemy import UUID as SA_UUID
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import selectinload
from database import get_db
from models.db_models import (
    KnowledgeBase,
    Document,
    TerminologyRegistryState,
    User,
)
from models.schemas import KnowledgeBaseCreate, KnowledgeBaseOut
from core.audit import AuditLogger, get_audit
from core.deps import (
    get_accessible_kb_ids,
    require_all_kb_scope,
    require_any_permission,
    require_kb_access,
)
from core.permissions import (
    CHAT_USE,
    KB_CREATE,
    KB_DELETE,
    KB_READ,
    KB_UPDATE,
    ROLE_MANAGE,
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _name(u):
    return (u.display_name or u.username) if u else None


async def _doc_count(db: AsyncSession, kb_id: uuid.UUID) -> int:
    """按真实文档行数统计（doc_count 已不再持久化）。"""
    return (await db.execute(
        select(func.count()).select_from(Document).where(Document.kb_id == kb_id)
    )).scalar_one()


@router.get("/list", response_model=list[KnowledgeBaseOut])
async def list_knowledge_bases(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_any_permission(CHAT_USE, KB_READ, ROLE_MANAGE)),
):
    # 非全权用户仅返回被授权的 KB；accessible 为 None 表示全部
    accessible = await get_accessible_kb_ids(user, db)
    stmt = (
        select(KnowledgeBase)
        .options(selectinload(KnowledgeBase.creator))
        .order_by(KnowledgeBase.created_at.desc())
    )
    if accessible is not None:
        if not accessible:
            return []
        stmt = stmt.where(KnowledgeBase.id.in_(accessible))
    rows = (await db.execute(stmt)).scalars().all()
    # doc_count 历史上是手工维护的冗余计数，易与真实行数漂移（上传 +1、并发删除丢更新）。
    # 这里直接按 documents 表的真实行数返回，确保展示永远准确，与文档管理列表一致。
    counts = dict((await db.execute(
        select(Document.kb_id, func.count()).group_by(Document.kb_id)
    )).all())
    for kb in rows:
        kb.created_by_name = _name(kb.creator)
        kb.doc_count = counts.get(kb.id, 0)
    return rows


@router.get("/tags", response_model=list[str])
async def list_document_tags(
    kb_ids: list[uuid.UUID] | None = Query(None),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_any_permission(CHAT_USE, KB_READ)),
):
    """返回（指定/全部可访问）知识库下文档用过的去重标签，供问答页标签选择器使用。"""
    accessible = await get_accessible_kb_ids(user, db)
    if accessible is not None:
        allowed = set(accessible)
        target = [k for k in kb_ids if k in allowed] if kb_ids else list(allowed)
        if not target:
            return []
    else:
        # 全权用户：有指定就按指定，否则全部文档
        target = list(kb_ids) if kb_ids else None

    sql = "SELECT DISTINCT jsonb_array_elements_text(tags) AS tag FROM documents"
    params = {}
    if target is not None:
        sql += " WHERE kb_id = ANY(:kb_ids)"
        params["kb_ids"] = target
    sql += " ORDER BY tag"

    stmt = text(sql)
    if target is not None:
        stmt = stmt.bindparams(bindparam("kb_ids", type_=ARRAY(SA_UUID())))
    rows = (await db.execute(stmt, params)).scalars().all()
    return [r for r in rows if r]


@router.post("/create", response_model=KnowledgeBaseOut)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_all_kb_scope(KB_CREATE)),
):
    kb = KnowledgeBase(
        name=payload.name,
        description=payload.description,
        icon_color=payload.icon_color,
        created_by=user.id,
    )
    db.add(kb)
    await db.flush()  # 取得 kb.id，以在同一事务初始化该 KB 的术语 revision 状态
    db.add(TerminologyRegistryState(
        kb_id=kb.id,
        revision=0,
        updated_by=user.id,
    ))
    audit.log(db, "kb.create", target_type="knowledge_base", target_id=kb.id, target_name=kb.name)
    await db.commit()
    await db.refresh(kb)
    # 创建场景下创建人即当前 user，直接取名避免再次查询关系
    kb.created_by_name = _name(user)
    kb.doc_count = 0  # 新建知识库尚无文档
    return kb


@router.put("/{kb_id}", response_model=KnowledgeBaseOut)
async def update_knowledge_base(
    kb_id: uuid.UUID,
    payload: KnowledgeBaseCreate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_kb_access(KB_UPDATE)),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    kb.name = payload.name
    kb.description = payload.description
    kb.icon_color = payload.icon_color
    audit.log(db, "kb.update", target_type="knowledge_base", target_id=kb.id, target_name=kb.name)
    await db.commit()
    await db.refresh(kb, attribute_names=["creator"])
    kb.created_by_name = _name(kb.creator)
    kb.doc_count = await _doc_count(db, kb_id)
    return kb


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_kb_access(KB_DELETE)),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    # 用真实文档行数校验（doc_count 为冗余计数，不计失败/处理中文档，不可作为依据）
    doc_count = (await db.execute(
        select(func.count()).select_from(Document).where(Document.kb_id == kb_id)
    )).scalar_one()
    if doc_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"该知识库下还有 {doc_count} 个文档，请先删除全部文档后再删除知识库",
        )
    audit.log(db, "kb.delete", target_type="knowledge_base", target_id=kb.id, target_name=kb.name)
    await db.delete(kb)
    await db.commit()
    return {"message": "删除成功"}
