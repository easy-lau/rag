import os
import json
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from database import get_db
from models.db_models import Document, DocumentChunk, KnowledgeBase, User, now_utc
from models.schemas import DocumentOut
from core.audit import AuditLogger, get_audit
from core.deps import require_kb_access
from core.document_jobs import enqueue_document_processing_job
from core.permissions import DOC_CREATE, DOC_DELETE, DOC_READ, DOC_UPDATE
from config import get_settings

router = APIRouter(prefix="/knowledge", tags=["documents"])

IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}


def _name(u):
    return (u.display_name or u.username) if u else None


def _normalize_tags(data) -> list[str]:
    """去重、去空白、保持顺序的标签列表。"""
    if not isinstance(data, list):
        return []
    seen, out = set(), []
    for t in data:
        t = str(t).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _parse_tags(raw: str | None) -> list[str]:
    """multipart 上传里标签以 JSON 字符串传入；容错解析为规范化的字符串列表。"""
    if not raw:
        return []
    try:
        return _normalize_tags(json.loads(raw))
    except (ValueError, TypeError):
        return []


def _ext_of(name: str | None) -> str | None:
    """取文件名扩展名（小写、去点），无扩展名返回 None。"""
    ext = os.path.splitext(name or "")[1].lower().lstrip(".")
    return ext or None


@router.get("/{kb_id}/documents", response_model=list[DocumentOut])
async def list_documents(
    kb_id: uuid.UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_kb_access(DOC_READ)),
):
    offset = (page - 1) * page_size
    rows = (await db.execute(
        select(Document)
        .options(selectinload(Document.creator), selectinload(Document.updater))
        .where(Document.kb_id == kb_id)
        .order_by(Document.created_at.desc())
        .offset(offset).limit(page_size)
    )).scalars().all()
    for d in rows:
        d.created_by_name = _name(d.creator)
        d.updated_by_name = _name(d.updater)
    return rows


@router.post("/{kb_id}/documents", response_model=DocumentOut)
async def upload_document(
    kb_id: uuid.UUID,
    file: UploadFile = File(...),
    tags: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(DOC_CREATE)),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    s = get_settings()
    os.makedirs(s.upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    saved_name = f"{uuid.uuid4()}.{ext}"
    saved_path = os.path.join(s.upload_dir, saved_name)

    contents = await file.read()
    with open(saved_path, "wb") as f_out:
        f_out.write(contents)

    doc = Document(kb_id=kb_id, filename=file.filename, file_type=ext, status="processing",
                   tags=_parse_tags(tags), created_by=user.id)
    db.add(doc)
    await db.flush()
    enqueue_document_processing_job(
        db,
        document=doc,
        job_type="file",
        source_path=saved_path,
        original_name=file.filename,
    )
    audit.log(db, "doc.upload", target_type="document", target_id=doc.id, target_name=doc.filename,
              detail={"kb_id": str(kb_id), "file_type": ext})
    await db.commit()
    await db.refresh(doc)
    return doc


@router.post("/{kb_id}/documents/image", response_model=DocumentOut)
async def upload_image_document(
    kb_id: uuid.UUID,
    file: UploadFile = File(...),
    tags: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(DOC_CREATE)),
):
    """上传图片/截图：保存原图 → 多模态模型转写为 Markdown → 分块入库；
    原图保留以便在编辑器中对照校对。"""
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if ext not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="仅支持 PNG / JPG / JPEG / WEBP / GIF / BMP 图片")

    s = get_settings()
    image_dir = os.path.join(s.upload_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    saved_name = f"{uuid.uuid4()}.{ext}"
    saved_path = os.path.join(image_dir, saved_name)

    contents = await file.read()
    with open(saved_path, "wb") as f_out:
        f_out.write(contents)

    doc = Document(
        kb_id=kb_id, filename=file.filename, file_type=ext,
        image_url=f"/api/uploads/images/{saved_name}", status="processing",
        tags=_parse_tags(tags), created_by=user.id,
    )
    db.add(doc)
    await db.flush()
    enqueue_document_processing_job(
        db,
        document=doc,
        job_type="image",
        source_path=saved_path,
        original_name=file.filename,
    )
    audit.log(db, "doc.upload_image", target_type="document", target_id=doc.id, target_name=doc.filename,
              detail={"kb_id": str(kb_id), "file_type": ext})
    await db.commit()
    await db.refresh(doc)
    return doc


class TextDocumentIn(BaseModel):
    title: str
    content: str
    source_url: str | None = None
    tags: list[str] = []


class TagsIn(BaseModel):
    tags: list[str] = []


@router.post("/{kb_id}/documents/text", response_model=DocumentOut)
async def create_text_document(
    kb_id: uuid.UUID,
    body: TextDocumentIn,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(DOC_CREATE)),
):
    kb = await db.get(KnowledgeBase, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    doc = Document(kb_id=kb_id, filename=body.title, file_type="md", raw_content=body.content,
                   source_url=body.source_url, status="processing",
                   tags=_normalize_tags(body.tags), created_by=user.id)
    db.add(doc)
    await db.flush()
    enqueue_document_processing_job(db, document=doc, job_type="text")
    audit.log(db, "doc.create_text", target_type="document", target_id=doc.id, target_name=doc.filename,
              detail={"kb_id": str(kb_id)})
    await db.commit()
    await db.refresh(doc)
    return doc


@router.get("/{kb_id}/documents/{doc_id}", response_model=DocumentOut)
async def get_document(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_kb_access(DOC_READ)),
):
    doc = await db.get(Document, doc_id)
    if not doc or doc.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not doc.raw_content:
        chunks = (await db.execute(
            select(DocumentChunk.content)
            .where(DocumentChunk.doc_id == doc_id)
            .order_by(DocumentChunk.chunk_index)
        )).scalars().all()
        doc.raw_content = "\n\n".join(chunks)
    return doc


@router.put("/{kb_id}/documents/{doc_id}", response_model=DocumentOut)
async def update_text_document(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    body: TextDocumentIn,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(DOC_UPDATE)),
):
    doc = await db.get(Document, doc_id)
    if not doc or doc.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 新任务拥有新的 revision。旧分块只有在新 revision 完整解析、向量化并
    # 原子写入后才会被替换；失败任务不会留下“先清空、后失败”的半成品状态。
    doc.filename = body.title
    doc.raw_content = body.content
    doc.source_url = body.source_url
    # 保留原始类型：编辑内容不应把 pdf/docx/xlsx 等改写成 md。
    # 优先按文件名扩展名识别（可自愈历史误标），无扩展名则维持原类型，最后兜底 md。
    doc.file_type = _ext_of(body.title) or doc.file_type or "md"
    doc.status = "processing"
    doc.chunk_count = 0
    doc.processing_revision += 1
    doc.tags = _normalize_tags(body.tags)
    doc.updated_by = user.id
    doc.updated_at = now_utc()
    enqueue_document_processing_job(db, document=doc, job_type="text")
    audit.log(db, "doc.update", target_type="document", target_id=doc.id, target_name=doc.filename,
              detail={"kb_id": str(kb_id)})
    await db.commit()
    await db.refresh(doc)
    return doc


@router.patch("/{kb_id}/documents/{doc_id}/tags", response_model=DocumentOut)
async def update_document_tags(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    body: TagsIn,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    user: User = Depends(require_kb_access(DOC_UPDATE)),
):
    """仅更新标签，不触发重新解析/嵌入——改标签不应让文档重新入库。"""
    doc = await db.get(Document, doc_id)
    if not doc or doc.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    doc.tags = _normalize_tags(body.tags)
    doc.updated_by = user.id
    doc.updated_at = now_utc()
    audit.log(
        db,
        "doc.update_tags",
        target_type="document",
        target_id=doc.id,
        target_name=doc.filename,
        detail={"kb_id": str(kb_id), "tag_count": len(doc.tags)},
    )
    await db.commit()
    await db.refresh(doc, attribute_names=["creator", "updater"])
    doc.created_by_name = _name(doc.creator)
    doc.updated_by_name = _name(doc.updater)
    return doc


@router.patch("/{kb_id}/documents/{doc_id}/toggle", response_model=DocumentOut)
async def toggle_document(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_kb_access(DOC_UPDATE)),
):
    doc = await db.get(Document, doc_id)
    if not doc or doc.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    doc.is_active = not doc.is_active
    audit.log(
        db,
        "doc.toggle",
        target_type="document",
        target_id=doc.id,
        target_name=doc.filename,
        detail={"kb_id": str(kb_id), "is_active": doc.is_active},
    )
    await db.commit()
    await db.refresh(doc)
    return doc


@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_kb_access(DOC_DELETE)),
):
    doc = await db.get(Document, doc_id)
    if not doc or doc.kb_id != kb_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    audit.log(db, "doc.delete", target_type="document", target_id=doc.id, target_name=doc.filename,
              detail={"kb_id": str(kb_id)})
    await db.delete(doc)
    await db.commit()
    return {"message": "删除成功"}
