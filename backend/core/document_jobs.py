"""Durable document-ingestion jobs and the worker that executes them.

The API only creates a document revision and its job in one database
transaction.  Parsing, vision and embeddings run in a separate worker process.
Every write is fenced by ``Document.processing_revision`` so a late worker can
never overwrite a newer user edit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from core.document_parser import (
    docx_to_markdown,
    excel_to_markdown,
    parse_file,
    parse_markdown_content,
)
from core.document_content import normalize_document_markdown
from core.embeddings import embed_batch
from core.vision import image_to_markdown
from database import AsyncSessionLocal
from models.db_models import (
    Document,
    DocumentChunk,
    DocumentProcessingJob,
    now_utc,
)


logger = logging.getLogger(__name__)
DocumentJobType = Literal["file", "text", "image", "prepare"]
_JOB_TYPES = frozenset({"file", "text", "image", "prepare"})


@dataclass(frozen=True)
class ClaimedDocumentJob:
    id: uuid.UUID
    document_id: uuid.UUID
    kb_id: uuid.UUID
    document_revision: int
    job_type: DocumentJobType
    payload: dict
    attempt_count: int


def _safe_payload(*, source_path: str | None, original_name: str | None) -> dict:
    payload: dict[str, str] = {}
    if source_path:
        payload["source_path"] = str(source_path)
    if original_name:
        payload["original_name"] = str(original_name)[:255]
    return payload


def enqueue_document_processing_job(
    db: AsyncSession,
    *,
    document: Document,
    job_type: DocumentJobType,
    source_path: str | None = None,
    original_name: str | None = None,
) -> DocumentProcessingJob:
    """Attach exactly one durable job to the current document revision.

    The caller owns the surrounding transaction; document state and job row
    therefore become visible atomically.  A replacement revision makes old
    jobs harmless even when they are already running.
    """

    if job_type not in _JOB_TYPES:
        raise ValueError("unsupported document processing job type")
    revision = int(document.processing_revision or 0)
    if revision < 1:
        raise ValueError("document processing revision must be positive")
    job = DocumentProcessingJob(
        document_id=document.id,
        kb_id=document.kb_id,
        document_revision=revision,
        job_type=job_type,
        payload=_safe_payload(source_path=source_path, original_name=original_name),
        status="queued",
        available_at=now_utc(),
    )
    db.add(job)
    return job


def _job_limits() -> tuple[int, int, float]:
    settings = get_settings()
    attempts = max(1, int(getattr(settings, "document_job_max_attempts", 3)))
    lease = max(30, int(getattr(settings, "document_job_lease_seconds", 900)))
    poll = max(0.1, float(getattr(settings, "document_job_poll_seconds", 1.0)))
    return attempts, lease, poll


async def _claim_next_job() -> ClaimedDocumentJob | None:
    max_attempts, lease_seconds, _ = _job_limits()
    now = now_utc()
    expired_payload: dict | None = None
    async with AsyncSessionLocal() as db:
        statement = (
            select(DocumentProcessingJob)
            .where(
                or_(
                    (DocumentProcessingJob.status == "queued")
                    & (DocumentProcessingJob.available_at <= now),
                    (DocumentProcessingJob.status == "running")
                    & (DocumentProcessingJob.lease_expires_at.is_not(None))
                    & (DocumentProcessingJob.lease_expires_at < now),
                )
            )
            .order_by(DocumentProcessingJob.available_at, DocumentProcessingJob.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        job = (await db.execute(statement)).scalar_one_or_none()
        if job is None:
            return None
        if job.attempt_count >= max_attempts:
            job.status = "failed"
            job.lease_expires_at = None
            job.last_error = "超过最大重试次数，未再次执行"
            doc = await db.get(Document, job.document_id, with_for_update=True)
            if doc is not None and doc.processing_revision == job.document_revision:
                doc.status = "failed"
            await db.commit()
            if job.job_type == "file":
                expired_payload = dict(job.payload or {})
        else:
            job.status = "running"
            job.attempt_count += 1
            job.lease_expires_at = now + timedelta(seconds=lease_seconds)
            job.last_error = None
            await db.commit()
            return ClaimedDocumentJob(
                id=job.id,
                document_id=job.document_id,
                kb_id=job.kb_id,
                document_revision=job.document_revision,
                job_type=job.job_type,  # type: ignore[arg-type]
                payload=dict(job.payload or {}),
                attempt_count=job.attempt_count,
            )
    if expired_payload is not None:
        await _discard_source_file(expired_payload)
    return None


def _source_path(payload: dict, *, require_exists: bool = True) -> Path:
    raw = str(payload.get("source_path") or "").strip()
    if not raw:
        raise ValueError("文件任务缺少 source_path")
    root = Path(get_settings().upload_dir).resolve()
    path = Path(raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("文件任务 source_path 不在上传目录内") from exc
    if require_exists and not path.is_file():
        raise FileNotFoundError("待处理源文件不存在")
    return path


async def _discard_source_file(payload: dict) -> None:
    """Delete only an ingestion-owned temporary source after a terminal state."""

    if not payload.get("source_path"):
        return
    try:
        path = _source_path(payload, require_exists=False)
        await asyncio.to_thread(path.unlink, missing_ok=True)
    except Exception as exc:  # Never turn an already committed job result into a retry.
        logger.warning("[文档任务] 清理临时源文件失败 error=%s", type(exc).__name__)


async def _load_current_document(job: ClaimedDocumentJob) -> Document | None:
    async with AsyncSessionLocal() as db:
        doc = await db.get(Document, job.document_id)
        if doc is None or doc.processing_revision != job.document_revision:
            return None
        return doc


async def _materialize_chunks(job: ClaimedDocumentJob, doc: Document) -> tuple[list[dict], str | None]:
    if job.job_type == "file":
        path = _source_path(job.payload)
        source_name = str(job.payload.get("original_name") or doc.filename)
        chunks = await asyncio.to_thread(parse_file, str(path), source_name=source_name)
        extension = path.suffix.lower()
        raw_content = (
            await asyncio.to_thread(excel_to_markdown, str(path))
            if extension in {".xlsx", ".xls"}
            else (
                await asyncio.to_thread(docx_to_markdown, str(path))
                if extension == ".docx"
                else None
            )
        )
    elif job.job_type == "text":
        chunks = await asyncio.to_thread(
            parse_markdown_content,
            str(doc.raw_content or ""),
            doc.filename,
        )
        raw_content = None
    elif job.job_type == "image":
        markdown = await image_to_markdown(str(_source_path(job.payload)))
        chunks = await asyncio.to_thread(parse_markdown_content, markdown, doc.filename)
        raw_content = markdown
    else:  # pragma: no cover - database constraint and type guard protect this
        raise ValueError("unsupported document processing job type")
    if not chunks:
        raise ValueError("文档解析结果为空")
    embeddings = await embed_batch([str(item["content"]) for item in chunks])
    if len(embeddings) != len(chunks):
        raise RuntimeError("文档分块与向量数量不一致")
    if raw_content is not None:
        raw_content = normalize_document_markdown(raw_content)
    return [
        {**dict(item), "embedding": embeddings[index]}
        for index, item in enumerate(chunks)
    ], raw_content


async def _prepare_draft_content(job: ClaimedDocumentJob, doc: Document) -> str:
    """抽取草稿审阅内容（raw_content），不产生分块/向量，也不删除源文件。

    上传接口只入队 prepare 任务：解析完成后文档仍是 draft，供编辑页审阅；
    「保存入库」才触发真正的分块与向量化（file/text 任务）。
    """

    if job.job_type == "image":
        markdown = await image_to_markdown(str(_source_path(job.payload)))
    else:
        path = _source_path(job.payload)
        extension = path.suffix.lower()
        source_name = str(job.payload.get("original_name") or doc.filename)
        if extension in {".xlsx", ".xls"}:
            markdown = await asyncio.to_thread(excel_to_markdown, str(path))
        elif extension == ".docx":
            markdown = await asyncio.to_thread(docx_to_markdown, str(path))
        else:
            chunks = await asyncio.to_thread(parse_file, str(path), source_name=source_name)
            markdown = "\n\n".join(str(item["content"]) for item in chunks)
    if not markdown.strip():
        raise ValueError("文档内容为空")
    return normalize_document_markdown(markdown)


async def _complete_prepare(job: ClaimedDocumentJob, raw_content: str) -> bool:
    """prepare 的终态：写入审阅内容但保持 draft，不产生任何检索分块。"""

    async with AsyncSessionLocal() as db:
        persisted = await db.get(DocumentProcessingJob, job.id, with_for_update=True)
        doc = await db.get(Document, job.document_id, with_for_update=True)
        if persisted is None or doc is None:
            return True
        if doc.processing_revision != job.document_revision:
            persisted.status = "superseded"
            persisted.lease_expires_at = None
            await db.commit()
            return True
        if persisted.status != "running" or persisted.attempt_count != job.attempt_count:
            return False
        doc.raw_content = raw_content
        persisted.status = "completed"
        persisted.completed_at = now_utc()
        persisted.lease_expires_at = None
        persisted.last_error = None
        await db.commit()
        return True


async def _complete_job(
    job: ClaimedDocumentJob,
    chunks_data: list[dict],
    raw_content: str | None,
) -> bool:
    async with AsyncSessionLocal() as db:
        persisted = await db.get(DocumentProcessingJob, job.id, with_for_update=True)
        doc = await db.get(Document, job.document_id, with_for_update=True)
        if persisted is None or doc is None:
            return True
        if doc.processing_revision != job.document_revision:
            persisted.status = "superseded"
            persisted.lease_expires_at = None
            await db.commit()
            return True
        if (
            persisted.status != "running"
            or persisted.attempt_count != job.attempt_count
        ):
            return False
        await db.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc.id))
        db.add_all([
            DocumentChunk(
                doc_id=doc.id,
                kb_id=doc.kb_id,
                content=str(item["content"]),
                embedding=item["embedding"],
                chunk_index=index,
                metadata_=item.get("metadata"),
            )
            for index, item in enumerate(chunks_data)
        ])
        doc.status = "ready"
        doc.chunk_count = len(chunks_data)
        if raw_content is not None:
            doc.raw_content = raw_content
        persisted.status = "completed"
        persisted.completed_at = now_utc()
        persisted.lease_expires_at = None
        persisted.last_error = None
        await db.commit()
        return True


async def _fail_job(job: ClaimedDocumentJob, exc: Exception) -> bool:
    max_attempts, _, _ = _job_limits()
    now = now_utc()
    message = f"{type(exc).__name__}: {exc}"[:2000]
    async with AsyncSessionLocal() as db:
        persisted = await db.get(DocumentProcessingJob, job.id, with_for_update=True)
        doc = await db.get(Document, job.document_id, with_for_update=True)
        if (
            persisted is None
            or persisted.status != "running"
            or persisted.attempt_count != job.attempt_count
        ):
            return False
        if doc is None or doc.processing_revision != job.document_revision:
            persisted.status = "superseded"
        elif persisted.attempt_count >= max_attempts:
            persisted.status = "failed"
            doc.status = "failed"
        else:
            persisted.status = "queued"
            persisted.available_at = now + timedelta(seconds=min(60, 2 ** persisted.attempt_count))
        persisted.lease_expires_at = None
        persisted.last_error = message
        await db.commit()
    logger.warning(
        "[文档任务] 失败 job=%s doc=%s attempt=%s error=%s",
        job.id, job.document_id, job.attempt_count, message,
    )
    return persisted.status in {"failed", "superseded"}


async def _renew_job_lease(job: ClaimedDocumentJob) -> bool:
    """Extend only the lease that belongs to this exact claimed attempt."""

    _, lease_seconds, _ = _job_limits()
    async with AsyncSessionLocal() as db:
        persisted = await db.get(DocumentProcessingJob, job.id, with_for_update=True)
        if (
            persisted is None
            or persisted.status != "running"
            or persisted.attempt_count != job.attempt_count
        ):
            return False
        persisted.lease_expires_at = now_utc() + timedelta(seconds=lease_seconds)
        await db.commit()
    return True


async def _maintain_job_lease(job: ClaimedDocumentJob, stop: asyncio.Event) -> None:
    _, lease_seconds, _ = _job_limits()
    interval = max(5.0, min(60.0, lease_seconds / 3))
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except TimeoutError:
            if not await _renew_job_lease(job):
                logger.warning("[文档任务] 任务租约不再归当前 worker 所有 job=%s", job.id)
                return


async def process_one_document_job() -> bool:
    job = await _claim_next_job()
    if job is None:
        return False
    lease_stop = asyncio.Event()
    lease_task = asyncio.create_task(
        _maintain_job_lease(job, lease_stop),
        name=f"document-job-lease-{job.id}",
    )
    terminal = False
    try:
        doc = await _load_current_document(job)
        if doc is None:
            terminal = await _complete_job(job, [], None)
            return True
        if job.job_type == "prepare":
            raw_content = await _prepare_draft_content(job, doc)
            terminal = await _complete_prepare(job, raw_content)
            if terminal:
                logger.info(
                    "[文档任务] 草稿内容准备完成 job=%s doc=%s",
                    job.id,
                    job.document_id,
                )
        else:
            chunks, raw_content = await _materialize_chunks(job, doc)
            terminal = await _complete_job(job, chunks, raw_content)
            if terminal:
                logger.info(
                    "[文档任务] 完成 job=%s doc=%s chunks=%s",
                    job.id,
                    job.document_id,
                    len(chunks),
                )
    except Exception as exc:
        terminal = await _fail_job(job, exc)
    finally:
        lease_stop.set()
        await lease_task
        if terminal and job.job_type == "file":
            await _discard_source_file(job.payload)
    return True


async def run_document_worker() -> None:
    """Run one sequential worker; deployment may scale workers via DB leases."""

    _, _, poll_seconds = _job_limits()
    logger.info("[文档任务] worker started poll_seconds=%s", poll_seconds)
    while True:
        try:
            worked = await process_one_document_job()
        except asyncio.CancelledError:
            raise
        except Exception:
            # A transient database/network failure must not permanently stop a
            # local embedded worker or the Supervisor-managed worker process.
            logger.exception("[文档任务] worker loop failed; will retry")
            worked = False
        if not worked:
            await asyncio.sleep(poll_seconds)


def main() -> None:
    asyncio.run(run_document_worker())


if __name__ == "__main__":  # pragma: no cover
    main()
