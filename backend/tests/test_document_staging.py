"""Regression coverage for the two-phase upload → 保存入库 lifecycle.

Upload endpoints must only stage a ``draft`` document row (``staged_path``);
ingestion (parse → chunk → embed → searchable) happens exclusively through the
``ingest`` endpoint.  Drafts must never carry processing jobs, and deleting a
draft must remove its staged source file.
"""

from __future__ import annotations

import io
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, UploadFile
from fastapi.datastructures import Headers

from api import document as document_api
from models.db_models import Document


def _user() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        is_superadmin=True,
        display_name="管理员",
        username="admin",
    )


def _upload_file(name: str = "制度.docx") -> UploadFile:
    return UploadFile(
        filename=name,
        file=io.BytesIO(b"staged content"),
        headers=Headers({"content-type": "application/octet-stream"}),
    )


class _FakeAudit:
    def __init__(self) -> None:
        self.events: list[str] = []

    # AuditLogger.log 是同步加入会话（随业务事务一起提交），log_independent 才异步落库。
    def log(self, _db, event: str, **_kwargs) -> None:
        self.events.append(event)

    async def log_independent(self, event: str, **_kwargs) -> None:
        self.events.append(event)


class _FakeDb:
    """Minimal in-memory stand-in for the AsyncSession surface used here."""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.refreshed: list[object] = []
        self.commits = 0
        self.loaded_document: Document | None = None

    async def get(self, _model, _pk):
        return SimpleNamespace(id=_pk)

    async def execute(self, _statement):
        class _Result:
            def scalar_one_or_none(self):
                return self._document

            def __init__(self, document):
                self._document = document

        return _Result(self.loaded_document)

    def add(self, value: object) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        pass

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, value: object) -> None:
        self.refreshed.append(value)

    async def delete(self, value: object) -> None:
        if value in self.added:
            self.added.remove(value)


def _draft_document(*, staged_path: str | None, status: str = "draft") -> Document:
    # 内存实例不会自动应用 mapped_column 的 Python default，显式补全序列化所需字段。
    return Document(
        id=uuid.uuid4(),
        kb_id=uuid.uuid4(),
        filename="制度.docx",
        file_type="docx",
        status=status,
        staged_path=staged_path,
        chunk_count=0,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        processing_revision=1,
        created_by=uuid.uuid4(),
    )


class UploadStagingTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_upload_stages_draft_and_enqueues_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            db = _FakeDb()
            audit = _FakeAudit()
            user = _user()
            enqueue = unittest.mock.Mock()
            passthrough = lambda _document, _user: _document

            with (
                patch("api.document.get_settings", return_value=SimpleNamespace(upload_dir=upload_dir)),
                patch("api.document.enqueue_document_processing_job", enqueue),
                patch.object(document_api, "_document_out", passthrough),
            ):
                result = await document_api.upload_document(
                    uuid.uuid4(), _upload_file(), None, db, audit, user
                )

            enqueue.assert_called_once()
            self.assertEqual(enqueue.call_args.kwargs["job_type"], "prepare")
            self.assertEqual(
                Path(enqueue.call_args.kwargs["source_path"]).parent.resolve(),
                Path(upload_dir).resolve(),
            )
            self.assertEqual(result.status, "draft")
            self.assertEqual(db.commits, 1)
            self.assertEqual(audit.events, ["doc.upload"])
            added = [item for item in db.added if isinstance(item, Document)]
            self.assertEqual(len(added), 1)
            self.assertTrue(added[0].staged_path)
            self.assertFalse(added[0].is_active)  # 草稿默认未启用
            self.assertTrue(Path(added[0].staged_path).is_file())
            self.assertEqual(Path(added[0].staged_path).read_bytes(), b"staged content")

    async def test_image_upload_stages_draft_and_enqueues_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            db = _FakeDb()
            audit = _FakeAudit()
            user = _user()
            enqueue = unittest.mock.Mock()
            passthrough = lambda _document, _user: _document

            with (
                patch("api.document.get_settings", return_value=SimpleNamespace(upload_dir=upload_dir)),
                patch("api.document.enqueue_document_processing_job", enqueue),
                patch.object(document_api, "_document_out", passthrough),
            ):
                result = await document_api.upload_image_document(
                    uuid.uuid4(), _upload_file("截图.png"), None, db, audit, user
                )

            enqueue.assert_called_once()
            self.assertEqual(enqueue.call_args.kwargs["job_type"], "prepare")
            self.assertEqual(result.status, "draft")
            self.assertEqual(audit.events, ["doc.upload_image"])
            added = [item for item in db.added if isinstance(item, Document)]
            self.assertEqual(len(added), 1)
            self.assertTrue(added[0].staged_path)
            self.assertFalse(added[0].is_active)
            self.assertTrue(added[0].image_url)


class IngestEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_ingest_enqueues_file_job_and_clears_staged_path(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            staged = Path(upload_dir) / "source.docx"
            staged.write_bytes(b"content")
            doc = _draft_document(staged_path=str(staged))
            db = _FakeDb()
            db.loaded_document = doc
            audit = _FakeAudit()
            enqueue = unittest.mock.Mock(return_value=SimpleNamespace(id=uuid.uuid4()))
            user = _user()
            doc.created_by = user.id

            with patch("api.document.enqueue_document_processing_job", enqueue):
                result = await document_api.ingest_document(
                    kb_id=doc.kb_id, doc_id=doc.id, db=db, audit=audit, user=user
                )

            enqueue.assert_called_once()
            call = enqueue.call_args.kwargs
            self.assertEqual(call["job_type"], "file")
            self.assertEqual(call["source_path"], str(staged))
            self.assertEqual(call["original_name"], doc.filename)
            self.assertEqual(result.status, "processing")
            self.assertTrue(doc.is_active)  # 入库即启用
            self.assertEqual(doc.staged_path, None)
            self.assertEqual(doc.processing_revision, 2)
            self.assertEqual(audit.events, ["doc.ingest"])

    async def test_ingest_file_with_edited_content_uses_text_job_and_discards_file(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            staged = Path(upload_dir) / "source.docx"
            staged.write_bytes(b"content")
            doc = _draft_document(staged_path=str(staged))
            doc.raw_content = "# 原内容\n\n第一段"
            db = _FakeDb()
            db.loaded_document = doc
            enqueue = unittest.mock.Mock(return_value=SimpleNamespace(id=uuid.uuid4()))
            user = _user()
            doc.created_by = user.id

            with patch("api.document.enqueue_document_processing_job", enqueue):
                await document_api.ingest_document(
                    kb_id=doc.kb_id,
                    doc_id=doc.id,
                    body=document_api.IngestDocumentIn(
                        title="制度（修订）",
                        content="# 修订后内容\n\n第二段",
                        source_url="https://example.com",
                        tags=["制度", "差旅"],
                    ),
                    db=db,
                    audit=_FakeAudit(),
                    user=user,
                )

            call = enqueue.call_args.kwargs
            self.assertEqual(call["job_type"], "text")
            self.assertIsNone(call["source_path"])
            self.assertEqual(call["original_name"], "制度（修订）")
            self.assertFalse(staged.exists())
            self.assertEqual(doc.filename, "制度（修订）")
            self.assertEqual(doc.raw_content, "# 修订后内容\n\n第二段")
            self.assertEqual(doc.source_url, "https://example.com")
            self.assertEqual(doc.tags, ["制度", "差旅"])

    async def test_ingest_file_with_unchanged_content_keeps_file_job(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            staged = Path(upload_dir) / "source.docx"
            staged.write_bytes(b"content")
            doc = _draft_document(staged_path=str(staged))
            doc.raw_content = "# 原内容\n\n第一段"
            db = _FakeDb()
            db.loaded_document = doc
            enqueue = unittest.mock.Mock(return_value=SimpleNamespace(id=uuid.uuid4()))
            user = _user()
            doc.created_by = user.id

            with patch("api.document.enqueue_document_processing_job", enqueue):
                await document_api.ingest_document(
                    kb_id=doc.kb_id,
                    doc_id=doc.id,
                    body=document_api.IngestDocumentIn(title="制度", content="# 原内容\n\n第一段"),
                    db=db,
                    audit=_FakeAudit(),
                    user=user,
                )

            call = enqueue.call_args.kwargs
            self.assertEqual(call["job_type"], "file")
            self.assertEqual(call["source_path"], str(staged))
            self.assertTrue(staged.exists())

    async def test_ingest_file_body_without_content_keeps_file_job(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            staged = Path(upload_dir) / "source.docx"
            staged.write_bytes(b"content")
            doc = _draft_document(staged_path=str(staged))
            db = _FakeDb()
            db.loaded_document = doc
            enqueue = unittest.mock.Mock(return_value=SimpleNamespace(id=uuid.uuid4()))
            user = _user()
            doc.created_by = user.id

            with patch("api.document.enqueue_document_processing_job", enqueue):
                await document_api.ingest_document(
                    kb_id=doc.kb_id,
                    doc_id=doc.id,
                    body=document_api.IngestDocumentIn(title="新标题"),
                    db=db,
                    audit=_FakeAudit(),
                    user=user,
                )

            self.assertEqual(enqueue.call_args.kwargs["job_type"], "file")
            self.assertTrue(staged.exists())

    async def test_ingest_image_with_edited_content_keeps_staged_image(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            staged = Path(upload_dir) / "source.png"
            staged.write_bytes(b"image")
            doc = _draft_document(staged_path=str(staged))
            doc.file_type = "png"
            doc.raw_content = "# 识别结果"
            db = _FakeDb()
            db.loaded_document = doc
            enqueue = unittest.mock.Mock(return_value=SimpleNamespace(id=uuid.uuid4()))
            user = _user()
            doc.created_by = user.id

            with patch("api.document.enqueue_document_processing_job", enqueue):
                await document_api.ingest_document(
                    kb_id=doc.kb_id,
                    doc_id=doc.id,
                    body=document_api.IngestDocumentIn(content="# 校对后的内容"),
                    db=db,
                    audit=_FakeAudit(),
                    user=user,
                )

            call = enqueue.call_args.kwargs
            self.assertEqual(call["job_type"], "text")
            self.assertEqual(doc.raw_content, "# 校对后的内容")
            self.assertTrue(staged.exists())


    async def test_ingest_image_doc_uses_text_job_from_prepared_content(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            staged = Path(upload_dir) / "source.png"
            staged.write_bytes(b"image")
            doc = _draft_document(staged_path=str(staged))
            doc.file_type = "png"
            doc.raw_content = "# 识别结果\n\n表格内容"
            db = _FakeDb()
            db.loaded_document = doc
            enqueue = unittest.mock.Mock(return_value=SimpleNamespace(id=uuid.uuid4()))
            user = _user()
            doc.created_by = user.id

            with patch("api.document.enqueue_document_processing_job", enqueue):
                await document_api.ingest_document(kb_id=doc.kb_id, doc_id=doc.id, db=db, audit=_FakeAudit(), user=user)

            self.assertEqual(enqueue.call_args.kwargs["job_type"], "text")
            self.assertIsNone(enqueue.call_args.kwargs["source_path"])

    async def test_ingest_image_rejects_unprepared_draft(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            staged = Path(upload_dir) / "source.png"
            staged.write_bytes(b"image")
            doc = _draft_document(staged_path=str(staged))
            doc.file_type = "png"
            doc.raw_content = None
            db = _FakeDb()
            db.loaded_document = doc
            user = _user()
            doc.created_by = user.id

            with self.assertRaisesRegex(HTTPException, "仍在准备中"):
                await document_api.ingest_document(kb_id=doc.kb_id, doc_id=doc.id, db=db, audit=_FakeAudit(), user=user)

    async def test_ingest_rejects_non_draft_documents(self) -> None:
        doc = _draft_document(staged_path="/tmp/x.docx", status="ready")
        db = _FakeDb()
        db.loaded_document = doc
        user = _user()
        doc.created_by = user.id

        with self.assertRaisesRegex(HTTPException, "仅草稿状态"):
            await document_api.ingest_document(kb_id=doc.kb_id, doc_id=doc.id, db=db, audit=_FakeAudit(), user=user)

    async def test_ingest_file_rejects_draft_without_staged_file(self) -> None:
        doc = _draft_document(staged_path=None)
        doc.raw_content = None
        db = _FakeDb()
        db.loaded_document = doc
        user = _user()
        doc.created_by = user.id

        with self.assertRaisesRegex(HTTPException, "缺少暂存源文件"):
            await document_api.ingest_document(kb_id=doc.kb_id, doc_id=doc.id, db=db, audit=_FakeAudit(), user=user)

    async def test_update_rejects_draft_to_keep_state_machine_clean(self) -> None:
        doc = _draft_document(staged_path="/tmp/x.docx")
        db = _FakeDb()
        db.loaded_document = doc
        user = _user()
        doc.created_by = user.id

        with self.assertRaisesRegex(HTTPException, "草稿尚未入库"):
            await document_api.update_text_document(
                doc.kb_id,
                doc.id,
                document_api.TextDocumentIn(title="新标题", content="新内容"),
                db,
                _FakeAudit(),
                user,
            )


class DocumentListVisibilityTests(unittest.IsolatedAsyncioTestCase):
    def _capture_db(self, is_superadmin: bool) -> tuple[dict, SimpleNamespace, "_CaptureDb"]:
        captured: dict[str, str] = {}
        user = SimpleNamespace(
            id=uuid.uuid4(),
            is_superadmin=is_superadmin,
            display_name="用户",
            username="user",
        )

        class _Result:
            def scalars(self):
                return self

            def all(self):
                return []

        class _CaptureDb:
            async def execute(self, statement):
                from sqlalchemy.dialects import postgresql

                captured["sql"] = str(statement.compile(
                    dialect=postgresql.dialect(),
                    compile_kwargs={"literal_binds": True},
                ))
                return _Result()

        return captured, user, _CaptureDb()

    async def test_list_hides_drafts_owned_by_others(self) -> None:
        captured, user, db = self._capture_db(is_superadmin=False)
        await document_api.list_documents(
            kb_id=uuid.uuid4(), page=1, page_size=20, db=db, user=user
        )
        self.assertIn("documents.status != 'draft'", captured["sql"])
        self.assertIn("documents.created_by = '", captured["sql"])

    async def test_list_superadmin_sees_everything(self) -> None:
        captured, user, db = self._capture_db(is_superadmin=True)
        await document_api.list_documents(
            kb_id=uuid.uuid4(), page=1, page_size=20, db=db, user=user
        )
        self.assertNotIn("status != 'draft'", captured["sql"])


class DraftDeleteTests(unittest.IsolatedAsyncioTestCase):
    async def test_deleting_draft_removes_staged_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            staged = Path(upload_dir) / "source.pdf"
            staged.write_bytes(b"pdf")
            doc = _draft_document(staged_path=str(staged))
            db = _FakeDb()
            db.loaded_document = doc
            user = _user()
            doc.created_by = user.id

            await document_api.delete_document(doc.kb_id, doc.id, db, _FakeAudit(), user)

            self.assertFalse(staged.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
