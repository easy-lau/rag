"""Regression coverage for durable document ingestion boundaries."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from api import document as document_api
from config import Settings
from core.document_jobs import (
    ClaimedDocumentJob,
    _source_path,
    enqueue_document_processing_job,
    process_one_document_job,
)


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, value: object) -> None:
        self.added.append(value)


class DocumentJobEnqueueTests(unittest.TestCase):
    def test_enqueue_binds_job_to_exact_document_revision(self) -> None:
        session = _FakeSession()
        doc = SimpleNamespace(
            id=uuid.uuid4(),
            kb_id=uuid.uuid4(),
            processing_revision=4,
        )

        job = enqueue_document_processing_job(
            session,  # type: ignore[arg-type]
            document=doc,  # type: ignore[arg-type]
            job_type="file",
            source_path="uploads/source.pdf",
            original_name="制度.pdf",
        )

        self.assertEqual(session.added, [job])
        self.assertEqual(job.document_id, doc.id)
        self.assertEqual(job.kb_id, doc.kb_id)
        self.assertEqual(job.document_revision, 4)
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.payload["source_path"], "uploads/source.pdf")

    def test_enqueue_rejects_missing_revision_or_unknown_type(self) -> None:
        session = _FakeSession()
        doc = SimpleNamespace(
            id=uuid.uuid4(), kb_id=uuid.uuid4(), processing_revision=0
        )
        with self.assertRaisesRegex(ValueError, "revision"):
            enqueue_document_processing_job(
                session, document=doc, job_type="text"  # type: ignore[arg-type]
            )
        doc.processing_revision = 1
        with self.assertRaisesRegex(ValueError, "unsupported"):
            enqueue_document_processing_job(
                session, document=doc, job_type="other"  # type: ignore[arg-type]
            )

    def test_source_path_cannot_escape_upload_root(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            source = Path(upload_dir) / "input.pdf"
            source.write_bytes(b"test")
            settings = SimpleNamespace(upload_dir=upload_dir)
            with patch("core.document_jobs.get_settings", return_value=settings):
                self.assertEqual(
                    _source_path({"source_path": str(source)}), source.resolve()
                )
                with self.assertRaisesRegex(ValueError, "不在上传目录"):
                    _source_path({"source_path": "/tmp/other.pdf"}, require_exists=False)


class DocumentWorkerRuntimeTests(unittest.TestCase):
    def test_embedded_worker_defaults_to_development_only(self) -> None:
        self.assertTrue(
            Settings(app_env="development").document_job_runs_embedded_worker
        )
        self.assertFalse(
            Settings(app_env="production").document_job_runs_embedded_worker
        )
        self.assertTrue(
            Settings(
                app_env="production", document_job_embedded_worker=True
            ).document_job_runs_embedded_worker
        )


class DocumentJobWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_file_completion_discards_its_temporary_source(self) -> None:
        job = ClaimedDocumentJob(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            kb_id=uuid.uuid4(),
            document_revision=1,
            job_type="file",
            payload={"source_path": "uploads/input.pdf"},
            attempt_count=1,
        )
        chunks = [{"content": "正文", "embedding": [0.1], "metadata": {}}]
        with (
            patch("core.document_jobs._claim_next_job", new=AsyncMock(return_value=job)),
            patch("core.document_jobs._load_current_document", new=AsyncMock(return_value=SimpleNamespace())),
            patch("core.document_jobs._materialize_chunks", new=AsyncMock(return_value=(chunks, None))),
            patch("core.document_jobs._complete_job", new=AsyncMock(return_value=True)) as complete,
            patch("core.document_jobs._discard_source_file", new=AsyncMock()) as discard,
        ):
            self.assertTrue(await process_one_document_job())

        complete.assert_awaited_once_with(job, chunks, None)
        discard.assert_awaited_once_with({"source_path": "uploads/input.pdf"})

    async def test_terminal_image_completion_keeps_reviewable_original_image(self) -> None:
        job = ClaimedDocumentJob(
            id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            kb_id=uuid.uuid4(),
            document_revision=1,
            job_type="image",
            payload={"source_path": "uploads/images/input.png"},
            attempt_count=1,
        )
        chunks = [{"content": "识别文本", "embedding": [0.1], "metadata": {}}]
        with (
            patch("core.document_jobs._claim_next_job", new=AsyncMock(return_value=job)),
            patch("core.document_jobs._load_current_document", new=AsyncMock(return_value=SimpleNamespace())),
            patch("core.document_jobs._materialize_chunks", new=AsyncMock(return_value=(chunks, "识别文本"))),
            patch("core.document_jobs._complete_job", new=AsyncMock(return_value=True)),
            patch("core.document_jobs._discard_source_file", new=AsyncMock()) as discard,
        ):
            self.assertTrue(await process_one_document_job())

        discard.assert_not_awaited()


class DocumentRouteBoundaryTests(unittest.TestCase):
    def test_api_has_no_in_process_document_processor(self) -> None:
        source = Path(document_api.__file__).read_text(encoding="utf-8")
        self.assertNotIn("asyncio.create_task", source)
        self.assertNotIn("def _process_document", source)
        self.assertNotIn("def reset_stuck_processing", source)
        self.assertIn("enqueue_document_processing_job", source)
