"""Authorization-adjacent regression coverage for document image delivery."""

import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from api.uploads import _resolve_image_path, get_document_image


class _DocumentResult:
    def __init__(self, document):
        self.document = document

    def scalars(self):
        return self

    def first(self):
        return self.document


class _FakeDb:
    def __init__(self, document):
        self.document = document

    async def execute(self, _statement):
        return _DocumentResult(self.document)


class UploadSecurityTests(unittest.IsolatedAsyncioTestCase):
    def test_image_path_rejects_non_generated_names(self) -> None:
        with self.assertRaisesRegex(HTTPException, "图片不存在"):
            _resolve_image_path("..%2Fsecret.png")

    async def test_document_image_checks_owning_kb_before_returning_file(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            filename = f"{uuid.uuid4()}.png"
            image_dir = Path(upload_dir) / "images"
            image_dir.mkdir()
            (image_dir / filename).write_bytes(b"image")
            document = SimpleNamespace(kb_id=uuid.uuid4(), status="ready", created_by=None)
            user = SimpleNamespace(id=uuid.uuid4(), is_superadmin=False)
            db = _FakeDb(document)
            access_check = AsyncMock()

            with (
                patch("api.uploads.get_settings", return_value=SimpleNamespace(upload_dir=upload_dir)),
                patch("api.uploads.ensure_kb_access", access_check),
            ):
                response = await get_document_image(filename, db, user)

            access_check.assert_awaited_once_with(user, document.kb_id, db)
            self.assertEqual(Path(response.path), (image_dir / filename).resolve())

    async def test_draft_image_rejected_for_non_owner(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            filename = f"{uuid.uuid4()}.png"
            image_dir = Path(upload_dir) / "images"
            image_dir.mkdir()
            (image_dir / filename).write_bytes(b"image")
            document = SimpleNamespace(
                kb_id=uuid.uuid4(),
                status="draft",
                created_by=uuid.uuid4(),  # 非当前用户
            )
            user = SimpleNamespace(id=uuid.uuid4(), is_superadmin=False)
            db = _FakeDb(document)

            with (
                patch("api.uploads.get_settings", return_value=SimpleNamespace(upload_dir=upload_dir)),
                patch("api.uploads.ensure_kb_access", new=AsyncMock()),
            ):
                with self.assertRaisesRegex(HTTPException, "图片不存在"):
                    await get_document_image(filename, db, user)

    async def test_draft_image_allowed_for_owner(self) -> None:
        with tempfile.TemporaryDirectory() as upload_dir:
            filename = f"{uuid.uuid4()}.png"
            image_dir = Path(upload_dir) / "images"
            image_dir.mkdir()
            (image_dir / filename).write_bytes(b"image")
            owner_id = uuid.uuid4()
            document = SimpleNamespace(
                kb_id=uuid.uuid4(),
                status="draft",
                created_by=owner_id,
            )
            user = SimpleNamespace(id=owner_id, is_superadmin=False)
            db = _FakeDb(document)

            with (
                patch("api.uploads.get_settings", return_value=SimpleNamespace(upload_dir=upload_dir)),
                patch("api.uploads.ensure_kb_access", new=AsyncMock()),
            ):
                response = await get_document_image(filename, db, user)

            self.assertEqual(Path(response.path), (image_dir / filename).resolve())
