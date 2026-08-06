"""Document response capabilities and actor-name projection tests."""

import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import HTTPException

from api.document import _document_out, _require_document_action
from core.permissions import DOC_DELETE, DOC_READ, DOC_UPDATE


class DocumentResponseTests(unittest.TestCase):
    def test_response_uses_policy_and_current_mutation_actor(self) -> None:
        owner_id = uuid.uuid4()
        user = SimpleNamespace(
            id=owner_id,
            username="owner",
            display_name="当前创建者",
            is_superadmin=False,
            permissions=[DOC_READ, DOC_UPDATE, DOC_DELETE],
        )
        stale_updater = SimpleNamespace(username="old", display_name="旧修改人")
        document = SimpleNamespace(
            id=uuid.uuid4(),
            kb_id=uuid.uuid4(),
            filename="权限测试.md",
            file_type="md",
            raw_content="content",
            source_url=None,
            image_url=None,
            chunk_count=1,
            status="ready",
            is_active=True,
            tags=[],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by=owner_id,
            updated_by=owner_id,
            updater=stale_updater,
        )

        response = _document_out(document, user)

        self.assertEqual(response.created_by_name, "当前创建者")
        self.assertEqual(response.updated_by_name, "当前创建者")
        self.assertEqual(
            response.permissions.model_dump(),
            {"read": True, "update": True, "delete": True},
        )


class DocumentDenialAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_owner_denial_is_audited_before_returning_403(self) -> None:
        class AuditSpy:
            def __init__(self):
                self.event = None

            async def log_independent(self, action, **payload):
                self.event = (action, payload)
                return True

        user = SimpleNamespace(
            id=uuid.uuid4(),
            username="editor",
            is_superadmin=False,
            permissions=[DOC_READ, DOC_UPDATE],
        )
        document = SimpleNamespace(
            id=uuid.uuid4(),
            kb_id=uuid.uuid4(),
            filename="其他人的文章.md",
            created_by=uuid.uuid4(),
        )
        audit = AuditSpy()

        with self.assertRaises(HTTPException) as denied:
            await _require_document_action(user, document, "update", audit)

        self.assertEqual(denied.exception.status_code, 403)
        self.assertEqual(audit.event[0], "doc.access_denied")
        self.assertEqual(audit.event[1]["detail"]["requested_action"], "update")
        self.assertEqual(audit.event[1]["detail"]["reason"], "not_document_owner")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
