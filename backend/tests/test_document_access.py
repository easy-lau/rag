"""Document ownership policy tests independent from HTTP and database state."""

import unittest
import uuid
from types import SimpleNamespace

from core.document_access import (
    DocumentAccessDenied,
    evaluate_document_permissions,
    require_document_action,
)
from core.permissions import DOC_DELETE, DOC_READ, DOC_UPDATE


def _user(*permissions: str, is_superadmin: bool = False):
    return SimpleNamespace(
        id=uuid.uuid4(),
        is_superadmin=is_superadmin,
        permissions=list(permissions),
    )


def _document(owner_id, *, status: str = "ready"):
    return SimpleNamespace(created_by=owner_id, status=status)


class DocumentAccessTests(unittest.TestCase):
    def test_owner_still_needs_each_function_capability(self) -> None:
        user = _user(DOC_READ, DOC_UPDATE)
        permissions = evaluate_document_permissions(user, _document(user.id))

        self.assertTrue(permissions.read)
        self.assertTrue(permissions.update)
        self.assertFalse(permissions.delete)

        with self.assertRaises(DocumentAccessDenied) as denied:
            require_document_action(user, _document(user.id), "delete")
        self.assertEqual(denied.exception.reason, "missing_capability")

    def test_non_owner_is_read_only_even_with_write_capabilities(self) -> None:
        user = _user(DOC_READ, DOC_UPDATE, DOC_DELETE)
        permissions = evaluate_document_permissions(user, _document(uuid.uuid4()))

        self.assertTrue(permissions.read)
        self.assertFalse(permissions.update)
        self.assertFalse(permissions.delete)

        for action in ("update", "delete"):
            with self.subTest(action=action):
                with self.assertRaises(DocumentAccessDenied) as denied:
                    require_document_action(user, _document(uuid.uuid4()), action)
                self.assertEqual(denied.exception.reason, "not_document_owner")

    def test_superadmin_can_manage_every_document_without_role_capabilities(self) -> None:
        user = _user(is_superadmin=True)
        for owner_id in (uuid.uuid4(), None):
            with self.subTest(owner_id=owner_id):
                permissions = evaluate_document_permissions(user, _document(owner_id))
                self.assertEqual(
                    permissions.as_dict(),
                    {"read": True, "update": True, "delete": True},
                )
                require_document_action(user, _document(owner_id), "update")
                require_document_action(user, _document(owner_id), "delete")

    def test_draft_is_readable_only_by_its_owner(self) -> None:
        owner = _user(DOC_READ)
        other = _user(DOC_READ, DOC_UPDATE, DOC_DELETE)
        draft = _document(owner.id, status="draft")

        owner_permissions = evaluate_document_permissions(owner, draft)
        self.assertTrue(owner_permissions.read)

        other_permissions = evaluate_document_permissions(other, draft)
        self.assertEqual(
            other_permissions.as_dict(),
            {"read": False, "update": False, "delete": False},
        )
        with self.assertRaises(DocumentAccessDenied):
            require_document_action(other, draft, "read")

    def test_superadmin_can_read_any_draft(self) -> None:
        user = _user(is_superadmin=True)
        draft = _document(uuid.uuid4(), status="draft")

        permissions = evaluate_document_permissions(user, draft)
        self.assertTrue(permissions.read)
        require_document_action(user, draft, "read")

    def test_ready_document_remains_readable_to_scoped_users(self) -> None:
        user = _user(DOC_READ)
        ready = _document(uuid.uuid4(), status="ready")

        self.assertTrue(evaluate_document_permissions(user, ready).read)


    def test_ownerless_document_is_not_writable_by_ordinary_user(self) -> None:
        user = _user(DOC_READ, DOC_UPDATE, DOC_DELETE)
        document = _document(None)

        permissions = evaluate_document_permissions(user, document)
        self.assertTrue(permissions.read)
        self.assertFalse(permissions.update)
        self.assertFalse(permissions.delete)

        with self.assertRaises(DocumentAccessDenied) as denied:
            require_document_action(user, document, "update")
        self.assertEqual(denied.exception.reason, "document_owner_missing")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
