"""Regression coverage for the pure RBAC policy helpers.

These tests intentionally avoid a database so the most security-sensitive
normalization and delegation rules can be checked in a fresh checkout.
"""

import unittest
import uuid
from types import SimpleNamespace

from fastapi import HTTPException
from pydantic import ValidationError

from api.roles import (
    _kb_scope_is_within_actor_access,
    _validate_superadmin_role_update,
)
from core.permissions import (
    CHAT_USE,
    DOC_CREATE,
    DOC_DELETE,
    DOC_READ,
    DOC_UPDATE,
    KB_CREATE,
    KB_DELETE,
    KB_READ,
    KB_SCOPE_ALL,
    KB_SCOPE_NONE,
    KB_SCOPE_SELECTED,
    KB_UPDATE,
    LEGACY_DOC_WRITE,
    LEGACY_KB_WRITE,
    MENU_CHAT,
    MENU_DOCUMENTS,
    MENU_TERMINOLOGY,
    ROLE_MANAGE,
    SETTINGS_WRITE,
    SUPERADMIN_ROLE_CODE,
    TERMINOLOGY_MANAGE,
    TERMINOLOGY_READ,
    USER_MANAGE,
    capability_catalog_payload,
    derive_menus,
    effective_permissions,
    non_superadmin_delegation_error,
    normalize_assignable_capabilities,
    normalize_scope_mode,
    validate_capability_scope,
)
from models.schemas import RoleUpdate


class PermissionPolicyTests(unittest.TestCase):
    def test_capability_dependencies_are_persisted_as_closure(self) -> None:
        for key in (DOC_CREATE, DOC_UPDATE, DOC_DELETE):
            with self.subTest(key=key):
                self.assertEqual(
                    normalize_assignable_capabilities([key]),
                    {key, DOC_READ, KB_READ},
                )
        for key in (KB_CREATE, KB_UPDATE, KB_DELETE):
            with self.subTest(key=key):
                self.assertEqual(
                    normalize_assignable_capabilities([key]),
                    {key, KB_READ},
                )
        self.assertEqual(
            normalize_assignable_capabilities([TERMINOLOGY_MANAGE]),
            {TERMINOLOGY_MANAGE, TERMINOLOGY_READ, KB_READ},
        )

    def test_crud_actions_are_independently_assignable(self) -> None:
        capabilities = normalize_assignable_capabilities([DOC_UPDATE, KB_UPDATE])
        self.assertIn(DOC_UPDATE, capabilities)
        self.assertIn(KB_UPDATE, capabilities)
        for unrelated in (DOC_CREATE, DOC_DELETE, KB_CREATE, KB_DELETE):
            self.assertNotIn(unrelated, capabilities)

    def test_legacy_write_keys_expand_only_while_loading(self) -> None:
        selected = set(effective_permissions(
            ["kb:write", "doc:write"], scope_mode=KB_SCOPE_SELECTED
        ))
        self.assertNotIn(KB_CREATE, selected)
        self.assertTrue({KB_UPDATE, KB_DELETE}.issubset(selected))
        self.assertTrue({DOC_CREATE, DOC_UPDATE, DOC_DELETE}.issubset(selected))
        self.assertIn(LEGACY_KB_WRITE, selected)
        self.assertIn(LEGACY_DOC_WRITE, selected)

        all_scope = set(effective_permissions([LEGACY_KB_WRITE], scope_mode=KB_SCOPE_ALL))
        self.assertTrue({KB_CREATE, KB_UPDATE, KB_DELETE}.issubset(all_scope))
        self.assertIn(LEGACY_KB_WRITE, all_scope)

        partial = set(effective_permissions(
            [KB_UPDATE, DOC_CREATE, DOC_UPDATE], scope_mode=KB_SCOPE_SELECTED
        ))
        self.assertNotIn(LEGACY_KB_WRITE, partial)
        self.assertNotIn(LEGACY_DOC_WRITE, partial)
        with self.assertRaisesRegex(ValueError, "旧写权限已拆分"):
            normalize_assignable_capabilities([LEGACY_DOC_WRITE])

    def test_menu_is_derived_not_assignable(self) -> None:
        menus = derive_menus([CHAT_USE, DOC_UPDATE, TERMINOLOGY_READ])
        self.assertIn(MENU_CHAT, menus)
        self.assertIn(MENU_DOCUMENTS, menus)
        self.assertIn(MENU_TERMINOLOGY, menus)
        with self.assertRaisesRegex(ValueError, "派生权限"):
            normalize_assignable_capabilities([MENU_CHAT])

    def test_scope_modes_reject_mixed_representation(self) -> None:
        self.assertEqual(normalize_scope_mode(None, []), (KB_SCOPE_NONE, set()))
        self.assertEqual(normalize_scope_mode(None, ["kb-a"]), (KB_SCOPE_SELECTED, {"kb-a"}))
        self.assertEqual(normalize_scope_mode(KB_SCOPE_ALL, []), (KB_SCOPE_ALL, set()))
        with self.assertRaisesRegex(ValueError, "不能同时"):
            normalize_scope_mode(KB_SCOPE_ALL, ["kb-a"])
        with self.assertRaisesRegex(ValueError, "至少"):
            normalize_scope_mode(KB_SCOPE_SELECTED, [])

    def test_none_scope_cannot_carry_kb_data_capabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "知识库数据权限"):
            validate_capability_scope({"kb:read"}, KB_SCOPE_NONE)
        with self.assertRaisesRegex(ValueError, "知识库数据权限"):
            validate_capability_scope({TERMINOLOGY_READ, KB_READ}, KB_SCOPE_NONE)
        # Chat can be used as a general conversation role without a KB scope.
        validate_capability_scope({CHAT_USE}, KB_SCOPE_NONE)
        # A selected scope may be attached before a role is given read/write
        # capabilities (for example a future chat-only RAG role).
        validate_capability_scope({CHAT_USE}, KB_SCOPE_SELECTED)

    def test_kb_create_requires_all_scope_at_role_save_time(self) -> None:
        for scope_mode in (KB_SCOPE_NONE, KB_SCOPE_SELECTED):
            with self.subTest(scope_mode=scope_mode), self.assertRaisesRegex(ValueError, "kb:create"):
                validate_capability_scope({KB_CREATE, KB_READ}, scope_mode)
        validate_capability_scope({KB_CREATE, KB_READ}, KB_SCOPE_ALL)
        validate_capability_scope({KB_UPDATE, KB_READ}, KB_SCOPE_SELECTED)
        validate_capability_scope({KB_DELETE, KB_READ}, KB_SCOPE_SELECTED)

    def test_non_superadmin_cannot_delegate_security_admin_or_all_scope(self) -> None:
        actor = {CHAT_USE, DOC_READ, DOC_CREATE, DOC_UPDATE, DOC_DELETE, KB_READ}
        self.assertIsNone(
            non_superadmin_delegation_error(actor, {CHAT_USE, DOC_READ}, KB_SCOPE_SELECTED)
        )
        self.assertIsNotNone(
            non_superadmin_delegation_error(actor, {CHAT_USE}, KB_SCOPE_ALL)
        )
        self.assertIsNotNone(
            non_superadmin_delegation_error(actor, {USER_MANAGE}, KB_SCOPE_NONE)
        )
        self.assertIsNotNone(
            non_superadmin_delegation_error(actor, {ROLE_MANAGE}, KB_SCOPE_NONE)
        )
        self.assertIsNotNone(
            non_superadmin_delegation_error(
                actor,
                {TERMINOLOGY_MANAGE, TERMINOLOGY_READ, KB_READ},
                KB_SCOPE_SELECTED,
            )
        )

    def test_selected_role_scope_must_be_within_actor_access(self) -> None:
        self.assertTrue(
            _kb_scope_is_within_actor_access(
                KB_SCOPE_SELECTED,
                ["kb-a"],
                ["kb-a", "kb-b"],
            )
        )
        self.assertFalse(
            _kb_scope_is_within_actor_access(
                KB_SCOPE_SELECTED,
                ["kb-a", "kb-outside"],
                ["kb-a", "kb-b"],
            )
        )
        # None is the established sentinel for actors with all-KB scope.
        self.assertTrue(
            _kb_scope_is_within_actor_access(
                KB_SCOPE_SELECTED,
                ["kb-outside"],
                None,
            )
        )

    def test_superadmin_role_identity_and_assignment_lock_are_immutable(self) -> None:
        role = SimpleNamespace(id=uuid.uuid4(), code=SUPERADMIN_ROLE_CODE)
        with self.assertRaisesRegex(HTTPException, "不可分配"):
            _validate_superadmin_role_update(
                role,
                desired_code=SUPERADMIN_ROLE_CODE,
                desired_assignable=True,
                linked_to_superadmin=True,
            )
        with self.assertRaisesRegex(HTTPException, "编码不可修改"):
            _validate_superadmin_role_update(
                role,
                desired_code="renamed_superadmin",
                desired_assignable=False,
                linked_to_superadmin=True,
            )

        normal_role = SimpleNamespace(id=uuid.uuid4(), code=None)
        with self.assertRaisesRegex(HTTPException, "系统保留"):
            _validate_superadmin_role_update(
                normal_role,
                desired_code=SUPERADMIN_ROLE_CODE,
                desired_assignable=False,
                linked_to_superadmin=False,
            )

    def test_catalog_marks_real_delegation_boundaries(self) -> None:
        catalog = capability_catalog_payload()
        definitions = {item["key"]: item for item in catalog["capabilities"]}
        for key in (
            KB_CREATE,
            KB_UPDATE,
            KB_DELETE,
            SETTINGS_WRITE,
            TERMINOLOGY_MANAGE,
            USER_MANAGE,
            ROLE_MANAGE,
        ):
            self.assertTrue(definitions[key]["superadmin_only"])
        all_scope = next(item for item in catalog["scope_modes"] if item["key"] == KB_SCOPE_ALL)
        self.assertTrue(all_scope["superadmin_only"])

        platform_admin = next(item for item in catalog["templates"] if item["code"] == "platform_operator")
        self.assertEqual(platform_admin["name"], "平台管理员")
        self.assertIn("log:read", platform_admin["permissions"])

        knowledge_editor = next(item for item in catalog["templates"] if item["code"] == "knowledge_editor")
        self.assertIn(DOC_CREATE, knowledge_editor["permissions"])
        self.assertIn(DOC_UPDATE, knowledge_editor["permissions"])
        self.assertNotIn(DOC_DELETE, knowledge_editor["permissions"])

    def test_role_update_assignable_is_omittable_but_not_nullable(self) -> None:
        self.assertIsNone(RoleUpdate().is_assignable)
        with self.assertRaises(ValidationError):
            RoleUpdate.model_validate({"is_assignable": None})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
