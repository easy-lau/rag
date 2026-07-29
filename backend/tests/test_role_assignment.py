"""Focused guards for assigning privileged role definitions to users."""

import unittest
import uuid
from types import SimpleNamespace

from fastapi import HTTPException

from api.users import _load_role_for_assignment
from core.permissions import KB_SCOPE_NONE, SUPERADMIN_ROLE_CODE


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _FakeDb:
    def __init__(self, *values):
        self.values = list(values)

    async def execute(self, _statement):
        return _ScalarResult(self.values.pop(0))


def _role(*, code=None, is_assignable=True):
    return SimpleNamespace(
        id=uuid.uuid4(),
        code=code,
        is_assignable=is_assignable,
        scope_mode=KB_SCOPE_NONE,
        permissions=[],
        knowledge_bases=[],
    )


class RoleAssignmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_superadmin_code_is_rejected_even_if_flag_was_corrupted(self) -> None:
        role = _role(code=SUPERADMIN_ROLE_CODE, is_assignable=True)
        current = SimpleNamespace(is_superadmin=True)
        with self.assertRaisesRegex(HTTPException, "不可分配"):
            await _load_role_for_assignment(role.id, current, _FakeDb(role))

    async def test_role_currently_linked_to_superadmin_is_also_rejected(self) -> None:
        role = _role(is_assignable=True)
        current = SimpleNamespace(is_superadmin=True)
        with self.assertRaisesRegex(HTTPException, "不可分配"):
            await _load_role_for_assignment(role.id, current, _FakeDb(role, uuid.uuid4()))

    async def test_normal_assignable_role_remains_available_to_superadmin(self) -> None:
        role = _role(is_assignable=True)
        current = SimpleNamespace(is_superadmin=True)
        assigned = await _load_role_for_assignment(role.id, current, _FakeDb(role, None))
        self.assertIs(assigned, role)
