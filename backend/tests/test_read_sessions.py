"""Regression tests for request/read transaction ownership boundaries."""

from __future__ import annotations

import unittest
from contextlib import asynccontextmanager

from core.read_sessions import isolated_read_session


class _RequestSession:
    def __init__(self) -> None:
        self.rollback_count = 0

    async def rollback(self) -> None:
        self.rollback_count += 1


class _ReadSession:
    def __init__(self) -> None:
        self.rollback_count = 0

    async def rollback(self) -> None:
        self.rollback_count += 1


class IsolatedReadSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_owned_read_failure_rolls_back_only_owned_session(self) -> None:
        request_session = _RequestSession()
        owned_sessions: list[_ReadSession] = []

        @asynccontextmanager
        async def factory():
            session = _ReadSession()
            owned_sessions.append(session)
            yield session

        with self.assertRaisesRegex(RuntimeError, "registry table missing"):
            async with isolated_read_session(
                request_db=request_session,
                session_factory=factory,
            ) as read_db:
                self.assertIs(read_db, owned_sessions[0])
                raise RuntimeError("registry table missing")

        self.assertEqual(request_session.rollback_count, 0)
        self.assertEqual(len(owned_sessions), 1)
        self.assertEqual(owned_sessions[0].rollback_count, 1)

    async def test_missing_factory_preserves_serial_compatibility_fallback(self) -> None:
        request_session = _RequestSession()

        async with isolated_read_session(
            request_db=request_session,
            session_factory=None,
        ) as read_db:
            self.assertIs(read_db, request_session)

        # The borrowed request transaction belongs to its caller.  The shared
        # boundary must never roll it back merely because no factory was passed.
        self.assertEqual(request_session.rollback_count, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
