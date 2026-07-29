"""Readiness health check coverage used by Docker Compose."""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from main import health


class _ConnectionContext:
    def __init__(self, connection=None, error=None):
        self.connection = connection
        self.error = error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self.connection

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False


class _Engine:
    def __init__(self, context):
        self.context = context

    def connect(self):
        return self.context


class HealthCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_checks_database_connection(self) -> None:
        connection = AsyncMock()
        fake_engine = _Engine(_ConnectionContext(connection=connection))

        with patch("main.engine", fake_engine):
            result = await health()

        connection.execute.assert_awaited_once()
        self.assertEqual(result, {"status": "ok", "database": "ok"})

    async def test_health_returns_503_when_database_is_unavailable(self) -> None:
        fake_engine = _Engine(_ConnectionContext(error=RuntimeError("offline")))

        with patch("main.engine", fake_engine):
            with self.assertRaises(HTTPException) as raised:
                await health()

        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
