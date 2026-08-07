import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from config import get_settings
from core.log_retention import cleanup_retained_logs


class LogRetentionCleanupTests(unittest.IsolatedAsyncioTestCase):
    def test_retention_defaults_are_configured(self):
        settings = get_settings()
        self.assertGreaterEqual(settings.operation_log_retention_days, 7)
        self.assertGreaterEqual(settings.intent_route_log_retention_days, 7)
        self.assertGreaterEqual(
            settings.operation_log_retention_days,
            settings.login_log_retention_days,
        )

    async def test_cleanup_deletes_expired_operation_and_route_logs(self):
        session = self._session()
        session.execute.side_effect = [
            SimpleNamespace(rowcount=12),
            SimpleNamespace(rowcount=3),
        ]
        with patch("core.log_retention.AsyncSessionLocal", return_value=session):
            deleted_operations, deleted_routes = await cleanup_retained_logs()

        self.assertEqual((deleted_operations, deleted_routes), (12, 3))
        self.assertEqual(session.execute.await_count, 2)
        self.assertEqual(session.commit.await_count, 2)

        settings = get_settings()
        now = datetime.now(timezone.utc)
        statements = [call[0][0] for call in session.execute.await_args_list]
        self.assertIn("delete from operation_logs", str(statements[0].compile()).casefold())
        self.assertIn("delete from intent_route_logs", str(statements[1].compile()).casefold())
        for statement, days in zip(
            statements,
            (settings.operation_log_retention_days, settings.intent_route_log_retention_days),
        ):
            params = dict(statement.compile().params)
            cutoff = next(
                value
                for value in params.values()
                if isinstance(value, datetime)
            )
            self.assertAlmostEqual(
                cutoff.timestamp(),
                (now - timedelta(days=days)).timestamp(),
                delta=60,
            )

    async def test_cleanup_with_no_expired_rows(self):
        session = self._session()
        session.execute.side_effect = [
            SimpleNamespace(rowcount=0),
            SimpleNamespace(rowcount=0),
        ]
        with patch("core.log_retention.AsyncSessionLocal", return_value=session):
            deleted_operations, deleted_routes = await cleanup_retained_logs()
        self.assertEqual((deleted_operations, deleted_routes), (0, 0))

    @staticmethod
    def _session() -> AsyncMock:
        session = AsyncMock()
        session.__aenter__.return_value = session
        session.__aexit__.return_value = False
        return session


if __name__ == "__main__":
    unittest.main()
