"""登录来源限流、聚合审计与可信客户端 IP 回归测试。"""

import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from starlette.requests import Request

from api.auth import login
from core.audit import client_ip
from core.login_security import (
    ACCOUNT_SCOPE,
    IP_SCOPE,
    LockedLoginSource,
    PAIR_SCOPE,
    ThrottleResult,
    _advance_bucket,
    bucket_retry_after,
    lock_login_source,
    record_login_attempt,
    register_login_failure,
    throttle_bucket_keys,
)
from models.schemas import LoginRequest


def _settings(**overrides):
    values = {
        "login_pair_failure_threshold": 3,
        "login_pair_window_minutes": 15,
        "login_pair_block_minutes": 15,
        "login_ip_failure_threshold": 5,
        "login_ip_window_minutes": 15,
        "login_ip_block_minutes": 60,
        "login_account_alert_threshold": 4,
        "login_account_alert_window_minutes": 15,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bucket(scope: str = PAIR_SCOPE):
    now = datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        scope=scope,
        bucket_key=f"{scope}-key",
        failure_count=0,
        window_started_at=now,
        blocked_until=None,
        last_failed_at=now,
        created_at=now,
        updated_at=now,
    )


def _user():
    return SimpleNamespace(
        id=uuid.uuid4(),
        username="admin",
        password_hash="hash",
        display_name="管理员",
        is_active=True,
        is_superadmin=True,
        role_id=None,
        role=None,
    )


def _request(peer: str, **headers: str) -> Request:
    raw_headers = [(key.lower().encode(), value.encode()) for key, value in headers.items()]
    return Request({"type": "http", "headers": raw_headers, "client": (peer, 12345)})


def _query_result(value):
    result = Mock()
    result.scalar_one_or_none.return_value = value
    result.scalar_one.return_value = value
    return result


def _rows_result(values):
    result = Mock()
    result.scalars.return_value.all.return_value = values
    return result


def _source(throttle: ThrottleResult | None = None) -> LockedLoginSource:
    return LockedLoginSource(buckets={}, throttle=throttle or ThrottleResult())


class LoginThrottleStateTests(unittest.TestCase):
    def test_bucket_keys_group_username_case_variants_without_merging_ips(self) -> None:
        first = throttle_bucket_keys("admin", "203.0.113.10")
        same_identity = throttle_bucket_keys(" ADMIN ", "203.0.113.10")
        other_ip = throttle_bucket_keys("admin", "203.0.113.11")

        self.assertEqual(first[PAIR_SCOPE], same_identity[PAIR_SCOPE])
        self.assertEqual(first[ACCOUNT_SCOPE], other_ip[ACCOUNT_SCOPE])
        self.assertNotEqual(first[PAIR_SCOPE], other_ip[PAIR_SCOPE])
        self.assertNotEqual(first[IP_SCOPE], other_ip[IP_SCOPE])

    def test_ip_bucket_is_shared_across_different_usernames(self) -> None:
        first = throttle_bucket_keys("admin", "203.0.113.10")
        second = throttle_bucket_keys("missing", "203.0.113.10")

        self.assertEqual(first[IP_SCOPE], second[IP_SCOPE])
        self.assertNotEqual(first[PAIR_SCOPE], second[PAIR_SCOPE])

    def test_ip_bucket_blocks_after_failures_across_multiple_usernames(self) -> None:
        bucket = _bucket(IP_SCOPE)
        now = bucket.window_started_at
        results = [
            _advance_bucket(
                bucket,
                now=now + timedelta(minutes=index),
                threshold=5,
                window_minutes=15,
                block_minutes=60,
            )
            for index in range(5)
        ]

        self.assertEqual(results[-1], (60 * 60, False))
        self.assertEqual(bucket_retry_after(bucket, now + timedelta(minutes=5)), 59 * 60)

    def test_pair_threshold_blocks_only_that_source_bucket(self) -> None:
        bucket = _bucket()
        now = bucket.window_started_at

        self.assertEqual(
            _advance_bucket(
                bucket,
                now=now,
                threshold=3,
                window_minutes=15,
                block_minutes=15,
            ),
            (0, False),
        )
        _advance_bucket(
            bucket,
            now=now + timedelta(minutes=1),
            threshold=3,
            window_minutes=15,
            block_minutes=15,
        )
        retry_after, alert = _advance_bucket(
            bucket,
            now=now + timedelta(minutes=2),
            threshold=3,
            window_minutes=15,
            block_minutes=15,
        )

        self.assertEqual(retry_after, 15 * 60)
        self.assertFalse(alert)
        self.assertEqual(
            bucket_retry_after(bucket, now + timedelta(minutes=3)),
            14 * 60,
        )

    def test_account_bucket_alerts_but_never_blocks(self) -> None:
        bucket = _bucket(ACCOUNT_SCOPE)
        now = bucket.window_started_at
        results = [
            _advance_bucket(
                bucket,
                now=now + timedelta(minutes=index),
                threshold=3,
                window_minutes=15,
                block_minutes=None,
            )
            for index in range(4)
        ]

        self.assertEqual(results, [(0, False), (0, False), (0, True), (0, False)])
        self.assertIsNone(bucket.blocked_until)

    def test_expired_window_discards_old_failures(self) -> None:
        bucket = _bucket()
        bucket.failure_count = 2
        now = bucket.window_started_at + timedelta(minutes=15)

        retry_after, _ = _advance_bucket(
            bucket,
            now=now,
            threshold=3,
            window_minutes=15,
            block_minutes=15,
        )

        self.assertEqual(retry_after, 0)
        self.assertEqual(bucket.failure_count, 1)
        self.assertEqual(bucket.window_started_at, now)


class ClientIpTests(unittest.TestCase):
    def test_loopback_proxy_can_supply_valid_real_ip(self) -> None:
        request = _request(
            "127.0.0.1",
            **{"x-real-ip": "222.209.6.131", "x-forwarded-for": "198.51.100.8"},
        )
        self.assertEqual(client_ip(request), "222.209.6.131")

    def test_non_loopback_peer_cannot_spoof_forwarding_headers(self) -> None:
        request = _request(
            "203.0.113.10",
            **{"x-real-ip": "222.209.6.131", "x-forwarded-for": "198.51.100.8"},
        )
        self.assertEqual(client_ip(request), "203.0.113.10")

    def test_invalid_proxy_header_falls_back_to_peer(self) -> None:
        request = _request("127.0.0.1", **{"x-real-ip": "not-an-ip"})
        self.assertEqual(client_ip(request), "127.0.0.1")


class LoginThrottleDatabaseFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_is_locked_before_failure_registration(self) -> None:
        now = datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc)
        settings = _settings()
        keys = throttle_bucket_keys("admin", "203.0.113.10")
        ip_bucket = _bucket(IP_SCOPE)
        ip_bucket.bucket_key = keys[IP_SCOPE]
        pair_bucket = _bucket(PAIR_SCOPE)
        pair_bucket.bucket_key = keys[PAIR_SCOPE]
        pair_bucket.failure_count = 2

        db = SimpleNamespace(
            execute=AsyncMock(
                side_effect=[Mock(), _rows_result([ip_bucket, pair_bucket])]
            )
        )
        source = await lock_login_source(
            db,
            username="admin",
            ip="203.0.113.10",
            now=now,
            settings=settings,
        )

        self.assertEqual(set(source.buckets), {IP_SCOPE, PAIR_SCOPE})
        lock_statement = db.execute.await_args_list[1].args[0]
        self.assertIsNotNone(lock_statement._for_update_arg)

        account_bucket = _bucket(ACCOUNT_SCOPE)
        account_bucket.bucket_key = keys[ACCOUNT_SCOPE]
        db.execute = AsyncMock(
            side_effect=[Mock(), _rows_result([account_bucket])]
        )
        result = await register_login_failure(
            db,
            source=source,
            username="admin",
            ip="203.0.113.10",
            now=now,
            settings=settings,
        )

        self.assertEqual(result.retry_after_seconds, 15 * 60)
        self.assertEqual(result.scope, PAIR_SCOPE)
        self.assertEqual(ip_bucket.failure_count, 1)
        self.assertEqual(account_bucket.failure_count, 1)


class LoginAuditAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_failure_updates_existing_log(self) -> None:
        existing = SimpleNamespace(
            attempt_count=2,
            last_attempt_at=None,
            user_agent=None,
            user_id=None,
        )
        result = Mock()
        result.scalar_one_or_none.return_value = existing
        db = SimpleNamespace(execute=AsyncMock(return_value=result), add=Mock())
        now = datetime(2026, 7, 30, 8, 0, tzinfo=timezone.utc)

        await record_login_attempt(
            db,
            user=SimpleNamespace(id="user-id"),
            username="admin",
            success=False,
            fail_reason="密码错误",
            ip="203.0.113.10",
            user_agent="browser",
            now=now,
        )

        self.assertEqual(existing.attempt_count, 3)
        self.assertEqual(existing.last_attempt_at, now)
        self.assertEqual(existing.user_id, "user-id")
        db.add.assert_not_called()


class LoginEndpointProtectionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _db_with_user(user):
        results = [_query_result(user)]
        if user is not None:
            results.append(_query_result(user))
        return SimpleNamespace(
            execute=AsyncMock(side_effect=results),
            commit=AsyncMock(),
        )

    async def test_unknown_user_uses_dummy_hash_and_generic_error(self) -> None:
        db = self._db_with_user(None)
        register = AsyncMock(return_value=ThrottleResult())
        audit = AsyncMock()
        with (
            patch("api.auth.lock_login_source", new=AsyncMock(return_value=_source())),
            patch("api.auth.verify_password", return_value=False) as verify,
            patch("api.auth.register_login_failure", new=register),
            patch("api.auth.record_login_attempt", new=audit),
        ):
            with self.assertRaises(HTTPException) as caught:
                await login(
                    LoginRequest(username="missing", password="secret"),
                    _request("127.0.0.1", **{"x-real-ip": "203.0.113.10"}),
                    db,
                )

        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.detail, "用户名或密码错误")
        verify.assert_called_once()
        register.assert_awaited_once()
        audit.assert_awaited_once()
        db.commit.assert_awaited_once()

    async def test_active_source_limit_is_generic_and_skips_bcrypt(self) -> None:
        db = SimpleNamespace(execute=AsyncMock(), commit=AsyncMock())
        audit = AsyncMock()
        with (
            patch(
                "api.auth.lock_login_source",
                new=AsyncMock(
                    return_value=_source(
                        ThrottleResult(
                            retry_after_seconds=300,
                            scope=PAIR_SCOPE,
                        )
                    )
                ),
            ),
            patch("api.auth.verify_password") as verify,
            patch("api.auth.record_login_attempt", new=audit),
        ):
            with self.assertRaises(HTTPException) as caught:
                await login(
                    LoginRequest(username="admin", password="correct"),
                    _request("127.0.0.1", **{"x-real-ip": "203.0.113.10"}),
                    db,
                )

        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.headers["Retry-After"], "300")
        self.assertEqual(caught.exception.headers["Cache-Control"], "no-store")
        verify.assert_not_called()
        audit.assert_awaited_once()
        db.commit.assert_awaited_once()
        db.execute.assert_not_awaited()

    async def test_disabled_user_uses_real_hash_and_generic_error(self) -> None:
        user = _user()
        user.is_active = False
        db = self._db_with_user(user)
        register = AsyncMock(return_value=ThrottleResult())
        audit = AsyncMock()
        with (
            patch("api.auth.lock_login_source", new=AsyncMock(return_value=_source())),
            patch("api.auth.verify_password", return_value=True) as verify,
            patch("api.auth.register_login_failure", new=register),
            patch("api.auth.record_login_attempt", new=audit),
        ):
            with self.assertRaises(HTTPException) as caught:
                await login(
                    LoginRequest(username="admin", password="correct"),
                    _request("127.0.0.1", **{"x-real-ip": "203.0.113.10"}),
                    db,
                )

        self.assertEqual(caught.exception.status_code, 401)
        self.assertEqual(caught.exception.detail, "用户名或密码错误")
        verify.assert_called_once_with("correct", "hash")
        register.assert_awaited_once()
        self.assertEqual(audit.await_args.kwargs["fail_reason"], "账号已禁用")
        db.commit.assert_awaited_once()

    async def test_threshold_failure_commits_then_returns_retry_after(self) -> None:
        user = _user()
        db = self._db_with_user(user)
        register = AsyncMock(
            return_value=ThrottleResult(
                retry_after_seconds=15 * 60,
                scope=PAIR_SCOPE,
            )
        )
        audit = AsyncMock()
        with (
            patch("api.auth.lock_login_source", new=AsyncMock(return_value=_source())),
            patch("api.auth.verify_password", return_value=False),
            patch("api.auth.register_login_failure", new=register),
            patch("api.auth.record_login_attempt", new=audit),
        ):
            with self.assertRaises(HTTPException) as caught:
                await login(
                    LoginRequest(username="admin", password="wrong"),
                    _request("127.0.0.1", **{"x-real-ip": "203.0.113.10"}),
                    db,
                )

        self.assertEqual(caught.exception.status_code, 429)
        self.assertEqual(caught.exception.headers["Retry-After"], str(15 * 60))
        register.assert_awaited_once()
        self.assertEqual(
            audit.await_args.kwargs["fail_reason"],
            "密码错误，来源临时受限",
        )
        db.commit.assert_awaited_once()

    async def test_success_clears_only_current_pair_and_writes_audit(self) -> None:
        user = _user()
        db = self._db_with_user(user)
        audit = AsyncMock()
        clear_pair = AsyncMock()

        with (
            patch("api.auth.lock_login_source", new=AsyncMock(return_value=_source())),
            patch("api.auth.verify_password", return_value=True),
            patch("api.auth.clear_login_pair_failures", new=clear_pair),
            patch("api.auth.record_login_attempt", new=audit),
            patch("api.auth._load_permissions", new=AsyncMock(return_value=[])),
            patch("api.auth.create_access_token", return_value="token"),
        ):
            response = await login(
                LoginRequest(username="admin", password="correct"),
                _request("127.0.0.1", **{"x-real-ip": "203.0.113.10"}),
                db,
            )

        self.assertEqual(response.access_token, "token")
        clear_pair.assert_awaited_once_with(
            db,
            username="admin",
            ip="203.0.113.10",
        )
        audit.assert_awaited_once()
        db.commit.assert_awaited_once()
        self.assertEqual(db.execute.await_count, 2)


if __name__ == "__main__":
    unittest.main()
