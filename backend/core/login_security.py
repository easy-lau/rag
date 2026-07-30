"""登录来源限流、审计聚合与日志保留策略。"""

import asyncio
import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, or_, select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from config import Settings, get_settings
from database import AsyncSessionLocal
from models.db_models import LoginLog, LoginThrottle, User


logger = logging.getLogger(__name__)
LOGIN_LOG_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60

PAIR_SCOPE = "pair"
IP_SCOPE = "ip"
ACCOUNT_SCOPE = "account"


@dataclass(frozen=True)
class ThrottleResult:
    """一次限流检查或失败登记的结果。"""

    retry_after_seconds: int = 0
    scope: str | None = None
    account_alert: bool = False


@dataclass
class LockedLoginSource:
    """当前事务已经锁定的 IP 与 IP + 用户名状态。"""

    buckets: dict[str, LoginThrottle]
    throttle: ThrottleResult


@dataclass(frozen=True)
class _ThrottleRule:
    scope: str
    bucket_key: str
    threshold: int
    window_minutes: int
    block_minutes: int | None


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _bucket_digest(*parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def throttle_bucket_keys(username: str, ip: str | None) -> dict[str, str]:
    """生成不可逆的稳定限流键；用户名大小写变体归入同一桶。"""
    normalized_username = username.strip().casefold()
    normalized_ip = (ip or "unknown").strip().casefold()
    return {
        PAIR_SCOPE: _bucket_digest(PAIR_SCOPE, normalized_ip, normalized_username),
        IP_SCOPE: _bucket_digest(IP_SCOPE, normalized_ip),
        ACCOUNT_SCOPE: _bucket_digest(ACCOUNT_SCOPE, normalized_username),
    }


def _rules_for(
    username: str,
    ip: str | None,
    settings: Settings,
) -> list[_ThrottleRule]:
    keys = throttle_bucket_keys(username, ip)
    return [
        _ThrottleRule(
            scope=IP_SCOPE,
            bucket_key=keys[IP_SCOPE],
            threshold=settings.login_ip_failure_threshold,
            window_minutes=settings.login_ip_window_minutes,
            block_minutes=settings.login_ip_block_minutes,
        ),
        _ThrottleRule(
            scope=PAIR_SCOPE,
            bucket_key=keys[PAIR_SCOPE],
            threshold=settings.login_pair_failure_threshold,
            window_minutes=settings.login_pair_window_minutes,
            block_minutes=settings.login_pair_block_minutes,
        ),
        _ThrottleRule(
            scope=ACCOUNT_SCOPE,
            bucket_key=keys[ACCOUNT_SCOPE],
            threshold=settings.login_account_alert_threshold,
            window_minutes=settings.login_account_alert_window_minutes,
            block_minutes=None,
        ),
    ]


def bucket_retry_after(bucket: LoginThrottle, now: datetime) -> int:
    """返回某一来源桶剩余限制秒数。"""
    blocked_until = _aware_utc(bucket.blocked_until)
    if blocked_until is None or blocked_until <= now:
        return 0
    return max(1, math.ceil((blocked_until - now).total_seconds()))


def _advance_bucket(
    bucket: LoginThrottle,
    *,
    now: datetime,
    threshold: int,
    window_minutes: int,
    block_minutes: int | None,
) -> tuple[int, bool]:
    """登记一次失败；返回（限制秒数，是否首次触发账号异常告警）。"""
    active_retry_after = bucket_retry_after(bucket, now)
    if active_retry_after:
        return active_retry_after, False

    bucket.blocked_until = None
    window_started = _aware_utc(bucket.window_started_at)
    if (
        window_started is None
        or now - window_started >= timedelta(minutes=window_minutes)
    ):
        bucket.failure_count = 0
        bucket.window_started_at = now

    bucket.failure_count = int(bucket.failure_count or 0) + 1
    bucket.last_failed_at = now
    bucket.updated_at = now

    if block_minutes is None:
        return 0, bucket.failure_count == threshold

    if bucket.failure_count < threshold:
        return 0, False

    bucket.blocked_until = now + timedelta(minutes=block_minutes)
    bucket.failure_count = 0
    bucket.window_started_at = now
    return block_minutes * 60, False


def _max_throttle_result(
    results: list[tuple[str, int]],
    *,
    account_alert: bool = False,
) -> ThrottleResult:
    scope, seconds = max(results, key=lambda item: item[1], default=(None, 0))
    return ThrottleResult(
        retry_after_seconds=seconds,
        scope=scope if seconds else None,
        account_alert=account_alert,
    )


async def _ensure_and_lock_buckets(
    db: AsyncSession,
    *,
    rules: list[_ThrottleRule],
    now: datetime,
) -> dict[str, LoginThrottle]:
    """原子确保状态桶存在，并按固定 scope 顺序加行锁。"""
    values = [
        {
            "scope": rule.scope,
            "bucket_key": rule.bucket_key,
            "failure_count": 0,
            "window_started_at": now,
            "blocked_until": None,
            "last_failed_at": now,
            "created_at": now,
            "updated_at": now,
        }
        for rule in rules
    ]
    await db.execute(
        pg_insert(LoginThrottle)
        .values(values)
        .on_conflict_do_nothing(
            index_elements=[LoginThrottle.scope, LoginThrottle.bucket_key]
        )
    )

    identities = [(rule.scope, rule.bucket_key) for rule in rules]
    rows = (
        await db.execute(
            select(LoginThrottle)
            .where(
                tuple_(LoginThrottle.scope, LoginThrottle.bucket_key).in_(identities)
            )
            .order_by(LoginThrottle.scope, LoginThrottle.bucket_key)
            .with_for_update()
        )
    ).scalars().all()
    buckets = {row.scope: row for row in rows}
    if len(buckets) != len(rules):
        raise RuntimeError("登录限流状态初始化不完整")
    return buckets


async def lock_login_source(
    db: AsyncSession,
    *,
    username: str,
    ip: str | None,
    now: datetime,
    settings: Settings | None = None,
) -> LockedLoginSource:
    """在 bcrypt 前锁定来源桶并检查限制，避免并发请求越过阈值。"""
    effective_settings = settings or get_settings()
    source_rules = [
        rule
        for rule in _rules_for(username, ip, effective_settings)
        if rule.scope in {IP_SCOPE, PAIR_SCOPE}
    ]
    buckets = await _ensure_and_lock_buckets(db, rules=source_rules, now=now)
    throttle = _max_throttle_result(
        [
            (scope, bucket_retry_after(bucket, now))
            for scope, bucket in buckets.items()
        ]
    )
    return LockedLoginSource(buckets=buckets, throttle=throttle)


async def register_login_failure(
    db: AsyncSession,
    *,
    source: LockedLoginSource,
    username: str,
    ip: str | None,
    now: datetime,
    settings: Settings | None = None,
) -> ThrottleResult:
    """在已经锁定的来源桶中登记失败，并单独更新只告警的账号桶。"""
    effective_settings = settings or get_settings()
    rules = _rules_for(username, ip, effective_settings)
    source_rules = [rule for rule in rules if rule.scope in source.buckets]
    account_rules = [rule for rule in rules if rule.scope == ACCOUNT_SCOPE]

    retry_results: list[tuple[str, int]] = []
    for rule in source_rules:
        bucket = source.buckets[rule.scope]
        retry_after, alert = _advance_bucket(
            bucket,
            now=now,
            threshold=rule.threshold,
            window_minutes=rule.window_minutes,
            block_minutes=rule.block_minutes,
        )
        retry_results.append((rule.scope, retry_after))

    account_buckets = await _ensure_and_lock_buckets(
        db,
        rules=account_rules,
        now=now,
    )
    account_rule = account_rules[0]
    _, account_alert = _advance_bucket(
        account_buckets[ACCOUNT_SCOPE],
        now=now,
        threshold=account_rule.threshold,
        window_minutes=account_rule.window_minutes,
        block_minutes=None,
    )

    if account_alert:
        logger.warning(
            "[登录安全] 用户名遭遇集中失败，仅告警不锁账号 username=%r source_ip=%s",
            username,
            ip or "unknown",
        )
    return _max_throttle_result(retry_results, account_alert=account_alert)


async def clear_login_pair_failures(
    db: AsyncSession,
    *,
    username: str,
    ip: str | None,
) -> None:
    """登录成功后只清理当前 IP + 用户名状态，不掩盖 IP 扫描或账号攻击告警。"""
    key = throttle_bucket_keys(username, ip)[PAIR_SCOPE]
    await db.execute(
        delete(LoginThrottle).where(
            LoginThrottle.scope == PAIR_SCOPE,
            LoginThrottle.bucket_key == key,
        )
    )


async def record_login_attempt(
    db: AsyncSession,
    *,
    user: User | None,
    username: str,
    success: bool,
    fail_reason: str | None,
    ip: str | None,
    user_agent: str | None,
    now: datetime,
) -> None:
    """写登录审计；短时间内相同失败合并为一条并累加次数。"""
    safe_user_agent = user_agent[:1024] if user_agent else None
    if not success:
        settings = get_settings()
        cutoff = now - timedelta(seconds=settings.login_log_aggregate_seconds)
        latest = (
            await db.execute(
                select(LoginLog)
                .where(
                    LoginLog.success.is_(False),
                    LoginLog.username == username,
                    LoginLog.ip == ip,
                    LoginLog.fail_reason == fail_reason,
                    func.coalesce(LoginLog.last_attempt_at, LoginLog.created_at) >= cutoff,
                )
                .order_by(LoginLog.last_attempt_at.desc())
                .limit(1)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if latest is not None:
            latest.attempt_count = int(latest.attempt_count or 1) + 1
            latest.last_attempt_at = now
            latest.user_agent = safe_user_agent
            if latest.user_id is None and user is not None:
                latest.user_id = user.id
            return

    db.add(
        LoginLog(
            user_id=user.id if user is not None else None,
            username=username,
            success=success,
            fail_reason=fail_reason,
            ip=ip,
            user_agent=safe_user_agent,
            attempt_count=1,
            last_attempt_at=now,
            created_at=now,
        )
    )


async def cleanup_expired_login_security_state() -> tuple[int, int]:
    """清理过期审计日志和已失效来源桶，避免攻击数据无限增长。"""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    log_cutoff = now - timedelta(days=settings.login_log_retention_days)
    throttle_cutoff = now - timedelta(hours=settings.login_throttle_retention_hours)
    async with AsyncSessionLocal() as session:
        log_result = await session.execute(
            delete(LoginLog).where(LoginLog.last_attempt_at < log_cutoff)
        )
        await session.commit()

        # 登录请求始终按 ip -> pair -> account 取锁。清理按 scope 分事务提交，
        # 避免批量 DELETE 与登录事务形成跨表或跨 scope 的循环等待。
        deleted_throttles = 0
        for scope in (IP_SCOPE, PAIR_SCOPE, ACCOUNT_SCOPE):
            throttle_result = await session.execute(
                delete(LoginThrottle).where(
                    LoginThrottle.scope == scope,
                    LoginThrottle.last_failed_at < throttle_cutoff,
                    or_(
                        LoginThrottle.blocked_until.is_(None),
                        LoginThrottle.blocked_until <= now,
                    ),
                )
            )
            deleted_throttles += int(throttle_result.rowcount or 0)
            await session.commit()
    return int(log_result.rowcount or 0), deleted_throttles


async def login_log_cleanup_loop() -> None:
    """应用运行期间每天清理一次登录安全数据。"""
    while True:
        try:
            deleted_logs, deleted_throttles = await cleanup_expired_login_security_state()
            if deleted_logs or deleted_throttles:
                logger.info(
                    "[登录安全] 已清理过期数据 logs=%s throttles=%s",
                    deleted_logs,
                    deleted_throttles,
                )
        except Exception as exc:
            logger.warning("[登录安全] 清理过期数据失败 error=%s", type(exc).__name__)
        await asyncio.sleep(LOGIN_LOG_CLEANUP_INTERVAL_SECONDS)
