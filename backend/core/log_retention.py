"""操作审计与路由决策日志的保留策略。

登录日志与 RAG 调用链分别由 login_security / rag_trace_store 管理；
这里统一清理 operation_logs 与 intent_route_logs，避免后台审计与
路由决策表随团队规模无限增长。路由日志只保留无正文摘要，供效果调优。
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from config import get_settings
from database import AsyncSessionLocal
from models.db_models import IntentRouteLog, OperationLog


logger = logging.getLogger(__name__)
LOG_RETENTION_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60


async def cleanup_retained_logs() -> tuple[int, int]:
    """删除超过保留期的操作审计与路由决策日志，返回 (操作日志, 路由日志) 删除条数。"""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    operation_cutoff = now - timedelta(days=settings.operation_log_retention_days)
    route_cutoff = now - timedelta(days=settings.intent_route_log_retention_days)
    async with AsyncSessionLocal() as session:
        operation_result = await session.execute(
            delete(OperationLog).where(OperationLog.created_at < operation_cutoff)
        )
        await session.commit()
        route_result = await session.execute(
            delete(IntentRouteLog).where(IntentRouteLog.created_at < route_cutoff)
        )
        await session.commit()
    return int(operation_result.rowcount or 0), int(route_result.rowcount or 0)


async def log_retention_cleanup_loop() -> None:
    """应用运行期间每天清理一次过期审计与路由日志。"""
    while True:
        try:
            deleted_operations, deleted_routes = await cleanup_retained_logs()
            if deleted_operations or deleted_routes:
                logger.info(
                    "[日志保留] 已清理过期数据 operations=%s routes=%s",
                    deleted_operations,
                    deleted_routes,
                )
        except Exception as exc:
            logger.warning("[日志保留] 清理过期数据失败 error=%s", type(exc).__name__)
        await asyncio.sleep(LOG_RETENTION_CLEANUP_INTERVAL_SECONDS)
