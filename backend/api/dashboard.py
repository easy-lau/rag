"""数据看板聚合接口与 AI 分析报告。

只输出聚合指标，不返回任何明细行（明细继续由调用链路/审计日志提供）；
质量维度以 RAG 调用链的 evidence_status 为系统侧判定，不等同于人工标注的正确率。
AI 分析报告只把聚合数字交给模型，绝不携带问题正文、文档正文或会话内容。
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import BigInteger, case, cast, func, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from core.deps import require_permission
from core.openai_client import get_client
from core.permissions import DASHBOARD_READ
from database import get_db
from models.db_models import (
    Document,
    DocumentChunk,
    KnowledgeBase,
    LoginLog,
    OperationLog,
    RagTraceEvent,
    RagTraceRun,
    User,
)

router = APIRouter(prefix="/admin/dashboard", tags=["dashboard"])

_MAX_RETENTION_DAYS = 30
# ``func.date_trunc`` 会把第一个参数当绑定参数，PostgreSQL 在 GROUP BY 中无法
# 将其与 SELECT 里的同表达式关联，因此把日粒度固化为 SQL 字面量。
_DAY_TRUNC = func.date_trunc(literal_column("'day'"), RagTraceRun.started_at)
_TOKEN_EVENT_NAMES = (
    "intent.model_result",
    "query.analysis.completed",
    "query.understanding.v3.completed",
    "generation.completed",
)
_REPORT_MAX_TOKENS = 1800

_REPORT_SYSTEM_PROMPT = (
    "你是一个企业 RAG 知识库系统的运营分析助手。你会收到一份近 7 天或 30 天的"
    "聚合统计 JSON（只含数字和少量用户名，不含任何用户问题、回答或文档正文）。"
    "请输出一份简洁的中文 Markdown 分析报告，结构为："
    "## 总体概况（一句话总结系统状态）"
    "## 规模与使用（用户、知识库、文档、问答量）"
    "## 回答质量（命中率、澄清率、无答案率、失败数，指出趋势和可能原因）"
    "## 性能与资源消耗（平均/P95 响应耗时、平均命中片段数、Token 总量与阶段构成）"
    "## 安全与运营（登录成功与失败、登录账号数、失败来源数、管理操作要点）"
    "## 建议（3-5 条可执行的改进动作，基于数据给出，不要编造指标）。"
    "只依据提供的 JSON 数据作答，数据缺失时明确说明，不要猜测具体数值。"
)


async def _collect_overview(db: AsyncSession, days: int) -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    user_total = (await db.execute(select(func.count()).select_from(User))).scalar_one()
    active_users = (await db.execute(
        select(func.count(func.distinct(RagTraceRun.user_id))).where(
            RagTraceRun.request_kind == "chat",
            RagTraceRun.started_at >= cutoff,
            RagTraceRun.user_id.is_not(None),
        )
    )).scalar_one()
    kb_total = (await db.execute(
        select(func.count()).select_from(KnowledgeBase)
    )).scalar_one()
    document_total = (await db.execute(
        select(func.count()).select_from(Document)
    )).scalar_one()
    chunk_total = (await db.execute(
        select(func.count()).select_from(DocumentChunk)
    )).scalar_one()
    new_documents = (await db.execute(
        select(func.count()).select_from(Document).where(
            Document.created_at >= cutoff
        )
    )).scalar_one()

    status_rows = (await db.execute(
        select(Document.status, func.count()).group_by(Document.status)
    )).all()
    documents_by_status = {
        str(status or "unknown"): int(count) for status, count in status_rows
    }

    daily_rows = (await db.execute(
        select(
            _DAY_TRUNC,
            func.count(),
            func.avg(RagTraceRun.duration_ms),
        )
        .where(
            RagTraceRun.request_kind == "chat",
            RagTraceRun.started_at >= cutoff,
        )
        .group_by(_DAY_TRUNC)
        .order_by(_DAY_TRUNC)
    )).all()
    qa_daily = [
        {
            "date": day.strftime("%Y-%m-%d"),
            "count": int(count),
            "avg_duration_ms": round(float(avg_ms)) if avg_ms is not None else None,
        }
        for day, count, avg_ms in daily_rows
    ]

    user_rows = (await db.execute(
        select(
            RagTraceRun.user_id,
            func.count(),
            func.sum(case(
                (RagTraceRun.evidence_status.in_(("hit", "partial")), 1),
                else_=0,
            )),
            func.avg(RagTraceRun.duration_ms),
            func.max(RagTraceRun.started_at),
        )
        .where(
            RagTraceRun.request_kind == "chat",
            RagTraceRun.started_at >= cutoff,
            RagTraceRun.user_id.is_not(None),
        )
        .group_by(RagTraceRun.user_id)
        .order_by(func.count().desc())
        .limit(10)
    )).all()
    user_ids = [row[0] for row in user_rows]
    name_by_id: dict = {}
    if user_ids:
        name_rows = (await db.execute(
            select(User.id, User.username).where(User.id.in_(user_ids))
        )).all()
        name_by_id = {row[0]: row[1] for row in name_rows}
    qa_per_user = [
        {
            "user_id": str(user_id),
            "username": name_by_id.get(user_id, "已删除用户"),
            "count": int(count),
            "hit_rate": round(int(hit_count or 0) / int(count), 4) if count else 0.0,
            "avg_duration_ms": round(float(avg_duration_ms)) if avg_duration_ms is not None else None,
            "last_active_at": last_active_at.isoformat() if last_active_at else None,
        }
        for user_id, count, hit_count, avg_duration_ms, last_active_at in user_rows
    ]

    evidence_rows = (await db.execute(
        select(RagTraceRun.evidence_status, func.count())
        .where(
            RagTraceRun.request_kind == "chat",
            RagTraceRun.started_at >= cutoff,
            RagTraceRun.evidence_status.is_not(None),
        )
        .group_by(RagTraceRun.evidence_status)
    )).all()
    by_evidence = {
        str(status): int(count) for status, count in evidence_rows
    }

    qa_total = sum(by_evidence.values())
    hit_count = by_evidence.get("hit", 0) + by_evidence.get("partial", 0)
    clarify_count = by_evidence.get("needs_clarification", 0)
    no_answer_count = (
        by_evidence.get("no_hit", 0) + by_evidence.get("insufficient_evidence", 0)
    )
    error_count = (await db.execute(
        select(func.count()).select_from(RagTraceRun).where(
            RagTraceRun.request_kind == "chat",
            RagTraceRun.started_at >= cutoff,
            RagTraceRun.status == "error",
        )
    )).scalar_one()

    perf_row = (await db.execute(
        select(
            func.avg(RagTraceRun.duration_ms),
            func.percentile_cont(0.95).within_group(RagTraceRun.duration_ms),
        ).where(
            RagTraceRun.request_kind == "chat",
            RagTraceRun.started_at >= cutoff,
            RagTraceRun.duration_ms.is_not(None),
        )
    )).first()
    hit_kb_row = (await db.execute(
        select(
            func.avg(RagTraceRun.hit_count),
            func.avg(RagTraceRun.selected_kb_count),
        ).where(
            RagTraceRun.request_kind == "chat",
            RagTraceRun.started_at >= cutoff,
            RagTraceRun.hit_count.is_not(None),
        )
    )).first()

    # Token 不在调用链摘要表中，而是由每次模型调用完成事件记录。这里只统计
    # 明确的一次性完成事件，避免把诊断快照里重复出现的 usage 再次累计。
    prompt_token_json = RagTraceEvent.payload["prompt_tokens"]
    completion_token_json = RagTraceEvent.payload["completion_tokens"]
    total_token_json = RagTraceEvent.payload["total_tokens"]
    prompt_token_metric = case(
        (
            func.jsonb_typeof(prompt_token_json) == "number",
            cast(prompt_token_json.astext, BigInteger),
        ),
        else_=0,
    )
    completion_token_metric = case(
        (
            func.jsonb_typeof(completion_token_json) == "number",
            cast(completion_token_json.astext, BigInteger),
        ),
        else_=0,
    )
    total_token_metric = case(
        (
            func.jsonb_typeof(total_token_json) == "number",
            cast(total_token_json.astext, BigInteger),
        ),
        else_=prompt_token_metric + completion_token_metric,
    )
    measured_call_metric = case(
        (
            or_(
                func.jsonb_typeof(prompt_token_json) == "number",
                func.jsonb_typeof(completion_token_json) == "number",
                func.jsonb_typeof(total_token_json) == "number",
            ),
            1,
        ),
        else_=0,
    )
    token_rows = (await db.execute(
        select(
            _DAY_TRUNC,
            RagTraceEvent.event,
            func.sum(prompt_token_metric),
            func.sum(completion_token_metric),
            func.sum(total_token_metric),
            func.sum(measured_call_metric),
        )
        .join(RagTraceEvent, RagTraceEvent.trace_id == RagTraceRun.trace_id)
        .where(
            RagTraceRun.request_kind == "chat",
            RagTraceRun.started_at >= cutoff,
            RagTraceEvent.event.in_(_TOKEN_EVENT_NAMES),
        )
        .group_by(_DAY_TRUNC, RagTraceEvent.event)
        .order_by(_DAY_TRUNC, RagTraceEvent.event)
    )).all()

    token_daily_by_date: dict[str, dict] = {}
    token_by_stage: dict[str, dict] = {}
    for day, event, prompt_tokens, completion_tokens, total_tokens, measured_calls in token_rows:
        date = day.strftime("%Y-%m-%d")
        daily = token_daily_by_date.setdefault(date, {
            "date": date,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "measured_calls": 0,
        })
        stage = token_by_stage.setdefault(str(event), {
            "stage": str(event),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "measured_calls": 0,
        })
        for target in (daily, stage):
            target["prompt_tokens"] += int(prompt_tokens or 0)
            target["completion_tokens"] += int(completion_tokens or 0)
            target["total_tokens"] += int(total_tokens or 0)
            target["measured_calls"] += int(measured_calls or 0)

    token_daily = list(token_daily_by_date.values())
    token_stages = sorted(
        token_by_stage.values(), key=lambda item: item["total_tokens"], reverse=True
    )
    prompt_token_total = sum(item["prompt_tokens"] for item in token_daily)
    completion_token_total = sum(item["completion_tokens"] for item in token_daily)
    token_total = sum(item["total_tokens"] for item in token_daily)
    measured_call_total = sum(item["measured_calls"] for item in token_daily)

    login_row = (await db.execute(
        select(
            func.coalesce(func.sum(case(
                (LoginLog.success.is_(True), LoginLog.attempt_count),
                else_=0,
            )), 0),
            func.coalesce(func.sum(case(
                (LoginLog.success.is_(False), LoginLog.attempt_count),
                else_=0,
            )), 0),
            func.count(func.distinct(case(
                (LoginLog.success.is_(True), LoginLog.user_id),
                else_=None,
            ))),
            func.count(func.distinct(case(
                (LoginLog.success.is_(False), LoginLog.ip),
                else_=None,
            ))),
        ).where(LoginLog.last_attempt_at >= cutoff)
    )).first()
    login_success = login_row[0] if login_row else 0
    login_failed = login_row[1] if login_row else 0
    login_users = login_row[2] if login_row else 0
    failed_sources = login_row[3] if login_row else 0

    operation_total = (await db.execute(
        select(func.count()).select_from(OperationLog).where(
            OperationLog.created_at >= cutoff
        )
    )).scalar_one()
    top_action_rows = (await db.execute(
        select(OperationLog.action, func.count())
        .where(OperationLog.created_at >= cutoff)
        .group_by(OperationLog.action)
        .order_by(func.count().desc())
        .limit(5)
    )).all()

    return {
        "days": days,
        "generated_at": now.isoformat(),
        "scale": {
            "users": int(user_total),
            "active_users": int(active_users),
            "knowledge_bases": int(kb_total),
            "documents": int(document_total),
            "documents_by_status": documents_by_status,
            "chunks": int(chunk_total),
            "new_documents": int(new_documents),
        },
        "qa": {
            "total": qa_total,
            "daily": qa_daily,
            "per_user": qa_per_user,
        },
        "quality": {
            "by_evidence": by_evidence,
            "hit_rate": round(hit_count / qa_total, 4) if qa_total else 0.0,
            "clarify_rate": round(clarify_count / qa_total, 4) if qa_total else 0.0,
            "no_answer_rate": round(no_answer_count / qa_total, 4) if qa_total else 0.0,
            "error_count": int(error_count),
        },
        "performance": {
            "avg_duration_ms": round(float(perf_row[0])) if perf_row and perf_row[0] is not None else None,
            "p95_duration_ms": round(float(perf_row[1])) if perf_row and perf_row[1] is not None else None,
            "avg_hit_count": round(float(hit_kb_row[0]), 2) if hit_kb_row and hit_kb_row[0] is not None else None,
            "avg_selected_kb_count": round(float(hit_kb_row[1]), 2) if hit_kb_row and hit_kb_row[1] is not None else None,
        },
        "tokens": {
            "prompt_tokens": prompt_token_total,
            "completion_tokens": completion_token_total,
            "total_tokens": token_total,
            "measured_calls": measured_call_total,
            "avg_tokens_per_qa": round(token_total / qa_total) if qa_total else 0,
            "daily": token_daily,
            "by_stage": token_stages,
        },
        "security": {
            "login_success": int(login_success),
            "login_failed": int(login_failed),
            "login_users": int(login_users),
            "failed_sources": int(failed_sources),
        },
        "operations": {
            "total": int(operation_total),
            "top_actions": [
                {"action": str(action), "count": int(count)}
                for action, count in top_action_rows
            ],
        },
    }


@router.get("/overview")
async def dashboard_overview(
    days: int = Query(7, ge=1, le=_MAX_RETENTION_DAYS),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(DASHBOARD_READ)),
):
    """返回系统规模、问答质量、性能、安全与运营的聚合统计（近 N 天）。"""
    return await _collect_overview(db, days)


@router.get("/report")
async def dashboard_ai_report(
    days: int = Query(7, ge=1, le=_MAX_RETENTION_DAYS),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(DASHBOARD_READ)),
):
    """基于聚合统计调用对话模型生成中文 Markdown 运营分析报告。"""
    settings = get_settings()
    if not (settings.llm_api_key and settings.llm_base_url and settings.chat_model):
        raise HTTPException(status_code=503, detail="未配置对话模型，无法生成 AI 分析报告")
    overview = await _collect_overview(db, days)
    payload = {
        "days": overview["days"],
        "scale": overview["scale"],
        "qa": {
            "total": overview["qa"]["total"],
            "daily": overview["qa"]["daily"],
            "per_user": overview["qa"]["per_user"],
        },
        "quality": overview["quality"],
        "performance": overview["performance"],
        "tokens": overview["tokens"],
        "security": overview["security"],
        "operations": overview["operations"],
    }
    try:
        client = get_client()
        response = await client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": _REPORT_SYSTEM_PROMPT},
                {"role": "user", "content": f"统计周期：近 {days} 天。聚合数据：\n{payload}"},
            ],
            temperature=0.3,
            max_tokens=_REPORT_MAX_TOKENS,
            timeout=float(settings.llm_request_timeout_seconds),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI 分析报告生成失败：{type(exc).__name__}",
        )
    report = (response.choices[0].message.content or "").strip()
    if not report:
        raise HTTPException(status_code=502, detail="AI 分析报告生成失败：模型返回空内容")
    return {"days": days, "report": report}
