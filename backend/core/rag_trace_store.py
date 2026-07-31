"""Asynchronous, bounded persistence for structured RAG trace events.

``trace_event`` must never wait for PostgreSQL while a user is receiving an
SSE answer.  It therefore writes to a bounded in-process queue; one worker
persists events in order and updates a compact run summary for the admin list.
If the queue or database is unavailable, application behavior is unaffected
and a warning is emitted instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, update

from config import get_settings
from core.rag_trace import trace_contains_business_content
from database import AsyncSessionLocal
from models.db_models import RagTraceEvent, RagTraceRun


logger = logging.getLogger(__name__)

_queue: asyncio.Queue[dict[str, Any]] | None = None
_worker_task: asyncio.Task | None = None
_cleanup_task: asyncio.Task | None = None
_accepting = False
_dropped_events = 0

_TERMINAL_SUCCESS_EVENTS = {"chat.response", "search_test.completed"}
_TERMINAL_ERROR_EVENTS = {"chat.error", "search_test.error", "chat.persistence_error"}
_TERMINAL_INTERRUPTED_EVENTS = {"chat.cancelled"}
_TERMINAL_EVENTS = (
    _TERMINAL_SUCCESS_EVENTS
    | _TERMINAL_ERROR_EVENTS
    | _TERMINAL_INTERRUPTED_EVENTS
)


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


def _as_uuid(value: Any) -> UUID | None:
    if value in (None, ""):
        return None
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _preview(value: Any, limit: int = 500) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    if not text:
        return None
    return text if len(text) <= limit else f"{text[:limit]}…"


def _request_kind(record: dict[str, Any]) -> str:
    event = str(record.get("event") or "")
    surface = str(record.get("surface") or "")
    if event.startswith("chat.") or surface == "chat":
        return "chat"
    if event.startswith("search_test.") or surface == "search_test":
        return "search_test"
    return "unknown"


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Limit pathological metadata without changing the normal trace schema."""

    if depth >= 10:
        return "[depth truncated]"
    if isinstance(value, str):
        limit = min(get_settings().rag_trace_content_max_chars, 50_000)
        return value if len(value) <= limit else f"{value[:limit]}…[truncated]"
    if isinstance(value, list):
        return [_bounded(item, depth=depth + 1) for item in value[:500]]
    if isinstance(value, dict):
        return {
            str(key): _bounded(item, depth=depth + 1)
            for key, item in list(value.items())[:500]
        }
    return value


def _queue_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    """Copy and cap one queued event by both structure and encoded bytes.

    ``trace_event`` has already converted the record to JSON-safe values.  The
    container log may retain the configured detail, while database persistence
    must have a stricter hard memory boundary when PostgreSQL is slow.
    """

    bounded = _bounded(record)
    settings = get_settings()
    limit = settings.rag_trace_max_event_bytes

    def encoded_size(value: dict[str, Any]) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )

    original_bytes = encoded_size(bounded)
    if original_bytes <= limit:
        return bounded

    essential_keys = (
        "trace_schema_version",
        "timestamp",
        "app_version",
        "app_revision",
        "event",
        "trace_id",
        "surface",
        "user_id",
        "conversation_id",
        "content_capture_enabled",
        "contains_business_content",
    )
    def essential_value(value: Any) -> Any:
        if isinstance(value, str) and len(value) > 256:
            return f"{value[:256]}…[truncated]"
        return value

    compact = {
        key: essential_value(bounded[key])
        for key in essential_keys
        if key in bounded
    }
    marker = {
        "persistence_payload_truncated": True,
        "persistence_original_bytes": original_bytes,
    }
    summary_keys = (
        "evidence_status",
        "retrieval_executed",
        "selected_kb_count",
        "displayed_result_count",
        "answer_source_count",
        "context_evidence_count",
        "hit_count",
        "candidate_count",
        "direct_evidence_count",
        "related_reference_count",
        "total_ms",
        "elapsed_ms",
        "answer_chars",
        "question_chars",
        "question_sha256",
        "query_chars",
        "query_sha256",
    )
    for key in summary_keys:
        if key not in bounded:
            continue
        trial = {**compact, key: bounded[key], **marker}
        if encoded_size(trial) <= limit:
            compact[key] = bounded[key]
    for key, value in bounded.items():
        if key in compact or key in marker:
            continue
        trial = {**compact, key: value, **marker}
        if encoded_size(trial) <= limit:
            compact[key] = value
    compact.update(marker)
    return compact


def _summary_updates(record: dict[str, Any]) -> dict[str, Any]:
    """Extract short, indexed run fields from one already-sanitized event."""

    event = str(record.get("event") or "unknown")
    updates: dict[str, Any] = {
        "current_stage": event,
        "updated_at": _parse_timestamp(record.get("timestamp")),
    }
    kind = _request_kind(record)
    if kind != "unknown":
        updates["request_kind"] = kind

    user_id = _as_uuid(record.get("user_id"))
    conversation_id = _as_uuid(record.get("conversation_id"))
    if user_id:
        updates["user_id"] = user_id
    if conversation_id:
        updates["conversation_id"] = conversation_id

    input_preview = _preview(record.get("question")) or _preview(record.get("query"))
    output_preview = _preview(record.get("answer")) or _preview(record.get("partial_answer"))
    if input_preview:
        updates["input_preview"] = input_preview
        updates["content_included"] = True
    if output_preview:
        updates["output_preview"] = output_preview
        updates["content_included"] = True
    # Prefer the marker generated before queue truncation and recursively inspect
    # old/hand-written records that predate it. This catches nested candidate,
    # context and metadata text without claiming every development event has text.
    if (
        record.get("contains_business_content") is True
        or trace_contains_business_content(record)
    ):
        updates["content_included"] = True

    selected_kbs = record.get("selected_kb_ids")
    if isinstance(selected_kbs, list):
        updates["selected_kb_count"] = len(selected_kbs)
    elif isinstance(record.get("selected_kb_count"), int):
        updates["selected_kb_count"] = record["selected_kb_count"]

    evidence_status = record.get("evidence_status")
    if isinstance(evidence_status, str) and evidence_status:
        updates["evidence_status"] = evidence_status[:32]

    # 审计口径：hit_count 只表示直接回答证据。展示候选、上下文 related
    # 资料和原始召回数量都有各自字段，不再从 results/sources 长度猜测命中。
    if evidence_status in {"no_hit", "skipped"}:
        updates["hit_count"] = 0
    else:
        for field in ("hit_count", "direct_evidence_count"):
            value = record.get(field)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            ):
                updates["hit_count"] = value
                break

    duration = record.get("total_ms")
    if not isinstance(duration, (int, float)) and event in _TERMINAL_SUCCESS_EVENTS:
        duration = record.get("elapsed_ms")
    if isinstance(duration, (int, float)) and duration >= 0:
        updates["duration_ms"] = round(duration)

    timestamp = _parse_timestamp(record.get("timestamp"))
    # 阶段错误（例如 retrieval.error）可能已由上一轮证据或其它通道恢复；
    # 它应在事件时间线中标红，但不能覆盖最终成功响应。只有明确终止请求的
    # chat/search_test 错误事件才决定整条调用链失败。
    if event in _TERMINAL_INTERRUPTED_EVENTS:
        updates.update(status="interrupted", completed_at=timestamp)
    elif event in _TERMINAL_ERROR_EVENTS:
        updates.update(status="error", completed_at=timestamp)
    elif event in _TERMINAL_SUCCESS_EVENTS:
        updates.update(status="success", completed_at=timestamp)
    return updates


def _evict_oldest_nonterminal_record(queue: asyncio.Queue[dict[str, Any]]) -> bool:
    """Make room for a terminal record without reordering retained events.

    The function is synchronous and contains no await, so the event-loop worker
    cannot consume the queue while it is temporarily drained and restored.
    Queue unfinished-task accounting is kept balanced for ``queue.join()``.
    """

    retained: list[dict[str, Any]] = []
    removed = False
    while True:
        try:
            queued = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        queue.task_done()
        event_name = str(queued.get("event") or "")
        if not removed and event_name not in _TERMINAL_EVENTS:
            removed = True
            continue
        retained.append(queued)
    for queued in retained:
        queue.put_nowait(queued)
    return removed


def enqueue_trace_record(record: dict[str, Any]) -> None:
    """Best-effort non-blocking enqueue called by :func:`trace_event`."""

    global _dropped_events
    if not _accepting or _queue is None:
        return

    event_name = str(record.get("event") or "")
    is_terminal = event_name in _TERMINAL_EVENTS
    # Keep a small part of the same FIFO queue available for terminal events.
    # This preserves per-trace ordering while preventing verbose candidate
    # events from making successful/failed/interrupted runs appear as running
    # until the stale-run cleanup executes 15 minutes later.
    reserve = (
        max(10, min(100, _queue.maxsize // 10))
        if _queue.maxsize > 0
        else 0
    )
    if (
        not is_terminal
        and _queue.maxsize > 0
        and _queue.qsize() >= max(0, _queue.maxsize - reserve)
    ):
        _dropped_events += 1
        if _dropped_events == 1 or _dropped_events % 100 == 0:
            logger.warning(
                "RAG 调用链持久化队列压力过高，已为终止事件预留容量并丢弃 %s 个非终止事件；"
                "问答主链路不受影响",
                _dropped_events,
            )
        return
    queued_record = _queue_safe_record(record)
    try:
        _queue.put_nowait(queued_record)
    except asyncio.QueueFull:
        # Reserved slots may themselves fill during a burst.  A terminal event
        # then evicts the oldest verbose non-terminal event and remains at the
        # tail, preserving the execution order of everything retained.
        if is_terminal and _evict_oldest_nonterminal_record(_queue):
            _dropped_events += 1
            _queue.put_nowait(queued_record)
            if _dropped_events == 1 or _dropped_events % 100 == 0:
                logger.warning(
                    "RAG 调用链队列已满，为保留终止状态已驱逐 %s 个非终止事件；"
                    "问答主链路不受影响",
                    _dropped_events,
                )
            return
        _dropped_events += 1
        if _dropped_events == 1 or _dropped_events % 100 == 0:
            logger.warning(
                "RAG 调用链持久化队列已满，已丢弃 %s 个事件；问答主链路不受影响",
                _dropped_events,
            )


async def _persist_batch(records: list[dict[str, Any]]) -> None:
    settings = get_settings()
    async with AsyncSessionLocal() as session:
        runs: dict[str, RagTraceRun] = {}
        for raw in records:
            trace_id = str(raw.get("trace_id") or "").strip()[:64]
            event_name = str(raw.get("event") or "unknown").strip()[:100]
            if not trace_id:
                continue

            run = runs.get(trace_id)
            if run is None:
                run = await session.get(RagTraceRun, trace_id)
                if run is None:
                    timestamp = _parse_timestamp(raw.get("timestamp"))
                    run = RagTraceRun(
                        trace_id=trace_id,
                        request_kind=_request_kind(raw),
                        status="running",
                        current_stage=event_name,
                        event_count=0,
                        observed_event_count=0,
                        storage_omitted_event_count=0,
                        storage_truncated=False,
                        content_included=False,
                        started_at=timestamp,
                        updated_at=timestamp,
                    )
                    session.add(run)
                runs[trace_id] = run

            run.observed_event_count += 1

            updates = _summary_updates(raw)
            # Once a stage has failed, a later response event must not make the
            # trace look healthy. It may still complete and retain its answer.
            if run.status == "error" and updates.get("status") == "success":
                updates.pop("status", None)
            for key, value in updates.items():
                setattr(run, key, value)
            timestamp = _parse_timestamp(raw.get("timestamp"))
            if timestamp < run.started_at:
                run.started_at = timestamp

            # Pipeline ``total_ms`` starts after synchronous intent routing, so
            # copying it from ``generation.completed`` under-reports what the
            # caller actually waited for.  A terminal event closes the run and
            # therefore has enough information to replace any stage-only value
            # with the wall-clock duration from the first observed event.
            if event_name in _TERMINAL_EVENTS:
                run.duration_ms = max(
                    0,
                    round((timestamp - run.started_at).total_seconds() * 1000),
                )

            max_events = settings.rag_trace_max_events_per_run
            is_terminal = event_name in _TERMINAL_EVENTS
            # Reserve the final slot for success/error/cancel. Candidate bursts
            # can be sampled, but an AI export must retain the request outcome.
            if not is_terminal and run.event_count >= max_events - 1:
                run.storage_truncated = True
                run.storage_omitted_event_count += 1
                continue
            sequence = run.event_count + 1
            if is_terminal and run.event_count >= max_events:
                victim = (
                    await session.execute(
                        select(RagTraceEvent)
                        .where(RagTraceEvent.trace_id == trace_id)
                        .order_by(RagTraceEvent.sequence.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if victim is None:
                    run.storage_truncated = True
                    run.storage_omitted_event_count += 1
                    continue
                sequence = victim.sequence
                await session.delete(victim)
                await session.flush()
                run.storage_truncated = True
                run.storage_omitted_event_count += 1
            else:
                run.event_count += 1
            session.add(RagTraceEvent(
                trace_id=trace_id,
                sequence=sequence,
                event=event_name,
                payload=_bounded(raw),
                created_at=timestamp,
            ))
        await session.commit()


async def _worker_loop() -> None:
    assert _queue is not None
    while True:
        first = await _queue.get()
        batch = [first]
        while len(batch) < 100:
            try:
                batch.append(_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        try:
            for attempt in range(2):
                try:
                    await _persist_batch(batch)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if attempt:
                        logger.error(
                            "RAG 调用链事件入库失败，丢弃本批 %s 条事件 error=%s",
                            len(batch),
                            type(exc).__name__,
                        )
                    else:
                        await asyncio.sleep(0.25)
        finally:
            for _ in batch:
                _queue.task_done()


async def cleanup_rag_traces() -> tuple[int, int]:
    """Delete expired traces and mark abandoned streams as interrupted."""

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=get_settings().rag_trace_retention_days)
    stale_before = now - timedelta(minutes=15)
    async with AsyncSessionLocal() as session:
        stale_result = await session.execute(
            update(RagTraceRun)
            .where(
                RagTraceRun.status == "running",
                RagTraceRun.updated_at < stale_before,
            )
            .values(status="interrupted", completed_at=RagTraceRun.updated_at)
        )
        delete_result = await session.execute(
            delete(RagTraceRun).where(RagTraceRun.started_at < cutoff)
        )
        await session.commit()
        return int(stale_result.rowcount or 0), int(delete_result.rowcount or 0)


async def _cleanup_loop() -> None:
    while True:
        try:
            stale, deleted = await cleanup_rag_traces()
            if stale or deleted:
                logger.info("RAG 调用链维护完成 interrupted=%s deleted=%s", stale, deleted)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("RAG 调用链维护失败 error=%s", type(exc).__name__)
        # 需要及时把用户断开连接后没有终止事件的流标记为 interrupted；
        # 保留期删除与该轻量维护共用同一事务，五分钟一次仅执行两条索引语句。
        await asyncio.sleep(5 * 60)


async def start_rag_trace_store() -> None:
    global _queue, _worker_task, _cleanup_task, _accepting
    settings = get_settings()
    # Retention is independent from collection. Turning off new trace capture
    # must not leave already stored business data beyond its configured TTL.
    if not _cleanup_task or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop(), name="rag-trace-cleanup")
    if not settings.rag_trace_enabled or not settings.rag_trace_persistence_enabled:
        return
    if _worker_task and not _worker_task.done():
        return
    _queue = asyncio.Queue(maxsize=settings.rag_trace_queue_size)
    _accepting = True
    _worker_task = asyncio.create_task(_worker_loop(), name="rag-trace-writer")


async def stop_rag_trace_store() -> None:
    global _accepting, _queue, _worker_task, _cleanup_task
    _accepting = False
    if _queue is not None:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(_queue.join(), timeout=10)
    for task in (_worker_task, _cleanup_task):
        if task:
            task.cancel()
    for task in (_worker_task, _cleanup_task):
        if task:
            with suppress(asyncio.CancelledError):
                await task
    _worker_task = None
    _cleanup_task = None
    _queue = None
