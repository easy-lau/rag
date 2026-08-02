"""Durable chat-turn state and idempotency helpers.

The HTTP/SSE handler deliberately keeps retrieval and generation outside this
module.  This module owns only the small state machine and the database
boundaries needed to reserve/recover a turn.  Keeping those rules in one place
prevents a retry path from accidentally dispatching the RAG pipeline again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from core.evidence_status import canonical_evidence_status
from models.db_models import ChatTurn, Message, now_utc


TURN_STATUSES = frozenset(
    {
        "accepted",
        "generating",
        "generated",
        "completed",
        "persist_failed",
        "failed",
        "cancelled",
    }
)

TERMINAL_TURN_STATUSES = frozenset({"completed", "failed", "cancelled"})
RECOVERABLE_TURN_STATUSES = frozenset({"generated", "persist_failed"})

# A failed/persist-failed turn can only move forward.  ``generated`` is kept
# separate from ``completed`` so the assistant message commit can be retried.
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "accepted": frozenset({"generating", "failed", "cancelled"}),
    "generating": frozenset({"generated", "failed", "cancelled"}),
    "generated": frozenset({"completed", "persist_failed", "failed", "cancelled"}),
    "persist_failed": frozenset({"generated", "completed", "persist_failed", "failed"}),
    "completed": frozenset({"completed"}),
    "failed": frozenset({"failed"}),
    "cancelled": frozenset({"cancelled"}),
}

MAX_PERSIST_ATTEMPTS = 3
# Longer than the configured maximum generation workflow (300s), leaving room
# for final SSE parsing and answer staging without a live worker being stolen.
DEFAULT_TURN_LEASE_SECONDS = 600
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_T = TypeVar("_T")


class TurnRequestConflict(ValueError):
    """The same request id was reused with a different question/turn id."""


class TurnReservationRace(RuntimeError):
    """A unique-key race could not be resolved by re-reading the turn."""


def normalize_request_id(value: object) -> str:
    """Return a bounded opaque idempotency key, generating one when omitted."""

    if value is None:
        return uuid.uuid4().hex
    text = str(value).strip()
    if not _REQUEST_ID_RE.fullmatch(text):
        raise ValueError(
            "request_id 必须为 1-128 位字母、数字或 . _ : - 组成的安全标识"
        )
    return text


def normalize_turn_id(value: object) -> uuid.UUID:
    if value is None:
        return uuid.uuid4()
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("turn_id 必须是有效 UUID") from exc


def question_digest(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _canonical_uuid_values(values: object) -> list[str]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return []
    normalized: set[str] = set()
    for value in values:
        try:
            normalized.add(str(uuid.UUID(str(value))))
        except (TypeError, ValueError, AttributeError):
            normalized.add(str(value).strip())
    return sorted(value for value in normalized if value)


def _canonical_search_config(value: object) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    tags = raw.get("tags")
    return {
        "method": str(raw.get("method") or "hybrid").strip().casefold(),
        "rerank": bool(raw.get("rerank", True)),
        "top_k": int(raw.get("top_k") or 5),
        "tags": sorted(
            {
                str(item).strip()
                for item in (tags if isinstance(tags, list) else [])
                if str(item).strip()
            }
        ),
    }


def build_turn_request_context(
    *,
    question: str,
    conversation_id: uuid.UUID,
    knowledge_base_ids: object = (),
    search_config: object = None,
    pending_route_revision: int = 0,
    pending_state_id: str | None = None,
) -> dict[str, Any]:
    """Build the canonical envelope protected by the idempotency key.

    ``request_id`` alone is only an opaque lookup key.  This envelope prevents
    an accidentally reused key from replaying an answer generated with a
    different KB scope, search mode, or clarification revision.
    """

    return {
        "schema_version": "chat_turn_request.v1",
        # Keep only the digest in the durable envelope; the transcript owns
        # business text and the idempotency ledger does not need another copy.
        "question_hash": question_digest(str(question)),
        "conversation_id": str(conversation_id),
        "knowledge_base_ids": _canonical_uuid_values(knowledge_base_ids),
        "search_config": _canonical_search_config(search_config),
        "pending_route": {
            "revision": max(0, int(pending_route_revision or 0)),
            "state_id": str(pending_state_id).strip()[:128]
            if pending_state_id
            else None,
        },
    }


def request_context_fingerprint(context: dict[str, Any]) -> str:
    encoded = json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_fingerprint(
    *,
    question: str,
    conversation_id: uuid.UUID,
    knowledge_base_ids: object = (),
    search_config: object = None,
    pending_route_revision: int = 0,
    pending_state_id: str | None = None,
) -> str:
    return request_context_fingerprint(
        build_turn_request_context(
            question=question,
            conversation_id=conversation_id,
            knowledge_base_ids=knowledge_base_ids,
            search_config=search_config,
            pending_route_revision=pending_route_revision,
            pending_state_id=pending_state_id,
        )
    )


def assert_turn_request_matches(
    turn: ChatTurn,
    context: dict[str, Any],
) -> None:
    """Reject every material parameter drift for an existing request id."""

    expected = request_context_fingerprint(context)
    stored = str(getattr(turn, "request_fingerprint", "") or "").strip()
    if stored:
        if stored != expected:
            raise TurnRequestConflict("同一 request_id 的请求参数不一致")
        return
    # Compatibility for in-memory/rolling-upgrade fixtures created before the
    # full envelope existed.  Production rows created by migration 0030 always
    # carry ``request_fingerprint``.
    if turn.question_hash != str(context.get("question_hash") or ""):
        raise TurnRequestConflict("同一 request_id 不能复用为不同问题")
    if str(turn.conversation_id) != str(context.get("conversation_id") or ""):
        raise TurnRequestConflict("同一 request_id 已属于其他会话")


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def turn_lease_expired(
    turn: ChatTurn,
    *,
    now: datetime | None = None,
    lease_seconds: int = DEFAULT_TURN_LEASE_SECONDS,
) -> bool:
    """Return whether an in-flight turn may safely be reclaimed."""

    if str(getattr(turn, "status", "") or "").casefold() not in {
        "accepted",
        "generating",
    }:
        return False
    stamp = _aware_utc(now or now_utc())
    expires_at = _aware_utc(getattr(turn, "lease_expires_at", None))
    if expires_at is not None:
        return expires_at <= stamp
    updated_at = _aware_utc(getattr(turn, "updated_at", None))
    if updated_at is None:
        return True
    return updated_at + timedelta(seconds=max(1, int(lease_seconds))) <= stamp


def renew_turn_lease(
    turn: ChatTurn,
    *,
    owner: str,
    now: datetime | None = None,
    lease_seconds: int = DEFAULT_TURN_LEASE_SECONDS,
    count_attempt: bool = False,
) -> ChatTurn:
    stamp = _aware_utc(now or now_utc())
    turn.lease_owner = str(owner)[:64]
    turn.lease_expires_at = stamp + timedelta(
        seconds=max(1, int(lease_seconds))
    )
    if count_attempt:
        turn.execution_attempts = int(
            getattr(turn, "execution_attempts", 0) or 0
        ) + 1
    turn.updated_at = stamp
    return turn


def reclaim_stale_turn(
    turn: ChatTurn,
    *,
    owner: str,
    now: datetime | None = None,
    lease_seconds: int = DEFAULT_TURN_LEASE_SECONDS,
) -> bool:
    """Reset an expired in-flight turn so the same request can execute again."""

    stamp = _aware_utc(now or now_utc())
    if not turn_lease_expired(
        turn,
        now=stamp,
        lease_seconds=lease_seconds,
    ):
        return False
    turn.status = "accepted"
    turn.trace_id = str(owner)[:64]
    turn.error_code = None
    renew_turn_lease(
        turn,
        owner=owner,
        now=stamp,
        lease_seconds=lease_seconds,
        count_attempt=True,
    )
    return True


def transition_turn(
    turn: ChatTurn,
    status: str,
    *,
    trace_id: str | None = None,
    evidence_status: str | None = None,
    retrieval_executed: bool | None = None,
    error_code: str | None = None,
    answer_content: str | None = None,
    answer_sources: list | None = None,
    search_snapshot: dict | None = None,
    tokens: int | None = None,
    assistant_message_id: uuid.UUID | None = None,
    now: datetime | None = None,
) -> ChatTurn:
    """Apply and validate one state transition in memory.

    The database caller must still use a row lock/CAS when two workers can
    update the same turn.  This pure function is intentionally deterministic so
    unit tests can exercise the protocol without PostgreSQL.
    """

    status = str(status).strip().casefold()
    if status not in TURN_STATUSES:
        raise ValueError(f"未知的 turn 状态: {status}")
    current = str(getattr(turn, "status", "accepted") or "accepted").casefold()
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if status not in allowed:
        raise ValueError(f"非法 turn 状态迁移: {current} -> {status}")
    turn.status = status
    if trace_id is not None:
        turn.trace_id = str(trace_id)[:64]
    if evidence_status is not None:
        # A state transition is a persistence boundary shared by API and
        # recovery paths.  Accept the one rolling legacy spelling on input,
        # but never write it back; unknown producer values fail closed to the
        # canonical infrastructure-error status.
        turn.evidence_status = (
            canonical_evidence_status(evidence_status) or "error"
        )[:32]
    if retrieval_executed is not None:
        turn.retrieval_executed = bool(retrieval_executed)
    if error_code is not None:
        turn.error_code = str(error_code)[:64]
    if answer_content is not None:
        turn.answer_content = answer_content
    if answer_sources is not None:
        turn.answer_sources = answer_sources
    if search_snapshot is not None:
        turn.search_snapshot = search_snapshot
    if tokens is not None:
        turn.tokens = int(tokens)
    if assistant_message_id is not None:
        turn.assistant_message_id = assistant_message_id
    stamp = now or now_utc()
    turn.updated_at = stamp
    if status == "generated":
        turn.generated_at = stamp
    if status == "completed":
        turn.completed_at = stamp
    if status in {"generated", "completed", "persist_failed", "failed", "cancelled"}:
        turn.lease_owner = None
        turn.lease_expires_at = None
    return turn


def turn_duration_ms(turn: ChatTurn) -> int | None:
    """Return completed end-to-end turn duration without trusting client clocks."""

    started_at = getattr(turn, "created_at", None)
    completed_at = getattr(turn, "completed_at", None)
    if not isinstance(started_at, datetime) or not isinstance(completed_at, datetime):
        return None
    return max(0, round((completed_at - started_at).total_seconds() * 1000))


def message_turn_metadata(turn: ChatTurn, *, status: str | None = None) -> dict[str, Any]:
    """Build nullable Message fields from the current turn snapshot."""

    effective_status = status or turn.status
    return {
        "turn_id": turn.id,
        "request_id": turn.request_id,
        "turn_status": effective_status,
        "trace_id": turn.trace_id,
        "evidence_status": turn.evidence_status,
        "retrieval_executed": turn.retrieval_executed,
        "error_code": turn.error_code,
        "delivery_status": "delivered" if effective_status == "completed" else "pending",
        "persistence_status": (
            "completed"
            if effective_status == "completed"
            else ("failed" if effective_status == "persist_failed" else "pending")
        ),
        "duration_ms": turn_duration_ms(turn),
        "search_snapshot": getattr(turn, "search_snapshot", None),
    }


async def find_turn(
    session: Any,
    conversation_id: uuid.UUID,
    request_id: str,
    *,
    for_update: bool = False,
) -> ChatTurn | None:
    """Load a turn; fake sessions without ``execute`` simply report no row."""

    execute = getattr(session, "execute", None)
    if not callable(execute):
        return None
    statement = select(ChatTurn).where(
        ChatTurn.conversation_id == conversation_id,
        ChatTurn.request_id == request_id,
    )
    if for_update:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    result = await execute(statement)
    scalar_one_or_none = getattr(result, "scalar_one_or_none", None)
    if callable(scalar_one_or_none):
        return scalar_one_or_none()
    scalars = getattr(result, "scalars", None)
    if callable(scalars):
        return scalars().first()
    return None


async def find_turn_for_user(
    session: Any,
    user_id: uuid.UUID,
    request_id: str,
    *,
    for_update: bool = False,
) -> ChatTurn | None:
    """Find the globally stable request for a user before creating a chat."""

    execute = getattr(session, "execute", None)
    if not callable(execute):
        return None
    statement = select(ChatTurn).where(
        ChatTurn.user_id == user_id,
        ChatTurn.request_id == request_id,
    )
    if for_update:
        statement = statement.with_for_update().execution_options(populate_existing=True)
    result = await execute(statement)
    scalar_one_or_none = getattr(result, "scalar_one_or_none", None)
    if callable(scalar_one_or_none):
        return scalar_one_or_none()
    scalars = getattr(result, "scalars", None)
    if callable(scalars):
        return scalars().first()
    return None


async def reserve_turn(
    session: Any,
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID | None,
    request_id: str,
    turn_id: uuid.UUID | None,
    question: str,
    trace_id: str,
    knowledge_base_ids: object = (),
    search_config: object = None,
    pending_route_revision: int = 0,
    pending_state_id: str | None = None,
) -> tuple[ChatTurn, bool]:
    """Get or create a conversation-scoped turn.

    Returns ``(turn, created)``.  A real database enforces the unique key; on
    an insert race we rollback and re-read the winner.  The fallback path keeps
    lightweight unit-test sessions usable without weakening production
    constraints.
    """

    digest = question_digest(question)
    request_context = build_turn_request_context(
        question=question,
        conversation_id=conversation_id,
        knowledge_base_ids=knowledge_base_ids,
        search_config=search_config,
        pending_route_revision=pending_route_revision,
        pending_state_id=pending_state_id,
    )
    fingerprint = request_context_fingerprint(request_context)
    requested_turn_id = turn_id
    if user_id is not None:
        user_existing = await find_turn_for_user(session, user_id, request_id)
        if user_existing is not None:
            assert_turn_request_matches(user_existing, request_context)
            if requested_turn_id is not None and str(user_existing.id) != str(requested_turn_id):
                raise TurnRequestConflict("同一 request_id 对应的 turn_id 不一致")
            return user_existing, False
    existing = await find_turn(session, conversation_id, request_id)
    if existing is not None:
        assert_turn_request_matches(existing, request_context)
        if requested_turn_id is not None and str(existing.id) != str(requested_turn_id):
            raise TurnRequestConflict("同一 request_id 对应的 turn_id 不一致")
        return existing, False

    turn_id = turn_id or uuid.uuid4()

    turn = ChatTurn(
        id=turn_id,
        conversation_id=conversation_id,
        user_id=user_id,
        request_id=request_id,
        question_hash=digest,
        request_fingerprint=fingerprint,
        request_context=request_context,
        trace_id=trace_id,
        status="accepted",
        execution_attempts=1,
    )
    renew_turn_lease(turn, owner=trace_id)
    add = getattr(session, "add", None)
    if not callable(add):
        raise RuntimeError("数据库会话不支持保存 chat turn")
    add(turn)
    flush = getattr(session, "flush", None)
    if callable(flush):
        try:
            await flush()
        except IntegrityError:
            rollback = getattr(session, "rollback", None)
            if callable(rollback):
                await rollback()
            winner = (
                await find_turn_for_user(session, user_id, request_id)
                if user_id is not None
                else None
            )
            if winner is None:
                winner = await find_turn(session, conversation_id, request_id)
            if winner is None:
                raise TurnReservationRace("chat turn 唯一键竞争后无法读取胜者")
            try:
                assert_turn_request_matches(winner, request_context)
            except TurnRequestConflict:
                raise TurnRequestConflict("同一 request_id 的请求参数不一致")
            if requested_turn_id is not None and str(winner.id) != str(requested_turn_id):
                raise TurnRequestConflict("同一 request_id 的请求参数不一致")
            return winner, False
    return turn, True


async def commit_with_retry(
    session: Any,
    *,
    attempts: int = MAX_PERSIST_ATTEMPTS,
    delay_seconds: float = 0.0,
    reapply: Callable[[Any], Awaitable[Any]] | None = None,
) -> Any:
    """Commit pre-applied work, rebuilding it after every rollback.

    SQLAlchemy expires/reverts pending ORM changes on ``rollback()``.  Retrying
    a bare ``commit()`` would therefore report success while saving nothing.
    Multiple attempts are enabled only when the caller supplies an idempotent
    ``reapply`` operation that reloads and reconstructs the transaction.
    """

    attempts = max(1, int(attempts)) if reapply is not None else 1
    last_error: BaseException | None = None
    reapplied_result: Any = None
    for index in range(attempts):
        try:
            if index > 0:
                assert reapply is not None
                reapplied_result = await reapply(session)
            await session.commit()
            return reapplied_result
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # persistence failures are surfaced to caller
            last_error = exc
            rollback = getattr(session, "rollback", None)
            if callable(rollback):
                try:
                    await rollback()
                except Exception:
                    pass
            if index + 1 < attempts and delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


async def run_transaction_with_retry(
    session_factory: Callable[[], Any],
    operation: Callable[[Any], Awaitable[_T]],
    *,
    attempts: int = MAX_PERSIST_ATTEMPTS,
    delay_seconds: float = 0.0,
) -> _T:
    """Run an operation in a fresh session for each commit attempt."""

    attempts = max(1, int(attempts))
    last_error: BaseException | None = None
    for index in range(attempts):
        async with session_factory() as session:
            try:
                result = await operation(session)
                await session.commit()
                return result
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                last_error = exc
                rollback = getattr(session, "rollback", None)
                if callable(rollback):
                    try:
                        await rollback()
                    except Exception:
                        pass
        if index + 1 < attempts and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


__all__ = [
    "TURN_STATUSES",
    "TERMINAL_TURN_STATUSES",
    "RECOVERABLE_TURN_STATUSES",
    "MAX_PERSIST_ATTEMPTS",
    "TurnRequestConflict",
    "TurnReservationRace",
    "normalize_request_id",
    "normalize_turn_id",
    "question_digest",
    "build_turn_request_context",
    "request_context_fingerprint",
    "request_fingerprint",
    "assert_turn_request_matches",
    "DEFAULT_TURN_LEASE_SECONDS",
    "turn_lease_expired",
    "renew_turn_lease",
    "reclaim_stale_turn",
    "transition_turn",
    "message_turn_metadata",
    "find_turn",
    "find_turn_for_user",
    "reserve_turn",
    "commit_with_retry",
    "run_transaction_with_retry",
]
