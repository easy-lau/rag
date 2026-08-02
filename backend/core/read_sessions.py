"""Ownership boundary for optional, short-lived database read work.

Chat requests keep a long-lived ``AsyncSession`` because it owns durable
conversation, turn, and message state.  Retrieval enrichments are different:
they are read-only, optional, and may legitimately degrade.  PostgreSQL marks
the *entire* transaction aborted after a failed statement, so an optional read
must never share that durable request transaction when a session factory is
available.

The factory is deliberately injectable.  Production passes ``AsyncSessionLocal``
and gets an owned connection/transaction per read operation; compatibility
callers without a factory retain their historical serial borrowed-session
behaviour.  The latter is intentionally not auto-rolled back here because the
borrower owns its transaction and may contain writes.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)

# ``AsyncSession`` and ``@asynccontextmanager`` factories both satisfy this
# protocol at runtime.  Keep the result type broad so focused test doubles and
# alternate SQLAlchemy factories remain supported without weakening ownership.
ReadSessionFactory = Callable[[], Any]


@asynccontextmanager
async def isolated_read_session(
    *,
    request_db: AsyncSession,
    session_factory: ReadSessionFactory | None,
) -> AsyncIterator[AsyncSession]:
    """Yield an owned read session, or the explicit serial compatibility fallback.

    When ``session_factory`` is supplied, every exit path rolls back the owned
    session before its context closes.  This releases a read-only transaction
    on success and, critically, clears an aborted transaction after a failed
    optional query.  It never calls ``rollback`` on ``request_db``.
    """

    if session_factory is None:
        # Compatibility callers remain serial and retain ownership of their
        # request session.  Do not add an implicit rollback here: it could
        # discard a conversation/turn write owned by the caller.
        yield request_db
        return

    async with session_factory() as read_db:
        try:
            yield read_db
        finally:
            rollback = getattr(read_db, "rollback", None)
            if callable(rollback):
                try:
                    await rollback()
                except Exception as exc:
                    # The context manager still owns close/connection cleanup.
                    # Do not mask the original retrieval/database failure.
                    logger.warning(
                        "[RAG read session] owned rollback failed error=%s",
                        type(exc).__name__,
                    )


__all__ = ["ReadSessionFactory", "isolated_read_session"]
