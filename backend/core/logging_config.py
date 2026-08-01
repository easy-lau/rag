"""Application logging configuration with conversation-scoped development logs.

Development keeps ordinary process/startup output in one system file, while
every chat stream is routed to the stable file for its ``conversation_id``.
Production still writes only to stdout for the container logging driver.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import TypeVar


_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEVELOPMENT_ENVS = {"dev", "development", "local", "test"}
_CONVERSATION_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_conversation_log_id: ContextVar[str | None] = ContextVar(
    "conversation_log_id",
    default=None,
)
_StreamItem = TypeVar("_StreamItem")


def _normalized_conversation_id(value: object) -> str | None:
    candidate = str(value or "").strip()
    return candidate if _CONVERSATION_ID_RE.fullmatch(candidate) else None


class ConversationDevelopmentFileHandler(logging.Handler):
    """Append each record to its conversation file, or the system file.

    The handler opens a file only for the duration of one emit.  A chat can
    create an arbitrary number of conversations, so retaining a file
    descriptor per conversation would eventually exhaust the worker limit.
    Appending atomically per formatted record keeps the implementation bounded
    while preserving one stable file per conversation across requests/reloads.
    """

    def __init__(self, directory: Path, system_path: Path) -> None:
        super().__init__()
        self._directory = directory
        self._system_path = system_path

    def _path_for_record(self, record: logging.LogRecord) -> Path:
        # Structured trace records carry their conversation explicitly. This
        # also routes request-phase traces emitted before the SSE generator has
        # entered its context; ordinary pipeline logs use the ContextVar.
        conversation_id = _normalized_conversation_id(
            getattr(record, "conversation_id", None)
        ) or _normalized_conversation_id(_conversation_log_id.get())
        if conversation_id is None:
            return self._system_path
        return self._directory / f"conversation-{conversation_id}.log"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            path = self._path_for_record(record)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")
        except Exception:
            self.handleError(record)


@asynccontextmanager
async def conversation_log_context(
    conversation_id: object,
) -> AsyncGenerator[None, None]:
    """Bind all logs emitted by an async chat stream to one conversation."""

    normalized = _normalized_conversation_id(conversation_id)
    token = _conversation_log_id.set(normalized)
    try:
        yield
    finally:
        _conversation_log_id.reset(token)


async def stream_in_conversation_log(
    stream: AsyncIterable[_StreamItem],
    *,
    conversation_id: object,
) -> AsyncGenerator[_StreamItem, None]:
    """Yield an SSE stream with an isolated conversation log context."""

    async with conversation_log_context(conversation_id):
        async for item in stream:
            yield item


def configure_application_logging(
    *,
    app_env: str,
    log_level: int,
    development_log_dir: str,
) -> Path | None:
    """Configure console logs and development conversation log routing.

    Returns the system/process log path. Chat records are instead written to
    ``conversation-<conversation_id>.log`` while their SSE stream is active.
    """

    formatter = logging.Formatter(_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [console_handler]
    system_path: Path | None = None
    development_handler: logging.Handler | None = None

    if app_env.strip().lower() in _DEVELOPMENT_ENVS:
        log_directory = Path(development_log_dir).expanduser().resolve()
        log_directory.mkdir(parents=True, exist_ok=True)
        run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        system_path = log_directory / f"backend-system-{run_stamp}-pid{os.getpid()}.log"
        development_handler = ConversationDevelopmentFileHandler(
            log_directory,
            system_path,
        )
        development_handler.setFormatter(formatter)
        handlers.append(development_handler)

    # Importing ``main`` in a reload worker can occur after a previous logging
    # setup. Replace handlers deterministically so a line is never duplicated.
    logging.basicConfig(level=log_level, handlers=handlers, force=True)

    # Uvicorn's access logger owns its handler and otherwise bypasses root.
    # Reuse the same router so request logs emitted during a chat stream land in
    # the corresponding conversation file rather than a second process file.
    if development_handler is not None:
        for logger_name in ("uvicorn.access", "uvicorn.error"):
            target = logging.getLogger(logger_name)
            if development_handler not in target.handlers:
                target.addHandler(development_handler)

    return system_path
