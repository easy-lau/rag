import asyncio
import logging
from pathlib import Path

from core.logging_config import configure_application_logging, stream_in_conversation_log


def test_development_logging_creates_a_run_file_and_records_root_logs(tmp_path: Path):
    log_path = configure_application_logging(
        app_env="development",
        log_level=logging.INFO,
        development_log_dir=str(tmp_path),
    )

    assert log_path is not None
    logging.getLogger("test.development_logging").info("development log check")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert log_path.parent == tmp_path.resolve()
    assert "development log check" in log_path.read_text(encoding="utf-8")


def test_development_logging_groups_stream_records_by_conversation(tmp_path: Path):
    system_path = configure_application_logging(
        app_env="development",
        log_level=logging.INFO,
        development_log_dir=str(tmp_path),
    )
    conversation_id = "f0d356cf-2438-4b3a-9061-bf8f19484394"

    async def stream():
        logging.getLogger("test.conversation_logging").info("first turn")
        yield "first"
        logging.getLogger("test.conversation_logging").info("second turn")
        yield "second"

    async def consume():
        return [
            item
            async for item in stream_in_conversation_log(
                stream(), conversation_id=conversation_id
            )
        ]

    assert asyncio.run(consume()) == ["first", "second"]
    conversation_path = tmp_path / f"conversation-{conversation_id}.log"
    assert conversation_path.exists()
    assert "first turn" in conversation_path.read_text(encoding="utf-8")
    assert "second turn" in conversation_path.read_text(encoding="utf-8")
    assert system_path is not None
    assert (
        not system_path.exists()
        or "first turn" not in system_path.read_text(encoding="utf-8")
    )


def test_production_logging_does_not_create_a_local_file(tmp_path: Path):
    log_path = configure_application_logging(
        app_env="production",
        log_level=logging.INFO,
        development_log_dir=str(tmp_path),
    )

    assert log_path is None
    assert list(tmp_path.iterdir()) == []
