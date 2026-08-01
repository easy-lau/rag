"""Alembic migration lineage regression tests."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import JSONB


BACKEND_DIR = Path(__file__).resolve().parents[1]
VERSIONS_DIR = BACKEND_DIR / "migrations" / "versions"


def _load_migration(filename: str):
    path = VERSIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"无法加载迁移：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MigrationRecorder:
    def __init__(self) -> None:
        self.tables: dict[str, tuple] = {}
        self.executed: list[str] = []
        self.added_columns: list[tuple[str, Column]] = []
        self.created_indexes: list[tuple[str, str, tuple[str, ...], bool]] = []
        self.altered_columns: list[tuple[str, str, dict]] = []

    def create_table(self, name, *items, **_kwargs) -> None:
        self.tables[name] = items

    def add_column(self, table_name: str, column: Column) -> None:
        self.added_columns.append((table_name, column))

    def create_index(
        self,
        name: str,
        table_name: str,
        columns,
        *,
        unique: bool = False,
        **_kwargs,
    ) -> None:
        self.created_indexes.append(
            (name, table_name, tuple(columns), unique)
        )

    def execute(self, statement) -> None:
        self.executed.append(str(statement))

    def alter_column(self, table_name: str, column_name: str, **kwargs) -> None:
        self.altered_columns.append((table_name, column_name, kwargs))


class MigrationLineageTests(unittest.TestCase):
    def test_trace_integrity_fields_live_in_followup_migration(self) -> None:
        migration_25 = _load_migration("0025_add_rag_trace_storage.py")
        migration_26 = _load_migration("0026_add_rag_trace_integrity_fields.py")

        create_recorder = _MigrationRecorder()
        with patch.object(migration_25, "op", create_recorder):
            migration_25.upgrade()
        initial_columns = {
            item.name
            for item in create_recorder.tables["rag_trace_runs"]
            if isinstance(item, Column)
        }
        self.assertIn("event_count", initial_columns)
        self.assertNotIn("observed_event_count", initial_columns)
        self.assertNotIn("storage_omitted_event_count", initial_columns)
        self.assertNotIn("storage_truncated", initial_columns)

        alter_recorder = _MigrationRecorder()
        with patch.object(migration_26, "op", alter_recorder):
            migration_26.upgrade()
        sql = "\n".join(alter_recorder.executed)
        for column in (
            "observed_event_count",
            "storage_omitted_event_count",
            "storage_truncated",
        ):
            with self.subTest(column=column):
                self.assertIn(f"ADD COLUMN IF NOT EXISTS {column}", sql)
                self.assertIn(f"ALTER COLUMN {column} SET NOT NULL", sql)
                self.assertIn(f"ALTER COLUMN {column} DROP DEFAULT", sql)
        self.assertIn(
            "observed_event_count = COALESCE(observed_event_count, event_count)",
            sql,
        )
        self.assertIn("WHERE observed_event_count IS NULL", sql)

    def test_document_structure_indexes_match_bounded_lookup_queries(self) -> None:
        migration_27 = _load_migration(
            "0027_add_document_structure_indexes.py"
        )
        recorder = _MigrationRecorder()

        with patch.object(migration_27, "op", recorder):
            migration_27.upgrade()

        sql = "\n".join(recorder.executed)
        for index_name in (
            "ix_document_chunks_doc_chunk_index",
            "ix_document_chunks_section_key_position",
            "ix_document_chunks_heading_position",
            "ix_document_chunks_table_part_position",
        ):
            with self.subTest(index_name=index_name):
                self.assertIn(f"CREATE INDEX IF NOT EXISTS {index_name}", sql)
        self.assertIn("ON document_chunks (doc_id, chunk_index)", sql)
        self.assertIn("metadata->>'section_key'", sql)
        self.assertIn("metadata->>'heading'", sql)
        self.assertIn("metadata->>'table_id'", sql)
        self.assertIn("~ '^[0-9]{1,9}$'", sql)

    def test_intent_route_state_migration_is_additive_and_unbound(self) -> None:
        migration_28 = _load_migration("0028_add_intent_route_state.py")
        recorder = _MigrationRecorder()

        with patch.object(migration_28, "op", recorder):
            migration_28.upgrade()

        columns = {
            (table_name, column.name): column
            for table_name, column in recorder.added_columns
        }
        pending_state = columns[("conversations", "pending_route_state")]
        self.assertIsInstance(pending_state.type, JSONB)
        self.assertTrue(pending_state.nullable)

        revision = columns[("conversations", "route_state_revision")]
        self.assertIsInstance(revision.type, Integer)
        self.assertFalse(revision.nullable)
        self.assertEqual(str(revision.server_default.arg), "0")

        trace_id = columns[("intent_route_logs", "trace_id")]
        self.assertIsInstance(trace_id.type, String)
        self.assertEqual(trace_id.type.length, 64)
        self.assertTrue(trace_id.nullable)
        self.assertFalse(trace_id.foreign_keys)

        route_summary = columns[("intent_route_logs", "route_summary")]
        self.assertIsInstance(route_summary.type, JSONB)
        self.assertTrue(route_summary.nullable)
        self.assertIn(
            (
                "ix_intent_route_logs_trace_id",
                "intent_route_logs",
                ("trace_id",),
                False,
            ),
            recorder.created_indexes,
        )

    def test_intent_evidence_status_expands_for_clarification_outcome(self) -> None:
        migration_29 = _load_migration(
            "0029_expand_intent_evidence_status.py"
        )
        upgrade_recorder = _MigrationRecorder()
        with patch.object(migration_29, "op", upgrade_recorder):
            migration_29.upgrade()

        self.assertEqual(len(upgrade_recorder.altered_columns), 1)
        table_name, column_name, kwargs = upgrade_recorder.altered_columns[0]
        self.assertEqual(
            (table_name, column_name),
            ("intent_route_logs", "evidence_status"),
        )
        self.assertEqual(kwargs["existing_type"].length, 16)
        self.assertEqual(kwargs["type_"].length, 32)
        self.assertTrue(kwargs["existing_nullable"])

        downgrade_recorder = _MigrationRecorder()
        with patch.object(migration_29, "op", downgrade_recorder):
            migration_29.downgrade()
        self.assertIn(
            "char_length(evidence_status) > 16",
            downgrade_recorder.executed[0],
        )
        _, _, downgrade_kwargs = downgrade_recorder.altered_columns[0]
        self.assertEqual(downgrade_kwargs["existing_type"].length, 32)
        self.assertEqual(downgrade_kwargs["type_"].length, 16)

    def test_chat_turn_migration_adds_recovery_ledger_and_history_state(self) -> None:
        migration_30 = _load_migration("0030_add_chat_turn_persistence.py")
        recorder = _MigrationRecorder()
        with patch.object(migration_30, "op", recorder):
            migration_30.upgrade()

        turn_columns = {
            item.name: item
            for item in recorder.tables["chat_turns"]
            if isinstance(item, Column)
        }
        for name in (
            "request_id",
            "request_fingerprint",
            "request_context",
            "resume_context",
            "status",
            "lease_owner",
            "lease_expires_at",
            "execution_attempts",
            "trace_id",
            "evidence_status",
            "retrieval_executed",
            "error_code",
            "answer_content",
            "answer_sources",
            "search_snapshot",
            "assistant_message_id",
        ):
            self.assertIn(name, turn_columns)
        message_columns = {
            column.name
            for table_name, column in recorder.added_columns
            if table_name == "messages"
        }
        self.assertIn("turn_status", message_columns)
        self.assertIn("search_snapshot", message_columns)

    def test_alembic_has_single_0030_head(self) -> None:
        config = Config(str(BACKEND_DIR / "alembic.ini"))
        config.set_main_option(
            "script_location",
            str(BACKEND_DIR / "migrations"),
        )
        scripts = ScriptDirectory.from_config(config)

        self.assertEqual(scripts.get_heads(), ["0030"])
        self.assertEqual(scripts.get_revision("0030").down_revision, "0029")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
