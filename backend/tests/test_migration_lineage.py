"""Alembic migration lineage regression tests."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Column


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

    def create_table(self, name, *items, **_kwargs) -> None:
        self.tables[name] = items

    def create_index(self, *_args, **_kwargs) -> None:
        return None

    def execute(self, statement) -> None:
        self.executed.append(str(statement))


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

    def test_alembic_has_single_0026_head(self) -> None:
        config = Config(str(BACKEND_DIR / "alembic.ini"))
        config.set_main_option(
            "script_location",
            str(BACKEND_DIR / "migrations"),
        )
        scripts = ScriptDirectory.from_config(config)

        self.assertEqual(scripts.get_heads(), ["0026"])
        self.assertEqual(scripts.get_revision("0026").down_revision, "0025")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
