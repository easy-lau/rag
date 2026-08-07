"""Alembic migration lineage regression tests."""

import importlib.util
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy import Column, ForeignKeyConstraint, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from models.db_models import (
    TerminologyConcept,
    TerminologyRegistryRevision,
    TerminologyRegistryState,
    TerminologyScopeBinding,
    TerminologyTerm,
)


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
        self.created_unique_constraints: list[tuple[str, str, tuple[str, ...]]] = []
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

    def create_unique_constraint(self, name: str, table_name: str, columns) -> None:
        self.created_unique_constraints.append((name, table_name, tuple(columns)))

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

    def test_active_task_state_migration_is_authorization_neutral(self) -> None:
        migration_34 = _load_migration(
            "0034_add_conversation_active_task_state.py"
        )
        recorder = _MigrationRecorder()

        with patch.object(migration_34, "op", recorder):
            migration_34.upgrade()

        columns = {
            (table_name, column.name): column
            for table_name, column in recorder.added_columns
        }
        state = columns[("conversations", "active_task_state")]
        self.assertIsInstance(state.type, JSONB)
        self.assertTrue(state.nullable)
        self.assertFalse(state.foreign_keys)
        revision = columns[("conversations", "active_task_revision")]
        self.assertIsInstance(revision.type, Integer)
        self.assertFalse(revision.nullable)
        self.assertEqual(str(revision.server_default.arg), "0")

    def test_alembic_has_single_0038_head(self) -> None:
        config = Config(str(BACKEND_DIR / "alembic.ini"))
        config.set_main_option(
            "script_location",
            str(BACKEND_DIR / "migrations"),
        )
        scripts = ScriptDirectory.from_config(config)

        self.assertEqual(scripts.get_heads(), ["0038"])
        self.assertEqual(scripts.get_revision("0032").down_revision, "0031")
        self.assertEqual(scripts.get_revision("0034").down_revision, "0033")
        self.assertEqual(scripts.get_revision("0035").down_revision, "0034")
        self.assertEqual(scripts.get_revision("0036").down_revision, "0035")
        self.assertEqual(scripts.get_revision("0037").down_revision, "0036")
        self.assertEqual(scripts.get_revision("0038").down_revision, "0037")

    def test_message_duration_migration_is_additive(self) -> None:
        migration_31 = _load_migration("0031_add_message_answer_duration.py")
        recorder = _MigrationRecorder()
        with patch.object(migration_31, "op", recorder):
            migration_31.upgrade()

        added = {
            column.name: column
            for table_name, column in recorder.added_columns
            if table_name == "messages"
        }
        self.assertIn("duration_ms", added)
        self.assertIsInstance(added["duration_ms"].type, Integer)

    def test_terminology_registry_migration_has_kb_owned_scope_proof_and_revision_seed(self) -> None:
        migration = _load_migration("0032_add_terminology_registry.py")
        recorder = _MigrationRecorder()
        with patch.object(migration, "op", recorder):
            migration.upgrade()

        self.assertIn(
            ("uq_documents_id_kb_id", "documents", ("id", "kb_id")),
            recorder.created_unique_constraints,
        )
        for table_name in (
            "terminology_concepts",
            "terminology_terms",
            "terminology_scope_bindings",
            "terminology_registry_state",
            "terminology_registry_revisions",
        ):
            with self.subTest(table_name=table_name):
                self.assertIn(table_name, recorder.tables)

        concept_columns = {
            item.name
            for item in recorder.tables["terminology_concepts"]
            if isinstance(item, Column)
        }
        self.assertTrue({"id", "kb_id", "code", "canonical_term"}.issubset(concept_columns))
        concept_constraints = recorder.tables["terminology_concepts"]
        self.assertTrue(any(
            isinstance(item, UniqueConstraint)
            and item.name == "uq_terminology_concepts_id_kb_id"
            for item in concept_constraints
        ))
        self.assertTrue(any(
            isinstance(item, UniqueConstraint)
            and item.name == "uq_terminology_concepts_kb_code"
            for item in concept_constraints
        ))

        term_columns = {
            item.name
            for item in recorder.tables["terminology_terms"]
            if isinstance(item, Column)
        }
        self.assertTrue({"concept_id", "kb_id", "term", "normalized_term", "match_mode"}.issubset(term_columns))
        term_constraints = recorder.tables["terminology_terms"]
        self.assertTrue(any(
            isinstance(item, UniqueConstraint)
            and item.name == "uq_terminology_terms_concept_kb_normalized"
            for item in term_constraints
        ))
        term_concept_fk = next(
            item for item in term_constraints
            if isinstance(item, ForeignKeyConstraint)
            and item.name == "fk_terminology_terms_concept_kb"
        )
        self.assertEqual(tuple(term_concept_fk.column_keys), ("concept_id", "kb_id"))
        self.assertEqual(
            tuple(element.target_fullname for element in term_concept_fk.elements),
            ("terminology_concepts.id", "terminology_concepts.kb_id"),
        )

        binding_constraints = recorder.tables["terminology_scope_bindings"]
        concept_fk = next(
            item for item in binding_constraints
            if isinstance(item, ForeignKeyConstraint)
            and item.name == "fk_terminology_scope_bindings_concept_kb"
        )
        self.assertEqual(tuple(concept_fk.column_keys), ("concept_id", "kb_id"))
        self.assertEqual(
            tuple(element.target_fullname for element in concept_fk.elements),
            ("terminology_concepts.id", "terminology_concepts.kb_id"),
        )
        composite_fk = next(
            item for item in binding_constraints
            if isinstance(item, ForeignKeyConstraint)
            and item.name == "fk_terminology_scope_bindings_document_kb"
        )
        self.assertEqual(tuple(composite_fk.column_keys), ("document_id", "kb_id"))
        self.assertEqual(
            tuple(element.target_fullname for element in composite_fk.elements),
            ("documents.id", "documents.kb_id"),
        )

        sql = "\n".join(recorder.executed)
        state_columns = {
            item.name
            for item in recorder.tables["terminology_registry_state"]
            if isinstance(item, Column)
        }
        self.assertTrue({"kb_id", "revision", "updated_at"}.issubset(state_columns))
        self.assertNotIn("id", state_columns)
        revision_columns = {
            item.name
            for item in recorder.tables["terminology_registry_revisions"]
            if isinstance(item, Column)
        }
        self.assertTrue({"kb_id", "revision", "change_payload"}.issubset(revision_columns))

        self.assertIn("INSERT INTO terminology_registry_state (kb_id, revision, updated_at)", sql)
        self.assertIn("SELECT id, 0, CURRENT_TIMESTAMP FROM knowledge_bases", sql)
        self.assertIn("ON CONFLICT (kb_id) DO NOTHING", sql)
        self.assertIn("CREATE UNIQUE INDEX uq_terminology_scope_bindings_identity", sql)
        self.assertIn("COALESCE(document_id", sql)

    def test_terminology_registry_migration_compiles_as_postgresql_ddl(self) -> None:
        """Compile 0032 itself without relying on a live developer database."""

        migration = _load_migration("0032_add_terminology_registry.py")
        output = io.StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql",
            opts={"as_sql": True, "output_buffer": output},
        )
        operations = Operations(context)
        with patch.object(migration, "op", operations):
            migration.upgrade()
        sql = output.getvalue()
        for fragment in (
            "ALTER TABLE documents ADD CONSTRAINT uq_documents_id_kb_id",
            "CREATE TABLE terminology_concepts",
            "kb_id UUID NOT NULL",
            "CREATE TABLE terminology_scope_bindings",
            "FOREIGN KEY(document_id, kb_id) REFERENCES documents (id, kb_id)",
            "FOREIGN KEY(concept_id, kb_id) REFERENCES terminology_concepts (id, kb_id)",
            "CREATE UNIQUE INDEX uq_terminology_scope_bindings_identity",
            "COALESCE(document_id, '00000000-0000-0000-0000-000000000000'::uuid)",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, sql)

    def test_conversation_repair_migration_compiles_as_postgresql_ddl(self) -> None:
        """0036 must compile offline; its JSON examples column cannot pass a raw Python list."""

        migration = _load_migration("0036_add_conversation_repair_intent.py")
        output = io.StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql",
            opts={
                "as_sql": True,
                "output_buffer": output,
                "literal_binds": True,
            },
        )
        operations = Operations(context)
        with patch.object(migration, "op", operations):
            migration.upgrade()
        sql = output.getvalue()
        for fragment in (
            "INSERT INTO intent_categories",
            "'conversation_repair'",
            "为什么要我选择",
            "'chat'",
            "90",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, sql)

    def test_result_reference_memory_migration_compiles_as_postgresql_ddl(self) -> None:
        """0037 must compile offline and stay additive on conversations."""

        migration = _load_migration("0037_add_conversation_result_reference_memory.py")
        output = io.StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql",
            opts={"as_sql": True, "output_buffer": output},
        )
        operations = Operations(context)
        with patch.object(migration, "op", operations):
            migration.upgrade()
        sql = output.getvalue()
        for fragment in (
            "ALTER TABLE conversations ADD COLUMN result_reference_memory JSONB",
            "result_reference_revision INTEGER",
            "DEFAULT 0",
            "NOT NULL",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, sql)

    def test_reference_correction_migration_compiles_as_postgresql_ddl(self) -> None:
        """0038 must compile offline; its JSON examples column needs op.inline_literal."""

        migration = _load_migration("0038_add_reference_correction_intent.py")
        output = io.StringIO()
        context = MigrationContext.configure(
            dialect_name="postgresql",
            opts={
                "as_sql": True,
                "output_buffer": output,
                "literal_binds": True,
            },
        )
        operations = Operations(context)
        with patch.object(migration, "op", operations):
            migration.upgrade()
        sql = output.getvalue()
        for fragment in (
            "INSERT INTO intent_categories",
            "'reference_correction'",
            "第四个不是《钉钉》吗",
            "'retrieve'",
            "95",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, sql)

    def test_terminology_orm_and_migration_keep_named_constraints_and_indexes_aligned(self) -> None:
        """Prevent a model-only rule from drifting away from rollout DDL."""

        migration = _load_migration("0032_add_terminology_registry.py")
        recorder = _MigrationRecorder()
        with patch.object(migration, "op", recorder):
            migration.upgrade()

        expected_constraints = {
            "terminology_concepts": {
                "ck_terminology_concepts_code_nonempty",
                "ck_terminology_concepts_canonical_term_nonempty",
                "uq_terminology_concepts_id_kb_id",
                "uq_terminology_concepts_kb_code",
            },
            "terminology_terms": {
                "ck_terminology_terms_match_mode",
                "ck_terminology_terms_term_nonempty",
                "ck_terminology_terms_normalized_term_nonempty",
                "uq_terminology_terms_concept_kb_normalized",
                "fk_terminology_terms_concept_kb",
            },
            "terminology_scope_bindings": {
                "ck_terminology_scope_bindings_product_key_nonempty",
                "ck_terminology_scope_bindings_version_key_nonempty",
                "ck_terminology_scope_bindings_project_key_nonempty",
                "fk_terminology_scope_bindings_concept_kb",
                "fk_terminology_scope_bindings_document_kb",
                "fk_terminology_scope_bindings_kb",
            },
            "terminology_registry_state": {
                "ck_terminology_registry_state_revision",
            },
            "terminology_registry_revisions": {
                "ck_terminology_registry_revisions_revision",
                "uq_terminology_registry_revisions_kb_revision",
            },
        }
        model_tables = {
            "terminology_concepts": TerminologyConcept.__table__,
            "terminology_terms": TerminologyTerm.__table__,
            "terminology_scope_bindings": TerminologyScopeBinding.__table__,
            "terminology_registry_state": TerminologyRegistryState.__table__,
            "terminology_registry_revisions": TerminologyRegistryRevision.__table__,
        }
        for table_name, expected in expected_constraints.items():
            with self.subTest(table_name=table_name):
                migration_names = {
                    item.name
                    for item in recorder.tables[table_name]
                    if getattr(item, "name", None)
                }
                model_names = {
                    item.name
                    for item in model_tables[table_name].constraints
                    if item.name
                }
                self.assertTrue(expected.issubset(migration_names))
                self.assertTrue(expected.issubset(model_names))

        expected_indexes = {
            "ix_terminology_concepts_kb_active",
            "ix_terminology_terms_kb_normalized_active",
            "ix_terminology_scope_bindings_kb_active",
            "ix_terminology_scope_bindings_document_active",
            "ix_terminology_registry_revisions_kb_created_at",
        }
        migration_indexes = {name for name, *_rest in recorder.created_indexes}
        model_indexes = {
            index.name
            for table in model_tables.values()
            for index in table.indexes
        }
        self.assertTrue(expected_indexes.issubset(migration_indexes))
        self.assertTrue(expected_indexes.issubset(model_indexes))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
