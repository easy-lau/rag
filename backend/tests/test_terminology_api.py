"""Policy and structural tests for the KB-owned terminology registry.

The registry is deliberately tested without a developer database: PostgreSQL
DDL checks prove the foreign-key boundary while focused service tests prove the
state/revision and authorization rules that must not depend on a UI path.
"""

from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, patch

from fastapi import HTTPException
from fastapi.routing import APIRoute
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from api import knowledge, terminology
from core.permissions import (
    DOC_READ,
    KB_READ,
    TERMINOLOGY_MANAGE,
    TERMINOLOGY_READ,
    normalize_assignable_capabilities,
)
from core.terminology_registry import (
    RegistryReadConsistencyError,
    assert_active_canonical_invariant,
    build_replayable_change_payload,
    ensure_document_binding_authorized,
    read_stable_registry_view,
)
from models.db_models import (
    Document,
    TerminologyConcept,
    TerminologyRegistryRevision,
    TerminologyRegistryState,
    TerminologyScopeBinding,
    TerminologyTerm,
)
from models.schemas import (
    KnowledgeBaseCreate,
    TerminologyScopeBindingUpdate,
    TerminologyTermCreate,
)


class TerminologyRegistryApiTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_routes_keep_backend_capability_and_kb_scope_dependencies(self) -> None:
        def route(method: str, path: str) -> APIRoute:
            return next(
                item
                for item in terminology.router.routes
                if isinstance(item, APIRoute)
                and item.path == path
                and method in item.methods
            )

        def permission_keys(item: APIRoute) -> set[str]:
            keys: set[str] = set()

            def visit(dependant) -> None:
                for child in dependant.dependencies:
                    closure = getattr(child.call, "__closure__", None) or ()
                    for cell in closure:
                        if isinstance(cell.cell_contents, str) and ":" in cell.cell_contents:
                            keys.add(cell.cell_contents)
                    visit(child)

            visit(item.dependant)
            return keys

        self.assertEqual(
            permission_keys(route("GET", "/knowledge/{kb_id}/terminology")),
            {TERMINOLOGY_READ},
        )
        for method, path in (
            ("POST", "/knowledge/{kb_id}/terminology/concepts"),
            ("PUT", "/knowledge/{kb_id}/terminology/concepts/{concept_id}"),
            ("POST", "/knowledge/{kb_id}/terminology/bindings"),
            ("PUT", "/knowledge/{kb_id}/terminology/bindings/{binding_id}"),
        ):
            with self.subTest(method=method, path=path):
                self.assertEqual(permission_keys(route(method, path)), {TERMINOLOGY_MANAGE})

    def test_registry_models_make_concepts_terms_and_bindings_kb_owned(self) -> None:
        concept_ddl = str(CreateTable(TerminologyConcept.__table__).compile(
            dialect=postgresql.dialect()
        ))
        term_ddl = str(CreateTable(TerminologyTerm.__table__).compile(
            dialect=postgresql.dialect()
        ))
        binding_ddl = str(CreateTable(TerminologyScopeBinding.__table__).compile(
            dialect=postgresql.dialect()
        ))
        binding_indexes = "\n".join(
            str(CreateIndex(index).compile(dialect=postgresql.dialect()))
            for index in TerminologyScopeBinding.__table__.indexes
        )
        document_ddl = str(CreateTable(Document.__table__).compile(
            dialect=postgresql.dialect()
        ))

        self.assertIn("kb_id UUID NOT NULL", concept_ddl)
        self.assertIn(
            "UNIQUE (id, kb_id)",
            concept_ddl,
        )
        self.assertIn("UNIQUE (kb_id, code)", concept_ddl)
        self.assertNotIn("UNIQUE (code)", concept_ddl)
        self.assertIn(
            "FOREIGN KEY(concept_id, kb_id) REFERENCES terminology_concepts (id, kb_id)",
            term_ddl,
        )
        self.assertIn(
            "FOREIGN KEY(concept_id, kb_id) REFERENCES terminology_concepts (id, kb_id)",
            binding_ddl,
        )
        self.assertIn(
            "FOREIGN KEY(document_id, kb_id) REFERENCES documents (id, kb_id)",
            binding_ddl,
        )
        self.assertIn("CONSTRAINT uq_documents_id_kb_id UNIQUE (id, kb_id)", document_ddl)
        self.assertIn("CREATE UNIQUE INDEX uq_terminology_scope_bindings_identity", binding_indexes)
        self.assertIn("COALESCE(document_id", binding_indexes)

    def test_registry_revision_is_partitioned_by_kb_without_global_singleton(self) -> None:
        state_ddl = str(CreateTable(TerminologyRegistryState.__table__).compile(
            dialect=postgresql.dialect()
        ))
        revision_ddl = str(CreateTable(TerminologyRegistryRevision.__table__).compile(
            dialect=postgresql.dialect()
        ))
        self.assertIn("kb_id UUID NOT NULL", state_ddl)
        self.assertIn("PRIMARY KEY (kb_id)", state_ddl)
        self.assertNotIn("id INTEGER", state_ddl)
        self.assertIn("UNIQUE (kb_id, revision)", revision_ddl)
        self.assertIn("CHECK (revision >= 0)", state_ddl)
        self.assertIn("CHECK (revision > 0)", revision_ddl)

    def test_orm_relationships_keep_kb_in_their_join_condition(self) -> None:
        for model, relation in (
            (TerminologyConcept, "terms"),
            (TerminologyConcept, "scope_bindings"),
            (TerminologyTerm, "concept"),
            (TerminologyScopeBinding, "concept"),
        ):
            with self.subTest(model=model.__name__, relation=relation):
                join = str(inspect(model).relationships[relation].primaryjoin)
                self.assertIn("kb_id", join)
                self.assertIn("concept_id", join)

    def test_terminology_permissions_are_scoped_and_management_is_transitive(self) -> None:
        self.assertEqual(
            normalize_assignable_capabilities([TERMINOLOGY_MANAGE]),
            {TERMINOLOGY_MANAGE, TERMINOLOGY_READ, KB_READ},
        )

    def test_terms_are_exactly_normalized_without_creating_semantic_aliases(self) -> None:
        term, normalized = terminology._normalize_reviewed_term(
            " 餐　补 ", field="术语"
        )
        self.assertEqual(term, "餐 补")
        self.assertEqual(normalized, "餐补")
        with self.assertRaisesRegex(HTTPException, "至少应包含两个字符"):
            terminology._normalize_reviewed_term("A", field="术语")

    def test_scope_keys_share_the_contract_normalizer_and_reject_empty_widening(self) -> None:
        payload = TerminologyScopeBindingUpdate.model_validate({
            "scope_product_key": " 钉钉　",
            "scope_version_key": " 8.2.75 ",
        })
        self.assertEqual(
            terminology._scope_values(payload, only_supplied=True),
            {"scope_product_key": "钉钉", "scope_version_key": "8.2.75"},
        )
        with self.assertRaisesRegex(HTTPException, "不能为空"):
            terminology._normalize_optional_scope("  ", field="scope_product_key")

    def test_initial_terms_always_include_a_strict_canonical_form(self) -> None:
        rows = terminology._merge_initial_terms(
            canonical_term="餐饮补贴",
            supplied=[
                TerminologyTermCreate(term="餐补", match_mode="strict_equivalent"),
                TerminologyTermCreate(term="餐饮补", match_mode="retrieval_only"),
            ],
        )
        self.assertEqual(rows[0][0], "餐饮补贴")
        self.assertEqual(rows[0][2], "strict_equivalent")
        self.assertEqual(len(rows), 3)
        with self.assertRaisesRegex(HTTPException, "冲突"):
            terminology._merge_initial_terms(
                canonical_term="餐饮补贴",
                supplied=[
                    TerminologyTermCreate(
                        term="餐饮补贴", match_mode="retrieval_only"
                    )
                ],
            )

    def test_reactivation_requires_an_active_strict_canonical_term(self) -> None:
        concept = TerminologyConcept(
            id=uuid.uuid4(),
            kb_id=uuid.uuid4(),
            code="meal_allowance",
            canonical_term="餐饮补贴",
            is_active=True,
        )
        canonical = TerminologyTerm(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=concept.kb_id,
            term="餐饮补贴",
            normalized_term="餐饮补贴",
            match_mode="strict_equivalent",
            is_active=True,
        )
        concept.terms = [canonical]
        assert_active_canonical_invariant(concept)

        # This represents a concept that was inactive while its canonical form
        # was weakened, then an administrator attempts to reactivate it.
        canonical.match_mode = "retrieval_only"
        with self.assertRaisesRegex(HTTPException, "规范术语"):
            assert_active_canonical_invariant(concept)

    async def test_reader_retries_until_state_graph_state_is_stable(self) -> None:
        revisions = iter((7, 8, 8, 8))
        graphs = iter(("stale-graph", "revision-eight-graph"))

        async def read_revision() -> int:
            return next(revisions)

        async def read_graph() -> str:
            return next(graphs)

        revision, graph = await read_stable_registry_view(
            read_revision=read_revision,
            read_graph=read_graph,
            max_attempts=3,
        )
        self.assertEqual(revision, 8)
        self.assertEqual(graph, "revision-eight-graph")

    async def test_reader_fails_closed_when_a_stable_registry_view_cannot_be_obtained(self) -> None:
        async def read_revision() -> int:
            read_revision.value += 1
            return read_revision.value

        read_revision.value = 0

        async def read_graph() -> str:
            return "moving"

        with self.assertRaises(RegistryReadConsistencyError):
            await read_stable_registry_view(
                read_revision=read_revision,
                read_graph=read_graph,
                max_attempts=2,
            )

    def test_registry_event_keeps_complete_replayable_before_after_payload(self) -> None:
        kb_id = uuid.uuid4()
        before = {
            "concept": {"id": "concept-1", "canonical_term": "餐补"},
            "terms": [{"id": "term-1", "match_mode": "retrieval_only"}],
            "bindings": [{"id": "binding-1", "is_active": False}],
        }
        after = {
            "concept": {"id": "concept-1", "canonical_term": "餐饮补贴"},
            "terms": [{"id": "term-1", "match_mode": "strict_equivalent"}],
            "bindings": [{"id": "binding-1", "is_active": True}],
        }
        payload = build_replayable_change_payload(
            action="terminology.concept.update",
            kb_id=kb_id,
            before=before,
            after=after,
        )
        self.assertEqual(payload["schema_version"], "terminology_event.v1")
        self.assertEqual(payload["kb_id"], str(kb_id))
        self.assertEqual(payload["before"], before)
        self.assertEqual(payload["after"], after)
        before["concept"]["canonical_term"] = "被外部篡改"
        self.assertEqual(payload["before"]["concept"]["canonical_term"], "餐补")

    async def test_mutation_service_keeps_event_audit_and_returned_graph_at_one_revision(self) -> None:
        class MutationDb:
            def __init__(self) -> None:
                self.added = []
                self.flush_count = 0
                self.commit_count = 0

            def add(self, item) -> None:
                self.added.append(item)

            async def flush(self) -> None:
                self.flush_count += 1

            async def commit(self) -> None:
                self.commit_count += 1

            async def rollback(self) -> None:  # pragma: no cover - failure path only
                raise AssertionError("successful mutation must not roll back")

        kb_id = uuid.uuid4()
        concept_id = uuid.uuid4()
        binding_id = uuid.uuid4()
        timestamp = datetime.now(timezone.utc)
        concept = TerminologyConcept(
            id=concept_id,
            kb_id=kb_id,
            code="meal_allowance",
            canonical_term="餐饮补贴",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
        term = TerminologyTerm(
            id=uuid.uuid4(),
            concept_id=concept_id,
            kb_id=kb_id,
            term="餐饮补贴",
            normalized_term="餐饮补贴",
            match_mode="strict_equivalent",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
        binding = TerminologyScopeBinding(
            id=binding_id,
            concept_id=concept_id,
            kb_id=kb_id,
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.terms = [term]
        concept.scope_bindings = [binding]
        state = TerminologyRegistryState(kb_id=kb_id, revision=4, updated_at=timestamp)
        db = MutationDb()
        audit = Mock()
        user = SimpleNamespace(id=uuid.uuid4())
        service = terminology.TerminologyRegistryMutationService(
            db=db,
            user=user,
            audit=audit,
            kb_id=kb_id,
        )
        before = {"concept": None, "terms": [], "bindings": []}
        after = {
            "concept": {"id": str(concept_id), "kb_id": str(kb_id)},
            "terms": [{"id": str(term.id), "kb_id": str(kb_id)}],
            "bindings": [{"id": str(binding_id), "kb_id": str(kb_id)}],
        }
        with patch.object(service, "_graph_rows", new=AsyncMock(return_value=[concept])):
            result = await service._record_and_commit(
                state=state,
                action="terminology.concept.create",
                target_kind="concept",
                target_id=concept_id,
                target_name=concept.code,
                before=before,
                after=after,
                conflict_detail="冲突",
            )

        event = next(item for item in db.added if isinstance(item, TerminologyRegistryRevision))
        self.assertEqual(state.revision, 5)
        self.assertEqual(event.kb_id, kb_id)
        self.assertEqual(event.revision, 5)
        self.assertEqual(event.change_payload["before"], before)
        self.assertEqual(event.change_payload["after"], after)
        self.assertEqual(result.registry_revision, 5)
        self.assertEqual(result.registry.registry_revision, 5)
        self.assertEqual(result.concept.id, concept_id)
        self.assertEqual(result.registry.bindings[0].id, binding_id)
        self.assertEqual(db.flush_count, 1)
        self.assertEqual(db.commit_count, 1)
        audit.log.assert_called_once()
        self.assertEqual(audit.log.call_args.kwargs["detail"]["registry_revision"], 5)
        self.assertEqual(audit.log.call_args.kwargs["detail"]["change_payload"], event.change_payload)

    async def test_service_reactivation_path_rejects_weakened_canonical_term(self) -> None:
        class RollbackDb:
            def __init__(self) -> None:
                self.rollback_count = 0

            async def rollback(self) -> None:
                self.rollback_count += 1

        kb_id = uuid.uuid4()
        timestamp = datetime.now(timezone.utc)
        concept = TerminologyConcept(
            id=uuid.uuid4(),
            kb_id=kb_id,
            code="meal_allowance",
            canonical_term="餐饮补贴",
            is_active=False,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.terms = [TerminologyTerm(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            term="餐饮补贴",
            normalized_term="餐饮补贴",
            match_mode="retrieval_only",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )]
        db = RollbackDb()
        service = terminology.TerminologyRegistryMutationService(
            db=db,
            user=SimpleNamespace(id=uuid.uuid4()),
            audit=Mock(),
            kb_id=kb_id,
        )
        service._state = AsyncMock(return_value=TerminologyRegistryState(kb_id=kb_id, revision=3))
        service._concept = AsyncMock(return_value=concept)
        service._record_and_commit = AsyncMock()

        from models.schemas import TerminologyConceptUpdate

        with self.assertRaisesRegex(HTTPException, "规范术语"):
            await service.update_concept(
                concept_id=concept.id,
                payload=TerminologyConceptUpdate(is_active=True),
            )
        self.assertEqual(db.rollback_count, 1)
        service._record_and_commit.assert_not_awaited()

    async def test_service_can_change_canonical_term_while_reactivating_without_weakening_it(self) -> None:
        class MutationDb:
            async def flush(self) -> None:
                return None

            async def rollback(self) -> None:  # pragma: no cover - success path
                raise AssertionError("valid combined transition must not roll back")

        kb_id = uuid.uuid4()
        timestamp = datetime.now(timezone.utc)
        concept = TerminologyConcept(
            id=uuid.uuid4(),
            kb_id=kb_id,
            code="meal_allowance",
            canonical_term="旧餐补",
            is_active=False,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.terms = [TerminologyTerm(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            term="旧餐补",
            normalized_term="旧餐补",
            match_mode="strict_equivalent",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )]
        service = terminology.TerminologyRegistryMutationService(
            db=MutationDb(),
            user=SimpleNamespace(id=uuid.uuid4()),
            audit=Mock(),
            kb_id=kb_id,
        )
        service._state = AsyncMock(return_value=TerminologyRegistryState(kb_id=kb_id, revision=3))
        service._concept = AsyncMock(return_value=concept)
        service._record_and_commit = AsyncMock(return_value="saved")

        from models.schemas import TerminologyConceptUpdate

        result = await service.update_concept(
            concept_id=concept.id,
            payload=TerminologyConceptUpdate(
                canonical_term="餐饮补贴",
                is_active=True,
            ),
        )
        self.assertEqual(result, "saved")
        self.assertTrue(concept.is_active)
        canonical = next(
            term for term in concept.terms if term.normalized_term == "餐饮补贴"
        )
        self.assertTrue(canonical.is_active)
        self.assertEqual(canonical.match_mode, "strict_equivalent")

    async def test_service_binding_activation_rechecks_canonical_invariant(self) -> None:
        class RollbackDb:
            def __init__(self) -> None:
                self.rollback_count = 0

            async def rollback(self) -> None:
                self.rollback_count += 1

        kb_id = uuid.uuid4()
        timestamp = datetime.now(timezone.utc)
        concept = TerminologyConcept(
            id=uuid.uuid4(),
            kb_id=kb_id,
            code="meal_allowance",
            canonical_term="餐饮补贴",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.terms = [TerminologyTerm(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            term="餐饮补贴",
            normalized_term="餐饮补贴",
            match_mode="retrieval_only",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )]
        binding = TerminologyScopeBinding(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            is_active=False,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.scope_bindings = [binding]
        db = RollbackDb()
        service = terminology.TerminologyRegistryMutationService(
            db=db,
            user=SimpleNamespace(id=uuid.uuid4()),
            audit=Mock(),
            kb_id=kb_id,
        )
        service._state = AsyncMock(return_value=TerminologyRegistryState(kb_id=kb_id, revision=3))
        service._binding_concept = AsyncMock(return_value=(concept, binding))
        service._record_and_commit = AsyncMock()

        from models.schemas import TerminologyScopeBindingUpdate

        with self.assertRaisesRegex(HTTPException, "规范术语"):
            await service.update_binding(
                binding_id=binding.id,
                payload=TerminologyScopeBindingUpdate(is_active=True),
            )
        self.assertEqual(db.rollback_count, 1)
        service._record_and_commit.assert_not_awaited()

    async def test_document_scoped_binding_requires_doc_read_and_kb_scope(self) -> None:
        kb_id = uuid.uuid4()
        denied = SimpleNamespace(is_superadmin=False, permissions=[TERMINOLOGY_MANAGE])
        with self.assertRaisesRegex(HTTPException, "文档查看权限"):
            await ensure_document_binding_authorized(
                user=denied,
                db=object(),
                kb_id=kb_id,
            )

        allowed = SimpleNamespace(
            is_superadmin=False,
            permissions=[TERMINOLOGY_MANAGE, DOC_READ],
        )
        with patch(
            "core.terminology_registry.ensure_kb_access",
            new=AsyncMock(),
        ) as ensure_scope:
            await ensure_document_binding_authorized(
                user=allowed,
                db=object(),
                kb_id=kb_id,
            )
        ensure_scope.assert_awaited_once_with(allowed, kb_id, ANY)

    def test_document_scoped_bindings_are_not_returned_without_doc_read(self) -> None:
        kb_id = uuid.uuid4()
        timestamp = datetime.now(timezone.utc)
        concept = TerminologyConcept(
            id=uuid.uuid4(),
            kb_id=kb_id,
            code="meal_allowance",
            canonical_term="餐饮补贴",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.terms = [TerminologyTerm(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            term="餐饮补贴",
            normalized_term="餐饮补贴",
            match_mode="strict_equivalent",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )]
        global_binding = TerminologyScopeBinding(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            document_id=None,
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
        document_binding = TerminologyScopeBinding(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            document_id=uuid.uuid4(),
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.scope_bindings = [global_binding, document_binding]
        service = terminology.TerminologyRegistryMutationService(
            db=object(),
            user=SimpleNamespace(id=uuid.uuid4(), is_superadmin=False, permissions=[]),
            audit=None,
            kb_id=kb_id,
        )
        registry = service._registry_from_graph(
            revision=2,
            concepts=[concept],
            include_inactive=True,
            include_document_scoped=False,
        )
        self.assertEqual([item.id for item in registry.bindings], [global_binding.id])
        self.assertEqual([item.id for item in registry.concepts], [concept.id])

    async def test_document_scoped_concept_cannot_be_mutated_indirectly_without_doc_read(self) -> None:
        class RollbackDb:
            def __init__(self) -> None:
                self.rollback_count = 0

            async def rollback(self) -> None:
                self.rollback_count += 1

        kb_id = uuid.uuid4()
        timestamp = datetime.now(timezone.utc)
        concept = TerminologyConcept(
            id=uuid.uuid4(),
            kb_id=kb_id,
            code="meal_allowance",
            canonical_term="餐饮补贴",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
        concept.terms = [TerminologyTerm(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            term="餐饮补贴",
            normalized_term="餐饮补贴",
            match_mode="strict_equivalent",
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )]
        concept.scope_bindings = [TerminologyScopeBinding(
            id=uuid.uuid4(),
            concept_id=concept.id,
            kb_id=kb_id,
            document_id=uuid.uuid4(),
            is_active=True,
            created_at=timestamp,
            updated_at=timestamp,
        )]
        db = RollbackDb()
        service = terminology.TerminologyRegistryMutationService(
            db=db,
            user=SimpleNamespace(id=uuid.uuid4(), is_superadmin=False, permissions=[]),
            audit=Mock(),
            kb_id=kb_id,
        )
        service._state = AsyncMock(return_value=TerminologyRegistryState(kb_id=kb_id, revision=3))
        service._concept = AsyncMock(return_value=concept)

        with self.assertRaisesRegex(HTTPException, "文档查看权限"):
            await service.create_term(
                concept_id=concept.id,
                payload=TerminologyTermCreate(term="餐补"),
            )
        self.assertEqual(db.rollback_count, 1)

    async def test_knowledge_base_creation_initializes_its_own_registry_state(self) -> None:
        class KnowledgeDb:
            def __init__(self) -> None:
                self.added = []
                self.commit_count = 0

            def add(self, item) -> None:
                self.added.append(item)

            async def flush(self) -> None:
                self.added[0].id = uuid.uuid4()

            async def commit(self) -> None:
                self.commit_count += 1

            async def refresh(self, _item) -> None:
                return None

        db = KnowledgeDb()
        user = SimpleNamespace(id=uuid.uuid4(), username="owner", display_name=None)
        audit = Mock()
        result = await knowledge.create_knowledge_base(
            payload=KnowledgeBaseCreate(name="术语隔离测试库"),
            db=db,
            audit=audit,
            user=user,
        )
        state = next(item for item in db.added if isinstance(item, TerminologyRegistryState))
        self.assertEqual(state.kb_id, result.id)
        self.assertEqual(state.revision, 0)
        self.assertEqual(state.updated_by, user.id)
        self.assertEqual(db.commit_count, 1)
        audit.log.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
