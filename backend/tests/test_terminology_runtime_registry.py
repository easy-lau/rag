"""Regression tests for the authorised terminology registry runtime adapter."""

from __future__ import annotations

import uuid
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace

from core.rag_v2.contracts import AnswerRequirementV2
from core.terminology_runtime import build_runtime_terminology_resolution
from core.terminology_runtime_registry import (
    _runtime_bindings_from_concepts,
    load_terminology_runtime_resolution,
)


KB_A = "11111111-1111-1111-1111-111111111111"
KB_B = "22222222-2222-2222-2222-222222222222"


def _term(
    *,
    identifier: str,
    term: str,
    mode: str = "strict_equivalent",
    active: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=identifier,
        term=term,
        match_mode=mode,
        is_active=active,
    )


def _concept(
    *,
    identifier: str,
    kb_id: str,
    code: str,
    canonical: str,
    document_id: str | None,
    source_mode: str = "strict_equivalent",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=identifier,
        kb_id=kb_id,
        code=code,
        canonical_term=canonical,
        is_active=True,
        terms=(
            _term(identifier=f"{identifier}_short", term="餐补", mode=source_mode),
            _term(identifier=f"{identifier}_full", term=canonical),
        ),
        scope_bindings=(SimpleNamespace(
            id=f"{identifier}_binding",
            kb_id=kb_id,
            document_id=document_id,
            scope_product_key=None,
            scope_version_key=None,
            scope_project_key=None,
            is_active=True,
        ),),
    )


def _requirement() -> AnswerRequirementV2:
    return AnswerRequirementV2(
        id="r1",
        description="普通员工餐补是多少",
        coverage_contract="single_claim",
        depends_on_requirement_ids=(),
        augmentation_requirement_ids=(),
    )


class RuntimeRegistryBindingTests(unittest.TestCase):
    def test_same_spelling_in_different_kbs_is_not_a_conflict_and_keeps_scope(self):
        requirement = _requirement()
        concepts = (
            _concept(
                identifier="concept_a",
                kb_id=KB_A,
                code="meal_a",
                canonical="餐饮补贴",
                document_id="doc_a",
            ),
            _concept(
                identifier="concept_b",
                kb_id=KB_B,
                code="meal_b",
                canonical="伙食补贴",
                document_id="doc_b",
            ),
        )
        bindings = _runtime_bindings_from_concepts(
            concepts=concepts,
            requirements=(requirement,),
            authorized_kb_ids=(KB_A, KB_B),
        )
        resolution = build_runtime_terminology_resolution(
            plan_fingerprint="a" * 64,
            scope_fingerprint="b" * 64,
            authorized_kb_ids=(KB_A, KB_B),
            registry_revisions={KB_A: 1, KB_B: 2},
            bindings=bindings,
        )

        variants = resolution.retrieval_variants(
            requirement=requirement,
            maximum_aliases=4,
        )

        self.assertEqual(resolution.status, "resolved")
        self.assertEqual(resolution.ambiguous_source_key_count, 0)
        self.assertEqual(len(variants), 2)
        self.assertEqual(
            {(item.kb_ids, item.document_ids) for item in variants},
            {((KB_A,), ("doc_a",)), ((KB_B,), ("doc_b",))},
        )
        self.assertTrue(all(len(item.kb_ids) == 1 for item in variants))

    def test_unauthorised_registry_row_is_dropped_before_pure_resolution(self):
        bindings = _runtime_bindings_from_concepts(
            concepts=(
                _concept(
                    identifier="concept_a",
                    kb_id=KB_A,
                    code="meal_a",
                    canonical="餐饮补贴",
                    document_id=None,
                ),
                _concept(
                    identifier="concept_b",
                    kb_id=KB_B,
                    code="meal_b",
                    canonical="伙食补贴",
                    document_id=None,
                ),
            ),
            requirements=(_requirement(),),
            authorized_kb_ids=(KB_A,),
        )

        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0].kb_id, KB_A)


class RuntimeRegistryFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_database_read_failure_degrades_aliases_without_throwing(self):
        class FailingSession:
            async def execute(self, *_args, **_kwargs):
                raise RuntimeError("database includes business-sensitive values")

        resolution = await load_terminology_runtime_resolution(
            db=FailingSession(),
            requirements=(_requirement(),),
            retrieval_kb_ids=(uuid.uuid4(),),
        )

        self.assertEqual(resolution.status, "degraded")
        self.assertEqual(
            resolution.retrieval_variants(
                requirement=_requirement(),
                maximum_aliases=3,
            ),
            (),
        )
        self.assertNotIn("business-sensitive", str(resolution.trace_summary()))

    async def test_failed_registry_read_does_not_touch_borrowed_request_session(self):
        """A missing registry table is optional, but its transaction is not.

        The production incident was caused by swallowing this read error on the
        request session.  PostgreSQL then rejected every later SAVEPOINT and
        source-refresh query.  The registry must own an isolated read session
        whenever a factory is supplied.
        """

        class RequestSession:
            def __init__(self):
                self.execute_calls = 0
                self.rollback_calls = 0

            async def execute(self, *_args, **_kwargs):
                self.execute_calls += 1
                raise AssertionError("request session must not read registry")

            async def rollback(self):
                self.rollback_calls += 1

        class MissingRegistryReadSession:
            def __init__(self):
                self.rollback_calls = 0

            async def execute(self, *_args, **_kwargs):
                raise RuntimeError("relation terminology_registry_state does not exist")

            async def rollback(self):
                self.rollback_calls += 1

        request_session = RequestSession()
        owned_sessions: list[MissingRegistryReadSession] = []

        @asynccontextmanager
        async def read_session_factory():
            session = MissingRegistryReadSession()
            owned_sessions.append(session)
            yield session

        resolution = await load_terminology_runtime_resolution(
            db=request_session,
            read_session_factory=read_session_factory,
            requirements=(_requirement(),),
            retrieval_kb_ids=(uuid.uuid4(),),
        )

        self.assertEqual(resolution.status, "degraded")
        self.assertEqual(request_session.execute_calls, 0)
        self.assertEqual(request_session.rollback_calls, 0)
        self.assertEqual(len(owned_sessions), 1)
        self.assertEqual(owned_sessions[0].rollback_calls, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
