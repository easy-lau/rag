import unittest
import uuid

from core.query_constraints import QueryConstraints, extract_query_constraints
from core.rag_v2.answer_policy import (
    decide_answer_policy,
    manage_authorized_candidates,
)
from core.rag_v2.context import EvidenceContext
from core.rag_v2.contracts import (
    AnswerRequirementV2,
    EvidenceBundle,
    EvidenceItem,
    EvidenceState,
    QueryPlanV2,
)
from core.rag_v2.evidence import FinalizedVisibleEvidence


class AnswerPolicyTests(unittest.TestCase):
    def _item(self, *, authorized=True, chunk_id=None, filename="通用配置.md"):
        return EvidenceItem(
            chunk_id=chunk_id or str(uuid.uuid4()),
            doc_id=str(self.doc_id),
            kb_id=str(self.kb_id),
            content="feature.enabled=true",
            authorized=authorized,
            constraint_status="neutral",
            metadata={"filename": filename},
        )

    def setUp(self):
        self.kb_id = uuid.uuid4()
        self.doc_id = uuid.uuid4()
        self.requirement = AnswerRequirementV2(
            id="answer_1",
            description="配置内容",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
            applicability_scope=QueryConstraints(),
        )
        self.plan = QueryPlanV2(
            original_query="查看配置内容",
            answer_shape="list",
            retrieval_queries=("配置内容",),
            requirements=(self.requirement,),
            confidence=0.73,
            source="model",
        )

    def _incomplete_finalized(self, item=None):
        bundle = EvidenceBundle(
            state=EvidenceState(
                availability="ok",
                confidence="retrieved",
                completeness="partial",
                reasons=("coverage_incomplete",),
            ),
            items=((item,) if item is not None else ()),
            context_item_ids=(),
            answer_source_ids=(),
            missing_requirement_ids=("answer_1",),
        )
        return FinalizedVisibleEvidence(
            bundle=bundle,
            context=EvidenceContext(text=""),
            assessment=None,
            route_item_ids=(),
            generation_allowed=False,
        )

    def test_incomplete_authorized_candidate_requires_durable_confirmation(self):
        item = self._item()
        candidates = manage_authorized_candidates((item,))
        decision = decide_answer_policy(
            finalized=self._incomplete_finalized(item),
            candidates=candidates,
            plan=self.plan,
            evidence_status="insufficient_evidence",
        )

        self.assertEqual(candidates.retrieval_status, "authorized_candidates_found")
        self.assertEqual(candidates.chunk_count, 1)
        self.assertEqual(decision.action, "clarify")
        self.assertEqual(decision.answerability_status, "evidence_incomplete")
        self.assertEqual(decision.semantic_confidence, 0.73)
        self.assertIsNotNone(decision.clarification_contract)
        choice = decision.clarification_contract.choices[0]
        self.assertEqual(choice["doc_ids"], [str(self.doc_id)])
        self.assertNotIn("feature.enabled", str(decision.to_dict(public=True)))

    def test_document_grouping_retains_every_chunk(self):
        first = self._item(chunk_id=str(uuid.uuid4()))
        second = self._item(chunk_id=str(uuid.uuid4()))
        candidates = manage_authorized_candidates((first, second))

        self.assertEqual(candidates.document_count, 1)
        self.assertEqual(candidates.chunk_count, 2)
        self.assertEqual(
            candidates.documents[0].chunk_ids,
            (first.chunk_id, second.chunk_id),
        )

    def test_unauthorized_only_never_exposes_candidate_identity(self):
        item = self._item(authorized=False, filename="受限数据库密码.md")
        candidates = manage_authorized_candidates((item,))
        decision = decide_answer_policy(
            finalized=self._incomplete_finalized(),
            candidates=candidates,
            plan=self.plan,
            evidence_status="insufficient_evidence",
        )

        self.assertEqual(candidates.retrieval_status, "unauthorized_only")
        self.assertEqual(candidates.documents, ())
        self.assertIsNone(candidates.clarification_contract())
        self.assertNotIn("受限数据库密码", str(candidates))
        self.assertEqual(decision.to_dict(public=False)["retrieval_status"], "unauthorized_only")
        self.assertEqual(decision.to_dict(public=True)["retrieval_status"], "no_match")
        self.assertNotIn("受限数据库密码", str(decision.to_dict(public=True)))


class ProjectScopeGrammarTests(unittest.TestCase):
    def test_possessive_prefix_does_not_create_project_scope(self):
        scope = extract_query_constraints("看下的云枢7配置")
        self.assertIsNone(scope.project)
        self.assertFalse(scope.explicit_project)

    def test_only_explicit_project_grammar_creates_project_scope(self):
        scope = extract_query_constraints("甲方项目的CloudPivot 8.2配置")
        self.assertEqual(scope.project, "甲方")
        self.assertTrue(scope.explicit_project)

    def test_question_word_before_bare_project_is_not_scope(self):
        scope = extract_query_constraints("当前有哪些项目？")
        self.assertIsNone(scope.project)


if __name__ == "__main__":
    unittest.main()
