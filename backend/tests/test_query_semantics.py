import json
import unittest

from core.query_analysis_compiler import compile_query_analysis_plan
from core.query_analysis_contract import (
    QUERY_ANALYSIS_SCHEMA_VERSION,
    parse_query_analysis,
)
from core.query_analysis_validation import validate_query_analysis_for_execution
from core.query_semantics import (
    KnowledgeRequestSemantics,
    RouteClarificationContinuation,
    document_catalog_request_for_question,
    document_catalog_surface_operation,
    resolve_turn_semantics,
)
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2


def _ref(source: str, span: str, *, turn_key: str = "current") -> dict[str, object]:
    start = source.index(span)
    return {
        "turn_key": turn_key,
        "start": start,
        "end": start + len(span),
        "span": span,
    }


def _generic_baseline(question: str) -> QueryPlanV2:
    return QueryPlanV2(
        original_query=question,
        answer_shape="fact",
        retrieval_queries=(question,),
        requirements=(AnswerRequirementV2(
            id="r1",
            description=question,
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        ),),
        confidence=0.7,
        source="fallback",
        reason="route_authorized_single_fact_baseline",
    )


class ResolvedTurnSemanticsTests(unittest.TestCase):
    def test_document_catalog_contract_and_prefetch_surface_are_generic(self):
        request = KnowledgeRequestSemantics(
            resource="document_catalog",
            operation="group",
            group_by="status",
            status_filter="any",
        )

        self.assertTrue(request.is_catalog_operation)
        self.assertEqual(
            document_catalog_surface_operation("按状态分别统计这些文档"),
            "group",
        )
        self.assertEqual(
            document_catalog_surface_operation("当前知识库有多少篇文章"),
            "count",
        )
        self.assertEqual(
            document_catalog_surface_operation("列出文档名称和状态"),
            "list",
        )
        self.assertIsNone(document_catalog_surface_operation("如何配置登录密码"))

    def test_clarification_continuation_separates_task_from_retrieval_rendering(self):
        continuation = RouteClarificationContinuation(
            schema_version="route_clarification_continuation.v1",
            original_query="我现在有哪些文章",
            current_answer="知识库里面的全部文章",
        )
        self.assertEqual(continuation.semantic_query, "我现在有哪些文章")
        self.assertEqual(continuation.answers, ("知识库里面的全部文章",))
        self.assertIn("我现在有哪些文章", continuation.canonical_retrieval_query)
        self.assertIn("知识库里面的全部文章", continuation.canonical_retrieval_query)
        self.assertNotIn("；", continuation.canonical_retrieval_query)
        self.assertEqual(
            document_catalog_request_for_question(
                continuation.canonical_retrieval_query
            ).operation,
            "list",
        )
        filtered = document_catalog_request_for_question("列出关于报销制度的文章")
        self.assertEqual(filtered.status_filter, "any")
        self.assertEqual(filtered.filter_terms, ("报销制度",))
        failed = document_catalog_request_for_question("列出失败的文章")
        self.assertEqual(failed.status_filter, "failed")

    def test_contextual_single_target_keeps_history_as_a_source_ref(self):
        current = "餐补呢"
        previous = "普通员工的住宿标准是多少"
        payload = {
            "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
            "relation": "followup",
            "self_contained": False,
            "context_turn_keys": ["t1"],
            "answer_candidates": [{
                "id": "a1",
                "target_source_ref": _ref(current, "餐补"),
                "qualifier_source_refs": [_ref(previous, "普通员工", turn_key="t1")],
                "bridge_candidate_ids": [],
            }],
            "bridge_candidates": [],
            "confidence": 0.95,
            "diagnostic": "当前问句省略主体，继承历史限定词。",
        }
        analysis = parse_query_analysis(
            json.dumps(payload, ensure_ascii=False),
            current_question=current,
            context_user_inputs={"t1": previous},
        )

        semantics = resolve_turn_semantics(analysis, current_question=current)

        self.assertFalse(semantics.self_contained)
        self.assertEqual(semantics.selected_context_turn_keys, ("t1",))
        self.assertEqual(semantics.request_kind, "single_fact")
        self.assertEqual(semantics.canonical_retrieval_queries, ("普通员工 餐补",))
        self.assertEqual(semantics.canonical_retrieval_query, "普通员工 餐补")
        self.assertEqual(
            semantics.answer_units[0].qualifier_source_refs[0].turn_key,
            "t1",
        )

        # The previous local follow-up heuristic may be false for ``餐补呢``.
        # A strict source-anchored contract must nevertheless be able to
        # replace only the generic current-turn baseline, without joining the
        # two user messages into a new sentence.
        baseline = _generic_baseline(current)
        validation = validate_query_analysis_for_execution(
            analysis,
            baseline_plan=baseline,
            current_question=current,
            deterministic_is_followup=False,
            allowed_context_turn_keys=("t1",),
        )
        self.assertTrue(validation.accepted)
        self.assertTrue(validation.replacement_authorized)
        compiled = compile_query_analysis_plan(
            analysis,
            execution_validation=validation,
            baseline_plan=baseline,
            current_question=current,
        )
        answer = next(
            item for item in compiled.plan.requirements if item.role == "answer"
        )
        self.assertEqual(answer.description, "普通员工 餐补")
        self.assertEqual(compiled.semantics.canonical_retrieval_query, "普通员工 餐补")
        self.assertIn("普通员工 餐补", compiled.plan.retrieval_queries)

    def test_multi_target_turn_has_independent_source_only_queries(self):
        question = "普通员工的住宿标准、餐补和出差补贴分别是多少"
        qualifier = _ref(question, "普通员工")
        payload = {
            "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
            "relation": "new",
            "self_contained": True,
            "context_turn_keys": [],
            "answer_candidates": [
                {
                    "id": f"a{index}",
                    "target_source_ref": _ref(question, target),
                    "qualifier_source_refs": [qualifier],
                    "bridge_candidate_ids": [],
                }
                for index, target in enumerate(("住宿标准", "餐补", "出差补贴"), start=1)
            ],
            "bridge_candidates": [],
            "confidence": 0.96,
            "diagnostic": "三个并列答案目标。",
        }
        analysis = parse_query_analysis(
            json.dumps(payload, ensure_ascii=False),
            current_question=question,
        )

        semantics = resolve_turn_semantics(analysis, current_question=question)

        self.assertEqual(semantics.request_kind, "finite_enumeration")
        self.assertEqual(semantics.answer_shape, "multi_part")
        self.assertEqual(
            semantics.canonical_retrieval_queries,
            ("普通员工 住宿标准", "普通员工 餐补", "普通员工 出差补贴"),
        )

    def test_configuration_state_and_procedure_are_not_the_same_contract(self):
        cases = (
            ("登录用户名枚举要配置什么", "登录用户名枚举", "configuration_state"),
            ("如何配置登录用户名枚举", "登录用户名枚举", "configuration_procedure"),
            ("修改参数应该怎么办", "参数", "configuration_procedure"),
        )
        for question, target, expected_kind in cases:
            with self.subTest(question=question):
                payload = {
                    "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
                    "relation": "new",
                    "self_contained": True,
                    "context_turn_keys": [],
                    "answer_candidates": [{
                        "id": "a1",
                        "target_source_ref": _ref(question, target),
                        "qualifier_source_refs": [],
                        "bridge_candidate_ids": [],
                    }],
                    "bridge_candidates": [],
                    "confidence": 0.95,
                    "diagnostic": "单一配置请求。",
                }
                analysis = parse_query_analysis(
                    json.dumps(payload, ensure_ascii=False),
                    current_question=question,
                )
                semantics = resolve_turn_semantics(
                    analysis,
                    current_question=question,
                )
                self.assertEqual(semantics.request_kind, expected_kind)
                self.assertEqual(
                    semantics.answer_shape,
                    "process" if expected_kind == "configuration_procedure" else "fact",
                )

    def test_scoped_standard_is_not_misclassified_as_a_whole_policy(self):
        question = "普通员工的出差标准是什么"
        payload = {
            "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
            "relation": "new",
            "self_contained": True,
            "context_turn_keys": [],
            "answer_candidates": [{
                "id": "a1",
                "target_source_ref": _ref(question, "出差标准"),
                "qualifier_source_refs": [_ref(question, "普通员工")],
                "bridge_candidate_ids": [],
            }],
            "bridge_candidates": [],
            "confidence": 0.95,
            "diagnostic": "有明确适用主体的单项标准。",
        }
        analysis = parse_query_analysis(
            json.dumps(payload, ensure_ascii=False),
            current_question=question,
        )
        semantics = resolve_turn_semantics(analysis, current_question=question)
        self.assertEqual(semantics.request_kind, "single_fact")
        self.assertEqual(semantics.answer_shape, "fact")


if __name__ == "__main__":
    unittest.main()
