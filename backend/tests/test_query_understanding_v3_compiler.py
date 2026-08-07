"""Contract tests for the source-catalog-backed V3 trusted compiler.

The tests deliberately exercise the compiler without the chat/pipeline
integration.  They prove that model output remains a selection of user spans,
while scope, bridge topology and the executable ledger remain backend-owned.
"""

from __future__ import annotations

import json
import unittest

from core.query_understanding_v3_catalog import SourceSpanCatalog
from core.query_understanding_v3_compiler import (
    BaselineFloor,
    compile_query_understanding,
    validate_query_understanding,
)
from core.query_understanding_v3_contract import (
    QueryUnderstandingV3ValidationError,
    parse_query_understanding,
)
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2


def _span_id(
    catalog: SourceSpanCatalog,
    text: str,
    *,
    occurrence: int = 0,
    source_key: str = "current",
) -> str:
    """Return one deterministic catalog entry by exact text/source."""

    matches = [
        entry
        for entry in catalog.entries
        if entry.source_key == source_key and entry.text == text
    ]
    if len(matches) <= occurrence:
        available = [
            (entry.source_key, entry.text)
            for entry in catalog.entries
        ]
        raise AssertionError(
            f"catalog has no {source_key} span {text!r}; available={available!r}"
        )
    return matches[occurrence].span_id


def _candidate(
    catalog: SourceSpanCatalog,
    *,
    targets: tuple[str, ...],
    qualifier_texts: tuple[str, ...] = (),
    answer_form: str = "fact",
) -> object:
    qualifier_span_ids = [_span_id(catalog, value) for value in qualifier_texts]
    payload = {
        "schema_version": "query_understanding.v3",
        "answer_candidates": [
            {
                "id": f"a{index}",
                "target_span_id": _span_id(catalog, target),
                "qualifier_span_ids": qualifier_span_ids,
            }
            for index, target in enumerate(targets, start=1)
        ],
        "knowledge_request": {
            "resource": "document_content",
            "operation": "answer",
            "filter_span_ids": [],
            "group_by": "none",
            "status_filter": "any",
            "result_handles": [],
            "answer_form": answer_form,
        },
    }
    return parse_query_understanding(json.dumps(payload, ensure_ascii=False), catalog=catalog)


def _runnable_fallback(question: str) -> QueryPlanV2:
    return QueryPlanV2(
        original_query=question,
        answer_shape="fact",
        retrieval_queries=(question,),
        requirements=(
            AnswerRequirementV2(
                id="fallback_r1",
                description=question,
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
            ),
        ),
        confidence=0.5,
        source="fallback",
        reason="deterministic_fallback_only",
    )


def _not_ready_fallback(question: str) -> QueryPlanV2:
    return QueryPlanV2(
        original_query=question,
        answer_shape="unknown",
        retrieval_queries=(),
        requirements=(),
        confidence=0.0,
        source="fallback",
        reason="deterministic_fallback_unresolved",
        needs_clarification=True,
        clarification_question="请补充问题中的必要限定条件。",
    )


class QueryUnderstandingV3CompilerTests(unittest.TestCase):
    def test_answer_form_compiles_across_topics_without_business_rules(self):
        cases = (
            ("员工想请假怎么办", "procedure", "process", "ordered_steps"),
            ("依赖安装失败后应该怎么处理", "procedure", "process", "ordered_steps"),
            ("云枢如何修改默认密码", "procedure", "process", "structured_collection"),
            ("介绍一下访问控制制度", "overview", "overview", "document_policy"),
            ("判断当前配置是否符合要求", "judgement", "judgement", "single_claim"),
        )
        for question, answer_form, answer_shape, coverage_contract in cases:
            with self.subTest(question=question):
                catalog = SourceSpanCatalog.build(current_question=question)
                understanding = _candidate(
                    catalog,
                    targets=(question,),
                    answer_form=answer_form,
                )
                compiled = compile_query_understanding(
                    catalog=catalog,
                    understanding=understanding,
                    baseline_floor=BaselineFloor(
                        current_question=question,
                        fallback_plan=_runnable_fallback(question),
                    ),
                )

                self.assertTrue(
                    compiled.validation.accepted,
                    compiled.validation.reason,
                )
                self.assertEqual(compiled.plan.answer_shape, answer_shape)
                answer = next(
                    item
                    for item in compiled.plan.requirements
                    if item.role == "answer"
                )
                self.assertEqual(
                    answer.effective_coverage_contract,
                    coverage_contract,
                )

    def test_surface_floor_upgrades_a_misclassified_solution_question(self):
        question = "员工想请假怎么办"
        catalog = SourceSpanCatalog.build(current_question=question)
        # Simulate a provider that returned a valid but overly broad fact form.
        understanding = _candidate(
            catalog,
            targets=(question,),
            answer_form="fact",
        )
        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=BaselineFloor(
                current_question=question,
                fallback_plan=_runnable_fallback(question),
            ),
        )
        self.assertEqual(compiled.plan.answer_shape, "process")
        answer = next(
            item for item in compiled.plan.requirements if item.role == "answer"
        )
        self.assertEqual(answer.effective_coverage_contract, "ordered_steps")

    def test_travel_multi_target_has_complete_coverage_and_shared_classification_augmentation(self):
        question = "普通员工的住宿标准、餐补和出差补贴这些分别是多少"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(
            catalog,
            targets=("住宿标准", "餐补", "出差补贴"),
            qualifier_texts=("普通员工",),
        )
        floor = BaselineFloor(
            current_question=question,
            fallback_plan=_runnable_fallback(question),
        )

        validation = validate_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )
        self.assertTrue(validation.accepted, validation.reason)
        self.assertEqual(validation.current_target_count, 3)

        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )

        self.assertFalse(compiled.used_fallback)
        self.assertEqual(compiled.plan.original_query, question)
        self.assertEqual(compiled.execution_bundle.mode, "ledgered")
        self.assertEqual(
            compiled.execution_bundle.task_graph.task_by_id["anchor_root"].query,
            question,
        )
        answers = [
            item for item in compiled.plan.requirements if item.role == "answer"
        ]
        bridges = [
            item for item in compiled.plan.requirements if item.role == "bridge"
        ]
        self.assertEqual([item.description for item in answers], [
            "普通员工 住宿标准",
            "普通员工 餐补",
            "普通员工 出差补贴",
        ])
        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0].description, "普通员工")
        self.assertEqual(bridges[0].bridge_kind, "classification")
        self.assertTrue(all(
            item.depends_on_requirement_ids == ()
            and item.augmentation_requirement_ids == (bridges[0].id,)
            for item in answers
        ))
        self.assertTrue(all(
            " ".join(catalog.resolve(span_id).text for span_id in (
                compiled.description_span_ids[requirement.id]
            )) == requirement.description
            for requirement in answers
        ))

    def test_single_meal_allowance_can_receive_classification_augmentation(self):
        question = "普通员工的餐补是多少"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(
            catalog,
            targets=("餐补",),
            qualifier_texts=("普通员工",),
        )
        floor = BaselineFloor(
            current_question=question,
            fallback_plan=_runnable_fallback(question),
        )

        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )

        self.assertTrue(compiled.validation.accepted, compiled.validation.reason)
        self.assertEqual(compiled.plan.answer_shape, "fact")
        answer = next(
            item for item in compiled.plan.requirements if item.role == "answer"
        )
        bridge = next(
            item for item in compiled.plan.requirements if item.role == "bridge"
        )
        self.assertEqual(answer.description, "普通员工 餐补")
        self.assertEqual(answer.depends_on_requirement_ids, ())
        self.assertEqual(answer.augmentation_requirement_ids, (bridge.id,))
        self.assertEqual(bridge.bridge_kind, "classification")

    def test_explicit_versions_are_compiled_as_separate_scope_partitions_not_a_merged_scope(self):
        question = "比较云枢6.0和云枢7.0的审批流程差异"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(catalog, targets=("审批流程",))
        # The deterministic fallback is intentionally unresolved.  A valid V3
        # candidate must still be able to build a runnable ledger, rather than
        # treating the fallback planner as a second semantic authority.
        floor = BaselineFloor(
            current_question=question,
            fallback_plan=_not_ready_fallback(question),
        )

        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )

        self.assertTrue(compiled.validation.accepted, compiled.validation.reason)
        self.assertFalse(compiled.used_fallback)
        self.assertEqual(compiled.plan.answer_shape, "comparison")
        answers = [
            item for item in compiled.plan.requirements if item.role == "answer"
        ]
        self.assertEqual(len(answers), 2)
        self.assertEqual(
            {item.applicability_scope.version for item in answers},
            {"6.0", "7.0"},
        )
        self.assertTrue(all(item.description == "审批流程" for item in answers))
        self.assertTrue(all(item.depends_on_requirement_ids == () for item in answers))
        self.assertTrue(all(
            len(item.applicability_scope.source_spans) >= 1
            for item in answers
        ))

    def test_sequential_version_targets_do_not_expand_to_a_cross_product(self):
        question = "ProductX6.0的安装要求和ProductX7.0的升级要求分别是什么"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(
            catalog,
            targets=("安装要求", "升级要求"),
        )
        floor = BaselineFloor(
            current_question=question,
            fallback_plan=_not_ready_fallback(question),
        )

        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )

        self.assertTrue(compiled.validation.accepted, compiled.validation.reason)
        answers = [
            item for item in compiled.plan.requirements if item.role == "answer"
        ]
        self.assertEqual(len(answers), 2)
        by_description = {
            item.description: item.applicability_scope.version for item in answers
        }
        self.assertEqual(by_description, {"安装要求": "6.0", "升级要求": "7.0"})

    def test_partial_multi_target_candidate_is_rejected_and_never_drops_a_current_target(self):
        question = "普通员工的住宿标准、餐补和出差补贴这些分别是多少"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(
            catalog,
            targets=("住宿标准", "餐补"),
            qualifier_texts=("普通员工",),
        )
        floor = BaselineFloor(
            current_question=question,
            fallback_plan=_runnable_fallback(question),
        )

        validation = validate_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )
        self.assertFalse(validation.accepted)
        self.assertEqual(validation.reason, "current_target_coverage_incomplete")

        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )
        self.assertTrue(compiled.used_fallback)
        self.assertIs(compiled.plan, floor.fallback_plan)

    def test_single_selected_target_cannot_hide_an_explicit_current_turn_enumeration(self):
        """Coverage is derived from the current sentence, not candidate count."""

        question = "普通员工的住宿标准、餐补和出差补贴这些分别是多少"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(
            catalog,
            targets=("餐补",),
            qualifier_texts=("普通员工",),
        )
        floor = BaselineFloor(
            current_question=question,
            fallback_plan=_runnable_fallback(question),
        )

        validation = validate_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )

        self.assertFalse(validation.accepted)
        self.assertEqual(validation.reason, "current_target_coverage_incomplete")

    def test_model_cannot_smuggle_scope_permission_or_proof_edge_fields(self):
        question = "普通员工的餐补是多少"
        catalog = SourceSpanCatalog.build(current_question=question)
        payload = {
            "schema_version": "query_understanding.v3",
            "answer_candidates": [{
                "id": "a1",
                "target_span_id": _span_id(catalog, "餐补"),
                "qualifier_span_ids": [_span_id(catalog, "普通员工")],
                "scope": {"version": "8.6"},
            }],
        }

        with self.assertRaises(QueryUnderstandingV3ValidationError):
            parse_query_understanding(
                json.dumps(payload, ensure_ascii=False),
                catalog=catalog,
            )

    def test_explicit_mapping_question_never_becomes_a_classification_augmentation(self):
        """The answer is the mapping itself, not a value that needs it first."""

        question = "普通员工对应什么职级"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(
            catalog,
            targets=("职级",),
            qualifier_texts=("普通员工",),
        )
        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=BaselineFloor(
                current_question=question,
                fallback_plan=_runnable_fallback(question),
            ),
        )

        self.assertTrue(compiled.validation.accepted, compiled.validation.reason)
        self.assertFalse(any(
            item.role == "bridge" for item in compiled.plan.requirements
        ))
        answer = next(item for item in compiled.plan.requirements if item.role == "answer")
        self.assertEqual(answer.augmentation_requirement_ids, ())

    def test_named_entity_or_no_qualifier_never_creates_a_bridge_or_proof_edge(self):
        question = "供应商甲的风险等级是什么"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(
            catalog,
            targets=("风险等级",),
            qualifier_texts=("供应商甲",),
        )
        floor = BaselineFloor(
            current_question=question,
            fallback_plan=_runnable_fallback(question),
        )

        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )

        self.assertTrue(compiled.validation.accepted, compiled.validation.reason)
        self.assertFalse(any(
            item.role == "bridge" for item in compiled.plan.requirements
        ))
        answer = next(
            item for item in compiled.plan.requirements if item.role == "answer"
        )
        self.assertEqual(answer.depends_on_requirement_ids, ())
        self.assertEqual(answer.augmentation_requirement_ids, ())

        direct_question = "餐补是多少"
        direct_catalog = SourceSpanCatalog.build(current_question=direct_question)
        direct_compiled = compile_query_understanding(
            catalog=direct_catalog,
            understanding=_candidate(direct_catalog, targets=("餐补",)),
            baseline_floor=BaselineFloor(
                current_question=direct_question,
                fallback_plan=_runnable_fallback(direct_question),
            ),
        )
        self.assertTrue(direct_compiled.validation.accepted)
        self.assertFalse(any(
            item.role == "bridge" for item in direct_compiled.plan.requirements
        ))

    def test_hard_clarification_guard_cannot_be_overridden_by_a_model_candidate(self):
        question = "普通员工的餐补是多少"
        catalog = SourceSpanCatalog.build(current_question=question)
        understanding = _candidate(
            catalog,
            targets=("餐补",),
            qualifier_texts=("普通员工",),
        )
        floor = BaselineFloor(
            current_question=question,
            fallback_plan=_not_ready_fallback(question),
            hard_clarification_reason="route_scope_conflict",
        )

        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=floor,
        )

        self.assertFalse(compiled.validation.accepted)
        self.assertEqual(compiled.validation.reason, "hard_clarification_guard")
        self.assertTrue(compiled.used_fallback)
        self.assertEqual(compiled.execution_bundle.mode, "not_ready")

    def test_model_cannot_split_history_scope_from_its_entity_qualifier(self):
        """A V3 source selection cannot erase t1's product/version envelope."""

        question = "餐补呢"
        catalog = SourceSpanCatalog.build(
            current_question=question,
            route_context=(
                {
                    "candidate_key": "t1",
                    "user_input": "普通员工在云枢8.6中的餐饮补贴是多少",
                },
            ),
        )
        payload = {
            "schema_version": "query_understanding.v3",
            "answer_candidates": [{
                "id": "a1",
                "target_span_id": _span_id(catalog, "餐补"),
                "qualifier_span_ids": [
                    _span_id(catalog, "普通员工", source_key="t1"),
                ],
            }],
        }
        understanding = parse_query_understanding(
            json.dumps(payload, ensure_ascii=False),
            catalog=catalog,
        )
        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=BaselineFloor(
                current_question=question,
                fallback_plan=_runnable_fallback(question),
            ),
        )

        self.assertTrue(compiled.used_fallback)
        self.assertFalse(compiled.validation.accepted)
        self.assertEqual(
            compiled.validation.reason,
            "historical_context_not_inheritable_explicit_scope",
        )
        self.assertTrue(compiled.validation.requires_clarification)

    def test_model_can_only_select_the_exact_inheritable_history_entity_span(self):
        """Whole historical questions are not a hidden qualifier vocabulary."""

        question = "餐补呢"
        previous = "普通员工的住宿标准是多少"
        catalog = SourceSpanCatalog.build(
            current_question=question,
            route_context=(
                {"candidate_key": "t1", "user_input": previous},
            ),
        )
        payload = {
            "schema_version": "query_understanding.v3",
            "answer_candidates": [{
                "id": "a1",
                "target_span_id": _span_id(catalog, "餐补"),
                "qualifier_span_ids": [
                    _span_id(catalog, previous, source_key="t1"),
                ],
            }],
        }
        understanding = parse_query_understanding(
            json.dumps(payload, ensure_ascii=False),
            catalog=catalog,
        )
        compiled = compile_query_understanding(
            catalog=catalog,
            understanding=understanding,
            baseline_floor=BaselineFloor(
                current_question=question,
                fallback_plan=_runnable_fallback(question),
            ),
        )

        self.assertFalse(compiled.validation.accepted)
        self.assertEqual(
            compiled.validation.reason,
            "historical_context_not_exact_entity_span",
        )
        self.assertTrue(compiled.validation.requires_clarification)


if __name__ == "__main__":
    unittest.main()
