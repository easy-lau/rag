import unittest

from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.task_execution import (
    TaskExecutionLedger,
    build_initial_retrieval_groups,
    build_retrieval_execution_schedule,
)
from core.rag_v2.task_graph import compile_retrieval_task_graph
from core.query_constraints import (
    ApplicabilityScope,
    ScopeCandidateRejection,
    ScopeSourceSpan,
)
from core.terminology_contracts import TerminologyBinding, TerminologyForm
from core.terminology_runtime import (
    RuntimeTerminologyBinding,
    TerminologyRuntimeResolution,
)


def _plan(requirements, *, question="普通员工的制度标准分别是多少"):
    return QueryPlanV2(
        original_query=question,
        answer_shape="multi_part" if len(requirements) > 1 else "fact",
        retrieval_queries=("旧数组位置不应参与执行",),
        requirements=tuple(requirements),
        confidence=0.95,
        source="local",
    )


def _answer(
    requirement_id,
    description,
    *,
    scope_version=None,
    dependencies=(),
    augmentations=None,
):
    kwargs = {}
    if augmentations is not None:
        kwargs["augmentation_requirement_ids"] = augmentations
    return AnswerRequirementV2(
        id=requirement_id,
        description=description,
        role="answer",
        importance="required",
        source="explicit",
        depends_on_requirement_ids=dependencies,
        scope_product="云枢" if scope_version else None,
        scope_version=scope_version,
        scope_explicit_version=bool(scope_version),
        **kwargs,
    )


def _bridge(requirement_id, subject="普通员工", *, kind="classification"):
    return AnswerRequirementV2(
        id=requirement_id,
        description=f"确认{subject}对应的适用分类",
        role="bridge",
        importance="helpful",
        source="inferred",
        bridge_subject=subject,
        bridge_kind=kind,
    )


def _candidate(
    chunk_id,
    content="有效正文",
    *,
    doc_id="doc-1",
    kb_id="kb-1",
    **extra,
):
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "chunk_index": 0,
        "content": content,
        **extra,
    }


def _scope_rejection(
    *,
    chunk_id="chunk-1",
    expected_scope_fingerprint="a" * 64,
    actual_identity_fingerprint="b" * 64,
    dimensions=("version",),
    reason_code="scope_mismatch_version",
):
    return ScopeCandidateRejection(
        kb_id="kb-1",
        doc_id="doc-1",
        chunk_id=chunk_id,
        expected_scope_fingerprint=expected_scope_fingerprint,
        actual_identity_fingerprint=actual_identity_fingerprint,
        mismatch_dimensions=dimensions,
        reason_code=reason_code,
    )


def _project_scope(project: str, *, start: int) -> ApplicabilityScope:
    return ApplicabilityScope(
        product="云枢",
        version="8.2.75",
        project=project,
        explicit_version=True,
        explicit_project=True,
        product_source=ScopeSourceSpan(
            dimension="product",
            start=start,
            end=start + 2,
            span="云枢",
        ),
        version_source=ScopeSourceSpan(
            dimension="version",
            start=start + 2,
            end=start + 8,
            span="8.2.75",
        ),
        project_source=ScopeSourceSpan(
            dimension="project",
            start=start + 8,
            end=start + 8 + len(project),
            span=project,
        ),
        extraction_reason="test_source_verified_project_scope",
    )


def _terminology_resolution() -> TerminologyRuntimeResolution:
    binding = TerminologyBinding(
        requirement_id="r1",
        concept_id="meal_allowance",
        concept_key="meal_allowance",
        display_name="餐饮补贴",
        source_term="餐补",
        source_relation_strength="strict_equivalent",
        query_forms=(
            TerminologyForm(
                term="餐补",
                rule_id="meal_short",
                relation_strength="strict_equivalent",
            ),
            TerminologyForm(
                term="餐饮补贴",
                rule_id="meal_full",
                relation_strength="strict_equivalent",
            ),
        ),
        evidence_forms=(
            TerminologyForm(
                term="餐补",
                rule_id="meal_short",
                relation_strength="strict_equivalent",
            ),
            TerminologyForm(
                term="餐饮补贴",
                rule_id="meal_full",
                relation_strength="strict_equivalent",
            ),
        ),
        scope_binding_ids=("binding_meal",),
    )
    return TerminologyRuntimeResolution(
        plan_fingerprint="a" * 64,
        scope_fingerprint="b" * 64,
        registry_revisions={"kb_a": 1},
        status="resolved",
        bindings=(RuntimeTerminologyBinding(
            binding=binding,
            kb_id="kb_a",
            document_id="doc_policy",
        ),),
        authorized_kb_ids=("kb_a",),
    )


class TaskExecutionLedgerTests(unittest.TestCase):
    def test_identical_query_with_different_project_scope_never_coalesces(self):
        first_scope = _project_scope("中青建安", start=0)
        second_scope = _project_scope("华东示范项目", start=20)
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工餐补标准是多少",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
                applicability_scope=first_scope,
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工餐补标准是多少",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
                applicability_scope=second_scope,
            ),
        )
        graph = compile_retrieval_task_graph(_plan(
            requirements,
            question="中青建安与华东示范项目的云枢8.2.75餐补",
        ))

        answer_groups = tuple(
            group
            for group in build_initial_retrieval_groups(graph)
            if any(task_id.startswith("answer_") for task_id in group.task_ids)
        )

        self.assertEqual(len(answer_groups), 2)
        self.assertEqual(
            {group.scope_project for group in answer_groups},
            {"中青建安", "华东示范项目"},
        )
        self.assertEqual(
            {group.scope_fingerprint for group in answer_groups},
            {first_scope.fingerprint, second_scope.fingerprint},
        )
        self.assertTrue(all(
            group.applicability_scope is not None
            and group.applicability_scope.project_source is not None
            for group in answer_groups
        ))

    def test_identical_query_with_identical_scope_coalesces_and_keeps_object(self):
        scope = _project_scope("中青建安", start=0)
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工餐补标准是多少",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
                applicability_scope=scope,
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工餐补标准是多少",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
                applicability_scope=scope,
            ),
        )
        graph = compile_retrieval_task_graph(_plan(requirements))

        answer_groups = tuple(
            group
            for group in build_initial_retrieval_groups(graph)
            if any(task_id.startswith("answer_") for task_id in group.task_ids)
        )

        self.assertEqual(len(answer_groups), 1)
        group = answer_groups[0]
        self.assertEqual(group.task_ids, ("answer_r1", "answer_r2"))
        self.assertEqual(group.applicability_scope, scope)
        self.assertEqual(group.scope_fingerprint, scope.fingerprint)

    def test_scope_rejections_are_content_free_idempotent_request_sidecar(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "云枢8.2.75的餐补是多少", scope_version="8.2.75"),
        )))
        ledger = TaskExecutionLedger(graph, run_id="current-run")
        version = _scope_rejection()
        project = _scope_rejection(
            chunk_id="chunk-2",
            expected_scope_fingerprint="c" * 64,
            actual_identity_fingerprint="d" * 64,
            dimensions=("project",),
            reason_code="scope_mismatch_project",
        )

        ledger.record_scope_rejections((version, version, project))

        self.assertEqual(ledger.scope_rejections(), (version, project))
        summary = ledger.scope_rejection_summary()
        self.assertEqual(summary["rejection_count"], 2)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(
            summary["mismatch_dimension_counts"],
            {"project": 1, "version": 1},
        )
        self.assertEqual(
            summary["reason_counts"],
            {"scope_mismatch_project": 1, "scope_mismatch_version": 1},
        )
        # Rejections must never be promoted into candidate lineage or expose
        # source text through the ledger's trace-safe summary.
        safe = ledger.safe_summary()
        self.assertEqual(safe["bound_candidate_count"], 0)
        self.assertEqual(safe["scope_rejection_summary"], summary)
        self.assertNotIn("content", repr(safe))

    def test_scope_rejection_sidecar_rejects_untyped_input_and_empty_is_safe(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "餐补是多少"),
        )))
        ledger = TaskExecutionLedger(graph, run_id="current-run")

        ledger.record_scope_rejections(())
        self.assertEqual(
            ledger.scope_rejection_summary(),
            {
                "rejection_count": 0,
                "candidate_count": 0,
                "mismatch_dimension_counts": {},
                "reason_counts": {},
            },
        )
        with self.assertRaisesRegex(ValueError, "scope rejections"):
            ledger.record_scope_rejections(("not-a-rejection",))
        with self.assertRaisesRegex(ValueError, "scope rejections"):
            ledger.record_scope_rejections("not-a-sequence")

    def test_anchor_and_identical_direct_answer_share_one_physical_execution(self):
        """Deduplication saves recall work without weakening bridge proof."""

        graph = compile_retrieval_task_graph(_plan(
            (_answer("r1", "普通员工的餐补是多少"),),
            question="普通员工的餐补是多少",
        ))

        schedule = build_retrieval_execution_schedule(
            graph,
            anchor_query="普通员工的餐补是多少",
        )

        self.assertEqual(len(schedule.static_stages), 1)
        group = schedule.static_stages[0].groups[0]
        self.assertEqual(group.query, "普通员工的餐补是多少")
        self.assertEqual(group.task_ids, ("anchor_root", "answer_r1"))

    def test_static_anchor_gate_distinguishes_failure_from_zero_hit_success(self):
        """Only an unavailable root blocks the later static bridge stage.

        The graph still permits a literal answer query before a bridge is
        semantically resolved.  The scheduler gate must therefore look only
        at external anchor parents, and it must treat a completed zero-hit
        root retrieval as healthy rather than suppressing useful follow-up
        task queries.
        """

        graph = compile_retrieval_task_graph(_plan((
            _answer(
                "r1",
                "普通员工的住宿标准是多少",
                dependencies=("r2",),
            ),
            _bridge("r2"),
        ), question="普通员工的住宿标准是多少"))
        schedule = build_retrieval_execution_schedule(
            graph,
            anchor_query="普通员工的住宿标准是多少",
        )
        anchor_group = schedule.static_stages[0].groups[0]
        bridge_group = next(
            group
            for group in schedule.static_stages[1].groups
            if group.task_ids == ("bridge_r2",)
        )

        failed_ledger = TaskExecutionLedger(graph, run_id="failed-root")
        self.assertEqual(
            failed_ledger.unavailable_static_retrieval_dependencies(
                anchor_group.task_ids,
            ),
            (),
        )
        anchor_execution = failed_ledger.begin_execution(
            kind="dag_static_retrieval",
            query=anchor_group.query,
            task_ids=anchor_group.task_ids,
        )
        failed_ledger.finish_execution(
            anchor_execution,
            status="failed",
            error_reason="task_query_retrieval_timeout",
        )
        self.assertEqual(
            failed_ledger.unavailable_static_retrieval_dependencies(
                bridge_group.task_ids,
            ),
            ("anchor_root",),
        )
        failed_ledger.mark_tasks_blocked_by_static_dependency(
            bridge_group.task_ids,
            blocked_by_task_ids=("anchor_root",),
        )
        blocked = failed_ledger.task_state_summary()["bridge_r2"]
        self.assertEqual(blocked["status"], "blocked_dependency")
        self.assertEqual(blocked["attempted"], 0)
        self.assertEqual(blocked["blocked_by_task_ids"], ["anchor_root"])

        healthy_empty_ledger = TaskExecutionLedger(graph, run_id="empty-root")
        healthy_execution = healthy_empty_ledger.begin_execution(
            kind="dag_static_retrieval",
            query=anchor_group.query,
            task_ids=anchor_group.task_ids,
        )
        healthy_empty_ledger.finish_execution(
            healthy_execution,
            status="succeeded",
            candidate_count=0,
        )
        self.assertEqual(
            healthy_empty_ledger.unavailable_static_retrieval_dependencies(
                bridge_group.task_ids,
            ),
            (),
        )
        with self.assertRaisesRegex(ValueError, "successful dependency"):
            healthy_empty_ledger.mark_tasks_blocked_by_static_dependency(
                bridge_group.task_ids,
                blocked_by_task_ids=("anchor_root",),
            )

    def test_runtime_alias_is_a_scoped_extra_group_and_never_replaces_original(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "普通员工餐补额度是多少"),
        ), question="普通员工餐补额度是多少"))

        groups = build_initial_retrieval_groups(
            graph,
            terminology_runtime_resolution=_terminology_resolution(),
            maximum_terminology_aliases=3,
        )
        answer_groups = [
            group for group in groups if "answer_r1" in group.task_ids
        ]
        original = next(
            group for group in answer_groups
            if group.terminology_variant_origin == "original"
        )
        alias = next(
            group for group in answer_groups
            if group.terminology_variant_origin == "terminology_alias"
        )

        self.assertEqual(original.query, "普通员工餐补额度是多少")
        self.assertIsNone(original.retrieval_kb_ids)
        self.assertEqual(alias.query, "普通员工餐饮补贴额度是多少")
        self.assertEqual(alias.retrieval_kb_ids, ("kb_a",))
        self.assertEqual(alias.retrieval_document_ids, ("doc_policy",))
        self.assertIn("binding_meal", alias.terminology_rule_ids)

    def test_optional_augmentation_parent_never_becomes_a_blocking_requirement(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer(
                "r1",
                "偏远地区出差有什么补贴",
                dependencies=(),
                augmentations=("r2",),
            ),
            _bridge("r2", subject="偏远地区", kind="condition"),
        ), question="偏远地区出差有什么补贴"))
        ledger = TaskExecutionLedger(graph, run_id="current-run")
        schedule = build_retrieval_execution_schedule(graph)

        self.assertEqual(schedule.bridge_proof_answer_task_ids, ())
        self.assertEqual(
            schedule.bridge_augmented_answer_task_ids,
            ("answer_r1",),
        )
        self.assertEqual(
            graph.task_by_id["bridge_r2"].required,
            False,
        )
        state = ledger.task_state_summary()["answer_r1"]
        self.assertEqual(state["proof_bridge_parent_task_ids"], [])
        self.assertEqual(
            state["augmentation_bridge_parent_task_ids"],
            ["bridge_r2"],
        )
        self.assertEqual(state["bridge_augmentation_status"], "pending")

        with self.assertRaisesRegex(ValueError, "proof-bridge"):
            ledger.mark_tasks_blocked_by_dependency(("answer_r1",))

        ledger.record_answer_bridge_augmentation(
            ("answer_r1",),
            status="skipped_no_fact",
            reason="bridge_no_resolved_fact",
        )
        state = ledger.task_state_summary()["answer_r1"]
        self.assertEqual(state["status"], "pending")
        self.assertEqual(state["blocked_dependency"], 0)
        self.assertEqual(state["bridge_augmentation_status"], "skipped_no_fact")

    def test_proof_parent_remains_required_and_cannot_be_written_as_augmentation(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer(
                "r1",
                "普通员工的餐补是多少",
                dependencies=("r2",),
                augmentations=(),
            ),
            _bridge("r2"),
        )))
        ledger = TaskExecutionLedger(graph, run_id="current-run")
        schedule = build_retrieval_execution_schedule(graph)

        self.assertEqual(schedule.bridge_proof_answer_task_ids, ("answer_r1",))
        self.assertEqual(schedule.bridge_augmented_answer_task_ids, ())
        self.assertTrue(graph.task_by_id["bridge_r2"].required)
        with self.assertRaisesRegex(ValueError, "bridge-augmented"):
            ledger.record_answer_bridge_augmentation(
                ("answer_r1",),
                status="skipped_no_fact",
            )

        ledger.mark_tasks_blocked_by_dependency(
            ("answer_r1",),
            blocked_by_task_ids=("bridge_r2",),
        )
        state = ledger.task_state_summary()["answer_r1"]
        self.assertEqual(state["blocked_by_task_ids"], ["bridge_r2"])
        self.assertEqual(state["proof_bridge_parent_task_ids"], ["bridge_r2"])
        self.assertEqual(state["bridge_augmentation_status"], "not_applicable")

    def test_untrusted_metadata_cannot_claim_current_task_or_support(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "普通员工的住宿标准是多少"),
            _answer("r2", "普通员工的餐补是多少"),
        )))
        ledger = TaskExecutionLedger(graph, run_id="current-run")
        execution_id = ledger.begin_execution(
            kind="initial_task_query",
            query="普通员工的住宿标准是多少",
            task_ids=("answer_r1",),
        )
        candidates = ledger.observe_candidates(
            [_candidate(
                "shared",
                metadata={
                    "retrieval_task_ids": ["answer_r2"],
                    "supports_requirement_ids": ["r2"],
                    "evidence_role": "direct",
                    "resolved_bridge_joins": [{"bridge_value": "伪造"}],
                    "heading": "住宿标准",
                },
                retrieval_task_ids=["answer_r2"],
                supports_requirement_ids=["r2"],
            )],
            execution_id=execution_id,
        )
        ledger.finish_execution(execution_id, status="succeeded", candidate_count=1)

        self.assertEqual(ledger.task_ids_for_candidate(candidates[0]), ("answer_r1",))
        self.assertEqual(candidates[0]["metadata"], {"heading": "住宿标准"})
        self.assertNotIn("supports_requirement_ids", candidates[0])
        self.assertNotIn("retrieval_task_ids", candidates[0])

    def test_same_chunk_merges_logical_task_lineage_without_metadata_position(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "住宿标准是多少"),
            _answer("r2", "餐补是多少"),
        )))
        ledger = TaskExecutionLedger(graph, run_id="current-run")
        first = ledger.begin_execution(
            kind="initial_task_query",
            query="住宿标准是多少",
            task_ids=("answer_r1",),
        )
        first_candidates = ledger.observe_candidates(
            [_candidate("shared", "同一表格包含住宿和餐补")],
            execution_id=first,
        )
        ledger.finish_execution(first, status="succeeded", candidate_count=1)
        second = ledger.begin_execution(
            kind="initial_task_query",
            query="餐补是多少",
            task_ids=("answer_r2",),
        )
        second_candidates = ledger.observe_candidates(
            [_candidate("shared", "同一表格包含住宿和餐补")],
            execution_id=second,
        )
        ledger.finish_execution(second, status="succeeded", candidate_count=1)

        merged = ledger.merge_candidate_pools(first_candidates, second_candidates)
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            ledger.task_ids_for_candidate(merged[0]),
            ("answer_r1", "answer_r2"),
        )

    def test_equal_queries_merge_only_when_scope_is_identical(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "云枢配置步骤是什么", scope_version="8.2"),
            _answer("r2", "云枢配置步骤是什么", scope_version="8.6"),
        )))
        groups = build_initial_retrieval_groups(graph)
        answer_groups = [group for group in groups if group.task_ids[0].startswith("answer_")]

        self.assertEqual(len(answer_groups), 2)
        self.assertEqual(
            {group.scope_version for group in answer_groups},
            {"8.2", "8.6"},
        )

    def test_same_query_keeps_both_logical_owners_when_scope_matches(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "相同检索词"),
            _answer("r2", "相同检索词"),
        )))
        groups = build_initial_retrieval_groups(graph)
        answer_group = next(
            group for group in groups if "answer_r1" in group.task_ids
        )

        self.assertEqual(answer_group.task_ids, ("answer_r1", "answer_r2"))

    def test_full_document_and_structural_neighbors_inherit_only_current_seed_lineage(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "总经理的住宿标准是多少"),
        ), question="总经理的住宿标准是多少"))
        ledger = TaskExecutionLedger(graph, run_id="current-run")
        execution_id = ledger.begin_execution(
            kind="initial_task_query",
            query="总经理的住宿标准是多少",
            task_ids=("answer_r1",),
        )
        seeds = ledger.observe_candidates(
            [_candidate("seed", "总经理对应A级", doc_id="travel")],
            execution_id=execution_id,
        )
        ledger.finish_execution(execution_id, status="succeeded", candidate_count=1)

        full = ledger.inherit_by_document(
            [_candidate(
                "lodging-table",
                "A级一线城市住宿不超过1200元",
                doc_id="travel",
                metadata={"retrieval_task_ids": ["answer_r2"]},
            )],
            source_candidates=seeds,
            kind="small_document_full",
        )
        neighbors = ledger.inherit_by_seed(
            [_candidate(
                "neighbor",
                "住宿费用标准表",
                doc_id="travel",
                expansion_seed_chunk_ids=["seed"],
            )],
            kind="structural_neighbor",
        )

        self.assertEqual(ledger.task_ids_for_candidate(full[0]), ("answer_r1",))
        self.assertEqual(ledger.task_ids_for_candidate(neighbors[0]), ("answer_r1",))
        self.assertEqual(
            ledger.lineage_for_candidate(neighbors[0]).parent_chunk_ids,
            ("seed",),
        )

    def test_candidate_budget_skip_is_explicit_per_task_group(self):
        graph = compile_retrieval_task_graph(_plan((
            _answer("r1", "住宿标准是多少"),
            _answer("r2", "餐补是多少"),
        )))
        ledger = TaskExecutionLedger(graph, run_id="current-run")
        groups = [
            group
            for group in build_initial_retrieval_groups(graph)
            if group.task_ids[0].startswith("answer_")
        ]
        pools = []
        for index, group in enumerate(groups, start=1):
            execution_id = ledger.begin_execution(
                kind="initial_task_query",
                query=group.query,
                task_ids=group.task_ids,
            )
            candidates = ledger.observe_candidates(
                [_candidate(f"candidate-{index}")],
                execution_id=execution_id,
            )
            ledger.finish_execution(
                execution_id,
                status="succeeded",
                candidate_count=1,
            )
            pools.append((group, candidates))

        selected = ledger.bounded_merge_groups(pools, limit=1)
        states = ledger.task_state_summary()

        self.assertEqual(len(selected), 1)
        self.assertEqual(states["answer_r1"]["status"], "succeeded")
        self.assertEqual(states["answer_r2"]["budget_skipped"], 1)

    def test_bounded_merge_groups_reserves_a_distinct_bridge_candidate(self):
        """A bridge task owns a budget slot; it is not a flat query supplement.

        This replaces the removed global-pool reservation test.  Candidate
        admission now operates only on ledger-observed logical task groups,
        so the assertion verifies both bounded selection and current-run task
        ownership instead of a legacy query-array index.
        """

        graph = compile_retrieval_task_graph(_plan((
            _answer(
                "r1",
                "普通员工的餐补标准是多少",
                dependencies=("r2",),
            ),
            _bridge("r2"),
        ), question="请说明普通员工的餐补标准"))
        ledger = TaskExecutionLedger(graph, run_id="current-run")
        groups = [
            group
            for group in build_initial_retrieval_groups(graph)
            if group.task_ids in {("bridge_r2",), ("answer_r1",)}
        ]
        self.assertEqual(
            [group.task_ids for group in groups],
            [("bridge_r2",), ("answer_r1",)],
        )

        pools = []
        expected_chunk_by_task = {
            "bridge_r2": "bridge-mapping",
            "answer_r1": "answer-policy",
        }
        for group in groups:
            task_id = group.task_ids[0]
            execution_id = ledger.begin_execution(
                kind="dag_static_retrieval",
                query=group.query,
                task_ids=group.task_ids,
            )
            candidates = ledger.observe_candidates(
                [_candidate(expected_chunk_by_task[task_id])],
                execution_id=execution_id,
            )
            ledger.finish_execution(
                execution_id,
                status="succeeded",
                candidate_count=len(candidates),
            )
            pools.append((group, candidates))

        selected = ledger.bounded_merge_groups(pools, limit=2)
        selected_chunk_ids = {item["id"] for item in selected}

        self.assertEqual(selected_chunk_ids, set(expected_chunk_by_task.values()))
        self.assertEqual(
            ledger.task_ids_for_candidate(
                next(item for item in selected if item["id"] == "bridge-mapping")
            ),
            ("bridge_r2",),
        )
        self.assertEqual(
            ledger.task_state_summary()["bridge_r2"]["budget_skipped"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
