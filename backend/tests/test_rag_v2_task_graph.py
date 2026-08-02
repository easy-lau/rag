import json
import unittest

from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.task_graph import (
    RETRIEVAL_TASK_GRAPH_SCHEMA_VERSION,
    RagExecutionBundle,
    RetrievalTask,
    RetrievalTaskGraph,
    compile_rag_execution_bundle,
    compile_retrieval_task_graph,
)
from core.rag_v2.task_execution import TaskExecutionLedger


def _answer(
    requirement_id: str,
    description: str,
    *,
    importance: str = "required",
    dependencies: tuple[str, ...] = (),
    augmentations: tuple[str, ...] | None = None,
    scope_product: str | None = None,
    scope_version: str | None = None,
) -> AnswerRequirementV2:
    kwargs = {}
    if augmentations is not None:
        kwargs["augmentation_requirement_ids"] = augmentations
    return AnswerRequirementV2(
        id=requirement_id,
        description=description,
        role="answer",
        importance=importance,
        depends_on_requirement_ids=dependencies,
        scope_product=scope_product,
        scope_version=scope_version,
        **kwargs,
    )


def _bridge(
    requirement_id: str,
    subject: str = "普通员工",
    *,
    kind: str = "classification",
    importance: str = "helpful",
    scope_product: str | None = None,
    scope_version: str | None = None,
) -> AnswerRequirementV2:
    return AnswerRequirementV2(
        id=requirement_id,
        description=f"确认{subject}对应的适用分类",
        role="bridge",
        importance=importance,
        source="inferred",
        bridge_subject=subject,
        bridge_kind=kind,
        scope_product=scope_product,
        scope_version=scope_version,
    )


def _plan(
    requirements: tuple[AnswerRequirementV2, ...],
    *,
    retrieval_queries: tuple[str, ...] = ("故意与任务无关的旧位置查询",),
) -> QueryPlanV2:
    return QueryPlanV2(
        original_query="普通员工分别查询多个标准",
        answer_shape="multi_part",
        retrieval_queries=retrieval_queries,
        requirements=requirements,
        confidence=0.95,
        source="local",
    )


class RetrievalTaskContractTests(unittest.TestCase):
    def test_compiler_binds_by_requirement_and_ignores_flat_query_positions(self) -> None:
        requirements = (
            _answer("r1", "普通员工的住宿标准是多少", dependencies=("r3",)),
            _answer("r2", "普通员工的餐补是多少", dependencies=("r3",)),
            _bridge("r3"),
        )
        plan = _plan(
            requirements,
            retrieval_queries=(
                "这是一个故意错位的旧查询",
                "另一个无关查询",
            ),
        )

        graph = compile_retrieval_task_graph(plan)
        tasks = graph.task_by_id

        self.assertEqual(
            tasks["answer_r1"].query,
            "普通员工的住宿标准是多少",
        )
        self.assertEqual(tasks["answer_r2"].query, "普通员工的餐补是多少")
        self.assertEqual(tasks["answer_r1"].target_requirement_ids, ("r1",))
        self.assertEqual(tasks["answer_r2"].target_requirement_ids, ("r2",))
        self.assertEqual(
            tasks["answer_r1"].dependency_task_ids,
            ("anchor_root", "bridge_r3"),
        )
        self.assertEqual(
            tasks["answer_r2"].dependency_task_ids,
            ("anchor_root", "bridge_r3"),
        )
        self.assertEqual(
            dict(tasks["answer_r1"].bridge_edge_modes),
            {"bridge_r3": "proof"},
        )
        self.assertEqual(
            tasks["answer_r1"].bridge_parent_task_ids(mode="proof"),
            ("bridge_r3",),
        )
        self.assertEqual(tasks["bridge_r3"].required, True)
        self.assertNotIn("故意错位", " ".join(task.query for task in graph.tasks))

        bridge_query = tasks["bridge_r3"].query
        self.assertIn("普通员工", bridge_query)
        self.assertIn("适用分类", bridge_query)
        self.assertNotIn("D级", bridge_query)
        self.assertNotIn("住宿补", bridge_query)
        self.assertNotIn("餐补标准", bridge_query)
        self.assertEqual(
            graph.requirement_ids_reachable_from(("anchor_root",)),
            frozenset({"r1", "r2", "r3"}),
        )

    def test_same_query_is_kept_as_two_logical_tasks(self) -> None:
        requirements = (
            _answer("r1", "同一个字面检索词"),
            _answer("r2", "同一个字面检索词"),
        )
        graph = compile_retrieval_task_graph(_plan(requirements))

        self.assertEqual(graph.task_by_id["answer_r1"].query, "同一个字面检索词")
        self.assertEqual(graph.task_by_id["answer_r2"].query, "同一个字面检索词")
        self.assertNotEqual(
            graph.task_by_id["answer_r1"].task_id,
            graph.task_by_id["answer_r2"].task_id,
        )
        self.assertEqual(
            sum(task.query == "同一个字面检索词" for task in graph.tasks),
            2,
        )

    def test_augmentation_only_bridge_is_nonblocking_and_nonrequired(self) -> None:
        requirements = (
            _answer(
                "r1",
                "偏远地区出差有什么补贴",
                dependencies=(),
                augmentations=("r2",),
            ),
            _bridge("r2", subject="偏远地区", kind="condition"),
        )
        plan = QueryPlanV2(
            original_query="偏远地区出差有什么补贴",
            answer_shape="fact",
            retrieval_queries=("偏远地区出差有什么补贴",),
            requirements=requirements,
            confidence=0.95,
            source="local",
        )

        graph = compile_retrieval_task_graph(plan)
        answer = graph.task_by_id["answer_r1"]
        bridge = graph.task_by_id["bridge_r2"]

        self.assertEqual(
            answer.dependency_task_ids,
            ("anchor_root", "bridge_r2"),
        )
        self.assertEqual(
            dict(answer.bridge_edge_modes),
            {"bridge_r2": "augmentation"},
        )
        self.assertEqual(answer.bridge_parent_task_ids(mode="proof"), ())
        self.assertEqual(
            answer.bridge_parent_task_ids(mode="augmentation"),
            ("bridge_r2",),
        )
        self.assertFalse(bridge.required)
        self.assertEqual(
            graph.answer_bridge_parent_task_ids(
                "answer_r1",
                mode="augmentation",
            ),
            ("bridge_r2",),
        )
        self.assertEqual(
            graph.safe_summary()["bridge_edge_counts"],
            {"proof": 0, "augmentation": 1},
        )

    def test_bridge_scope_is_copied_without_inventing_answer_terms(self) -> None:
        requirements = (
            _answer(
                "r1",
                "云枢8.6普通员工的餐补是多少",
                dependencies=("r2",),
                scope_product="云枢",
                scope_version="8.6",
            ),
            _bridge(
                "r2",
                scope_product="云枢",
                scope_version="8.6",
            ),
        )
        graph = compile_retrieval_task_graph(_plan(requirements))
        bridge = graph.task_by_id["bridge_r2"]
        answer = graph.task_by_id["answer_r1"]

        self.assertEqual(bridge.scope_product, "云枢")
        self.assertEqual(bridge.scope_version, "8.6")
        self.assertTrue(bridge.scope_explicit_version)
        self.assertEqual(answer.scope_version, "8.6")
        self.assertEqual(
            bridge.query,
            "云枢 8.6 普通员工 对应的适用分类 等级 类别 职级 角色 版本 档位 阶段",
        )
        self.assertNotIn("D级", bridge.query)

    def test_scoped_answer_cannot_consume_an_unscoped_bridge(self) -> None:
        # Scope is an execution dependency, not just answer presentation.  A
        # bridge retrieved without the answer's version/project scope could
        # materialize an invalid second hop.
        requirements = (
            _answer(
                "r1",
                "云枢8.6普通员工的餐补是多少",
                dependencies=("r2",),
                scope_product="云枢",
                scope_version="8.6",
            ),
            _bridge("r2"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "scoped answer bridge must carry the same applicability scope",
        ):
            _plan(requirements)

    def test_one_bridge_cannot_join_two_explicit_version_answers(self) -> None:
        requirements = (
            _answer(
                "r1",
                "CloudPivot 6 安全配置",
                dependencies=("r3",),
                scope_product="CloudPivot",
                scope_version="6",
            ),
            _answer(
                "r2",
                "CloudPivot 7 安全配置",
                dependencies=("r3",),
                scope_product="CloudPivot",
                scope_version="7",
            ),
            _bridge(
                "r3",
                scope_product="CloudPivot",
                scope_version="6",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "one bridge requirement cannot serve answers with different applicability scopes",
        ):
            _plan(requirements)

    def test_bridge_query_uses_the_validated_relation_family(self) -> None:
        requirements = (
            _answer("r1", "审批额度由项目等级决定是多少", dependencies=("r2",)),
            _bridge("r2", subject="项目等级", kind="condition"),
        )

        graph = compile_retrieval_task_graph(_plan(requirements))
        query = graph.task_by_id["bridge_r2"].query

        self.assertIn("项目等级", query)
        self.assertIn("适用条件", query)
        self.assertNotIn("职级", query)

    def test_untyped_bridge_cannot_enter_executable_task_graph(self) -> None:
        requirements = (
            _answer("r1", "普通员工的餐补是多少", dependencies=("r2",)),
            AnswerRequirementV2(
                id="r2",
                description="确认普通员工的关系",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
            ),
        )

        with self.assertRaisesRegex(ValueError, "validated bridge_kind"):
            compile_retrieval_task_graph(_plan(requirements))

    def test_execution_bundle_makes_plan_graph_handoff_atomic(self) -> None:
        typed_requirements = (
            _answer("r1", "普通员工的餐补是多少", dependencies=("r2",)),
            _bridge("r2"),
        )
        typed_plan = _plan(typed_requirements)
        bundle = compile_rag_execution_bundle(typed_plan)

        self.assertEqual(bundle.mode, "ledgered")
        self.assertTrue(bundle.uses_task_ledger)
        self.assertIsNotNone(bundle.task_graph)
        self.assertEqual(bundle.task_graph.requirements, typed_plan.requirements)

        untyped_requirements = (
            _answer("r1", "普通员工的餐补是多少", dependencies=("r2",)),
            AnswerRequirementV2(
                id="r2",
                description="确认普通员工的关系",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
            ),
        )
        not_ready = compile_rag_execution_bundle(_plan(untyped_requirements))
        self.assertEqual(not_ready.mode, "not_ready")
        self.assertEqual(not_ready.reason, "untyped_bridge_semantics")
        self.assertIsNone(not_ready.task_graph)

        # A ready plan is not merely paired with a graph: its candidate
        # provenance is recorded by this request's ledger.  Persisted
        # candidate metadata is intentionally irrelevant to this assertion.
        ledger = TaskExecutionLedger(bundle.task_graph, run_id="task-graph-test")
        execution_id = ledger.begin_execution(
            kind="initial_task_query",
            query=typed_plan.requirements[0].description,
            task_ids=("answer_r1",),
        )
        observed = ledger.observe_candidates(({
            "kb_id": "kb-test",
            "doc_id": "doc-test",
            "chunk_id": "chunk-test",
            "content": "普通员工的餐补标准为100元/天。",
            "metadata": {"retrieval_task_ids": ["bridge_r2"]},
        },), execution_id=execution_id)
        ledger.finish_execution(execution_id, status="succeeded", candidate_count=1)
        self.assertEqual(
            ledger.task_ids_for_candidate(observed[0]),
            ("answer_r1",),
        )

        other_plan = _plan((_answer("r1", "另一个问题"),))
        with self.assertRaisesRegex(ValueError, "requirements must match"):
            RagExecutionBundle(
                plan=other_plan,
                mode="ledgered",
                reason="invalid_pair",
                task_graph=bundle.task_graph,
            )

    def test_anchor_is_recall_only_and_has_no_requirement_owner(self) -> None:
        requirements = (_answer("r1", "直接事实是什么"),)
        anchor = RetrievalTask(
            task_id="anchor_root",
            role="anchor",
            query="原始问题",
        )
        answer = RetrievalTask(
            task_id="answer_r1",
            role="answer",
            query="直接事实是什么",
            target_requirement_ids=("r1",),
            required=True,
        )
        graph = RetrievalTaskGraph(
            requirements=requirements,
            tasks=(anchor, answer),
        )

        self.assertEqual(anchor.target_requirement_ids, ())
        self.assertFalse(anchor.required)
        self.assertEqual(graph.safe_summary()["task_counts"]["anchor"], 1)
        self.assertNotIn("r1", anchor.target_requirement_ids)

        with self.assertRaisesRegex(ValueError, "anchor tasks cannot target"):
            RetrievalTask(
                task_id="anchor_bad",
                role="anchor",
                query="原始问题",
                target_requirement_ids=("r1",),
            )

    def test_graph_serialization_is_json_safe_and_summary_is_content_light(self) -> None:
        requirements = (
            _answer("r1", "普通员工的餐补是多少", dependencies=("r2",)),
            _bridge("r2"),
        )
        graph = compile_retrieval_task_graph(_plan(requirements))
        serialized = graph.to_dict()
        json.dumps(serialized, ensure_ascii=False)

        self.assertEqual(
            serialized["schema_version"],
            RETRIEVAL_TASK_GRAPH_SCHEMA_VERSION,
        )
        self.assertEqual(serialized["requirements"][0]["id"], "r1")
        self.assertNotIn("description", serialized["requirements"][0])
        summary = graph.safe_summary()
        self.assertNotIn("餐补", json.dumps(summary, ensure_ascii=False))
        self.assertEqual(
            serialized["tasks"][-1]["bridge_edge_modes"],
            {"bridge_r2": "proof"},
        )


class RetrievalTaskGraphValidationTests(unittest.TestCase):
    def test_duplicate_task_ids_are_rejected(self) -> None:
        requirement = _answer("r1", "事实")
        task = RetrievalTask(
            task_id="answer_r1",
            role="answer",
            query="事实",
            target_requirement_ids=("r1",),
        )
        with self.assertRaisesRegex(ValueError, "duplicate task ids"):
            RetrievalTaskGraph(
                requirements=(requirement,),
                tasks=(task, task),
            )

    def test_unknown_target_and_dangling_dependency_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requirement that does not exist"):
            RetrievalTaskGraph(
                requirements=(_answer("r1", "事实"),),
                tasks=(
                    RetrievalTask(
                        task_id="answer_missing",
                        role="answer",
                        query="事实",
                        target_requirement_ids=("r9",),
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "dangling dependency"):
            RetrievalTaskGraph(
                requirements=(_answer("r1", "事实"),),
                tasks=(
                    RetrievalTask(
                        task_id="answer_r1",
                        role="answer",
                        query="事实",
                        target_requirement_ids=("r1",),
                        dependency_task_ids=("bridge_missing",),
                    ),
                ),
            )

    def test_role_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "role does not match"):
            RetrievalTaskGraph(
                requirements=(_answer("r1", "事实"),),
                tasks=(
                    RetrievalTask(
                        task_id="bridge_r1",
                        role="bridge",
                        query="分类",
                        target_requirement_ids=("r1",),
                    ),
                ),
            )

    def test_cycle_is_rejected(self) -> None:
        requirements = (
            _answer("r1", "事实一"),
            _answer("r2", "事实二"),
        )
        tasks = (
            RetrievalTask(
                task_id="answer_r1",
                role="answer",
                query="事实一",
                target_requirement_ids=("r1",),
                dependency_task_ids=("answer_r2",),
            ),
            RetrievalTask(
                task_id="answer_r2",
                role="answer",
                query="事实二",
                target_requirement_ids=("r2",),
                dependency_task_ids=("answer_r1",),
            ),
        )
        with self.assertRaisesRegex(ValueError, "contains a cycle"):
            RetrievalTaskGraph(requirements=requirements, tasks=tasks)

    def test_required_answer_must_have_exactly_one_owner(self) -> None:
        requirement = _answer("r1", "事实")
        with self.assertRaisesRegex(ValueError, "every required answer"):
            RetrievalTaskGraph(
                requirements=(requirement,),
                tasks=(
                    RetrievalTask(
                        task_id="anchor_root",
                        role="anchor",
                        query="事实",
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "multiple answer tasks"):
            RetrievalTaskGraph(
                requirements=(requirement,),
                tasks=(
                    RetrievalTask(
                        task_id="answer_one",
                        role="answer",
                        query="事实",
                        target_requirement_ids=("r1",),
                        required=True,
                    ),
                    RetrievalTask(
                        task_id="answer_two",
                        role="answer",
                        query="事实",
                        target_requirement_ids=("r1",),
                        required=True,
                    ),
                ),
            )

    def test_answer_bridge_edges_must_match_requirement_edges(self) -> None:
        requirements = (
            _answer("r1", "餐补", dependencies=("r2",)),
            _bridge("r2"),
        )
        with self.assertRaisesRegex(ValueError, "bridge edge modes"):
            RetrievalTaskGraph(
                requirements=requirements,
                tasks=(
                    RetrievalTask(
                        task_id="answer_r1",
                        role="answer",
                        query="餐补",
                        target_requirement_ids=("r1",),
                        required=True,
                    ),
                    RetrievalTask(
                        task_id="bridge_r2",
                        role="bridge",
                        query="分类",
                        target_requirement_ids=("r2",),
                        required=True,
                    ),
                ),
            )

    def test_unreferenced_bridge_is_rejected_as_dangling(self) -> None:
        requirements = (
            _answer("r1", "事实", dependencies=()),
            _bridge("r2"),
        )
        # The requirement validator rejects this before task ownership checks,
        # which is the desired fail-closed boundary for an unreferenced bridge.
        with self.assertRaisesRegex(ValueError, "unreferenced bridge"):
            RetrievalTaskGraph(
                requirements=requirements,
                tasks=(
                    RetrievalTask(
                        task_id="answer_r1",
                        role="answer",
                        query="事实",
                        target_requirement_ids=("r1",),
                    ),
                    RetrievalTask(
                        task_id="bridge_r2",
                        role="bridge",
                        query="分类",
                        target_requirement_ids=("r2",),
                        required=False,
                    ),
                ),
            )

    def test_augmentation_edge_mode_must_match_requirement_edge(self) -> None:
        requirements = (
            _answer("r1", "条件补贴", dependencies=(), augmentations=("r2",)),
            _bridge("r2", subject="偏远地区", kind="condition"),
        )
        with self.assertRaisesRegex(ValueError, "bridge edge modes"):
            RetrievalTaskGraph(
                requirements=requirements,
                tasks=(
                    RetrievalTask(
                        task_id="anchor_root",
                        role="anchor",
                        query="条件补贴",
                    ),
                    RetrievalTask(
                        task_id="bridge_r2",
                        role="bridge",
                        query="偏远地区适用条件",
                        target_requirement_ids=("r2",),
                        dependency_task_ids=("anchor_root",),
                        required=False,
                    ),
                    RetrievalTask(
                        task_id="answer_r1",
                        role="answer",
                        query="条件补贴",
                        target_requirement_ids=("r1",),
                        dependency_task_ids=("anchor_root", "bridge_r2"),
                        bridge_edge_modes={"bridge_r2": "proof"},
                        required=True,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
