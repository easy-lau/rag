import unittest

from core.query_constraints import extract_query_constraints
from core.rag_v2.bridge_resolution import (
    ResolvedBridgeFact,
    bridge_fact_matches_candidate_scope,
    build_bridge_expansion_queries,
    candidate_supports_resolved_answer_set,
    content_contains_bridge_value,
    content_contains_positive_subject,
    content_matches_answer_target,
    extract_bridge_values,
)
from core.rag_v2.contracts import AnswerRequirementV2
from core.rag_v2.evidence import assemble_evidence_bundle


def _candidate(
    chunk_id: str,
    *,
    doc_id: str = "doc-a",
    kb_id: str = "kb-a",
    chunk_index: int = 0,
    content: str = "具体条款：住宿标准为450元/天。",
    **values,
) -> dict:
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "chunk_index": chunk_index,
        "content": content,
        **values,
    }


def _multi_hop_requirements(
    answer_description: str,
    bridge_description: str,
) -> tuple[AnswerRequirementV2, AnswerRequirementV2]:
    return (
        AnswerRequirementV2(
            id="r1",
            description=answer_description,
            depends_on_requirement_ids=("r2",),
        ),
        AnswerRequirementV2(
            id="r2",
            description=bridge_description,
            role="bridge",
            importance="helpful",
            source="inferred",
        ),
    )


class EvidenceBundleAssemblyTests(unittest.TestCase):
    def test_normal_v2_shape_requires_typed_requirements(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty requirements"):
            assemble_evidence_bundle(
                query="查询标准",
                answer_shape="fact",
                candidates=[_candidate("seed")],
            )

    def test_multi_hop_shape_requires_bridge_requirement(self) -> None:
        with self.assertRaisesRegex(ValueError, "answer-to-bridge dependency"):
            assemble_evidence_bundle(
                query="对象对应的额度是多少",
                answer_shape="multi_hop",
                candidates=[_candidate("seed")],
                requirements=(
                    AnswerRequirementV2(
                        id="r1",
                        description="对象对应的额度是多少",
                        depends_on_requirement_ids=(),
                    ),
                ),
            )

    def test_single_requirement_retrieval_seed_is_not_requirement_support(self) -> None:
        requirement = AnswerRequirementV2(id="r1", description="某项标准")
        bundle = assemble_evidence_bundle(
            query="某项标准",
            candidates=[
                _candidate(
                    "seed",
                    content="明确条款：上限为450元/天。",
                    candidate_origins=["initial_retrieval"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=("某项标准",),
        )

        self.assertEqual(bundle.items[0].role, "background")
        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_overview_maps_all_bounded_anchored_document_chunks(self) -> None:
        requirement = AnswerRequirementV2(id="r1", description="制度概览")
        bundle = assemble_evidence_bundle(
            query="制度概览",
            answer_shape="overview",
            candidates=[
                _candidate(
                    "seed",
                    content="制度标题",
                    candidate_origins=["initial_retrieval"],
                )
            ],
            overview_candidates=[
                _candidate("section", chunk_index=1, content="具体章节内容")
            ],
            requirements=(requirement,),
            retrieval_queries=("制度概览",),
        )

        self.assertEqual(set(bundle.answer_source_ids), {"seed", "section"})
        self.assertTrue(all(
            item.supports_requirement_ids == ("r1",)
            for item in bundle.answer_sources
        ))

    def test_query_indexes_preserve_visible_per_requirement_mapping(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="交通要求"),
            AnswerRequirementV2(id="r2", description="住宿要求"),
        )
        bundle = assemble_evidence_bundle(
            query="请分别查询交通和住宿要求",
            candidates=[
                _candidate(
                    "lodging",
                    content="住宿要求：每天不超过450元。",
                    expansion_query_indexes=[1],
                ),
                _candidate(
                    "transport",
                    chunk_index=1,
                    content="交通要求：乘坐高铁二等座。",
                    expansion_query_indexes=[0],
                ),
            ],
            requirements=requirements,
            retrieval_queries=("交通要求", "住宿要求"),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["transport"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["transport"].role, "direct")
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(set(bundle.answer_source_ids), {"transport", "lodging"})
        self.assertEqual(bundle.state.completeness, "complete")

    def test_merged_query_indexes_cannot_manufacture_missing_answer(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工出差的住宿标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工出差的交通标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r3",
                description="普通员工出差的餐补标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r4",
                description=(
                    "确认普通员工对应的适用分类、等级、类别或阶段"
                    "（用于确定住宿标准）"
                ),
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        all_query_indexes = [0, 1, 2, 3]
        bundle = assemble_evidence_bundle(
            query="普通员工出差的住宿、交通和餐补标准分别是多少？",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "lodging",
                    content="D级住宿标准：一线城市不超过450元/天。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "transport",
                    chunk_index=1,
                    content="D级交通标准：飞机经济舱、高铁二等座。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "classification",
                    chunk_index=2,
                    content="职级分类：普通员工对应D级。",
                    expansion_query_indexes=all_query_indexes,
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(
                requirement.description for requirement in requirements
            ),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["transport"].supports_requirement_ids, ("r2",))
        self.assertEqual(
            by_id["classification"].supports_requirement_ids,
            ("r4",),
        )
        self.assertEqual(bundle.missing_requirement_ids, ("r3",))
        self.assertEqual(bundle.state.completeness, "partial")

    def test_common_subject_terms_cannot_cover_coordinated_answers(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="普通员工住宿标准"),
            AnswerRequirementV2(id="r2", description="普通员工交通标准"),
            AnswerRequirementV2(id="r3", description="普通员工餐补标准"),
        )
        bundle = assemble_evidence_bundle(
            query="普通员工住宿、交通和餐补标准分别是多少？",
            answer_shape="multi_part",
            candidates=[
                _candidate(
                    "common-subject",
                    content="普通员工制度适用于公司全体员工。",
                    expansion_query_indexes=[0, 1, 2],
                )
            ],
            requirements=requirements,
            retrieval_queries=tuple(
                requirement.description for requirement in requirements
            ),
            completeness="complete",
        )

        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1", "r2", "r3"))
        self.assertEqual(bundle.state.completeness, "unknown")

    def test_merged_query_indexes_keep_each_visible_coordinated_answer(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工出差的住宿标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工出差的交通标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r3",
                description="普通员工出差的餐补标准是多少",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r4",
                description=(
                    "确认普通员工对应的适用分类、等级、类别或阶段"
                    "（用于确定住宿标准）"
                ),
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        all_query_indexes = [0, 1, 2, 3]
        bundle = assemble_evidence_bundle(
            query="普通员工出差的住宿、交通和餐补标准分别是多少？",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "lodging",
                    content="D级住宿标准：一线城市不超过450元/天。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "transport",
                    chunk_index=1,
                    content="D级交通标准：飞机经济舱、高铁二等座。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "meal",
                    chunk_index=2,
                    content="D级餐补标准：每天100元。",
                    filename="公司出差管理标准.docx",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "classification",
                    chunk_index=3,
                    content="职级分类：普通员工对应D级。",
                    expansion_query_indexes=all_query_indexes,
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(
                requirement.description for requirement in requirements
            ),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["transport"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["meal"].supports_requirement_ids, ("r3",))
        self.assertEqual(
            by_id["classification"].supports_requirement_ids,
            ("r4",),
        )
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_reimbursement_fragments_cannot_cover_each_other(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="报销提交时限是多久",
            ),
            AnswerRequirementV2(
                id="r2",
                description="需要提供哪些凭证",
            ),
        )
        all_query_indexes = [0, 1]
        deadline = _candidate(
            "deadline",
            content="费用报销时限：出差结束后5个工作日内提交。",
            expansion_query_indexes=all_query_indexes,
        )
        receipts = _candidate(
            "receipts",
            chunk_index=1,
            content="报销凭证：必须提供正规发票、行程单及住宿发票。",
            expansion_query_indexes=all_query_indexes,
        )
        values = {
            "query": "报销提交时限是多久？需要提供哪些凭证？",
            "answer_shape": "multi_part",
            "requirements": requirements,
            "retrieval_queries": tuple(
                requirement.description for requirement in requirements
            ),
            "completeness": "complete",
        }

        complete = assemble_evidence_bundle(
            candidates=[deadline, receipts],
            **values,
        )
        by_id = {item.chunk_id: item for item in complete.items}
        self.assertEqual(by_id["deadline"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["receipts"].supports_requirement_ids, ("r2",))
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(complete.state.completeness, "complete")

        deadline_only = assemble_evidence_bundle(
            candidates=[deadline],
            **values,
        )
        self.assertEqual(deadline_only.missing_requirement_ids, ("r2",))
        self.assertEqual(deadline_only.state.completeness, "partial")

        receipts_only = assemble_evidence_bundle(
            candidates=[receipts],
            **values,
        )
        self.assertEqual(receipts_only.missing_requirement_ids, ("r1",))
        self.assertEqual(receipts_only.state.completeness, "partial")

    def test_bridge_query_index_cannot_promote_value_only_chunk(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐补金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "amount",
                    content="餐饮补贴：D级为100元/天。",
                    candidate_origins=["initial_retrieval"],
                    expansion_query_indexes=[0, 1],
                ),
                _candidate(
                    "classification",
                    chunk_index=1,
                    content="职级分类：普通岗位对应D级。",
                    expansion_query_indexes=[1],
                ),
            ],
            requirements=requirements,
            retrieval_queries=(
                "普通岗位的餐饮补贴是多少",
                "普通岗位 对应的适用分类 等级 类别 阶段",
            ),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["amount"].supports_requirement_ids, ("r1",))
        self.assertEqual(
            by_id["classification"].supports_requirement_ids,
            ("r2",),
        )
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_explicit_support_ids_are_filtered_to_known_requirements(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="目标一"),
            AnswerRequirementV2(id="r2", description="目标二"),
        )
        bundle = assemble_evidence_bundle(
            query="查询目标",
            candidates=[
                _candidate(
                    "explicit",
                    content="目标一：已完成",
                    role="direct",
                    supports_requirement_ids=["r1", "unknown", "INVALID"],
                )
            ],
            requirements=requirements,
            retrieval_queries=("其他查询",),
            rerank_succeeded=True,
            completeness="complete",
        )

        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ("r1",))
        self.assertEqual(item.metadata["supports_requirement_ids"], ["r1"])
        self.assertEqual(item.role, "direct")
        self.assertEqual(bundle.answer_source_ids, ("explicit",))
        self.assertEqual(bundle.missing_requirement_ids, ("r2",))
        self.assertEqual(bundle.state.completeness, "partial")

    def test_lexical_coverage_is_assessed_per_chunk_not_concatenated(self) -> None:
        requirement = AnswerRequirementV2(
            id="r1",
            description="alpha beta",
        )
        bundle = assemble_evidence_bundle(
            query="alpha beta",
            candidates=[
                _candidate("alpha", content="alpha"),
                _candidate("beta", chunk_index=1, content="beta"),
            ],
            requirements=(requirement,),
            retrieval_queries=(),
            completeness="complete",
        )

        self.assertTrue(all(not item.supports_requirement_ids for item in bundle.items))
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))
        self.assertEqual(bundle.state.completeness, "unknown")

    def test_entity_overlap_alone_cannot_satisfy_compound_requirement(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "bridge",
                    content="职级分类：普通岗位对应D级。",
                )
            ],
            requirements=requirements,
            retrieval_queries=("普通岗位的餐饮补贴是多少",),
            completeness="complete",
        )

        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ("r2",))
        self.assertEqual(item.role, "bridge")
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_answer_query_index_cannot_promote_bridge_only_chunk(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "bridge",
                    content="职级分类：普通岗位对应D级。",
                    expansion_query_indexes=[0],
                )
            ],
            requirements=requirements,
            retrieval_queries=("普通岗位的餐饮补贴是多少",),
            completeness="complete",
        )

        item = bundle.items[0]
        self.assertEqual(item.supports_requirement_ids, ("r2",))
        self.assertEqual(item.role, "bridge")
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_multi_hop_joins_answer_seed_to_resolved_bridge_value(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "answer",
                    content="餐饮补贴：D级为100元/天。",
                    candidate_origins=["initial_retrieval"],
                ),
                _candidate(
                    "bridge",
                    chunk_index=1,
                    content="职级分类：普通岗位对应D级。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=("普通岗位的餐饮补贴是多少",),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["answer"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["answer"].role, "complement")
        self.assertEqual(by_id["bridge"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["bridge"].role, "bridge")
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(set(bundle.answer_source_ids), {"answer", "bridge"})

    def test_multi_hop_rejects_unjoined_answer_value(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通岗位对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="普通岗位的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "wrong-answer",
                    content="餐饮补贴：A级为200元/天。",
                    candidate_origins=["initial_retrieval"],
                ),
                _candidate(
                    "bridge",
                    chunk_index=1,
                    content="职级分类：普通岗位对应D级。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=("普通岗位的餐饮补贴是多少",),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["wrong-answer"].supports_requirement_ids, ())
        self.assertEqual(by_id["wrong-answer"].role, "background")
        self.assertEqual(bundle.answer_source_ids, ("bridge",))
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_cross_domain_implicit_mappings_join_only_the_resolved_value(self) -> None:
        cases = (
            (
                "合同工住宿标准",
                "合同工属于L2类。",
                "住宿标准：L2类为300元/天。",
            ),
            (
                "试用期年假天数",
                "试用期属于入职阶段P0。",
                "年假天数：P0为0天。",
            ),
            (
                "外包人员的系统权限是什么",
                "外包人员归属于访客角色R1。",
                "系统权限：R1仅可查看公开数据。",
            ),
        )

        from core.rag_v2.query_plan import plan_query_locally

        for question, bridge_content, answer_content in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)
                bundle = assemble_evidence_bundle(
                    query=question,
                    answer_shape=plan.answer_shape,
                    candidates=[
                        _candidate(
                            "answer",
                            content=answer_content,
                            candidate_origins=[
                                "initial_retrieval",
                                "small_document_full",
                            ],
                            full_document_chunk_count=2,
                        ),
                        _candidate(
                            "bridge",
                            chunk_index=1,
                            content=bridge_content,
                        ),
                    ],
                    requirements=plan.requirements,
                    retrieval_queries=plan.retrieval_queries,
                    completeness="complete",
                )

                self.assertEqual(bundle.missing_requirement_ids, ())
                self.assertEqual(
                    set(bundle.answer_source_ids),
                    {"answer", "bridge"},
                )

    def test_table_bridge_uses_only_subject_row_and_ignores_leave_approval(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的住宿标准是多少",
            "确认总经理对应的适用分类、等级、类别或阶段（用于确定住宿标准）",
        )
        classification = (
            "| 职级 | 适用人员 |\n"
            "| --- | --- |\n"
            "| A级 | 董事长、总经理、副总经理 |\n"
            "| B级 | 部门总监、高级经理 |\n"
            "| C级 | 部门经理、主管 |\n"
            "| D级 | 普通员工、专员 |"
        )
        leave_approval = (
            "| 请假时长 | 审批人 |\n"
            "| --- | --- |\n"
            "| 1天以内 | 直属主管 |\n"
            "| 5天以上 | 总经理 |"
        )

        self.assertEqual(
            extract_bridge_values(requirements[1].description, classification),
            ("A级",),
        )
        self.assertEqual(
            extract_bridge_values(requirements[1].description, leave_approval),
            (),
        )

        bundle = assemble_evidence_bundle(
            query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("classification", content=classification),
                _candidate(
                    "leave-approval",
                    doc_id="doc-leave",
                    content=leave_approval,
                ),
                _candidate(
                    "grade-a",
                    chunk_index=1,
                    content="住宿标准：A级一线城市不超过1200元/天。",
                ),
                _candidate(
                    "grade-d",
                    chunk_index=2,
                    content="住宿标准：D级一线城市不超过450元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["classification"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["grade-a"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["grade-d"].supports_requirement_ids, ())
        self.assertEqual(by_id["leave-approval"].supports_requirement_ids, ())
        self.assertEqual(
            set(bundle.answer_source_ids),
            {"classification", "grade-a"},
        )
        # Final typed evidence coverage is authoritative over a pipeline-time
        # partial ceiling once the complete bridge path is visible.
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_manager_mentions_do_not_manufacture_a_grade_mapping(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的住宿标准是多少",
            "确认总经理对应的适用分类、等级、类别或阶段（用于确定住宿标准）",
        )
        bundle = assemble_evidence_bundle(
            query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "approval",
                    content="A级、B级人员出差需总经理审批。",
                ),
                _candidate(
                    "appendix",
                    chunk_index=1,
                    content="本标准未尽事宜，由总经理办公会研究决定。",
                ),
                _candidate(
                    "grade-a",
                    chunk_index=2,
                    content="住宿标准：A级一线城市不超过1200元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        self.assertTrue(all(
            "r2" not in item.supports_requirement_ids for item in bundle.items
        ))
        self.assertEqual(bundle.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertNotEqual(bundle.state.completeness, "complete")

    def test_cross_domain_named_taxonomies_join_without_business_special_cases(self) -> None:
        cases = (
            (
                "星云产品的数据导出权限是什么",
                "确认星云产品对应的产品级别",
                "产品目录：星云产品属于企业版。",
                "数据导出权限：企业版允许导出业务数据。",
                "数据导出权限：基础版仅允许导出汇总数据。",
            ),
            (
                "供应商甲的风险处置措施是什么",
                "确认供应商甲对应的风险等级",
                "风险评估：供应商甲认定为高风险。",
                "风险处置措施：高风险供应商暂停准入并启动复核。",
                "风险处置措施：低风险供应商保持常规监测。",
            ),
            (
                "合同工的住宿标准是多少",
                "确认合同工对应的岗位等级",
                "用工分类：合同工属于L2类。",
                "住宿标准：L2类为300元/天。",
                "住宿标准：L3类为500元/天。",
            ),
        )

        for question, bridge_description, bridge, answer, wrong in cases:
            with self.subTest(question=question):
                requirements = _multi_hop_requirements(
                    question,
                    bridge_description,
                )
                bundle = assemble_evidence_bundle(
                    query=question,
                    answer_shape="multi_hop",
                    candidates=[
                        _candidate("bridge", content=bridge),
                        _candidate("answer", chunk_index=1, content=answer),
                        _candidate("wrong", chunk_index=2, content=wrong),
                    ],
                    requirements=requirements,
                    retrieval_queries=tuple(
                        item.description for item in requirements
                    ),
                    completeness="partial",
                )
                by_id = {item.chunk_id: item for item in bundle.items}
                self.assertEqual(by_id["bridge"].supports_requirement_ids, ("r2",))
                self.assertEqual(by_id["answer"].supports_requirement_ids, ("r1",))
                self.assertEqual(by_id["wrong"].supports_requirement_ids, ())
                self.assertEqual(bundle.missing_requirement_ids, ())
                self.assertEqual(bundle.state.completeness, "complete")

    def test_local_plan_and_evidence_join_named_risk_taxonomy_end_to_end(
        self,
    ) -> None:
        from core.rag_v2.query_plan import plan_query_locally

        question = "供应商甲的风险处置措施是什么"
        plan = plan_query_locally(question)
        candidates = [
            _candidate(
                "bridge",
                content="风险评估：供应商甲认定为高风险。",
                candidate_origins=["small_document_full"],
                full_document_chunk_count=3,
            ),
            _candidate(
                "answer",
                chunk_index=1,
                content="风险处置措施：高风险供应商暂停准入并启动复核。",
                candidate_origins=["small_document_full"],
                full_document_chunk_count=3,
            ),
            _candidate(
                "wrong",
                chunk_index=2,
                content="风险处置措施：低风险供应商保持常规监测。",
                candidate_origins=["small_document_full"],
                full_document_chunk_count=3,
            ),
        ]

        bundle = assemble_evidence_bundle(
            query=question,
            answer_shape=plan.answer_shape,
            candidates=candidates,
            requirements=plan.requirements,
            retrieval_queries=plan.retrieval_queries,
        )

        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual(set(bundle.answer_source_ids), {"bridge", "answer"})
        self.assertNotIn("wrong", bundle.answer_source_ids)

    def test_bridge_extraction_rejects_negation_exclusion_and_cross_claims(self) -> None:
        cases = (
            ("确认普通员工对应的职级", "普通员工不属于D级。"),
            ("确认普通员工对应的职级", "除普通员工外，其他人员属于D级。"),
            ("确认普通员工对应的职级", "除专员、普通员工及助理外，其他人员属于D级。"),
            ("确认普通员工对应的职级", "D级适用于除普通员工外的人员。"),
            ("确认星云产品对应的产品级别", "星云产品属于非企业版。"),
            ("确认供应商甲对应的风险等级", "供应商甲并非高风险。"),
            (
                "确认普通员工对应的职级",
                "普通员工信息如下。高级经理对应A级。",
            ),
            (
                "确认普通员工对应的职级",
                "普通员工名单见附件。供应商甲认定为高风险。",
            ),
        )

        for description, content in cases:
            with self.subTest(content=content):
                self.assertEqual(extract_bridge_values(description, content), ())

    def test_bridge_subject_is_an_exact_entity_or_table_list_item(self) -> None:
        description = "确认普通员工对应的职级"
        for content in (
            "非普通员工属于D级。",
            "高级普通员工属于D级。",
            "普通员工家属属于D级。",
            "| 职级 | 适用人员 |\n| --- | --- |\n| D级 | 非普通员工 |",
            "| 职级 | 适用人员 |\n| --- | --- |\n| D级 | 普通员工家属 |",
        ):
            with self.subTest(content=content):
                self.assertEqual(extract_bridge_values(description, content), ())

        self.assertEqual(
            extract_bridge_values(
                description,
                "| 职级 | 适用人员 |\n| --- | --- |\n"
                "| D级 | 专员、普通员工、助理 |",
            ),
            ("D级",),
        )
        for claim, expected in (
            ("普通员工的餐补为100元", True),
            ("普通员工可以查看数据", True),
            ("非普通员工的餐补为100元", False),
            ("高级普通员工的餐补为100元", False),
            ("普通员工家属的餐补为100元", False),
        ):
            with self.subTest(claim=claim):
                self.assertEqual(
                    content_contains_positive_subject(claim, "普通员工"),
                    expected,
                )
        self.assertTrue(content_contains_positive_subject(
            "总经理在北京的住宿上限为1200元",
            "北京",
        ))

    def test_uncanonicalized_bridge_never_falls_back_to_lexical_completion(
        self,
    ) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="供应商甲的风险处置措施",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="核对供应商甲的风险等级",
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        bundle = assemble_evidence_bundle(
            query="供应商甲的风险处置措施是什么",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "noise",
                    content="供应商甲风险等级申请流程已经发布。",
                ),
                _candidate(
                    "answer",
                    chunk_index=1,
                    content="风险处置措施：高风险供应商暂停准入。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=(
                "供应商甲的风险处置措施",
                "核对供应商甲的风险等级",
            ),
            completeness="complete",
        )

        self.assertTrue(all(
            "r2" not in item.supports_requirement_ids for item in bundle.items
        ))
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1", "r2"))
        self.assertNotEqual(bundle.state.completeness, "complete")

    def test_shorter_bridge_subject_cannot_erase_residual_query_context(self) -> None:
        """A taxonomy prefix is not the whole user entity/applicability scope."""

        family_requirements = _multi_hop_requirements(
            "普通员工家属的住宿标准是什么",
            "确认普通员工对应的职级",
        )
        family_expansion_queries = build_bridge_expansion_queries(
            family_requirements,
            [_candidate("family-query-bridge", content="普通员工对应D级。")],
        )
        self.assertEqual(len(family_expansion_queries), 1)
        self.assertIn("D级家属", family_expansion_queries[0])
        family_bundle = assemble_evidence_bundle(
            query="普通员工家属的住宿标准是什么",
            answer_shape="multi_hop",
            candidates=[
                _candidate("family-bridge", content="普通员工对应D级。"),
                _candidate(
                    "family-answer",
                    doc_id="family-answer-doc",
                    content="住宿标准：D级不超过450元/天。",
                    filename="公司住宿标准.docx",
                ),
                _candidate(
                    "employee-direct",
                    doc_id="employee-direct-doc",
                    content="普通员工的住宿标准为450元/天。",
                    filename="普通员工住宿标准.docx",
                ),
            ],
            requirements=family_requirements,
            retrieval_queries=tuple(
                item.description for item in family_requirements
            ),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in family_bundle.items}
        self.assertEqual(by_id["family-bridge"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["family-answer"].supports_requirement_ids, ())
        self.assertEqual(by_id["employee-direct"].supports_requirement_ids, ())
        self.assertEqual(family_bundle.missing_requirement_ids, ("r1",))
        self.assertEqual(family_bundle.state.completeness, "partial")

        # The same structural rule still permits a real activity context when
        # it is grounded by the candidate's document topic instead of guessed
        # away by the planner.
        travel_requirements = _multi_hop_requirements(
            "普通员工出差的住宿标准是什么",
            "确认普通员工对应的职级",
        )
        travel_bundle = assemble_evidence_bundle(
            query="普通员工出差的住宿标准是什么",
            answer_shape="multi_hop",
            candidates=[
                _candidate("travel-bridge", content="普通员工对应D级。"),
                _candidate(
                    "travel-answer",
                    doc_id="travel-answer-doc",
                    content="住宿标准：D级不超过450元/天。",
                    filename="公司出差管理标准.docx",
                ),
            ],
            requirements=travel_requirements,
            retrieval_queries=tuple(
                item.description for item in travel_requirements
            ),
            completeness="partial",
        )

        self.assertEqual(travel_bundle.missing_requirement_ids, ())
        self.assertEqual(travel_bundle.state.completeness, "complete")
        self.assertEqual(
            set(travel_bundle.answer_source_ids),
            {"travel-bridge", "travel-answer"},
        )

    def test_bridge_value_matching_uses_exact_positive_boundaries(self) -> None:
        cases = (
            ("R1仅可查看", "R1", True),
            ("R10仅可查看", "R1", False),
            ("D级100元/天", "D级", True),
            ("D级 100元/天", "D级", True),
            ("企业版允许导出", "企业版", True),
            ("非企业版允许导出", "企业版", False),
            ("高风险暂停准入", "高风险", True),
            ("非高风险暂停准入", "高风险", False),
        )
        for content, value, expected in cases:
            with self.subTest(content=content, value=value):
                self.assertEqual(
                    content_contains_bridge_value(content, value),
                    expected,
                )

    def test_target_normalization_preserves_business_action_names(self) -> None:
        cases = (
            ("申请权限是什么", "导出权限：允许导出"),
            ("查询权限是什么", "删除权限：允许删除"),
        )
        for question, unrelated in cases:
            with self.subTest(question=question):
                self.assertFalse(content_matches_answer_target(
                    question,
                    unrelated,
                    bridge_subjects=(),
                ))
                self.assertTrue(content_matches_answer_target(
                    question,
                    f"{question.removesuffix('是什么')}：已开启",
                    bridge_subjects=(),
                ))

        self.assertFalse(content_matches_answer_target(
            "普通员工的餐补金额是多少",
            "聚餐活动补录：D级为100元",
            bridge_subjects=("普通员工",),
        ))
        self.assertFalse(content_matches_answer_target(
            "普通员工的餐补金额是多少",
            "聚餐补录：D级为100元",
            bridge_subjects=("普通员工",),
        ))
        self.assertTrue(content_matches_answer_target(
            "普通员工的餐补金额是多少",
            "餐饮补贴：D级为100元",
            bridge_subjects=("普通员工",),
        ))

    def test_answer_target_bridge_and_result_must_share_one_claim(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的餐饮补贴是多少",
            "确认普通员工对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("bridge", content="普通员工对应D级。"),
                _candidate(
                    "same-claim",
                    chunk_index=1,
                    content="餐饮补贴：D级为100元/天。",
                ),
                _candidate(
                    "split-claim",
                    chunk_index=2,
                    content="餐饮补贴标准如下。D级为100元/天。",
                    filename="餐饮补贴标准.docx",
                ),
                _candidate(
                    "title-only",
                    chunk_index=3,
                    content="D级为100元/天。",
                    filename="餐饮补贴标准.docx",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["same-claim"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["split-claim"].supports_requirement_ids, ())
        self.assertEqual(by_id["title-only"].supports_requirement_ids, ())

    def test_resolved_answer_set_requires_all_bridges_in_one_claim(self) -> None:
        answer = AnswerRequirementV2(
            id="r1",
            description="总经理在北京的住宿标准是多少",
        )
        facts = (
            ResolvedBridgeFact(
                requirement_id="r2",
                subject="总经理",
                value="A级",
                source_chunk_id="manager-map",
                source_doc_id="doc-a",
                source_kb_id="kb-a",
            ),
            ResolvedBridgeFact(
                requirement_id="r3",
                subject="北京",
                value="一线城市",
                source_chunk_id="city-map",
                source_doc_id="doc-a",
                source_kb_id="kb-a",
            ),
        )
        same_sentence = _candidate(
            "answer",
            content="住宿标准：A级在一线城市不超过1200元/天。",
        )
        same_row = _candidate(
            "table-answer",
            content=(
                "## 住宿标准\n"
                "| 职级 | 城市类别 | 上限 |\n"
                "| --- | --- | --- |\n"
                "| A级 | 一线城市 | 1200元/天 |"
            ),
        )
        split_claims = _candidate(
            "split-answer",
            content=(
                "住宿标准：A级不超过1200元/天。"
                "住宿标准：一线城市不超过1200元/天。"
            ),
        )

        for candidate in (same_sentence, same_row):
            self.assertTrue(candidate_supports_resolved_answer_set(
                answer,
                candidate,
                facts,
                bridge_subjects=("总经理", "北京"),
            ))
        self.assertFalse(candidate_supports_resolved_answer_set(
            answer,
            split_claims,
            facts,
            bridge_subjects=("总经理", "北京"),
        ))

    def test_chunk_body_breadcrumb_is_table_semantic_context(self) -> None:
        """DOCX table chunks retain headings as body breadcrumbs, not metadata."""

        answer = AnswerRequirementV2(
            id="r1",
            description="负责人对应的住宿标准是多少",
            depends_on_requirement_ids=("r2",),
        )
        fact = ResolvedBridgeFact(
            requirement_id="r2",
            subject="负责人",
            value="P1级",
            source_chunk_id="role-map",
            source_doc_id="policy-doc",
            source_kb_id="kb-a",
        )
        candidate = _candidate(
            "lodging-table",
            doc_id="policy-doc",
            chunk_index=1,
            content=(
                "【差旅制度 › 住宿费用标准】\n"
                "| 职级 | 一线城市（元/天） | 二线城市（元/天） |\n"
                "| --- | --- | --- |\n"
                "| P1级 | ≤1200 | ≤800 |"
            ),
        )

        self.assertTrue(candidate_supports_resolved_answer_set(
            answer,
            candidate,
            (fact,),
            bridge_subjects=("负责人",),
        ))

    def test_resolved_table_claims_ignore_other_taxonomy_rows(self) -> None:
        """A matrix is not contradictory merely because other classes differ."""

        requirements = _multi_hop_requirements(
            "负责人对应的住宿标准是多少",
            "确认负责人对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="负责人对应的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("role-map", content="负责人对应P1级。"),
                _candidate(
                    "lodging-table",
                    chunk_index=1,
                    content=(
                        "【差旅制度 › 住宿费用标准】\n"
                        "| 职级 | 一线城市（元/天） |\n"
                        "| --- | --- |\n"
                        "| P1级 | ≤1200 |\n"
                        "| P2级 | ≤800 |"
                    ),
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["lodging-table"].supports_requirement_ids, ("r1",))
        self.assertNotIn("conflicting_active_answer_claims", bundle.state.reasons)

    def test_direct_self_contained_answer_bypasses_only_its_inferred_bridge(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的餐饮补贴是多少",
            "确认普通员工对应的职级",
        )
        direct = assemble_evidence_bundle(
            query="普通员工的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[_candidate(
                "direct",
                content="普通员工的餐饮补贴为100元/天。",
            )],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )
        split = assemble_evidence_bundle(
            query="普通员工的餐饮补贴是多少",
            answer_shape="multi_hop",
            candidates=[_candidate(
                "split",
                content="普通员工信息如下。餐饮补贴为100元/天。",
            )],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        self.assertEqual(direct.missing_requirement_ids, ())
        self.assertEqual(direct.answer_source_ids, ("direct",))
        self.assertEqual(split.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(split.answer_source_ids, ())
        for content in (
            "非普通员工的餐饮补贴为100元/天。",
            "高级普通员工的餐饮补贴为100元/天。",
            "普通员工家属的餐饮补贴为100元/天。",
        ):
            with self.subTest(content=content):
                embedded = assemble_evidence_bundle(
                    query="普通员工的餐饮补贴是多少",
                    answer_shape="multi_hop",
                    candidates=[_candidate("embedded", content=content)],
                    requirements=requirements,
                    retrieval_queries=tuple(
                        item.description for item in requirements
                    ),
                    completeness="partial",
                )
                self.assertIn("r2", embedded.missing_requirement_ids)

    def test_bridge_subject_in_title_cannot_create_document_root_anchor(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的出差标准是什么",
            "确认总经理对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="总经理的出差标准是什么",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "mapping",
                    content="总经理对应A级。",
                    filename="总经理请假制度.md",
                    candidate_origins=["initial_retrieval"],
                ),
                _candidate(
                    "approval",
                    chunk_index=1,
                    content="## 审批\nA级审批上限为5天。",
                    filename="总经理请假制度.md",
                    candidate_origins=["small_document_full"],
                ),
                _candidate(
                    "leave",
                    chunk_index=2,
                    content="## 休假\nA级每年可休10天。",
                    filename="总经理请假制度.md",
                    candidate_origins=["small_document_full"],
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["mapping"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["approval"].supports_requirement_ids, ())
        self.assertEqual(by_id["leave"].supports_requirement_ids, ())
        self.assertIn("r1", bundle.missing_requirement_ids)

    def test_same_document_does_not_override_explicit_scope_conflicts(self) -> None:
        dimensions = (
            ({"scope_products": ("alpha",)}, {"product": "beta"}),
            ({"scope_versions": ("8.2",)}, {"version": "8.6"}),
            ({"scope_projects": ("project-a",)}, {"project": "project-b"}),
        )
        for fact_scope, candidate_scope in dimensions:
            fact = ResolvedBridgeFact(
                requirement_id="r2",
                subject="对象甲",
                value="L2类",
                source_chunk_id="map",
                source_doc_id="same-doc",
                source_kb_id="kb-a",
                **fact_scope,
            )
            candidate = _candidate(
                "answer",
                doc_id="same-doc",
                metadata=candidate_scope,
            )
            with self.subTest(fact_scope=fact_scope):
                self.assertFalse(
                    bridge_fact_matches_candidate_scope(fact, candidate)
                )

    def test_bridge_expansion_queries_keep_each_explicit_scope(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的住宿标准是多少",
            "确认总经理对应的职级",
        )
        queries = build_bridge_expansion_queries(
            requirements,
            (
                _candidate(
                    "map-82",
                    doc_id="doc-82",
                    content="总经理对应A级。",
                    metadata={"product": "alpha", "version": "8.2"},
                ),
                _candidate(
                    "map-86",
                    doc_id="doc-86",
                    content="总经理对应B级。",
                    metadata={"product": "alpha", "version": "8.6"},
                ),
            ),
        )

        self.assertEqual(len(queries), 2)
        self.assertTrue(any("A级" in query and "8.2" in query for query in queries))
        self.assertTrue(any("B级" in query and "8.6" in query for query in queries))

    def test_conflicting_same_document_bridge_values_fail_closed(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的住宿标准是多少",
            "确认普通员工对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("map-a", content="普通员工对应A级。"),
                _candidate(
                    "map-d",
                    content="普通员工对应D级。",
                ),
                _candidate(
                    "answer-a",
                    chunk_index=1,
                    content="住宿标准：A级为1200元/天。",
                ),
                _candidate(
                    "answer-d",
                    chunk_index=1,
                    content="住宿标准：D级为450元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["map-a"].role, "conflicting")
        self.assertEqual(by_id["map-d"].role, "conflicting")
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(bundle.state.completeness, "unknown")

    def test_different_documents_keep_independent_complete_bridge_graphs(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的住宿标准是多少",
            "确认普通员工对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("map-a", content="普通员工对应A级。"),
                _candidate(
                    "answer-a",
                    chunk_index=1,
                    content="住宿标准：A级为1200元/天。",
                ),
                _candidate(
                    "map-d",
                    doc_id="doc-b",
                    content="普通员工对应D级。",
                ),
                _candidate(
                    "answer-d",
                    doc_id="doc-b",
                    chunk_index=1,
                    content="住宿标准：D级为450元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["map-a"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["answer-a"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["map-d"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["answer-d"].supports_requirement_ids, ("r1",))
        self.assertEqual(
            set(bundle.answer_source_ids),
            {"map-a", "answer-a", "map-d", "answer-d"},
        )
        self.assertEqual(bundle.missing_requirement_ids, ())

    def test_bridge_join_never_crosses_incompatible_document_version(self) -> None:
        requirements = _multi_hop_requirements(
            "总经理的住宿标准是多少",
            "确认总经理对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate(
                    "map-82",
                    doc_id="map-doc-82",
                    content="总经理对应A级。",
                    metadata={"version": "8.2"},
                ),
                _candidate(
                    "answer-82",
                    doc_id="policy-doc-82",
                    content="住宿标准：A级为1200元/天。",
                    metadata={"version": "8.2"},
                ),
                _candidate(
                    "answer-86",
                    doc_id="policy-doc-86",
                    content="住宿标准：A级为1600元/天。",
                    metadata={"version": "8.6"},
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["answer-82"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["answer-86"].supports_requirement_ids, ())
        self.assertEqual(
            set(bundle.answer_source_ids),
            {"map-82", "answer-82"},
        )

    def test_answer_target_abbreviation_matches_only_same_semantic_anchor(self) -> None:
        requirements = _multi_hop_requirements(
            "普通员工的餐补金额是多少",
            "确认普通员工对应的职级",
        )
        bundle = assemble_evidence_bundle(
            query="普通员工的餐补金额是多少",
            answer_shape="multi_hop",
            candidates=[
                _candidate("bridge", content="普通员工对应D级。"),
                _candidate(
                    "meal",
                    chunk_index=1,
                    content="餐饮补贴：D级为100元/天。",
                ),
                _candidate(
                    "lodging",
                    chunk_index=2,
                    content="住宿补贴：D级为450元/天。",
                ),
            ],
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="complete",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        self.assertEqual(by_id["meal"].supports_requirement_ids, ("r1",))
        self.assertEqual(by_id["lodging"].supports_requirement_ids, ())
        self.assertEqual(set(bundle.answer_source_ids), {"bridge", "meal"})

    def test_broad_manager_travel_question_keeps_all_value_bearing_sections(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="总经理的出差标准是什么",
                depends_on_requirement_ids=("r2",),
                coverage_mode="collection",
            ),
            AnswerRequirementV2(
                id="r2",
                description=(
                    "确认总经理对应的适用分类、等级、类别或阶段"
                    "（用于确定出差标准）"
                ),
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        filename = "公司出差管理标准.docx"
        full_document_chunk_count = 8
        candidates = [
            _candidate(
                "classification",
                content=(
                    "| 职级 | 适用人员 |\n"
                    "| --- | --- |\n"
                    "| A级 | 董事长、总经理、副总经理 |\n"
                    "| D级 | 普通员工、专员 |"
                ),
                filename=filename,
                candidate_origins=["initial_retrieval"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "flight",
                chunk_index=1,
                content="## 飞机\n| 职级 | 国内航班 |\n| --- | --- |\n| A级 | 头等舱或公务舱 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "train",
                chunk_index=2,
                content="## 火车\n| 职级 | 标准 |\n| --- | --- |\n| A级 | 高铁一等座、火车软卧 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "city",
                chunk_index=3,
                content="## 市内交通\n| 职级 | 标准 |\n| --- | --- |\n| A级 | 出租车、网约车、公务用车 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "lodging",
                chunk_index=4,
                content="## 住宿费用\n| 职级 | 一线城市 |\n| --- | --- |\n| A级 | 不超过1200元/天 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "meal",
                chunk_index=5,
                content="## 餐饮补贴\n| 职级 | 标准 |\n| --- | --- |\n| A级 | 200元/天 |",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "communication",
                chunk_index=6,
                content=(
                    "## 通讯补贴\n通讯补贴为50元/天，所有职级统一适用。"
                ),
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
            _candidate(
                "leave",
                doc_id="leave-doc",
                content="请假超过5天由总经理审批。",
                filename="员工请假管理办法.docx",
            ),
            _candidate(
                "appendix",
                chunk_index=7,
                content="本标准未尽事宜由总经理办公会研究决定。",
                filename=filename,
                candidate_origins=["small_document_full"],
                full_document_chunk_count=full_document_chunk_count,
            ),
        ]
        bundle = assemble_evidence_bundle(
            query="总经理的出差标准是什么",
            answer_shape="multi_hop",
            candidates=candidates,
            requirements=requirements,
            retrieval_queries=tuple(item.description for item in requirements),
            completeness="partial",
        )

        by_id = {item.chunk_id: item for item in bundle.items}
        answer_ids = {
            "flight",
            "train",
            "city",
            "lodging",
            "meal",
            "communication",
        }
        self.assertTrue(all(
            by_id[chunk_id].supports_requirement_ids == ("r1",)
            for chunk_id in answer_ids
        ))
        self.assertEqual(by_id["classification"].supports_requirement_ids, ("r2",))
        self.assertEqual(by_id["leave"].supports_requirement_ids, ())
        self.assertEqual(by_id["appendix"].supports_requirement_ids, ())
        self.assertEqual(set(bundle.answer_source_ids), answer_ids | {"classification"})
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_collection_requires_a_verified_full_snapshot(self) -> None:
        requirement = AnswerRequirementV2(
            id="r1",
            description="供应商管理要求是什么",
            depends_on_requirement_ids=(),
            coverage_mode="collection",
        )
        bundle = assemble_evidence_bundle(
            query="供应商管理要求是什么",
            answer_shape="overview",
            candidates=[
                _candidate(
                    "single-clause",
                    content="供应商管理要求：必须每年复审一次。",
                    candidate_origins=["initial_retrieval"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=(requirement.description,),
            completeness="complete",
        )

        self.assertEqual(bundle.answer_source_ids, ("single-clause",))
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))
        self.assertEqual(bundle.state.completeness, "partial")
        self.assertIn("collection_snapshot_unproven", bundle.state.reasons)

    def test_collection_includes_fragment_is_not_exhaustive(self) -> None:
        cases = (
            (
                "公司出差标准是什么",
                "公司出差标准：交通包括飞机、高铁。",
            ),
            (
                "供应商管理要求是什么",
                "供应商管理要求包括年度复审、审计留痕。",
            ),
        )

        for query, content in cases:
            with self.subTest(query=query):
                requirement = AnswerRequirementV2(
                    id="r1",
                    description=query,
                    coverage_mode="collection",
                )
                bundle = assemble_evidence_bundle(
                    query=query,
                    answer_shape="overview",
                    candidates=[
                        _candidate(
                            "partial-list",
                            content=content,
                            candidate_origins=["initial_retrieval"],
                        )
                    ],
                    requirements=(requirement,),
                    retrieval_queries=(query,),
                    completeness="complete",
                )

                self.assertEqual(bundle.answer_source_ids, ("partial-list",))
                self.assertEqual(bundle.missing_requirement_ids, ("r1",))
                self.assertEqual(bundle.state.completeness, "partial")
                self.assertIn(
                    "collection_snapshot_unproven",
                    bundle.state.reasons,
                )

    def test_collection_accepts_target_bound_exhaustive_enumeration(self) -> None:
        query = "供应商管理要求是什么"
        requirement = AnswerRequirementV2(
            id="r1",
            description=query,
            coverage_mode="collection",
        )
        bundle = assemble_evidence_bundle(
            query=query,
            answer_shape="overview",
            candidates=[
                _candidate(
                    "closed-list",
                    content=(
                        "供应商管理要求仅包括以下两项："
                        "年度复审、审计留痕。"
                    ),
                    candidate_origins=["initial_retrieval"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=(query,),
            completeness="partial",
        )

        self.assertEqual(bundle.answer_source_ids, ("closed-list",))
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_collection_accepts_target_bound_complete_process(self) -> None:
        query = "采购申请流程是什么"
        requirement = AnswerRequirementV2(
            id="r1",
            description=query,
            coverage_mode="collection",
        )
        bundle = assemble_evidence_bundle(
            query=query,
            answer_shape="overview",
            candidates=[
                _candidate(
                    "closed-process",
                    content=(
                        "采购申请流程如下：\n"
                        "1. 提交申请。\n"
                        "2. 负责人审批。\n"
                        "3. 系统归档。"
                    ),
                    candidate_origins=["initial_retrieval"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=(query,),
            completeness="partial",
        )

        self.assertEqual(bundle.answer_source_ids, ("closed-process",))
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_collection_requires_every_part_of_target_bound_table(self) -> None:
        query = "系统支持的登录方式有哪些"
        requirement = AnswerRequirementV2(
            id="r1",
            description=query,
            coverage_mode="collection",
        )
        candidates = [
            _candidate(
                f"login-part-{index}",
                chunk_index=index,
                content=(
                    "| 系统支持的登录方式 | 说明 |\n"
                    "| --- | --- |\n"
                    f"| {method} | {description} |"
                ),
                filename="系统使用手册.md",
                candidate_origins=[
                    "initial_retrieval" if index == 0 else "same_section"
                ],
                section_path=["系统使用手册.md", "系统支持的登录方式"],
                table_id="login-methods",
                table_part_index=index,
                table_part_count=2,
            )
            for index, (method, description) in enumerate((
                ("密码登录", "使用账号密码"),
                ("单点登录", "使用企业身份源"),
            ))
        ]

        complete = assemble_evidence_bundle(
            query=query,
            answer_shape="list",
            candidates=candidates,
            requirements=(requirement,),
            retrieval_queries=(query,),
            completeness="partial",
        )
        incomplete = assemble_evidence_bundle(
            query=query,
            answer_shape="list",
            candidates=candidates[:1],
            requirements=(requirement,),
            retrieval_queries=(query,),
            completeness="complete",
        )

        self.assertEqual(
            set(complete.answer_source_ids),
            {"login-part-0", "login-part-1"},
        )
        self.assertEqual(complete.missing_requirement_ids, ())
        self.assertEqual(complete.state.completeness, "complete")
        self.assertEqual(incomplete.missing_requirement_ids, ("r1",))
        self.assertEqual(incomplete.state.completeness, "partial")

    def test_collection_downgrades_when_context_budget_drops_one_clause(self) -> None:
        requirement = AnswerRequirementV2(
            id="r1",
            description="供应商管理要求是什么",
            depends_on_requirement_ids=(),
            coverage_mode="collection",
        )
        candidates = [
            _candidate(
                "review",
                chunk_index=0,
                content="## 复审要求\n供应商管理要求：每年复审一次。",
                filename="供应商管理要求.md",
                candidate_origins=["initial_retrieval", "small_document_full"],
                full_document_chunk_count=2,
            ),
            _candidate(
                "audit",
                chunk_index=1,
                content="## 审计要求\n供应商管理要求：必须保留审计记录三年。",
                filename="供应商管理要求.md",
                candidate_origins=["small_document_full"],
                full_document_chunk_count=2,
            ),
        ]
        bundle = assemble_evidence_bundle(
            query="供应商管理要求是什么",
            answer_shape="overview",
            candidates=candidates,
            requirements=(requirement,),
            retrieval_queries=(requirement.description,),
            completeness="complete",
            max_context_chunks=1,
        )

        self.assertEqual(len(bundle.answer_source_ids), 1)
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))
        self.assertEqual(bundle.state.completeness, "partial")
        self.assertIn("collection_context_incomplete", bundle.state.reasons)

    def test_required_coverage_precedes_higher_scored_background_under_budget(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="alpha target"),
            AnswerRequirementV2(id="r2", description="beta target"),
        )
        bundle = assemble_evidence_bundle(
            query="two targets",
            candidates=[
                _candidate("background", content="generic notes", score=100),
                _candidate(
                    "r1",
                    chunk_index=1,
                    content="alpha target: first mapped result",
                    score=0.1,
                    role="direct",
                    supports_requirement_ids=["r1"],
                ),
                _candidate(
                    "r2",
                    chunk_index=2,
                    content="beta target: second mapped result",
                    score=0.1,
                    role="direct",
                    supports_requirement_ids=["r2"],
                ),
            ],
            requirements=requirements,
            retrieval_queries=("alpha target", "beta target"),
            rerank_succeeded=True,
            completeness="complete",
            max_context_chunks=2,
        )

        self.assertEqual(bundle.context_item_ids, ("r1", "r2"))
        self.assertEqual(bundle.answer_source_ids, ("r1", "r2"))
        self.assertEqual(bundle.missing_requirement_ids, ())
        self.assertEqual(bundle.state.completeness, "complete")

    def test_invalid_query_indexes_cannot_manufacture_requirement_support(self) -> None:
        requirement = AnswerRequirementV2(id="r1", description="目标要求")
        bundle = assemble_evidence_bundle(
            query="目标要求",
            candidates=[
                _candidate(
                    "invalid-indexes",
                    content="完全无关正文",
                    expansion_query_indexes=[True, -1, 9, "bad"],
                )
            ],
            requirements=(requirement,),
            retrieval_queries=("目标要求",),
            completeness="complete",
        )

        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.items[0].role, "background")
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_unexecuted_reranker_cannot_mark_candidate_verified_or_direct(self) -> None:
        bundle = assemble_evidence_bundle(
            query="配置",
            candidates=[
                _candidate(
                    "legacy",
                    content="配置说明",
                    rerank_status="verified",
                    evidence_role="direct",
                )
            ],
            rerank_succeeded=None,
        )

        self.assertEqual(bundle.items[0].confidence, "retrieved")
        self.assertEqual(bundle.items[0].role, "background")

    def test_stale_support_annotations_are_ignored_without_verification(self) -> None:
        requirement = AnswerRequirementV2(id="r1", description="fresh target")
        bundle = assemble_evidence_bundle(
            query="fresh target",
            candidates=[
                _candidate(
                    "stale",
                    content="old unrelated context",
                    role="direct",
                    supports_requirement_ids=["r1"],
                    rerank_status="verified",
                )
            ],
            requirements=(requirement,),
            retrieval_queries=("fresh target",),
            rerank_succeeded=None,
            completeness="complete",
        )

        self.assertEqual(bundle.items[0].supports_requirement_ids, ())
        self.assertEqual(bundle.items[0].role, "background")
        self.assertEqual(bundle.answer_source_ids, ())
        self.assertEqual(bundle.missing_requirement_ids, ("r1",))

    def test_dependency_edge_is_coverage_critical_for_every_answer_shape(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="answer target",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="bridge target",
                role="bridge",
                importance="helpful",
                source="inferred",
            ),
        )
        values = dict(
            query="resolve target",
            candidates=[
                _candidate(
                    "answer",
                    content="answer target: mapped value",
                    expansion_query_indexes=[0],
                )
            ],
            requirements=requirements,
            retrieval_queries=("answer target", "bridge target"),
            completeness="complete",
        )

        multi_hop = assemble_evidence_bundle(answer_shape="multi_hop", **values)
        ordinary = assemble_evidence_bundle(answer_shape="multi_part", **values)

        self.assertEqual(multi_hop.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(multi_hop.state.completeness, "unknown")
        self.assertEqual(ordinary.missing_requirement_ids, ("r1", "r2"))
        self.assertEqual(ordinary.state.completeness, "unknown")

    def test_hard_mismatch_and_unauthorized_candidates_are_excluded(self) -> None:
        constraints = extract_query_constraints("云枢8.2.75消息接口怎么配置")
        candidates = [
            _candidate(
                "a-2",
                chunk_index=2,
                content="所属产品：云枢；产品版本：8.2.75。配置项B。",
                rerank_status="verified",
            ),
            _candidate(
                "b-0",
                doc_id="doc-b",
                content="所属产品：云枢；产品版本：7.0。旧配置项。",
                rerank_status="verified",
            ),
            _candidate(
                "a-0",
                chunk_index=0,
                content="所属产品：云枢；产品版本：8.2.75。",
                rerank_status="verified",
            ),
            _candidate("secret", doc_id="doc-secret", authorized=False),
        ]

        bundle = assemble_evidence_bundle(
            query="云枢8.2.75消息接口怎么配置",
            candidates=candidates,
            constraints=constraints,
            rerank_succeeded=True,
        )

        self.assertEqual([item.chunk_id for item in bundle.items], ["a-0", "a-2"])
        self.assertTrue(all(item.doc_id == "doc-a" for item in bundle.items))
        self.assertIn("hard_constraint_mismatch_excluded", bundle.state.reasons)
        self.assertIn("unauthorized_candidate_excluded", bundle.state.reasons)
        self.assertEqual(bundle.state.confidence, "verified")

    def test_explicit_product_version_excludes_unknown_scope(self) -> None:
        constraints = extract_query_constraints("云枢8.6登录配置")
        bundle = assemble_evidence_bundle(
            query="云枢8.6登录配置",
            candidates=[
                _candidate(
                    "exact",
                    content="所属产品：云枢；产品版本：8.6。登录配置A。",
                ),
                _candidate(
                    "unknown",
                    doc_id="doc-generic",
                    content="通用登录配置B。",
                ),
            ],
            constraints=constraints,
        )

        self.assertEqual([item.chunk_id for item in bundle.items], ["exact"])
        self.assertIn("hard_constraint_unknown_excluded", bundle.state.reasons)

    def test_items_are_grouped_by_document_and_sorted_by_chunk_index(self) -> None:
        bundle = assemble_evidence_bundle(
            query="配置标准",
            candidates=[
                _candidate("a-3", chunk_index=3),
                _candidate("b-2", doc_id="doc-b", chunk_index=2),
                _candidate("a-1", chunk_index=1),
                _candidate("b-0", doc_id="doc-b", chunk_index=0),
            ],
        )

        self.assertEqual(
            [(item.doc_id, item.chunk_index) for item in bundle.items],
            [("doc-a", 1), ("doc-a", 3), ("doc-b", 0), ("doc-b", 2)],
        )

    def test_rerank_failure_downgrades_but_does_not_erase_candidates(self) -> None:
        bundle = assemble_evidence_bundle(
            query="普通员工住宿标准",
            candidates=[
                _candidate(
                    "retrieved",
                    evidence_role="irrelevant",
                    rerank_status="unverified",
                )
            ],
            rerank_succeeded=False,
            completeness="complete",
        )

        self.assertEqual(bundle.state.availability, "degraded")
        self.assertEqual(bundle.state.confidence, "retrieved")
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual([item.chunk_id for item in bundle.context_items], ["retrieved"])
        self.assertEqual(bundle.answer_sources, ())
        self.assertIn("rerank_degraded", bundle.state.reasons)

    def test_expansion_degradation_is_independent_from_confidence_and_completeness(self) -> None:
        bundle = assemble_evidence_bundle(
            query="普通员工住宿标准",
            candidates=[
                _candidate("verified", rerank_status="verified")
            ],
            rerank_succeeded=True,
            expansion_succeeded=False,
            completeness="complete",
        )

        self.assertEqual(bundle.state.availability, "degraded")
        self.assertEqual(bundle.state.confidence, "verified")
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual(len(bundle.items), 1)
        self.assertIn("expansion_degraded", bundle.state.reasons)

    def test_missing_requirements_mark_partial_without_clearing_context(self) -> None:
        bundle = assemble_evidence_bundle(
            query="交通和住宿标准",
            candidates=[_candidate("住宿")],
            completeness="complete",
            missing_requirement_ids=("transport",),
        )

        self.assertEqual(bundle.state.completeness, "partial")
        self.assertEqual(bundle.missing_requirement_ids, ("transport",))
        self.assertEqual(bundle.context_item_ids, ("住宿",))

    def test_concrete_table_outranks_boilerplate_unless_explicitly_requested(self) -> None:
        candidates = [
            _candidate(
                "overview",
                chunk_index=0,
                content="一、总则\n为规范公司员工出差管理，制定本制度。",
                evidence_role="direct",
            ),
            _candidate(
                "table",
                chunk_index=1,
                content="| 职级 | 住宿标准 |\n| --- | --- |\n| D级 | 450元/天 |",
                evidence_role="related",
            ),
        ]

        concrete = assemble_evidence_bundle(
            query="普通员工住宿标准是多少",
            candidates=candidates,
            max_context_chunks=1,
        )
        overview = assemble_evidence_bundle(
            query="这份制度的总则是什么",
            candidates=candidates,
            answer_shape="overview",
            requirements=(
                AnswerRequirementV2(
                    id="r1",
                    description="这份制度的总则是什么",
                ),
            ),
            retrieval_queries=("这份制度的总则是什么",),
            max_context_chunks=1,
        )

        self.assertEqual(concrete.context_item_ids, ("table",))
        self.assertEqual(overview.context_item_ids, ("overview",))

    def test_overview_accepts_only_full_document_chunks_anchored_by_retrieval(self) -> None:
        bundle = assemble_evidence_bundle(
            query="请概述这份制度的主要内容",
            answer_shape="overview",
            candidates=[
                _candidate("anchor", content="公司出差管理标准")
            ],
            overview_candidates=[
                _candidate("full-1", chunk_index=1, content="一、总则：规范出差管理。"),
                _candidate(
                    "foreign",
                    doc_id="doc-foreign",
                    content="不属于已授权召回文档的全文。",
                ),
            ],
            requirements=(
                AnswerRequirementV2(
                    id="r1",
                    description="请概述这份制度的主要内容",
                ),
            ),
            retrieval_queries=("请概述这份制度的主要内容",),
            rerank_succeeded=True,
        )

        self.assertEqual(
            {item.chunk_id for item in bundle.items},
            {"anchor", "full-1"},
        )
        full = next(item for item in bundle.items if item.chunk_id == "full-1")
        self.assertEqual(full.confidence, "retrieved")
        self.assertIn("overview_full_document", full.origins)
        self.assertIn("unanchored_overview_candidate_excluded", bundle.state.reasons)

    def test_context_chunk_and_character_budgets_are_hard_limits(self) -> None:
        bundle = assemble_evidence_bundle(
            query="标准",
            candidates=[
                _candidate("first", content="1234567890", score=1.0),
                _candidate("second", chunk_index=1, content="abcdefghij", score=0.5),
            ],
            completeness="complete",
            max_context_chunks=1,
            max_context_chars=5,
        )

        self.assertEqual(bundle.context_item_ids, ("first",))
        self.assertEqual(bundle.context_items[0].content, "12345")
        self.assertTrue(bundle.context_items[0].metadata["context_truncated"])
        self.assertIn("context_budget_limited", bundle.state.reasons)
        self.assertEqual(bundle.state.completeness, "partial")

    def test_all_filtered_candidates_are_a_normal_empty_result(self) -> None:
        bundle = assemble_evidence_bundle(
            query="云枢8.2.75配置",
            candidates=[
                _candidate("wrong", constraint_status="mismatch")
            ],
        )

        self.assertEqual(bundle.state.availability, "ok")
        self.assertEqual(bundle.state.confidence, "none")
        self.assertEqual(bundle.items, ())
        self.assertEqual(bundle.context_items, ())

    def test_empty_retrieval_is_no_evidence_not_infrastructure_unavailable(self) -> None:
        bundle = assemble_evidence_bundle(
            query="知识库中不存在的问题",
            candidates=[],
        )

        self.assertEqual(bundle.state.availability, "ok")
        self.assertEqual(bundle.state.confidence, "none")
        self.assertEqual(bundle.state.completeness, "unknown")
        self.assertIn("no_usable_authorized_evidence", bundle.state.reasons)

    def test_source_metadata_needed_by_sse_is_preserved(self) -> None:
        bundle = assemble_evidence_bundle(
            query="配置",
            candidates=[
                _candidate(
                    "source",
                    filename="配置说明.md",
                    file_type="markdown",
                    source_url="https://kb.example/doc/source",
                    doc_tags=["安全", "配置"],
                    retrieval_score=0.82,
                    answer_support=0.91,
                )
            ],
        )

        metadata = bundle.items[0].metadata
        self.assertEqual(metadata["filename"], "配置说明.md")
        self.assertEqual(metadata["file_type"], "markdown")
        self.assertEqual(metadata["source_url"], "https://kb.example/doc/source")
        self.assertEqual(metadata["doc_tags"], ["安全", "配置"])
        self.assertEqual(metadata["retrieval_score"], 0.82)
        self.assertEqual(metadata["answer_support"], 0.91)


if __name__ == "__main__":
    unittest.main()
