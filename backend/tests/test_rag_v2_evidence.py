import unittest

from core.query_constraints import extract_query_constraints
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


class EvidenceBundleAssemblyTests(unittest.TestCase):
    def test_normal_v2_shape_requires_typed_requirements(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty requirements"):
            assemble_evidence_bundle(
                query="查询标准",
                answer_shape="fact",
                candidates=[_candidate("seed")],
            )

    def test_multi_hop_shape_requires_bridge_requirement(self) -> None:
        with self.assertRaisesRegex(ValueError, "bridge requirement"):
            assemble_evidence_bundle(
                query="对象对应的额度是多少",
                answer_shape="multi_hop",
                candidates=[_candidate("seed")],
                requirements=(
                    AnswerRequirementV2(
                        id="r1",
                        description="对象对应的额度是多少",
                    ),
                ),
            )

    def test_single_requirement_uses_current_retrieval_seed_as_complement(self) -> None:
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

        self.assertEqual(bundle.items[0].role, "complement")
        self.assertEqual(bundle.items[0].supports_requirement_ids, ("r1",))
        self.assertEqual(bundle.answer_source_ids, ("seed",))

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
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工出差的交通标准是多少",
            ),
            AnswerRequirementV2(
                id="r3",
                description="普通员工出差的餐补标准是多少",
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
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "transport",
                    chunk_index=1,
                    content="D级交通标准：飞机经济舱、高铁二等座。",
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
        self.assertEqual(bundle.state.completeness, "partial")

    def test_merged_query_indexes_keep_each_visible_coordinated_answer(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工出差的住宿标准是多少",
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工出差的交通标准是多少",
            ),
            AnswerRequirementV2(
                id="r3",
                description="普通员工出差的餐补标准是多少",
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
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "transport",
                    chunk_index=1,
                    content="D级交通标准：飞机经济舱、高铁二等座。",
                    expansion_query_indexes=all_query_indexes,
                ),
                _candidate(
                    "meal",
                    chunk_index=2,
                    content="D级餐补标准：每天100元。",
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
            AnswerRequirementV2(id="r1", description="查询普通岗位的餐补金额"),
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
                    content="无词面重叠的正文",
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
        self.assertEqual(bundle.state.completeness, "partial")

    def test_entity_overlap_alone_cannot_satisfy_compound_requirement(self) -> None:
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="查询普通岗位的餐饮补贴金额",
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
                            candidate_origins=["initial_retrieval"],
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
                    content="first mapped result",
                    score=0.1,
                    role="direct",
                    supports_requirement_ids=["r1"],
                ),
                _candidate(
                    "r2",
                    chunk_index=2,
                    content="second mapped result",
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

    def test_multi_hop_helpful_bridge_is_coverage_critical(self) -> None:
        requirements = (
            AnswerRequirementV2(id="r1", description="answer target"),
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
                    content="mapped by current retrieval",
                    expansion_query_indexes=[0],
                )
            ],
            requirements=requirements,
            retrieval_queries=("answer target", "bridge target"),
            completeness="complete",
        )

        multi_hop = assemble_evidence_bundle(answer_shape="multi_hop", **values)
        ordinary = assemble_evidence_bundle(answer_shape="fact", **values)

        self.assertEqual(multi_hop.missing_requirement_ids, ("r2",))
        self.assertEqual(multi_hop.state.completeness, "partial")
        self.assertEqual(ordinary.missing_requirement_ids, ())
        self.assertEqual(ordinary.state.completeness, "complete")

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
