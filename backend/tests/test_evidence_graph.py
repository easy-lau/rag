import unittest
from dataclasses import replace

from core.rag_v2.contracts import (
    AnswerRequirementV2,
    BridgeClaimBinding,
    EvidenceBundle,
    EvidenceClaim,
    EvidenceItem,
    EvidenceState,
    VerifiedCollectionClosure,
)
from core.rag_v2.evidence_graph import (
    assess_evidence_coverage_graph,
    build_evidence_coverage_graph,
    derive_verified_collection_closures,
)


def _item(
    chunk_id: str,
    content: str,
    *,
    doc_id: str = "travel-policy",
    kb_id: str = "kb-travel",
    chunk_index: int = 0,
    role: str = "background",
    contribution_kind: str | None = None,
    supports: tuple[str, ...] = (),
    section_key: str | None = None,
    origins: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> EvidenceItem:
    values = dict(metadata or {})
    if section_key is not None:
        values["section_key"] = section_key
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        kb_id=kb_id,
        content=content,
        chunk_index=chunk_index,
        role=role,  # type: ignore[arg-type]
        contribution_kind=contribution_kind,  # type: ignore[arg-type]
        supports_requirement_ids=supports,
        origins=origins,
        metadata=values,
    )


def _bundle(
    items: tuple[EvidenceItem, ...],
    *,
    visible_item_ids: tuple[str, ...] | None = None,
) -> EvidenceBundle:
    return EvidenceBundle(
        state=EvidenceState(
            availability="ok",
            confidence="verified",
            completeness="partial",
        ),
        items=items,
        context_item_ids=(
            visible_item_ids
            if visible_item_ids is not None
            else tuple(item.chunk_id for item in items)
        ),
    )


def _assessment_for(assessment, requirement_id: str):
    return next(
        value
        for value in assessment.requirement_assessments
        if value.requirement_id == requirement_id
    )


def _manager_requirements():
    return (
        AnswerRequirementV2(
            id="r1",
            description="总经理的完整出差标准",
            coverage_mode="collection",
            coverage_contract="document_policy",
            depends_on_requirement_ids=("r2",),
        ),
        AnswerRequirementV2(
            id="r2",
            description="确认总经理对应的职级",
            role="bridge",
            importance="helpful",
            source="inferred",
            bridge_subject="总经理",
            bridge_kind="classification",
        ),
    )


def _manager_policy_items() -> tuple[EvidenceItem, ...]:
    full_origin = ("small_document_full",)
    full_count = 8
    return (
        _item(
            "classification",
            "职级分类：总经理对应A级。",
            chunk_index=0,
            role="bridge",
            contribution_kind="bridge_fact",
            supports=("r2",),
            section_key="classification",
            origins=full_origin,
            metadata={
                "full_document_chunk_count": full_count,
                "document_policy_root_requirement_ids": ["r1"],
            },
        ),
        _item(
            "air",
            "A级国内航班可乘头等舱或公务舱。",
            chunk_index=1,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="transport",
            origins=full_origin,
            metadata={"full_document_chunk_count": full_count},
        ),
        _item(
            "hotel",
            "A级一线城市住宿不超过1200元/天。",
            chunk_index=2,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="lodging",
            origins=full_origin,
            metadata={"full_document_chunk_count": full_count},
        ),
        _item(
            "hotel-note",
            "注：一线城市包括北京、上海、广州、深圳。",
            chunk_index=3,
            section_key="lodging",
            origins=full_origin,
            metadata={"full_document_chunk_count": full_count},
        ),
        _item(
            "meal",
            "A级餐饮补贴为200元/天。",
            chunk_index=4,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="meal",
            origins=full_origin,
            metadata={"full_document_chunk_count": full_count},
        ),
        _item(
            "meal-note",
            "注：不足一天按实际用餐次数比例计算。",
            chunk_index=5,
            section_key="meal",
            origins=full_origin,
            metadata={"full_document_chunk_count": full_count},
        ),
        _item(
            "universal-subsidy",
            "通讯补贴和出差补贴为所有职级统一标准。",
            chunk_index=6,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="subsidy",
            origins=full_origin,
            metadata={"full_document_chunk_count": full_count},
        ),
        _item(
            "approval",
            "A级人员出差需总经理审批。",
            chunk_index=7,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="approval",
            origins=full_origin,
            metadata={"full_document_chunk_count": full_count},
        ),
    )


def _manager_policy_claims() -> tuple[EvidenceClaim, ...]:
    """The fixture carries real graph claims, never legacy join metadata."""

    binding = BridgeClaimBinding("r2", "classification", "A级")
    return (
        EvidenceClaim(
            id="manager-grade",
            requirement_id="r2",
            evidence_item_id="classification",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="bridge_fact",
            applicability="bridge_value",
        ),
        *(
            EvidenceClaim(
                id=f"manager-{item_id}",
                requirement_id="r1",
                evidence_item_id=item_id,
                document_key=("kb-travel", "travel-policy"),
                contribution_kind="answer_claim",
                applicability="bridge_value",
                bridge_bindings=(binding,),
            )
            for item_id in ("air", "hotel", "meal", "approval")
        ),
        EvidenceClaim(
            id="manager-universal-subsidy",
            requirement_id="r1",
            evidence_item_id="universal-subsidy",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="document_universal",
        ),
    )


class EvidenceCoverageGraphTests(unittest.TestCase):
    def test_manager_policy_closes_bridge_groups_notes_and_full_snapshot(self):
        requirements = _manager_requirements()
        graph = build_evidence_coverage_graph(
            _bundle(_manager_policy_items()),
            requirements,
            claims=_manager_policy_claims(),
        )
        assessment = assess_evidence_coverage_graph(graph)

        self.assertEqual(assessment.completeness, "complete")
        self.assertEqual(_assessment_for(assessment, "r1").completeness, "complete")
        self.assertEqual(graph.document_root_keys["r1"], ("kb-travel", "travel-policy"))
        self.assertIn(("kb-travel", "travel-policy"), graph.complete_document_keys)

    def test_hidden_section_notes_make_the_policy_partial(self):
        items = _manager_policy_items()
        visible = tuple(
            item.chunk_id
            for item in items
            if item.chunk_id not in {"hotel-note", "meal-note"}
        )
        graph = build_evidence_coverage_graph(
            _bundle(items, visible_item_ids=visible),
            _manager_requirements(),
            claims=_manager_policy_claims(),
        )
        assessment = assess_evidence_coverage_graph(graph)
        r1 = _assessment_for(assessment, "r1")

        self.assertEqual(assessment.completeness, "partial")
        self.assertEqual(r1.completeness, "partial")
        self.assertTrue({"hotel-note", "meal-note"}.issubset(r1.missing_item_ids))
        self.assertIn("document_policy_snapshot_incomplete", r1.reasons)

    def test_regular_employee_meal_requires_exact_d_grade_bridge_and_note(self):
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的餐饮补贴是多少",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通员工对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        bridge = _item(
            "grade",
            "职级分类：普通员工、专员对应D级。",
            role="bridge",
            contribution_kind="bridge_fact",
            supports=("r2",),
            section_key="classification",
        )
        meal = _item(
            "meal",
            "D级餐饮补贴100元/天。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="meal",
        )
        note = _item(
            "meal-note",
            "注：不足一天按实际用餐次数比例计算。",
            section_key="meal",
        )
        graph = build_evidence_coverage_graph(
            _bundle((bridge, meal, note)),
            requirements,
            claims=(
                EvidenceClaim(
                    id="employee-grade",
                    requirement_id="r2",
                    evidence_item_id="grade",
                    document_key=("kb-travel", "travel-policy"),
                    contribution_kind="bridge_fact",
                    applicability="bridge_value",
                ),
                EvidenceClaim(
                    id="employee-meal",
                    requirement_id="r1",
                    evidence_item_id="meal",
                    document_key=("kb-travel", "travel-policy"),
                    contribution_kind="answer_claim",
                    applicability="bridge_value",
                    bridge_bindings=(
                        BridgeClaimBinding("r2", "grade", "D级"),
                    ),
                ),
            ),
        )
        assessment = assess_evidence_coverage_graph(graph)

        self.assertEqual(assessment.completeness, "complete")
        self.assertEqual(_assessment_for(assessment, "r1").completeness, "complete")

    def test_document_universal_claim_does_not_depend_on_unused_grade_bridge(self):
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的出差补贴是多少",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通员工对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        subsidy = _item(
            "subsidy",
            "出差补贴100元/天，所有职级统一。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="subsidy",
        )
        graph = build_evidence_coverage_graph(
            _bundle((subsidy,)),
            requirements,
            claims=(EvidenceClaim(
                id="universal-subsidy",
                requirement_id="r1",
                evidence_item_id="subsidy",
                document_key=("kb-travel", "travel-policy"),
                contribution_kind="answer_claim",
                applicability="document_universal",
            ),),
        )
        assessment = assess_evidence_coverage_graph(graph)

        self.assertEqual(assessment.completeness, "complete")
        self.assertEqual(_assessment_for(assessment, "r1").completeness, "complete")

    def test_condition_bound_direct_clause_closes_without_optional_condition_bridge(self):
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="偏远地区出差有什么补贴",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="偏远地区对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="偏远地区",
                bridge_kind="condition",
            ),
        )
        source_clause = _item(
            "remote-clause",
            "偏远地区或艰苦地区出差，可申请额外补贴，标准另行审批。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="special-region",
        )
        graph = build_evidence_coverage_graph(
            _bundle((source_clause,)),
            requirements,
            claims=(EvidenceClaim(
                id="remote-clause-claim",
                requirement_id="r1",
                evidence_item_id="remote-clause",
                document_key=("kb-travel", "travel-policy"),
                contribution_kind="answer_claim",
                applicability="condition_bound",
            ),),
        )
        assessment = assess_evidence_coverage_graph(graph)

        self.assertEqual(assessment.completeness, "complete")
        self.assertEqual(_assessment_for(assessment, "r1").completeness, "complete")

    def test_explicit_cross_section_condition_binds_only_its_target_group(self):
        requirements = (AnswerRequirementV2(id="r1", description="住宿标准"),)
        lodging = _item(
            "lodging",
            "一线城市住宿标准450元/天。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="lodging",
        )
        city_condition = _item(
            "city-condition",
            "一线城市包括北京、上海、广州、深圳。",
            contribution_kind="qualifier",
            section_key="city-definition",
            metadata={"condition_for_section_key": "lodging"},
        )
        unrelated = _item(
            "unrelated-condition",
            "二线城市包括省会城市。",
            contribution_kind="qualifier",
            section_key="city-definition-2",
            metadata={"condition_for_section_key": "meal"},
        )
        claim = EvidenceClaim(
            id="lodging-claim",
            requirement_id="r1",
            evidence_item_id="lodging",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="condition_bound",
            condition_group_id="section:kb-travel:travel-policy:city-definition",
        )
        graph = build_evidence_coverage_graph(
            _bundle((lodging, city_condition, unrelated)),
            requirements,
            claims=(claim,),
        )
        assessment = assess_evidence_coverage_graph(graph)
        lodging_group = next(
            group for group in graph.structural_groups
            if group.id == "section:kb-travel:travel-policy:lodging"
        )

        self.assertEqual(lodging_group.condition_item_ids, ("city-condition",))
        self.assertNotIn("unrelated-condition", lodging_group.condition_item_ids)
        self.assertEqual(_assessment_for(assessment, "r1").completeness, "complete")

    def test_leave_document_manager_approval_cannot_satisfy_travel_policy(self):
        requirements = _manager_requirements()
        travel_bridge = _item(
            "travel-grade",
            "总经理对应A级。",
            role="bridge",
            contribution_kind="bridge_fact",
            supports=("r2",),
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 1},
        )
        leave_approval = _item(
            "leave-approval",
            "总经理审批请假申请。",
            doc_id="leave-policy",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="approval",
        )
        leave_claim = EvidenceClaim(
            id="leave-claim",
            requirement_id="r1",
            evidence_item_id="leave-approval",
            document_key=("kb-travel", "leave-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
        )
        graph = build_evidence_coverage_graph(
            _bundle((travel_bridge, leave_approval)),
            requirements,
            claims=(leave_claim,),
            document_root_keys={"r1": ("kb-travel", "travel-policy")},
        )
        assessment = assess_evidence_coverage_graph(graph)
        r1 = _assessment_for(assessment, "r1")

        self.assertEqual(r1.completeness, "partial")
        self.assertEqual(r1.supporting_claim_ids, ())
        self.assertIn("no_typed_answer_claim", r1.reasons)

    def test_unbound_note_in_a_multi_table_section_cannot_cross_bind_tables(self):
        requirements = (
            AnswerRequirementV2(id="r1", description="住宿补贴"),
            AnswerRequirementV2(id="r2", description="餐饮补贴"),
        )
        lodging_table = _item(
            "lodging-table",
            "住宿补贴表。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="allowances",
            metadata={"table_id": "lodging", "table_part_index": 0, "table_part_count": 1},
        )
        meal_table = _item(
            "meal-table",
            "餐饮补贴表。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r2",),
            section_key="allowances",
            metadata={"table_id": "meal", "table_part_index": 0, "table_part_count": 1},
        )
        note = _item(
            "unbound-note",
            "注：以上标准按实际天数折算。",
            section_key="allowances",
        )
        graph = build_evidence_coverage_graph(
            _bundle((lodging_table, meal_table, note), visible_item_ids=("lodging-table", "meal-table")),
            requirements,
            claims=(
                EvidenceClaim(
                    id="lodging-table-claim",
                    requirement_id="r1",
                    evidence_item_id="lodging-table",
                    document_key=("kb-travel", "travel-policy"),
                    contribution_kind="answer_claim",
                    applicability="direct_subject",
                ),
                EvidenceClaim(
                    id="meal-table-claim",
                    requirement_id="r2",
                    evidence_item_id="meal-table",
                    document_key=("kb-travel", "travel-policy"),
                    contribution_kind="answer_claim",
                    applicability="direct_subject",
                ),
            ),
        )
        assessment = assess_evidence_coverage_graph(graph)

        self.assertEqual(assessment.completeness, "complete")
        self.assertTrue(all(
            "unbound-note" not in group.companion_item_ids
            for group in graph.structural_groups
        ))

    def test_document_policy_requires_a_complete_visible_snapshot(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="完整制度",
            coverage_mode="collection",
            coverage_contract="document_policy",
        )
        root = _item(
            "root",
            "公司制度正文。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 3},
        )
        graph = build_evidence_coverage_graph(
            _bundle((root,)),
            (requirement,),
            claims=(EvidenceClaim(
                id="document-root-claim",
                requirement_id="r1",
                evidence_item_id="root",
                document_key=("kb-travel", "travel-policy"),
                contribution_kind="answer_claim",
                applicability="direct_subject",
            ),),
            document_root_keys={"r1": ("kb-travel", "travel-policy")},
        )
        assessment = assess_evidence_coverage_graph(graph)
        r1 = _assessment_for(assessment, "r1")

        self.assertEqual(r1.completeness, "partial")
        self.assertIn("document_policy_snapshot_incomplete", r1.reasons)

    def test_verified_collection_closure_closes_only_its_requirement_scope(self):
        """A verified answer scope may omit an irrelevant document appendix."""

        requirement = AnswerRequirementV2(
            id="r1",
            description="完整差旅标准",
            coverage_mode="collection",
            coverage_contract="document_policy",
        )
        flight = _item(
            "flight",
            "飞机标准。",
            chunk_index=0,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 3},
        )
        lodging = _item(
            "lodging",
            "住宿标准。",
            chunk_index=1,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 3},
        )
        appendix = _item(
            "appendix",
            "本标准未尽事宜另行解释。",
            chunk_index=2,
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 3},
        )
        closure = VerifiedCollectionClosure(
            requirement_id="r1",
            claim_item_ids=("flight", "lodging"),
            source_kind="full_document_snapshot",
            source_document_key=("kb-travel", "travel-policy"),
        )
        graph = build_evidence_coverage_graph(
            _bundle(
                (flight, lodging, appendix),
                visible_item_ids=("flight", "lodging"),
            ),
            (requirement,),
            claims=(
                EvidenceClaim(
                    id="flight-claim",
                    requirement_id="r1",
                    evidence_item_id="flight",
                    document_key=("kb-travel", "travel-policy"),
                    contribution_kind="answer_claim",
                    applicability="direct_subject",
                ),
                EvidenceClaim(
                    id="lodging-claim",
                    requirement_id="r1",
                    evidence_item_id="lodging",
                    document_key=("kb-travel", "travel-policy"),
                    contribution_kind="answer_claim",
                    applicability="direct_subject",
                ),
            ),
            document_root_keys={"r1": ("kb-travel", "travel-policy")},
            collection_closures=(closure,),
        )

        self.assertEqual(
            _assessment_for(assess_evidence_coverage_graph(graph), "r1").completeness,
            "complete",
        )

    def test_verified_collection_closure_fails_when_any_claim_is_not_visible(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="系统支持的登录方式有哪些",
            coverage_mode="collection",
            coverage_contract="structured_collection",
        )
        password = _item(
            "password-row",
            "| 登录方式 | 说明 |\n| --- | --- |\n| 密码登录 | 使用账号密码登录 |",
            chunk_index=0,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            metadata={
                "table_id": "login-methods",
                "table_part_index": 0,
                "table_part_count": 2,
            },
        )
        sso = _item(
            "sso-row",
            "| 单点登录 | 通过企业身份提供方登录 |",
            chunk_index=1,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            metadata={
                "table_id": "login-methods",
                "table_part_index": 1,
                "table_part_count": 2,
            },
        )
        closure = VerifiedCollectionClosure(
            requirement_id="r1",
            claim_item_ids=("password-row", "sso-row"),
            source_kind="complete_table",
            source_document_key=("kb-travel", "travel-policy"),
            source_table_key="table:kb-travel:travel-policy:login-methods",
        )
        claims = (
            EvidenceClaim(
                id="password-claim",
                requirement_id="r1",
                evidence_item_id="password-row",
                document_key=("kb-travel", "travel-policy"),
                contribution_kind="answer_claim",
                applicability="direct_subject",
            ),
            EvidenceClaim(
                id="sso-claim",
                requirement_id="r1",
                evidence_item_id="sso-row",
                document_key=("kb-travel", "travel-policy"),
                contribution_kind="answer_claim",
                applicability="direct_subject",
            ),
        )
        graph = build_evidence_coverage_graph(
            _bundle((password, sso), visible_item_ids=("password-row",)),
            (requirement,),
            claims=claims,
            collection_closures=(closure,),
        )

        assessment = _assessment_for(assess_evidence_coverage_graph(graph), "r1")
        self.assertEqual(assessment.completeness, "partial")
        self.assertIn("sso-row", assessment.missing_item_ids)

    def test_collection_closure_cannot_reference_an_untyped_item(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="完整差旅标准",
            coverage_mode="collection",
        )
        answer = _item(
            "answer",
            "交通标准。",
            chunk_index=0,
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 2},
        )
        appendix = _item(
            "appendix",
            "无关附录。",
            chunk_index=1,
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 2},
        )
        closure = VerifiedCollectionClosure(
            requirement_id="r1",
            claim_item_ids=("answer", "appendix"),
            source_kind="full_document_snapshot",
            source_document_key=("kb-travel", "travel-policy"),
        )

        with self.assertRaisesRegex(ValueError, "typed answer claims"):
            build_evidence_coverage_graph(
                _bundle((answer, appendix)),
                (requirement,),
                collection_closures=(closure,),
            )

    def test_unregistered_terminology_cannot_claim_strict_proof(self):
        requirement = AnswerRequirementV2(id="r1", description="餐饮补贴")
        candidate = _item(
            "meal",
            "餐补100元/天。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            metadata={"claim_proof_kind": {"r1": "terminology_strict"}},
        )
        derived = build_evidence_coverage_graph(_bundle((candidate,)), (requirement,))
        self.assertEqual(
            _assessment_for(assess_evidence_coverage_graph(derived), "r1").completeness,
            "partial",
        )

        strict_claim = EvidenceClaim(
            id="strict-meal",
            requirement_id="r1",
            evidence_item_id="meal",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
            proof_kind="terminology_strict",
            strict_terminology_rule_ids=("term_meal_allowance",),
        )
        graph = build_evidence_coverage_graph(
            _bundle((candidate,)),
            (requirement,),
            claims=(strict_claim,),
        )
        self.assertEqual(
            _assessment_for(assess_evidence_coverage_graph(graph), "r1").completeness,
            "complete",
        )

    def test_untrusted_metadata_cannot_manufacture_a_graph_claim(self):
        """Only an explicit EvidenceClaim is positive graph proof.

        These fields were formerly consumed by the ``claims=None`` fallback.
        They are deliberately realistic projections so this protects against a
        future caller reintroducing one of the legacy metadata channels.
        """

        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的住宿标准是多少",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通员工对应的职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        mapping = _item(
            "forged-grade",
            "普通员工对应D级。",
            role="bridge",
            contribution_kind="bridge_fact",
            supports=("r2",),
            metadata={
                "resolved_bridge_joins": [{
                    "answer_requirement_id": "r1",
                    "bridge_requirement_id": "r2",
                    "bridge_source_chunk_id": "forged-grade",
                    "bridge_value": "D级",
                }],
            },
        )
        amount = _item(
            "forged-amount",
            "D级住宿标准为450元/天。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            metadata={
                "claim_applicability": {"r1": "bridge_value"},
                "claim_proof_kind": {"r1": "terminology_strict"},
                "strict_terminology_rule_ids": {"r1": ["forged-rule"]},
                "answer_claim_assertions": {
                    "r1": [{
                        "status": "active",
                        "result_kind": "scalar",
                        "normalized_result": "450元/天",
                        "claim_key": "住宿标准",
                    }],
                },
            },
        )

        graph = build_evidence_coverage_graph(
            _bundle((mapping, amount)),
            requirements,
        )
        assessment = assess_evidence_coverage_graph(graph)

        self.assertEqual(graph.claims, ())
        self.assertEqual(assessment.completeness, "partial")
        self.assertEqual(_assessment_for(assessment, "r1").completeness, "partial")
        self.assertIn(
            "no_typed_answer_claim",
            _assessment_for(assessment, "r1").reasons,
        )

    def test_closed_semantic_claims_expose_conflict_without_merging_routes(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="普通员工的住宿标准是多少",
        )
        grade_a = _item(
            "policy-a",
            "住宿标准：A级为1200元/天。",
            doc_id="policy-a",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        grade_d = _item(
            "policy-d",
            "住宿标准：D级为450元/天。",
            doc_id="policy-d",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        graph = build_evidence_coverage_graph(
            _bundle((grade_a, grade_d)),
            (requirement,),
            claims=(
                EvidenceClaim(
                    id="a-claim",
                    requirement_id="r1",
                    evidence_item_id="policy-a",
                    document_key=("kb-travel", "policy-a"),
                    contribution_kind="answer_claim",
                    applicability="direct_subject",
                    result_kind="scalar",
                    normalized_result="1200元/天",
                    claim_key="住宿标准",
                ),
                EvidenceClaim(
                    id="d-claim",
                    requirement_id="r1",
                    evidence_item_id="policy-d",
                    document_key=("kb-travel", "policy-d"),
                    contribution_kind="answer_claim",
                    applicability="direct_subject",
                    result_kind="scalar",
                    normalized_result="450元/天",
                    claim_key="住宿标准",
                ),
            ),
        )
        assessment = assess_evidence_coverage_graph(graph)

        self.assertEqual(assessment.completeness, "complete")
        self.assertEqual(len(assessment.answer_conflicts), 1)
        conflict = assessment.answer_conflicts[0]
        self.assertEqual(conflict.requirement_id, "r1")
        self.assertEqual(conflict.claim_key, "住宿标准")
        self.assertEqual(
            set(conflict.normalized_results),
            {"1200元/天", "450元/天"},
        )
        self.assertEqual(set(conflict.claim_ids), {"a-claim", "d-claim"})

    def test_same_anchor_conflicting_scalar_claims_remain_a_conflict(self):
        """Route aggregation must not hide a contradiction inside one source.

        Post-evidence document projection may aggregate complementary typed
        assertions from the same source route.  The coverage graph still owns
        scalar/categorical conflict detection, including two incompatible
        values carried by the same physical source item.
        """

        requirement = AnswerRequirementV2(
            id="r1",
            description="普通员工的餐补标准是多少",
        )
        policy = _item(
            "same-source-policy",
            "普通员工餐补标准存在两个互斥金额。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        graph = build_evidence_coverage_graph(
            _bundle((policy,)),
            (requirement,),
            claims=(
                EvidenceClaim(
                    id="same-source-100",
                    requirement_id="r1",
                    evidence_item_id="same-source-policy",
                    document_key=("kb-travel", "travel-policy"),
                    contribution_kind="answer_claim",
                    applicability="direct_subject",
                    result_kind="scalar",
                    normalized_result="100元/天",
                    claim_key="餐补标准",
                ),
                EvidenceClaim(
                    id="same-source-200",
                    requirement_id="r1",
                    evidence_item_id="same-source-policy",
                    document_key=("kb-travel", "travel-policy"),
                    contribution_kind="answer_claim",
                    applicability="direct_subject",
                    result_kind="scalar",
                    normalized_result="200元/天",
                    claim_key="餐补标准",
                ),
            ),
        )
        assessment = assess_evidence_coverage_graph(graph)

        self.assertEqual(len(assessment.answer_conflicts), 1)
        conflict = assessment.answer_conflicts[0]
        self.assertEqual(conflict.claim_key, "餐补标准")
        self.assertEqual(
            set(conflict.normalized_results),
            {"100元/天", "200元/天"},
        )
        self.assertEqual(
            set(conflict.claim_ids),
            {"same-source-100", "same-source-200"},
        )

    def test_configuration_values_merge_when_equal_and_conflict_when_different(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="默认密码强制修改应该如何配置",
        )
        left = _item(
            "config-left",
            "force_change_default_password: true",
            doc_id="config-left",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        right = _item(
            "config-right",
            "force_change_default_password: false",
            doc_id="config-right",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )

        def assessment(right_value: str):
            graph = build_evidence_coverage_graph(
                _bundle((left, right)),
                (requirement,),
                claims=(
                    EvidenceClaim(
                        id="config-left-claim",
                        requirement_id="r1",
                        evidence_item_id="config-left",
                        document_key=("kb-travel", "config-left"),
                        contribution_kind="answer_claim",
                        applicability="direct_subject",
                        result_kind="config_assignment",
                        normalized_result="force_change_default_password=true",
                        claim_key="force_change_default_password",
                    ),
                    EvidenceClaim(
                        id="config-right-claim",
                        requirement_id="r1",
                        evidence_item_id="config-right",
                        document_key=("kb-travel", "config-right"),
                        contribution_kind="answer_claim",
                        applicability="direct_subject",
                        result_kind="config_assignment",
                        normalized_result=(
                            f"force_change_default_password={right_value}"
                        ),
                        claim_key="force_change_default_password",
                    ),
                ),
            )
            return assess_evidence_coverage_graph(graph)

        self.assertEqual(assessment("true").answer_conflicts, ())
        conflicts = assessment("false").answer_conflicts
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].result_kind, "config_assignment")
        self.assertEqual(
            set(conflicts[0].normalized_results),
            {
                "force_change_default_password=true",
                "force_change_default_password=false",
            },
        )

    def test_bridge_value_claim_must_bind_every_declared_dependency(self):
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="目标金额",
                depends_on_requirement_ids=("r2", "r3"),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认职级",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="员工",
                bridge_kind="classification",
            ),
            AnswerRequirementV2(
                id="r3",
                description="确认地区",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="地区",
                bridge_kind="condition",
            ),
        )
        answer = _item(
            "answer",
            "D级一线城市金额。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        incomplete_binding = EvidenceClaim(
            id="answer-claim",
            requirement_id="r1",
            evidence_item_id="answer",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="bridge_value",
            bridge_bindings=(BridgeClaimBinding("r2", "grade", "D级"),),
        )

        with self.assertRaisesRegex(ValueError, "declared edge path"):
            build_evidence_coverage_graph(
                _bundle((answer,)),
                requirements,
                claims=(incomplete_binding,),
            )

    def test_independent_direct_claim_is_a_distinct_route_from_a_bridge_value(self):
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="目标金额",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认法定地区",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="地区",
                bridge_kind="condition",
            ),
        )
        answer = _item(
            "answer",
            "某地区金额。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        direct_claim = EvidenceClaim(
            id="answer-claim",
            requirement_id="r1",
            evidence_item_id="answer",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
        )

        graph = build_evidence_coverage_graph(
            _bundle((answer,)),
            requirements,
            claims=(direct_claim,),
        )

        self.assertEqual(
            _assessment_for(assess_evidence_coverage_graph(graph), "r1").completeness,
            "complete",
        )

    def test_condition_bound_claim_cannot_borrow_a_condition_from_another_document(self):
        requirement = AnswerRequirementV2(id="r1", description="住宿标准")
        lodging = _item(
            "lodging",
            "住宿标准450元。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            section_key="lodging",
        )
        foreign_condition = _item(
            "foreign-city-definition",
            "一线城市包括北京。",
            doc_id="leave-policy",
            contribution_kind="qualifier",
            section_key="city-definition",
        )
        invalid_claim = EvidenceClaim(
            id="lodging-claim",
            requirement_id="r1",
            evidence_item_id="lodging",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="condition_bound",
            condition_group_id="section:kb-travel:leave-policy:city-definition",
        )

        with self.assertRaisesRegex(ValueError, "cannot borrow conditions across documents"):
            build_evidence_coverage_graph(
                _bundle((lodging, foreign_condition)),
                (requirement,),
                claims=(invalid_claim,),
            )

    def test_graph_is_optional_on_bundle_and_uses_immutable_mappings(self):
        requirement = AnswerRequirementV2(id="r1", description="补贴标准")
        evidence_item = _item(
            "subsidy",
            "补贴标准100元。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        bundle = _bundle((evidence_item,))
        graph = build_evidence_coverage_graph(bundle, (requirement,))
        bound_bundle = replace(bundle, coverage_graph=graph)

        self.assertIs(bound_bundle.coverage_graph, graph)
        self.assertNotIn("coverage_graph", bundle.to_dict())
        with self.assertRaises(TypeError):
            graph.evidence_document_keys["subsidy"] = ("other", "doc")

    def test_single_claim_contract_closes_with_one_direct_typed_claim(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="普通员工餐饮补贴是多少",
            coverage_contract="single_claim",
        )
        item = _item(
            "meal",
            "普通员工餐饮补贴为100元/天。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        graph = build_evidence_coverage_graph(
            _bundle((item,)),
            (requirement,),
            claims=(EvidenceClaim(
                id="meal-claim",
                requirement_id="r1",
                evidence_item_id="meal",
                document_key=("kb-travel", "travel-policy"),
                contribution_kind="answer_claim",
                applicability="direct_subject",
            ),),
        )

        assessment = assess_evidence_coverage_graph(graph)
        self.assertEqual(assessment.completeness, "complete")
        self.assertEqual(_assessment_for(assessment, "r1").completeness, "complete")

    def test_structured_collection_cannot_close_from_full_document_snapshot_alone(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="系统支持的登录方式有哪些",
            coverage_mode="collection",
            coverage_contract="structured_collection",
        )
        item = _item(
            "snapshot",
            "系统支持的登录方式说明。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 1},
        )
        claim = EvidenceClaim(
            id="login-methods-claim",
            requirement_id="r1",
            evidence_item_id="snapshot",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
        )
        preliminary = build_evidence_coverage_graph(
            _bundle((item,)),
            (requirement,),
            claims=(claim,),
        )

        self.assertIn(("kb-travel", "travel-policy"), preliminary.complete_document_keys)
        self.assertEqual(derive_verified_collection_closures(preliminary), ())
        assessment = assess_evidence_coverage_graph(preliminary)
        self.assertEqual(_assessment_for(assessment, "r1").completeness, "partial")
        self.assertIn(
            "structured_collection_closure_unproven",
            _assessment_for(assessment, "r1").reasons,
        )

    def test_ordered_steps_requires_sequence_declaration_not_a_complete_table(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="采购申请流程是什么",
            coverage_mode="collection",
            coverage_contract="ordered_steps",
        )
        table = _item(
            "steps-table",
            "| 流程 | 操作 |\n| --- | --- |\n| 采购申请 | 提交申请 |",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            metadata={
                "table_id": "purchase-steps",
                "table_part_index": 0,
                "table_part_count": 1,
            },
        )
        table_claim = EvidenceClaim(
            id="steps-table-claim",
            requirement_id="r1",
            evidence_item_id="steps-table",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
        )
        table_graph = build_evidence_coverage_graph(
            _bundle((table,)),
            (requirement,),
            claims=(table_claim,),
        )
        self.assertEqual(derive_verified_collection_closures(table_graph), ())
        self.assertEqual(
            _assessment_for(
                assess_evidence_coverage_graph(table_graph),
                "r1",
            ).completeness,
            "partial",
        )

        source = _item(
            "steps-source",
            "采购申请流程如下：\n1. 提交申请。\n2. 负责人审批。\n3. 系统归档。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        source_claim = EvidenceClaim(
            id="steps-source-claim",
            requirement_id="r1",
            evidence_item_id="steps-source",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
        )
        preliminary = build_evidence_coverage_graph(
            _bundle((source,)),
            (requirement,),
            claims=(source_claim,),
        )
        closures = derive_verified_collection_closures(preliminary)
        self.assertEqual(len(closures), 1)
        self.assertEqual(closures[0].source_kind, "source_declaration")
        closed_graph = build_evidence_coverage_graph(
            _bundle((source,)),
            (requirement,),
            claims=(source_claim,),
            collection_closures=closures,
        )
        self.assertEqual(
            _assessment_for(
                assess_evidence_coverage_graph(closed_graph),
                "r1",
            ).completeness,
            "complete",
        )

    def test_document_policy_requires_rooted_full_document_snapshot_not_table(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="完整公司出差管理标准",
            coverage_mode="collection",
            coverage_contract="document_policy",
        )
        table = _item(
            "travel-table",
            "| 职级 | 餐补 |\n| --- | --- |\n| D级 | 100元/天 |",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            metadata={
                "table_id": "travel-meal",
                "table_part_index": 0,
                "table_part_count": 1,
            },
        )
        table_claim = EvidenceClaim(
            id="travel-table-claim",
            requirement_id="r1",
            evidence_item_id="travel-table",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
        )
        table_graph = build_evidence_coverage_graph(
            _bundle((table,)),
            (requirement,),
            claims=(table_claim,),
            document_root_keys={"r1": ("kb-travel", "travel-policy")},
        )
        self.assertEqual(derive_verified_collection_closures(table_graph), ())
        table_assessment = _assessment_for(
            assess_evidence_coverage_graph(table_graph),
            "r1",
        )
        self.assertEqual(table_assessment.completeness, "partial")
        self.assertIn("document_policy_snapshot_incomplete", table_assessment.reasons)

        snapshot = _item(
            "travel-policy-root",
            "公司出差管理标准完整正文。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            origins=("small_document_full",),
            metadata={"full_document_chunk_count": 1},
        )
        snapshot_claim = EvidenceClaim(
            id="travel-policy-root-claim",
            requirement_id="r1",
            evidence_item_id="travel-policy-root",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
        )
        preliminary = build_evidence_coverage_graph(
            _bundle((snapshot,)),
            (requirement,),
            claims=(snapshot_claim,),
            document_root_keys={"r1": ("kb-travel", "travel-policy")},
        )
        closures = derive_verified_collection_closures(preliminary)
        self.assertEqual(len(closures), 1)
        self.assertEqual(closures[0].source_kind, "full_document_snapshot")
        closed_graph = build_evidence_coverage_graph(
            _bundle((snapshot,)),
            (requirement,),
            claims=(snapshot_claim,),
            document_root_keys={"r1": ("kb-travel", "travel-policy")},
            collection_closures=closures,
        )
        self.assertEqual(
            _assessment_for(
                assess_evidence_coverage_graph(closed_graph),
                "r1",
            ).completeness,
            "complete",
        )

    def test_document_root_is_rejected_for_single_claim_contract(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="餐补标准是多少",
            coverage_contract="single_claim",
        )
        item = _item(
            "meal",
            "餐补标准为100元/天。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
        )
        with self.assertRaisesRegex(ValueError, "document-policy"):
            build_evidence_coverage_graph(
                _bundle((item,)),
                (requirement,),
                document_root_keys={"r1": ("kb-travel", "travel-policy")},
            )

    def test_generic_facet_anchor_cannot_become_document_policy_root(self):
        requirement = AnswerRequirementV2(
            id="r1",
            description="完整公司出差管理标准",
            coverage_mode="collection",
            coverage_contract="document_policy",
        )
        lodging_facet = _item(
            "lodging-facet",
            "住宿费用标准：D级不超过450元/天。",
            role="direct",
            contribution_kind="answer_claim",
            supports=("r1",),
            metadata={
                "section_key": "lodging",
                # This broader topic-inheritance annotation is intentionally
                # not a policy-root certificate.
                "document_root_answer_requirement_ids": ["r1"],
            },
        )
        facet_claim = EvidenceClaim(
            id="lodging-facet-claim",
            requirement_id="r1",
            evidence_item_id="lodging-facet",
            document_key=("kb-travel", "travel-policy"),
            contribution_kind="answer_claim",
            applicability="direct_subject",
        )
        graph = build_evidence_coverage_graph(
            _bundle((lodging_facet,)),
            (requirement,),
            claims=(facet_claim,),
        )
        assessment = _assessment_for(assess_evidence_coverage_graph(graph), "r1")
        self.assertNotIn("r1", graph.document_root_keys)
        self.assertEqual(assessment.completeness, "partial")
        self.assertIn("document_policy_root_unproven", assessment.reasons)


if __name__ == "__main__":
    unittest.main()
