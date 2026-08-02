import unittest

from core.conversation_context import detect_followup
from core.query_surface_structure import (
    current_turn_candidate_targets_are_complete,
    has_current_turn_local_enumeration_antecedent,
    is_elliptical_current_clause_target,
    is_procedure_question,
    parse_query_surface_frame,
    parse_distributive_enumeration,
)
from core.rag_v2.query_plan import plan_query_locally


class CurrentTurnSurfaceStructureTests(unittest.TestCase):
    def test_explicit_distributive_forms_share_one_parser_boundary(self) -> None:
        cases = {
            "住宿和餐补这些分别回答": ("住宿", "餐补"),
            "住宿和餐补这些分别说明一下": ("住宿", "餐补"),
            "住宿还有餐补这些各项如何配置": ("住宿", "餐补"),
            "普通员工的住宿标准和餐补还有出差补贴这些分别是多少": (
                "普通员工的住宿标准",
                "餐补",
                "出差补贴",
            ),
        }
        for question, expected_parts in cases.items():
            with self.subTest(question=question):
                structure = parse_distributive_enumeration(question)
                self.assertIsNotNone(structure)
                assert structure is not None
                self.assertEqual(structure.parts, expected_parts)
                self.assertTrue(structure.has_local_anaphora_antecedent)
                self.assertEqual(
                    detect_followup(
                        question,
                        has_previous_turn=True,
                        previous_user_question="登录用户名枚举如何配置",
                    )[0],
                    False,
                )
                self.assertEqual(plan_query_locally(question).answer_shape, "multi_part")

    def test_local_list_can_block_history_inheritance_without_forcing_split(self) -> None:
        for question in (
            "住宿、餐补和出差补贴这些是多少",
            "住宿和餐补这些配置分别如何处理",
            "住宿和餐补那些配置分别如何处理",
        ):
            with self.subTest(question=question):
                self.assertTrue(has_current_turn_local_enumeration_antecedent(question))
                self.assertFalse(
                    detect_followup(
                        question,
                        has_previous_turn=True,
                        previous_user_question="登录用户名枚举如何配置",
                    )[0]
                )

    def test_historical_or_incomplete_forms_never_claim_a_local_antecedent(self) -> None:
        cases = (
            "风险还有哪些分别是什么",
            "上面的住宿和餐补这些分别是多少",
            "这些配置分别如何处理",
            "住宿和餐补这些分别",
        )
        for question in cases:
            with self.subTest(question=question):
                self.assertFalse(has_current_turn_local_enumeration_antecedent(question))

        self.assertTrue(
            detect_followup(
                "上面的住宿和餐补这些分别是多少",
                has_previous_turn=True,
                previous_user_question="普通员工的出差标准是什么",
            )[0]
        )
        self.assertTrue(
            detect_followup(
                "这些配置分别如何处理",
                has_previous_turn=True,
                previous_user_question="登录用户名枚举如何配置",
            )[0]
        )

    def test_local_anaphora_never_invents_a_suffix(self) -> None:
        plan = plan_query_locally("住宿、餐补还有出差补贴这些分别是多少")

        self.assertEqual(
            plan.retrieval_queries,
            ("住宿是多少", "餐补是多少", "出差补贴是多少"),
        )
        self.assertNotIn("住宿补贴", plan.retrieval_queries)

    def test_shared_subject_bridge_attaches_to_each_policy_sibling(self) -> None:
        plan = plan_query_locally("普通员工的交通、住宿和餐补这些分别是多少")
        answers = [item for item in plan.requirements if item.role == "answer"]

        self.assertEqual(len(answers), 3)
        self.assertEqual(
            [item.depends_on_requirement_ids for item in answers],
            [(), (), ()],
        )
        self.assertEqual(
            [item.augmentation_requirement_ids for item in answers],
            [("r4",), ("r4",), ("r4",)],
        )


class QuerySurfaceFrameTests(unittest.TestCase):
    """The frame is grammar only: no KB lookup or business-rule inference."""

    def test_question_shell_is_removed_without_losing_the_business_head(self) -> None:
        cases = {
            "偏远地区出差有什么补贴": (
                "出差补贴",
                ("偏远地区", "出差"),
                (("偏远地区", "condition"),),
                "enumeration",
            ),
            "艰苦地区出差有什么补贴": (
                "出差补贴",
                ("艰苦地区", "出差"),
                (("艰苦地区", "condition"),),
                "enumeration",
            ),
            "连续超过30天出差有什么补贴": (
                "出差补贴",
                ("连续超过30天", "出差"),
                (("连续超过30天", "condition"),),
                "enumeration",
            ),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                frame = parse_query_surface_frame(question)
                self.assertEqual(frame.answer_target, expected[0])
                self.assertEqual(frame.context_terms, expected[1])
                self.assertEqual(
                    tuple((item.text, item.kind) for item in frame.qualifiers),
                    expected[2],
                )
                self.assertEqual(frame.question_operator, expected[3])

    def test_entity_scope_and_explicit_relation_are_typed_from_surface_form(self) -> None:
        cases = {
            "普通员工的餐补是多少": (
                "餐补",
                (("普通员工", "entity"),),
                "value",
            ),
            "供应商甲的风险处置措施是什么": (
                "风险处置措施",
                (("供应商甲", "entity"),),
                "value",
            ),
            "客户A的服务策略是什么": (
                "服务策略",
                (("客户A", "entity"),),
                "value",
            ),
            "D级的餐补是多少": (
                "餐补",
                (("D级", "scope"),),
                "value",
            ),
            "普通员工对应什么职级": (
                "职级",
                (("普通员工", "entity"),),
                "relation",
            ),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                frame = parse_query_surface_frame(question)
                self.assertEqual(frame.answer_target, expected[0])
                self.assertEqual(
                    tuple((item.text, item.kind) for item in frame.qualifiers),
                    expected[1],
                )
                self.assertEqual(frame.question_operator, expected[2])

    def test_how_question_is_not_by_itself_a_procedure_contract(self) -> None:
        """Interrogative form is not enough to demand process closure.

        A question about how a conditional standard is disposed of is one
        policy claim.  In contrast, a concrete operational action can request
        a process.  This boundary is shared with the planner so it cannot be
        changed by an evidence-layer special case.
        """

        cases = {
            "如何配置VPN": True,
            "采购申请流程是什么": True,
            "连续出差超过30天住宿标准如何处理": False,
            "超出报销标准如何处理": False,
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                frame = parse_query_surface_frame(question)
                self.assertIsNotNone(frame)
                self.assertEqual(is_procedure_question(question, frame=frame), expected)

    def test_source_anchored_candidates_must_cover_the_complete_local_list(self) -> None:
        question = "普通员工的住宿标准、餐补和出差补贴这些分别是多少"
        employee = (question.index("普通员工"), question.index("普通员工") + len("普通员工"))
        targets = tuple(
            (question.index(value), question.index(value) + len(value))
            for value in ("住宿标准", "餐补", "出差补贴")
        )
        self.assertTrue(
            current_turn_candidate_targets_are_complete(
                question,
                target_ranges=targets,
                qualifier_ranges=(employee,),
            )
        )
        self.assertFalse(
            current_turn_candidate_targets_are_complete(
                question,
                target_ranges=targets[:2],
                qualifier_ranges=(employee,),
            )
        )
        self.assertFalse(
            current_turn_candidate_targets_are_complete(
                question,
                target_ranges=(targets[0], (targets[1][0], targets[1][0] + 1), targets[2]),
                qualifier_ranges=(employee,),
            )
        )

    def test_source_anchored_candidates_can_cover_separate_current_clauses(self) -> None:
        question = "报销提交时限是多久？需要提供哪些凭证？"
        report = (question.index("报销"), question.index("报销") + len("报销"))
        targets = (
            (question.index("提交时限"), question.index("提交时限") + len("提交时限")),
            (question.index("凭证"), question.index("凭证") + len("凭证")),
        )

        self.assertTrue(
            current_turn_candidate_targets_are_complete(
                question,
                target_ranges=targets,
                qualifier_ranges=(report,),
            )
        )
        # The punctuation boundary alone is not enough: business content left
        # uncovered in a sibling clause must still reject the candidate set.
        self.assertFalse(
            current_turn_candidate_targets_are_complete(
                question,
                target_ranges=(
                    targets[0],
                    (question.index("哪些"), question.index("哪些") + len("哪些")),
                ),
                qualifier_ranges=(report,),
            )
        )

    def test_elliptical_action_target_requires_an_earlier_clause_and_pure_shell(self) -> None:
        question = "报销提交时限是多久？需要提供哪些凭证？"
        report = (question.index("报销"), question.index("报销") + len("报销"))
        target = (question.index("凭证"), question.index("凭证") + len("凭证"))
        self.assertTrue(
            is_elliptical_current_clause_target(
                question,
                target_range=target,
                qualifier_ranges=(report,),
            )
        )

        # A qualifier already inside the later clause is not an omitted head;
        # it belongs to the ordinary baseline task and cannot trigger a
        # cross-clause lexical rewrite.
        repeated = "报销提交时限是多久？报销需要提供哪些凭证？"
        repeated_target = (
            repeated.rindex("凭证"),
            repeated.rindex("凭证") + len("凭证"),
        )
        repeated_qualifier = (
            repeated.rindex("报销"),
            repeated.rindex("报销") + len("报销"),
        )
        self.assertFalse(
            is_elliptical_current_clause_target(
                repeated,
                target_range=repeated_target,
                qualifier_ranges=(repeated_qualifier,),
            )
        )

        # Punctuation itself cannot make an arbitrary noun a dependent
        # request.  Residual business words must prevent normalisation.
        unrelated = "报销提交时限是多久？其他需要提供哪些凭证？"
        unrelated_target = (
            unrelated.index("凭证"),
            unrelated.index("凭证") + len("凭证"),
        )
        unrelated_report = (
            unrelated.index("报销"),
            unrelated.index("报销") + len("报销"),
        )
        self.assertFalse(
            is_elliptical_current_clause_target(
                unrelated,
                target_range=unrelated_target,
                qualifier_ranges=(unrelated_report,),
            )
        )
    def test_composed_entity_and_condition_keep_separate_boundaries(self) -> None:
        cases = {
            "总经理在北京的住宿标准是多少": (
                "住宿标准",
                ("北京",),
                (("总经理", "entity"), ("北京", "condition")),
            ),
            "合同工在一线城市住宿标准是多少": (
                "住宿标准",
                ("一线城市",),
                (("合同工", "entity"), ("一线城市", "condition")),
            ),
            "需要提供哪些凭证": (
                "凭证",
                (),
                (),
            ),
            "报销需要提供哪些凭证": (
                "报销凭证",
                (),
                (),
            ),
            "报销凭证需要提供哪些": (
                "报销凭证",
                (),
                (),
            ),
            "申请需要提交哪些材料": (
                "申请材料",
                (),
                (),
            ),
            "报销提交时限是多久": (
                "报销提交时限",
                (),
                (),
            ),
            # ``需要满足`` / ``符合`` / ``具备`` are a question shell.  The
            # answer noun is explicitly present in the current text and must
            # remain the same for planning and source-claim adjudication.
            "报销需要满足什么条件": (
                "报销条件",
                (),
                (),
            ),
            "报销需符合哪些要求": (
                "报销要求",
                (),
                (),
            ),
            "普通员工应具备哪些条件": (
                "条件",
                (),
                (("普通员工", "entity"),),
            ),
            "供应商管理要求有哪些": (
                "管理要求",
                (),
                (("供应商", "entity"),),
            ),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                frame = parse_query_surface_frame(question)
                self.assertEqual(frame.answer_target, expected[0])
                self.assertEqual(frame.context_terms, expected[1])
                self.assertEqual(
                    tuple((item.text, item.kind) for item in frame.qualifiers),
                    expected[2],
                )

    def test_condition_like_prefix_never_becomes_an_entity_by_default(self) -> None:
        for question in (
            "北京出差住宿标准是多少",
            "在北京出差的住宿标准是多少",
            "超过48小时的处理要求是什么",
            "项目验收阶段的资料有哪些",
        ):
            with self.subTest(question=question):
                frame = parse_query_surface_frame(question)
                self.assertNotIn(
                    "entity",
                    tuple(item.kind for item in frame.qualifiers),
                )
