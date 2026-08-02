import unittest

from core.rag_v2.query_plan import plan_query_locally


class ConservativeLocalQueryPlannerTests(unittest.TestCase):
    def test_recognizes_generic_answer_shapes(self) -> None:
        cases = {
            "某项制度是什么": "overview",
            "这个集合有哪些元素": "list",
            "如何完成这项操作": "process",
            "比较方案 A 与方案 B 的差异": "comparison",
            "这个条件是否成立": "judgement",
            "最终值是多少": "fact",
            "该值由前一项决定，最终结果是多少": "multi_hop",
            "第一项是什么？第二项如何处理？": "multi_part",
        }

        for question, expected in cases.items():
            with self.subTest(question=question):
                plan = plan_query_locally(question)
                self.assertEqual(plan.answer_shape, expected)

    def test_internal_question_placeholder_does_not_create_a_mapping_bridge(self) -> None:
        """A legacy fallback sentence is not a user entity/attribute query."""

        plan = plan_query_locally("回答用户当前问题")

        self.assertEqual([item.role for item in plan.requirements], ["answer"])

    def test_procedure_contract_requires_an_operational_structure(self) -> None:
        """Do not turn every `如何处理` into an exhaustive process request."""

        cases = {
            "如何配置VPN": ("process", "collection", "structured_collection"),
            "采购申请流程是什么": ("process", "collection", "structured_collection"),
            "连续出差超过30天住宿标准如何处理": (
                "fact",
                "single",
                "single_claim",
            ),
            "超出报销标准如何处理": ("fact", "single", "single_claim"),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                plan = plan_query_locally(question)
                answer = next(item for item in plan.requirements if item.role == "answer")
                self.assertEqual(
                    (plan.answer_shape, answer.coverage_mode, answer.coverage_contract),
                    expected,
                )

    def test_broad_overview_does_not_use_narrow_fact_path(self) -> None:
        plan = plan_query_locally("某项标准是什么")

        self.assertEqual(plan.answer_shape, "overview")
        self.assertFalse(plan.allows_narrow_fact_path)
        self.assertEqual(plan.requirements[0].coverage_mode, "collection")

    def test_list_requirement_uses_collection_coverage(self) -> None:
        plan = plan_query_locally("这个集合有哪些元素")

        self.assertEqual(plan.answer_shape, "list")
        self.assertEqual(plan.requirements[0].coverage_mode, "collection")

    def test_broad_open_policy_uses_document_snapshot_not_list_closure(
        self,
    ) -> None:
        """Open policy heads require a full policy snapshot to be complete."""

        for question in (
            "公司的出差标准是什么",
            "总经理的出差标准是什么",
            "供应商管理要求是什么",
            "供应商管理要求有哪些",
            "公司出差管理制度有哪些规定",
        ):
            with self.subTest(question=question):
                answer = next(
                    item
                    for item in plan_query_locally(question).requirements
                    if item.role == "answer"
                )
                self.assertEqual(answer.coverage_mode, "collection")
                self.assertEqual(answer.coverage_contract, "document_policy")

        for question in (
            "登录方式有哪些",
            "偏远地区出差有什么补贴",
            "采购申请流程是什么",
            "审批流程有哪些步骤",
        ):
            with self.subTest(question=question):
                answer = next(
                    item
                    for item in plan_query_locally(question).requirements
                    if item.role == "answer"
                )
                self.assertEqual(answer.coverage_mode, "collection")
                self.assertEqual(answer.coverage_contract, "structured_collection")

    def test_what_is_does_not_automatically_expand_to_document_overview(
        self,
    ) -> None:
        for question in (
            "公司的联系电话是什么",
            "系统登录入口是什么",
            "该字段是什么",
            "what is the server address",
        ):
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertNotEqual(plan.answer_shape, "overview")
                self.assertNotEqual(plan.reason, "explicit_overview_signal")

    def test_explicit_or_policy_overview_signals_remain_overview(self) -> None:
        for question in (
            "请概述这份制度的主要内容",
            "介绍一下向量数据库",
            "供应商政策是什么",
            "what is the employee policy",
        ):
            with self.subTest(question=question):
                self.assertEqual(
                    plan_query_locally(question).answer_shape,
                    "overview",
                )

    def test_only_explicit_scalar_lookup_uses_narrow_fact_path(self) -> None:
        plan = plan_query_locally("最终数值是多少")

        self.assertEqual(plan.answer_shape, "fact")
        self.assertTrue(plan.allows_narrow_fact_path)

    def test_relational_question_is_not_collapsed_to_fact(self) -> None:
        plan = plan_query_locally("该对象对应的数值是多少")

        self.assertEqual(plan.answer_shape, "multi_hop")
        self.assertFalse(plan.allows_narrow_fact_path)
        self.assertEqual(
            [item.role for item in plan.requirements],
            ["answer", "bridge"],
        )

    def test_entity_group_classification_is_an_optional_augmentation(self) -> None:
        cases = (
            "普通员工的餐补是多少",
            "合同工住宿标准",
            "外包人员的系统权限是什么",
            "管理员的访问权限是什么",
            "董事长的住宿标准是什么",
            "总经理的出差标准是什么",
            "副总经理的住宿标准是什么",
            "部门经理的住宿标准是什么",
            "主管的出差标准是什么",
            "总监的住宿标准是什么",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertEqual(plan.answer_shape, "fact")
                self.assertEqual(len(plan.requirements), 2)
                self.assertEqual(plan.requirements[0].role, "answer")
                self.assertEqual(plan.requirements[1].role, "bridge")
                self.assertEqual(plan.requirements[1].source, "inferred")
                self.assertEqual(
                    plan.requirements[0].depends_on_requirement_ids,
                    (),
                )
                self.assertEqual(
                    plan.requirements[0].augmentation_requirement_ids,
                    (plan.requirements[1].id,),
                )
                self.assertEqual(len(plan.retrieval_queries), 2)

    def test_named_entity_strategy_question_is_a_direct_attribute(self) -> None:
        plan = plan_query_locally("供应商甲的风险处置措施是什么")

        self.assertEqual(plan.answer_shape, "fact")
        # A named supplier's disposition is one entity attribute.  Treating
        # an instance as a population that must first map to a class creates
        # an unnecessary evidence dependency and can suppress direct proof.
        answer = plan.requirements[0]
        self.assertEqual(answer.coverage_mode, "single")
        self.assertEqual(answer.coverage_contract, "single_claim")
        self.assertEqual(answer.depends_on_requirement_ids, ())
        self.assertEqual(answer.augmentation_requirement_ids, ())
        self.assertFalse(any(item.role == "bridge" for item in plan.requirements))

    def test_compact_prepositional_conditions_never_gain_classification_bridges(
        self,
    ) -> None:
        cases = {
            "总经理在北京住宿标准": ("总经理",),
            "合同工在一线城市住宿标准": ("合同工",),
            "客户A在项目P1审批额度": (),
        }
        for question, expected_subjects in cases.items():
            with self.subTest(question=question):
                plan = plan_query_locally(question)
                bridges = tuple(
                    item.bridge_subject
                    for item in plan.requirements
                    if item.role == "bridge"
                )
                self.assertEqual(bridges, expected_subjects)
                self.assertEqual(
                    plan.requirements[0].depends_on_requirement_ids,
                    (),
                )
                self.assertEqual(
                    plan.requirements[0].augmentation_requirement_ids,
                    tuple(item.id for item in plan.requirements[1:]),
                )

    def test_ambiguous_compact_preposition_does_not_invent_a_condition(
        self,
    ) -> None:
        plan = plan_query_locally("总经理在岗补贴标准")

        bridges = [
            item for item in plan.requirements if item.role == "bridge"
        ]
        self.assertEqual([item.bridge_subject for item in bridges], ["总经理"])

    def test_condition_scope_is_direct_and_never_a_classification_augmentation(
        self,
    ) -> None:
        """Places, durations, stages and thresholds stay in the direct task."""

        cases = (
            "偏远地区出差有什么补贴",
            "艰苦地区出差有什么补贴",
            "连续超过30天出差有什么补贴",
            "北京出差住宿标准是多少",
            "试用期年假天数",
            "项目验收阶段的资料有哪些",
        )
        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)
                answers = [item for item in plan.requirements if item.role == "answer"]
                self.assertEqual(plan.retrieval_queries[0], question)
                self.assertFalse(any(
                    item.role == "bridge" for item in plan.requirements
                ))
                self.assertTrue(all(
                    item.depends_on_requirement_ids == () for item in answers
                ))
                self.assertTrue(all(
                    item.augmentation_requirement_ids == () for item in answers
                ))

    def test_implicit_mapping_coverage_distinguishes_policy_and_entity_attributes(
        self,
    ) -> None:
        for question in ("总经理的出差标准是什么",):
            with self.subTest(question=question):
                collection_plan = plan_query_locally(question)
                self.assertEqual(collection_plan.answer_shape, "fact")
                self.assertEqual(
                    collection_plan.requirements[0].coverage_mode,
                    "collection",
                )
                self.assertEqual(
                    collection_plan.requirements[0].coverage_contract,
                    "document_policy",
                )
                self.assertEqual(
                    collection_plan.requirements[1].coverage_mode,
                    "single",
                )

        for question in (
            "总经理的住宿标准是多少",
            "总经理的住宿多少钱",
            "普通员工餐补金额",
            "客户A的审批额度是什么",
            "客户A的审批额度是多少",
            "外包人员的系统权限是什么",
            "供应商甲的风险处置措施是什么",
            "客户A的服务策略是什么",
        ):
            with self.subTest(question=question):
                plan = plan_query_locally(question)
                answer = next(
                    item for item in plan.requirements if item.role == "answer"
                )
                self.assertEqual(answer.coverage_mode, "single")
                self.assertEqual(answer.coverage_contract, "single_claim")

    def test_explicit_applicability_scope_is_not_part_of_bridge_subject(self) -> None:
        cases = (
            "云枢8.6普通员工的餐补标准是多少",
            "CloudPivot 8.6 普通员工的住宿标准是多少",
            "2025版普通员工的交通标准是多少",
            "中青建安的云枢8.2.75普通员工餐补标准是多少",
            "中青建安云枢8.2.75普通员工的餐补标准是多少",
            "关于中青建安的 CloudPivot 8.6 普通员工住宿标准是多少",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertEqual(plan.answer_shape, "fact")
                bridge = next(
                    item for item in plan.requirements if item.role == "bridge"
                )
                answer = next(
                    item for item in plan.requirements if item.role == "answer"
                )
                self.assertEqual(answer.depends_on_requirement_ids, ())
                self.assertEqual(answer.augmentation_requirement_ids, (bridge.id,))
                self.assertTrue(bridge.description.startswith("确认普通员工对应"))
                self.assertNotIn("8.6", bridge.description)
                self.assertNotIn("8.2.75", bridge.description)
                self.assertNotIn("2025", bridge.description)
                self.assertNotIn("中青建安", bridge.description)

    def test_scope_suffix_without_a_subject_falls_back_to_original_question(
        self,
    ) -> None:
        plan = plan_query_locally("普通员工的云枢8.6配置权限是什么")

        self.assertEqual(plan.answer_shape, "fact")
        bridge = next(item for item in plan.requirements if item.role == "bridge")
        answer = next(item for item in plan.requirements if item.role == "answer")
        self.assertEqual(answer.depends_on_requirement_ids, ())
        self.assertEqual(answer.augmentation_requirement_ids, (bridge.id,))
        self.assertTrue(bridge.description.startswith("确认普通员工对应"))
        self.assertIn("云枢8.6配置权限", bridge.description)

    def test_direct_configuration_and_topic_phrases_do_not_gain_a_bridge(self) -> None:
        cases = {
            "如何配置员工权限": "process",
            "交通费用标准": "unknown",
            "云枢8.6登录配置": "unknown",
            "普通岗位的管理标准是什么": "overview",
            "员工标准是什么": "overview",
            "供应商政策是什么": "overview",
        }

        for question, expected_shape in cases.items():
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertEqual(plan.answer_shape, expected_shape)
                self.assertFalse(any(
                    item.role == "bridge" for item in plan.requirements
                ))

    def test_organization_owners_do_not_gain_an_invented_bridge(self) -> None:
        cases = (
            "公司的出差管理标准是什么",
            "公司的报销标准是什么",
            "企业的费用标准是什么",
            "集团的审批规则是什么",
            "组织的访问控制规则是什么",
            "部门的密码策略是什么",
            "机构的访问规则是什么",
            "地区的住宿规则是什么",
            "供应商的风险处置措施是什么",
            "客户的服务方案是什么",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertNotEqual(plan.answer_shape, "multi_hop")
                self.assertFalse(any(
                    item.role == "bridge" for item in plan.requirements
                ))

    def test_named_entities_and_places_do_not_gain_classification_augmentation(self) -> None:
        for question, expects_augmentation in (
            ("组织A的访问控制规则是什么", False),
            ("部门D01的审批额度是多少", False),
            ("北京地区的住宿标准是多少", False),
        ):
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertNotEqual(plan.answer_shape, "multi_hop")
                answers = [item for item in plan.requirements if item.role == "answer"]
                bridges = [item for item in plan.requirements if item.role == "bridge"]
                self.assertEqual(bool(bridges), expects_augmentation)
                if expects_augmentation:
                    self.assertEqual(
                        answers[0].augmentation_requirement_ids,
                        (bridges[0].id,),
                    )
                else:
                    self.assertEqual(answers[0].augmentation_requirement_ids, ())

    def test_direct_entity_attributes_do_not_invent_an_intermediate_bridge(
        self,
    ) -> None:
        for question in (
            "供应商甲的联系人名称是什么",
            "客户A的名称是什么",
            "用户U01的邮箱地址是什么",
            "供应商甲的风险等级是什么",
            "客户A的服务类别是什么",
        ):
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertNotEqual(plan.answer_shape, "multi_hop")
                self.assertFalse(any(
                    item.role == "bridge" for item in plan.requirements
                ))

    def test_explicit_scope_or_classification_does_not_gain_a_bridge(self) -> None:
        cases = (
            "云枢8.6版本的参数上限是多少",
            "D级的餐补是多少",
            "专业版产品的并发上限是多少",
            "重点项目的审批额度是多少",
            "管理员角色的访问权限是什么",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertNotEqual(plan.answer_shape, "multi_hop")
                self.assertFalse(any(
                    item.role == "bridge" for item in plan.requirements
                ))

    def test_unknown_and_invalid_input_are_conservative(self) -> None:
        unknown = plan_query_locally("opaque token")
        invalid = plan_query_locally(None)

        self.assertEqual(unknown.answer_shape, "unknown")
        self.assertFalse(unknown.allows_narrow_fact_path)
        self.assertEqual(invalid.answer_shape, "unknown")
        self.assertTrue(invalid.needs_clarification)
        self.assertFalse(invalid.allows_narrow_fact_path)

    def test_undecomposable_relationship_requests_specific_clarification(self) -> None:
        plan = plan_query_locally("该值取决于前一项")

        self.assertTrue(plan.needs_clarification)
        self.assertEqual(plan.answer_shape, "unknown")
        self.assertIn("对象", plan.clarification_question)
        self.assertIn("对应关系", plan.clarification_question)
        self.assertIn("标准或数值", plan.clarification_question)

    def test_multi_part_queries_are_split_without_adding_topics(self) -> None:
        plan = plan_query_locally("第一项是什么？第二项如何处理？")

        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertEqual(
            plan.retrieval_queries,
            ("第一项是什么", "第二项如何处理"),
        )
        self.assertEqual(
            [(item.id, item.description) for item in plan.requirements],
            [
                ("r1", "第一项是什么"),
                ("r2", "第二项如何处理"),
            ],
        )
        self.assertTrue(all(item.is_required_answer for item in plan.requirements))
        self.assertTrue(all(item.source == "explicit" for item in plan.requirements))

        newline_plan = plan_query_locally("检查第一项\n处理第二项")
        self.assertEqual(newline_plan.answer_shape, "multi_part")
        self.assertEqual(
            newline_plan.retrieval_queries,
            ("检查第一项", "处理第二项"),
        )
        self.assertEqual(
            [item.description for item in newline_plan.requirements],
            ["检查第一项", "处理第二项"],
        )

    def test_coordinated_items_become_independent_required_answers(self) -> None:
        cases = {
            "报销时限和所需凭证分别是什么？": (
                "报销时限是什么",
                "所需凭证是什么",
            ),
            "采购额度、审批人和报销材料分别是什么？": (
                "采购额度是什么",
                "审批人是什么",
                "报销材料是什么",
            ),
            "账号锁定与密码策略各自如何配置？": (
                "账号锁定如何配置",
                "密码策略如何配置",
            ),
        }

        for question, expected_queries in cases.items():
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertEqual(plan.answer_shape, "multi_part")
                self.assertEqual(plan.retrieval_queries, expected_queries)
                self.assertEqual(
                    [item.description for item in plan.requirements],
                    list(expected_queries),
                )
                self.assertTrue(all(
                    item.is_required_answer for item in plan.requirements
                ))

    def test_safe_coordinated_targets_do_not_require_distributive_marker(
        self,
    ) -> None:
        cases = {
            "普通员工的住宿标准和餐补是多少": (
                (
                    "普通员工的住宿标准是多少",
                    "普通员工的餐补是多少",
                    "普通员工 对应的适用分类 等级 类别 职级 角色 版本 档位 阶段",
                ),
                ("answer", "answer", "bridge"),
                ((), (), None),
                (("r3",), ("r3",), None),
            ),
            "合同工的住宿标准与交通补贴是多少": (
                (
                    "合同工的住宿标准是多少",
                    "合同工的交通补贴是多少",
                    "合同工 对应的适用分类 等级 类别 职级 角色 版本 档位 阶段",
                ),
                ("answer", "answer", "bridge"),
                ((), (), None),
                (("r3",), ("r3",), None),
            ),
            "云枢8.6版本的并发上限和附件上限是多少": (
                (
                    "云枢8.6版本的并发上限是多少",
                    "云枢8.6版本的附件上限是多少",
                ),
                ("answer", "answer"),
                ((), ()),
                ((), ()),
            ),
        }

        for question, (queries, roles, dependencies, augmentations) in cases.items():
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertEqual(plan.answer_shape, "multi_part")
                self.assertEqual(plan.retrieval_queries, queries)
                self.assertEqual(
                    tuple(item.role for item in plan.requirements),
                    roles,
                )
                self.assertEqual(
                    tuple(
                        item.depends_on_requirement_ids
                        for item in plan.requirements
                    ),
                    dependencies,
                )
                self.assertEqual(
                    tuple(
                        item.augmentation_requirement_ids
                        for item in plan.requirements
                    ),
                    augmentations,
                )
                self.assertIn(
                    "implicit_coordinated_answer_targets",
                    plan.reason,
                )

    def test_implicit_coordination_keeps_ambiguous_compounds_intact(self) -> None:
        cases = (
            "审批和报销流程是什么",
            "公司的制度和标准是什么",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertNotEqual(
                    plan.reason,
                    "implicit_coordinated_answer_targets",
                )
                self.assertEqual(len(plan.requirements), 1)

    def test_coordinated_identity_scope_keeps_one_critical_bridge(self) -> None:
        cases = (
            "普通员工的交通、住宿和餐补标准分别是多少？",
            "普通员工交通、住宿和餐补标准分别是多少？",
            "普通员工出差的住宿、交通和餐补标准分别是多少？",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertEqual(plan.answer_shape, "multi_part")
                self.assertEqual(
                    [item.role for item in plan.requirements],
                    ["answer", "answer", "answer", "bridge"],
                )
                self.assertTrue(all(
                    item.is_required_answer for item in plan.requirements[:3]
                ))
                self.assertEqual(plan.requirements[3].importance, "helpful")
                self.assertEqual(plan.requirements[3].source, "inferred")
                self.assertIn("普通员工", plan.requirements[3].description)
                self.assertTrue(all(
                    item.depends_on_requirement_ids == ()
                    for item in plan.requirements[:3]
                ))
                self.assertTrue(all(
                    item.augmentation_requirement_ids == ("r4",)
                    for item in plan.requirements[:3]
                ))
                self.assertEqual(
                    len(plan.retrieval_queries),
                    len(plan.requirements),
                )

    def test_local_anaphora_after_enumeration_is_trimmed_and_split(self) -> None:
        plan = plan_query_locally(
            "普通员工的住宿标准和餐补还有出差补贴这些分别是多少"
        )

        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertEqual(
            [item.description for item in plan.requirements if item.role == "answer"],
            [
                "普通员工的住宿标准是多少",
                "普通员工的餐补是多少",
                "普通员工的出差补贴是多少",
            ],
        )
        self.assertNotIn("这些", " ".join(plan.retrieval_queries))
        bridge_ids = {
            dependency
            for item in plan.requirements
            if item.role == "answer"
            for dependency in (item.augmentation_requirement_ids or ())
        }
        self.assertEqual(len(bridge_ids), 1)

    def test_coordinated_explicit_scope_does_not_invent_a_bridge(self) -> None:
        cases = (
            "公司的交通、住宿和餐补标准分别是多少？",
            "云枢8.6版本的并发上限和附件上限分别是多少？",
            "公司员工手册的借阅权限和保管要求分别是什么？",
        )

        for question in cases:
            with self.subTest(question=question):
                plan = plan_query_locally(question)

                self.assertEqual(plan.answer_shape, "multi_part")
                self.assertTrue(all(
                    item.role == "answer" for item in plan.requirements
                ))

    def test_coordinated_entity_scopes_receive_the_complete_shared_target(
        self,
    ) -> None:
        plan = plan_query_locally(
            "供应商甲、供应商乙的风险处置措施分别是什么"
        )

        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertEqual(
            plan.retrieval_queries[:2],
            (
                "供应商甲的风险处置措施是什么",
                "供应商乙的风险处置措施是什么",
            ),
        )
        answers = [item for item in plan.requirements if item.role == "answer"]
        bridges = [item for item in plan.requirements if item.role == "bridge"]
        self.assertEqual(
            [item.description for item in answers],
            list(plan.retrieval_queries[:2]),
        )
        self.assertEqual(bridges, [])
        self.assertNotIn("供应商甲是什么", plan.retrieval_queries)

    def test_coordinated_complete_targets_are_not_merged_into_each_other(
        self,
    ) -> None:
        plan = plan_query_locally(
            "供应商甲的风险处置措施和风险等级分别是什么"
        )

        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertEqual(
            plan.retrieval_queries[:2],
            (
                "供应商甲的风险处置措施是什么",
                "供应商甲的风险等级是什么",
            ),
        )
        self.assertNotIn("风险处置措施等级", " ".join(plan.retrieval_queries))
        answers = [item for item in plan.requirements if item.role == "answer"]
        self.assertEqual(answers[0].depends_on_requirement_ids, ())
        self.assertEqual(answers[0].augmentation_requirement_ids, ())
        self.assertEqual(answers[1].depends_on_requirement_ids, ())

    def test_chinese_semicolon_and_numbered_queries_have_matching_requirements(
        self,
    ) -> None:
        semicolon_plan = plan_query_locally("查询甲项；查询乙项；查询丙项")

        self.assertEqual(
            semicolon_plan.retrieval_queries,
            ("查询甲项", "查询乙项", "查询丙项"),
        )
        self.assertEqual(
            [item.description for item in semicolon_plan.requirements],
            ["查询甲项", "查询乙项", "查询丙项"],
        )

        numbered_plan = plan_query_locally(
            "1. 查询第一项 2. 查询第二项 3. 查询第三项"
        )
        self.assertEqual(
            numbered_plan.retrieval_queries,
            ("查询第一项", "查询第二项", "查询第三项"),
        )
        self.assertEqual(
            [(item.id, item.description) for item in numbered_plan.requirements],
            [
                ("r1", "查询第一项"),
                ("r2", "查询第二项"),
                ("r3", "查询第三项"),
            ],
        )

    def test_english_multi_part_query_has_one_requirement_per_question(self) -> None:
        plan = plan_query_locally(
            "What is the first setting? How do I configure the second setting?"
        )

        self.assertEqual(plan.answer_shape, "multi_part")
        self.assertEqual(
            plan.retrieval_queries,
            (
                "What is the first setting",
                "How do I configure the second setting",
            ),
        )
        self.assertEqual(
            [item.description for item in plan.requirements],
            list(plan.retrieval_queries),
        )

    def test_duplicate_multi_part_queries_are_deduplicated_in_requirements(
        self,
    ) -> None:
        plan = plan_query_locally("第一项是什么？第一项是什么？第二项是什么？")

        self.assertEqual(
            plan.retrieval_queries,
            ("第一项是什么", "第二项是什么"),
        )
        self.assertEqual(
            [(item.id, item.description) for item in plan.requirements],
            [("r1", "第一项是什么"), ("r2", "第二项是什么")],
        )

    def test_multi_part_queries_and_requirements_are_limited_to_eight(self) -> None:
        question = " ".join(
            f"{index}. 子问题{index}" for index in range(1, 11)
        )

        plan = plan_query_locally(question)

        self.assertEqual(
            plan.retrieval_queries,
            tuple(f"子问题{index}" for index in range(1, 9)),
        )
        self.assertEqual(
            [item.id for item in plan.requirements],
            [f"r{index}" for index in range(1, 9)],
        )
        self.assertEqual(
            [item.description for item in plan.requirements],
            list(plan.retrieval_queries),
        )

    def test_non_multi_part_plan_keeps_single_original_query_requirement(self) -> None:
        question = "某项制度是什么"

        plan = plan_query_locally(question)

        self.assertEqual(plan.answer_shape, "overview")
        self.assertEqual(plan.retrieval_queries, (question,))
        self.assertEqual(
            [(item.id, item.description) for item in plan.requirements],
            [("r1", question)],
        )

    def test_single_trailing_question_mark_is_not_multi_part(self) -> None:
        plan = plan_query_locally("某项制度是什么？")

        self.assertEqual(plan.answer_shape, "overview")

    def test_explicit_version_comparison_is_compiled_to_one_scope_per_answer(
        self,
    ) -> None:
        question = "比较 CloudPivot 6 和 CloudPivot 7 的安全配置"

        plan = plan_query_locally(question)

        answers = [
            item for item in plan.requirements if item.is_required_answer
        ]
        self.assertEqual(plan.answer_shape, "comparison")
        self.assertEqual(
            [item.description for item in answers],
            ["CloudPivot 6 安全配置", "CloudPivot 7 安全配置"],
        )
        self.assertEqual(
            [(item.scope_product, item.scope_version) for item in answers],
            [("CloudPivot", "6"), ("CloudPivot", "7")],
        )
        self.assertNotEqual(
            answers[0].scope_fingerprint,
            answers[1].scope_fingerprint,
        )
        self.assertTrue(all(
            item.applicability_scope.version_source is not None
            and item.applicability_scope.version_source.origin == "current_query"
            for item in answers
        ))

    def test_comparison_scope_fallback_still_never_merges_side_constraints(
        self,
    ) -> None:
        # The target shell is deliberately unusual.  The planner may retain
        # it verbatim, but must never collapse source-owned 6/7 into one
        # unscoped answer that could admit version 8.
        plan = plan_query_locally(
            "请比较 CloudPivot 6 与 CloudPivot 7，然后说明谁更适合"
        )

        answers = [
            item for item in plan.requirements if item.is_required_answer
        ]
        self.assertEqual(len(answers), 2)
        self.assertEqual(
            {item.scope_version for item in answers},
            {"6", "7"},
        )
        self.assertTrue(all(
            item.applicability_scope.has_scope_constraint
            for item in answers
        ))


if __name__ == "__main__":
    unittest.main()
