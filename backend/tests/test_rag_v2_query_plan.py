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

    def test_broad_overview_does_not_use_narrow_fact_path(self) -> None:
        plan = plan_query_locally("某项标准是什么")

        self.assertEqual(plan.answer_shape, "overview")
        self.assertFalse(plan.allows_narrow_fact_path)

    def test_only_explicit_scalar_lookup_uses_narrow_fact_path(self) -> None:
        plan = plan_query_locally("最终数值是多少")

        self.assertEqual(plan.answer_shape, "fact")
        self.assertTrue(plan.allows_narrow_fact_path)

    def test_relational_question_is_not_collapsed_to_fact(self) -> None:
        plan = plan_query_locally("该对象对应的数值是多少")

        self.assertEqual(plan.answer_shape, "multi_hop")
        self.assertFalse(plan.allows_narrow_fact_path)

    def test_unknown_and_invalid_input_are_conservative(self) -> None:
        unknown = plan_query_locally("opaque token")
        invalid = plan_query_locally(None)

        self.assertEqual(unknown.answer_shape, "unknown")
        self.assertFalse(unknown.allows_narrow_fact_path)
        self.assertEqual(invalid.answer_shape, "unknown")
        self.assertTrue(invalid.needs_clarification)
        self.assertFalse(invalid.allows_narrow_fact_path)

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


if __name__ == "__main__":
    unittest.main()
