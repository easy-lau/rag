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

        newline_plan = plan_query_locally("检查第一项\n处理第二项")
        self.assertEqual(newline_plan.answer_shape, "multi_part")
        self.assertEqual(
            newline_plan.retrieval_queries,
            ("检查第一项", "处理第二项"),
        )

    def test_single_trailing_question_mark_is_not_multi_part(self) -> None:
        plan = plan_query_locally("某项制度是什么？")

        self.assertEqual(plan.answer_shape, "overview")


if __name__ == "__main__":
    unittest.main()
