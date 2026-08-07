import unittest
import uuid

from core.semantic_memory import (
    MAX_FACTS,
    ResolvedEntityMemory,
    extract_resolved_entity_memory,
    has_entity_reuse,
    parse_resolved_entity_memory,
)

GRADE_TABLE = (
    "【公司出差管理标准.docx › 二、职级分类】\n"
    "| 职级 | 适用人员 |\n"
    "| --- | --- |\n"
    "| A级 | 董事长、总经理、副总经理 |\n"
    "| B级 | 部门总监、高级经理 |\n"
    "| C级 | 部门经理、主管 |\n"
    "| D级 | 普通员工、专员 |"
)
FLIGHT_TABLE = (
    "【公司出差管理标准.docx › 三、交通费用标准 › 3.1 飞机】\n"
    "| 职级 | 国内航班 | 国际航班 |\n"
    "| --- | --- | --- |\n"
    "| A级 | 头等舱或公务舱 | 公务舱 |\n"
    "| B级 | 公务舱（航程>3小时）<br>经济舱（航程≤3小时） | 经济舱 |\n"
    "| C级 | 经济舱 | 经济舱 |\n"
    "| D级 | 经济舱 | 经济舱 |"
)


class SemanticMemoryExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kb_id = uuid.uuid4()
        self.doc_id = uuid.uuid4()
        self.turn_id = uuid.uuid4()

    def _source(self, *, chunk_id: uuid.UUID, content: str) -> dict:
        return {
            "kb_id": str(self.kb_id),
            "doc_id": str(self.doc_id),
            "id": str(chunk_id),
            "filename": "公司出差管理标准.docx",
            "content": content,
        }

    def test_mention_in_member_cell_binds_back_to_category(self) -> None:
        memory = extract_resolved_entity_memory(
            sources=[self._source(chunk_id=uuid.uuid4(), content=GRADE_TABLE)],
            question="普通员工出差时可以乘坐的交通工具有哪些",
            source_turn_id=self.turn_id,
            trace_id="trace-1",
        )

        self.assertIsNotNone(memory)
        assert memory is not None
        self.assertEqual(
            [(fact.mention, fact.attribute, fact.value) for fact in memory.facts],
            [("普通员工", "职级", "D级")],
        )
        fact = memory.facts[0]
        self.assertEqual(fact.filename, "公司出差管理标准.docx")
        self.assertEqual(fact.kb_id, self.kb_id)
        self.assertEqual(fact.doc_id, self.doc_id)
        self.assertEqual(fact.source_turn_id, self.turn_id)

    def test_category_mention_in_question_binds_all_columns(self) -> None:
        memory = extract_resolved_entity_memory(
            sources=[self._source(chunk_id=uuid.uuid4(), content=FLIGHT_TABLE)],
            question="D级可以坐国内航班经济舱吗",
            source_turn_id=self.turn_id,
            trace_id="trace-1",
        )

        self.assertIsNotNone(memory)
        assert memory is not None
        self.assertEqual(
            [(fact.mention, fact.attribute, fact.value) for fact in memory.facts],
            [
                ("D级", "国内航班", "经济舱"),
                ("D级", "国际航班", "经济舱"),
            ],
        )

    def test_entity_reuse_detects_repeated_mention_in_later_turn(self) -> None:
        memory = extract_resolved_entity_memory(
            sources=[self._source(chunk_id=uuid.uuid4(), content=GRADE_TABLE)],
            question="普通员工出差时可以乘坐的交通工具有哪些",
            source_turn_id=self.turn_id,
            trace_id="trace-1",
        )

        self.assertIsNotNone(memory)
        assert memory is not None
        self.assertTrue(has_entity_reuse("普通员工可以乘坐头等舱吗", memory))
        self.assertFalse(has_entity_reuse("员工请假流程是什么", memory))
        self.assertFalse(has_entity_reuse("", memory))

    def test_memory_keeps_full_answer_source_identities(self) -> None:
        first = uuid.uuid4()
        second = uuid.uuid4()
        memory = extract_resolved_entity_memory(
            sources=[
                self._source(chunk_id=first, content=GRADE_TABLE),
                self._source(chunk_id=second, content=FLIGHT_TABLE),
            ],
            question="普通员工出差时可以乘坐的交通工具有哪些",
            source_turn_id=self.turn_id,
            trace_id="trace-1",
        )

        self.assertIsNotNone(memory)
        assert memory is not None
        self.assertEqual(memory.source_chunk_ids, (first, second))

    def test_round_trip_is_strict_and_content_free(self) -> None:
        memory = extract_resolved_entity_memory(
            sources=[self._source(chunk_id=uuid.uuid4(), content=GRADE_TABLE)],
            question="普通员工出差时可以乘坐的交通工具有哪些",
            source_turn_id=self.turn_id,
            trace_id="trace-1",
        )

        self.assertIsNotNone(memory)
        assert memory is not None
        payload = memory.to_dict()
        self.assertEqual(parse_resolved_entity_memory(payload), memory)
        self.assertNotIn("content", str(payload))
        self.assertIsNone(
            parse_resolved_entity_memory({**payload, "source_chunk_ids": []})
        )
        self.assertIsNone(
            parse_resolved_entity_memory({"schema_version": "other"})
        )

    def test_no_mention_match_yields_no_memory(self) -> None:
        memory = extract_resolved_entity_memory(
            sources=[self._source(chunk_id=uuid.uuid4(), content=FLIGHT_TABLE)],
            question="普通员工出差时可以乘坐的交通工具有哪些",
            source_turn_id=self.turn_id,
            trace_id="trace-1",
        )

        # ``普通员工`` only occurs in the grade table, not the flight table.
        self.assertIsNone(memory)

    def test_fact_limit_is_bounded(self) -> None:
        many_rows = (
            "| 类别 | 值 |\n"
            "| --- | --- |\n"
            + "".join(f"| X{index} | 人员{index} |\n" for index in range(80))
        )
        memory = extract_resolved_entity_memory(
            sources=[self._source(chunk_id=uuid.uuid4(), content=many_rows)],
            question="X1 X2 X3 X4 X5 X6 X7 X8 X9 X10 X11 X12 X13 X14 X15 X16 X17 X18 X19 X20 X21 X22 X23 X24 X25 X26 X27 X28 X29 X30 X31 X32 X33 X34 X35 X36 X37 X38 X39 X40 X41 X42 X43 X44 X45 X46 X47 X48 X49 X50 X51 X52 X53 X54 X55 X56 X57 X58 X59 X60 X61",
            source_turn_id=self.turn_id,
            trace_id="trace-1",
        )

        self.assertIsNotNone(memory)
        assert memory is not None
        self.assertLessEqual(len(memory.facts), MAX_FACTS)

    def test_rejects_malformed_entity_memory(self) -> None:
        self.assertIsNone(parse_resolved_entity_memory(None))
        self.assertIsNone(parse_resolved_entity_memory("bad"))
        self.assertIsNone(parse_resolved_entity_memory({}))
        self.assertIsNone(
            parse_resolved_entity_memory(
                {
                    "schema_version": "rag_semantic_memory.v1",
                    "facts": "bad",
                    "source_chunk_ids": [],
                }
            )
        )
        self.assertIsNone(
            parse_resolved_entity_memory(
                {
                    "schema_version": "rag_semantic_memory.v1",
                    "facts": [],
                    "source_chunk_ids": [str(uuid.uuid4())],
                }
            )
        )


class SemanticMemoryValidationTests(unittest.TestCase):
    def test_empty_memory_requires_sources_and_facts(self) -> None:
        with self.assertRaises(ValueError):
            ResolvedEntityMemory(facts=(), source_chunk_ids=())

    def test_duplicate_source_chunks_are_rejected(self) -> None:
        chunk_id = uuid.uuid4()
        with self.assertRaises(ValueError):
            ResolvedEntityMemory(
                facts=(),
                source_chunk_ids=(chunk_id, chunk_id),
            )


if __name__ == "__main__":
    unittest.main()
