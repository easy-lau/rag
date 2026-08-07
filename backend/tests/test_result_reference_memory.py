import unittest
import uuid

from core.conversation_context import (
    ConversationContext,
    apply_result_reference_memory_context,
)
from core.result_reference_memory import (
    RESULT_REFERENCE_MEMORY_SCHEMA_VERSION,
    ResolvedResultReference,
    build_result_reference_memory,
    build_reference_correction_acknowledgement,
    is_reference_correction,
    parse_result_reference_memory,
    resolve_result_reference_memory,
)
from models.db_models import Document


class _ScalarRows:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _DB:
    def __init__(self, rows):
        self.rows = rows

    async def execute(self, _statement):
        return _ScalarRows(self.rows)


def _document(*, doc_id, kb_id, filename="钉钉"):
    return Document(
        id=doc_id,
        kb_id=kb_id,
        filename=filename,
        file_type="md",
        status="ready",
        is_active=True,
        tags=[],
    )


class ResultReferenceMemoryParseTests(unittest.TestCase):
    def setUp(self):
        self.kb_id = uuid.uuid4()
        self.doc_ids = [uuid.uuid4() for _ in range(3)]
        self.turn_id = uuid.uuid4()
        self.memory = build_result_reference_memory(
            root_query="我现在知识库里面有多少文章",
            list_label="知识库文档目录",
            items=[
                {"kb_id": self.kb_id, "doc_id": self.doc_ids[0], "filename": "二开发送钉钉工作通知"},
                {"kb_id": self.kb_id, "doc_id": self.doc_ids[1], "filename": "云枢6配置参数说明"},
                {"kb_id": self.kb_id, "doc_id": self.doc_ids[2], "filename": "钉钉"},
            ],
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
        )

    def test_round_trip_is_strict(self):
        payload = self.memory.to_dict()

        parsed = parse_result_reference_memory(payload)

        self.assertEqual(parsed, self.memory)
        self.assertEqual(parsed.items[2].filename, "钉钉")
        self.assertNotIn("content", payload)

    def test_rejects_unknown_schema_or_extra_fields(self):
        payload = self.memory.to_dict()
        self.assertIsNone(parse_result_reference_memory(
            {**payload, "schema_version": "rag_result_reference_memory.v9"}
        ))
        self.assertIsNone(parse_result_reference_memory(
            {**payload, "dispatch_authorized": True}
        ))

    def test_rejects_duplicate_or_unsorted_items(self):
        payload = self.memory.to_dict()
        payload["items"] = [
            {**payload["items"][0], "index": 1},
            {**payload["items"][0], "index": 2},
        ]
        self.assertIsNone(parse_result_reference_memory(payload))
        payload = self.memory.to_dict()
        payload["items"][0]["index"] = 2
        self.assertIsNone(parse_result_reference_memory(payload))

    def test_build_requires_at_least_one_item(self):
        with self.assertRaises(ValueError):
            build_result_reference_memory(
                root_query="有多少文章",
                list_label="目录",
                items=[],
                source_turn_id=self.turn_id,
                trace_id=uuid.uuid4().hex,
            )


class ResultReferenceCorrectionTests(unittest.TestCase):
    def test_correction_language_structure(self):
        self.assertTrue(is_reference_correction("第四个不是《钉钉》吗"))
        self.assertTrue(is_reference_correction("你刚才说错了，应该是第五个"))
        self.assertTrue(is_reference_correction("第五个才对吧"))
        self.assertTrue(is_reference_correction("你返回错了吧，我想看第四个"))
        self.assertFalse(is_reference_correction("我想看第四个"))

    def test_acknowledgement_names_the_correct_document(self):
        kb_id = uuid.uuid4()
        memory = build_result_reference_memory(
            root_query="有多少文章",
            list_label="目录",
            items=[
                {"kb_id": kb_id, "doc_id": uuid.uuid4(), "filename": "云枢7配置"},
                {"kb_id": kb_id, "doc_id": uuid.uuid4(), "filename": "云枢6配置参数说明"},
                {"kb_id": kb_id, "doc_id": uuid.uuid4(), "filename": "二开发送钉钉工作通知"},
                {"kb_id": kb_id, "doc_id": uuid.uuid4(), "filename": "钉钉"},
            ],
            source_turn_id=uuid.uuid4(),
            trace_id=uuid.uuid4().hex,
        )
        surface = __import__(
            "core.result_reference", fromlist=["parse_result_reference_surface"]
        ).parse_result_reference_surface("第四个不是《钉钉》吗")
        item = memory.item_for_surface(surface)
        self.assertIsNotNone(item)
        self.assertEqual(item.filename, "钉钉")

        acknowledgement = build_reference_correction_acknowledgement(
            surface=surface,
            item=item,
            question="第四个不是《钉钉》吗",
        )

        self.assertIsNotNone(acknowledgement)
        self.assertIn("第四个", acknowledgement)
        self.assertIn("钉钉", acknowledgement)
        self.assertIsNone(build_reference_correction_acknowledgement(
            surface=surface,
            item=item,
            question="我想看第四个",
        ))


class ResultReferenceMemoryResolutionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.kb_id = uuid.uuid4()
        self.other_kb_id = uuid.uuid4()
        self.doc_ids = [uuid.uuid4() for _ in range(4)]
        self.turn_id = uuid.uuid4()
        self.memory = build_result_reference_memory(
            root_query="我现在知识库里面有多少文章",
            list_label="知识库文档目录",
            items=[
                {"kb_id": self.kb_id, "doc_id": self.doc_ids[0], "filename": "二开发送钉钉工作通知"},
                {"kb_id": self.kb_id, "doc_id": self.doc_ids[1], "filename": "云枢6配置参数说明"},
                {"kb_id": self.kb_id, "doc_id": self.doc_ids[2], "filename": "云枢7配置"},
                {"kb_id": self.kb_id, "doc_id": self.doc_ids[3], "filename": "钉钉"},
            ],
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
        )

    async def test_ordinal_reference_resolves_and_reauthorizes(self):
        db = _DB(_document(doc_id=self.doc_ids[3], kb_id=self.kb_id))

        resolved = await resolve_result_reference_memory(
            db,
            value=self.memory.to_dict(),
            question="我想看第四个",
            selected_kb_ids=[self.kb_id],
        )

        self.assertIsNotNone(resolved)
        self.assertIsInstance(resolved, ResolvedResultReference)
        self.assertEqual(resolved.item.index, 4)
        self.assertEqual(resolved.source["doc_id"], str(self.doc_ids[3]))
        self.assertEqual(resolved.source["filename"], "钉钉")
        self.assertFalse(resolved.correction)
        self.assertIsNone(resolved.acknowledgement)
        self.assertEqual(resolved.safe_summary()["index"], 4)

    async def test_last_reference_resolves(self):
        db = _DB(_document(doc_id=self.doc_ids[3], kb_id=self.kb_id))

        resolved = await resolve_result_reference_memory(
            db,
            value=self.memory.to_dict(),
            question="最后一个是什么",
            selected_kb_ids=[self.kb_id],
        )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.item.index, 4)
        self.assertEqual(resolved.source["filename"], "钉钉")

    async def test_correction_resolves_with_acknowledgement(self):
        db = _DB(_document(doc_id=self.doc_ids[3], kb_id=self.kb_id))

        resolved = await resolve_result_reference_memory(
            db,
            value=self.memory.to_dict(),
            question="第四个不是《钉钉》吗",
            selected_kb_ids=[self.kb_id],
        )

        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.correction)
        self.assertIsNotNone(resolved.acknowledgement)
        self.assertIn("钉钉", resolved.acknowledgement)

    async def test_regulation_clause_is_not_a_result_reference(self):
        db = _DB(_document(doc_id=self.doc_ids[0], kb_id=self.kb_id))

        resolved = await resolve_result_reference_memory(
            db,
            value=self.memory.to_dict(),
            question="《员工手册》第3条说了什么",
            selected_kb_ids=[self.kb_id],
        )

        self.assertIsNone(resolved)

    async def test_out_of_range_ordinal_is_rejected(self):
        db = _DB(_document(doc_id=self.doc_ids[0], kb_id=self.kb_id))

        resolved = await resolve_result_reference_memory(
            db,
            value=self.memory.to_dict(),
            question="我想看第九个",
            selected_kb_ids=[self.kb_id],
        )

        self.assertIsNone(resolved)

    async def test_unauthorized_kb_is_rejected(self):
        db = _DB(_document(doc_id=self.doc_ids[0], kb_id=self.kb_id))

        resolved = await resolve_result_reference_memory(
            db,
            value=self.memory.to_dict(),
            question="我想看第四个",
            selected_kb_ids=[self.other_kb_id],
        )

        self.assertIsNone(resolved)

    async def test_inactive_or_unready_document_is_rejected(self):
        db = _DB(_document(doc_id=self.doc_ids[0], kb_id=self.kb_id))
        db.rows.is_active = False

        resolved = await resolve_result_reference_memory(
            db,
            value=self.memory.to_dict(),
            question="我想看第四个",
            selected_kb_ids=[self.kb_id],
        )

        self.assertIsNone(resolved)

    async def test_plain_question_does_not_resolve(self):
        db = _DB(_document(doc_id=self.doc_ids[0], kb_id=self.kb_id))

        resolved = await resolve_result_reference_memory(
            db,
            value=self.memory.to_dict(),
            question="普通员工出差可以坐什么车",
            selected_kb_ids=[self.kb_id],
        )

        self.assertIsNone(resolved)


class ResultReferenceMemoryContextTests(unittest.TestCase):
    def setUp(self):
        self.kb_id = uuid.uuid4()
        self.doc_id = uuid.uuid4()
        self.turn_id = uuid.uuid4()
        self.memory = build_result_reference_memory(
            root_query="有多少文章",
            list_label="目录",
            items=[
                {"kb_id": self.kb_id, "doc_id": self.doc_id, "filename": "钉钉"},
            ],
            source_turn_id=self.turn_id,
            trace_id=uuid.uuid4().hex,
        )

    def _resolved(self, *, correction=False):
        return ResolvedResultReference(
            memory=self.memory,
            item=self.memory.items[0],
            source={
                "source_kind": "document_result_reference",
                "doc_id": str(self.doc_id),
                "kb_id": str(self.kb_id),
                "filename": "钉钉",
            },
            surface=__import__(
                "core.result_reference",
                fromlist=["parse_result_reference_surface"],
            ).parse_result_reference_surface("我想看第四个"),
            correction=correction,
            acknowledgement=(
                "你说得对，第四个是《钉钉》。" if correction else None
            ),
        )

    def test_apply_sets_memory_mode_and_sources(self):
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="我想看第四个",
            history_messages=(),
            carryover_sources=(),
        )

        applied = apply_result_reference_memory_context(
            context=context,
            question="我想看第四个",
            resolved_reference=self._resolved(),
        )

        self.assertEqual(applied.query_resolution_mode, "result_reference_memory")
        self.assertTrue(applied.is_followup)
        self.assertEqual(len(applied.result_reference_memory_sources), 1)
        self.assertEqual(
            applied.result_reference_memory_sources[0]["filename"],
            "钉钉",
        )
        self.assertIsNone(applied.result_reference_memory_acknowledgement)

    def test_apply_with_correction_sets_acknowledgement(self):
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="第四个不是《钉钉》吗",
            history_messages=(),
            carryover_sources=(),
        )

        applied = apply_result_reference_memory_context(
            context=context,
            question="第四个不是《钉钉》吗",
            resolved_reference=self._resolved(correction=True),
        )

        self.assertEqual(applied.query_resolution_mode, "result_reference_memory")
        self.assertIsNotNone(applied.result_reference_memory_acknowledgement)

    def test_apply_with_none_keeps_context(self):
        context = ConversationContext(
            is_followup=False,
            followup_reason="standalone_question",
            standalone_query="普通员工出差交通工具有哪些",
            history_messages=(),
            carryover_sources=(),
        )

        applied = apply_result_reference_memory_context(
            context=context,
            question="普通员工出差交通工具有哪些",
            resolved_reference=None,
        )

        self.assertIs(applied, context)


if __name__ == "__main__":
    unittest.main()
