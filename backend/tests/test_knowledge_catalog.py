from __future__ import annotations

import json
import unittest
import uuid

from core.knowledge_catalog import run_knowledge_catalog_stream
from core.query_semantics import KnowledgeRequestSemantics
from models.db_models import Document, KnowledgeBase


class _Result:
    def __init__(self, *, scalar: int | None = None, rows=()):
        self.scalar = scalar
        self.rows = list(rows)

    def scalar_one(self):
        if self.scalar is None:
            raise AssertionError("result has no scalar")
        return self.scalar

    def all(self):
        return list(self.rows)


class _CatalogDB:
    def __init__(self, results):
        self.results = list(results)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if not self.results:
            raise AssertionError("unexpected catalog query")
        return self.results.pop(0)


async def _events(stream) -> list[dict]:
    events: list[dict] = []
    async for chunk in stream:
        events.append(json.loads(chunk.removeprefix("data: ").strip()))
    return events


class KnowledgeCatalogRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.kb_id = uuid.uuid4()
        self.kb = KnowledgeBase(id=self.kb_id, name="华中大区知识库")
        self.ready = Document(
            id=uuid.uuid4(),
            kb_id=self.kb_id,
            filename="云枢6配置参数说明",
            file_type="md",
            status="ready",
            is_active=True,
            tags=["云枢6"],
        )
        self.inactive = Document(
            id=uuid.uuid4(),
            kb_id=self.kb_id,
            filename="云枢7配置",
            file_type="docx",
            status="failed",
            is_active=False,
            tags=["云枢7"],
        )

    async def test_count_uses_only_authorized_scope_and_metadata_evidence(self) -> None:
        db = _CatalogDB([
            _Result(scalar=2),
            _Result(rows=[(self.ready, self.kb), (self.inactive, self.kb)]),
        ])
        request = KnowledgeRequestSemantics(
            resource="document_catalog",
            operation="count",
            filter_span_ids=("s_current_004",),
            filter_terms=("云枢配置",),
        )

        events = await _events(run_knowledge_catalog_stream(
            question="我现在有关于云枢配置的知识库有几个文章",
            kb_ids=[self.kb_id],
            search_config={},
            conversation_id=str(uuid.uuid4()),
            db=db,
            knowledge_request=request,
        ))

        search = next(item for item in events if item["type"] == "search_results")
        answer = "".join(
            item["content"] for item in events if item["type"] == "text_delta"
        )
        self.assertEqual(search["total"], 2)
        self.assertTrue(all(
            item["source_kind"] == "document_metadata"
            for item in search["answer_sources"]
        ))
        self.assertIn("共有 2 篇文章", answer)
        self.assertIn("《云枢6配置参数说明》", answer)
        self.assertIn("《云枢7配置》", answer)
        self.assertNotIn("embedding", answer.casefold())

        for statement in db.statements:
            params = statement.compile().params
            flattened = str(params)
            self.assertIn(str(self.kb_id), flattened)

    async def test_list_and_group_render_natural_catalog_answers(self) -> None:
        cases = (
            (
                KnowledgeRequestSemantics(
                    resource="document_catalog",
                    operation="list",
                ),
                [
                    _Result(scalar=2),
                    _Result(rows=[(self.ready, self.kb), (self.inactive, self.kb)]),
                ],
                "共找到 2 篇文章",
            ),
            (
                KnowledgeRequestSemantics(
                    resource="document_catalog",
                    operation="group",
                    group_by="status",
                ),
                [
                    _Result(scalar=2),
                    _Result(rows=[(self.ready, self.kb), (self.inactive, self.kb)]),
                    _Result(rows=[("ready", 1), ("inactive", 1)]),
                ],
                "按状态统计如下",
            ),
        )
        for request, results, expected in cases:
            with self.subTest(operation=request.operation):
                events = await _events(run_knowledge_catalog_stream(
                    question="列出并统计当前文档",
                    kb_ids=[self.kb_id],
                    search_config={},
                    conversation_id=str(uuid.uuid4()),
                    db=_CatalogDB(results),
                    knowledge_request=request,
                ))
                answer = "".join(
                    item["content"]
                    for item in events
                    if item["type"] == "text_delta"
                )
                self.assertIn(expected, answer)
                if request.operation == "group":
                    self.assertIn("已就绪：1 篇", answer)
                    self.assertIn("已停用：1 篇", answer)


if __name__ == "__main__":
    unittest.main()
