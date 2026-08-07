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
        self.draft = Document(
            id=uuid.uuid4(),
            kb_id=self.kb_id,
            filename="8.1升级8.6.20",
            file_type="md",
            status="draft",
            is_active=True,
            tags=["升级"],
        )

    def _assert_excludes_drafts_and_inactive(self, statements) -> None:
        compiled = [
            (str(stmt.compile()), dict(stmt.compile().params))
            for stmt in statements
        ]
        for sql, params in compiled:
            self.assertIn("is_active is true", sql.casefold())
            self.assertNotIn("is_active is false", sql.casefold())
            self.assertTrue(
                any(value == "draft" for value in params.values()),
                f"expected a draft status bound parameter in: {params}",
            )

    async def test_default_catalog_hides_drafts_and_inactive_documents(self) -> None:
        db = _CatalogDB([
            _Result(scalar=1),
            _Result(rows=[(self.ready, self.kb)]),
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
        self.assertEqual(search["total"], 1)
        self.assertTrue(all(
            item["source_kind"] == "document_metadata"
            for item in search["answer_sources"]
        ))
        self.assertIn("共有 1 篇文章", answer)
        self.assertIn("《云枢6配置参数说明》", answer)
        self.assertNotIn("《云枢7配置》", answer)
        self.assertNotIn("8.1升级8.6.20", answer)
        self.assertNotIn("embedding", answer.casefold())
        self._assert_excludes_drafts_and_inactive(db.statements)

    async def test_explicit_inactive_filter_still_lists_disabled_documents(self) -> None:
        db = _CatalogDB([
            _Result(scalar=1),
            _Result(rows=[(self.inactive, self.kb)]),
        ])
        request = KnowledgeRequestSemantics(
            resource="document_catalog",
            operation="list",
            status_filter="inactive",
        )

        events = await _events(run_knowledge_catalog_stream(
            question="列出当前停用的文档",
            kb_ids=[self.kb_id],
            search_config={},
            conversation_id=str(uuid.uuid4()),
            db=db,
            knowledge_request=request,
        ))

        answer = "".join(
            item["content"] for item in events if item["type"] == "text_delta"
        )
        self.assertIn("《云枢7配置》", answer)
        self.assertNotIn("8.1升级8.6.20", answer)
        for statement in db.statements:
            self.assertIn("is_active is false", str(statement.compile()).casefold())
            self.assertTrue(
                any(value == "draft" for value in dict(statement.compile().params).values())
            )
        process = next(item for item in events if item["type"] == "search_process")
        self.assertEqual(process["execution_path"], "catalog")
        self.assertEqual(
            [(step["key"], step["label"]) for step in process["steps"]],
            [
                ("analyze", "问题分析"),
                ("retrieve", "目录查询"),
                ("generate", "生成"),
            ],
        )
        active_steps = [
            item["step"]
            for item in events
            if item["type"] == "search_step" and item["status"] == "active"
        ]
        self.assertEqual(active_steps, ["analyze", "retrieve", "generate"])

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
                    _Result(scalar=1),
                    _Result(rows=[(self.ready, self.kb)]),
                ],
                "共找到 1 篇文章",
            ),
            (
                KnowledgeRequestSemantics(
                    resource="document_catalog",
                    operation="group",
                    group_by="status",
                ),
                [
                    _Result(scalar=1),
                    _Result(rows=[(self.ready, self.kb)]),
                    _Result(rows=[("ready", 1)]),
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
                    self.assertNotIn("已停用", answer)


if __name__ == "__main__":
    unittest.main()
