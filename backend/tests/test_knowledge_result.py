from __future__ import annotations

import json
import unittest
import uuid

from core.knowledge_result import run_knowledge_result_stream
from core.query_semantics import KnowledgeRequestSemantics
from models.db_models import Document, DocumentChunk, KnowledgeBase


class _Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def all(self):
        return list(self.rows)

    def scalars(self):
        return self


class _DB:
    def __init__(self, results):
        self.results = list(results)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        if not self.results:
            raise AssertionError("unexpected query")
        return self.results.pop(0)


async def _events(stream):
    output = []
    async for chunk in stream:
        output.append(json.loads(chunk.removeprefix("data: ").strip()))
    return output


class KnowledgeResultRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_uses_only_bound_authorized_document(self):
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        kb = KnowledgeBase(id=kb_id, name="测试知识库")
        document = Document(
            id=doc_id,
            kb_id=kb_id,
            filename="第一篇文章.md",
            file_type="md",
            status="ready",
            is_active=True,
        )
        chunks = [
            DocumentChunk(
                id=uuid.uuid4(),
                doc_id=doc_id,
                kb_id=kb_id,
                content="第一段正文。",
                chunk_index=0,
            ),
            DocumentChunk(
                id=uuid.uuid4(),
                doc_id=doc_id,
                kb_id=kb_id,
                content="第二段正文。",
                chunk_index=1,
            ),
        ]
        request = KnowledgeRequestSemantics(
            resource="document_result",
            operation="read",
            result_handles=("r_t1_001",),
            result_labels=("第一篇文章.md",),
        )
        db = _DB([
            _Result(rows=[(document, kb)]),
            _Result(rows=chunks),
        ])

        events = await _events(run_knowledge_result_stream(
            question="我想看第一个文章",
            kb_ids=[kb_id],
            search_config={},
            conversation_id=str(uuid.uuid4()),
            db=db,
            knowledge_request=request,
            result_sources=[{
                "handle": "r_t1_001",
                "kb_id": str(kb_id),
                "doc_id": str(doc_id),
                "filename": "第一篇文章.md",
            }],
        ))

        search = next(item for item in events if item["type"] == "search_results")
        answer = "".join(
            item["content"] for item in events if item["type"] == "text_delta"
        )
        self.assertEqual(search["evidence_status"], "hit")
        self.assertEqual(len(search["answer_sources"]), 2)
        self.assertTrue(all(
            item["doc_id"] == str(doc_id)
            for item in search["answer_sources"]
        ))
        self.assertIn("《第一篇文章.md》", answer)
        self.assertIn("第一段正文", answer)
        self.assertIn("第二段正文", answer)
        process = next(item for item in events if item["type"] == "search_process")
        self.assertEqual(process["execution_path"], "result_reference")
        self.assertEqual(
            [(step["key"], step["label"]) for step in process["steps"]],
            [
                ("analyze", "问题分析"),
                ("retrieve", "结果读取"),
                ("generate", "生成"),
            ],
        )
        active_steps = [
            item["step"]
            for item in events
            if item["type"] == "search_step" and item["status"] == "active"
        ]
        self.assertEqual(active_steps, ["analyze", "retrieve", "generate"])

    async def test_read_renders_legacy_collapsed_markdown_as_sections_and_code(self):
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        kb = KnowledgeBase(id=kb_id, name="测试知识库")
        document = Document(
            id=doc_id,
            kb_id=kb_id,
            filename="升级说明.docx",
            file_type="docx",
            status="ready",
            is_active=True,
        )
        chunk = DocumentChunk(
            id=uuid.uuid4(),
            doc_id=doc_id,
            kb_id=kb_id,
            content=(
                "【升级说明.docx › 后端升级】\n"
                "注意事项 * 复制<code>application.yml</code> * 修改配置。 "
                "```shell cp -R application-prod.yml application.yml ```"
            ),
            chunk_index=0,
            metadata_={"section_path": ["升级说明.docx", "后端升级"]},
        )
        request = KnowledgeRequestSemantics(
            resource="document_result",
            operation="read",
            result_handles=("r_t1_001",),
            result_labels=("升级说明.docx",),
        )
        db = _DB([
            _Result(rows=[(document, kb)]),
            _Result(rows=[chunk]),
        ])

        events = await _events(run_knowledge_result_stream(
            question="看一下第一个",
            kb_ids=[kb_id],
            search_config={},
            conversation_id=str(uuid.uuid4()),
            db=db,
            knowledge_request=request,
            result_sources=[{
                "handle": "r_t1_001",
                "kb_id": str(kb_id),
                "doc_id": str(doc_id),
                "filename": "升级说明.docx",
            }],
        ))

        answer = "".join(
            item["content"] for item in events if item["type"] == "text_delta"
        )
        self.assertIn("## 《升级说明.docx》", answer)
        self.assertIn("### 后端升级", answer)
        self.assertIn("- 复制`application.yml`", answer)
        self.assertIn("```shell\ncp -R application-prod.yml application.yml\n```", answer)
        self.assertNotIn("<code>", answer)


if __name__ == "__main__":
    unittest.main()
