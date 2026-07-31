import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy.dialects.postgresql import ARRAY, UUID

from core.retriever import (
    MAX_SCOPED_DOCUMENTS,
    MAX_SCOPED_EXACT_TOTAL_CHUNKS,
    MAX_SCOPED_QUERIES,
    MAX_SCOPED_RESULTS,
    MAX_SMALL_DOCUMENT_CHARS,
    MAX_SMALL_DOCUMENT_CHUNKS,
    PER_DOCUMENT_RERANK_CHUNKS,
    _build_trigram_terms,
    _candidate_pool_size,
    _normalize_result_scores,
    fetch_small_document_candidates,
    fetch_structural_neighbors,
    hybrid_search,
    search_within_documents,
)


class _Rows:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def mappings(self):
        return self

    def all(self) -> list[dict]:
        return self._rows


def _db_with_rows(rows: list[dict]):
    return SimpleNamespace(execute=AsyncMock(return_value=_Rows(rows)))


class _NestedTransaction:
    def __init__(self, owner):
        self.owner = owner

    async def __aenter__(self):
        self.owner.savepoint_enters += 1
        return self

    async def __aexit__(self, exc_type, _exc, _traceback):
        self.owner.savepoint_exits += 1
        if exc_type is not None:
            self.owner.savepoint_rollbacks += 1
        return False


class _SavepointDB:
    def __init__(self, execute_results):
        self.execute = AsyncMock(side_effect=execute_results)
        self.savepoint_enters = 0
        self.savepoint_exits = 0
        self.savepoint_rollbacks = 0

    def begin_nested(self):
        return _NestedTransaction(self)


class TrigramQueryTests(unittest.TestCase):
    def test_long_chinese_question_produces_topic_product_and_version_terms(self) -> None:
        terms = _build_trigram_terms(
            "解决登录用户名枚举 要配置什么 我是云枢8.6"
        )

        self.assertIn("解决登录用户名枚举 要配置什么 我是云枢8.6", terms)
        self.assertIn("登录用户名枚举", terms)
        self.assertIn("云枢8.6", terms)
        self.assertIn("云枢", terms)
        self.assertIn("8.6", terms)
        self.assertEqual(len(terms), len({term.casefold() for term in terms}))

    def test_empty_query_has_no_terms(self) -> None:
        self.assertEqual(_build_trigram_terms(" \n\t "), [])

    def test_candidate_pool_overfetches_but_is_bounded(self) -> None:
        self.assertEqual(_candidate_pool_size(1), 40)
        self.assertEqual(_candidate_pool_size(10), 80)
        self.assertEqual(_candidate_pool_size(100), 240)


class ResultCompatibilityTests(unittest.TestCase):
    def test_legacy_score_is_kept_and_observability_fields_are_present(self) -> None:
        result = _normalize_result_scores({"id": "a", "score": 0.42})

        self.assertEqual(result["score"], 0.42)
        self.assertEqual(result["retrieval_score"], 0.42)
        for field in (
            "vector_score",
            "vector_rank",
            "keyword_score",
            "keyword_rank",
            "trigram_score",
            "trigram_rank",
        ):
            self.assertIn(field, result)
            self.assertIsNone(result[field])

    def test_retrieval_score_is_the_backward_compatible_score(self) -> None:
        result = _normalize_result_scores(
            {"score": 0.1, "retrieval_score": 0.25, "vector_rank": 1}
        )

        self.assertEqual(result["score"], 0.25)
        self.assertEqual(result["retrieval_score"], 0.25)
        self.assertEqual(result["vector_rank"], 1)

    def test_active_channels_only_contains_channels_with_a_rank(self) -> None:
        result = _normalize_result_scores(
            {
                "score": 0.03,
                "vector_rank": None,
                "keyword_rank": 2,
                "trigram_rank": 1,
            }
        )

        self.assertEqual(result["active_channels"], ["keyword", "trigram"])


class RetrieverSqlTests(unittest.IsolatedAsyncioTestCase):
    def assert_document_scope_filters(self, sql: str, *, minimum: int) -> None:
        """Every retrieval channel must bind chunks to a ready document in its KB."""
        self.assertGreaterEqual(
            sql.count("d.kb_id = dc.kb_id"),
            minimum,
        )
        self.assertGreaterEqual(
            sql.count("d.status = 'ready'"),
            minimum,
        )

    async def test_hybrid_uses_three_channels_chunk_fusion_and_stable_order(self) -> None:
        kb_id = uuid.uuid4()
        row = {
            "id": uuid.uuid4(),
            "doc_id": uuid.uuid4(),
            "content": "error_reply_same: true",
            "score": 0.04,
            "retrieval_score": 0.04,
            "vector_score": 0.81,
            "vector_rank": 2,
            "keyword_score": 0.53,
            "keyword_rank": 1,
            "trigram_score": 0.92,
            "trigram_rank": 1,
        }
        db = _db_with_rows([row])

        with patch("core.retriever.embed_text", new=AsyncMock(return_value=[0.1, 0.2])):
            results = await hybrid_search(
                db,
                "解决登录用户名枚举 要配置什么 我是云枢8.6",
                [kb_id],
                top_k=5,
                method="hybrid",
            )

        statement, params = db.execute.await_args.args
        sql = str(statement)
        self.assertIn("AS MATERIALIZED", sql)
        self.assertIn("dc.embedding::halfvec(2560)", sql)
        self.assertIn("CAST(:emb AS halfvec(2560))", sql)
        self.assertIn("dc.embedding <=> CAST(:emb AS vector)", sql)
        self.assertIn("word_similarity", sql)
        self.assertIn("FROM unnest(:trigram_terms)", sql)
        self.assertIn("SELECT id, doc_id FROM vector_r", sql)
        self.assertIn("SELECT id, doc_id FROM keyword_r", sql)
        self.assertIn("SELECT id, doc_id FROM trigram_r", sql)
        self.assertGreaterEqual(sql.count("PARTITION BY doc_id"), 3)
        self.assertIn("document_chunk_rank <= :per_document_chunks", sql)
        self.assertIn("fused_document_chunk_rank <= :per_document_chunks", sql)
        self.assertIn("fused_diverse.doc_id ASC", sql)
        self.assertIn("dc.chunk_index ASC", sql)
        self.assertIn("dc.id ASC", sql)
        self.assertIn("fused_diverse.retrieval_score AS score", sql)
        self.assert_document_scope_filters(sql, minimum=4)

        self.assertEqual(params["kb_ids"], [kb_id])
        self.assertIn("登录用户名枚举", params["trigram_terms"])
        self.assertEqual(params["candidate_pool"], 40)
        self.assertEqual(
            params["per_document_chunks"],
            PER_DOCUMENT_RERANK_CHUNKS,
        )
        self.assertEqual(params["top_k"], 5)

        kb_type = statement._bindparams["kb_ids"].type
        trigram_type = statement._bindparams["trigram_terms"].type
        self.assertIsInstance(kb_type, ARRAY)
        self.assertIsInstance(kb_type.item_type, UUID)
        self.assertIsInstance(trigram_type, ARRAY)

        self.assertEqual(results[0]["score"], 0.04)
        self.assertEqual(results[0]["retrieval_score"], 0.04)
        self.assertEqual(results[0]["trigram_rank"], 1)
        self.assertEqual(
            results[0]["active_channels"], ["vector", "keyword", "trigram"]
        )

    async def test_keyword_skips_embedding_and_disables_vector_channel(self) -> None:
        kb_id = uuid.uuid4()
        db = _db_with_rows([
            {
                "id": uuid.uuid4(),
                "doc_id": uuid.uuid4(),
                "content": "用户名枚举配置",
                "score": 0.03,
                "retrieval_score": 0.03,
                "vector_score": None,
                "vector_rank": None,
                "keyword_score": 0.4,
                "keyword_rank": 1,
                "trigram_score": 0.8,
                "trigram_rank": 2,
            }
        ])
        embed = AsyncMock(return_value=[0.1, 0.2])

        with patch("core.retriever.embed_text", new=embed):
            results = await hybrid_search(
                db,
                "用户名枚举怎么配置",
                [kb_id],
                top_k=5,
                method="keyword",
            )

        embed.assert_not_awaited()
        db.execute.assert_awaited_once()
        _, params = db.execute.await_args.args
        self.assertFalse(params["vector_enabled"])
        self.assertIsNone(params["emb"])
        self.assertEqual(results[0]["active_channels"], ["keyword", "trigram"])

    async def test_hybrid_keeps_solution_chunk_beside_placeholder_from_same_document(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        rows = [
            {
                "id": uuid.uuid4(),
                "doc_id": doc_id,
                "content": "【问题描述】无",
                "chunk_index": 1,
                "retrieval_score": 0.04,
                "vector_rank": 1,
            },
            {
                "id": uuid.uuid4(),
                "doc_id": doc_id,
                "content": "【解决方案】error_reply_same: true",
                "chunk_index": 3,
                "retrieval_score": 0.03,
                "trigram_rank": 2,
            },
        ]
        db = _db_with_rows(rows)

        with patch("core.retriever.embed_text", new=AsyncMock(return_value=[0.1, 0.2])):
            results = await hybrid_search(
                db,
                "云枢中如何配置登录用户名枚举",
                [kb_id],
                top_k=5,
                method="hybrid",
            )

        self.assertEqual(len(results), 2)
        self.assertEqual({item["doc_id"] for item in results}, {doc_id})
        self.assertTrue(
            any("error_reply_same" in item["content"] for item in results)
        )

    async def test_hybrid_embedding_failure_still_executes_lexical_search(self) -> None:
        kb_id = uuid.uuid4()
        db = _db_with_rows([
            {
                "id": uuid.uuid4(),
                "doc_id": uuid.uuid4(),
                "content": "error_reply_same: true",
                "score": 0.02,
                "retrieval_score": 0.02,
                "vector_score": None,
                "vector_rank": None,
                "keyword_score": None,
                "keyword_rank": None,
                "trigram_score": 0.91,
                "trigram_rank": 1,
            }
        ])
        embed = AsyncMock(side_effect=RuntimeError("embedding unavailable"))

        with patch("core.retriever.embed_text", new=embed):
            results = await hybrid_search(
                db,
                "云枢8.6用户名枚举配置",
                [kb_id],
                top_k=5,
                method="hybrid",
            )

        embed.assert_awaited_once_with("云枢8.6用户名枚举配置")
        db.execute.assert_awaited_once()
        statement, params = db.execute.await_args.args
        self.assertIn("AND :vector_enabled", str(statement))
        self.assertFalse(params["vector_enabled"])
        self.assertIsNone(params["emb"])
        self.assertEqual(results[0]["active_channels"], ["trigram"])

    async def test_vector_and_keyword_paths_bound_candidates_by_document(self) -> None:
        kb_id = uuid.uuid4()
        for method, score_name, rank_name in (
            ("vector", "vector_score", "vector_rank"),
            ("keyword", "keyword_score", "keyword_rank"),
        ):
            with self.subTest(method=method):
                db = _db_with_rows([{
                    "id": uuid.uuid4(),
                    "doc_id": uuid.uuid4(),
                    "score": 0.7,
                    "retrieval_score": 0.7,
                    score_name: 0.7,
                    rank_name: 1,
                }])
                with patch(
                    "core.retriever.embed_text",
                    new=AsyncMock(return_value=[0.1, 0.2]),
                ):
                    results = await hybrid_search(
                        db, "测试查询", [kb_id], top_k=5, method=method
                    )

                statement = db.execute.await_args.args[0]
                sql = str(statement)
                if method == "vector":
                    self.assertIn("AS MATERIALIZED", sql)
                    self.assertIn("dc.embedding::halfvec(2560)", sql)
                    self.assertIn("CAST(:emb AS halfvec(2560))", sql)
                    self.assertIn("dc.embedding <=> CAST(:emb AS vector)", sql)
                self.assertIn("PARTITION BY doc_id", sql)
                self.assertIn("doc_id ASC", sql)
                self.assertIn("id ASC", sql)
                self.assert_document_scope_filters(
                    sql,
                    minimum=2 if method == "vector" else 4,
                )
                if method == "keyword":
                    self.assertIn(
                        "document_chunk_rank <= :per_document_chunks",
                        sql,
                    )
                self.assertEqual(results[0][score_name], 0.7)
                self.assertEqual(results[0][rank_name], 1)
                self.assertIn("trigram_score", results[0])

    async def test_empty_inputs_skip_embedding_and_database(self) -> None:
        db = _db_with_rows([])
        embed = AsyncMock()

        with patch("core.retriever.embed_text", new=embed):
            result = await hybrid_search(db, "  ", [uuid.uuid4()])

        self.assertEqual(result, [])
        embed.assert_not_awaited()
        db.execute.assert_not_awaited()


class DocumentScopedRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_document_full_candidates_are_scoped_and_complete(self) -> None:
        kb_id = uuid.uuid4()
        other_kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        unauthorized_doc_id = uuid.uuid4()
        contents = ["普通员工属于D级", "D级住宿不超过450元"]
        total_chars = sum(len(content) for content in contents)
        rows = [
            {
                "id": uuid.uuid4(),
                "doc_id": doc_id,
                "kb_id": kb_id,
                "content": content,
                "chunk_index": index,
                "metadata": {"heading": "出差标准"},
                "filename": "出差制度.md",
                "actual_chunk_count": len(contents),
                "actual_char_count": total_chars,
                "doc_order": 1,
            }
            for index, content in enumerate(contents)
        ]
        # 即使异常测试替身返回了计划外行，Python 防御层也不得把它带入候选池。
        rows.append({
            **rows[0],
            "id": uuid.uuid4(),
            "doc_id": unauthorized_doc_id,
            "kb_id": other_kb_id,
            "content": "未授权内容",
        })
        db = _SavepointDB([_Rows(rows)])

        with patch("core.retriever.trace_event") as trace_mock:
            results = await fetch_small_document_candidates(
                db,
                kb_ids=[kb_id],
                doc_ids=[doc_id],
                max_chunks=999,
                max_chars=999_999,
                trace_id="trace-small-doc",
            )

        statement, params = db.execute.await_args.args
        sql = str(statement)
        self.assertIn("FROM unnest(:doc_ids) WITH ORDINALITY", sql)
        self.assertIn("d.id = ANY(:doc_ids)", sql)
        self.assertIn("d.kb_id = ANY(:kb_ids)", sql)
        self.assertIn("d.is_active = TRUE", sql)
        self.assertIn("d.status = 'ready'", sql)
        self.assertIn("JOIN LATERAL", sql)
        self.assertIn("LIMIT :probe_chunk_limit", sql)
        self.assertIn("COUNT(probed.id) BETWEEN 1 AND :max_chunks", sql)
        self.assertIn("SUM(char_length(COALESCE(probed.content, '')))", sql)
        self.assertIn("dc.doc_id = ANY(:doc_ids)", sql)
        self.assertIn("dc.kb_id = ANY(:kb_ids)", sql)
        self.assertEqual(params["doc_ids"], [doc_id])
        self.assertEqual(params["kb_ids"], [kb_id])
        self.assertEqual(params["max_chunks"], MAX_SMALL_DOCUMENT_CHUNKS)
        self.assertEqual(params["max_chars"], MAX_SMALL_DOCUMENT_CHARS)
        self.assertEqual(
            params["probe_chunk_limit"],
            MAX_SMALL_DOCUMENT_CHUNKS + 1,
        )
        self.assertEqual(params["row_limit"], MAX_SMALL_DOCUMENT_CHUNKS)
        self.assertEqual([item["chunk_index"] for item in results], [0, 1])
        self.assertTrue(all(item["doc_id"] == doc_id for item in results))
        self.assertTrue(all(
            item["candidate_origin"] == "small_document_full" for item in results
        ))
        self.assertTrue(all(
            item["candidate_origins"] == ["small_document_full"]
            for item in results
        ))
        self.assertTrue(all(
            item["full_document_chunk_count"] == 2 for item in results
        ))
        self.assertTrue(all(
            item["full_document_char_count"] == total_chars for item in results
        ))
        self.assertEqual(db.savepoint_enters, 1)
        completed = next(
            call for call in trace_mock.call_args_list
            if call.args
            and call.args[0] == "retrieval.small_document_candidates_completed"
        )
        self.assertEqual(completed.kwargs["loaded_document_count"], 1)
        self.assertEqual(completed.kwargs["candidate_count"], 2)

    async def test_small_document_cross_document_budget_never_returns_half_doc(self) -> None:
        kb_id = uuid.uuid4()
        first_doc = uuid.uuid4()
        second_doc = uuid.uuid4()
        rows = []
        for doc_order, doc_id, prefix in (
            (1, first_doc, "甲"),
            (2, second_doc, "乙"),
        ):
            contents = [prefix * 2, prefix * 2]
            rows.extend({
                "id": uuid.uuid4(),
                "doc_id": doc_id,
                "kb_id": kb_id,
                "content": content,
                "chunk_index": index,
                "filename": f"{prefix}.md",
                "actual_chunk_count": 2,
                "actual_char_count": 4,
                "doc_order": doc_order,
            } for index, content in enumerate(contents))
        db = _db_with_rows(rows)

        results = await fetch_small_document_candidates(
            db,
            kb_ids=[kb_id],
            doc_ids=[first_doc, second_doc],
            max_chunks=3,
            max_chars=8,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual({item["doc_id"] for item in results}, {first_doc})

    async def test_small_document_integrity_mismatch_skips_entire_document(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        db = _db_with_rows([{
            "id": uuid.uuid4(),
            "doc_id": doc_id,
            "kb_id": kb_id,
            "content": "只返回一片",
            "chunk_index": 0,
            "filename": "不完整.md",
            "actual_chunk_count": 2,
            "actual_char_count": len("只返回一片") + 3,
            "doc_order": 1,
        }])

        results = await fetch_small_document_candidates(
            db,
            kb_ids=[kb_id],
            doc_ids=[doc_id],
        )

        self.assertEqual(results, [])

    async def test_small_document_loader_rejects_empty_scope(self) -> None:
        db = _db_with_rows([])

        self.assertEqual(
            await fetch_small_document_candidates(
                db,
                kb_ids=[],
                doc_ids=[uuid.uuid4()],
            ),
            [],
        )
        self.assertEqual(
            await fetch_small_document_candidates(
                db,
                kb_ids=[uuid.uuid4()],
                doc_ids=[],
            ),
            [],
        )
        db.execute.assert_not_awaited()

    async def test_scoped_search_binds_documents_kbs_and_hard_budgets(self) -> None:
        kb_ids = [uuid.uuid4(), uuid.uuid4()]
        doc_ids = [uuid.uuid4() for _ in range(MAX_SCOPED_DOCUMENTS + 2)]
        chunk_id = uuid.uuid4()
        row = {
            "id": chunk_id,
            "doc_id": doc_ids[0],
            "kb_id": kb_ids[0],
            "content": "D级住宿标准为一线城市不超过450元",
            "chunk_index": 4,
            "metadata": {"heading": "住宿费用标准"},
            "filename": "公司出差管理标准.docx",
            "retrieval_score": 0.03,
            "score": 0.03,
            "vector_score": 0.91,
            "vector_rank": 1,
            "keyword_score": None,
            "keyword_rank": None,
            "trigram_score": 0.8,
            "trigram_rank": 1,
        }
        db = _db_with_rows([row])

        with patch(
            "core.retriever.embed_text",
            new=AsyncMock(return_value=[0.1, 0.2]),
        ) as embed:
            results = await search_within_documents(
                db,
                queries=["普通员工出差标准", "D级住宿交通补贴", "不得执行的第三条"],
                kb_ids=kb_ids,
                doc_ids=doc_ids,
                method="hybrid",
                per_document_limit=99,
                total_limit=99,
            )

        self.assertEqual(embed.await_count, MAX_SCOPED_QUERIES)
        self.assertEqual(db.execute.await_count, MAX_SCOPED_QUERIES + 1)
        stats_statement, stats_params = db.execute.await_args_list[0].args
        self.assertIn("FROM documents d", str(stats_statement))
        self.assertIn("d.id = ANY(:doc_ids)", str(stats_statement))
        self.assertEqual(stats_params["doc_ids"], doc_ids[:MAX_SCOPED_DOCUMENTS])
        for call in db.execute.await_args_list[1:]:
            statement, params = call.args
            sql = str(statement)
            self.assertIn("dc.doc_id = ANY(:doc_ids)", sql)
            self.assertIn("dc.kb_id = ANY(:kb_ids)", sql)
            self.assertIn("d.id = dc.doc_id", sql)
            self.assertIn("d.kb_id = dc.kb_id", sql)
            self.assertIn("d.is_active = TRUE", sql)
            self.assertIn("d.status = 'ready'", sql)
            self.assertIn("dc.embedding <=> CAST(:emb AS vector)", sql)
            self.assertIn("to_tsvector('simple', dc.content)", sql)
            self.assertIn("word_similarity", sql)
            self.assertIn("vector_candidates AS MATERIALIZED", sql)
            self.assertIn("keyword_candidates AS MATERIALIZED", sql)
            self.assertNotIn("scoped_scored", sql)
            self.assertEqual(params["doc_ids"], doc_ids[:MAX_SCOPED_DOCUMENTS])
            self.assertEqual(params["kb_ids"], kb_ids)
            self.assertEqual(params["per_document_limit"], 4)
            self.assertEqual(params["total_limit"], MAX_SCOPED_RESULTS)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], chunk_id)
        self.assertEqual(results[0]["candidate_origin"], "document_scoped")
        self.assertEqual(results[0]["candidate_origins"], ["document_scoped"])
        self.assertEqual(results[0]["expansion_query_indexes"], [0, 1])

    async def test_each_statement_uses_savepoint_and_partial_query_success_survives(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        row = {
            "id": uuid.uuid4(),
            "doc_id": doc_id,
            "kb_id": kb_id,
            "content": "D级住宿450元",
            "chunk_index": 4,
            "retrieval_score": 0.02,
            "score": 0.02,
            "vector_score": 0.8,
            "vector_rank": 1,
        }
        db = _SavepointDB([
            _Rows([{
                "total_chunk_count": 15,
                "max_document_chunk_count": 15,
            }]),
            RuntimeError("first scoped SQL failed"),
            _Rows([row]),
        ])

        with patch(
            "core.retriever.embed_text",
            new=AsyncMock(return_value=[0.1, 0.2]),
        ):
            results = await search_within_documents(
                db,
                queries=["普通员工出差标准", "D级住宿标准"],
                kb_ids=[kb_id],
                doc_ids=[doc_id],
            )

        self.assertEqual([item["id"] for item in results], [row["id"]])
        self.assertEqual(results[0]["expansion_query_indexes"], [1])
        self.assertEqual(db.savepoint_enters, 3)
        self.assertEqual(db.savepoint_exits, 3)
        self.assertEqual(db.savepoint_rollbacks, 1)

    async def test_all_scoped_sql_failures_raise_after_isolated_attempts(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        db = _SavepointDB([
            _Rows([{
                "total_chunk_count": 15,
                "max_document_chunk_count": 15,
            }]),
            RuntimeError("query one failed"),
            RuntimeError("query two failed"),
        ])

        with (
            patch(
                "core.retriever.embed_text",
                new=AsyncMock(return_value=[0.1, 0.2]),
            ),
            self.assertRaisesRegex(RuntimeError, "query two failed"),
        ):
            await search_within_documents(
                db,
                queries=["查询一", "查询二"],
                kb_ids=[kb_id],
                doc_ids=[doc_id],
            )

        self.assertEqual(db.savepoint_enters, 3)
        self.assertEqual(db.savepoint_exits, 3)
        self.assertEqual(db.savepoint_rollbacks, 2)

    async def test_large_document_guard_skips_exact_vector_and_trigram(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        lexical_row = {
            "id": uuid.uuid4(),
            "doc_id": doc_id,
            "kb_id": kb_id,
            "content": "D级住宿标准",
            "chunk_index": 30,
            "retrieval_score": 0.02,
            "score": 0.02,
            "keyword_score": 0.8,
            "keyword_rank": 1,
        }
        db = _SavepointDB([
            _Rows([{
                "total_chunk_count": MAX_SCOPED_EXACT_TOTAL_CHUNKS + 1,
                "max_document_chunk_count": MAX_SCOPED_EXACT_TOTAL_CHUNKS + 1,
            }]),
            _Rows([lexical_row]),
        ])
        embed = AsyncMock(return_value=[0.1, 0.2])

        with (
            patch("core.retriever.embed_text", new=embed),
            patch("core.retriever.trace_event") as trace_mock,
        ):
            results = await search_within_documents(
                db,
                queries=["D级出差标准"],
                kb_ids=[kb_id],
                doc_ids=[doc_id],
                method="hybrid",
                trace_id="trace-large-doc",
            )

        embed.assert_not_awaited()
        _, params = db.execute.await_args_list[-1].args
        self.assertFalse(params["vector_enabled"])
        self.assertFalse(params["trigram_enabled"])
        self.assertTrue(params["keyword_enabled"])
        self.assertEqual(params["trigram_terms"], [])
        self.assertEqual(results[0]["active_channels"], ["keyword"])
        completed = next(
            call for call in trace_mock.call_args_list
            if call.args and call.args[0] == "retrieval.document_scoped_completed"
        )
        self.assertTrue(completed.kwargs["scan_guard_triggered"])
        self.assertEqual(completed.kwargs["scan_guard_reason"], "total_chunk_limit")

    async def test_scoped_vector_failure_keeps_lexical_channels(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        db = _db_with_rows([{
            "id": uuid.uuid4(),
            "doc_id": doc_id,
            "kb_id": kb_id,
            "content": "D级经济舱",
            "chunk_index": 3,
            "retrieval_score": 0.02,
            "score": 0.02,
            "keyword_score": 0.5,
            "keyword_rank": 1,
            "trigram_score": 0.7,
            "trigram_rank": 1,
            "vector_score": None,
            "vector_rank": None,
        }])

        with patch(
            "core.retriever.embed_text",
            new=AsyncMock(side_effect=RuntimeError("embedding unavailable")),
        ):
            results = await search_within_documents(
                db,
                queries=["D级出差标准"],
                kb_ids=[kb_id],
                doc_ids=[doc_id],
                method="vector",
            )

        _, params = db.execute.await_args.args
        self.assertFalse(params["vector_enabled"])
        self.assertTrue(params["keyword_enabled"])
        self.assertTrue(params["trigram_enabled"])
        self.assertIsNone(params["emb"])
        self.assertEqual(results[0]["active_channels"], ["keyword", "trigram"])

    async def test_structural_expansion_is_scoped_and_merges_origins(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        seed_id = uuid.uuid4()
        shared_id = uuid.uuid4()
        section_id = uuid.uuid4()
        rows = [
            {
                "id": shared_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "content": "表格后一片",
                "chunk_index": 3,
                "metadata": {"heading": "职级分类", "type": "table"},
                "structural_origin": "table_sibling",
                "seed_chunk_id": seed_id,
                "structure_distance": 1,
            },
            {
                "id": shared_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "content": "表格后一片",
                "chunk_index": 3,
                "metadata": {"heading": "职级分类", "type": "table"},
                "structural_origin": "adjacent",
                "seed_chunk_id": seed_id,
                "structure_distance": 1,
            },
            {
                "id": section_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "content": "同章节补充",
                "chunk_index": 4,
                "metadata": {"heading": "职级分类"},
                "structural_origin": "same_section",
                "seed_chunk_id": seed_id,
                "structure_distance": 2,
            },
        ]
        db = _SavepointDB([_Rows(rows)])
        seed = {
            "id": seed_id,
            "doc_id": doc_id,
            "kb_id": kb_id,
            "chunk_index": 2,
            # 旧版元数据没有 section_key/table_id，必须按 heading/type 回退。
            "metadata": {"heading": "职级分类", "type": "table"},
        }

        results = await fetch_structural_neighbors(
            db,
            kb_ids=[kb_id],
            seed_candidates=[seed],
            neighbor_radius=99,
            same_section_limit=99,
            table_sibling_radius=99,
            total_limit=99,
        )

        statement, params = db.execute.await_args.args
        sql = str(statement)
        self.assertNotIn("eligible_chunks AS MATERIALIZED", sql)
        self.assertIn("eligible_documents AS MATERIALIZED", sql)
        self.assertGreaterEqual(sql.count("CROSS JOIN LATERAL"), 5)
        self.assertIn("dc.doc_id = ANY(:doc_ids)", sql)
        self.assertIn("dc.kb_id = ANY(:kb_ids)", sql)
        self.assertIn("d.id = dc.doc_id", sql)
        self.assertIn("d.kb_id = dc.kb_id", sql)
        self.assertIn("d.is_active = TRUE", sql)
        self.assertIn("d.status = 'ready'", sql)
        self.assertIn("dc.chunk_index IN", sql)
        self.assertIn("dc.metadata->>'heading' = seed.heading", sql)
        self.assertIn("dc.metadata->>'table_id' = seed.table_id", sql)
        self.assertIn("dc.chunk_index < seed.chunk_index", sql)
        self.assertIn("dc.chunk_index > seed.chunk_index", sql)
        self.assertIn("~ '^[0-9]{1,9}$'", sql)
        self.assertNotIn("NULLIF(ec.metadata->>'table_part_index', '')::integer", sql)
        self.assertEqual(params["same_section_limit"], 2)
        self.assertTrue(params["neighbors_enabled"])
        self.assertTrue(params["tables_enabled"])
        self.assertEqual(params["seed_specs"][0]["block_type"], "table")

        self.assertEqual(len(results), 2)
        shared = next(item for item in results if item["id"] == shared_id)
        self.assertEqual(
            shared["candidate_origins"],
            ["table_sibling", "adjacent"],
        )
        self.assertEqual(shared["expansion_seed_chunk_ids"], [str(seed_id)])
        self.assertEqual(db.savepoint_enters, 1)
        self.assertEqual(db.savepoint_exits, 1)

    async def test_structural_expansion_treats_dirty_table_part_as_unusable(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        seed_id = uuid.uuid4()
        db = _SavepointDB([_Rows([])])

        results = await fetch_structural_neighbors(
            db,
            kb_ids=[kb_id],
            seed_candidates=[{
                "id": seed_id,
                "doc_id": doc_id,
                "kb_id": kb_id,
                "chunk_index": 50,
                "metadata": {
                    "block_type": "table",
                    "table_id": "travel-standard",
                    "table_part_index": "1 OR invalid",
                },
            }],
        )

        statement, params = db.execute.await_args.args
        sql = str(statement)
        self.assertEqual(results, [])
        self.assertIsNone(params["seed_specs"][0]["table_part_index"])
        self.assertIn("~ '^[0-9]{1,9}$'", sql)
        self.assertIn("ELSE NULL", sql)
        self.assertNotIn("eligible_chunks AS MATERIALIZED", sql)

    async def test_structural_expansion_skips_invalid_or_unscoped_seeds(self) -> None:
        db = _db_with_rows([])
        results = await fetch_structural_neighbors(
            db,
            kb_ids=[uuid.uuid4()],
            seed_candidates=[{"id": "invalid", "doc_id": "also-invalid"}],
        )

        self.assertEqual(results, [])
        db.execute.assert_not_awaited()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
