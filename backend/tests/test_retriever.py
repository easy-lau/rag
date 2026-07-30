import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sqlalchemy.dialects.postgresql import ARRAY, UUID

from core.retriever import (
    PER_DOCUMENT_RERANK_CHUNKS,
    _build_trigram_terms,
    _candidate_pool_size,
    _normalize_result_scores,
    hybrid_search,
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
