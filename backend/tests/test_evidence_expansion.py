import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.evidence_expansion import (
    ExpansionBudget,
    expand_evidence_candidates,
    merge_expansion_candidates,
)


def _candidate(
    *,
    chunk_id=None,
    doc_id=None,
    index: int = 0,
    content: str = "候选内容",
    origin: str | None = None,
) -> dict:
    item = {
        "id": chunk_id or uuid.uuid4(),
        "doc_id": doc_id or uuid.uuid4(),
        "kb_id": uuid.uuid4(),
        "chunk_index": index,
        "content": content,
        "filename": "制度.md",
        "retrieval_score": 0.03,
        "score": 0.03,
        "metadata": {"heading": "制度"},
    }
    if origin:
        item["candidate_origin"] = origin
    return item


class CandidateMergeTests(unittest.TestCase):
    def test_same_chunk_merges_all_origins_without_inflating_global_score(self) -> None:
        chunk_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        seed_id = uuid.uuid4()
        initial = _candidate(
            chunk_id=chunk_id,
            doc_id=doc_id,
            origin="current_retrieval",
        )
        scoped = {
            **initial,
            "candidate_origin": "document_scoped",
            "candidate_origins": ["document_scoped"],
            "retrieval_score": 0.9,
            "score": 0.9,
            "document_scoped_score": 0.9,
            "expansion_query_indexes": [1],
            "expansion_seed_chunk_ids": [str(seed_id)],
        }
        adjacent = {
            **initial,
            "candidate_origin": "adjacent",
            "candidate_origins": ["adjacent"],
            "retrieval_score": None,
            "score": None,
            "expansion_seed_chunk_ids": [str(seed_id)],
            "expansion_sources": [{
                "origin": "adjacent",
                "seed_chunk_id": str(seed_id),
                "distance": 1,
            }],
        }

        outcome = merge_expansion_candidates(initial_candidates=[initial], added_candidates=[scoped, adjacent])

        self.assertEqual(len(outcome.candidates), 1)
        item = outcome.candidates[0]
        self.assertEqual(item["candidate_origin"], "current_retrieval")
        self.assertEqual(
            item["candidate_origins"],
            ["global_retrieval", "document_scoped", "adjacent"],
        )
        self.assertEqual(item["retrieval_score"], 0.03)
        self.assertEqual(item["document_scoped_score"], 0.9)
        self.assertEqual(item["expansion_query_indexes"], [1])
        self.assertEqual(item["expansion_seed_chunk_ids"], [str(seed_id)])
        self.assertEqual(outcome.added_candidate_count, 0)
        self.assertEqual(outcome.deduplicated_count, 2)

    def test_additions_joint_pool_and_character_budget_are_hard_bounded(self) -> None:
        doc_id = uuid.uuid4()
        initial = [
            _candidate(doc_id=doc_id, index=index, content="首轮")
            for index in range(25)
        ]
        additions = [
            _candidate(
                doc_id=doc_id,
                index=100 + index,
                content="扩展片段",
                origin="document_scoped",
            )
            for index in range(20)
        ]

        outcome = merge_expansion_candidates(
            initial,
            additions,
            budget=ExpansionBudget(
                max_added_candidates=99,
                max_joint_candidates=99,
                max_added_chars=99_999,
            ),
        )

        self.assertEqual(len(outcome.candidates), 30)
        self.assertEqual(outcome.added_candidate_count, 5)
        self.assertEqual(outcome.budget_dropped_count, 15)

        char_limited = merge_expansion_candidates(
            [],
            [
                _candidate(content="1234", origin="document_scoped"),
                _candidate(content="5678", origin="document_scoped"),
            ],
            budget=ExpansionBudget(max_added_chars=6),
        )
        self.assertEqual(char_limited.added_candidate_count, 1)
        self.assertEqual(char_limited.added_chars, 4)
        self.assertEqual(char_limited.budget_dropped_count, 1)

    def test_complete_small_document_has_priority_over_ordinary_pool(self) -> None:
        doc_id = uuid.uuid4()
        full_document = [
            _candidate(
                doc_id=doc_id,
                index=index,
                content="片段",
                origin="small_document_full",
            )
            for index in range(30)
        ]
        for item in full_document:
            item["candidate_origins"] = ["small_document_full"]
            item["retrieval_score"] = None
            item["score"] = None

        # 第 11 片既是首轮命中也是全文片段；应保留首轮分数并只占一个位置。
        global_seed = {
            **full_document[10],
            "candidate_origin": "current_retrieval",
            "candidate_origins": [],
            "retrieval_score": 0.08,
            "score": 0.08,
        }
        unrelated = [_candidate(index=100 + index) for index in range(5)]

        outcome = merge_expansion_candidates(
            [global_seed, *unrelated],
            [],
            priority_added_candidates=full_document,
        )

        self.assertEqual(len(outcome.candidates), 30)
        self.assertEqual(
            {item["id"] for item in outcome.candidates},
            {item["id"] for item in full_document},
        )
        self.assertEqual(outcome.added_candidate_count, 29)
        self.assertEqual(outcome.counts_by_origin["small_document_full"], 30)
        self.assertEqual(outcome.budget_dropped_count, 5)
        merged_seed = next(
            item for item in outcome.candidates if item["id"] == global_seed["id"]
        )
        self.assertEqual(merged_seed["retrieval_score"], 0.08)
        self.assertEqual(
            merged_seed["candidate_origins"],
            ["global_retrieval", "small_document_full"],
        )


class EvidenceExpansionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_document_loads_all_chunks_and_skips_redundant_search(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        seed = _candidate(doc_id=doc_id, index=1, content="普通员工属于D级")
        seed["kb_id"] = kb_id
        seed["rerank_candidate_index"] = 1
        full_document = []
        for index, content in enumerate(("职级说明", "普通员工属于D级", "D级住宿450元")):
            item = _candidate(doc_id=doc_id, index=index, content=content)
            item["kb_id"] = kb_id
            if index == 1:
                item["id"] = seed["id"]
            item.update({
                "candidate_origin": "small_document_full",
                "candidate_origins": ["small_document_full"],
                "retrieval_score": None,
                "score": None,
                "full_document_chunk_count": 3,
                "full_document_char_count": sum(map(len, (
                    "职级说明", "普通员工属于D级", "D级住宿450元",
                ))),
            })
            full_document.append(item)

        with (
            patch(
                "core.evidence_expansion.fetch_small_document_candidates",
                new=AsyncMock(return_value=full_document),
            ) as full_document_search,
            patch(
                "core.evidence_expansion.search_within_documents",
                new=AsyncMock(),
            ) as scoped_search,
            patch(
                "core.evidence_expansion.fetch_structural_neighbors",
                new=AsyncMock(),
            ) as structural_search,
        ):
            outcome = await expand_evidence_candidates(
                SimpleNamespace(),
                question="普通员工的出差标准是什么",
                kb_ids=[kb_id],
                initial_candidates=[seed],
                plan={
                    "should_expand": True,
                    "target_candidate_indexes": [1],
                    "secondary_queries": ["D级出差标准"],
                },
            )

        self.assertEqual(full_document_search.await_args.kwargs["doc_ids"], [doc_id])
        self.assertEqual(full_document_search.await_args.kwargs["max_chunks"], 30)
        self.assertEqual(full_document_search.await_args.kwargs["max_chars"], 16_000)
        scoped_search.assert_not_awaited()
        structural_search.assert_not_awaited()
        self.assertEqual([item["chunk_index"] for item in outcome.candidates], [0, 1, 2])
        self.assertEqual(outcome.added_candidate_count, 2)
        self.assertEqual(outcome.counts_by_origin["small_document_full"], 3)
        self.assertEqual(outcome.full_document_candidates, full_document)
        self.assertEqual(outcome.errors, ())

    async def test_large_seed_document_keeps_semantic_and_structural_path(self) -> None:
        kb_id = uuid.uuid4()
        small_doc = uuid.uuid4()
        large_doc = uuid.uuid4()
        small_seed = _candidate(doc_id=small_doc, index=0, content="普通员工属于D级")
        small_seed["kb_id"] = kb_id
        small_seed["rerank_candidate_index"] = 1
        large_seed = _candidate(doc_id=large_doc, index=10, content="大型制度目录")
        large_seed["kb_id"] = kb_id
        large_seed["rerank_candidate_index"] = 2
        full_small = {
            **small_seed,
            "candidate_origin": "small_document_full",
            "candidate_origins": ["small_document_full"],
            "retrieval_score": None,
            "score": None,
            "full_document_chunk_count": 1,
            "full_document_char_count": len(small_seed["content"]),
        }
        scoped = _candidate(
            doc_id=large_doc,
            index=20,
            content="大文档语义命中",
            origin="document_scoped",
        )
        scoped["kb_id"] = kb_id
        scoped["candidate_origins"] = ["document_scoped"]
        structural = _candidate(
            doc_id=large_doc,
            index=21,
            content="大文档相邻片段",
            origin="adjacent",
        )
        structural["kb_id"] = kb_id
        structural["candidate_origins"] = ["adjacent"]

        with (
            patch(
                "core.evidence_expansion.fetch_small_document_candidates",
                new=AsyncMock(return_value=[full_small]),
            ),
            patch(
                "core.evidence_expansion.search_within_documents",
                new=AsyncMock(return_value=[scoped]),
            ) as scoped_search,
            patch(
                "core.evidence_expansion.fetch_structural_neighbors",
                new=AsyncMock(return_value=[structural]),
            ) as structural_search,
        ):
            outcome = await expand_evidence_candidates(
                SimpleNamespace(),
                question="制度标准是什么",
                kb_ids=[kb_id],
                initial_candidates=[small_seed, large_seed],
                plan={
                    "should_expand": True,
                    "target_candidate_indexes": [1, 2],
                    "secondary_queries": ["制度详细标准"],
                },
            )

        self.assertEqual(scoped_search.await_args.kwargs["doc_ids"], [large_doc])
        self.assertTrue(all(
            item["doc_id"] == large_doc
            for item in structural_search.await_args.kwargs["seed_candidates"]
        ))
        candidate_ids = {item["id"] for item in outcome.candidates}
        self.assertIn(full_small["id"], candidate_ids)
        self.assertIn(scoped["id"], candidate_ids)
        self.assertIn(structural["id"], candidate_ids)
        self.assertEqual(outcome.counts_by_origin["small_document_full"], 1)
        self.assertEqual(outcome.counts_by_origin["document_scoped"], 1)
        self.assertEqual(outcome.counts_by_origin["adjacent"], 1)

    async def test_scoped_failure_still_allows_structural_candidates(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        seed = _candidate(doc_id=doc_id, index=2, content="普通员工属于D级")
        structural = _candidate(
            doc_id=doc_id,
            index=3,
            content="D级经济舱",
            origin="adjacent",
        )
        structural["candidate_origins"] = ["adjacent"]
        plan = {
            "needed": True,
            "target_candidate_indexes": [1],
            "queries": ["D级交通标准"],
        }

        with (
            patch(
                "core.evidence_expansion.fetch_small_document_candidates",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "core.evidence_expansion.search_within_documents",
                new=AsyncMock(side_effect=RuntimeError("scoped SQL failed")),
            ),
            patch(
                "core.evidence_expansion.fetch_structural_neighbors",
                new=AsyncMock(return_value=[structural]),
            ) as structural_search,
        ):
            outcome = await expand_evidence_candidates(
                SimpleNamespace(),
                question="普通员工的出差标准是什么",
                kb_ids=[kb_id],
                initial_candidates=[seed],
                plan=plan,
            )

        structural_search.assert_awaited_once()
        self.assertTrue(outcome.expanded)
        self.assertEqual(outcome.errors, ("document_scoped:RuntimeError",))
        self.assertIn(structural["id"], {item["id"] for item in outcome.candidates})

    async def test_reranker_plan_maps_one_based_indexes_before_using_current_order(self) -> None:
        kb_id = uuid.uuid4()
        doc_a = uuid.uuid4()
        doc_b = uuid.uuid4()
        # 当前列表已经重排：原始第 2 条排在前面，原始第 1 条排在后面。
        original_second = _candidate(doc_id=doc_b, content="其它候选")
        original_second["rerank_candidate_index"] = 2
        original_first = _candidate(doc_id=doc_a, content="普通员工属于D级")
        original_first["rerank_candidate_index"] = 1
        original_first["bridge_facts"] = [{
            "subject": "普通员工",
            "relation": "属于",
            "object": "D级",
        }]
        plan = SimpleNamespace(
            needed=True,
            target_candidate_indexes=(1,),
            queries=(),
            missing_requirement_ids=("travel_standard",),
        )

        with (
            patch(
                "core.evidence_expansion.fetch_small_document_candidates",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "core.evidence_expansion.search_within_documents",
                new=AsyncMock(return_value=[]),
            ) as scoped_search,
            patch(
                "core.evidence_expansion.fetch_structural_neighbors",
                new=AsyncMock(return_value=[]),
            ) as structural_search,
        ):
            outcome = await expand_evidence_candidates(
                SimpleNamespace(),
                question="普通员工的出差标准是什么",
                kb_ids=[kb_id],
                initial_candidates=[original_second, original_first],
                plan=plan,
            )

        self.assertTrue(outcome.expanded)
        self.assertEqual([seed["id"] for seed in outcome.seed_candidates], [original_first["id"]])
        self.assertEqual(scoped_search.await_args.kwargs["doc_ids"], [doc_a])
        self.assertEqual(
            scoped_search.await_args.kwargs["queries"],
            ["普通员工的出差标准是什么", "普通员工的出差标准是什么 普通员工 D级"],
        )
        self.assertEqual(
            structural_search.await_args.kwargs["seed_candidates"][0]["id"],
            original_first["id"],
        )

    async def test_duck_typed_plan_only_expands_documents_from_initial_candidates(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        seed = _candidate(doc_id=doc_id, index=2, content="普通员工属于D级")
        seed["kb_id"] = kb_id
        scoped = _candidate(
            doc_id=doc_id,
            index=3,
            content="D级飞机为经济舱",
            origin="document_scoped",
        )
        scoped["kb_id"] = kb_id
        scoped["candidate_origins"] = ["document_scoped"]
        structural = _candidate(
            doc_id=doc_id,
            index=4,
            content="D级火车为二等座",
            origin="adjacent",
        )
        structural["kb_id"] = kb_id
        structural["candidate_origins"] = ["adjacent"]
        invalid_doc_id = uuid.uuid4()
        plan = SimpleNamespace(
            should_expand=True,
            seed_chunk_ids=[seed["id"]],
            # 任意额外 doc_id 不得进入 SQL 作用域。
            seed_doc_ids=[doc_id, invalid_doc_id],
            bridge_terms=["D级"],
            secondary_queries=["D级交通住宿餐饮补贴标准"],
            required_facets=["交通", "住宿", "餐饮"],
        )

        with (
            patch(
                "core.evidence_expansion.fetch_small_document_candidates",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "core.evidence_expansion.search_within_documents",
                new=AsyncMock(return_value=[scoped]),
            ) as scoped_search,
            patch(
                "core.evidence_expansion.fetch_structural_neighbors",
                new=AsyncMock(return_value=[structural]),
            ) as structural_search,
        ):
            outcome = await expand_evidence_candidates(
                SimpleNamespace(),
                question="普通员工的出差标准是什么",
                kb_ids=[kb_id],
                initial_candidates=[seed],
                plan=plan,
            )

        self.assertTrue(outcome.expanded)
        self.assertEqual(scoped_search.await_args.kwargs["doc_ids"], [doc_id])
        self.assertEqual(
            scoped_search.await_args.kwargs["queries"],
            ["普通员工的出差标准是什么", "D级交通住宿餐饮补贴标准"],
        )
        structural_seeds = structural_search.await_args.kwargs["seed_candidates"]
        self.assertEqual([item["id"] for item in structural_seeds], [seed["id"], scoped["id"]])
        self.assertEqual(len(outcome.candidates), 3)
        self.assertEqual(outcome.added_candidate_count, 2)
        self.assertEqual(outcome.counts_by_origin["global_retrieval"], 1)
        self.assertEqual(outcome.counts_by_origin["document_scoped"], 1)
        self.assertEqual(outcome.counts_by_origin["adjacent"], 1)

    async def test_false_plan_returns_bounded_initial_pool_without_database_calls(self) -> None:
        candidates = [_candidate(index=index) for index in range(35)]
        with (
            patch(
                "core.evidence_expansion.search_within_documents",
                new=AsyncMock(),
            ) as scoped_search,
            patch(
                "core.evidence_expansion.fetch_structural_neighbors",
                new=AsyncMock(),
            ) as structural_search,
        ):
            outcome = await expand_evidence_candidates(
                SimpleNamespace(),
                question="一个片段已经完整回答",
                kb_ids=[uuid.uuid4()],
                initial_candidates=candidates,
                plan={"should_expand": False},
            )

        self.assertFalse(outcome.expanded)
        self.assertEqual(len(outcome.candidates), 30)
        self.assertEqual(outcome.budget_dropped_count, 5)
        scoped_search.assert_not_awaited()
        structural_search.assert_not_awaited()

    async def test_invalid_plan_seed_ids_cannot_expand_arbitrary_document(self) -> None:
        actual_doc = uuid.uuid4()
        seed = _candidate(doc_id=actual_doc)
        with (
            patch(
                "core.evidence_expansion.fetch_small_document_candidates",
                new=AsyncMock(return_value=[]),
            ) as full_document_search,
            patch(
                "core.evidence_expansion.search_within_documents",
                new=AsyncMock(return_value=[]),
            ) as scoped_search,
            patch(
                "core.evidence_expansion.fetch_structural_neighbors",
                new=AsyncMock(return_value=[]),
            ),
        ):
            await expand_evidence_candidates(
                SimpleNamespace(),
                question="制度标准",
                kb_ids=[uuid.uuid4()],
                initial_candidates=[seed],
                plan={
                    "should_expand": True,
                    "seed_doc_ids": [uuid.uuid4()],
                    "secondary_queries": ["补充查询"],
                },
            )

        self.assertEqual(
            full_document_search.await_args.kwargs["doc_ids"],
            [actual_doc],
        )
        self.assertEqual(scoped_search.await_args.kwargs["doc_ids"], [actual_doc])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
