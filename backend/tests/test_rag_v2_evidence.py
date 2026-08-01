import unittest

from core.query_constraints import extract_query_constraints
from core.rag_v2.evidence import assemble_evidence_bundle


def _candidate(
    chunk_id: str,
    *,
    doc_id: str = "doc-a",
    kb_id: str = "kb-a",
    chunk_index: int = 0,
    content: str = "具体条款：住宿标准为450元/天。",
    **values,
) -> dict:
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "kb_id": kb_id,
        "chunk_index": chunk_index,
        "content": content,
        **values,
    }


class EvidenceBundleAssemblyTests(unittest.TestCase):
    def test_hard_mismatch_and_unauthorized_candidates_are_excluded(self) -> None:
        constraints = extract_query_constraints("云枢8.2.75消息接口怎么配置")
        candidates = [
            _candidate(
                "a-2",
                chunk_index=2,
                content="所属产品：云枢；产品版本：8.2.75。配置项B。",
                rerank_status="verified",
            ),
            _candidate(
                "b-0",
                doc_id="doc-b",
                content="所属产品：云枢；产品版本：7.0。旧配置项。",
                rerank_status="verified",
            ),
            _candidate(
                "a-0",
                chunk_index=0,
                content="所属产品：云枢；产品版本：8.2.75。",
                rerank_status="verified",
            ),
            _candidate("secret", doc_id="doc-secret", authorized=False),
        ]

        bundle = assemble_evidence_bundle(
            query="云枢8.2.75消息接口怎么配置",
            candidates=candidates,
            constraints=constraints,
            rerank_succeeded=True,
        )

        self.assertEqual([item.chunk_id for item in bundle.items], ["a-0", "a-2"])
        self.assertTrue(all(item.doc_id == "doc-a" for item in bundle.items))
        self.assertIn("hard_constraint_mismatch_excluded", bundle.state.reasons)
        self.assertIn("unauthorized_candidate_excluded", bundle.state.reasons)
        self.assertEqual(bundle.state.confidence, "verified")

    def test_explicit_product_version_excludes_unknown_scope(self) -> None:
        constraints = extract_query_constraints("云枢8.6登录配置")
        bundle = assemble_evidence_bundle(
            query="云枢8.6登录配置",
            candidates=[
                _candidate(
                    "exact",
                    content="所属产品：云枢；产品版本：8.6。登录配置A。",
                ),
                _candidate(
                    "unknown",
                    doc_id="doc-generic",
                    content="通用登录配置B。",
                ),
            ],
            constraints=constraints,
        )

        self.assertEqual([item.chunk_id for item in bundle.items], ["exact"])
        self.assertIn("hard_constraint_unknown_excluded", bundle.state.reasons)

    def test_items_are_grouped_by_document_and_sorted_by_chunk_index(self) -> None:
        bundle = assemble_evidence_bundle(
            query="配置标准",
            candidates=[
                _candidate("a-3", chunk_index=3),
                _candidate("b-2", doc_id="doc-b", chunk_index=2),
                _candidate("a-1", chunk_index=1),
                _candidate("b-0", doc_id="doc-b", chunk_index=0),
            ],
        )

        self.assertEqual(
            [(item.doc_id, item.chunk_index) for item in bundle.items],
            [("doc-a", 1), ("doc-a", 3), ("doc-b", 0), ("doc-b", 2)],
        )

    def test_rerank_failure_downgrades_but_does_not_erase_candidates(self) -> None:
        bundle = assemble_evidence_bundle(
            query="普通员工住宿标准",
            candidates=[
                _candidate(
                    "retrieved",
                    evidence_role="irrelevant",
                    rerank_status="unverified",
                )
            ],
            rerank_succeeded=False,
            completeness="complete",
        )

        self.assertEqual(bundle.state.availability, "degraded")
        self.assertEqual(bundle.state.confidence, "retrieved")
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual([item.chunk_id for item in bundle.context_items], ["retrieved"])
        self.assertEqual([item.chunk_id for item in bundle.answer_sources], ["retrieved"])
        self.assertIn("rerank_degraded", bundle.state.reasons)

    def test_expansion_degradation_is_independent_from_confidence_and_completeness(self) -> None:
        bundle = assemble_evidence_bundle(
            query="普通员工住宿标准",
            candidates=[
                _candidate("verified", rerank_status="verified")
            ],
            rerank_succeeded=True,
            expansion_succeeded=False,
            completeness="complete",
        )

        self.assertEqual(bundle.state.availability, "degraded")
        self.assertEqual(bundle.state.confidence, "verified")
        self.assertEqual(bundle.state.completeness, "complete")
        self.assertEqual(len(bundle.items), 1)
        self.assertIn("expansion_degraded", bundle.state.reasons)

    def test_missing_requirements_mark_partial_without_clearing_context(self) -> None:
        bundle = assemble_evidence_bundle(
            query="交通和住宿标准",
            candidates=[_candidate("住宿")],
            completeness="complete",
            missing_requirement_ids=("transport",),
        )

        self.assertEqual(bundle.state.completeness, "partial")
        self.assertEqual(bundle.missing_requirement_ids, ("transport",))
        self.assertEqual(bundle.context_item_ids, ("住宿",))

    def test_concrete_table_outranks_boilerplate_unless_explicitly_requested(self) -> None:
        candidates = [
            _candidate(
                "overview",
                chunk_index=0,
                content="一、总则\n为规范公司员工出差管理，制定本制度。",
                evidence_role="direct",
            ),
            _candidate(
                "table",
                chunk_index=1,
                content="| 职级 | 住宿标准 |\n| --- | --- |\n| D级 | 450元/天 |",
                evidence_role="related",
            ),
        ]

        concrete = assemble_evidence_bundle(
            query="普通员工住宿标准是多少",
            candidates=candidates,
            max_context_chunks=1,
        )
        overview = assemble_evidence_bundle(
            query="这份制度的总则是什么",
            candidates=candidates,
            answer_shape="overview",
            max_context_chunks=1,
        )

        self.assertEqual(concrete.context_item_ids, ("table",))
        self.assertEqual(overview.context_item_ids, ("overview",))

    def test_overview_accepts_only_full_document_chunks_anchored_by_retrieval(self) -> None:
        bundle = assemble_evidence_bundle(
            query="请概述这份制度的主要内容",
            answer_shape="overview",
            candidates=[
                _candidate("anchor", content="公司出差管理标准")
            ],
            overview_candidates=[
                _candidate("full-1", chunk_index=1, content="一、总则：规范出差管理。"),
                _candidate(
                    "foreign",
                    doc_id="doc-foreign",
                    content="不属于已授权召回文档的全文。",
                ),
            ],
            rerank_succeeded=True,
        )

        self.assertEqual(
            {item.chunk_id for item in bundle.items},
            {"anchor", "full-1"},
        )
        full = next(item for item in bundle.items if item.chunk_id == "full-1")
        self.assertEqual(full.confidence, "retrieved")
        self.assertIn("overview_full_document", full.origins)
        self.assertIn("unanchored_overview_candidate_excluded", bundle.state.reasons)

    def test_context_chunk_and_character_budgets_are_hard_limits(self) -> None:
        bundle = assemble_evidence_bundle(
            query="标准",
            candidates=[
                _candidate("first", content="1234567890", score=1.0),
                _candidate("second", chunk_index=1, content="abcdefghij", score=0.5),
            ],
            completeness="complete",
            max_context_chunks=1,
            max_context_chars=5,
        )

        self.assertEqual(bundle.context_item_ids, ("first",))
        self.assertEqual(bundle.context_items[0].content, "12345")
        self.assertTrue(bundle.context_items[0].metadata["context_truncated"])
        self.assertIn("context_budget_limited", bundle.state.reasons)
        self.assertEqual(bundle.state.completeness, "partial")

    def test_all_filtered_candidates_are_a_normal_empty_result(self) -> None:
        bundle = assemble_evidence_bundle(
            query="云枢8.2.75配置",
            candidates=[
                _candidate("wrong", constraint_status="mismatch")
            ],
        )

        self.assertEqual(bundle.state.availability, "ok")
        self.assertEqual(bundle.state.confidence, "none")
        self.assertEqual(bundle.items, ())
        self.assertEqual(bundle.context_items, ())

    def test_empty_retrieval_is_no_evidence_not_infrastructure_unavailable(self) -> None:
        bundle = assemble_evidence_bundle(
            query="知识库中不存在的问题",
            candidates=[],
        )

        self.assertEqual(bundle.state.availability, "ok")
        self.assertEqual(bundle.state.confidence, "none")
        self.assertEqual(bundle.state.completeness, "unknown")
        self.assertIn("no_usable_authorized_evidence", bundle.state.reasons)

    def test_source_metadata_needed_by_sse_is_preserved(self) -> None:
        bundle = assemble_evidence_bundle(
            query="配置",
            candidates=[
                _candidate(
                    "source",
                    filename="配置说明.md",
                    file_type="markdown",
                    source_url="https://kb.example/doc/source",
                    doc_tags=["安全", "配置"],
                    retrieval_score=0.82,
                    answer_support=0.91,
                )
            ],
        )

        metadata = bundle.items[0].metadata
        self.assertEqual(metadata["filename"], "配置说明.md")
        self.assertEqual(metadata["file_type"], "markdown")
        self.assertEqual(metadata["source_url"], "https://kb.example/doc/source")
        self.assertEqual(metadata["doc_tags"], ["安全", "配置"])
        self.assertEqual(metadata["retrieval_score"], 0.82)
        self.assertEqual(metadata["answer_support"], 0.91)


if __name__ == "__main__":
    unittest.main()
