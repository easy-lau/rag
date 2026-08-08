import unittest

from core.evidence_admission import (
    MAX_DOC_VECTOR_GAP,
    MIN_VECTOR_SCORE,
    admit_evidence_candidates,
    assess_document_relevance,
)


def _candidate(doc_id: str, **signals) -> dict:
    return {
        "id": f"chunk-{doc_id}-{len(signals)}",
        "doc_id": doc_id,
        "kb_id": "kb-a",
        "content": "generic retrieval candidate",
        **signals,
    }


class DocumentRelevanceAdmissionTests(unittest.TestCase):
    def test_candidate_admission_rejects_rank_only_noise(self) -> None:
        admission = admit_evidence_candidates(
            [
                _candidate(
                    "noise",
                    retrieval_score=1 / 61,
                    fusion_rank=1,
                ),
            ],
            query="查询用户列表使用哪个接口",
        )

        self.assertEqual(admission.status, "rejected")
        self.assertFalse(admission.candidates)
        self.assertEqual(len(admission.rejections), 1)
        self.assertEqual(
            admission.rejections[0].reason,
            "document_relevance_gate",
        )

    def test_candidate_admission_keeps_strong_document_and_rejects_noise(
        self,
    ) -> None:
        admission = admit_evidence_candidates(
            [
                _candidate("target", vector_score=0.86),
                _candidate("noise", vector_score=0.60),
            ],
            query="查询用户列表使用哪个接口",
        )

        self.assertEqual(admission.status, "admitted")
        self.assertEqual(
            [item["doc_id"] for item in admission.candidates],
            ["target"],
        )
        self.assertEqual(
            [item.doc_id for item in admission.rejections],
            ["noise"],
        )

    def test_structured_record_score_anchors_its_source_document(self) -> None:
        record = _candidate(
            "catalog",
            source_kind="knowledge_record",
            record_id="record-1",
            structured_score=0.42,
        )
        chunk = _candidate("catalog", vector_score=0.40)

        admission = admit_evidence_candidates(
            [record, chunk],
            query="查询用户列表使用哪个接口",
        )

        self.assertEqual(len(admission.candidates), 2)
        self.assertEqual(
            admission.candidates[1]["admission_reason"],
            "structured_record_document_anchor",
        )

    def test_empty_and_invalid_document_candidates_admit_nothing(self) -> None:
        empty = assess_document_relevance([])
        invalid = assess_document_relevance([
            {"vector_score": 0.99},
            object(),
        ])

        self.assertEqual(empty.admitted_doc_ids, ())
        self.assertEqual(empty.rejected_doc_ids, ())
        self.assertEqual(empty.reason, "no_candidates")
        self.assertEqual(invalid.admitted_doc_ids, ())
        self.assertEqual(invalid.rejected_doc_ids, ())
        self.assertEqual(invalid.reason, "no_valid_documents")

    def test_low_vector_nearest_neighbors_produce_empty_admission(self) -> None:
        decision = assess_document_relevance([
            _candidate("doc-a", vector_score=MIN_VECTOR_SCORE - 0.001),
            _candidate("doc-b", vector_score=0.51),
        ])

        self.assertEqual(decision.admitted_doc_ids, ())
        self.assertEqual(decision.rejected_doc_ids, ("doc-a", "doc-b"))
        self.assertEqual(
            decision.reason,
            "no_document_met_lexical_or_vector_gate",
        )

    def test_document_best_vector_separates_target_from_lower_noise(self) -> None:
        decision = assess_document_relevance([
            _candidate("target", vector_score=0.86),
            _candidate("target", vector_score=0.82),
            _candidate("noise", vector_score=0.824),
            _candidate("lower-noise", vector_score=0.79),
        ])

        self.assertEqual(decision.admitted_doc_ids, ("target",))
        self.assertEqual(
            decision.rejected_doc_ids,
            ("noise", "lower-noise"),
        )
        self.assertEqual(
            decision.reason,
            "admitted_by_vector_score_and_global_gap",
        )

    def test_multiple_lexical_hit_documents_are_all_retained(self) -> None:
        decision = assess_document_relevance([
            _candidate("keyword-doc", keyword_rank=1, keyword_score=0.01, vector_score=0.2),
            _candidate("trigram-doc", trigram_score=0.18),
            _candidate("channel-doc", active_channels=["trigram"], trigram_score=0.12),
            _candidate("noise", vector_score=MIN_VECTOR_SCORE - 0.01),
        ])

        self.assertEqual(
            decision.admitted_doc_ids,
            ("keyword-doc", "trigram-doc", "channel-doc"),
        )
        self.assertEqual(decision.rejected_doc_ids, ("noise",))
        self.assertEqual(decision.reason, "admitted_by_lexical_evidence")

    def test_absolute_vector_threshold_is_inclusive(self) -> None:
        decision = assess_document_relevance([
            _candidate("boundary", vector_score=MIN_VECTOR_SCORE),
            _candidate("below", vector_score=MIN_VECTOR_SCORE - 0.000001),
        ])

        self.assertEqual(decision.admitted_doc_ids, ("boundary",))
        self.assertEqual(decision.rejected_doc_ids, ("below",))

    def test_global_vector_gap_boundary_is_inclusive(self) -> None:
        global_best = 0.9
        at_gap = global_best - MAX_DOC_VECTOR_GAP
        decision = assess_document_relevance([
            _candidate("best", vector_score=global_best),
            _candidate("at-gap", vector_score=at_gap),
            _candidate("outside-gap", vector_score=at_gap - 0.000001),
        ])

        self.assertEqual(decision.admitted_doc_ids, ("best", "at-gap"))
        self.assertEqual(decision.rejected_doc_ids, ("outside-gap",))

    def test_rank_fusion_score_does_not_masquerade_as_raw_vector_score(self) -> None:
        decision = assess_document_relevance([
            _candidate(
                "rank-only",
                score=0.99,
                retrieval_score=0.99,
            )
        ])

        self.assertEqual(decision.admitted_doc_ids, ())
        self.assertEqual(decision.rejected_doc_ids, ("rank-only",))

    def test_lexical_and_vector_paths_report_combined_reason(self) -> None:
        decision = assess_document_relevance([
            _candidate("lexical", keyword_score=0.1),
            _candidate("vector", vector_score=0.9),
        ])

        self.assertEqual(decision.admitted_doc_ids, ("lexical", "vector"))
        self.assertEqual(
            decision.reason,
            "admitted_by_lexical_or_vector_evidence",
        )

    def test_rank_or_channel_without_a_score_is_not_lexical_evidence(self) -> None:
        decision = assess_document_relevance([
            _candidate("rank-only", keyword_rank=1),
            _candidate("channel-only", active_channels=["trigram"]),
            _candidate("near-zero", keyword_score=1e-9),
        ])

        self.assertEqual(decision.admitted_doc_ids, ())
        self.assertEqual(
            decision.rejected_doc_ids,
            ("rank-only", "channel-only", "near-zero"),
        )

    def test_invalid_thresholds_are_rejected(self) -> None:
        for kwargs in (
            {"min_vector_score": -0.1},
            {"min_vector_score": float("nan")},
            {"max_doc_vector_gap": 1.1},
            {"max_doc_vector_gap": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                assess_document_relevance([], **kwargs)

    def test_output_order_follows_first_document_appearance(self) -> None:
        decision = assess_document_relevance([
            # Rank metadata alone is intentionally not evidence.  Use scored
            # lexical observations so this test exercises output ordering
            # after the relevance gate admits both documents.
            _candidate("second", trigram_rank=2, trigram_score=0.12),
            _candidate("first", keyword_rank=1, keyword_score=0.01),
            _candidate("second", keyword_rank=3, keyword_score=0.01),
        ])

        self.assertEqual(decision.admitted_doc_ids, ("second", "first"))
        self.assertEqual(decision.to_dict()["admitted_doc_ids"], ["second", "first"])

    def test_high_recall_admission_keeps_generic_lexical_hits_for_evidence_graph(self) -> None:
        decision = assess_document_relevance(
            [
                _candidate(
                    "travel",
                    filename="公司出差管理标准.docx",
                    content="员工出差交通、住宿和餐饮补贴标准",
                    keyword_score=0.02,
                ),
                _candidate(
                    "leave",
                    filename="员工请假管理办法.docx",
                    content="员工请假审批和休假要求",
                    trigram_score=0.18,
                ),
            ],
            query="不存在的火星基地量子补贴标准是什么",
        )
        self.assertEqual(decision.admitted_doc_ids, ("travel", "leave"))
        self.assertEqual(decision.reason, "admitted_by_lexical_evidence")

    def test_unscoped_product_query_keeps_one_representative_per_version(self) -> None:
        decision = assess_document_relevance(
            [
                _candidate(
                    "cloudpivot-7",
                    vector_score=0.88,
                    filename="云枢7配置说明.md",
                    content="云枢 登录 强制 修改 密码 应该 怎么办 配置",
                    metadata={"产品名称": "云枢", "产品版本": "7"},
                ),
                _candidate(
                    "cloudpivot-6",
                    vector_score=0.80,
                    filename="云枢6配置说明.md",
                    content="云枢 登录 强制 修改 密码 应该 怎么办 配置",
                    metadata={"产品名称": "云枢", "产品版本": "6"},
                ),
            ],
            query="产品：云枢，登录强制修改密码应该怎么办",
        )
        self.assertEqual(
            decision.admitted_doc_ids,
            ("cloudpivot-7", "cloudpivot-6"),
        )
        self.assertEqual(
            decision.reason,
            "admitted_by_each_explicit_version_representative",
        )

    def test_query_topic_gate_keeps_strong_vector_without_exact_terms(self) -> None:
        decision = assess_document_relevance(
            [
                _candidate(
                    "semantic",
                    content="员工出差管理制度",
                    vector_score=0.90,
                ),
            ],
            query="差旅费用怎么核算",
        )

        self.assertEqual(decision.admitted_doc_ids, ("semantic",))


if __name__ == "__main__":
    unittest.main()
