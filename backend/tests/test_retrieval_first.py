import asyncio
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.evidence_contract import (
    CandidateRejection,
    EvidencePack,
    RetrievalChannelReport,
)
from core.evidence_retrieval_service import (
    EvidenceRetrievalService,
    _fuse_ranked_channels,
)
from core.retrieval_first_runner import run_retrieval_first_stream


KB_ID = "11111111-1111-4111-8111-111111111111"
DOC_1 = "22222222-2222-4222-8222-222222222222"
DOC_2 = "33333333-3333-4333-8333-333333333333"
CHUNK_1 = "44444444-4444-4444-8444-444444444444"
CHUNK_2 = "55555555-5555-4555-8555-555555555555"


def _chunk(
    *,
    chunk_id: str = CHUNK_1,
    doc_id: str = DOC_1,
    content: str = "分页查询组织下的员工 Code /mozi/employee/page",
) -> dict:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "kb_id": KB_ID,
        "filename": "云枢使用的专有钉接口列表",
        "source_kind": "document_chunk",
        "vector_score": 0.86,
        "retrieval_score": 0.42,
        "content": content,
    }


def _record(
    subject: str = "分页查询组织下的员工 Code",
    endpoint: str = "/mozi/employee/page",
    *,
    record_id: str = "66666666-6666-4666-8666-666666666666",
    chunk_id: str = CHUNK_1,
    doc_id: str = DOC_1,
) -> dict:
    return {
        "id": record_id,
        "record_id": record_id,
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "kb_id": KB_ID,
        "filename": "接口目录",
        "record_type": "table_row",
        "source_kind": "knowledge_record",
        "subject": subject,
        "content": f"{subject} {endpoint}",
        "structured_score": 0.7,
        "retrieval_score": 0.7,
    }


def _complete(candidate: dict, *, evidence_set_id: str = "set_1") -> dict:
    return {
        **candidate,
        "jointly_selected": True,
        "joint_rerank_status": "verified_joint",
        "coverage_status": "complete",
        "evidence_role": "direct",
        "joint_support_score": 0.9,
        "evidence_set_id": evidence_set_id,
    }


async def _empty_search(*_args, **_kwargs):
    return []


class EvidenceRetrievalServiceTests(unittest.TestCase):
    def test_channel_fusion_uses_rank_not_incompatible_raw_scores(self) -> None:
        first = _chunk()
        low_raw_score = {**first, "retrieval_score": 0.01}
        other = _chunk(chunk_id=CHUNK_2, doc_id=DOC_2, content="其他候选")

        fused = _fuse_ranked_channels(
            (
                ("vector", [low_raw_score, other]),
                ("keyword", [first]),
            ),
            limit=5,
        )

        self.assertEqual(fused[0]["chunk_id"], CHUNK_1)
        self.assertEqual(fused[0]["retrieval_channels"], ["vector", "keyword"])
        self.assertEqual(fused[0]["channel_ranks"], {"vector": 1, "keyword": 1})

    def test_empty_successful_retrieval_is_true_no_hit(self) -> None:
        service = EvidenceRetrievalService(
            structured_search=_empty_search,
            chunk_search=_empty_search,
        )

        pack = asyncio.run(self._retrieve(service, verify=False))

        self.assertEqual(pack.outcome, "no_hit")
        self.assertEqual(pack.retrieval_status, "no_hit")
        self.assertFalse(pack.candidates)

    def test_primary_retrieval_failure_is_service_unavailable(self) -> None:
        async def failed_chunks(*_args, **_kwargs):
            raise TimeoutError("embedding timeout")

        service = EvidenceRetrievalService(
            structured_search=_empty_search,
            chunk_search=failed_chunks,
        )

        pack = asyncio.run(self._retrieve(service, verify=False))

        self.assertEqual(pack.outcome, "service_unavailable")
        self.assertEqual(pack.reason, "primary_retrieval_channel_failed")

    def test_rank_only_candidates_are_insufficient_without_calling_verifier(
        self,
    ) -> None:
        verifier = Mock(side_effect=AssertionError("verifier must not run"))
        noise = {
            **_chunk(),
            "vector_score": None,
            "keyword_score": None,
            "trigram_score": None,
            "retrieval_score": 1 / 61,
        }

        async def chunks(*_args, **_kwargs):
            return [noise]

        service = EvidenceRetrievalService(
            structured_search=_empty_search,
            chunk_search=chunks,
            verifier=verifier,
        )

        pack = asyncio.run(self._retrieve(service, verify=True))

        self.assertEqual(pack.retrieval_status, "hit")
        self.assertEqual(pack.admission_status, "rejected")
        self.assertEqual(pack.outcome, "insufficient_evidence")
        self.assertFalse(pack.selected_evidence)
        verifier.assert_not_called()

    def test_verifier_failure_falls_back_only_to_admitted_candidates(self) -> None:
        target = _chunk()
        noise = _chunk(
            chunk_id=CHUNK_2,
            doc_id=DOC_2,
            content="无关升级说明",
        )
        noise["vector_score"] = 0.60
        verifier_inputs: list[list[dict]] = []

        async def chunks(*_args, **_kwargs):
            return [target, noise]

        async def verifier(_query, candidates, *_args, **_kwargs):
            verifier_inputs.append(candidates)
            return SimpleNamespace(
                succeeded=False,
                results=candidates,
                error="provider timeout",
                failure_kind="timeout",
            )

        service = EvidenceRetrievalService(
            structured_search=_empty_search,
            chunk_search=chunks,
            verifier=verifier,
        )

        pack = asyncio.run(self._retrieve(service, verify=True))

        self.assertEqual([item["doc_id"] for item in verifier_inputs[0]], [DOC_1])
        self.assertEqual([item["doc_id"] for item in pack.selected_evidence], [DOC_1])
        self.assertEqual(len(pack.candidates), 2)
        self.assertEqual(len(pack.admission_rejections), 1)

    def test_verifier_failure_preserves_retrieved_evidence(self) -> None:
        candidate = _record()

        async def records(*_args, **_kwargs):
            return [candidate]

        async def verifier(*_args, **_kwargs):
            return SimpleNamespace(
                succeeded=False,
                results=[candidate],
                error="provider timeout",
                failure_kind="timeout",
                model="test-reranker",
            )

        service = EvidenceRetrievalService(
            structured_search=records,
            chunk_search=_empty_search,
            verifier=verifier,
        )

        pack = asyncio.run(self._retrieve(service, verify=True))

        self.assertEqual(pack.outcome, "answered")
        self.assertEqual(pack.retrieval_status, "hit")
        self.assertEqual(pack.verification_status, "unverified")
        self.assertEqual(pack.reason, "verification_unavailable")
        self.assertEqual(pack.selected_evidence[0]["record_id"], candidate["record_id"])
        self.assertEqual(pack.trace["verification_failure_kind"], "timeout")

    def test_failed_verifier_cannot_promote_or_force_clarification(self) -> None:
        first = _record()
        second = _record(
            "根据标签获取人员列表",
            "/employees/by-tag",
            record_id="77777777-7777-4777-8777-777777777777",
            chunk_id=CHUNK_2,
            doc_id=DOC_2,
        )

        async def records(*_args, **_kwargs):
            return [first, second]

        async def verifier(*_args, **_kwargs):
            return SimpleNamespace(
                succeeded=False,
                results=[
                    {**_complete(first), "rerank_candidate_index": 1},
                    {**second, "rerank_candidate_index": 2},
                ],
                decision_status="ambiguous",
                ambiguity_candidate_indexes=(1, 2),
                error="invalid verifier contract",
                failure_kind="contract_validation",
            )

        service = EvidenceRetrievalService(
            structured_search=records,
            chunk_search=_empty_search,
            verifier=verifier,
        )

        pack = asyncio.run(self._retrieve(service, verify=True))

        self.assertEqual(pack.outcome, "answered")
        self.assertEqual(pack.verification_status, "unverified")
        self.assertFalse(pack.ambiguity_candidates)
        self.assertEqual(len(pack.selected_evidence), 2)

    def test_disabled_verifier_keeps_ranked_evidence_without_calling_model(self) -> None:
        verifier = Mock(side_effect=AssertionError("verifier must not run"))

        async def chunks(*_args, **_kwargs):
            return [_chunk()]

        service = EvidenceRetrievalService(
            structured_search=_empty_search,
            chunk_search=chunks,
            verifier=verifier,
        )

        pack = asyncio.run(self._retrieve(service, verify=False))

        self.assertEqual(pack.outcome, "answered")
        self.assertEqual(pack.verification_status, "not_requested")
        self.assertEqual(pack.reason, "deterministic_admission_evidence")
        verifier.assert_not_called()

    def test_explicit_empty_document_scope_denies_all_candidates(self) -> None:
        async def chunks(*_args, **_kwargs):
            return [_chunk()]

        service = EvidenceRetrievalService(
            structured_search=_empty_search,
            chunk_search=chunks,
        )

        pack = asyncio.run(service.retrieve(
            db=object(),
            original_query="查询用户列表",
            resolved_query="查询用户列表",
            kb_ids=[uuid.UUID(KB_ID)],
            method="hybrid",
            top_k=5,
            verify=False,
            trace_id="trace-empty-scope",
            evidence_scope_filter={"doc_ids": []},
        ))

        self.assertEqual(pack.outcome, "no_hit")
        self.assertFalse(pack.candidates)

    def test_complete_verification_promotes_direct_evidence(self) -> None:
        candidate = _record()

        async def records(*_args, **_kwargs):
            return [candidate]

        async def verifier(*_args, **_kwargs):
            return SimpleNamespace(
                succeeded=True,
                results=[_complete(candidate)],
                coverage_status="complete",
                error=None,
            )

        service = EvidenceRetrievalService(
            structured_search=records,
            chunk_search=_empty_search,
            verifier=verifier,
        )

        pack = asyncio.run(self._retrieve(service, verify=True))

        self.assertEqual(pack.outcome, "answered")
        self.assertEqual(pack.verification_status, "verified")
        self.assertEqual(pack.reason, "verified_evidence")

    def test_explicit_semantic_ambiguity_requests_clarification(self) -> None:
        first = _record()
        second = _record(
            "根据标签获取人员列表",
            "/employees/by-tag",
            record_id="77777777-7777-4777-8777-777777777777",
            chunk_id=CHUNK_2,
            doc_id=DOC_2,
        )

        async def records(*_args, **_kwargs):
            return [first, second]

        async def verifier(*_args, **_kwargs):
            return SimpleNamespace(
                succeeded=True,
                results=[
                    {**first, "rerank_candidate_index": 1},
                    {**second, "rerank_candidate_index": 2},
                ],
                coverage_status="insufficient",
                decision_status="ambiguous",
                ambiguity_candidate_indexes=(1, 2),
                error=None,
            )

        service = EvidenceRetrievalService(
            structured_search=records,
            chunk_search=_empty_search,
            verifier=verifier,
        )

        pack = asyncio.run(self._retrieve(service, verify=True))

        self.assertEqual(pack.outcome, "needs_clarification")
        self.assertEqual(pack.verification_status, "ambiguous")
        self.assertFalse(pack.selected_evidence)
        self.assertEqual(len(pack.ambiguity_candidates), 2)

    def test_second_stage_expands_only_verifier_selected_chunk(self) -> None:
        chunk = _chunk(content="接口目录表格")
        correct = _record()
        verifier_calls: list[list[dict]] = []
        expanded_ids: list[tuple[uuid.UUID, ...]] = []

        async def chunks(*_args, **_kwargs):
            return [chunk]

        async def scoped(_db, _query, chunk_ids, *, top_k):
            expanded_ids.append(tuple(chunk_ids))
            self.assertEqual(top_k, 8)
            return [correct]

        async def verifier(_query, candidates, *_args, **_kwargs):
            verifier_calls.append(candidates)
            if len(verifier_calls) == 1:
                return SimpleNamespace(
                    succeeded=True,
                    results=[{
                        **chunk,
                        "rerank_candidate_index": 1,
                        "joint_rerank_status": "verified",
                        "evidence_role": "related",
                        "contribution_role": "background",
                        "answer_support": "invalid-provider-score",
                        "topic_relevance": {"invalid": True},
                    }],
                    coverage_status="partial",
                    selected_candidate_indexes=(1,),
                    error=None,
                )
            return SimpleNamespace(
                succeeded=True,
                results=[_complete(correct)],
                coverage_status="complete",
                error=None,
            )

        service = EvidenceRetrievalService(
            structured_search=_empty_search,
            chunk_search=chunks,
            scoped_record_search=scoped,
            verifier=verifier,
        )

        pack = asyncio.run(self._retrieve(service, verify=True))

        self.assertEqual(expanded_ids, [(uuid.UUID(CHUNK_1),)])
        self.assertEqual(len(verifier_calls), 2)
        self.assertEqual(pack.verification_status, "verified")
        self.assertEqual(pack.selected_evidence[0]["record_id"], correct["record_id"])

    @staticmethod
    async def _retrieve(
        service: EvidenceRetrievalService,
        *,
        verify: bool,
    ) -> EvidencePack:
        return await service.retrieve(
            db=object(),
            original_query="我需要查询用户列表是用的哪个接口",
            resolved_query="我需要查询用户列表是用的哪个接口",
            kb_ids=[uuid.UUID(KB_ID)],
            method="hybrid",
            top_k=5,
            verify=verify,
            trace_id="trace-test",
        )


class RetrievalFirstChatAdapterTests(unittest.TestCase):
    @staticmethod
    def _pack(
        *,
        verification_status: str,
        outcome: str = "answered",
        candidate: dict | None = None,
    ) -> EvidencePack:
        candidate = candidate or _record()
        selected = (candidate,) if outcome == "answered" else ()
        ambiguous = (
            (candidate, _record(
                "根据标签获取人员列表",
                "/employees/by-tag",
                record_id="77777777-7777-4777-8777-777777777777",
                chunk_id=CHUNK_2,
                doc_id=DOC_2,
            ))
            if outcome == "needs_clarification"
            else ()
        )
        return EvidencePack(
            original_query="查询用户列表使用哪个接口",
            resolved_query="查询用户列表使用哪个接口",
            retrieval_status="hit",
            verification_status=verification_status,
            outcome=outcome,
            candidates=ambiguous or (candidate,),
            selected_evidence=selected,
            ambiguity_candidates=ambiguous,
            reason=(
                "multiple_semantic_evidence_sets"
                if outcome == "needs_clarification"
                else "test"
            ),
            channel_reports=(
                RetrievalChannelReport(
                    name="document_chunks",
                    status="succeeded",
                    candidate_count=1,
                ),
            ),
        )

    @staticmethod
    async def _events(pack: EvidencePack, *, rerank: bool = True) -> list[dict]:
        service = SimpleNamespace(retrieve=Mock())

        async def retrieve(**_kwargs):
            return pack

        service.retrieve = retrieve
        with patch(
            "core.retrieval_first_runner.EvidenceRetrievalService",
            return_value=service,
        ):
            return [
                json.loads(chunk.removeprefix("data: ").strip())
                async for chunk in run_retrieval_first_stream(
                    question="查询用户列表使用哪个接口",
                    kb_ids=[uuid.UUID(KB_ID)],
                    search_config={"method": "hybrid", "top_k": 5, "rerank": rerank},
                    conversation_id=str(uuid.uuid4()),
                    db=object(),
                )
            ]

    def test_unavailable_verifier_returns_unverified_sources_and_answer(self) -> None:
        pack = self._pack(verification_status="unverified")

        async def synthesis(**_kwargs):
            yield "使用 /mozi/employee/page 接口。"

        async def collect():
            with patch(
                "core.retrieval_first_runner._stream_grounded_synthesis",
                new=synthesis,
            ):
                return await self._events(pack)

        events = asyncio.run(collect())
        search = next(item for item in events if item.get("type") == "search_results")
        answer = "".join(
            item.get("content", "")
            for item in events
            if item.get("type") == "text_delta"
        )

        self.assertEqual(search["evidence_status"], "unverified")
        self.assertEqual(search["outcome_status"], "answered")
        self.assertEqual(search["answer_source_count"], 1)
        self.assertEqual(search["answer_sources"][0]["evidence_role"], "unverified")
        self.assertIsNone(search["error_code"])
        self.assertEqual(answer, "使用 /mozi/employee/page 接口。")

    def test_generation_failure_degrades_to_extractive_answer(self) -> None:
        pack = self._pack(verification_status="unverified")

        async def failed_synthesis(**_kwargs):
            raise TimeoutError("chat model timeout")
            yield "unreachable"

        async def collect():
            with patch(
                "core.retrieval_first_runner._stream_grounded_synthesis",
                new=failed_synthesis,
            ):
                return await self._events(pack)

        events = asyncio.run(collect())
        answer = "".join(
            item.get("content", "")
            for item in events
            if item.get("type") == "text_delta"
        )

        self.assertIn("检索到了以下相关知识库内容", answer)
        self.assertIn("/mozi/employee/page", answer)

    def test_verified_exact_evidence_uses_deterministic_answer(self) -> None:
        candidate = _record(subject="查询用户列表使用哪个接口")
        pack = self._pack(verification_status="verified", candidate=candidate)

        events = asyncio.run(self._events(pack))
        search = next(item for item in events if item.get("type") == "search_results")
        answer = "".join(
            item.get("content", "")
            for item in events
            if item.get("type") == "text_delta"
        )

        self.assertEqual(search["evidence_status"], "hit")
        self.assertEqual(search["verification_status"], "verified")
        self.assertIn("匹配到", answer)

    def test_real_ambiguity_emits_clarification_contract(self) -> None:
        pack = self._pack(
            verification_status="ambiguous",
            outcome="needs_clarification",
        )

        events = asyncio.run(self._events(pack))
        clarification = next(
            item for item in events if item.get("type") == "clarification_state"
        )
        search = next(item for item in events if item.get("type") == "search_results")

        self.assertEqual(clarification["status"], "proposed")
        self.assertEqual(search["evidence_status"], "needs_clarification")
        self.assertEqual(search["answer_source_count"], 0)

    def test_rejected_admission_never_becomes_answer_context(self) -> None:
        noise = _chunk(content="无关系统升级说明")
        noise["vector_score"] = 0.51
        pack = EvidencePack(
            original_query="查询用户列表使用哪个接口",
            resolved_query="查询用户列表使用哪个接口",
            retrieval_status="hit",
            admission_status="rejected",
            verification_status="not_requested",
            outcome="insufficient_evidence",
            candidates=(noise,),
            admission_rejections=(CandidateRejection(
                candidate_id=CHUNK_1,
                doc_id=DOC_1,
                source_kind="document_chunk",
                reason="document_relevance_gate",
            ),),
            reason="no_candidate_met_admission_gate",
        )

        events = asyncio.run(self._events(pack))
        search = next(item for item in events if item.get("type") == "search_results")
        answer = "".join(
            item.get("content", "")
            for item in events
            if item.get("type") == "text_delta"
        )

        self.assertEqual(search["evidence_status"], "insufficient_evidence")
        self.assertEqual(search["admission_status"], "rejected")
        self.assertEqual(search["rejected_candidate_count"], 1)
        self.assertEqual(search["answer_sources"], [])
        self.assertEqual(search["context_evidence_count"], 0)
        self.assertIn("没有候选通过相关性准入", answer)


if __name__ == "__main__":
    unittest.main()
