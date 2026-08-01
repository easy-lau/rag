import asyncio
import json
import time
import unittest
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from core.query_route_compiler import (
    RouteCategoryPolicy,
    RouteCompilerConfig,
    compile_rag_task_contract,
)
from core.query_route_contract import parse_rag_route_decision
from core.rag_v2.pipeline import (
    _bounded_merge_global_candidate_pools,
    _plan_with_contract_requirements,
    run_rag_v2_stream,
)
from core.rag_v2.query_plan import plan_query_locally


def _settings():
    return SimpleNamespace(
        top_k=5,
        chat_model="test-chat",
        temperature=0,
        max_tokens=256,
        llm_request_timeout_seconds=10,
        llm_max_attempts=1,
        llm_retry_base_delay_seconds=0,
        rag_trace_include_content=True,
        rag_v2_retrieval_timeout_seconds=1,
        rag_v2_expansion_timeout_seconds=0.02,
        rag_v2_retrieval_workflow_timeout_seconds=1,
        rag_v2_generation_workflow_timeout_seconds=1,
    )


def _task_contract(
    question: str,
    *,
    requirements=None,
    relation: str = "new",
):
    contextual = relation != "new"
    route = parse_rag_route_decision(
        {
            "schema_version": "rag_route_decision.v1",
            "readiness": "ready",
            "intent_code": "knowledge_qa",
            "relation": relation,
            "evidence_scope": "enterprise_kb",
            "query_resolution": {
                "mode": "contextualize" if contextual else "current",
                "context_turn_keys": ["t1"] if contextual else [],
            },
            "requirements": requirements
            or [{
                "role": "answer",
                "origin": "user_text",
                "description": question,
            }],
            "clarification": {"question": "", "unresolved": []},
            "confidence": 0.98,
            "rationale": "v2 pipeline test",
        },
        allowed_intent_codes=["knowledge_qa"],
        available_turn_keys=(("t1",) if contextual else ()),
    )
    return compile_rag_task_contract(
        route,
        RouteCategoryPolicy(
            code="knowledge_qa",
            name="知识问答",
            action="retrieve",
        ),
        RouteCompilerConfig(),
        question=question,
        selected_kb_count=1,
        available_turn_keys=(("t1",) if contextual else ()),
        source="test",
    )


def _writing_task_contract(question: str):
    route = parse_rag_route_decision(
        {
            "schema_version": "rag_route_decision.v1",
            "readiness": "ready",
            "intent_code": "writing",
            "relation": "new",
            "evidence_scope": "mixed",
            "query_resolution": {"mode": "current", "context_turn_keys": []},
            "requirements": [{
                "role": "answer",
                "origin": "user_text",
                "description": question,
            }],
            "clarification": {"question": "", "unresolved": []},
            "confidence": 0.98,
            "rationale": "knowledge-grounded writing test",
        },
        allowed_intent_codes=["writing"],
    )
    return compile_rag_task_contract(
        route,
        RouteCategoryPolicy(
            code="writing",
            name="知识写作",
            action="writing",
        ),
        RouteCompilerConfig(),
        question=question,
        selected_kb_count=1,
        source="test",
        knowledge_writing=True,
    )
def _candidate(
    *,
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    chunk_index: int,
    content: str,
    filename: str = "制度.md",
    score: float = 0.08,
) -> dict:
    return {
        "id": uuid.uuid4(),
        "kb_id": kb_id,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "content": content,
        "filename": filename,
        "file_type": "markdown",
        "source_url": None,
        "doc_tags": [],
        "metadata": {},
        "score": score,
        "retrieval_score": score,
        # Mirror the production retriever contract.  Individual relevance tests
        # override these raw observations when exercising score thresholds.
        "vector_score": 0.86,
        "vector_rank": 1,
        "keyword_score": None,
        "keyword_rank": None,
        "trigram_score": None,
        "trigram_rank": None,
        "active_channels": ["vector"],
        "candidate_origin": "current_retrieval",
        "candidate_origins": ["current_retrieval"],
    }


def _full_document(
    *,
    kb_id: uuid.UUID,
    doc_id: uuid.UUID,
    contents: list[str],
    filename: str = "制度.md",
) -> list[dict]:
    total_chars = sum(len(content) for content in contents)
    rows = []
    for index, content in enumerate(contents):
        item = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=index,
            content=content,
            filename=filename,
            score=0.0,
        )
        item.update(
            score=None,
            retrieval_score=None,
            vector_score=None,
            vector_rank=None,
            active_channels=[],
            candidate_origin="small_document_full",
            candidate_origins=["small_document_full"],
            full_document_chunk_count=len(contents),
            full_document_char_count=total_chars,
        )
        rows.append(item)
    return rows


def _payloads(chunks: list[str]) -> list[dict]:
    return [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: ")
    ]


class _FakeCompletions:
    def __init__(self, answer="已根据资料回答"):
        self.answer = answer
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=self.answer),
                    finish_reason="stop",
                )],
                usage=None,
            )

        return stream()


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def with_options(self, **_kwargs):
        return self


class _HangingCompletions:
    def __init__(self, *, first_delta: str | None = None):
        self.first_delta = first_delta
        self.calls: list[dict] = []
        self.cancelled = False

    async def create(self, **kwargs):
        self.calls.append(kwargs)

        async def stream():
            if self.first_delta is not None:
                yield SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content=self.first_delta),
                        finish_reason=None,
                    )],
                    usage=None,
                )
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        return stream()


class _HangingClient:
    def __init__(self, *, first_delta: str | None = None):
        self.completions = _HangingCompletions(first_delta=first_delta)
        self.chat = SimpleNamespace(completions=self.completions)

    def with_options(self, **_kwargs):
        return self


class RagV2PipelineTests(unittest.IsolatedAsyncioTestCase):
    def test_global_plan_candidate_budget_reserves_novel_bridge_seed(self) -> None:
        kb_id = uuid.uuid4()
        primary = [
            _candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=index,
                content=f"主查询候选{index}",
            )
            for index in range(24)
        ]
        bridge = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="跨文档桥接候选",
        )

        merged = _bounded_merge_global_candidate_pools(
            primary,
            [[dict(primary[0]), bridge]],
        )

        self.assertEqual(len(merged), 24)
        self.assertIn(str(bridge["id"]), {
            str(item["id"]) for item in merged
        })

    def test_duplicate_global_candidate_merges_all_plan_query_indexes(self) -> None:
        kb_id = uuid.uuid4()
        shared = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="同一片段支持多个明确子问题。",
        )
        primary = dict(shared, expansion_query_indexes=[0])
        supplemental = dict(shared, expansion_query_indexes=[1, 2])

        merged = _bounded_merge_global_candidate_pools(
            [primary],
            [[supplemental]],
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["expansion_query_indexes"], [0, 1, 2])

    def test_single_broad_contract_requirement_does_not_erase_local_multi_part(
        self,
    ) -> None:
        question = "第一项是什么？第二项如何处理？"
        plan = plan_query_locally(question)
        contract = _task_contract(
            question,
            requirements=[{
                "role": "answer",
                "origin": "user_text",
                "description": "回答用户提出的问题",
            }],
        )

        resolved = _plan_with_contract_requirements(plan, contract)

        self.assertEqual(resolved.answer_shape, "multi_part")
        self.assertEqual(
            [(item.id, item.description) for item in resolved.requirements],
            [
                ("r1", "第一项是什么"),
                ("r2", "第二项如何处理"),
            ],
        )
        self.assertEqual(resolved.retrieval_queries, (
            "第一项是什么",
            "第二项如何处理",
        ))
        self.assertIn("local_multi_part_requirements_preserved", resolved.reason)

    def test_local_multi_part_keeps_additional_contract_bridge(self) -> None:
        question = "第一项是什么？第二项如何处理？"
        plan = plan_query_locally(question)
        contract = _task_contract(
            question,
            requirements=[
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "回答用户提出的问题",
                },
                {
                    "role": "bridge",
                    "origin": "semantically_entailed",
                    "description": "确认两项使用同一适用范围",
                },
            ],
        )

        resolved = _plan_with_contract_requirements(plan, contract)

        self.assertEqual(
            [(item.id, item.role, item.importance) for item in resolved.requirements],
            [
                ("r1", "answer", "required"),
                ("r2", "answer", "required"),
                ("r3", "bridge", "helpful"),
            ],
        )
        self.assertEqual(
            resolved.requirements[2].description,
            "确认两项使用同一适用范围",
        )

    def test_coordinated_identity_answers_and_bridge_survive_contract_merge(
        self,
    ) -> None:
        question = "普通员工出差的住宿、交通和餐补标准分别是多少？"
        plan = plan_query_locally(question)
        contract = _task_contract(question)

        resolved = _plan_with_contract_requirements(plan, contract)

        self.assertEqual(resolved.answer_shape, "multi_hop")
        self.assertEqual(
            [item.role for item in resolved.requirements],
            ["answer", "answer", "answer", "bridge"],
        )
        self.assertEqual(
            [item.id for item in resolved.requirements],
            ["r1", "r2", "r3", "r4"],
        )
        self.assertEqual(
            len(resolved.retrieval_queries),
            len(resolved.requirements),
        )
        self.assertTrue(all(
            item.is_required_answer for item in resolved.requirements[:3]
        ))
        self.assertIn(
            "local_multi_part_requirements_preserved",
            resolved.reason,
        )

    def test_authoritative_coordinated_plan_stays_within_requirement_budget(
        self,
    ) -> None:
        question = (
            "普通员工的交通、住宿、餐补、通讯、驻外、夜班和高温补贴"
            "分别是多少？"
        )
        plan = plan_query_locally(question)
        contract = _task_contract(
            question,
            requirements=[
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": question,
                },
                {
                    "role": "bridge",
                    "origin": "semantically_entailed",
                    "description": "确认员工与制度适用范围之间的关系",
                },
            ],
        )

        self.assertEqual(len(plan.requirements), 8)
        resolved = _plan_with_contract_requirements(plan, contract)

        self.assertEqual(len(resolved.requirements), 8)
        self.assertEqual(len(resolved.retrieval_queries), 8)
        self.assertNotIn(
            "确认员工与制度适用范围之间的关系",
            {item.description for item in resolved.requirements},
        )

    def test_single_answer_contract_cannot_erase_local_implicit_bridge(self) -> None:
        question = "合同工住宿标准"
        plan = plan_query_locally(question)
        compiled = _task_contract(question)
        # Simulate an older/model-produced contract that compressed the route
        # to one answer target.  The execution planner remains the final local
        # safety boundary and must restore its deterministic bridge.
        answer_only = replace(
            compiled,
            requirements=(compiled.requirements[0],),
        )

        resolved = _plan_with_contract_requirements(plan, answer_only)

        self.assertEqual(resolved.answer_shape, "multi_hop")
        self.assertEqual(
            [item.role for item in resolved.requirements],
            ["answer", "bridge"],
        )
        self.assertTrue(any(
            "合同工" in query for query in resolved.retrieval_queries[1:]
        ))

    def test_contract_bridge_resolves_local_planning_clarification(self) -> None:
        question = "该值取决于前一项"
        plan = plan_query_locally(question)
        contract = _task_contract(
            question,
            requirements=[
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "查询该值",
                },
                {
                    "role": "bridge",
                    "origin": "semantically_entailed",
                    "description": "确认前一项与该值之间的决定关系",
                },
            ],
        )

        self.assertTrue(plan.needs_clarification)

        resolved = _plan_with_contract_requirements(plan, contract)

        self.assertFalse(resolved.needs_clarification)
        self.assertIsNone(resolved.clarification_question)
        self.assertEqual(resolved.answer_shape, "multi_hop")
        self.assertEqual(
            [item.role for item in resolved.requirements],
            ["answer", "bridge"],
        )
        self.assertIn(
            "task_contract_resolved_planning_clarification",
            resolved.reason,
        )

    def test_contextualized_single_requirement_uses_standalone_query(self) -> None:
        standalone_query = "那住宿呢。普通员工的出差标准是什么"
        plan = plan_query_locally(standalone_query)
        contract = _task_contract("那住宿呢", relation="followup")

        resolved = _plan_with_contract_requirements(plan, contract)

        self.assertEqual(len(resolved.requirements), 1)
        self.assertEqual(
            resolved.requirements[0].description,
            standalone_query,
        )

    async def _run(
        self,
        *,
        question: str,
        kb_id: uuid.UUID,
        initial: list[dict] | Exception,
        full_document: list[dict] | Exception,
        scoped: list[dict] | Exception | None = None,
        requirements=None,
        evidence_scope_filter: dict | None = None,
        blocking_full_document: bool = False,
        vector_channel_failed: bool = False,
        carryover_sources: list[dict] | None = None,
        standalone_query: str | None = None,
        selected_tags: list[str] | None = None,
        initial_sequence: list[list[dict] | Exception] | None = None,
        initial_delay_seconds: float = 0,
        blocking_scoped: bool = False,
        settings_override=None,
        client_override=None,
        task_contract_override=None,
        expected_error: type[BaseException] | None = None,
    ):
        client = client_override or _FakeClient()
        search = AsyncMock()
        if initial_sequence is not None:
            search.side_effect = initial_sequence
        elif isinstance(initial, Exception):
            search.side_effect = initial
        elif initial_delay_seconds > 0:
            async def delayed_search(*_args, **_kwargs):
                await asyncio.sleep(initial_delay_seconds)
                return initial

            search.side_effect = delayed_search
        elif vector_channel_failed:
            async def degraded_search(*_args, **kwargs):
                diagnostics = kwargs.get("diagnostics")
                if isinstance(diagnostics, dict):
                    diagnostics["vector_channel_failed"] = True
                    diagnostics["vector_error_type"] = "TimeoutError"
                return initial

            search.side_effect = degraded_search
        else:
            search.return_value = initial
        fetch_full = AsyncMock()
        if blocking_full_document:
            async def block_forever(*_args, **_kwargs):
                await asyncio.sleep(60)

            fetch_full.side_effect = block_forever
        elif isinstance(full_document, Exception):
            fetch_full.side_effect = full_document
        else:
            fetch_full.return_value = full_document
        scoped_search = AsyncMock()
        if blocking_scoped:
            async def block_scoped(*_args, **_kwargs):
                await asyncio.sleep(60)

            scoped_search.side_effect = block_scoped
        elif isinstance(scoped, Exception):
            scoped_search.side_effect = scoped
        else:
            scoped_search.return_value = scoped or []
        structural_search = AsyncMock(return_value=[])

        trace = Mock()
        with (
            patch(
                "core.rag_v2.pipeline.get_settings",
                return_value=settings_override or _settings(),
            ),
            patch("core.rag_v2.pipeline.hybrid_search", new=search),
            patch(
                "core.rag_v2.pipeline.fetch_small_document_candidates",
                new=fetch_full,
            ),
            patch(
                "core.rag_v2.pipeline.search_within_documents",
                new=scoped_search,
            ),
            patch(
                "core.rag_v2.pipeline.fetch_structural_neighbors",
                new=structural_search,
            ),
            patch("core.rag_v2.pipeline.get_client", return_value=client),
            patch("core.rag_v2.pipeline.trace_event", new=trace),
        ):
            chunks = []

            async def collect_chunks():
                async for chunk in run_rag_v2_stream(
                    question=question,
                    kb_ids=[kb_id],
                    search_config={
                        "top_k": 5,
                        "method": "hybrid",
                        "rerank": True,
                        "tags": selected_tags or [],
                    },
                    conversation_id="v2-test-conversation",
                    db=SimpleNamespace(),
                    intent={"intent_code": "knowledge_qa"},
                    task_contract=(
                        task_contract_override
                        or _task_contract(
                            question,
                            requirements=requirements,
                        )
                    ),
                    evidence_scope_filter=evidence_scope_filter,
                    carryover_sources=carryover_sources,
                    is_followup=bool(carryover_sources),
                    standalone_query=standalone_query,
                ):
                    chunks.append(chunk)

            if expected_error is None:
                await collect_chunks()
            else:
                with self.assertRaises(expected_error):
                    await collect_chunks()
        self._last_trace = trace
        return _payloads(chunks), client, search, fetch_full, scoped_search

    async def test_cross_domain_business_matrix_reaches_grounded_generation(
        self,
    ) -> None:
        cases = (
            (
                "ordinary_employee_transport",
                "普通员工的交通标准是什么",
                (
                    "职级分类：普通员工对应D级。",
                    "交通标准：D级乘飞机经济舱、高铁二等座。",
                ),
                ("普通员工对应D级", "高铁二等座"),
            ),
            (
                "contractor_lodging",
                "合同工住宿标准是多少",
                (
                    "用工分类：合同工属于L2类。",
                    "住宿标准：L2类不超过300元/天。",
                ),
                ("合同工属于L2类", "300元/天"),
            ),
            (
                "reimbursement_deadline_receipts",
                "报销提交时限是多久？需要提供哪些凭证？",
                (
                    "费用报销时限：出差结束后5个工作日内提交。",
                    "报销凭证：必须提供正规发票、行程单及住宿发票。",
                ),
                ("5个工作日", "正规发票"),
            ),
            (
                "leave_approval_process",
                "员工请假审批流程是什么",
                (
                    "请假审批流程：员工提交申请，直属主管审批，三天以上再由部门负责人审批。",
                ),
                ("直属主管审批", "部门负责人审批"),
            ),
            (
                "purchase_approval_limit",
                "采购申请单笔审批额度是多少",
                (
                    "采购审批制度：单笔采购申请金额不超过5000元的，由部门经理审批。",
                ),
                ("5000元", "部门经理审批"),
            ),
            (
                "login_username_enumeration",
                "如何配置登录用户名枚举防护",
                (
                    "登录用户名枚举防护配置：将 error_reply_same 设置为 true，"
                    "使账号不存在与密码错误返回相同提示。",
                ),
                ("error_reply_same", "相同提示"),
            ),
        )

        for name, question, support_contents, expected_markers in cases:
            with self.subTest(scenario=name):
                kb_id = uuid.uuid4()
                candidates = [
                    _candidate(
                        kb_id=kb_id,
                        doc_id=uuid.uuid4(),
                        chunk_index=0,
                        filename=f"{name}-{index}.md",
                        content=content,
                    )
                    for index, content in enumerate(support_contents, start=1)
                ]
                distractor = _candidate(
                    kb_id=kb_id,
                    doc_id=uuid.uuid4(),
                    chunk_index=0,
                    filename=f"{name}-unrelated.md",
                    content="禁止纳入：访客停车区域和固定资产盘点说明。",
                )
                distractor.update(vector_score=0.05, vector_rank=20)

                payloads, client, *_ = await self._run(
                    question=question,
                    kb_id=kb_id,
                    initial=[*candidates, distractor],
                    full_document=[],
                )

                result = next(
                    item for item in payloads if item["type"] == "search_results"
                )
                self.assertIn(
                    result["evidence_status"],
                    {"hit", "partial", "unverified"},
                )
                self.assertEqual(result["missing_requirement_ids"], [])
                self.assertGreaterEqual(len(result["answer_sources"]), 1)
                self.assertEqual(len(client.completions.calls), 1)
                prompt = "\n".join(
                    message["content"]
                    for message in client.completions.calls[0]["messages"]
                )
                for marker in expected_markers:
                    self.assertIn(marker, prompt)
                self.assertNotIn("禁止纳入", prompt)
                self.assertEqual(
                    "".join(
                        item.get("content", "")
                        for item in payloads
                        if item.get("type") == "text_delta"
                    ),
                    "已根据资料回答",
                )
                self.assertEqual(payloads[-1]["type"], "done")

    async def test_knowledge_grounded_writing_uses_v2_retrieval_and_writing_prompt(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        question = "根据制度起草一段报销通知"
        initial = [_candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            content="报销通知要求：费用发生后5个工作日内提交正规发票。",
        )]

        payloads, client, *_ = await self._run(
            question=question,
            kb_id=kb_id,
            initial=initial,
            full_document=[],
            task_contract_override=_writing_task_contract(question),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertTrue(result["retrieval_executed"])
        self.assertEqual(len(client.completions.calls), 1)
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("用户要求执行写作任务", system_prompt)
        self.assertIn("不得添加证据未支持的企业事实", system_prompt)

    async def test_unresolved_query_plan_clarifies_before_retrieval_or_generation(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        question = "该值取决于前一项"

        payloads, client, search, fetch_full, scoped = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        clarification = next(
            item for item in payloads if item["type"] == "evidence_clarification"
        )
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertFalse(result["retrieval_executed"])
        self.assertEqual(result["results"], [])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(clarification["choices"], [])
        self.assertIn("对应关系", clarification["question"])
        self.assertIn("标准或数值", clarification["question"])
        search.assert_not_awaited()
        fetch_full.assert_not_awaited()
        scoped.assert_not_awaited()
        self.assertEqual(client.completions.calls, [])
        trace_events = [
            call.args[0]
            for call in self._last_trace.call_args_list
            if call.args
        ]
        self.assertNotIn("retrieval.plan", trace_events)
        self.assertNotIn("generation.context", trace_events)
        self.assertIn("generation.skipped", trace_events)

    async def test_scope_selection_hard_limits_same_kb_documents(self) -> None:
        kb_id = uuid.uuid4()
        selected_doc_id = uuid.uuid4()
        unrelated_doc_id = uuid.uuid4()
        selected = _candidate(
            kb_id=kb_id,
            doc_id=selected_doc_id,
            chunk_index=0,
            content="所选范围内的明确配置。",
            filename="目标版本.docx",
        )
        unrelated = _candidate(
            kb_id=kb_id,
            doc_id=unrelated_doc_id,
            chunk_index=0,
            content="同一知识库中的其它版本配置。",
            filename="其它版本.docx",
        )
        scope_filter = {
            "mode": "single",
            "kb_ids": [str(kb_id)],
            "doc_ids": [str(selected_doc_id)],
            "choices": [{
                "key": "c1",
                "label": "目标版本 —《目标版本.docx》",
                "products": ["目标产品"],
                "canonical_products": ["目标产品"],
                "versions": ["1.0"],
                "projects": [],
                "filenames": ["目标版本.docx"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(selected_doc_id)],
                "anchor_doc_ids": [str(selected_doc_id)],
                "companion_doc_ids": [],
            }],
        }

        payloads, client, search, _fetch, scoped_search = await self._run(
            question="配置是什么",
            kb_id=kb_id,
            initial=[],
            # Even a faulty expansion adapter returning another same-KB
            # document must not escape the selected document allow-list.
            full_document=[selected, unrelated],
            scoped=[selected, unrelated],
            evidence_scope_filter=scope_filter,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(selected_doc_id)},
        )
        self.assertTrue(result["evidence_scope_anchor_hit"])
        self.assertEqual(
            result["evidence_scope_anchor_doc_ids"],
            [str(selected_doc_id)],
        )
        self.assertNotIn("其它版本配置", "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        ))
        search.assert_not_awaited()
        self.assertGreaterEqual(scoped_search.await_count, 1)

    async def test_scope_selection_without_context_anchor_stays_incomplete(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        selected_doc_id = uuid.uuid4()
        scope_filter = {
            "mode": "single",
            "kb_ids": [str(kb_id)],
            "doc_ids": [str(selected_doc_id)],
            "choices": [{
                "key": "c1",
                "label": "目标版本 —《目标版本.docx》",
                "products": ["目标产品"],
                "canonical_products": ["目标产品"],
                "versions": ["1.0"],
                "projects": [],
                "filenames": ["目标版本.docx"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(selected_doc_id)],
                "anchor_doc_ids": [str(selected_doc_id)],
                "companion_doc_ids": [],
            }],
        }

        payloads, _client, search, _fetch, scoped_search = await self._run(
            question="配置是什么",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            scoped=[],
            evidence_scope_filter=scope_filter,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["answer_sources"], [])
        self.assertFalse(result["evidence_scope_anchor_hit"])
        self.assertEqual(result["evidence_scope_anchor_doc_ids"], [])
        search.assert_not_awaited()
        self.assertGreaterEqual(scoped_search.await_count, 1)

    async def test_followup_uses_fresh_retrieval_inside_carryover_document(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        carryover = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content="上一轮只说明普通员工对应D级。",
            filename="公司出差管理标准.docx",
            score=0.99,
        )
        carryover["metadata"] = {"retrieval_score": 0.99}
        fresh = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=4,
            content="D级住宿上限：一线城市450元/天。",
            filename="公司出差管理标准.docx",
            score=0.07,
        )

        payloads, client, search, _fetch, scoped_search = await self._run(
            question="那住宿呢",
            standalone_query="那住宿呢。普通员工的出差标准是什么",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            scoped=[fresh],
            carryover_sources=[carryover],
            task_contract_override=_task_contract(
                "那住宿呢",
                relation="followup",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertGreater(result["answer_source_count"], 0)
        self.assertTrue(result["carryover_anchor_succeeded"])
        self.assertFalse(result["carryover_seed_used"])
        self.assertEqual(result["carryover_candidate_count"], 1)
        self.assertEqual(len(client.completions.calls), 1)
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("住宿上限", prompt)
        self.assertNotIn("上一轮只说明", prompt)
        search.assert_awaited_once()
        anchor_call = next(
            call
            for call in scoped_search.await_args_list
            if call.kwargs.get("surface") == "chat_v2_carryover"
        )
        self.assertEqual(anchor_call.kwargs["queries"], [
            "那住宿呢。普通员工的出差标准是什么"
        ])
        self.assertEqual(anchor_call.kwargs["doc_ids"], [doc_id])

    async def test_followup_carryover_anchor_failure_rejects_previous_seed(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        carryover = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=4,
            content="D级住宿上限：一线城市450元/天。",
            filename="公司出差管理标准.docx",
            score=0.99,
        )
        carryover["metadata"] = {"retrieval_score": 0.99}
        foreign = _candidate(
            kb_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="未授权知识库中的住宿标准。",
            score=1.0,
        )

        payloads, client, _search, _fetch, _scoped = await self._run(
            question="那住宿呢",
            standalone_query="那住宿呢。普通员工的出差标准是什么",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            scoped=TimeoutError("carryover scoped search timeout"),
            carryover_sources=[carryover, foreign],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "error")
        self.assertEqual(result["evidence_availability"], "unavailable")
        self.assertEqual(result["answer_source_count"], 0)
        self.assertEqual(result["answer_sources"], [])
        self.assertFalse(result["carryover_anchor_succeeded"])
        self.assertFalse(result["carryover_seed_used"])
        self.assertEqual(result["carryover_candidate_count"], 0)
        self.assertEqual(client.completions.calls, [])
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertIn("服务暂时不可用", answer)
        self.assertNotIn("一线城市450元", answer)
        self.assertNotIn("未授权知识库", answer)

    async def test_unmatched_carryover_document_cannot_reenter_via_expansion(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        old_doc_id = uuid.uuid4()
        current_doc_id = uuid.uuid4()
        carryover = _candidate(
            kb_id=kb_id,
            doc_id=old_doc_id,
            chunk_index=0,
            content="上一轮旧文档只说明旧指标。",
            filename="旧制度.docx",
            score=0.99,
        )
        current = _candidate(
            kb_id=kb_id,
            doc_id=current_doc_id,
            chunk_index=0,
            content="目标指标当前值为20。",
            filename="当前制度.docx",
        )
        current.update(vector_score=0.91, vector_rank=1)
        old_full = _full_document(
            kb_id=kb_id,
            doc_id=old_doc_id,
            contents=["目标指标旧值为10，已废止。"],
            filename="旧制度.docx",
        )
        current_full = _full_document(
            kb_id=kb_id,
            doc_id=current_doc_id,
            contents=["目标指标当前值为20。"],
            filename="当前制度.docx",
        )

        payloads, client, _search, fetch_full, _scoped = await self._run(
            question="目标指标是多少",
            standalone_query="目标指标是多少",
            kb_id=kb_id,
            initial=[current],
            full_document=[*old_full, *current_full],
            scoped=[],
            carryover_sources=[carryover],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(
            {item["doc_id"] for item in result["results"]},
            {str(current_doc_id)},
        )
        self.assertEqual(
            fetch_full.await_args.kwargs["doc_ids"],
            [current_doc_id],
        )
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("当前值为20", prompt)
        self.assertNotIn("旧值为10", prompt)
        self.assertNotIn("上一轮旧文档", prompt)

    async def test_rejected_fresh_carryover_hit_cannot_reenter_via_expansion(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        old_doc_id = uuid.uuid4()
        current_doc_id = uuid.uuid4()
        carryover = _candidate(
            kb_id=kb_id,
            doc_id=old_doc_id,
            chunk_index=0,
            content="上一轮旧标准。",
            filename="旧制度.docx",
            score=0.99,
        )
        low_fresh = _candidate(
            kb_id=kb_id,
            doc_id=old_doc_id,
            chunk_index=1,
            content="旧制度的弱相关说明。",
            filename="旧制度.docx",
        )
        low_fresh.update(vector_score=0.55, vector_rank=2)
        current = _candidate(
            kb_id=kb_id,
            doc_id=current_doc_id,
            chunk_index=0,
            content="目标指标当前值为20。",
            filename="当前制度.docx",
        )
        current.update(vector_score=0.92, vector_rank=1)
        old_full = _full_document(
            kb_id=kb_id,
            doc_id=old_doc_id,
            contents=["目标指标旧值为10，已废止。"],
            filename="旧制度.docx",
        )
        current_full = _full_document(
            kb_id=kb_id,
            doc_id=current_doc_id,
            contents=["目标指标当前值为20。"],
            filename="当前制度.docx",
        )

        payloads, client, _search, fetch_full, _scoped = await self._run(
            question="目标指标是多少",
            standalone_query="目标指标是多少",
            kb_id=kb_id,
            initial=[current],
            full_document=[*old_full, *current_full],
            scoped=[low_fresh],
            carryover_sources=[carryover],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertFalse(result["carryover_anchor_succeeded"])
        self.assertEqual(
            {item["doc_id"] for item in result["results"]},
            {str(current_doc_id)},
        )
        self.assertEqual(
            fetch_full.await_args.kwargs["doc_ids"],
            [current_doc_id],
        )
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("旧值为10", prompt)
        self.assertNotIn("上一轮旧标准", prompt)

    async def test_selected_tags_soft_boost_v2_without_admitting_noise(self) -> None:
        kb_id = uuid.uuid4()
        tagged_doc_id = uuid.uuid4()
        other_doc_id = uuid.uuid4()
        tagged = _candidate(
            kb_id=kb_id,
            doc_id=tagged_doc_id,
            chunk_index=0,
            content="标签命中的制度事实。",
            filename="标签制度.md",
            score=0.08,
        )
        tagged.update(doc_tags=["重点"], keyword_score=0.1, vector_score=0.9)
        other = _candidate(
            kb_id=kb_id,
            doc_id=other_doc_id,
            chunk_index=0,
            content="未命中标签的同主题事实。",
            filename="普通制度.md",
            score=0.1,
        )
        other.update(keyword_score=0.1, vector_score=0.9)

        payloads, _client, _search, _fetch, _scoped = await self._run(
            question="制度事实是什么",
            kb_id=kb_id,
            initial=[other, tagged],
            full_document=[],
            selected_tags=["重点"],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["answer_sources"][0]["doc_id"], str(tagged_doc_id))
        # The tag only changes ordering; both candidates still pass the normal
        # lexical/vector gate and remain in the bounded result set.
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(tagged_doc_id), str(other_doc_id)},
        )

    async def test_followup_low_score_fresh_anchor_does_not_reuse_previous_seed(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        carryover = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=4,
            content="D级住宿上限：一线城市450元/天。",
            filename="公司出差管理标准.docx",
            score=0.98,
        )
        low_score_fresh = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            content="公司出差管理标准总则。",
            filename="公司出差管理标准.docx",
            score=0.01,
        )
        low_score_fresh.update(
            vector_score=0.55,
            vector_rank=1,
            active_channels=["vector"],
        )

        payloads, client, *_ = await self._run(
            question="那住宿呢",
            standalone_query="那住宿呢。普通员工的出差标准是什么",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            scoped=[low_score_fresh],
            carryover_sources=[carryover],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "no_hit")
        self.assertFalse(result["carryover_seed_used"])
        self.assertFalse(result["carryover_anchor_succeeded"])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(client.completions.calls, [])
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertIn("未找到", answer)
        self.assertNotIn("一线城市450元", answer)

    async def test_document_relevance_gate_excludes_lower_vector_noise(self) -> None:
        kb_id = uuid.uuid4()
        target_doc_id = uuid.uuid4()
        noise_doc_id = uuid.uuid4()
        target = _candidate(
            kb_id=kb_id,
            doc_id=target_doc_id,
            chunk_index=0,
            content="目标制度中的明确标准。",
            filename="目标制度.docx",
        )
        target.update(vector_score=0.88, vector_rank=1, active_channels=["vector"])
        noise = _candidate(
            kb_id=kb_id,
            doc_id=noise_doc_id,
            chunk_index=0,
            content="无关制度内容。",
            filename="无关制度.docx",
        )
        noise.update(vector_score=0.80, vector_rank=2, active_channels=["vector"])

        payloads, client, *_ = await self._run(
            question="查询目标制度标准",
            kb_id=kb_id,
            initial=[target, noise],
            # Simulate an adapter returning both documents during expansion;
            # the admitted-document boundary must still hold.
            full_document=[target, noise],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(target_doc_id)},
        )
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("无关制度内容", prompt)

    async def test_full_document_tables_cannot_evict_initial_retrieval_seed(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        relevant_content = "目标事实的值为唯一答案。"
        full_document = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=[
                relevant_content,
                *[
                    f"| 扩展字段{i} | 扩展值{i} |\n| --- | --- |"
                    for i in range(16)
                ],
            ],
        )
        initial = dict(full_document[0])
        initial.update(
            score=0.09,
            retrieval_score=0.09,
            vector_score=0.9,
            vector_rank=1,
            active_channels=["vector"],
            candidate_origin="current_retrieval",
            candidate_origins=["current_retrieval"],
        )

        payloads, client, *_ = await self._run(
            question="目标事实的值是多少",
            kb_id=kb_id,
            initial=[initial],
            full_document=full_document,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertIn(
            str(initial["id"]),
            {item["id"] for item in result["answer_sources"]},
        )
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn(relevant_content, prompt)

    async def test_final_context_budget_reconciles_serialized_sources(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        initial = [
            _candidate(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_index=index,
                content=f"目标制度条款{index}：" + ("甲" * 970),
                filename="目标制度.docx",
            )
            for index in range(16)
        ]

        payloads, client, *_ = await self._run(
            question="目标制度的完整内容是什么",
            kb_id=kb_id,
            initial=initial,
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        generation_context = next(
            call
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "generation.context"
        )
        serialized_context = generation_context.kwargs["context"]
        context_source_ids = {
            str(item["chunk_id"])
            for item in generation_context.kwargs["context_sources"]
        }
        all_context_sources = generation_context.kwargs[
            "all_context_sources"
        ]
        answer_source_ids = {
            str(item["id"])
            for item in result["answer_sources"]
        }

        self.assertLessEqual(len(serialized_context), 16_000)
        self.assertGreater(
            generation_context.kwargs["context_budget_dropped_count"],
            0,
        )
        self.assertLess(result["answer_source_count"], len(initial))
        self.assertEqual(answer_source_ids, context_source_ids)
        self.assertEqual(
            result["answer_source_count"],
            len(generation_context.kwargs["context_sources"]),
        )
        self.assertTrue(all(
            source["supports_requirement_ids"] == ["r1"]
            for source in result["answer_sources"]
        ))
        # An oversized source is omitted completely: a truncated prefix may
        # not contain the clause that established requirement support, and a
        # hidden background prefix must never influence generation.
        self.assertNotIn("角色：background", serialized_context)
        self.assertEqual(len(all_context_sources), len(context_source_ids))
        self.assertTrue(all(
            source["evidence_contribution_role"]
            in {"direct", "bridge", "complement"}
            and source["supports_requirement_ids"]
            and source["included_in_answer_sources"]
            for source in all_context_sources
        ))
        self.assertIn(
            "generation_context_budget_limited",
            result["evidence_state"]["reasons"],
        )
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        for source in result["answer_sources"]:
            self.assertIn(f"片段：{source['chunk_index']}；", prompt)

    async def test_low_vector_nearest_neighbors_are_normal_no_hit(self) -> None:
        kb_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="最近邻但主题无关。",
        )
        candidate.update(
            vector_score=0.55,
            vector_rank=1,
            active_channels=["vector"],
        )

        payloads, client, _search, fetch_full, _scoped = await self._run(
            question="完全不同的主题",
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "no_hit")
        self.assertEqual(result["answer_source_count"], 0)
        self.assertEqual(client.completions.calls, [])
        self.assertIn(
            "知识库中未找到与“完全不同的主题”相关的内容",
            "".join(
                item.get("content", "")
                for item in payloads
                if item.get("type") == "text_delta"
            ),
        )
        generation_context = next(
            call
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "generation.context"
        )
        self.assertTrue(generation_context.kwargs["deterministic"])
        self.assertIsNone(generation_context.kwargs["model"])
        self.assertEqual(generation_context.kwargs["context"], "")
        fetch_full.assert_not_awaited()

    async def test_unknown_policy_subject_with_generic_lexical_hits_is_no_hit(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        travel = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="公司出差管理标准.docx",
            content="员工出差交通、住宿和餐饮补贴标准。",
        )
        travel.update(
            vector_score=None,
            vector_rank=None,
            keyword_score=0.02,
            keyword_rank=1,
            active_channels=["keyword"],
        )
        leave = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="员工请假管理办法.docx",
            content="员工请假审批、休假天数和销假要求。",
        )
        leave.update(
            vector_score=None,
            vector_rank=None,
            trigram_score=0.18,
            trigram_rank=1,
            active_channels=["trigram"],
        )

        payloads, client, search, fetch_full, scoped = await self._run(
            question="不存在的火星基地量子补贴标准是什么",
            kb_id=kb_id,
            initial=[travel, leave],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "no_hit")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(client.completions.calls, [])
        self.assertFalse(any(
            item["type"] == "evidence_clarification" for item in payloads
        ))
        # The unknown owner is not a valid identity/classification qualifier,
        # so the planner performs only the original retrieval instead of a
        # meaningless synthetic bridge lookup.
        self.assertEqual(search.await_count, 1)
        fetch_full.assert_not_awaited()
        scoped.assert_not_awaited()

    async def test_no_hit_fallback_displays_raw_followup_not_standalone_query(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="与住宿完全无关的内容。",
        )
        candidate.update(
            vector_score=0.2,
            vector_rank=1,
            active_channels=["vector"],
        )

        payloads, client, *_ = await self._run(
            question="那住宿呢",
            standalone_query="那住宿呢。普通员工的出差标准是什么",
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
        )

        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertIn("那住宿呢", answer)
        self.assertNotIn("普通员工的出差标准是什么", answer)
        self.assertEqual(client.completions.calls, [])

    async def test_explicit_year_scope_excludes_other_year_before_context(self) -> None:
        kb_id = uuid.uuid4()
        rows = []
        for version in ("2024", "2025"):
            row = _candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=0,
                filename=f"差旅制度{version}版.docx",
                content=f"差旅制度{version}版：餐补标准。",
            )
            row.update(
                metadata={"version": version},
                vector_score=0.86,
                vector_rank=1 if version == "2025" else 2,
                active_channels=["trigram"],
            )
            rows.append(row)

        payloads, client, *_ = await self._run(
            question="查询差旅制度2025版的餐补标准",
            kb_id=kb_id,
            initial=rows,
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["answer_source_count"], 1)
        self.assertIn("2025", result["answer_sources"][0]["content"])
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("2024版", prompt)

    async def test_explicit_two_version_comparison_keeps_both_and_drops_third(self) -> None:
        kb_id = uuid.uuid4()
        rows = []
        for version in ("6", "7", "8"):
            row = _candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=0,
                filename=f"CloudPivot {version} 安全配置",
                content=f"所属产品：CloudPivot；产品版本：{version}。安全配置。",
            )
            row.update(
                metadata={"product": "CloudPivot", "version": version},
                vector_score=0.85,
                vector_rank=int(version) - 5,
                active_channels=["trigram"],
            )
            rows.append(row)

        payloads, client, *_ = await self._run(
            question="比较 CloudPivot 6 和 CloudPivot 7 的安全配置",
            kb_id=kb_id,
            initial=rows,
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        versions = {
            str(source["content"]).split("产品版本：", 1)[1].split("。", 1)[0]
            for source in result["answer_sources"]
        }
        self.assertEqual(versions, {"6", "7"})
        self.assertEqual(len(client.completions.calls), 1)

    async def test_multiple_required_items_report_missing_coverage(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        full = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=["第一项的值是A。"],
        )

        payloads, client, *_ = await self._run(
            question="请分别回答第一项和第二项",
            kb_id=kb_id,
            initial=[{
                **full[0],
                "vector_score": 0.86,
                "vector_rank": 1,
                "active_channels": ["vector"],
            }],
            full_document=full,
            requirements=[
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "查询第一项的值",
                },
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "查询第二项的值",
                },
            ],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["coverage_status"], "partial")
        self.assertEqual(result["covered_requirement_ids"], ["r1"])
        self.assertEqual(result["missing_requirement_ids"], ["r2"])
        self.assertEqual(result["missing_requirement_count"], 1)
        self.assertEqual(
            result["answer_sources"][0]["supports_requirement_ids"],
            ["r1"],
        )

    async def test_expansion_deadline_retains_first_pass_evidence(self) -> None:
        kb_id = uuid.uuid4()
        initial = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=4,
            content="首轮已经召回的明确条款。",
        )

        payloads, _client, *_ = await self._run(
            question="请说明明确条款",
            kb_id=kb_id,
            initial=[initial],
            full_document=[],
            blocking_full_document=True,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(result["answer_source_count"], 1)
        self.assertIn(
            "expansion_deadline_or_adapter_failed",
            result["evidence_state"]["reasons"],
        )

    async def test_vector_channel_failure_with_lexical_hit_is_degraded_not_empty(self) -> None:
        kb_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="词面通道命中的明确条款。",
        )
        candidate.update(
            trigram_rank=1,
            trigram_score=0.12,
            active_channels=["trigram"],
        )

        payloads, _client, *_ = await self._run(
            question="明确条款",
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
            vector_channel_failed=True,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(result["answer_source_count"], 1)
        self.assertIn("retrieval_degraded", result["evidence_state"]["reasons"])

    async def test_vector_channel_failure_without_lexical_hit_is_unavailable(self) -> None:
        kb_id = uuid.uuid4()
        payloads, client, *_ = await self._run(
            question="仅语义可召回的问题",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            vector_channel_failed=True,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "error")
        self.assertEqual(result["evidence_availability"], "unavailable")
        self.assertEqual(client.completions.calls, [])

    async def test_overview_loads_complete_small_document_in_source_order(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        contents = [
            "公司制度",
            "一、总则：规范管理。",
            "二、分类：普通岗位对应D级。",
            "三、交通：D级乘坐经济舱和高铁二等座。",
            "四、住宿：D级上限450元/天。",
            "五、餐饮：D级补贴100元/天。",
        ]
        full = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=contents,
            filename="公司管理标准.docx",
        )
        initial = [dict(full[0]), dict(full[2])]
        initial[0].update(
            score=0.09,
            retrieval_score=0.09,
            vector_score=0.86,
            vector_rank=1,
            active_channels=["vector"],
        )
        initial[1].update(
            score=0.08,
            retrieval_score=0.08,
            vector_score=0.84,
            vector_rank=2,
            active_channels=["vector"],
        )

        payloads, client, _search, _fetch, scoped = await self._run(
            question="普通岗位的管理标准是什么",
            kb_id=kb_id,
            initial=initial,
            full_document=full,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["evidence_availability"], "ok")
        self.assertEqual(result["evidence_confidence"], "retrieved")
        self.assertEqual(result["evidence_completeness"], "complete")
        self.assertEqual(
            [item["chunk_index"] for item in result["answer_sources"]],
            list(range(len(contents))),
        )
        self.assertEqual(len(client.completions.calls), 1)
        context = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        for expected in ("经济舱", "450元", "100元"):
            self.assertIn(expected, context)
        scoped.assert_not_awaited()

    async def test_fact_with_bridge_requirement_keeps_cross_chunk_evidence(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        full = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=[
                "职级分类：普通岗位对应D级。",
                "餐饮补贴：D级为100元/天。",
            ],
        )
        initial = [dict(full[1])]
        initial[0].update(
            score=0.09,
            retrieval_score=0.09,
            vector_score=0.86,
            vector_rank=1,
            active_channels=["vector"],
        )

        payloads, client, *_ = await self._run(
            question="普通岗位的餐饮补贴是多少",
            kb_id=kb_id,
            initial=initial,
            full_document=full,
            requirements=[
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "查询普通岗位的餐饮补贴金额",
                },
                {
                    "role": "bridge",
                    "origin": "semantically_entailed",
                    "description": "确认普通岗位对应的职级",
                },
            ],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(len(result["answer_sources"]), 2)
        by_role = {
            source["evidence_contribution_role"]: source
            for source in result["answer_sources"]
        }
        self.assertEqual(by_role["bridge"]["supports_requirement_ids"], ["r2"])
        self.assertEqual(
            by_role["complement"]["supports_requirement_ids"],
            ["r1"],
        )
        self.assertEqual(result["covered_requirement_ids"], ["r2", "r1"])
        self.assertEqual(result["missing_requirement_ids"], [])
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("普通岗位对应D级", prompt)
        self.assertIn("D级为100元/天", prompt)
        self.assertIn('"answer_shape": "multi_hop"', prompt)

    async def test_multi_hop_global_plan_queries_join_different_documents(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        answer_doc_id = uuid.uuid4()
        bridge_doc_id = uuid.uuid4()
        weak_doc_id = uuid.uuid4()
        answer = _candidate(
            kb_id=kb_id,
            doc_id=answer_doc_id,
            chunk_index=0,
            filename="差旅补贴标准.md",
            content="D级餐补标准为100元/天。",
        )
        bridge = _candidate(
            kb_id=kb_id,
            doc_id=bridge_doc_id,
            chunk_index=0,
            filename="员工职级分类.md",
            content="普通员工对应D级。",
        )
        unauthorized = _candidate(
            kb_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="未授权分类.md",
            content="普通员工对应A级。",
        )
        weak = _candidate(
            kb_id=kb_id,
            doc_id=weak_doc_id,
            chunk_index=0,
            filename="无关制度.md",
            content="访客停车区域说明。",
        )
        weak.update(vector_score=0.42, vector_rank=3)

        payloads, client, search, *_ = await self._run(
            question="普通员工的餐补标准是多少",
            kb_id=kb_id,
            initial=[],
            initial_sequence=[
                [answer],
                [bridge, unauthorized, weak],
            ],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["evidence_completeness"], "complete")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(answer_doc_id), str(bridge_doc_id)},
        )
        by_doc_id = {
            item["doc_id"]: item for item in result["answer_sources"]
        }
        bridge_source = by_doc_id[str(bridge_doc_id)]
        self.assertEqual(
            bridge_source["metadata"]["expansion_query_indexes"],
            [1],
        )
        self.assertIn(
            "global_plan_query_supplement",
            bridge_source["candidate_origins"],
        )
        self.assertEqual(search.await_count, 2)
        self.assertEqual(search.await_args_list[0].args[1], "普通员工的餐补标准是多少")
        self.assertIn("普通员工", search.await_args_list[1].args[1])
        self.assertEqual(
            search.await_args_list[1].kwargs["surface"],
            "chat_v2_plan_query",
        )
        self.assertIs(
            search.await_args_list[0].args[0],
            search.await_args_list[1].args[0],
        )
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("D级餐补标准为100元", prompt)
        self.assertIn("普通员工对应D级", prompt)
        self.assertNotIn("普通员工对应A级", prompt)
        self.assertNotIn("访客停车", prompt)

    async def test_global_plan_query_timeout_keeps_primary_evidence_degraded(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        answer_doc_id = uuid.uuid4()
        answer = _candidate(
            kb_id=kb_id,
            doc_id=answer_doc_id,
            chunk_index=0,
            filename="差旅补贴标准.md",
            content="D级餐补标准为100元/天。",
        )

        payloads, client, search, *_ = await self._run(
            question="普通员工的餐补标准是多少",
            kb_id=kb_id,
            initial=[],
            initial_sequence=[
                [answer],
                TimeoutError("plan query timed out"),
            ],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(answer_doc_id)},
        )
        self.assertIn(
            "plan_query_retrieval_timeout",
            result["evidence_state"]["reasons"],
        )
        self.assertEqual(search.await_count, 2)
        self.assertEqual(len(client.completions.calls), 1)
        answer_text = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertNotIn("服务暂时不可用", answer_text)

    async def test_multi_hop_selected_scope_never_runs_global_plan_query(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        selected_doc_id = uuid.uuid4()
        outside_doc_id = uuid.uuid4()
        selected = _candidate(
            kb_id=kb_id,
            doc_id=selected_doc_id,
            chunk_index=0,
            filename="已选择的8.6制度.md",
            content="普通员工对应D级；D级餐补标准为100元/天。",
        )
        outside = _candidate(
            kb_id=kb_id,
            doc_id=outside_doc_id,
            chunk_index=0,
            filename="范围外的7.0制度.md",
            content="普通员工对应C级；C级餐补标准为200元/天。",
        )
        scope_filter = {
            "mode": "single",
            "kb_ids": [str(kb_id)],
            "doc_ids": [str(selected_doc_id)],
            "choices": [{
                "key": "c1",
                "label": "云枢 8.6 —《已选择的8.6制度.md》",
                "products": ["云枢"],
                "canonical_products": ["CloudPivot"],
                "versions": ["8.6"],
                "projects": [],
                "filenames": ["已选择的8.6制度.md"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(selected_doc_id)],
                "anchor_doc_ids": [str(selected_doc_id)],
                "companion_doc_ids": [],
            }],
        }

        payloads, client, search, _fetch, scoped_search = await self._run(
            question="普通员工的餐补标准是多少",
            kb_id=kb_id,
            initial=[],
            scoped=[selected, outside],
            full_document=[selected, outside],
            evidence_scope_filter=scope_filter,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(selected_doc_id)},
        )
        search.assert_not_awaited()
        scoped_search.assert_awaited()
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("100元", prompt)
        self.assertNotIn("200元", prompt)

    async def test_global_plan_query_respects_explicit_version_constraint(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        answer_doc_id = uuid.uuid4()
        bridge_doc_id = uuid.uuid4()
        wrong_version_doc_id = uuid.uuid4()
        answer = _candidate(
            kb_id=kb_id,
            doc_id=answer_doc_id,
            chunk_index=0,
            filename="云枢8.6差旅标准.md",
            content="所属产品：云枢；版本：8.6。D级餐补为100元/天。",
        )
        bridge = _candidate(
            kb_id=kb_id,
            doc_id=bridge_doc_id,
            chunk_index=0,
            filename="云枢8.6职级分类.md",
            content="所属产品：云枢；版本：8.6。普通员工对应D级。",
        )
        wrong_version = _candidate(
            kb_id=kb_id,
            doc_id=wrong_version_doc_id,
            chunk_index=0,
            filename="云枢7职级分类.md",
            content="所属产品：云枢；版本：7。普通员工对应C级。",
        )

        payloads, client, search, *_ = await self._run(
            question="云枢8.6普通员工的餐补标准是多少",
            kb_id=kb_id,
            initial=[],
            initial_sequence=[
                [answer],
                [bridge, wrong_version],
            ],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(search.await_count, 2)
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(answer_doc_id), str(bridge_doc_id)},
        )
        self.assertNotIn(
            str(wrong_version_doc_id),
            {item["doc_id"] for item in result["results"]},
        )
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("版本：8.6", prompt)
        self.assertNotIn("版本：7", prompt)

    async def test_narrow_fact_uses_one_small_document_without_scoped_search(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        full = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=[
                "对象与等级的映射关系。",
                "该等级对应的最终数值为100。",
            ],
        )
        initial = [dict(full[1])]
        initial[0].update(
            score=0.09,
            retrieval_score=0.09,
            vector_score=0.86,
            vector_rank=1,
            active_channels=["vector"],
        )

        payloads, client, _search, fetch_full, scoped = await self._run(
            question="最终数值是多少",
            kb_id=kb_id,
            initial=initial,
            full_document=full,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(len(result["answer_sources"]), 1)
        self.assertIn("最终数值为100", result["answer_sources"][0]["content"])
        self.assertEqual(len(client.completions.calls), 1)
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("对象与等级的映射关系", prompt)
        self.assertIn("最终数值为100", prompt)
        fetch_full.assert_awaited_once()
        scoped.assert_not_awaited()

    async def test_expansion_failures_retain_initial_authorized_evidence(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        initial = [_candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=3,
            content="明确条款：某项上限为450元/天。",
        )]

        payloads, client, *_ = await self._run(
            question="请说明某项标准",
            kb_id=kb_id,
            initial=initial,
            full_document=TimeoutError("small document timeout"),
            scoped=TimeoutError("scoped search timeout"),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(len(result["answer_sources"]), 1)
        self.assertIn(
            "明确条款",
            client.completions.calls[0]["messages"][1]["content"],
        )
        self.assertNotIn(
            "服务暂时不可用",
            client.completions.calls[0]["messages"][0]["content"],
        )

    async def test_retrieval_workflow_deadline_is_shared_across_stages(self) -> None:
        kb_id = uuid.uuid4()
        current = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="项目成员的内部额度为200元。",
            filename="内部管理标准.md",
        )
        carryover = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=1,
            content="项目成员内部额度的适用说明。",
            filename="内部管理标准.md",
        )
        settings = _settings()
        settings.rag_v2_retrieval_timeout_seconds = 0.5
        settings.rag_v2_expansion_timeout_seconds = 0.5
        settings.rag_v2_retrieval_workflow_timeout_seconds = 0.08

        started = time.perf_counter()
        payloads, client, search, fetch_full, scoped = await self._run(
            question="项目成员的内部额度是多少",
            kb_id=kb_id,
            initial=[current],
            full_document=[],
            carryover_sources=[carryover],
            initial_delay_seconds=0.02,
            blocking_scoped=True,
            settings_override=settings,
        )
        elapsed = time.perf_counter() - started

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertLess(elapsed, 0.3)
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertGreater(result["answer_source_count"], 0)
        self.assertIn(
            "retrieval_workflow_deadline_exhausted",
            result["evidence_state"]["reasons"],
        )
        self.assertEqual(len(client.completions.calls), 1)
        self.assertEqual(search.await_count, 2)
        scoped.assert_awaited_once()
        # The carryover phase consumed the shared remainder.  Expansion must
        # fail before starting new I/O, while the first-pass evidence survives.
        fetch_full.assert_not_awaited()

    async def test_generation_workflow_deadline_bounds_all_attempts(self) -> None:
        kb_id = uuid.uuid4()
        initial = [_candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="项目成员的内部额度为200元。",
            filename="内部管理标准.md",
        )]
        settings = _settings()
        settings.llm_request_timeout_seconds = 1
        settings.llm_max_attempts = 3
        settings.rag_v2_generation_workflow_timeout_seconds = 0.05
        client = _HangingClient()

        started = time.perf_counter()
        payloads, *_ = await self._run(
            question="项目成员的内部额度是多少",
            kb_id=kb_id,
            initial=initial,
            full_document=[],
            settings_override=settings,
            client_override=client,
            expected_error=asyncio.TimeoutError,
        )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.25)
        self.assertEqual(len(client.completions.calls), 1)
        self.assertTrue(client.completions.cancelled)
        self.assertLessEqual(client.completions.calls[0]["timeout"], 0.05)
        self.assertFalse(any(
            item["type"] == "text_delta" for item in payloads
        ))

    async def test_generation_deadline_after_first_delta_never_replays(self) -> None:
        kb_id = uuid.uuid4()
        initial = [_candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="项目成员的内部额度为200元。",
            filename="内部管理标准.md",
        )]
        settings = _settings()
        settings.llm_request_timeout_seconds = 1
        settings.llm_max_attempts = 3
        settings.rag_v2_generation_workflow_timeout_seconds = 0.05
        client = _HangingClient(first_delta="第一段")

        payloads, *_ = await self._run(
            question="项目成员的内部额度是多少",
            kb_id=kb_id,
            initial=initial,
            full_document=[],
            settings_override=settings,
            client_override=client,
            expected_error=asyncio.TimeoutError,
        )

        deltas = [
            item["content"]
            for item in payloads
            if item["type"] == "text_delta"
        ]
        self.assertEqual(deltas, ["第一段"])
        self.assertEqual(len(client.completions.calls), 1)
        self.assertTrue(client.completions.cancelled)

    async def test_primary_retrieval_failure_is_not_reported_as_no_hit(self) -> None:
        kb_id = uuid.uuid4()

        payloads, client, _search, fetch_full, scoped = await self._run(
            question="查询当前制度",
            kb_id=kb_id,
            initial=TimeoutError("database unavailable"),
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "error")
        self.assertEqual(result["evidence_availability"], "unavailable")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(client.completions.calls, [])
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertIn(
            "服务暂时不可用",
            answer,
        )
        self.assertNotIn(
            "未找到相关内容",
            answer,
        )
        fetch_full.assert_not_awaited()
        scoped.assert_not_awaited()

    async def test_primary_retrieval_failure_rejects_authorized_followup_seed(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        seed = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=2,
            content="住宿标准：普通员工一线城市每天不超过450元。",
        )

        payloads, client, search, fetch_full, scoped = await self._run(
            question="那住宿呢",
            kb_id=kb_id,
            initial=TimeoutError("database unavailable"),
            full_document=[],
            carryover_sources=[seed],
            standalone_query="那住宿呢。普通员工的出差标准是什么",
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "error")
        self.assertEqual(result["evidence_availability"], "unavailable")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(result["answer_source_count"], 0)
        self.assertFalse(result["carryover_seed_used"])
        self.assertEqual(result["carryover_anchor_succeeded"], False)
        self.assertEqual(result["carryover_candidate_count"], 0)
        self.assertEqual(client.completions.calls, [])
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertIn("服务暂时不可用", answer)
        self.assertNotIn("450元", answer)
        fetch_full.assert_not_awaited()
        scoped.assert_not_awaited()
        search.assert_awaited_once()

    async def test_candidate_without_raw_quality_signal_is_no_hit(self) -> None:
        kb_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="缺少召回质量观测的候选正文。",
        )
        for field in (
            "vector_score",
            "vector_rank",
            "keyword_score",
            "keyword_rank",
            "trigram_score",
            "trigram_rank",
            "active_channels",
        ):
            candidate.pop(field, None)

        payloads, client, _search, fetch_full, scoped = await self._run(
            question="查询当前制度",
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "no_hit")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(client.completions.calls, [])
        fetch_full.assert_not_awaited()
        scoped.assert_not_awaited()
        completed = next(
            call
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.completed"
        )
        self.assertEqual(
            completed.kwargs["relevance_reason"],
            "adapter_quality_signal_missing",
        )

    async def test_verified_scope_bounds_uncalibrated_candidate(self) -> None:
        kb_id = uuid.uuid4()
        selected_doc_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=selected_doc_id,
            chunk_index=0,
            content="用户已选择范围内的受限正文。",
            filename="已选择范围.md",
        )
        for field in (
            "vector_score",
            "vector_rank",
            "keyword_score",
            "keyword_rank",
            "trigram_score",
            "trigram_rank",
            "active_channels",
        ):
            candidate.pop(field, None)
        scope_filter = {
            "mode": "single",
            "kb_ids": [str(kb_id)],
            "doc_ids": [str(selected_doc_id)],
            "choices": [{
                "key": "c1",
                "label": "已选择范围 —《已选择范围.md》",
                "products": [],
                "canonical_products": [],
                "versions": [],
                "projects": [],
                "filenames": ["已选择范围.md"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(selected_doc_id)],
                "anchor_doc_ids": [str(selected_doc_id)],
                "companion_doc_ids": [],
            }],
        }

        payloads, client, search, _fetch, scoped = await self._run(
            question="配置是什么",
            kb_id=kb_id,
            initial=[],
            scoped=[candidate],
            full_document=[],
            evidence_scope_filter=scope_filter,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["answer_source_count"], 1)
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(selected_doc_id)},
        )
        self.assertTrue(result["evidence_scope_anchor_hit"])
        self.assertEqual(len(client.completions.calls), 1)
        search.assert_not_awaited()
        scoped.assert_awaited()

    async def test_explicit_version_mismatch_never_enters_context(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        initial = [_candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            filename="CloudPivot 7 配置.md",
            content="所属产品：CloudPivot；产品版本：7。旧范围配置值为true。",
        )]

        payloads, client, *_ = await self._run(
            question="CloudPivot 8.6 的配置值是多少",
            kb_id=kb_id,
            initial=initial,
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "version_mismatch")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(client.completions.calls, [])
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertNotIn("旧范围配置值为true", answer)
        self.assertIn("指定产品、版本或适用范围", answer)

    async def test_retriever_output_outside_authorized_kb_is_rejected(self) -> None:
        authorized_kb_id = uuid.uuid4()
        leaked = _candidate(
            kb_id=uuid.uuid4(),
            doc_id=uuid.uuid4(),
            chunk_index=0,
            content="不属于当前用户范围的敏感正文。",
        )

        payloads, client, _search, fetch_full, _scoped = await self._run(
            question="查询当前制度",
            kb_id=authorized_kb_id,
            initial=[leaked],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "no_hit")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(client.completions.calls, [])
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertNotIn("敏感正文", answer)
        fetch_full.assert_not_awaited()

    async def test_mutually_exclusive_scopes_clarify_before_generation(self) -> None:
        kb_id = uuid.uuid4()
        first_doc = uuid.uuid4()
        second_doc = uuid.uuid4()
        initial = [
            _candidate(
                kb_id=kb_id,
                doc_id=first_doc,
                chunk_index=0,
                filename="CloudPivot 6 安全配置.md",
                content="所属产品：CloudPivot；产品版本：6。安全配置方法A。",
            ),
            _candidate(
                kb_id=kb_id,
                doc_id=second_doc,
                chunk_index=0,
                filename="CloudPivot 7 安全配置.md",
                content="所属产品：CloudPivot；产品版本：7。安全配置方法B。",
            ),
        ]

        payloads, client, *_ = await self._run(
            question="如何设置安全配置",
            kb_id=kb_id,
            initial=initial,
            full_document=[],
        )

        result_index = next(
            index for index, item in enumerate(payloads) if item["type"] == "search_results"
        )
        clarification_index = next(
            index
            for index, item in enumerate(payloads)
            if item["type"] == "evidence_clarification"
        )
        result = payloads[result_index]
        clarification = payloads[clarification_index]
        self.assertLess(result_index, clarification_index)
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertEqual(result["answer_sources"], [])
        self.assertTrue(clarification["needs_clarification"])
        self.assertEqual(len(clarification["choices"]), 2)
        self.assertIn("都对比", clarification["question"])
        self.assertEqual(len(client.completions.calls), 0)
        self.assertEqual(payloads[-1]["type"], "done")

    async def test_broad_document_topic_clarifies_and_displays_each_choice(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        leave_doc = uuid.uuid4()
        travel_doc = uuid.uuid4()
        leave_chunks = [
            _candidate(
                kb_id=kb_id,
                doc_id=leave_doc,
                chunk_index=index,
                filename="员工请假管理办法.docx",
                content=f"员工请假制度第{index + 1}部分：审批、休假和销假要求。",
            )
            for index in range(6)
        ]
        travel = _candidate(
            kb_id=kb_id,
            doc_id=travel_doc,
            chunk_index=0,
            filename="公司出差管理标准.docx",
            content="员工出差交通、住宿和餐饮补贴标准。",
        )

        payloads, client, *_ = await self._run(
            question="员工标准是什么",
            kb_id=kb_id,
            initial=[*leave_chunks, travel],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        clarification = next(
            item for item in payloads if item["type"] == "evidence_clarification"
        )
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(clarification["dimension"], "document")
        self.assertEqual(
            {choice["doc_ids"][0] for choice in clarification["choices"]},
            {str(leave_doc), str(travel_doc)},
        )
        displayed_doc_ids = {item["doc_id"] for item in result["results"]}
        self.assertEqual(displayed_doc_ids, {str(leave_doc), str(travel_doc)})
        self.assertLessEqual(len(result["results"]), 5)
        self.assertEqual(client.completions.calls, [])

    async def test_low_score_mutually_exclusive_scope_is_not_dropped_before_clarification(self) -> None:
        kb_id = uuid.uuid4()
        first_doc = uuid.uuid4()
        second_doc = uuid.uuid4()
        first = _candidate(
            kb_id=kb_id,
            doc_id=first_doc,
            chunk_index=0,
            filename="CloudPivot 6 安全配置.md",
            content="所属产品：CloudPivot；产品版本：6。安全配置方法A。",
        )
        first.update(
            vector_score=0.90,
            vector_rank=1,
            active_channels=["vector"],
        )
        second = _candidate(
            kb_id=kb_id,
            doc_id=second_doc,
            chunk_index=0,
            filename="CloudPivot 7 安全配置.md",
            content="所属产品：CloudPivot；产品版本：7。安全配置方法B。",
        )
        # Deliberately outside MAX_DOC_VECTOR_GAP.  The source identity still
        # makes it an independent applicable scope, so it must survive long
        # enough for the clarification gate to present both choices.
        second.update(
            vector_score=0.82,
            vector_rank=2,
            active_channels=["vector"],
        )

        payloads, client, *_ = await self._run(
            question="如何设置安全配置",
            kb_id=kb_id,
            initial=[first, second],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        clarification = next(
            item for item in payloads if item["type"] == "evidence_clarification"
        )
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertEqual(len(clarification["choices"]), 2)
        self.assertEqual(
            {choice["doc_ids"][0] for choice in clarification["choices"]},
            {str(first_doc), str(second_doc)},
        )
        self.assertEqual(client.completions.calls, [])

    async def test_uncalibrated_ambiguity_candidate_cannot_bypass_gate(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        calibrated_doc_id = uuid.uuid4()
        uncalibrated_doc_id = uuid.uuid4()
        calibrated = _candidate(
            kb_id=kb_id,
            doc_id=calibrated_doc_id,
            chunk_index=0,
            filename="CloudPivot 6 安全配置.md",
            content="所属产品：CloudPivot；产品版本：6。可信配置方法A。",
        )
        uncalibrated = _candidate(
            kb_id=kb_id,
            doc_id=uncalibrated_doc_id,
            chunk_index=0,
            filename="CloudPivot 7 安全配置.md",
            content="所属产品：CloudPivot；产品版本：7。无质量观测配置方法B。",
        )
        for field in (
            "vector_score",
            "vector_rank",
            "keyword_score",
            "keyword_rank",
            "trigram_score",
            "trigram_rank",
            "active_channels",
        ):
            uncalibrated.pop(field, None)

        payloads, client, *_ = await self._run(
            question="如何设置安全配置",
            kb_id=kb_id,
            initial=[calibrated, uncalibrated],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertNotEqual(result["evidence_status"], "needs_clarification")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(calibrated_doc_id)},
        )
        self.assertFalse(any(
            item["type"] == "evidence_clarification" for item in payloads
        ))
        self.assertEqual(len(client.completions.calls), 1)
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("可信配置方法A", prompt)
        self.assertNotIn("无质量观测配置方法B", prompt)


if __name__ == "__main__":
    unittest.main()
