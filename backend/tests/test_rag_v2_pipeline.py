import asyncio
import json
import time
import unittest
import uuid
from contextlib import asynccontextmanager
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
    AnchorRetrievalSnapshot,
    _admit_and_bind_expansion_candidates,
    _evidence_execution_decision,
    _should_use_general_model_fallback,
    _should_model_adjudicate_evidence,
    retrieve_anchor_retrieval_snapshot,
    run_rag_v2_stream,
)
from core.rag_v2.contracts import AnswerRequirementV2, QueryPlanV2
from core.rag_v2.query_plan import (
    partition_plan_by_applicability_scopes,
    plan_query_locally,
)
from core.rag_v2.task_graph import (
    compile_rag_execution_bundle,
    compile_retrieval_task_graph,
)
from core.rag_v2.task_execution import PhysicalRetrievalGroup, TaskExecutionLedger
from core.query_constraints import (
    ApplicabilityScope,
    ScopeSourceSpan,
    admit_candidates_for_scopes,
)
from core.structured_output import clear_structured_output_capability_cache


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
        rag_v2_task_query_parallelism=3,
    )


def _task_contract(
    question: str,
    *,
    requirements=None,
    relation: str = "new",
    query_mode: str | None = None,
    evidence_scope: str = "enterprise_kb",
):
    mode = query_mode or ("contextualize" if relation != "new" else "current")
    contextual = mode == "contextualize"
    route = parse_rag_route_decision(
        {
            "schema_version": "rag_route_decision.v1",
            "readiness": "ready",
            "intent_code": "knowledge_qa",
            "relation": relation,
            "evidence_scope": evidence_scope,
            "query_resolution": {
                "mode": mode,
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


def _direct_test_execution_bundle(
    question: str,
    *,
    requirements=None,
):
    """Build an explicit, ledgerable test plan when local planning is out of scope.

    The pipeline tests exercise retrieval, authorization and evidence behavior;
    they must not rely on the runner's removed "plan it for me" entrypoint.
    A test that needs bridge semantics supplies a fully typed bundle itself.
    """

    answer_descriptions: list[str] = []
    for item in requirements or ():
        if not isinstance(item, dict) or item.get("role") != "answer":
            continue
        description = str(item.get("description") or "").strip()
        if description and description not in answer_descriptions:
            answer_descriptions.append(description)
    if not answer_descriptions:
        answer_descriptions.append(question)
    answer_shape = "fact" if len(answer_descriptions) == 1 else "multi_part"
    plan = QueryPlanV2(
        original_query=question,
        answer_shape=answer_shape,
        retrieval_queries=tuple(answer_descriptions),
        requirements=tuple(
            AnswerRequirementV2(
                id=f"r{index}",
                description=description,
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
            )
            for index, description in enumerate(answer_descriptions, start=1)
        ),
        confidence=0.95,
        source="local",
    )
    bundle = compile_rag_execution_bundle(plan)
    if not bundle.uses_task_ledger:
        raise AssertionError("direct test plan must be ledgered")
    return bundle


def _typed_bridge_execution_bundle(
    question: str,
    *,
    answer_description: str | None = None,
    bridge_subject: str,
    bridge_kind: str,
    bridge_description: str | None = None,
    edge_mode: str = "proof",
    scope_product: str | None = None,
    scope_version: str | None = None,
) -> object:
    """Compile one explicit bridge route for a pipeline integration test.

    Test callers must declare the semantic relation they intend to exercise;
    the runner is deliberately no longer a fallback planner.  This helper
    mirrors the production handoff exactly: a typed ``QueryPlanV2`` is
    compiled once into an immutable ledgered execution bundle.
    """

    if edge_mode not in {"proof", "augmentation"}:
        raise ValueError("test bridge edge mode must be proof or augmentation")
    answer_description = answer_description or question
    bridge_description = bridge_description or (
        f"确认{bridge_subject}对应的适用{bridge_kind}"
    )
    answer = AnswerRequirementV2(
        id="r1",
        description=answer_description,
        role="answer",
        importance="required",
        source="explicit",
        depends_on_requirement_ids=("r2",) if edge_mode == "proof" else (),
        augmentation_requirement_ids=("r2",)
        if edge_mode == "augmentation" else (),
        scope_product=scope_product,
        scope_version=scope_version,
        scope_explicit_version=scope_version is not None,
    )
    bridge = AnswerRequirementV2(
        id="r2",
        description=bridge_description,
        role="bridge",
        importance="helpful",
        source="inferred",
        bridge_subject=bridge_subject,
        bridge_kind=bridge_kind,
        scope_product=scope_product,
        scope_version=scope_version,
        scope_explicit_version=scope_version is not None,
    )
    plan = QueryPlanV2(
        original_query=question,
        # An augmentation can release a second-hop retrieval path, but it is
        # never a proof prerequisite.  Calling that plan multi_hop would make
        # the test helper encode the old hard-dependency semantics it is
        # supposed to validate.
        answer_shape="multi_hop" if edge_mode == "proof" else "fact",
        # This is deliberately inert legacy projection.  Every assertion in
        # this module must exercise the task graph, not query-array position.
        retrieval_queries=("test legacy projection must not execute",),
        requirements=(answer, bridge),
        confidence=0.95,
        source="model",
    )
    bundle = compile_rag_execution_bundle(plan)
    if not bundle.uses_task_ledger:
        raise AssertionError("typed test bridge plan must be ledgered")
    return bundle
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
    def test_execution_strategy_uses_evidence_topology_not_question_terms(self) -> None:
        doc_id = str(uuid.uuid4())
        candidates = [
            {
                "id": f"chunk-{index}",
                "doc_id": doc_id,
                "content": f"第{index + 1}段制度正文",
                "full_document_chunk_count": 3,
                "candidate_origins": (
                    ["small_document_full", "initial_retrieval"]
                    if index == 1
                    else ["small_document_full"]
                ),
            }
            for index in range(3)
        ]

        decision = _evidence_execution_decision(
            deterministic_closed=False,
            candidates=candidates,
            full_document_candidates=candidates,
        )

        self.assertEqual(decision.strategy, "bounded_small_document")
        self.assertEqual(decision.eligible_candidate_indexes, (1, 2, 3))
        self.assertEqual(decision.anchor_candidate_indexes, (2,))

    async def test_small_document_adjudication_failure_keeps_unverified_source_context(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        full_document = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            filename="内部制度.docx",
            contents=[
                "第一章 制度适用与基本原则。",
                "特殊场景应结合审批记录，由责任部门复核后处理。",
            ],
        )
        initial = dict(full_document[0])
        initial.update(
            vector_score=0.91,
            vector_rank=1,
            active_channels=["vector"],
        )
        settings = SimpleNamespace(
            **vars(_settings()),
            rag_v2_model_evidence_adjudication_enabled=True,
            rag_v2_model_evidence_adjudication_timeout_seconds=1,
        )
        failed = SimpleNamespace(
            succeeded=False,
            error="TimeoutError: evidence adjudication timed out",
        )

        with (
            patch(
                "core.rag_v2.pipeline.select_small_document_evidence_with_coverage",
                new=AsyncMock(return_value=failed),
            ) as small_adjudicator,
            patch(
                "core.rag_v2.pipeline.joint_rerank_with_coverage",
                new=AsyncMock(),
            ) as joint_adjudicator,
        ):
            payloads, client, *_ = await self._run(
                question="遇到例外时应该怎么办",
                kb_id=kb_id,
                initial=[initial],
                full_document=full_document,
                settings_override=settings,
            )

        small_adjudicator.assert_awaited_once()
        joint_adjudicator.assert_not_awaited()
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_execution_strategy"], "bounded_small_document")
        self.assertEqual(result["model_adjudication_state"], "failed")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertTrue(result["unverified_generation"])
        self.assertEqual(result["source_verification"], "unverified")
        self.assertEqual(result["answer_source_count"], 2)
        self.assertEqual(result["direct_evidence_count"], 0)
        self.assertEqual(result["hit_count"], 0)
        self.assertEqual(result["unverified_reference_count"], 2)
        self.assertEqual(result["covered_requirement_ids"], [])
        self.assertTrue(all(
            source["source_verification"] == "unverified"
            for source in result["answer_sources"]
        ))
        fallback_trace = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "evidence.unverified_fallback"
        )
        self.assertTrue(fallback_trace["activated"])
        self.assertEqual(fallback_trace["input_candidate_count"], 2)
        self.assertEqual(fallback_trace["authorized_candidate_count"], 2)
        self.assertEqual(fallback_trace["requirement_bound_candidate_count"], 2)
        self.assertEqual(fallback_trace["converted_candidate_count"], 2)
        self.assertEqual(fallback_trace["selected_candidate_count"], 2)
        self.assertEqual(fallback_trace["exclusion_reason_counts"], {})
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("特殊场景应结合审批记录", prompt)
        self.assertIn("语义支持关系尚未由重排模型验证", prompt)
        self.assertIn("必须逐个范围独立回答", prompt)

    async def test_configuration_assignment_is_direct_evidence_without_ordered_steps(
        self,
    ) -> None:
        """A YAML assignment must close a configuration answer globally."""

        question = "云枢如何修改默认密码"
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirement = AnswerRequirementV2(
            id="r1",
            description=question,
            role="answer",
            importance="required",
            source="explicit",
            coverage_mode="collection",
            coverage_contract="structured_collection",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        )
        execution_bundle = compile_rag_execution_bundle(QueryPlanV2(
            original_query=question,
            answer_shape="process",
            retrieval_queries=(question,),
            requirements=(requirement,),
            confidence=0.95,
            source="local",
        ))
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            filename="云枢6配置参数说明",
            content=(
                "```yaml\n"
                "cloudpivot:\n"
                "  organization:\n"
                "    defaultPwd: Authine@123456 # 默认密码配置\n"
                "  switch:\n"
                "    force_change_default_password: true # 默认密码强制修改\n"
                "```"
            ),
        )
        candidate["metadata"] = {"product": "云枢", "version": "6"}

        payloads, client, *_ = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[candidate],
            full_document=[candidate],
            execution_bundle=execution_bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        process = next(item for item in payloads if item["type"] == "search_process")
        self.assertEqual(
            [step["key"] for step in process["steps"]],
            ["analyze", "expand", "retrieve", "rerank", "generate"],
        )
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(doc_id)},
        )
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("force_change_default_password", prompt)

    async def test_flattened_configuration_block_closes_without_model_adjudication(
        self,
    ) -> None:
        """Importer-flattened YAML keeps configuration evidence deterministic."""

        question = "云枢6如何修改默认密码"
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirement = AnswerRequirementV2(
            id="r1",
            description=question,
            role="answer",
            importance="required",
            source="explicit",
            coverage_mode="collection",
            coverage_contract="structured_collection",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        )
        execution_bundle = compile_rag_execution_bundle(QueryPlanV2(
            original_query=question,
            answer_shape="process",
            retrieval_queries=(question,),
            requirements=(requirement,),
            confidence=0.95,
            source="local",
        ))
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            filename="云枢6配置参数说明",
            content=(
                "```yaml cloudpivot: organization: defaultPwd: "
                "Authine@123456 #默认密码配置 login: error_reply_same: true "
                "#解决登录用户名枚举 switch: force_change_default_password: true "
                "#默认密码强制修改 ```"
            ),
        )
        candidate["metadata"] = {"product": "云枢", "version": "6"}

        payloads, client, *_ = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[candidate],
            full_document=[candidate],
            execution_bundle=execution_bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(doc_id)},
        )
        self.assertTrue(any(
            "defaultPwd" in str(call)
            for call in client.completions.calls
        ))

    async def test_unversioned_configuration_routes_are_not_silently_merged(
        self,
    ) -> None:
        """Different version assignments must remain separate answer routes."""

        question = "云枢如何修改默认密码"
        kb_id = uuid.uuid4()
        requirement = AnswerRequirementV2(
            id="r1",
            description=question,
            role="answer",
            importance="required",
            source="explicit",
            coverage_mode="collection",
            coverage_contract="structured_collection",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        )
        execution_bundle = compile_rag_execution_bundle(QueryPlanV2(
            original_query=question,
            answer_shape="process",
            retrieval_queries=(question,),
            requirements=(requirement,),
            confidence=0.95,
            source="local",
        ))
        candidates = []
        for version, default_key in (("6", "defaultPwd"), ("7", "defaultPwd1")):
            candidate = _candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=0,
                filename=f"云枢{version}配置",
                content=(
                    f"云枢{version}解决方案：\n"
                    f"cloudpivot.organization.{default_key}: Authine@123456 # 默认密码配置\n"
                    "cloudpivot.switch.force_change_default_password: true # 默认密码强制修改"
                ),
            )
            candidate["metadata"] = {"product": "云枢", "version": version}
            candidates.append(candidate)

        payloads, *_ = await self._run(
            question=question,
            kb_id=kb_id,
            initial=candidates,
            full_document=candidates,
            execution_bundle=execution_bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertTrue(result.get("clarification"))

    async def test_scope_metadata_cannot_close_a_shared_answer_target(self) -> None:
        """Applicability matches are filters, not proof of the requested act."""

        question = "在平台A中如何给自定义用户发送通知消息"
        kb_id = uuid.uuid4()
        base_requirement = AnswerRequirementV2(
            id="r1",
            description=question,
            role="answer",
            importance="required",
            source="explicit",
            coverage_mode="collection",
            coverage_contract="ordered_steps",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        )
        base_plan = QueryPlanV2(
            original_query=question,
            answer_shape="process",
            retrieval_queries=(question,),
            requirements=(base_requirement,),
            confidence=0.95,
            source="model",
        )
        versions = ("6", "6.0.1", "7", "8.2.75")
        scoped_plan = partition_plan_by_applicability_scopes(
            base_plan,
            tuple(
                ApplicabilityScope(product="平台A", version=version)
                for version in versions
            ),
            comparison=True,
        )
        candidates = []
        for version in versions:
            candidate = _candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=0,
                filename=f"平台A {version} 说明",
                content=f"所属产品：平台A\n产品版本：{version}",
            )
            candidate["metadata"] = {"product": "平台A", "version": version}
            candidates.append(candidate)
        settings = SimpleNamespace(
            **vars(_settings()),
            rag_v2_model_evidence_adjudication_enabled=True,
            rag_v2_model_evidence_adjudication_timeout_seconds=1,
        )
        failed = SimpleNamespace(
            succeeded=False,
            error="simulated adjudication failure",
        )

        with patch(
            "core.rag_v2.pipeline.joint_rerank_with_coverage",
            new=AsyncMock(return_value=failed),
        ) as adjudicator:
            payloads, *_ = await self._run(
                question=question,
                kb_id=kb_id,
                initial=candidates,
                full_document=candidates,
                execution_bundle=compile_rag_execution_bundle(scoped_plan),
                settings_override=settings,
            )

        adjudicator.assert_awaited_once()
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_execution_strategy"], "joint_adjudication")
        self.assertEqual(result["model_adjudication_state"], "failed")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["coverage_status"], "partial")
        self.assertTrue(result["unverified_generation"])
        self.assertEqual(result["direct_evidence_count"], 0)
        self.assertEqual(result["hit_count"], 0)
        self.assertEqual(result["unverified_reference_count"], len(versions))
        self.assertEqual(result["covered_requirement_ids"], [])
        self.assertEqual(
            {source["metadata"].get("version") for source in result["answer_sources"]},
            set(versions),
        )
        self.assertTrue(all(
            source["source_verification"] == "unverified"
            for source in result["answer_sources"]
        ))

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
        initial_sequence: list[list[dict] | Exception] | None = None,
        search_side_effect=None,
        scoped_sequence: list[list[dict] | Exception] | None = None,
        scoped_side_effect=None,
        initial_delay_seconds: float = 0,
        blocking_scoped: bool = False,
        settings_override=None,
        client_override=None,
        task_contract_override=None,
        execution_bundle=None,
        task_read_session_factory=None,
        request_db=None,
        anchor_retrieval_snapshot=None,
        anchor_retrieval_revision=None,
        top_k: int = 5,
        expected_error: type[BaseException] | None = None,
    ):
        client = client_override or _FakeClient()
        search = AsyncMock()
        if search_side_effect is not None:
            search.side_effect = search_side_effect
        elif initial_sequence is not None:
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
        if scoped_side_effect is not None:
            scoped_search.side_effect = scoped_side_effect
        elif scoped_sequence is not None:
            scoped_search.side_effect = scoped_sequence
        elif blocking_scoped:
            async def block_scoped(*_args, **_kwargs):
                await asyncio.sleep(60)

            scoped_search.side_effect = block_scoped
        elif isinstance(scoped, Exception):
            scoped_search.side_effect = scoped
        else:
            scoped_search.return_value = scoped or []
        structural_search = AsyncMock(return_value=[])
        task_contract = (
            task_contract_override
            or _task_contract(
                question,
                requirements=requirements,
            )
        )
        # The runner's only executable handoff is an immutable, ledgered
        # bundle.  Tests that do not need a hand-crafted plan still compile
        # the same request-local contract here instead of exercising the
        # deprecated plan/graph arguments.
        if execution_bundle is not None:
            effective_execution_bundle = execution_bundle
        else:
            local_bundle = compile_rag_execution_bundle(
                plan_query_locally(standalone_query or question)
            )
            effective_execution_bundle = (
                local_bundle
                if local_bundle.uses_task_ledger
                else _direct_test_execution_bundle(
                    standalone_query or question,
                    requirements=requirements,
                )
            )

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
                        "top_k": top_k,
                        "method": "hybrid",
                        "rerank": True,
                    },
                    conversation_id="v2-test-conversation",
                    db=(request_db if request_db is not None else SimpleNamespace()),
                    intent={"intent_code": "knowledge_qa"},
                    task_contract=task_contract,
                    evidence_scope_filter=evidence_scope_filter,
                    carryover_sources=carryover_sources,
                    is_followup=bool(carryover_sources),
                    standalone_query=standalone_query,
                    execution_bundle=effective_execution_bundle,
                    task_read_session_factory=task_read_session_factory,
                    anchor_retrieval_snapshot=anchor_retrieval_snapshot,
                    anchor_retrieval_revision=anchor_retrieval_revision,
                ):
                    chunks.append(chunk)

            if expected_error is None:
                await collect_chunks()
            else:
                with self.assertRaises(expected_error):
                    await collect_chunks()
        self._last_trace = trace
        return _payloads(chunks), client, search, fetch_full, scoped_search

    def _scoped_expansion_context(
        self,
        *,
        scope: ApplicabilityScope,
        seed: dict,
    ) -> tuple[TaskExecutionLedger, PhysicalRetrievalGroup, list[dict]]:
        """Create one genuinely admitted first-pass seed for expansion tests."""

        requirement = AnswerRequirementV2(
            id="r1",
            description="CloudPivot 范围内的配置标准",
            role="answer",
            importance="required",
            source="explicit",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
            applicability_scope=scope,
        )
        graph = compile_retrieval_task_graph(QueryPlanV2(
            original_query=requirement.description,
            answer_shape="fact",
            retrieval_queries=(requirement.description,),
            requirements=(requirement,),
            confidence=0.95,
            source="local",
        ))
        ledger = TaskExecutionLedger(graph, run_id="scope-expansion-test")
        group = PhysicalRetrievalGroup(
            group_id="answer_r1",
            query=requirement.description,
            task_ids=("answer_r1",),
            scope_product=None,
            scope_version=None,
            scope_explicit_version=False,
            applicability_scope=scope,
        )
        admission = admit_candidates_for_scopes([seed], (scope,))
        self.assertEqual(len(admission.candidates), 1)
        execution_id = ledger.begin_execution(
            kind="initial_task_query",
            query=group.query,
            task_ids=group.task_ids,
        )
        admitted_seed = ledger.observe_candidates(
            admission.candidates,
            execution_id=execution_id,
        )
        ledger.finish_execution(
            execution_id,
            status="succeeded",
            candidate_count=len(admitted_seed),
        )
        return ledger, group, admitted_seed

    def test_expansion_scope_rejection_never_acquires_lineage(self):
        """A rejected V7 sibling cannot borrow a V6 task through document id."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        scope = ApplicabilityScope(
            product="CloudPivot",
            version="6",
            explicit_version=True,
        )
        seed = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            content="所属产品：CloudPivot；产品版本：6。V6 登录保护已启用。",
        )
        seed["metadata"] = {"product": "CloudPivot", "version": "6"}
        ledger, group, admitted_seed = self._scoped_expansion_context(
            scope=scope,
            seed=seed,
        )
        rejected = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content="所属产品：CloudPivot；产品版本：7。V7 登录保护已启用。",
        )
        rejected["metadata"] = {"product": "CloudPivot", "version": "7"}

        bound, admitted_count, rejection_count, _dropped = (
            _admit_and_bind_expansion_candidates(
                [rejected],
                identity_sources=admitted_seed,
                task_ledger=ledger,
                task_groups=(group,),
                fallback=scope,
                kind="small_document_full",
                relationship="document",
            )
        )

        self.assertEqual(bound, [])
        self.assertEqual(admitted_count, 0)
        self.assertGreaterEqual(rejection_count, 1)
        self.assertIsNone(ledger.lineage_for_candidate(rejected))
        self.assertTrue(ledger.scope_rejections())
        self.assertNotIn("登录保护", repr(ledger.scope_rejection_summary()))

    def test_headless_same_document_expansion_inherits_identity_then_lineage(self):
        """A headerless V6 sibling stays usable after the legal two-step flow."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        scope = ApplicabilityScope(
            product="CloudPivot",
            version="6",
            explicit_version=True,
        )
        seed = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            content="所属产品：CloudPivot；产品版本：6。适用范围说明。",
        )
        seed["metadata"] = {"product": "CloudPivot", "version": "6"}
        ledger, group, admitted_seed = self._scoped_expansion_context(
            scope=scope,
            seed=seed,
        )
        headless = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content="登录保护的配置值为 enabled。",
        )

        bound, admitted_count, rejection_count, dropped_count = (
            _admit_and_bind_expansion_candidates(
                [headless],
                identity_sources=admitted_seed,
                task_ledger=ledger,
                task_groups=(group,),
                fallback=scope,
                kind="small_document_full",
                relationship="document",
            )
        )

        self.assertEqual(admitted_count, 1)
        self.assertEqual(rejection_count, 0)
        self.assertEqual(dropped_count, 0)
        self.assertEqual(len(bound), 1)
        inherited = bound[0]["metadata"]["inherited_document_identity"]
        self.assertEqual(inherited["version"], ["6"])
        self.assertEqual(
            ledger.task_ids_for_candidate(bound[0]),
            ("answer_r1",),
        )

    def test_structural_expansion_cannot_cross_project_seed_scope(self):
        """A same-id structural proposal from another project stays unbound."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        scope = ApplicabilityScope(
            product="CloudPivot",
            version="6",
            project="中青建安",
            explicit_version=True,
            explicit_project=True,
            product_source=ScopeSourceSpan(
                dimension="product",
                start=0,
                end=10,
                span="CloudPivot",
            ),
            version_source=ScopeSourceSpan(
                dimension="version",
                start=11,
                end=12,
                span="6",
            ),
            project_source=ScopeSourceSpan(
                dimension="project",
                start=13,
                end=17,
                span="中青建安",
            ),
        )
        seed = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            content="中青建安项目：CloudPivot 6 登录保护标准。",
        )
        seed["metadata"] = {
            "product": "CloudPivot",
            "version": "6",
            "project": "中青建安",
        }
        ledger, group, admitted_seed = self._scoped_expansion_context(
            scope=scope,
            seed=seed,
        )
        other_project = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content="华东示范项目的配置值为 enabled。",
        )
        other_project["metadata"] = {
            "product": "CloudPivot",
            "version": "6",
            "project": "华东示范项目",
        }
        other_project["expansion_seed_chunk_ids"] = [str(seed["id"])]

        bound, admitted_count, rejection_count, _dropped = (
            _admit_and_bind_expansion_candidates(
                [other_project],
                identity_sources=admitted_seed,
                lineage_sources=admitted_seed,
                task_ledger=ledger,
                task_groups=(group,),
                fallback=scope,
                kind="structural_neighbor",
                relationship="seed",
            )
        )

        self.assertEqual(bound, [])
        self.assertEqual(admitted_count, 0)
        self.assertGreaterEqual(rejection_count, 1)
        self.assertIsNone(ledger.lineage_for_candidate(other_project))

    async def test_supplied_execution_bundle_is_the_only_runner_handoff(self) -> None:
        question = "普通员工的餐补是多少？"
        kb_id = uuid.uuid4()
        bundle = compile_rag_execution_bundle(plan_query_locally(question))
        self.assertTrue(bundle.uses_task_ledger)
        payloads, *_ = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[_candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=0,
                content="普通员工餐补标准为100元/天。",
            )],
            full_document=[],
            execution_bundle=bundle,
        )
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertIn(result["evidence_status"], {"hit", "partial", "unverified", "no_hit"})

    async def test_execution_bundle_activates_the_request_ledger(self) -> None:
        question = "普通员工的餐补是多少？"
        kb_id = uuid.uuid4()
        bundle = compile_rag_execution_bundle(plan_query_locally(question))
        self.assertEqual(bundle.mode, "ledgered")

        await self._run(
            question=question,
            kb_id=kb_id,
            initial=[_candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=0,
                content="普通员工对应D级，餐补标准为100元/天。",
            )],
            full_document=[],
            execution_bundle=bundle,
        )

        query_plan_traces = [
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "query.plan"
        ]
        self.assertEqual(len(query_plan_traces), 1)
        self.assertTrue(query_plan_traces[0]["task_graph_execution"])
        self.assertEqual(
            query_plan_traces[0]["execution_bundle"]["mode"],
            "ledgered",
        )

    async def test_explicit_nonrunnable_bundle_streams_a_closed_clarification_without_retrieval(self) -> None:
        question = "该值取决于前一项"
        kb_id = uuid.uuid4()
        nonrunnable = compile_rag_execution_bundle(plan_query_locally(question))
        self.assertEqual(nonrunnable.mode, "not_ready")
        payloads, _client, search, _fetch, _scoped = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[_candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=0,
                content="不应触发检索。",
            )],
            full_document=[],
            execution_bundle=nonrunnable,
        )
        search.assert_not_awaited()
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertFalse(result["retrieval_executed"])
        clarification = next(
            item for item in payloads if item["type"] == "clarification_state"
        )
        self.assertEqual(clarification["schema_version"], "rag_clarification_state.v1")
        self.assertEqual(clarification["status"], "proposed")
        process = next(item for item in payloads if item["type"] == "search_process")
        self.assertEqual(
            [step["key"] for step in process["steps"]],
            ["analyze", "generate"],
        )
        self.assertEqual(
            [
                item["step"]
                for item in payloads
                if item["type"] == "search_step" and item["status"] == "active"
            ],
            ["analyze", "generate"],
        )

    async def test_task_graph_replay_closes_manager_grade_to_lodging_chain(self) -> None:
        """Replay the log-15 failure without relying on query-array positions."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="总经理的住宿标准是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认总经理对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="总经理",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            # Deliberately unusable legacy projection: the graph must remain
            # the sole execution authority for this supplied API plan.
            retrieval_queries=("不应被执行的旧数组查询",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        self.assertTrue(bundle.uses_task_ledger)
        mapping = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content=(
                "【公司出差管理标准.docx › 二、职级分类】\n"
                "| 职级 | 适用人员 |\n| --- | --- |\n| A级 | 总经理 |"
            ),
            filename="公司出差管理标准.docx",
        )
        lodging = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=6,
            content=(
                "【公司出差管理标准.docx › 四、住宿费用标准】\n"
                "| 职级 | 一线城市（元/天） | 二线城市（元/天） | 其他城市（元/天） |\n"
                "| --- | --- | --- | --- |\n| A级 | ≤1200 | ≤800 | ≤500 |"
            ),
            filename="公司出差管理标准.docx",
        )
        distractor = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=11,
            content="出差结束后5个工作日内提交费用报销申请。",
            filename="公司出差管理标准.docx",
        )

        payloads, client, search, _fetch, _scoped = await self._run(
            question="总经理的住宿标准是多少",
            kb_id=kb_id,
            initial=[mapping, lodging, distractor],
            full_document=[mapping, lodging, distractor],
            execution_bundle=bundle,
            task_contract_override=_task_contract(
                "总经理的住宿标准是多少",
                requirements=[{
                    "role": "answer",
                    "origin": "user_text",
                    "description": "总经理的住宿标准是多少",
                }],
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertIn("≤1200", prompt)
        self.assertIn("≤800", prompt)
        self.assertIn("≤500", prompt)
        self.assertNotIn("5个工作日", prompt)
        executed_queries = [call.args[1] for call in search.await_args_list]
        self.assertNotIn("不应被执行的旧数组查询", executed_queries)
        self.assertTrue(all(
            call.kwargs.get("surface") == "chat_v2_task_graph"
            for call in search.await_args_list
        ))
        trace_events = [
            call.args[0]
            for call in self._last_trace.call_args_list
            if call.args
        ]
        self.assertIn("retrieval.task_query_completed", trace_events)

    async def test_task_graph_executes_every_multi_part_answer_not_only_first_two(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的住宿标准是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工的餐补是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r3",
                description="普通员工的出差补贴是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r4",
                description="确认普通员工对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="普通员工的住宿、餐补和出差补贴分别是多少",
            answer_shape="multi_hop",
            retrieval_queries=("旧投影一", "旧投影二"),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        self.assertTrue(bundle.uses_task_ledger)
        candidates = [
            _candidate(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_index=1,
                content=(
                    "【公司出差管理标准.docx › 二、职级分类】\n"
                    "| 职级 | 适用人员 |\n| --- | --- |\n| D级 | 普通员工 |"
                ),
            ),
            _candidate(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_index=6,
                content=(
                    "【公司出差管理标准.docx › 四、住宿费用标准】\n"
                    "住宿标准：D级一线城市不超过450元/天。"
                ),
            ),
            _candidate(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_index=8,
                content=(
                    "【公司出差管理标准.docx › 五、餐饮补贴标准】\n"
                    "餐饮补贴标准：D级100元/天。"
                ),
            ),
            _candidate(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_index=9,
                content=(
                    "【公司出差管理标准.docx › 六、其他补贴】\n"
                    "出差补贴标准：D级100元/天。"
                ),
            ),
        ]

        payloads, client, search, _fetch, _scoped = await self._run(
            question="普通员工的住宿、餐补和出差补贴分别是多少",
            kb_id=kb_id,
            initial=candidates,
            full_document=candidates,
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertIn("450元/天", prompt)
        self.assertIn("餐饮补贴标准", prompt)
        self.assertIn("出差补贴标准", prompt)
        # The literal user-worded paths are always retrieved.  This fixture
        # deliberately returns every table section for the bridge query, so
        # source-local closure can still avoid redundant materialised Wave 2
        # calls after the fact has been proven.
        executed_queries = [call.args[1] for call in search.await_args_list]
        self.assertEqual(search.await_count, 5)  # anchor + bridge + 3 direct answers
        self.assertTrue(any("适用分类" in query for query in executed_queries))
        self.assertTrue({
            "普通员工的住宿标准是多少",
            "普通员工的餐补是多少",
            "普通员工的出差补贴是多少",
        }.issubset(executed_queries))

    async def test_task_graph_dynamic_bridge_second_hop_keeps_answer_owner(self) -> None:
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="总经理的住宿标准是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认总经理对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="总经理",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            retrieval_queries=("旧数组不可用",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        self.assertTrue(bundle.uses_task_ledger)
        mapping = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content=(
                "【公司出差管理标准.docx › 二、职级分类】\n"
                "| 职级 | 适用人员 |\n| --- | --- |\n| A级 | 总经理 |"
            ),
        )
        lodging = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=6,
            content="住宿标准：A级一线城市不超过1200元/天。",
        )

        payloads, client, search, _fetch, _scoped = await self._run(
            question="总经理的住宿标准是多少",
            kb_id=kb_id,
            initial=[mapping],
            # The anchor and the exactly-equal literal answer query are one
            # physical retrieval group with two logical owners.  The bridge
            # stays separate, and only its source-grounded ``A级`` fact may
            # release the materialised Wave-2 answer query.
            initial_sequence=[[mapping], [mapping], [lodging]],
            full_document=[],
            scoped=[],
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertIn("1200元/天", prompt)
        self.assertIn("A级", search.await_args_list[-1].args[1])
        task_trace = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args
            and call.args[0] == "retrieval.task.completed"
            and call.kwargs.get("wave") == 2
        )
        self.assertEqual(task_trace["task_ids"], ["answer_r1"])
        self.assertEqual(task_trace["parent_task_ids"], ["bridge_r2"])
        self.assertEqual(task_trace["parent_chunk_ids"], [str(mapping["id"])])

    async def test_document_scoped_bridge_supplement_keeps_execution_lineage(
        self,
    ) -> None:
        """A late bridge fact must retain its own task-owned execution id.

        The first static pass may establish only that the document is relevant.
        If the bounded document fallback then finds ``普通员工 -> D级``, bridge
        resolution must use the fallback execution that actually retrieved the
        fact.  Reusing only the empty static bridge execution violates the
        ledger's fact-origin proof and used to surface as a top-level
        ``ValueError`` instead of running the released second-hop query.
        """

        question = "普通员工的住宿标准是多少"
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        bundle = _typed_bridge_execution_bundle(
            question,
            bridge_subject="普通员工",
            bridge_kind="classification",
        )
        mapping = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content=(
                "【公司出差管理标准.docx › 二、职级分类】\n"
                "普通员工对应D级。"
            ),
            filename="公司出差管理标准.docx",
        )
        lodging = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=6,
            content="住宿标准：D级一线城市不超过450元/天。",
            filename="公司出差管理标准.docx",
        )

        async def controlled_search(_db, query, *_args, **_kwargs):
            if "适用分类" in query:
                # The static bridge group misses; the same-document fallback
                # below is the only valid bridge owner for this fact.
                return []
            if "D级" in query:
                return [lodging]
            if query == question:
                # This row is only an anchor/initial-retrieval seed.  It must
                # not be allowed to impersonate the bridge retrieval.
                return [mapping]
            return []

        async def controlled_scoped_search(*_args, **kwargs):
            scoped_query = kwargs["queries"][0]
            if "适用分类" in scoped_query:
                return [mapping]
            return []

        payloads, client, search, fetch_full, scoped_search = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            scoped_side_effect=controlled_scoped_search,
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertEqual(result["evidence_status"], "hit")
        self.assertIn("450元/天", prompt)
        self.assertTrue(any(
            "适用分类" in call.kwargs["queries"][0]
            for call in scoped_search.await_args_list
        ))
        self.assertTrue(any(
            "D级" in call.args[1]
            for call in search.await_args_list
        ))
        fetch_full.assert_awaited_once()
        bridge_trace = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.bridge.resolved"
        )
        self.assertEqual(bridge_trace["status"], "resolved")
        self.assertTrue(bridge_trace["source_execution_ids"])

    async def test_bridge_released_answers_use_resolved_value_and_run_in_parallel(self) -> None:
        """Wave 2 must not start before the bridge is semantic-ready.

        The Event barrier would time out under the historical serial executor:
        its first child request could never observe the other two starts.  It
        also catches a regression where the children use the raw subject rather
        than the resolved ``D级`` value.
        """

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的住宿标准是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工的餐补是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r3",
                description="普通员工的出差补贴是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r4",),
            ),
            AnswerRequirementV2(
                id="r4",
                description="确认普通员工对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="普通员工的住宿、餐补和出差补贴分别是多少",
            answer_shape="multi_hop",
            retrieval_queries=("旧投影不得执行",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        self.assertTrue(bundle.uses_task_ledger)
        mapping = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content=(
                "| 职级 | 适用人员 |\n| --- | --- |\n| D级 | 普通员工 |"
            ),
        )
        lodging = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=6,
            content="住宿标准：D级一线城市不超过450元/天。",
        )
        meal = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=8,
            content="餐饮补贴标准：D级100元/天。",
        )
        travel = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=9,
            content="出差补贴标准：D级100元/天。",
        )
        all_children_started = asyncio.Event()
        dynamic_queries: list[str] = []
        dynamic_session_ids: list[int] = []
        read_sessions: list[SimpleNamespace] = []

        async def controlled_search(_db, query, *_args, **_kwargs):
            if "适用分类" in query:
                return [mapping]
            if query == plan.original_query:
                return [mapping]
            if query in {
                "普通员工的住宿标准是多少",
                "普通员工的餐补是多少",
                "普通员工的出差补贴是多少",
            }:
                # Literal direct paths are valid but this fixture deliberately
                # makes them empty, so only the source-proven bridge can
                # release the D级 materialised retrievals below.
                return []
            if "D级" not in query:
                raise AssertionError(f"unsafe child query: {query}")
            dynamic_queries.append(query)
            dynamic_session_ids.append(id(_db))
            if len(dynamic_queries) == 3:
                all_children_started.set()
            await asyncio.wait_for(all_children_started.wait(), timeout=0.4)
            if "住宿" in query:
                return [lodging]
            if "餐补" in query:
                return [meal]
            if "出差补贴" in query:
                return [travel]
            raise AssertionError(f"unexpected dynamic query: {query}")

        @asynccontextmanager
        async def task_read_session_factory():
            session = SimpleNamespace(rollback_count=0)

            async def rollback():
                session.rollback_count += 1

            session.rollback = rollback
            read_sessions.append(session)
            yield session

        payloads, client, search, _fetch, _scoped = await self._run(
            question=plan.original_query,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=bundle,
            task_read_session_factory=task_read_session_factory,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertEqual(len(dynamic_queries), 3)
        self.assertTrue(all("D级" in query for query in dynamic_queries))
        self.assertEqual(len(set(dynamic_session_ids)), 3)
        self.assertTrue(read_sessions)
        self.assertTrue(all(session.rollback_count == 1 for session in read_sessions))
        self.assertIn("450元/天", prompt)
        self.assertIn("餐饮补贴标准", prompt)
        self.assertIn("出差补贴标准", prompt)
        bridge_resolution_index = next(
            index
            for index, call in enumerate(self._last_trace.call_args_list)
            if call.args and call.args[0] == "retrieval.bridge.resolved"
        )
        wave_two_index = next(
            index
            for index, call in enumerate(self._last_trace.call_args_list)
            if call.args
            and call.args[0] == "retrieval.dag.wave_started"
            and call.kwargs.get("wave") == 2
        )
        self.assertLess(
            bridge_resolution_index,
            wave_two_index,
        )
        executed_queries = [call.args[1] for call in search.await_args_list]
        self.assertTrue({
            "普通员工的住宿标准是多少",
            "普通员工的餐补是多少",
            "普通员工的出差补贴是多少",
        }.issubset(executed_queries))

    async def test_anchor_preflight_snapshot_reused_by_final_task_graph(self) -> None:
        """A V3 preflight cache saves one anchor I/O, not final provenance."""

        question = "普通员工的餐补是多少"
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=3,
            content="普通员工的餐补标准为100元/天。",
        )
        # A snapshot cannot be populated with a raw adapter row.  The explicit
        # flag mirrors `_authorized_candidates` after the preflight boundary.
        candidate["authorized"] = True
        snapshot = AnchorRetrievalSnapshot(
            revision="analysis-revision-1",
            query=question,
            kb_ids=(kb_id,),
            document_ids=None,
            method="hybrid",
            candidate_limit=6,
            candidates=(candidate,),
        )

        payloads, client, search, _fetch, _scoped = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
            execution_bundle=_direct_test_execution_bundle(question),
            anchor_retrieval_snapshot=snapshot,
            anchor_retrieval_revision="analysis-revision-1",
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertEqual(len(client.completions.calls), 1)
        # The graph's direct task is physically coalesced with its anchor, so
        # successful cache reuse means there is no duplicate hybrid call.
        search.assert_not_awaited()
        reuse_events = [
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.anchor_preflight.reused"
        ]
        self.assertEqual(len(reuse_events), 1)
        self.assertEqual(reuse_events[0]["candidate_count"], 1)

    async def test_anchor_preflight_query_mismatch_is_rejected_then_falls_back(self) -> None:
        """A stale cache must never be reinterpreted for a different graph."""

        question = "普通员工的餐补是多少"
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=3,
            content="普通员工的餐补标准为100元/天。",
        )
        candidate["authorized"] = True
        snapshot = AnchorRetrievalSnapshot(
            revision="analysis-revision-2",
            query="普通员工的住宿标准是多少",
            kb_ids=(kb_id,),
            document_ids=None,
            method="hybrid",
            candidate_limit=6,
            candidates=(candidate,),
        )

        payloads, _client, search, _fetch, _scoped = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
            execution_bundle=_direct_test_execution_bundle(question),
            anchor_retrieval_snapshot=snapshot,
            anchor_retrieval_revision="analysis-revision-2",
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(search.await_count, 1)
        rejection_events = [
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.anchor_preflight.rejected"
        ]
        self.assertEqual(len(rejection_events), 1)
        self.assertEqual(rejection_events[0]["reason"], "query_mismatch")

    async def test_anchor_preflight_revision_and_scope_identity_are_strict(self) -> None:
        """Revision and authorized range are cache identity, never hints."""

        kb_id = uuid.uuid4()
        other_kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        snapshot = AnchorRetrievalSnapshot(
            revision="analysis-revision-identity",
            query="普通员工的餐补是多少",
            kb_ids=(kb_id,),
            document_ids=None,
            method="hybrid",
            candidate_limit=6,
        )
        common = {
            "query": "普通员工的餐补是多少",
            "kb_ids": (kb_id,),
            "document_ids": None,
            "method": "hybrid",
            "candidate_limit": 6,
        }
        self.assertEqual(
            snapshot.match_reason(revision="other-revision", **common),
            "revision_mismatch",
        )
        self.assertEqual(
            snapshot.match_reason(
                revision="analysis-revision-identity",
                **{**common, "kb_ids": (other_kb_id,)},
            ),
            "kb_scope_mismatch",
        )
        self.assertEqual(
            snapshot.match_reason(
                revision="analysis-revision-identity",
                **{**common, "document_ids": (doc_id,)},
            ),
            "document_scope_mismatch",
        )

    async def test_anchor_preflight_timeout_falls_back_to_normal_v2_anchor(self) -> None:
        """A timed-out optional preflight must not turn a healthy final run red."""

        question = "普通员工的餐补是多少"
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=3,
            content="普通员工的餐补标准为100元/天。",
        )

        @asynccontextmanager
        async def read_session_factory():
            session = SimpleNamespace()

            async def rollback():
                return None

            session.rollback = rollback
            yield session

        with patch(
            "core.rag_v2.pipeline.hybrid_search",
            new=AsyncMock(side_effect=asyncio.TimeoutError()),
        ):
            snapshot = await retrieve_anchor_retrieval_snapshot(
                db=SimpleNamespace(),
                revision="analysis-revision-3",
                query=question,
                kb_ids=(kb_id,),
                method="hybrid",
                candidate_limit=6,
                timeout_seconds=0.2,
                task_read_session_factory=read_session_factory,
            )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.status, "timeout")
        self.assertFalse(snapshot.reusable)

        payloads, _client, search, _fetch, _scoped = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
            execution_bundle=_direct_test_execution_bundle(question),
            anchor_retrieval_snapshot=snapshot,
            anchor_retrieval_revision="analysis-revision-3",
        )
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(search.await_count, 1)
        rejection_events = [
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.anchor_preflight.rejected"
        ]
        self.assertEqual(rejection_events[0]["reason"], "snapshot_not_ready")

    async def test_anchor_preflight_uses_owned_read_session_not_request_session(self) -> None:
        """Concurrent V3 preflight cannot borrow the request's DB session."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=2,
            content="餐饮补贴标准：D级100元/天。",
        )

        class RequestSession:
            async def execute(self, *_args, **_kwargs):
                raise AssertionError("preflight must not execute on request session")

        read_sessions: list[SimpleNamespace] = []

        @asynccontextmanager
        async def read_session_factory():
            session = SimpleNamespace(rollback_count=0)

            async def rollback():
                session.rollback_count += 1

            session.rollback = rollback
            read_sessions.append(session)
            yield session

        search = AsyncMock(return_value=[candidate])
        with patch("core.rag_v2.pipeline.hybrid_search", new=search):
            snapshot = await retrieve_anchor_retrieval_snapshot(
                db=RequestSession(),
                revision="analysis-revision-4",
                query="普通员工的餐补是多少",
                kb_ids=(kb_id,),
                method="hybrid",
                candidate_limit=6,
                timeout_seconds=0.2,
                task_read_session_factory=read_session_factory,
            )
        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.reusable)
        self.assertEqual(len(snapshot.candidates), 1)
        self.assertIs(search.await_args.args[0], read_sessions[0])
        self.assertEqual(read_sessions[0].rollback_count, 1)
        self.assertTrue(snapshot.candidates[0]["authorized"])

    async def test_registry_read_failure_isolated_from_request_and_baseline_retrieval(self) -> None:
        """Optional registry failure cannot poison a usable RAG request.

        This mirrors a database that has not yet received migration 0032.  The
        registry read must degrade aliases inside its own session, while normal
        static retrieval and the resulting grounded answer continue on fresh
        read sessions.  In particular the request session must receive neither
        a registry query nor a rollback.
        """

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=4,
            content="普通员工的餐补标准为100元/天。",
        )

        class RequestSession:
            def __init__(self):
                self.execute_calls = 0
                self.rollback_calls = 0

            async def execute(self, *_args, **_kwargs):
                self.execute_calls += 1
                raise AssertionError("RAG read must not use request session")

            async def rollback(self):
                self.rollback_calls += 1

        class ReadSession:
            def __init__(self, *, registry_missing: bool):
                self.registry_missing = registry_missing
                self.rollback_calls = 0

            async def execute(self, *_args, **_kwargs):
                if self.registry_missing:
                    raise RuntimeError(
                        "relation terminology_registry_state does not exist"
                    )
                return SimpleNamespace()

            async def rollback(self):
                self.rollback_calls += 1

        request_session = RequestSession()
        read_sessions: list[ReadSession] = []

        @asynccontextmanager
        async def read_session_factory():
            session = ReadSession(registry_missing=not read_sessions)
            read_sessions.append(session)
            yield session

        payloads, client, search, _fetch_full, _scoped = await self._run(
            question="普通员工的餐补是多少",
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
            task_read_session_factory=read_session_factory,
            request_db=request_session,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertEqual(len(client.completions.calls), 1)
        self.assertGreater(search.await_count, 0)
        self.assertEqual(request_session.execute_calls, 0)
        self.assertEqual(request_session.rollback_calls, 0)
        self.assertGreaterEqual(len(read_sessions), 2)
        self.assertEqual(read_sessions[0].rollback_calls, 1)
        self.assertTrue(all(
            session.rollback_calls == 1 for session in read_sessions
        ))

    async def test_bridge_no_fact_or_conflict_skips_only_augmentation(self) -> None:
        """Bridge uncertainty skips Wave 2 but never suppresses direct search."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的餐补是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认普通员工对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="普通员工的餐补是多少",
            answer_shape="fact",
            retrieval_queries=("不得执行的旧检索词",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        self.assertTrue(bundle.uses_task_ledger)
        cases = {
            "no_fact": _candidate(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_index=1,
                content="公司制度适用于全体员工。",
            ),
            "conflict": _candidate(
                kb_id=kb_id,
                doc_id=doc_id,
                chunk_index=2,
                content=(
                    "| 职级 | 适用人员 |\n| --- | --- |\n"
                    "| D级 | 普通员工 |\n| C级 | 普通员工 |"
                ),
            ),
        }
        expected_augmentation_statuses = {
            "no_fact": "skipped_no_fact",
            "conflict": "skipped_conflict",
        }
        for expected_status, bridge_candidate in cases.items():
            with self.subTest(status=expected_status):
                executed_queries: list[str] = []

                async def controlled_search(_db, query, *_args, **_kwargs):
                    executed_queries.append(query)
                    return [bridge_candidate]

                payloads, _client, _search, _fetch, _scoped = await self._run(
                    question=plan.original_query,
                    kb_id=kb_id,
                    initial=[],
                    full_document=[],
                    search_side_effect=controlled_search,
                    execution_bundle=bundle,
                )

                result = next(
                    item for item in payloads if item["type"] == "search_results"
                )
                self.assertNotEqual(result["evidence_status"], "error")
                self.assertIn(plan.requirements[0].description, executed_queries)
                self.assertFalse(any("D级" in query for query in executed_queries))
                completed = next(
                    call.kwargs
                    for call in self._last_trace.call_args_list
                    if call.args and call.args[0] == "retrieval.completed"
                )
                states = completed["task_execution"]["task_states"]
                self.assertEqual(states["bridge_r2"]["bridge_status"], expected_status)
                self.assertEqual(states["answer_r1"]["status"], "succeeded")
                self.assertEqual(
                    states["answer_r1"]["bridge_augmentation_status"],
                    expected_augmentation_statuses[expected_status],
                )
                self.assertEqual(states["answer_r1"]["blocked_by_task_ids"], [])

    async def test_direct_employee_allowance_survives_missing_optional_mapping(self) -> None:
        """A source-level subject/target fact remains answerable without a bridge.

        This is the real class of regression behind ``普通员工的餐补是多少``:
        an inferred classification task may return no fact, but a clause that
        directly states the original subject and answer target must still be
        admitted.  A near-prefix relation such as ``普通员工家属`` must not
        borrow that subject identity.
        """

        kb_id = uuid.uuid4()
        direct_doc_id = uuid.uuid4()
        family_doc_id = uuid.uuid4()
        direct = _candidate(
            kb_id=kb_id,
            doc_id=direct_doc_id,
            chunk_index=0,
            content="普通员工餐补标准为100元/天。",
            filename="员工餐补制度.md",
        )
        family = _candidate(
            kb_id=kb_id,
            doc_id=family_doc_id,
            chunk_index=0,
            content="普通员工家属餐补标准为999元/天。",
            filename="家属福利制度.md",
        )
        executed_queries: list[str] = []

        async def controlled_search(_db, query, *_args, **_kwargs):
            executed_queries.append(query)
            if "适用分类" in query:
                return []
            return [direct, family]

        payloads, client, _search, _fetch, _scoped = await self._run(
            question="普通员工的餐补是多少",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=_typed_bridge_execution_bundle(
                "普通员工的餐补是多少",
                bridge_subject="普通员工",
                bridge_kind="classification",
                edge_mode="augmentation",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(direct_doc_id)},
        )
        self.assertEqual(len(client.completions.calls), 1)
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("100元/天", prompt)
        self.assertNotIn("999元/天", prompt)
        self.assertFalse(any("D级" in query for query in executed_queries))
        completed = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.completed"
        )
        states = completed["task_execution"]["task_states"]
        self.assertEqual(states["bridge_r2"]["bridge_status"], "no_fact")
        self.assertEqual(
            states["answer_r1"]["bridge_augmentation_status"],
            "skipped_no_fact",
        )

    async def test_direct_condition_clause_survives_an_unresolved_inferred_bridge(self) -> None:
        """An implicit taxonomy guess cannot hide a directly applicable rule.

        This reproduces the class of defect behind questions such as
        ``偏远地区出差有什么补贴``: a planner may offer a classification bridge
        as recall enhancement, but the condition clause itself is a complete
        source-grounded answer and must remain usable when no mapping exists.
        """

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="偏远地区出差有什么补贴",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认偏远地区对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="偏远地区",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="偏远地区出差有什么补贴",
            answer_shape="fact",
            retrieval_queries=("旧投影不得执行",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        self.assertTrue(bundle.uses_task_ledger)
        direct_rule = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=9,
            content="偏远地区或艰苦地区出差，可申请额外补贴，标准另行审批。",
        )
        executed_queries: list[str] = []

        async def controlled_search(_db, query, *_args, **_kwargs):
            executed_queries.append(query)
            if "适用分类" in query:
                return []
            return [direct_rule]

        payloads, client, _search, _fetch, _scoped = await self._run(
            question=plan.original_query,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("r1", result["missing_requirement_ids"])
        self.assertIn("可申请额外补贴", prompt)
        self.assertIn(plan.requirements[0].description, executed_queries)
        self.assertFalse(any("D级" in query for query in executed_queries))
        completed = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.completed"
        )
        states = completed["task_execution"]["task_states"]
        self.assertEqual(states["bridge_r2"]["bridge_status"], "no_fact")
        self.assertEqual(states["answer_r1"]["status"], "succeeded")
        self.assertEqual(
            states["answer_r1"]["bridge_augmentation_status"],
            "skipped_no_fact",
        )

    async def test_failed_bridge_does_not_abort_direct_or_independent_answer_branches(self) -> None:
        """A failed optional bridge leaves both direct and sibling paths intact."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="普通员工的餐补是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=("r3",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="请假审批需要哪些审批",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=(),
            ),
            AnswerRequirementV2(
                id="r3",
                description="确认普通员工对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="普通员工餐补和请假审批分别是什么",
            answer_shape="multi_part",
            retrieval_queries=("不得执行的旧检索词",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        self.assertTrue(bundle.uses_task_ledger)
        anchor = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            content="公司制度目录。",
        )
        independent_answer = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=5,
            content="请假审批需要直属上级和部门负责人审批。",
        )
        executed_queries: list[str] = []

        async def controlled_search(_db, query, *_args, **_kwargs):
            executed_queries.append(query)
            if "适用分类" in query:
                raise asyncio.TimeoutError()
            if "请假审批" in query:
                return [independent_answer]
            return [anchor]

        payloads, client, _search, _fetch, _scoped = await self._run(
            question=plan.original_query,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertNotEqual(result["evidence_status"], "error")
        self.assertIn("直属上级", prompt)
        self.assertIn("r1", result["missing_requirement_ids"])
        self.assertNotIn("r2", result["missing_requirement_ids"])
        self.assertFalse(any("D级" in query for query in executed_queries))
        completed = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.completed"
        )
        states = completed["task_execution"]["task_states"]
        self.assertEqual(states["bridge_r3"]["bridge_status"], "failed")
        self.assertEqual(states["answer_r1"]["status"], "succeeded")
        self.assertEqual(
            states["answer_r1"]["bridge_augmentation_status"],
            "skipped_failed",
        )
        self.assertEqual(states["answer_r2"]["status"], "succeeded")

    async def test_unclosed_bridge_scope_does_not_clarify_or_release_answer_wave(
        self,
    ) -> None:
        """Bridge facts alone are prerequisites, not user-facing choices.

        Two version-labelled mapping fragments may be useful diagnostics, but
        neither closes the requested lodging answer.  Clarifying from these
        raw/pre-answer candidates would recreate the old behaviour where the
        system asks the user to choose documents before it has any grounded
        answer alternatives.  The safe terminal state is insufficient
        evidence: no Wave 2 query, no visible answer source and no model call.
        """

        kb_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="总经理的住宿标准是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认总经理对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="总经理",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            retrieval_queries=("不得执行的旧检索词",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        self.assertTrue(bundle.uses_task_ledger)
        version_six = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=1,
            filename="CloudPivot 6 出差标准.md",
            content=(
                "所属产品：CloudPivot；产品版本：6。"
                "职级分类：总经理对应A级。"
            ),
        )
        version_seven = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=1,
            filename="CloudPivot 7 出差标准.md",
            content=(
                "所属产品：CloudPivot；产品版本：7。"
                "职级分类：总经理对应A级。"
            ),
        )
        executed_queries: list[str] = []

        async def controlled_search(_db, query, *_args, **_kwargs):
            executed_queries.append(query)
            if "A级" in query:
                raise AssertionError(
                    "ambiguous static scope must not release Wave 2"
                )
            return [version_six, version_seven]

        payloads, client, _search, _fetch, _scoped = await self._run(
            question=plan.original_query,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["answer_sources"], [])
        self.assertFalse(any(
            item["type"] == "clarification_state" for item in payloads
        ))
        self.assertEqual(len(client.completions.calls), 1)
        # The identical anchor and literal answer query are physically
        # coalesced, but the ledger still retains both logical owners.
        self.assertEqual(len(executed_queries), 2)  # anchor/answer + bridge
        self.assertIn("总经理的住宿标准是多少", executed_queries)
        self.assertTrue(any("适用分类" in query for query in executed_queries))
        self.assertFalse(any("A级" in query for query in executed_queries))
        completed = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.completed"
        )
        bridge_state = completed["task_execution"]["task_states"]["bridge_r2"]
        self.assertEqual(bridge_state["bridge_status"], "resolved")
        self.assertEqual(
            bridge_state["bridge_materialization_status"],
            "blocked_scope_ambiguity",
        )
        self.assertTrue(any(
            call.args
            and call.args[0] == "retrieval.bridge_materialization_blocked"
            for call in self._last_trace.call_args_list
        ))

    async def test_scope_closed_bridge_alternatives_clarify_without_wave_two(
        self,
    ) -> None:
        """Only complete source-local branches may turn scope into a choice.

        The bridge facts are intentionally the same grade.  Their downstream
        lodging clauses differ, proving that equality of an intermediate value
        does not license an unscoped Wave-2 query.  Each document nevertheless
        closes its own bridge+answer route, so final evidence—not raw mappings—
        may request a version clarification.
        """

        kb_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="总经理的住宿标准是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r2",),
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认总经理对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="总经理",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query="总经理的住宿标准是多少",
            answer_shape="multi_hop",
            retrieval_queries=("不得执行的旧检索词",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        version_six = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=1,
            filename="CloudPivot 6 出差标准.md",
            content=(
                "所属产品：CloudPivot；产品版本：6。"
                "职级分类：总经理对应A级。"
                "住宿标准：A级不超过1200元/天。"
            ),
        )
        version_seven = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=1,
            filename="CloudPivot 7 出差标准.md",
            content=(
                "所属产品：CloudPivot；产品版本：7。"
                "职级分类：总经理对应A级。"
                "住宿标准：A级不超过800元/天。"
            ),
        )
        executed_queries: list[str] = []

        async def controlled_search(_db, query, *_args, **_kwargs):
            executed_queries.append(query)
            if "A级" in query:
                raise AssertionError(
                    "scope alternatives must not release a dynamic Wave 2 query"
                )
            return [version_six, version_seven]

        payloads, client, _search, _fetch, _scoped = await self._run(
            question=plan.original_query,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        clarification = next(
            item for item in payloads if item["type"] == "clarification_state"
        )
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertEqual(len(clarification["choices"]), 2)
        self.assertEqual(client.completions.calls, [])
        self.assertEqual(len(executed_queries), 2)
        self.assertFalse(any("A级" in query for query in executed_queries))
        source_docs = {
            doc_id
            for choice in clarification["choices"]
            for doc_id in choice["doc_ids"]
        }
        self.assertEqual(
            source_docs,
            {str(version_six["doc_id"]), str(version_seven["doc_id"])},
        )

    async def test_explicit_scope_admits_one_bridge_path_and_releases_wave_two(
        self,
    ) -> None:
        """An explicit answer/bridge scope selects one safe dynamic route."""

        kb_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="CloudPivot 6 总经理的住宿标准是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r2",),
                scope_product="CloudPivot",
                scope_version="6",
                scope_explicit_version=True,
            ),
            AnswerRequirementV2(
                id="r2",
                description="确认 CloudPivot 6 总经理对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="总经理",
                bridge_kind="classification",
                scope_product="CloudPivot",
                scope_version="6",
                scope_explicit_version=True,
            ),
        )
        plan = QueryPlanV2(
            original_query="CloudPivot 6 总经理的住宿标准是多少",
            answer_shape="multi_hop",
            retrieval_queries=("不得执行的旧检索词",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        version_six_mapping = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=1,
            filename="CloudPivot 6 职级分类.md",
            content=(
                "所属产品：CloudPivot；产品版本：6。"
                "职级分类：总经理对应A级。"
            ),
        )
        version_seven_mapping = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=1,
            filename="CloudPivot 7 职级分类.md",
            content=(
                "所属产品：CloudPivot；产品版本：7。"
                "职级分类：总经理对应A级。"
            ),
        )
        version_six_lodging = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=2,
            filename="CloudPivot 6 住宿标准.md",
            content=(
                "所属产品：CloudPivot；产品版本：6。"
                "住宿标准：A级不超过1200元/天。"
            ),
        )
        dynamic_queries: list[str] = []

        async def controlled_search(_db, query, *_args, **_kwargs):
            if "A级" in query:
                dynamic_queries.append(query)
                return [version_six_lodging]
            return [version_six_mapping, version_seven_mapping]

        payloads, client, _search, _fetch, _scoped = await self._run(
            question=plan.original_query,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertTrue(
            dynamic_queries,
            [
                (call.args, call.kwargs)
                for call in self._last_trace.call_args_list
            ],
        )
        self.assertEqual(
            result["evidence_status"],
            "hit",
            [
                (call.args, call.kwargs)
                for call in self._last_trace.call_args_list
            ],
        )
        self.assertTrue(client.completions.calls, payloads)
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertEqual(len(dynamic_queries), 1)
        self.assertIn("CloudPivot 6", dynamic_queries[0])
        self.assertIn("A级", dynamic_queries[0])
        self.assertNotIn("CloudPivot 7", dynamic_queries[0])
        self.assertIn("1200元/天", prompt)
        self.assertFalse(any(
            item["type"] == "clarification_state" for item in payloads
        ))
        completed = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.completed"
        )
        self.assertEqual(
            completed["task_execution"]["task_states"]["bridge_r2"]
            ["bridge_materialization_status"],
            "eligible",
        )

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
                # The scenario matrix is an execution/evidence integration
                # test, not a hidden planner test.  Declare the two factual
                # classification routes explicitly; every other scenario is
                # a standalone or multi-part answer contract.
                if name in {
                    "ordinary_employee_transport",
                    "contractor_lodging",
                }:
                    execution_bundle = _typed_bridge_execution_bundle(
                        question,
                        bridge_subject=(
                            "普通员工"
                            if name == "ordinary_employee_transport"
                            else "合同工"
                        ),
                        bridge_kind="classification",
                    )
                elif name == "reimbursement_deadline_receipts":
                    execution_bundle = _direct_test_execution_bundle(
                        question,
                        requirements=[
                            {
                                "role": "answer",
                                "description": "报销提交时限是多久",
                            },
                            {
                                "role": "answer",
                                # The semantic contract owns terminology
                                # normalization: the source's governed entity
                                # is ``报销凭证``, not an arbitrary word-order
                                # copy of the user's question.
                                "description": "报销凭证需要提供哪些",
                            },
                        ],
                    )
                else:
                    execution_bundle = _direct_test_execution_bundle(question)
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
                    execution_bundle=execution_bundle,
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

    async def test_scope_selection_hard_limits_same_kb_documents(self) -> None:
        kb_id = uuid.uuid4()
        selected_doc_id = uuid.uuid4()
        unrelated_doc_id = uuid.uuid4()
        selected = _candidate(
            kb_id=kb_id,
            doc_id=selected_doc_id,
            chunk_index=0,
            content="配置参数：mode设置为strict。",
            filename="目标版本.docx",
        )
        unrelated = _candidate(
            kb_id=kb_id,
            doc_id=unrelated_doc_id,
            chunk_index=0,
            content="配置参数：mode设置为legacy。",
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
        self.assertNotIn("mode设置为legacy", "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        ))
        search.assert_not_awaited()
        self.assertGreaterEqual(scoped_search.await_count, 1)

    async def test_scope_selection_skips_model_when_bounded_evidence_is_closed(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        selected_doc_id = uuid.uuid4()
        selected = _candidate(
            kb_id=kb_id,
            doc_id=selected_doc_id,
            chunk_index=0,
            content="验证码有效期配置为 10 分钟。",
            filename="目标配置说明.md",
        )
        scope_filter = {
            "mode": "single",
            "kb_ids": [str(kb_id)],
            "doc_ids": [str(selected_doc_id)],
            "choices": [{
                "key": "c2",
                "label": "《目标配置说明》",
                "products": [],
                "canonical_products": [],
                "versions": [],
                "projects": [],
                "filenames": ["目标配置说明.md"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(selected_doc_id)],
                "anchor_doc_ids": [str(selected_doc_id)],
                "companion_doc_ids": [],
            }],
        }
        adjudicator = AsyncMock()
        settings = SimpleNamespace(
            **vars(_settings()),
            rag_v2_model_evidence_adjudication_enabled=True,
            rag_v2_model_evidence_adjudication_timeout_seconds=1,
        )

        with patch(
            "core.rag_v2.pipeline.joint_rerank_with_coverage",
            new=adjudicator,
        ):
            payloads, _client, _search, _fetch, _scoped = await self._run(
                question="验证码有效期时间是多少",
                kb_id=kb_id,
                initial=[],
                scoped=[selected],
                full_document=[selected],
                evidence_scope_filter=scope_filter,
                settings_override=settings,
            )

        adjudicator.assert_not_awaited()
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit", result)
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(selected_doc_id)},
        )

    def test_model_evidence_adjudication_requires_opt_in(self) -> None:
        enabled = SimpleNamespace(rag_v2_model_evidence_adjudication_enabled=True)

        self.assertTrue(_should_model_adjudicate_evidence(
            settings=enabled,
            search_config={"rerank": True},
        ))
        self.assertFalse(_should_model_adjudicate_evidence(
            settings=enabled,
            search_config={"rerank": False},
        ))

    async def test_process_answer_closes_before_model_adjudication(self) -> None:
        """The evidence model is an enhancer, never a process-answer gate."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        contents = [
            "XX公司员工请假管理办法。第一章总则。",
            "请假类别包括事假、病假和年休假。",
            "审批权限根据请假时长确定。",
            "1天以内由直属主管审批，5天以上由总经理审批。",
            (
                "公司请假流程如下："
                "（一）员工填写请假申请单；"
                "（二）按审批权限逐级提交；"
                "（三）审批通过后交人力资源部备案；"
                "（四）假期结束后办理销假。"
                "突发情况应在返岗后1个工作日内补办，"
                "本办法自2026年1月1日起施行。"
            ),
        ]
        full_document = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=contents,
            filename="员工请假管理办法.docx",
        )
        process_seed = dict(full_document[4])
        process_seed.update(
            score=0.08,
            retrieval_score=0.08,
            vector_score=0.91,
            vector_rank=1,
            active_channels=["vector"],
            candidate_origin="current_retrieval",
            candidate_origins=["current_retrieval"],
        )
        settings = SimpleNamespace(
            **vars(_settings()),
            rag_v2_model_evidence_adjudication_enabled=True,
            rag_v2_model_evidence_adjudication_timeout_seconds=0.01,
        )

        adjudicator = AsyncMock()

        async def never_finishes(*_args, **_kwargs):
            await asyncio.sleep(1)

        adjudicator.side_effect = never_finishes

        with patch(
            "core.rag_v2.pipeline.joint_rerank_with_coverage",
            new=adjudicator,
        ):
            payloads, client, *_ = await self._run(
                question="公司的请假流程是什么",
                kb_id=kb_id,
                initial=[process_seed],
                full_document=full_document,
                settings_override=settings,
            )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit", result)
        self.assertIsNone(result["rerank_succeeded"])
        adjudicator.assert_not_awaited()
        self.assertEqual(
            {item["chunk_index"] for item in result["answer_sources"]},
            {4},
        )
        self.assertEqual(len(client.completions.calls), 1)
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("员工填写请假申请单", prompt)
        self.assertIn("假期结束后办理销假", prompt)

    async def test_complete_document_overview_closes_before_model_adjudication(
        self,
    ) -> None:
        """A bounded DOCX snapshot is deterministic overview evidence.

        Plain DOCX chunks do not always retain section-heading metadata.  The
        complete snapshot cardinality and a current-query document root still
        form a closed document-policy route, so an optional evidence model may
        not become the availability gate for this request class.
        """

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        contents = [
            "XX公司员工请假管理办法。第一章总则。",
            "请假类别包括事假、病假和年休假。",
            "审批权限根据请假时长确定。",
            "1天以内由直属主管审批，5天以上由总经理审批。",
            "请假流程包括申请、逐级审批、人力资源备案和销假。",
        ]
        full_document = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=contents,
            filename="员工请假管理办法.docx",
        )
        initial = []
        for index, source in enumerate(full_document):
            candidate = dict(source)
            candidate.update(
                score=0.03,
                retrieval_score=0.03,
                vector_score=0.86,
                vector_rank=index + 1,
                active_channels=["vector"],
                candidate_origin="current_retrieval",
                candidate_origins=["current_retrieval"],
            )
            initial.append(candidate)
        requirement = AnswerRequirementV2(
            id="a1",
            description="员工请假管理办法",
            role="answer",
            importance="required",
            source="explicit",
            coverage_mode="collection",
            coverage_contract="document_policy",
            depends_on_requirement_ids=(),
            augmentation_requirement_ids=(),
        )
        execution_bundle = compile_rag_execution_bundle(QueryPlanV2(
            original_query="我要查询知识库里面的员工请假管理办法",
            answer_shape="overview",
            retrieval_queries=("我要查询知识库里面的员工请假管理办法",),
            requirements=(requirement,),
            confidence=0.9,
            source="model",
        ))
        settings = SimpleNamespace(
            **vars(_settings()),
            rag_v2_model_evidence_adjudication_enabled=True,
        )
        adjudicator = AsyncMock()

        with patch(
            "core.rag_v2.pipeline.joint_rerank_with_coverage",
            new=adjudicator,
        ):
            payloads, client, *_ = await self._run(
                question="我要查询知识库里面的员工请假管理办法",
                kb_id=kb_id,
                initial=initial,
                full_document=full_document,
                execution_bundle=execution_bundle,
                settings_override=settings,
            )

        adjudicator.assert_not_awaited()
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit", result)
        self.assertIsNone(result["rerank_succeeded"])
        self.assertEqual(len(result["answer_sources"]), len(contents))
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("病假和年休假", prompt)
        self.assertIn("人力资源备案和销假", prompt)

    async def test_scope_slice_selection_drops_sibling_section_in_same_document(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        shared_doc_id = uuid.uuid4()
        selected = _candidate(
            kb_id=kb_id,
            doc_id=shared_doc_id,
            chunk_index=1,
            filename="安全配置多版本.md",
            content=(
                "所属产品：CloudPivot；产品版本：2025。"
                "安全配置：必须启用2025安全模式。"
            ),
        )
        selected["metadata"] = {"section_key": "section-2025"}
        sibling = _candidate(
            kb_id=kb_id,
            doc_id=shared_doc_id,
            chunk_index=0,
            filename="安全配置多版本.md",
            content=(
                "所属产品：CloudPivot；产品版本：2024。"
                "安全配置：必须启用2024旧模式。"
            ),
        )
        sibling["metadata"] = {"section_key": "section-2024"}
        scope_filter = {
            "mode": "single",
            "kb_ids": [str(kb_id)],
            "doc_ids": [str(shared_doc_id)],
            "choices": [{
                "key": "c1",
                "label": "CloudPivot 2025 —《安全配置多版本.md》",
                "products": ["CloudPivot"],
                "canonical_products": ["CloudPivot"],
                "versions": ["2025"],
                "projects": [],
                "filenames": ["安全配置多版本.md"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(shared_doc_id)],
                "anchor_doc_ids": [str(shared_doc_id)],
                "companion_doc_ids": [],
                "scope_slices": [{
                    "kb_id": str(kb_id),
                    "doc_id": str(shared_doc_id),
                    "section_key": "section-2025",
                    "chunk_ids": [str(selected["id"])],
                    "is_anchor": True,
                }],
            }],
        }

        payloads, client, search, _fetch, scoped_search = await self._run(
            question="安全配置是什么",
            kb_id=kb_id,
            initial=[],
            # Simulate adapters returning every section from the selected
            # physical document.  The persisted slice remains the final
            # execution boundary for results, context and answer sources.
            scoped=[selected, sibling],
            full_document=[selected, sibling],
            evidence_scope_filter=scope_filter,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertTrue(result["evidence_scope_anchor_hit"])
        self.assertEqual(
            {item["chunk_id"] for item in result["results"]},
            {str(selected["id"])},
        )
        self.assertEqual(
            {item["chunk_id"] for item in result["answer_sources"]},
            {str(selected["id"])},
        )
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("2025安全模式", prompt)
        self.assertNotIn("2024旧模式", prompt)
        search.assert_not_awaited()
        self.assertGreaterEqual(scoped_search.await_count, 1)

    async def test_same_document_unresolved_answer_scope_requests_refinement(
        self,
    ) -> None:
        """No fake version buttons when one document lacks scoped lineage.

        The headers are only nearby retrieval context.  They do not create a
        source-proven applicability binding for the answer clauses, so the
        final evidence graph may identify two answer routes but must ask for a
        free-text refinement instead of manufacturing selectable versions.
        """
        kb_id = uuid.uuid4()
        shared_doc_id = uuid.uuid4()
        header_2024 = _candidate(
            kb_id=kb_id,
            doc_id=shared_doc_id,
            chunk_index=0,
            filename="旧版多范围配置.md",
            content="所属产品：CloudPivot；产品版本：2024。",
        )
        answer_2024 = _candidate(
            kb_id=kb_id,
            doc_id=shared_doc_id,
            chunk_index=1,
            filename="旧版多范围配置.md",
            content="安全配置：必须启用2024兼容模式。",
        )
        header_2025 = _candidate(
            kb_id=kb_id,
            doc_id=shared_doc_id,
            chunk_index=2,
            filename="旧版多范围配置.md",
            content="所属产品：CloudPivot；产品版本：2025。",
        )
        answer_2025 = _candidate(
            kb_id=kb_id,
            doc_id=shared_doc_id,
            chunk_index=3,
            filename="旧版多范围配置.md",
            content="安全配置：必须启用2025严格模式。",
        )
        legacy_chunks = [
            header_2024,
            answer_2024,
            header_2025,
            answer_2025,
        ]

        first_payloads, first_client, *_ = await self._run(
            question="安全配置是什么",
            kb_id=kb_id,
            initial=legacy_chunks,
            full_document=[],
        )

        result = next(
            item for item in first_payloads if item["type"] == "search_results"
        )
        clarification = next(
            item
            for item in first_payloads
            if item["type"] == "clarification_state"
        )
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(clarification["dimension"], "scope")
        self.assertEqual(clarification["choices"], [])
        self.assertEqual(clarification["selection_mode"], "refine")
        self.assertEqual(first_client.completions.calls, [])

    async def test_single_closed_route_with_product_and_rule_claims_does_not_refine(
        self,
    ) -> None:
        """One source route may carry both scope identity and the answer rule.

        The final graph emits distinct typed assertions for ``CloudPivot`` and
        for the normative configuration rule.  They are complementary
        semantics on one source route, not two competing rules requiring a
        synthetic scope clarification.
        """

        kb_id = uuid.uuid4()
        document_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=document_id,
            chunk_index=0,
            filename="CloudPivot 6 安全配置.md",
            content=(
                "所属产品：CloudPivot；产品版本：6。"
                "安全配置：必须启用6版登录保护。"
            ),
        )
        candidate.update(
            metadata={"product": "CloudPivot", "version": "6"},
            vector_score=0.85,
            vector_rank=1,
            active_channels=["trigram"],
        )

        payloads, client, *_ = await self._run(
            question="CloudPivot 6 的安全配置是什么",
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertIsNone(result["clarification"])
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(document_id)},
        )
        self.assertEqual(len(client.completions.calls), 1)

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
        """A follow-up may reuse a document boundary, never an old fact."""

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
            # The current retrieval must contain a complete target/value
            # assertion.  A prior-turn D级 mapping is only a document anchor
            # and cannot close this new lodging question by itself.
            content="普通员工的住宿标准为：一线城市450元/天。",
            filename="公司出差管理标准.docx",
            score=0.07,
        )

        payloads, client, search, _fetch, scoped_search = await self._run(
            question="那住宿呢",
            standalone_query="普通员工的住宿标准是多少",
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
        # The task graph may normalize the user wording differently across
        # planner versions.  The stable contract is that the answer prompt
        # contains the current turn's complete fact, not a historical lexical
        # paraphrase such as "住宿上限" from a prior source.
        self.assertIn("普通员工的住宿标准为：一线城市450元/天。", prompt)
        self.assertNotIn("上一轮只说明", prompt)
        # The DAG may issue an anchor, literal answer query and optional
        # bridge query.  Their count is an implementation detail; assert the
        # semantic boundary instead: every global query is for the current
        # reconstructed question, and no historical source became evidence.
        self.assertTrue(search.await_args_list)
        self.assertTrue(all(
            "普通员工" in str(call.args[1])
            for call in search.await_args_list
        ))
        self.assertTrue(all(
            call.kwargs.get("surface") == "chat_v2_task_graph"
            for call in search.await_args_list
        ))
        anchor_call = next(
            call
            for call in scoped_search.await_args_list
            if call.kwargs.get("surface") == "chat_v2_carryover"
        )
        # Carryover only constrains the document boundary.  Its anchor query
        # must use the compiled, current-turn standalone question rather than
        # reconstructing an old umbrella question from history.
        self.assertEqual(anchor_call.kwargs["queries"], [
            "普通员工的住宿标准是多少"
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
        self.assertEqual(len(client.completions.calls), 1)
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("服务暂时不可用", system_prompt)
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

    async def test_document_tags_do_not_change_v2_retrieval_order(self) -> None:
        kb_id = uuid.uuid4()
        tagged_doc_id = uuid.uuid4()
        other_doc_id = uuid.uuid4()
        tagged = _candidate(
            kb_id=kb_id,
            doc_id=tagged_doc_id,
            chunk_index=0,
            content="制度事实：处理时限为5个工作日。",
            filename="标签制度.md",
            score=0.08,
        )
        tagged.update(doc_tags=["重点"], keyword_score=0.1, vector_score=0.9)
        other = _candidate(
            kb_id=kb_id,
            doc_id=other_doc_id,
            chunk_index=0,
            content="制度事实：处理时限为3个工作日。",
            filename="普通制度.md",
            score=0.1,
        )
        other.update(keyword_score=0.1, vector_score=0.9)

        payloads, _client, _search, _fetch, _scoped = await self._run(
            question="制度事实是什么",
            kb_id=kb_id,
            initial=[other, tagged],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["results"][0]["doc_id"], str(other_doc_id))
        # 文档标签可供管理和约束识别使用，但不再作为用户偏好参与检索排序。
        self.assertEqual(
            {item["doc_id"] for item in result["results"]},
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
        # The only fresh same-document row was retrieved and authorised, but
        # failed the current-turn relevance gate.  It therefore never obtains
        # task lineage, cannot enter expansion/context/answer evidence, and
        # the correct terminal state is a true no-hit rather than an
        # "insufficient evidence" claim about evidence we did not admit.
        self.assertEqual(result["evidence_status"], "no_hit")
        self.assertFalse(result["carryover_seed_used"])
        self.assertFalse(result["carryover_anchor_succeeded"])
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(result["results"], [])
        self.assertEqual(len(client.completions.calls), 1)
        admission_trace = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args
            and call.args[0] == "retrieval.carryover_anchor_admission"
        )
        self.assertEqual(admission_trace["raw_candidate_count"], 1)
        self.assertEqual(admission_trace["scope_admitted_candidate_count"], 1)
        self.assertEqual(admission_trace["admitted_candidate_count"], 0)
        self.assertEqual(
            admission_trace["relevance_reason"],
            "no_document_met_lexical_or_vector_gate",
        )
        self.assertEqual(
            admission_trace["reason"],
            "carryover_anchor_below_relevance_gate",
        )
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("知识库中未找到相关内容", system_prompt)
        self.assertNotIn("一线城市450元", answer)

    async def test_document_relevance_gate_excludes_lower_vector_noise(self) -> None:
        kb_id = uuid.uuid4()
        target_doc_id = uuid.uuid4()
        noise_doc_id = uuid.uuid4()
        target = _candidate(
            kb_id=kb_id,
            doc_id=target_doc_id,
            chunk_index=0,
            content="目标制度标准：处理时限为5天。",
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

    async def test_final_context_budget_rejects_unclosed_serialized_sources(
        self,
    ) -> None:
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
            # A narrow fact path otherwise intentionally restricts its
            # initial evidence to the user-facing top-k.  Exercise the
            # renderer budget with more independently closed candidates.
            top_k=16,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        generation_context = next(
            call
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "generation.context"
        )
        serialized_context = generation_context.kwargs["context"]
        # These 16 opaque fragments cannot prove a request for the *complete*
        # policy.  When their serialised size exceeds the context budget, the
        # final visible-evidence pass must reject the incomplete route rather
        # than select a subset and let the model imply that it saw the full
        # policy.  Terminal traces intentionally have no hidden
        # ``all_context_sources`` field: ``context_sources`` is the sole
        # model-visible source contract and stays empty while the answer model
        # expresses the structured insufficient-evidence state.
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(result["answer_source_count"], 0)
        self.assertEqual(serialized_context, "")
        self.assertEqual(generation_context.kwargs["context_sources"], [])
        self.assertEqual(generation_context.kwargs["model"], "test-chat")
        self.assertEqual(len(client.completions.calls), 1)
        self.assertIn(
            "context_budget_limited",
            result["evidence_state"]["reasons"],
        )
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("无法组成可核验的完整答案链", system_prompt)

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
        self.assertEqual(len(client.completions.calls), 1)
        self.assertIn(
            "知识库中未找到相关内容",
            client.completions.calls[0]["messages"][0]["content"],
        )
        self.assertEqual(
            client.completions.calls[0]["messages"][-1]["content"],
            "完全不同的主题",
        )
        generation_context = next(
            call
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "generation.context"
        )
        self.assertEqual(generation_context.kwargs["model"], "test-chat")
        self.assertEqual(generation_context.kwargs["context"], "")
        fetch_full.assert_not_awaited()

    async def test_unknown_policy_subject_keeps_recall_but_never_promotes_answer_sources(
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
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(
            item["evidence_role"] == "related" for item in result["results"]
        ))
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(len(client.completions.calls), 1)
        self.assertFalse(any(
            item["type"] == "clarification_state" for item in payloads
        ))
        # High-recall admission may inspect the admitted documents, but no
        # candidate becomes an answer source without evidence closure.
        self.assertEqual(search.await_count, 1)
        fetch_full.assert_awaited()
        scoped.assert_awaited()

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

        self.assertEqual(len(client.completions.calls), 1)
        user_prompt = client.completions.calls[0]["messages"][-1]["content"]
        self.assertEqual(user_prompt, "那住宿呢")
        self.assertNotIn("普通员工的出差标准是什么", user_prompt)

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
                content=(
                    f"所属产品：CloudPivot；产品版本：{version}。"
                    f"安全配置：必须启用{version}版登录保护。"
                ),
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
        """A timed-out optional expansion cannot erase a closed direct claim."""

        kb_id = uuid.uuid4()
        initial = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=4,
            content="差旅通讯补贴标准为50元/天。",
        )

        payloads, _client, *_ = await self._run(
            question="差旅通讯补贴标准是多少",
            kb_id=kb_id,
            initial=[initial],
            full_document=[],
            blocking_full_document=True,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
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
            content="明确条款的处理时限为5天。",
        )
        candidate.update(
            trigram_rank=1,
            trigram_score=0.12,
            active_channels=["trigram"],
        )

        payloads, _client, *_ = await self._run(
            question="明确条款的处理时限是多少",
            kb_id=kb_id,
            initial=[candidate],
            full_document=[],
            vector_channel_failed=True,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
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
        self.assertEqual(len(client.completions.calls), 1)

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

    async def test_overview_incomplete_snapshot_never_generates_from_partial_policy(self) -> None:
        """A rooted policy title cannot turn five of six chunks into an overview."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        full = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=[
                "公司制度",
                "一、总则：规范管理。",
                "二、分类：普通岗位对应D级。",
                "三、交通：D级乘坐经济舱和高铁二等座。",
                "四、住宿：D级上限450元/天。",
                "五、餐饮：D级补贴100元/天。",
            ],
            filename="公司管理标准.docx",
        )
        initial = [dict(full[0]), dict(full[2])]
        for index, candidate in enumerate(initial, start=1):
            candidate.update(
                score=0.1 - index * 0.01,
                retrieval_score=0.1 - index * 0.01,
                vector_score=0.88 - index * 0.01,
                vector_rank=index,
                active_channels=["vector"],
            )

        payloads, client, *_ = await self._run(
            question="普通岗位的管理标准是什么",
            kb_id=kb_id,
            initial=initial,
            # The retriever's declared cardinality remains six, but only five
            # source chunks arrive.  The final graph must reject it instead of
            # allowing a plausible-looking, silently incomplete overview.
            full_document=full[:-1],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(len(client.completions.calls), 1)
        self.assertIn(
            "collection_snapshot_unproven",
            result["evidence_state"]["reasons"],
        )

    async def test_overview_snapshot_is_atomic_under_context_budget(self) -> None:
        """An over-budget policy is unavailable, never cropped into a false overview."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        contents = [
            "公司制度",
            "一、分类：普通岗位对应D级。",
            *[
                f"第{index}章 D级管理规则：第{index}项。"
                for index in range(2, 17)
            ],
        ]
        full = _full_document(
            kb_id=kb_id,
            doc_id=doc_id,
            contents=contents,
            filename="公司管理标准.docx",
        )
        initial = [dict(full[0]), dict(full[1])]
        for index, candidate in enumerate(initial, start=1):
            candidate.update(
                score=0.1 - index * 0.01,
                retrieval_score=0.1 - index * 0.01,
                vector_score=0.88 - index * 0.01,
                vector_rank=index,
                active_channels=["vector"],
            )

        payloads, client, *_ = await self._run(
            question="普通岗位的管理标准是什么",
            kb_id=kb_id,
            initial=initial,
            full_document=full,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(len(client.completions.calls), 1)
        self.assertIn(
            "context_budget_limited",
            result["evidence_state"]["reasons"],
        )

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
        answer = dict(full[1])
        answer.update(
            score=0.09,
            retrieval_score=0.09,
            vector_score=0.86,
            vector_rank=1,
            active_channels=["vector"],
        )
        mapping = dict(full[0])
        mapping.update(
            score=0.08,
            retrieval_score=0.08,
            vector_score=0.84,
            vector_rank=2,
            active_channels=["vector"],
        )

        payloads, client, *_ = await self._run(
            question="普通岗位的餐饮补贴是多少",
            kb_id=kb_id,
            initial=[],
            initial_sequence=[[answer], [mapping], [answer]],
            full_document=[],
            execution_bundle=_typed_bridge_execution_bundle(
                "普通岗位的餐饮补贴是多少",
                bridge_subject="普通岗位",
                bridge_kind="classification",
                bridge_description="确认普通岗位对应的职级",
            ),
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
        # Bridge facts remain visible provenance, but only answer
        # requirements participate in final answer-coverage accounting.
        self.assertEqual(result["covered_requirement_ids"], ["r1"])
        self.assertEqual(result["missing_requirement_ids"], [])
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("普通岗位对应D级", prompt)
        self.assertIn("D级为100元/天", prompt)
        self.assertIn('"answer_shape": "multi_hop"', prompt)

    async def test_bridge_first_dynamic_second_hop_cross_domain_matrix(self) -> None:
        cases = (
            {
                "name": "manager_to_grade_to_lodging",
                "question": "总经理的住宿标准是多少",
                "bridge_subject": "总经理",
                "bridge_requirement": (
                    "确认总经理对应的适用分类、等级、类别或阶段"
                    "（用于确定住宿标准）"
                ),
                "bridge_content": "职级分类：总经理对应A级。",
                "resolved_value": "A级",
                "target_term": "住宿",
                "answer_content": "住宿标准：A级一线城市不超过1200元/天。",
                "answer_marker": "1200元/天",
            },
            {
                "name": "product_to_tier_to_permission",
                "question": "星云产品的数据导出权限是什么",
                "bridge_subject": "星云产品",
                "bridge_requirement": "确认星云产品对应的产品级别",
                "bridge_content": "产品目录：星云产品属于企业版。",
                "resolved_value": "企业版",
                "target_term": "导出权限",
                "answer_content": "数据导出权限：企业版允许导出业务数据。",
                "answer_marker": "允许导出业务数据",
            },
            {
                "name": "supplier_to_risk_to_disposition",
                "question": "供应商甲的风险处置措施是什么",
                "bridge_subject": "供应商甲",
                "bridge_requirement": "确认供应商甲对应的风险等级",
                "bridge_content": "风险评估：供应商甲认定为高风险。",
                "resolved_value": "高风险",
                "target_term": "处置",
                "answer_content": "风险处置措施：高风险供应商暂停准入并启动复核。",
                "answer_marker": "暂停准入并启动复核",
            },
        )

        for case in cases:
            with self.subTest(scenario=case["name"]):
                kb_id = uuid.uuid4()
                bridge_doc_id = uuid.uuid4()
                answer_doc_id = uuid.uuid4()
                bridge = _candidate(
                    kb_id=kb_id,
                    doc_id=bridge_doc_id,
                    chunk_index=0,
                    filename=f"{case['name']}-mapping.md",
                    content=case["bridge_content"],
                )
                answer = _candidate(
                    kb_id=kb_id,
                    doc_id=answer_doc_id,
                    chunk_index=0,
                    filename=f"{case['name']}-policy.md",
                    content=case["answer_content"],
                )
                unauthorized = _candidate(
                    kb_id=uuid.uuid4(),
                    doc_id=uuid.uuid4(),
                    chunk_index=0,
                    filename="unauthorized-policy.md",
                    content=(
                        f"{case['resolved_value']} {case['target_term']}："
                        "禁止纳入的未授权答案。"
                    ),
                )

                async def search_effect(_db, retrieval_query, *_args, **_kwargs):
                    if (
                        case["resolved_value"] in retrieval_query
                        and case["target_term"] in retrieval_query
                    ):
                        return [answer, unauthorized]
                    return [bridge]

                payloads, client, search, *_ = await self._run(
                    question=case["question"],
                    kb_id=kb_id,
                    initial=[],
                    full_document=[],
                    search_side_effect=search_effect,
                    execution_bundle=_typed_bridge_execution_bundle(
                        case["question"],
                        bridge_subject=case["bridge_subject"],
                        bridge_kind="classification",
                        bridge_description=case["bridge_requirement"],
                    ),
                )

                result = next(
                    item for item in payloads if item["type"] == "search_results"
                )
                self.assertEqual(result["evidence_status"], "hit")
                self.assertEqual(result["evidence_completeness"], "complete")
                self.assertEqual(result["missing_requirement_ids"], [])
                self.assertEqual(
                    {item["doc_id"] for item in result["answer_sources"]},
                    {str(bridge_doc_id), str(answer_doc_id)},
                )
                bridge_query_calls = [
                    call
                    for call in search.await_args_list
                    if call.kwargs.get("surface") == "chat_v2_task_graph_bridge"
                ]
                self.assertEqual(len(bridge_query_calls), 1)
                bridge_query_call = bridge_query_calls[0]
                resolved_query = bridge_query_call.args[1]
                self.assertIn(case["resolved_value"], resolved_query)
                self.assertIn(case["target_term"], resolved_query)
                self.assertEqual(
                    bridge_query_call.kwargs["surface"],
                    "chat_v2_task_graph_bridge",
                )
                answer_source = next(
                    item
                    for item in result["answer_sources"]
                    if item["doc_id"] == str(answer_doc_id)
                )
                self.assertIn(
                    "task_graph_bridge_answer",
                    answer_source["candidate_origins"],
                )
                wave_two = next(
                    call.kwargs
                    for call in self._last_trace.call_args_list
                    if call.args
                    and call.args[0] == "retrieval.task.completed"
                    and call.kwargs.get("wave") == 2
                )
                self.assertEqual(wave_two["task_ids"], ["answer_r1"])
                self.assertEqual(wave_two["parent_task_ids"], ["bridge_r2"])
                prompt = "\n".join(
                    message["content"]
                    for message in client.completions.calls[0]["messages"]
                )
                self.assertIn(case["bridge_content"], prompt)
                self.assertIn(case["answer_marker"], prompt)
                self.assertNotIn("禁止纳入的未授权答案", prompt)
                self.assertFalse(any(
                    item["type"] == "clarification_state"
                    for item in payloads
                ))

    async def test_bridge_second_hop_stays_inside_selected_documents(self) -> None:
        kb_id = uuid.uuid4()
        selected_doc_id = uuid.uuid4()
        outside_doc_id = uuid.uuid4()
        bridge = _candidate(
            kb_id=kb_id,
            doc_id=selected_doc_id,
            chunk_index=0,
            filename="已选择的职级与住宿制度.md",
            content="职级分类：总经理对应A级。",
        )
        answer = _candidate(
            kb_id=kb_id,
            doc_id=selected_doc_id,
            chunk_index=1,
            filename="已选择的职级与住宿制度.md",
            content="住宿标准：A级一线城市不超过1200元/天。",
        )
        outside = _candidate(
            kb_id=kb_id,
            doc_id=outside_doc_id,
            chunk_index=0,
            filename="范围外制度.md",
            content="住宿标准：A级一线城市不超过9999元/天。",
        )
        scope_filter = {
            "mode": "single",
            "kb_ids": [str(kb_id)],
            "doc_ids": [str(selected_doc_id)],
            "choices": [{
                "key": "c1",
                "label": "已选择的职级与住宿制度",
                "products": [],
                "canonical_products": [],
                "versions": [],
                "projects": [],
                "filenames": ["已选择的职级与住宿制度.md"],
                "kb_ids": [str(kb_id)],
                "doc_ids": [str(selected_doc_id)],
                "anchor_doc_ids": [str(selected_doc_id)],
                "companion_doc_ids": [],
            }],
        }

        async def scoped_effect(_db, queries, *_args, **_kwargs):
            query = queries[0]
            if "A级" in query and "住宿" in query:
                # Deliberately return a range-escaping row as well: the
                # production range gate must remove it after retrieval.
                return [answer, outside]
            return [bridge]

        payloads, client, global_search, _fetch, scoped_search = await self._run(
            question="总经理的住宿标准是多少",
            kb_id=kb_id,
            initial=[],
            scoped=[],
            scoped_side_effect=scoped_effect,
            full_document=[],
            evidence_scope_filter=scope_filter,
            execution_bundle=_typed_bridge_execution_bundle(
                "总经理的住宿标准是多少",
                bridge_subject="总经理",
                bridge_kind="classification",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(selected_doc_id)},
        )
        bridge_query_calls = [
            call
            for call in scoped_search.await_args_list
            if call.kwargs.get("surface") == "chat_v2_task_graph_bridge_scope"
        ]
        self.assertEqual(len(bridge_query_calls), 1)
        dynamic_call = bridge_query_calls[0]
        self.assertEqual(
            dynamic_call.kwargs["surface"],
            "chat_v2_task_graph_bridge_scope",
        )
        self.assertEqual(
            dynamic_call.kwargs["doc_ids"],
            [selected_doc_id],
        )
        self.assertIn("A级", dynamic_call.kwargs["queries"][0])
        global_search.assert_not_awaited()
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("1200元/天", prompt)
        self.assertNotIn("9999元/天", prompt)

    async def test_bridge_second_hop_inherits_product_version_and_project_scope(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        question = "中青建安的云枢8.2.75普通员工餐补标准是多少"
        bridge = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="中青建安云枢8.2.75职级.md",
            content="普通员工对应D级。",
        )
        bridge["metadata"] = {
            "product": "云枢",
            "version": "8.2.75",
            "project": "中青建安",
        }
        correct = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="中青建安云枢8.2.75餐补.md",
            content="餐补标准：D级为100元/天。",
        )
        correct["metadata"] = {
            "product": "云枢",
            "version": "8.2.75",
            "project": "中青建安",
        }
        wrong_version = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="中青建安云枢7餐补.md",
            content="餐补标准：D级为700元/天。",
        )
        wrong_version["metadata"] = {
            "product": "云枢",
            "version": "7",
            "project": "中青建安",
        }
        wrong_project = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="华东示范项目云枢8.2.75餐补.md",
            content="餐补标准：D级为900元/天。",
        )
        wrong_project["metadata"] = {
            "product": "云枢",
            "version": "8.2.75",
            "project": "华东示范项目",
        }

        async def search_effect(_db, retrieval_query, *_args, **_kwargs):
            if "D级" in retrieval_query and "餐补" in retrieval_query:
                return [correct, wrong_version, wrong_project]
            return [bridge]

        payloads, client, search, *_ = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=search_effect,
            execution_bundle=_typed_bridge_execution_bundle(
                question,
                bridge_subject="普通员工",
                bridge_kind="classification",
                bridge_description="确认普通员工对应的职级",
                scope_product="云枢",
                scope_version="8.2.75",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["missing_requirement_ids"], [])
        bridge_query_calls = [
            call
            for call in search.await_args_list
            if call.kwargs.get("surface") == "chat_v2_task_graph_bridge"
        ]
        self.assertEqual(len(bridge_query_calls), 1)
        resolved_query = bridge_query_calls[0].args[1]
        for marker in ("中青建安", "云枢8.2.75", "D级", "餐补"):
            self.assertIn(marker, resolved_query)
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("100元/天", prompt)
        self.assertNotIn("700元/天", prompt)
        self.assertNotIn("900元/天", prompt)
        displayed_doc_ids = {item["doc_id"] for item in result["results"]}
        self.assertNotIn(str(wrong_version["doc_id"]), displayed_doc_ids)
        self.assertNotIn(str(wrong_project["doc_id"]), displayed_doc_ids)

    async def test_single_hop_does_not_trigger_dynamic_bridge_search(self) -> None:
        kb_id = uuid.uuid4()
        answer = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="采购审批制度.md",
            content="采购申请单笔审批额度不超过5000元。",
        )

        payloads, client, search, *_ = await self._run(
            question="采购申请单笔审批额度是多少",
            kb_id=kb_id,
            initial=[answer],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(search.await_count, 1)
        self.assertGreaterEqual(len(result["answer_sources"]), 1)
        self.assertFalse(any(
            "resolved_bridge_query" in item["candidate_origins"]
            for item in result["results"]
        ))
        self.assertFalse(any(
            call.args and call.args[0] == "retrieval.bridge_expansion_planned"
            for call in self._last_trace.call_args_list
        ))
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("5000元", prompt)

    async def test_bridge_second_hop_timeout_is_bounded_partial(self) -> None:
        kb_id = uuid.uuid4()
        bridge_doc_id = uuid.uuid4()
        bridge = _candidate(
            kb_id=kb_id,
            doc_id=bridge_doc_id,
            chunk_index=0,
            filename="员工职级分类.md",
            content="职级分类：总经理对应A级。",
        )

        async def search_effect(_db, retrieval_query, *_args, **_kwargs):
            if "A级" in retrieval_query and "住宿" in retrieval_query:
                raise TimeoutError("resolved bridge query timed out")
            return [bridge]

        payloads, client, search, *_ = await self._run(
            question="总经理的住宿标准是多少",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=search_effect,
            execution_bundle=_typed_bridge_execution_bundle(
                "总经理的住宿标准是多少",
                bridge_subject="总经理",
                bridge_kind="classification",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(
            sum(
                call.kwargs.get("surface") == "chat_v2_task_graph_bridge"
                for call in search.await_args_list
            ),
            1,
        )
        # A proof-edge timeout leaves the first-hop mapping diagnosable, but
        # it cannot be promoted into answer context or generation evidence.
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(result["evidence_completeness"], "partial")
        self.assertEqual(result["missing_requirement_ids"], ["r1"])
        self.assertEqual(
            {item["doc_id"] for item in result["results"]},
            {str(bridge_doc_id)},
        )
        self.assertEqual(result["answer_sources"], [])
        completed = next(
            call
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.completed"
        )
        self.assertEqual(completed.kwargs["bridge_query_planned_count"], 1)
        self.assertEqual(completed.kwargs["bridge_query_attempted_count"], 1)
        self.assertEqual(completed.kwargs["bridge_query_succeeded_count"], 0)
        self.assertEqual(completed.kwargs["bridge_query_candidate_count"], 0)
        self.assertEqual(len(client.completions.calls), 1)

    async def test_bridge_spec_materialization_failure_preserves_first_hop_diagnostic(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        bridge_doc_id = uuid.uuid4()
        bridge = _candidate(
            kb_id=kb_id,
            doc_id=bridge_doc_id,
            chunk_index=0,
            filename="员工职级分类.md",
            content="职级分类：普通员工对应D级。",
        )

        with patch(
            "core.rag_v2.pipeline.build_bridge_expansion_specs_from_facts",
            side_effect=RuntimeError("bridge resolver defect"),
        ):
            payloads, client, search, *_ = await self._run(
                question="普通员工的餐补标准是多少",
                kb_id=kb_id,
                initial=[bridge],
                full_document=[],
                execution_bundle=_typed_bridge_execution_bundle(
                    "普通员工的餐补标准是多少",
                    bridge_subject="普通员工",
                    bridge_kind="classification",
                    bridge_description="确认普通员工对应的职级",
                ),
            )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(
            result["decision_reason"],
            "rag_v2_related_evidence_unclosed",
        )
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(
            {item["doc_id"] for item in result["results"]},
            {str(bridge_doc_id)},
        )
        self.assertEqual(result["answer_sources"], [])
        self.assertFalse(any(
            call.kwargs.get("surface") in {
                "chat_v2_task_graph_bridge",
                "chat_v2_task_graph_bridge_scope",
            }
            for call in search.await_args_list
        ))
        self.assertEqual(len(client.completions.calls), 1)
        bridge_error = next(
            call
            for call in self._last_trace.call_args_list
            if call.args
            and call.args[0] == "retrieval.bridge_spec_materialization_error"
        )
        self.assertEqual(
            bridge_error.kwargs["released_proof_answer_task_ids"],
            ["answer_r1"],
        )
        self.assertFalse(any(
            call.args and call.args[0] == "retrieval.error"
            for call in self._last_trace.call_args_list
        ))

    async def test_proof_bridge_joins_different_authorized_documents(
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
                [answer],
            ],
            full_document=[],
            execution_bundle=_typed_bridge_execution_bundle(
                "普通员工的餐补标准是多少",
                bridge_subject="普通员工",
                bridge_kind="classification",
                bridge_description="确认普通员工对应的职级",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["evidence_completeness"], "complete")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(answer_doc_id), str(bridge_doc_id)},
        )
        bridge_source = next(
            item for item in result["answer_sources"]
            if item["doc_id"] == str(bridge_doc_id)
        )
        self.assertEqual(bridge_source["supports_requirement_ids"], ["r2"])
        dynamic_call = next(
            call
            for call in search.await_args_list
            if call.kwargs.get("surface") == "chat_v2_task_graph_bridge"
        )
        self.assertIn("D级", dynamic_call.args[1])
        dynamic_trace = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args
            and call.args[0] == "retrieval.task.completed"
            and call.kwargs.get("wave") == 2
        )
        self.assertEqual(dynamic_trace["task_ids"], ["answer_r1"])
        self.assertEqual(dynamic_trace["parent_task_ids"], ["bridge_r2"])
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("D级餐补标准为100元", prompt)
        self.assertIn("普通员工对应D级", prompt)
        self.assertNotIn("普通员工对应A级", prompt)
        self.assertNotIn("访客停车", prompt)

    async def test_competing_cross_document_answer_graphs_clarify_with_companions(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        grade_a_doc = uuid.uuid4()
        grade_d_doc = uuid.uuid4()
        policy_a_doc = uuid.uuid4()
        policy_d_doc = uuid.uuid4()
        answer_a = _candidate(
            kb_id=kb_id,
            doc_id=policy_a_doc,
            chunk_index=0,
            filename="管理岗住宿标准.md",
            content="住宿标准：A级为1200元/天。",
        )
        answer_d = _candidate(
            kb_id=kb_id,
            doc_id=policy_d_doc,
            chunk_index=0,
            filename="员工住宿标准.md",
            content="住宿标准：D级为450元/天。",
        )
        map_a = _candidate(
            kb_id=kb_id,
            doc_id=grade_a_doc,
            chunk_index=0,
            filename="管理岗分类.md",
            content="普通员工对应A级。",
        )
        map_d = _candidate(
            kb_id=kb_id,
            doc_id=grade_d_doc,
            chunk_index=0,
            filename="员工职级分类.md",
            content="普通员工对应D级。",
        )

        async def search_effect(_db, query, *_args, **_kwargs):
            if "A级" in query and "住宿" in query:
                return [answer_a]
            if "D级" in query and "住宿" in query:
                return [answer_d]
            if "适用分类" in query:
                return [map_a, map_d]
            return [answer_a, answer_d]

        payloads, client, *_ = await self._run(
            question="普通员工的住宿标准是多少",
            kb_id=kb_id,
            initial=[],
            search_side_effect=search_effect,
            full_document=[],
            execution_bundle=_typed_bridge_execution_bundle(
                "普通员工的住宿标准是多少",
                bridge_subject="普通员工",
                bridge_kind="classification",
                bridge_description="确认普通员工对应的职级",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        clarification = next(
            item for item in payloads if item["type"] == "clarification_state"
        )
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertEqual(clarification["dimension"], "document")
        self.assertEqual(len(clarification["choices"]), 2)
        by_anchor = {
            choice["anchor_doc_ids"][0]: choice
            for choice in clarification["choices"]
        }
        self.assertEqual(
            by_anchor[str(policy_a_doc)]["companion_doc_ids"],
            [str(grade_a_doc)],
        )
        self.assertEqual(
            by_anchor[str(policy_d_doc)]["companion_doc_ids"],
            [str(grade_d_doc)],
        )
        self.assertEqual(
            set(by_anchor[str(policy_a_doc)]["doc_ids"]),
            {str(policy_a_doc), str(grade_a_doc)},
        )
        self.assertEqual(client.completions.calls, [])

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
        # The surviving D-grade clause does not prove that the user-supplied
        # subject belongs to D grade.  A timed-out bridge query may degrade
        # availability, but it must never expose the unjoined amount as an
        # answer source or invoke generation with it.
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(result["answer_sources"], [])
        # Retrieval no longer executes a positional global query plan.  The
        # failed bridge is represented by its typed task lifecycle, which
        # keeps the surviving amount candidate from being mistaken for a
        # joined answer.
        self.assertIn(
            "task_query_retrieval_timeout",
            result["evidence_state"]["reasons"],
        )
        self.assertIn(
            "bridge_retrieval_failed",
            result["evidence_state"]["reasons"],
        )
        self.assertNotIn(
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

    async def test_task_graph_explicit_version_isolated_across_direct_and_bridge_paths(
        self,
    ) -> None:
        """Physical DAG fan-out is not a semantic regression surface.

        The contract is that every direct/bridge route stays in the requested
        8.6 applicability range.  The scheduler may add an anchor or a
        bridge-materialised second hop, so a fixed HTTP-call count would make
        this test reject a correct implementation merely for changing its
        execution plan.
        """
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

        executed_queries: list[str] = []

        async def scoped_search(_db, query, *_args, **_kwargs):
            executed_queries.append(query)
            if "对应的适用分类" in query:
                # The wrong row is intentionally returned by the adapter so
                # task-local scope admission must reject it before ledger,
                # context, results, or generation can see it.
                return [bridge, wrong_version]
            return [answer]

        payloads, client, search, *_ = await self._run(
            question="云枢8.6普通员工的餐补标准是多少",
            kb_id=kb_id,
            initial=[],
            search_side_effect=scoped_search,
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertTrue(any(
            "对应的适用分类" in query for query in executed_queries
        ))
        self.assertTrue(any(
            "D级" in query and "8.6" in query for query in executed_queries
        ))
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
        completed_scope_admissions = [
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.task_query_completed"
            and call.kwargs.get("task_ids") == ["bridge_r2"]
        ]
        self.assertTrue(completed_scope_admissions)
        self.assertGreaterEqual(
            completed_scope_admissions[0]["scope_rejection_count"],
            1,
        )

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
        """Expansion is optional only after the initial fact is closed."""

        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        initial = [_candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=3,
            content="差旅通讯补贴标准为50元/天。",
        )]

        payloads, client, *_ = await self._run(
            question="差旅通讯补贴标准是多少",
            kb_id=kb_id,
            initial=initial,
            full_document=TimeoutError("small document timeout"),
            scoped=TimeoutError("scoped search timeout"),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(len(result["answer_sources"]), 1)
        self.assertIn(
            "通讯补贴标准为50元",
            client.completions.calls[0]["messages"][1]["content"],
        )
        self.assertNotIn(
            "服务暂时不可用",
            client.completions.calls[0]["messages"][0]["content"],
        )

    async def test_expansion_failure_keeps_related_but_unclosed_candidate_non_answer(self) -> None:
        """A related source without a target/value claim must stay insufficient."""

        kb_id = uuid.uuid4()
        initial = [_candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=3,
            content="差旅补贴制度说明。",
        )]

        payloads, client, *_ = await self._run(
            question="请说明通讯补贴标准",
            kb_id=kb_id,
            initial=initial,
            full_document=TimeoutError("small document timeout"),
            scoped=TimeoutError("scoped search timeout"),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(len(client.completions.calls), 1)
        self.assertIn(
            "coverage_graph:required_answer_claim_missing",
            result["evidence_state"]["reasons"],
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
        self.assertEqual(result["evidence_status"], "hit")
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
        self.assertEqual(len(client.completions.calls), 1)
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("服务暂时不可用", system_prompt)
        self.assertNotIn("未找到相关内容", system_prompt)
        fetch_full.assert_not_awaited()
        scoped.assert_not_awaited()

    async def test_successful_zero_hit_anchor_releases_bridge_static_query(self) -> None:
        """A healthy empty root retrieval is not an upstream failure.

        This guards the scheduler boundary itself: the bridge must remain
        runnable after an empty anchor response, otherwise a mapping that only
        its own task query can find would be silently lost.  The final answer
        still requires the bridge-proven second hop.
        """

        question = "普通员工的住宿标准是多少"
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        mapping = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=1,
            content="普通员工对应D级。",
        )
        lodging = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=6,
            content="住宿标准：D级一线城市不超过450元/天。",
        )
        executed_queries: list[str] = []

        async def controlled_search(_db, retrieval_query, *_args, **_kwargs):
            executed_queries.append(retrieval_query)
            if "适用分类" in retrieval_query:
                return [mapping]
            if "D级" in retrieval_query and "住宿" in retrieval_query:
                return [lodging]
            # The anchor/direct physical group completed successfully with no
            # candidates.  It must not block the bridge's own static query.
            return []

        payloads, client, _search, *_ = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=_typed_bridge_execution_bundle(
                question,
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(result["missing_requirement_ids"], [])
        self.assertTrue(any("适用分类" in query for query in executed_queries))
        self.assertTrue(any("D级" in query and "住宿" in query for query in executed_queries))
        self.assertEqual(len(client.completions.calls), 1)
        self.assertFalse(any(
            call.args and call.args[0] == "retrieval.task_query_blocked"
            for call in self._last_trace.call_args_list
        ))

    async def test_failed_anchor_blocks_static_dependents_before_dispatch(self) -> None:
        """A root adapter failure blocks bridge dispatch through the ledger."""

        question = "普通员工的住宿标准是多少"
        kb_id = uuid.uuid4()
        payloads, client, search, fetch_full, scoped = await self._run(
            question=question,
            kb_id=kb_id,
            initial=RuntimeError("retriever unavailable"),
            full_document=[],
            execution_bundle=_typed_bridge_execution_bundle(
                question,
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "error")
        self.assertEqual(result["evidence_availability"], "unavailable")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(len(client.completions.calls), 1)
        self.assertEqual(search.await_count, 1)
        fetch_full.assert_not_awaited()
        scoped.assert_not_awaited()
        task_error = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.task_query_error"
        )
        self.assertEqual(task_error["reason"], "task_query_retrieval_failed")
        blocked = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.task_query_blocked"
        )
        self.assertEqual(blocked["task_ids"], ["bridge_r2"])
        self.assertEqual(blocked["blocked_by_task_ids"], ["anchor_root"])
        self.assertEqual(
            blocked["reason"],
            "upstream_static_dependency_unavailable",
        )

    async def test_failed_bridge_keeps_independent_closed_answer_partial(self) -> None:
        """One failed branch must not erase an independently closed answer.

        The root succeeds, so the later static stage remains runnable.  The
        bridge branch times out, while an unrelated direct answer has a
        complete current-run claim.  The result must stay partial/degraded:
        retain the usable address source, keep the meal amount closed, and
        never report a false complete answer.
        """

        question = "公司办公地址和普通员工的餐补分别是多少"
        kb_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        requirements = (
            AnswerRequirementV2(
                id="r1",
                description="公司的办公地址是什么",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=(),
                augmentation_requirement_ids=(),
            ),
            AnswerRequirementV2(
                id="r2",
                description="普通员工的餐补是多少",
                role="answer",
                importance="required",
                source="explicit",
                depends_on_requirement_ids=("r3",),
                augmentation_requirement_ids=(),
            ),
            AnswerRequirementV2(
                id="r3",
                description="确认普通员工对应的适用分类",
                role="bridge",
                importance="helpful",
                source="inferred",
                bridge_subject="普通员工",
                bridge_kind="classification",
            ),
        )
        plan = QueryPlanV2(
            original_query=question,
            answer_shape="multi_part",
            retrieval_queries=("legacy projection must not execute",),
            requirements=requirements,
            confidence=0.95,
            source="model",
        )
        bundle = compile_rag_execution_bundle(plan)
        address = _candidate(
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_index=0,
            content="公司办公地址为北京市朝阳区。",
        )

        async def controlled_search(_db, retrieval_query, *_args, **_kwargs):
            if "适用分类" in retrieval_query:
                raise TimeoutError("bridge retrieval timed out")
            if "办公地址" in retrieval_query:
                return [address]
            return []

        payloads, client, _search, *_ = await self._run(
            question=question,
            kb_id=kb_id,
            initial=[],
            full_document=[],
            search_side_effect=controlled_search,
            execution_bundle=bundle,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["evidence_availability"], "degraded")
        self.assertEqual(result["missing_requirement_ids"], ["r2"])
        self.assertEqual(
            {item["content"] for item in result["answer_sources"]},
            {"公司办公地址为北京市朝阳区。"},
        )
        self.assertEqual(len(client.completions.calls), 1)
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("公司办公地址为北京市朝阳区", prompt)
        self.assertNotIn("餐补为", prompt)
        self.assertIn(
            "task_query_retrieval_timeout",
            result["evidence_state"]["reasons"],
        )

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
        self.assertEqual(len(client.completions.calls), 1)
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("服务暂时不可用", system_prompt)
        self.assertNotIn("450元", answer)
        fetch_full.assert_not_awaited()
        scoped.assert_not_awaited()
        search.assert_awaited_once()
        blocked = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "retrieval.task_query_blocked"
        )
        self.assertEqual(blocked["task_ids"], ["bridge_r2"])
        self.assertEqual(blocked["blocked_by_task_ids"], ["anchor_root"])

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
        self.assertEqual(len(client.completions.calls), 1)
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
            content="配置参数：mode设置为strict。",
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
        self.assertEqual(result["evidence_status"], "scope_mismatch")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(len(client.completions.calls), 1)
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertNotIn("旧范围配置值为true", answer)
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("产品、版本或适用范围冲突", system_prompt)

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
        self.assertEqual(len(client.completions.calls), 1)
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertNotIn("敏感正文", answer)
        fetch_full.assert_not_awaited()

    async def test_mutually_exclusive_scopes_clarify_before_generation(self) -> None:
        """Clarification requires two closed answer alternatives, not titles."""

        kb_id = uuid.uuid4()
        first_doc = uuid.uuid4()
        second_doc = uuid.uuid4()
        initial = [
            _candidate(
                kb_id=kb_id,
                doc_id=first_doc,
                chunk_index=0,
                filename="CloudPivot 6 安全配置值.md",
                content=(
                    "所属产品：CloudPivot；产品版本：6。"
                    "安全配置值为方法A。"
                ),
            ),
            _candidate(
                kb_id=kb_id,
                doc_id=second_doc,
                chunk_index=0,
                filename="CloudPivot 7 安全配置值.md",
                content=(
                    "所属产品：CloudPivot；产品版本：7。"
                    "安全配置值为方法B。"
                ),
            ),
        ]

        payloads, client, *_ = await self._run(
            question="安全配置值是多少",
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
            if item["type"] == "clarification_state"
        )
        result = payloads[result_index]
        clarification = payloads[clarification_index]
        self.assertLess(result_index, clarification_index)
        self.assertEqual(result["evidence_status"], "needs_clarification")
        self.assertEqual(result["answer_sources"], [])
        self.assertTrue(clarification["needs_clarification"])
        self.assertEqual(len(clarification["choices"]), 2)
        self.assertEqual(clarification["selection_mode"], "choice")
        self.assertEqual(len(client.completions.calls), 0)
        self.assertEqual(payloads[-1]["type"], "done")

    async def test_broad_document_topic_does_not_clarify_from_raw_candidates(
        self,
    ) -> None:
        """Related document titles are diagnostic-only until answer closure."""
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
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["answer_sources"], [])
        self.assertFalse(any(
            item["type"] == "clarification_state" for item in payloads
        ))
        displayed_doc_ids = {item["doc_id"] for item in result["results"]}
        # Related diagnostics remain bounded by the normal display budget;
        # unlike a clarification they are not required to enumerate every
        # raw candidate/document.
        self.assertTrue(displayed_doc_ids)
        self.assertTrue(displayed_doc_ids.issubset({
            str(leave_doc),
            str(travel_doc),
        }))
        self.assertLessEqual(len(result["results"]), 5)
        self.assertEqual(len(client.completions.calls), 1)

    async def test_low_score_scope_competitor_does_not_force_clarification(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        first_doc = uuid.uuid4()
        second_doc = uuid.uuid4()
        first = _candidate(
            kb_id=kb_id,
            doc_id=first_doc,
            chunk_index=0,
                filename="CloudPivot 6 安全配置值.md",
                content=(
                    "所属产品：CloudPivot；产品版本：6。"
                    "安全配置值为方法A。"
                ),
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
                filename="CloudPivot 7 安全配置值.md",
                content=(
                    "所属产品：CloudPivot；产品版本：7。"
                    "安全配置值为方法B。"
                ),
        )
        # Deliberately outside MAX_DOC_VECTOR_GAP.  Scope identity alone must
        # not turn a low-quality raw neighbour into a user-facing choice; only
        # candidates admitted into the final closed answer graph may do that.
        second.update(
            vector_score=0.82,
            vector_rank=2,
            active_channels=["vector"],
        )

        payloads, client, *_ = await self._run(
            question="安全配置值是多少",
            kb_id=kb_id,
            initial=[first, second],
            full_document=[],
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "hit")
        self.assertEqual(
            {item["doc_id"] for item in result["answer_sources"]},
            {str(first_doc)},
        )
        self.assertFalse(any(
            item["type"] == "clarification_state" for item in payloads
        ))
        self.assertEqual(len(client.completions.calls), 1)

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
            filename="CloudPivot 6 安全配置值.md",
            content=(
                "所属产品：CloudPivot；产品版本：6。"
                "安全配置值为方法A。"
            ),
        )
        uncalibrated = _candidate(
            kb_id=kb_id,
            doc_id=uncalibrated_doc_id,
            chunk_index=0,
            filename="CloudPivot 7 安全配置值.md",
            content="所属产品：CloudPivot；产品版本：7。安全配置值为方法B。",
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
            question="安全配置值是多少",
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
            item["type"] == "clarification_state" for item in payloads
        ))
        self.assertEqual(len(client.completions.calls), 1)
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("安全配置值为方法A", prompt)
        self.assertNotIn("安全配置值为方法B", prompt)

    async def test_no_hit_general_fallback_calls_model_without_sources(self) -> None:
        kb_id = uuid.uuid4()
        settings = _settings()
        settings.rag_general_fallback_mode = "no_hit"
        settings.rag_general_fallback_model = "fast-fallback-model"

        payloads, client, *_ = await self._run(
            question="如何设计一个通用的排班流程",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            settings_override=settings,
            task_contract_override=_task_contract(
                "如何设计一个通用的排班流程",
                evidence_scope="mixed",
            ),
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "no_hit")
        self.assertEqual(result["answer_provenance"], "general_model")
        self.assertEqual(result["general_fallback_mode"], "no_hit")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(result["answer_source_count"], 0)
        self.assertEqual(len(client.completions.calls), 1)
        self.assertEqual(
            client.completions.calls[0]["model"],
            "fast-fallback-model",
        )
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("明确标记为非知识库依据", prompt)
        self.assertNotIn("知识库证据：", prompt)
        answer = "".join(
            item.get("content", "")
            for item in payloads
            if item.get("type") == "text_delta"
        )
        self.assertNotIn("未获得知识库证据支持", answer)
        self.assertIn("已根据资料回答", answer)

    async def test_deepseek_v4_general_fallback_disables_thinking(self) -> None:
        clear_structured_output_capability_cache()
        kb_id = uuid.uuid4()
        settings = _settings()
        settings.rag_general_fallback_mode = "no_hit"
        settings.rag_general_fallback_model = "deepseek-v4-pro"
        settings.llm_base_url = "https://llm.example/v1"

        _payloads_result, client, *_ = await self._run(
            question="如何设计一个通用的排班流程",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            settings_override=settings,
            task_contract_override=_task_contract(
                "如何设计一个通用的排班流程",
                evidence_scope="mixed",
            ),
        )

        self.assertEqual(len(client.completions.calls), 1)
        self.assertEqual(
            client.completions.calls[0]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    async def test_source_bound_no_hit_does_not_let_general_model_guess(self) -> None:
        kb_id = uuid.uuid4()
        settings = _settings()
        settings.rag_general_fallback_mode = "no_hit"
        settings.rag_general_fallback_model = "fast-fallback-model"

        payloads, client, *_ = await self._run(
            question="云枢8.6的登录参数应该怎么修改",
            kb_id=kb_id,
            initial=[],
            full_document=[],
            settings_override=settings,
        )

        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "no_hit")
        self.assertEqual(result["answer_provenance"], "knowledge_base")
        self.assertEqual(
            result["general_fallback_blocked_reason"],
            "source_bound_scope_requires_knowledge_evidence",
        )
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(len(client.completions.calls), 1)
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        self.assertIn("知识库中未找到相关内容", system_prompt)
        self.assertIn("禁止使用自己的知识编造企业事实", system_prompt)

    async def test_insufficient_evidence_fallback_requires_broadest_mode(self) -> None:
        kb_id = uuid.uuid4()
        leave_doc = uuid.uuid4()
        travel_doc = uuid.uuid4()
        candidates = [
            _candidate(
                kb_id=kb_id,
                doc_id=leave_doc,
                chunk_index=index,
                filename="员工请假管理办法.docx",
                content=f"员工请假制度第{index + 1}部分：审批、休假和销假要求。",
            )
            for index in range(6)
        ]
        candidates.append(_candidate(
            kb_id=kb_id,
            doc_id=travel_doc,
            chunk_index=0,
            filename="公司出差管理标准.docx",
            content="员工出差交通、住宿和餐饮补贴标准。",
        ))

        no_hit_only = _settings()
        no_hit_only.rag_general_fallback_mode = "no_hit"
        strict_payloads, strict_client, *_ = await self._run(
            question="员工标准是什么",
            kb_id=kb_id,
            initial=candidates,
            full_document=[],
            settings_override=no_hit_only,
            task_contract_override=_task_contract(
                "员工标准是什么",
                evidence_scope="mixed",
            ),
        )
        strict_result = next(
            item for item in strict_payloads if item["type"] == "search_results"
        )
        self.assertEqual(strict_result["evidence_status"], "insufficient_evidence")
        self.assertEqual(strict_result["answer_provenance"], "knowledge_base")
        self.assertEqual(len(strict_client.completions.calls), 1)

        broad = _settings()
        broad.rag_general_fallback_mode = "no_hit_or_insufficient"
        payloads, client, *_ = await self._run(
            question="员工标准是什么",
            kb_id=kb_id,
            initial=candidates,
            full_document=[],
            settings_override=broad,
            task_contract_override=_task_contract(
                "员工标准是什么",
                evidence_scope="mixed",
            ),
        )
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertEqual(result["answer_provenance"], "general_model")
        self.assertEqual(result["answer_sources"], [])
        self.assertEqual(len(client.completions.calls), 1)
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertNotIn("员工请假制度第1部分", prompt)
        self.assertNotIn("员工出差交通、住宿和餐饮补贴标准", prompt)
        generation_context = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "generation.context"
        )
        self.assertEqual(generation_context["context"], "")
        self.assertEqual(generation_context["context_sources"], [])
        self.assertEqual(generation_context["all_context_sources"], [])

    async def test_adjudication_failure_keeps_general_fallback_closed(self) -> None:
        kb_id = uuid.uuid4()
        candidates = [
            _candidate(
                kb_id=kb_id,
                doc_id=uuid.uuid4(),
                chunk_index=0,
                filename="员工制度摘要.docx",
                content="员工制度包含差旅与费用管理章节。",
            )
        ]
        settings = _settings()
        settings.rag_v2_model_evidence_adjudication_enabled = True
        settings.rag_general_fallback_mode = "no_hit_or_insufficient"
        settings.rag_general_fallback_model = "fast-fallback-model"
        failed_outcome = SimpleNamespace(
            succeeded=False,
            error="ValueError: invalid evidence response",
        )

        with patch(
            "core.rag_v2.pipeline.joint_rerank_with_coverage",
            new=AsyncMock(return_value=failed_outcome),
        ) as adjudicate:
            payloads, client, *_ = await self._run(
                question="普通员工的完整出差标准是什么",
                kb_id=kb_id,
                initial=candidates,
                full_document=[],
                settings_override=settings,
            )

        adjudicate.assert_awaited_once()
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["evidence_status"], "partial")
        self.assertEqual(result["model_adjudication_state"], "failed")
        self.assertEqual(
            result["model_adjudication_error"],
            "ValueError: invalid evidence response",
        )
        self.assertIsNone(result["general_fallback_blocked_reason"])
        self.assertEqual(result["answer_provenance"], "knowledge_base")
        self.assertTrue(result["unverified_generation"])
        self.assertEqual(result["source_verification"], "unverified")
        self.assertEqual(result["direct_evidence_count"], 0)
        self.assertEqual(result["hit_count"], 0)
        self.assertEqual(result["unverified_reference_count"], 1)
        self.assertEqual(len(result["answer_sources"]), 1)
        self.assertEqual(len(client.completions.calls), 1)
        self.assertEqual(client.completions.calls[0]["model"], "test-chat")
        prompt = "\n".join(
            message["content"]
            for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("语义支持关系尚未由重排模型验证", prompt)
        self.assertIn("员工制度包含差旅与费用管理章节", prompt)
        generation_context = next(
            call.kwargs
            for call in self._last_trace.call_args_list
            if call.args and call.args[0] == "generation.context"
        )
        self.assertEqual(generation_context["model_adjudication_state"], "failed")
        self.assertIn("员工制度包含差旅与费用管理章节", generation_context["context"])
        self.assertEqual(len(generation_context["context_sources"]), 1)
        self.assertEqual(len(generation_context["all_context_sources"]), 1)

    async def test_successful_adjudication_with_insufficient_evidence_does_not_degrade(
        self,
    ) -> None:
        kb_id = uuid.uuid4()
        candidate = _candidate(
            kb_id=kb_id,
            doc_id=uuid.uuid4(),
            chunk_index=0,
            filename="员工制度目录.docx",
            content="员工制度包含若干管理章节。",
        )
        settings = _settings()
        settings.rag_v2_model_evidence_adjudication_enabled = True
        outcome = SimpleNamespace(
            succeeded=True,
            error=None,
            results=[{
                **candidate,
                "evidence_role": "related",
                "contribution_role": "background",
                "supports_requirement_ids": [],
                "rerank_status": "verified",
            }],
            coverage_status="insufficient",
            missing_requirement_ids=("r1",),
        )

        with patch(
            "core.rag_v2.pipeline.joint_rerank_with_coverage",
            new=AsyncMock(return_value=outcome),
        ) as adjudicate:
            payloads, client, *_ = await self._run(
                question="普通员工的完整出差标准是什么",
                kb_id=kb_id,
                initial=[candidate],
                full_document=[],
                settings_override=settings,
            )

        adjudicate.assert_awaited_once()
        result = next(item for item in payloads if item["type"] == "search_results")
        self.assertEqual(result["model_adjudication_state"], "succeeded")
        self.assertEqual(result["evidence_status"], "insufficient_evidence")
        self.assertFalse(result["unverified_generation"])
        self.assertEqual(result["answer_sources"], [])
        prompt = "\n".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        self.assertIn("无法组成可核验的完整答案链", prompt)
        self.assertNotIn("员工制度包含若干管理章节", prompt)

    def test_general_fallback_never_opens_for_error_or_scope_mismatch(self) -> None:
        for status in ("error", "scope_mismatch", "needs_clarification", "hit"):
            with self.subTest(status=status):
                self.assertFalse(_should_use_general_model_fallback(
                    evidence_status=status,
                    configured_mode="no_hit_or_insufficient",
                ))

        self.assertTrue(_should_use_general_model_fallback(
            evidence_status="no_hit",
            configured_mode="no_hit",
        ))
        self.assertTrue(_should_use_general_model_fallback(
            evidence_status="insufficient_evidence",
            configured_mode="no_hit_or_insufficient",
        ))
        self.assertFalse(_should_use_general_model_fallback(
            evidence_status="insufficient_evidence",
            configured_mode="no_hit_or_insufficient",
            model_adjudication_succeeded=False,
        ))


if __name__ == "__main__":
    unittest.main()
