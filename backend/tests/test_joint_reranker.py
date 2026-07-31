import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.reranker import (
    AnswerRequirement,
    RerankOutcome,
    joint_rerank_with_coverage,
    rerank_with_status,
    select_small_document_evidence_with_coverage,
)


def _client_with_payload(payload: dict):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )


def _client_with_raw(raw: str):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )


def _payload_response(payload: dict):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
        ]
    )


def _client_with_payload_sequence(*payloads: dict):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    side_effect=[_payload_response(payload) for payload in payloads]
                )
            )
        )
    )


def _settings():
    return SimpleNamespace(
        chat_model="test-chat",
        llm_request_timeout_seconds=10,
        rag_trace_include_content=True,
        rag_trace_include_candidate_details=False,
    )


def _assessment(
    index: int,
    *,
    topic: float = 0.9,
    support: float = 0.8,
    constraint: str = "neutral",
    role: str = "related",
    contribution: str | None = None,
    supports: list[str] | None = None,
    bridge_facts: list[dict] | None = None,
) -> dict:
    value = {
        "index": index,
        "topic_relevance": topic,
        "answer_support": support,
        "constraint_status": constraint,
        "evidence_role": role,
        "reason": "候选提供了可核验的信息",
    }
    if contribution is not None:
        value["contribution_role"] = contribution
    if supports is not None:
        value["supports_requirement_ids"] = supports
    if bridge_facts is not None:
        value["bridge_facts"] = bridge_facts
    return value


def _requirement(
    identifier: str,
    description: str,
    *,
    importance: str = "required",
    source: str = "explicit",
) -> dict:
    return {
        "id": identifier,
        "description": description,
        "importance": importance,
        "source": source,
    }


class FirstPassPlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_parses_safe_bridge_and_expansion_plan(self) -> None:
        query = "普通员工的出差标准是什么"
        results = [
            {
                "id": "grade",
                "content": "普通员工、专员属于D级。",
                "score": 0.02,
            }
        ]
        payload = {
            "requirements": [
                _requirement("r1", "确定普通员工适用等级"),
                _requirement("r2", "取得该等级的出差标准"),
            ],
            "results": [
                _assessment(
                    1,
                    support=0.55,
                    contribution="bridge",
                    supports=["r1"],
                    bridge_facts=[
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                )
            ],
            "expansion": {
                "needed": True,
                "target_candidate_indexes": [1],
                "queries": ["D级 出差标准"],
                "missing_requirement_ids": ["r2"],
                "reason": "当前只有职级映射，缺少标准明细",
            },
        }
        client = _client_with_payload(payload)

        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await rerank_with_status(query, results)

        self.assertTrue(outcome.succeeded)
        self.assertEqual([item.id for item in outcome.requirements], ["r1", "r2"])
        self.assertIsNotNone(outcome.expansion_plan)
        self.assertTrue(outcome.expansion_plan.needed)
        self.assertEqual(outcome.expansion_plan.target_candidate_indexes, (1,))
        self.assertEqual(outcome.results[0]["contribution_role"], "bridge")
        self.assertEqual(
            outcome.results[0]["bridge_facts"][0]["object"],
            "D级",
        )

    async def test_fabricated_bridge_term_fails_closed(self) -> None:
        query = "普通员工的出差标准是什么"
        results = [{"id": "grade", "content": "普通员工属于D级", "score": 0.02}]
        payload = {
            "requirements": [_requirement("r1", "确定职级")],
            "results": [
                _assessment(
                    1,
                    contribution="bridge",
                    supports=["r1"],
                    bridge_facts=[
                        {
                            "subject": "普通员工",
                            "relation": "属于",
                            "object": "不存在的S级",
                        }
                    ],
                )
            ],
        }

        with (
            patch("core.reranker.get_client", return_value=_client_with_payload(payload)),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await rerank_with_status(query, results)

        self.assertFalse(outcome.succeeded)
        self.assertIn("不在问题或候选正文", outcome.error or "")
        self.assertEqual(outcome.results[0]["rerank_status"], "unverified")

    async def test_fabricated_expansion_index_fails_closed(self) -> None:
        results = [{"id": "grade", "content": "普通员工属于D级", "score": 0.02}]
        payload = {
            "requirements": [_requirement("r1", "取得出差标准")],
            "results": [
                _assessment(
                    1,
                    contribution="bridge",
                    supports=[],
                    bridge_facts=[
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                )
            ],
            "expansion": {
                "needed": True,
                "target_candidate_indexes": [2],
                "queries": ["D级 出差标准"],
                "missing_requirement_ids": ["r1"],
            },
        }

        with (
            patch("core.reranker.get_client", return_value=_client_with_payload(payload)),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await rerank_with_status("普通员工的出差标准是什么", results)

        self.assertFalse(outcome.succeeded)
        self.assertIn("不存在的候选索引", outcome.error or "")

    async def test_inferred_required_requirement_is_demoted_to_helpful(self) -> None:
        payload = {
            "requirements": [
                _requirement(
                    "r1",
                    "模型推断的可选背景",
                    importance="required",
                    source="inferred",
                )
            ],
            "results": [_assessment(1, role="direct")],
        }
        with (
            patch("core.reranker.get_client", return_value=_client_with_payload(payload)),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await rerank_with_status(
                "请解释这条明确事实",
                [{"id": "a", "content": "明确事实", "score": 0.1}],
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.requirements[0].importance, "helpful")

    async def test_route_locked_requirement_cannot_be_downgraded_by_reranker(self) -> None:
        locked = (
            AnswerRequirement(
                id="r1",
                description="取得完整出差标准",
                importance="required",
                source="explicit",
            ),
        )
        payload = {
            "requirements": [
                _requirement(
                    "r1",
                    "取得完整出差标准",
                    importance="helpful",
                    source="explicit",
                )
            ],
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                )
            ],
        }

        with (
            patch("core.reranker.get_client", return_value=_client_with_payload(payload)),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await rerank_with_status(
                "普通员工的出差标准是什么",
                [{"id": "answer", "content": "D级出差标准", "score": 0.1}],
                locked,
            )

        self.assertFalse(outcome.succeeded)
        self.assertIn("不得修改路由器锁定的 requirements", outcome.error or "")
        self.assertEqual(outcome.results[0]["rerank_status"], "unverified")

    async def test_complete_standalone_answer_disables_model_expansion(self) -> None:
        results = [
            {"id": "answer", "content": "报销应在5个工作日内提交。", "score": 0.1},
            {"id": "bridge", "content": "普通员工属于D级。", "score": 0.02},
        ]
        payload = {
            "requirements": [_requirement("r1", "报销时限")],
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                ),
                _assessment(
                    2,
                    support=0.5,
                    contribution="bridge",
                    supports=[],
                    bridge_facts=[
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                ),
            ],
            "expansion": {
                "needed": True,
                "target_candidate_indexes": [2],
                "queries": ["D级 出差标准"],
                "missing_requirement_ids": [],
                "reason": "模型过度请求扩展",
            },
        }
        with (
            patch("core.reranker.get_client", return_value=_client_with_payload(payload)),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await rerank_with_status("报销应在多久内提交", results)

        self.assertTrue(outcome.succeeded)
        self.assertIsNotNone(outcome.expansion_plan)
        self.assertFalse(outcome.expansion_plan.needed)
        self.assertTrue(outcome.expansion_plan.model_requested)
        self.assertIn("无需", outcome.expansion_plan.overridden_reason or "")

    async def test_expansion_cannot_introduce_conflicting_version(self) -> None:
        results = [{"id": "v86", "content": "云枢8.6登录配置概览", "score": 0.1}]
        payload = {
            "requirements": [_requirement("r1", "登录配置参数")],
            "results": [
                _assessment(
                    1,
                    contribution="bridge",
                    supports=[],
                    bridge_facts=[
                        {"subject": "云枢8.6", "relation": "包含", "object": "登录配置"}
                    ],
                )
            ],
            "expansion": {
                "needed": True,
                "target_candidate_indexes": [1],
                "queries": ["云枢7 登录配置参数"],
                "missing_requirement_ids": ["r1"],
            },
        }
        with (
            patch("core.reranker.get_client", return_value=_client_with_payload(payload)),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await rerank_with_status("云枢8.6登录怎么配置", results)

        self.assertFalse(outcome.succeeded)
        self.assertIn("冲突", outcome.error or "")


class SmallDocumentSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, payload, *, eligible=(1, 2, 3), anchors=(1,)):
        results = [
            {"id": "grade", "doc_id": "travel", "content": "普通员工属于D级。"},
            {"id": "train", "doc_id": "travel", "content": "D级乘高铁二等座。"},
            {"id": "hotel", "doc_id": "travel", "content": "D级住宿450元/天。"},
            {"id": "other", "doc_id": "other", "content": "员工请假需审批。"},
        ]
        requirements = [
            _requirement("r1", "回答普通员工出差标准"),
            _requirement(
                "r2",
                "确认普通员工对应职级",
                importance="helpful",
                source="inferred",
            ),
        ]
        client = _client_with_payload(payload)
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await select_small_document_evidence_with_coverage(
                "普通员工的出差标准是什么",
                results,
                requirements,
                bridge_requirement_ids=("r2",),
                eligible_candidate_indexes=eligible,
                anchor_candidate_indexes=anchors,
            )
        return outcome, client

    async def test_compact_selector_builds_complete_target_document_set(self) -> None:
        payload = {
            "selected": [
                {
                    "index": 1,
                    "role": "bridge",
                    "supports_requirement_ids": ["r2"],
                    "bridge_facts": [
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                },
                {
                    "index": 2,
                    "role": "answer",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [],
                },
                {
                    "index": 3,
                    "role": "answer",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [],
                },
            ],
            "coverage_complete": True,
        }

        outcome, client = await self._run(payload)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "complete")
        self.assertEqual(outcome.selected_candidate_indexes, (1, 2, 3))
        selected = [item for item in outcome.results if item.get("jointly_selected")]
        self.assertEqual({item["id"] for item in selected}, {"grade", "train", "hotel"})
        self.assertTrue(all(item["evidence_role"] == "direct" for item in selected))
        self.assertTrue(all(
            item["assessment_mode"] == "small_document_binary_selection"
            for item in outcome.results
        ))
        competitor = next(item for item in outcome.results if item["id"] == "other")
        self.assertFalse(competitor["jointly_selected"])

        call = client.chat.completions.create.await_args
        response_format = call.kwargs["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        index_schema = response_format["json_schema"]["schema"]["properties"][
            "selected"
        ]["items"]["properties"]["index"]
        self.assertEqual(index_schema["enum"], [1, 2, 3])
        self.assertNotIn(
            "evidence_sets",
            response_format["json_schema"]["schema"]["properties"],
        )
        user_payload = json.loads(call.kwargs["messages"][1]["content"])
        self.assertEqual(user_payload["output_contract"], "json")
        self.assertEqual(user_payload["bridge_requirement_ids"], ["r2"])
        self.assertEqual(user_payload["eligible_candidate_indexes"], [1, 2, 3])
        self.assertEqual(user_payload["anchor_candidate_indexes"], [1])
        self.assertLessEqual(call.kwargs["timeout"], 15)
        self.assertLessEqual(call.kwargs["max_tokens"], 700)
        grade = next(item for item in selected if item["id"] == "grade")
        train = next(item for item in selected if item["id"] == "train")
        self.assertEqual(grade["contribution_role"], "bridge")
        self.assertEqual(grade["bridge_facts"][0]["object"], "D级")
        self.assertEqual(train["contribution_role"], "complement")

    async def test_complete_claim_without_required_bridge_fails_closed(self) -> None:
        payload = {
            "selected": [{
                "index": 2,
                "role": "answer",
                "supports_requirement_ids": ["r1"],
                "bridge_facts": [],
            }],
            "coverage_complete": True,
        }

        outcome, _client = await self._run(payload)

        self.assertFalse(outcome.succeeded)
        self.assertIn("bridge", outcome.error or "")
        self.assertFalse(any(
            item.get("jointly_selected") for item in outcome.results
        ))

    async def test_competitor_cannot_be_selected_outside_target_allowlist(self) -> None:
        payload = {
            "selected": [
                {
                    "index": 1,
                    "role": "bridge",
                    "supports_requirement_ids": ["r2"],
                    "bridge_facts": [
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                },
                {
                    "index": 4,
                    "role": "answer",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [],
                },
            ],
            "coverage_complete": True,
        }

        outcome, _client = await self._run(payload)

        self.assertFalse(outcome.succeeded)
        self.assertIn("index", outcome.error or "")
        self.assertFalse(any(
            item.get("jointly_selected") for item in outcome.results
        ))

    async def test_valid_incomplete_selection_remains_no_evidence(self) -> None:
        payload = {
            "selected": [{
                "index": 1,
                "role": "bridge",
                "supports_requirement_ids": ["r2"],
                "bridge_facts": [
                    {"subject": "普通员工", "relation": "属于", "object": "D级"}
                ],
            }],
            "coverage_complete": False,
        }

        outcome, _client = await self._run(payload)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "insufficient")
        self.assertEqual(outcome.selected_candidate_indexes, ())
        self.assertEqual(outcome.missing_requirement_ids, ("r1",))

    async def test_complete_claim_without_selected_anchor_fails_closed(self) -> None:
        payload = {
            "selected": [
                {
                    "index": 2,
                    "role": "bridge",
                    "supports_requirement_ids": ["r2"],
                    "bridge_facts": [
                        {"subject": "D级", "relation": "乘坐", "object": "高铁二等座"}
                    ],
                },
                {
                    "index": 3,
                    "role": "answer",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [],
                },
            ],
            "coverage_complete": True,
        }

        outcome, _client = await self._run(payload, anchors=(1,))

        self.assertFalse(outcome.succeeded)
        self.assertIn("锚点", outcome.error or "")

    async def test_fabricated_bridge_fact_fails_closed(self) -> None:
        payload = {
            "selected": [
                {
                    "index": 1,
                    "role": "bridge",
                    "supports_requirement_ids": ["r2"],
                    "bridge_facts": [
                        {"subject": "普通员工", "relation": "属于", "object": "A级"}
                    ],
                },
                {
                    "index": 2,
                    "role": "answer",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [],
                },
            ],
            "coverage_complete": True,
        }

        outcome, _client = await self._run(payload)

        self.assertFalse(outcome.succeeded)
        self.assertIn("object", outcome.error or "")

    async def test_single_required_question_keeps_anchored_bridge(self) -> None:
        results = [
            {"id": "grade", "doc_id": "travel", "content": "普通员工属于D级。"},
            {
                "id": "standard",
                "doc_id": "travel",
                "content": "D级乘经济舱，住宿450元/天。" + "细则" * 1600,
            },
        ]
        payload = {
            "selected": [
                {
                    "index": 1,
                    "role": "bridge",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                },
                {
                    "index": 2,
                    "role": "answer",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [],
                },
            ],
            "coverage_complete": True,
        }
        client = _client_with_payload(payload)
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await select_small_document_evidence_with_coverage(
                "普通员工的出差标准是什么",
                results,
                [_requirement("r1", "回答普通员工出差标准")],
                eligible_candidate_indexes=(1, 2),
                anchor_candidate_indexes=(1,),
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "complete")
        self.assertEqual(outcome.selected_candidate_indexes, (1, 2))
        selected = {
            item["id"]: item
            for item in outcome.results
            if item.get("jointly_selected")
        }
        self.assertEqual(selected["grade"]["contribution_role"], "bridge")
        self.assertEqual(selected["standard"]["contribution_role"], "complement")
        self.assertEqual(selected["grade"]["bridge_facts"][0]["object"], "D级")
        request_payload = json.loads(
            client.chat.completions.create.await_args.kwargs["messages"][1]["content"]
        )
        self.assertFalse(request_payload["candidates"][1]["content_truncated"])
        self.assertGreater(len(request_payload["candidates"][1]["content"]), 3000)

    async def test_answer_with_bridge_facts_fails_closed(self) -> None:
        payload = {
            "selected": [
                {
                    "index": 1,
                    "role": "bridge",
                    "supports_requirement_ids": ["r2"],
                    "bridge_facts": [
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                },
                {
                    "index": 2,
                    "role": "answer",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [
                        {"subject": "D级", "relation": "乘坐", "object": "高铁二等座"}
                    ],
                },
            ],
            "coverage_complete": True,
        }

        outcome, _client = await self._run(payload)

        self.assertFalse(outcome.succeeded)
        self.assertIn("必须为空", outcome.error or "")

    async def test_json_fallback_cannot_exceed_bridge_fact_schema_limit(self) -> None:
        fact = {"subject": "普通员工", "relation": "属于", "object": "D级"}
        payload = {
            "selected": [
                {
                    "index": 1,
                    "role": "bridge",
                    "supports_requirement_ids": ["r2"],
                    "bridge_facts": [fact, fact, fact],
                },
                {
                    "index": 2,
                    "role": "answer",
                    "supports_requirement_ids": ["r1"],
                    "bridge_facts": [],
                },
            ],
            "coverage_complete": True,
        }

        outcome, _client = await self._run(payload)

        self.assertFalse(outcome.succeeded)
        self.assertIn("契约上限", outcome.error or "")


class JointCoverageTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, query, results, requirements, payload):
        client = _client_with_payload(payload)
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            return await joint_rerank_with_coverage(
                query,
                results,
                requirements,
            )

    async def test_joint_request_prefers_strict_schema_enum_contract(self) -> None:
        requirements = [_requirement("r1", "取得住宿标准")]
        results = [{"id": "hotel", "content": "D级住宿450元/天。", "score": 0.02}]
        payload = {
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1],
                    "joint_answer_support": 0.9,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]}
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "住宿标准完整",
                }
            ],
            "selected_set_id": "set_1",
        }
        client = _client_with_payload(payload)
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "普通员工住宿标准是什么",
                results,
                requirements,
            )

        self.assertTrue(outcome.succeeded)
        response_format = client.chat.completions.create.await_args.kwargs[
            "response_format"
        ]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        results_schema = schema["properties"]["results"]
        self.assertEqual(results_schema["minItems"], 0)
        self.assertEqual(results_schema["maxItems"], 1)
        result_properties = results_schema["items"]["properties"]
        self.assertEqual(
            set(result_properties["evidence_role"]["enum"]),
            {"direct", "related", "irrelevant"},
        )
        self.assertEqual(
            result_properties["supports_requirement_ids"]["items"]["enum"],
            ["r1"],
        )
        self.assertLessEqual(
            client.chat.completions.create.await_args.kwargs["max_tokens"],
            2800,
        )

    async def test_compact_joint_response_safely_omits_irrelevant_candidates(self) -> None:
        requirements = [_requirement("r1", "完整回答普通员工出差标准")]
        results = [
            {"id": f"chunk-{index}", "content": f"无关章节 {index}", "score": 0.02}
            for index in range(1, 16)
        ]
        results[2]["content"] = "普通员工属于D级"
        results[3]["content"] = "D级乘坐经济舱和高铁二等座"
        results[6]["content"] = "D级住宿一线450元、二线350元、其他250元"
        results[7]["content"] = "D级餐饮补贴100元/天"
        results[8]["content"] = "通讯50元/天，出差补贴100元/天"
        selected_indexes = (3, 4, 7, 8, 9)
        payload = {
            "results": [
                _assessment(
                    index,
                    role="direct",
                    contribution=("bridge" if index == 3 else "complement"),
                    supports=["r1"],
                    bridge_facts=(
                        [{"subject": "普通员工", "relation": "属于", "object": "D级"}]
                        if index == 3
                        else []
                    ),
                )
                for index in selected_indexes
            ],
            "evidence_sets": [{
                "id": "set_1",
                "candidate_indexes": list(selected_indexes),
                "joint_answer_support": 0.94,
                "coverage": [{
                    "requirement_id": "r1",
                    "candidate_indexes": list(selected_indexes),
                }],
                "coverage_status": "complete",
                "missing_requirement_ids": [],
                "reason": "职级映射与D级各项标准共同覆盖问题",
            }],
            "selected_set_id": "set_1",
        }

        outcome = await self._run(
            "普通员工的出差标准是什么",
            results,
            requirements,
            payload,
        )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "complete")
        self.assertEqual(outcome.selected_candidate_indexes, selected_indexes)
        self.assertEqual(len(outcome.results), 15)
        selected = [item for item in outcome.results if item["jointly_selected"]]
        omitted = [item for item in outcome.results if not item["jointly_selected"]]
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(omitted), 10)
        self.assertTrue(all(item["rerank_status"] == "unverified" for item in omitted))
        self.assertTrue(all(item["joint_rerank_status"] == "omitted" for item in omitted))
        self.assertTrue(all(item["evidence_role"] == "irrelevant" for item in omitted))
        self.assertTrue(all(item["supports_requirement_ids"] == [] for item in omitted))

    async def test_bridge_and_complement_form_complete_joint_evidence(self) -> None:
        requirements = [
            _requirement("r1", "确定普通员工适用等级"),
            _requirement("r2", "取得该等级交通住宿与补贴标准"),
        ]
        results = [
            {"id": "grade", "content": "普通员工、专员属于D级。", "score": 0.02},
            {
                "id": "standard",
                "content": "D级：经济舱、住宿不超过450元、补贴100元。",
                "score": 0.02,
            },
        ]
        payload = {
            "results": [
                _assessment(
                    1,
                    support=0.55,
                    contribution="bridge",
                    supports=["r1"],
                    bridge_facts=[
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                ),
                _assessment(
                    2,
                    support=0.9,
                    contribution="complement",
                    supports=["r2"],
                    bridge_facts=[],
                ),
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1, 2],
                    "joint_answer_support": 0.92,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]},
                        {"requirement_id": "r2", "candidate_indexes": [2]},
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "职级映射和D级标准联合覆盖问题",
                }
            ],
            "selected_set_id": "set_1",
        }

        outcome = await self._run(
            "普通员工的出差标准是什么",
            results,
            requirements,
            payload,
        )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "complete")
        self.assertEqual(outcome.selected_candidate_indexes, (1, 2))
        self.assertEqual(outcome.missing_requirement_ids, ())
        self.assertTrue(all(item["jointly_selected"] for item in outcome.results))
        self.assertTrue(all(item["evidence_role"] == "direct" for item in outcome.results))
        self.assertTrue(all(item["rerank_status"] == "verified_joint" for item in outcome.results))

    async def test_known_contribution_alias_is_normalized_deterministically(self) -> None:
        requirements = [_requirement("r1", "取得补贴标准")]
        results = [
            {
                "id": "allowance",
                "content": "D级出差补贴为100元/天。",
                "score": 0.02,
            }
        ]
        payload = {
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="supplementary",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1],
                    "joint_answer_support": 0.9,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]}
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "补贴片段覆盖必要需求",
                }
            ],
            "selected_set_id": "set_1",
        }
        client = _client_with_payload(payload)
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "普通员工的出差补贴是多少",
                results,
                requirements,
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "complete")
        self.assertEqual(outcome.results[0]["contribution_role"], "complement")
        self.assertEqual(
            outcome.results[0]["contribution_role_original"], "supplementary"
        )
        self.assertEqual(
            outcome.results[0]["contribution_role_resolution"],
            "normalized_alias",
        )
        client.chat.completions.create.assert_awaited_once()

    async def test_unknown_contribution_role_is_downgraded_without_losing_batch(self) -> None:
        requirements = [
            _requirement("r1", "确定适用职级"),
            _requirement("r2", "取得交通标准"),
        ]
        results = [
            {"id": "unknown", "content": "普通员工属于D级。", "score": 0.02},
            {"id": "travel", "content": "D级乘坐经济舱。", "score": 0.02},
        ]
        payload = {
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="primary",
                    supports=["r1"],
                    bridge_facts=[],
                ),
                _assessment(
                    2,
                    contribution="complement",
                    supports=["r2"],
                    bridge_facts=[],
                ),
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1, 2],
                    "joint_answer_support": 0.9,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]},
                        {"requirement_id": "r2", "candidate_indexes": [2]},
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "模型声称两个片段共同覆盖",
                }
            ],
            "selected_set_id": "set_1",
        }
        client = _client_with_payload(payload)
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "普通员工的出差标准是什么",
                results,
                requirements,
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "partial")
        self.assertEqual(outcome.selected_candidate_indexes, (2,))
        by_id = {item["id"]: item for item in outcome.results}
        invalid = by_id["unknown"]
        self.assertEqual(invalid["contribution_role"], "irrelevant")
        self.assertEqual(invalid["contribution_role_original"], "primary")
        self.assertEqual(
            invalid["contribution_role_resolution"], "downgraded_unknown"
        )
        self.assertEqual(invalid["supports_requirement_ids"], [])
        self.assertEqual(invalid["evidence_role"], "irrelevant")
        self.assertFalse(invalid["jointly_selected"])
        self.assertTrue(by_id["travel"]["jointly_selected"])
        client.chat.completions.create.assert_awaited_once()

    async def test_invalid_structure_is_repaired_once_with_short_prompt(self) -> None:
        requirements = [_requirement("r1", "取得住宿标准")]
        results = [
            {
                "id": "hotel",
                "content": "UNIQUE_CANDIDATE_BODY D级一线城市住宿450元/天。",
                "score": 0.02,
            }
        ]
        valid_payload = {
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1],
                    "joint_answer_support": 0.9,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]}
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "住宿标准完整",
                }
            ],
            "selected_set_id": "set_1",
        }
        invalid_payload = json.loads(json.dumps(valid_payload))
        invalid_payload["evidence_sets"][0]["coverage_status"] = "fully_covered"
        client = _client_with_payload_sequence(invalid_payload, valid_payload)
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "普通员工住宿标准是什么",
                results,
                requirements,
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "complete")
        self.assertEqual(client.chat.completions.create.await_count, 2)
        repair_call = client.chat.completions.create.await_args_list[1]
        self.assertEqual(repair_call.kwargs["timeout"], 8.0)
        self.assertNotIn(
            "UNIQUE_CANDIDATE_BODY",
            repair_call.kwargs["messages"][1]["content"],
        )
        self.assertIn(
            "coverage_status 无效",
            repair_call.kwargs["messages"][1]["content"],
        )

    async def test_schema_fallback_repair_uses_lowercase_json_contract(self) -> None:
        class ProviderContractError(Exception):
            def __init__(self, status_code: int, message: str) -> None:
                super().__init__(message)
                self.status_code = status_code

        requirements = [_requirement("r1", "取得住宿标准")]
        results = [{"id": "hotel", "content": "D级住宿450元/天。", "score": 0.02}]
        valid_payload = {
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1],
                    "joint_answer_support": 0.9,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]}
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "住宿标准完整",
                }
            ],
            "selected_set_id": "set_1",
        }
        invalid_payload = json.loads(json.dumps(valid_payload))
        invalid_payload["results"][0]["evidence_role"] = "DIRECT_EVIDENCE"
        json_object_attempt = 0

        async def create(**kwargs):
            nonlocal json_object_attempt
            response_format = kwargs["response_format"]
            if response_format["type"] == "json_schema":
                raise ProviderContractError(
                    400,
                    "Invalid value: json_schema. Supported values are: "
                    "text, json_object",
                )
            json_object_attempt += 1
            messages_text = "\n".join(
                str(message.get("content") or "") for message in kwargs["messages"]
            )
            if json_object_attempt == 1:
                return _payload_response(invalid_payload)
            if "json" not in messages_text:
                raise ProviderContractError(
                    400,
                    "messages must contain the word 'json' to use json_object",
                )
            return _payload_response(valid_payload)

        create_mock = AsyncMock(side_effect=create)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_mock))
        )
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "普通员工住宿标准是什么",
                results,
                requirements,
            )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "complete")
        self.assertEqual(create_mock.await_count, 3)
        self.assertEqual(
            create_mock.await_args_list[0].kwargs["response_format"]["type"],
            "json_schema",
        )
        self.assertTrue(
            all(
                call.kwargs["response_format"] == {"type": "json_object"}
                for call in create_mock.await_args_list[1:]
            )
        )
        repair_messages = create_mock.await_args_list[2].kwargs["messages"]
        self.assertIn(
            "json",
            "\n".join(str(message.get("content") or "") for message in repair_messages),
        )
        self.assertIn(
            "evidence_role 必须为",
            repair_messages[1]["content"],
        )

    async def test_failed_structure_repair_falls_back_without_promotion(self) -> None:
        requirements = [_requirement("r1", "答案")]
        results = [{"id": "candidate", "content": "候选正文", "score": 0.02}]
        invalid_payload = {
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": "not-an-array",
            "selected_set_id": None,
        }
        client = _client_with_payload_sequence(invalid_payload, invalid_payload)
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "答案是什么",
                results,
                requirements,
            )

        self.assertFalse(outcome.succeeded)
        self.assertIn("联合重排结构修复失败", outcome.error or "")
        self.assertEqual(client.chat.completions.create.await_count, 2)
        self.assertEqual(outcome.coverage_status, "insufficient")
        self.assertFalse(outcome.results[0]["jointly_selected"])
        self.assertIsNone(outcome.results[0]["evidence_role"])

    async def test_repair_provider_exception_falls_back_without_third_attempt(self) -> None:
        requirements = [_requirement("r1", "答案")]
        results = [{"id": "candidate", "content": "候选正文", "score": 0.02}]
        invalid_payload = {
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": "not-an-array",
            "selected_set_id": None,
        }
        create = AsyncMock(
            side_effect=[_payload_response(invalid_payload), TimeoutError("repair timeout")]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "答案是什么",
                results,
                requirements,
            )

        self.assertFalse(outcome.succeeded)
        self.assertIn("TimeoutError", outcome.error or "")
        self.assertEqual(create.await_count, 2)
        self.assertFalse(outcome.results[0]["jointly_selected"])
        self.assertIsNone(outcome.results[0]["evidence_role"])

    async def test_code_downgrades_claimed_complete_when_required_field_missing(self) -> None:
        requirements = [
            _requirement("r1", "确定等级"),
            _requirement("r2", "交通标准"),
            _requirement("r3", "住宿标准"),
        ]
        results = [
            {"id": "grade", "content": "普通员工属于D级", "score": 0.02},
            {"id": "travel", "content": "D级乘坐经济舱", "score": 0.02},
        ]
        payload = {
            "results": [
                _assessment(
                    1,
                    contribution="bridge",
                    supports=["r1"],
                    bridge_facts=[
                        {"subject": "普通员工", "relation": "属于", "object": "D级"}
                    ],
                ),
                _assessment(
                    2,
                    contribution="complement",
                    supports=["r2"],
                    bridge_facts=[],
                ),
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1, 2],
                    "joint_answer_support": 0.95,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]},
                        {"requirement_id": "r2", "candidate_indexes": [2]},
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "模型错误声称完整",
                }
            ],
            "selected_set_id": "set_1",
        }

        outcome = await self._run(
            "普通员工的出差标准是什么",
            results,
            requirements,
            payload,
        )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "partial")
        self.assertEqual(outcome.missing_requirement_ids, ("r3",))
        self.assertEqual(outcome.evidence_sets[0].model_coverage_status, "complete")
        self.assertEqual(outcome.evidence_sets[0].coverage_status, "partial")
        self.assertTrue(all(item["evidence_role"] == "related" for item in outcome.results))

    async def test_version_mismatch_cannot_enter_joint_set(self) -> None:
        requirements = [_requirement("r1", "云枢8.6登录配置")]
        results = [
            {
                "id": "v6",
                "filename": "云枢6配置.md",
                "content": "产品版本：云枢6全系，error_reply_same: true",
                "score": 0.02,
            }
        ]
        payload = {
            "results": [
                _assessment(
                    1,
                    constraint="exact",
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1],
                    "joint_answer_support": 0.99,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]}
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "模型错误采用旧版本",
                }
            ],
            "selected_set_id": "set_1",
        }

        outcome = await self._run(
            "云枢8.6登录怎么配置",
            results,
            requirements,
            payload,
        )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "insufficient")
        self.assertEqual(outcome.selected_candidate_indexes, ())
        self.assertEqual(outcome.results[0]["constraint_status"], "mismatch")
        self.assertFalse(outcome.results[0]["jointly_selected"])
        self.assertNotEqual(outcome.results[0]["evidence_role"], "direct")

    async def test_coverage_claim_must_match_candidate_supports(self) -> None:
        requirements = [_requirement("r1", "答案必要字段")]
        results = [{"id": "a", "content": "仅仅主题相近", "score": 0.02}]
        payload = {
            "results": [
                _assessment(
                    1,
                    contribution="complement",
                    supports=[],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1],
                    "joint_answer_support": 0.99,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]}
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "模型错误声称候选覆盖了必要字段",
                }
            ],
            "selected_set_id": "set_1",
        }

        outcome = await self._run("答案是什么", results, requirements, payload)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "insufficient")
        self.assertEqual(outcome.covered_requirement_ids, ())
        self.assertEqual(outcome.missing_requirement_ids, ("r1",))
        self.assertFalse(outcome.results[0]["jointly_selected"])

    async def test_unknown_product_scope_cannot_enter_joint_set(self) -> None:
        requirements = [_requirement("r1", "云枢登录配置")]
        results = [
            {"id": "generic", "content": "设置统一登录失败提示", "score": 0.02}
        ]
        payload = {
            "results": [
                _assessment(
                    1,
                    constraint="exact",
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [1],
                    "joint_answer_support": 0.9,
                    "coverage": [
                        {"requirement_id": "r1", "candidate_indexes": [1]}
                    ],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "适用产品未知",
                }
            ],
            "selected_set_id": "set_1",
        }

        outcome = await self._run(
            "云枢登录怎么配置",
            results,
            requirements,
            payload,
        )

        self.assertEqual(outcome.coverage_status, "insufficient")
        self.assertEqual(outcome.results[0]["constraint_status"], "unknown")
        self.assertFalse(outcome.results[0]["jointly_selected"])

    async def test_fabricated_evidence_set_index_fails_without_promotion(self) -> None:
        requirements = [_requirement("r1", "答案")]
        results = [{"id": "a", "content": "答案正文", "score": 0.02}]
        payload = {
            "results": [
                _assessment(
                    1,
                    role="direct",
                    contribution="standalone_answer",
                    supports=["r1"],
                    bridge_facts=[],
                )
            ],
            "evidence_sets": [
                {
                    "id": "set_1",
                    "candidate_indexes": [2],
                    "joint_answer_support": 0.9,
                    "coverage": [],
                    "coverage_status": "complete",
                    "missing_requirement_ids": [],
                    "reason": "伪造索引",
                }
            ],
            "selected_set_id": "set_1",
        }

        outcome = await self._run("答案是什么", results, requirements, payload)

        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "insufficient")
        self.assertFalse(outcome.results[0]["jointly_selected"])
        self.assertNotEqual(outcome.results[0].get("evidence_role"), "direct")

    async def test_timeout_does_not_promote_unverified_expansion_candidate(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(side_effect=TimeoutError("upstream timeout"))
                )
            )
        )
        raw_expansion = {
            "id": "expanded",
            "content": "扩展片段",
            "score": 0.02,
            "evidence_role": "direct",
        }
        with (
            patch("core.reranker.get_client", return_value=client),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "问题",
                [raw_expansion],
                [_requirement("r1", "答案")],
            )

        self.assertFalse(outcome.succeeded)
        self.assertIn("TimeoutError", outcome.error or "")
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.assertFalse(outcome.results[0]["jointly_selected"])
        self.assertIsNone(outcome.results[0]["evidence_role"])
        self.assertEqual(outcome.results[0]["rerank_status"], "unverified")

    async def test_malformed_joint_json_falls_back_without_promotion(self) -> None:
        with (
            patch("core.reranker.get_client", return_value=_client_with_raw("{bad json")),
            patch("core.reranker.get_settings", return_value=_settings()),
        ):
            outcome = await joint_rerank_with_coverage(
                "答案是什么",
                [{"id": "expanded", "content": "候选", "score": 0.02}],
                [_requirement("r1", "答案")],
            )

        self.assertFalse(outcome.succeeded)
        self.assertIn("JSONDecodeError", outcome.error or "")
        self.assertEqual(outcome.coverage_status, "insufficient")
        self.assertFalse(outcome.results[0]["jointly_selected"])

    def test_rerank_outcome_old_constructor_remains_compatible(self) -> None:
        outcome = RerankOutcome(results=[], succeeded=False, error="disabled")

        self.assertEqual(outcome.requirements, ())
        self.assertIsNone(outcome.coverage_status)
        self.assertEqual(outcome.selected_candidate_indexes, ())

    async def test_accepts_answer_requirement_instances(self) -> None:
        outcome = await joint_rerank_with_coverage(
            "问题",
            [],
            [AnswerRequirement(id="r1", description="答案")],
        )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.coverage_status, "insufficient")
        self.assertEqual(outcome.missing_requirement_ids, ("r1",))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
