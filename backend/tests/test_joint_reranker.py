import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.reranker import (
    AnswerRequirement,
    RerankOutcome,
    joint_rerank_with_coverage,
    rerank_with_status,
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
