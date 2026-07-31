import copy
import json
import unittest

from core.query_route_contract import (
    MAX_REQUIREMENTS,
    ROUTE_DECISION_SCHEMA_VERSION,
    RouteDecisionValidationError,
    build_rag_route_decision_schema,
    build_rag_route_response_format,
    parse_rag_route_decision,
)


def _payload(**updates):
    value = {
        "schema_version": ROUTE_DECISION_SCHEMA_VERSION,
        "readiness": "ready",
        "intent_code": "knowledge_qa",
        "relation": "new",
        "evidence_scope": "enterprise_kb",
        "query_resolution": {
            "mode": "current",
            "context_turn_keys": [],
        },
        "requirements": [
            {
                "role": "answer",
                "origin": "user_text",
                "description": "取得普通员工适用的完整出差标准",
            }
        ],
        "clarification": {"question": "", "unresolved": []},
        "confidence": 0.94,
        "rationale": "用户询问企业制度",
    }
    value.update(updates)
    return value


class RagRouteDecisionSchemaTests(unittest.TestCase):
    def test_dynamic_schema_is_exact_and_limits_model_choices(self) -> None:
        schema = build_rag_route_decision_schema(
            allowed_intent_codes=["knowledge_qa", "general_chat"],
            available_turn_keys=["t1", "t2"],
        )

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(
            schema["properties"]["intent_code"]["enum"],
            ["knowledge_qa", "general_chat"],
        )
        query_schema = schema["properties"]["query_resolution"]
        self.assertFalse(query_schema["additionalProperties"])
        self.assertEqual(
            query_schema["properties"]["context_turn_keys"]["items"]["enum"],
            ["t1", "t2"],
        )
        self.assertNotIn(
            "uniqueItems",
            query_schema["properties"]["context_turn_keys"],
        )
        requirement_schema = schema["properties"]["requirements"]["items"]
        unresolved_schema = schema["properties"]["clarification"]["properties"][
            "unresolved"
        ]["items"]
        self.assertFalse(requirement_schema["additionalProperties"])
        self.assertFalse(unresolved_schema["additionalProperties"])

    def test_no_turn_candidates_schema_only_allows_empty_arrays(self) -> None:
        schema = build_rag_route_decision_schema(
            allowed_intent_codes=["other"],
            available_turn_keys=[],
        )
        candidate_schema = schema["properties"]["query_resolution"]["properties"][
            "context_turn_keys"
        ]
        self.assertEqual(candidate_schema["maxItems"], 0)
        self.assertNotIn("enum", candidate_schema["items"])

    def test_response_format_wraps_schema_as_strict_json_schema(self) -> None:
        response_format = build_rag_route_response_format(
            allowed_intent_codes=["knowledge_qa"],
            available_turn_keys=["t1"],
        )
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(
            response_format["json_schema"]["schema"]["properties"][
                "schema_version"
            ]["const"],
            ROUTE_DECISION_SCHEMA_VERSION,
        )


class RagRouteDecisionParserTests(unittest.TestCase):
    def test_parses_exact_followup_contract_and_round_trips(self) -> None:
        payload = _payload(
            relation="followup",
            query_resolution={"mode": "current", "context_turn_keys": ["t1"]},
            requirements=[
                {
                    "role": "bridge",
                    "origin": "semantically_entailed",
                    "description": "确定普通员工对应的出差职级",
                },
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": "取得该职级对应的完整出差标准",
                },
            ],
        )

        decision = parse_rag_route_decision(
            json.dumps(payload, ensure_ascii=False),
            allowed_intent_codes=["knowledge_qa", "other"],
            available_turn_keys=["t1", "t2"],
        )

        self.assertEqual(decision.relation, "followup")
        self.assertEqual(decision.query_resolution.context_turn_keys, ("t1",))
        self.assertEqual([item.role for item in decision.requirements], ["bridge", "answer"])
        self.assertEqual(decision.to_dict(), payload)

    def test_rejects_extra_or_missing_fields_at_every_level(self) -> None:
        cases = []
        top_extra = _payload()
        top_extra["need_retrieval"] = True
        cases.append(top_extra)

        query_extra = _payload()
        query_extra["query_resolution"]["rewritten_query"] = "不允许"
        cases.append(query_extra)

        requirement_extra = _payload()
        requirement_extra["requirements"][0]["importance"] = "required"
        cases.append(requirement_extra)

        unresolved_missing = _payload(
            readiness="needs_clarification",
            requirements=[],
            clarification={
                "question": "你指的是哪一轮？",
                "unresolved": [{"role": "context_turn", "reason": "missing"}],
            },
        )
        cases.append(unresolved_missing)

        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(RouteDecisionValidationError):
                    parse_rag_route_decision(
                        payload,
                        allowed_intent_codes=["knowledge_qa"],
                        available_turn_keys=["t1", "t2"],
                    )

    def test_rejects_markdown_wrapping_and_duplicate_json_keys(self) -> None:
        raw = json.dumps(_payload(), ensure_ascii=False)
        with self.assertRaises(RouteDecisionValidationError):
            parse_rag_route_decision(
                f"```json\n{raw}\n```",
                allowed_intent_codes=["knowledge_qa"],
            )

        duplicate = raw.replace(
            '"schema_version": "rag_route_decision.v1",',
            '"schema_version": "rag_route_decision.v1", '
            '"schema_version": "rag_route_decision.v1",',
            1,
        )
        with self.assertRaises(RouteDecisionValidationError):
            parse_rag_route_decision(
                duplicate,
                allowed_intent_codes=["knowledge_qa"],
            )

    def test_ready_and_clarification_shapes_are_mutually_exclusive(self) -> None:
        ready_with_question = _payload(
            clarification={
                "question": "还需要确认",
                "unresolved":[
                    {"role": "subject", "reason": "missing", "candidate_keys": []}
                ],
            }
        )
        clarification_without_slot = _payload(
            readiness="needs_clarification",
            clarification={"question": "请补充信息", "unresolved": []},
        )

        for payload in (ready_with_question, clarification_without_slot):
            with self.assertRaises(RouteDecisionValidationError):
                parse_rag_route_decision(
                    payload,
                    allowed_intent_codes=["knowledge_qa"],
                )

        valid = _payload(
            readiness="needs_clarification",
            requirements=[],
            clarification={
                "question": "请补充要查询的制度主题。",
                "unresolved": [
                    {"role": "subject", "reason": "missing", "candidate_keys": []}
                ],
            },
        )
        decision = parse_rag_route_decision(
            valid,
            allowed_intent_codes=["knowledge_qa"],
        )
        self.assertEqual(decision.readiness, "needs_clarification")

    def test_relation_and_query_binding_invariants_fail_closed(self) -> None:
        new_with_history = _payload(
            query_resolution={"mode": "current", "context_turn_keys": ["t1"]}
        )
        contextualize_without_history = _payload(
            readiness="needs_clarification",
            relation="followup",
            query_resolution={"mode": "contextualize", "context_turn_keys": []},
            requirements=[],
            clarification={
                "question": "你指的是哪一轮？",
                "unresolved": [
                    {"role": "context_turn", "reason": "missing", "candidate_keys": []}
                ],
            },
        )

        for payload in (
            new_with_history,
            contextualize_without_history,
        ):
            with self.assertRaises(RouteDecisionValidationError):
                parse_rag_route_decision(
                    payload,
                    allowed_intent_codes=["knowledge_qa"],
                    available_turn_keys=["t1"],
                )

        # relation only describes semantic continuity.  A self-contained
        # follow-up does not need a historical binding when the current query
        # already contains everything needed for execution.
        followup_without_history = _payload(relation="followup")
        decision = parse_rag_route_decision(
            followup_without_history,
            allowed_intent_codes=["knowledge_qa"],
            available_turn_keys=["t1"],
        )
        self.assertEqual(decision.relation, "followup")
        self.assertEqual(decision.query_resolution.context_turn_keys, ())

    def test_candidate_keys_are_request_local_unique_and_bounded(self) -> None:
        valid = _payload(
            relation="followup",
            query_resolution={
                "mode": "contextualize",
                "context_turn_keys": ["t1", "t2", "t3"],
            },
        )
        decision = parse_rag_route_decision(
            valid,
            allowed_intent_codes=["knowledge_qa"],
            available_turn_keys=["t1", "t2", "t3"],
        )
        self.assertEqual(
            decision.query_resolution.context_turn_keys,
            ("t1", "t2", "t3"),
        )

        invalid_cases = []
        for keys in (["t1", "t1"], ["t4"], ["message-id"], ["t1", "t2", "t3", "t4"]):
            payload = copy.deepcopy(valid)
            payload["query_resolution"]["context_turn_keys"] = keys
            invalid_cases.append(payload)
        for payload in invalid_cases:
            with self.subTest(keys=payload["query_resolution"]["context_turn_keys"]):
                with self.assertRaises(RouteDecisionValidationError):
                    parse_rag_route_decision(
                        payload,
                        allowed_intent_codes=["knowledge_qa"],
                        available_turn_keys=["t1", "t2", "t3"],
                    )

    def test_unresolved_candidate_reason_rules_are_strict(self) -> None:
        ambiguous = _payload(
            readiness="needs_clarification",
            relation="followup",
            requirements=[],
            clarification={
                "question": "你指的是哪一轮？",
                "unresolved": [
                    {
                        "role": "context_turn",
                        "reason": "ambiguous",
                        "candidate_keys": ["t1", "t2"],
                    }
                ],
            },
        )
        decision = parse_rag_route_decision(
            ambiguous,
            allowed_intent_codes=["knowledge_qa"],
            available_turn_keys=["t1", "t2"],
        )
        self.assertEqual(
            decision.clarification.unresolved[0].candidate_keys,
            ("t1", "t2"),
        )

        one_candidate = copy.deepcopy(ambiguous)
        one_candidate["clarification"]["unresolved"][0]["candidate_keys"] = ["t1"]
        missing_with_candidate = copy.deepcopy(ambiguous)
        missing_with_candidate["clarification"]["unresolved"][0]["reason"] = "missing"
        missing_decision = parse_rag_route_decision(
            missing_with_candidate,
            allowed_intent_codes=["knowledge_qa"],
            available_turn_keys=["t1", "t2"],
        )
        self.assertEqual(
            missing_decision.clarification.unresolved[0].candidate_keys,
            ("t1", "t2"),
        )

        unavailable_with_candidate = copy.deepcopy(ambiguous)
        unavailable_with_candidate["clarification"]["unresolved"][0][
            "reason"
        ] = "unavailable"
        for payload in (one_candidate, unavailable_with_candidate):
            with self.assertRaises(RouteDecisionValidationError):
                parse_rag_route_decision(
                    payload,
                    allowed_intent_codes=["knowledge_qa"],
                    available_turn_keys=["t1", "t2"],
                )

    def test_requirement_limit_and_confidence_are_enforced(self) -> None:
        too_many = _payload(
            requirements=[
                {
                    "role": "answer",
                    "origin": "user_text",
                    "description": f"目标 {index}",
                }
                for index in range(MAX_REQUIREMENTS + 1)
            ]
        )
        boolean_confidence = _payload(confidence=True)
        non_finite_confidence = _payload(confidence=float("nan"))
        for payload in (too_many, boolean_confidence, non_finite_confidence):
            with self.assertRaises(RouteDecisionValidationError):
                parse_rag_route_decision(
                    payload,
                    allowed_intent_codes=["knowledge_qa"],
                )

    def test_ready_route_requires_at_least_one_answer_target(self) -> None:
        no_requirements = _payload(requirements=[])
        bridge_only = _payload(
            requirements=[
                {
                    "role": "bridge",
                    "origin": "semantically_entailed",
                    "description": "先确定普通员工对应的职级",
                }
            ]
        )

        for payload in (no_requirements, bridge_only):
            with self.subTest(requirements=payload["requirements"]):
                with self.assertRaises(RouteDecisionValidationError):
                    parse_rag_route_decision(
                        payload,
                        allowed_intent_codes=["knowledge_qa"],
                    )


if __name__ == "__main__":
    unittest.main()
