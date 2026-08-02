import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.query_analysis_contract import (
    QUERY_ANALYSIS_SCHEMA_VERSION,
    QueryAnalysisValidationError,
    build_query_analysis_response_format,
    build_query_analysis_schema,
    parse_query_analysis,
)
from core.query_analyzer import analyze_query


QUESTION = "普通员工的住宿标准、餐补和出差补贴分别是多少？"


def _ref(source: str, span: str, *, occurrence: int = 0, turn_key: str = "current") -> dict:
    start = -1
    offset = 0
    for _ in range(occurrence + 1):
        start = source.index(span, offset)
        offset = start + len(span)
    return {
        "turn_key": turn_key,
        "start": start,
        "end": start + len(span),
        "span": span,
    }


def _payload(*, question: str = QUESTION, **updates) -> dict:
    value = {
        "schema_version": QUERY_ANALYSIS_SCHEMA_VERSION,
        "relation": "new",
        "self_contained": True,
        "context_turn_keys": [],
        "confidence": 0.95,
        "diagnostic": "三个并列目标共享同一人员限定词。",
    }
    if "answer_candidates" not in updates or "bridge_candidates" not in updates:
        employee = _ref(question, "普通员工")
        value.setdefault("answer_candidates", [
            {
                "id": "a1",
                "target_source_ref": _ref(question, "住宿标准"),
                "qualifier_source_refs": [employee],
                "bridge_candidate_ids": ["b1"],
            },
            {
                "id": "a2",
                "target_source_ref": _ref(question, "餐补"),
                "qualifier_source_refs": [employee],
                "bridge_candidate_ids": ["b1"],
            },
            {
                "id": "a3",
                "target_source_ref": _ref(question, "出差补贴"),
                "qualifier_source_refs": [employee],
                "bridge_candidate_ids": ["b1"],
            },
        ])
        value.setdefault("bridge_candidates", [
            {
                "id": "b1",
                "subject_source_ref": employee,
            }
        ])
    value.update(updates)
    return value


class QueryAnalysisV2SchemaTests(unittest.TestCase):
    def test_dynamic_schema_is_exact_and_exposes_offset_source_refs(self):
        schema = build_query_analysis_schema(available_turn_keys=["t1", "t2"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(
            schema["properties"]["context_turn_keys"]["items"]["enum"],
            ["t1", "t2"],
        )
        source = schema["properties"]["answer_candidates"]["items"]["properties"][
            "target_source_ref"
        ]
        self.assertEqual(
            set(source["required"]),
            {"turn_key", "start", "end", "span"},
        )
        self.assertEqual(source["properties"]["start"]["minimum"], 0)
        self.assertFalse(
            schema["properties"]["bridge_candidates"]["items"][
                "additionalProperties"
            ]
        )
        response_format = build_query_analysis_response_format(
            available_turn_keys=["t1"]
        )
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])

    def test_parser_round_trips_source_anchored_candidate_graph(self):
        payload = _payload()
        analysis = parse_query_analysis(
            json.dumps(payload, ensure_ascii=False), current_question=QUESTION
        )
        self.assertEqual(analysis.schema_version, "query_analysis.v2")
        self.assertEqual(
            [item.id for item in analysis.answer_candidates], ["a1", "a2", "a3"]
        )
        self.assertEqual(
            analysis.answer_candidates[0].target_source_ref.span,
            "住宿标准",
        )
        self.assertEqual(
            analysis.bridge_candidates[0].subject_source_ref.span,
            "普通员工",
        )
        self.assertEqual(analysis.to_dict(), payload)
        self.assertEqual(analysis.safe_summary()["answer_candidate_count"], 3)
        self.assertEqual(analysis.safe_summary()["bridge_candidate_count"], 1)

    def test_parser_preserves_raw_whitespace_for_offset_validation(self):
        question = "普通员工  的餐补是多少？"
        payload = _payload(
            question=question,
            answer_candidates=[
                {
                    "id": "a1",
                    "target_source_ref": _ref(question, "餐补"),
                    "qualifier_source_refs": [_ref(question, "普通员工")],
                    "bridge_candidate_ids": ["b1"],
                }
            ],
            bridge_candidates=[
                {
                    "id": "b1",
                    "subject_source_ref": _ref(question, "普通员工"),
                }
            ],
            diagnostic="一个目标和一个限定词。",
        )
        analysis = parse_query_analysis(
            json.dumps(payload, ensure_ascii=False), current_question=question
        )
        self.assertEqual(
            analysis.answer_candidates[0].target_source_ref.start,
            question.index("餐补"),
        )

    def test_parser_rejects_execution_fields_and_nonliteral_or_wrong_offsets(self):
        extra = _payload()
        extra["answer_candidates"][0]["coverage_mode"] = "single"
        executable_bridge = _payload()
        executable_bridge["bridge_candidates"][0]["kind"] = "classification"
        guessed = _payload()
        guessed["answer_candidates"][0]["target_source_ref"]["span"] = "D级住宿标准"
        wrong_offset = _payload()
        wrong_offset["answer_candidates"][0]["target_source_ref"]["start"] += 1
        for raw in (extra, executable_bridge, guessed, wrong_offset):
            with self.subTest(raw=raw):
                with self.assertRaises(QueryAnalysisValidationError):
                    parse_query_analysis(
                        json.dumps(raw, ensure_ascii=False),
                        current_question=QUESTION,
                    )

    def test_target_must_come_from_current_and_history_is_exactly_route_bound(self):
        current = "那餐补呢？"
        history = "普通员工的住宿标准是多少？"
        employee = _ref(history, "普通员工", turn_key="t1")
        payload = _payload(
            question=current,
            relation="followup",
            self_contained=False,
            context_turn_keys=["t1"],
            answer_candidates=[
                {
                    "id": "a1",
                    "target_source_ref": _ref(current, "餐补"),
                    "qualifier_source_refs": [employee],
                    "bridge_candidate_ids": ["b1"],
                }
            ],
            bridge_candidates=[
                {"id": "b1", "subject_source_ref": employee}
            ],
            diagnostic="当前目标继承历史人员限定词。",
        )
        analysis = parse_query_analysis(
            json.dumps(payload, ensure_ascii=False),
            current_question=current,
            context_user_inputs={"t1": history},
        )
        self.assertFalse(analysis.self_contained)
        self.assertEqual(analysis.context_turn_keys, ("t1",))

        historical_target = copy.deepcopy(payload)
        historical_target["answer_candidates"][0]["target_source_ref"] = _ref(
            history, "住宿标准", turn_key="t1"
        )
        with self.assertRaisesRegex(QueryAnalysisValidationError, "当前输入"):
            parse_query_analysis(
                json.dumps(historical_target, ensure_ascii=False),
                current_question=current,
                context_user_inputs={"t1": history},
            )

        unbound = copy.deepcopy(payload)
        unbound["context_turn_keys"] = []
        with self.assertRaisesRegex(QueryAnalysisValidationError, "绑定历史上下文"):
            parse_query_analysis(
                json.dumps(unbound, ensure_ascii=False),
                current_question=current,
                context_user_inputs={"t1": history},
            )

    def test_bridge_subject_must_be_a_qualifier_of_every_referencing_answer(self):
        payload = _payload()
        payload["answer_candidates"][0]["qualifier_source_refs"] = []
        with self.assertRaisesRegex(QueryAnalysisValidationError, "bridge 的主体"):
            parse_query_analysis(
                json.dumps(payload, ensure_ascii=False), current_question=QUESTION
            )

    def test_parser_rejects_dangling_candidates_duplicate_targets_and_context_leakage(self):
        dangling = _payload()
        dangling["bridge_candidates"].append({
            "id": "b2",
            "subject_source_ref": _ref(QUESTION, "普通员工"),
        })
        duplicate_target = _payload()
        duplicate_target["answer_candidates"][1]["target_source_ref"] = _ref(
            QUESTION, "住宿标准"
        )
        history_leak = _payload(relation="followup", self_contained=False)
        history_leak["context_turn_keys"] = ["t1"]
        history_leak["answer_candidates"][0]["qualifier_source_refs"] = [
            _ref("经理的餐补是多少？", "经理", turn_key="t1")
        ]
        history_leak["bridge_candidates"][0]["subject_source_ref"] = _ref(
            "经理的餐补是多少？", "经理", turn_key="t1"
        )
        for raw, context in (
            (dangling, None),
            (duplicate_target, None),
            (history_leak, {"t1": "经理的餐补是多少？"}),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(QueryAnalysisValidationError):
                    parse_query_analysis(
                        json.dumps(raw, ensure_ascii=False),
                        current_question=QUESTION,
                        context_user_inputs=context,
                    )


class QueryAnalyzerV2Tests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, *, mode="active", timeout=1.0):
        return SimpleNamespace(
            rag_query_analyzer_mode=mode,
            rag_query_analyzer_timeout_seconds=timeout,
            intent_model="analysis-model",
            chat_model="chat-model",
            rag_trace_include_content=True,
            rag_trace_content_max_chars=50000,
        )

    def _response(self, content, finish_reason="stop"):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )],
            model="analysis-model",
            id="response-1",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    async def test_analyzer_uses_v2_strict_schema_and_returns_validated_graph(self):
        create = AsyncMock(return_value=self._response(
            json.dumps(_payload(), ensure_ascii=False)
        ))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        traces = []
        with (
            patch("core.query_analyzer.get_settings", return_value=self._settings()),
            patch("core.query_analyzer.get_client", return_value=client),
            patch(
                "core.query_analyzer.trace_event",
                side_effect=lambda event, **payload: traces.append((event, payload)),
            ),
        ):
            result = await analyze_query(question=QUESTION, trace_id="trace-1")
        self.assertTrue(result.accepted)
        self.assertEqual(
            result.analysis.safe_summary()["answer_candidate_count"], 3
        )
        request = create.await_args.kwargs
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        schema = request["response_format"]["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            "query_analysis.v2",
        )
        self.assertIn("start", request["messages"][0]["content"])
        self.assertEqual(
            [event for event, _ in traces],
            ["query.analysis.requested", "query.analysis.completed", "query.analysis.validated"],
        )

    async def test_analyzer_rejects_invalid_model_json_without_raising(self):
        create = AsyncMock(return_value=self._response("not-json"))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with (
            patch("core.query_analyzer.get_settings", return_value=self._settings()),
            patch("core.query_analyzer.get_client", return_value=client),
            patch("core.query_analyzer.trace_event"),
        ):
            result = await analyze_query(question=QUESTION)
        self.assertFalse(result.accepted)
        self.assertEqual(result.fallback_reason, "invalid_json")

    async def test_analyzer_times_out_and_cancels_the_provider_task(self):
        cancelled = asyncio.Event()

        async def wait_forever(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=wait_forever))
            )
        )
        with (
            patch("core.query_analyzer.get_settings", return_value=self._settings(timeout=0.1)),
            patch("core.query_analyzer.get_client", return_value=client),
            patch("core.query_analyzer.trace_event"),
        ):
            result = await analyze_query(question=QUESTION)
        self.assertFalse(result.accepted)
        self.assertEqual(result.fallback_reason, "timeout")
        self.assertTrue(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
