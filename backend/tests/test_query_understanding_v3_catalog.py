"""Contract tests for the bounded ``query_understanding.v3`` source catalog.

The model never receives, or is allowed to return, an offset-based source
reference.  It can only select server-issued span identifiers.  These tests
protect that boundary before the V3 compiler is wired into a chat request.
"""

from __future__ import annotations

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.query_understanding_v3_analyzer import analyze_query_understanding
from core.query_understanding_v3_catalog import (
    SourceSpanCatalogError,
    build_source_span_catalog,
)
from core.query_understanding_v3_contract import (
    QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
    QueryUnderstandingV3ValidationError,
    build_query_understanding_response_format,
    parse_query_understanding,
)


QUESTION = "普通员工  的住宿标准、餐补和出差补贴分别是多少？"


def _span_id(catalog, text, *, source_key="current"):
    return next(
        item.span_id
        for item in catalog.entries
        if item.source_key == source_key and item.text == text
    )


def _payload(catalog, **updates):
    employee = _span_id(catalog, "普通员工")
    value = {
        "schema_version": QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
        "answer_candidates": [
            {
                "id": "a1",
                "target_span_id": _span_id(catalog, "住宿标准"),
                "qualifier_span_ids": [employee],
            },
            {
                "id": "a2",
                "target_span_id": _span_id(catalog, "餐补"),
                "qualifier_span_ids": [employee],
            },
            {
                "id": "a3",
                "target_span_id": _span_id(catalog, "出差补贴"),
                "qualifier_span_ids": [employee],
            },
        ],
        "knowledge_request": {
            "resource": "document_content",
            "operation": "answer",
            "filter_span_ids": [],
            "group_by": "none",
            "status_filter": "any",
            "result_handles": [],
            "answer_form": "fact",
        },
    }
    value.update(updates)
    return value


class SourceSpanCatalogTests(unittest.TestCase):
    def test_catalog_is_deterministic_and_model_view_has_no_offsets_or_source_texts(self):
        first = build_source_span_catalog(current_question=QUESTION)
        second = build_source_span_catalog(current_question=QUESTION)

        self.assertEqual(
            [(item.span_id, item.source_key, item.start, item.end, item.text)
             for item in first.entries],
            [(item.span_id, item.source_key, item.start, item.end, item.text)
             for item in second.entries],
        )
        self.assertEqual(_span_id(first, "普通员工"), "s_current_002")
        self.assertEqual(_span_id(first, "住宿标准"), "s_current_003")
        self.assertEqual(_span_id(first, "餐补"), "s_current_004")
        self.assertEqual(_span_id(first, "出差补贴"), "s_current_005")
        for item in first.entries:
            self.assertEqual(
                item.text,
                first.source_text_for(item.source_key)[item.start:item.end],
            )

        model_view = first.model_payload()
        self.assertEqual(model_view["schema_version"], "source_span_catalog.v1")
        self.assertTrue(model_view["spans"])
        self.assertEqual(
            set(model_view["spans"][0]), {"span_id", "source", "text"}
        )
        serialized = json.dumps(model_view, ensure_ascii=False)
        self.assertNotIn('"start"', serialized)
        self.assertNotIn('"end"', serialized)
        self.assertNotIn('"source_key"', serialized)

    def test_catalog_only_accepts_explicit_route_context_user_fragments(self):
        catalog = build_source_span_catalog(
            current_question="那餐补呢？",
            route_context=[{
                "candidate_key": "t1",
                "user_input": "普通员工的住宿标准是多少？",
                "assistant_answer": "D级，住宿上限450元。",
            }],
        )
        historical = [item for item in catalog.entries if item.source_key == "t1"]
        self.assertTrue(historical)
        self.assertIn("普通员工", [item.text for item in historical])
        self.assertNotIn("D级", json.dumps(catalog.model_payload(), ensure_ascii=False))
        self.assertEqual(catalog.resolve(historical[0].span_id).source_kind, "route_context")

        with self.assertRaisesRegex(SourceSpanCatalogError, "候选键"):
            build_source_span_catalog(
                current_question="当前问题",
                route_context=[{"candidate_key": "message-42", "user_input": "历史问题"}],
            )

    def test_chinese_and_unicode_whitespace_keep_source_boundaries_stable(self):
        question = "　普通员工\u3000的 餐补 是多少？  "
        catalog = build_source_span_catalog(current_question=question)
        self.assertEqual(_span_id(catalog, "普通员工"), "s_current_002")
        self.assertEqual(_span_id(catalog, "餐补"), "s_current_003")
        self.assertEqual(
            [(item.span_id, item.start, item.end, item.text) for item in catalog.entries],
            [(item.span_id, item.start, item.end, item.text)
            for item in build_source_span_catalog(current_question=question).entries],
        )

    def test_compact_relation_exposes_server_derived_target_and_qualifier_spans(self):
        """Catalogue grammar must not force a compact relation into one span."""

        catalog = build_source_span_catalog(current_question="普通员工对应什么职级")
        self.assertIn("普通员工", [item.text for item in catalog.current_entries])
        self.assertIn("职级", [item.text for item in catalog.current_entries])

    def test_exact_range_lookup_binds_only_the_catalogued_contextual_spans(self):
        catalog = build_source_span_catalog(
            current_question="那住宿呢",
            route_context=[{
                "candidate_key": "t1",
                "user_input": "普通员工的餐饮补贴是多少",
            }],
        )

        current = catalog.find_exact_span(
            source_key="current",
            start=1,
            end=3,
        )
        historical = catalog.find_exact_span(
            source_key="t1",
            start=0,
            end=4,
        )
        self.assertIsNotNone(current)
        self.assertIsNotNone(historical)
        self.assertEqual(current.text, "住宿")
        self.assertEqual(historical.text, "普通员工")
        self.assertIsNone(
            catalog.find_exact_span(source_key="t1", start=1, end=4),
            "nearby text must not be used as an approximate source binding",
        )

    def test_catalog_question_exposes_exact_metadata_filter_literal(self):
        catalog = build_source_span_catalog(
            current_question="我现在有关于云枢配置的知识库有几个文章",
        )

        self.assertIn(
            "云枢配置",
            [item.text for item in catalog.current_entries],
        )
        self.assertNotIn(
            "于云枢配置",
            [item.text for item in catalog.current_entries],
        )


class QueryUnderstandingV3ContractTests(unittest.TestCase):
    def test_parser_round_trips_catalog_ids_without_model_source_authority(self):
        catalog = build_source_span_catalog(current_question=QUESTION)
        payload = _payload(catalog)
        analysis = parse_query_understanding(
            json.dumps(payload, ensure_ascii=False), catalog=catalog
        )

        self.assertEqual(analysis.schema_version, QUERY_UNDERSTANDING_V3_SCHEMA_VERSION)
        self.assertEqual(
            [item.target_span_id for item in analysis.answer_candidates],
            [_span_id(catalog, "住宿标准"), _span_id(catalog, "餐补"), _span_id(catalog, "出差补贴")],
        )
        self.assertEqual(
            analysis.answer_candidates[0].target_span.text,
            "住宿标准",
        )
        self.assertEqual(analysis.to_dict(), payload)
        self.assertEqual(analysis.safe_summary()["answer_candidate_count"], 3)

        response_format = build_query_understanding_response_format(catalog=catalog)
        schema = response_format["json_schema"]["schema"]
        self.assertTrue(response_format["json_schema"]["strict"])
        target = schema["properties"]["answer_candidates"]["items"]["properties"]["target_span_id"]
        self.assertEqual(
            set(target), {"type", "enum"}
        )
        self.assertTrue(all(item.startswith("s_current_") for item in target["enum"]))
        serialized_schema = json.dumps(schema, ensure_ascii=False)
        for forbidden in (
            '"diagnostic"', '"start"', '"end"', '"text"', '"scope"',
            '"bridge"', '"relation"', '"self_contained"', '"confidence"',
        ):
            self.assertNotIn(forbidden, serialized_schema)

    def test_parser_rejects_unknown_cross_history_and_forged_source_fields(self):
        catalog = build_source_span_catalog(
            current_question="那餐补呢？",
            route_context=[{
                "candidate_key": "t1",
                "user_input": "普通员工的住宿标准是多少？",
            }],
        )
        history_employee = _span_id(catalog, "普通员工", source_key="t1")
        historical_target = _span_id(catalog, "住宿标准", source_key="t1")
        current_target = _span_id(catalog, "餐补")
        base = {
            "schema_version": QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
            "answer_candidates": [{
                "id": "a1",
                "target_span_id": current_target,
                "qualifier_span_ids": [history_employee],
            }],
        }
        accepted = parse_query_understanding(
            json.dumps(base, ensure_ascii=False), catalog=catalog
        )
        self.assertEqual(accepted.referenced_context_keys, ("t1",))
        self.assertFalse(accepted.self_contained)
        self.assertEqual(accepted.relation, "followup")

        unknown = copy.deepcopy(base)
        unknown["answer_candidates"][0]["target_span_id"] = "s_current_999"
        historical = copy.deepcopy(base)
        historical["answer_candidates"][0]["target_span_id"] = historical_target
        forged_text = copy.deepcopy(base)
        forged_text["answer_candidates"][0]["target_text"] = "餐补"
        forged_offset = copy.deepcopy(base)
        forged_offset["answer_candidates"][0]["start"] = 0
        forged_kb = copy.deepcopy(base)
        forged_kb["knowledge_base_id"] = "kb-1"
        forged_fact = copy.deepcopy(base)
        forged_fact["amount"] = 100
        forged_original = copy.deepcopy(base)
        forged_original["answer_candidates"][0]["source_text"] = "餐补"
        forged_relation = copy.deepcopy(base)
        forged_relation["relation"] = "followup"
        forged_confidence = copy.deepcopy(base)
        forged_confidence["confidence"] = 0.9
        for raw, expected in (
            (unknown, "span_id"),
            (historical, "当前输入"),
            (forged_text, "字段不精确"),
            (forged_offset, "字段不精确"),
            (forged_kb, "字段不精确"),
            (forged_fact, "字段不精确"),
            (forged_original, "字段不精确"),
            (forged_relation, "字段不精确"),
            (forged_confidence, "字段不精确"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(QueryUnderstandingV3ValidationError, expected):
                    parse_query_understanding(
                        json.dumps(raw, ensure_ascii=False), catalog=catalog
                    )

    def test_parser_safely_normalizes_json_object_transport_noise(self):
        catalog = build_source_span_catalog(
            current_question="云枢7修改密码如何配置",
        )
        current_target = next(iter(catalog.current_span_ids))
        payload = {
            "schema_version": QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
            "answer_candidates": [{
                "target_span_id": current_target,
                "qualifier_span_ids": [current_target],
            }],
        }

        analysis = parse_query_understanding(
            json.dumps(payload, ensure_ascii=False),
            catalog=catalog,
        )

        self.assertEqual(analysis.answer_candidates[0].id, "a1")
        self.assertEqual(analysis.answer_candidates[0].qualifier_span_ids, ())
        self.assertTrue(analysis.self_contained)

    def test_catalog_capability_is_source_bound_and_contains_no_object_ids(self):
        catalog = build_source_span_catalog(
            current_question="我现在有关于云枢配置的知识库有几个文章",
        )
        payload = {
            "schema_version": QUERY_UNDERSTANDING_V3_SCHEMA_VERSION,
            "answer_candidates": [{
                "id": "a1",
                "target_span_id": _span_id(catalog, "知识库有几个文章"),
                "qualifier_span_ids": [],
            }],
            "knowledge_request": {
                "resource": "document_catalog",
                "operation": "count",
                "filter_span_ids": [_span_id(catalog, "云枢配置")],
                "group_by": "none",
                "status_filter": "any",
            },
        }

        analysis = parse_query_understanding(
            json.dumps(payload, ensure_ascii=False),
            catalog=catalog,
        )

        self.assertTrue(analysis.knowledge_request.is_catalog_operation)
        self.assertEqual(analysis.knowledge_request.operation, "count")
        self.assertEqual(analysis.knowledge_request.filter_terms, ("云枢配置",))
        serialized = json.dumps(
            build_query_understanding_response_format(catalog=catalog),
            ensure_ascii=False,
        )
        self.assertNotIn("kb_id", serialized)
        self.assertNotIn("doc_id", serialized)
        self.assertNotIn("sql", serialized.casefold())

    def test_legacy_v3_response_can_only_fall_back_to_content_answer(self):
        catalog = build_source_span_catalog(current_question=QUESTION)
        legacy = _payload(catalog)
        legacy.pop("knowledge_request")

        analysis = parse_query_understanding(
            json.dumps(legacy, ensure_ascii=False),
            catalog=catalog,
        )

        self.assertFalse(analysis.knowledge_request.is_catalog_operation)
        self.assertEqual(analysis.knowledge_request.operation, "answer")

    def test_invalid_catalog_capability_combinations_are_rejected(self):
        catalog = build_source_span_catalog(current_question=QUESTION)
        for request in (
            {
                "resource": "document_catalog",
                "operation": "group",
                "filter_span_ids": [],
                "group_by": "none",
                "status_filter": "any",
            },
            {
                "resource": "document_content",
                "operation": "count",
                "filter_span_ids": [],
                "group_by": "none",
                "status_filter": "any",
            },
        ):
            payload = _payload(catalog, knowledge_request=request)
            with self.subTest(request=request):
                with self.assertRaisesRegex(
                    QueryUnderstandingV3ValidationError,
                    "组合非法",
                ):
                    parse_query_understanding(
                        json.dumps(payload, ensure_ascii=False),
                        catalog=catalog,
                    )

    def test_parser_rejects_duplicate_and_overlapping_model_selections(self):
        catalog = build_source_span_catalog(current_question=QUESTION)
        payload = _payload(catalog)
        employee = _span_id(catalog, "普通员工")

        duplicate_qualifier = copy.deepcopy(payload)
        duplicate_qualifier["answer_candidates"][0]["qualifier_span_ids"] = [
            employee, employee,
        ]
        duplicate_target = copy.deepcopy(payload)
        duplicate_target["answer_candidates"][1]["target_span_id"] = (
            duplicate_target["answer_candidates"][0]["target_span_id"]
        )
        overlapping = copy.deepcopy(payload)
        full_turn = next(
            item.span_id
            for item in catalog.entries
            if item.source_key == "current" and item.text == QUESTION
        )
        overlapping["answer_candidates"] = [{
            "id": "a1",
            "target_span_id": full_turn,
            "qualifier_span_ids": [employee],
        }]
        for raw, expected in (
            (duplicate_qualifier, "重复"),
            (duplicate_target, "重复"),
            (overlapping, "重叠"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(QueryUnderstandingV3ValidationError, expected):
                    parse_query_understanding(
                        json.dumps(raw, ensure_ascii=False), catalog=catalog
                    )


class QueryUnderstandingV3AnalyzerTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinal_result_reference_uses_deterministic_handle_without_model(self):
        route_context = [{
            "candidate_key": "t1",
            "user_input": "列出当前文章",
            "assistant_answer": "1. A\n2. B",
            "result_items": [
                {
                    "handle": "r_t1_001",
                    "ordinal": 1,
                    "resource": "document",
                    "label": "A.md",
                    "status": "ready",
                },
                {
                    "handle": "r_t1_002",
                    "ordinal": 2,
                    "resource": "document",
                    "label": "B.md",
                    "status": "ready",
                },
            ],
        }]
        settings = SimpleNamespace(
            rag_query_analyzer_mode="active",
            rag_query_analyzer_timeout_seconds=1.0,
            intent_model="slow-model",
            chat_model="",
        )
        with (
            patch("core.query_understanding_v3_analyzer.get_settings", return_value=settings),
            patch("core.query_understanding_v3_analyzer.get_client") as get_client,
            patch("core.query_understanding_v3_analyzer.trace_event"),
        ):
            result = await analyze_query_understanding(
                question="我想看第一个文章",
                route_context=route_context,
            )

        self.assertTrue(result.accepted)
        self.assertEqual(result.origin, "deterministic")
        self.assertEqual(result.analysis.relation, "followup")
        self.assertTrue(result.analysis.knowledge_request.is_result_operation)
        self.assertEqual(result.analysis.knowledge_request.operation, "read")
        self.assertEqual(
            result.analysis.knowledge_request.result_handles,
            ("r_t1_001",),
        )
        get_client.assert_not_called()

    async def test_analyzer_sends_only_catalog_and_parses_span_id_response(self):
        catalog = build_source_span_catalog(current_question=QUESTION)
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(_payload(catalog), ensure_ascii=False)),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            model="test-model",
            id="response-1",
        )
        recorded = {}

        class FakeCompletions:
            async def create(self, **kwargs):
                recorded.update(kwargs)
                return response

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions()),
        )
        settings = SimpleNamespace(
            rag_query_analyzer_mode="active",
            rag_query_analyzer_timeout_seconds=1.0,
            intent_model="test-model",
            chat_model="",
        )
        with (
            patch("core.query_understanding_v3_analyzer.get_settings", return_value=settings),
            patch("core.query_understanding_v3_analyzer.get_client", return_value=client),
            patch("core.query_understanding_v3_analyzer.trace_event"),
        ):
            result = await analyze_query_understanding(question=QUESTION)

        self.assertTrue(result.accepted)
        self.assertEqual(result.analysis.answer_candidates[0].target_span.text, "住宿标准")
        outbound = json.loads(recorded["messages"][1]["content"])
        self.assertEqual(set(outbound), {"span_catalog"})
        self.assertNotIn("user_question", outbound)
        self.assertNotIn('"start"', recorded["messages"][1]["content"])
        self.assertIn("span_id", recorded["messages"][0]["content"])
        self.assertIn("json object", recorded["messages"][0]["content"].casefold())

    async def test_analyzer_retries_json_object_with_json_instruction(self):
        catalog = build_source_span_catalog(current_question=QUESTION)
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(_payload(catalog), ensure_ascii=False)),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            model="test-model",
            id="response-json-object",
        )
        calls = []

        class UnsupportedSchema(Exception):
            status_code = 400

            def __init__(self):
                super().__init__("This response_format type is unavailable now")
                self.body = {"error": {"message": str(self)}}

        class FakeCompletions:
            async def create(self, **kwargs):
                calls.append(kwargs)
                if kwargs.get("response_format", {}).get("type") == "json_schema":
                    raise UnsupportedSchema()
                return response

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        settings = SimpleNamespace(
            rag_query_analyzer_mode="active",
            rag_query_analyzer_timeout_seconds=1.0,
            intent_model="test-model",
            chat_model="",
            llm_base_url="https://llm.example/v1",
        )
        with (
            patch("core.query_understanding_v3_analyzer.get_settings", return_value=settings),
            patch("core.query_understanding_v3_analyzer.get_client", return_value=client),
            patch("core.query_understanding_v3_analyzer.trace_event"),
        ):
            result = await analyze_query_understanding(question=QUESTION)

        self.assertTrue(result.accepted)
        self.assertEqual(result.structured_output_mode, "json_object")
        self.assertTrue(result.json_object_fallback_used)
        self.assertEqual(calls[-1]["response_format"], {"type": "json_object"})
        self.assertTrue(
            any(
                message.get("role") == "system"
                and "json" in str(message.get("content") or "").casefold()
                for message in calls[-1]["messages"]
                if isinstance(message, dict)
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
