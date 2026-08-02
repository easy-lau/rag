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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
