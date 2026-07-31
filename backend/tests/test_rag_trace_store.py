"""RAG trace summary extraction and route authorization regression tests."""

import inspect
import asyncio
import json
import unittest
import uuid
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.routing import APIRoute

from api.rag_traces import (
    _encode_bounded_trace_export,
    _load_bounded_export_events,
    _rag_trace_export_payload,
    _require_trace_content_access,
    _run_out,
    _utc_filter,
    export_rag_trace,
    router,
)
from core.permissions import LOG_READ, MENU_LOGIN_LOGS, MENU_RAG_TRACES, effective_permissions
import core.rag_trace_store as trace_store
from core.rag_trace_store import _bounded, _queue_safe_record, _summary_updates
from models.db_models import RagTraceEvent, RagTraceRun


def _route(method: str, path: str) -> APIRoute:
    for route in router.routes:
        if isinstance(route, APIRoute) and route.path == path and method in route.methods:
            return route
    raise AssertionError(f"route not found: {method} {path}")


def _required_permission_keys(route: APIRoute) -> set[str]:
    keys: set[str] = set()

    def visit(dependant) -> None:
        for child in dependant.dependencies:
            call = child.call
            if inspect.isfunction(call):
                key = inspect.getclosurevars(call).nonlocals.get("key")
                if isinstance(key, str):
                    keys.add(key)
            visit(child)

    visit(route.dependant)
    return keys


class _TraceRunResult:
    def __init__(self, value):
        self.value = value

    def one_or_none(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value


class _TraceEventsResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return self

    def all(self):
        return self.values


class _TraceRowsResult:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class _TraceExportDb:
    def __init__(self, run_row, events):
        metadata = [
            (
                event.id,
                event.sequence,
                event.event,
                len(json.dumps(event.payload, ensure_ascii=False).encode("utf-8")),
            )
            for event in events
        ]
        self.results = iter((
            _TraceRunResult(run_row),
            _TraceRowsResult(metadata),
            _TraceEventsResult(events),
        ))
        self.commits = 0

    async def execute(self, _query):
        return next(self.results)

    async def commit(self):
        self.commits += 1


class _AuditStub:
    def __init__(self):
        self.calls = []

    def log(self, db, action, **kwargs):
        self.calls.append((db, action, kwargs))


class _PersistSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def get(self, _model, _key):
        return None

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commits += 1


def _stored_trace(*, content_included: bool = False):
    timestamp = datetime(2026, 7, 30, 10, 0, tzinfo=UTC)
    run = RagTraceRun(
        trace_id="trace-export-1",
        request_kind="chat",
        user_id=None,
        conversation_id=uuid.UUID("33c21ffe-80c3-4905-9409-b50425e01ad2"),
        status="success",
        current_stage="chat.response",
        event_count=2,
        observed_event_count=2,
        storage_omitted_event_count=0,
        storage_truncated=False,
        content_included=content_included,
        input_preview="已保存的问题" if content_included else None,
        output_preview="已保存的回答" if content_included else None,
        evidence_status="hit",
        selected_kb_count=1,
        hit_count=2,
        duration_ms=1250,
        started_at=timestamp,
        completed_at=timestamp + timedelta(seconds=2),
        updated_at=timestamp + timedelta(seconds=2),
    )
    first_payload = {
        "trace_schema_version": 1,
        "app_version": "v1.2.3",
        "app_revision": "abc123",
        "event": "conversation.context_resolved",
        "trace_id": run.trace_id,
        "question_sha256": "a" * 64,
        "standalone_query_sha256": "a" * 64,
        "is_followup": True,
        "followup_reason": "missing_action_object",
    }
    if content_included:
        first_payload["question"] = "已保存的问题"
    events = [
        RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=1,
            event="conversation.context_resolved",
            payload=first_payload,
            created_at=timestamp,
        ),
        RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=2,
            event="intent.routing_decision",
            payload={
                "trace_schema_version": 1,
                "app_version": "v1.2.3",
                "app_revision": "abc123",
                "prompt_version": "2026-07-30.v2",
                "event": "intent.routing_decision",
                "trace_id": run.trace_id,
                "intent": {
                    "intent_code": "knowledge_qa",
                    "retrieval_policy": "required",
                    "need_retrieval": True,
                },
                "decision_reason": "classified_retrieval",
            },
            created_at=timestamp + timedelta(milliseconds=10),
        ),
    ]
    return run, events, timestamp


class RagTraceStoreTests(unittest.TestCase):
    def test_request_and_response_build_safe_list_summary(self) -> None:
        request = _summary_updates({
            "event": "chat.request",
            "timestamp": "2026-07-30T10:00:00+00:00",
            "question": "  云枢 8.6\n如何配置  ",
            "selected_kb_ids": ["kb-1", "kb-2"],
        })
        response = _summary_updates({
            "event": "chat.response",
            "timestamp": "2026-07-30T10:00:02+00:00",
            "answer": "回答正文",
            "evidence_status": "hit",
            "displayed_result_count": 3,
            "context_evidence_count": 2,
            "hit_count": 2,
            "direct_evidence_count": 2,
            "total_ms": 2150,
        })

        self.assertEqual(request["request_kind"], "chat")
        self.assertEqual(request["input_preview"], "云枢 8.6 如何配置")
        self.assertEqual(request["selected_kb_count"], 2)
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["output_preview"], "回答正文")
        self.assertEqual(response["hit_count"], 2)
        self.assertEqual(response["duration_ms"], 2150)

    def test_trace_hit_count_never_falls_back_to_display_or_candidate_lengths(self) -> None:
        display_only = _summary_updates({
            "event": "evidence.selection",
            "displayed_result_count": 5,
            "candidate_count": 9,
            "results": [{}, {}, {}, {}, {}],
            "sources": [{}, {}],
        })
        direct = _summary_updates({
            "event": "chat.response",
            "evidence_status": "partial",
            "displayed_result_count": 5,
            "context_evidence_count": 2,
            "direct_evidence_count": 1,
        })
        no_hit = _summary_updates({
            "event": "chat.response",
            "evidence_status": "no_hit",
            "displayed_result_count": 5,
            "direct_evidence_count": 3,
        })

        self.assertNotIn("hit_count", display_only)
        self.assertEqual(direct["hit_count"], 1)
        self.assertEqual(no_hit["hit_count"], 0)

    def test_error_event_is_terminal_and_sanitized_metadata_is_bounded(self) -> None:
        result = _summary_updates({
            "event": "search_test.error",
            "timestamp": "2026-07-30T10:00:02+00:00",
            "total_ms": 900,
        })
        bounded = _bounded({"nested": ["x" * 60_000]})

        self.assertEqual(result["status"], "error")
        self.assertIsNotNone(result["completed_at"])
        self.assertTrue(bounded["nested"][0].endswith("…[truncated]"))

    def test_recoverable_stage_error_does_not_finish_the_request(self) -> None:
        result = _summary_updates({
            "event": "retrieval.error",
            "timestamp": "2026-07-30T10:00:01+00:00",
        })

        self.assertNotIn("status", result)
        self.assertNotIn("completed_at", result)

    def test_cancelled_stream_is_immediately_interrupted(self) -> None:
        result = _summary_updates({
            "event": "chat.cancelled",
            "timestamp": "2026-07-30T10:00:01+00:00",
        })

        self.assertEqual(result["status"], "interrupted")
        self.assertIsNotNone(result["completed_at"])

    def test_queue_record_has_a_hard_encoded_size_boundary(self) -> None:
        settings = SimpleNamespace(
            rag_trace_content_max_chars=50_000,
            rag_trace_max_event_bytes=16_384,
        )
        with patch("core.rag_trace_store.get_settings", return_value=settings):
            result = _queue_safe_record({
                "event": "generation.context",
                "trace_id": "trace-large",
                "context": "云" * 50_000,
                "context_chars": 50_000,
                "context_sha256": "a" * 64,
                "displayed_result_count": 7,
            })

        encoded = json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertLessEqual(len(encoded), settings.rag_trace_max_event_bytes)
        self.assertTrue(result["persistence_payload_truncated"])
        self.assertEqual(result["trace_id"], "trace-large")
        self.assertEqual(result["displayed_result_count"], 7)

    def test_queue_reserves_fifo_capacity_for_terminal_events(self) -> None:
        settings = SimpleNamespace(
            rag_trace_content_max_chars=50_000,
            rag_trace_max_event_bytes=16_384,
        )
        previous_queue = trace_store._queue
        previous_accepting = trace_store._accepting
        previous_dropped = trace_store._dropped_events
        queue = asyncio.Queue(maxsize=100)
        for index in range(90):
            queue.put_nowait({"event": "retrieval.candidate", "trace_id": f"seed-{index}"})
        try:
            trace_store._queue = queue
            trace_store._accepting = True
            trace_store._dropped_events = 0
            with patch("core.rag_trace_store.get_settings", return_value=settings):
                trace_store.enqueue_trace_record({
                    "event": "retrieval.candidate",
                    "trace_id": "non-terminal",
                })
                self.assertEqual(queue.qsize(), 90)
                trace_store.enqueue_trace_record({
                    "event": "chat.response",
                    "trace_id": "terminal",
                })
            self.assertEqual(queue.qsize(), 91)
            self.assertEqual(queue.get_nowait()["trace_id"], "seed-0")
            terminal = list(queue._queue)[-1]
            self.assertEqual(terminal["event"], "chat.response")
            self.assertEqual(terminal["trace_id"], "terminal")
        finally:
            trace_store._queue = previous_queue
            trace_store._accepting = previous_accepting
            trace_store._dropped_events = previous_dropped

    def test_terminal_event_evicts_oldest_nonterminal_when_queue_is_full(self) -> None:
        settings = SimpleNamespace(
            rag_trace_content_max_chars=50_000,
            rag_trace_max_event_bytes=16_384,
        )
        previous_queue = trace_store._queue
        previous_accepting = trace_store._accepting
        previous_dropped = trace_store._dropped_events
        queue = asyncio.Queue(maxsize=100)
        queue.put_nowait({"event": "retrieval.candidate", "trace_id": "oldest-detail"})
        for index in range(99):
            queue.put_nowait({"event": "chat.response", "trace_id": f"terminal-{index}"})
        try:
            trace_store._queue = queue
            trace_store._accepting = True
            trace_store._dropped_events = 0
            with patch("core.rag_trace_store.get_settings", return_value=settings):
                trace_store.enqueue_trace_record({
                    "event": "chat.cancelled",
                    "trace_id": "latest-terminal",
                })

            retained = list(queue._queue)
            self.assertEqual(queue.qsize(), 100)
            self.assertNotIn("oldest-detail", {item["trace_id"] for item in retained})
            self.assertEqual(retained[-1]["trace_id"], "latest-terminal")
            self.assertEqual(trace_store._dropped_events, 1)
        finally:
            trace_store._queue = previous_queue
            trace_store._accepting = previous_accepting
            trace_store._dropped_events = previous_dropped

    def test_log_read_derives_both_read_only_admin_menus(self) -> None:
        permissions = set(effective_permissions([LOG_READ]))
        self.assertIn(MENU_LOGIN_LOGS, permissions)
        self.assertIn(MENU_RAG_TRACES, permissions)

    def test_trace_list_and_detail_require_log_read(self) -> None:
        self.assertEqual(_required_permission_keys(_route("GET", "/rag-traces")), {LOG_READ})
        self.assertEqual(_required_permission_keys(_route("GET", "/rag-traces/{trace_id}")), {LOG_READ})
        self.assertEqual(
            _required_permission_keys(_route("GET", "/rag-traces/{trace_id}/export")),
            {LOG_READ},
        )

    def test_trace_export_is_ai_friendly_and_uses_only_stored_sanitized_rows(self) -> None:
        run, events, timestamp = _stored_trace(content_included=False)

        payload = _rag_trace_export_payload(
            run,
            events,
            exported_at=timestamp + timedelta(minutes=1),
        )

        self.assertEqual(payload["export_schema_version"], 1)
        # 导出文件自身格式仍是 v1，同时明确保留历史事件的 Trace v1，
        # 新旧调用链可以识别但不能混合统计。
        self.assertEqual(
            payload["diagnostic_index"]["versions"]["trace_schema_version"],
            [1],
        )
        self.assertEqual(payload["trace"]["trace_id"], run.trace_id)
        self.assertNotIn("username", payload["trace"])
        self.assertFalse(payload["data_policy"]["content_included"])
        self.assertTrue(payload["data_policy"]["no_external_rehydration"])
        self.assertNotIn("question", payload["events"][0]["payload"])
        self.assertEqual(payload["events"][0]["trace_id"], run.trace_id)
        self.assertEqual(payload["events"][0]["payload"], events[0].payload)
        self.assertEqual(payload["diagnostic_index"]["stage_sequences"], {
            "conversation": [1],
            "intent": [2],
        })
        self.assertEqual(
            payload["diagnostic_index"]["versions"]["prompt_version"],
            ["2026-07-30.v2"],
        )
        self.assertTrue(
            payload["diagnostic_index"]["integrity"]["summary_matches_persisted_rows"]
        )
        self.assertFalse(payload["diagnostic_index"]["integrity"]["export_truncated"])
        snapshot = payload["diagnostic_index"]["snapshot"]
        self.assertEqual(snapshot["conversation_context"]["standalone_query_sha256"], "a" * 64)
        self.assertEqual(
            snapshot["routing_decision"]["intent"]["retrieval_policy"],
            "required",
        )
        self.assertIsNone(snapshot["retrieval_expansion_plan"])
        self.assertIsNone(snapshot["retrieval_document_scoped_result"])
        self.assertIsNone(snapshot["retrieval_structure_expansion"])
        self.assertIsNone(snapshot["retrieval_expansion_result"])
        self.assertIsNone(snapshot["rerank_joint_result"])
        self.assertEqual(snapshot["rerank_joint_history"], [])
        self.assertIsNone(snapshot["evidence_coverage"])
        self.assertEqual(snapshot["evidence_coverage_history"], [])
        self.assertEqual(payload["ai_analysis_guide"]["expected_sections"][-1], "优化建议及回归测试")
        self.assertIn("不可信数据", payload["ai_analysis_guide"]["untrusted_data_warning"])

    def test_diagnostic_snapshot_keeps_algorithm_and_prompt_fingerprints(self) -> None:
        run, events, timestamp = _stored_trace(content_included=False)
        run.event_count = 5
        run.observed_event_count = 5
        events.extend([
            RagTraceEvent(
                id=uuid.uuid4(),
                trace_id=run.trace_id,
                sequence=3,
                event="retrieval.plan",
                payload={
                    "event": "retrieval.plan",
                    "retrieval_algorithm": "vector_fts_trigram_rrf",
                    "rrf_k": 60,
                    "trigram_min_score": 0.18,
                    "rerank_candidate_min": 12,
                    "rerank_candidate_multiplier": 4,
                    "rerank_candidate_max": 60,
                },
                created_at=timestamp,
            ),
            RagTraceEvent(
                id=uuid.uuid4(),
                trace_id=run.trace_id,
                sequence=4,
                event="rerank.completed",
                payload={
                    "event": "rerank.completed",
                    "model": "chat-model",
                    "prompt_version": "rerank-v2",
                    "topic_relevance_threshold": 0.62,
                    "answer_support_threshold": 0.78,
                },
                created_at=timestamp,
            ),
            RagTraceEvent(
                id=uuid.uuid4(),
                trace_id=run.trace_id,
                sequence=5,
                event="generation.context",
                payload={
                    "event": "generation.context",
                    "model": "chat-model",
                    "temperature": 0.2,
                    "max_tokens": 2048,
                    "request_timeout_seconds": 45,
                    "max_attempts": 2,
                    "context_sha256": "b" * 64,
                    "system_prompt_sha256": "c" * 64,
                },
                created_at=timestamp,
            ),
        ])

        snapshot = _rag_trace_export_payload(
            run,
            events,
            exported_at=timestamp,
        )["diagnostic_index"]["snapshot"]

        self.assertEqual(snapshot["retrieval_plan"]["retrieval_algorithm"], "vector_fts_trigram_rrf")
        self.assertEqual(snapshot["retrieval_plan"]["rrf_k"], 60)
        self.assertEqual(snapshot["rerank_result"]["prompt_version"], "rerank-v2")
        self.assertEqual(snapshot["rerank_result"]["answer_support_threshold"], 0.78)
        self.assertEqual(snapshot["generation_context"]["temperature"], 0.2)
        self.assertEqual(snapshot["generation_context"]["system_prompt_sha256"], "c" * 64)

    def test_diagnostic_snapshot_summarizes_expansion_joint_rerank_and_coverage_safely(self) -> None:
        run, events, timestamp = _stored_trace(content_included=False)
        secret = "普通员工属于D级，住宿标准为内部金额"

        def trace_event(sequence: int, event: str, payload: dict) -> RagTraceEvent:
            return RagTraceEvent(
                id=uuid.uuid4(),
                trace_id=run.trace_id,
                sequence=sequence,
                event=event,
                payload={"event": event, **payload},
                created_at=timestamp + timedelta(milliseconds=sequence),
            )

        events.extend([
            trace_event(3, "retrieval.expansion_planned", {
                "should_expand": True,
                "seed_document_count": 1,
                "seed_chunk_count": 2,
                "secondary_query_count": 1,
                "bridge_term_count": 1,
                "required_facet_count": 4,
                "max_added_candidates": 24,
                "max_joint_rerank_candidates": 30,
                "max_added_chars": 30_000,
                # These fields are intentionally not part of the diagnostic
                # allow-list even if an old producer persisted them.
                "secondary_queries": [secret],
                "bridge_terms": [secret],
                "reason": secret,
            }),
            trace_event(4, "retrieval.document_scoped_completed", {
                "succeeded": True,
                "query_count": 1,
                "successful_query_count": 1,
                "failed_query_count": 0,
                "scoped_document_count": 1,
                "scoped_chunk_count": 15,
                "max_document_chunk_count": 15,
                "candidate_count": 8,
                "vector_fallback_count": 0,
                "scan_guard_triggered": False,
                "scan_guard_reason": "total_chunk_limit",
                "channel_candidate_counts": {
                    "vector": 5,
                    "keyword": 2,
                    "trigram": 1,
                    "document_excerpt": 99,
                },
                "candidate_contents": [secret],
                "elapsed_ms": 21,
            }),
            trace_event(5, "retrieval.structure_expanded", {
                "seed_chunk_count": 2,
                "scoped_document_count": 1,
                "candidate_count": 6,
                "counts_by_origin": {
                    "adjacent": 2,
                    "same_section": 2,
                    "table_sibling": 2,
                    secret: 99,
                },
                "elapsed_ms": 4,
                "content": secret,
            }),
            trace_event(6, "retrieval.expansion_completed", {
                "initial_candidate_count": 12,
                "added_candidate_count": 10,
                "combined_candidate_count": 22,
                "counts_by_origin": {
                    "document_scoped": 6,
                    "adjacent": 2,
                    "same_section": 2,
                },
                "deduplicated_count": 3,
                "budget_dropped_count": 1,
                "error_count": 0,
                "added_chars": 8_000,
                "elapsed_ms": 32,
            }),
            trace_event(7, "rerank.joint_completed", {
                "requested": True,
                "attempted": True,
                "succeeded": True,
                "pass_name": "joint",
                "model": "rerank/model-v2",
                "prompt_version": "joint-v1",
                "candidate_count": 22,
                "selected_candidate_count": 3,
                "requirement_count": 4,
                "missing_requirement_count": 2,
                "evidence_set_count": 2,
                "selected_evidence_set_id": "set-final",
                "selected_candidate_indexes": [1, 4, 7],
                "coverage_status": "partial",
                "joint_support_score": 0.78,
                "covered_requirement_ids": ["req-1", "req-2"],
                "missing_requirement_ids": ["req-3", "req-4"],
                "elapsed_ms": 45,
                "requirements": [secret],
                "error": secret,
            }),
            trace_event(9, "evidence.coverage", {
                "pass": "initial",
                "coverage_status": "partial",
                "required_requirement_count": 4,
                "covered_requirement_count": 1,
                "missing_requirement_count": 3,
                "selected_candidate_count": 1,
                "missing_requirement_ids": ["req-2", "req-3", "req-4"],
                "missing_requirements": [secret],
            }),
            trace_event(8, "evidence.coverage_assessed", {
                "pass_name": "final",
                "coverage_status": "complete",
                "required_requirement_count": 4,
                "covered_requirement_count": 4,
                "missing_requirement_count": 0,
                "selected_candidate_count": 3,
                "selected_evidence_set_id": "set-final",
                "joint_support_score": 0.91,
                "covered_requirement_ids": ["req-1", "req-2", "req-3", "req-4"],
                "missing_requirement_ids": [],
                "expansion_attempted": True,
                "expansion_succeeded": True,
                "retry_exhausted": False,
                "context_budget_dropped_count": 1,
                "context_budget_chars": 12_345,
                "trigger": "model_plan",
                "elapsed_ms": 2,
                "answer_preview": secret,
            }),
        ])
        run.event_count = len(events)
        run.observed_event_count = len(events)

        snapshot = _rag_trace_export_payload(
            run,
            events,
            exported_at=timestamp,
        )["diagnostic_index"]["snapshot"]

        self.assertTrue(snapshot["retrieval_expansion_plan"]["should_expand"])
        self.assertEqual(
            snapshot["retrieval_document_scoped_result"]["channel_candidate_counts"],
            {"vector": 5, "keyword": 2, "trigram": 1},
        )
        self.assertFalse(
            snapshot["retrieval_document_scoped_result"]["scan_guard_triggered"]
        )
        self.assertEqual(
            snapshot["retrieval_document_scoped_result"]["scoped_chunk_count"],
            15,
        )
        self.assertEqual(
            snapshot["retrieval_structure_expansion"]["counts_by_origin"],
            {"adjacent": 2, "same_section": 2, "table_sibling": 2},
        )
        self.assertEqual(
            snapshot["retrieval_expansion_result"]["combined_candidate_count"],
            22,
        )
        self.assertEqual(snapshot["retrieval_expansion_result"]["error_count"], 0)
        self.assertEqual(snapshot["rerank_joint_result"]["model"], "rerank/model-v2")
        self.assertEqual(
            snapshot["rerank_joint_result"]["selected_candidate_indexes"],
            [1, 4, 7],
        )
        self.assertTrue(snapshot["rerank_joint_result"]["requested"])
        self.assertEqual(snapshot["rerank_joint_result"]["pass_name"], "joint")
        self.assertEqual(snapshot["rerank_joint_result"]["selected_candidate_count"], 3)
        self.assertEqual(snapshot["evidence_coverage"]["pass_name"], "final")
        self.assertEqual(snapshot["evidence_coverage"]["coverage_status"], "complete")
        self.assertTrue(snapshot["evidence_coverage"]["expansion_succeeded"])
        self.assertEqual(snapshot["evidence_coverage"]["context_budget_dropped_count"], 1)
        self.assertEqual(snapshot["evidence_coverage"]["context_budget_chars"], 12_345)
        self.assertEqual(snapshot["evidence_coverage"]["trigger"], "model_plan")
        self.assertEqual(len(snapshot["evidence_coverage_history"]), 2)
        self.assertNotIn(secret, json.dumps(snapshot, ensure_ascii=False))

    def test_trace_export_downloads_every_stored_event_with_expected_filename(self) -> None:
        run, events, _ = _stored_trace(content_included=True)
        db = _TraceExportDb(run, events)
        audit = _AuditStub()

        response = asyncio.run(export_rag_trace(
            trace_id=run.trace_id,
            db=db,
            user=SimpleNamespace(is_superadmin=True),
            audit=audit,
        ))
        payload = json.loads(response.body)

        self.assertEqual(response.media_type, "application/json")
        self.assertEqual(
            response.headers["content-disposition"],
            f'attachment; filename="rag-trace-{run.trace_id}.json"',
        )
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(len(payload["events"]), 2)
        self.assertIsInstance(payload, dict)
        self.assertIn("diagnostic_index", payload)
        self.assertIn("integrity", payload["diagnostic_index"])
        self.assertEqual(payload["events"][0]["payload"]["question"], "已保存的问题")
        self.assertTrue(payload["data_policy"]["content_included"])
        self.assertEqual(db.commits, 1)
        self.assertEqual(audit.calls[0][1], "rag_trace.export")
        self.assertEqual(audit.calls[0][2]["detail"], {
            "event_count": 2,
            "persisted_event_count": 2,
            "truncated": False,
            "content_included": True,
            "encoded_bytes": len(response.body),
        })
        self.assertEqual(response.headers["x-rag-trace-truncated"], "false")
        self.assertEqual(response.headers["x-rag-trace-omitted-events"], "0")
        self.assertEqual(response.headers["x-rag-trace-bytes"], str(len(response.body)))

    def test_content_trace_detail_and_export_require_superadmin(self) -> None:
        run, _events, _ = _stored_trace(content_included=True)

        with self.assertRaises(HTTPException) as raised:
            _require_trace_content_access(
                run,
                SimpleNamespace(is_superadmin=False),
            )

        self.assertEqual(raised.exception.status_code, 403)
        _require_trace_content_access(
            run,
            SimpleNamespace(is_superadmin=True),
        )

    def test_masked_trace_summary_explicitly_reports_content_access(self) -> None:
        run, _events, _timestamp = _stored_trace(content_included=True)

        output = _run_out(
            run,
            "auditor",
            reveal_content=False,
            content_accessible=False,
        )

        self.assertTrue(output.content_included)
        self.assertFalse(output.content_accessible)
        self.assertIsNone(output.input_preview)
        self.assertIsNone(output.output_preview)

    def test_trace_export_limit_keeps_core_events_before_verbose_candidates(self) -> None:
        run, events, timestamp = _stored_trace(content_included=False)
        candidate = RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=1,
            event="retrieval.candidate",
            payload={"event": "retrieval.candidate", "candidate_content": "x" * 1000},
            created_at=timestamp,
        )
        routing = events[1]
        routing.sequence = 2
        response = RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=3,
            event="chat.response",
            payload={"event": "chat.response", "evidence_status": "hit"},
            created_at=timestamp,
        )
        all_events = [candidate, routing, response]
        metadata = [
            (event.id, event.sequence, event.event, len(json.dumps(event.payload)))
            for event in all_events
        ]
        db = SimpleNamespace(
            execute=AsyncMock(side_effect=[
                _TraceRowsResult(metadata),
                _TraceEventsResult([routing, response]),
            ])
        )

        with patch("api.rag_traces.TRACE_EXPORT_MAX_EVENTS", 2):
            selected, stats = asyncio.run(
                _load_bounded_export_events(db, run.trace_id)
            )

        self.assertEqual([event.event for event in selected], ["intent.routing_decision", "chat.response"])
        self.assertTrue(stats["truncated"])
        self.assertEqual(stats["omitted_event_count"], 1)

    def test_trace_export_limit_preserves_new_expansion_and_coverage_core_events(self) -> None:
        run, _events, timestamp = _stored_trace(content_included=False)
        event_names = [
            "retrieval.expansion_planned",
            "retrieval.document_scoped_completed",
            "retrieval.structure_expanded",
            "retrieval.expansion_completed",
            "rerank.joint_completed",
            "evidence.coverage_assessed",
        ]
        core_events = [
            RagTraceEvent(
                id=uuid.uuid4(),
                trace_id=run.trace_id,
                sequence=index,
                event=event_name,
                payload={"event": event_name, "candidate_count": index},
                created_at=timestamp + timedelta(milliseconds=index),
            )
            for index, event_name in enumerate(event_names, start=1)
        ]
        verbose = RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=7,
            event="retrieval.candidate",
            payload={"event": "retrieval.candidate", "candidate_content": "x" * 1000},
            created_at=timestamp,
        )
        terminal = RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=8,
            event="chat.response",
            payload={"event": "chat.response", "evidence_status": "hit"},
            created_at=timestamp,
        )
        all_events = [*core_events, verbose, terminal]
        metadata = [
            (event.id, event.sequence, event.event, len(json.dumps(event.payload)))
            for event in all_events
        ]
        db = SimpleNamespace(execute=AsyncMock(side_effect=[
            _TraceRowsResult(metadata),
            _TraceEventsResult([*core_events, terminal]),
        ]))

        with patch("api.rag_traces.TRACE_EXPORT_MAX_EVENTS", 7):
            selected, stats = asyncio.run(_load_bounded_export_events(db, run.trace_id))

        self.assertEqual(
            [event.event for event in selected],
            [*event_names, "chat.response"],
        )
        self.assertTrue(stats["truncated"])
        self.assertEqual(stats["omitted_event_count"], 1)

    def test_export_encoder_enforces_real_bytes_and_keeps_latest_terminal(self) -> None:
        run, events, timestamp = _stored_trace(content_included=False)
        candidate = RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=3,
            event="retrieval.candidate",
            payload={
                "event": "retrieval.candidate",
                "candidate_content": "云枢配置" * 5_000,
            },
            created_at=timestamp,
        )
        terminal = RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=4,
            event="chat.response",
            payload={"event": "chat.response", "evidence_status": "hit"},
            created_at=timestamp,
        )
        all_events = [*events, candidate, terminal]
        run.event_count = len(all_events)
        run.observed_event_count = len(all_events)
        export_stats = {
            "persisted_event_count": len(all_events),
            "selected_event_count": len(all_events),
            "selected_payload_bytes_estimate": None,
            "max_events": 500,
            "max_payload_bytes": 8_000,
            "truncated": False,
            "omitted_event_count": 0,
        }

        with patch("api.rag_traces.TRACE_EXPORT_MAX_PAYLOAD_BYTES", 8_000):
            encoded, stats = _encode_bounded_trace_export(
                run,
                all_events,
                export_stats,
            )

        payload = json.loads(encoded)
        self.assertLessEqual(len(encoded), 8_000)
        self.assertNotIn("retrieval.candidate", [item["event"] for item in payload["events"]])
        self.assertEqual(payload["events"][-1]["event"], "chat.response")
        self.assertTrue(stats["truncated"])
        self.assertEqual(stats["omitted_event_count"], 1)
        self.assertEqual(stats["encoded_bytes"], len(encoded))

    def test_one_event_export_limit_selects_latest_terminal(self) -> None:
        run, _events, timestamp = _stored_trace(content_included=False)
        first_terminal = RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=3,
            event="chat.error",
            payload={"event": "chat.error"},
            created_at=timestamp,
        )
        latest_terminal = RagTraceEvent(
            id=uuid.uuid4(),
            trace_id=run.trace_id,
            sequence=4,
            event="chat.response",
            payload={"event": "chat.response"},
            created_at=timestamp,
        )
        metadata = [
            (item.id, item.sequence, item.event, len(json.dumps(item.payload)))
            for item in (first_terminal, latest_terminal)
        ]
        db = SimpleNamespace(execute=AsyncMock(side_effect=[
            _TraceRowsResult(metadata),
            _TraceEventsResult([latest_terminal]),
        ]))

        with patch("api.rag_traces.TRACE_EXPORT_MAX_EVENTS", 1):
            selected, stats = asyncio.run(_load_bounded_export_events(db, run.trace_id))

        self.assertEqual([item.event for item in selected], ["chat.response"])
        self.assertTrue(stats["truncated"])
        self.assertEqual(stats["omitted_event_count"], 1)

    def test_persistence_reserves_terminal_slot_and_tracks_omissions(self) -> None:
        timestamp = "2026-07-30T10:00:00+00:00"
        records = [
            {"trace_id": "trace-cap", "event": "chat.request", "timestamp": timestamp},
            {"trace_id": "trace-cap", "event": "retrieval.plan", "timestamp": timestamp},
            {"trace_id": "trace-cap", "event": "retrieval.candidate", "timestamp": timestamp},
            {"trace_id": "trace-cap", "event": "chat.response", "timestamp": timestamp},
        ]
        session = _PersistSession()
        settings = SimpleNamespace(
            rag_trace_max_events_per_run=3,
            rag_trace_content_max_chars=50_000,
        )

        with (
            patch("core.rag_trace_store.AsyncSessionLocal", return_value=session),
            patch("core.rag_trace_store.get_settings", return_value=settings),
        ):
            asyncio.run(trace_store._persist_batch(records))

        run = next(item for item in session.added if isinstance(item, RagTraceRun))
        stored_events = [
            item for item in session.added if isinstance(item, RagTraceEvent)
        ]
        self.assertEqual([item.event for item in stored_events], [
            "chat.request",
            "retrieval.plan",
            "chat.response",
        ])
        self.assertEqual([item.sequence for item in stored_events], [1, 2, 3])
        self.assertEqual(run.observed_event_count, 4)
        self.assertEqual(run.event_count, 3)
        self.assertEqual(run.storage_omitted_event_count, 1)
        self.assertTrue(run.storage_truncated)
        self.assertEqual(run.status, "success")
        self.assertEqual(session.commits, 1)

    def test_disabled_collection_still_starts_retention_cleanup(self) -> None:
        previous_queue = trace_store._queue
        previous_worker = trace_store._worker_task
        previous_cleanup = trace_store._cleanup_task
        previous_accepting = trace_store._accepting
        created = []

        class _TaskStub:
            def done(self):
                return False

        def create_task(coroutine, *, name):
            coroutine.close()
            created.append(name)
            return _TaskStub()

        settings = SimpleNamespace(
            rag_trace_enabled=False,
            rag_trace_persistence_enabled=True,
        )
        try:
            trace_store._queue = None
            trace_store._worker_task = None
            trace_store._cleanup_task = None
            trace_store._accepting = False
            with (
                patch("core.rag_trace_store.get_settings", return_value=settings),
                patch("core.rag_trace_store.asyncio.create_task", side_effect=create_task),
            ):
                asyncio.run(trace_store.start_rag_trace_store())

            self.assertEqual(created, ["rag-trace-cleanup"])
            self.assertIsNone(trace_store._worker_task)
            self.assertFalse(trace_store._accepting)
        finally:
            trace_store._queue = previous_queue
            trace_store._worker_task = previous_worker
            trace_store._cleanup_task = previous_cleanup
            trace_store._accepting = previous_accepting

    def test_trace_time_filter_normalizes_naive_and_aware_values_to_utc(self) -> None:
        naive = datetime(2026, 7, 30, 10, 0, 0)
        aware = datetime(
            2026,
            7,
            30,
            18,
            0,
            0,
            tzinfo=timezone(timedelta(hours=8)),
        )

        self.assertEqual(_utc_filter(naive).tzinfo, UTC)
        self.assertEqual(
            _utc_filter(aware),
            datetime(2026, 7, 30, 10, 0, 0, tzinfo=UTC),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
