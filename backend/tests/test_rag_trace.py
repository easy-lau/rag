import hashlib
import json
import unittest
import uuid
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.rag_trace import (
    TRACE_SCHEMA_VERSION,
    content_fields,
    exception_log_text,
    log_exception_safely,
    redact_sensitive_text,
    trace_contains_business_content,
    trace_event,
    trace_query_constraints,
)


def _settings(
    *,
    enabled=True,
    include_content=False,
    max_chars=50000,
    app_version="dev",
    app_revision="",
):
    return SimpleNamespace(
        rag_trace_enabled=enabled,
        rag_trace_include_content=include_content,
        rag_trace_content_max_chars=max_chars,
        app_version=app_version,
        app_revision=app_revision,
    )


class RagTraceTests(unittest.TestCase):
    def test_content_is_hashed_but_hidden_when_content_logging_is_disabled(self):
        with patch("core.rag_trace.get_settings", return_value=_settings()):
            fields = content_fields("question", "云枢8.6")

        self.assertEqual(fields["question_chars"], 5)
        self.assertEqual(
            fields["question_sha256"],
            hashlib.sha256("云枢8.6".encode("utf-8")).hexdigest(),
        )
        self.assertNotIn("question", fields)

    def test_development_content_is_included_and_bounded(self):
        with patch(
            "core.rag_trace.get_settings",
            return_value=_settings(include_content=True, max_chars=1000),
        ):
            fields = content_fields("answer", "回答内容")

        self.assertEqual(fields["answer"], "回答内容")

    def test_content_fields_redacts_credentials_before_hashing_or_storage(self):
        raw = "password=plain-secret https://user:pass@example.com/v1?q=secret"
        sanitized = "password=[REDACTED] https://example.com/v1"
        with patch(
            "core.rag_trace.get_settings",
            return_value=_settings(include_content=True),
        ):
            fields = content_fields("question", raw)

        self.assertEqual(fields["question"], sanitized)
        self.assertEqual(fields["question_chars"], len(sanitized))
        self.assertEqual(
            fields["question_sha256"],
            hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
        )
        self.assertNotEqual(
            fields["question_sha256"],
            hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )
        self.assertNotIn("plain-secret", json.dumps(fields))
        self.assertNotIn("user:pass", json.dumps(fields))

    def test_nested_candidate_and_context_are_detected_as_business_content(self):
        self.assertTrue(trace_contains_business_content({
            "results": [
                {
                    "ranking": {"score": 0.91},
                    "candidate_content": "企业内部配置说明",
                }
            ]
        }))
        self.assertTrue(trace_contains_business_content({
            "generation": {"context": "已筛选知识片段"}
        }))
        self.assertFalse(trace_contains_business_content({
            "generation": {"model": "gpt-compatible", "total_tokens": 128}
        }))

    def test_trace_event_emits_machine_readable_json(self):
        with (
            patch("core.rag_trace.get_settings", return_value=_settings()),
            self.assertLogs("rag.trace", level="INFO") as captured,
        ):
            trace_event("retrieval.plan", trace_id="trace-1", top_k=5)

        payload = json.loads(captured.output[0].partition("rag.trace:")[2])
        self.assertEqual(payload["event"], "retrieval.plan")
        self.assertEqual(payload["trace_id"], "trace-1")
        self.assertEqual(payload["top_k"], 5)

    def test_trace_event_recursively_serializes_uuid_and_decimal(self):
        document_id = uuid.uuid4()
        with (
            patch("core.rag_trace.get_settings", return_value=_settings()),
            self.assertLogs("rag.trace", level="INFO") as captured,
        ):
            trace_event(
                "rerank.candidate",
                candidate={
                    "document": {"id": document_id},
                    "ranking": [{"retrieval_score": Decimal("0.03125")}],
                },
            )

        payload = json.loads(captured.output[0].partition("rag.trace:")[2])
        self.assertEqual(payload["candidate"]["document"]["id"], str(document_id))
        retrieval_score = payload["candidate"]["ranking"][0]["retrieval_score"]
        self.assertEqual(retrieval_score, 0.03125)
        self.assertIsInstance(retrieval_score, float)

    def test_trace_event_includes_schema_and_application_version(self):
        self.assertEqual(TRACE_SCHEMA_VERSION, 2)
        settings = _settings(app_version="1.2.3", app_revision="abc1234")
        with (
            patch("core.rag_trace.get_settings", return_value=settings),
            self.assertLogs("rag.trace", level="INFO") as captured,
        ):
            trace_event("generation.completed", trace_id="trace-version")

        payload = json.loads(captured.output[0].partition("rag.trace:")[2])
        self.assertEqual(payload["trace_schema_version"], TRACE_SCHEMA_VERSION)
        self.assertTrue(payload["timestamp"].endswith("+00:00"))
        self.assertEqual(payload["app_version"], "1.2.3")
        self.assertEqual(payload["app_revision"], "abc1234")
        self.assertFalse(payload["content_capture_enabled"])

    def test_development_trace_recursively_redacts_credentials_and_url_queries(self):
        settings = _settings(include_content=True)
        with (
            patch("core.rag_trace.get_settings", return_value=settings),
            self.assertLogs("rag.trace", level="INFO") as captured,
        ):
            trace_event(
                "retrieval.candidate",
                trace_id="trace-secret",
                metadata={
                    "api_key": "sk-private",
                    "nested": {
                        "password": "plain-password",
                        "access_token": "access-private",
                        "note": (
                            "Bearer abc.def and "
                            "https://user:pass@example.com/v1?q=secret#fragment"
                        ),
                    },
                },
            )

        payload = json.loads(captured.output[0].partition("rag.trace:")[2])
        self.assertTrue(payload["content_capture_enabled"])
        self.assertEqual(payload["metadata"]["api_key"], "[REDACTED]")
        self.assertEqual(payload["metadata"]["nested"]["password"], "[REDACTED]")
        self.assertEqual(payload["metadata"]["nested"]["access_token"], "[REDACTED]")
        note = payload["metadata"]["nested"]["note"]
        self.assertIn("Bearer [REDACTED]", note)
        self.assertIn("https://example.com/v1", note)
        self.assertNotIn("user:pass", note)
        self.assertNotIn("q=secret", note)

    def test_production_constraints_remove_echoed_query_fragment(self):
        with patch("core.rag_trace.get_settings", return_value=_settings()):
            constraints = trace_query_constraints(
                {
                    "product": "云枢",
                    "version": "8.6",
                    "matched_text": "我是云枢8.6",
                    "extraction_reason": "由原问题识别",
                }
            )

        self.assertEqual(constraints["product"], "云枢")
        self.assertEqual(constraints["version"], "8.6")
        self.assertNotIn("matched_text", constraints)
        self.assertNotIn("extraction_reason", constraints)

    def test_production_exception_trace_keeps_type_but_hides_message(self):
        with (
            patch("core.rag_trace.get_settings", return_value=_settings()),
            self.assertLogs("rag.trace", level="INFO") as captured,
        ):
            trace_event(
                "chat.error",
                error=RuntimeError("https://provider.example secret response"),
            )

        payload = json.loads(captured.output[0].partition("rag.trace:")[2])
        self.assertEqual(payload["error"], {"type": "RuntimeError"})

    def test_production_normal_log_hides_exception_message_and_traceback(self):
        target_logger = Mock()
        exc = RuntimeError("https://provider.example secret response")
        with patch("core.rag_trace.get_settings", return_value=_settings()):
            summary = exception_log_text(exc)
            log_exception_safely(
                target_logger,
                "request failed trace=%s",
                "trace-1",
                exc=exc,
            )

        self.assertEqual(summary, "RuntimeError")
        target_logger.error.assert_called_once_with(
            "request failed trace=%s error=%s",
            "trace-1",
            "RuntimeError",
        )
        target_logger.exception.assert_not_called()

    def test_development_normal_log_keeps_exception_details(self):
        target_logger = Mock()
        exc = RuntimeError("provider diagnostic")
        with patch(
            "core.rag_trace.get_settings",
            return_value=_settings(include_content=True),
        ):
            summary = exception_log_text(exc)
            log_exception_safely(
                target_logger,
                "request failed",
                exc=exc,
            )

        self.assertEqual(summary, "RuntimeError: provider diagnostic")
        target_logger.exception.assert_called_once_with(
            "request failed error=%s",
            "RuntimeError: provider diagnostic",
        )

    def test_development_exception_details_are_useful_but_credentials_are_redacted(self):
        exc = RuntimeError(
            "request https://user:pass@example.com/v1?api_key=sk-private "
            "Authorization=Bearer-raw"
        )
        with patch(
            "core.rag_trace.get_settings",
            return_value=_settings(include_content=True),
        ):
            summary = exception_log_text(exc)

        self.assertIn("RuntimeError: request https://example.com/v1", summary)
        self.assertNotIn("user:pass", summary)
        self.assertNotIn("sk-private", summary)
        self.assertNotIn("Bearer-raw", summary)

    def test_non_http_connection_urls_drop_userinfo_query_and_fragment(self):
        sanitized = redact_sensitive_text(
            "DATABASE_URL=postgresql+asyncpg://rag:db-pass@db.example:5432/rag?ssl=true "
            "redis=redis://:cache-pass@cache.example:6379/0?decode=true "
            "broker=amqp://worker:mq-pass@mq.example/vhost#internal"
        )

        self.assertIn(
            "DATABASE_URL=postgresql+asyncpg://db.example:5432/rag",
            sanitized,
        )
        self.assertIn("redis=redis://cache.example:6379/0", sanitized)
        self.assertIn("broker=amqp://mq.example/vhost", sanitized)
        for secret in (
            "rag:db-pass",
            "cache-pass",
            "worker:mq-pass",
            "ssl=true",
            "decode=true",
            "#internal",
        ):
            self.assertNotIn(secret, sanitized)

    def test_serialization_error_does_not_interrupt_business_flow(self):
        error_logger = Mock()
        with (
            patch("core.rag_trace.get_settings", return_value=_settings()),
            patch("core.rag_trace.logger.error", error_logger),
        ):
            # 可观测日志失败只能被记录，不能向调用方传播异常。
            trace_event(
                "retrieval.plan",
                trace_id="trace-broken",
                unsupported_value=object(),
            )

        error_logger.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
