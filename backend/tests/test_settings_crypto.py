"""Regression coverage for encrypted, database-backed model settings."""

import unittest
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from api.settings import (
    ModelConnectionTest,
    ModelConnectionTestOut,
    ModelListRequest,
    SettingsUpdate,
    _ModelListFetchError,
    _fetch_model_ids,
    _load,
    _model_list_config,
    _run_model_connection_test,
    _secret_status,
    _test_config,
    apply_stored_settings,
    list_models,
    test_model_connection,
    update_settings,
)
from core.settings_crypto import (
    ENCRYPTED_VALUE_PREFIX,
    SettingsEncryptionError,
    decrypt_setting_secret,
    encrypt_setting_secret,
)
from config import Settings
from core.openai_client import get_embedding_client
from models.db_models import SystemSetting


class SettingsCryptoTests(unittest.TestCase):
    def test_legacy_model_environment_variables_are_ignored(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "must-not-be-used",
                "LLM_BASE_URL": "https://legacy.example.test/v1",
                "CHAT_MODEL": "legacy-model",
                "INTENT_MODEL": "legacy-intent-model",
                "RERANK_MODEL": "legacy-rerank-model",
                "EMBEDDING_DIMENSIONS": "1536",
                "TEMPERATURE": "1.9",
                "TOP_K": "20",
                "SITE_TITLE": "legacy-title",
            },
        ):
            settings = Settings()

        self.assertEqual(settings.llm_api_key, "")
        self.assertEqual(settings.llm_base_url, "https://api.openai.com/v1")
        self.assertEqual(settings.chat_model, "gpt-4o")
        self.assertEqual(settings.intent_model, "")
        self.assertEqual(settings.rerank_model, "")
        self.assertEqual(settings.embedding_dimensions, 2560)
        self.assertEqual(settings.temperature, 0.7)
        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.site_title, "RAG 检索系统")

    def test_round_trip_uses_versioned_ciphertext(self) -> None:
        ciphertext = encrypt_setting_secret("sk-secret-value", "test-master-key")

        self.assertTrue(ciphertext.startswith(ENCRYPTED_VALUE_PREFIX))
        self.assertNotIn("sk-secret-value", ciphertext)
        self.assertEqual(
            decrypt_setting_secret(ciphertext, "test-master-key"),
            "sk-secret-value",
        )

    def test_ciphertext_rejects_wrong_master_key_and_unknown_version(self) -> None:
        ciphertext = encrypt_setting_secret("sk-secret-value", "test-master-key")

        with self.assertRaises(SettingsEncryptionError):
            decrypt_setting_secret(ciphertext, "another-master-key")
        with self.assertRaises(SettingsEncryptionError):
            decrypt_setting_secret("enc:v2:future", "test-master-key")

    def test_legacy_plaintext_is_readable_for_one_time_migration(self) -> None:
        self.assertEqual(
            decrypt_setting_secret("legacy-secret", "test-master-key"),
            "legacy-secret",
        )


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.added = []
        self.commits = 0

    async def execute(self, _query):
        return _Result(self.rows)

    async def get(self, _model, key):
        return next((row for row in self.rows if row.key == key), None)

    def add(self, row):
        self.rows.append(row)
        self.added.append(row)

    async def commit(self):
        self.commits += 1


class _SessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False


def _runtime_settings(**overrides):
    values = {
        "config_encryption_key": "test-master-key",
        "llm_api_key": "",
        "llm_base_url": "https://llm.example.test/v1",
        "chat_model": "chat-model",
        "intent_model": "",
        "rerank_model": "",
        "temperature": 0.7,
        "max_tokens": 2048,
        "embedding_api_key": "",
        "embedding_base_url": "https://embedding.example.test/v1",
        "embedding_model": "embedding-model",
        "embedding_dimensions": 2560,
        "vision_api_key": "",
        "vision_base_url": "https://vision.example.test/v1",
        "vision_model": "vision-model",
        "top_k": 5,
        "rerank_enabled": True,
        "show_sources": True,
        "site_title": "RAG",
        "site_description": "desc",
        "site_logo": "",
        "browser_title": "",
        "site_copyright": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class StoredSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_migrates_legacy_plaintext_key_and_loads_runtime_value(self) -> None:
        row = SystemSetting(key="llm_api_key", value="legacy-secret")
        session = _Session([row])
        settings = _runtime_settings()

        with (
            patch("api.settings.get_settings", return_value=settings),
            patch("database.AsyncSessionLocal", return_value=_SessionContext(session)),
        ):
            await apply_stored_settings()

        self.assertTrue(row.value.startswith(ENCRYPTED_VALUE_PREFIX))
        self.assertEqual(settings.llm_api_key, "legacy-secret")
        self.assertEqual(session.commits, 1)

    async def test_startup_rejects_ciphertext_when_master_key_is_wrong(self) -> None:
        row = SystemSetting(
            key="llm_api_key",
            value=encrypt_setting_secret("legacy-secret", "right-key"),
        )
        session = _Session([row])
        settings = _runtime_settings(config_encryption_key="wrong-key")

        with (
            patch("api.settings.get_settings", return_value=settings),
            patch("database.AsyncSessionLocal", return_value=_SessionContext(session)),
            self.assertRaises(SettingsEncryptionError),
        ):
            await apply_stored_settings()

    async def test_startup_rejects_legacy_plaintext_without_master_key(self) -> None:
        row = SystemSetting(key="llm_api_key", value="legacy-secret")
        session = _Session([row])
        settings = _runtime_settings(config_encryption_key="")

        with (
            patch("api.settings.get_settings", return_value=settings),
            patch("database.AsyncSessionLocal", return_value=_SessionContext(session)),
            self.assertRaises(SettingsEncryptionError),
        ):
            await apply_stored_settings()

    async def test_update_encrypts_api_key_and_does_not_audit_its_value(self) -> None:
        session = _Session()
        audit = Mock()
        settings = _runtime_settings()

        with (
            patch("api.settings.get_settings", return_value=settings),
            patch("api.settings._load", new=AsyncMock(return_value={"llm_api_key": "已配置（加密保存）"})),
        ):
            result = await update_settings(
                SettingsUpdate(llm_api_key="sk-new-secret"),
                db=session,
                audit=audit,
                _=None,
            )

        self.assertEqual(result["llm_api_key"], "已配置（加密保存）")
        self.assertEqual(session.commits, 1)
        self.assertEqual(settings.llm_api_key, "sk-new-secret")
        self.assertEqual(len(session.added), 1)
        self.assertTrue(session.added[0].value.startswith(ENCRYPTED_VALUE_PREFIX))
        self.assertNotIn("sk-new-secret", session.added[0].value)
        audit.log.assert_called_once()
        self.assertNotIn("sk-new-secret", repr(audit.log.call_args))

    async def test_update_rejects_new_key_without_encryption_master_key(self) -> None:
        session = _Session()
        with patch(
            "api.settings.get_settings",
            return_value=_runtime_settings(config_encryption_key=""),
        ):
            with self.assertRaises(HTTPException) as raised:
                await update_settings(
                    SettingsUpdate(llm_api_key="sk-new-secret"),
                    db=session,
                    audit=Mock(),
                    _=None,
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(session.commits, 0)

    async def test_update_normalizes_and_applies_intent_model(self) -> None:
        session = _Session()
        audit = Mock()
        settings = _runtime_settings()

        with (
            patch("api.settings.get_settings", return_value=settings),
            patch(
                "api.settings._load",
                new=AsyncMock(return_value={"intent_model": "intent-model"}),
            ),
        ):
            result = await update_settings(
                SettingsUpdate(intent_model="  intent-model  "),
                db=session,
                audit=audit,
                _=None,
            )

        self.assertEqual(result["intent_model"], "intent-model")
        self.assertEqual(settings.intent_model, "intent-model")
        self.assertEqual(session.added[0].key, "intent_model")
        self.assertEqual(session.added[0].value, "intent-model")
        self.assertEqual(
            audit.log.call_args.kwargs["detail"],
            {"changed": ["intent_model"]},
        )

    async def test_update_normalizes_and_applies_rerank_model(self) -> None:
        session = _Session()
        audit = Mock()
        settings = _runtime_settings()

        with (
            patch("api.settings.get_settings", return_value=settings),
            patch(
                "api.settings._load",
                new=AsyncMock(return_value={"rerank_model": "fast-reranker"}),
            ),
        ):
            result = await update_settings(
                SettingsUpdate(rerank_model="  fast-reranker  "),
                db=session,
                audit=audit,
                _=None,
            )

        self.assertEqual(result["rerank_model"], "fast-reranker")
        self.assertEqual(settings.rerank_model, "fast-reranker")
        self.assertEqual(session.added[0].key, "rerank_model")
        self.assertEqual(session.added[0].value, "fast-reranker")
        self.assertEqual(
            audit.log.call_args.kwargs["detail"],
            {"changed": ["rerank_model"]},
        )

    async def test_update_accepts_empty_rerank_model_for_chat_fallback(self) -> None:
        row = SystemSetting(key="rerank_model", value="old-reranker")
        session = _Session([row])
        settings = _runtime_settings(rerank_model="old-reranker")

        with (
            patch("api.settings.get_settings", return_value=settings),
            patch(
                "api.settings._load",
                new=AsyncMock(return_value={"rerank_model": ""}),
            ),
        ):
            await update_settings(
                SettingsUpdate(rerank_model="   "),
                db=session,
                audit=Mock(),
                _=None,
            )

        self.assertEqual(row.value, "")
        self.assertEqual(settings.rerank_model, "")

    async def test_load_returns_saved_rerank_model(self) -> None:
        session = _Session([
            SystemSetting(key="rerank_model", value="fast-reranker"),
        ])

        with patch("api.settings.get_settings", return_value=_runtime_settings()):
            result = await _load(session)

        self.assertEqual(result["rerank_model"], "fast-reranker")

    async def test_update_rejects_base_url_change_without_a_new_key(self) -> None:
        session = _Session()
        settings = _runtime_settings(llm_api_key="stored-secret")

        with patch("api.settings.get_settings", return_value=settings):
            with self.assertRaises(HTTPException) as raised:
                await update_settings(
                    SettingsUpdate(llm_base_url="https://other.example.test/v1"),
                    db=session,
                    audit=Mock(),
                    _=None,
                )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("重新填写", raised.exception.detail)
        self.assertEqual(session.commits, 0)

    async def test_update_rejects_unverified_embedding_dimension(self) -> None:
        session = _Session()
        settings = _runtime_settings(embedding_api_key="embedding-secret")
        failed_test = ModelConnectionTestOut(
            ok=False,
            message="向量模型返回 1536 维，当前知识库需要 2560 维",
            embedding_dimensions=1536,
            latency_ms=10,
            error_code="embedding_dimension_mismatch",
        )

        with (
            patch("api.settings.get_settings", return_value=settings),
            patch("api.settings._run_model_connection_test", new=AsyncMock(return_value=failed_test)),
        ):
            with self.assertRaises(HTTPException) as raised:
                await update_settings(
                    SettingsUpdate(embedding_model="another-embedding-model"),
                    db=session,
                    audit=Mock(),
                    _=None,
                )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("1536", raised.exception.detail)
        self.assertEqual(session.commits, 0)


class SettingsDisplayAndTestConfigTests(unittest.TestCase):
    def test_secret_status_never_returns_raw_value(self) -> None:
        self.assertEqual(
            _secret_status("very-short-secret", ""),
            "已配置（加密保存）",
        )
        self.assertEqual(
            _secret_status(None, "environment-secret"),
            "已配置（环境变量）",
        )

    def test_test_connection_uses_saved_key_when_form_key_is_empty(self) -> None:
        settings = _runtime_settings(llm_api_key="stored-secret")
        payload = ModelConnectionTest(service="llm", api_key="", base_url="", model="")

        with patch("api.settings.get_settings", return_value=settings):
            api_key, base_url, model, host = _test_config(payload)

        self.assertEqual(api_key, "stored-secret")
        self.assertEqual(base_url, "https://llm.example.test/v1")
        self.assertEqual(model, "chat-model")
        self.assertEqual(host, "llm.example.test")

    def test_connection_test_refuses_to_send_saved_key_to_changed_base_url(self) -> None:
        settings = _runtime_settings(llm_api_key="stored-secret")
        payload = ModelConnectionTest(
            service="llm",
            api_key="",
            base_url="https://other.example.test/v1",
            model="chat-model",
        )

        with patch("api.settings.get_settings", return_value=settings):
            with self.assertRaises(HTTPException) as raised:
                _test_config(payload)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("重新填写 API Key", raised.exception.detail)

    def test_connection_test_uses_same_endpoint_key_fallback_as_runtime(self) -> None:
        settings = _runtime_settings(
            llm_api_key="shared-secret",
            embedding_api_key="",
            embedding_base_url="https://llm.example.test/v1",
        )
        payload = ModelConnectionTest(service="embedding", api_key="", base_url="", model="")

        with patch("api.settings.get_settings", return_value=settings):
            api_key, *_ = _test_config(payload)

        self.assertEqual(api_key, "shared-secret")

    def test_connection_test_rejects_base_url_with_embedded_credentials(self) -> None:
        settings = _runtime_settings(llm_api_key="stored-secret")
        payload = ModelConnectionTest(
            service="llm",
            base_url="https://attacker:password@example.test/v1",
            model="chat-model",
        )

        with patch("api.settings.get_settings", return_value=settings):
            with self.assertRaises(HTTPException) as raised:
                _test_config(payload)

        self.assertEqual(raised.exception.status_code, 400)

    def test_model_list_uses_saved_key_without_requiring_a_model(self) -> None:
        settings = _runtime_settings(llm_api_key="stored-secret")
        payload = ModelListRequest(service="llm", api_key="", base_url="")

        with patch("api.settings.get_settings", return_value=settings):
            api_key, base_url, host = _model_list_config(payload)

        self.assertEqual(api_key, "stored-secret")
        self.assertEqual(base_url, "https://llm.example.test/v1")
        self.assertEqual(host, "llm.example.test")

    def test_model_list_refuses_to_send_saved_key_to_changed_base_url(self) -> None:
        settings = _runtime_settings(llm_api_key="stored-secret")
        payload = ModelListRequest(
            service="llm",
            api_key="",
            base_url="https://other.example.test/v1",
        )

        with patch("api.settings.get_settings", return_value=settings):
            with self.assertRaises(HTTPException) as raised:
                _model_list_config(payload)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("重新填写 API Key", raised.exception.detail)


class ModelConnectionTestTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_test_returns_safe_failure_and_closes_client(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("raw upstream detail")))
            ),
            close=AsyncMock(),
        )
        payload = ModelConnectionTest(service="llm")

        with patch("api.settings.AsyncOpenAI", return_value=client):
            result = await _run_model_connection_test(
                payload,
                api_key="sk-never-log-this",
                base_url="https://llm.example.test/v1",
                model="chat-model",
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "RuntimeError")
        self.assertNotIn("raw upstream detail", result.message)
        client.close.assert_awaited_once()

    async def test_vision_test_reports_base_connection_failure_separately(self) -> None:
        create = AsyncMock(side_effect=RuntimeError("raw upstream detail"))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=AsyncMock(),
        )

        with patch("api.settings.AsyncOpenAI", return_value=client):
            result = await _run_model_connection_test(
                ModelConnectionTest(service="vision"),
                api_key="sk-never-log-this",
                base_url="https://vision.example.test/v1",
                model="vision-model",
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "vision_text_RuntimeError")
        self.assertIn("基础文本接口连接失败", result.message)
        self.assertEqual(create.await_count, 1)
        client.close.assert_awaited_once()

    async def test_vision_test_reports_image_protocol_failure_after_text_success(self) -> None:
        create = AsyncMock(
            side_effect=[SimpleNamespace(), RuntimeError("raw upstream detail")]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=AsyncMock(),
        )

        with patch("api.settings.AsyncOpenAI", return_value=client):
            result = await _run_model_connection_test(
                ModelConnectionTest(service="vision"),
                api_key="sk-never-log-this",
                base_url="https://vision.example.test/v1",
                model="vision-model",
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "vision_image_RuntimeError")
        self.assertIn("文本接口连接成功，但图片输入失败", result.message)
        self.assertEqual(create.await_count, 2)
        image_content = create.await_args_list[1].kwargs["messages"][0]["content"]
        self.assertEqual(image_content[1]["type"], "image_url")
        self.assertTrue(
            image_content[1]["image_url"]["url"].startswith(
                "data:image/png;base64,iVBOR"
            )
        )
        self.assertGreater(len(image_content[1]["image_url"]["url"]), 400)
        client.close.assert_awaited_once()

    async def test_vision_test_succeeds_only_after_text_and_image_requests(self) -> None:
        create = AsyncMock(return_value=SimpleNamespace())
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=AsyncMock(),
        )

        with patch("api.settings.AsyncOpenAI", return_value=client):
            result = await _run_model_connection_test(
                ModelConnectionTest(service="vision"),
                api_key="sk-never-log-this",
                base_url="https://vision.example.test/v1",
                model="vision-model",
            )

        self.assertTrue(result.ok)
        self.assertIn("图片输入测试成功", result.message)
        self.assertEqual(create.await_count, 2)
        client.close.assert_awaited_once()

    async def test_connection_endpoint_audits_without_api_key(self) -> None:
        session = _Session()
        audit = Mock()
        result = ModelConnectionTestOut(
            ok=True,
            message="模型连接成功",
            latency_ms=12,
        )

        with (
            patch(
                "api.settings._test_config",
                return_value=("sk-never-log-this", "https://llm.example.test/v1", "chat-model", "llm.example.test"),
            ),
            patch("api.settings._run_model_connection_test", new=AsyncMock(return_value=result)),
        ):
            actual = await test_model_connection(
                ModelConnectionTest(service="llm"),
                db=session,
                audit=audit,
                _=None,
            )

        self.assertIs(actual, result)
        self.assertEqual(session.commits, 1)
        audit.log.assert_called_once()
        self.assertNotIn("sk-never-log-this", repr(audit.log.call_args))


class ModelListTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_list_keeps_only_safe_unique_ids_and_closes_client(self) -> None:
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=AsyncMock(
                    return_value=SimpleNamespace(
                        data=[
                            SimpleNamespace(id="z-model"),
                            SimpleNamespace(id=" alpha-model "),
                            SimpleNamespace(id="alpha-model"),
                            SimpleNamespace(id="bad\nmodel"),
                            SimpleNamespace(id="x" * 257),
                            SimpleNamespace(id=None),
                        ]
                    )
                )
            ),
            close=AsyncMock(),
        )

        with patch("api.settings.AsyncOpenAI", return_value=client):
            models, latency_ms = await _fetch_model_ids(
                service="llm",
                api_key="sk-never-log-this",
                base_url="https://llm.example.test/v1",
            )

        self.assertEqual(models, ["alpha-model", "z-model"])
        self.assertGreaterEqual(latency_ms, 0)
        client.models.list.assert_awaited_once_with(timeout=15)
        client.close.assert_awaited_once()

    async def test_model_list_failure_does_not_expose_upstream_details(self) -> None:
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=AsyncMock(side_effect=RuntimeError("raw upstream secret detail"))
            ),
            close=AsyncMock(),
        )

        with patch("api.settings.AsyncOpenAI", return_value=client):
            with self.assertRaises(_ModelListFetchError) as raised:
                await _fetch_model_ids(
                    service="llm",
                    api_key="sk-never-log-this",
                    base_url="https://llm.example.test/v1",
                )

        self.assertEqual(raised.exception.error_code, "RuntimeError")
        self.assertNotIn("raw upstream secret detail", str(raised.exception))
        client.close.assert_awaited_once()

    async def test_model_list_endpoint_audits_success_without_api_key_or_models(self) -> None:
        session = _Session()
        audit = Mock()
        payload = ModelListRequest(service="llm", api_key="sk-never-log-this")

        with (
            patch(
                "api.settings._model_list_config",
                return_value=(
                    "sk-never-log-this",
                    "https://llm.example.test/v1",
                    "llm.example.test",
                ),
            ),
            patch(
                "api.settings._fetch_model_ids",
                new=AsyncMock(return_value=(["chat-model", "other-model"], 12)),
            ),
        ):
            actual = await list_models(payload, db=session, audit=audit, _=None)

        self.assertEqual(actual.models, ["chat-model", "other-model"])
        self.assertEqual(actual.latency_ms, 12)
        self.assertEqual(session.commits, 1)
        audit.log.assert_called_once()
        audit_detail = audit.log.call_args.kwargs["detail"]
        self.assertEqual(audit_detail["model_count"], 2)
        self.assertNotIn("models", audit_detail)
        self.assertNotIn("sk-never-log-this", repr(audit.log.call_args))

    async def test_model_list_endpoint_audits_safe_failure(self) -> None:
        session = _Session()
        audit = Mock()
        payload = ModelListRequest(service="llm", api_key="sk-never-log-this")

        with (
            patch(
                "api.settings._model_list_config",
                return_value=(
                    "sk-never-log-this",
                    "https://llm.example.test/v1",
                    "llm.example.test",
                ),
            ),
            patch(
                "api.settings._fetch_model_ids",
                new=AsyncMock(side_effect=_ModelListFetchError("AuthenticationError", 18)),
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            await list_models(payload, db=session, audit=audit, _=None)

        self.assertEqual(raised.exception.status_code, 502)
        self.assertIn("/models", raised.exception.detail)
        self.assertNotIn("AuthenticationError", raised.exception.detail)
        self.assertEqual(session.commits, 1)
        audit.log.assert_called_once()
        audit_detail = audit.log.call_args.kwargs["detail"]
        self.assertFalse(audit_detail["ok"])
        self.assertEqual(audit_detail["error_code"], "AuthenticationError")
        self.assertNotIn("sk-never-log-this", repr(audit.log.call_args))


class ClientCredentialBoundaryTests(unittest.TestCase):
    def test_embedding_does_not_receive_llm_key_for_a_different_endpoint(self) -> None:
        settings = _runtime_settings(
            llm_api_key="llm-secret",
            embedding_api_key="",
            llm_base_url="https://llm.example.test/v1",
            embedding_base_url="https://embedding.example.test/v1",
        )
        client_factory = Mock()

        with (
            patch("core.openai_client.get_settings", return_value=settings),
            patch("core.openai_client.AsyncOpenAI", client_factory),
        ):
            get_embedding_client()

        self.assertEqual(client_factory.call_args.kwargs["api_key"], "")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
