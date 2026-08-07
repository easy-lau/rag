"""Load encrypted, database-backed settings into the current process.

The API server and document worker run as separate Supervisor processes in
production.  Each process therefore has its own cached ``Settings`` instance
and must load the database-backed model configuration for itself.
"""

import logging

from sqlalchemy import select

from config import get_settings
from core.settings_crypto import (
    SECRET_SETTING_KEYS,
    SettingsEncryptionError,
    decrypt_setting_secret,
    encrypt_setting_secret,
    is_encrypted_setting_value,
)
from models.db_models import SystemSetting


logger = logging.getLogger(__name__)


# Keep the database-backed runtime contract independent from the HTTP layer so
# standalone processes can load the same settings without importing an API
# router.  ``api.settings.SettingsUpdate`` verifies this set at import time.
EDITABLE_SETTING_KEYS = frozenset({
    "llm_api_key",
    "llm_base_url",
    "chat_model",
    "llm_structured_output_mode",
    "llm_disable_thinking",
    "intent_model",
    "rerank_model",
    "temperature",
    "max_tokens",
    "rerank_timeout_seconds",
    "embedding_api_key",
    "embedding_base_url",
    "embedding_model",
    "vision_api_key",
    "vision_base_url",
    "vision_model",
    "top_k",
    "rerank_enabled",
    "rag_general_fallback_mode",
    "rag_general_fallback_model",
    "show_sources",
    "site_title",
    "site_description",
    "site_logo",
    "browser_title",
    "site_copyright",
})


def coerce_setting_value(key: str, value: str, settings) -> object:
    if key == "rag_general_fallback_mode":
        normalized = value.strip().casefold()
        if normalized not in {
            "off",
            "no_hit",
            "no_hit_or_insufficient",
        }:
            raise ValueError("invalid general fallback mode")
        return normalized
    current = getattr(
        settings,
        key,
        "auto" if key == "llm_structured_output_mode" else None,
    )
    if isinstance(current, bool):
        return value.strip().lower() in ("true", "1", "yes")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


async def apply_stored_settings() -> None:
    """Load database settings into the current process and migrate old keys."""
    from database import AsyncSessionLocal

    settings = get_settings()
    migrated_keys: list[str] = []
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(SystemSetting))).scalars().all()
        for row in rows:
            if row.value is None or row.key not in EDITABLE_SETTING_KEYS:
                continue

            value = row.value
            if row.key in SECRET_SETTING_KEYS:
                try:
                    value = decrypt_setting_secret(
                        value,
                        settings.config_encryption_key,
                    )
                except SettingsEncryptionError as exc:
                    logger.error(
                        "[系统设置] 无法加载加密密钥 key=%s: %s",
                        row.key,
                        exc,
                    )
                    raise

                if value and not is_encrypted_setting_value(row.value):
                    if settings.config_encryption_key:
                        row.value = encrypt_setting_secret(
                            value,
                            settings.config_encryption_key,
                        )
                        migrated_keys.append(row.key)
                    else:
                        raise SettingsEncryptionError(
                            "检测到历史明文模型密钥；请设置 CONFIG_ENCRYPTION_KEY 后重新启动以完成加密迁移"
                        )

            try:
                setattr(
                    settings,
                    row.key,
                    coerce_setting_value(row.key, value, settings),
                )
            except (TypeError, ValueError):
                logger.warning("[系统设置] 忽略无法解析的字段 key=%s", row.key)

        if migrated_keys:
            await db.commit()

    if migrated_keys:
        logger.info(
            "[系统设置] 已将历史明文模型密钥迁移为加密存储：%s",
            ", ".join(migrated_keys),
        )
