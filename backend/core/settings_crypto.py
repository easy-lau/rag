"""系统设置中敏感值的加密存储。

数据库备份可包含 settings 表，因此模型 API Key 不能以明文持久化。
加密主密钥只来自部署环境的 CONFIG_ENCRYPTION_KEY，不写入数据库。
"""

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken


SECRET_SETTING_KEYS = frozenset({
    "llm_api_key",
    "embedding_api_key",
    "vision_api_key",
})

ENCRYPTED_VALUE_PREFIX = "enc:v1:"


class SettingsEncryptionError(RuntimeError):
    """加密主密钥缺失或无法解密配置时抛出。"""


def is_encrypted_setting_value(value: str | None) -> bool:
    return bool(value and value.startswith(ENCRYPTED_VALUE_PREFIX))


def is_versioned_encrypted_setting_value(value: str | None) -> bool:
    """判断值是否使用了 ``enc:`` 格式（包含未知版本）。"""
    return bool(value and value.startswith("enc:"))


def _fernet(master_key: str) -> Fernet:
    if not master_key or not master_key.strip():
        raise SettingsEncryptionError("CONFIG_ENCRYPTION_KEY 未设置")

    # 接受 openssl rand -hex 32 生成的普通字符串，也避免把原始主密钥直接当作 Fernet 格式。
    derived_key = base64.urlsafe_b64encode(
        hashlib.sha256(master_key.encode("utf-8")).digest()
    )
    return Fernet(derived_key)


def encrypt_setting_secret(value: str, master_key: str) -> str:
    token = _fernet(master_key).encrypt(value.encode("utf-8")).decode("ascii")
    return f"{ENCRYPTED_VALUE_PREFIX}{token}"


def decrypt_setting_secret(value: str, master_key: str) -> str:
    if is_versioned_encrypted_setting_value(value) and not is_encrypted_setting_value(value):
        # 不能把未来或损坏版本误认为历史明文后再次加密，否则会永久丢失原密钥。
        raise SettingsEncryptionError("系统设置中的加密密钥版本不受支持")

    if not is_encrypted_setting_value(value):
        # 兼容升级前 settings 表中的明文值；应用启动后会在主密钥存在时自动重加密。
        return value

    token = value[len(ENCRYPTED_VALUE_PREFIX):]
    try:
        return _fernet(master_key).decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError) as exc:
        raise SettingsEncryptionError("系统设置中的加密密钥无法解密") from exc
