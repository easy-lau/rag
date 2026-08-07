import logging
import os
import time
import uuid
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from core.audit import AuditLogger, get_audit
from core.deps import require_permission
from core.openai_client import get_service_credentials
from core.permissions import SETTINGS_READ, SETTINGS_WRITE
from core.reranker import clear_rerank_circuit_breakers
from core.runtime_settings import (
    EDITABLE_SETTING_KEYS,
    coerce_setting_value as _coerce_setting_value,
)
from core.settings_crypto import (
    SECRET_SETTING_KEYS,
    SettingsEncryptionError,
    encrypt_setting_secret,
)
from core.structured_output import (
    clear_structured_output_capability_cache,
    create_structured_completion,
)
from database import get_db
from models.db_models import SystemSetting, User


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])

LOGO_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "svg", "ico"}
SECRET_CONFIGURED_LABEL = "已配置（加密保存）"
ENV_SECRET_CONFIGURED_LABEL = "已配置（环境变量）"
PUBLIC_SITE_SETTING_KEYS = frozenset({
    "site_title",
    "site_description",
    "site_logo",
    "browser_title",
    "site_copyright",
})
MODEL_SERVICE_FIELDS = {
    "llm": ("llm_api_key", "llm_base_url", "chat_model"),
    "embedding": ("embedding_api_key", "embedding_base_url", "embedding_model"),
    "vision": ("vision_api_key", "vision_base_url", "vision_model"),
}

# 64×64、非透明且包含清晰边框/交叉线的 PNG。部分兼容网关会拒绝透明 1×1
# 占位图，因此测试图片必须更接近正式上传的普通截图，避免误判多模态能力。
_CONNECTION_TEST_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAABP0lEQVR42u3aSw7CMAxFUXbAkBUwYVssn0UUJCSEaJs4znPrW4I6QuC8I/WTJj5d7w/0cRqAARACzpfb+5hSfj7x6oCEhu9sJkAqw0+wVcD8pwnTv74pAbIZFsNUAHkMazHqgAyGQgATYF9DeWgrYC9DddAGwPYGy3BtgC0NxoGaAdsY7EN4ANGGpuJOQJyhtawfEGFwFOwCaA2+Ur0AlcFdRADoN/T8XQPoCdGJlwF8UfpPPyWgNZDk4hED7LFUty89wBJOePMNAZQjah9/UYC1oPKHdyBgHjdi+hQLKBhU9cMBiwZh8QE49inEvojZt1H2g4w9lWBP5tjTafYLDfuVkv1Sz15WYS9ssZcW2Yu77OV19gYHe4uJvcnH3mZlb3SzWw3YzR7sdht2wxO+5ew4TX9Tyk8dMDp3B+APAU9fQ3JlaZ+uUgAAAABJRU5ErkJggg=="
)


class SettingsOut(BaseModel):
    llm_api_key: str
    llm_base_url: str
    chat_model: str
    llm_structured_output_mode: Literal[
        "auto", "json_schema", "json_object", "plain_json"
    ]
    llm_disable_thinking: bool
    intent_model: str
    rerank_model: str
    temperature: float
    max_tokens: int
    rerank_timeout_seconds: float
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    vision_api_key: str
    vision_base_url: str
    vision_model: str
    top_k: int
    rerank_enabled: bool
    rag_general_fallback_mode: Literal[
        "off",
        "no_hit",
        "no_hit_or_insufficient",
    ]
    rag_general_fallback_model: str
    show_sources: bool
    site_title: str
    site_description: str
    site_logo: str
    browser_title: str
    site_copyright: str


class SettingsUpdate(BaseModel):
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    chat_model: str | None = None
    llm_structured_output_mode: Literal[
        "auto", "json_schema", "json_object", "plain_json"
    ] | None = None
    llm_disable_thinking: bool | None = None
    intent_model: str | None = Field(None, max_length=255)
    rerank_model: str | None = Field(None, max_length=255)
    temperature: float | None = None
    max_tokens: int | None = None
    rerank_timeout_seconds: float | None = Field(None, ge=1.0, le=120.0)
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    vision_api_key: str | None = None
    vision_base_url: str | None = None
    vision_model: str | None = None
    top_k: int | None = None
    rerank_enabled: bool | None = None
    rag_general_fallback_mode: Literal[
        "off",
        "no_hit",
        "no_hit_or_insufficient",
    ] | None = None
    rag_general_fallback_model: str | None = Field(None, max_length=255)
    show_sources: bool | None = None
    site_title: str | None = None
    site_description: str | None = None
    site_logo: str | None = None
    browser_title: str | None = None
    site_copyright: str | None = None


if frozenset(SettingsUpdate.model_fields) != EDITABLE_SETTING_KEYS:
    raise RuntimeError("SettingsUpdate 与数据库运行时设置字段不一致")


class ModelConnectionTest(BaseModel):
    service: Literal["llm", "embedding", "vision"]
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


class ModelConnectionTestOut(BaseModel):
    ok: bool
    message: str
    embedding_dimensions: int | None = None
    latency_ms: int
    error_code: str | None = None
    structured_output_mode: Literal[
        "json_schema", "json_object", "plain_json"
    ] | None = None
    structured_output_attempted_modes: list[str] = Field(default_factory=list)
    thinking_disabled: bool | None = None


class ModelListRequest(BaseModel):
    """读取 OpenAI 兼容服务的模型目录。

    ``api_key`` 只用于本次请求：允许管理员在尚未保存配置前先验证并选择模型，
    但绝不写入数据库、响应或审计日志。
    """

    service: Literal["llm", "embedding", "vision"]
    api_key: str | None = None
    base_url: str | None = None


class ModelListOut(BaseModel):
    models: list[str]
    latency_ms: int


class _ModelListFetchError(Exception):
    """上游 /models 请求失败时的安全错误载体。

    只保留异常类型和耗时，避免上游响应中可能携带的密钥、请求头或服务商细节
    经由日志、审计或 HTTP 响应泄漏。
    """

    def __init__(self, error_code: str, latency_ms: int):
        super().__init__(error_code)
        self.error_code = error_code
        self.latency_ms = latency_ms


class SiteSettingsOut(BaseModel):
    site_title: str
    site_description: str
    site_logo: str
    browser_title: str
    site_copyright: str


def _is_secret_placeholder(value: str) -> bool:
    return not value.strip() or "***" in value or value.startswith("已配置")


def _base_url_identity(value: str) -> tuple[str, str, int | None, str]:
    """校验模型 Base URL，并返回用于比较凭据边界的规范化标识。"""
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Base URL 格式或端口不正确") from exc

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(
            status_code=400,
            detail="Base URL 必须是无账号、查询参数和片段的 http:// 或 https:// 地址",
        )

    default_port = 443 if parsed.scheme == "https" else 80
    normalized_port = None if port in (None, default_port) else port
    path = parsed.path.rstrip("/")
    return parsed.scheme.lower(), parsed.hostname.lower(), normalized_port, path


def _validate_model_base_url_updates(updates: dict[str, object], settings) -> None:
    """Base URL 变化时不允许静默复用旧服务密钥到新地址。"""
    for service, (api_key_field, base_url_field, model_field) in MODEL_SERVICE_FIELDS.items():
        if model_field in updates:
            model = str(updates[model_field]).strip()
            if not model:
                raise HTTPException(status_code=400, detail="模型名称不能为空")
            updates[model_field] = model

        if base_url_field not in updates:
            continue

        new_base_url = str(updates[base_url_field]).strip()
        updates[base_url_field] = new_base_url
        new_identity = _base_url_identity(new_base_url)
        saved_api_key, saved_base_url = get_service_credentials(service, settings)
        if (
            saved_api_key
            and new_identity != _base_url_identity(saved_base_url)
            and not updates.get(api_key_field)
        ):
            raise HTTPException(
                status_code=400,
                detail="更改 Base URL 时必须重新填写该服务的 API Key",
            )


# Model IDs must be real provider identifiers.  Placeholder/template values
# (e.g. an ORM ``objectId`` string or a server-side template variable) were
# the root cause of the 8-05 ``model not found: objectId`` outage.  They are
# rejected at the save boundary instead of failing at request time.
_PLACEHOLDER_MODEL_MARKERS = (
    "objectid",
    "{{",
    "}}",
    "{%",
    "%}",
    "<model>",
    "model_id",
    "your_",
    "todo",
    "placeholder",
    "demo-model",
)
_MODEL_FIELDS = (
    "chat_model",
    "intent_model",
    "rerank_model",
    "embedding_model",
    "vision_model",
    "rag_general_fallback_model",
)
# intent/rerank/fallback models may be empty to mean "reuse the chat model".
_OPTIONAL_MODEL_FIELDS = frozenset({
    "intent_model",
    "rerank_model",
    "rag_general_fallback_model",
})
_TIMEOUT_FIELDS = ("rerank_timeout_seconds",)
_BOOL_FIELDS = ("llm_disable_thinking", "rerank_enabled", "show_sources")


def _validate_settings_contract(
    updates: dict[str, object],
    settings: Any,
) -> None:
    """Reject invalid runtime settings at the save boundary (contract layer).

    This is a whitelist contract, not a per-field ``if``: every model name is
    checked for placeholders, every timeout stays in ``[1, 120]`` seconds and
    every boolean field must be a real boolean.  A bad value must fail loudly
    here instead of being silently stored and exploding at request time.
    """

    for field in _MODEL_FIELDS:
        if field not in updates:
            continue
        raw = str(updates[field] or "").strip()
        updates[field] = raw
        if not raw:
            if field in _OPTIONAL_MODEL_FIELDS:
                continue
            raise HTTPException(status_code=400, detail="模型名称不能为空")
        lowered = raw.casefold()
        for marker in _PLACEHOLDER_MODEL_MARKERS:
            if marker in lowered:
                raise HTTPException(
                    status_code=400,
                    detail=f"{field} 不能包含占位符或模板变量",
                )
    for field in _TIMEOUT_FIELDS:
        if field not in updates:
            continue
        value = updates[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail="超时配置必须为数字")
        numeric = float(value)
        if not 1.0 <= numeric <= 120.0:
            raise HTTPException(
                status_code=400,
                detail="超时配置必须在 1~120 秒之间",
            )
    for field in _BOOL_FIELDS:
        if field in updates and not isinstance(updates[field], bool):
            raise HTTPException(status_code=400, detail="布尔配置必须为 true/false")


async def _validate_embedding_update(updates: dict[str, object], settings) -> None:
    """Embedding 连接发生变化时，在保存前验证可用性及 2560 维约束。"""
    api_key_field, base_url_field, model_field = MODEL_SERVICE_FIELDS["embedding"]
    fields = (api_key_field, base_url_field, model_field)
    if not any(
        field in updates and updates[field] != getattr(settings, field)
        for field in fields
    ):
        return

    saved_api_key, saved_base_url = get_service_credentials("embedding", settings)
    api_key = str(updates.get(api_key_field) or saved_api_key)
    base_url = str(updates.get(base_url_field) or saved_base_url)
    model = str(
        updates[model_field] if model_field in updates else getattr(settings, model_field)
    )
    result = await _run_model_connection_test(
        ModelConnectionTest(service="embedding"),
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.message)


def _secret_status(db_value: str | None, env_value: str) -> str:
    if db_value is not None:
        # 无论历史明文还是新版密文，接口都绝不返回密钥内容。
        return SECRET_CONFIGURED_LABEL if db_value else ""
    return ENV_SECRET_CONFIGURED_LABEL if env_value else ""


def _value_from_database(db_map: dict[str, str | None], key: str, settings):
    value = db_map.get(key)
    default = getattr(
        settings,
        key,
        "auto" if key == "llm_structured_output_mode" else None,
    )
    if value is None:
        return default
    try:
        return _coerce_setting_value(key, value, settings)
    except (TypeError, ValueError):
        logger.warning("[系统设置] 忽略无法解析的字段 key=%s", key)
        return default


async def _load(db: AsyncSession) -> dict:
    rows = (await db.execute(select(SystemSetting))).scalars().all()
    db_map = {row.key: row.value for row in rows}
    settings = get_settings()

    return {
        "llm_api_key": _secret_status(db_map.get("llm_api_key"), settings.llm_api_key),
        "llm_base_url": _value_from_database(db_map, "llm_base_url", settings),
        "chat_model": _value_from_database(db_map, "chat_model", settings),
        "llm_structured_output_mode": _value_from_database(
            db_map, "llm_structured_output_mode", settings
        ),
        "llm_disable_thinking": _value_from_database(
            db_map, "llm_disable_thinking", settings
        ),
        "intent_model": _value_from_database(db_map, "intent_model", settings),
        "rerank_model": _value_from_database(db_map, "rerank_model", settings),
        "temperature": _value_from_database(db_map, "temperature", settings),
        "max_tokens": _value_from_database(db_map, "max_tokens", settings),
        "rerank_timeout_seconds": _value_from_database(
            db_map,
            "rerank_timeout_seconds",
            settings,
        ),
        "embedding_api_key": _secret_status(db_map.get("embedding_api_key"), settings.embedding_api_key),
        "embedding_base_url": _value_from_database(db_map, "embedding_base_url", settings),
        "embedding_model": _value_from_database(db_map, "embedding_model", settings),
        "vision_api_key": _secret_status(db_map.get("vision_api_key"), settings.vision_api_key),
        "vision_base_url": _value_from_database(db_map, "vision_base_url", settings),
        "vision_model": _value_from_database(db_map, "vision_model", settings),
        "top_k": _value_from_database(db_map, "top_k", settings),
        "rerank_enabled": _value_from_database(db_map, "rerank_enabled", settings),
        "rag_general_fallback_mode": _value_from_database(
            db_map,
            "rag_general_fallback_mode",
            settings,
        ),
        "rag_general_fallback_model": _value_from_database(
            db_map,
            "rag_general_fallback_model",
            settings,
        ),
        "show_sources": _value_from_database(db_map, "show_sources", settings),
        "site_title": _value_from_database(db_map, "site_title", settings),
        "site_description": _value_from_database(db_map, "site_description", settings),
        "site_logo": _value_from_database(db_map, "site_logo", settings),
        "browser_title": _value_from_database(db_map, "browser_title", settings),
        "site_copyright": _value_from_database(db_map, "site_copyright", settings),
    }


async def _load_site(db: AsyncSession) -> dict:
    """公开端点只读取品牌字段，绝不返回模型或密钥配置。"""
    rows = await db.execute(
        select(SystemSetting.key, SystemSetting.value).where(
            SystemSetting.key.in_(PUBLIC_SITE_SETTING_KEYS)
        )
    )
    db_map = dict(rows.all())
    settings = get_settings()
    return {
        "site_title": _value_from_database(db_map, "site_title", settings),
        "site_description": _value_from_database(db_map, "site_description", settings),
        "site_logo": _value_from_database(db_map, "site_logo", settings),
        "browser_title": _value_from_database(db_map, "browser_title", settings),
        "site_copyright": _value_from_database(db_map, "site_copyright", settings),
    }


def _resolve_model_service_credentials(
    service: Literal["llm", "embedding", "vision"],
    *,
    submitted_api_key: str | None,
    submitted_base_url: str | None,
) -> tuple[str, str, str]:
    """解析一次模型服务调用所需凭据，并守住已保存 Key 的地址边界。"""
    settings = get_settings()
    saved_api_key, saved_base_url = get_service_credentials(service, settings)
    submitted_key = (submitted_api_key or "").strip()
    submitted_key = "" if _is_secret_placeholder(submitted_key) else submitted_key
    base_url = (submitted_base_url or "").strip() or saved_base_url

    base_url_identity = _base_url_identity(base_url)
    if submitted_key:
        api_key = submitted_key
    else:
        if not saved_api_key:
            raise HTTPException(status_code=400, detail="请先填写 API Key")
        if base_url_identity != _base_url_identity(saved_base_url):
            raise HTTPException(
                status_code=400,
                detail="更改 Base URL 时必须重新填写 API Key，系统不会把已保存密钥发送到新地址",
            )
        api_key = saved_api_key

    return api_key, base_url, base_url_identity[1]


def _test_config(payload: ModelConnectionTest) -> tuple[str, str, str, str]:
    settings = get_settings()
    _, _, model_field = MODEL_SERVICE_FIELDS[payload.service]
    api_key, base_url, host = _resolve_model_service_credentials(
        payload.service,
        submitted_api_key=payload.api_key,
        submitted_base_url=payload.base_url,
    )
    model = (payload.model or "").strip() or getattr(settings, model_field)

    if not model:
        raise HTTPException(status_code=400, detail="请先填写模型名称或推理接入点 ID")
    return api_key, base_url, model, host


def _model_list_config(payload: ModelListRequest) -> tuple[str, str, str]:
    """模型列表不需要预先填写 model，只复用凭据和 Base URL 校验规则。"""
    return _resolve_model_service_credentials(
        payload.service,
        submitted_api_key=payload.api_key,
        submitted_base_url=payload.base_url,
    )


def _safe_model_ids(response) -> list[str]:
    """从 OpenAI SDK 响应中提取可安全交给 UI 展示的模型 ID。"""
    model_ids: set[str] = set()
    for item in getattr(response, "data", ()) or ():
        model_id = getattr(item, "id", None)
        if not isinstance(model_id, str):
            continue
        model_id = model_id.strip()
        # 保留主流 OpenAI 兼容服务的模型 ID 字符集，同时过滤控制字符和异常长值。
        if (
            not model_id
            or len(model_id) > 256
            or any(ord(char) < 32 or ord(char) == 127 for char in model_id)
        ):
            continue
        model_ids.add(model_id)
    return sorted(model_ids, key=str.casefold)


async def _fetch_model_ids(
    *,
    service: Literal["llm", "embedding", "vision"],
    api_key: str,
    base_url: str,
) -> tuple[list[str], int]:
    """调用 OpenAI 兼容的 ``/models``，只返回模型 ID 和耗时。"""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    started = time.monotonic()
    try:
        response = await client.models.list(timeout=15)
        return _safe_model_ids(response), round((time.monotonic() - started) * 1000)
    except Exception as exc:
        error_code = type(exc).__name__
        latency_ms = round((time.monotonic() - started) * 1000)
        logger.warning(
            "[系统设置] 模型列表获取失败 service=%s error=%s",
            service,
            error_code,
        )
        raise _ModelListFetchError(error_code, latency_ms) from None
    finally:
        try:
            await client.close()
        except Exception:
            # 客户端关闭失败不应覆盖已经得到的列表，也不记录可能包含敏感信息的异常文本。
            logger.warning("[系统设置] 模型列表客户端关闭失败 service=%s", service)


async def _run_model_connection_test(
    payload: ModelConnectionTest,
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> ModelConnectionTestOut:
    """发起最小真实模型请求；多模态按“文本连接→图片输入”两阶段验证。"""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    started = time.monotonic()

    def elapsed_ms() -> int:
        return round((time.monotonic() - started) * 1000)

    def log_failure(exc: Exception, *, stage: str) -> None:
        status_code = getattr(exc, "status_code", None)
        request_id = getattr(exc, "request_id", None)
        if not isinstance(request_id, str) or len(request_id) > 128 or any(
            ord(char) < 32 or ord(char) == 127 for char in request_id
        ):
            request_id = None
        logger.warning(
            "[系统设置] 模型连接测试失败 service=%s stage=%s error=%s status=%s request_id=%s",
            payload.service,
            stage,
            type(exc).__name__,
            status_code or "-",
            request_id or "-",
        )

    try:
        if payload.service == "embedding":
            response = await client.embeddings.create(
                model=model,
                input="系统配置连接测试",
                timeout=15,
            )
            dimensions = len(response.data[0].embedding) if response.data else 0
            expected = get_settings().embedding_dimensions
            if dimensions != expected:
                return ModelConnectionTestOut(
                    ok=False,
                    message=f"向量模型返回 {dimensions} 维，当前知识库需要 {expected} 维",
                    embedding_dimensions=dimensions,
                    latency_ms=elapsed_ms(),
                    error_code="embedding_dimension_mismatch",
                )
            return ModelConnectionTestOut(
                ok=True,
                message="向量模型连接成功",
                embedding_dimensions=dimensions,
                latency_ms=elapsed_ms(),
            )

        if payload.service == "vision":
            # 第一步只验证地址、密钥、模型名称和 Chat Completions 文本接口。
            try:
                await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "请只回复 OK"}],
                    max_tokens=8,
                    timeout=15,
                )
            except Exception as exc:
                log_failure(exc, stage="text")
                return ModelConnectionTestOut(
                    ok=False,
                    message="多模态模型的基础文本接口连接失败，请检查 API Key、Base URL、模型名称和网络连通性",
                    latency_ms=elapsed_ms(),
                    error_code=f"vision_text_{type(exc).__name__}",
                )

            # 第二步使用与正式图片识别相同的 Chat Completions image_url Data URL 协议。
            try:
                await client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请简短描述图片中的形状和颜色"},
                            {"type": "image_url", "image_url": {"url": _CONNECTION_TEST_IMAGE}},
                        ],
                    }],
                    temperature=0,
                    max_tokens=32,
                    timeout=15,
                )
            except Exception as exc:
                log_failure(exc, stage="image")
                return ModelConnectionTestOut(
                    ok=False,
                    message=(
                        "模型文本接口连接成功，但图片输入失败；当前服务可能不兼容 "
                        "Chat Completions 的 image_url Base64 格式，或上游服务暂时异常"
                    ),
                    latency_ms=elapsed_ms(),
                    error_code=f"vision_image_{type(exc).__name__}",
                )

            return ModelConnectionTestOut(
                ok=True,
                message="多模态模型连接及图片输入测试成功",
                latency_ms=elapsed_ms(),
            )

        structured = await create_structured_completion(
            client,
            request={
                "model": model,
                "messages":[
                    {
                        "role": "system",
                        "content": "输出一个 JSON 对象，内容为 {\"ok\":true}。",
                    },
                    {"role": "user", "content": "请执行结构化输出连接测试。"},
                ],
                "temperature": 0,
                "max_tokens": 16,
            },
            strict_response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "connection_probe_v1",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}},
                    },
                },
            },
            timeout_seconds=15,
            provider_identity=base_url,
            model=model,
        )
        return ModelConnectionTestOut(
            ok=True,
            message=f"模型连接及结构化输出测试成功（{structured.mode}）",
            latency_ms=elapsed_ms(),
            structured_output_mode=structured.mode,
            structured_output_attempted_modes=list(structured.attempted_modes),
            thinking_disabled=structured.thinking_disabled,
        )
    except Exception as exc:
        error_code = type(exc).__name__
        log_failure(exc, stage="connection")
        return ModelConnectionTestOut(
            ok=False,
            message="模型连接失败，请检查 API Key、Base URL、模型名称和网络连通性",
            latency_ms=elapsed_ms(),
            error_code=error_code,
        )
    finally:
        await client.close()


@router.get("", response_model=SettingsOut)
async def get_settings_api(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(SETTINGS_READ)),
):
    return await _load(db)


@router.get("/site", response_model=SiteSettingsOut)
async def get_site_settings(db: AsyncSession = Depends(get_db)):
    return await _load_site(db)


@router.post("/test-connection", response_model=ModelConnectionTestOut)
async def test_model_connection(
    payload: ModelConnectionTest,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(SETTINGS_WRITE)),
):
    """测试尚未保存的表单配置，密钥不写入日志、审计明细或数据库。"""
    api_key, base_url, model, host = _test_config(payload)
    result = await _run_model_connection_test(
        payload,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    # 成功与失败都留下一条不含密钥的审计记录，便于排查服务器网络或模型配置问题。
    audit.log(
        db,
        "settings.connection_test",
        target_type="settings",
        detail={
            "service": payload.service,
            "model": model,
            "host": host,
            "ok": result.ok,
            "latency_ms": result.latency_ms,
            "error_code": result.error_code,
        },
    )
    await db.commit()
    return result


@router.post("/models", response_model=ModelListOut)
async def list_models(
    payload: ModelListRequest,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(SETTINGS_WRITE)),
):
    """获取 OpenAI 兼容服务的模型 ID 列表，供设置页选择模型。

    调用可使用本次表单临时填写的 API Key；若未填写，则仅在 Base URL 未改变时
    使用已保存的 Key。这样既支持“先获取、再保存”，也不会将旧 Key 发送至新地址。
    """
    api_key, base_url, host = _model_list_config(payload)
    try:
        models, latency_ms = await _fetch_model_ids(
            service=payload.service,
            api_key=api_key,
            base_url=base_url,
        )
    except _ModelListFetchError as exc:
        audit.log(
            db,
            "settings.model_list",
            target_type="settings",
            detail={
                "service": payload.service,
                "host": host,
                "ok": False,
                "latency_ms": exc.latency_ms,
                "error_code": exc.error_code,
            },
        )
        await db.commit()
        raise HTTPException(
            status_code=502,
            detail=(
                "无法获取模型列表，请检查 API Key、Base URL、网络连通性，"
                "或确认服务商支持 OpenAI 兼容的 /models 接口"
            ),
        ) from None

    # 仅记录调用结果摘要；候选 Key、已保存 Key 与完整模型列表均不进入审计。
    audit.log(
        db,
        "settings.model_list",
        target_type="settings",
        detail={
            "service": payload.service,
            "host": host,
            "ok": True,
            "latency_ms": latency_ms,
            "model_count": len(models),
        },
    )
    await db.commit()
    return ModelListOut(models=models, latency_ms=latency_ms)


@router.post("/logo")
async def upload_logo(
    file: UploadFile = File(...),
    _: User = Depends(require_permission(SETTINGS_WRITE)),
):
    """上传站点图标，保存到 upload_dir/branding/，返回可访问的静态 URL。"""
    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if ext not in LOGO_EXTS:
        raise HTTPException(status_code=400, detail="仅支持 PNG / JPG / JPEG / WEBP / GIF / SVG / ICO 图片")

    settings = get_settings()
    branding_dir = os.path.join(settings.upload_dir, "branding")
    os.makedirs(branding_dir, exist_ok=True)
    saved_name = f"{uuid.uuid4()}.{ext}"
    saved_path = os.path.join(branding_dir, saved_name)

    contents = await file.read()
    with open(saved_path, "wb") as f_out:
        f_out.write(contents)

    return {"url": f"/api/uploads/branding/{saved_name}"}


@router.put("", response_model=SettingsOut)
async def update_settings(
    payload: SettingsUpdate,
    db: AsyncSession = Depends(get_db),
    audit: AuditLogger = Depends(get_audit),
    _: User = Depends(require_permission(SETTINGS_WRITE)),
):
    updates = payload.model_dump(exclude_none=True)

    # 前端只在填写新密钥时提交；掩码、空值均不能覆盖已有密钥。
    for key_field in SECRET_SETTING_KEYS:
        value = updates.get(key_field)
        if value is not None and _is_secret_placeholder(value):
            updates.pop(key_field, None)

    settings = get_settings()
    for optional_model_field in ("intent_model", "rerank_model"):
        if optional_model_field in updates:
            # 空值表示复用对话模型；统一去除首尾空白，避免模型 ID 隐性失效。
            updates[optional_model_field] = str(updates[optional_model_field]).strip()
    _validate_model_base_url_updates(updates, settings)
    _validate_settings_contract(updates, settings)
    if any(key in SECRET_SETTING_KEYS for key in updates) and not settings.config_encryption_key:
        raise HTTPException(
            status_code=503,
            detail="服务未配置 CONFIG_ENCRYPTION_KEY，无法安全保存 API Key",
        )
    await _validate_embedding_update(updates, settings)

    for key, value in updates.items():
        stored_value = str(value)
        if key in SECRET_SETTING_KEYS:
            try:
                stored_value = encrypt_setting_secret(stored_value, settings.config_encryption_key)
            except SettingsEncryptionError as exc:
                raise HTTPException(status_code=503, detail="无法加密保存 API Key") from exc

        row = await db.get(SystemSetting, key)
        if row:
            row.value = stored_value
        else:
            db.add(SystemSetting(key=key, value=stored_value))

    if updates:
        audit.log(
            db,
            "settings.update",
            target_type="settings",
            detail={"changed": sorted(updates.keys())},
        )
    await db.commit()

    # 当前 API 进程即时生效；独立文档 worker 会在领取下一条任务时重新加载。
    # 重启后两个进程也都会从数据库恢复配置。
    for key, value in updates.items():
        setattr(settings, key, value)
    if any(
        key in updates
        for key in (
            "llm_base_url",
            "chat_model",
            "llm_structured_output_mode",
            "llm_disable_thinking",
        )
    ):
        # 端点、模型或管理员选择变化后，旧能力结论不能污染新配置。
        clear_structured_output_capability_cache()
    if any(
        key in updates
        for key in (
            "llm_base_url",
            "chat_model",
            "rerank_model",
            "llm_structured_output_mode",
            "llm_disable_thinking",
            "rerank_timeout_seconds",
        )
    ):
        # 熔断键虽然包含端点、模型和合同，但旧进程状态没有配置 revision。
        # 配置变化时整体清理，避免新模型继承旧模型的失败窗口。
        clear_rerank_circuit_breakers()

    return await _load(db)
