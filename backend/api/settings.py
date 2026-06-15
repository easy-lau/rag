import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models.db_models import SystemSetting, User
from config import get_settings
from pydantic import BaseModel
from core.deps import require_permission
from core.permissions import SETTINGS_READ, SETTINGS_WRITE

router = APIRouter(prefix="/settings", tags=["settings"])

LOGO_EXTS = {"png", "jpg", "jpeg", "webp", "gif", "svg", "ico"}


class SettingsOut(BaseModel):
    llm_api_key: str
    llm_base_url: str
    chat_model: str
    temperature: float
    max_tokens: int
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    vision_api_key: str
    vision_base_url: str
    vision_model: str
    top_k: int
    rerank_enabled: bool
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
    temperature: float | None = None
    max_tokens: int | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    vision_api_key: str | None = None
    vision_base_url: str | None = None
    vision_model: str | None = None
    top_k: int | None = None
    rerank_enabled: bool | None = None
    show_sources: bool | None = None
    site_title: str | None = None
    site_description: str | None = None
    site_logo: str | None = None
    browser_title: str | None = None
    site_copyright: str | None = None


def _mask(key: str) -> str:
    if not key or len(key) < 8:
        return key or ""
    return key[:5] + "***" + key[-3:]


async def apply_stored_settings() -> None:
    """启动时把数据库 settings 表里保存的配置加载进运行时 Settings，
    使设置页保存的 Key / Base URL / 模型在重启后依然生效。"""
    from database import AsyncSessionLocal
    s = get_settings()
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(SystemSetting))).scalars().all()
    for row in rows:
        if row.value is None or not hasattr(s, row.key):
            continue
        current = getattr(s, row.key)
        try:
            if isinstance(current, bool):
                coerced = str(row.value).strip().lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                coerced = int(row.value)
            elif isinstance(current, float):
                coerced = float(row.value)
            else:
                coerced = row.value
            setattr(s, row.key, coerced)
        except (ValueError, TypeError):
            continue


async def _load(db: AsyncSession) -> dict:
    rows = (await db.execute(select(SystemSetting))).scalars().all()
    db_map = {r.key: r.value for r in rows}
    s = get_settings()

    def v(key, default):
        return db_map.get(key, str(default))

    return {
        "llm_api_key":      _mask(db_map.get("llm_api_key") or s.llm_api_key),
        "llm_base_url":     v("llm_base_url", s.llm_base_url),
        "chat_model":       v("chat_model", s.chat_model),
        "temperature":      float(v("temperature", s.temperature)),
        "max_tokens":       int(v("max_tokens", s.max_tokens)),
        "embedding_api_key": _mask(db_map.get("embedding_api_key") or s.embedding_api_key),
        "embedding_base_url": v("embedding_base_url", s.embedding_base_url),
        "embedding_model":  v("embedding_model", s.embedding_model),
        "vision_api_key":   _mask(db_map.get("vision_api_key") or s.vision_api_key),
        "vision_base_url":  v("vision_base_url", s.vision_base_url),
        "vision_model":     v("vision_model", s.vision_model),
        "top_k":            int(v("top_k", s.top_k)),
        "rerank_enabled":   v("rerank_enabled", s.rerank_enabled).lower() == "true",
        "show_sources":     v("show_sources", s.show_sources).lower() == "true",
        "site_title":       v("site_title", s.site_title),
        "site_description": v("site_description", s.site_description),
        "site_logo":        v("site_logo", s.site_logo),
        "browser_title":    v("browser_title", s.browser_title),
        "site_copyright":   v("site_copyright", s.site_copyright),
    }


class SiteSettingsOut(BaseModel):
    site_title: str
    site_description: str
    site_logo: str
    browser_title: str
    site_copyright: str


async def _load_site(db: AsyncSession) -> dict:
    """读取公开品牌字段：从数据库 settings 表取这 4 个 key，缺失则回退 config 默认。"""
    rows = (await db.execute(select(SystemSetting))).scalars().all()
    db_map = {r.key: r.value for r in rows}
    s = get_settings()
    return {
        "site_title":       db_map.get("site_title") or s.site_title,
        "site_description": db_map.get("site_description") or s.site_description,
        "site_logo":        db_map.get("site_logo") or s.site_logo,
        "browser_title":    db_map.get("browser_title") or s.browser_title,
        "site_copyright":   db_map.get("site_copyright") or s.site_copyright,
    }


@router.get("", response_model=SettingsOut)
async def get_settings_api(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(require_permission(SETTINGS_READ)),
):
    return await _load(db)


@router.get("/site", response_model=SiteSettingsOut)
async def get_site_settings(db: AsyncSession = Depends(get_db)):
    """公开端点（无鉴权）：供登录页、普通用户、浏览器标题读取站点品牌信息。
    仅返回 4 个非敏感品牌字段，绝不暴露 API Key 等完整设置。"""
    return await _load_site(db)


@router.post("/logo")
async def upload_logo(
    file: UploadFile = File(...),
    _: User = Depends(require_permission(SETTINGS_WRITE)),
):
    """上传站点图标，保存到 upload_dir/branding/，返回可访问的静态 URL。"""
    ext = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
    if ext not in LOGO_EXTS:
        raise HTTPException(status_code=400, detail="仅支持 PNG / JPG / JPEG / WEBP / GIF / SVG / ICO 图片")

    s = get_settings()
    branding_dir = os.path.join(s.upload_dir, "branding")
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
    _: User = Depends(require_permission(SETTINGS_WRITE)),
):
    updates = payload.model_dump(exclude_none=True)

    # 绝不用空值或掩码占位符覆盖已存储的 API Key
    for key_field in ("llm_api_key", "embedding_api_key", "vision_api_key"):
        val = updates.get(key_field)
        if val is not None and (not val.strip() or "***" in val):
            updates.pop(key_field, None)

    for key, value in updates.items():
        row = await db.get(SystemSetting, key)
        if row:
            row.value = str(value)
        else:
            db.add(SystemSetting(key=key, value=str(value)))
    await db.commit()

    # 刷新内存配置
    s = get_settings()
    for key, value in updates.items():
        if hasattr(s, key):
            setattr(s, key, value)

    return await _load(db)
