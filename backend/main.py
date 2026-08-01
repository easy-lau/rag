import os
import logging
import asyncio
from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from api import (
    auth,
    chat,
    document,
    intent_routing,
    knowledge,
    login_logs,
    operation_logs,
    rag_traces,
    roles,
    search,
    settings,
    uploads,
    users,
)
from config import get_settings
from core.logging_config import configure_application_logging
from core.settings_crypto import SettingsEncryptionError
from core.login_security import login_log_cleanup_loop
from core.rag_trace_store import start_rag_trace_store, stop_rag_trace_store
from database import engine

_settings = get_settings()
_log_level = getattr(logging, _settings.log_level.strip().upper(), logging.INFO)
_development_log_path = configure_application_logging(
    app_env=_settings.app_env,
    log_level=_log_level,
    development_log_dir=_settings.development_log_dir,
)
if _development_log_path is not None:
    logging.getLogger(__name__).info("[startup] 开发日志文件: %s", _development_log_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载数据库中保存的设置（API Key、Base URL、模型等）
    try:
        await settings.apply_stored_settings()
    except SettingsEncryptionError:
        # 密钥丢失或密文损坏时不能静默回退到环境变量，否则会以错误配置继续处理用户请求。
        logging.getLogger(__name__).critical("[startup] 无法解密数据库中的模型密钥，拒绝启动应用")
        raise
    except Exception as exc:
        # 数据库设置是运行时模型配置的唯一来源，加载失败时不能以默认配置继续对外提供服务。
        logging.getLogger(__name__).error(
            "[startup] 加载数据库系统设置失败，拒绝启动应用 error=%s",
            type(exc).__name__,
        )
        raise
    # 重置上次进程退出时卡在『处理中』的文档（其后台任务已随进程丢失）
    try:
        await document.reset_stuck_processing()
    except Exception as e:
        print(f"[startup] 重置卡死文档失败: {e}")
    login_log_cleanup_task = asyncio.create_task(login_log_cleanup_loop())
    await start_rag_trace_store()
    try:
        yield
    finally:
        await stop_rag_trace_store()
        login_log_cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await login_log_cleanup_task


app = FastAPI(title="RAG 检索系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Conversation-ID",
        "X-RAG-Trace-ID",
        "X-RAG-Request-ID",
        "X-RAG-Turn-ID",
        "X-RAG-Trace-Truncated",
        "X-RAG-Trace-Omitted-Events",
        "Retry-After",
    ],
)

app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(roles.router, prefix="/api")
app.include_router(login_logs.router, prefix="/api")
app.include_router(operation_logs.router, prefix="/api")
app.include_router(rag_traces.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(document.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(intent_routing.router, prefix="/api")
app.include_router(uploads.router, prefix="/api")

# 品牌图需要在登录页公开显示；文档原图由 uploads 路由鉴权后返回，不能
# 再把整个 upload_dir 作为无鉴权静态目录挂载。
_upload_dir = _settings.upload_dir
os.makedirs(os.path.join(_upload_dir, "images"), exist_ok=True)
_branding_dir = os.path.join(_upload_dir, "branding")
os.makedirs(_branding_dir, exist_ok=True)
app.mount(
    "/api/uploads/branding",
    StaticFiles(directory=_branding_dir),
    name="branding_uploads",
)


@app.get("/api/health")
async def health():
    try:
        async with asyncio.timeout(2):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
    except Exception as exc:
        logging.getLogger(__name__).warning("健康检查连接数据库失败: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database unavailable",
        ) from exc
    return {"status": "ok", "database": "ok"}
