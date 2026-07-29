import os
import logging
import asyncio
from contextlib import asynccontextmanager
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
    roles,
    search,
    settings,
    uploads,
    users,
)
from config import get_settings
from database import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载数据库中保存的设置（API Key、Base URL、模型等）
    try:
        await settings.apply_stored_settings()
    except Exception as e:
        print(f"[startup] 加载已存设置失败: {e}")
    # 重置上次进程退出时卡在『处理中』的文档（其后台任务已随进程丢失）
    try:
        await document.reset_stuck_processing()
    except Exception as e:
        print(f"[startup] 重置卡死文档失败: {e}")
    yield


app = FastAPI(title="RAG 检索系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-ID"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(roles.router, prefix="/api")
app.include_router(login_logs.router, prefix="/api")
app.include_router(operation_logs.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(document.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(settings.router, prefix="/api")
app.include_router(intent_routing.router, prefix="/api")
app.include_router(uploads.router, prefix="/api")

# 品牌图需要在登录页公开显示；文档原图由 uploads 路由鉴权后返回，不能
# 再把整个 upload_dir 作为无鉴权静态目录挂载。
_upload_dir = get_settings().upload_dir
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
