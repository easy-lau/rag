import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
    users,
)
from config import get_settings

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

# 原始图片静态服务：上传的图片转写为文档后，原图仍可在编辑器对照、并供未来多模态问答回传
_upload_dir = get_settings().upload_dir
os.makedirs(os.path.join(_upload_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(_upload_dir, "branding"), exist_ok=True)
app.mount("/api/uploads", StaticFiles(directory=_upload_dir), name="uploads")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
