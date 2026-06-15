# RAG 检索系统

企业知识库问答系统：文档（PDF / Word / PPT / Excel / Markdown / 图片）入库 → 向量 + 关键词混合检索 + 重排 → LLM 依据检索内容作答。含用户 / 角色 / 权限（RBAC）、登录日志、站点品牌设置。

## 技术栈

- **后端**：FastAPI · SQLAlchemy(async) · PostgreSQL + pgvector · Alembic
- **前端**：Vue 3 · Vite · Naive UI · Tailwind
- **模型**：任意 OpenAI 兼容的 Chat / Embedding / Vision 接口

## 目录

```
backend/    FastAPI 后端（api/ core/ models/ migrations/）
frontend/   Vue 3 前端
docker-compose.yml      一键部署（postgres + backend + frontend）
docker-compose.db.yml   仅起数据库（本地开发用）
```

## 环境变量

复制 `.env.example` 为 `.env`，按文件内注释填写（`.env` 已被 gitignore，切勿提交）。**必填三项**：

- `POSTGRES_PASSWORD` —— ⚠️ 只用字母和数字，**不要含 `@ : / ? # %` 等符号**（会破坏数据库连接串，导致后端连不上库）
- `JWT_SECRET` —— 随机长字符串（`openssl rand -hex 32`）
- `ADMIN_INIT_PASSWORD` —— 初始管理员密码（用户名固定 `admin`；仅首次初始化数据库时生效）

模型相关（`LLM_*` / `EMBEDDING_*` / `VISION_*`）可留空，部署后在「设置」页里填。docker 部署**不要**手动设 `DATABASE_URL`，compose 会自动指向 `postgres` 容器。

## 部署到一台新服务器（Docker，推荐）

前置：目标服务器已装 Docker + Docker Compose。

```bash
git clone <你的仓库地址> rag && cd rag
cp .env.example .env && vim .env        # 填写上面的环境变量
docker compose up --build -d            # 构建并启动 postgres + 后端 + 前端
docker compose exec backend alembic upgrade head   # 执行数据库迁移
```

访问 `http://<服务器IP>`（前端 80 端口），用户名 `admin` / 密码 = `ADMIN_INIT_PASSWORD`。

常用命令：

```bash
docker compose logs -f          # 看日志
docker compose ps               # 看状态
docker compose down             # 停止
docker compose up --build -d    # 更新代码后重新构建
docker compose exec backend alembic upgrade head   # 拉到新迁移后执行
```

## 本地开发

```bash
# 1) 只起数据库
docker compose -f docker-compose.db.yml up -d

# 2) 后端
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --reload --port 8000

# 3) 前端
cd frontend && npm install && npm run dev
```

## 备份

数据在两个 Docker volume：`pg_data`（数据库）、`upload_data`（上传文件）。定期备份：

```bash
docker compose exec postgres pg_dump -U rag rag_db > backup.sql
```
