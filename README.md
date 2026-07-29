# RAG 检索系统

企业知识库问答系统：文档（PDF / Word / PPT / Excel / Markdown / 图片）入库 → 向量与关键词混合检索 → 重排 → LLM 基于知识库作答。包含用户、角色、细粒度权限、智能路由、登录日志、操作审计和站点设置。

## 技术栈

- 后端：FastAPI · SQLAlchemy(async) · PostgreSQL + pgvector · Alembic
- 前端：Vue 3 · Vite · Naive UI · Tailwind
- 模型：OpenAI 兼容的 Chat / Embedding / Vision 接口

## 前端 UI 开发约定

新增或修改前端页面、组件、弹窗、下拉和响应式交互前，请先阅读 [UI 开发规范](docs/UI开发规范.md)。该规范是本项目 UI 的统一验收标准。

## Docker Compose 架构

仓库只保留一个 `docker-compose.yml`，直接拉取阿里云 ACR 中的前后端合一发布镜像：

```text
浏览器 :8001 → app(Nginx + FastAPI) → postgres(pgvector)
                    ↑
            migrate(Alembic，一次性任务)
```

- 全栈部署：`docker compose pull && docker compose up -d`
- 本地只启动数据库：`docker compose up -d postgres`
- PostgreSQL 只绑定宿主机 `127.0.0.1:5433`
- FastAPI `8000` 只在应用容器内部开放，Nginx 对外提供 `8001`
- 数据保存于 `rag_pg_data` 和 `rag_upload_data` 两个 Docker volume

## GitHub Release 自动发布镜像

工作流文件为 [.github/workflows/release-image.yml](/Users/liuxing/Engineering/Personal/rag/.github/workflows/release-image.yml:1)。发布 GitHub Release 后，会构建前后端合一镜像并推送到：

```text
crpi-ixo2xnp7izugsz4z.cn-hangzhou.personal.cr.aliyuncs.com/easylau/rag:<版本号>
crpi-ixo2xnp7izugsz4z.cn-hangzhou.personal.cr.aliyuncs.com/easylau/rag:latest
```

在 GitHub 仓库的 `Settings → Secrets and variables → Actions` 中配置：

```text
ACR_USERNAME   阿里云容器镜像服务用户名
ACR_PASSWORD   阿里云容器镜像服务密码或访问凭证
```

Release 标签 `v1.2.3` 会生成镜像版本 `1.2.3`。工作流同时构建 `linux/amd64` 和 `linux/arm64`。

Compose 已固定使用阿里云镜像仓库；服务器仅需在 `.env` 中选择要部署的版本：

```dotenv
IMAGE_TAG=1.2.3
```

镜像仓库当前为公开仓库，服务器可直接拉取并更新：

```bash
docker compose pull
docker compose up -d --force-recreate migrate app
```

如后续将仓库改为私有仓库，再先执行：

```bash
docker login crpi-ixo2xnp7izugsz4z.cn-hangzhou.personal.cr.aliyuncs.com
```

## 新服务器首次部署

前置条件：Linux 服务器已安装 Git、Docker Engine 和 Docker Compose v2.20+。

```bash
docker version
docker compose version
docker context show

sudo mkdir -p /opt/rag
sudo chown "$USER":"$USER" /opt/rag
git clone https://github.com/easy-lau/rag.git /opt/rag
cd /opt/rag

cp .env.example .env
chmod 600 .env
```

这些命令应登录目标服务器后执行，`docker context show` 通常应为 `default`；不要在来源不明的远程 Context 上直接部署。

生成三个不同的随机值：

```bash
openssl rand -hex 24
openssl rand -hex 32
openssl rand -hex 16
```

编辑 `.env`，至少填写：

```dotenv
POSTGRES_PASSWORD=<第一个随机值>
JWT_SECRET=<第二个随机值>
ADMIN_INIT_PASSWORD=<第三个随机值或自定义强密码>
# 可填 latest；生产环境建议固定为已发布版本，例如 1.2.3。
IMAGE_TAG=latest
```

不要在 Docker 部署的 `.env` 中设置 `DATABASE_URL`。配置检查通过后，拉取镜像、迁移并启动：

```bash
docker compose config --quiet
docker compose pull
docker compose up -d
```

检查启动结果：

```bash
docker compose ps -a
docker compose exec -T app alembic current
curl --fail --silent --show-error http://127.0.0.1:8001/api/health
docker compose logs --tail=100 postgres migrate app
```

正常结果：

- `postgres`、`app` 为 `running/healthy`
- `migrate` 为 `Exited (0)`，这是一次性迁移成功后的正常状态
- Alembic 显示 `0020 (head)` 或仓库后续更新后的最新 head

默认访问地址为 `http://服务器IP:8001`。初始用户名为 `admin`，密码是首次迁移前填写的 `ADMIN_INIT_PASSWORD`；数据库初始化后再修改该环境变量不会修改现有管理员密码。

也可以执行仓库内的安全包装脚本完成同样的首次构建与启动：

```bash
bash deploy.sh
```

脚本默认明确使用 `default` Docker Context，不会沿用机器上误切换的远程 Context。如确实需要指定其他 Context：

```bash
RAG_DOCKER_CONTEXT=my-server bash deploy.sh
```

该 Compose 依赖 `service_completed_successfully`，只支持 Docker Compose v2，不要使用 `docker stack deploy`。

## 公网 HTTPS

直接使用 `服务器IP:8001` 适合内网或临时验收。正式公网环境应做到：

1. 将 `.env` 中的 `APP_BIND_HOST` 改为 `127.0.0.1`。
2. 域名 A/AAAA 记录指向服务器。
3. 使用宿主机 Caddy 或 Nginx 将 HTTPS 反向代理到 `127.0.0.1:8001`。
4. 防火墙只开放管理所需的 SSH 端口以及 `80/443`，不要开放 `5433/8000/8001`。

Caddy 最小配置示例：

```caddyfile
rag.example.com {
    reverse_proxy 127.0.0.1:8001
}
```

Caddy 会自动申请和续期证书。项目内层 Nginx 已配置 50 MB 上传限制、SSE 长连接以及代理请求头。

## 模型配置

生产部署时请在容器 `.env` 中填写 LLM、Embedding 和 Vision（如需）模型参数；修改后需重建 `app` 容器使其生效。需要特别注意：

- Embedding 模型接口必须实际返回 `2560` 维向量。
- 当前代码不会主动向兼容接口传递 `dimensions` 参数，仅填写维度环境变量不能改变模型输出。
- 模型 API Key 会进入系统配置和数据库备份，备份文件应按敏感数据保护。
- 聊天流默认单次超时 `60` 秒；仅在首个回答文本前发生超时、连接异常、限流或上游 5xx 时按 `1` 秒、`2` 秒退避重试，最多 `3` 次总尝试。回答开始输出后不会重试，避免重复内容。
- 向量化默认单次超时 `60` 秒，发生超时、连接异常、限流或上游 5xx 时按 `1` 秒、`2` 秒退避重试，最多 `3` 次总尝试。可用 `.env` 的 `EMBEDDING_*` 重试参数覆盖。

## 本地开发

先按首次部署章节创建根目录 `.env`，然后只启动 PostgreSQL：

```bash
docker compose up -d postgres
```

后端连接宿主机的 `127.0.0.1:5433`：

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL='postgresql+asyncpg://rag:<你的POSTGRES_PASSWORD>@127.0.0.1:5433/rag_prod'
alembic upgrade head
uvicorn main:app --reload --port 8000
```

另一个终端启动前端：

```bash
cd frontend
npm install
npm run dev
```

停止本地数据库：

```bash
docker compose stop postgres
```

只启动 PostgreSQL 时，Compose 不要求 JWT 和管理员密码；完整启动 `migrate/app` 时，镜像入口会再次校验三个必填值并在缺失时拒绝启动。

## 日志与日常操作

```bash
docker compose ps -a
docker compose logs -f app
docker compose restart app
docker compose stop
docker compose up -d
```

不要执行 `docker compose down -v`，`-v` 会删除数据库和上传文件卷。

## 备份与升级

升级前停止写入，并同时备份数据库与上传文件：

```bash
cd /opt/rag
docker compose stop app migrate

RAG_BACKUP_DIR="backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RAG_BACKUP_DIR"

docker compose exec -T postgres \
  sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > "$RAG_BACKUP_DIR/database.dump"

docker compose cp app:/app/uploads "$RAG_BACKUP_DIR/uploads"
git rev-parse HEAD > "$RAG_BACKUP_DIR/git-commit.txt"
```

把备份复制到服务器之外的加密存储并定期演练恢复。然后执行更新：

```bash
git pull --ff-only origin main
docker compose pull
docker compose up -d --force-recreate --remove-orphans \
  migrate app

docker compose exec -T app alembic current
curl --fail --silent --show-error http://127.0.0.1:8001/api/health
docker compose logs --tail=100 migrate app
```

权限迁移等不兼容版本禁止新旧后端同时运行；迁移失败时不要强行启动新后端，应先处理迁移错误或使用匹配的旧代码与升级前备份恢复。

### 恢复演练

恢复前应切换到 `git-commit.txt` 记录的匹配代码版本，并使用匹配的 `.env`。以下命令会覆盖数据库中的同名对象，应只在明确的恢复维护窗口执行：

```bash
cd /opt/rag
RAG_RESTORE_DIR="/opt/rag/backups/20260729-120000"

docker compose pull
docker compose stop app migrate
docker compose up -d postgres

docker compose exec -T postgres \
  sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-privileges' \
  < "$RAG_RESTORE_DIR/database.dump"

# 以下会清空当前 upload_data 卷后复制备份，仅在确认恢复目标正确时执行。
docker compose run --rm --no-deps \
  --entrypoint sh \
  -v "$RAG_RESTORE_DIR/uploads:/restore:ro" \
  app -c 'find /app/uploads -mindepth 1 -depth -delete && cp -a /restore/. /app/uploads/'

docker compose up -d --force-recreate migrate app

docker compose exec -T app alembic current
curl --fail --silent --show-error http://127.0.0.1:8001/api/health
```

`COMPOSE_PROJECT_NAME` 决定数据库和上传卷的前缀。同一台服务器部署多套环境时必须使用不同项目名；升级旧环境前先通过 `docker compose ls` 和 `docker volume ls` 核对原项目名，不能随意修改，否则会挂载一组新的空卷，看起来像“数据丢失”。
