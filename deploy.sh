#!/usr/bin/env bash
set -Eeuo pipefail

RAG_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAG_PROJECT_DIR"
RAG_DOCKER_CONTEXT="${RAG_DOCKER_CONTEXT:-default}"
RAG_DOCKER=(docker --context "$RAG_DOCKER_CONTEXT")

if ! command -v docker >/dev/null 2>&1; then
  echo "错误：未找到 docker，请先安装 Docker Engine 和 Docker Compose v2。" >&2
  exit 1
fi

if ! docker context inspect "$RAG_DOCKER_CONTEXT" >/dev/null 2>&1; then
  echo "错误：Docker Context '$RAG_DOCKER_CONTEXT' 不存在。" >&2
  exit 1
fi

if ! "${RAG_DOCKER[@]}" compose version >/dev/null 2>&1; then
  echo "错误：未找到 docker compose v2。" >&2
  exit 1
fi

if ! "${RAG_DOCKER[@]}" buildx version >/dev/null 2>&1; then
  echo "错误：未找到 docker buildx，请先安装 docker-buildx-plugin。" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "错误：缺少 .env。请先执行 cp .env.example .env，并填写三个必填密钥。" >&2
  exit 1
fi

for RAG_REQUIRED_KEY in POSTGRES_PASSWORD JWT_SECRET ADMIN_INIT_PASSWORD; do
  if ! grep -Eq "^${RAG_REQUIRED_KEY}=.+$" .env; then
    echo "错误：.env 中的 ${RAG_REQUIRED_KEY} 未填写。" >&2
    exit 1
  fi
done

echo ">>> 目标 Docker Context：$RAG_DOCKER_CONTEXT"

echo ">>> 校验 Compose 配置"
"${RAG_DOCKER[@]}" compose config --quiet

echo ">>> 构建镜像"
"${RAG_DOCKER[@]}" compose build --pull

echo ">>> 启动数据库、执行迁移并启动应用"
"${RAG_DOCKER[@]}" compose up -d --force-recreate --remove-orphans \
  migrate app

echo ">>> 等待端到端健康检查"
for _ in $(seq 1 60); do
  if "${RAG_DOCKER[@]}" compose exec -T app \
    python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/api/health', timeout=3)" \
    >/dev/null 2>&1; then
    "${RAG_DOCKER[@]}" compose ps -a
    echo ">>> 部署完成。请按 .env 中的 APP_BIND_HOST/APP_PORT 或 HTTPS 域名访问。"
    exit 0
  fi
  sleep 2
done

echo "错误：应用在 120 秒内未通过健康检查。" >&2
"${RAG_DOCKER[@]}" compose ps -a
"${RAG_DOCKER[@]}" compose logs --tail=100 postgres migrate app
exit 1
