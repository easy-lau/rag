#!/bin/sh
set -eu

if [ -z "${DATABASE_URL:-}" ]; then
  echo "错误：DATABASE_URL 未设置。" >&2
  exit 1
fi

if [ -z "${JWT_SECRET:-}" ]; then
  echo "错误：JWT_SECRET 未设置，请先填写根目录 .env。" >&2
  exit 1
fi

if [ -z "${ADMIN_INIT_PASSWORD:-}" ]; then
  echo "错误：ADMIN_INIT_PASSWORD 未设置，请先填写根目录 .env。" >&2
  exit 1
fi

exec "$@"
