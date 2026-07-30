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

if [ -z "${CONFIG_ENCRYPTION_KEY:-}" ]; then
  echo "错误：CONFIG_ENCRYPTION_KEY 未设置，请先填写根目录 .env。" >&2
  exit 1
fi

if [ -z "${ADMIN_INIT_PASSWORD:-}" ]; then
  echo "错误：ADMIN_INIT_PASSWORD 未设置，请先填写根目录 .env。" >&2
  exit 1
fi

# 安全默认是不信任任何外层代理。只有显式配置的来源 CIDR 才能改写 Nginx 的
# remote_addr；随后传给 FastAPI 的 X-Real-IP 始终由同容器 Nginx 重新生成。
real_ip_config="/etc/nginx/conf.d/00-real-ip.conf"
python - "$real_ip_config" "${TRUSTED_PROXY_CIDRS:-}" <<'PY'
import ipaddress
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
raw = sys.argv[2]
networks = []
for value in raw.replace(",", " ").split():
    try:
        networks.append(str(ipaddress.ip_network(value, strict=False)))
    except ValueError as exc:
        raise SystemExit(f"错误：TRUSTED_PROXY_CIDRS 包含无效地址 {value!r}") from exc

lines = [f"set_real_ip_from {network};" for network in networks]
if networks:
    lines.extend(["real_ip_header X-Forwarded-For;", "real_ip_recursive on;"])
target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
PY

exec "$@"
