FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 依赖全部走预编译 wheel：asyncpg / bcrypt / cryptography 均有 manylinux 包，
# 且 asyncpg 运行时不依赖 libpq，故无需 gcc / libpq-dev，省掉缓慢的 apt 安装。
# pip 默认走清华镜像（国内快）；海外部署可用
#   docker compose build --build-arg PIP_INDEX_URL=https://pypi.org/simple
# 覆盖。
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -i ${PIP_INDEX_URL} -r requirements.txt

COPY . .

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

# Alembic 由 Compose 中的一次性 migrate 服务执行，避免应用重启或多副本并发迁移。
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
