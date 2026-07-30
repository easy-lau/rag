# 前端构建阶段
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./

ARG APP_VERSION=dev
ARG APP_REVISION=
ENV VITE_APP_VERSION=${APP_VERSION} \
    VITE_APP_REVISION=${APP_REVISION}

RUN npm run build

# 前后端合一运行镜像
FROM python:3.11-slim-bookworm

WORKDIR /app

ARG APP_VERSION=dev
ARG APP_REVISION=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION=${APP_VERSION} \
    APP_REVISION=${APP_REVISION}

ARG PIP_INDEX_URL=https://pypi.org/simple

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/conf.d/default.conf

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
        --retries 5 \
        --timeout 60 \
        --index-url "${PIP_INDEX_URL}" \
        -r requirements.txt

COPY backend/ ./
COPY --from=frontend-builder /frontend/dist/ /usr/share/nginx/html/
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/supervisord.conf /etc/supervisor/conf.d/rag.conf

RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /app/uploads \
    && nginx -t

EXPOSE 8001

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/rag.conf"]
