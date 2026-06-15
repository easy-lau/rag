#!/bin/bash
set -e

SERVER="root@10.168.120.42"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/data/engineering/rag"

echo ">>> 切换到远程 Docker Context..."
docker context use remote

echo ">>> 同步代码到服务器..."
rsync -avz \
  --exclude 'node_modules' \
  --exclude '__pycache__' \
  --exclude '.env' \
  --exclude '*.pyc' \
  --exclude 'dist' \
  --exclude '.git' \
  -e "ssh -i $SSH_KEY -p 22" \
  ./ $SERVER:$REMOTE_DIR/

echo ">>> 同步 .env 文件..."
scp -P 22 -i $SSH_KEY .env $SERVER:$REMOTE_DIR/.env

echo ">>> 构建并启动..."
docker compose up --build -d

echo ">>> 完成！访问 http://10.168.120.42"
echo ">>> 查看日志：docker compose logs -f"
