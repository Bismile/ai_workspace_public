#!/bin/bash
# kill_port.sh — 释放占用指定端口的进程
#
# 用法：
#   bash kill_port.sh 6006      # 释放 TensorBoard 端口
#   bash kill_port.sh 8888      # 释放 Jupyter 端口

set -euo pipefail

PORT="${1:-}"
if [[ -z "$PORT" ]]; then
    echo "用法: bash kill_port.sh <端口号>"
    exit 1
fi

PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [[ -z "$PIDS" ]]; then
    echo "[INFO] 端口 $PORT 未被占用"
    exit 0
fi

echo "[INFO] 占用端口 $PORT 的进程: $PIDS"
echo "$PIDS" | xargs kill -9
echo "[INFO] 已终止"
