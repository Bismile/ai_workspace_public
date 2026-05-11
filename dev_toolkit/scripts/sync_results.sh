#!/bin/bash
# sync_results.sh — 把实验结果（图/日志）同步到指定目标目录
#
# 用法：
#   bash sync_results.sh <源目录> <目标目录>
#   bash sync_results.sh results/ /data/flg/shared/results/my_exp/
#   bash sync_results.sh results/ user@server:/home/user/results/   # 支持远程

set -euo pipefail

SRC="${1:-results/}"
DST="${2:-}"

if [[ -z "$DST" ]]; then
    echo "用法: bash sync_results.sh <源目录> <目标目录>"
    exit 1
fi

echo "[INFO] 同步 $SRC → $DST"
rsync -avz --progress \
    --include="*.png" \
    --include="*.jpg" \
    --include="*.pdf" \
    --include="*.csv" \
    --include="*.json" \
    --include="*.log" \
    --include="*/" \
    --exclude="*" \
    "$SRC" "$DST"
echo "[INFO] 同步完成"
