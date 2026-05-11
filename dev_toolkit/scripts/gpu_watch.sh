#!/bin/bash
# gpu_watch.sh — 实时监控 GPU 显存/利用率
#
# 用法：
#   bash gpu_watch.sh          # 默认 2 秒刷新
#   bash gpu_watch.sh 5        # 5 秒刷新

set -euo pipefail

INTERVAL="${1:-2}"

watch -n "$INTERVAL" "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits | \
    awk -F',' '{printf \"GPU%s %-24s  util=%3s%%  mem=%5s/%5sMiB  temp=%s°C\n\", \$1,\$2,\$3,\$4,\$5,\$6}'"
