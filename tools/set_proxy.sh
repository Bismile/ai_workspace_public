#!/usr/bin/env bash
# tools/set_proxy.sh — 统一更新工作区代理配置
#
# 用法:
#   ./tools/set_proxy.sh http://127.0.0.1:20809   # 设置代理
#   ./tools/set_proxy.sh ""                        # 清除代理
#   ./tools/set_proxy.sh                           # 同上（清除）
#
# 会更新的地方：
#   tools/config.sh          （manage_repos.sh 运行时读取）
#   prompts/**/*.md          （提示词里的代理示例）
#
# 不会更新的地方：
#   research/   projects/    （第三方/子仓库，不属于本 workspace）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$SCRIPT_DIR/config.sh"
NEW_PROXY="${1:-}"

# ── 读取当前代理地址（用于替换 prompts/ 里的旧值）──────────
OLD_PROXY=""
if [[ -f "$CONFIG_FILE" ]]; then
    OLD_PROXY=$(grep -E '^PROXY=' "$CONFIG_FILE" 2>/dev/null \
        | head -1 | sed 's/^PROXY="\(.*\)"/\1/' || true)
fi

# ── 更新 tools/config.sh ───────────────────────────────────
{
cat << 'HEREDOC'
# manage_repos.sh 本地配置（此文件不进 git，按需修改）
#
# 代理地址：填写后所有 git/push 操作自动挂载，留空则不挂代理
HEREDOC
if [[ -n "$NEW_PROXY" ]]; then
    echo "PROXY=\"$NEW_PROXY\""
else
    echo "# PROXY=\"http://127.0.0.1:20809\""
    echo "PROXY=\"\""
fi
} > "$CONFIG_FILE"

if [[ -n "$NEW_PROXY" ]]; then
    echo "✅ tools/config.sh → PROXY=\"$NEW_PROXY\""
else
    echo "✅ tools/config.sh → PROXY=\"\"（已清除）"
fi

# ── 扫描 prompts/ 并替换代理地址 ──────────────────────────
# 清除时用占位符替换，保留示例的可读性
REPLACE_WITH="${NEW_PROXY:-http://<host>:<port>}"

if [[ -z "$OLD_PROXY" ]]; then
    echo "ℹ️  prompts/ 无需更新（原代理地址为空）"
elif [[ "$OLD_PROXY" == "$NEW_PROXY" ]]; then
    echo "ℹ️  prompts/ 无需更新（与旧地址相同）"
else
    echo "🔄 扫描 prompts/ ..."
    COUNT=0
    while IFS= read -r -d '' f; do
        if grep -qF "$OLD_PROXY" "$f" 2>/dev/null; then
            sed -i "s|${OLD_PROXY}|${REPLACE_WITH}|g" "$f"
            echo "  ✔ $(realpath --relative-to="$REPO_ROOT" "$f")"
            COUNT=$((COUNT + 1))
        fi
    done < <(find "$REPO_ROOT/prompts" -type f \
        \( -name "*.md" -o -name "*.txt" -o -name "*.sh" -o -name "*.yaml" \) \
        -print0 2>/dev/null)
    echo "  共更新 $COUNT 个文件"
fi

echo ""
echo "当前 shell 若需立即生效，请手动运行："
if [[ -n "$NEW_PROXY" ]]; then
    echo "  export ALL_PROXY=$NEW_PROXY"
else
    echo "  unset ALL_PROXY https_proxy http_proxy"
fi
