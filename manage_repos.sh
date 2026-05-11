#!/usr/bin/env bash
# manage_repos.sh — ai_workspace submodule 管理脚本
# 用法:
#   ./manage_repos.sh add          <路径|domain/name>  <URL> [描述]  # 添加 research submodule
#   ./manage_repos.sh rm           <路径|domain/name>               # 移除 submodule
#   ./manage_repos.sh new-domain   <domain>  <中文名>               # 新建 research 分区
#   ./manage_repos.sh new-project  <项目名>  [描述]                 # 本地新建 projects/ 仓库
#   ./manage_repos.sh link-project <项目名>  <GitHub URL> [描述]    # 将本地项目注册为 submodule
#   ./manage_repos.sh push-project <项目名>  [commit消息]           # 推送 project 并更新主仓库指针
#   ./manage_repos.sh update       [路径|domain/name]               # 拉取最新 (留空=全部)
#   ./manage_repos.sh list                                          # 列出所有 submodule
#   ./manage_repos.sh push         [commit消息]                     # 提交变更并推送主仓库
#   ./manage_repos.sh status                                        # 查看所有 submodule 状态
#   ./manage_repos.sh readme                                        # 重建所有 README 目录表
#
# 代理配置：tools/config.sh（不进 git，运行 tools/set_proxy.sh 统一更新）
#
# 路径简写示例（research 仓库）：
#   domain/name           → research/domain/repos/name
#   domain/subdir/name    → research/domain/repos/subdir/name
#   embodied_ai/Motus     → research/embodied_ai/repos/Motus

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
README_UPDATER="${REPO_ROOT}/tools/update_readme.py"

# 读本地配置（tools/config.sh，不进 git；留空 PROXY 则不挂代理）
PROXY=""
CONFIG_FILE="${REPO_ROOT}/tools/config.sh"
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE"

# 按需挂代理
if [[ -n "${PROXY:-}" ]]; then
    export https_proxy="$PROXY" http_proxy="$PROXY"
fi

cd "$REPO_ROOT"

# ── 路径解析：支持简写 domain/name → research/domain/repos/name ──
resolve_path() {
    local input="$1"
    # 已经是完整路径
    if [[ "$input" == research/* || "$input" == projects/* ]]; then
        echo "$input"; return
    fi
    local domain="${input%%/*}"
    local rest="${input#*/}"
    # 检查是否是已知 domain（用 grep 快速匹配 JSON key）
    if grep -q "\"${domain}\"" tools/domains.json 2>/dev/null; then
        echo "research/${domain}/repos/${rest}"
    else
        echo "$input"  # 未知，原样返回让 git 报错
    fi
}

# README 更新辅助（静默失败：python3 不存在时不阻断流程）
_update_readme() {
    if command -v python3 &>/dev/null && [[ -f "$README_UPDATER" ]]; then
        python3 "$README_UPDATER" "$@" || echo "⚠️  README 更新失败（不影响 git 操作）"
    fi
}

usage() {
    sed -n '/^# 用法/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 1
}

# ─────────────────────────────────────────────
# add <path|domain/name> <url> [描述]
# ─────────────────────────────────────────────
cmd_add() {
    local raw="$1" url="$2" desc="${3:-}"
    if [[ -z "$raw" || -z "$url" ]]; then
        echo "用法: $0 add <路径或domain/name> <GitHub URL> [描述]" >&2; exit 1
    fi
    local path
    path="$(resolve_path "$raw")"
    if [[ "$path" != "$raw" ]]; then
        echo "ℹ️  路径展开: $raw → $path"
    fi
    if git config --file .gitmodules --get "submodule.${path}.url" &>/dev/null; then
        echo "❌ submodule '${path}' 已存在，如需更换 URL 请先 rm 再 add" >&2; exit 1
    fi
    echo "➕ 添加 submodule: $path  ←  $url"
    git submodule add "$url" "$path"
    _update_readme add "$path" "$url" "$desc"
    echo "✅ 完成。记得运行 '$0 push' 将变更同步到远端。"
}

# ─────────────────────────────────────────────
# rm <path|domain/name>
# ─────────────────────────────────────────────
cmd_rm() {
    local raw="$1"
    if [[ -z "$raw" ]]; then
        echo "用法: $0 rm <路径或domain/name>" >&2; exit 1
    fi
    local path
    path="$(resolve_path "$raw")"
    if [[ "$path" != "$raw" ]]; then
        echo "ℹ️  路径展开: $raw → $path"
    fi
    if ! git config --file .gitmodules --get "submodule.${path}.url" &>/dev/null; then
        echo "❌ 未找到 submodule '$path'" >&2; exit 1
    fi
    echo "🗑  移除 submodule: $path"

    # 1. 从 .gitmodules 移除
    git submodule deinit -f "$path"
    # 2. 从 index 移除
    git rm -f "$path"
    # 3. 清理 .git/modules 缓存
    rm -rf ".git/modules/$path"

    _update_readme rm "$path"
    echo "✅ submodule '$path' 已移除。记得运行 '$0 push' 提交变更。"
}

# ─────────────────────────────────────────────
# new-project <name> [描述]
# ─────────────────────────────────────────────
cmd_new_project() {
    local name="${1:-}" desc="${2:-}"
    if [[ -z "$name" ]]; then
        echo "用法: $0 new-project <项目名> [描述]" >&2; exit 1
    fi
    if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "❌ 项目名只能含字母、数字、- 、_" >&2; exit 1
    fi
    local path="projects/$name"
    if [[ -e "$path" ]]; then
        echo "❌ $path 已存在" >&2; exit 1
    fi
    echo "📁 创建本地项目: $path"
    mkdir -p "$path"
    cd "$path"
    git init
    git checkout -b main 2>/dev/null || git checkout -b main 2>/dev/null || true
    # 初始文件
    { echo "# $name"; [[ -n "$desc" ]] && printf '\n%s\n' "$desc"; } > README.md
    cat > .gitignore << 'GIEOF'
__pycache__/
*.py[cod]
.env
.venv/
dist/
build/
*.log
GIEOF
    git add .
    git commit -m "init: $name"
    cd "$REPO_ROOT"
    echo "✅ projects/$name 已创建（纯本地 git 仓库，尚未注册为 submodule）"
    echo ""
    echo "   开发完成后，在 GitHub 创建同名仓库，再运行："
    echo "   ./manage_repos.sh link-project $name https://github.com/Bismile/$name.git [描述]"
}

# ─────────────────────────────────────────────
# link-project <name> <GitHub URL> [描述]
# ─────────────────────────────────────────────
cmd_link_project() {
    local name="${1:-}" url="${2:-}" desc="${3:-}"
    if [[ -z "$name" || -z "$url" ]]; then
        echo "用法: $0 link-project <项目名> <GitHub URL> [描述]" >&2; exit 1
    fi
    local path="projects/$name"
    if [[ ! -d "$path/.git" && ! -f "$path/.git" ]]; then
        echo "❌ $path 不存在或不是 git 仓库（先用 new-project 创建或手动 git init）" >&2; exit 1
    fi
    if git config --file .gitmodules --get "submodule.${path}.url" &>/dev/null; then
        echo "❌ $path 已经是 submodule，无需重复注册" >&2; exit 1
    fi
    echo "🔗 推送 $path 到 $url ..."
    # 1. 推送本地提交到 GitHub
    (
        cd "$path"
        if git remote get-url origin &>/dev/null 2>&1; then
            git remote set-url origin "$url"
        else
            git remote add origin "$url"
        fi
        git push -u origin HEAD
    )
    # 2. 注册为 submodule（--force 允许在已有目录上操作）
    echo "📌 注册为 submodule ..."
    git submodule add --force "$url" "$path"
    # 3. 更新 README
    _update_readme add "$path" "$url" "$desc"
    echo "✅ projects/$name 已注册为 submodule 并推送到 $url"
    echo "   记得运行 '$0 push' 提交主仓库的变更。"
}

# ─────────────────────────────────────────────
# update [path|domain/name]
# ─────────────────────────────────────────────
cmd_update() {
    local raw="${1:-}"
    local path=""
    if [[ -n "$raw" ]]; then
        path="$(resolve_path "$raw")"
        [[ "$path" != "$raw" ]] && echo "ℹ️  路径展开: $raw → $path"
    fi
    if [[ -n "$path" ]]; then
        echo "🔄 更新 submodule: $path"
        git submodule update --remote --merge -- "$path"
    else
        echo "🔄 更新所有 submodule（并行，最多 8 个）..."
        git submodule update --remote --merge --jobs 8
    fi
    echo "✅ 更新完成。记得运行 '$0 push' 提交新的 commit 指针。"
}

# ─────────────────────────────────────────────
# list
# ─────────────────────────────────────────────
cmd_list() {
    echo "📋 当前 submodule 列表："
    echo "────────────────────────────────────────────────────────────────"
    printf "%-50s  %s\n" "本地路径" "远端 URL"
    echo "────────────────────────────────────────────────────────────────"
    git config --file .gitmodules --get-regexp '^submodule\..*\.path$' | \
    while read -r key path; do
        local name="${key#submodule.}"; name="${name%.path}"
        local url
        url=$(git config --file .gitmodules --get "submodule.${name}.url")
        printf "%-50s  %s\n" "$path" "$url"
    done
    echo "────────────────────────────────────────────────────────────────"
    local count
    count=$(git config --file .gitmodules --get-regexp '^submodule\..*\.path$' | wc -l)
    echo "共 $count 个 submodule"
}

# ─────────────────────────────────────────────
# push [message]
# ─────────────────────────────────────────────
cmd_push() {
    local msg="${1:-chore: update submodules}"
    echo "📦 暂存变更..."
    git add -A

    if git diff --cached --quiet; then
        echo "ℹ️  没有待提交的变更"
    else
        git commit -m "$msg"
        echo "✅ 已提交: $msg"
    fi

    echo "🚀 推送到远端..."
    git push origin HEAD
    echo "✅ 推送完成"
}

# ─────────────────────────────────────────────
# status
# ─────────────────────────────────────────────
cmd_status() {
    echo "📊 submodule 状态："
    git submodule status
}

# ─────────────────────────────────────────────
# new-domain <domain> <中文名>
# ─────────────────────────────────────────────
cmd_new_domain() {
    local domain="${1:-}" zh="${2:-}"
    if [[ -z "$domain" || -z "$zh" ]]; then
        echo "用法: $0 new-domain <domain名> <中文名>" >&2; exit 1
    fi
    if [[ ! "$domain" =~ ^[a-z0-9_]+$ ]]; then
        echo "❌ domain 名只能含小写字母、数字、下划线" >&2; exit 1
    fi
    echo "🗂  创建新分区: $domain（$zh）"
    _update_readme new-domain "$domain" "$zh"
    echo "✅ 完成。接下来用 '$0 add research/${domain}/repos/<仓库> <URL> 描述' 添加仓库。"
    echo "   记得运行 '$0 push' 提交变更。"
}

# ─────────────────────────────────────────────
# push-project <name> [message]
# ─────────────────────────────────────────────
cmd_push_project() {
    local name="${1:-}" msg="${2:-}"
    if [[ -z "$name" ]]; then
        echo "用法: $0 push-project <项目名> [commit消息]" >&2; exit 1
    fi
    local path="projects/$name"
    if [[ ! -d "$path/.git" && ! -f "$path/.git" ]]; then
        echo "❌ $path 不存在或不是 git 仓库" >&2; exit 1
    fi
    cd "$path"
    [[ -z "$msg" ]] && msg="chore: update $name"
    echo "📦 $name: 暂存变更..."
    git add -A
    if git diff --cached --quiet; then
        echo "ℹ️  没有待提交的变更"
    else
        git commit -m "$msg"
        echo "✅ 已提交: $msg"
    fi
    echo "🚀 推送 $name 到远端..."
    git push origin HEAD
    echo "✅ $name 推送完成"
    cd "$REPO_ROOT"
    # 更新主仓库的 submodule 指针
    git add "$path"
    if ! git diff --cached --quiet; then
        git commit -m "chore: update $name submodule pointer"
        echo "✅ 主仓库 submodule 指针已更新"
    fi
}

# ─────────────────────────────────────────────
# readme  — 重建所有 README 目录表
# ─────────────────────────────────────────────
cmd_readme() {
    echo "📝 重建所有 README 目录表..."
    _update_readme rebuild
}

# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "$CMD" in
    add)           cmd_add          "$@" ;;
    rm)            cmd_rm           "$@" ;;
    new-domain)    cmd_new_domain   "$@" ;;
    new-project)   cmd_new_project  "$@" ;;
    link-project)  cmd_link_project "$@" ;;
    update)        cmd_update       "$@" ;;
    list)          cmd_list              ;;
    push)          cmd_push         "$@" ;;
    push-project)  cmd_push_project "$@" ;;
    status)        cmd_status            ;;
    readme)        cmd_readme            ;;
    *)             usage ;;
esac
