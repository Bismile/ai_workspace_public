#!/bin/bash
# new_project.sh — 从模板快速初始化新 ML 项目
#
# 用法：
#   bash new_project.sh <项目名> [模板名]
#
# 示例：
#   bash new_project.sh my_diffusion diffusion
#   bash new_project.sh my_cls base
#   bash new_project.sh my_exp          # 默认用 base 模板

set -euo pipefail

# ---- 配置 ----
TOOLKIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$TOOLKIT_DIR/.." && pwd)"
PROJECTS_DIR="$WORKSPACE_DIR/projects"
TEMPLATES_DIR="$TOOLKIT_DIR/project_templates"
SNIPPETS_DIR="$TOOLKIT_DIR/snippets"

PROJECT_NAME="${1:-}"
TEMPLATE="${2:-base}"

# ---- 参数检查 ----
if [[ -z "$PROJECT_NAME" ]]; then
    echo "用法: bash new_project.sh <项目名> [模板]"
    echo "可用模板: $(ls "$TEMPLATES_DIR" | tr '\n' ' ')"
    echo "项目将创建在: $PROJECTS_DIR/"
    exit 1
fi

TEMPLATE_PATH="$TEMPLATES_DIR/$TEMPLATE"
if [[ ! -d "$TEMPLATE_PATH" ]]; then
    echo "[ERROR] 模板不存在: $TEMPLATE_PATH"
    echo "可用模板: $(ls "$TEMPLATES_DIR" | tr '\n' ' ')"
    exit 1
fi

mkdir -p "$PROJECTS_DIR"
PROJECT_PATH="$PROJECTS_DIR/$PROJECT_NAME"

if [[ -d "$PROJECT_PATH" ]]; then
    echo "[ERROR] 目录已存在: $PROJECT_PATH"
    exit 1
fi

# 后续操作在 PROJECTS_DIR 里进行
PROJECT_NAME="$PROJECT_PATH"

# ---- 复制模板 ----
echo "[INFO] 从模板 '$TEMPLATE' 创建项目 '$PROJECT_NAME' ..."
cp -r "$TEMPLATE_PATH" "$PROJECT_NAME"

# ---- 复制 snippets ----
echo "[INFO] 复制 snippets ..."
mkdir -p "$PROJECT_NAME/utils" "$PROJECT_NAME/data" "$PROJECT_NAME/trainers"
cp "$SNIPPETS_DIR/utils/seed.py"       "$PROJECT_NAME/utils/"
cp "$SNIPPETS_DIR/utils/checkpoint.py" "$PROJECT_NAME/utils/"
cp "$SNIPPETS_DIR/utils/logger.py"     "$PROJECT_NAME/utils/"
cp "$SNIPPETS_DIR/data/dataloader_base.py" "$PROJECT_NAME/data/"
cp "$SNIPPETS_DIR/training/train_loop.py"  "$PROJECT_NAME/trainers/base_trainer.py"

# ---- 创建必要目录 ----
mkdir -p "$PROJECT_NAME/checkpoints" "$PROJECT_NAME/logs" "$PROJECT_NAME/configs"

# ---- 创建 .gitignore ----
cat > "$PROJECT_NAME/.gitignore" << 'EOF'
checkpoints/
logs/
wandb/
__pycache__/
*.pyc
*.egg-info/
.DS_Store
*.pt
*.pth
data/
.venv/
EOF

# ---- 初始化 git ----
echo "[INFO] git init ..."
cd "$PROJECT_NAME"
git init -q
git add .
git commit -q -m "init: from $TEMPLATE template"
cd -

echo ""
echo "✅ 项目创建完成: $PROJECT_NAME/"
echo ""
echo "目录结构："
find "$PROJECT_NAME" -not -path "*/.git/*" -not -name ".git" | sort | sed "s|$PROJECT_NAME/||" | head -30
echo ""
echo "下一步："
echo "  cd $PROJECT_NAME"
echo "  uv venv .venv && source .venv/bin/activate"
echo "  uv pip install -r requirements.txt"
