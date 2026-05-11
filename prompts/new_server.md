# 新服务器初始化 Prompt

把这个文件内容贴给 agent，agent 会自动完成初始化。流程分两阶段：
1. agent 先跑 Step 1 检查，把结果（df -h、GPU、网络）**展示给你**，然后停下来
2. **你填写下方路径表格**，agent 再继续 Step 2 以后

---

## 路径配置（Step 1 结果出来后填写，agent 等待）

> Agent：运行完 Step 1 后，把结果展示给用户，**停下来等用户填写这张表格**，不要继续执行后面的步骤。

| 配置项 | 值（用户填写） |
|--------|--------------|
| username | `flg` |
| 代理地址 | `http://127.0.0.1:20809`（没有则留空） |
| ai_workspace 所在磁盘 | 例：`/data` |
| 大文件磁盘（数据集/权重/缓存） | 例：`/data1` |
| ai_workspace 仓库地址 | `https://github.com/<user>/ai_workspace` |

填完后，告诉 agent 继续，它会根据表格自动推导所有路径：
- workspace → `<ai_workspace磁盘>/<username>/ai_workspace/`
- 数据集/权重/缓存 → `<大文件磁盘>/<username>/datasets/`、`ckpt/`、`cache/`

---

## 指令

我刚拿到一台新服务器，请按步骤帮我完成初始化。每步完成后告诉我结果再继续下一步。

---

### Step 1 — 检查存储、硬件、网络（完成后停下来等我填表格）

```bash
# 存储分布（决定路径怎么分配）
df -h | grep -E "Filesystem|/dev"

# 当前用户
whoami && echo $HOME

# 硬件
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvcc --version 2>/dev/null | grep release || echo "nvcc not found"

# 网络
curl -s --max-time 5 https://github.com && echo "GitHub: OK" || echo "GitHub: FAIL"
curl -s --max-time 5 https://huggingface.co && echo "HuggingFace: OK" || echo "HuggingFace: FAIL"
```

**根据网络结果确定后续是否需要代理：**
- GitHub OK → 后续 git clone 不加代理
- GitHub FAIL → 后续加 `ALL_PROXY=<代理地址>`

**等用户填好上方表格后再继续。**

---

### Step 2 — 创建目录结构

> Agent：用表格中的路径替换下方占位符

```bash
# 替换 <ws> = ai_workspace磁盘/username，<data> = 大文件磁盘/username
mkdir -p <ws>/ai_workspace/projects
mkdir -p <data>/datasets
mkdir -p <data>/ckpt
mkdir -p <data>/cache/huggingface
mkdir -p <data>/cache/uv
```

---

### Step 3 — 配置 bashrc

```bash
cat >> ~/.bashrc << 'EOF'

export PIP_CACHE_DIR=<data>/cache/pip
export TMPDIR=<data>/temp
export PATH="$HOME/.local/bin:$PATH"
# 代理按需手动 export，不写入 bashrc：
# export ALL_PROXY=http://127.0.0.1:20809
# HF_ENDPOINT=https://hf-mirror.com（与代理不可共用）
EOF
mkdir -p <data>/temp
source ~/.bashrc
```

---

### Step 4 — 安装 uv

```bash
# 检查是否已安装
uv --version 2>/dev/null && echo "已安装" || \
    curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

---

### Step 5 — 克隆 ai_workspace

```bash
# 根据 Step 1 网络结果决定是否加 ALL_PROXY
git clone <仓库地址> <ws>/ai_workspace
cd <ws>/ai_workspace
git submodule update --init --recursive
```

---

### Step 6 — 配置 VS Code（让 Copilot 新建终端自动激活项目 .venv）

在 `/home/<username>/.vscode-server/data/User/settings.json` 写入（文件不存在则新建）：

```json
{
    "python.terminal.activateEnvironment": true
}
```

---

### Step 7 — 更新 prefix.md

修改 `prompts/prefix.md` 中"当前服务器环境"表格为本机实际路径，重点字段：

| 字段 | 本机值 |
|------|--------|
| 工作区 | `<ws>/ai_workspace/` |
| 数据集 | `<data>/datasets/` |
| 模型权重 (ckpt_root) | `<data>/ckpt/` |
| HuggingFace 缓存 | `<data>/cache/huggingface/` |
| 实验输出 (exp_root) | 各项目目录下的 `exp/`，在项目 `config/local.yaml` 配置 |
| 代理 | 按需填，不写入 bashrc |

---

## 已知服务器记录

### 主力机（2026-05）

| 配置项 | 值 |
|--------|-----|
| 系统 | Ubuntu 22.04 |
| CUDA | 12.4 |
| GPU | 8× RTX 4090 (48G) |
| 工作区 | `/data/flg/ai_workspace/` |
| 数据集 | `/data1/flg/datasets/` |
| 模型权重 (ckpt_root) | `/data1/flg/ckpt/` |
| HF 缓存 | `/data1/flg/cache/huggingface/` |
| uv 缓存 | `/data1/flg/cache/uv/` |
| 代理 | `http://127.0.0.1:20809` |

---

## 常用命令速查

### uv

```bash
# 有 pyproject.toml 的项目（推荐）
cd projects/my_exp
UV_CACHE_DIR=<data>/cache/uv uv sync   # 创建 .venv 并安装所有依赖
uv run python train.py                  # 直接运行，无需 activate

# 没有 pyproject.toml 的脚本
uv venv .venv && source .venv/bin/activate
uv pip install torch torchvision
```

### 模型下载

```bash
# HuggingFace 镜像（不挂代理）
HF_ENDPOINT=https://hf-mirror.com hf download <org>/<model> \
    --local-dir <data>/ckpt/<org>/<model>

# 挂代理直连（与镜像二选一）
ALL_PROXY=http://127.0.0.1:20809 hf download <org>/<model> \
    --local-dir <data>/ckpt/<org>/<model>

# git clone
ALL_PROXY=http://127.0.0.1:20809 git clone https://github.com/...
```
