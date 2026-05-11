# Agent 前置 Prompt

你正在协助一位 ML 研究员开发项目。

## Workspace 结构

```
/data/flg/ai_workspace/
├── manage_repos.sh  # 仓库管理脚本（add/rm/new-project/push 等）
├── tools/           # 自动化工具（update_readme.py、sync_public.py 等）
├── prompts/         # Agent 前置 Prompt（本文件所在位置）
├── dev_toolkit/     # 可复用代码工具笱（snippets、scripts、templates）
├── research/        # 第三方领域仓库索引（git submodule）
└── projects/        # 自开发项目（每个独立 git repo，以 submodule 形式注册）
```

自开发项目统一放 `projects/`，用 `manage_repos.sh new-project` 初始化并注册，用 `dev_toolkit/scripts/new_project.sh` 复制 snippets 脚手架。

## 当前服务器环境

| 配置项 | 值 |
|--------|----|
| 工作区 | `/data/flg/ai_workspace/` |
| 自开发项目 | `/data/flg/ai_workspace/projects/` |
| 数据集 | `/data1/flg/datasets/` |
| 模型/权重缓存 | `/data1/flg/cache/` |
| 模型权重（ckpt_root） | `/data1/flg/ckpt/`，按 `<org>/<model>` 组织 |
| HuggingFace 缓存 | `/data1/flg/cache/huggingface/` |
| 实验输出（exp_root） | 各项目目录下的 `exp/`，通过项目 `config/local.yaml` 配置 |
| 代理 | `http://127.0.0.1:20809`（按需手动 export，不写入 bashrc） |
| 包管理 | `uv`（有 pyproject.toml 的项目用 `uv sync`；单包用 `uv pip install`） |

**网络操作按需挂代理：(注意进行huggingface下载操作时,本地节点和hfmirror不可共用,通常使用其一即可)**
```bash
export ALL_PROXY=http://127.0.0.1:20809
ALL_PROXY=http://127.0.0.1:20809 git clone https://...
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download ...   # HF 镜像备选
```

**写代码时的路径约定：**
- 数据集用 `/data1/flg/datasets/<dataset_name>/`，不要硬编码其他路径
- 模型权重从 `/data1/flg/ckpt/<org>/<model>/` 加载
- HuggingFace 模型用 `cache_dir="/data1/flg/cache/huggingface/"` 参数
- 实验输出（checkpoint、log、eval 结果）写到项目 `exp/` 下，通过 `config/local.yaml` 的 `exp_root` 变量配置，不硬编码绝对路径

## 工作原则

- 优先给出可直接运行的代码，不写伪代码或省略关键细节
- 有不确定的地方直接说，不猜
- 代码用 PyTorch，变量命名清晰，只注释关键逻辑
- 对于复杂修改，说明改了什么、为什么

## 可复用工具库

路径：`/data/flg/ai_workspace/dev_toolkit/`

### snippets/ — 直接 copy 进项目用

| 文件 | 功能 |
|------|------|
| `snippets/utils/seed.py` | 全局随机种子，支持确定性模式 |
| `snippets/utils/checkpoint.py` | checkpoint 保存/恢复/自动查找最新 |
| `snippets/utils/logger.py` | stdout+文件双路 logger，AverageMeter |
| `snippets/utils/ddp.py` | DDP 初始化、rank/world_size 获取、cleanup |
| `snippets/utils/config.py` | dataclass config + argparse 自动绑定 |
| `snippets/utils/wandb_logger.py` | Wandb 封装，rank 0 写入，支持图像/artifact |
| `snippets/training/train_loop.py` | BaseTrainer 骨架（AMP/DDP/自动恢复） |
| `snippets/training/lr_schedule.py` | warmup + cosine/linear decay scheduler |
| `snippets/training/ema.py` | EMA 权重维护（teacher 模型常用） |
| `snippets/training/amp.py` | 混合精度封装（fp16/bf16/fp32 统一接口，含自动检测） |
| `snippets/training/fsdp.py` | FSDP 包装、混合精度配置、checkpoint 保存/恢复 |
| `snippets/training/deepspeed_utils.py` | DeepSpeed ZeRO-1/2/3 配置生成 + 初始化 + checkpoint |
| `snippets/training/memory.py` | gradient checkpointing 封装、显存追踪、峰值分析 |
| `snippets/data/dataloader_base.py` | BaseDataset + 标准 ImageNet transform |
| `snippets/data/video_dataset.py` | 视频帧读取 Dataset（cv2/decord 两路） |
| `snippets/eval/fid.py` | 用 clean-fid 计算 FID，支持生成图批量保存 |
| `snippets/eval/linear_probe.py` | 冻结 backbone + linear head 评估表征质量 |

### scripts/ — shell 工具

| 文件 | 功能 |
|------|------|
| `scripts/new_project.sh <名> [模板]` | 从模板初始化新项目，自动复制 snippets |
| `scripts/gpu_watch.sh` | 实时监控 GPU 显存/利用率 |
| `scripts/kill_port.sh <端口>` | 释放占用端口的进程 |
| `scripts/sync_results.sh` | 把 results/ 下的图表同步到指定目录 |

### project_templates/ — 新项目起点

| 模板 | 适用场景 |
|------|---------|
| `project_templates/base/` | 通用分类/回归任务 |

---

## 任务工作流

**每次开始一个新任务时，先在项目根目录创建 `PLAN.md`，然后再动手写代码。**

文件格式：

```markdown
# 任务：<一句话描述>

## 目标
- 要做什么，完成标准是什么

## 方案
- 选择了哪种实现路径，为什么

## 步骤
- [ ] 步骤 1
- [ ] 步骤 2
- ...

## 待确认
- 有哪些决策需要用户确认（列出来，不要擅自决定）

## 备注
- 遇到的问题、临时记录、下次继续的上下文
```

规则：
- 写完 `PLAN.md` 后，把"待确认"里的问题发给用户，等确认后再继续
- 每完成一个步骤，立即在 `PLAN.md` 里勾掉对应条目
- 如果中途方案变了，更新"方案"一节，不要静默改变
- 用户可能会直接编辑 `PLAN.md` 来修改计划，每次行动前先重新读一遍

---

当你需要某个功能时，**先检查上面的列表**，如果有对应 snippet，直接读文件内容后复用，不要从头重写。

---

## 参考仓库（用户填写后生效）

> 将本次需要参考的仓库路径和用途填入下表，agent 将主动阅读对应代码作为实现参考。
> 路径相对于 `/data/flg/ai_workspace/`，例如 `research/generative_vision/repos/DiT/`

| 仓库路径 | 参考目的 |
|---------|---------|
| （填写） | （填写） |


<details>
<summary>research/ 仓库速查（点击展开）</summary>

**generative_vision/repos/**

| 仓库 | 简介 |
|------|------|
| `DiT/` | Diffusion Transformer 原版实现 |
| `LightningDiT/` | 高效 DiT 训练框架（更快收敛） |
| `DDT/` | Decoupled Diffusion Transformer |
| `JiT/` | 基于 flow matching 的图像生成 |
| `REPA/` | representation alignment 加速扩散模型训练 |
| `RAE/` | Residual Autoencoder |
| `Internal-Guidance/` | 扩散模型内部特征引导 |
| `drifting/` | 扩散模型训练探索 |
| `mae/` | Masked Autoencoder (MAE) |
| `dino/dino/` | DINO v1 self-supervised ViT |
| `dino/dinov2/` | DINOv2 self-supervised ViT |
| `denoising-diffusion-pytorch/` | DDPM PyTorch 简洁实现 |
| `ml-tarflow/` | normalizing flow 实现 |

**embodied_ai/repos/**

| 仓库 | 简介 |
|------|------|
| `UniVLA/` | 视觉-语言-动作统一模型 |
| `cosmos-policy/` | NVIDIA Cosmos 策略模型 |
| `DreamDojo/` | 世界模型驱动的机器人学习 |
| `dreamzero/` | 零样本世界模型规划 |
| `Genie-Envisioner/` | 交互式世界模型 |
| `Ctrl-World/` | 可控世界模型 |
| `video-prediction-policy/` | 视频预测作为机器人策略 |
| `mimic-video/` | 视频模仿学习 |
| `Motus/` | 机器人运动生成 |
| `FastWAM/` | 快速 Waypoint-based Action Model |
| `LARYBench/` | 长视野机器人任务 benchmark |
| `robocasa/` | 机器人操作数据集 / 仿真环境 |
| `lingbot-va/` | 语言引导的机器人视觉-动作模型 |

**video_generation/repos/**

| 仓库 | 简介 |
|------|------|
| `HunyuanVideo/` | 腾讯混元视频生成模型 |
| `Wan2.2/` | 阿里通义万象视频生成 2.2 |

**multimodal_vlm/repos/**

| 仓库 | 简介 |
|------|------|
| `Qwen3-VL/` | Qwen3 多模态视觉语言模型 |
| `tuna-2/` | Meta TUNA-2 多模态模型 |

</details>

---

## 项目操作入口

### 场景 A：新建项目

```bash
cd /data/flg/ai_workspace

# 1. 创建项目（本地 git init，注册进 workspace）
./manage_repos.sh new-project <项目名> [描述]

# 2. 可选：用 dev_toolkit 模板复制 snippets 脚手架
bash dev_toolkit/scripts/new_project.sh <项目名>

# 3. 进入项目，配置虚拟环境
cd projects/<项目名>
uv venv .venv && source .venv/bin/activate
uv pip install torch torchvision

# 4. 开发完成后推送到 GitHub（先在 GitHub 建好空仓库）
cd /data/flg/ai_workspace
./manage_repos.sh link-project <项目名> https://github.com/<YourOrg>/<项目名>.git [描述]
```

Agent 应在项目根目录创建 `PLAN.md`，再动手写代码。

### 场景 B：在已有仓库（外部或 fork）上开发

```bash
# 方式一（推荐）：用 manage_repos.sh 注册并 clone
cd /data/flg/ai_workspace
./manage_repos.sh add projects/<项目名> https://github.com/... [描述]

# 方式二：已手动 git clone 的，补注册为 submodule
cd /data/flg/ai_workspace
git submodule add --force <URL> projects/<项目名>
./manage_repos.sh readme && ./manage_repos.sh push "add: <项目名>"

# 建议在项目根目录新建 PLAN.md 记录修改意图和进度
```

Agent 应先阅读项目结构和关键文件，理解原有逻辑后再修改，并在 `PLAN.md` 里记录改了什么。
