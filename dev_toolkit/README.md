# 开发工具箱

可复用的代码片段与项目模板，供 agent 和开发者快速调用。

**Agent 使用入口：** `../prompts/prefix.md`（workspace 根目录）— 加到对话上下文，agent 即可自动找到并调用所有工具。

---

## 目录结构

```
ai_workspace/
├── prompts/
│   ├── prefix.md              # ← Agent 前置 prompt，包含所有工具索引
│   └── new_server.md          # ← 新服务器初始化流程
├── dev_toolkit/
│   ├── snippets/
│   ├── utils/                 # seed、checkpoint、logger、ddp、config
│   ├── training/              # train_loop（BaseTrainer）、lr_schedule、ema
│   ├── data/                  # dataloader_base、video_dataset
│   └── eval/                  # fid、linear_probe
├── project_templates/
│   └── base/                  # 通用 ML 项目骨架
└── scripts/
    ├── new_project.sh         # 从模板初始化新项目
    ├── gpu_watch.sh           # 实时 GPU 监控
    ├── kill_port.sh           # 释放占用端口
    └── sync_results.sh        # 同步实验结果到目标目录
```

---

## 快速使用

### 新建项目

```bash
bash dev_toolkit/scripts/new_project.sh <项目名> [模板]
```

自动完成：复制模板 → 复制 snippets → 创建目录结构 → `.gitignore` → `git init`。

### 给 Agent 挂载工具库

把 `prompts/prefix.md` 加到对话上下文，agent 即知道所有 snippet 位置，会自动读文件后复用。

---

## 子仓库

| 仓库 | 来源 | 简介 |
|------|------|------|
