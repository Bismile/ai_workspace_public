"""
显存优化工具：gradient checkpointing、activation offload、显存分析。

核心思路：
    - gradient checkpointing：用重计算换显存，前向不保存中间激活，
      反向时在该层重新跑一次前向。大约省 ~60% 激活显存，代价是 ~30% 速度。
    - activation offload：把激活卸载到 CPU，极端省显存，速度更慢（一般不用）。
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


# ---- Gradient Checkpointing ----

def enable_gradient_checkpointing(model: nn.Module, granularity: str = "block"):
    """
    对常见结构自动开启 gradient checkpointing。

    Args:
        granularity:
            "block"   — 对每个 Transformer block 单独 checkpoint（最常用）
            "full"    — 整个模型做一次 checkpoint（省显存最多，但限制最大）
            "manual"  — 不自动开，手动在 forward 里用 checkpoint() 包
    """
    if granularity == "full":
        model.gradient_checkpointing_enable()   # HuggingFace 模型有此方法
        return

    # 对 HuggingFace / 自定义模型的通用方式
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    else:
        # 遍历顶层子模块，给每个加上 checkpoint wrapper
        for name, module in model.named_children():
            if _is_transformer_block(module):
                _wrap_checkpoint(model, name, module)


def _is_transformer_block(module: nn.Module) -> bool:
    """粗略判断是否为 Transformer block（含 attention + ffn）。"""
    has_attn = any("attn" in n.lower() or "attention" in n.lower()
                   for n, _ in module.named_modules())
    has_ffn  = any("mlp" in n.lower() or "ffn" in n.lower() or "feed" in n.lower()
                   for n, _ in module.named_modules())
    return has_attn and has_ffn


def _wrap_checkpoint(parent: nn.Module, child_name: str, child: nn.Module):
    """把 child 替换为 CheckpointWrapper。"""
    wrapped = CheckpointWrapper(child)
    setattr(parent, child_name, wrapped)


class CheckpointWrapper(nn.Module):
    """把任意模块包装为 gradient checkpoint 版本。"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        # checkpoint 不支持 keyword args，需要包一层
        def _fwd(*a):
            return self.module(*a, **kwargs)
        return checkpoint(_fwd, *args, use_reentrant=False)


# ---- checkpoint_sequential 用法（顺序模型）----

def sequential_checkpoint(layers: nn.Sequential, x: torch.Tensor, segments: int = 4):
    """
    把顺序模型切成 segments 段，每段做一次 gradient checkpoint。
    segments 越多，省显存越多，速度越慢。

    适合 ResNet / 纯顺序结构。
    """
    return checkpoint_sequential(layers, segments, x, use_reentrant=False)


# ---- 显存分析工具 ----

def print_gpu_memory(prefix: str = ""):
    """打印当前 GPU 显存占用（所有卡）。"""
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"{prefix}[GPU {i}] allocated={alloc:.2f}GB  reserved={reserved:.2f}GB")


def memory_summary(device: int = 0) -> str:
    return torch.cuda.memory_summary(device=device, abbreviated=True)


class MemoryTracker:
    """在训练循环中追踪峰值显存。

    Usage:
        tracker = MemoryTracker()
        tracker.reset()
        with tracker.track("forward"):
            loss = model(batch)
        tracker.report()
    """

    def __init__(self, device: int = 0):
        self.device = device
        self.records: dict[str, float] = {}

    def reset(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        self.records = {}

    class _ctx:
        def __init__(self, tracker, label):
            self.tracker = tracker
            self.label = label

        def __enter__(self):
            torch.cuda.reset_peak_memory_stats(self.tracker.device)

        def __exit__(self, *_):
            peak = torch.cuda.max_memory_allocated(self.tracker.device) / 1e9
            self.tracker.records[self.label] = peak

    def track(self, label: str):
        return self._ctx(self, label)

    def report(self):
        for label, gb in self.records.items():
            print(f"  [{label}] peak GPU mem = {gb:.3f} GB")
