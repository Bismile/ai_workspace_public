"""
混合精度（AMP）训练封装。

三种模式：
    1. torch.amp（原生，推荐）— 支持 fp16 / bf16，自动管理 GradScaler
    2. 纯 bf16（无 scaler）  — A100/H100 推荐，bf16 不会溢出，不需要 loss scaling
    3. 手动 fp16             — 需要 GradScaler，适合旧卡（V100）

用法：见文件末尾示例。
"""

from contextlib import contextmanager
from typing import Literal

import torch
import torch.nn as nn


class AMPTrainer:
    """
    把 AMP 相关逻辑封装成统一接口，切换模式只需改 dtype 参数。

    dtype:
        "fp32"  — 不启用混合精度（baseline/debug 用）
        "fp16"  — torch.amp fp16 + GradScaler（V100 / 兼容性优先）
        "bf16"  — torch.amp bf16，无 GradScaler（A100/H100 推荐）
    """

    def __init__(self, dtype: Literal["fp32", "fp16", "bf16"] = "bf16"):
        self.dtype = dtype
        self.enabled = dtype != "fp32"
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "fp16"))

    def autocast(self):
        if not self.enabled:
            from contextlib import nullcontext
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def backward(self, loss: torch.Tensor):
        self.scaler.scale(loss).backward()

    def step(self, optimizer, model: nn.Module | None = None, max_grad_norm: float = 1.0):
        """unscale → clip grad → optimizer step → scaler update。"""
        if self.scaler.is_enabled():
            self.scaler.unscale_(optimizer)
        if model is not None and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self):
        return self.scaler.state_dict()

    def load_state_dict(self, state):
        self.scaler.load_state_dict(state)


# ---- 工具函数 ----

def detect_best_dtype() -> str:
    """自动选择当前 GPU 最适合的精度。"""
    if not torch.cuda.is_available():
        return "fp32"
    cap = torch.cuda.get_device_capability()
    # Ampere（sm_80）及以上原生支持 bf16
    if cap[0] >= 8:
        return "bf16"
    # Volta / Turing 支持 fp16
    if cap[0] >= 7:
        return "fp16"
    return "fp32"


def convert_model_dtype(model: nn.Module, dtype: str = "bf16") -> nn.Module:
    """把整个模型参数转换为指定精度（推理时用，训练一般不整体转）。"""
    t = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    return model.to(t)


# ---- 使用示例 ----
"""
amp = AMPTrainer(dtype="bf16")   # 或 detect_best_dtype()

for batch in loader:
    optimizer.zero_grad()

    with amp.autocast():
        loss = model(batch)

    amp.backward(loss)
    amp.step(optimizer, model=model, max_grad_norm=1.0)

# checkpoint 时保存 scaler 状态
torch.save({"scaler": amp.state_dict(), ...}, "ckpt.pt")
amp.load_state_dict(ckpt["scaler"])
"""
