"""Warmup + cosine/linear decay learning rate scheduler。

用法：
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=500, total_steps=10000)
    # 每个 step 调用一次
    scheduler.step()
"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def warmup_linear_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def build_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    mode: str = "cosine",
    last_step: int = -1,
) -> LambdaLR:
    """
    Args:
        mode: "cosine" | "linear"
        last_step: 断点续训时传入当前 step（从 0 开始），-1 表示从头训
    """
    fn = warmup_cosine_schedule if mode == "cosine" else warmup_linear_schedule

    def lr_lambda(step):
        return fn(step, warmup_steps, total_steps)

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_step)
