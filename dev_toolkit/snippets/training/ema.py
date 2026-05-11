"""EMA（指数移动平均）权重维护，常用于 diffusion teacher / DINO teacher。

用法：
    ema = EMA(model, decay=0.9999)
    # 每个 optimizer step 后调用
    ema.update(model)
    # 评估时切换到 EMA 权重
    with ema.apply(model):
        evaluate(model)
"""

import copy
from contextlib import contextmanager
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        # 同步 buffers（如 BN 的 running_mean）
        for s_buf, buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(buf)

    @contextmanager
    def apply(self, model: nn.Module):
        """临时把模型权重换成 EMA 权重，退出时恢复。"""
        original = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow.state_dict())
        try:
            yield
        finally:
            model.load_state_dict(original)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)
