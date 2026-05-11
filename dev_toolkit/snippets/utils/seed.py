import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """设置全局随机种子，保证实验可复现。

    Args:
        seed: 随机种子值
        deterministic: 是否启用 CUDA 确定性模式（会降低速度）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        # benchmark=True 在固定输入尺寸时能加速
        torch.backends.cudnn.benchmark = True
