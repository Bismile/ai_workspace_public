"""FID 计算工具，基于 clean-fid。

依赖：pip install clean-fid

用法：
    # 方式一：从两个图像目录直接计算
    score = compute_fid("path/to/real", "path/to/fake")

    # 方式二：批量保存生成图后计算
    saver = ImageSaver("outputs/gen")
    for batch in generate(...):
        saver.save_batch(batch)   # batch: (B,C,H,W) tensor in [-1,1] 或 [0,1]
    score = saver.compute_fid("path/to/real")
"""

import os
from pathlib import Path

import torch
from torchvision.utils import save_image


def compute_fid(real_dir: str, fake_dir: str, device: str = "cuda") -> float:
    """用 clean-fid 计算 FID。"""
    from cleanfid import fid
    score = fid.compute_fid(real_dir, fake_dir, device=device)
    return score


def compute_fid_from_stats(real_dir: str, fake_dir: str, dataset_name: str = None) -> float:
    """如果预先计算了统计数据，用名字加速。"""
    from cleanfid import fid
    if dataset_name:
        return fid.compute_fid(fake_dir, dataset_name=dataset_name)
    return fid.compute_fid(real_dir, fake_dir)


class ImageSaver:
    """把生成图批量保存为 PNG，然后计算 FID。"""

    def __init__(self, save_dir: str, exist_ok: bool = True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.count = 0

    def save_batch(self, images: torch.Tensor, normalize: bool = True):
        """
        Args:
            images: (B, C, H, W)，值域 [-1,1]（normalize=True）或 [0,1]（normalize=False）
        """
        for img in images:
            save_image(
                img,
                self.save_dir / f"{self.count:06d}.png",
                normalize=normalize,
                value_range=(-1, 1) if normalize else (0, 1),
            )
            self.count += 1

    def compute_fid(self, real_dir: str, device: str = "cuda") -> float:
        return compute_fid(real_dir, str(self.save_dir), device=device)

    def __len__(self):
        return self.count
