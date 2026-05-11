import os
import shutil
from pathlib import Path
import torch


def save_checkpoint(
    state: dict,
    save_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
    keep_last_n: int = 3,
):
    """保存训练 checkpoint，可选保留最近 N 个。

    state 推荐包含：
        {
            'epoch': int,
            'step': int,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),  # 可选
            'config': vars(cfg),                  # 可选，保存超参
            'best_metric': float,
        }
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / filename
    torch.save(state, path)

    if is_best:
        best_path = save_dir / "best.pt"
        shutil.copyfile(path, best_path)

    # 清理旧 checkpoint，只保留最近 keep_last_n 个
    if keep_last_n > 0:
        ckpts = sorted(save_dir.glob("checkpoint_step*.pt"))
        for old in ckpts[:-keep_last_n]:
            old.unlink()


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, device="cuda"):
    """加载 checkpoint，返回 epoch 和 step。"""
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    best_metric = ckpt.get("best_metric", None)

    return epoch, step, best_metric


def find_latest_checkpoint(save_dir: str, pattern: str = "checkpoint_step*.pt") -> str | None:
    """找到最新的 checkpoint 文件路径，没有则返回 None。"""
    ckpts = sorted(Path(save_dir).glob(pattern))
    return str(ckpts[-1]) if ckpts else None
