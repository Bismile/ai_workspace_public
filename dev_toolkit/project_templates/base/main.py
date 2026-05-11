"""项目入口。"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path

from trainers.base_trainer import BaseTrainer
from utils.seed import set_seed


@dataclass
class Config:
    # 数据
    data_root: str = "data/"
    num_workers: int = 8

    # 模型（子类覆盖）
    model_name: str = "base"

    # 训练
    num_epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True
    seed: int = 42

    # 调度
    warmup_epochs: int = 5

    # 保存与日志
    save_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    resume: str = "auto"   # "auto" | "" | 具体路径
    log_every: int = 50
    eval_every: int = 1
    save_every: int = 5
    keep_last_n: int = 3


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    cfg = Config()
    for key, val in vars(cfg).items():
        t = type(val) if val is not None else str
        parser.add_argument(f"--{key}", type=t, default=val)
    args = parser.parse_args()
    return Config(**vars(args))


def main():
    cfg = parse_args()

    # 子类示例：from trainers.my_trainer import MyTrainer
    # trainer = MyTrainer(cfg)
    # trainer.train()
    raise NotImplementedError("请在 trainers/ 中实现具体 Trainer，然后在此处调用")


if __name__ == "__main__":
    main()
