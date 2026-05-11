"""
通用训练循环骨架。

用法：
    继承 BaseTrainer，实现 compute_loss() 和可选的 evaluate()。
    不要在这里加业务逻辑，业务逻辑放在子类。
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.logger import get_logger, AverageMeter
from utils.seed import set_seed


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger(__name__, log_file=f"{cfg.log_dir}/train.log")

        set_seed(cfg.seed)

        self.model = self.build_model().to(self.device)
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = GradScaler(enabled=cfg.use_amp)

        self.epoch = 0
        self.step = 0
        self.best_metric = float("inf")

        # 自动恢复
        if cfg.resume == "auto":
            ckpt_path = find_latest_checkpoint(cfg.save_dir)
            if ckpt_path:
                self.epoch, self.step, self.best_metric = load_checkpoint(
                    ckpt_path, self.model, self.optimizer, self.scheduler, self.device
                )
                self.logger.info(f"Resumed from {ckpt_path} (epoch={self.epoch}, step={self.step})")
        elif cfg.resume:
            self.epoch, self.step, self.best_metric = load_checkpoint(
                cfg.resume, self.model, self.optimizer, self.scheduler, self.device
            )

    # ---- 子类必须实现 ----

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def build_optimizer(self):
        raise NotImplementedError

    def build_scheduler(self):
        return None

    def build_loaders(self) -> tuple[DataLoader, Optional[DataLoader]]:
        """返回 (train_loader, val_loader)，val_loader 可为 None。"""
        raise NotImplementedError

    def compute_loss(self, batch) -> tuple[torch.Tensor, dict]:
        """返回 (loss_tensor, log_dict)。log_dict 的值会被打印。"""
        raise NotImplementedError

    # ---- 可选覆盖 ----

    def evaluate(self, val_loader: DataLoader) -> dict:
        """返回 metrics dict，key 'main_metric' 用于判断 is_best。"""
        return {}

    # ---- 训练主循环 ----

    def train(self):
        train_loader, val_loader = self.build_loaders()
        cfg = self.cfg

        for epoch in range(self.epoch, cfg.num_epochs):
            self.epoch = epoch
            self.model.train()
            loss_meter = AverageMeter("loss")
            t0 = time.time()

            for batch in train_loader:
                self.optimizer.zero_grad()

                with autocast(enabled=cfg.use_amp):
                    loss, log_dict = self.compute_loss(batch)

                self.scaler.scale(loss).backward()

                if cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_meter.update(loss.item())
                self.step += 1

                if self.step % cfg.log_every == 0:
                    self.logger.info(
                        f"epoch={epoch} step={self.step} {loss_meter} "
                        + " ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0
            self.logger.info(f"Epoch {epoch} done in {elapsed:.1f}s | {loss_meter}")

            # 验证 & 保存
            if val_loader is not None and (epoch + 1) % cfg.eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    metrics = self.evaluate(val_loader)
                self.logger.info(f"[Eval] epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

                is_best = metrics.get("main_metric", float("inf")) < self.best_metric
                if is_best:
                    self.best_metric = metrics["main_metric"]
            else:
                is_best = False

            if (epoch + 1) % cfg.save_every == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "step": self.step,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                        "best_metric": self.best_metric,
                        "config": vars(cfg),
                    },
                    save_dir=cfg.save_dir,
                    filename=f"checkpoint_step{self.step:07d}.pt",
                    is_best=is_best,
                    keep_last_n=cfg.keep_last_n,
                )
