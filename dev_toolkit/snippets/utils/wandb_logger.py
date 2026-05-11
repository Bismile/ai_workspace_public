"""
Wandb 日志封装，屏蔽 rank 判断和初始化细节。

用法：
    logger = WandbLogger(project="my_exp", name="run_01", config=vars(cfg))
    logger.log({"loss": 0.5, "lr": 1e-4}, step=100)
    logger.log_image("samples", images, step=100)   # images: list of PIL/np/tensor
    logger.finish()
"""

import os
from typing import Any

import torch


class WandbLogger:
    """
    封装 wandb，只有 rank 0 实际写入，其他 rank 是空操作。
    rank 自动从环境变量读取（torchrun 设置），单卡直接用也没问题。
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict | None = None,
        dir: str = "wandb/",
        resume: bool = False,
        run_id: str | None = None,
        enabled: bool = True,
    ):
        self.rank = int(os.environ.get("RANK", 0))
        self.enabled = enabled and (self.rank == 0)
        self._run = None

        if self.enabled:
            import wandb
            self._run = wandb.init(
                project=project,
                name=name,
                config=config,
                dir=dir,
                resume="allow" if resume else None,
                id=run_id,
            )

    def log(self, metrics: dict[str, Any], step: int | None = None):
        if self.enabled and self._run is not None:
            self._run.log(metrics, step=step)

    def log_image(self, key: str, images, step: int | None = None, caption: str | list | None = None):
        """
        images: list of (PIL.Image | np.ndarray | torch.Tensor (C,H,W) in [0,1])
        """
        if not self.enabled:
            return
        import wandb
        import numpy as np

        wb_images = []
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                img = (img * 255).astype("uint8")
            cap = (caption[i] if isinstance(caption, list) else caption) or ""
            wb_images.append(wandb.Image(img, caption=cap))
        self.log({key: wb_images}, step=step)

    def log_histogram(self, key: str, tensor: torch.Tensor, step: int | None = None):
        if not self.enabled:
            return
        import wandb
        self.log({key: wandb.Histogram(tensor.detach().cpu().float().numpy())}, step=step)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model"):
        """上传文件（如 checkpoint）为 wandb artifact。"""
        if not self.enabled:
            return
        import wandb
        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def summary(self, key: str, value: Any):
        """记录到 run summary（最终指标，显示在 wandb 表格里）。"""
        if self.enabled and self._run is not None:
            self._run.summary[key] = value

    def finish(self):
        if self.enabled and self._run is not None:
            self._run.finish()

    @property
    def run_id(self) -> str | None:
        return self._run.id if self._run else None
