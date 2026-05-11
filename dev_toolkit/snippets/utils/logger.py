import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """获取同时输出到 stdout 和文件的 logger。

    Usage:
        logger = get_logger(__name__, log_file="logs/train.log")
        logger.info("epoch 1 done")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 避免重复添加 handler

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件输出
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """跟踪指标的滑动平均，常用于 loss/acc 统计。

    Usage:
        meter = AverageMeter("loss")
        for batch in loader:
            loss = ...
            meter.update(loss.item(), n=batch_size)
        print(meter)  # loss: 0.3214 (avg)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.val:.4f} (avg {self.avg:.4f})"
