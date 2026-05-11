"""
基础 Dataset 骨架。

用法：
    继承 BaseDataset，实现 __len__ 和 __getitem__。
    标准 ImageNet 归一化和常用 augmentation 已内置。
"""

from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet 归一化参数（通用）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_strong_augment(size: int = 224) -> transforms.Compose:
    """RandAugment 风格的强增强，用于自监督预训练。"""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=size // 10 * 2 + 1)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BaseDataset(Dataset):
    """可继承的基础 Dataset，统一处理 transform 和路径逻辑。"""

    def __init__(self, root: str, split: str = "train", transform: Optional[Callable] = None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        """返回样本列表，格式自定义（如 (path, label) 对）。"""
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
