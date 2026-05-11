"""线性探测（Linear Probe）评估表征质量。

用法：
    acc = linear_probe_eval(
        backbone=model.encoder,       # 冻结的特征提取器
        train_loader=train_loader,    # 标签数据集
        val_loader=val_loader,
        feature_dim=768,
        num_classes=1000,
        epochs=100,
    )
    print(f"Linear probe acc: {acc:.4f}")
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader, device: str = "cuda"):
    """一次性提取所有特征，适合特征维度不太大的情况。"""
    backbone.eval()
    feats, labels = [], []
    for batch in loader:
        imgs, lbls = batch[0].to(device), batch[1]
        f = backbone(imgs)
        # 支持 (B, D) 或 (B, N, D) 取 CLS token
        if f.dim() == 3:
            f = f[:, 0]
        feats.append(f.cpu())
        labels.append(lbls)
    return torch.cat(feats), torch.cat(labels)


def linear_probe_eval(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    feature_dim: int,
    num_classes: int,
    epochs: int = 100,
    lr: float = 0.01,
    device: str = "cuda",
) -> float:
    """
    Returns:
        top-1 accuracy on val_loader
    """
    print("Extracting train features...")
    train_feats, train_labels = extract_features(backbone, train_loader, device)
    print("Extracting val features...")
    val_feats, val_labels = extract_features(backbone, val_loader, device)

    # 归一化特征（对 linear probe 很重要）
    mean = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0, keepdim=True).clamp(min=1e-6)
    train_feats = (train_feats - mean) / std
    val_feats = (val_feats - mean) / std

    head = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 用内存中的特征构造 DataLoader
    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(train_feats, train_labels)
    t_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        head.train()
        for x, y in t_loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(head(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    head.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(val_feats, val_labels), batch_size=512):
            x, y = x.to(device), y.to(device)
            correct += (head(x).argmax(1) == y).sum().item()
            total += y.size(0)

    return correct / total
