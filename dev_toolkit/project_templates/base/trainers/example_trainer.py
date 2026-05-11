"""
具体 Trainer 示例。继承 BaseTrainer，实现业务逻辑。
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 把 snippets/ 里的 BaseTrainer 复制到本项目的 trainers/ 中
# from trainers.base_trainer import BaseTrainer
# from data.dataset import MyDataset


class MyTrainer:  # 继承 BaseTrainer
    """示例：简单分类任务 Trainer。"""

    def build_model(self) -> nn.Module:
        import torchvision.models as models
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(2048, self.cfg.num_classes)
        return model

    def build_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def build_scheduler(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(self.optimizer, T_max=self.cfg.num_epochs)

    def build_loaders(self):
        # train_ds = MyDataset(self.cfg.data_root, split="train", transform=get_train_transform())
        # val_ds   = MyDataset(self.cfg.data_root, split="val",   transform=get_val_transform())
        # return build_dataloader(train_ds, self.cfg.batch_size), \
        #        build_dataloader(val_ds,   self.cfg.batch_size, shuffle=False)
        raise NotImplementedError

    def compute_loss(self, batch):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self.model(images)
        loss = nn.functional.cross_entropy(logits, labels)

        acc = (logits.argmax(1) == labels).float().mean()
        return loss, {"acc": acc.item()}

    def evaluate(self, val_loader: DataLoader) -> dict:
        correct = total = 0
        for images, labels in val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return {"main_metric": 1 - correct / total, "acc": correct / total}
