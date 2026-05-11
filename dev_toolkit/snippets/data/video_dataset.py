"""视频帧读取 Dataset，支持 cv2 和 decord 两路后端。

用法：
    ds = VideoDataset("data/videos", clip_len=16, frame_stride=2, backend="decord")
    frames, label = ds[0]  # frames: (T, C, H, W) float32 in [0,1]
"""

from pathlib import Path
from typing import Callable, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    """
    目录结构期望：
        root/
          class_a/
            video1.mp4
            video2.mp4
          class_b/
            ...

    若无分类结构，label 统一返回 -1。
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 16,
        frame_stride: int = 1,
        size: int = 224,
        transform: Optional[Callable] = None,
        backend: Literal["cv2", "decord"] = "decord",
        extensions: tuple = (".mp4", ".avi", ".mov", ".mkv"),
    ):
        self.root = Path(root)
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.backend = backend
        self.transform = transform or transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self._scan(extensions)

    def _scan(self, extensions):
        classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if classes:
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for cls in classes:
                for ext in extensions:
                    for v in (self.root / cls).glob(f"*{ext}"):
                        self.samples.append((v, self.class_to_idx[cls]))
        else:
            for ext in extensions:
                for v in self.root.glob(f"**/*{ext}"):
                    self.samples.append((v, -1))

    def _load_frames_decord(self, path: Path) -> np.ndarray:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path))
        total = len(vr)
        need = self.clip_len * self.frame_stride
        start = max(0, (total - need) // 2)
        indices = list(range(start, min(total, start + need), self.frame_stride))[:self.clip_len]
        # 不足则循环填充
        while len(indices) < self.clip_len:
            indices += indices
        indices = indices[:self.clip_len]
        frames = vr.get_batch(indices).numpy()  # (T, H, W, C)
        return frames

    def _load_frames_cv2(self, path: Path) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        need = self.clip_len * self.frame_stride
        start = max(0, (total - need) // 2)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for i in range(self.clip_len):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + i * self.frame_stride)
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames)  # (T, H, W, C)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.backend == "decord":
            frames = self._load_frames_decord(path)
        else:
            frames = self._load_frames_cv2(path)

        from PIL import Image
        tensor_frames = torch.stack([
            self.transform(Image.fromarray(f)) for f in frames
        ])  # (T, C, H, W)

        return tensor_frames, label
