"""数据加载与增强。

- JEPA：同一图像的两个独立增强视图（RandomResizedCrop + 颜色抖动等）。
- MAE：原图 + 随机像素 mask。

静态图像经 ToTensor + Normalize 后送入 SNN，编码器内部做恒定输入的时间展开。
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# 数据集归一化统计
MEAN_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "stl10": ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
}


class TwoViewDataset(Dataset):
    """返回同一图像的两个增强视图，用于 augmentation-based JEPA。"""

    def __init__(self, base: Dataset, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.transform(img), self.transform(img)


class MaskedImageDataset(Dataset):
    """返回（masked 图像, 原图），用于 MAE。"""

    def __init__(self, base: Dataset, transform, mask_ratio: float = 0.75):
        self.base = base
        self.transform = transform
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        x = self.transform(img)  # [C, H, W] 归一化后
        # 像素级随机 mask（置 0）
        mask = torch.rand_like(x[0:1]) > self.mask_ratio  # [1, H, W]
        masked = x * mask.float()
        return masked, x


def get_transforms(name: str, image_size: int, train: bool):
    mean, std = MEAN_STD[name]
    if train:
        # JEPA 训练用增强（对齐 eb_jepa 图像 JEPA）
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.05)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_datasets(name: str, data_dir: str, image_size: int):
    """构建训练/测试数据集。"""
    mean, std = MEAN_STD[name]
    train_transform = get_transforms(name, image_size, train=True)
    eval_transform = get_transforms(name, image_size, train=False)

    if name == "cifar10":
        train_base = datasets.CIFAR10(data_dir, train=True, download=True)
        test_base = datasets.CIFAR10(data_dir, train=False, download=True)
    elif name == "stl10":
        train_base = datasets.STL10(data_dir, split="unlabeled", download=True)
        test_base = datasets.STL10(data_dir, split="test", download=True)
    else:
        raise ValueError(f"未知数据集: {name}")

    return train_base, test_base, train_transform, eval_transform


def get_dataloaders(name: str, data_dir: str, image_size: int, batch_size: int, num_workers: int = 4, target: str = "jepa", mask_ratio: float = 0.75):
    """构建 DataLoader。

    Returns:
        (train_loader, test_loader)。JEPA 返回双视图，MAE 返回 (masked, 原图)。
    """
    train_base, test_base, train_tf, eval_tf = build_datasets(name, data_dir, image_size)

    if target == "mae":
        train_ds = MaskedImageDataset(train_base, train_tf, mask_ratio)
    else:
        train_ds = TwoViewDataset(train_base, train_tf)

    # 测试集统一用单视图（线性探测/评估用）
    test_ds = TwoViewDataset(test_base, eval_tf) if target == "jepa" else MaskedImageDataset(test_base, eval_tf, mask_ratio=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, test_loader
