"""ANN 编码器：与 SpikingEncoder 同结构，但用 ReLU 替代脉冲。

用途：ANN-JEPA 基线——隔离「脉冲」的贡献。若 SNN 与 ANN 精度差异大，说明
脉冲动力学是瓶颈；若差异小，说明脉冲在精度上不损失（仅省能耗）。
"""

from typing import List

import torch.nn as nn


class ANNEncoder(nn.Module):
    """与 SpikingEncoder 对齐的 ANN 编码器（Conv-BN-ReLU-MaxPool）。"""

    def __init__(self, in_channels: int = 3, channels: List[int] = [32, 64, 128], embedding_dim: int = 512):
        super().__init__()
        prev = in_channels
        blocks = []
        for c in channels:
            blocks += [
                nn.Conv2d(prev, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            prev = c
        self.conv_blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels[-1], embedding_dim, bias=False)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.mean(dim=[2, 3])  # 全局平均池化 [N, C]
        return self.fc(x)       # [N, D] 连续嵌入
