"""MAE 解码器：从潜表征重建像素。

SNN-MAE 基线中，脉冲编码器输出潜表征，ANN 转置卷积解码器重建原始图像。
解码器使用普通 ANN（像素重建无需脉冲动力学）。
"""

import torch.nn as nn


class Decoder(nn.Module):
    """转置卷积解码器。

    Args:
        embedding_dim: 潜表征维度。
        in_channels: 输入图像通道数。
        init_resolution: 转置卷积起点分辨率（CIFAR-10 为 8，STL-10 为 12）。
    """

    def __init__(self, embedding_dim: int, in_channels: int = 3, init_resolution: int = 8):
        super().__init__()
        hidden = 128
        self.init_resolution = init_resolution
        self.fc = nn.Linear(embedding_dim, hidden * init_resolution * init_resolution)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(hidden, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        # z: [N, embedding_dim]
        x = self.fc(z)
        x = x.view(x.shape[0], -1, self.init_resolution, self.init_resolution)
        return self.deconv(x)
