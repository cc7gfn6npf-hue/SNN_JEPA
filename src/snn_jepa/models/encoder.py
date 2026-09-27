"""脉冲神经网络编码器。

将静态图像编码为时间窗口内的脉冲计数嵌入（spike-count embedding）。

设计要点：
- 多步模式（step_mode='m'）：输入 [T, N, C, H, W]，GPU 上并行计算 T 个时间步。
- 静态图像以「恒定输入」方式沿时间维复制 T 次，等价于速率编码。
- 输出对时间维求和，得到离散、非负、有界的脉冲计数嵌入——这正是项目要检验的
  「脉冲计数表征是否满足各向同性高斯假设（H2'）」的对象。
"""

from typing import List

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class SpikingEncoder(nn.Module):
    """LIF/PLIF 脉冲前馈编码器，输出脉冲计数嵌入。

    Args:
        in_channels: 输入图像通道数（CIFAR-10/STL-10 为 3）。
        channels: 各卷积层输出通道数列表。
        embedding_dim: 输出嵌入维度。
        timesteps: 时间窗口步数 T。
        neuron_type: 'PLIF'（参数化，tau 可学习）或 'LIF'（固定 tau）。
        tau: 膜时间常数（LIF 用，PLIF 作为 init_tau）。
        v_threshold: 放电阈值。
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128],
        embedding_dim: int = 512,
        timesteps: int = 4,
        neuron_type: str = "PLIF",
        tau: float = 2.0,
        v_threshold: float = 1.0,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.embedding_dim = embedding_dim

        def make_neuron():
            sf = surrogate.Sigmoid(alpha=4.0)
            if neuron_type == "PLIF":
                return neuron.ParametricLIFNode(
                    init_tau=tau,
                    v_threshold=v_threshold,
                    surrogate_function=sf,
                    detach_reset=True,
                )
            return neuron.LIFNode(
                tau=tau,
                v_threshold=v_threshold,
                surrogate_function=sf,
                detach_reset=True,
            )

        prev = in_channels
        self.conv_blocks = nn.ModuleList()
        for c in channels:
            block = nn.Sequential(
                layer.Conv2d(prev, c, kernel_size=3, stride=1, padding=1, bias=False),
                layer.BatchNorm2d(c),
                make_neuron(),
                layer.MaxPool2d(2),
            )
            self.conv_blocks.append(block)
            prev = c

        self.fc = layer.Linear(channels[-1], embedding_dim, bias=False)
        self.lif_out = make_neuron()

        # 多步模式：整网接受 [T, N, ...] 输入
        functional.set_step_mode(self, "m")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 静态图像 [N, C, H, W]。

        Returns:
            脉冲计数嵌入 [N, embedding_dim]。
        """
        # 恒定输入编码：沿时间维复制
        x = x.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)  # [T, N, C, H, W]
        functional.reset_net(self)

        for block in self.conv_blocks:
            x = block(x)  # 每块输出 [T, N, c, h, w]

        # 全局平均池化（对空间维求脉冲均值），得到 [T, N, C]
        x = x.mean(dim=[3, 4])

        x = self.fc(x)          # [T, N, embedding_dim]
        x = self.lif_out(x)     # 输出脉冲 [T, N, embedding_dim]

        # 脉冲计数：沿时间维求和
        return x.sum(dim=0)     # [N, embedding_dim]


def build_encoder(cfg) -> SpikingEncoder:
    """从配置构建编码器（兼容 Hydra DictConfig 或普通 dict）。"""
    return SpikingEncoder(
        in_channels=cfg.get("in_channels", 3),
        channels=list(cfg.get("channels", [32, 64, 128])),
        embedding_dim=cfg.get("embedding_dim", 512),
        timesteps=cfg.get("timesteps", 4),
        neuron_type=cfg.get("neuron_type", "PLIF"),
        tau=cfg.get("tau", 2.0),
        v_threshold=cfg.get("threshold", 1.0),
    )
