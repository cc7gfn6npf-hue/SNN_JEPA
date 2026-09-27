"""脉冲神经网络编码器。

将静态图像编码为时间窗口内的脉冲计数嵌入（spike-count embedding）。

设计要点（阶段 1 修正）：
- 单步模式 + 手动循环 T 步（官方验证的可靠配置，规避多步 BN 的潜在问题）。
- 静态图像以「恒定输入」重复 T 步，等价于速率编码。
- 默认 IF 神经元（无泄漏，恒定输入下膜电位线性累积、必然放电），
  解决 LIF 泄漏 + BN 零均值导致的「全零坍塌」与 surrogate 梯度消失问题；
  LIF/PLIF 保留为可选项，供阶段 2 生物机制（发放率稳态/不应期门控）使用。
- 输出对时间步求和，得到离散、非负、有界的脉冲计数嵌入。
"""

from typing import List

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron, surrogate


class SpikingEncoder(nn.Module):
    """脉冲前馈编码器，输出脉冲计数嵌入。

    Args:
        in_channels: 输入图像通道数。
        channels: 各卷积层输出通道数列表。
        embedding_dim: 输出嵌入维度。
        timesteps: 时间窗口步数 T。
        neuron_type: 'IF'（无泄漏）/ 'LIF'（固定 tau）/ 'PLIF'（参数化 tau）。
        tau: 膜时间常数（LIF/PLIF 使用）。
        v_threshold: 放电阈值。
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128],
        embedding_dim: int = 512,
        timesteps: int = 4,
        neuron_type: str = "IF",
        tau: float = 2.0,
        v_threshold: float = 0.5,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.embedding_dim = embedding_dim

        def make_neuron():
            sf = surrogate.ATan()
            if neuron_type == "LIF":
                return neuron.LIFNode(
                    tau=tau, v_threshold=v_threshold, surrogate_function=sf, detach_reset=True
                )
            if neuron_type == "PLIF":
                return neuron.ParametricLIFNode(
                    init_tau=tau, v_threshold=v_threshold, surrogate_function=sf, detach_reset=True
                )
            return neuron.IFNode(v_threshold=v_threshold, surrogate_function=sf, detach_reset=True)

        prev = in_channels
        self.conv_blocks = nn.ModuleList()
        for c in channels:
            block = nn.Sequential(
                nn.Conv2d(prev, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c),
                make_neuron(),
                nn.MaxPool2d(2),
            )
            self.conv_blocks.append(block)
            prev = c

        self.fc = nn.Linear(channels[-1], embedding_dim, bias=False)
        self.lif_out = make_neuron()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：静态图像 [N, C, H, W] -> 脉冲计数嵌入 [N, embedding_dim]。"""
        functional.reset_net(self)
        out = None
        for _ in range(self.timesteps):
            xt = x  # 恒定输入（速率编码）
            for block in self.conv_blocks:
                xt = block(xt)
            xt = xt.mean(dim=[2, 3])  # 全局平均池化 [N, C]
            xt = self.fc(xt)          # [N, D]
            xt = self.lif_out(xt)     # 输出脉冲 [N, D]
            out = xt if out is None else out + xt
        return out  # 脉冲计数 [N, D]


def build_encoder(cfg) -> SpikingEncoder:
    """从配置构建编码器（兼容 Hydra DictConfig 或普通 dict）。"""
    return SpikingEncoder(
        in_channels=cfg.get("in_channels", 3),
        channels=list(cfg.get("channels", [32, 64, 128])),
        embedding_dim=cfg.get("embedding_dim", 512),
        timesteps=cfg.get("timesteps", 4),
        neuron_type=cfg.get("neuron_type", "IF"),
        tau=cfg.get("tau", 2.0),
        v_threshold=cfg.get("threshold", 0.5),
    )
