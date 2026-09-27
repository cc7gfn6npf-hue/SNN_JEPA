"""预测器：在潜空间预测目标嵌入。

JEPA 中预测器从上下文嵌入预测目标嵌入。此处使用轻量 ANN MLP（潜空间预测无需
脉冲动力学），符合项目架构「预测器：轻量 MLP」的设定。
"""

from typing import List, Optional

import torch.nn as nn


class Predictor(nn.Module):
    """轻量 MLP 预测器。

    Args:
        input_dim: 输入嵌入维度。
        hidden_dims: 隐层维度列表。
        output_dim: 输出维度，默认等于 input_dim。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256],
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def build_predictor(cfg, embedding_dim: int) -> Predictor:
    return Predictor(
        input_dim=embedding_dim,
        hidden_dims=list(cfg.get("hidden_dims", [256])),
        output_dim=cfg.get("output_dim", embedding_dim),
    )
