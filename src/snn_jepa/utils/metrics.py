"""表征质量指标：有效秩与稳定秩。

有效秩（effective rank, Roy & Vetterli, 2007）为谱熵的指数，
用于量化表征矩阵的维度利用率，是诊断表征坍塌的核心指标。
稳定秩（stable rank）对尺度不敏感，二者互补。
"""

import math

import torch


def stable_rank(features: torch.Tensor) -> float:
    """稳定秩 srank = (||Z||_F)^2 / (||Z||_2)^2，对尺度鲁棒。"""
    Z = features.detach()
    fro = torch.norm(Z, p="fro")
    spec = torch.linalg.matrix_norm(Z, ord=2)
    if spec.item() == 0.0:
        return 0.0
    return float((fro**2) / (spec**2))


def effective_rank(features: torch.Tensor, eps: float = 1e-8) -> float:
    """有效秩 = exp(H)，H 为奇异值归一化后的谱熵（Roy & Vetterli, 2007）。"""
    Z = features.detach()
    s = torch.linalg.svdvals(Z)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    entropy = float(-(p * torch.log(p)).sum())
    return math.exp(entropy)


def spectral_entropy(features: torch.Tensor, eps: float = 1e-8) -> float:
    """谱熵 H，可直接用于监控坍塌程度（H 越低坍塌越严重）。"""
    Z = features.detach()
    s = torch.linalg.svdvals(Z)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(-(p * torch.log(p)).sum())
