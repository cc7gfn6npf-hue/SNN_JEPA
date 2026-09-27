"""损失函数：JEPA 潜空间预测 + 抗坍塌正则 + MAE 像素重建。

抗坍塌正则 MVP 采用 VICReg（方差项 + 协方差项），对应项目的 H2'/H5 基线；
阶段 2 将接入 SIGReg 与生物机制（发放率稳态 / 突触缩放 / 不应期门控）。
"""

import math

import torch
import torch.nn.functional as F


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg 方差项：迫使每个维度标准差 ≥ gamma，防止维度坍塌。

    Args:
        z: 嵌入 [N, D]。
    """
    std = z.std(dim=0)  # [D]
    return F.relu(gamma - std).pow(2).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg 协方差项：迫使维度间去相关。

    Args:
        z: 嵌入 [N, D]。
    """
    z = z - z.mean(dim=0)
    n = z.shape[0]
    cov = (z.t() @ z) / (n - 1)  # [D, D]
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / z.shape[1]


def vicreg_loss(z: torch.Tensor, lambda_std: float, lambda_cov: float, gamma: float = 1.0) -> torch.Tensor:
    """VICReg 正则项（不含不变性项，不变性由 JEPA 预测损失承担）。"""
    return lambda_std * variance_loss(z, gamma) + lambda_cov * covariance_loss(z)


def jepa_loss(z1: torch.Tensor, z2: torch.Tensor, pred: torch.Tensor, lambda_std: float, lambda_cov: float) -> torch.Tensor:
    """JEPA 总损失 = 潜空间 L2 预测 + VICReg 正则。

    Args:
        z1: 上下文嵌入 [N, D]。
        z2: 目标嵌入 [N, D]（应已 stop-gradient）。
        pred: 预测器输出 [N, D]。
    """
    pred_loss = F.mse_loss(pred, z2)
    reg = vicreg_loss(z1, lambda_std, lambda_cov)
    return pred_loss + reg, pred_loss, reg


def mae_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAE 像素重建损失（MSE）。"""
    return F.mse_loss(recon, target)


def sigreg_loss(z: torch.Tensor, num_directions: int = 8) -> torch.Tensor:
    """SIGReg（LeJEPA 简版）：随机投影 + 特征函数匹配，强制嵌入匹配各向同性高斯 N(0, I)。

    依据 Cramér–Wold 定理：一个 d 维分布是各向同性高斯，当且仅当它在所有一维投影上
    都是标准正态。因此把嵌入投影到 M 个随机方向，逐个用特征函数（Epps–Pulley 型）
    匹配 N(0,1)，即可从结构上消除「全零坍塌」与「维度坍塌（srank=1.0）」。

    单投影损失（与固定 N(0,1) 匹配，含方差约束）：
        L(h) = (1/N²) Σ_{j,k} exp(-(h_j - h_k)²/2) - (√2/N) Σ_j exp(-h_j²/4) + 1/√2

    Args:
        z: 嵌入 [N, D]。
        num_directions: 随机投影方向数 M。
    """
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)  # 中心化（投影均值自动≈0）
    u = torch.randn(d, num_directions, device=z.device)
    u = u / (u.norm(dim=0, keepdim=True) + 1e-8)  # 单位随机方向
    proj = zc @ u  # [N, M]

    total = torch.zeros((), device=z.device)
    for m in range(num_directions):
        h = proj[:, m]  # 未标准化，直接匹配 N(0,1)
        diff = h.unsqueeze(0) - h.unsqueeze(1)
        pair = torch.exp(-(diff ** 2) / 2.0).sum()          # Σ_{j,k}
        s2 = torch.exp(-(h ** 2) / 4.0).sum()               # Σ_j
        total = total + pair / (n ** 2) - math.sqrt(2.0) * s2 / n + 1.0 / math.sqrt(2.0)

    return total / num_directions


def jepa_sigreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    pred: torch.Tensor,
    lam: float,
    num_directions: int = 8,
):
    """JEPA 总损失（SIGReg 正则版）= 潜空间 L2 预测 + λ·SIGReg。"""
    pred_loss = F.mse_loss(pred, z2)
    reg = sigreg_loss(z1, num_directions)
    return pred_loss + lam * reg, pred_loss, reg
