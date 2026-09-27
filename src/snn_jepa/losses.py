"""损失函数：JEPA 潜空间预测 + 抗坍塌正则 + MAE 像素重建。

抗坍塌正则 MVP 采用 VICReg（方差项 + 协方差项），对应项目的 H2'/H5 基线；
阶段 2 将接入 SIGReg 与生物机制（发放率稳态 / 突触缩放 / 不应期门控）。
"""

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
