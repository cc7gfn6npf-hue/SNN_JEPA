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


def homeostasis_loss(z: torch.Tensor, target_rate: float = 0.25, timesteps: int = 4) -> torch.Tensor:
    """发放率稳态正则（H2'）：惩罚偏离目标发放率的维度，生物对应突触缩放（synaptic scaling）。

    生物机制：神经元通过突触缩放维持目标发放率——发放率过高则下调输入权重，
    过低则上调。在嵌入层面等价于约束各维度发放率均匀：既防止维度坍塌（发放率
    过低→沉默），又防止单维度碾压（发放率过高→过活跃），实现能量均匀化。

    与 SIGReg 的关键区别：SIGReg 强制匹配各向同性高斯（含负半部分），但脉冲计数
    嵌入非负有界，理论上不可能匹配；发放率稳态只约束发放率均匀，契合脉冲表征的
    物理特性，是更「生物」且更适配 SNN 的抗坍塌机制。

    Args:
        z: 脉冲计数嵌入 [N, D]。
        target_rate: 目标发放率（0~1，相对 T 的比例）。
        timesteps: 时间步数 T。
    """
    rate = z.mean(dim=0) / timesteps  # [D] 每维平均发放率（0~1）
    return ((rate - target_rate) ** 2).mean()


def jepa_bio_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    pred: torch.Tensor,
    lambda_homeo: float,
    lambda_decorr: float,
    target_rate: float = 0.25,
    timesteps: int = 4,
):
    """JEPA 总损失（生物抗坍塌版）= 潜空间 L2 预测 + 发放率稳态 + 侧向去相关。

    发放率稳态（H2'）替代 VICReg 方差项；侧向去相关（H5，生物对应侧向抑制/突触缩放）
    替代 VICReg 协方差项。二者合起来构成「生物版 VICReg」，但机制内源、无人工技巧。
    """
    pred_loss = F.mse_loss(pred, z2)
    reg = lambda_homeo * homeostasis_loss(z1, target_rate, timesteps) + lambda_decorr * covariance_loss(z1)
    return pred_loss + reg, pred_loss, reg
