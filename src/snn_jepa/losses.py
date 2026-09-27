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
    """发放率稳态正则（H2'，增强版）：约束发放率的均值与 batch 内分布。

    生物对应突触缩放（synaptic scaling）：神经元将发放率维持在目标区间——
    过低（沉默）上调突触，过高（过活跃）下调突触。增强版在「均值稳态」基础上
    增加「方差上下界」：下界防止维度在 batch 内恒定（坍塌），上界防止单维度
    过活跃（srank=1.0 的能量碾压）。这是脉冲表征（非负有界）下更完整的稳态约束。

    与 SIGReg 的关键区别：SIGReg 强制匹配各向同性高斯（含负半部分），但脉冲计数
    嵌入非负有界，理论上不可能匹配；发放率稳态只约束发放率分布（均值 + 上下界），
    契合脉冲表征的物理特性，是更「生物」且更适配 SNN 的抗坍塌机制。

    Args:
        z: 脉冲计数嵌入 [N, D]。
        target_rate: 目标发放率（0~1，相对 T 的比例）。
        timesteps: 时间步数 T。
    """
    rate = z / timesteps  # [N, D] 每样本每维发放率（0~1）
    mean_rate = rate.mean(dim=0)  # [D] batch 平均
    std_rate = rate.std(dim=0)    # [D] batch 内标准差
    mean_loss = ((mean_rate - target_rate) ** 2).mean()
    lower_loss = F.relu(0.05 - std_rate).pow(2).mean()   # 方差下界：防沉默/坍塌
    upper_loss = F.relu(std_rate - 0.35).pow(2).mean()   # 方差上界：防单维度碾压
    return mean_loss + lower_loss + upper_loss


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


def rate_distribution_loss(
    z: torch.Tensor,
    num_directions: int = 8,
    target_rate: float = 0.25,
    timesteps: int = 4,
) -> torch.Tensor:
    """发放率分布匹配（H2' 高阶版）：随机投影后，匹配发放率的均值与方差到目标分布。

    对应生物的稳态可塑性维持全体神经元的发放率分布（高阶约束，非仅均值）。
    关键：约束「各投影方向的方差相等」，直接防止能量集中（srank=1.0）——这是低阶
    发放率稳态（仅均值/方差区间）做不到的。相比 SIGReg 匹配高斯（含负半部分），本
    机制匹配非负的发放率分布，契合脉冲表征的非负有界特性，是脉冲域 SIGReg 的生物替代。

    Args:
        z: 脉冲计数嵌入 [N, D]。
        num_directions: 随机投影方向数。
        target_rate: 目标发放率。
        timesteps: 时间步数 T。
    """
    n, d = z.shape
    rate = z / timesteps  # [N, D] 发放率 ∈ [0,1]
    # 目标方差：对应 Binomial(T, target_rate) 的自然发放方差
    target_var = target_rate * (1.0 - target_rate) / timesteps
    u = torch.randn(d, num_directions, device=z.device)
    u = u / (u.norm(dim=0, keepdim=True) + 1e-8)
    proj = rate @ u  # [N, M]
    total = torch.zeros((), device=z.device)
    for m in range(num_directions):
        h = proj[:, m]
        total = total + (h.mean() - target_rate) ** 2 + (h.var(unbiased=False) - target_var) ** 2
    return total / num_directions


def jepa_rate_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    pred: torch.Tensor,
    lam: float,
    target_rate: float = 0.25,
    timesteps: int = 4,
    num_directions: int = 8,
):
    """JEPA 总损失（发放率分布匹配版）= 潜空间 L2 预测 + λ·发放率分布匹配。"""
    pred_loss = F.mse_loss(pred, z2)
    reg = rate_distribution_loss(z1, num_directions, target_rate, timesteps)
    return pred_loss + lam * reg, pred_loss, reg


def weak_sigreg_loss(z: torch.Tensor, num_directions: int = 8) -> torch.Tensor:
    """弱 SIGReg（二阶矩白化）：约束投影的协方差 = 单位矩阵。

    对应生物的 flashlight 稳态可塑性（颗粒细胞逆行信使），是 SIGReg 的**生物可实现形式**
    （arXiv:2607.21622 证明其与 STDP⁺ 一起精确实现 SIGReg 梯度，无需反向传播）。
    只约束二阶矩——投影方差=1（白化）+ 投影间去相关，不要求完整高斯，因此更契合
    脉冲表征（离散非负有界）。

    关键区别（相比失败的 rate_distribution_loss）：目标方差 = 1（白化），而非
    发放率的 Binomial 方差 0.047；作用在脉冲计数 z（方差可达 T/4）而非 rate（≤0.25）。

    Args:
        z: 脉冲计数嵌入 [N, D]。
        num_directions: 随机投影方向数。
    """
    n, d = z.shape
    zc = z - z.mean(dim=0)
    u = torch.randn(d, num_directions, device=z.device)
    u = u / (u.norm(dim=0, keepdim=True) + 1e-8)
    f = zc @ u  # [N, M] 投影

    var = f.var(dim=0, unbiased=False)  # [M]
    var_loss = ((var - 1.0) ** 2).mean()  # 白化：各投影方差 = 1

    f_norm = f / (f.std(dim=0, unbiased=False) + 1e-8)
    corr = (f_norm.t() @ f_norm) / n  # [M, M] 相关矩阵
    off = corr - torch.diag(torch.diag(corr))
    cov_loss = (off ** 2).sum() / max(num_directions, 1)  # 去相关

    return var_loss + cov_loss


def jepa_weak_sigreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    pred: torch.Tensor,
    lam: float,
    num_directions: int = 8,
):
    """JEPA 总损失（弱 SIGReg 版）= 潜空间 L2 预测 + λ·投影协方差白化。"""
    pred_loss = F.mse_loss(pred, z2)
    reg = weak_sigreg_loss(z1, num_directions)
    return pred_loss + lam * reg, pred_loss, reg
