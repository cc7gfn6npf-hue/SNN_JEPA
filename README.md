# SNN-JEPA

在脉冲神经网络（SNN）中实现 Joint-Embedding Predictive Architecture（JEPA），系统检验**潜空间预测**是否比**输入空间重建**（MAE 式）产生更接近生物皮层（V1/V4/IT）的神经表征。

## 核心命题

JEPA 为防表征坍塌而引入的人工技巧，在皮层中都有内源对应物。本项目检验：用皮层内源机制显式替换这些人工技巧，是否能在不损失表征质量的前提下，使表征更贴合皮层编码。

| 人工技巧 | 生物对应机制 | 可检验假设 |
|---|---|---|
| 方差正则（VICReg/SIGReg 方差项） | 发放率稳态（firing-rate homeostasis） | H2' |
| 协方差去相关 | 突触缩放 + 侧向抑制 | H5 |
| stop-gradient / EMA | 不应期梯度门控 | H4' |

## 假设

- **H1**：SNN-JEPA 在 V4/IT 层级的线性神经预测精度显著高于 SNN-MAE。
- **H2'**：发放率稳态可等价替代方差正则防坍塌，稳态强度与表征有效秩单调相关。
- **H4'**：不应期梯度门控可替代 EMA + stop-gradient，效果取决于不应期时间常数 τ_ref。
- **H5**：突触缩放/侧向去相关可等价替代协方差正则项，提升维度解耦与线性探测精度。

## 架构

- **上下文编码器**：LIF/PLIF 脉冲前馈网络 → 时间窗口脉冲计数嵌入。
- **目标编码器**：同构，参数由 EMA 或非对称更新获得。
- **预测器**：轻量 MLP，在潜空间预测目标嵌入。
- **损失**：JEPA 潜空间 L2 损失 + 抗坍塌正则（SIGReg / VICReg / 生物机制）。
- **基线**：SNN-MAE（输入重建）、ANN-JEPA、RPL。

## 目录结构

```
SNN_JEPA/
├── configs/            # Hydra 配置
│   ├── config.yaml
│   ├── data/           # cifar10 / stl10
│   ├── model/          # encoder（含 predictor）
│   └── experiment/     # jepa_vicreg / jepa_sigreg
├── src/snn_jepa/       # Python 包
│   └── utils/          # seed / metrics（有效秩、srank）
├── scripts/            # 训练/评估入口（阶段 1）
├── experiments/        # 运行日志与 checkpoint
├── data/               # 数据集
└── docs/               # 阶段报告、复现计划
```

## 快速开始

```bash
# 1. 安装依赖（torch 需匹配你的 CUDA 版本，见 docs/reproduction_plan.md）
conda env create -f environment.yml
conda activate snn-jepa

# 2. 复现图像 JEPA（eb_jepa 的 VICReg/SIGReg）验证环境
#    详见 docs/reproduction_plan.md

# 3. 阶段 1 起：
#    python scripts/train.py --config-name config
```

## 状态

- [x] 阶段 0：环境与复现（仓库结构、依赖、复现计划）—— 进行中
- [ ] 阶段 1：基线搭建（SNN-MAE / SNN-JEPA）
- [ ] 阶段 2：抗坍塌机制开发与消融（B→C→A）
- [ ] 阶段 3：神经数据验证（Allen → NSD/Algonauts）
- [ ] 阶段 4：论文产出

## 关键决策记录（详见 docs/phase0_report.md）

1. 算力：云 GPU（CUDA），非本机 MPS。
2. 复现锚点：**图像 JEPA**（`facebookresearch/eb_jepa`、`rbalestr-lab/lejepa`），**非** SG-JEPA（SG-JEPA 为动态图论文）。
3. 神经数据：Allen Brain Observatory 先行 → NSD/Algonauts。
4. 抗坍塌：SIGReg = Sketched Isotropic Gaussian Regularization（LeJEPA 2025）。
