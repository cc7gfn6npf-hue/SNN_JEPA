# 阶段 0 复现计划：图像 JEPA + SNN 编码器兼容性验证

## 目标

1. 搭建可复现环境（PyTorch + SpikingJelly + Hydra + WandB，固定种子）。
2. 复现**图像域 JEPA**（VICReg 与 SIGReg 两种抗坍塌正则）在 CIFAR-10 上的结果，作为环境正确性与性能上界基线。
3. 验证 **SNN 编码器 + JEPA 损失可兼容**：脉冲计数嵌入在 JEPA 目标下稳定训练、不坍塌、线性探测显著优于随机。

> 说明：原项目书中的"SG-JEPA"实为动态图论文（Zhang et al., arXiv:2607.18412），与视觉/神经可预测性目标不符，已降级为引用。复现锚点改为 `facebookresearch/eb_jepa`（图像 JEPA，含 VICReg/SIGReg）与 `rbalestr-lab/lejepa`（SIGReg 官方实现）。

## 步骤

### 0.1 环境安装

**版本锁定（已确认）：CUDA 12.1 + PyTorch 2.3.0 + SpikingJelly 0.0.0.0.14，Python 3.10。**

```bash
# AutoDL 方式（推荐）：直接选镜像 pytorch-2.3.0+cu121，已预装 torch/torchvision
# 无需手动装 torch，只装其余依赖：
pip install -r requirements.txt

# 手动方式（非 AutoDL 镜像时）：
conda env create -f environment.yml && conda activate snn-jepa
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 校验
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import spikingjelly; print('spikingjelly', spikingjelly.__version__)"
```

> 说明：稳定版 SpikingJelly 默认使用纯 torch 后端，无需安装 cupy/triton。若后续阶段 2/3 追求多步神经元训练加速，再评估升级到 torch 2.6+ 开发版（含 triton 后端，官方测试 triton==3.3.1）。

### 0.2 可复现性冒烟测试

```bash
python -c "
from snn_jepa.utils.seed import set_seed
import torch
set_seed(42)
a = torch.randn(100).sum().item()
set_seed(42)
b = torch.randn(100).sum().item()
assert a == b, 'seed not reproducible'
print('seed reproducible:', a)
"
```

### 0.3 复现图像 JEPA（性能上界基线）

克隆并运行 `facebookresearch/eb_jepa` 的 CIFAR-10 示例，分别用 VICReg 与 SIGReg：

```bash
git clone https://github.com/facebookresearch/eb_jepa.git /tmp/eb_jepa
# 按 eb_jepa/examples/image_jepa/README.md 运行 CIFAR-10 训练 + 线性探测
```

**参照指标**（eb_jepa README 报告）：VICReg 最佳 ~90%、SIGReg 最佳 ~91% 的 CIFAR-10 线性探测精度。本机复现误差在 ±1–2% 内视为环境正确。

### 0.4 SNN 编码器 + JEPA 损失兼容性（关键退出条件）

用一个小型 LIF/PLIF 编码器替换 eb_jepa 的 CNN 编码器，保持 JEPA 损失不变：

- 静态图像以恒定输入或首步注入方式编码为时间窗口内的脉冲序列；
- 输出取时间窗口脉冲计数（spike count）作为嵌入；
- 分别接 VICReg 与 SIGReg 正则。

**通过标准**：
- 训练损失收敛且不坍塌（有效秩不趋近于 1）；
- CIFAR-10 线性探测精度显著高于随机（> 40% 即视为兼容，不追求追平 ANN）；
- WandB 正常记录有效秩/谱熵曲线。

## 退出条件（阶段 0 完成标志）

- [ ] 固定种子下两次运行 loss 曲线逐位一致。
- [ ] 图像 JEPA（VICReg/SIGReg）复现，CIFAR-10 线性探测与报告值误差 ≤ 2%。
- [ ] SNN 编码器在 JEPA 损失下稳定训练、不坍塌、线性探测显著优于随机。
- [ ] Hydra 配置 + WandB 日志链路贯通。

## 参考

- eb_jepa: https://github.com/facebookresearch/eb_jepa
- LeJEPA (SIGReg): Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics", arXiv:2511.08544；代码 https://github.com/rbalestr-lab/lejepa
- VICReg: Bardes et al., 2022
