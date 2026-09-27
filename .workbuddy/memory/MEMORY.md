# SNN-JEPA 项目长期笔记

## 项目定位

在脉冲神经网络（SNN）中实现 JEPA，检验潜空间预测是否比输入重建（MAE）更贴近生物皮层表征。核心叙事：**Biologically-Grounded Anti-Collapse**。

## 关键决策（勿重复讨论）

1. **算力**：云 GPU（AutoDL）。本机是 Apple Silicon Mac，MPS 不适合训练，仅用于代码开发/调试。
   - **版本锁定**：CUDA 12.1 + PyTorch 2.3.0（镜像 pytorch-2.3.0+cu121）+ SpikingJelly 0.0.0.0.14 + Python 3.10。
2. **复现锚点**：图像 JEPA（`facebookresearch/eb_jepa`、`rbalestr-lab/lejepa`）。**SG-JEPA 是动态图论文，仅作引用，不复现。**
3. **神经数据**：Allen Brain Observatory（allensdk，零审批）先行 → NSD/Algonauts 2023。
4. **抗坍塌机制排序**：发放率稳态(B) → 突触缩放/去相关(C) → 不应期梯度门控(A)。
5. **论文定位与增补（2026-09-27）**：首发 CCN/COSYNE，升级 NeurIPS/eLife。三大缺口：能量分析（为什么用SNN）、时间维度（spiking真实性）、novelty硬度。当前主线=生物可塑性规则做硬 + 能量分析 + 多基线/多种子（≥3）；时序主线留作第二篇。详见 `docs/research_plan.md`。

## 术语锚点

- SIGReg = Sketched Isotropic Gaussian Regularization（LeJEPA, arXiv:2511.08544），单超参 λ，Cramér–Wold + Epps–Pulley，把嵌入拉向各向同性高斯，无需 stop-gradient/EMA。
- VISReg（Wu et al. 2026）修复 SIGReg 的"坍塌时梯度消失"与"尺度/形状耦合"。
- 有效秩（Roy & Vetterli 2007）、srank：见 `src/snn_jepa/utils/metrics.py`。

## 风险清单

- 脉冲计数嵌入（离散、非负、有界）未必满足 SIGReg 的各向同性高斯假设——这是 H2' 的核心检验点，无论成立与否都是贡献。
- 静态图像经速率编码后，脉冲特性退化为激活函数，生物合理性论证需在讨论中明确分层。
- **全零坍塌（SNN 训练经典陷阱）**：LIF 泄漏 + BN 零均值 + 阈值过高 → 放电率过低 → surrogate 梯度消失 → 表征全零。对策：单步模式 + IF 神经元（无泄漏）+ ATan 代理梯度 + 降阈值（0.5）。诊断信号：eff_rank=0、reg 恒等于 variance 最大值。
- **数据集下载**：torchvision 0.18 的 CIFAR/STL 源在国内超时，需手动下载（旧地址或镜像）到 data 目录。
- **维度坍塌实证（阶段1，2026-09-27）**：未修复的 JEPA 在 CIFAR-10 上 srank=1.0（能量集中单一方向），线性探测 32.53% < MAE 43.93%。**H1 已修正为条件性表述**（有效抗坍塌下 JEPA 优于 MAE）。阶段 2 优先：SIGReg 拉 srank。
