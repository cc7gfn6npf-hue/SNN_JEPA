# 阶段 0 报告：环境与复现

## 一、目标

搭建可复现的 SNN+JEPA 实验环境，完成图像 JEPA 复现锚点，并验证"脉冲编码器 + JEPA 损失"的兼容性。

## 二、方法

- 环境：PyTorch（CUDA）+ SpikingJelly + Hydra + WandB，固定随机种子。
- 复现锚点：图像域 JEPA（`facebookresearch/eb_jepa` 的 VICReg/SIGReg，`rbalestr-lab/lejepa` 的 SIGReg）。
- 兼容性验证：以 LIF/PLIF 编码器替换 CNN 编码器，输出脉冲计数嵌入，接 JEPA 损失 + 抗坍塌正则。

## 三、当前状态

**已完成（2026-09-27）：**

- [x] 仓库结构与 git 初始化
- [x] 依赖文件：`environment.yml`、`requirements.txt`
- [x] Hydra 配置骨架：`configs/`（data / model / experiment）
- [x] 工具模块：`src/snn_jepa/utils/seed.py`（固定种子）、`metrics.py`（有效秩/稳定秩/谱熵）
- [x] 复现计划：`docs/reproduction_plan.md`

**进行中 / 待办：**

- [ ] 云 GPU 平台确认与 torch/CUDA 版本锁定
- [ ] eb_jepa 图像 JEPA 复现（VICReg/SIGReg）
- [ ] SNN 编码器 + JEPA 损失兼容性验证
- [ ] WandB 日志链路与有效秩监控

## 四、关键决策记录

1. **算力**：云 GPU（CUDA），非本机 Apple Silicon MPS。规模上限解除，训练集可用 STL-10（+ 可选 Tiny-ImageNet 子集）。
2. **复现锚点**：图像 JEPA，非 SG-JEPA（动态图论文）。
3. **神经数据**：Allen Brain Observatory 先行（零审批、即开即用），再上 NSD/Algonauts。
4. **抗坍塌机制排序**：发放率稳态（B）→ 突触缩放/去相关（C）→ 不应期梯度门控（A），阶段 2 前定夺。
5. **核心叙事**：Biologically-Grounded Anti-Collapse——人工抗坍塌技巧的皮层内源对应物。

## 五、结论

阶段 0 的仓库骨架、依赖与复现计划已就绪。环境验证依赖云 GPU 平台的最终确认，是当前唯一的外部阻塞项。

## 六、下一步（阶段 0 收尾 → 阶段 1）

1. 确认云 GPU 平台（AutoDL / Colab Pro+ / Lambda / 自有集群）与 CUDA 版本，锁定 torch/spikingjelly 版本。
2. 在云 GPU 上跑通 0.3/0.4 复现步骤，记录参照指标与有效秩曲线。
3. 通过退出条件后，进入阶段 1：实现 SNN-MAE 与 SNN-JEPA 基线。
