# Human-as-Humanoid: Enabling Zero-Shot Humanoid Learning from Ego-Exo Human Videos

- 本地 PDF：`papers/vla-architecture/Human-as-Humanoid_2606.32009.pdf`
- arXiv：https://arxiv.org/abs/2606.32009
- 年份：2026（6 月, CoRL 2026）
- 团队：HKUST(GZ), DeepCybo 等
- 阶段：硬件-软件联合设计 —— 人类视频零样本→人形动作

## 一句话总结

Human-as-Humanoid 提出硬件-软件联合设计：设计 PrimeU 人形使其身体比例对齐人类，配合 ego-exo 双视角视频→60-DoF 动作 pipeline，实现零样本迁移——无需任何机器人演示数据。CoRL 2026。

## 核心技术

1. **PrimeU 硬件** — 肩宽比 0.97, 臂长比 1.02, 手长比 1.00 人类对齐人形
2. **Ego-Exo Pipeline** — ~20 FPS: 外视角跟踪→mesh 重建→分阶段 IK 重映射→60-DoF action
3. **DS-HKC Loss** — FK-aware dual-space 监督（关节空间 + 任务空间）
4. **4.8-7.2× throughput** over 传统动捕遥操作

## 底层原理与数学推导

分阶段 IK: 先求解臂部 7-DoF IK 使腕部位姿匹配人类腕部，再求解手部 20-DoF IK 使指尖位置匹配。DS-HKC loss 同时约束关节角度和末端位姿。

## 物理直觉解释

将人类到机器人的迁移从 AI 问题降为运动学问题——如果机器人的身体比例和人类一样，那么人类的动作可以直接通过 IK 映射到机器人，不需要大规模学习 embodiment gap。

## 工程细节与实操指南

- PrimeU: 双臂 7-DoF + 双灵巧手 20-DoF + 颈 3-DoF + 腰 3-DoF
- 传感: 头戴+腕部 RealSense D435
- Pipeline: 外视角人体跟踪→mesh-aware 重建→staged IK→60-DoF chunks
- 测试任务: 戒指放置、魔方装箱、倒水、叠杯、拧瓶盖

## 消融实验与分析

| 消融 | 结论 |
|------|------|
| Ego-Exo vs 仅 Exo | 双视角对精细手部操作必要 |
| Staged IK vs 统一 IK | 分阶段求解稳定性和精度均更优 |
| DS-HKC vs 仅关节空间 | 任务空间约束对末端精度关键 |

## 技术权衡

| 优势 | 劣势 |
|------|------|
| 零样本，无需机器人 demo | 依赖定制硬件 PrimeU |
| 4.8-7.2× throughput over 动捕 | 仅覆盖上身，不含腿部 locomotion |

## 技术价值与演进定位

核心洞察：**硬件对齐可以大幅降低 AI 负担**。和 HumanPlus (Stanford, 2024) 形成互补——Human-as-Humanoid 用硬件-软件联合设计，HumanPlus 用 RL 弥合 embodiment gap。

## 与其他论文的关系

- **HumanPlus (Stanford 2024)** — 人类视频→人形, 纯 RL 路线
- **ROVE (XPeng 2026)** — 人形 VLA 后训练, 互补

## 精读问题

1. 身体比例不对齐时 pipeline 哪一步最先崩？
2. Ego-exo 双视角 vs 单视角的精度差异量化？
