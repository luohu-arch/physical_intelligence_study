# PAIWorld: A 3D-Consistent World Foundation Model for Robotic Manipulation

- 本地 PDF：`papers/vla-architecture/PAIWorld_2606.18375.pdf`
- arXiv：https://arxiv.org/abs/2606.18375
- 年份：2026（6 月）
- 团队：中科院工业 AI 研究所
- 阶段：3D 一致世界模型 —— Cosmos-Predict2.5 (~14B) + 几何感知跨视角注意力

## 一句话总结

PAIWorld 解决世界模型的多视角 3D 不一致问题——Geo-RoPE 几何旋转位置编码 + Latent 3D-REPA 3D 蒸馏，在 DiT backbone 上同时注入视角间通信和 3D 几何先验。WorldArena 第一，超加性增益 2.64 > 0.93+0.72。

## 核心技术

1. **Geo-RoPE** — 相机射线方向+位姿编码到 RoPE，几何引导的跨视角 token 匹配
2. **Latent 3D-REPA** — 从 Depth Anything 3 蒸馏 3D 特征，pairwise cosine similarity 对齐
3. **超加性效应** — 跨视角注意力 + 3D 蒸馏联合 > 单独之和
4. **下游应用** — Model-based planning, world action models, multi-view policy post-training

## 底层原理与数学推导

Geo-RoPE 双子空间设计：Ray subspace（像素级 3D 射线方向）+ Pose subspace（视角级相机位姿特征）。Latent 3D-REPA 用随机锚点采样降低复杂度 O(n²)→O(n)。

## 物理直觉解释

你在不同角度看到一个杯子，知道是同一个杯子，因为你对 3D 空间有理解。传统世界模型从不同角度生成视频时，杯子会漂移、变形、变色，因为它不理解 3D。PAIWorld 通过 Geo-RoPE 让模型"知道每个像素在 3D 空间中的位置"，通过 3D-REPA 让模型"理解 3D 结构"。

## 工程细节与实操指南

- Backbone: NVIDIA Cosmos-Predict2.5 (~14B), DiT-based
- 训练: 200 H200 GPUs, 30K iters, ~7 days
- 数据: 2.5M multi-view robot manipulation clips (AgiBot-World, RoboMIND, Galaxea, RoboTwin, RoboCOIN)

## 消融实验与分析

| 消融 | MEt3R 提升 | 结论 |
|------|-----------|------|
| 仅跨视角注意力 | +0.93 | 通信通道必要但不充分 |
| 仅 Latent 3D-REPA | +0.72 | 几何先验有用 |
| **两者联合** | **+2.64** | **超加性——1+1>2** |

## 技术权衡

| 优势 | 劣势 |
|------|------|
| WorldArena 第一，多视角 3D 一致性 SOTA | 14B backbone, 200 H200, 7 days — 训练成本极高 |
| 超加性效应验证了联合设计的必要性 | 仅限已知相机外参场景 |

## 技术价值与演进定位

PAIWorld 是 3D 一致世界模型的标杆——和 FP3（3D 策略）形成互补，后者关注"如何用 3D 做动作"，前者关注"如何在 3D 中做想象"。

## 与其他论文的关系

- **WEAVER (2026)** — 多视角世界模型，关注保真度/效率；PAIWorld 关注 3D 一致性
- **RISE (RSS 2026)** — 想象中 RL, PAIWorld 提供了更好的 imagination engine
- **Cosmos-Predict** — PAIWorld 的 backbone, 加入了多视角和 3D 能力

## 精读问题

1. Geo-RoPE 对相机外参标定误差的鲁棒性？
2. 14B backbone 的推理延迟是否支持实时规划？
