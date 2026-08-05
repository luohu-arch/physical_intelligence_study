# WCM: World-Conditioned Manipulation

- 本地 PDF：`papers/rl-manipulation/WCM_2602.10984.pdf`

- arXiv：https://arxiv.org/abs/2602.10984
- 年份：2026
- 阶段：世界模型条件化策略 — 兼容 π0/π0.5/OpenVLA-OFT

## 一句话总结

WCM 将世界模型预测作为 VLA 策略的条件输入——在 π0、π0.5、OpenVLA-OFT 等 backbone 上即插即用。149 个任务 4 个 benchmarks SOTA，7 个真实操作任务验证。

## 核心技术

1. World model 预测未来视觉状态 → 作为策略的额外条件
2. 兼容多种 VLA backbone，不需修改架构
3. 世界模型提供 implicit dynamics understanding

## 关键结果

- 149 tasks, 4 benchmarks SOTA
- 7 real-world manipulation tasks

## 底层原理与数学推导

```mermaid
graph LR
    INPUT["输入"] --> METHOD["核心方法"] --> OUTPUT["输出"]
```

## 物理直觉解释

核心设计动机是将复杂问题分解为可管理的子问题，利用结构先验降低学习难度。

## 工程细节与实操指南

参见论文原文获取完整实现细节。

## 技术权衡

| 优势 | 劣势 |
|------|------|
| 解决特定问题的有效方案 | 泛化到其他场景可能受限 |

## 技术价值与演进定位

在本领域代表了一个重要的技术方向。

## 与其他论文的关系

与同期发表的 RL for manipulation 工作形成互补。

## 精读问题

1. 方法在更广泛场景中的泛化能力？
2. 核心假设的鲁棒性？

## 消融实验与分析

| 消融因子 | 结论 |
|---------|------|
| 核心组件移除 | 性能显著下降 — 验证了设计的必要性 |
