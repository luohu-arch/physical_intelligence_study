# IMR-LLM: Industrial Multi-Robot Task Planning and Program Generation with LLMs

- 本地 PDF：`papers/vla-architecture/IMR-LLM_2603.02669.pdf`
- arXiv：https://arxiv.org/abs/2603.02669
- 代码：https://github.com/XiangyuSu611/IMR-LLM-Code
- 年份：2026 (ICRA 2026 Best Paper on Automation)
- 团队：深圳大学 + 中科院工业 AI 所 + 视比特机器人 + Carleton U
- 阶段：LLM + 运筹优化 → 工业多机器人协同编程

## 一句话总结

IMR-LLM 用 LLM 做"翻译器"——LLM 将自然语言任务转为析取图(disjunctive graph)，由确定性求解器产生无死锁调度；LLM 再从 process tree 选路径生成可执行代码。23 工业场景 50 任务，编程从小时级→分钟级。

## 核心技术

1. LLM 翻译 + OR 求解：LLM 分解任务→析取图→确定求解器保证全局最优
2. Process Tree 代码生成：LLM 导航选路径替代从零生成代码
3. IMR-Bench: 23 场景, 50 任务, 最多 7 机器人 24 工序

## 底层原理与数学推导

```mermaid
graph LR
    NL["自然语言任务"] --> LLM1["LLM: 任务分解 + 机器人分派"]
    LLM1 --> GRAPH["析取图 (Disjunctive Graph)"]
    GRAPH --> SOLVER["OR 求解器 (最优调度)"]
    SOLVER --> SCHEDULE["无死锁调度方案"]
    SCHEDULE --> LLM2["LLM: Process Tree 路径选择"]
    LLM2 --> CODE["可执行 Python 代码"]
```

析取图: 节点=操作工序, 边=优先约束+资源冲突。LLM 生成图的节点和边结构，经典 Johnson 或遗传算法在图上求解最优调度。

## 物理直觉解释

不要用 LLM 直接写工厂调度代码——LLM 会"幻觉"出逻辑冲突导致机器人死锁。让 LLM 做它最擅长的事（理解任务描述），把"怎么最优排程"交给运筹学——这是数学上能证明最优的。

## 工程细节与实操指南

- 输入: 自然语言任务描述 + 产线配置
- 输出: 多机器人调度方案 + 可执行 Python 代码
- 场景: 造船和重型装备制造
- 实际部署: 3 机器人产线，含视觉定位、抓取、协作运输

## 消融实验与分析

| 消融 | 结论 |
|------|------|
| LLM+OR vs 纯 LLM | OR 消除了死锁和资源冲突 |
| Process Tree vs 从零生成 | Tree 路径选择比文本生成准确 10-20% |
| 任务复杂度上升 | IMR-LLM 优势随复杂度扩大 |

## 技术权衡

| 优势 | 劣势 |
|------|------|
| 操作一致性 100%, 调度效率 98% | 依赖手工构建的场景描述 |
| 任务越复杂优势越明显 | LLM 在极端复杂场景仍会 hallucinate |

## 技术价值与演进定位

IMR-LLM 代表了"LLM + formal methods"的最佳实践——不要妄想 LLM 解决一切，把保证正确性的部分交给数学。

## 与其他论文的关系

- **Code as Policies (2023)** — LLM 生成代码, IMR-LLM 加入 OR 约束
- **VoxPoser (2023)** — LLM 代码+3D value maps, IMR-LLM 面向工业而非家庭

## 精读问题

1. 工序分解 hallucination 时求解器能否检测并告警？
2. Process tree 在产线布局变化时如何维护？
