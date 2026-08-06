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

**1. 析取图调度模型**：多机器人产线调度被形式化为最小化完工时间（makespan）的组合优化问题

$$
\min \; C_{max} \quad \text{s.t.} \quad C_{j,k} \ge C_{i,l} + p_{j,k} \; (\forall \text{ 工序优先约束}), \quad \text{且同一机器人上的工序互斥}
$$

节点表示工序，实线边（conjunctive）编码工序依赖，虚线边（disjunctive）编码同一机器人上工序的互斥顺序选择。LLM 负责从自然语言中抽取这两类约束（分解 + 机器人分派），确定性 OR 求解器在图上求全局最优调度，数学上保证无死锁。

**2. Process Tree 程序生成**：调度方案确定后，LLM 不从头写代码，而是沿操作过程树（operation process tree）选择已定义的操作路径并组装成可执行 Python 程序，配合符号状态检查（工件位置 + 已执行操作）验证程序正确性。成功判定 SR = SE ∧ GCR（调度正确且全局约束满足）。

## 物理直觉解释

**不要用 LLM 直接写工厂调度代码——LLM 会"幻觉"出逻辑冲突导致机器人死锁**。想象两条机械臂要交替完成"打磨→焊接→转运"：LLM 逐句生成动作代码时，很可能让两台机器人同时抢同一块工件、或让机器人 B 等一个永远不会发生的信号——这正是纯 LLM 方案在复杂多机器人任务上 SR 归零（w/order 消融 0.00）的原因。IMR-LLM 的解法是**职责分离**：让 LLM 做它最擅长的事（把自然语言任务翻译成结构化的依赖图），把"怎么最优排程"交给运筹学——调度是数学上能证明最优的，而翻译错误至少可以被求解器暴露出来。

**析取图像一张"工序互斥表"**：每个机器人在同一时刻只能干一件事（disjunctive 边），工序之间有先后依赖（conjunctive 边）。LLM 负责把"先把船体打磨完，再交给焊接机器人"这类句子翻译成图的边；求解器在图上搜索使总工期最短且不冲突的排程。这就像交警不指挥每辆车的每一个动作，而是拿到各路线的时间表后统一调度——全局最优与无死锁由算法保证，而不是靠 LLM 的"临场发挥"。

**Process Tree 代码生成像"填空题而非作文题"**：LLM 从零生成代码是作文（自由发挥、容易跑题）；沿过程树选择操作路径是填空（选项给定、只需选择组装），实测在 Exe 指标上比从零生成更准，且符号状态检查（工件位置 + 已执行工序）能验证程序真的按计划执行完。

- 输入: 自然语言任务描述 + 产线配置
- 输出: 多机器人调度方案 + 可执行 Python 代码
- 场景: 造船和重型装备制造
- 实际部署: 3 机器人产线，含视觉定位、抓取、协作运输

## 消融实验与分析

IMR-Bench 上按任务类别分组的 SR（成功完成全部工序且全局约束满足）：

| 消融 | Single Robot SR | Simple Multi-Robot SR | Complex Multi-Robot SR |
|------|----------------|----------------------|----------------------|
| Ours（完整，GPT-4o） | 0.90 | 0.87 | 0.68 |
| Ours（Qwen3-32B） | 1.00 | 0.87 | 0.60 |
| w/order（LLM 直接排执行顺序，无 OR 求解器） | 0.90 | 0.47 | 0.00 |
| w/dependency（无依赖约束建模） | 0.90 | 0.60 | 0.36 |
| w/o T（无 Process Tree 程序生成） | 0.80 | 0.64 | 0.44 |
| 基线 LiP-O / LaMMA-O / SMART-LLM | 0.90 / 0.80 / 0.50 | 0.73 / 0.46 / 0.20 | 0.24 / 0.20 / 0.00 |

**核心结论**：消融清晰显示 IMR-LLM 的优势随任务复杂度单调扩大——单机器人任务各方法几乎打平（SR 0.80-1.00），复杂多机器人任务上完整方法 0.68 而"LLM 直接排程"归零（0.00）、无依赖建模仅 0.36、无 Process Tree 0.44，说明析取图求解器与依赖约束建模是多机器人协同的核心保障；与基线对比（LiP-O 0.24、LaMMA-O 0.20、SMART-LLM 0.00）进一步印证纯 LLM 或简化形式化方法都无法应对工业级多机器人调度。


## 工程细节与实操指南

- **Input**: 自然语言任务描述 + 产线配置（机器人数量/类型/工作空间）
- **LLM**: Task decomposition → operation → robot assignment → disjunctive graph
- **OR Solver**: Johnson algorithm / genetic algorithm on disjunctive graph → deadlock-free schedule (100% consistency, 98% efficiency)
- **Code Gen**: Process tree navigation (not open-ended generation) → executable Python code (87-90% success)
- **Benchmark**: IMR-Bench — 23 real industrial scenes, 50 tasks, up to 7 robots × 24 operations
- **Real deployment**: 3-robot production line with visual positioning, grasping, collaborative transport
- **Speedup**: Manual programming hours → minutes

## 技术权衡

| 优势 | 劣势 |
|------|------|
| 复杂多机器人任务 SR 0.68，远超纯 LLM 基线（0.00-0.24） | 依赖手工构建的场景描述与产线配置 |
| 任务越复杂优势越明显（单机器人 0.90 → 复杂多机器人 0.68 vs 基线崩溃） | LLM 在极端复杂场景仍会 hallucinate，需求解器兜底 |
| 调度正确性由 OR 求解器保证（无死锁、全局最优） | Process Tree 的路径库需要随产线工艺维护 |

## 技术价值与演进定位

IMR-LLM 代表了"LLM + formal methods"的最佳实践——不要妄想 LLM 解决一切，把保证正确性的部分交给数学。它对工业机器人编程的价值是范式级的：把"工艺专家写调度 + 程序员写代码"的数天流程压缩为"自然语言描述 → 分钟级自动编程"，同时以析取图 + 符号状态检查维持正确性保证。对机器人领域而言，它与 Code as Policies / VoxPoser 一脉相承（LLM 生成机器人程序），但首次把"LLM 不可靠"这一事实显式纳入设计——LLM 只产结构、数学保证最优，这一"翻译器 + 求解器"的分工模式可推广到任意需要 LLM 与确定性算法协作的自动化场景。

## 与其他论文的关系

- **Code as Policies (2023)** — LLM 生成机器人代码的鼻祖：在家庭场景直接生成 Python 策略代码，IMR-LLM 将其扩展到工业多机器人，并加入 OR 求解器消除 LLM 的逻辑幻觉
- **VoxPoser (2023)** — LLM 代码 + 3D value maps 的交互式操作方案：面向家庭单臂，IMR-LLM 面向造船/重装产线的多机器人协同
- **SMART-LLM / LaMMA / LiP** — 同任务分解路线的直接基线：SMART-LLM 复杂多机器人 SR 0.00，LaMMA-O 0.20，LiP-O 0.24，均远低于 IMR-LLM 的 0.68
- **COHERENT** — 单一调用完成全部子问题的集中式规划：与 IMR-LLM 的显式分解 + 求解器路线形成对照，后者在大规模任务上更稳

## 精读问题

1. 工序分解 hallucination 时求解器能否检测并告警？析取图约束冲突如何自动暴露 LLM 翻译错误？
2. Process tree 在产线布局变化时如何维护？路径库的更新成本与自动生成方案？
3. 复杂度拐点在哪——任务规模超过多少工序/机器人时，纯 LLM 方案必然崩溃而 IMR-LLM 仍保持 SR 稳定？
4. LLM 翻译成析取图时的错误类型分布（漏边/多边/错边）？求解器对翻译错误的鲁棒性边界？
5. 符号状态检查（SE）与真实物理执行的差距——程序在仿真中正确是否保证产线上无碰撞？
6. 若把 OR 求解器替换为学习型调度（GNN + 启发式），能否在大规模场景超越确定性最优？
