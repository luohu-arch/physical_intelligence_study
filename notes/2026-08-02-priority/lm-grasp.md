# LM-GRASP: Instance-Specific Language Models for Combinatorial Construction via Online Imitation Learning

- arXiv: https://arxiv.org/abs/2607.28135
- Source: https://arxiv.org/abs/2607.28135
- Project:
- Local PDF: `/Users/luogu/physical_intelligence/papers/2026-08-02-priority/lm-grasp-instance-specific-language-models-for-combinatorial-construction-via-on_2607.28135.pdf`
- Year: 2026
- Category: language-conditioned manipulation / online imitation
- Priority: medium

## 一句话总结

LM-GRASP 把组合构造问题中的 grasp/action selection 表述成 instance-specific language model 问题，并通过 online imitation learning 改进策略，目标是在复杂组合任务中比传统 GRASP 构造启发式更灵活。

## 核心技术

1. **Instance-specific language model**：不是训练一个全局策略解决所有实例，而是针对具体 construction instance 建立语言/决策模型。
2. **Online imitation learning**：通过在线模仿专家或改进决策，逐步提升构造动作选择。
3. **Combinatorial construction framing**：任务不只是单步 grasp，而是组合决策序列。
4. **Classical GRASP comparison**：论文讨论 classical Greedy Randomized Adaptive Search Procedure 的限制，并用 LM-GRASP 替代或增强。

## 底层原理与数学推导

组合构造可视为序列决策：

$$
\tau = (a_1, a_2, ..., a_T), \quad a_t \in A(s_t)
$$

目标是最大化构造质量或任务 reward：

$$
\max_{\pi} \mathbb{E}_{a_t \sim \pi(\cdot|s_t, I)} [R(\tau)]
$$

LM-GRASP 的语言模型承担的是 action ranking / action proposal 功能；online imitation 则把专家轨迹或改进动作反馈加入训练，减少纯启发式构造在复杂实例上的失败。

```mermaid
flowchart LR
    INST[Problem instance] --> LM[Instance-specific LM]
    STATE[Current construction state] --> LM
    LM --> RANK[Rank candidate actions]
    EXP[Online expert feedback] --> UPDATE[Imitation update]
    RANK --> EXEC[Execute construction step]
    EXEC --> STATE
    UPDATE --> LM
```

## 物理直觉解释

组合构造像搭积木或装配：每一步看似局部，但会影响后续可行性。传统启发式容易陷入局部贪心。语言模型的作用不是“聊天”，而是作为一种可学习的启发式，把当前实例结构、历史状态和候选动作联系起来。

## 工程细节与实操指南

- 阅读时要区分 robotics grasp 和 optimization GRASP，标题中 GRASP 可能同时有双关含义。
- 如果迁移到机械臂任务，需要把离散构造动作映射成可执行 grasp/place primitive。
- Online imitation 的数据效率和专家来源是复现关键。
- 适合用于 construction、assembly、packing 等 long-horizon manipulation。

## 消融实验与分析

重点看：

- instance-specific LM 是否优于全局模型或传统 GRASP。
- online imitation 是否比 offline imitation 更稳。
- 长程组合任务中，早期错误是否能被后续恢复。
- 语言模型输出是否可解释，还是只是黑箱 action scorer。

## 技术权衡（Trade-off）

| 优势 | 劣势与工程代价 |
|------|---------------|
| 适合组合构造类长程任务 | 和真实机器人低层控制仍有距离 |
| instance-specific 模型可利用任务结构 | 每个实例适配可能增加计算成本 |
| online imitation 可逐步改进 | 需要专家或高质量反馈 |
| 可与 language-conditioned manipulation 结合 | 物理执行误差未必在模型内处理 |

## 技术价值与演进定位

LM-GRASP 的价值在于拓宽 VLA 之外的 language-conditioned manipulation 视角：语言模型可以做任务实例上的动作排序/启发式学习，而不一定直接输出连续机器人动作。

## 与其他论文的关系

- 和 RoboBRIDGE：LM-GRASP 可作为高层 planner/skill selector，RoboBRIDGE 负责接到执行系统。
- 和 SemAnCorr：SemAnCorr 解决物体局部 frame，LM-GRASP 解决组合动作选择。
- 和 VoxPoser：都把 LLM 用作结构化决策，而不是直接端到端控制。

## 精读问题

1. 这里的 language model 输入输出具体是什么？
2. online imitation 的 expert 来自算法、人工，还是 oracle？
3. 如果部署到机械臂，动作 primitive 如何定义？
4. 是否能用于装配/packing 这类真实 manipulation 任务？
