# 2026 VLA / Physical AI Research Brief

更新时间：2026-05-10

## 结论

2026 年值得优先关注的主线不是单纯“更大的 VLA”，而是三条路线的分化：

1. **公开论文路线**：以 `π0.7` 为代表，强调 steerable generalist robotic foundation model、组合泛化和跨 embodiment 泛化。
2. **商业闭源 mastery 路线**：以 `GEN-1` 为代表，强调成功率、速度、少量机器人适配数据和真实部署阈值，但缺少公开论文细节。
3. **full-stack dexterous manipulation 路线**：以 `GENE-26.5` 为代表，把模型、机器人手、触觉数据、仿真和低延迟控制栈一起设计。

## 1. π0.7

- 状态：有公开 arXiv 论文。
- 机构：Physical Intelligence。
- 重点：steerable VLA、emergent compositional generalization、cross-embodiment generalization。
- 核心架构变化：从单一语言指令条件，扩展为语言 + 策略元数据 + 子目标图像等多模态上下文条件。
- 为什么重要：它把 VLA 的目标从“跟随已见指令”推进到“通过上下文 steering 执行未见组合任务”。
- 推荐优先级：最高。

来源：

- https://arxiv.org/abs/2604.15483
- https://www.pi.website/blog/pi07

## 2. GEN-1

- 状态：未找到公开 arXiv 论文。
- 机构：Generalist AI。
- 重点：physical task mastery、99% success claim、约 3x speed、约 1 小时 robot-specific data。
- 核心架构线索：大型多模态 embodied model，实时动作输出；大规模 human wearable data；强调物理 commonsense 和即时纠偏。
- 重要变化：把研究目标从泛化 benchmark 推到“是否可商业部署”的成功率/速度阈值。
- 风险：缺少论文、代码、开放评测细节，必须作为产业观察而非可复现实证结论。

来源：

- https://generalistai.com/blog/apr-02-2026-GEN-1
- https://www.humanoidsdaily.com/news/generalist-ai-unveils-gen-1-the-quest-for-robot-mastery-and-intelligent-improvisation

## 3. GENE-26.5

- 状态：未找到公开 arXiv 论文，但官方博客细节较多。
- 机构：Genesis AI。
- 重点：human-level manipulation、full-stack robotics、human-like dexterous hand、tactile/human data、high-fidelity simulation。
- 核心架构线索：语言、视觉、本体感知、触觉、动作的联合轨迹建模；官方明确提到 flow matching。
- 重要变化：把 VLA/robot foundation model 的竞争点从“模型结构”扩展到“硬件-数据-仿真-控制-模型”的整体闭环。
- 风险：演示任务多，但缺少开放 benchmark 和论文级可复现细节。

来源：

- https://www.genesis.ai/blog/gene-26-5-advancing-robotic-manipulation-to-human-level
- https://www.genesis.ai/press/press-release-gene-265

## 对当前学习路线的影响

建议把 2026 新研究加入现有路线的最后一段：

1. 先读 `Flow Matching`，理解连续生成基础。
2. 再读 `FAST Tokenizer`，理解动作 token 化的新方向。
3. 然后读 `π0.7`，这是当前最值得精读的公开 2026 论文。
4. 最后读 `GEN-1` 和 `GENE-26.5` 的技术观察笔记，重点不是公式，而是看产业界如何定义数据、硬件、控制栈和部署指标。

## 需要持续追踪的问题

- GEN-1 是否会释放论文、技术报告或评测协议？
- GENE-26.5 是否会公布模型结构、数据规模、仿真-真实相关性细节？
- π0.7 的 steerability 是否能在开源模型中复现？
- 2026 的主线会偏向“更大模型”，还是“更完整数据/控制/硬件闭环”？

