# LLM 多智能体协作机制综述（大纲卡）

- arXiv: https://arxiv.org/abs/2501.06322
- Source: https://arxiv.org/abs/2501.06322
- 本地 PDF：`papers/briefs/MAS-Collab-Survey_2501.06322.pdf`
- Year: 2025 (Tran et al., 820+ 引用——该方向引用最高的综述)
- Category: briefs / survey outline
- Priority: medium
- 拉取日期：2026-10-10（多智能体方向开卷卡，只做大纲不入正文赛道）

## 一句话总结

按"协作机制"而非"应用"组织的 LLM-MAS 综述：把通信（何时/向谁/说什么）、决策（独立/集中/分散怎么合成行动）、任务分解（显式 plan vs 涌现分工）三维展开，覆盖辩论、角色扮演、拓扑结构（pipeline/star/hierarchy/graph）四类机制；结论务实——多智能体不是银弹，有效性依赖任务-机制匹配，且强调评估标准化的缺失。作为本库多智能体方向的地图使用：正文赛道各篇（maporl/dr-mas/goa/more-agents）分别落在它的"决策-学习"与"通信-结构"象限。

## 技术要点（机制分类骨架）

- **通信机制**：辩论（Debate 系：多轮互评收敛）、角色扮演（角色 prompt 分工）、共享记忆/黑板、拓扑结构（pipeline/星型/层级/图——GoA 是图拓扑的后续进化）。
- **决策机制**：独立决策+投票（More Agents 的理论化母体）、集中编排（orchestrator/planner）、分散协商（去中心化共识）。
- **协作拓扑什么时候重要**：论文给出任务特征 × 机制匹配的讨论——需要互补信息/分工的任务上结构化协作有效，独立可解任务上简单投票已够。
- **评估与失败模式**：MAS 评估基准碎片化；失败模式（级联错误、幻觉共识、过度自信 agent 主导）分类——后来的 GoA（judge 偏差进图）与 MAPoRL（SFT 学不会协作）都在具体化这些担忧。
- **训练侧空白（写作时点 2025-01）**：综述明确指出协作 LLM 的训练（而非 prompt）是开放问题——MAPoRL（2025-02）与 Dr. MAS（2026-02）正是这个空白的填补记录。

## 与本库的关系

- 为 notes/rl/agentic-training/maporl.md、dr-mas.md（训练线）与 notes/rl/agentic-algo/goa.md、more-agents.md（推理时线）提供分类坐标系：训练线 = 综述指出的空白方向；推理时线 = 其通信/拓扑机制章节的深化。
- arlarena / ragen-2：综述的"失败模式"章节与这两篇的实证稳定性研究互补。

## 备注

- 大纲卡规格：不进正文赛道、不进精读队列；引用它定位新论文时说"综述 §X 机制"即可。
- 2025-01 截稿，未覆盖 2025 下半年后的 RL 共训（MAPoRL 后续）、agent 编排框架（Dr. MAS 框架）与 2026 的图化协作（GoA）——用本库四篇正文补齐。
