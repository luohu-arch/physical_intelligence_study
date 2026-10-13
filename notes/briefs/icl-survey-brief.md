# In-Context Learning 综述双卡（大纲卡）

- arXiv: https://arxiv.org/abs/2301.00234（总纲）；https://arxiv.org/abs/2311.00237（机制篇）
- 本地 PDF：`papers/briefs/ICL-Survey_2301.00234.pdf`；`papers/briefs/ICL-Mystery-Survey_2311.00237.pdf`
- Year: 2023→2024（双双 EMNLP 2024 正会；前者 ~4000 引用是该方向最高）
- Category: briefs / survey outline
- Priority: medium
- 拉取日期：2026-10-10（ICL 方向开卷卡，只做大纲不入正文赛道）

## 一句话总结

两篇互补的 ICL 综述配对入库：**Dong et al.《A Survey on In-Context Learning》**（总纲，复旦/RUC 系）按"形式化—训练策略—演示设计"组织工程全貌（预训练如何涌现 ICL、示范选择/顺序/格式的影响、ICL 理论三派：贝叶斯/梯度下降/注意力路由）；**Zhou et al.《The Mystery of In-Context Learning》**（机制篇，利兹/杜伦系）专注解释与分析——机制可解释性（induction heads、task vectors）、数学基础、影响因素实证，外加风险面（真实性/偏见/毒性）。作为本库 ICL 线的坐标系：四篇正文笔记（icl-what-algorithm/icl-task-vectors/many-shot-icl/algorithm-distillation）分别落在机制篇的"GD 构造/表征定位/规模律/RL 扩展"四个小节。

## 技术要点（两张卡的分工）

**总纲卡（Dong 2301.00234）**：
- ICL 形式化：上下文示例→预测器的映射，无参数更新；能力来源的三个假说族（预训练任务分布覆盖、隐式贝叶斯推断、隐式优化）。
- 训练侧：什么预训练数据/目标催生 ICL（元学习式混合任务 vs 纯语言建模）；指令微调对 ICL 的放大。
- 演示工程：选择（相似度/多样性）、顺序（近期发现的顺序敏感性）、格式（CoT/模板）——few-shot 时代的手艺清单。
- 读者价值：工程旋钮全集；many-shot 之后"顺序钝感"等结论正在改写它的部分章节。

**机制卡（Zhou 2311.00237）**：
- 机制可解释性：induction heads（前缀匹配+复制电路）、task vectors/function vectors（演示压缩为向量）。
- 数学基础：贝叶斯推断视角（Xie et al.）、GD 等价视角（Akyürek/Dai/von Oswald）、表达力上界（Garg 等）——本库 icl-what-algorithm 的直接上游。
- 实证因素：示例数量/顺序/格式/模型规模的效应清单。
- 风险与安全：ICL 的偏见放大、毒性注入、truthfulness——many-shot 时代被放大的面。

## 与本库的关系

- 四篇 ICL 正文（icl-what-algorithm、icl-task-vectors、many-shot-icl、algorithm-distillation）+ 机器人锚点 icrt 构成正文线；本双卡提供分类坐标与引用入口。
- statelinformation / wam-ttt / robottt（TTT vs ICL 之争）在机制卡的"数学基础"小节有理论对照。

## 备注

- 大纲卡规格：不进正文赛道、不进精读队列；引用新论文时按"综述 §章节"定位。
- Dong 卡更新到 v6（2024-10），Zhou 卡 v3（2024-10）——都早于 many-shot 之后的长上下文 ICL 浪潮与 2025-26 的具身 ICL，缺口由本库正文笔记补。
