# World Action Models: The Next Frontier in Embodied AI（综述大纲卡）

- 本地 PDF：`papers/briefs/WAM-Survey_2605.12090.pdf`
- arXiv：https://arxiv.org/abs/2605.12090
- 年份：2026（5 月）
- 团队：复旦大学（Fudan University）/ Shanghai Innovation Institute / National University of Singapore
- 类型：综述大纲卡（当索引用，不写深度笔记）

## 这篇综述讲什么

首个系统性 WAM 综述：定义 World Action Model 为「统一预测状态建模与动作生成、建模未来状态与动作联合分布」的具身基础模型，并与 VLA（只做观测→动作映射）、纯世界模型（只预测未来）做概念切割。核心分类轴是架构二分法 **Cascaded WAM**（世界模型先产未来状态/视觉计划，动作模型再解码）vs **Joint WAM**（单一系统联合预测），并沿数据生态（遥操作 / UMI / 仿真 / 人类 egocentric 视频）和双轨评估（世界建模能力 + 动作策略能力）展开。69 页。

## 章节大纲

1. **Introduction**（p1）— VLA 的反应式局限 → 引出「把世界模型整合进动作生成」的 WAM 范式。
2. **Definitions and Formalism**（p3）— Foundational Paradigms 基础范式；WAM vs 相关概念的消歧（VLA / 世界模型 / 视频生成模型）。
3. **VLAs and World Models: Foundations and Early Integration**（p6）— 三条谱系溯源：Action-conditioned WM（RSSM→Dreamer→TransDreamer TSSM）；Language-conditioned WM；Embodied WM（从纯视频学动作语义，Genie latent action、SWIM 等）。再加 World Model for VLA 的两种用法：for Learning（辅助模仿/RL 训练）与 for Evaluation。
4. **Architecture**（p13）— 核心章。Cascaded WAM：Explicit Planning via Pixel-Space Representations（UniPi 起点的两阶段蓝图，动作经 learned inverse dynamics 或几何闭式解提取）与 Implicit Planning via Latent Representations；Joint WAM：Autoregressive Generation（GR-1/GR-2、GR-MG、WorldVLA、CoT-VLA 等自回归 token 流）与 Diffusion-based Generation（多种耦合模式，含 Unified Stream 等）。
5. **Training data**（p26）— 四类数据生态：Robot-Centric Teleoperation Data（放大规模 + 多模态接触具身化）；Portable Human Demonstration Data（UMI-style）；Simulation Data（程序化生成、3D/4D 具身建模、触觉物理）；Human and Ego-Centric Data（被动观察的动作语义、姿态估计桥接动作缺口、通用预训练配比）。
6. **Evaluation**（p34）— 双轨。How to Evaluate World Modeling Capability：Visual Fidelity / Physical Commonsense / Action Plausibility；How to Evaluate Action Policy：通用操作、双臂+人形、移动操作、接触与形变操作、真机五类 benchmark。
7. **Open challenges and Opportunities**（p41）
8. **Conclusions**（p44）

## 值得查的表/图

- **Figure 2（p4）总路线图/taxonomy**：全领域方法一棵树，按 Cascaded/Joint 分支标注几乎所有 2024–2026 代表作（GR-1/GR-2、WorldVLA、CoT-VLA、UWM、FLARE、LingBot-VA、DreamZero、Cosmos Policy、Motus、DUST、AdaWorldPolicy…）；Figure 1 是其时间线版本。
- **Table 1（p16）Cascaded WAM 方法对比**：是否需要动作标注、视觉计划形式、动作提取方式等维度对比级联式方法。
- **Table 2（p19）/ Table 3（p25）Joint WAM 方法对比**：按自回归与扩散两族，逐行给参数量、backbone、输入输出模态、评测环境——查某方法属于哪支最快。
- **Tables 4–7（pp.28–33）四类数据集汇总**：robot-centric 遥操作 / UMI / 仿真 / 人类 egocentric，统一 Modality 列——写数据相关论文时的数据集索引。
- **Table 8（p36）+ Table 9（p40）评估汇总**：Table 8 是世界建模能力的指标与 benchmark；Table 9 是动作策略 benchmark 大表（对象数/任务轨迹数/观测模态/机器人/simulator/Eval Focus 标签）。

## 与本库的关系

- **Dreamer 系列 / DayDreamer / RISE**：归入 §3 谱系溯源的 Action-conditioned World Models 一线（RSSM 家族，Dreamer 系列为规划范式主干）；RISE 出现在 Figure 1/2 时间线的 RL 分支。以后写 model-based RL 世界模型对照 WAM 时引这部分。
- **LeWorldModel**：§3 JEPA 谱系段被引为「end-to-end JEPA + SIGReg 单正则替代多项目标」的代表。
- **WoW**：§3 Embodied World Model 段，作为「VLM critic 评估生成视频并回炉 prompt 的自优化闭环 SOPHIA」被引。
- **UniPi**：§4 Cascaded / Pixel-Space Explicit Planning 的起点方法（two-stage blueprint 出处）。
- **GR-MG**：§4 Joint / Autoregressive Generation，Table 2 中 GPT-style + InstructPix2Pix 目标图路线。
- **WorldVLA**：§4 Joint / Autoregressive Generation，Table 2 中 Chameleon-based MLLM（7B）联合预测 future VQ tokens 与离散动作。
- **DreamZero / LingBot-VA**：出现在 Figure 1/2 总时间线及 §4 扩散式 Joint WAM 一带。
- 本库其余 world-model 笔记（I-JEPA、V-JEPA 单独论、TD-MPC2、SuSIE、SimDist、SD-JEPA、TacWAM、WCM、EgoGenesis、NoGaussianRequired、WorldArena 作为论文本体）未被收录或仅作背景引用；WorldArena 仅在 Table 3 各方法的评测环境列出现。
