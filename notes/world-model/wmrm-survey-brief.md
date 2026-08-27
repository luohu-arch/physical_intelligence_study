# World Models for Robotic Manipulation: A Survey（综述大纲卡）

- 本地 PDF：`papers/world-model/WMRM-Survey_2606.00113.pdf`
- arXiv：https://arxiv.org/abs/2606.00113
- 年份：2026（6 月）
- 团队：香港理工大学机械工程系牵头，联合 HIT（深圳）/ Great Bay University / CityU HK / 港大 / NTU / KTH 等
- 类型：综述大纲卡（当索引用，不写深度笔记）

## 这篇综述讲什么

专做「机器人操作」的世界模型综述。给出操作性定义——世界模型是**动作条件预测系统** p(未来 | 当前观测, 动作)，并明确它不是感知模块、逆模型、策略或奖励函数。两条主分类轴：按**表示**分 5 个家族（按结构归纳偏置递增排序），按**预测-动作接口**分 Integrated 与 Explicit Planner 两类；再把世界模型重新定位为跨学习生命周期的基础设施（合成经验、候选过滤、搜索评估、学习环境、结果验证）。系统整理了 34 个操作数据集与三档评估指标，25 页。

## 章节大纲

1. **Introduction**（p1）
2. **Definition and Scope of World Models**（p2）— 从 forward models 到 model-based learning；当代具身 AI 中的世界模型；面向操作的可操作定义。
3. **Representations**（p4）— 五个表示家族：Image/Video、Learned Latent、Motion Fields & Scene Flow、Geometric & Spatiotemporal、Physics-Informed Dynamics；末尾给「如何选表示」的讨论。
4. **A Functional Taxonomy by Prediction–Action Interface**（p7）— Integrated Prediction–Action Models（预测与动作一个系统内完成）vs Explicit Predictive Planners（WM 只负责 rollout，策略在外部规划）；各自的设计取舍。
5. **World Models as Learning and Decision Infrastructure**（p9）— 五种基础设施角色：Synthetic Experience Generation / Candidate-Action Filtering & Refinement / Search-Based Action Evaluation / Learned Environments for Policy Evaluation & Improvement / Outcome Scoring & Feasibility Verification。
6. **World Models Across the Learning Lifecycle**（p12）— Pretraining（可复用的 latent/video/3D prior）、Post-Training（合成数据、world 内 RL、奖励与偏好打分、幻觉过滤；核心风险 simulator exploitation）、Inference Adaptation（搜索、candidate rerank、TTT、记忆更新、自纠错）。
7. **Datasets for World-Model Learning**（p15）— 按用途递进：Video Prediction Pioneers → Task-Centric Simulation Benchmarks → Demonstration Collection & IL → Large-Scale Real-Robot Pretraining Corpora → Multimodal/Contact-Rich Data → Autonomous Data Paradigms。
8. **Benchmarks and Evaluation**（p18）— Predictive Fidelity Metrics；Downstream Task/System Benchmarks；Infrastructure & Simulator Reliability；基准测试的开放问题。
9. **Conclusion**（p21）

## 值得查的表/图

- **TABLE I（p13）全方法分类总表** — 这张表是本综述的核心索引：每行给 Functional role（Predictive prior / Integrated / Planner / Infrastructure）、Context（RL/IL/VLA）、Representation、Predicted signal、Predictive function、Lifecycle stage。查某个方法被归为什么角色直接翻这张表。
- **TABLE II（p16）34 个操作数据集** + **Figure 6（p17）时间线图**：按功能角色分类，横轴为首次公开年份。
- **Fig. 2（p5）表示谱系图**：五个表示家族按结构化 inductive bias 排序，一眼定位某方法是重建式还是纯预测式。
- **Fig. 4（p10）五种基础设施角色示意图**：对应 §5 的分类框架。
- **Fig. 5（p14）生命周期图**：Pretraining → Post-Training → Inference 三阶段各用什么机制。
- **TABLE III（p19）评估指标汇总**：predictive fidelity 指标逐项列出。

## 与本库的关系

- **Dreamer v3**：TABLE I 归入 **Integrated** 类（latent rollout，imagination-based policy learning，Train 阶段）；正文 §Integrated Prediction–Action Models 引 Dreamer 系列为 latent rollout 训练范式的来源。
- **DayDreamer**：§4 同段，作为「Dreamer 式 latent rollout 可迁移到真机」的证据方法。
- **TD-MPC2**：§Explicit Predictive Planners，作为 latent MPC 规划路线代表（与 MoDem 一线并列）。
- **V-JEPA 2**：TABLE I 归入 **Predictive prior** 类（Latent 表示，future latent state，Predictive prior + planning，Pre/Infer 阶段）。
- **UniPi**：TABLE I 归入 **Planner** 类（Video 表示，future video trajectory，trajectory planning，Infer 阶段）。
- **SuSIE**：TABLE I 归入 **Planner** 类（Image 表示，subgoal image，subgoal planning，Infer 阶段）；正文 §8 提及其用 image-editing 扩散生成子目标图。
- **GR-MG**：正文 §Explicit Predictive Planners 段提一笔（与 SuSIE 同一 image-level 子目标接口），未进 TABLE I 主表。
- **WorldVLA**：TABLE I 归入 **Integrated** 类（Joint world action model，Train 阶段）。
- 本库其余 world-model 笔记（I-JEPA、DreamZero、LingBot-VA、SimDist、SD-JEPA、TacWAM、WoW、WorldArena 等）未被检索到。写操作方向的世界模型论文时，引用这篇来确定「方法属于哪类基础设施角色」最合适。
