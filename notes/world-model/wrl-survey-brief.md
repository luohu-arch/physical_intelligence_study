# World Model for Robot Learning: A Comprehensive Survey（综述大纲卡）

- 本地 PDF：`papers/world-model/WRL-Survey_2605.00080.pdf`
- arXiv：https://arxiv.org/abs/2605.00080
- 年份：2026（5 月）
- 团队：NTU 领衔，联合 UC Berkeley / Stanford / 东京大学 / Oxford / Microsoft / ETH Zurich / Princeton / Harvard
- 类型：综述大纲卡（当索引用，不写深度笔记）

## 这篇综述讲什么

机器人学习视角的世界模型全景综述。定义世界模型为「描述环境在智能体动作下如何演化的预测结构」，强调价值在支撑策略学习/规划而非视觉保真度。三大应用角色：**World Model for Policy**（按耦合方式分五类范式）、**World Model as Simulator**（RL 与评估）、**World Model for Robotic Video Generation**（从想象力生成到可控/结构化/foundation 级四段演进），再外延导航与自动驾驶。配三级评估框架与 LIBERO/RoboTwin/CALVIN/SIMPLER 代表结果表。43 页。

## 章节大纲

1. **Introduction**（p1）
2. **Background**（p4）— World Model vs Video Generation Model 概念切分；Robot Policy 的两代形态：Visuomotor Policy 与 Vision-Language-Action Policy。
3. **World Model for Policy**（p7）— 核心章，五类耦合范式：IDM-style 逆动力学策略（视频生成先产 plan 再逆出动作）；Unified Policies with a Single World-Model Backbone；MoE/MoT-Style Policies with Expert World-Model Backbones；Unified Vision-Language-Action Models；Policies with Latent-Space World Modeling（把 JEPA 式表征空间预测内化进 VLA）。
4. **World Model as Simulator**（p15）— World Model for Reinforcement Learning；World Model for Evaluation。
5. **World Model for Robotic Video Generation**（p18）— 四段技术演进：Video Generation as Imagination for Policy Learning → Toward Action-Controllable Video World Models → Structure-Aware Generation with Interaction and Geometry Priors → From Video Backbones to Foundation World Models；末尾 Technical Progression and Open Challenges。
6. **World Model for Other Applications**（p23）— Navigation；Autonomous Driving。
7. **Benchmarks, Datasets, and Results**（p24）— 三级评估：action-conditioned generation/open-loop predictive quality、closed-loop task utility、physical consistency/controllability/executability 诊断；数据集资源盘点；LIBERO 4-suite 与 RoboTwin/CALVIN/SIMPLER 代表成绩。
8. **Challenges and Future Directions**（p30）— Causal Conditioning Gaps、Efficiency Bottlenecks、Multi-Modal Perception Bottlenecks、Classical Control Integration、Symbolic Structure Integration、Open Challenges in Evaluation Metrics。

## 值得查的表/图

- **Table 1（p8）世界模型策略的架构范式对比** — 本综述核心索引：Paradigm / Representative Work / Future Generation at Inference / Backbone（VGM vs UMM vs MLLM）/ Coupling Style，逐行列出全部五类范式代表方法。
- **Figure 6 + Table 2（pp.19–20）robotic video world model 统一视图**：四能力档位分组对比 §5 全部方法。
- **Tables 3–4（pp.26–27）数据集双表**：Table 3 给核心属性，Table 4 给「每个数据集对哪些具身世界建模能力有贡献」的相关性矩阵——设计预训练配比时查这个。
- **Figure 2（p3）时间线图**：world-model-based policy 方法的演进树（解耦 → 统一骨干 → MoE/MoT …）。
- **Figure 4（p13）两条 MLLM 内化路线**：Unified VLA 与 latent 内化的对照示意。
- **Tables 5–6（pp.28–29）代表性结果**：按世界模型整合方式分组的 LIBERO 成绩表与 RoboTwin/CALVIN/SIMPLER 成绩表——查某范式在标准 benchmark 上什么水平直接看这两张。
- **Figure 3（p10）五种架构范式示意图**（IDM-style 等）。

## 与本库的关系

- **UniPi**：§3 + Table 1 **IDM-style Decoupled** 范式的首行代表（explicit video rollout，VGM backbone）。
- **LingBot-VA**：§3 + Table 1 归入 **MoE/MoT fusion** 范式（visual predictive context）；亦出现在结果表中。
- **DreamZero**：§3 + Table 1 归入 **Single-backbone Shared backbone** 范式（chunk-wise joint rollout）；并列入 LIBERO 结果表对照。
- **WorldVLA**：§3 + Table 1 归入 **Unified VLA** 范式（future image prediction mainly train-time，joint co-training）；LIBERO 结果表中该组的一员。
- **WoW**：§5 Robotic Video Generation 一带反复讨论（Table 2 中标注其能力档位），属 foundation world model 阶段的讨论对象。
- **TD-MPC2 / LeWorldModel**：§5 开头作为 latent-space MPC 增强长时程推理的论据被引（§4 未以其为主线）。
- **Dreamer 系列**：在 §5 robotic video generation 的方法表与正文出现（作为 generation-as-simulator 讨论的对照系），不是本综述 RL 章的主角。
- **V-JEPA**：§3.5 Latent-Space World Modeling 的概念锚点——明确说 JEPA 家族是表征空间预测原理的来源，该节目标是把它变成可用的策略学习机制；本库 SD-JEPA 类工作若写 related work 可引此节定位。
- 本库其余（I-JEPA、SuSIE、SimDist、TacWAM、SD-JEPA、WorldArena 论文本体、WCM、EgoGenesis、NoGaussianRequired）未被检索到。查「我的方法在世界模型-策略耦合谱系里属于哪类」用这篇最直接。
