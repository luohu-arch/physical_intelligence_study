# World-Action Models for Robot Learning and Control: A Survey（综述大纲卡）

- 本地 PDF：`papers/briefs/WAM-Survey2_2609.16074.pdf`
- arXiv：https://arxiv.org/abs/2609.16074
- 年份：2026（9 月，arXiv v1 2026-09-13）
- 团队：MBZUAI + Caltech + Amazon FAR + University of Virginia + Georgia Tech + NYU + UC Berkeley（Zuxing Lu / Hongjia Zhai 共一，Xingxing Zuo 通讯；IEEE 期刊格式，19 页）
- 项目页：https://rcl-robotics.github.io/Awesome-World-Action-Models
- 类型：综述大纲卡（当索引用，不写深度笔记）

## 这篇综述讲什么

机器人视角的 WAM 综述（与库内 5 月的复旦 WAM-Survey 卡互为补充）：把 World-Action Model 定义为「耦合未来世界预测与可执行动作生成」的预测控制模型——联合预测未来状态与动作，或先预测状态再经逆动力学接口反推动作（plan-then-act），并在 POMDP 形式化下与 WM（只学转移 $\hat z_{t+1}=f_{\text{WM}}(z_t,a_t)$）、VLA（只映射 $\hat a_{t:t+H}=f_{\text{VLA}}(o_{<t},\ell)$）做公式级切割。与复旦版以 Cascaded/Joint 架构二分为主轴不同，本篇用**五轴 taxonomy**（模型组件 / 转移建模范式 / 架构结构 / 训练流水线 / 数据模态与 scaling）组织方法，再按应用域（操作 / 导航 / 自动驾驶）和评估资源（数据集 / benchmark / 指标）两章展开。覆盖 268 篇引用，含 2026 年最新一代（Fast-WAM、GigaWorld-Policy、ImageWAM、Dyna-2、π0.6/π0.7 等）。挑战章落在动作对齐、世界-动作分解、空间多视角一致性、长程记忆、神经模拟器闭环学习、推理效率六点。

## 章节大纲

1. **Introduction**（p1）— 反应式 VLA 的局限 → WAM = 预测结构 + 动作生成；Fig. 1 给出从 latent WM/MBRL、VGM、LAPA、VLA 四线汇聚到 Unified WAM 的演化时间线。
2. **Background: From World Models to World-Action Models**（p3）— POMDP 形式化与 WM/VLA/WAM 三定义（Eq. 1–3）；四条基础研究线：WM 与 MBRL（PlaNet RSSM、显式/隐式两支）、视频生成模型（视觉预测规划器）、Latent Action Pre-training（LAPA，无动作视频学潜动作）、VLA（自回归 vs flow matching）；最后归纳 WAM 四组件（policy / forward dynamics / inverse dynamics / visual planning）与联合分布 vs plan-then-act 两种因子分解。
3. **Architecture Design & Model Taxonomy**（p4，核心章）— 五轴：**A. 模型组件**（语言编码器 CLIP/T5/LLM、视觉编码器 2D-Video VAE/表示编码器 DINO-SigLIP-VJEPA/3D 多视角/VLM、backbone 从零统一 Transformer vs 预训练视频生成模型、动作解码器 tokenizer/LAM/扩散-流头）；**B. 转移建模范式**（Joint Prediction：GR-1/GR-2→VPP/UWM→WorldVLA/Motus→ViPRA/Cosmos Policy vs Inverse Dynamics：Pathdreamer→Seer→DreamZero→隐式 IDM 的 LingBot-VA/Fast-WAM/GigaWorld-Policy，另有目标条件 IDM 一支 Act2Goal/π0.7）；**C. 架构结构**（End-to-end 单骨干 vs Dual System 双专家，后者经 MoT 路由或 cross-attention 两种接口，含 ImageWAM/Helix）；**D. 训练流水线**（预训练三件事：时空表示/动作表示/视频-动作对齐；后训练三件事：策略微调/数据增广/想象环境 RL，含 DreamGen、WorldArena 的动作接地轨迹合成）；**E. 数据模态与 scaling**（语言/多视角/3D 几何/接触-听觉/本体感觉五模态；互联网视频→egocentric 演示→具身轨迹的数据金字塔，Dyna-2 百万小时）。
4. **Applications**（p10）— 按域看预测的四种用途（表示学习 / look-ahead 规划 / 合成数据 / 想象策略优化）：**操作**（video-action 与统一策略、结构化状态模型 Gaussian/粒子/物理/视触听、想象式策略优化 RL 系）；**导航**（look-ahead 规划 Pathdreamer/DreamWalker/NWM、latent 规划与 WM-RL X-MOBILITY/WMP/RWM）；**自动驾驶**（可控视频生成与占据预测、统一驾驶 agent 与 WM-RL DreamerAD/World4Drive）。
5. **Datasets, Benchmarks, and Evaluation Metrics**（p11）— 指标家族总表（任务级 SR/SPL/DS 到生成级 FID/FVD/LPIPS，及「预测质量 ≠ 控制效用」的核心论点）；五类资源分节：机器人演示与人类视频语料（DROID/OXE/AgiBot vs Ego4D/EgoDex/SSv2）、仿真操作套件（LIBERO/CALVIN/RLBench/RoboCasa/RoboTwin…）、驾驶数据与仿真器（开环 nuScenes/Waymo vs 闭环 NAVSIM/CARLA/Bench2Drive）、导航环境与 benchmark（Habitat/R2R/RxR/ScaleVLN）、基础 RL 基准（DMControl/Atari 100k/Procgen + 真机桥接 FMB）。
6. **Open Challenges and Future Directions**（p14）— 动作意图与动作对齐解耦、世界-动作分解（agent/universe 因子分解及「联合优化会生成被自家策略利用的乐观状态」的批评）、空间与多视角一致性（3D 显式表示）、长程记忆（MEM/EventVLA/MemoryVAM/Harness VLA 两路线）、神经模拟器与闭环策略学习、推理延迟与效率（少步生成、KV-cache 复用）。
7. **Conclusion**（p15）。

## 最值得查的图表

- **Table I（p11）全领域方法总表**：按操作/导航/自动驾驶三域分组，每行给 Paradigm 标签（VAM / Uni / WM-Plan / RL / 3D）+ Backbone + 评测环境——查某方法属于哪支哪域最快的一张表，操作域就收了 38 行。
- **Fig. 1（p1）演化时间线**：从 TD-MPC/DayDreamer/Pathdreamer（Before）→ 2023 RT/Diffusion Policy/IRIS → 2024 GR-1/Genie/Sora → 2025 π0/DreamerV3/UWM/VPP → 2026 Cosmos Policy/Fast-WAM/DreamZero/GigaWorld-Policy，标注venue——写相关工作时间轴直接抄。
- **Fig. 5（p7）架构 2×2 分类图**：横轴 Joint Prediction vs Inverse Dynamics × 纵轴 End-to-end（统一骨干）vs Dual System（video/action 双专家，MoT 或 cross-attention）——本篇 taxonomy 的浓缩版，与复旦版 Fig. 2 的 Cascaded/Joint 树对照读。
- 次查：Fig. 6（p8）预训练/后训练流水线图、Fig. 7（p9）数据金字塔（互联网视频→egocentric→具身轨迹）、Table II（p12）全部评估指标定义速查、Table III（p13）全部数据集/benchmark 规模与指标速查。

## 与库内论文的重叠

直接被综述收录（10 篇库内笔记可按其分类对号入座）：

- **DreamerV3**（notes/world-model/dreamer-v3.md）→ 引文 [18]（Nature 2025 期刊版）：§II-B MBRL 隐式路线（latent 正则一支）+ Fig. 1 时间线；§VI-E 想象 rollout 的 RSSM 谱系背景。
- **TD-MPC2**（notes/world-model/td-mpc2.md）→ 引文 [58]：§II-B 隐式世界模型谱系（与 Dreamer 系并列）；§V-F 作为「RL 基准上先验证预测模型支持规划/价值学习再迁移机器人」的方法学引用。
- **DayDreamer**（notes/world-model/daydreamer.md）→ 引文 [3]：§I/§II-B 溯源 + Fig. 1 时间线 "Before" 分支——实体机器人上世界模型学习的起点。
- **DreamZero**（notes/world-model/dreamzero.md）→ 引文 [23]（综述所著录题名即 "World action models are zero-shot policies"）：§III-B IDM 范式的 chunk 级自回归视频生成代表（"diffusion 式 predict-then-act 推理开销大"）；§III-C 端到端一支；§III-D 微调（约 30 分钟真机数据跨本体迁移）与视频-动作对齐（video gen + IDM 双路线）；§VI-F KV-cache 复用的效率引用。
- **LingBot-VA**（notes/architecture/lingbot-va.md）→ 引文 [24] "Causal world modeling for robot control"：§III-B 被归入「隐式 IDM」（MoT + 异步推理，比显式 predict-then-act 高效）；§III-C 双系统 MoT 路由的两个代表之一（另一个 Fast-WAM）；Table I 操作域（Uni / causal-diff MoT / AgiBot）。
- **WorldVLA**（notes/world-model/worldvla.md）→ 引文 [39]：§III-B Joint Prediction（共享自回归 token 空间的典型设计，与 Motus 分立两支）；§III-C 端到端系统；Table I（Uni / Chameleon AR / LIBERO）。
- **RISE**（notes/world-model/rise.md）→ 引文 [154] "Self-improving robot policy with compositional world model"：§IV-A 操作第三组「想象式策略优化」（World4RL/DreamPlan/VAMPO/WMPO/EVA/RISE 一列）；Table I（RL / latent / contact-rich）。
- **LeWorldModel**（notes/world-model/leworldmodel.md）→ 引文 [75]：§II-B 隐式世界模型 JEPA 一支（与 DINO-WM 并列的 [74][75]）；§VI-E 作为 goal-based 想象目标优化的引用。
- **V-JEPA 2**（notes/world-model/v-jepa2.md）→ 综述直接引的是其后续 **V-JEPA 2.1** [90]（arXiv 2603.14482），列于 §III-A 视觉编码器的 Representation Encoders（DINO/CLIP/SigLIP/VJEPA，"表示空间而非像素生成器"路线）；V-JEPA 2 本体未直接引用，属同谱系映射。
- **WorldArena**（notes/world-model/worldarena.md）→ 引文 [120]：§III-D 后训练-数据增广的「动作接地轨迹合成」路线（与 DreamGen 并列，用 IDM/LAM 给世界模型生成的无标注 rollout 恢复伪动作）。

名称相近但为不同论文（注意区分）：

- **MemoryWAM**（notes/memory/memorywam.md）：综述 §VI-D 长程记忆引的是 **MemoryVAM** [254]（arXiv 2606.20679，Jiang et al., "Integrating Memory into Video Action Model"）——与本库 MemoryWAM（2606.20562，Yang et al.）编号仅差 117、主题相同，但是不同论文；MemoryWAM 本体未被本综述收录。§VI-D 的两条记忆路线（模型内建 MEM/EventVLA/MemoryVAM vs agentic 外挂 Harness VLA）正是库内 memory 线（MemBodied/MemoryWAM/EchoVLA/RoboMemory 等）在该综述中的归类位置。

未被收录、但可按其分类落位的库内论文：

- **TacWAM**（notes/world-model/tacwam.md）：综述触觉线引的是 HiTac-WAM [138]/FAWAM [139]/VT-WM [143] 等 2026-08 前后的工作，TacWAM（2607.28391）未被引——分类落位在 §III-E「Contact and Auditory Feedback」+ §IV-A「Structured State Models」。
- **SD-JEPA**（notes/world-model/sd-jepa.md）：综述 JEPA-WAM 新作引了 JEPA-WAM [112]/LaWAM [113]/VLA-JEPA [115]（§III-D 表示预测进入 WAM 框架一段），SD-JEPA（2605.31111）未被引，同属该 JEPA-WAM 分支。
- **WoW / SimDist / WCM / SuSIE / UniPi / GR-MG / EgoGenesis / PAIWorld / WEAVER / NoGaussianRequired / I-JEPA / V-JEPA(1)**：均未被引用或仅作间接背景——查这些论文的定位仍以 5 月复旦版综述卡（wam-survey-brief.md）和 wm-comprehensive-brief.md 为准。

## 相关链接

- arXiv：https://arxiv.org/abs/2609.16074
- 项目页（Awesome 列表）：https://rcl-robotics.github.io/Awesome-World-Action-Models
- 库内对照综述卡：`notes/briefs/wam-survey-brief.md`（复旦版，Cascaded/Joint 主轴）、`notes/briefs/wm-comprehensive-brief.md`、`notes/briefs/wmrm-survey-brief.md`、`notes/briefs/wrl-survey-brief.md`
