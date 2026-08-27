# World Models: A Comprehensive Survey（综述大纲卡）

- 本地 PDF：`papers/briefs/WM-Comprehensive_2606.00133.pdf`
- arXiv：https://arxiv.org/abs/2606.00133
- 年份：2026（6 月）
- 团队：Augusta University / University of Georgia 牵头，26 位作者多机构联合
- 类型：综述大纲卡（当索引用，不写深度笔记）

## 这篇综述讲什么

目前覆盖面最广的世界模型综述（147 页）：不限于机器人，横跨 RL、自动驾驶、视频生成、医学、金融等。用**四个分类轴**组织全领域——架构（表示/动力学/模态/学习范式/下游用途）、方法论家族（状态空间循环 / Transformer / 扩散 / 物理感知 / 语言增强）、推理策略（想象规划 / 策略学习 / 反事实 / 不确定性规划）、应用领域。机器人只是其中一个域；它的价值在于跨域统一框架与 Dreamer/Cosmos/MuZero/Sora/Genie/V-JEPA 2 这类锚定系统的横向对比。

## 章节大纲

1. **Introduction**（p5）
2. **Background and Conceptual Foundations of World Model**（p7）— 定义与基本概念（Schmidhuber 起源）；世界模型的关键组件；世界模型 vs model-free RL 的根本区别；latent space 的作用。
3. **Categorization of World Models by Architecture**（p16）— 五个子轴：by Representation（pixel 级 / 连续 latent / 离散 token / joint-embedding 预测 / object-centric 结构化 / 3D-occupancy）；by Dynamics（确定性 / 随机 / 隐式生成式 / 表征空间预测式 / 记忆增强）；by Modality（visual-only / language-only / 3D 几何 / 触觉本体感知 / 多模态融合）；by Learning Paradigms（自监督 / online MBRL / offline 与 batch / foundation 大规模预训练 / 监督模仿 / 混合多阶段）；by Downstream Use（RL 规划 / 自动驾驶 / 机器人具身 / 医疗影像 / 视频生成 / 语言推理）。各子轴末尾均有 Discussion and Open Challenges。
4. **Categorization of World Models by Methodological Families**（p36）— State-space and recurrent latent（RSSM/Dreamer 家族主线）、Transformer-based、Diffusion-Based、Physics-informed and structured、Language-augmented and multi-modal。
5. **Categorization of World Models by Reasoning Strategy**（p52）— Imagination-based planning（learning 期 background planning vs decision-time forward search，compounding errors 与 objective mismatch）；Policy learning with a world model；Counterfactual reasoning（abduction–action–prediction 管线、非可辨识性极限）；Planning under uncertainty。
6. **Categorization of World Models by Application Domains**（p63）— Robotics、Autonomous driving、Video prediction & scene understanding、Multimodal agents、RL & Games、Scientific modeling、Medical imaging/video、Educational measurement、Business & finance。
7. **Evaluation Protocols and Benchmarks**（p96）— Common Evaluation Metrics；Benchmark Environments and Datasets。
8. **Major Challenges and Limitations**（p99）
9. **Discussion and Future Directions**（p106）

## 值得查的表/图

- **Figure 1（p8）全景图**：三层结构——隐式表示与未来预测的概念分类 + 1974–2024 里程碑时间线 + 各应用域部署样例。
- **Table 1（p21）按输入模态分类的代表模型表**。
- **Table 2（p32）按学习范式分类的代表模型表**（含 Data req. 列）。
- **Table 3（p64）机器人领域代表模型按 paradigm 分组** — Latent dynamics（DayDreamer、TD-MPC2、RoboDreamer…）/ Predictive-JEPA（I-JEPA、V-JEPA、V-JEPA 2、DINO-WM）/ Generative sim.（UniSim、COSMOS、UWM、Cosmos Policy）/ Structured-3D（DreMa、PointWorld），并标注是否真机验证。机器人方向查这张最快。
- **Table 4（p65）JEPA 家族对比表** — I-JEPA / V-JEPA / DINO-WM / V-JEPA 2 四行 × Input / Encoder / Action-conditioned / Offline / Zero-shot 五列，写 JEPA 相关论文的对比基线表可直接参照其维度设计。
- **Table 5（p69）自动驾驶 WM 评测基准**、**Tables 6+9（pp.70,98–99）视频预测代表模型与基准**、**Table 10（p99）主要 benchmark 环境汇总**。
- **Figure 2（p53）latent 世界模型中的 imagination-based planning 示意图**。

## 与本库的关系

- **DayDreamer**：Table 3 归入 **Latent dynamics** paradigm（RSSM 部署于 4 类真机平台，真机标注 ✓）；§robotics 正文以其论证 latent imagination 在真实噪声/延迟下仍有效。
- **TD-MPC2**：Table 3 归入 **Latent dynamics** paradigm（latent MPC + value learning）。
- **I-JEPA**：Table 3 **Predictive/JEPA** paradigm + Table 4 JEPA 家族对比行（Image 输入、learned ViT、无动作条件、可 offline 训练）。
- **V-JEPA**：Table 3 + Table 4 同上（video 输入、无动作条件）。
- **V-JEPA 2**：Table 3 + Table 4（video、ViT-H、动作条件 ✓ offline ✓ zero-shot ✓，1.2B action-conditioned MPC）；正文 sim-to-real 一节亦引 V-JEPA 2-AC。
- **LeWorldModel**：§Architecture 的 representation 分类中，归入 joint-embedding (JEPA) 式「从像素端到端学预测 embedding、避免显式重建」一支。
- **SimDist**：§Architecture 中「Cross-domain transfer and adaptation」段，作为「仿真预训练 + 只微调动力学模块解决 sim-to-real」的模块化案例。
- **SuSIE**：§Methodological Families 内被引一次，作为语言条件 image-editing 扩散做子目标预测 world model 的轻量路线。
- 未检索到本库这些论文：UniPi、GR-MG、WorldVLA、DreamZero、LingBot-VA、WoW、SD-JEPA、TacWAM、WorldArena、RISE、WCM。本库 robotics 方向写 related work 时，这篇适合用来给方法一个跨域坐标（JEPA 家族谱系、四大范式分组），但操作细节应引 WMRM/WRL/WAM 三篇专向综述。
