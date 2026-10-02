# Action Chunking Engineering：88 方法精选工作台（大纲卡）

- 来源：GitHub 精选库 `Peaceful-World-X/Action-Chunking-Survey`（https://github.com/Peaceful-World-X/Action-Chunking-Survey）
- 类型：大纲卡（领域内**无正式综述论文**，此库是事实上的综述：88 个方法、ACT 2304 → ChunkFlow 2607、附交互对比工作台与机制动画）
- 年份：2026（持续更新）
- 拉取日期：2026-10-02

## 为什么值得收藏

action chunking 没有专门综述——VLA 综述（2508.13073 等）只给一小节。这个库把 2023-2026 全部方法收进一张可筛选的表，并按**工程机制**（不是发表年份）分类：chunk 长度怎么定、边界噪声怎么去、推理怎么实时化、RL 怎么和 chunk 结合。库内 mobile-aloha-act（ACT 原文）、diffusion-policy、pi0/fast-tokenizer、oxyGen 等笔记的共同主题就是这张表的局部。

## 机制分类大纲（88 方法，按工作台分类轴）

1. **Horizon/长度自适应**（chunk 长度不再拍脑袋）：ChunkFlow（2607，元动作归纳）、A³、ACH、ACSAC、AQC、AAC、AAC-DB、AutoH、EQRL、MoH、SEAM、SGAC、TAS、C-LASI、CDM、ADH、P2A、CAD、H2O2、AdaptiveH
2. **实时/异步推理（RTC 家族）**——把 chunk 推理摊平成流水线，解决"chunk 太长反应慢"：RTC-Inference（2509）/RTC-Training、Soft-RTC、D-RTC、VLASH、VLA-RAIL、TIDAL、ACNet、PAINT、REMAC、FASTER、SuP、FLASH、RTI-DP、RDP、TubeDP、QUART-Online、DiP-ACT、SwiftPP、RSP、SSR、SSD、Soft-ACT、CStac、Duplet
3. **连续/平滑表示**（用样条/小波替代离散 chunk）：BSP（B-spline）、SplineP、ABPolicy、Spline Policy、DWS、RTR、LiPo、Legato、NIAF、Wavelet 系列
4. **chunk 边界噪声/去视角耦合**（ACT 的复合误差问题）：NSA、DVAC、BID、PTE、PPC、DCDP、A2C2、SV-VLA、PATCH、ChunkBC、ChunkingLCB、CSA、Corrector（VLA-Corrector）、VLA-Markup、DEHP、RoboChat、PaceVLA、SSD-OOD、ARRoW、RecVLA、SVRA、SPARSE
5. **RL × chunk**（chunk 与 RL 互融：鲁棒性、探索、策略梯度）：PAC-ACT、SEAR、QC、RL-ACP、AC-PPO、CO-RFT、AC³、MAC、DQC、AQC(RL 侧)、SERNF、VQ-ACE、VGAS、SCD、Boost-VLA、QSA-RL、MoE-AC、Multi-ACT、RefineACT、VLA-a-a
6. **频域/分层策略**：FAFM、FocalPolicy、HiPolicy、HiFlow、PACE（Phase-Aware Chunk Execution，2606）、ChunkDVAE、CoE、CogACT、EACT
7. **Tokenization/表征**：OAT、FASTer-VQ、VQActFlow、CoA、PF-DAG、CCT、ChungusVLA、SpecVLA、Athena、A/1、AliVLA、VLA4AD
8. **分析与理论**（何时/为何有效）：Why Does Action Chunking Improve Behavioral Cloning?（2608.02547，把收益归因于"延迟重规划"效应）、Action Chunking and Exploratory Data Collection Yield Exponential Improvements（2507.09061，开环稳定任务下指数优势 + 探索数据收集）、Stabilizing ACT（IEEE，执行期不稳定性）

## 用法建议

- **按问题查**：写笔记遇到"重规划频率/chunk 长度"协议项时，用分类轴 1/2 查对应方法族；机器人四问的 Correction 维度基本等价于 RTC 家族 + 自适应长度两族要解的问题。
- **两篇理论文值得深度入库**（已登记 watchlist 候选）：2608.02547（why it works）、2507.09061（when it works）。
- 与库内笔记的连接：ACT（mobile-aloha-act）、Diffusion Policy chunk、π0 动作专家、FAST-tokenizer、OxyGen 70Hz、F1/pa0 系列 Evidence 行的 chunk 数值，都能在这张表找到同族方法做横向对比。

## 与本库的关系

- **mobile-aloha-act**：表中 2304 起点（ACT 原文）。
- **diffusion-policy / pi0 / fast-tokenizer / oxyGen**：分别对应"扩散 chunk 表示 / flow-matching 动作专家 / tokenization 加速 / RTC 系统化"四条线的库内锚点。
- **VLA 综述 2508.13073**：正式综述里讲 chunking 的一节（Parallel Decoding via Chunking），可作交叉索引。
