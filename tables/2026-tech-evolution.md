# 2026 VLA & Manipulation 技术演进总结

> 最后更新：2026-07-16

## 各大顶会最佳论文/核心趋势

| 会议 | 论文 | 方向 | 核心贡献 |
|------|------|------|---------|
| **ICRA 2026** Best Paper | **SymSkill** (GRASP Lab) | 符号+技能共同发明 | 5 分钟无标签数据 → 符号抽象 + 多步长程操作，打通符号逻辑与柔顺操作 |
| **ICRA 2026** Finalist | **FP3** (清华高阳) | 3D Foundation Policy | 首个 3D 点云操作基模，1.3B, 82.5% 零样本, 仅需 80 demos 微调 |
| **ICRA 2026** Best RL | View-Invariant Policy (JHU+TRI) | 视角鲁棒策略 | 策略条件化到相机外参，解决相机移动后策略失效问题 |
| **ICRA 2026** Finalist | **Dexora** (清华/北大/上交) | 开源双臂灵巧 VLA | 10 万仿真 + 1 万遥操作，混合外骨骼+Vision Pro 遥操作 |
| **ICRA 2026** Finalist | Bi-Adapt (NUS) | 双臂少样本泛化 | 少量试错即可迁移已有技能到全新物体类别 |
| **RSS 2026** | **LingBot-VA** (蚂蚁灵波+HKUST) | 自回归视频-动作世界模型 | MoT 架构统一视频预测+动作生成, 50 demos 超越 π0.5 +20pp |
| **CVPR 2026** Highlight | Action-Sketcher (PKU+BAAI) | Sketch-as-reasoning VLA | — |
| **CVPR 2026** Highlight | XL-VLA (UCSD+Amazon) | 跨具身灵巧操作 VLA | — |
| **CoRL 2025** Best Paper | Fabrica + UniFP | 双臂装配 / 统一力位控制 | — |
| **CoRL 2025** Finalist | π0.5 (Physical Intelligence) | 层级 VLA | 首次在未曾见过的家庭环境中完成长程操作 |
| **NeurIPS 2025** Spotlight | Knowledge Insulation | VLA 训练范式 | 梯度隔离 VLM+Action Expert → π0.6/π0.7 的模板 |
| **NeurIPS 2025** Spotlight | SafeVLA | VLA 安全 | 首个 safety-first VLA 研究集群 |
| **ICLR 2026** | WholeBodyVLA (OpenDriveLab) | 人形全身操作 VLA | 从无动作 egocentric 视频学习, 78% AgiBot X2 成功率 |
| **ICLR 2026** | InstructVLA | VLA 指令微调 | 33% SimplerEnv 提升, 96% OpenVLA 提升 |
| **ICLR 2026** | LeRobot (HuggingFace) | 开源基础设施 | 端到端机器人学习库: ACT, Diffusion Policy, 硬件抽象 |

## 七大技术演进趋势

### 趋势 1: VLA 从 2D → 3D

传统 VLA 用 2D 图像，将 3D 空间关系压缩到像素中。FP3 用点云直接建模 3D 空间——零样本从 55% 提升到 95%（同域）。

```
OpenVLA/RT-2 (2D image → action) 
  → 3D Foresight (2D + depth/scene flow auxiliary)
    → FP3 (3D point cloud → action, ICRA 2026 Finalist)
      → WholeBodyVLA (3D + whole-body, ICLR 2026)
```

**关键洞察**：2D 压缩损失了 VLA 最关键的空间 grounding 信息。3D 是下一阶段 VLA 的必要升级方向。

### 趋势 2: 从单臂 → 双臂 → 全身

CoRL 2025 的 Fabrica (双臂装配), ICRA 2026 的 Dexora (双臂灵巧 VLA), ICLR 2026 的 WholeBodyVLA (全身人形操作)。

```
Mobile ALOHA (双臂, 2024) 
  → π0 (多配置单臂, 2024)
    → Dexora (双臂灵巧, ICRA 2026 Finalist)
      → WholeBodyVLA (全身人形, ICLR 2026)
```

### 趋势 3: 从模仿学习 → 自我改进 RL

模仿学习的性能上限 = 示教质量。RL 可以超越这个上限。

```
ACT/DP (纯模仿, 2023-2024)
  → RECAP (离线 RL 蒸馏, 2025)
    → RL Token (在线 RL 精调, 2026)
      → SimpleVLA-RL (全模型 RL 后训练 + emergent pushcut, 2025)
```

### 趋势 4: 从反应式 → 规划式 → 符号推理

VLA 从 "直接输出动作" 进化到 "先推理再行动"。

```
RT-2 (直接动作, 2023) 
  → PALM (affordance reasoning + progress tracking, CVPR 2026)
    → SymSkill (符号+技能共同发明, ICRA 2026 Best Paper)
```

SymSkill 是里程碑——5 分钟无标签数据自主发现操作符和谓词，在线实时符号规划+故障恢复。

### 趋势 5: 世界模型回归

不是简单的视频生成，而是因果世界模型（causal world model）。

```
Dreamer v3 (latent imagination, 2023)
  → UniPi (video-as-policy, 2023)
    → DreamVLA (compact knowledge prediction, 2025)
      → LingBot-VA (视频-动作因果世界模型, RSS 2026)
```

LingBot-VA 是代表：消融去掉视频预测模块 → 成功率从 93% 降到 48%。因果性是关键。

### 趋势 6: 数据效率的质变

| 论文 | 数据需求 |
|------|---------|
| SymSkill (ICRA 2026 Best) | **5 分钟无标签数据** |
| LingBot-VA (RSS 2026) | **50 demos** (最低 10 demos) |
| FP3 (ICRA 2026 Finalist) | **80 demos** |
| BridgeVLA (NeurIPS 2025) | **3 条轨迹/任务** |

2023 年的 ACT 需要 ~50 demos，2024 年的 π0 需要 68 任务的海量数据。2026 年：个位数 demos 可以达到实用级别。

### 趋势 7: 从端到端黑盒 → 结构化推理

```
RT-2 (端到端 black-box, 2023)
  → CoT-VLA (chain-of-thought VLA)
    → PALM (四类 affordance + progress)
      → SymSkill (离散符号抽象)
```

## VLA 架构格局（2026 中）

```
                ┌── 端到端黑盒路线 ─── RT-2, OpenVLA, π0
                │
VLA ───────────┼── 推理增强路线 ─── PALM, OneTwoVLA, TriVLA, G0.5 (Native CoT)
                │
                ├── 3D 感知路线 ─── FP3, 3D Foresight, WholeBodyVLA
                │
                ├── 世界模型路线 ─── LingBot-VA, DreamVLA, UniPi
                │
                ├── RL 改进路线 ─── RL Token, SimpleVLA-RL, RECAP
                │
                ├── 符号推理路线 ─── SymSkill, VoxPoser
                │
                ├── 动作表示革新 ─── FAST, MINT (频域), XR-1 (VQ), G0.5 (ActionCodec)
                │
                └── 高效/工业落地 ─── LingBot-VLA 2.0, Green-VLA, FLOWER, Fabrica
```

**融合趋势**：这些路线不是互斥的。未来最强 VLA 可能是 3D perception + world model + reasoning + RL self-improvement 的组合。

---

## 2026 年 7 月最新进展

### 工业界：LingBot-VLA 2.0（蚂蚁灵波，2026.07.08 开源）

**arXiv 2607.06403** | 当前最重量级的工业级 VLA 开源发布：

| 维度 | 规格 |
|------|------|
| 预训练数据 | **60,000 小时**（50K 真机 + 10K 人类 egocentric 视频） |
| 本体覆盖 | **17 家厂商、20+ 种构型**（Franka, AgileX, Unitree G1, Galaxea R1, 星动纪元等） |
| 动作空间 | **统一 55 维**（双臂关节+末端位姿+夹爪+灵巧手+腰部+头部+底盘） |
| 核心架构 | MoE（共享 expert + 多路由 expert），Sigmoid 路由替代 Softmax |
| 预测动力学 | Dual-query distillation：当前状态 Q_t + 未来状态 Q_{t+δ}，几何监督 + 时序监督 |
| 推理延迟 | **<130ms** on RTX 4090 |
| 性能 | GM-100 超越 π0.5（66.2% vs 59.1% progress）和 GR00T N1.7（36.3%） |
| 开源 | GitHub + HuggingFace + 魔搭社区 |

**核心意义**：LingBot-VA (RSS 2026 因果世界模型) 是学术突破，LingBot-VLA 2.0 是将其工业化、规模化的结果——同一团队在 6 个月内从 paper 到开源产品。意味着 "世界模型 VLA" 路线已经进入实用阶段。

### ICML 2026（7 月）VLA 精选

| 论文 | 核心贡献 |
|------|---------|
| **LaST₀** | Latent Spatio-Temporal CoT：在 latent 空间做时空推理（不显式生成文字），双系统 MoT（低频推理+高频执行），比 SOTA 高 13-14% |
| **Contrastive Rep. Regularization** | RS-CL loss：将 VLM 表征与本体感知信号对齐，RobotCasa-Kitchen 69.7% SOTA |
| **VLA-FixBench** | VLA 故障诊断与恢复 benchmark：VLM-VLA 协作定位时空偏差并回滚，真机 +35% |
| **Anchor-Centric Adaptation** | 识别 "diversity trap"：先锚定核心场景再扩展边界，同等预算优于均匀采样 |

### RSS 2026 精选

| 论文 | 核心贡献 |
|------|---------|
| **FlashSAC (Best Paper)** | 低 UTD off-policy RL, 2.5M 模型, 60+ 任务 SOTA, 人形 <20min sim-to-real |
| **MINT** | DCT 频谱分解: 意图 vs 执行, one-shot 迁移 +60% |
| **RISE** | 组合世界模型: imagination RL, 真机 +35-45% |
| **SimDist** | 仿真蒸馏世界模型, 15-30min 数据, peg 25%→85% |
| BiDemoSyn | 1 次示教→数千条双臂数据 |
| SID (HKUST) | 2 次示教 ~90% 成功率 |

### 产业/开源 7 月

| 工作 | 核心 |
|------|------|
| **LingBot-VLA 2.0** | 60K h, 20+ 构型, 55D, MoE, <130ms, 开源, 超越 π0.5 |
| **WEAVER (CMU+Mila)** | 离线 improvement +38% over π0.5, zero-interaction |
| **PAIWorld (中科院)** | 3D 一致世界模型, WorldArena 第一 |
| **Green-VLA (Sber)** | 五阶段训练, CALVIN 4.62 |
| **ROVE (小鹏)** | 人形人在环 RL, OVE 过滤不完美干预 |

### ICRA 2026 补充

| 论文 | 奖项 |
|------|------|
| **IMR-LLM** | Best Paper on Automation |
| **FAST-LIVO2** | King-Sun Fu Memorial Best Paper |
| **FlashSAC** → 实为 RSS 2026 Best Paper |

## 对你（申请者）的最新定位

1. **World model 爆发窗口**：RISE + WEAVER + PAIWorld + SimDist + LingBot-VA，5 篇世界模型论文同时在 2026 顶会出现——这个方向正在从学术研究走向实用。你的积累直接命中这个窗口
2. **RL for VLA 是新蓝海**：FlashSAC (RSS Best) + RL Token + ROVE + SimpleVLA-RL。VLA 从模仿走向 RL 自改进是确定趋势，竞争者还不多
3. **频域动作表示是差异点**：FAST + MINT 都用 DCT——刚起步，切入空间大
4. **3D 不可避免**：FP3 + PAIWorld + View-Invariant 同时证明 2D 天花板已到
