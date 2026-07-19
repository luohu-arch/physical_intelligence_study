# Paper Taxonomy — VLA & World Model Research

按 5 大核心研究问题对所有论文进行分类。共 58 篇，覆盖 2022-2026。

## 五条研究主线

```
动作表示 ──→ 泛化 ──→ 长程推理 ──→ RL 自改进 ──→ 世界模型
    │           │           │            │            │
    └───────────┴───────────┴────────────┴────────────┘
                        │
                  终极目标：机器人看到一个新任务，
                  看一遍人类怎么做，在脑子里想象各种可能，
                  挑最好的方案，执行中遇到问题自己修正
```

---

## 问题一：动作应该怎么表示和生成？

机器人策略的"输出格式"是什么——这是 VLA 最底层的设计选择。

| 方案 | 代表论文 | 年份 | 核心逻辑 |
|------|---------|------|---------|
| 离散 token | RT-1 | 2022 | 256-bin 均匀离散化 |
| 扩散去噪 | Diffusion Policy | 2023 | 噪声→动作，DDPM，捕获多模态分布 |
| | Octo | 2024 | 27M/93M 扩散策略，消费级 GPU |
| Flow Matching | Flow Matching (理论) | 2022 | OT 路径替代扩散，10 步替代 100 步 |
| | π0 | 2024 | VLM + FM Action Expert, SE(3) 动作 |
| | π0.5 | 2025 | 异构 co-training, 97.6% 非目标数据 |
| | π0.7 | 2026 | 多模态上下文 + steerability |
| | FLOWER | 2025 | 950M, 裁剪 50% LLM 层 |
| | Green-VLA | 2026 | 五阶段训练, 64D 统一动作空间 |
| | LLaDA-VLA | 2025 | 扩散 VLM 替代自回归 VLM |
| 频域压缩 | FAST Tokenizer | 2025 | DCT, 前 10% 系数保留 95% 能量 |
| 频域解耦 | MINT (RSS 2026) | 2026 | DCT 分解意图 vs 执行 |
| VQ 编码 | G0.5 (ActionCodec) | 2026 | 27D 跨本体 VQ codebook |
| | XR-1 (ICML Oral) | 2026 | UVMC: 视觉+动作联合 VQ, 共享 codebook |
| CVAE | ACT / Mobile ALOHA | 2024 | CVAE encoder 压缩真值动作 |
| 回归 | OpenVLA, PaLM-E | 2024 | VLM 直接回归连续动作 |
| 自回归统一 | RT-2 | 2023 | 动作 token 融入 VLM 词表 |
| | G0.5 (Native CoT) | 2026 | 推理+动作同一自回归序列 |
| | UniVLA (discrete) | 2025 | 视觉-语言-动作全离散 token 统一 |
| 3D 动作 | FP3 (ICRA Finalist) | 2025 | 点云→DiT→SE(3) 动作 |
| 力控 | UniFP (CoRL Best) | 2025 | 位置+力联合输出，无传感器 |

**演进**：离散 token → 扩散 → Flow Matching → 频域/VQ → 视觉-动作联合编码

---

## 问题二：VLA 怎么从"能做"到"在哪都能做"？

泛化是 VLA 从 lab 到真实世界的核心瓶颈。

| 泛化维度 | 代表论文 | 年份 | 方案 |
|---------|---------|------|------|
| 跨机器人 | π0.5 (97.6% 非目标) | 2025 | 异构数据大规模 co-training |
| | Octo (OXE 预训练) | 2024 | 多具身联合训练 |
| | GR00T N1 (数据金字塔) | 2025 | only 10% 真实数据 |
| | LingBot-VLA 2.0 (60K h) | 2026 | 20+ 构型, 工业级 scale |
| | XR-1 (6 具身) | 2026 | UVMC 跨本体 codebook |
| | RoboCat (自改进) | 2023 | 执行→生成数据→重训练 |
| 跨任务 | MoS-VLA (基函数+L1) | 2025 | 1 次演示适配新任务 |
| | MINT (Intent Token) | 2026 | 从 1 次演示提取意图 |
| | XR-1 (20 demos) | 2026 | 新任务 20 条演示 |
| 视角不变 | FP3 (3D 点云) | 2025 | 3D → 天然视角不变 |
| | 3D Foresight | 2026 | 2D+深度+场景流辅助 |
| | View-Invariant (ICRA Best) | 2025 | Plücker ray 相机条件化 |
| 零样本 | Human-as-Humanoid (CoRL) | 2026 | 人类视频→人形, 零机器人 demo |
| | SuSIE | 2023 | 图像编辑扩散模型→子目标图像 |
| 环境泛化 | Green-VLA (R2 RL 对齐) | 2026 | 五阶段渐进缩小 gap |
| 全身人形 | WholeBodyVLA (ICLR) | 2026 | 全身 loco-manipulation VLA |

**演进**：单具身→跨具身, 2D→3D, 百条 demos→个位数 demos, 同域→零样本

---

## 问题三：长程任务中"我做到哪了"怎么解决？

多步操作的核心难点——模型不知道自己完成了多少、下一步该做什么。

| 方案 | 代表论文 | 年份 | 核心机制 |
|------|---------|------|---------|
| 符号+技能 | SymSkill (ICRA 双奖) | 2026 | 5min 数据→谓词+操作符→A*规划 |
| Affordance+Progress | PALM (CVPR) | 2026 | 四类 affordance + 连续 progress 0→1 |
| 轨迹草图 | RT-Trajectory | 2023 | 2D 粗粒度轨迹作为条件 |
| LLM→3D Value Map | VoxPoser | 2023 | LLM 代码→3D 可组合 value maps |
| Native CoT | G0.5 | 2026 | 推理和动作同一自回归序列 |
| Latent CoT | LaST₀ (ICML) | 2026 | latent 时空推理, 无文字延迟 |
| 双系统 | GR00T N1 | 2025 | VLM(10Hz) + DiT-FM(120Hz) |
| | OneTwoVLA | 2025 | Decision Token 自适应切换 |
| 三系统 | TriVLA | 2025 | VLM + 视频扩散 + FM |
| 视觉前瞻 | F1-VLA | 2025 | 先预测目标状态→再解码动作 |
| | DreamVLA | 2025 | 紧凑世界知识预测替代像素级预测 |
| 层级装配 | Fabrica (CoRL Best) | 2025 | 规划+R : 宏观用 PDDL, 微观用 SE(3) RL |

**演进**：端到端黑盒 → affordance → 符号规划 → 多系统协同

---

## 问题四：如何超越模仿学习的上限？

示教数据有限、有噪声、不一定最优——RL 可以突破。

| 方案 | 代表论文 | 年份 | 核心机制 |
|------|---------|------|---------|
| 想象中 RL | RISE (RSS) | 2026 | 组合世界模型=训练环境, +35-45% |
| 离线改进 | WEAVER | 2026 | 从 buffer 蒸馏, 零真机交互, +38% |
| 在线 RL 精调 | RL Token (PI) | 2026 | 小 actor-critic on VLA 特征 |
| 全模型 RL | SimpleVLA-RL | 2025 | PPO on VLA, pushcut emergent behavior |
| | RECAP | 2025 | 离线 RL 蒸馏→全 VLA |
| 人在环 RL | ROVE (小鹏) | 2026 | OVE optimistic value filtering |
| 仿真蒸馏 | SimDist (RSS) | 2026 | 仿真预训练→仅微调 dynamics |
| RL 底层算法 | FlashSAC (RSS Best) | 2026 | 低 UTD + 大模型 + 归一化 |
| 自改进循环 | RoboCat | 2023 | 执行→生成数据→重训练 |
| 力控制 | UniFP (CoRL Best) | 2025 | 位置+力统一, 无需力传感器 |

**演进**：纯模仿→online RL→imagination RL（零真机）→ 世界模型=训练环境

---

## 问题五：世界模型到底该怎么建？

从"预测像素"到"能帮助策略改进的想象引擎"。

| 维度 | 代表论文 | 年份 | 核心贡献 |
|------|---------|------|---------|
| **Latent 想象** | Dreamer v3 | 2023 | RSSM + symlog, 150+ tasks |
| | DayDreamer | 2022 | 真实机器人 1h 在线学习 |
| **Decoder-free** | TD-MPC2 | 2023 | TD-learning 替代像素重建 |
| **Latent prediction SSL** | I-JEPA | 2023 | context→target latent 预测 |
| | V-JEPA | 2024 | 视频时空 latent 预测 |
| **Video-as-policy** | UniPi | 2023 | 视频扩散→逆动力学→动作 |
| | SuSIE | 2023 | 图像编辑替代视频生成 |
| **2D 世界模型+策略** | GR-MG | 2024 | RGB 预测+goal 生成+action chunking |
| **因果世界模型** | LingBot-VA (RSS) | 2026 | 视频+动作统一自回归 |
| **组合世界模型** | RISE (RSS) | 2026 | 动态+价值分离, imagination RL |
| **3D 一致世界模型** | PAIWorld | 2026 | Geo-RoPE + 3D 蒸馏, WorldArena 1st |
| **多目标世界模型** | WEAVER | 2026 | 保真度+效率+长程, +38% offline |
| **仿真蒸馏** | SimDist (RSS) | 2026 | 仿真→真实仅微调 dynamics |
| **World model post-training** | UniVLA (discrete) | 2025 | VLA+世界模型→LIBERO 95.5% |

**演进**：latent 想象→视频生成→因果序列模型, 单一→组合, 2D→多视角 3D

---

## 辅助方向

### 数据与基础设施

| 论文 | 年份 | 贡献 |
|------|------|------|
| Open X-Embodiment | 2023 | 22 种机器人, 100 万+轨迹, VLA 的 ImageNet |
| Mobile ALOHA / ACT | 2024 | $32K 低成本遥操作 |
| RoboCat | 2023 | 自改进循环 |

### 感知（SLAM / 传感器）

| 论文 | 年份 | 贡献 |
|------|------|------|
| FAST-LIO2 | 2022 | 紧耦合 LiDAR-惯导里程计 |
| FAST-LIVO2 (ICRA Best) | 2026 | 激光-惯性-视觉直接里程计 |

### 工业/评估

| 论文 | 年份 | 贡献 |
|------|------|------|
| IMR-LLM (ICRA Best Auto) | 2026 | LLM+运筹学→工业多机器人调度 |
| Fabrica (CoRL Best) | 2025 | CAD→双臂自主装配 |
| Physion-Eval | 2026 | 视频物理 reasoning 评估 benchmark |

### 商业/产业

| 论文 | 年份 | 状态 |
|------|------|------|
| Gemini Robotics | 2025 | 闭源, Google DeepMind |
| GEN-1 | 2026 | 闭源, Generalist AI |
| GENE-26.5 | 2026 | 闭源, Genesis AI |
| LingBot-VLA 2.0 | 2026 | 开源, 60K h, 20+ 构型 |

---

## 五条线的交汇

```
     问题一              问题二              问题三
   动作怎么表示       ──→  怎么泛化       ──→  长程怎么做
       │                    │                    │
       │    动作 token 化    │    跨具身 codebook  │    符号规划+CoT
       │    → 频域/VQ       │    → 3D 视角不变    │    → 多系统分工
       │                    │                    │
       └────────────────────┴────────────────────┘
                            │
                    问题四 + 问题五
                  想象中 RL + 世界模型
                    互为因果：
         世界模型提供想象环境 → RL 在想象中自改进
         RL 产生高质量数据 → 世界模型继续提升
```

每篇论文解决这个拼图中的一块。当前 58 篇覆盖了大部分，缺的主要是灵巧手操作和触觉感知两个子方向。
