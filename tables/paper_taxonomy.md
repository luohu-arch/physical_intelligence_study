# Paper Taxonomy — VLA & World Model Research

按技术路线和演进关系对所有 39 篇论文进行分类。

## 分类总览

| Track | 论文数 | 时间跨度 | 核心命题 |
|-------|--------|---------|---------|
| [VLA 架构演进](#1-vla-架构演进) | 10 | 2022-2026 | 动作如何生成：离散→连续→统一自回归 |
| [推理-动作融合](#2-推理-动作融合) | 7 | 2023-2025 | 从"感知→动作"到"思考→动作"，System 1+2 |
| [世界模型](#3-世界模型) | 8 | 2022-2024 | 学习环境动力学，latent imagination |
| [视觉前瞻 VLA](#4-视觉前瞻-vla) | 3 | 2025 | 先预测未来状态，再生成动作 |
| [高效 VLA](#5-高效-vla) | 5 | 2024-2025 | sub-billion 参数，低算力预训练 |
| [数据与规模化](#6-数据与规模化) | 3 | 2023-2024 | 数据集、遥操作、自改进 |
| [产业与商业](#7-产业与商业) | 4 | 2025-2026 | 闭源/商业模型，全栈整合 |

---

## 1. VLA 架构演进

**核心命题：动作如何生成？离散 token → 连续扩散 → Flow Matching → 统一自回归**

```
RT-1 (2022) ──→ RT-2 (2023) ──→ OpenVLA (2024) ──→ π0 (2024) ──→ π0.5 (2025) ──→ π0.7 (2026)
  │                │                  │                  │               │               │
  │ 离散动作token   │ VLM词表融合       │ 开源7B            │ VLM+FM Expert  │ 异构co-train   │ steerability
  │                │                  │                  │               │               │
  └──→ G0.5 (2026): 统一自回归，VLM-as-Actor，回归RT-2范式但解决效率瓶颈
```

| # | 论文 | 年份 | 团队 | 架构范式 | 动作表征 |
|---|------|------|------|---------|---------|
| 1 | RT-1 | 2022 | Google DeepMind | Transformer + 离散动作 | 256-bin 离散化 |
| 2 | RT-2 | 2023 | Google DeepMind | VLM + 动作 token 融合 | 离散 token in VLM vocab |
| 3 | OpenVLA | 2024 | Stanford/UCB/TRI | VLM 直接回归动作 | 连续回归 (7B) |
| 4 | π0 | 2024 | Physical Intelligence | VLM + Flow Matching Expert | SE(3) Flow Matching, 10步 |
| 5 | π0.5 | 2025 | Physical Intelligence | 同 π0，异构 co-training | 97.6% 非目标数据 |
| 6 | π0.7 | 2026 | Physical Intelligence | 多模态上下文 + steerability | Flow Matching, 继承 π0 |
| 7 | G0.5 | 2026 | 星海图 | 统一自回归 (VLM-as-Actor) | VQ ActionCodec, 27D |
| 8 | Diffusion Policy | 2023 | Columbia | 扩散动作生成 | DDPM 连续动作 |
| 9 | Flow Matching | 2022 | (基础理论) | CNF 最优传输 | ODE, 10步替代扩散 |
| 10 | FAST Tokenizer | 2025 | Physical Intelligence | DCT 频域压缩 | 前10%系数, 序列压缩80% |

**演进主线：**
- 2022: RT-1 证明 Transformer + 离散动作可行，但精度受限
- 2023: RT-2 将动作融入 VLM 词表；Diffusion Policy 用扩散替代离散 → 连续动作精度飞跃
- 2024: OpenVLA 开源 7B；π0 用 Flow Matching (10步) 替代扩散 (16-100步)
- 2025: π0.5 证明数据多样性 > 数据规模；FAST 用频域压缩解决长序列
- 2026: G0.5 回归统一自回归 + VQ Codec 解决效率；π0.7 实现 steerability

---

## 2. 推理-动作融合

**核心命题：从"感知→动作"到"思考→动作"。System 1 (快速反应) + System 2 (深度推理)**

```
PaLM-E (2023) ──→ RT-Trajectory (2023) ──→ GR00T N1 (2025) ──→ TriVLA (2025)
    │                    │                       │                    │
    │ LLM理解具身输入    │ 轨迹草图条件            │ System 2+1 双系统   │ System 1+2+3 三系统
    │                    │                       │                    │
    └──→ VoxPoser (2023): LLM写出3D价值图代码     │                    │
                                                   │                    │
                                          OneTwoVLA (2025)     Gemini Robotics (2025)
                                          Decision Token 自适应  双模型(ER+动作)分离
```

| # | 论文 | 年份 | 核心思路 | 推理方式 |
|---|------|------|---------|---------|
| 1 | PaLM-E | 2023 | 传感器→LLM token space | LLM 文本推理 |
| 2 | RT-Trajectory | 2023 | 2D 轨迹草图 condition | 粗粒度轨迹作为中间表征 |
| 3 | VoxPoser | 2023 | LLM 写代码→3D value map→MPC | System 2 空间推理 |
| 4 | GR00T N1 | 2025 | VLM(10Hz) + DiT-FM(120Hz) 双系统 | System 2 规划 + System 1 执行 |
| 5 | OneTwoVLA | 2025 | Decision Token 自适应切换 | 单模型内 S1+S2 融合 |
| 6 | TriVLA | 2025 | VLM + 视频扩散 + FM 三系统 | 情景记忆驱动的 S1+S2+S3 |
| 7 | Gemini Robotics | 2025 | ER(空间推理) + 动作 双模型 | 推理与动作显式分离 |

**演进主线：**
- 2023: PaLM-E 让 LLM 看懂传感器；VoxPoser/RT-Trajectory 探索不同中间表征
- 2025: GR00T N1 确立"双系统"范式；OneTwoVLA 尝试单模型融合；TriVLA 扩展到三系统

---

## 3. 世界模型

**核心命题：学习环境动力学预测。从 latent imagination 到视频生成规划**

```
                  ┌──→ Dreamer v3 (2023): RSSM + latent imagination, 150+ tasks
                  │
DayDreamer (2022) ──→ TD-MPC2 (2023): decoder-free + MPC, 替代路线
(真实机器人验证)     │
                  ├──→ I-JEPA (2023) ──→ V-JEPA (2024): latent prediction SSL → 视频
                  │
                  ├──→ UniPi (2023): video-as-policy, 视频扩散规划
                  │       │
                  │       └──→ SuSIE (2023): 图像编辑替代视频生成, 更轻量
                  │
                  └──→ GR-MG (2024): 2D 世界模型 + 策略联合训练
```

| # | 论文 | 年份 | 世界模型类型 | 核心机制 |
|---|------|------|------------|---------|
| 1 | DayDreamer | 2022 | Online RSSM | 真实机器人在线学习 |
| 2 | Dreamer v3 | 2023 | RSSM + symlog | latent imagination, actor-critic |
| 3 | TD-MPC2 | 2023 | Decoder-free | latent MPC + TD-learning |
| 4 | I-JEPA | 2023 | Latent prediction SSL | context→target latent 预测 |
| 5 | V-JEPA | 2024 | Video latent prediction | 过去帧→未来帧 latent |
| 6 | UniPi | 2023 | Video diffusion planner | 文本→视频→逆动力学→动作 |
| 7 | SuSIE | 2023 | Image editing planner | 观测+语言→子目标图像→BC |
| 8 | GR-MG | 2024 | 2D future prediction | RGB预测+goal生成+action chunking |

**演进主线：**
- 2022: DayDreamer 首次验证真实机器人世界模型可行 (1h 学走路)
- 2023: Dreamer v3 建立通用基线；TD-MPC2 证明无解码器路线更高效
- 2023: I-JEPA 开辟第三条 SSL 路径（非对比非生成）；UniPi 开创 video-as-policy
- 2024: V-JEPA 扩展到视频；GR-MG 将世界模型融入策略联合训练

---

## 4. 视觉前瞻 VLA

**核心命题：先预测未来视觉状态，再生成动作。2025 年新兴路线**

```
UniPi (2023) ──→ SuSIE (2023) ──→ F1-VLA (2025) ──→ DreamVLA (2025)
(video规划)      (图像编辑子目标)    (显式视觉前瞻)     (紧凑世界知识预测)
```

| # | 论文 | 年份 | 预测对象 | 动作生成 |
|---|------|------|---------|---------|
| 1 | F1-VLA | 2025 | 完整未来图像 | 逆动力学 + 动作解码 |
| 2 | DreamVLA | 2025 | 紧凑世界知识 (动态/深度/语义) | DiT 解码动作 |
| 3 | UniVLA (discrete) | 2025 | 视频因果动态 (隐式) | 自回归离散 token |

**核心分歧：** 像素级预测 (F1) vs 紧凑知识预测 (DreamVLA) vs 隐式 latent 预测 (UniVLA)

---

## 5. 高效 VLA

**核心命题：打破参数量竞赛。sub-billion 参数，消费级 GPU 可训练/部署**

| # | 论文 | 年份 | 参数量 | 关键技巧 | 代表性结果 |
|---|------|------|--------|---------|-----------|
| 1 | Octo | 2024 | 27M/93M | Transformer + Diffusion Head | 消费级 GPU 部署 |
| 2 | FLOWER | 2025 | 950M | LLM层裁剪50% + Global-AdaLN | CALVIN 4.53 SOTA, 200 H100-h |
| 3 | UniVLA (OpenDriveLab) | 2025 | - | DINOv2 latent action + VQ-VAE | 1/20 预训练算力, LIBERO 95.2% |
| 4 | LLaDA-VLA | 2025 | - | 扩散 VLM 替代自回归 VLM | CALVIN 4.01 > OpenVLA 3.27 |
| 5 | SimpleVLA-RL | 2025 | - | RL post-training on VLA | LIBERO SOTA, RoboTwin > π0 |

**核心洞察：** 精心设计的 sub-billion 模型 + 高效训练策略可以在关键 benchmark 上超越数十亿参数方法。

---

## 6. 数据与规模化

**核心命题：数据从哪里来，如何规模化**

| # | 论文 | 年份 | 贡献 | 影响 |
|---|------|------|------|------|
| 1 | Open X-Embodiment | 2023 | 22种机器人, 100万+轨迹, 统一格式 | VLA 的"ImageNet 时刻" |
| 2 | Mobile ALOHA / ACT | 2024 | $32K 低成本全身遥操作 | 引爆数据采集生态 |
| 3 | RoboCat | 2023 | 自改进循环: 执行→生成数据→重训练 | 数据闭环范式 |

---

## 7. 产业与商业

**核心命题：从 benchmark 到部署。闭源商业模型的声明与推测**

| # | 论文 | 年份 | 团队 | 状态 | 核心声明 |
|---|------|------|------|------|---------|
| 1 | Gemini Robotics | 2025 | Google DeepMind | 闭源 | ~100 demos 微调新任务 |
| 2 | G0.5 | 2026 | 星海图 | 闭源(权重待开源) | 7 大基准 SOTA |
| 3 | GEN-1 | 2026 | Generalist AI | 闭源 | 99% 成功率, 3× 速度 |
| 4 | GENE-26.5 | 2026 | Genesis AI | 闭源 | 全栈灵巧操作 (硬件+模型) |

**注意：** GEN-1 和 GENE-26.5 无公开论文，仅有商业声明。对待方式：产业观察，非可复现研究。

---

## 全局技术演进时间线

```
2022 ─┬─ RT-1 (离散动作起点)
      ├─ DayDreamer (真实世界模型)
      └─ Flow Matching (理论基础)

2023 ─┬─ RT-2 (VLA范式确立)
      ├─ PaLM-E (LLM具身化)
      ├─ VoxPoser (模块化空间推理)
      ├─ RT-Trajectory (轨迹草图)
      ├─ RoboCat (自改进)
      ├─ Diffusion Policy (连续动作扩散)
      ├─ Dreamer v3 (通用世界模型)
      ├─ TD-MPC2 (隐式世界模型)
      ├─ I-JEPA (latent prediction SSL)
      ├─ UniPi (video-as-policy)
      ├─ SuSIE (图像编辑规划)
      └─ Open X-Embodiment (数据规模化)

2024 ─┬─ OpenVLA (开源7B VLA)
      ├─ Octo (轻量扩散策略)
      ├─ Mobile ALOHA/ACT (低成本遥操作)
      ├─ π0 (VLM + Flow Matching)
      ├─ V-JEPA (视频 JEPA)
      └─ GR-MG (2D世界模型+策略)

2025 ─┬─ π0.5 (开放世界泛化)
      ├─ GR00T N1 (双系统架构)
      ├─ OneTwoVLA (自适应推理)
      ├─ TriVLA (三系统)
      ├─ Gemini Robotics (闭源标杆)
      ├─ FAST (频域动作压缩)
      ├─ F1-VLA (视觉前瞻)
      ├─ DreamVLA (紧凑知识预测)
      ├─ FLOWER (高效 VLA)
      ├─ UniVLA×2 (离散统一 / 潜在动作)
      ├─ LLaDA-VLA (扩散 VLM 骨干)
      ├─ MoS-VLA (one-shot 适配)
      └─ SimpleVLA-RL (RL 后训练)

2026 ─┬─ π0.7 (steerability)
      ├─ G0.5 (统一自回归)
      ├─ GEN-1 (商业闭源)
      └─ GENE-26.5 (全栈灵巧操作)
```
