# Paper Taxonomy — VLA & World Model Research

全部 100 篇论文，按 8 个赛道分类。

---

## Architecture（VLA 架构 + 动作生成）— notes/architecture/

**VLA 基础模型**：rt-1, rt-2, openvla, pi0, pi05, pi07, g05, instructvla
**动作生成（扩散/Flow）**：diffusion-policy, flow-matching, fast-tokenizer, mint
**开源工业 VLA**：xr-1, lingbot-vla2, green-vla, octo, flower, bridgevla
**高效 VLA**：llada-vla, univla, univla-latent-actions, mos-vla, wholebodyvla
**3D VLA**：fp3, view-invariant-policy
**VLA 后训练 + RL**：simplevla-rl, rl-token, rove
**数据多样性**：diversity-is-all-you-need

29 篇

---

## Reasoning（推理-动作融合）— notes/reasoning/

**推理-动作融合**：palm-e, rt-trajectory, voxposer
**双/三系统推理**：gr00t-n1, onetwovla, trivla, gemini-robotics
**视觉前瞻 VLA**：f1-vla, dreamvla
**符号推理**：symskill, imr-llm
**人形全身 VLA**：human-as-humanoid, unifp

13 篇

---

## World Model（世界模型）— notes/world-model/

**潜空间世界模型**：dreamer-v3, daydreamer, td-mpc2
**JEPA 系列**：ijepa, vjepa, v-jepa2, leworldmodel, sd-jepa, no-gaussian-required
**视频即策略**：unipi, susie
**2D 世界模型+策略**：gr-mg
**3D 世界模型**：paiworld, weaver
**World Action Model**：lingbot-va, worldvla, dreamzero, egogenesis, tacwam, simdist, rise, wcm
**评估**：worldarena
**综述**：wam-survey, wmrm-survey, wrl-survey, wm-comprehensive

26 篇

---

## RL（RL 方法）— notes/rl/

**核心 RL 算法**：flashsac, rl-100
**VLA + RL**：rl-token, simplevla-rl, rove, z-1, vlac
**Sim2Real**：viserdex, phys2real
**灵巧操作 RL**：dexora, torl-vla, hapticvla, grits
**分层/规划 RL**：omniretarget, dlo-routing
**技能发现**：diayn
15 篇

---

## Memory（机器人记忆）— notes/memory/

**快速权重记忆**：robottt
**分层压缩记忆**：memorywam
**空间地图记忆**：serf, echovla
**多模块记忆**：robomemory
**状态化训练记忆**：statelinformation
**世界模型持续学习**：wam-ttt

7 篇

---

## Data（数据与基础设施）— notes/data/

**规模化数据集**：open-x-embodiment
**低成本遥操作**：mobile-aloha-act
**自改进循环**：robocat

3 篇

---

## Perception（感知）— notes/perception/

**LiDAR SLAM**：fast-lio2

1 篇

---

## Industry（产业简报）— notes/industry/

**商业闭源模型**：gen-1, gene-26-5, 2026-vla-research-brief

3 篇

---

## 全局统计

| 赛道 | 篇数 | 核心问题 |
|------|------|---------|
| Architecture | 32 | 动作怎么表示和生成 |
| Reasoning | 13 | 长程任务中的时序推理 |
| World Model | 26 | 如何建模环境动力学 |
| RL | 16 | 如何超越模仿学习上限 |
| Memory | 7 | 如何让机器人不忘记 |
| Data | 3 | 数据从哪里来 |
| Perception | 1 | 如何定位和建图 |
| Industry | 3 | 商业落地现状 |

**演进方向**：动作表示升级（频域/VQ/联合编码）→ 世界模型从被动预测变主动训练环境（WAM）→ RL 从 demo 走向自我改进 → 记忆从外挂变成架构核心 → 3D 几何回归
