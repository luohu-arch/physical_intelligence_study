# Paper Taxonomy — VLA & World Model Research

全部 118 篇论文，按 7 个深度赛道 + 1 个大纲卡赛道分类（RL 赛道 34 篇分 9 个物理子目录：core/vla/sim2real/dexterous/planning + 4 条 agentic 线）。

---

## Architecture（VLA 架构 + 动作生成）— notes/architecture/

**VLA 基础模型**：rt-1, rt-2, openvla, pi0, pi05, pi07, g05, instructvla
**动作生成（扩散/Flow）**：diffusion-policy, flow-matching, fast-tokenizer, mint
**开源工业 VLA**：xr-1, lingbot-vla2, green-vla, octo, flower, bridgevla
**高效 VLA**：llada-vla, univla, univla-latent-actions, mos-vla, wholebodyvla
**3D VLA**：fp3, view-invariant-policy
**VLA 后训练 + RL**：simplevla-rl, rl-token, rove
**数据多样性**：diversity-is-all-you-need
**Agentic VLA / 执行中心 VLM**：art-vla-agent, capek05

31 篇

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
**World Action Model**：lingbot-va, worldvla, dreamzero, egogenesis, tacwam, simdist, rise, wcm, wow
**评估**：worldarena

23 篇

---

## RL（RL 方法）— notes/rl/（物理子目录按子线组织）

**core/（核心算法与技能发现，3 篇）**：flashsac, rl-100, diayn
**vla/（VLA + RL，5 篇）**：rl-token, simplevla-rl, rove, z-1, vlac
**sim2real/（2 篇）**：viserdex, phys2real
**dexterous/（灵巧操作 RL，4 篇）**：dexora, torl-vla, hapticvla, grits
**planning/（分层/规划 RL，2 篇）**：omniretarget, dlo-routing
**agentic-robot/（机器人侧 agentic 闭环，5 篇）**：harbor, enpire, agentic-robotics-loop, playful-agentic, humanvid-selfimprove
**agentic-training/（LLM 侧训练系统，5 篇）**：lego-rl, polar, lite-researcher, arlarena, harness-1
**agentic-algo/（算法与稳定性，5 篇）**：ragen-2, g2po, reasoning-to-agentic, dataprm, coskill
**agentic-app/（应用，3 篇）**：tool-r0, openclaw-rl, tongyi-deepresearch

34 篇（rove 从 architecture/ 归位到 rl/vla/）

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

## Briefs（综述与行业大纲卡）— notes/briefs/

**世界模型综述卡**：wam-survey-brief, wmrm-survey-brief, wrl-survey-brief, wm-comprehensive-brief
**行业简报**：gen-1, gene-26-5, 2026-vla-research-brief

（大纲卡按 *-brief.md 命名，门禁自动豁免深度检查）

7 篇

---

## 全局统计

| 赛道 | 篇数 | 核心问题 |
|------|------|---------|
| Architecture | 30 | 动作怎么表示和生成 |
| Reasoning | 13 | 长程任务中的时序推理 |
| World Model | 23 | 如何建模环境动力学 |
| RL | 34 | 如何超越模仿学习上限（含 agentic 闭环与 LLM 侧训练系统） |
| Memory | 7 | 如何让机器人不忘记 |
| Data | 3 | 数据从哪里来 |
| Perception | 1 | 如何定位和建图 |
| Briefs | 7 | 综述索引与商业落地简报 |

**演进方向**：动作表示升级（频域/VQ/联合编码）→ 世界模型从被动预测变主动训练环境（WAM）→ RL 从 demo 走向自我改进 → 自我改进再升级为 agentic 闭环（coding agent 驱动 harness 自动化，HARBOR/ENPIRE 线）→ 记忆从外挂变成架构核心 → 3D 几何回归
