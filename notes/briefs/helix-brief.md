# Figure Helix（01 / 02 / 2.5）：商业 VLA 全身控制线（大纲卡）

- Source：https://www.figure.ai/news/helix（Helix 01）、Helix 02（2026-01，webuildhumanoids/figure news）、Helix 2.5（2026，figure.ai/news/helix-2-5-zero-shot-30-home-generalization）
- Project：https://www.figure.ai
- Category：commercial physical AI / no open paper（tech report + blog）
- Priority：high
- 拉取日期：2026-10-03

## 一句话总结

Figure 的 Helix 系列是商业 VLA 全身控制的代表线：**Helix 01**（2025-02，首个对整个人形上半身——腕/指/头——输出高频连续控制的 VLA）；**Helix 02**（2026-01，扩展到全身，新增学习式全身控制器 System 0，搭载 Figure 03）；**Helix 2.5**（2026，单一基础模型驱动三种全身行为，零样本泛化到 30 个真实家庭，提出 index pretraining）。

## 技术要点（来自 tech report 与官方披露）

- **双系统架构**（Helix 01）：System 2 = ~7B VLM，7-9 Hz 慢速场景/语言理解；System 1 = ~80M 潜动作解码器，200 Hz 高速全身控制——与库内 GR00T N1、OneTwoVLA、TriVLA 的双/三系统线同构，但工业落地最激进。
- **System 0**（Helix 02）：学习式全身底层控制器（行走/平衡/接触），把 VLA 从上半身扩展到 loco-manipulation——与库内 WholeBodyVLA 的分层频率设计（VLA 10Hz + LMO 50Hz）可对照。
- **Index pretraining**（Helix 2.5）：面向家庭泛化的预训练组织方式（细节未公开）。
- **零样本家庭泛化**：30 个真实家庭、未见环境直接执行——库内机器人四问的 Deployment 维度的最强商业证据之一。

## 证据与开放性评估

- **无 arXiv 论文、无开放权重/数据**——所有数字（200Hz、30 家庭、7B+80M）来自官方 tech report，无第三方独立评测。
- demo 口碑极好（F.03 排序 demo 2026-05 引发病毒式讨论；HN 有技术讨论帖），但按库规归入"营销+真范式"档：双系统+全身控制范式真实，具体指标不可验证。
- 复现锚点：学术侧用 GR00T N1（开放）+ π0.7 系列替代验证同类设计。

## 与本库的关系

- **gr00t-n1 / onetwovla / trivla**：双/三系统 VLA 的开放对照。
- **wholebodyvla**：分层频率全身控制的学术对照（System 0 ≈ LMO 层）。
- **pi0 / pi05 / pi07**：Physical Intelligence 同为商业 VLA 标杆，但 π 系列有论文+开放权重，证据等级高一档。
- **gen-1 / gene-26-5**：同档商业 brief 卡（无论文，指标不可验）。
