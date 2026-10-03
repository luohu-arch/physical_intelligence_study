---
name: pull-vla-research
description: Use when discovering recent VLA, robot foundation model, or Physical AI papers and organizing them into a local study library. Use when curating paper watchlists, creating Markdown reading notes for embodied AI research, or maintaining CSV paper matrices. Triggers on arXiv paper discovery, robotics literature review, or VLA/embodied AI research curation requests.
---

# Pull VLA Research

## Overview

Keep a local Physical AI / VLA paper library current. Prioritizes papers that extend the existing RT-1, PaLM-E, RT-2, Open X-Embodiment, Octo, OpenVLA, Diffusion Policy, Flow Matching, and FAST learning track.

**Type:** Flexible — adapt the workflow to the specific request, but always verify arXiv IDs and apply the scoring rubric before adding papers.

## When to Use

- Pulling new VLA/robotics papers from arXiv or watchlists
- Creating Markdown reading notes for embodied AI papers
- Updating the paper watchlist CSV
- Maintaining a local PDF library of robotics research
- Searching for recent Physical AI publications

**Skip when:** the request is about reading or annotating existing papers (no pulling needed), or the paper is outside physical-world robotics/embodied AI.

## Workflow Checklist

- [ ] Inspect workspace structure:
  - `papers/architecture/`, `papers/reasoning/`, `papers/world-model/` — track-organized PDFs (RT-1, π0, Dreamer, etc.)
  - `papers/YYYY-MM-DD/` — incrementally pulled PDFs
  - `notes/architecture/`, `notes/reasoning/`, `notes/world-model/` — reading notes by track
  - `notes/YYYY-MM-DD/` — notes for incrementally pulled papers
  - `tables/` — CSV paper matrices
  - `plans/` — reading plans
- [ ] Read `references/watchlist.csv` for the curated default paper list
- [ ] Run `scripts/pull_vla_papers.py` from the target workspace
- [ ] Prefer arXiv PDFs when available
- [ ] Create or update artifacts:
  - [ ] `papers/YYYY-MM-DD/<slug>_<arxiv-id>.pdf`
  - [ ] `notes/YYYY-MM-DD/<slug>.md`
  - [ ] `tables/vla_research_watchlist.csv`
- [ ] Only update `plans/reading_plan.md` after confirming new papers fit the learning path

## Quick Start

From the target study workspace:

```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --mode watchlist --download --notes --workspace .
```

Search arXiv for additional candidates:

```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --mode search --query "vision language action robot manipulation" --max-results 20 --workspace .
```

Run both curated watchlist and search:

```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --mode both --download --notes --workspace .
```

Backfill existing notes with arXiv metadata (abstracts, authors, published date) and local PDF paths:

```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --mode watchlist --backfill --workspace .
```

Generate deep-dive note templates (with math/physics/engineering sections) instead of basic templates:

```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --mode watchlist --download --notes --deep --workspace .
```

## Note Templates

Two note template levels are available:

| Flag | Template | Sections |
|------|----------|----------|
| `--notes` (default) | Basic | Why This Matters, Abstract, Reading Questions, Key Ideas, Architecture, Implementation, My Notes |
| `--notes --deep` | Deep Dive | 一句话总结, **九问速览**, 核心技术, 底层原理与数学推导, 物理直觉解释, 工程细节与实操指南, **实验协议清单**, 消融实验与分析, 技术权衡(Trade-off), 技术价值与演进定位, 与论文关系, 精读问题 |

Use `--deep` for high-priority papers that need detailed technical analysis. Use the basic template for screening candidates. Existing notes are never overwritten — use `--backfill` to fill metadata into already-created notes.

**After generating deep templates:** the AI must fill in each section by reading the paper PDF and/or extracting relevant content from `vla.md`. The `--backfill` mode only handles metadata (abstract, authors, PDF paths); the deep technical content requires AI analysis of the paper.


## Mermaid House Style（全库图表规范，v2）

所有深度笔记的 mermaid 架构图必须用以下规范绘制（在 GitHub/VS Code 原生渲染）：

**1) 主题头**（每个图开头，必填）：

```
%%{init: {
  'theme':'base',
  'themeVariables':{
    'primaryColor':'#fafbfd','primaryBorderColor':'#4a5d7d','primaryTextColor':'#1f2937',
    'fontFamily':'"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif',
    'fontSize':'14px','clusterBkg':'#fbfcfe','clusterBorder':'#b9c6d8','edgeLabelBackground':'#ffffff'
  },
  'flowchart':{'curve':'basis','nodeSpacing':26,'rankSpacing':42,'padding':10}
}}%%
```

**2) 语义配色 classDef**（按笔记事实上色，颜色是信息不是装饰）：

| 语义 | classDef | 用途 |
|---|---|---|
| data | `fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20` | 输入/数据/观测 |
| frozen | `fill:#e3f2fd,stroke:#1565c0,color:#0d47a1` | 预训练冻结模块 |
| train | `fill:#fff3e0,stroke:#ef6c00,stroke-width:2.5px,color:#e65100` | 可训练模块（加粗描边） |
| loss | `fill:#ffebee,stroke:#c62828,stroke-dasharray:6 3,color:#b71c1c` | 损失/监督信号（虚线） |
| act | `fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2.5px,color:#4a148c` | 动作输出路径 |
| loop | `fill:#eceff1,stroke:#546e7a,stroke-dasharray:4 3,color:#37474f` | 闭环/反馈/重规划 |
| env | `fill:#e0f2f1,stroke:#00695c,color:#004d40` | 环境/世界模型/仿真 |
| mem | `fill:#fffde7,stroke:#f9a825,color:#f57f17` | 记忆/历史/上下文 |
| reward | `fill:#fce4ec,stroke:#ad1457,color:#880e4f` | 奖励/评分/critic |
| key | `fill:#fff8e1,stroke:#ff8f00,stroke-width:3px,color:#e65100` | 本文核心 novelty（金框） |

**2.5) 边线型语义**（v3 图内底部"线型"图例行自动展示，只列图中实际用到的）：

| 线型 | 颜色 | 语义 |
|---|---|---|
| 粗实线 4.4px | #263238 深灰 | 主数据流（mermaid `==>`） |
| 实线 3.2px | #546e7a 中灰 | 一般信息流（`-->`） |
| 细实线 2.2px | #9fb0bb 浅灰 | 弱关联（`---`） |
| 长虚线 12-7 | #1565c0 蓝 | 闭环反馈：执行→观测回灌（`-.->` 无训练语义） |
| 点划线 3-4.5-11-4.5 | #c62828 红 | 损失/监督/蒸馏/TD/梯度（`-.->` 标签含关键词时自动归入） |

连线一律**直角折线**（对向=直线/肘形，同侧=U 形环绕，相邻侧=L 形），转折段自动避让卡片；箭头固定 13px 不随线宽缩放。

规则：**实线=前向数据流（推理时存在），虚线=非数据流信号**；边标签药丸描边与文字跟随边色。自动转换的 loss 关键词：蒸馏|损失|TD|监督|梯度|优化|反传|学习信号|expectile|n-step|回传。

**3) 结构规范**：`flowchart LR/TD`（管线横排、层级竖排）；subgraph 分组建框（模块/阶段/双系统）；主数据流 `==>` 粗箭头；反馈/监督 `-.->` 虚线并带边标签；端点用体育场形 `([文本])`；数据库形 `[(名称)]` 只用于数据集。

**4) 硬约束**：节点/标签只用中文与 ASCII；禁用希腊字母与数学 Unicode（写 pi/alpha/beta，不写 π/α/β）；数字下标用 x0/x_t 形式。门禁 mermaid-safe 会拦截。

## v3 SVG 架构图（全库默认，2026-10 起全量铺开）

深度笔记的架构图一律用 v3 SVG（图标 + 渐变 + 阴影，观感接近论文原图）；mermaid v2 只用于 briefs 大纲卡与快速草图。全库已由 `scripts/mermaid_to_spec.py` 从 v2 mermaid 自动转换完成（每篇的 `figures/<id>/arch.spec.json` 含布局 spec 与原始 mermaid 备份，可手调后用 fancy_diagram 重渲染）。

**1) 工具链**：`scripts/mermaid_to_spec.py`（mermaid→spec 自动转换+分层布局+图标分配，`--dry`/`--all`）+ `scripts/fancy_diagram.py`（spec→SVG，手调用）+ 图标库 `assets/ml-paper-icons/`（512 个 SVG，ISC 许可，duotone/lucide/phosphor/tabler 四族，已 vendor 含 LICENSE）。自动转换：

```bash
python3 skills/pull-vla-research/scripts/fancy_diagram.py notes/<track>/figures/<id>/arch.spec.json --png
```

spec 是 JSON：title/subtitle/foot、canvas、panels（虚线分组）、nodes（label/sub/icon/cls/x/y/w/h/tag）、edges（from/to/style/label + 锚点 out/in/pos 与偏移 loff 可省略自动判断）、legend。语义类沿用 v2 十类同色系；图标用语义名（brain/lightning/robot/target/db/chart/state/flask/noise…，别名表在脚本 ALIASES）。

**2) 嵌入格式**（替换原 mermaid 块，spec 与 arch.svg 存 `notes/<track>/figures/<id>/`（PNG 仅按需 `--png` 生成用于目检，不入库））：

```
![<id> 架构图 v3](figures/<id>/arch.svg)

*架构速览：<一句话，主数据流 + 关键机制>*
```

门禁 diagram 检查已接受 mermaid 或 v3 SVG 两者之一；文本层面 SVG 同样可 diff。

**3) 验证流程**（必须全过再嵌入）：`xml.etree` 解析过 → `qlmanage -t -s 1600` 或本地 http 服务 + 浏览器截图（GitHub 渲染的 ground truth 是浏览器）→ 视觉模型只问 BAD（文字溢出/箭头穿卡/图标缺失/布局空洞）。soffice 对部分 SVG 加载失败，不要用它做 SVG 校验。

**4) 布局经验**：列间距 ≥100px（箭头通道，防"穿卡"观感）；长反馈回路用 out bottom/in bottom + k≥70 压底走线；标签药丸放不进缝隙时用 loff 挪到通道外侧；同通道双向边（如 Q⇄蒸馏）错开 pos_out/pos_in；CJK 宽度系数 1.06（粗体实测）；同行节点 ≥5 个时间距 ×1.4。

**5) 外部画图 skill 路线图**（2026-10 评估结论）：
- K-Dense `pptx-posters`：强门禁 manifest 审批工作流，面向**印刷级会议海报**（打印/无障碍/包安全检查），不适合笔记内嵌图；真要做 poster 时再启用。
- 本机 `presentations:pptx`（python-pptx）+ `soffice` 转 PNG：适合做汇报 slide / 求职材料里的框架图，不做笔记。
- 本机 `tikz-figure-code` / `thesis-figure-skill`：写论文时的 TikZ 路线。
- 笔记内嵌图一律 mermaid v2（默认）或 v3 SVG（旗舰），不引入 PPT/图片依赖。

## Reading Methodology (阅读方法论)

**阅读顺序**（不要从第一页逐字读）：Abstract → Figure 1 → Introduction → Main Results → Experimental Setup → Method Overview → Method Details → Ablation → Failure Cases → Appendix（第一档）→ Related Work。第一遍 10–20 分钟只求填出 Problem→Insight→Mechanism→Evidence 链条；讲不出这四环就不钻公式。

**九问速览**（每篇必填，插在一句话总结之后）：

```markdown
## 九问速览

1. **Problem**：
2. **Bottleneck**：
3. **Insight**：
4. **Method**：
5. **Evidence**：
6. **Ablation**：
7. **Assumption**：
8. **Failure**：
9. **Opportunity**：

| 维度 | 论文答案 |
|---|---|
| Perception |  |
| Closed-loop |  |
| Correction |  |
| Deployment |  |
```

机器人四问比网络结构更重要：模型知道多少环境信息（Perception）、失败能否发现（Closed-loop）、发现后能否重规划（Correction）、训练与部署条件差距（Deployment）。

**实验协议清单**（每篇必填，插在工程细节之后；数据主要来自**附录**第一档内容）：

15 项：观测 / 动作空间 / 控制频率 / 重规划频率 / 动作 horizon / 数据 / 奖励 / Reset / 成功定义 / 评估次数 / 随机种子 / 扰动测试 / 真机 / 算力 / 特权信息。查不到的写「未报告」，**严禁编造**。表后附「附录陷阱自查」六行：privileged 信息、reward shaping、reset 难度、eval budget、底层控制栈、数据优势——机器人论文"看起来是算法提升、实际是 protocol 不一样"大多藏在这六处。

**附录三档法**：第一档必看（implementation/hyperparameters/eval protocol/reward/observation-action space/control frequency/success criteria/seeds/additional ablations/failure cases）；第二档复现或 follow-up 时再看（逐层架构/prompt 模板/完整 task list）；第三档可跳（长证明/大量定性图）。判断标准：不看这节会不会误判论文结论。**正文决定 insight，附录决定可信度。**

**Method 拆法**：按 Input→Representation→Decision→Action→Training Signal 信息流拆，先问哪些 pretrained/frozen/训练、loss 是什么、train/infer 是否一致，再读公式；公式只精读 objective、policy 定义、体现 novelty 的三类。**消融比 SOTA 表更值得细看**——检查证据是否支持机制（No X / Correct X / Shuffled X / 长短 horizon 对照），只有 72%→76% 的整体提升不算证明。

## Quality Gate

Validate note quality before committing to the study library:

```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --validate --workspace .
```

The gate checks every note for:
- [x] All 12 deep-dive sections present (含 九问速览、实验协议清单)
- [x] Zero `待补充` placeholders
- [x] Mermaid architecture diagram present
- [x] Trade-off table in 技术权衡 section
- [x] Local PDF path filled
- [x] arXiv link and year metadata
- [i] LaTeX math presence (informational only)

Non-paper entries (commercial briefs, overviews) get relaxed checks. Run `--validate` after writing or updating notes to ensure quality consistency.

## Selection Criteria

Prioritize papers that contribute at least one of:

- **New VLA architectures:** unified action-language modeling, dual/triple systems, reasoning/acting integration
- **Better action representation:** FAST-like tokenization, latent actions, flow/diffusion action heads
- **Better physical grounding:** embodied reasoning, spatial reasoning, tactile/contact-rich manipulation, foresight/world modeling
- **Better data scaling:** Open X-Embodiment extensions, human video pretraining, synthetic data, self-improvement loops
- **Better adaptation:** one-shot/few-shot skill adaptation, cross-embodiment adaptation, RL post-training
- **Better deployability:** smaller efficient policies, open weights/code, lower training/inference cost

Deprioritize papers that are only tangentially about agents, generic VLMs, or non-robotic multimodal benchmarks unless they explicitly improve physical-world action.

## Common Mistakes

- **Wrong paper downloaded:** Similar VLA paper titles can collide. Always verify arXiv IDs before downloading.
- **Mixing curriculum and incremental papers:** Curriculum papers from `vla.md` belong in track folders (`papers/architecture/`, `papers/reasoning/`, `papers/world-model/`). Incremental pulls go in date-stamped subfolders.
- **Overwriting user notes:** Do not overwrite existing notes with user annotations unless explicitly requested; append new sections instead.
- **Adding off-topic papers:** Papers about generic VLMs or non-robotic benchmarks should be filtered out unless they explicitly improve physical action.
- **Ignoring memory hints:** The script learns from your behavior. If it suggests `--deep` or `--backfill`, those flags have been useful in past sessions.

## Memory System

The script maintains a `.memory/preferences.json` file that learns from your behavior across sessions:

| What It Learns | How |
|---------------|-----|
| Note depth preference | Tracks whether you use `--deep` |
| Auto-download preference | Tracks whether you use `--download` |
| Category priorities | Counts which paper categories you pull most often |
| Auto-boosted papers | New papers in your frequently-read categories get a priority boost |

**How it works:**
- Each run records your flag usage and paper selections
- On the next run, the script prints suggestions for missing flags
- Papers in your top-3 most-read categories get auto-boosted priority
- Memory is stored in `.memory/preferences.json` — never blocks, always has safe defaults
- Delete `.memory/` to reset all learned preferences

## Resources

- `references/watchlist.csv` — curated high-value recent papers and technical reports
- `references/scoring.md` — rubric for deciding whether a new candidate belongs in the learning library (1-5 scale)
- `scripts/pull_vla_papers.py` — deterministic downloader, note generator, and metadata backfiller (`--backfill` mode)
