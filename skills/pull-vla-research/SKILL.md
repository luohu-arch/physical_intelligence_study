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
