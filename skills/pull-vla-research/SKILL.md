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
  - `papers/curriculum/` — PDFs from `vla.md` core curriculum (RT-1, RT-2, OpenVLA, etc.)
  - `papers/YYYY-MM-DD/` — incrementally pulled PDFs
  - `notes/curriculum/` — reading notes for curriculum papers
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
| `--notes --deep` | Deep Dive | 一句话总结, 核心技术, 底层原理与数学推导, 物理直觉解释, 工程细节与实操指南, 技术权衡(Trade-off), 技术价值与演进定位, 与论文关系, 精读问题 |

Use `--deep` for high-priority papers that need detailed technical analysis. Use the basic template for screening candidates. Existing notes are never overwritten — use `--backfill` to fill metadata into already-created notes.

**After generating deep templates:** the AI must fill in each section by reading the paper PDF and/or extracting relevant content from `vla.md`. The `--backfill` mode only handles metadata (abstract, authors, PDF paths); the deep technical content requires AI analysis of the paper.

## Quality Gate

Validate note quality before committing to the study library:

```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --validate --workspace .
```

The gate checks every note for:
- [x] All 9 deep-dive sections present
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
- **Mixing curriculum and incremental papers:** Curriculum papers from `vla.md` belong in `papers/curriculum/` and `notes/curriculum/`. Incremental pulls go in date-stamped subfolders.
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
