---
name: evaluate-paper-value
description: Use before batch-downloading papers to evaluate whether each work is worth the time investment of deep reading and note-taking. Triggers when the user asks to pull papers in bulk, evaluate paper quality, filter a paper list, or decide whether specific papers merit downloading. Use proactively when the user mentions "pull papers", "batch download", "which papers are worth reading", or similar curation phrases.
---

# Evaluate Paper Value

## Overview

Before downloading and analyzing papers at scale, evaluate each candidate across four dimensions to decide whether it's worth the time investment. This prevents pulling low-value papers that waste storage, compute, and — most importantly — the user's attention.

**Type:** Flexible — adapt scoring thresholds to the specific research context, but always evaluate all four dimensions before making a pull decision.

## When to Use

- Before running `--mode watchlist --download` or `--mode search --download` to filter candidates
- When the user provides a list of arXiv links or paper titles and asks which are worth reading
- When the user says "pull latest papers" or "batch download" without already curating the list
- When evaluating a single unfamiliar paper the user is considering adding

**Skip when:** the user has already confirmed specific papers to pull, or the list is small (< 3 papers) and all are clearly relevant.

## Evaluation Framework

For each paper, score across four dimensions on a 1-5 scale (1 = lowest, 5 = highest). The overall score determines the action.

### Dimension 1: Peer Review & Publication Venue

How trustworthy is this work? Peer-reviewed papers have survived scrutiny; company tech reports and arXiv-only preprints haven't.

| Score | Criteria |
|-------|----------|
| 5 | Published at top venue (NeurIPS, ICML, ICLR, CVPR, RSS, CoRL, ICRA with oral) |
| 4 | Published at solid venue (ICRA, IROS, ECCV, T-RO, IJRR, RAL, AAAI) |
| 3 | arXiv only but from established academic lab with consistent publication record |
| 2 | arXiv only from industry team with no prior peer-reviewed work in this area |
| 1 | Company blog post, tech report without open weights/data, or unknown source |

**Red flag:** Company tech reports that announce products, not research. These are marketing, not science — skip unless they define a genuinely new paradigm with open artifacts.

### Dimension 2: Impact & Community Reception

Is this paper influential, or just another incremental tweak?

| Score | Criteria |
|-------|----------|
| 5 | 500+ citations, widely used as baseline, defines a named paradigm (e.g., Dreamer, JEPA, RT-2) |
| 4 | 100-500 citations, strong community interest, spawns follow-up work |
| 3 | 10-100 citations, cited by known labs, discussed in survey papers |
| 2 | < 10 citations, low engagement, no notable follow-ups |
| 1 | Zero citations, no code, no discussion anywhere |

**Note on recency:** Papers from the last 6 months naturally have fewer citations. For these, check: (a) are well-known labs building on it? (b) is it discussed at seminars/conferences? (c) does it appear in recent survey papers? A 2026 paper with 5 citations but already cited by top labs is a 4, not a 2.

### Dimension 3: Methodological Contribution

Is this a genuine idea or an engineering assembly?

| Score | Criteria |
|-------|----------|
| 5 | Introduces a new paradigm or fundamental insight that reorients the field (e.g., JEPA's latent prediction, flow matching for actions, Dreamer's RSSM) |
| 4 | Significant architectural innovation or novel combination that meaningfully advances the state of the art |
| 3 | Solid improvement over existing methods with clear ablation showing what matters |
| 2 | Incremental tweak — new dataset applied to existing method, hyperparameter tuning, or "we added module X to model Y" |
| 1 | Pure engineering — bigger model, more data, no new idea. Or worse: rebranding existing work. |

**Key heuristic:** If you remove this paper from the literature, does the field lose a distinct direction? If yes → 4-5. If another paper would have filled the same gap within months → 2-3. If nothing would change → 1.

### Dimension 4: Research Alignment

Does this paper advance the user's specific research interests?

| Score | Criteria |
|-------|----------|
| 5 | Directly on the core track: VLA architectures, robot foundation models, embodied world models, action representation |
| 4 | Strong relevance: robot manipulation, visuomotor policy learning, cross-embodiment transfer |
| 3 | Adjacent but applicable: video prediction for robotics, 3D scene understanding for manipulation, human-to-robot transfer |
| 2 | Tangential: general computer vision, NLP, generic multimodal models without robotics evaluation |
| 1 | Irrelevant: no physical action, no robot experiments, no embodied component |

**Must have robot experiments:** Papers claiming "embodied" or "physical" relevance should evaluate on robot benchmarks (LIBERO, CALVIN, SimplerEnv, DROID, real robots) or standard embodied simulators. Papers that only evaluate on video/image benchmarks while claiming robotics relevance are at most a 2.

## Decision Matrix

Combine the four scores into an overall judgment:

| Overall Score | Action | Meaning |
|---------------|--------|---------|
| 16-20 | **Must Pull** | High-value work. Download PDF, write deep-dive notes, add to curriculum. |
| 11-15 | **Pull with Context** | Worth reading. Pull PDF, write basic notes, but don't prioritize deep analysis unless it connects to an active project. |
| 6-10 | **Watchlist Only** | Borderline. Add to watchlist CSV for potential future reading, don't download now. |
| 1-5 | **Skip** | Not worth the disk space or attention. Move on. |

**Hard Gates** — any of these automatically cap the action at "Watchlist Only" regardless of overall score:
- Company tech report without peer review AND without open model weights → max "Watchlist Only"
- Zero robot experiments while claiming robotics relevance → max "Watchlist Only"
- Claims contradicted by community consensus or known benchmark results → Skip

## Workflow

When evaluating a batch of candidate papers:

1. **Quick triage** — For each paper, check title + abstract + venue. Reject obvious mismatches immediately (wrong domain, marketing fluff).
2. **Score surviving candidates** — Run the full 4-dimension evaluation on papers that pass triage.
3. **Sort by overall score** — Pull in descending order. Time is finite; start with the most valuable.
4. **Report the decisions** — Output a clear table showing the score breakdown and action for each paper.
5. **Flag uncertain cases** — If scoring is ambiguous (e.g., can't determine peer review status), mark with reason and defer to the user.

### Output Format

Present decisions in a table:

```
| # | Paper | Venue | Impact | Method | Align | Total | Action |
|---|-------|-------|--------|--------|-------|-------|--------|
| 1 | Dreamer v3 | 5 | 5 | 5 | 5 | 20 | Must Pull |
| 2 | Some Paper  | 2 | 3 | 3 | 4 | 12 | Pull with Context |
| 3 | Weak Paper  | 1 | 2 | 2 | 3 | 8  | Watchlist Only |
```

Add a one-line rationale for each decision, especially for Skip and Watchlist Only.

## Integration with pull-vla-research

When the user asks to do a bulk pull:

1. **Run evaluation first** — Apply this framework to the candidate list BEFORE invoking `pull_vla_papers.py`.
2. **Filter the watchlist** — Remove Skip candidates from the download queue. Only pass Must Pull and Pull with Context papers to the download script.
3. **Prioritize Must Pull** — Use `--deep` notes for Must Pull papers, basic notes for Pull with Context.
4. **Update watchlist** — Track Watchlist Only papers in the CSV for potential future reading.

This skill's scoring is a pre-filter. The `scoring.md` rubric in `pull-vla-research/references/` is the post-download quality gate. Use both.

## Concrete Examples

These examples encode the evaluation standards from past sessions. Use them to calibrate scoring.

### Must Pull (16-20)

| Paper | Breakdown | Why |
|-------|-----------|-----|
| Dreamer v3 | 5+5+5+5=20 | RSSM + world model learning from pixels is a foundational paradigm for robot world models. Published, highly cited, direct relevance. |
| I-JEPA | 5+5+5+5=20 | CVPR 2023. Defined the JEPA paradigm — a third path beyond contrastive and generative SSL. Directly influences robot world model design. |
| π0 | 4+5+5+5=19 | Flow matching for action generation is a paradigm shift. Not yet peer-reviewed but from Physical Intelligence (established lab) with open model. |

### Pull with Context (11-15)

| Paper | Breakdown | Why |
|-------|-----------|-----|
| π0.5 | 3+3+4+5=15 | Important co-training insight (97.6% non-target data), but incremental over π0. Solid engineering contribution. |
| V-JEPA | 4+4+4+4=16 → capped at 15 (no robot experiments) | Extends I-JEPA to video. Strong method, but evaluates on video benchmarks only — robotics relevance is theoretical. |
| TD-MPC2 | 4+4+4+4=16 → 15 | Decoder-free world model + MPC. Published, solid method, but more relevant to locomotion than manipulation. |
| UniPi | 4+3+4+5=16 → 15 | Video-as-policy is a distinct paradigm, but only evaluated on simpler benchmarks. Important idea, limited validation. |

### Watchlist Only (6-10)

| Paper | Breakdown | Why |
|-------|-----------|-----|
| SuSIE | 4+3+3+4=14 → capped at 10 (engineering assembly) | ICLR 2024 but core idea is "InstructPix2Pix + policy" — an assembly, not a new method. Useful as context for video-as-policy, not a standalone contribution. |
| DayDreamer | 3+3+3+4=13 → capped at 10 (incremental) | Real-robot Dreamer deployment is practical but methodologically incremental. Good engineering, limited novelty. |
| GR-MG | 3+2+3+5=13 → capped at 10 (engineering assembly) | GR-1 + goal generation + action chunking = assembly of existing components. Useful as baseline reference for 3D Foresight. |

### Skip (1-5)

| Paper | Breakdown | Why |
|-------|-----------|-----|
| G0.5 (Galaxea) | 1+2+2+4=9 → Skip (hard gate: company tech report, no peer review, no open model) | Company product announcement dressed as research. No peer review, closed artifacts, claims unverifiable. Even though alignment score is high, the lack of scientific rigor makes this marketing, not research. |

Note on G0.5: The hard gate triggers because it's a company tech report without peer review AND without open model weights. The 27D ActionCodec and Native CoT ideas are interesting, but until peer-reviewed or released with verifiable artifacts, this is a press release — not a research paper. Track for awareness, never pull.

## Key Principles

1. **Peer review is the strongest signal.** A NeurIPS/ICLR/CVPR/RSS paper has survived months of expert scrutiny. An arXiv-only preprint hasn't. A company blog post actively *wants* you to not scrutinize it.

2. **Ideas > engineering.** A small paper with one genuinely new idea is worth more than a massive engineering effort with no intellectual contribution. The field moves forward through ideas, not through parameter counts.

3. **Robotics requires robot experiments.** A paper claiming to advance "embodied AI" must actually evaluate on embodied tasks. Video prediction and language benchmarks don't count.

4. **When in doubt, watchlist.** It's cheap to track a paper for later; it's expensive to read and take notes on the wrong paper. Default to conservative.

5. **Calibrate for recency.** A 2026 paper cannot have Dreamer v3-level citations. Judge it by who is paying attention, not just citation count.
