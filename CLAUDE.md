# Physical Intelligence — VLA Research Study Library

Local study library for Vision-Language-Action (VLA), robot foundation model, and Physical AI research.

## Repository Map

```
papers/
  curriculum/      — PDFs from vla.md core curriculum (RT-1, RT-2, OpenVLA, etc.)
  YYYY-MM-DD/      — Incrementally pulled PDFs, organized by pull date
notes/
  curriculum/      — Reading notes for curriculum papers
  YYYY-MM-DD/      — Reading notes for incrementally pulled papers
tables/            — CSV watchlists and paper matrices
plans/             — Reading plans and curricula
skills/            — Automation skills for paper discovery and organization
vla.md             — Master VLA research overview and curriculum
```

## Available Skills

- **pull-vla-research** (`skills/pull-vla-research/SKILL.md`) — Discover, download, and organize recent VLA/robotics papers from arXiv. Creates PDFs, Markdown notes, and CSV watchlists. Run via `python3 skills/pull-vla-research/scripts/pull_vla_papers.py`.

## Common Tasks

**Pull new papers from the watchlist:**
```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode watchlist --download --notes --deep
```

**Search arXiv for recent papers:**
```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode search --query "VLA robot manipulation flow matching" --max-results 10
```

**Add a new paper to track —** edit `skills/pull-vla-research/references/watchlist.csv`, add a row with `id, title, year, category, priority, why, arxiv_id, source_url`. Then run the pull command above.

**Backfill missing metadata into existing notes:**
```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --backfill
```

**Validate all notes pass the quality gate:**
```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --validate
```

**Regenerate everything (ignore incremental skip):**
```bash
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode watchlist --download --notes --deep --force
```

## Key Conventions

- Curriculum papers from `vla.md` live in `papers/curriculum/` and `notes/curriculum/`
- Incrementally pulled papers go in date-stamped subfolders (`papers/YYYY-MM-DD/`, `notes/YYYY-MM-DD/`)
- Never mix curriculum and incremental papers in the same directory
- Never overwrite user-annotated notes; append new sections
- Always verify arXiv IDs before downloading (similar VLA paper titles collide often)
- Use `references/scoring.md` rubric (1-5 scale) to filter borderline candidates
- Use Markdown and CSV as the source-of-truth study artifacts

## Content Integrity

- **Never fabricate content.** Every claim in notes must be traceable to the paper PDF, arXiv abstract, or official project page.
- **Mark uncertainty explicitly.** For information that cannot be confirmed from available sources, use `待确认` with the reason and source limitation. Examples:
  - `待确认：论文未公开此超参数，基于默认实践推测`
  - `待确认：仅从摘要推断，完整方法需阅读全文`
  - `待确认：商业声明，无独立评测验证`
- **Distinguish source quality.** Clearly separate: paper claims (from PDF), abstract summaries (from arXiv page), official blog posts, and third-party reports. Do not present blog claims as paper conclusions.
- **Prefer the paper.** When arXiv abstract and paper PDF conflict, the PDF is authoritative. When only the abstract is available, limit claims to what the abstract actually states.
- **Run `--validate` after writing notes** to catch structural issues before committing.

## Gotchas

- **Watchlist template vs output table are different files.** The input template (`skills/pull-vla-research/references/watchlist.csv`) has no `local_pdf`/`local_note` columns. The output table (`tables/vla_research_watchlist.csv`) is auto-generated with paths. Add new papers to the template, not the output table.
- **arXiv title collisions are common.** Multiple papers can have near-identical titles ("Unified VLA...", "VLA: A..."). Always verify via arxiv_id before downloading — never match by title alone.
- **Mermaid blocks cannot contain Unicode math symbols.** Characters like `√`, `ᾱ`, `α`, `β`, `σ`, subscripts (₀, ₁) cause parse errors in VS Code. Use ASCII equivalents or descriptive text instead.
- **LaTeX `\\` requires double backslash.** In Markdown code blocks, `\\` is rendered as-is, but inside `$$` math blocks, `\` is an escape character — use `\\\\` for line breaks in aligned environments.
- **`write_note` overwrites the entire file.** If a user has manually edited a note, re-running `--notes` will destroy those edits. Use `--force` only when you intend to regenerate from scratch. Incremental skip protects against accidental overwrites in normal runs.
- **Backfill fills empty fields only.** It won't replace existing content, but it also won't update stale metadata (e.g., if the arXiv abstract was revised). To refresh metadata, manually delete the section and re-run backfill.
