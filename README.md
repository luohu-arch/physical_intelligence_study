# VLA Research Study Library

Local study library for VLA, robot foundation model, and Physical AI papers. Automated discovery, download, note generation, and quality validation via the `pull-vla-research` skill.

## Quick Start

```bash
cd /path/to/physical-intelligence

# Pull papers from the watchlist — download PDFs + generate deep notes
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode watchlist --download --notes --deep

# Validate all notes pass the quality gate
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --validate
```

## Directory Structure

```
papers/
  curriculum/          Curriculum paper PDFs (RT-1, RT-2, OpenVLA, etc.)
  2026-05-10/          Incrementally pulled PDFs (date-stamped)
  2026-05-10-2/        Same-day re-pull (auto-suffixed)
notes/
  curriculum/          Deep-dive reading notes for curriculum papers
  2026-05-10/          Notes for incremental papers
tables/
  vla_research_watchlist.csv   Output table with paths and status
skills/pull-vla-research/
  references/watchlist.csv      Input watchlist — add new papers here
  scripts/pull_vla_papers.py    CLI entry point
  scripts/lib/                  Modular library (arxiv, notes, validate, etc.)
```

## Commands

### Pull Papers from Watchlist

```bash
# Download only (no notes)
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode watchlist --download

# Full pipeline: download + deep notes + backfill metadata
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode watchlist --download --notes --deep --backfill

# Flat mode — no date subfolder, outputs go directly to papers/ and notes/
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode watchlist --download --notes --flat

# Force re-download/re-generate (ignores incremental skip)
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode watchlist --download --notes --deep --force
```

### Search arXiv

```bash
# Search with custom queries
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode search --query "VLA robot manipulation 2025" --max-results 10 --download

# Combine watchlist + search
python3 skills/pull-vla-research/scripts/pull_vla_papers.py \
  --mode both --query "flow matching robotics" --download --notes --deep
```

### Backfill Existing Notes

```bash
# Add arXiv metadata (abstract, authors, published date) to notes that lack them
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --backfill
```

### Quality Gate

```bash
# Validate all notes against the 7-check gate
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --validate

# Validate a specific subdirectory
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --validate --notes-dir notes/curriculum
```

### Library Stats and Cross-References

```bash
# Paper count, category/priority breakdown, math & diagram coverage
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --stats

# Suggest cross-references between notes
python3 skills/pull-vla-research/scripts/pull_vla_papers.py --crossref
```

## Key Flags

| Flag | Effect |
|------|--------|
| `--mode watchlist` | Process papers from `references/watchlist.csv` |
| `--mode search` | Search arXiv with `--query` |
| `--mode both` | Watchlist + search |
| `--download` | Download PDFs from arXiv |
| `--notes` | Generate Markdown reading notes |
| `--deep` | Full 9-section deep notes (otherwise basic template) |
| `--backfill` | Fetch arXiv metadata and fill into existing notes |
| `--validate` | Run the quality gate (7 structural checks) |
| `--force` | Re-download/re-generate even if local files exist |
| `--flat` | No date-stamped subfolder |
| `--date-folder 2026-05-20` | Override date folder (default: today) |
| `--max-results N` | arXiv search result limit (default: 20) |
| `--min-year YYYY` | Earliest publication year for search (default: 2025) |

## Incremental Pulling

The script automatically skips papers that already have a valid PDF and note. To pull new papers:

1. Add new entries to `skills/pull-vla-research/references/watchlist.csv`
2. Run the pull command — only new entries are processed
3. If run again on the same day, date folder auto-suffixes: `2026-05-10-2`, `2026-05-10-3`, etc.

Use `--force` to bypass incremental skip and regenerate everything.

## Quality Gate Checks

1. All 9 deep-dive sections present
2. Zero `待补充` placeholders
3. At least 1 LaTeX math block
4. At least 1 mermaid architecture diagram
5. Mermaid blocks free of Unicode math characters
6. Trade-off comparison table
7. Valid local PDF path and arXiv metadata
