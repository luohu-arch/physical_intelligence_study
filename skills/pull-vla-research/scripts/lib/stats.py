"""Statistics for the VLA research study library.

Computes paper counts by category, priority, and year, as well as
note-level metrics such as math blocks, mermaid diagrams, and
reading progress (deep vs. basic notes).
"""

from __future__ import annotations

from pathlib import Path

from .workspace import load_watchlist


def compute_stats(workspace: Path) -> str:
    """Compute and return a formatted summary of workspace statistics.

    Gathers the following metrics:
      - Papers grouped by category, priority, and publication year
      - Total notes, split into deep vs. basic
      - Count of ``$$`` math blocks and ``mermaid`` diagrams across all notes
      - Reading progress (notes with deep content vs. total)

    Args:
        workspace: Root path of the study workspace.

    Returns:
        A multi-line string with the formatted statistics summary.
    """
    lines: list[str] = []
    lines.append("=" * 50)
    lines.append("Study Library Statistics")
    lines.append("=" * 50)

    # -- Watchlist-based counts --
    watchlist_path = workspace / "tables" / "vla_research_watchlist.csv"
    if watchlist_path.exists():
        rows = load_watchlist(watchlist_path)

        # By category
        categories: dict[str, int] = {}
        for row in rows:
            cat = row.get("category", "unknown") or "unknown"
            categories[cat] = categories.get(cat, 0) + 1

        # By priority
        priorities: dict[str, int] = {}
        for row in rows:
            pri = row.get("priority", "unknown") or "unknown"
            priorities[pri] = priorities.get(pri, 0) + 1

        # By year
        years: dict[str, int] = {}
        for row in rows:
            yr = row.get("year", "unknown") or "unknown"
            years[yr] = years.get(yr, 0) + 1

        lines.append(f"\nTotal papers in watchlist: {len(rows)}")
        lines.append("")
        lines.append("  By category:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            lines.append(f"    {cat}: {count}")

        lines.append("")
        lines.append("  By priority:")
        for pri, count in sorted(priorities.items(), key=lambda x: -x[1]):
            lines.append(f"    {pri}: {count}")

        lines.append("")
        lines.append("  By year:")
        for yr, count in sorted(years.items(), key=lambda x: -x[1]):
            lines.append(f"    {yr}: {count}")
    else:
        lines.append("\n  (no watchlist.csv found — skipping paper-level stats)")

    # -- Note-level metrics --
    notes_dir = workspace / "notes"
    total_notes = 0
    deep_notes = 0
    basic_notes = 0
    total_math_blocks = 0
    total_mermaid = 0

    if notes_dir.exists() and notes_dir.is_dir():
        note_files = sorted(notes_dir.rglob("*.md"))
        note_files = [f for f in note_files if not f.name.startswith(".")]

        for nf in note_files:
            total_notes += 1
            text = nf.read_text(encoding="utf-8")

            # Count math blocks (paired $$)
            math_count = text.count("$$") // 2
            total_math_blocks += math_count

            # Count mermaid diagrams
            mermaid_count = text.count("```mermaid")
            total_mermaid += mermaid_count

            # Heuristic: deep notes contain "核心技术" and "底层原理与数学推导"
            has_deep_sections = "核心技术" in text and "底层原理与数学推导" in text
            if has_deep_sections:
                deep_notes += 1
            else:
                basic_notes += 1

    lines.append(f"\nNotes: {total_notes}")
    lines.append(f"  Deep notes: {deep_notes}")
    lines.append(f"  Basic notes: {basic_notes}")
    reading_progress = f"{deep_notes}/{total_notes}" if total_notes else "0/0"
    lines.append(f"  Reading progress: {reading_progress} with deep content")
    lines.append(f"  Total math blocks ($$ pairs): {total_math_blocks}")
    lines.append(f"  Total mermaid diagrams: {total_mermaid}")

    # -- Closing --
    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)
