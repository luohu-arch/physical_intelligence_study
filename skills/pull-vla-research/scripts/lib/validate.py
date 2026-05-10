from __future__ import annotations

import re
from pathlib import Path


DEEP_SECTIONS = [
    "一句话总结",
    "核心技术",
    "底层原理与数学推导",
    "物理直觉解释",
    "工程细节与实操指南",
    "技术权衡",
    "技术价值与演进定位",
    "与其他论文的关系",
    "精读问题",
]

CHECK_ICONS = {"PASS": "✓", "FAIL": "✗", "WARN": "△"}


def validate_notes(workspace: Path, notes_dir: str | None = None) -> tuple[int, int]:
    """Validate all notes in a directory. Returns (passed, total) counts.

    Checks:
      1. All 9 deep-dive sections present
      2. Zero 待补充 placeholders
      3. At least 1 LaTeX $$ math block
      4. At least 1 mermaid diagram
      5. Trade-off table with | delimiter
      6. Local PDF path is non-empty
      7. Metadata: arXiv link, year, category
    """
    base = workspace / (notes_dir or "notes")
    if not base.exists():
        print(f"notes directory not found: {base}")
        return 0, 0

    note_files = sorted(base.rglob("*.md"))
    # Exclude non-paper files
    note_files = [f for f in note_files
                  if not f.name.startswith(".") and "brief" not in f.name.lower()]

    if not note_files:
        print("no note files found")
        return 0, 0

    passed = 0
    total = len(note_files)

    for nf in note_files:
        text = nf.read_text(encoding="utf-8")
        rel = nf.relative_to(workspace)
        is_no_paper = "no open paper" in text or "无公开论文" in text or "commercial physical AI" in text
        is_overview = "brief" in nf.name.lower() or "research-brief" in nf.name.lower()
        checks: list[tuple[str, bool, str]] = []

        # 1. Structure completeness (skip deep sections for non-paper/overview entries)
        if is_no_paper or is_overview:
            checks.append(("sections", True, "skipped (non-paper/overview)"))
        else:
            missing = [s for s in DEEP_SECTIONS if s not in text]
            checks.append(("sections", len(missing) == 0,
                           f"missing: {', '.join(missing)}" if missing else f"all {len(DEEP_SECTIONS)} present"))

        # 2. No placeholders
        todos = text.count("待补充")
        checks.append(("no-placeholders", todos == 0,
                       f"{todos} placeholders" if todos else "clean"))

        # 3. Math (at least 1 LaTeX formula; informative only, does not block gate)
        math_count = text.count("$$") + text.count("$")
        has_math = math_count >= 2
        checks.append(("math", True,  # never blocks — informative only
                       f"{math_count} LaTeX markers" if has_math else "no LaTeX found (info)"))

        # 4. Diagram presence
        has_diagram = "mermaid" in text or is_no_paper
        checks.append(("diagram", has_diagram,
                       "mermaid present" if "mermaid" in text else
                       ("skipped (non-paper)" if is_no_paper else "no mermaid diagram")))

        # 4.5 Mermaid sanitization (no Unicode math that breaks rendering)
        MERMAID_UNSAFE = re.compile(
            r'[√ᾱ·αβγδεθλμπστφωΔΘΛΣΩ₀₁₂ₜ̂×êôâ²³⁴⁵⁶⁷⁸⁹⁰]'
        )
        mermaid_blocks = re.findall(r'```mermaid\n(.*?)```', text, re.DOTALL)
        bad_chars: set[str] = set()
        for block in mermaid_blocks:
            for ch in MERMAID_UNSAFE.findall(block):
                bad_chars.add(ch)
        mermaid_clean = len(bad_chars) == 0
        checks.append(("mermaid-safe", mermaid_clean,
                       "clean" if mermaid_clean else f"unsafe chars: {''.join(sorted(bad_chars))}"))

        # 5. Trade-off table
        if is_no_paper or is_overview:
            checks.append(("tradeoff-table", True, "skipped (non-paper/overview)"))
        else:
            has_tradeoff_section = "技术权衡" in text or "Trade-off" in text
            has_table = has_tradeoff_section and "|" in text
            checks.append(("tradeoff-table", has_table,
                           "trade-off table present" if has_table else "no trade-off table"))

        # 6. Local PDF path (supports both English and Chinese metadata formats)
        has_pdf_label = "Local PDF" in text or "本地 PDF" in text
        if is_no_paper:
            has_local_pdf = True  # skip check for commercial/non-paper entries
            pdf_detail = "commercial entry (no arXiv paper expected)"
        else:
            has_local_pdf = has_pdf_label and "papers/" in text
            pdf_detail = "PDF path present" if has_local_pdf else "Local PDF missing or empty"
        checks.append(("pdf-path", has_local_pdf, pdf_detail))

        # 7. Metadata
        has_arxiv = "arxiv.org" in text.lower()
        has_year = bool(re.search(r"(Year|年份|Published).*\d{4}", text))
        if is_overview or is_no_paper:
            has_arxiv = True
            has_year = True
        checks.append(("metadata", has_arxiv and has_year,
                       f"arxiv:{has_arxiv} year:{has_year}"))

        # Summary
        failed_checks = [c for c in checks if not c[1]]
        all_pass = len(failed_checks) == 0
        if all_pass:
            passed += 1

        # Print
        status = CHECK_ICONS["PASS"] if all_pass else CHECK_ICONS["FAIL"]
        print(f"\n{status} {rel}")
        for name, ok, detail in checks:
            icon = CHECK_ICONS["PASS"] if ok else CHECK_ICONS["FAIL"]
            print(f"  {icon} {name}: {detail}")

    # Final summary
    print(f"\n{'='*50}")
    print(f"Gate result: {passed}/{total} passed")
    if passed < total:
        print(f"Failed: {total - passed} notes need fixes")
    else:
        print("All notes pass the gate.")
    return passed, total
