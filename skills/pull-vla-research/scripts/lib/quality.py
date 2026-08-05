"""Note quality grading — checks beyond section presence.

Usage: python -m skills.pull-vla-research.scripts.lib.quality [workspace]
Outputs per-note quality grade (A/B/C/D) and flags concrete issues.
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class QualityReport:
    note: str
    grade: str  # A, B, C, D
    score: int  # 0-100
    flags: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        icon = {"A": "★", "B": "▲", "C": "●", "D": "✗"}[self.grade]
        flags_str = "\n    ".join(self.flags) if self.flags else "none"
        return f"{icon} {self.note} [{self.grade}] score={self.score}\n  flags: {flags_str}"


# ── Section-level criteria ──────────────────────────────────────────

def _check_ablation_quality(text: str) -> tuple[int, list[str]]:
    """Checks ablation section has quantitative data, not just hand-waving."""
    section = _extract_section(text, "消融实验与分析")
    if not section:
        return 0, ["消融章节缺失"]

    lines = [l for l in section.split("\n") if "|" in l and l.strip().startswith("|")]
    # Count numeric cells (contains digit or %)
    numeric_cells = sum(
        1 for l in lines
        for cell in l.split("|")[1:-1]
        if re.search(r'\d+%|\d+\.\d+|\+\d+|\d+/\d+', cell)
    )
    score = 0
    flags = []

    # At least one data row in ablation table
    data_rows = len([l for l in lines if re.search(r'\|.+\|.+\|.+\|', l)])
    if data_rows < 2:
        flags.append(f"消融表仅有{data_rows}行数据（建议≥3行）")
        score += 1
    else:
        score += 8

    # Has quantitative values (numbers)
    if numeric_cells < 3:
        flags.append(f"消融表定量数据不足（仅{numeric_cells}个数字单元，建议≥3）")
    else:
        score += 12

    # Has a 核心结论 summary line
    if "核心结论" not in section:
        flags.append("消融缺'核心结论'总结行")
    else:
        score += 5

    # Check for template text
    template_markers = ["显著下降 — 验证了设计的必要性", "核心组件的必要性"]
    for tm in template_markers:
        if tm in section:
            flags.append(f"消融含模板文本: '{tm}'")
            score -= 3

    return min(score, 25), flags


def _check_physics_intuition(text: str) -> tuple[int, list[str]]:
    """Physics intuition should be multi-paragraph with concrete analogies."""
    section = _extract_section(text, "物理直觉解释")
    if not section:
        return 0, ["物理直觉章节缺失"]

    score = 0
    flags = []

    # Count substantive paragraphs (not just one-liners)
    paras = [p.strip() for p in section.split("\n\n") if p.strip() and not p.strip().startswith("##")]
    substantive = [p for p in paras if len(p) > 100]

    if len(substantive) >= 3:
        score += 10
    elif len(substantive) == 2:
        score += 7
    elif len(substantive) == 1:
        score += 4
        if len(substantive[0]) < 200:
            flags.append(f"物理直觉仅1段且<200字")
    else:
        flags.append("物理直觉内容过短")
        score += 1

    # Check for template text
    template_markers = [
        "本工作的核心设计动机是将复杂问题分解为可管理的子问题",
        "利用结构先验降低学习难度",
    ]
    for tm in template_markers:
        if tm in section:
            flags.append(f"物理直觉含模板文本")
            score -= 5
            break

    # Check for concrete analogy (bold text pattern like **...**)
    if re.search(r'\*\*[^*]+\*\*', section):
        score += 3

    return min(score, 15), flags


def _check_questions(text: str) -> tuple[int, list[str]]:
    """Check 精读问题 are specific and research-actionable."""
    section = _extract_section(text, "精读问题")
    if not section:
        return 0, ["精读问题章节缺失"]

    score = 0
    flags = []

    # Count numbered questions
    questions = re.findall(r'\d+\.\s+\*\*[^*]+\*\*|\d+\.\s+[^?]+\?', section)
    n = len(questions)

    if n >= 5:
        score += 10
    elif n >= 4:
        score += 8
    elif n >= 3:
        score += 6
    elif n >= 2:
        score += 4
        flags.append(f"精读问题仅{n}个（建议≥3个可研究方向）")
    else:
        score += 2
        flags.append(f"精读问题仅{n}个")

    # Check for too-general questions (template patterns)
    generic_patterns = [
        r"泛化边界在哪里",
        r"核心方法的泛化",
        r"主要失败模式是什么",
    ]
    generic_count = sum(1 for p in generic_patterns if re.search(p, section))
    if generic_count >= 2 and n <= 3:
        flags.append("精读问题过于泛化（含模板文本）")
        score -= 3

    return min(score, 10), flags


def _check_math_depth(text: str) -> tuple[int, list[str]]:
    """Math section should have equations beyond the mermaid diagram."""
    section = _extract_section(text, "底层原理与数学推导")
    if not section:
        return 0, ["底层原理章节缺失"]

    score = 0
    flags = []

    # Count LaTeX math blocks ($$ or $)
    display_math = len(re.findall(r'\$\$[^$]+\$\$', section))
    inline_math = len(re.findall(r'\$[^$]+\$', section))

    if display_math >= 3:
        score += 10
    elif display_math >= 1:
        score += 6
    else:
        flags.append("底层原理缺少LaTeX公式（仅mermaid图）")

    if inline_math >= 5:
        score += 5
    elif inline_math >= 2:
        score += 3

    return min(score, 15), flags


def _check_related_work_detail(text: str) -> tuple[int, list[str]]:
    """Related work should have specific comparisons, not just names."""
    section = _extract_section(text, "与其他论文的关系")
    if not section:
        return 0, ["与其他论文的关系章节缺失"]

    lines = [l.strip() for l in section.split("\n") if l.strip().startswith("-")]
    detailed = sum(1 for l in lines if "—" in l or "：" in l or ":" in l)

    score = 0
    flags = []
    if len(lines) >= 4 and detailed >= 3:
        score += 10
    elif len(lines) >= 2:
        score += 5
    else:
        flags.append("相关论文关系过于简短")

    # Check for template
    if "与同期发表的 RL for manipulation 工作形成互补" in section:
        flags.append("相关论文含模板文本")
        score -= 3

    return min(score, 10), flags


def _check_tech_value_uniqueness(text: str) -> tuple[int, list[str]]:
    """技术价值 should be non-template, with concrete positioning."""
    section = _extract_section(text, "技术价值与演进定位")
    if not section:
        return 0, ["技术价值章节缺失"]

    flags = []
    score = 5
    if "在本领域代表了一个重要的技术方向" in section:
        flags.append("技术价值含模板文本")
        score = 1
    if len(section) > 200:
        score += 5
    return min(score, 10), flags


# ── Helpers ─────────────────────────────────────────────────────────

def _extract_section(text: str, section_name: str) -> str | None:
    """Extract content of a ## section."""
    pattern = rf'## {re.escape(section_name)}.*?(?=\n## |\Z)'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(0) if m else None


# ── Main grading function ───────────────────────────────────────────

def grade_note(note_path: Path) -> QualityReport:
    """Grade a single note and return QualityReport."""
    text = note_path.read_text(encoding="utf-8")
    rel = str(note_path.relative_to(note_path.parent.parent.parent))

    # Skip non-paper entries
    is_no_paper = "no open paper" in text or "无公开论文" in text or "commercial physical AI" in text
    is_overview = "brief" in note_path.name.lower() or "research-brief" in note_path.name.lower()
    if is_no_paper or is_overview:
        return QualityReport(note=rel, grade="S", score=100,
                            flags=["skipped (non-paper)"])

    total_score = 0
    all_flags = []
    highlights = []

    checks = [
        ("消融", _check_ablation_quality(text)),
        ("物理直觉", _check_physics_intuition(text)),
        ("精读问题", _check_questions(text)),
        ("数学推导", _check_math_depth(text)),
        ("相关论文", _check_related_work_detail(text)),
        ("技术价值", _check_tech_value_uniqueness(text)),
    ]

    for name, (score, flags) in checks:
        total_score += score
        all_flags.extend([f"[{name}] {f}" for f in flags])

    # Grade mapping
    if total_score >= 85:
        grade = "A"
    elif total_score >= 65:
        grade = "B"
    elif total_score >= 45:
        grade = "C"
    else:
        grade = "D"

    # Highlight good things
    if total_score >= 85:
        highlights.append("深度分析: 消融定量+多层直觉+具体精读问题")
    ablation_score = checks[0][1][0]
    if ablation_score >= 20:
        highlights.append("消融质量高: 多项定量数据+核心结论")

    return QualityReport(note=rel, grade=grade, score=total_score,
                        flags=all_flags, highlights=highlights)


def grade_all(workspace: Path) -> dict[str, QualityReport]:
    """Grade all notes and return summary."""
    notes_dir = workspace / "notes"
    reports = {}

    for nf in sorted(notes_dir.rglob("*.md")):
        if nf.name.startswith("."):
            continue
        report = grade_note(nf)
        reports[str(nf.relative_to(workspace))] = report

    return reports


def print_quality_report(reports: dict[str, QualityReport]) -> tuple[int, int, int, int]:
    """Print report and return (A, B, C, D) counts."""
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "S": 0}
    by_grade: dict[str, list[QualityReport]] = {"A": [], "B": [], "C": [], "D": [], "S": []}

    for report in reports.values():
        counts[report.grade] += 1
        by_grade[report.grade].append(report)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Quality Report: {len(reports)} notes graded")
    print(f"  A (deep):    {counts['A']}")
    print(f"  B (standard):{counts['B']}")
    print(f"  C (shallow): {counts['C']}")
    print(f"  D (bare):    {counts['D']}")
    if counts['S']:
        print(f"  S (skipped): {counts['S']}")

    # Print D and C grades (issues to fix)
    for grade in ["D", "C"]:
        if by_grade[grade]:
            print(f"\n── {grade}-grade notes (needs improvement) ──")
            for r in by_grade[grade][:10]:  # cap at 10
                print(r)

    # Print top A
    if by_grade["A"]:
        print(f"\n── A-grade notes ──")
        for r in by_grade["A"][:5]:
            print(f"  ★ {r.note} (score={r.score})")

    print(f"{'='*60}\n")
    return counts["A"], counts["B"], counts["C"], counts["D"]


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    reports = grade_all(ws)
    print_quality_report(reports)
