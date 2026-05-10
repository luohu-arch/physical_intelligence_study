from __future__ import annotations

from pathlib import Path

from .arxiv import fetch_url, normalize_arxiv_id, slugify
from .workspace import output_dir


def download_pdf(row: dict, workspace: Path, date_folder: str | None) -> str:
    arxiv_id = normalize_arxiv_id(row["arxiv_id"])
    if not arxiv_id:
        return ""
    slug = slugify(row["title"])
    path = output_dir(workspace, "papers", date_folder) / f"{slug}_{arxiv_id}.pdf"
    if path.exists() and path.stat().st_size > 1024:
        return str(path)
    pdf_url = row.get("pdf_url") or f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    data = fetch_url(pdf_url, timeout=180)
    path.write_bytes(data)
    return str(path)


def has_pdf_eof(path: Path) -> bool:
    if not path.exists():
        return False
    tail = path.read_bytes()[-2048:]
    return b"%%EOF" in tail


def write_note(row: dict, workspace: Path, pdf_path: str | None, date_folder: str | None, deep: bool = False) -> str:
    slug = row.get("id") or slugify(row["title"])
    path = output_dir(workspace, "notes", date_folder) / f"{slug}.md"
    if path.exists():
        return str(path)
    title = row["title"]
    if deep:
        body = _deep_note_body(row, title, pdf_path)
    else:
        body = _basic_note_body(row, title, pdf_path)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _basic_note_body(row: dict, title: str, pdf_path: str | None) -> str:
    return f"""# {title}

- arXiv: https://arxiv.org/abs/{row['arxiv_id']}
- Source: {row.get('source_url', '')}
- Project: {row.get('project_url', '')}
- Local PDF: `{pdf_path or ''}`
- Year: {row.get('year', '')}
- Category: {row.get('category', '')}
- Priority: {row.get('priority', '')}

## Why This Matters

{row.get('why', '')}

## Abstract / Summary

{row.get('summary') or '待补充。'}

## Reading Questions

1. What problem does this paper solve compared with RT-2, OpenVLA, Octo, or pi0/FAST?
2. What is the core architectural or training change?
3. What evidence shows the method improves real robot control or meaningful benchmarks?
4. What are the assumptions, costs, and failure modes?

## Key Ideas

待补充。

## Relation To Current Study Map

待补充。

## Core Architecture / Important Changes

待补充。若没有公开论文，基于官方发布、技术博客和可信报道整理，不把未经验证的营销说法当作论文结论。

## Implementation / Reproduction Notes

待补充。

## My Notes

待补充。
"""


def _deep_note_body(row: dict, title: str, pdf_path: str | None) -> str:
    return f"""# {title}

- arXiv: https://arxiv.org/abs/{row['arxiv_id']}
- Source: {row.get('source_url', '')}
- Project: {row.get('project_url', '')}
- Local PDF: `{pdf_path or ''}`
- Year: {row.get('year', '')}
- Category: {row.get('category', '')}
- Priority: {row.get('priority', '')}

## 一句话总结

待补充。

## 核心技术

待补充。

## 底层原理与数学推导

待补充。

## 物理直觉解释

待补充。

## 工程细节与实操指南

待补充。

## 技术权衡（Trade-off）

| 优势 | 劣势与工程代价 |
|------|---------------|
| 待补充 | 待补充 |

## 技术价值与演进定位

待补充。

## 与其他论文的关系

待补充。

## 精读问题

1. 待补充。
"""


def find_pdf_for_note(workspace: Path, row: dict) -> str:
    """Find an already-downloaded PDF that matches a watchlist entry."""
    arxiv_id = normalize_arxiv_id(row.get("arxiv_id", ""))
    if not arxiv_id:
        return ""
    papers_dir = workspace / "papers"
    if not papers_dir.exists():
        return ""
    slug = slugify(row["title"])
    expected_name = f"{slug}_{arxiv_id}.pdf"
    for subdir in sorted(papers_dir.iterdir(), reverse=True):
        if not subdir.is_dir() or subdir.name == "curriculum":
            continue
        candidate = subdir / expected_name
        if candidate.exists() and candidate.stat().st_size > 1024:
            return str(candidate)
    return ""


def find_note_for_row(workspace: Path, row: dict) -> Path | None:
    """Find an existing note file matching a watchlist entry by id or slug."""
    note_path_str = row.get("local_note", "")
    if note_path_str:
        p = Path(note_path_str)
        if p.exists():
            return p
    # Search notes/ subdirectories for a matching file
    rid = row.get("id", "")
    slug = slugify(row["title"])
    notes_dir = workspace / "notes"
    if not notes_dir.exists():
        return None
    for subdir in sorted(notes_dir.iterdir(), reverse=True):
        if not subdir.is_dir():
            continue
        for candidate_name in (f"{rid}.md", f"{slug}.md"):
            candidate = subdir / candidate_name
            if candidate.exists():
                return candidate
    return None


def backfill_notes(workspace: Path, rows: list[dict], fetched_meta: dict[str, dict]) -> int:
    """Backfill existing notes with arXiv metadata and local PDF paths.

    Only fills in empty/placeholder fields. Never overwrites user-written content.
    Returns the number of notes updated.
    """
    updated = 0
    for row in rows:
        note_path = find_note_for_row(workspace, row)
        if note_path is None:
            continue

        text = note_path.read_text(encoding="utf-8")

        # Backfill Local PDF path
        local_pdf = row.get("local_pdf", "")
        if not local_pdf:
            local_pdf = find_pdf_for_note(workspace, row)
        if local_pdf:
            if "Local PDF: ``" in text:
                text = text.replace("Local PDF: ``", f"Local PDF: `{local_pdf}`")
            elif "本地 PDF：" in text and "`" not in text.split("本地 PDF：")[1].split("\n")[0]:
                text = text.replace("本地 PDF：", f"本地 PDF：`{local_pdf}`")
            elif "Local PDF:" not in text and "本地 PDF" not in text:
                # Add the PDF line after arXiv line
                pdf_line = f"- Local PDF: `{local_pdf}`\n"
                if "- arXiv:" in text:
                    text = text.replace("- arXiv:", f"{pdf_line}- arXiv:")
                elif "- Source:" in text:
                    text = text.replace("- Source:", f"{pdf_line}- Source:")
            row["local_pdf"] = local_pdf
        row["local_note"] = str(note_path)

        arxiv_id = normalize_arxiv_id(row.get("arxiv_id", ""))
        meta = fetched_meta.get(arxiv_id, {}) if arxiv_id else {}

        # Backfill abstract if the section is a placeholder
        abstract = meta.get("summary", "")
        if abstract:
            # Replace "待补充。" placeholder
            old_block = "## Abstract / Summary\n\n待补充。"
            new_block = f"## Abstract / Summary\n\n{abstract}"
            if old_block in text:
                text = text.replace(old_block, new_block)
            else:
                # Replace empty abstract section (template with blank summary)
                old_empty = "## Abstract / Summary\n\n\n\n## Reading Questions"
                new_empty = f"## Abstract / Summary\n\n{abstract}\n\n## Reading Questions"
                if old_empty in text:
                    text = text.replace(old_empty, new_empty)

        # Backfill authors and published date into metadata header if missing
        authors = meta.get("authors", "")
        published = meta.get("published", "")
        if authors and "- Authors:" not in text:
            text = text.replace(
                "- Year:",
                f"- Authors: {authors}\n- Year:",
            )
        if published and f"- Published: {published}" not in text:
            text = text.replace(
                "- Year:",
                f"- Published: {published}\n- Year:",
            )

        note_path.write_text(text, encoding="utf-8")
        updated += 1

    return updated
