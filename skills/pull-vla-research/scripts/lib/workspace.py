from __future__ import annotations

import csv
from pathlib import Path

from .arxiv import normalize_arxiv_id


def load_watchlist(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_metadata(watch: list[dict], fetched: list[dict]) -> list[dict]:
    by_id = {x["arxiv_id"]: x for x in fetched}
    out = []
    for row in watch:
        arxiv_id = normalize_arxiv_id(row.get("arxiv_id", ""))
        meta = by_id.get(arxiv_id, {})
        merged = dict(row)
        for key in ["title", "year", "source_url"]:
            if not merged.get(key) and meta.get(key):
                merged[key] = meta[key]
        for key in ["published", "authors", "summary", "pdf_url"]:
            merged[key] = meta.get(key, "")
        merged["arxiv_id"] = arxiv_id
        out.append(merged)
    return out


def resolve_date_folder(workspace: Path, base_date: str) -> str:
    """Return base_date if its papers/notes dirs are absent or empty, else base_date-2, -3, etc."""
    _mkdirs(workspace)
    papers_dir = workspace / "papers" / base_date
    notes_dir = workspace / "notes" / base_date

    def _has_content(d: Path) -> bool:
        return d.exists() and any(True for _ in d.iterdir())

    if not _has_content(papers_dir) and not _has_content(notes_dir):
        return base_date

    n = 2
    while True:
        candidate = f"{base_date}-{n}"
        if not (workspace / "papers" / candidate).exists() and not (workspace / "notes" / candidate).exists():
            return candidate
        n += 1


def _mkdirs(workspace: Path) -> None:
    for name in ["papers", "notes", "tables"]:
        (workspace / name).mkdir(parents=True, exist_ok=True)


def ensure_dirs(workspace: Path) -> None:
    _mkdirs(workspace)
    for name in ["papers", "notes", "tables"]:
        (workspace / name).mkdir(parents=True, exist_ok=True)


def output_dir(workspace: Path, base: str, date_folder: str | None) -> Path:
    path = workspace / base
    if date_folder:
        path = path / date_folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_matrix(rows: list[dict], workspace: Path) -> str:
    path = workspace / "tables" / "vla_research_watchlist.csv"
    fields = [
        "id",
        "title",
        "year",
        "published",
        "category",
        "priority",
        "arxiv_id",
        "source_url",
        "project_url",
        "local_pdf",
        "local_note",
        "why",
        "my_status",
    ]
    # Preserve existing local_pdf and local_note from the output table
    existing: dict[str, dict] = {}
    if path.exists():
        existing = {r.get("id", ""): r for r in load_watchlist(path)}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rid = row.get("id", "")
            out = {k: row.get(k, "") for k in fields}
            # Carry forward local paths from previous runs if not set in current run
            if rid in existing:
                for key in ("local_pdf", "local_note"):
                    if not out.get(key) and existing[rid].get(key):
                        out[key] = existing[rid][key]
            writer.writerow(out)
    return str(path)
