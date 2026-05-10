#!/usr/bin/env python3
"""Pull recent VLA / Physical AI papers into a local study library.

Outputs (incremental pulls go in date-stamped folders):
  papers/<YYYY-MM-DD>/<slug>_<arxiv-id>.pdf
  notes/<YYYY-MM-DD>/<slug>.md
  tables/vla_research_watchlist.csv

Curriculum papers from vla.md live separately in papers/curriculum/ and notes/curriculum/.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from lib import (
    # arxiv
    DEFAULT_QUERIES, arxiv_api_by_ids, normalize_arxiv_id, search_rows,
    # workspace
    ensure_dirs, load_watchlist, merge_metadata, resolve_date_folder, write_matrix,
    # notes
    backfill_notes, download_pdf, has_pdf_eof, write_note,
    # memory
    load_memory, memory_status, record_session, save_memory, suggest_flags,
    # validate
    validate_notes,
    # crossref
    generate_crossrefs,
    # stats
    compute_stats,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".", help="Study workspace root")
    parser.add_argument("--mode", choices=["watchlist", "search", "both"], default="watchlist")
    parser.add_argument("--watchlist", default=None, help="CSV watchlist path")
    parser.add_argument("--query", action="append", default=[], help="Extra arXiv search query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--min-year", type=int, default=2025)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--notes", action="store_true")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--crossref", action="store_true", help="Generate cross-reference suggestions between notes")
    parser.add_argument("--stats", action="store_true", help="Show paper library statistics")
    parser.add_argument("--notes-dir", default=None, help="Subdirectory under notes/")
    parser.add_argument("--date-folder", default=date.today().isoformat())
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download / re-generate even if local files exist")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    ensure_dirs(workspace)

    # --validate: gate check
    if args.validate:
        passed, total = validate_notes(workspace, args.notes_dir)
        if passed < total:
            print(f"\nGate: BLOCKED — {total - passed} notes need fixes")
            return 1
        return 0

    # --stats: library statistics
    if args.stats:
        print(compute_stats(workspace))
        return 0

    # --crossref: generate cross-references
    if args.crossref:
        refs = generate_crossrefs(workspace)
        for note_id, suggestions in sorted(refs.items()):
            if suggestions:
                print(f"\n{note_id}:")
                for s in suggestions:
                    print(f"  ← {s}")
        if not any(refs.values()):
            print("No cross-references found.")
        return 0

    # --pull / --backfill flow
    date_folder = None if args.flat else args.date_folder
    # Resolve date suffix for same-day multi-pull (only when downloading or generating notes)
    if date_folder and (args.download or args.notes):
        date_folder = resolve_date_folder(workspace, date_folder)
    mem = load_memory(workspace)
    status = memory_status(mem)
    if "session" in status:
        print(f"memory: {status}", file=sys.stderr)
    mem_flags = suggest_flags(mem)
    if mem_flags:
        missing = [f for f in mem_flags if not getattr(args, f.lstrip("-").replace("-", "_"), False)]
        if missing:
            print(f"memory hint: try {' '.join(missing)}", file=sys.stderr)

    skill_root = Path(__file__).resolve().parents[1]
    watchlist_path = Path(args.watchlist) if args.watchlist else skill_root / "references" / "watchlist.csv"

    rows: list[dict] = []
    if args.mode in {"watchlist", "both"}:
        watch = load_watchlist(watchlist_path)
        try:
            fetched = arxiv_api_by_ids([r["arxiv_id"] for r in watch if r.get("arxiv_id")])
        except Exception as exc:
            print(f"warning: arXiv metadata fetch failed: {exc}", file=sys.stderr)
            fetched = []
        rows.extend(merge_metadata(watch, fetched))

    if args.mode in {"search", "both"}:
        try:
            rows.extend(search_rows(args.query or DEFAULT_QUERIES, args.max_results, args.min_year))
        except Exception as exc:
            print(f"warning: arXiv search failed: {exc}", file=sys.stderr)

    # Dedup
    deduped: dict[str, dict] = {}
    for row in rows:
        key = row.get("arxiv_id") or row.get("id") or row.get("title")
        deduped[key] = row
    rows = list(deduped.values())

    # --backfill
    if args.backfill:
        backfill_csv = workspace / "tables" / "vla_research_watchlist.csv"
        backfill_rows = load_watchlist(backfill_csv) if backfill_csv.exists() else rows
        backfill_ids = [normalize_arxiv_id(r["arxiv_id"]) for r in backfill_rows if r.get("arxiv_id")]
        try:
            backfill_meta_list = arxiv_api_by_ids(backfill_ids)
        except Exception as exc:
            print(f"warning: arXiv metadata fetch for backfill failed: {exc}", file=sys.stderr)
            backfill_meta_list = []
        backfill_lookup = {normalize_arxiv_id(r["arxiv_id"]): r for r in backfill_meta_list}
        n = backfill_notes(workspace, backfill_rows, backfill_lookup)
        if n > 0:
            print(f"backfilled {n} existing notes")
        write_matrix(backfill_rows, workspace)
        if not args.download and not args.notes:
            flags_used = {"download": False, "notes": False, "deep": args.deep, "backfill": True}
            record_session(mem, backfill_rows, flags_used)
            save_memory(workspace, mem)
            print(f"wrote {len(backfill_rows)} rows")
            return 0

    # Merge existing output-table paths so we can skip already-processed rows
    output_table = workspace / "tables" / "vla_research_watchlist.csv"
    existing_by_id: dict[str, dict] = {}
    if output_table.exists():
        existing_by_id = {r.get("id", ""): r for r in load_watchlist(output_table)}

    # Download + notes
    skipped = 0
    for row in rows:
        row.setdefault("my_status", "todo" if row.get("priority") != "screen" else "screen")

        # Carry forward existing paths from previous runs
        prev = existing_by_id.get(row.get("id", ""), {})
        existing_pdf = prev.get("local_pdf", "")
        existing_note = prev.get("local_note", "")
        pdf_path = row.get("local_pdf", "") or existing_pdf
        note_path = row.get("local_note", "") or existing_note

        # Incremental skip: if both PDF and note exist and are valid, skip unless --force
        pdf_ok = pdf_path and Path(pdf_path).exists() and has_pdf_eof(Path(pdf_path))
        note_ok = note_path and Path(note_path).exists()
        download_needed = args.download and row.get("arxiv_id") and (args.force or not pdf_ok)
        notes_needed = args.notes and (args.force or not note_ok)

        if args.download and row.get("arxiv_id") and not download_needed:
            pass  # pdf_ok already true, reuse existing path
        elif args.download and row.get("arxiv_id"):
            try:
                pdf_path = download_pdf(row, workspace, date_folder)
                if pdf_path and not has_pdf_eof(Path(pdf_path)):
                    print(f"warning: PDF may be incomplete: {pdf_path}", file=sys.stderr)
            except Exception as exc:
                print(f"download failed for {row.get('arxiv_id')}: {exc}", file=sys.stderr)
        else:
            pdf_path = ""  # reset if not downloading

        row["local_pdf"] = pdf_path

        if notes_needed:
            row["local_note"] = write_note(row, workspace, pdf_path, date_folder, deep=args.deep)
        elif args.notes:
            row["local_note"] = note_path  # reuse existing
            skipped += 1

    if skipped:
        print(f"skipped {skipped} already-processed row(s) — use --force to regenerate", file=sys.stderr)

    matrix = write_matrix(rows, workspace)
    print(f"wrote {len(rows)} rows to {matrix}")

    flags_used = {"download": args.download, "notes": args.notes,
                  "deep": args.deep, "backfill": args.backfill}
    record_session(mem, rows, flags_used)
    save_memory(workspace, mem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
