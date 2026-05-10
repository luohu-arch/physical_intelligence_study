# VLA Research — shared library
from .arxiv import (
    ATOM, DEFAULT_QUERIES,
    arxiv_api_by_ids, arxiv_search, fetch_url,
    normalize_arxiv_id, parse_arxiv_entries, search_rows, slugify,
)
from .workspace import (
    ensure_dirs, load_watchlist, merge_metadata, output_dir, resolve_date_folder,
    write_matrix,
)
from .notes import (
    backfill_notes, download_pdf, find_note_for_row, find_pdf_for_note,
    has_pdf_eof, write_note,
)
from .memory import (
    get_category_alias, get_learned_sections, get_style_rules,
    load_memory, memory_status, observe_correction, observe_missed_section,
    record_session, save_memory, suggest_flags,
)
from .validate import DEEP_SECTIONS, validate_notes
from .crossref import generate_crossrefs
from .stats import compute_stats

__all__ = [
    # arxiv
    "ATOM", "DEFAULT_QUERIES",
    "arxiv_api_by_ids", "arxiv_search", "fetch_url",
    "normalize_arxiv_id", "parse_arxiv_entries", "search_rows", "slugify",
    # workspace
    "ensure_dirs", "load_watchlist", "merge_metadata", "output_dir",
    "resolve_date_folder", "write_matrix",
    # notes
    "backfill_notes", "download_pdf", "find_note_for_row", "find_pdf_for_note",
    "has_pdf_eof", "write_note",
    # memory
    "get_category_alias", "get_learned_sections", "get_style_rules",
    "load_memory", "memory_status", "observe_correction", "observe_missed_section",
    "record_session", "save_memory", "suggest_flags",
    # validate
    "DEEP_SECTIONS", "validate_notes",
    # crossref
    "generate_crossrefs",
    # stats
    "compute_stats",
]
