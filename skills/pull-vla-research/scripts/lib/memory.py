from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any


MEMORY_DIR = ".memory"
MEMORY_VERSION = 2

DEFAULT_PREFS: dict[str, Any] = {
    "note_depth": None,               # "deep" | "basic" | None (undecided)
    "note_language": None,            # "zh-CN" | "en" | None
    "require_math_derivation": None,  # true | false | None
    "require_architecture_diagram": None,
    "require_tradeoff_table": None,
    "require_physical_intuition": None,
    "content_priorities": [],         # ordered: ["math","intuition","engineering",...]
    "auto_download": None,
    "auto_backfill": None,
}

DEFAULT_JUDGMENTS: dict[str, Any] = {
    "category_aliases": {},           # e.g. "humanoid" → "humanoid foundation model"
    "priority_overrides": {},         # paper_id → priority (from manual reclassification)
    "skip_reasons": {},               # paper_id → why skipped
    "interpretations": [],            # [{context, input, resolution, date}, ...]
}

DEFAULT_PATTERNS: dict[str, Any] = {
    "missed_sections": {},            # section_name → count (user filled what was empty)
    "common_corrections": {},         # edit_pattern → count
    "style_rules": [],                # ["use Chinese","prefer mermaid over ASCII",...]
    "section_order": [],              # learned preferred section order
}


def memory_dir(workspace: Path) -> Path:
    d = workspace / MEMORY_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(default)
        _deep_merge(merged, data)
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(default)


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_memory(workspace: Path) -> dict[str, Any]:
    """Load all three memory files."""
    md = memory_dir(workspace)
    return {
        "preferences": _load_json(md / "preferences.json", DEFAULT_PREFS),
        "judgments": _load_json(md / "judgments.json", DEFAULT_JUDGMENTS),
        "patterns": _load_json(md / "patterns.json", DEFAULT_PATTERNS),
    }


def save_memory(workspace: Path, mem: dict[str, Any]) -> None:
    md = memory_dir(workspace)
    _save_json(md / "preferences.json", mem.get("preferences", {}))
    _save_json(md / "judgments.json", mem.get("judgments", {}))
    _save_json(md / "patterns.json", mem.get("patterns", {}))


def record_session(mem: dict[str, Any], rows: list[dict], flags: dict[str, bool]) -> None:
    """Observe and record user behavior from this session."""
    prefs = mem.setdefault("preferences", {})
    patterns = mem.setdefault("patterns", {})

    # Observe flag-based preferences
    if flags.get("deep"):
        prefs["note_depth"] = "deep"
    if flags.get("download"):
        prefs["auto_download"] = True
    if flags.get("backfill"):
        prefs["auto_backfill"] = True

    # Count session
    prefs.setdefault("_session_count", 0)
    prefs["_session_count"] += 1
    prefs["_last_session"] = date.today().isoformat()


def observe_correction(mem: dict[str, Any], context: str, before: str, after: str) -> None:
    """Record a user correction for future reuse.

    Args:
        context:  what was being decided (e.g. 'category', 'section_depth', 'priority')
        before:   the system's default/guess
        after:    what the user changed it to
    """
    judgments = mem.setdefault("judgments", {})
    interpretations = judgments.setdefault("interpretations", [])
    interpretations.append({
        "context": context,
        "before": before,
        "after": after,
        "date": date.today().isoformat(),
    })
    # Keep last 50 to avoid unbounded growth
    if len(interpretations) > 50:
        judgments["interpretations"] = interpretations[-50:]

    # Update category aliases
    if context == "category":
        aliases = judgments.setdefault("category_aliases", {})
        aliases[before.strip().lower()] = after


def observe_missed_section(mem: dict[str, Any], section_name: str) -> None:
    """Record that a section was initially empty and user filled it."""
    patterns = mem.setdefault("patterns", {})
    missed = patterns.setdefault("missed_sections", {})
    missed[section_name] = missed.get(section_name, 0) + 1


def get_learned_sections(mem: dict[str, Any]) -> list[str]:
    """Return sections that were historically missed and should be auto-filled."""
    patterns = mem.get("patterns", {})
    missed = patterns.get("missed_sections", {})
    return [k for k, v in missed.items() if v >= 2]


def get_category_alias(mem: dict[str, Any], raw_category: str) -> str | None:
    """Resolve a fuzzy category using past judgments."""
    judgments = mem.get("judgments", {})
    aliases = judgments.get("category_aliases", {})
    return aliases.get(raw_category.strip().lower())


def get_style_rules(mem: dict[str, Any]) -> list[str]:
    """Return learned style rules (e.g. 'use Chinese', 'prefer mermaid')."""
    patterns = mem.get("patterns", {})
    return patterns.get("style_rules", [])


def memory_status(mem: dict[str, Any]) -> str:
    """Human-readable summary of what memory has learned."""
    prefs = mem.get("preferences", {})
    judgments = mem.get("judgments", {})
    patterns = mem.get("patterns", {})

    parts = []
    if prefs.get("note_depth"):
        parts.append(f"note_depth={prefs['note_depth']}")
    if prefs.get("note_language"):
        parts.append(f"language={prefs['note_language']}")
    if prefs.get("require_architecture_diagram"):
        parts.append("want_diagrams")
    if prefs.get("require_tradeoff_table"):
        parts.append("want_tradeoff_tables")
    if prefs.get("require_math_derivation"):
        parts.append("want_math_derivations")

    aliases = judgments.get("category_aliases", {})
    if aliases:
        parts.append(f"{len(aliases)} category aliases learned")

    sessions = prefs.get("_session_count", 0)
    if sessions:
        parts.insert(0, f"session #{sessions}")

    missed = patterns.get("missed_sections", {})
    frequent = get_learned_sections(mem)
    if frequent:
        parts.append(f"{len(frequent)} auto-fill sections: {', '.join(frequent[:3])}")

    return " | ".join(parts) if parts else "fresh memory (no preferences learned yet)"


def suggest_flags(mem: dict[str, Any]) -> list[str]:
    """Return suggested CLI flags based on learned preferences."""
    prefs = mem.get("preferences", {})
    suggestions = []
    if prefs.get("note_depth") == "deep":
        suggestions.append("--deep")
    if prefs.get("auto_download"):
        suggestions.append("--download")
    if prefs.get("auto_backfill"):
        suggestions.append("--backfill")
    return suggestions


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
