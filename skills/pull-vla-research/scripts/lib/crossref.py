"""Cross-reference detection between notes.

Scans all notes in the workspace and identifies which papers
mention other papers by name or slug, generating suggested
"与其它论文的关系" entries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .arxiv import slugify


def generate_crossrefs(workspace: Path) -> dict[str, list[dict[str, Any]]]:
    """Scan all notes and detect cross-references between papers.

    For each paper note, identifies which other papers are mentioned
    by title or slug within the note text. Returns suggestions suitable
    for populating "与其它论文的关系" sections.

    Args:
        workspace: Root path of the study workspace (contains notes/).

    Returns:
        A dict mapping ``note_id`` (filename stem) to a list of suggestion
        dicts, each with keys:
          - ``note_id``: the referenced note's id
          - ``title``: the referenced note's title
          - ``context``: a short excerpt from the text where the reference
            was detected (first 80 chars)
    """
    notes_dir = workspace / "notes"
    if not notes_dir.exists() or not notes_dir.is_dir():
        return {}

    # Collect all note files and their titles
    note_info: dict[str, dict[str, Any]] = {}
    for subdir in sorted(notes_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for f in sorted(subdir.glob("*.md")):
            if f.name.startswith("."):
                continue
            note_id = f.stem
            text = f.read_text(encoding="utf-8")
            # Extract the title (first # heading)
            title = ""
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    title = stripped.lstrip("# ").strip()
                    break
            note_slug = slugify(title, 48) if title else note_id
            note_info[note_id] = {
                "path": f,
                "title": title,
                "slug": note_slug,
                "text": text,
            }

    # For each note, scan text for mentions of other notes
    crossrefs: dict[str, list[dict[str, Any]]] = {}
    for note_id, info in note_info.items():
        mentions: list[dict[str, Any]] = []
        for other_id, other_info in note_info.items():
            if other_id == note_id:
                continue
            # Check if the other paper's title is mentioned
            found = False
            context = ""
            if other_info["title"] and other_info["title"] in info["text"]:
                found = True
                idx = info["text"].index(other_info["title"])
                context = info["text"][max(0, idx - 20): idx + len(other_info["title"]) + 20]
            elif other_id in info["text"]:
                found = True
                idx = info["text"].index(other_id)
                context = info["text"][max(0, idx - 20): idx + len(other_id) + 20]
            elif other_info["slug"] != note_id and other_info["slug"] in info["text"]:
                found = True
                idx = info["text"].index(other_info["slug"])
                context = info["text"][max(0, idx - 20): idx + len(other_info["slug"]) + 20]

            if found:
                mentions.append({
                    "note_id": other_id,
                    "title": other_info["title"],
                    "context": context.strip()[:80],
                })

        if mentions:
            crossrefs[note_id] = mentions

    return crossrefs
