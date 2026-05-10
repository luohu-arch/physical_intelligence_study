from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


ATOM = "{http://www.w3.org/2005/Atom}"
DEFAULT_QUERIES = [
    "vision language action robot manipulation",
    "robot foundation model humanoid manipulation",
    "embodied reasoning robot manipulation",
    "action tokenizer vision language action",
    "flow policy vision language action robot",
    "reinforcement learning vision language action robot",
]


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len].strip("-") or "paper"


def normalize_arxiv_id(value: str) -> str:
    value = value.strip()
    value = value.replace("https://arxiv.org/abs/", "")
    value = value.replace("https://arxiv.org/pdf/", "")
    value = value.removesuffix(".pdf")
    value = re.sub(r"v\d+$", "", value)
    return value


def fetch_url(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pull-vla-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def arxiv_api_by_ids(ids: list[str]) -> list[dict]:
    ids = [normalize_arxiv_id(x) for x in ids if x.strip()]
    if not ids:
        return []
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids)
    return parse_arxiv_entries(fetch_url(url))


def arxiv_search(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = "https://export.arxiv.org/api/query?" + params
    return parse_arxiv_entries(fetch_url(url))


def parse_arxiv_entries(data: bytes) -> list[dict]:
    root = ET.fromstring(data)
    rows = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id", default="")
        arxiv_id = normalize_arxiv_id(raw_id.rsplit("/", 1)[-1])
        title = " ".join(entry.findtext(f"{ATOM}title", default="").split())
        summary = " ".join(entry.findtext(f"{ATOM}summary", default="").split())
        published = entry.findtext(f"{ATOM}published", default="")
        authors = [
            a.findtext(f"{ATOM}name", default="")
            for a in entry.findall(f"{ATOM}author")
        ]
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "year": published[:4],
                "published": published[:10],
                "authors": "; ".join(authors),
                "summary": summary,
                "source_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            }
        )
    return rows


def search_rows(queries: list[str], max_results: int, min_year: int) -> list[dict]:
    seen = set()
    out = []
    keywords = re.compile(
        r"(vision-language-action|vla|robot|robotic|embodied|manipulation|humanoid|action)",
        re.I,
    )
    for q in queries:
        for row in arxiv_search(q, max_results=max_results):
            if row["arxiv_id"] in seen:
                continue
            seen.add(row["arxiv_id"])
            year = int(row["year"] or 0)
            if year < min_year:
                continue
            text = row["title"] + " " + row["summary"]
            if not keywords.search(text):
                continue
            row.update(
                {
                    "id": slugify(row["title"], 48),
                    "category": "search-candidate",
                    "priority": "screen",
                    "why": "Candidate from arXiv search; evaluate with references/scoring.md before adding to core plan.",
                    "project_url": "",
                }
            )
            out.append(row)
        time.sleep(3)
    return out
