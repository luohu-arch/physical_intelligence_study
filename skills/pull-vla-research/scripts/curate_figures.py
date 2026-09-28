#!/usr/bin/env python3
"""审查已嵌入的 Figure 1：图注是结果图/纯 teaser 的，换成论文内真正的架构图。

用法: python3 curate_figures.py [--dry]
规则:
  WRONG = 图注含结果类词（results/success rate/benchmark summary/evaluation/
          performance/comparison across）且不含方法结构词
  对 WRONG 的论文扫描 Figure 2..8 图注，按方法结构词评分取最高分者替换；
  无合适替代则保持原状并在报告中标注。
"""
import argparse
import glob
import os
import re
import sys

import fitz

sys.path.insert(0, os.path.dirname(__file__))
from extract_figures import find_caption, render  # noqa: E402
from embed_figures import optimize  # noqa: E402

RESULTS_WORDS = re.compile(
    r"benchmark summary|success rate|results?\b|evaluation|performance|"
    r"compares favorably|comparisons? across|scores?\b|ablation", re.I)
STRUCT_WORDS = re.compile(
    r"architect|framework|pipeline|overview of|training (and|path|procedure)|"
    r"consists of|modules?|method|structure|diagram|workflow|design of", re.I)
STRUCT_SCORE = [
    (re.compile(r"architect", re.I), 3),
    (re.compile(r"framework|pipeline|overview of", re.I), 2),
    (re.compile(r"training|method|modules?|consists|workflow|structure|design", re.I), 1),
]


def classify(cap):
    r, s = RESULTS_WORDS.search(cap), STRUCT_WORDS.search(cap)
    return bool(r) and not s


def score(cap):
    return sum(w for pat, w in STRUCT_SCORE if pat.search(cap))


def captions_in_pdf(pdf, max_fig=8):
    doc = fitz.open(pdf)
    found = {}
    for page in doc:
        for n in range(1, max_fig + 1):
            if n in found:
                continue
            for lab in (f"Figure {n}", f"Fig. {n}", f"FIGURE {n}"):
                a, sp, txt = find_caption(page, lab)
                if a is not None:
                    found[n] = (page.number + 1, " ".join(txt.split())[:110])
                    break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    kept, swapped, kept_wrong = [], [], []
    for note in sorted(glob.glob("notes/**/*.md", recursive=True)):
        if "brief" in os.path.basename(note):
            continue
        nid = os.path.basename(note)[:-3]
        track = os.path.dirname(note)[len("notes/"):]  # 支持 rl/<sub> 多级赛道
        text = open(note, encoding="utf-8").read()
        m = re.search(r"!\[[^\]]*\]\(figures/%s/(fig\d+)\.png\)\n\*论文 Figure (\d+)（p(\d+)）：([^*]+)\*" % nid, text)
        if not m:
            continue
        fig_file, fig_no, page_no, cap = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        if not classify(cap):
            kept.append((nid, cap[:60]))
            continue
        pm = re.search(r"(?:Local PDF|本地 PDF)[：:]\s*`?([^`\n]+)`?", text)
        pdf = pm.group(1).strip() if pm else ""
        if not pdf or not os.path.exists(pdf):
            kept_wrong.append((nid, "no pdf", cap[:50]))
            continue
        caps = captions_in_pdf(pdf)
        best_n, best_s, best_cap = None, 0, ""
        for n, (pg, c) in caps.items():
            if n == fig_no or classify(c):
                continue
            sc = score(c)
            if sc > best_s:
                best_n, best_s, best_cap = n, sc, c
        if best_n is None:
            kept_wrong.append((nid, "no better fig", cap[:50]))
            continue
        r = render(pdf, f"/tmp/_cur_{nid}.png", best_n, 200)
        if not r:
            kept_wrong.append((nid, f"fig{best_n} render fail", cap[:50]))
            continue
        figdir = f"notes/{track}/figures/{nid}"
        os.makedirs(figdir, exist_ok=True)
        if not a.dry:
            optimize(f"/tmp/_cur_{nid}.png", f"{figdir}/fig{best_n}.png")
            old = f"{figdir}/{fig_file}"
            if os.path.exists(old) and f"fig{best_n}.png" != fig_file:
                os.remove(old)
            new_ref = (f"![{nid} 架构图](figures/{nid}/fig{best_n}.png)\n"
                       f"*论文 Figure {best_n}（p{r['page']}）：{r['caption'].rstrip('. ')}*")
            text = text[:m.start()] + new_ref + text[m.end():]
            open(note, "w", encoding="utf-8").write(text)
        swapped.append((nid, f"fig{fig_no}->fig{best_n}", best_cap[:70]))
    print(f"kept={len(kept)} swapped={len(swapped)} kept-wrong={len(kept_wrong)}")
    print("\n-- SWAPPED --")
    for row in swapped:
        print("  ", row[0], row[1], "|", row[2])
    print("\n-- KEPT although suboptimal --")
    for row in kept_wrong:
        print("  ", row[0], row[1], "|", row[2])


if __name__ == "__main__":
    main()
