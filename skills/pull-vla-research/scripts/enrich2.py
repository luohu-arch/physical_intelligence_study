#!/usr/bin/env python3
"""enrich2: 全库图表升级。

pass 1 — 用 v2 提取器重裁全部已嵌入架构图（修复图内标签被切）
pass 2 — 为每篇笔记选主结果表（图注含结果关键词的 Table 1-6），嵌入消融章节
幂等：已有表格引用的笔记跳过 pass 2。
"""
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
from extract2 import render_v2, find_anchor  # noqa: E402
from embed_figures import optimize  # noqa: E402

RESULTS_KW = re.compile(
    r"compar|results?|success|baseline|average|overall|vs\.?|ablation|"
    r"performance|benchmark|win|SR\b|score|evaluat|accuracy|pass@|"
    r"libero|swe-bench|alfworld|webshop|sokoban|calvin", re.I)
NON_RESULTS_KW = re.compile(
    r"environment|hyperparam|prompt|dataset (list|detail)|task list|"
    r"notation|implementation detail|training (config|setup)", re.I)


def table_captions(pdf, max_n=8):
    """返回 {n: (page_no, caption)}。"""
    import fitz
    doc = fitz.open(pdf)
    out = {}
    for page in doc:
        for n in range(1, max_n + 1):
            if n in out:
                continue
            for lab in (f"Table {n}", f"TABLE {n}"):
                a, sp, txt = find_anchor(page, lab)
                if a is not None:
                    out[n] = (page.number + 1, " ".join(txt.split()))
                    break
    return out


def pick_results_table(pdf):
    caps = table_captions(pdf)
    best, best_s = None, -999
    for n, (pg, c) in caps.items():
        kw = len(RESULTS_KW.findall(c))
        bad = len(NON_RESULTS_KW.findall(c))
        s = kw * 10 - bad * 15 - n
        if kw >= 1 and s > best_s:
            best, best_s = n, s
    if best is None:
        # 兜底: Table 1 若非明显非结果表
        if 1 in caps and not NON_RESULTS_KW.search(caps[1][1]):
            best = 1
    return best, caps.get(best, (None, ""))


def main():
    fig_ok, fig_fail = 0, 0
    tab_add, tab_skip, tab_fail = 0, 0, 0
    for note in sorted(glob.glob("notes/**/*.md", recursive=True)):
        if "brief" in os.path.basename(note):
            continue
        nid = os.path.basename(note)[:-3]
        subdir = os.path.dirname(note)[len("notes/"):]
        text = open(note, encoding="utf-8").read()
        pm = re.search(r"(?:Local PDF|本地 PDF)[：:]\s*`?([^`\n]+)`?", text)
        pdf = pm.group(1).strip() if pm else ""
        if not pdf or not os.path.exists(pdf):
            continue
        figdir = f"notes/{subdir}/figures/{nid}"
        # pass 1: 重裁现有图
        m = re.search(r"figures/%s/(fig\d+)\.png\)\n\*论文 Figure (\d+)（p(\d+)）" % nid, text)
        if m and os.path.isdir(figdir):
            fn = int(m.group(2))
            r = render_v2(pdf, "/tmp/_e2_fig.png", fn, dpi=200)
            if r:
                optimize("/tmp/_e2_fig.png", f"{figdir}/{m.group(1)}.png")
                new = (f"*论文 Figure {fn}（p{r['page']}）：{r['caption'].rstrip('. ')}*")
                text = re.sub(r"\*论文 Figure %d（p\d+）：[^*]+\*" % fn, new, text, count=1)
                fig_ok += 1
            else:
                fig_fail += 1
        # pass 2: 主结果表
        if re.search(r"figures/%s/tab" % nid, text):
            tab_skip += 1
        else:
            tn, (pg, cap) = pick_results_table(pdf)
            if tn is None:
                tab_fail += 1
            else:
                r = render_v2(pdf, "/tmp/_e2_tab.png", tn, dpi=200, table=True)
                if r:
                    os.makedirs(figdir, exist_ok=True)
                    optimize("/tmp/_e2_tab.png", f"{figdir}/tab{tn}.png")
                    embed = (f"![{nid} 主结果表](figures/{nid}/tab{tn}.png)\n"
                             f"*论文 Table {tn}（p{r['page']}）：{r['caption'].rstrip('. ')}*\n\n")
                    if re.search(r"## 消融实验与分析\n\n", text):
                        text = re.sub(r"(## 消融实验与分析\n\n)", r"\1" + embed, text, count=1)
                    else:
                        text = re.sub(r"(## 消融实验与分析\n)", r"\1\n" + embed, text, count=1)
                    tab_add += 1
                else:
                    tab_fail += 1
        open(note, "w", encoding="utf-8").write(text)
    print(f"fig-re={fig_ok} fig-fail={fig_fail} tab-add={tab_add} "
          f"tab-none={tab_fail} tab-skip={tab_skip}")


if __name__ == "__main__":
    main()
