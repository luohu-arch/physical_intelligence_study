#!/usr/bin/env python3
"""repair4: 按视觉审查清单修复 BAD 裁切 + 错选表格。

- fig-edge / tab-edge: v4 重裁覆盖同名 png, 更新图注页码
- tab-repick: 逐候选表(分数降序)v4 提取+结构校验, 首个有效者重写嵌入;
  全部无效则移除嵌入与 png
- --add-all: 为尚无表格的笔记补充(校验通过的)主结果表
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from extract3 import render_v4  # noqa: E402
from embed_figures import optimize  # noqa: E402

RESULTS_KW = re.compile(
    r"compar|results?|success|baseline|average|overall|vs\.?|ablation|"
    r"performance|benchmark|win|SR\b|score|evaluat|accuracy|pass@|"
    r"libero|swe-bench|alfworld|webshop|sokoban|calvin", re.I)
NON_RESULTS_KW = re.compile(
    r"environment|hyperparam|prompt|dataset (list|detail)|task list|"
    r"notation|implementation detail|training (config|setup)", re.I)

# 视觉审查(w01-w19 + chk1 复核)判定的修复清单
FIG_EDGE = {  # nid: fig_no
    "fast-tokenizer": 1, "instructvla": 1, "univla-latent-actions": 2,
    "view-invariant-policy": 1, "wholebodyvla": 4,
    "human-as-humanoid": 3, "imr-llm": 1, "symskill": 1,
    "tool-r0": 1, "enpire": 2, "hapticvla": 1, "dlo-routing": 1,
    "rl-token": 1, "wam-ttt": 1, "dexora": 1, "simplevla-rl": 2,
    "vlarl": 1, "d-jepa": 1, "vla-dreamer": 1, "paiworld": 2,
    "worldvla": 1, "unipi": 1, "humanvid-selfimprove": 1,
    "dextacwam": 2, "td-mpc2": 3,
}
TAB_EDGE = {  # nid: tab_no (裁切问题, 同号重裁; 无效则转 repick)
    "diffusion-policy": 1, "instructvla": 2, "lingbot-vla2": 5,
    "xr-1": 2, "echovla": 2, "memorywam": 1, "dreamvla": 1,
    "gr00t-n1": 4, "human-as-humanoid": 3, "facet0": 2,
    "harness-1": 2, "openclaw-rl": 3, "leworldmodel": 6, "susie": 1,
    "v-jepa2": 4, "vjepa": 5, "weaver": 3, "coskill": 1,
}
TAB_REPICK = ["art-vla-agent", "capek05", "tongyi-deepresearch",
              "dataprm", "robocat"]

_NOTE_CACHE = {}


def note_of(nid):
    if nid not in _NOTE_CACHE:
        hits = glob.glob(f"notes/**/{nid}.md", recursive=True)
        _NOTE_CACHE[nid] = hits[0] if hits else None
    return _NOTE_CACHE[nid]


def pdf_of(text):
    pm = re.search(r"(?:Local PDF|本地 PDF)[：:]\s*`?([^`\n]+)`?", text)
    return pm.group(1).strip() if pm else ""


def figdir_of(note):
    d = os.path.dirname(note)  # notes/<track>
    nid = os.path.basename(note)[:-3]
    return f"{d}/figures/{nid}", nid


def table_captions(pdf, max_n=10):
    from extract3 import find_anchor
    import fitz
    doc = fitz.open(pdf)
    out = {}
    for page in doc:
        for n in range(1, max_n + 1):
            if n in out:
                continue
            for lab in (f"Table {n}", f"TABLE {n}"):
                for strict in (True, False):
                    a, sp, txt = find_anchor(page, lab, strict)
                    if a is not None:
                        out[n] = (page.number + 1, " ".join(txt.split()))
                        break
    return out


def candidates(pdf):
    caps = table_captions(pdf)
    scored = []
    for n, (pg, c) in caps.items():
        kw = len(RESULTS_KW.findall(c))
        bad = len(NON_RESULTS_KW.findall(c))
        s = kw * 10 - bad * 15 - n
        scored.append((s, n, c))
    scored.sort(reverse=True)
    return scored


def repick(note, pdf, figdir, nid, text, log):
    """逐候选重选表格; 成功重写嵌入, 全败则移除。"""
    m = re.search(r"!\[[^\]]*主结果表\]\(figures/%s/tab(\d+)\.png\)\n+"
                  r"\*论文 Table \d+（p\d+）：[^\n]*\*\n*" % nid, text)
    old_n = int(m.group(1)) if m else None
    for s, n, cap in candidates(pdf):
        if s < 0 and n != 1:
            continue
        r = render_v4(pdf, "/tmp/_r4_tab.png", n, dpi=200, table=True)
        if r and r.get("valid"):
            os.makedirs(figdir, exist_ok=True)
            optimize("/tmp/_r4_tab.png", f"{figdir}/tab{n}.png")
            embed = (f"![{nid} 主结果表](figures/{nid}/tab{n}.png)\n\n"
                     f"*论文 Table {n}（p{r['page']}）："
                     f"{r['caption'].rstrip('. ')}*\n\n")
            if m:
                text = re.sub(r"!\[[^\]]*主结果表\]\(figures/%s/tab\d+\.png\)\n+"
                              r"\*论文 Table \d+（p\d+）：[^\n]*\*\n*" % nid,
                              embed, text, count=1)
            else:
                text = re.sub(r"(## 消融实验与分析\n\n)", r"\1" + embed,
                              text, count=1)
            if old_n and old_n != n:
                old = f"{figdir}/tab{old_n}.png"
                if os.path.exists(old):
                    os.remove(old)
            log.append(f"  {nid}: repick tab{old_n}->tab{n} (p{r['page']})")
            return text
    if m:  # 无有效表 → 移除嵌入
        text = re.sub(r"!\[[^\]]*主结果表\]\(figures/%s/tab\d+\.png\)\n+"
                      r"\*论文 Table \d+（p\d+）：[^\n]*\*\n*" % nid,
                      "", text, count=1)
        if old_n:
            old = f"{figdir}/tab{old_n}.png"
            if os.path.exists(old):
                os.remove(old)
        log.append(f"  {nid}: 移除嵌入(无有效结果表)")
    else:
        log.append(f"  {nid}: 无候选")
    return text


def process(nid, jobs, log):
    note = note_of(nid)
    if not note:
        log.append(f"  {nid}: NOTE NOT FOUND")
        return
    text = open(note, encoding="utf-8").read()
    pdf = pdf_of(text)
    if not pdf or not os.path.exists(pdf):
        log.append(f"  {nid}: PDF MISSING ({pdf})")
        return
    figdir, _ = figdir_of(note)
    for kind, no in jobs:
        if kind == "fig":
            r = render_v4(pdf, "/tmp/_r4_fig.png", no, dpi=200)
            if r:
                optimize("/tmp/_r4_fig.png", f"{figdir}/fig{no}.png")
                text = re.sub(r"\*论文 Figure %d（p\d+）：[^*]*\*" % no,
                              f"*论文 Figure {no}（p{r['page']}）："
                              f"{r['caption'].rstrip('. ')}*", text, count=1)
                log.append(f"  {nid}: fig{no} re-cropped (p{r['page']})")
            else:
                log.append(f"  {nid}: fig{no} MISS!")
        else:
            r = render_v4(pdf, "/tmp/_r4_tab.png", no, dpi=200, table=True)
            if r and r.get("valid"):
                optimize("/tmp/_r4_tab.png", f"{figdir}/tab{no}.png")
                text = re.sub(r"\*论文 Table %d（p\d+）：[^*]*\*" % no,
                              f"*论文 Table {no}（p{r['page']}）："
                              f"{r['caption'].rstrip('. ')}*", text, count=1)
                log.append(f"  {nid}: tab{no} re-cropped (p{r['page']})")
            else:
                why = "invalid" if r else "miss"
                log.append(f"  {nid}: tab{no} {why} -> repick")
                text = repick(note, pdf, figdir, nid, text, log)
    open(note, "w", encoding="utf-8").write(text)


def add_all(log):
    added = skipped = failed = 0
    for note in sorted(glob.glob("notes/**/*.md", recursive=True)):
        base = os.path.basename(note)
        if "brief" in base:
            continue
        nid = base[:-3]
        text = open(note, encoding="utf-8").read()
        if re.search(r"figures/%s/tab" % nid, text):
            skipped += 1
            continue
        pdf = pdf_of(text)
        if not pdf or not os.path.exists(pdf):
            continue
        figdir, _ = figdir_of(note)
        m = re.search(r"## 消融实验与分析\n\n", text)
        if not m:
            continue
        before = len(log)
        text = repick(note, pdf, figdir, nid, text, log)
        open(note, "w", encoding="utf-8").write(text)
        if any("repick tab" in l for l in log[before:]):
            added += 1
        elif any("无候选" in l for l in log[before:]):
            failed += 1
    log.append(f"add-all: added={added} none={failed} skip={skipped}")


def main():
    log = []
    todo = {}
    for nid, no in FIG_EDGE.items():
        todo.setdefault(nid, []).append(("fig", no))
    for nid, no in TAB_EDGE.items():
        todo.setdefault(nid, []).append(("tab", no))
    for nid in TAB_REPICK:
        todo.setdefault(nid, []).append(("repick", 0))
    for nid in sorted(todo):
        jobs = todo[nid]
        jobs = [j if j[0] != "repick" else None for j in jobs]
        # repick 单独处理
        real = [j for j in jobs if j]
        if real:
            process(nid, real, log)
        if any(j is None for j in jobs):
            note = note_of(nid)
            if note:
                text = open(note, encoding="utf-8").read()
                pdf = pdf_of(text)
                figdir, _ = figdir_of(note)
                if pdf and os.path.exists(pdf):
                    text = repick(note, pdf, figdir, nid, text, log)
                    open(note, "w", encoding="utf-8").write(text)
    print("\n".join(log))
    if "--add-all" in sys.argv:
        log2 = []
        add_all(log2)
        print("\n".join(log2))


if __name__ == "__main__":
    main()
