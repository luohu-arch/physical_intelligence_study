#!/usr/bin/env python3
"""批量提取各笔记对应 PDF 的 Figure 1 并嵌入笔记「核心技术」章节。

用法: python3 embed_figures.py [--force]
输出: notes/<track>/figures/<note-id>/fig1.png + 笔记内嵌引用
幂等: 已含 figures/<note-id>/ 引用的笔记跳过（--force 重做）。
"""
import argparse
import glob
import os
import re
import sys

import fitz
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from extract_figures import render  # noqa: E402

MAX_W = 1100
COLORS = 128
TMP = "/tmp/_fig_embed.png"


def optimize(src, dst):
    im = Image.open(src)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, int(im.height * MAX_W / im.width)), Image.LANCZOS)
    try:
        im = im.quantize(colors=COLORS, method=Image.MEDIANCUT)
    except Exception:
        pass
    im.save(dst, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ok = miss_pdf = fail = skip = 0
    for note in sorted(glob.glob("notes/**/*.md", recursive=True)):
        if "brief" in os.path.basename(note):
            continue
        nid = os.path.basename(note)[:-3]
        track = os.path.dirname(note)[len("notes/"):]  # 支持 rl/<sub> 多级赛道
        text = open(note, encoding="utf-8").read()
        if not a.force and f"figures/{nid}/" in text:
            skip += 1
            continue
        m = re.search(r"(?:Local PDF|本地 PDF)[：:]\s*`?([^`\n]+)`?", text)
        pdf = m.group(1).strip() if m else ""
        if not pdf or not os.path.exists(pdf):
            miss_pdf += 1
            continue
        r = render(pdf, TMP, 1, 200)
        if not r:
            fail += 1
            continue
        figdir = f"notes/{track}/figures/{nid}"
        os.makedirs(figdir, exist_ok=True)
        dst = f"{figdir}/fig1.png"
        optimize(TMP, dst)
        cap = r["caption"].rstrip(". ")
        embed = (f"![{nid} 架构图](figures/{nid}/fig1.png)\n"
                 f"*论文 Figure 1（p{r['page']}）：{cap}*\n\n")
        if not a.force and f"figures/{nid}/" not in text:
            # 插在「## 核心技术」标题后的第一个空行处
            new = re.sub(r"(## 核心技术\n\n)", r"\1" + embed, text, count=1)
            if new == text:  # 结构异常则追加到文件末尾
                new = text.rstrip() + "\n\n" + embed
            open(note, "w", encoding="utf-8").write(new)
        ok += 1
    print(f"embedded={ok} skip(already)={skip} no-pdf={miss_pdf} no-fig={fail}")


if __name__ == "__main__":
    main()
