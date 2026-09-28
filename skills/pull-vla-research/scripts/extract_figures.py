#!/usr/bin/env python3
"""从论文 PDF 提取关键图表（架构图等）为 PNG，供笔记嵌入。

用法:
  python3 extract_figures.py --pdf <pdf> --out <png> [--fig 1] [--dpi 220]

策略: 定位 "Figure N" / "Fig. N" 图注文本块，取其所在栏（按图注 x 范围）内、
图注上方属于图形元素（嵌入位图 + 大尺寸矢量绘图）的包围盒并集，高分辨率渲染。
找不到图形元素时退化为图注上方固定高度区域（0.42 页高）。
"""
import argparse
import sys

import fitz


def find_caption(page, label):
    """返回 (锚点矩形, 栏宽矩形, 图注文本)。

    锚点 = "Figure N" 文本本身的 search rect（纵向裁剪依据，兼容图内标签
    与图注合并成同一文本块的 PDF）；栏宽 = 所在块的 bbox（横向过滤依据）。
    """
    for variants in (f"{label}:", f"{label}.", f"{label} "):
        for r in page.search_for(variants):
            for b in page.get_text("blocks"):
                bx = fitz.Rect(b[:4])
                head = " ".join(b[4].strip().lower().split())
                if bx.intersects(r) and label.lower() in head[:80]:
                    return r, bx, b[4]
    return None, None, None


def graphics_union(page, anchor, span):
    """图注锚点上方、栏宽范围内的图形元素包围盒并集。"""
    col_x0, col_x1 = min(anchor.x0, span.x0) - 5, max(anchor.x1, span.x1) + 5
    boxes = []
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                if r.y1 <= anchor.y0 + 4 and r.width > 30 and r.height > 20:
                    if r.x1 > col_x0 and r.x0 < col_x1:
                        boxes.append(r)
        except Exception:
            pass
    for d in page.get_drawings():
        r = d["rect"]
        if r.y1 <= anchor.y0 + 4 and r.width > 25 and r.height > 15:
            if r.x1 > col_x0 and r.x0 < col_x1:
                boxes.append(r)
    if not boxes:
        return None
    u = boxes[0]
    for b in boxes[1:]:
        u |= b
    return u


def render(pdf_path, out_path, fig_no, dpi):
    doc = fitz.open(pdf_path)
    label_candidates = [f"Figure {fig_no}", f"Fig. {fig_no}", f"FIGURE {fig_no}"]
    for page in doc:
        anchor = span = None
        for lab in label_candidates:
            anchor, span, cap_text = find_caption(page, lab)
            if anchor is not None:
                break
        if anchor is None:
            continue
        clip = graphics_union(page, anchor, span)
        if clip is None or clip.height < 60 or clip.width < 100:
            # 退化：图注锚点上方 0.42 页高、锚点所在块栏宽
            clip = fitz.Rect(span.x0, max(40, anchor.y0 - page.rect.height * 0.42),
                             span.x1, anchor.y0 - 2)
        clip = clip & page.rect
        if clip.is_empty or clip.height < 50:
            continue
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
        pix.save(out_path)
        return {"page": page.number + 1, "size": (round(clip.width), round(clip.height)),
                "caption": " ".join(cap_text.split())[:90], "file": out_path}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=220)
    a = ap.parse_args()
    r = render(a.pdf, a.out, a.fig, a.dpi)
    if r:
        print("OK", r)
    else:
        print("MISS", a.pdf)
        sys.exit(1)


if __name__ == "__main__":
    main()
