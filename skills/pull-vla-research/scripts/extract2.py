#!/usr/bin/env python3
"""v2 图表提取：修复过度裁剪（图内文字标签被切）+ 支持表格（图注在上）。

改进:
- 图内窄文本块(面板标签 (a)/标题行)并入裁剪区域
- 更大边距, 列宽对齐
- --table 模式: Table 图注在上方, 裁注释下方区域
"""
import argparse
import sys

import fitz

PAD = 14


def find_anchor(page, label):
    """返回 (锚点矩形=标签文本, 栏宽矩形=所在块, 图注全文)。"""
    for v in (f"{label}:", f"{label}.", f"{label} ", f"{label}:"):
        for r in page.search_for(v):
            for b in page.get_text("blocks"):
                bx = fitz.Rect(b[:4])
                head = " ".join(b[4].strip().lower().split())
                if bx.intersects(r) and label.lower() in head[:80]:
                    return r, bx, b[4]
    return None, None, None


def collect(page, col_x0, col_x1, y0, y1, is_above, min_h=10):
    """收集 [y0,y1] 区间内、栏宽范围内的图形与窄文本块。"""
    boxes = []
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                if _in(r, col_x0, col_x1, y0, y1):
                    boxes.append(r)
        except Exception:
            pass
    for d in page.get_drawings():
        r = d["rect"]
        if r.width > 18 and r.height > min_h and _in(r, col_x0, col_x1, y0, y1):
            boxes.append(r)
    # 图内文字: 窄于栏宽 85% 且在目标区间的文本块
    colw = col_x1 - col_x0
    for b in page.get_text("blocks"):
        bx = fitz.Rect(b[:4])
        txt = b[4].strip()
        if (bx.width < colw * 0.85 and len(txt) < 200
                and bx.y0 >= y0 - 6 and bx.y1 <= y1 + 6
                and bx.x1 > col_x0 and bx.x0 < col_x1):
            boxes.append(bx)
    return boxes


def _in(r, x0, x1, y0, y1):
    return r.y0 >= y0 - 8 and r.y1 <= y1 + 8 and r.x1 > x0 and r.x0 < x1


def render_v2(pdf_path, out_path, fig_no, dpi=200, table=False, pad=PAD):
    doc = fitz.open(pdf_path)
    labels = [f"Table {fig_no}", f"TABLE {fig_no}"] if table else \
             [f"Figure {fig_no}", f"Fig. {fig_no}", f"FIGURE {fig_no}"]
    for page in doc:
        for lab in labels:
            a, sp, txt = find_anchor(page, lab)
            if a is None:
                continue
            H = page.rect.height
            if table:
                # 表注在上方, 表体在下方; 界至下一密集文本块或 0.45 页
                y0, y1 = a.y1 + 2, min(page.rect.height - 30, a.y1 + H * 0.45)
                # 找下方第一个"整栏宽"文本块(正文)作下界
                colw = sp.x1 - sp.x0
                for b in page.get_text("blocks"):
                    bx = fitz.Rect(b[:4])
                    if (bx.y0 > a.y1 + 4 and bx.width > colw * 0.9
                            and len(b[4]) > 250 and bx.y0 < y1):
                        y1 = bx.y0 - 4
                        break
            else:
                y0, y1 = max(30, a.y0 - H * 0.52), a.y0 - 2
            col_x0, col_x1 = min(sp.x0, a.x0) - 4, max(sp.x1, a.x1) + 4
            boxes = collect(page, col_x0, col_x1, y0, y1, not table,
                            min_h=0.8 if table else 10)
            if not boxes:
                continue
            u = boxes[0]
            for b in boxes[1:]:
                u |= b
            # v3: 不对称扩展——顶部多留(图题文字)、底部贴近图注、水平放宽
            pt, pb, ps = (pad, pad, pad) if table else (26, 10, 22)
            x0 = max(u.x0 - ps, max(28, page.rect.x0 + 20))
            x1 = min(u.x1 + ps, page.rect.x1 - 20)
            if not table:
                y1c = min(u.y1 + pb, a.y0 - 3)
            else:
                y1c = u.y1 + pb
            clip = fitz.Rect(x0, u.y0 - pt, x1, y1c)
            clip = clip & page.rect
            if clip.is_empty or clip.height < 45 or clip.width < 90:
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip, alpha=False)
            pix.save(out_path)
            return {"page": page.number + 1, "caption": " ".join(txt.split())[:100]}
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    r = render_v2(a.pdf, a.out, a.fig, a.dpi, a.table)
    print("OK", r) if r else (print("MISS"), sys.exit(1))
