#!/usr/bin/env python3
"""v4 图表提取：v3 审查发现的失败类修复。

- 图: 先收集图形核心(图片+绘图), 仅吸收紧邻(±20pt)的图内文字标签;
  页眉/作者行拒绝并向下推挤 → 不再混入正文
- 表: 行级图注锚点(支持 "Table N:" / "Table N." / "Table N |" / "Table N 无分隔"
  / "表N"); 规则线聚类(x 中心过滤 + gap<=90)定位表体;
  兜底: 单元格矩形密集区 / 大图截图表 / 跨页表体(下一页顶部) / 纯文本连续行;
  结构校验(core_ok 且 rows>=2), 无效返回 valid=False 供上层换选
"""
import re
import sys

import fitz

PAD = 14
HEADER_RE = re.compile(
    r"published as a conference paper|proceedings of|arXiv:\d{4}\.\d{4,5}"
    r"|under review|©\s*\d{4}|ieee (?:icra|ra-l|international)|confidential"
    r"|volume \d+.*number \d+|journal of|[A-Z][a-z]+ [A-Z][a-z]+ et al\.", re.I)


def _headerish(txt, y0):
    return y0 < 62 or bool(HEADER_RE.search(txt[:120]))


def _xov(bx, u):
    ov = min(bx.x1, u.x1) - max(bx.x0, u.x0)
    return max(0.0, ov) / max(1.0, bx.width)


def find_anchor(page, label, strict=True):
    """行级题注锚点。strict: label + 分隔符(: . — – |)开头;
    宽松模式再接受 label + 大写字母/括号/中文(DeepMind 无分隔符风格)。"""
    base = r"\s*".join(re.escape(w) for w in label.split())
    if strict:
        pat = re.compile(base + r"\s*[:.\u2013\u2014|]", re.I)
    else:
        pat = re.compile(base + r"[\s][(A-Z\u4e00-\u9fff]")
    try:
        td = page.get_text("dict")
    except Exception:
        return None, None, None
    for b in td["blocks"]:
        if b.get("type") != 0:
            continue
        for line in b["lines"]:
            txt = "".join(s["text"] for s in line["spans"]).strip()
            nrm = " ".join(txt.split())
            if pat.match(nrm):
                return fitz.Rect(line["bbox"]), fitz.Rect(b["bbox"]), txt
    return None, None, None


def _in(r, x0, x1, y0, y1):
    return r.y0 >= y0 - 8 and r.y1 <= y1 + 8 and r.x1 > x0 and r.x0 < x1


def _union(boxes):
    u = fitz.Rect(boxes[0])
    for b in boxes[1:]:
        u |= b
    return u


def _absorb_rows(u, page, y_lo, x_lo, x_hi):
    """吸收 [y_lo±16] 垂直范围内、水平邻近的文本行进 union。"""
    for b in page.get_text("blocks"):
        bx = fitz.Rect(b[:4])
        t = b[4].strip()
        cy = (bx.y0 + bx.y1) / 2
        if (bx.y0 > y_lo and u.y0 - 16 <= cy <= u.y1 + 16
                and bx.x1 > u.x0 - 30 and bx.x0 < u.x1 + 30
                and len(t) < 400 and not _headerish(t, bx.y0)
                and bx.x1 > x_lo and bx.x0 < x_hi):
            u |= bx
    return u


def render_v4(pdf_path, out_path, fig_no, dpi=200, table=False, pad=PAD,
              fig_side=34, fig_top=20, fig_bot=12, fig_pb=10):
    doc = fitz.open(pdf_path)
    kind = "Table" if table else "Figure"
    cjk = "表" if table else "图"
    labels = [f"{kind} {fig_no}", f"{cjk} {fig_no}", f"{cjk}{fig_no}"]
    if not table:
        labels.append(f"Fig. {fig_no}")
    for strict in (True, False):
        for pidx in range(len(doc)):
            page = doc[pidx]
            for lab in labels:
                a, sp, txt = find_anchor(page, lab, strict)
                if a is None:
                    continue
                H = page.rect.height
                if table:
                    # 多行图注: 块底距锚点顶 <90pt 视为图注延续
                    cap_bottom = sp.y1 if sp.y1 - a.y0 < 90 else a.y1
                    y0 = cap_bottom + 2
                    y1 = min(page.rect.height - 26, cap_bottom + H * 0.55)
                else:
                    y0, y1 = max(30, a.y0 - H * 0.52), a.y0 - 2
                col_x0, col_x1 = min(sp.x0, a.x0) - 4, max(sp.x1, a.x1) + 4
                # 图形核心: 嵌入图片 + 绘图(表模式接受 0 高度细规则线)
                G = []
                for img in page.get_images(full=True):
                    try:
                        for r in page.get_image_rects(img[0]):
                            if _in(r, col_x0, col_x1, y0, y1):
                                G.append(r)
                    except Exception:
                        pass
                try:
                    drawings = page.get_drawings()
                except Exception:
                    drawings = []
                for d in drawings:
                    r = d["rect"]
                    if table:
                        ok = r.width > 18
                    else:
                        ok = r.width > 18 and r.height > 10
                    if ok and _in(r, col_x0, col_x1, y0, y1):
                        G.append(r)
                crosspage = False
                nxt = None
                if table:
                    # 规则线聚类: 以最宽线为中心, x 偏离<=90 的连续细线(gap<=90)
                    lines = [d["rect"] for d in drawings
                             if d["rect"].width > 40 and d["rect"].height < 3.5
                             and d["rect"].y0 > cap_bottom - 2
                             and d["rect"].y1 <= y1
                             and d["rect"].x1 > col_x0 and d["rect"].x0 < col_x1]
                    cells = [d["rect"] for d in drawings
                             if d["rect"].y0 > cap_bottom - 2
                             and d["rect"].y1 <= y1
                             and d["rect"].x1 > col_x0 and d["rect"].x0 < col_x1
                             and d["rect"].width > 6 and d["rect"].height > 2]
                    if lines:
                        wide = max(lines, key=lambda r: r.width)
                        cx = (wide.x0 + wide.x1) / 2
                        lines = [r for r in lines
                                 if abs((r.x0 + r.x1) / 2 - cx) <= 90]
                        lines.sort(key=lambda r: r.y0)
                        # 0 高度规则线在并集中被视为空矩形, 需增肥后再并
                        fat = [fitz.Rect(r.x0, r.y0 - 1.2, r.x1, r.y1 + 1.2)
                               for r in lines]
                        cluster = [fat[0]]
                        for r in fat[1:]:
                            if r.y0 - cluster[-1].y1 <= 90:
                                cluster.append(r)
                            else:
                                break
                        u = fitz.Rect(cluster[0])
                        for r in cluster[1:]:
                            u |= r
                        core_ok = len(cluster) >= 2 or len(cells) >= 8
                    elif len(cells) >= 8:
                        u = _union(cells)
                        core_ok = True
                    else:
                        big_imgs = [r for r in G
                                    if r.width > 150 and r.height > 60]
                        nxt = doc[pidx + 1] if pidx + 1 < len(doc) else None
                        cp = None
                        if nxt is not None:
                            try:
                                cp_lines = [d["rect"] for d in nxt.get_drawings()
                                            if d["rect"].width > 40
                                            and d["rect"].height < 3.5
                                            and d["rect"].y1 <= nxt.rect.height * 0.45
                                            and d["rect"].x1 > col_x0
                                            and d["rect"].x0 < col_x1]
                            except Exception:
                                cp_lines = []
                            cp_lines.sort(key=lambda r: r.y0)
                        if len(cp_lines) >= 2:
                            cp = fitz.Rect(cp_lines[0].x0, cp_lines[0].y0 - 1.2,
                                           cp_lines[0].x1, cp_lines[0].y1 + 1.2)
                            for r in cp_lines[1:]:
                                fr = fitz.Rect(r.x0, r.y0 - 1.2, r.x1, r.y1 + 1.2)
                                if fr.y0 - cp.y1 <= 90:
                                    cp |= fr
                            if cp.height < 40:
                                cp = None
                        if big_imgs:
                            u = _union(big_imgs)
                            core_ok = True
                        elif cp is not None:
                            u = fitz.Rect(cp)
                            u = _absorb_rows(u, nxt, -1, col_x0 - 20, col_x1 + 20)
                            core_ok = True
                            crosspage = True
                        else:
                            # 纯文本表: 图注下方连续文本行
                            trows = sorted(
                                [fitz.Rect(b[:4]) for b in page.get_text("blocks")
                                 if fitz.Rect(b[:4]).y0 > cap_bottom
                                 and fitz.Rect(b[:4]).y1 <= y1
                                 and fitz.Rect(b[:4]).x1 > col_x0
                                 and fitz.Rect(b[:4]).x0 < col_x1],
                                key=lambda r: r.y0)
                            contig = []
                            for r in trows:
                                if not contig or r.y0 - contig[-1].y1 <= 18:
                                    contig.append(r)
                                else:
                                    break
                            if len(contig) >= 3:
                                u = _union(contig)
                                core_ok = True
                            else:
                                continue
                    if not crosspage:
                        u = _absorb_rows(u, page, cap_bottom,
                                         col_x0 - 20, col_x1 + 20)
                        for r in G:
                            if r.y0 >= u.y0 - 12 and r.y1 <= u.y1 + 12:
                                u |= r
                else:
                    if not G:
                        continue  # 无图形核心 → 换页/换写法, 绝不用纯文本凑
                    ug = _union(G)
                    u = fitz.Rect(ug)
                    # 吸收图内标签: 顶部标题行/底部注行(垂直紧邻+水平重叠)
                    # 或侧边图例(垂直大部分在图内+水平紧邻)
                    for b in page.get_text("blocks"):
                        bx = fitz.Rect(b[:4])
                        t = b[4].strip()
                        if _headerish(t, bx.y0) or len(t) > 250:
                            continue
                        if bx.x1 < col_x0 + 6 or bx.x0 > col_x1 - 6:
                            continue  # 栏外的侧边竖排水印等
                        vov = max(0.0, min(bx.y1, u.y1 + fig_bot) - max(bx.y0, u.y0 - fig_top))
                        near_v = (bx.y0 >= u.y0 - fig_top
                                  and bx.y1 <= min(u.y1 + fig_bot, a.y0 - 1)
                                  and _xov(bx, u) >= 0.3)
                        near_h = (vov >= 0.6 * bx.height
                                  and bx.x0 < u.x1 + fig_side and bx.x1 > u.x0 - fig_side)
                        if near_v or near_h:
                            u |= bx
                if table:
                    pt, pb, ps = pad, pad, pad
                else:
                    pt, pb, ps = 26, fig_pb, 22
                tgt = nxt if crosspage else page
                x0 = max(u.x0 - ps, max(26, tgt.rect.x0 + 16))
                x1 = min(u.x1 + ps, tgt.rect.x1 - 16)
                y_top = u.y0 - pt
                if table and not crosspage:
                    y_top = max(y_top, cap_bottom + 2)
                y_bot = u.y1 + pb if table else min(u.y1 + pb, a.y0 - 3)
                # 页眉/作者行推挤: 短文本行(高<80pt, 排除 arXiv 侧边竖排水印)
                # 与裁剪区顶部相交才下推, 且推挤后高度不足则忽略
                for b in tgt.get_text("blocks"):
                    bx = fitz.Rect(b[:4])
                    t = b[4].strip()
                    if (_headerish(t, bx.y0) and bx.height < 80
                            and bx.x1 > x0 and bx.x0 < x1
                            and bx.y1 > y_top - 2 and bx.y0 < min(y_bot, u.y0)):
                        if bx.y1 + 4 < y_bot - 40:
                            y_top = max(y_top, bx.y1 + 4)
                clip = fitz.Rect(x0, y_top, x1, y_bot) & tgt.rect
                if clip.is_empty or clip.height < 45 or clip.width < 90:
                    continue
                info = {"page": page.number + 1,
                        "caption": " ".join(txt.split())[:100], "valid": True}
                if table:
                    rows = sum(1 for b in tgt.get_text("blocks")
                               if fitz.Rect(b[:4]).intersects(clip)
                               and len(b[4].strip()) > 3)
                    img_tab = any(r.height > 50 and r.width > 150
                                  for r in (G if not crosspage else [])
                                  if r.intersects(clip))
                    if not core_ok or (rows < 2 and not img_tab):
                        info["valid"] = False
                        return info
                pix = tgt.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72),
                                     clip=clip, alpha=False)
                pix.save(out_path)
                if crosspage:
                    info["page"] = f"{page.number + 1}-{nxt.number + 1}"
                return info
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    r = render_v4(a.pdf, a.out, a.fig, a.dpi, a.table)
    print("OK", r) if r else (print("MISS"), sys.exit(1))
