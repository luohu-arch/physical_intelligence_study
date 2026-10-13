#!/usr/bin/env python3
"""v1 风格渲染器(用户锁定的视觉风格, 2026-10 重生成版):
以 3f15aa9 原版为基底 —— 贝塞尔曲线连线 / 原线宽 / 7.5px 箭头 / 无图例行,
叠加此后积累的全部正确性修复:
  - 徽章-文字几何(y+56 起, 高度公式与 spec 一致)
  - 药丸避障求解器(卡片/彼此/锚点区/路径, own 豁免, 4 级兜底 + WARN 注释)
  - 画布自动扩高(底部走廊兜底)
  - panel 标签支持两行
  - 曲线穿卡的边自动回退到正交路由(防箭头消失)
用法: python3 fancy_diagram_v1.py spec.json --out arch.svg [--png]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fancy_diagram import (PALETTE, EDGE_STYLE as _EDGE_STYLE_ORIG, FONT, esc, tw,
                           anchor_pt, auto_sides, edge_path, icon_g,
                           resolve_and_render_pills)  # noqa: E402

# v1 原版线宽与线型(3f15aa9): 细线 + 长/短虚线, 箭头 7.5px
EDGE_STYLE = {
    "main":  ("#607d8b", 2.4, ""),
    "thick": ("#37474f", 3.2, ""),
    "thin":  ("#90a4ae", 1.8, ""),
    "fb":    ("#1565c0", 2.2, "7 5"),
    "loss":  ("#c62828", 2.2, "6 4"),
}


def bezier_pts(sa, na, sb, nb, k=None):
    """原版 edge_path 的贝塞尔曲线, 返回 (d, mid, 采样点列)."""
    NORM = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
    k = k if k is not None else max(46, (abs(sb[0] - sa[0]) + abs(sb[1] - sa[1])) * 0.42)
    (u1, v1), (u2, v2) = NORM[na], NORM[nb]
    c1 = (sa[0] + u1 * k, sa[1] + v1 * k)
    c2 = (sb[0] + u2 * k, sb[1] + v2 * k)
    d = (f"M {sa[0]:.1f} {sa[1]:.1f} C {c1[0]:.1f} {c1[1]:.1f}, "
         f"{c2[0]:.1f} {c2[1]:.1f}, {sb[0]:.1f} {sb[1]:.1f}")
    pts = []
    for i in range(17):
        t = i / 16
        x = (1 - t) ** 3 * sa[0] + 3 * (1 - t) ** 2 * t * c1[0] + 3 * (1 - t) * t ** 2 * c2[0] + t ** 3 * sb[0]
        y = (1 - t) ** 3 * sa[1] + 3 * (1 - t) ** 2 * t * c1[1] + 3 * (1 - t) * t ** 2 * c2[1] + t ** 3 * sb[1]
        pts.append((x, y))
    mid = pts[8]
    return d, mid, pts


def seg_hits_rect(p, q, r, shrink=4):
    x0, y0, x1, y1 = r[0] + shrink, r[1] + shrink, r[2] - shrink, r[3] - shrink
    steps = max(2, int(max(abs(q[0] - p[0]), abs(q[1] - p[1])) / 6) + 1)
    for i in range(steps + 1):
        t = i / steps
        if x0 <= p[0] + (q[0] - p[0]) * t <= x1 and y0 <= p[1] + (q[1] - p[1]) * t <= y1:
            return True
    return False


def render(spec):
    W, H = spec["canvas"]
    nodes0 = {n["id"]: n for n in spec["nodes"]}
    for e in spec.get("edges", []):
        if e["from"] in nodes0 and e["to"] in nodes0:
            a, b = nodes0[e["from"]], nodes0[e["to"]]
            sa_, sb_ = auto_sides(a, b)
            na, nb = e.get("out", sa_), e.get("in", sb_)
            k = e.get("k", 60)
            if na == nb == "bottom":
                H = max(H, max(a["y"] + a["h"], b["y"] + b["h"]) + k + 100)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="{FONT}">',
           f'<style>text{{font-family:{FONT};}} .lbl{{font-weight:700;}}</style>',
           "<defs>"]
    for cls, (a, b, s, ink) in PALETTE.items():
        out.append(f'<linearGradient id="g-{cls}" x1="0" y1="0" x2="0" y2="1">'
                   f'<stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>')
    out.append('<linearGradient id="g-title" x1="0" y1="0" x2="1" y2="0">'
               '<stop offset="0" stop-color="#00695c"/><stop offset="1" stop-color="#1565c0"/></linearGradient>')
    for st, (c, _, _) in EDGE_STYLE.items():
        out.append(f'<marker id="arr-{st}" viewBox="0 0 10 10" refX="9" refY="5" '
                   f'markerWidth="7.5" markerHeight="7.5" orient="auto-start-reverse">'
                   f'<path d="M0 0 L10 5 L0 10 z" fill="{c}"/></marker>')
    out.append("</defs>")
    out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    out.append(f'<rect x="34" y="30" width="5" height="42" rx="2.5" fill="url(#g-title)"/>')
    out.append(f'<text x="52" y="52" font-size="25" font-weight="800" fill="#1a2733">{esc(spec["title"])}</text>')
    if spec.get("subtitle"):
        out.append(f'<text x="52" y="74" font-size="13.5" fill="#78909c">{esc(spec["subtitle"])}</text>')
    if spec.get("foot"):
        out.append(f'<text x="{W - 34}" y="{H - 18}" font-size="11.5" fill="#a0aab4" '
                   f'text-anchor="end">{esc(spec["foot"])}</text>')

    for p in spec.get("panels", []):
        out.append(f'<rect x="{p["x"]}" y="{p["y"]}" width="{p["w"]}" height="{p["h"]}" rx="18" '
                   f'fill="#fbfdff" stroke="#b7c6d6" stroke-width="1.4" stroke-dasharray="8 5"/>')
        plabels = p["label"].split("\n") if "\n" in p["label"] else [p["label"]]
        plw = max(tw(l, 12.5) for l in plabels) + 24
        ph = 26 + 17 * (len(plabels) - 1)
        out.append(f'<rect x="{p["x"] + 16}" y="{p["y"] + 13}" width="{plw}" '
                   f'height="{ph}" rx="13" fill="#eef3f8" stroke="#b7c6d6"/>')
        for li, pl in enumerate(plabels):
            out.append(f'<text x="{p["x"] + 27}" y="{p["y"] + 30 + li * 17}" font-size="12.5" '
                       f'fill="#4a6579">{esc(pl)}</text>')

    nodes = {n["id"]: n for n in spec["nodes"]}

    # 边: 贝塞尔为主, 穿卡时回退正交
    pill_jobs, paths, path_segs = [], [], []
    for e in spec.get("edges", []):
        a, b = nodes[e["from"]], nodes[e["to"]]
        sa_, sb_ = auto_sides(a, b)
        na, nb = e.get("out", sa_), e.get("in", sb_)
        sa = anchor_pt(a, na, e.get("pos_out", 0.5))
        sb = anchor_pt(b, nb, e.get("pos_in", 0.5))
        rects = [(n2["x"], n2["y"], n2["x"] + n2["w"], n2["y"] + n2["h"])
                 for nid2, n2 in nodes.items() if nid2 not in (e["from"], e["to"])]
        d, mid, pts = bezier_pts(sa, na, sb, nb, e.get("k"))
        crossing = any(seg_hits_rect(pts[i], pts[i + 1], r) for r in rects for i in range(len(pts) - 1))
        if crossing:
            d, mid, opts = edge_path(sa, na, sb, nb, e.get("k"), rects, (W, H))
            pts = opts
        st = e.get("style", "main")
        c, w, dash = EDGE_STYLE[st]
        dash_a = f' stroke-dasharray="{dash}"' if dash else ""
        paths.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}"{dash_a} '
                     f'marker-end="url(#arr-{st})" opacity="0.92"/>')
        seg_idx = list(range(len(path_segs), len(path_segs) + max(0, len(pts) - 1)))
        path_segs.extend((pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        if e.get("label"):
            off = e.get("loff", (0, 0))
            lp = e.get("labpos")
            pill_jobs.append({"text": e["label"],
                              "cx": (lp[0] if lp else mid[0] + off[0]),
                              "cy": (lp[1] if lp else mid[1] + off[1]),
                              "color": EDGE_STYLE[st][0], "ax": sa[0], "ay": sa[1],
                              "bx": sb[0], "by": sb[1], "pin": bool(e.get("labpos")),
                              "own": seg_idx})
    out += paths

    for n in spec["nodes"]:
        cls = n.get("cls", "loop")
        a, b, s, ink = PALETTE[cls]
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        bw = 3.2 if cls in ("key", "train") else 2.0
        dash = ' stroke-dasharray="7 4"' if cls == "loss" else ""
        out.append(f'<rect x="{x + 2.5}" y="{y + 3.5}" width="{w}" height="{h}" rx="14" '
                   f'fill="#37474f" opacity="0.13"/>')
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="url(#g-{cls})" '
                   f'stroke="{s}" stroke-width="{bw}"{dash}/>')
        icon = n.get("icon")
        if icon:
            bx, by = x + w / 2, y
            out.append(f'<circle cx="{bx}" cy="{by}" r="24" fill="#ffffff" stroke="{s}" stroke-width="1.8"/>')
            out.append(f'<circle cx="{bx}" cy="{by}" r="27.5" fill="none" stroke="{s}" '
                       f'stroke-width="1" opacity="0.35"/>')
            out.append(icon_g(icon, ink, bx, by, 27))
            ty = y + 56
        else:
            ty = y + h / 2 - 8
        lines = n["label"].split("\n")
        if len(lines) == 1:
            out.append(f'<text x="{x + w / 2}" y="{ty + 7}" font-size="16" class="lbl" fill="{ink}" '
                       f'text-anchor="middle">{esc(n["label"])}</text>')
        else:
            out.append(f'<text x="{x + w / 2}" y="{ty}" font-size="15" class="lbl" fill="{ink}" '
                       f'text-anchor="middle">' +
                       f'<tspan x="{x + w / 2}" dy="0">{esc(lines[0])}</tspan>'
                       f'<tspan x="{x + w / 2}" dy="20">{esc(lines[1])}</tspan></text>')
        if n.get("sub"):
            sy = ty + (30 if len(lines) > 1 else 26)
            if len(lines) > 1:
                out.append(f'<text x="{x + w / 2}" y="{sy}" font-size="12" fill="#5f6b7a" '
                           f'text-anchor="middle">{esc(n["sub"])}</text>')
            else:
                for i, part in enumerate(n["sub"].split("\n")):
                    out.append(f'<text x="{x + w / 2}" y="{sy + i * 17}" font-size="12" fill="#5f6b7a" '
                               f'text-anchor="middle">{esc(part)}</text>')
        if n.get("tag"):
            t = esc(n["tag"])
            tw_ = tw(n["tag"], 11.5) + 18
            out.append(f'<g><rect x="{x + w - tw_ - 8}" y="{y + 8}" width="{tw_:.1f}" height="21" rx="10.5" '
                       f'fill="{s}" opacity="0.14"/>'
                       f'<text x="{x + w - tw_ / 2 - 8:.1f}" y="{y + 22.5}" font-size="11.5" font-weight="700" '
                       f'fill="{ink}" text-anchor="middle">{t}</text></g>')
    out += resolve_and_render_pills(pill_jobs, spec["nodes"], W, H, path_segs)
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out")
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    svg = render(spec)
    out = args.out or args.spec.replace(".spec.json", ".svg")
    open(out, "w").write(svg + "\n")
    print("SVG", out)
    if args.png:
        subprocess.run(["qlmanage", "-t", "-s", "1600", "-o", os.path.dirname(out) or ".", out])


if __name__ == "__main__":
    import subprocess  # noqa: E402
    main()
