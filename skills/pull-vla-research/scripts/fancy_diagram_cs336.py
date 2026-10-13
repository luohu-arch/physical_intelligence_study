#!/usr/bin/env python3
"""cs336 报告风格渲染器(GitHub 文档风扁平 SVG):
  - 卡片 #f6f8fa 底 + #d0d7de 1px 边框, rx=6, 无阴影/渐变/图标徽章
  - 语义类用左侧 3px 色条 + 极淡底色差异(保留库内 data/train/loss 等语义, 视觉不吵)
  - 边: 1.5px 直线 + 7px 小箭头; loss/fb 用橙/蓝虚线
  - 边标签: 线中点旁 12px 灰字, 白色描边 halo(替代药丸)
  - 无图例行; 小标题(13px)+副标题(12px), 画布紧凑(内容边界+24 边距)
用法: python3 fancy_diagram_cs336.py spec.json --out arch.svg
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fancy_diagram import esc, tw, anchor_pt, auto_sides  # noqa: E402

# 语义类 -> (左条色, 极淡底色, 文字主色)
CLS = {
    "data":   ("#0969da", "#f0f6fe", "#1f2328"),
    "train":  ("#1a7f37", "#f2f9f3", "#1f2328"),
    "frozen": ("#57606a", "#f6f8fa", "#1f2328"),
    "loss":   ("#8a5a00", "#fdf8ec", "#1f2328"),
    "act":    ("#8250df", "#f8f5fe", "#1f2328"),
    "loop":   ("#0b57d0", "#f0f4fe", "#1f2328"),
    "env":    ("#57606a", "#f6f8fa", "#1f2328"),
    "mem":    ("#bf3989", "#fdf3f8", "#1f2328"),
    "reward": ("#8a5a00", "#fdf8ec", "#1f2328"),
    "key":    ("#0b57d0", "#eef4fe", "#0b2966"),
}
EDGE = {
    "main":  ("#57606a", 1.5, ""),
    "thick": ("#1f2328", 2.0, ""),
    "thin":  ("#8c959f", 1.2, ""),
    "fb":    ("#0b57d0", 1.5, "6 4"),
    "loss":  ("#8a5a00", 1.5, "5 3"),
}
FONT = "'PingFang SC','Hiragino Sans GB','Microsoft YaHei','Segoe UI',sans-serif"


def render(spec):
    nodes = {n["id"]: n for n in spec["nodes"]}
    xs = [n["x"] for n in spec["nodes"]] + [n["x"] + n["w"] for n in spec["nodes"]]
    ys = [n["y"] for n in spec["nodes"]] + [n["y"] + n["h"] for n in spec["nodes"]]
    for p in spec.get("panels", []):
        xs += [p["x"], p["x"] + p["w"]]
        ys += [p["y"], p["y"] + p["h"]]
    x0, y0, x1, y1 = min(xs) - 16, min(ys) - 16, max(xs) + 16, max(ys) + 16
    # 标题带
    head_h = 46 if spec.get("title") else 16
    dx, dy = -x0 + 20, -y0 + head_h
    W, H = int(x1 - x0 + 40), int(y1 - y0 + head_h + 30)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="{FONT}" role="img">',
           f'<title>{esc(spec.get("title", ""))}</title>',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    if spec.get("title"):
        out.append(f'<text x="20" y="28" font-size="14" font-weight="700" '
                   f'fill="#1f2328">{esc(spec["title"])}</text>')
        if spec.get("subtitle"):
            out.append(f'<text x="{W - 20}" y="28" font-size="12" fill="#57606a" '
                       f'text-anchor="end">{esc(spec["subtitle"])}</text>')

    def T(n):
        return (n["x"] + dx, n["y"] + dy)

    for p in spec.get("panels", []):
        px, py = p["x"] + dx, p["y"] + dy
        out.append(f'<rect x="{px}" y="{py}" width="{p["w"]}" height="{p["h"]}" rx="8" '
                   f'fill="#f6f8fa" stroke="#d8dee4" stroke-dasharray="0"/>')
        for li, l in enumerate(p["label"].split("\n")[:2]):
            out.append(f'<text x="{px + 14}" y="{py + 20 + li * 15}" font-size="11.5" '
                       f'font-weight="600" fill="#57606a">{esc(l)}</text>')

    edges_out, labels_out = [], []
    for e in spec.get("edges", []):
        a, b = nodes[e["from"]], nodes[e["to"]]
        sa_, sb_ = auto_sides(a, b)
        na, nb = e.get("out", sa_), e.get("in", sb_)
        sa = (anchor_pt(a, na, e.get("pos_out", 0.5))[0] + dx,
              anchor_pt(a, na, e.get("pos_out", 0.5))[1] + dy)
        sb = (anchor_pt(b, nb, e.get("pos_in", 0.5))[0] + dx,
              anchor_pt(b, nb, e.get("pos_in", 0.5))[1] + dy)
        c, w, dash = EDGE[e.get("style", "main")]
        st = e.get("style", "main")
        d_a = f' stroke-dasharray="{dash}"' if dash else ""
        edges_out.append(f'<line x1="{sa[0]:.1f}" y1="{sa[1]:.1f}" x2="{sb[0]:.1f}" y2="{sb[1]:.1f}" '
                         f'stroke="{c}" stroke-width="{w}"{d_a} marker-end="url(#arw-{st})"/>')
        if e.get("label"):
            mx, my = (sa[0] + sb[0]) / 2, (sa[1] + sb[1]) / 2
            off = e.get("loff", (0, 0))
            lp = e.get("labpos")
            if lp:
                mx, my = lp[0] + dx, lp[1] + dy
            else:
                mx, my = mx + off[0], my + off[1] - 8
            labels_out.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="11.5" fill="#57606a" '
                              f'text-anchor="middle" paint-order="stroke" stroke="#ffffff" '
                              f'stroke-width="3">{esc(e["label"])}</text>')

    out.append('<defs>')
    for st, (c, _, _) in EDGE.items():
        out.append(f'<marker id="arw-{st}" viewBox="0 0 10 10" refX="9" refY="5" '
                   f'markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    out.append('</defs>')
    out += [f'<g>{e}</g>' for e in edges_out]

    for n in spec["nodes"]:
        cls = n.get("cls", "loop")
        bar, fill, ink = CLS.get(cls, CLS["loop"])
        x, y = T(n)
        w, h = n["w"], n["h"]
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                   f'fill="{fill}" stroke="#d0d7de" stroke-width="1"/>')
        out.append(f'<path d="M {x} {y + 6} q 0 -6 6 -6 h 0 v 0" fill="none"/>')
        out.append(f'<line x1="{x}" y1="{y + 6}" x2="{x}" y2="{y + h - 6}" stroke="{bar}" stroke-width="3"/>')
        out.append(f'<path d="M {x + 3} {y} a 3 3 0 0 0 -3 3" stroke="none"/>')
        labs = n["label"].split("\n")
        subs = n.get("sub", "").split("\n") if n.get("sub") else []
        ty = y + (h - (len(labs) * 18 + len(subs) * 15 + (4 if subs else 0))) / 2 + 13
        for i, l in enumerate(labs):
            out.append(f'<text x="{x + w / 2}" y="{ty + i * 18}" font-size="13" font-weight="700" '
                       f'fill="{ink}" text-anchor="middle">{esc(l)}</text>')
        for i, s in enumerate(subs):
            out.append(f'<text x="{x + w / 2}" y="{ty + len(labs) * 18 + 2 + i * 15}" font-size="11.5" '
                       f'fill="#57606a" text-anchor="middle">{esc(s)}</text>')
    out += labels_out
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out")
    args = ap.parse_args()
    spec = json.load(open(args.spec))
    svg = render(spec)
    out = args.out or args.spec.replace(".spec.json", ".svg")
    open(out, "w").write(svg + "\n")
    print("SVG", out)


if __name__ == "__main__":
    main()
