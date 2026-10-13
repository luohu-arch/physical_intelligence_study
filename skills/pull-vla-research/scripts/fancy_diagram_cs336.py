#!/usr/bin/env python3
"""cs336 报告风格渲染器(GitHub 文档风扁平 SVG), 库内唯一绘图渲染器:
  - 规范源自 cs336/html-report-work/FRAGMENT_SPEC.md: 卡片 #f6f8fa + 边 #d0d7de,
    墨 #1f2328 / 次 #57606a / 强调 #0b57d0 / 警示 #8a5a00, 文字 >=11.5px, 内联零依赖
  - 语义类用左侧 3px 色条 + 极淡底色(保留 data/train/loss 等语义)
  - 边: 1.5px 直线 + 7px 箭头; fb/loss 虚线; 直线穿卡自动改 L 形绕行
  - 边标签: 线中点旁 11.5px 灰字 + 白色描边 halo; labpos/loff 可手调
  - 无图例; 画布=内容+路由外接框(viewBox 原生偏移, 不平移坐标)
用法: python3 fancy_diagram_cs336.py spec.json --out arch.svg
"""
import argparse
import json

CLS = {
    "data":   ("#0969da", "#f0f6fe"),
    "train":  ("#1a7f37", "#f2f9f3"),
    "frozen": ("#57606a", "#f6f8fa"),
    "loss":   ("#8a5a00", "#fdf8ec"),
    "act":    ("#8250df", "#f8f5fe"),
    "loop":   ("#0b57d0", "#f0f4fe"),
    "env":    ("#57606a", "#f6f8fa"),
    "mem":    ("#bf3989", "#fdf3f8"),
    "reward": ("#8a5a00", "#fdf8ec"),
    "key":    ("#0b57d0", "#eef4fe"),
}
EDGE = {
    "main":  ("#57606a", 1.5, ""),
    "thick": ("#1f2328", 2.0, ""),
    "thin":  ("#8c959f", 1.2, ""),
    "fb":    ("#0b57d0", 1.5, "6 4"),
    "loss":  ("#8a5a00", 1.5, "5 3"),
}
FONT = "'PingFang SC','Hiragino Sans GB','Microsoft YaHei','Segoe UI',sans-serif"

# 排版呼吸感(2026-10-13 用户反馈: 密集 -> 加大行距/段距/字号, 卡片间隙外扩)
GAP_SCALE = 1.12   # 卡片坐标外扩系数(卡片尺寸不变, 只拉开间距)
L_FS, L_PITCH = 14.5, 21      # 卡片标题字号/行距
S_FS, S_PITCH = 12.5, 17.5    # 副文字号/行距
LS_GAP = 8                    # 标题块与副文块的段间距
E_FS = 12.5                   # 边标签字号


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def anchor_pt(n, side, pos=0.5):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    if side == "top":
        return (x + w * pos, y)
    if side == "bottom":
        return (x + w * pos, y + h)
    if side == "left":
        return (x, y + h * pos)
    return (x + w, y + h * pos)


def auto_sides(a, b):
    dx = (b["x"] + b["w"] / 2) - (a["x"] + a["w"] / 2)
    dy = (b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)
    if abs(dy) >= abs(dx):
        return ("bottom", "top") if dy > 0 else ("top", "bottom")
    return ("right", "left") if dx > 0 else ("left", "right")


def _seg_hits(p, q, r, shrink=4):
    x0, y0, x1, y1 = r[0] + shrink, r[1] + shrink, r[2] - shrink, r[3] - shrink
    steps = max(2, int(max(abs(q[0] - p[0]), abs(q[1] - p[1])) / 6) + 1)
    for i in range(steps + 1):
        t = i / steps
        if x0 <= p[0] + (q[0] - p[0]) * t <= x1 and y0 <= p[1] + (q[1] - p[1]) * t <= y1:
            return True
    return False


def _route(sa, sb, na, nb, rects):
    """直线优先; 穿过非端点卡片时改 L 形绕行(垂直边走底/顶通道, 水平边走侧通道)."""
    if not any(_seg_hits(sa, sb, r) for r in rects):
        return [sa, sb]
    if na in ("bottom", "top") or nb in ("bottom", "top"):
        my = (sa[1] + sb[1]) / 2 if na in ("bottom", "top") and nb in ("bottom", "top") else (
            sa[1] + 70 if na == "bottom" else sa[1] - 70)
        while any(_seg_hits((sa[0], my), (sb[0], my), r) for r in rects):
            my += 90 if na == "bottom" else -90
        return [sa, (sa[0], my), (sb[0], my), sb]
    mx = min(sa[0], sb[0]) - 60
    while any(_seg_hits((mx, sa[1]), (mx, sb[1]), r) for r in rects):
        mx -= 90
    return [sa, (mx, sa[1]), (mx, sb[1]), sb]


def render(spec):
    import copy
    for n in spec["nodes"]:
        n["x"], n["y"] = round(n["x"] * GAP_SCALE), round(n["y"] * GAP_SCALE)
    for p in spec.get("panels", []):
        p["x"], p["y"] = round(p["x"] * GAP_SCALE), round(p["y"] * GAP_SCALE)
    for e in spec.get("edges", []):
        if e.get("labpos"):
            e["labpos"] = [round(e["labpos"][0] * GAP_SCALE), round(e["labpos"][1] * GAP_SCALE)]
    nodes = {n["id"]: n for n in spec["nodes"]}
    rpts = []

    edges_out, labels_out = [], []
    for e in spec.get("edges", []):
        a, b = nodes[e["from"]], nodes[e["to"]]
        sa_, sb_ = auto_sides(a, b)
        na, nb = e.get("out", sa_), e.get("in", sb_)
        sa = anchor_pt(a, na, e.get("pos_out", 0.5))
        sb = anchor_pt(b, nb, e.get("pos_in", 0.5))
        rects = [(n2["x"], n2["y"], n2["x"] + n2["w"], n2["y"] + n2["h"])
                 for nid2, n2 in nodes.items() if nid2 not in (e["from"], e["to"])]
        pts = _route(sa, sb, na, nb, rects)
        rpts += pts
        c, w, dash = EDGE[e.get("style", "main")]
        st = e.get("style", "main")
        d_a = f' stroke-dasharray="{dash}"' if dash else ""
        d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}" + "".join(f" L {q[0]:.1f} {q[1]:.1f}" for q in pts[1:])
        edges_out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}"{d_a} '
                         f'marker-end="url(#arw-{st})"/>')
        if e.get("label"):
            mx, my = (sa[0] + sb[0]) / 2, (sa[1] + sb[1]) / 2
            off = e.get("loff", (0, 0))
            lp = e.get("labpos")
            if lp:
                mx, my = lp[0], lp[1]
            else:
                mx, my = mx + off[0], my + off[1] - 8
            rpts += [(mx, my), (mx + 120, my + 12)]
            labels_out.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="12.5" fill="#57606a" '
                              f'text-anchor="middle" paint-order="stroke" stroke="#ffffff" '
                              f'stroke-width="3">{esc(e["label"])}</text>')

    xs = [n["x"] for n in spec["nodes"]] + [n["x"] + n["w"] for n in spec["nodes"]] + [p[0] for p in rpts]
    ys = [n["y"] for n in spec["nodes"]] + [n["y"] + n["h"] for n in spec["nodes"]] + [p[1] for p in rpts]
    for p in spec.get("panels", []):
        xs += [p["x"], p["x"] + p["w"]]
        ys += [p["y"], p["y"] + p["h"]]
    pad, head, foot = 26, 56, 28
    x0, y0, x1, y1 = min(xs) - pad, min(ys) - head, max(xs) + pad, max(ys) + pad
    W, H = int(x1 - x0), int(y1 - y0 + foot)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="{x0} {y0} {W} {H}" font-family="{FONT}" role="img">',
           f'<title>{esc(spec.get("title", ""))}</title>',
           f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" fill="#ffffff"/>']
    if spec.get("title"):
        out.append(f'<text x="{x0 + 22}" y="{y0 + 30}" font-size="14" font-weight="700" '
                   f'fill="#1f2328">{esc(spec["title"])}</text>')
        if spec.get("subtitle"):
            out.append(f'<text x="{x1 - 22}" y="{y0 + 30}" font-size="12" fill="#57606a" '
                       f'text-anchor="end">{esc(spec["subtitle"])}</text>')
    out.append('<defs>')
    for st, (c, _, _) in EDGE.items():
        out.append(f'<marker id="arw-{st}" viewBox="0 0 10 10" refX="9" refY="5" '
                   f'markerWidth="7" markerHeight="7" orient="auto">'
                   f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    out.append('</defs>')

    for p in spec.get("panels", []):
        out.append(f'<rect x="{p["x"]}" y="{p["y"]}" width="{p["w"]}" height="{p["h"]}" rx="8" '
                   f'fill="#f6f8fa" stroke="#d8dee4"/>')
        for li, l in enumerate(p["label"].split("\n")[:2]):
            out.append(f'<text x="{p["x"] + 14}" y="{p["y"] + 20 + li * 15}" font-size="11.5" '
                       f'font-weight="600" fill="#57606a">{esc(l)}</text>')

    out += edges_out

    for n in spec["nodes"]:
        cls = n.get("cls", "loop")
        bar, fill = CLS.get(cls, CLS["loop"])
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
                   f'fill="{fill}" stroke="#d0d7de" stroke-width="1"/>')
        out.append(f'<line x1="{x}" y1="{y + 6}" x2="{x}" y2="{y + h - 6}" stroke="{bar}" stroke-width="3"/>')
        labs = n["label"].split("\n")
        subs = n.get("sub", "").split("\n") if n.get("sub") else []
        ty = y + (h - (len(labs) * L_PITCH + (LS_GAP if subs else 0) + len(subs) * S_PITCH)) / 2 + 14
        for i, l in enumerate(labs):
            out.append(f'<text x="{x + w / 2}" y="{ty + i * L_PITCH}" font-size="14.5" font-weight="700" '
                       f'fill="#1f2328" text-anchor="middle">{esc(l)}</text>')
        for i, s in enumerate(subs):
            out.append(f'<text x="{x + w / 2}" y="{ty + len(labs) * L_PITCH + LS_GAP + i * S_PITCH}" font-size="12.5" '
                       f'fill="#57606a" text-anchor="middle">{esc(s)}</text>')
    out += labels_out
    if spec.get("foot"):
        out.append(f'<text x="{x1 - 22}" y="{y1 + 16}" font-size="10.5" fill="#8c959f" '
                   f'text-anchor="end">{esc(spec["foot"])}</text>')
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
