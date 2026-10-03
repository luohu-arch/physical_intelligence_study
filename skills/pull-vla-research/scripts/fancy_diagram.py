#!/usr/bin/env python3
"""fancy_diagram.py — v3 SVG 架构图生成器（内嵌 ml-paper-icons 图标）。

用法:
  python3 fancy_diagram.py spec.json            # 输出 spec 同目录 arch.svg
  python3 fancy_diagram.py spec.json --png      # 同时 soffice 转 arch.png（校验用）

spec 结构:
{
  "title": "...", "subtitle": "...", "foot": "可选脚注",
  "canvas": [1280, 640],
  "panels": [{"label": "...", "x":..,"y":..,"w":..,"h":..}],   # 虚线分组框(可选)
  "nodes": [
    {"id":"img","label":"多视角 RGB","sub":"3 相机 224x224","icon":"camera",
     "cls":"data","x":40,"y":100,"w":210,"h":86,"tag":"3B"}
  ],
  "edges": [
    {"from":"img","to":"vlm","label":"跨注意力","style":"main",     # main|thick|thin|fb|loss
     "out":"right","pos_out":0.5,"in":"left","pos_in":0.5}        # 锚点可省略自动判断
  ],
  "legend": ["data","train","frozen"]                             # 要展示的语义类
}

语义类与 mermaid v2 house style 同色系; 图标优先 duotone(双色填充) -> lucide -> phosphor -> tabler。
"""
import argparse
import html
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ICON_ROOT = os.path.join(HERE, "..", "assets", "ml-paper-icons")

# v2 语义配色延续: (渐变浅端, 渐变深端, 边框, 文字)
PALETTE = {
    "data":   ("#f1f9f1", "#e2f0e2", "#2e7d32", "#1b5e20"),
    "frozen": ("#f0f7fe", "#dcecfb", "#1565c0", "#0d47a1"),
    "train":  ("#fff6e8", "#ffedd4", "#ef6c00", "#b34700"),
    "loss":   ("#fdf0f1", "#f9e0e2", "#c62828", "#b71c1c"),
    "act":    ("#f8f1fb", "#f0e4f7", "#6a1b9a", "#4a148c"),
    "loop":   ("#f2f5f7", "#e6ebee", "#546e7a", "#37474f"),
    "env":    ("#eef8f6", "#dcf0ec", "#00695c", "#004d40"),
    "mem":    ("#fffef0", "#fdf6d8", "#b8860b", "#8a6d00"),
    "reward": ("#fdf2f7", "#f9e2ed", "#ad1457", "#880e4f"),
    "key":    ("#fffaf0", "#fff1d6", "#ef6c00", "#9a4d00"),
}
CLS_CN = {"data": "数据/输入", "frozen": "冻结模块", "train": "可训练模块", "loss": "损失/监督",
          "act": "动作生成", "loop": "循环/反馈", "env": "环境", "mem": "记忆",
          "reward": "奖励", "key": "关键组件"}
EDGE_STYLE = {  # style -> (stroke, width, dash)
    "main":  ("#607d8b", 2.4, ""),
    "thick": ("#37474f", 3.2, ""),
    "thin":  ("#90a4ae", 1.8, ""),
    "fb":    ("#1565c0", 2.2, "7 5"),
    "loss":  ("#c62828", 2.2, "6 4"),
}
FONT = "'PingFang SC','Hiragino Sans GB','Microsoft YaHei','Segoe UI',sans-serif"


def esc(s):
    return html.escape(str(s), quote=True)


def tw(s, fs):
    """估算文本宽度: CJK 全宽, 其余约 0.55em。"""
    return sum(fs * (1.06 if ord(ch) > 0x2E80 else 0.58) for ch in str(s))


ALIASES = {  # 语义名 -> 候选(按优先级), 逐个尝试 duotone(d-)/lucide/phosphor/tabler
    "chat": ["d-chat-circle-dots", "message-square"],
    "robot": ["d-robot", "bot"],
    "state": ["d-scan", "scan"],
    "target": ["d-target", "target", "crosshair"],
    "flow": ["d-flow-arrow", "arrow-right"],
    "noise": ["d-waveform", "shuffle", "sparkle"],
    "chart": ["d-chart-bar", "chart-column"],
    "sigma": ["d-sigma", "sigma"],
    "gear": ["d-gear-six", "settings"],
    "flask": ["d-flask", "flask-conical"],
    "brain": ["d-brain", "brain"],
    "lightning": ["d-lightning", "zap"],
    "reward": ["d-medal", "trophy", "medal"],
    "eye": ["d-eye", "eye"],
    "clock": ["d-clock-countdown", "clock"],
    "graph": ["d-graph", "git-fork"],
    "tree": ["d-tree-structure", "network"],
    "db": ["d-database", "database"],
    "cpu": ["d-cpu", "cpu"],
    "spark": ["d-sparkle", "sparkles"],
    "check": ["d-check-circle", "check"],
    "search": ["d-magnifying-glass", "search"],
    "idea": ["d-lightbulb-filament", "lightbulb"],
    "stack": ["d-stack", "layers"],
    "puzzle": ["d-puzzle-piece", "puzzle"],
    "cross": ["d-x-circle", "x"],
    "refresh": ["d-arrows-clockwise", "refresh-cw"],
    "box": ["box", "boxes"],
    "circuit": ["d-circuitry", "circuit-board"],
    "trophy": ["d-trophy", "trophy"],
    "video": ["d-video-camera", "video"],
    "users": ["d-users-three", "users"],
    "atom": ["d-atom", "atom"],
    "cube": ["d-cube", "box"],
    "gauge": ["d-clock-countdown", "gauge"],
}


def load_icon(name, ink):
    """读图标 SVG, 返回可内嵌的 <g>(已换色)。找不到时报近似名。"""
    cands = []
    for base in ALIASES.get(name, [name]):
        for fam, prefix in (("duotone", "d-"), ("lucide", ""), ("phosphor", ""), ("tabler", "")):
            cands.append((base, os.path.join(ICON_ROOT, fam, f"{prefix}{base}.svg")))
    path = next((c for _, c in cands if os.path.exists(c)), None)
    if not path:
        have = set()
        for _, c in cands:
            d = os.path.dirname(c)
            if os.path.isdir(d):
                have |= {os.path.splitext(f)[0].lower() for f in os.listdir(d)}
        near = sorted(n for n in have if name.split("-")[0] in n)[:6]
        raise SystemExit(f"icon '{name}' 不存在; 同族近似: {near or '无'}")
    raw = open(path, encoding="utf-8").read()
    m = re.search(r"<svg[^>]*viewBox=\"([\d. ]+)\"[^>]*>(.*?)</svg>", raw, re.S)
    vb = [float(v) for v in m.group(1).split()]
    body = m.group(2).replace("currentColor", ink)
    return vb, body


def icon_g(name, ink, cx, cy, size):
    """把图标缩放到 size x size, 中心 (cx,cy)。"""
    vb, body = load_icon(name, ink)
    s = size / max(vb[2], vb[3])
    x = cx - vb[2] * s / 2 - vb[0] * s
    y = cy - vb[3] * s / 2 - vb[1] * s
    return f'<g transform="translate({x:.1f},{y:.1f}) scale({s:.4f})">{body}</g>'


def anchor_pt(n, side, pos):
    x, y, w, h = n["x"], n["y"], n["w"], n["h"]
    return {"top":    (x + w * pos, y), "bottom": (x + w * pos, y + h),
            "left":   (x, y + h * pos), "right":  (x + w, y + h * pos)}[side]


def auto_sides(a, b):
    dx, dy = (b["x"] + b["w"] / 2) - (a["x"] + a["w"] / 2), (b["y"] + b["h"] / 2) - (a["y"] + a["h"] / 2)
    if abs(dx) > abs(dy) + 60:
        return ("right", "left") if dx > 0 else ("left", "right")
    return ("bottom", "top") if dy > 0 else ("top", "bottom")


NORM = {"top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}


def edge_path(sa, na, sb, nb, k=None):
    k = k if k is not None else max(46, (abs(sb[0] - sa[0]) + abs(sb[1] - sa[1])) * 0.42)
    (u1, v1), (u2, v2) = NORM[na], NORM[nb]
    c1 = (sa[0] + u1 * k, sa[1] + v1 * k)
    c2 = (sb[0] + u2 * k, sb[1] + v2 * k)
    return (f"M {sa[0]:.1f} {sa[1]:.1f} C {c1[0]:.1f} {c1[1]:.1f}, "
            f"{c2[0]:.1f} {c2[1]:.1f}, {sb[0]:.1f} {sb[1]:.1f}",
            ((sa[0] + c1[0] + c2[0] + sb[0]) / 4, (sa[1] + c1[1] + c2[1] + sb[1]) / 4))


def render(spec):
    W, H = spec["canvas"]
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

    # 标题区
    out.append(f'<rect x="34" y="30" width="5" height="42" rx="2.5" fill="url(#g-title)"/>')
    out.append(f'<text x="52" y="52" font-size="25" font-weight="800" fill="#1a2733">{esc(spec["title"])}</text>')
    if spec.get("subtitle"):
        out.append(f'<text x="52" y="74" font-size="13.5" fill="#78909c">{esc(spec["subtitle"])}</text>')
    if spec.get("foot"):
        out.append(f'<text x="{W - 34}" y="{H - 18}" font-size="11.5" fill="#a0aab4" '
                   f'text-anchor="end">{esc(spec["foot"])}</text>')

    # 分组面板
    for p in spec.get("panels", []):
        out.append(f'<rect x="{p["x"]}" y="{p["y"]}" width="{p["w"]}" height="{p["h"]}" rx="18" '
                   f'fill="#fbfdff" stroke="#b7c6d6" stroke-width="1.4" stroke-dasharray="8 5"/>')
        plw = tw(p["label"], 12.5) + 24
        out.append(f'<rect x="{p["x"] + 16}" y="{p["y"] + 13}" width="{plw}" '
                   f'height="26" rx="13" fill="#eef3f8" stroke="#b7c6d6"/>')
        out.append(f'<text x="{p["x"] + 27}" y="{p["y"] + 30}" font-size="12.5" '
                   f'fill="#4a6579">{esc(p["label"])}</text>')

    nodes = {n["id"]: n for n in spec["nodes"]}

    # 边(先画, 压在卡片下; 标签最后)
    labels, paths = [], []
    for e in spec.get("edges", []):
        a, b = nodes[e["from"]], nodes[e["to"]]
        sa_, sb_ = auto_sides(a, b)
        na, nb = e.get("out", sa_), e.get("in", sb_)
        sa = anchor_pt(a, na, e.get("pos_out", 0.5))
        sb = anchor_pt(b, nb, e.get("pos_in", 0.5))
        d, mid = edge_path(sa, na, sb, nb, e.get("k"))
        st = e.get("style", "main")
        c, w, dash = EDGE_STYLE[st]
        dash_a = f' stroke-dasharray="{dash}"' if dash else ""
        paths.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}"{dash_a} '
                     f'marker-end="url(#arr-{st})" opacity="0.92"/>')
        if e.get("label"):
            t = esc(e["label"])
            wl = tw(e["label"], 12) + 18
            lx, ly = mid[0], mid[1]
            off = e.get("loff", (0, 0))
            labels.append(f'<g transform="translate({lx + off[0]:.1f},{ly + off[1]:.1f})">'
                          f'<rect x="{-wl / 2:.1f}" y="-11" width="{wl:.1f}" height="22" rx="11" '
                          f'fill="#ffffff" stroke="#cfd8dc"/>'
                          f'<text y="4" font-size="12" fill="#455a64" text-anchor="middle">{t}</text></g>')
    out += paths

    # 节点卡片
    for n in spec["nodes"]:
        cls = n.get("cls", "loop")
        a, b, s, ink = PALETTE[cls]
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        bw = 2.8 if cls in ("key", "train") else 1.6
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
            ty = y + 44
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
            sub = esc(n["sub"])
            if len(lines) > 1:
                out.append(f'<text x="{x + w / 2}" y="{sy}" font-size="12" fill="#5f6b7a" '
                           f'text-anchor="middle">{sub}</text>')
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
    out += labels

    # 图例
    legend = spec.get("legend")
    if legend:
        lx, ly = 40, H - 34
        out.append(f'<text x="{lx}" y="{ly + 4.5}" font-size="12" fill="#78909c">图例</text>')
        lx += 46
        for cls in legend:
            a, b, s, ink = PALETTE[cls]
            label = CLS_CN[cls]
            wch = tw(label, 11.5) + 38
            out.append(f'<g><rect x="{lx}" y="{ly - 10}" width="{wch}" height="22" rx="11" '
                       f'fill="url(#g-{cls})" stroke="{s}" stroke-width="1.3"/>'
                       f'<circle cx="{lx + 13}" cy="{ly + 1}" r="4.2" fill="{s}"/>'
                       f'<text x="{lx + 24}" y="{ly + 5}" font-size="11.5" fill="{ink}">{esc(label)}</text></g>')
            lx += wch + 12
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", help="输出路径 (默认 spec 同目录 arch.svg)")
    ap.add_argument("--png", action="store_true", help="soffice 转 png (校验/ montage 用)")
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding="utf-8"))
    svg = render(spec)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.spec)), "arch.svg")
    open(out, "w", encoding="utf-8").write(svg)
    print("SVG", out)
    if a.png:
        d = os.path.dirname(os.path.abspath(out))
        subprocess.run(["qlmanage", "-t", "-s", "1600", "-o", d, out],
                       check=True, capture_output=True, timeout=120)
        png = os.path.join(d, os.path.basename(out) + ".png")  # qlmanage 生成 <name>.svg.png
        os.replace(png, os.path.splitext(out)[0] + ".png")
        print("PNG", os.path.splitext(out)[0] + ".png")


if __name__ == "__main__":
    main()
