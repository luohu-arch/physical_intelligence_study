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
EDGE_STYLE = {  # style -> (stroke, width, dash); 实线=前向数据流, 虚线=非数据流信号
    "main":  ("#546e7a", 3.2, ""),
    "thick": ("#263238", 4.4, ""),
    "thin":  ("#9fb0bb", 2.2, ""),
    "fb":    ("#1565c0", 2.8, "12 7"),           # 长虚线: 闭环反馈
    "loss":  ("#c62828", 2.8, "3 4.5 11 4.5"),   # 点划线: 损失/监督/蒸馏
}
EDGE_CN = {"thick": "主数据流", "main": "信息流", "thin": "弱关联", "fb": "闭环反馈", "loss": "损失/监督"}
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


def _avoid_v(x, y0, y1, rects):
    """垂直转折段 x 若穿卡, 平移到卡侧空位。"""
    lo, hi = min(y0, y1), max(y0, y1)
    for _ in range(2):
        hit = None
        for r in rects:
            if r[0] + 10 < x < r[2] - 10 and hi > r[1] + 8 and lo < r[3] - 8:
                hit = r
                break
        if not hit:
            break
        cands = sorted([hit[2] + 18, hit[0] - 18], key=lambda c: abs(c - x))
        x = cands[0]
    return x


def _avoid_h(y, x0, x1, rects):
    lo, hi = min(x0, x1), max(x0, x1)
    for _ in range(2):
        hit = None
        for r in rects:
            if r[1] + 10 < y < r[3] - 10 and hi > r[0] + 8 and lo < r[2] - 8:
                hit = r
                break
        if not hit:
            break
        cands = sorted([hit[3] + 18, hit[1] - 18], key=lambda c: abs(c - y))
        y = cands[0]
    return y


def edge_path(sa, na, sb, nb, k=None, rects=()):
    """直角折线路由: 对向=直线/肘形, 同侧=U 形环绕, 相邻侧=L 形。"""
    k = k if k is not None else 60
    opp = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    if nb == opp[na]:                                   # 对向
        if na in ("left", "right"):
            if abs(sa[1] - sb[1]) < 16:
                pts = [sa, sb]
            else:
                mx = _avoid_v((sa[0] + sb[0]) / 2, sa[1], sb[1], rects)
                lo, hi = min(sa[0], sb[0]) + 24, max(sa[0], sb[0]) - 24
                if hi <= lo or not (lo <= mx <= hi):
                    mx = (sa[0] + sb[0]) / 2   # 绕行走廊放不下: 回中
                pts = [sa, (mx, sa[1]), (mx, sb[1]), sb]
        else:
            if abs(sa[0] - sb[0]) < 16:
                pts = [sa, sb]
            else:
                my = _avoid_h((sa[1] + sb[1]) / 2, sa[0], sb[0], rects)
                lo, hi = min(sa[1], sb[1]) + 24, max(sa[1], sb[1]) - 24
                if hi <= lo or not (lo <= my <= hi):
                    my = (sa[1] + sb[1]) / 2
                pts = [sa, (sa[0], my), (sb[0], my), sb]
    elif na == nb:                                      # 同侧 U 形环绕
        if na == "bottom":
            ly = max(sa[1], sb[1]) + k
            pts = [sa, (sa[0], ly), (sb[0], ly), sb]
        elif na == "top":
            ly = min(sa[1], sb[1]) - k
            pts = [sa, (sa[0], ly), (sb[0], ly), sb]
        elif na == "right":
            lx = max(sa[0], sb[0]) + k
            pts = [sa, (lx, sa[1]), (lx, sb[1]), sb]
        else:
            lx = min(sa[0], sb[0]) - k
            pts = [sa, (lx, sa[1]), (lx, sb[1]), sb]
    else:                                               # 相邻侧 L 形
        if na in ("left", "right"):
            corner = (sb[0], sa[1])
        else:
            corner = (sa[0], sb[1])
        pts = [sa, corner, sb]
    d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}" + "".join(f" L {q[0]:.1f} {q[1]:.1f}" for q in pts[1:])
    lens = [((pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2) ** .5
            for i in range(len(pts) - 1)]
    half, acc, mid = sum(lens) / 2, 0.0, pts[-1]
    for i, L in enumerate(lens):
        if acc + L >= half and L > 0:
            t = (half - acc) / L
            mid = (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t,
                   pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t)
            break
        acc += L
    return d, mid


def resolve_and_render_pills(jobs, nodes, W, H):
    """标签药丸避障: 依次取第一个不压卡片/已放药丸/两端锚点禁区的候选偏移。"""
    placed = []
    node_rects = [(n["x"] - 2, n["y"] - 2, n["x"] + n["w"] + 2, n["y"] + n["h"] + 2) for n in nodes]
    CAND = [(0, 0), (0, -28), (0, 28), (0, -56), (0, 56), (26, -28), (-26, -28),
            (26, 28), (-26, 28), (0, -84), (0, 84), (0, -112), (0, 112),
            (64, 0), (-64, 0), (64, -28), (-64, -28), (64, 28), (-64, 28)]
    out = []

    def _find(j, wl, allow_node):
        for dx, dy in CAND:
            cx, cy = j["cx"] + dx, j["cy"] + dy
            r = (cx - wl / 2, cy - 12, cx + wl / 2, cy + 12)
            if r[0] < 6 or r[2] > W - 6 or r[1] < 96 or r[3] > H - 24:
                continue
            if not allow_node and any(r[0] < nr[2] and nr[0] < r[2] and r[1] < nr[3] and nr[1] < r[3]
                                      for nr in node_rects):
                continue
            if any(r[0] < pr[2] + 10 and pr[0] - 10 < r[2] and r[1] < pr[3] + 8 and pr[1] - 8 < r[3]
                   for pr in placed):
                continue
            if any(abs(cx - ax) < wl / 2 + 18 and abs(cy - ay) < 30
                   for ax, ay in ((j["ax"], j["ay"]), (j["bx"], j["by"]))):
                continue
            return cx, cy, r
        return None

    for j in jobs:
        wl = tw(j["text"], 12) + 20
        best = _find(j, wl, allow_node=False)
        if best is None:   # 全躲失败: 宁可压卡也不压别的药丸/箭头
            best = _find(j, wl, allow_node=True)
        if best is None:   # 仍失败: 就地放
            best = (j["cx"], j["cy"], (j["cx"] - wl / 2, j["cy"] - 12, j["cx"] + wl / 2, j["cy"] + 12))
        cx, cy, r = best
        placed.append(r)
        t = esc(j["text"])
        out.append(f'<g transform="translate({cx:.1f},{cy:.1f})">'
                   f'<rect x="{-wl / 2:.1f}" y="-11" width="{wl:.1f}" height="22" rx="11" '
                   f'fill="#ffffff" stroke="{j["color"]}" stroke-width="1.2"/>'
                   f'<text y="4" font-size="12" fill="{j["color"]}" font-weight="600" '
                   f'text-anchor="middle">{t}</text></g>')
    return out


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
        out.append(f'<marker id="arr-{st}" viewBox="0 0 10 10" refX="8.5" refY="5" '
                   f'markerWidth="13" markerHeight="13" markerUnits="userSpaceOnUse" '
                   f'orient="auto-start-reverse">'
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
    pill_jobs, paths = [], []
    for e in spec.get("edges", []):
        a, b = nodes[e["from"]], nodes[e["to"]]
        sa_, sb_ = auto_sides(a, b)
        na, nb = e.get("out", sa_), e.get("in", sb_)
        sa = anchor_pt(a, na, e.get("pos_out", 0.5))
        sb = anchor_pt(b, nb, e.get("pos_in", 0.5))
        rects = [(n2["x"] + 6, n2["y"] + 6, n2["x"] + n2["w"] - 6, n2["y"] + n2["h"] - 6)
                 for n2 in spec["nodes"]]
        d, mid = edge_path(sa, na, sb, nb, e.get("k"), rects)
        st = e.get("style", "main")
        c, w, dash = EDGE_STYLE[st]
        dash_a = f' stroke-dasharray="{dash}"' if dash else ""
        paths.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{w}"{dash_a} '
                     f'marker-end="url(#arr-{st})" opacity="0.92"/>')
        if e.get("label"):
            off = e.get("loff", (0, 0))
            pill_jobs.append({"text": e["label"], "cx": mid[0] + off[0], "cy": mid[1] + off[1],
                              "color": EDGE_STYLE[st][0], "ax": sa[0], "ay": sa[1],
                              "bx": sb[0], "by": sb[1]})
    out += paths

    # 节点卡片
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
    out += resolve_and_render_pills(pill_jobs, spec["nodes"], W, H)

    # 图例
    legend = spec.get("legend")
    used_styles = []
    for e in spec.get("edges", []):
        st = e.get("style", "main")
        if st not in used_styles:
            used_styles.append(st)
    if legend or used_styles:
        ly = H - 56 if (legend and used_styles) else H - 34
        if legend:
            lx = 40
            out.append(f'<text x="{lx}" y="{ly + 4.5}" font-size="12" fill="#78909c">图例</text>')
            lx += 46
            for cls in legend:
                a, b, s_, ink = PALETTE[cls]
                label = CLS_CN[cls]
                wch = tw(label, 11.5) + 38
                out.append(f'<g><rect x="{lx}" y="{ly - 10}" width="{wch}" height="22" rx="11" '
                           f'fill="url(#g-{cls})" stroke="{s_}" stroke-width="1.3"/>'
                           f'<circle cx="{lx + 13}" cy="{ly + 1}" r="4.2" fill="{s_}"/>'
                           f'<text x="{lx + 24}" y="{ly + 5}" font-size="11.5" fill="{ink}">{esc(label)}</text></g>')
                lx += wch + 12
        if used_styles:
            lx, ly2 = 40, H - 16
            out.append(f'<text x="{lx}" y="{ly2 + 4}" font-size="11.5" fill="#78909c">线型</text>')
            lx += 40
            for st in used_styles:
                c, w, dash = EDGE_STYLE[st]
                dash_a = f' stroke-dasharray="{dash}"' if dash else ""
                label = EDGE_CN[st]
                seg = 44
                out.append(f'<g><path d="M {lx} {ly2} h {seg}" stroke="{c}" stroke-width="{w}"'
                           f'{dash_a} marker-end="url(#arr-{st})"/>'
                           f'<text x="{lx + seg + 8}" y="{ly2 + 4}" font-size="11.5" '
                           f'fill="{c}">{esc(label)}</text></g>')
                lx += seg + 14 + tw(label, 11.5) + 26
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
