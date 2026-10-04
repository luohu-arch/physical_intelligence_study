#!/usr/bin/env python3
"""给 collide 报告的兜底药丸(tier>=3)网格搜索空位并写 labpos。
用法: python3 fix_pills.py <fid> ...   # 不带参数则处理全部有告警的
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fancy_diagram as F  # noqa: E402
from fancy_diagram import tw  # noqa: E402


def process(spec_path):
    spec = json.load(open(spec_path))
    fid = spec_path.split("/")[-2]
    nodes = {n["id"]: n for n in spec["nodes"]}
    W0, H0 = spec["canvas"]
    # 与 render 一致的画布扩展
    W, H = W0, H0
    for e in spec.get("edges", []):
        if e["from"] in nodes and e["to"] in nodes:
            a, b = nodes[e["from"]], nodes[e["to"]]
            sa_, sb_ = F.auto_sides(a, b)
            na, nb = e.get("out", sa_), e.get("in", sb_)
            k = e.get("k", 60)
            if na == nb == "bottom":
                H = max(H, max(a["y"] + a["h"], b["y"] + b["h"]) + k + 100)
            elif na == nb == "right":
                W = max(W, max(a["x"] + a["w"], b["x"] + b["w"]) + k + 100)

    node_rects = [(n["x"] - 4, n["y"] - 4, n["x"] + n["w"] + 4, n["y"] + n["h"] + 4)
                  for n in spec["nodes"]]
    pill_jobs, path_segs, edges_ref = [], [], []
    for e in spec.get("edges", []):
        if e["from"] not in nodes or e["to"] not in nodes:
            continue
        a, b = nodes[e["from"]], nodes[e["to"]]
        sa_, sb_ = F.auto_sides(a, b)
        na, nb = e.get("out", sa_), e.get("in", sb_)
        sa = F.anchor_pt(a, na, e.get("pos_out", 0.5))
        sb = F.anchor_pt(b, nb, e.get("pos_in", 0.5))
        rects = [(n2["x"], n2["y"], n2["x"] + n2["w"], n2["y"] + n2["h"])
                 for nid2, n2 in nodes.items() if nid2 not in (e["from"], e["to"])]
        d, mid, epts = F.edge_path(sa, na, sb, nb, e.get("k"), rects, (W, H))
        seg_idx = list(range(len(path_segs), len(path_segs) + max(0, len(epts) - 1)))
        path_segs.extend((epts[i], epts[i + 1]) for i in range(len(epts) - 1))
        if e.get("label"):
            off = e.get("loff", (0, 0))
            lp = e.get("labpos")
            pill_jobs.append({"text": e["label"], "cx": (lp[0] if lp else mid[0] + off[0]),
                              "cy": (lp[1] if lp else mid[1] + off[1]), "color": "#000",
                              "ax": sa[0], "ay": sa[1], "bx": sb[0], "by": sb[1],
                              "pin": bool(e.get("labpos")), "own": seg_idx})
            edges_ref.append(e)

    def seg_near_rect(p, q, r, pad=3.0):
        x0, y0, x1, y1 = r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad
        steps = max(2, int(max(abs(q[0] - p[0]), abs(q[1] - p[1])) / 6) + 1)
        for i in range(steps + 1):
            t = i / steps
            if x0 <= p[0] + (q[0] - p[0]) * t <= x1 and y0 <= p[1] + (q[1] - p[1]) * t <= y1:
                return True
        return False

    # 先按渲染顺序模拟放置, 找出落到 tier>=3 的药丸
    placed = []
    CAND = F.CAND if hasattr(F, "CAND") else None
    if CAND is None:  # resolver 内部 CAND, 复制一份
        CAND = [(0, 0), (0, -28), (0, 28), (0, -56), (0, 56), (26, -28), (-26, -28),
                (26, 28), (-26, 28), (0, -84), (0, 84), (0, -112), (0, 112),
                (64, 0), (-64, 0), (64, -28), (-64, -28), (64, 28), (-64, 28),
                (0, -140), (0, 140), (90, -56), (-90, -56), (90, 56), (-90, 56),
                (90, 0), (-90, 0), (120, -28), (-120, -28), (120, 28), (-120, 28)]

    def ok_rect(r, j, allow_node, avoid_path):
        own = set(j.get("own", ()))
        if r[0] < 6 or r[2] > W - 6 or r[1] < 96 or r[3] > H - 24:
            return False
        if not allow_node and any(r[0] < nr[2] and nr[0] < r[2] and r[1] < nr[3] and nr[1] < r[3]
                                  for nr in node_rects):
            return False
        if any(r[0] < pr[2] + 10 and pr[0] - 10 < r[2] and r[1] < pr[3] + 8 and pr[1] - 8 < r[3]
               for pr in placed):
            return False
        for ax, ay in ((j["ax"], j["ay"]), (j["bx"], j["by"])):
            if abs((r[0] + r[2]) / 2 - ax) < (r[2] - r[0]) / 2 + 18 and abs((r[1] + r[3]) / 2 - ay) < 30:
                return False
        if avoid_path:
            for idx, (p, q) in enumerate(path_segs):
                if idx in own:
                    continue
                if seg_near_rect(p, q, r):
                    return False
        return True

    fixed, unfixed = [], []
    for j, e in zip(pill_jobs, edges_ref):
        wl = tw(j["text"], 12) + 20
        tier = 0
        for allow_node, avoid_path in ((False, True), (False, False), (True, False)):
            found = None
            for dx, dy in CAND:
                cx, cy = j["cx"] + dx, j["cy"] + dy
                r = (cx - wl / 2, cy - 12, cx + wl / 2, cy + 12)
                if ok_rect(r, j, allow_node, avoid_path):
                    found = (cx, cy, r)
                    break
            if found:
                tier = 1 if (False, True) == (allow_node, avoid_path) else (2 if not allow_node else 3)
                break
        if not found:
            tier = 4
            found = (j["cx"], j["cy"], (j["cx"] - wl / 2, j["cy"] - 12, j["cx"] + wl / 2, j["cy"] + 12))
        if tier >= 3 and not j.get("pin"):
            # 网格粗搜: 以锚点为中心 +-260, 步长 13
            best = None
            for gx in range(-260, 261, 13):
                for gy in range(-260, 261, 13):
                    cx, cy = j["cx"] + gx, j["cy"] + gy
                    r = (cx - wl / 2, cy - 12, cx + wl / 2, cy + 12)
                    if ok_rect(r, j, allow_node=False, avoid_path=True):
                        best = (cx, cy)
                        break
                if best:
                    break
            if best:
                e["labpos"] = [round(best[0]), round(best[1])]
                fixed.append((e["from"], e["to"], j["text"][:20], best))
                placed.append((best[0] - wl / 2, best[1] - 12, best[0] + wl / 2, best[1] + 12))
                continue
            unfixed.append((e["from"], e["to"], j["text"]))
        placed.append(found[2])
    if fixed:
        json.dump(spec, open(spec_path, "w"), ensure_ascii=False, indent=1)
    return fixed, unfixed


def main():
    args = sys.argv[1:]
    paths = [p for p in sorted(glob.glob("notes/**/figures/*/arch.spec.json", recursive=True))
             if not args or p.split("/")[-2] in args]
    tot_f = tot_u = 0
    for p in paths:
        f, u = process(p)
        if f or u:
            print(p.split("/")[-2], "fixed:", [(a, b, t) for a, b, t, _ in f])
            if u:
                print("   UNFIXED:", u)
        tot_f += len(f)
        tot_u += len(u)
    print(f"\ntotal fixed {tot_f}, unfixed {tot_u}")


if __name__ == "__main__":
    main()
