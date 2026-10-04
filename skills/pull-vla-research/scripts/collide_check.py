#!/usr/bin/env python3
"""collide_check: 全量可见元素碰撞检查（比几何检查覆盖更多盲区）。

检查项:
 1. 节点-节点 / 节点-画布越界
 2. 面板-面板重叠
 3. 面板框住非成员节点（或成员在外面）
 4. 面板标签 chip 压节点/药丸/tag
 5. tag 角标药丸 与 边标签药丸 重叠
 6. 边标签药丸 互叠 / 压节点（重查）
 7. 连线折线段穿过无关节点卡片（U形/肘形绕行失控）
 8. 卡内文字估宽溢出（宽松系数）
输出: 每张图的问题计数与明细, 按严重度排序。
"""
import glob
import json
import re
import sys

sys.path.insert(0, "skills/pull-vla-research/scripts")
from fancy_diagram import edge_path, anchor_pt, auto_sides, tw  # noqa: E402


def seg_rect_hit(p1, p2, r, shrink=8):
    x0, y0, x1, y1 = r[0] + shrink, r[1] + shrink, r[2] - shrink, r[3] - shrink
    if x0 >= x1 or y0 >= y1:
        return False
    # 只处理正交/斜线段 vs 轴对齐矩形: 采样法足够
    steps = max(2, int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) / 6))
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t
        if x0 < x < x1 and y0 < y < y1:
            return True
    return False


def rect_overlap(a, b, pad=2):
    return a[0] < b[2] - pad and b[0] < a[2] - pad and a[1] < b[3] - pad and b[1] < a[3] - pad


def check(spec, svg_path):
    import xml.etree.ElementTree as ET
    issues = []
    ns = "{http://www.w3.org/2000/svg}"
    nsre = "http://www.w3.org/2000/svg"
    root = ET.parse(svg_path).getroot()
    W, H = spec["canvas"]
    nodes = spec["nodes"]
    nrect = {n["id"]: (n["x"], n["y"], n["x"] + n["w"], n["y"] + n["h"]) for n in nodes}

    pills, panels_r = [], []
    for g in root.iter(f"{ns}g"):
        m = re.match(r"translate\(([-\d.]+),([-\d.]+)\)", g.get("transform", ""))
        if not m:
            continue
        cx, cy = float(m.group(1)), float(m.group(2))
        rect = g.find(f"{ns}rect")
        if rect is None:
            continue
        w_, h_ = float(rect.get("width")), float(rect.get("height"))
        dash = rect.get("stroke-dasharray")
        if dash and w_ > 120 and h_ > 80:
            panels_r.append((cx - w_/2, cy - h_/2, cx + w_/2, cy + h_/2))
        else:
            pills.append((cx - w_/2, cy - h_/2, cx + w_/2, cy + h_/2, rect.get("stroke")))
    # 图例区的药丸/线型示例跳过(画布底部 70px)
    pills = [p for p in pills if p[3] < H - 74]
    panels_r = [p for p in panels_r if p[3] < H - 74]

    def ov(a, b, pad=2):
        return a[0] < b[2]-pad and b[0] < a[2]-pad and a[1] < b[3]-pad and b[1] < a[3]-pad

    for i, a in enumerate(nodes):
        r = nrect[a["id"]]
        if r[2] > W+8 or r[3] > H+8 or r[0] < -8 or r[1] < 40:
            issues.append(("out", a["id"]))
        for b in nodes[i+1:]:
            if ov(r, nrect[b["id"]]):
                issues.append(("n-ovl", f'{a["id"]}~{b["id"]}'))
    for i in range(len(panels_r)):
        for j in range(i+1, len(panels_r)):
            if ov(panels_r[i], panels_r[j], pad=4):
                issues.append(("p-ovl", f"panel{i}~panel{j}"))
    for i, a in enumerate(pills):
        for nid, r in nrect.items():
            if ov(a[:4], r):
                issues.append(("pill-node", f"{i}x{nid}"))
                break
    for i in range(len(pills)):
        for j in range(i+1, len(pills)):
            if ov(pills[i][:4], pills[j][:4]):
                issues.append(("pill-pill", f"{i}~{j}"))
    for i, a in enumerate(panels_r):
        for nid, r in nrect.items():
            if ov(a, r, pad=0):
                # 节点在面板内不一定错(成员); 无法区分成员 -> 只报"骑线"(部分相交)
                inside = r[0] > a[0]+16 and r[2] < a[2]-16 and r[1] > a[1]+40 and r[3] < a[3]-16
                if not inside:
                    issues.append(("panel-straddle", f"{nid}"))
    # 线穿卡: 用渲染一致 rects(shrink 6) 重算路径
    rects_r = list(nrect.values())
    by_id = {n["id"]: n for n in nodes}
    for e in spec.get("edges", []):
        if e["from"] not in by_id or e["to"] not in by_id:
            continue
        a, b = by_id[e["from"]], by_id[e["to"]]
        sa_, sb_ = auto_sides(a, b)
        na, nb = e.get("out", sa_), e.get("in", sb_)
        sa = anchor_pt(a, na, e.get("pos_out", 0.5))
        sb = anchor_pt(b, nb, e.get("pos_in", 0.5))
        rects_e = [nrect[i2] for i2 in nrect if i2 not in (e["from"], e["to"])]
        d, _ = edge_path(sa, na, sb, nb, e.get("k"), rects_e, (W, H))
        pts = [tuple(map(float, mm)) for mm in re.findall(r"([-.\d]+) ([-.\d]+)", d)]
        ends = {e["from"], e["to"]}
        for k in range(len(pts)-1):
            for nid, r in nrect.items():
                if nid in ends:
                    continue
                if seg_rect_hit(pts[k], pts[k+1], r, shrink=4):
                    issues.append(("line-x-card", f'{e["from"]}->{e["to"]} x {nid}'))
                    break
    for n in nodes:
        labs = n["label"].split("\n")
        subs = n.get("sub", "").split("\n") if n.get("sub") else []
        for ln, fs in ([(l, 15.5) for l in labs] + [(l, 12) for l in subs]):
            if ln and tw(ln, fs) * 1.15 + 34 > n["w"]:
                issues.append(("text-ovf", f'{n["id"]}'))
                break
    return issues


def main():
    rows = []
    for sj in sorted(glob.glob("notes/**/figures/*/arch.spec.json", recursive=True)):
        spec = json.load(open(sj))
        iss = check(spec, sj.replace(".spec.json", ".svg"))
        if iss:
            rows.append((len(iss), sj, iss))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"{len(rows)} diagrams with issues, {total} total")
    for c, sj, iss in rows:
        nid = sj.split("/figures/")[0].split("/")[-1] + "/" + sj.split("/figures/")[1].split("/")[0]
        kinds = {}
        for k, _ in iss:
            kinds[k] = kinds.get(k, 0) + 1
        print(f"  {c:3d} {nid} {kinds}")
    json.dump([{"note": r[1], "issues": [[k, d] for k, d in r[2]]} for r in rows],
              open("/tmp/collide_report.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
