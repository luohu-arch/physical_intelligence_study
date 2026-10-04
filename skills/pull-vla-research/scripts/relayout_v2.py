#!/usr/bin/env python3
"""relayout_v2: 语义几何重排（确定性，无视觉/无 LLM）。

对未修改过的 spec（git 干净）从 mermaid_original 重建布局，规则：
1. rank 沿主数据流（最长路径，忽略回边）
2. 语义列定位: data 输入靠左/顶部起点, env 最右, loss/mem 收进侧翼面板行
3. 严格对齐: 同列 x 一致、列内 y 等距(38)、列间隙统一(>=108)、内容整体居中
4. 回边(反馈/监督)统一走底部走廊: out bottom/in bottom, k 压过最低行
5. 面板从 mermaid subgraph 成员重建, bbox 精确包住成员
6. 画布比例 1.15~2.2, 宽<=1650; 超限自动转置(TB<->LR)重排一次

用法: relayout_v2.py [--dry]   处理所有 git 未修改的 arch.spec.json
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mermaid_to_spec import parse_mermaid, wrap_cjk, tw, pick_icon  # noqa: E402
from fancy_diagram import render  # noqa: E402

MARGIN_L, CONTENT_TOP = 46, 132
GAPX, GAPY, SPACING = 108, 96, 38

# 语义优先级: 数值越小越靠主链起点
CLS_ORDER = {"data": 0, "frozen": 1, "train": 2, "act": 3, "key": 2, "mem": 4,
             "reward": 4, "loss": 5, "env": 6, "loop": 3}


def ranks_of(nodes, edges):
    adj = {i: [] for i in nodes}
    for e in edges:
        if e["a"] in nodes and e["b"] in nodes and e["a"] != e["b"]:
            adj[e["a"]].append(e["b"])
    back = set()
    color = {i: 0 for i in nodes}

    def dfs(u):
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1:
                back.add((u, v))
            elif color[v] == 0:
                dfs(v)
        color[u] = 2
    for i in nodes:
        if color[i] == 0:
            dfs(i)
    rank = {i: 0 for i in nodes}
    fwd = {i: [] for i in nodes}
    for e in edges:
        if (e["a"], e["b"]) not in back and e["a"] != e["b"]:
            fwd[e["a"]].append(e["b"])
    for _ in range(len(nodes) + 1):
        ch = False
        for u in nodes:
            for v in fwd[u]:
                if rank[v] < rank[u] + 1:
                    rank[v] = rank[u] + 1
                    ch = True
        if not ch:
            break
    return rank, back


def deghost(parsed):
    """边指向 subgraph id 时会产生幽灵节点: 边重定向到面板首个成员, 删除幽灵。"""
    nodes, edges, panels_src = parsed["nodes"], parsed["edges"], parsed["panels"]
    for gid in list(nodes):
        if gid in panels_src and panels_src[gid]["members"]:
            rep = panels_src[gid]["members"][0]
            for e in edges:
                if e["a"] == gid:
                    e["a"] = rep
                if e["b"] == gid:
                    e["b"] = rep
            del nodes[gid]
    return parsed


def layout(parsed, transpose=False):
    parsed = deghost(parsed)
    nodes, edges, panels_src = parsed["nodes"], parsed["edges"], parsed["panels"]
    lr = (parsed["dir"] in ("LR", "RL")) ^ transpose
    rank, back = ranks_of(nodes, edges)

    geom = {}
    for nid, n in nodes.items():
        lines = [l for seg in re.split(r"<br\s*/?>", n["label"]) for l in wrap_cjk(seg, 12)][:3]
        label_l, sub_l = lines[:2], lines[2:]
        w = max(150, min(280, max(tw(l, 16.5) for l in lines) * 1.03 + 54))
        L, S = len(label_l), len(sub_l)
        h = max(94, 52 + (L - 1) * 21 + (24 if S else 8) + S * 17 + 14)
        geom[nid] = {"w": round(w), "h": round(h), "label": "\n".join(label_l),
                     "sub": "\n".join(sub_l), "cls": n["cls"], "rank": rank[nid],
                     "sub_g": n["sub"]}

    maxr = max(g["rank"] for g in geom.values())
    # 同层排序: 语义优先 + (轻量)邻居亲和
    order = {}
    for r in range(maxr + 1):
        ns = [nid for nid, g in geom.items() if g["rank"] == r]
        ns.sort(key=lambda i: (CLS_ORDER.get(geom[i]["cls"], 3), i))
        # 邻居亲和: 与上一层顺序对齐(稳定两轮)
        for _ in range(2):
            frozen = {nid: k for k, nid in enumerate(ns)}
            prev = {nid: k for k, nid in enumerate(order.get(r - 1, []))} if r else {}

            def aff(nid):
                arr = []
                for e in edges:
                    if e["a"] == nid and e["b"] in prev:
                        arr.append(prev[e["b"]])
                    elif e["b"] == nid and e["a"] in prev:
                        arr.append(prev[e["a"]])
                return (sum(arr) / len(arr)) if arr else frozen[nid]
            ns.sort(key=aff)
        order[r] = ns
    # 位置
    colsize = [max((geom[n]["w"] if lr else geom[n]["h"]) for n in order[r]) for r in range(maxr + 1)]
    across = 0
    xs = {}
    for r in range(maxr + 1):
        tot = sum((geom[n]["h"] if lr else geom[n]["w"]) for n in order[r]) + SPACING * (len(order[r]) - 1)
        across = max(across, tot)
    pos_main = MARGIN_L
    for r in range(maxr + 1):
        base = pos_main
        tot = sum((geom[n]["h"] if lr else geom[n]["w"]) for n in order[r]) + SPACING * (len(order[r]) - 1)
        cur = (CONTENT_TOP + (across - tot) / 2) if lr else \
              (MARGIN_L + (across - tot) / 2)
        for nid in order[r]:
            g = geom[nid]
            if lr:
                g["x"] = base + (colsize[r] - g["w"]) / 2
                g["y"] = cur
                cur += g["h"] + SPACING
            else:
                g["x"] = cur
                g["y"] = base + (colsize[r] - g["h"]) / 2
                cur += g["w"] + SPACING
        pos_main += colsize[r] + (GAPX if lr else GAPY)
    along = sum(colsize) + (GAPX if lr else GAPY) * maxr
    W = MARGIN_L + (along if lr else across) + 46
    H = CONTENT_TOP + (across if lr else along) + 96

    # 面板: 从 mermaid subgraph 重建
    panels = []
    for sid, p in panels_src.items():
        mem = [m for m in p["members"] if m in geom]
        if len(mem) < 1:
            continue
        x0 = min(geom[m]["x"] for m in mem) - 26
        y0 = min(geom[m]["y"] for m in mem) - 46
        x1 = max(geom[m]["x"] + geom[m]["w"] for m in mem) + 26
        y1 = max(geom[m]["y"] + geom[m]["h"] for m in mem) + 22
        panels.append({"label": wrap_cjk(p["label"], 20)[0] if p["label"] else sid,
                       "x": round(x0), "y": round(y0), "w": round(x1 - x0), "h": round(y1 - y0)})

    # 边: 前向=直线/肘形(渲染器处理), 回边=底部走廊
    ARROW_STYLE = {"==>": "thick", "-->": "main", "---": "thin", "==": "thick", "-.->": "fb"}
    spec_edges = []
    min_bottom = max(g["y"] + g["h"] for g in geom.values())
    for e in edges:
        a, b = e["a"], e["b"]
        if a not in geom or b not in geom:
            continue
        st = ARROW_STYLE.get(e["arrow"], "main")
        lab = e["label"][:22] if e["label"] else ""
        if re.search(r"蒸馏|损失|TD|监督|梯度|优化|反传|学习信号|expectile|n-step|回传", lab or "", re.I):
            st = "loss"
        se = {"from": a, "to": b, "style": st}
        if lab:
            se["label"] = lab
        if (a, b) in back or rank[b] < rank[a]:
            k = 84 + (min_bottom - min(geom[a]["y"] + geom[a]["h"], geom[b]["y"] + geom[b]["h"]))
            se.update({"out": "bottom", "in": "bottom", "pos_out": 0.3, "pos_in": 0.6, "k": round(k)})
        spec_edges.append(se)

    nodes_out = [{"id": nid, "label": g["label"], "sub": g["sub"] or "",
                  "icon": pick_icon(g["label"] + " " + (g["sub"] or ""), g["cls"]),
                  "cls": g["cls"], "x": round(g["x"]), "y": round(g["y"]),
                  "w": g["w"], "h": g["h"]} for nid, g in geom.items()]
    return {"nodes": nodes_out, "edges": spec_edges, "panels": panels,
            "canvas": [int(W), int(H)], "legend": sorted({g["cls"] for g in geom.values()})}


def main():
    dry = "--dry" in sys.argv
    mod = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    modified = {l.split()[-1] for l in mod.split("\n") if l.endswith("arch.spec.json")}
    ok = skip = fail = 0
    for sj in sorted(glob.glob("notes/**/figures/*/arch.spec.json", recursive=True)):
        rel = sj.replace("/Users/luogu/physical_intelligence/", "")
        if rel in modified:
            skip += 1          # agent 已改过的保留人工成果
            continue
        old = json.load(open(sj))
        src = old.get("mermaid_original")
        if not src:
            skip += 1
            continue
        parsed = parse_mermaid(src)
        if parsed["error"]:
            fail += 1
            print("PARSE-FAIL", sj)
            continue
        spec = layout(parsed)
        W, H = spec["canvas"]

        def score(w, h):
            return max(w / h, h / w) / 2.0 + max(w, h) / 1650.0
        transposed = False
        if score(W, H) > 1.8:                      # 任一方向不佳就算另一方向择优
            alt = layout(parsed, transpose=True)
            if score(*alt["canvas"]) < score(W, H):
                spec, (W, H), transposed = alt, alt["canvas"], True
        # 不做画布钳制(缩画布不挪节点会造成越界); 极端比例靠 transpose 缓解
        spec["title"] = old["title"]
        spec["subtitle"] = old["subtitle"]
        spec["foot"] = old["foot"]
        spec["source"] = "relayout-v2" + ("+T" if transposed else "")
        spec["mermaid_original"] = src
        if dry:
            print(("T " if transposed else "  "), sj, spec["canvas"])
            ok += 1
            continue
        try:
            open(sj.replace(".spec.json", ".svg"), "w").write(render(spec))
        except Exception as e:
            fail += 1
            print("RENDER-FAIL", sj, e)
            continue
        json.dump(spec, open(sj, "w"), ensure_ascii=False, indent=1)
        ok += 1
    print(f"relayout-v2: {ok} done, {skip} skipped(agent-touched/hand), {fail} failed")


if __name__ == "__main__":
    main()
