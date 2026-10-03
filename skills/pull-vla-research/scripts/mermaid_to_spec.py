#!/usr/bin/env python3
"""mermaid_to_spec: 把笔记里的 v2 mermaid 块自动转成 v3 SVG spec。

管线: 解析 mermaid flowchart (节点/类/边/subgraph)
    -> 分层布局 (rank = 最长路径, 同层 barycenter 排序, LR=横排 TB=竖排)
    -> 图标按关键词分配
    -> fancy_diagram.render 出 SVG (+qlmanage PNG)

用法:
  mermaid_to_spec.py notes/rl/core/foo.md            # 转换单篇 (写 figures/foo/arch.*)
  mermaid_to_spec.py --dry                            # 全库解析报告
  mermaid_to_spec.py --all                            # 全库转换
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fancy_diagram import render, tw  # noqa: E402

# ---------- mermaid 解析 ----------

FILL2CLS = {"#e8f5e9": "data", "#e3f2fd": "frozen", "#fff3e0": "train",
            "#ffebee": "loss", "#f3e5f5": "act", "#eceff1": "loop",
            "#e0f2f1": "env", "#fffde7": "mem", "#fce4ec": "reward",
            "#fff8e1": "key", "#fafbfd": "loop"}
CANON = set(FILL2CLS.values())

NODE_RE = re.compile(r'^([A-Za-z0-9_.\-]+)\s*(?:::([A-Za-z0-9_\-]+))?\s*(.*)$')
SHAPE_RE = re.compile(r'[([<{]+\s*"([^"]*)"\s*[)\]>}]+|[([<{]+([^"\n]*?)[)\]>}]+')


def parse_mermaid(src):
    """返回 {nodes, edges, panels, error}。nodes: id->{label, cls, sub(member)}"""
    lines = []
    skip = False
    for ln in src.split("\n"):
        t = ln.strip()
        if t.startswith("%%{init"):
            skip = True
            if "}%%" in t:
                skip = False
            continue
        if skip:
            if "}%%" in t:
                skip = False
            continue
        if not t or t.startswith("%%"):
            continue
        lines.append(t)

    direction = "LR"
    nodes, edges, panels = {}, [], {}
    stack = []          # subgraph id 栈
    panel_parent = {}   # subgraph id -> 父 subgraph id
    cls_of = {}         # node id -> class
    classdefs = {}      # class name -> fill

    m = re.match(r'(?:flowchart|graph)\s+(LR|RL|TB|TD|BT)', lines[0] if lines else "")
    if m:
        direction = {"TD": "TB", "BT": "TB"}.get(m.group(1), m.group(1))
        lines = lines[1:]
    else:
        return {"error": "非 flowchart 图或缺少方向声明"}

    def reg_node(tok):
        tok = tok.strip()
        if not tok:
            return None
        mm = NODE_RE.match(tok)
        if not mm:
            return None
        nid, inline_cls, rest = mm.group(1), mm.group(2), mm.group(3).strip()
        if nid in ("subgraph", "end", "direction", "class", "classDef", "style", "linkStyle", "click"):
            return None
        label = nid
        sm = re.search(r'"([^"]*)"', rest)
        if sm:
            label = sm.group(1)
        else:
            sm2 = SHAPE_RE.match(rest) if rest else None
            if sm2:
                label = (sm2.group(1) or sm2.group(2) or nid).strip()
        if nid not in nodes:
            nodes[nid] = {"label": label, "cls": None, "sub": None}
        elif label != nid:
            nodes[nid]["label"] = label
        if inline_cls:
            cls_of[nid] = inline_cls
        if stack and nodes[nid]["sub"] is None:
            nodes[nid]["sub"] = stack[-1]
        return nid

    def norm_arrows(t):
        """归一化边写法: -- text --> / -. text .-> / == text ==> / -->|label|
        文本型标签抽出存 labs, 箭头归一成 -->/-.->/==>/--- 供 split。"""
        labs = []

        def keep(arrow):
            def _k(m):
                labs.append(re.sub(r"<br\s*/?>", " ", m.group(1)).strip())
                return arrow
            return _k
        t = re.sub(r'-\.\s*"([^"]*)"\s*\.->', keep("-.->"), t)
        t = re.sub(r'-\.\s+([^.]+?)\s+\.->', keep("-.->"), t)
        t = re.sub(r'==\s*"([^"]*)"\s*==>', keep("==>"), t)
        t = re.sub(r'={2}\s+([^=]+?)\s+={2}>', keep("==>"), t)
        t = re.sub(r'-{2}\s*"([^"]*)"\s*-{2}>', keep("-->"), t)
        t = re.sub(r'-{2}\s+([^->][^>]*?)\s+-{2}>', keep("-->"), t)
        t = re.sub(r'(-{2,}>|-\.\->|==>|---)\|([^|]*)\|',
                   lambda m: labs.append(re.sub(r"<br\s*/?>", " ", m.group(2)).strip()) or m.group(1), t)
        return t, labs

    for t in lines:
        sm = re.match(r'subgraph\s+([A-Za-z0-9_.\-]+)\s*(?:\[[\'"]?(.*?)[\'"]?\])?\s*$', t)
        if sm:
            sid, slab = sm.group(1), sm.group(2)
            panels[sid] = {"label": (slab or sid).strip(), "members": [], "parent": stack[-1] if stack else None}
            if stack:
                panel_parent[sid] = stack[-1]
            stack.append(sid)
            continue
        if t == "end" or t.startswith("end "):
            if stack:
                stack.pop()
            continue
        if t.startswith("direction"):
            continue
        cm = re.match(r'classDef\s+([A-Za-z0-9_\-]+)\s+(.*)$', t)
        if cm:
            name, body = cm.group(1), cm.group(2)
            fm = re.search(r'fill:\s*(#[0-9a-fA-F]{6})', body)
            classdefs[name] = FILL2CLS.get((fm.group(1).lower() if fm else ""), "loop")
            continue
        cm = re.match(r'class\s+([A-Za-z0-9_.,\- ]+)\s+([A-Za-z0-9_\-]+)\s*$', t)
        if cm:
            for nid in re.split(r'[,\s]+', cm.group(1)):
                if nid:
                    cls_of[nid] = cm.group(2)
            continue
        if t.startswith(("style", "linkStyle", "click")):
            continue

        # 边/节点行
        tt, labels = norm_arrows(t)
        parts = re.split(r'(-->|-\.\->|==>|---|==)', tt)
        if len(parts) >= 3:
            ids = []
            arrow = None
            for p in parts:
                if p in ("-->", "-.->", "==>", "---", "=="):
                    arrow = p
                else:
                    nid = reg_node(p)
                    if nid:
                        ids.append(nid)
            lab = labels[0] if labels else ""
            for a, b in zip(ids, ids[1:]):
                edges.append({"a": a, "b": b, "arrow": arrow, "label": lab})
        else:
            reg_node(tt)

    for nid, n in nodes.items():
        c = cls_of.get(nid)
        if c in CANON:
            n["cls"] = c
        elif c in classdefs:
            n["cls"] = classdefs[c]
        else:
            n["cls"] = "loop"
        if n["sub"]:
            panels.setdefault(n["sub"], {"label": n["sub"], "members": [], "parent": None})
            panels[n["sub"]]["members"].append(nid)
    # 只保留顶层面板为绘制面板; 成员并含嵌套面板成员
    top_panels = {sid: p for sid, p in panels.items() if p["parent"] is None and p["members"]}
    for sid, p in list(panels.items()):
        if p["parent"]:
            cur = p["parent"]
            while cur and cur in panel_parent:
                cur = panel_parent[cur]
            if cur in top_panels:
                top_panels[cur]["members"].extend(p["members"])
    return {"nodes": nodes, "edges": edges, "panels": top_panels, "dir": direction, "error": None}


# ---------- 图标分配 ----------

KEYWORD_ICONS = [
    (r"多视角|RGB|相机|摄像|图像|视觉|image|camera|observation|像素", "camera"),
    (r"语言|指令|命令|prompt|文本|对话|问句|utterance|text|指令词", "chat"),
    (r"状态|state|观测|obs|本体|proprio", "state"),
    (r"VLM|LLM|主干|backbone|Gemma|Qwen|GPT|语言模型|推理|reason|think|思维", "brain"),
    (r"编码|encoder|SigLIP|ViT|CLIP|感知|percept", "eye"),
    (r"critic|Q函数|Q头|评估|value|verifier|判别", "target"),
    (r"奖励|reward|评分|打分|score", "reward"),
    (r"环境|env|仿真|sim|Isaac|MuJoCo|物理|转移|transition|动力学|dynamic", "gear"),
    (r"世界模型|world model|dream|想象|imagine|latent model", "graph"),
    (r"记忆|memory|历史|history|上下文|context|经验|replay|buffer", "db"),
    (r"噪声|noise|先验|prior|latent|隐变量|z_0|x_0", "noise"),
    (r"数据集|dataset|演示|demo|数据流|corpus|采集", "db"),
    (r"蒸馏|distill|teacher|student", "flask"),
    (r"损失|loss|监督|目标函数|TD|训练信号", "sigma"),
    (r"chunk|块|token|分词|tokenizer|离散|码本|codebook|VQ", "stack"),
    (r"机器人|robot|机械臂|臂|夹爪|gripper|末端|执行|控制|control|跟踪", "robot"),
    (r"规划|plan|搜索|search tree|蒙特|MCTS|选项|option|技能|skill", "tree"),
    (r"检索|retrieve|RAG|知识库|召回", "search"),
    (r"评测|benchmark|成功|success|指标|metric|评估基准", "check"),
    (r"动作|action|act|策略|policy|actor|π|decoder|解码|输出", "lightning"),
]
CLS_DEFAULT_ICON = {"data": "db", "frozen": "brain", "train": "cpu", "loss": "sigma",
                    "act": "lightning", "loop": "refresh", "env": "gear", "mem": "db",
                    "reward": "medal", "key": "spark"}


def pick_icon(label, cls):
    for pat, icon in KEYWORD_ICONS:
        if re.search(pat, label, re.I):
            return icon
    return CLS_DEFAULT_ICON.get(cls, "cpu")


# ---------- 布局 ----------

GAPX, GAPY, SPACING, MARGIN_L, CONTENT_TOP = 104, 92, 40, 44, 130
CROWD = 5  # 同 rank 节点数达到该值则加大间距
TIGHT = {"gapx": 66, "gapy": 60, "spacing": 24, "max_units": 10, "wcap": 260}


def wrap_cjk(text, max_units=13.5):
    units = 0
    lines, cur = [], ""
    for ch in text:
        u = 1.0 if ord(ch) > 0x2E80 else 0.55
        if units + u > max_units and cur:
            lines.append(cur)
            cur, units = ch, u
        else:
            cur += ch
            units += u
    if cur:
        lines.append(cur)
    return lines[:3]


def layout(parsed, tight=False):
    nodes, edges = parsed["nodes"], parsed["edges"]
    lr = parsed["dir"] in ("LR", "RL")
    gapx = TIGHT["gapx"] if tight else GAPX
    gapy = TIGHT["gapy"] if tight else GAPY
    spacing = TIGHT["spacing"] if tight else SPACING
    max_units = TIGHT["max_units"] if tight else 13.5
    wcap = TIGHT["wcap"] if tight else 290

    # rank: 最长路径 (忽略回边)
    adj = {i: [] for i in nodes}
    indeg = {i: 0 for i in nodes}
    for e in edges:
        if e["a"] in nodes and e["b"] in nodes and e["a"] != e["b"]:
            adj[e["a"]].append(e["b"])
            indeg[e["b"]] += 1
    back = set()
    for e in edges:
        if e["a"] == e["b"]:
            back.add((e["a"], e["b"]))
    # DFS 找回边
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
        changed = False
        for u in nodes:
            for v in fwd[u]:
                if rank[v] < rank[u] + 1:
                    rank[v] = rank[u] + 1
                    changed = True
        if not changed:
            break

    # 节点尺寸
    geom = {}
    for nid, n in nodes.items():
        pre = re.split(r"<br\s*/?>", n["label"])
        lines = [l for seg in pre for l in wrap_cjk(seg, max_units)][:3]
        label_l = lines[:2]
        sub_l = lines[2:]
        w = max(140, min(wcap, max(tw(l, 16.5) for l in lines) * 1.03 + 56))
        L, S = len(label_l), len(sub_l)
        h = max(92, 50 + (L - 1) * 21 + (24 if S else 8) + S * 17 + 14)
        geom[nid] = {"w": round(w), "h": round(h), "label": "\n".join(label_l),
                     "sub": "\n".join(sub_l), "cls": n["cls"], "rank": rank[nid]}

    maxr = max(g["rank"] for g in geom.values())
    ranks = [[] for _ in range(maxr + 1)]
    for nid, g in geom.items():
        ranks[g["rank"]].append(nid)

    # 同层排序: barycenter 2 轮
    order = {r: list(ns) for r, ns in enumerate(ranks)}
    pos = {}
    for r, ns in order.items():
        for i, nid in enumerate(ns):
            pos[nid] = i
    for _ in range(2):
        for r in range(maxr + 1):
            def bc(nid):
                ps = [pos[e["a"]] for e in edges if e["b"] == nid and rank[e["a"]] == r - 1]
                qs = [pos[e["b"]] for e in edges if e["a"] == nid and rank[e["b"]] == r - 1]
                arr = ps + qs
                return sum(arr) / len(arr) if arr else pos[nid]
            order[r].sort(key=bc)
            for i, nid in enumerate(order[r]):
                pos[nid] = i
        for r in range(maxr, -1, -1):
            def bc2(nid):
                ps = [pos[e["b"]] for e in edges if e["a"] == nid and rank[e["b"]] == r + 1]
                qs = [pos[e["a"]] for e in edges if e["b"] == nid and rank[e["a"]] == r + 1]
                arr = ps + qs
                return sum(arr) / len(arr) if arr else pos[nid]
            order[r].sort(key=bc2)
            for i, nid in enumerate(order[r]):
                pos[nid] = i

    # 坐标
    colsize = []
    for r in range(maxr + 1):
        if lr:
            colsize.append(max(geom[nid]["w"] for nid in order[r]))
        else:
            colsize.append(max(geom[nid]["h"] for nid in order[r]))
    across_total = max(sum((geom[nid]["h"] if lr else geom[nid]["w"]) for nid in order[r])
                       + spacing * (len(order[r]) - 1) for r in range(maxr + 1))
    x = MARGIN_L
    y = CONTENT_TOP
    for r in range(maxr + 1):
        sp_r0 = spacing * (1.4 if len(order[r]) >= CROWD else 1.0)
        tot = sum(geom[nid]["h" if lr else "w"] for nid in order[r]) + sp_r0 * (len(order[r]) - 1)
        # 跨轴方向居中, 消除整排顶对齐导致的失衡
        if lr:
            y = CONTENT_TOP + (across_total - tot) / 2
            cur, base = y, x
        else:
            x = MARGIN_L + (across_total - tot) / 2
            cur, base = x, y
        for nid in order[r]:
            g = geom[nid]
            sp_r = spacing * (1.4 if len(order[r]) >= CROWD else 1.0)
            if lr:
                g["x"] = base + (colsize[r] - g["w"]) / 2
                g["y"] = cur
                cur += g["h"] + sp_r
            else:
                g["x"] = cur
                g["y"] = base + (colsize[r] - g["h"]) / 2
                cur += g["w"] + sp_r
            g["_tot"] = tot
        if lr:
            x += colsize[r] + gapx
        else:
            y += colsize[r] + gapy

    along = sum(colsize) + (gapx if lr else gapy) * maxr          # 沿 rank 方向总长
    across = across_total
    if lr:
        W, H = MARGIN_L + along + 44, CONTENT_TOP + across + 90
    else:
        W, H = MARGIN_L + across + 44, CONTENT_TOP + along + 90

    # 面板 bbox
    panels = []
    for sid, p in parsed["panels"].items():
        mem = [m for m in p["members"] if m in geom]
        if not mem:
            continue
        x0 = min(geom[m]["x"] for m in mem) - 24
        y0 = min(geom[m]["y"] for m in mem) - 44
        x1 = max(geom[m]["x"] + geom[m]["w"] for m in mem) + 24
        y1 = max(geom[m]["y"] + geom[m]["h"] for m in mem) + 20
        panels.append({"label": wrap_cjk(p["label"], 22)[0] if p["label"] else sid,
                       "x": round(x0), "y": round(y0), "w": round(x1 - x0), "h": round(y1 - y0)})

    # 边
    ARROW_STYLE = {"==>": "thick", "-->": "main", "---": "thin", "==": "thick", "-.->": "fb"}
    spec_edges, back_i = [], 0
    for e in edges:
        a, b = e["a"], e["b"]
        if a not in geom or b not in geom:
            continue
        st = ARROW_STYLE.get(e["arrow"], "main")
        lab = e["label"][:24] if e["label"] else ""
        if re.search(r"蒸馏|损失|TD|监督|梯度|优化|反传|学习信号|expectile|n-step|回传", lab or "", re.I):
            st = "loss"
        se = {"from": a, "to": b, "style": st}
        if lab:
            se["label"] = lab
        if (a, b) in back and rank[b] < rank[a]:
            above = geom[a]["y"] - CONTENT_TOP > 100
            se.update({"out": "top" if above else "bottom", "in": "top" if above else "bottom",
                       "pos_out": 0.3, "pos_in": 0.55, "k": 74})
            back_i += 1
        elif rank[a] == rank[b] and geom[a]["x"] < geom[b]["x"] - 10:
            se.update({"out": "bottom", "in": "bottom", "pos_out": 0.75, "pos_in": 0.25, "k": 52})
        elif rank[a] == rank[b]:
            se.update({"out": "bottom", "in": "bottom", "pos_out": 0.25, "pos_in": 0.75, "k": 52})
        if len(lab) > 10:
            se["loff"] = [0, -18]
        spec_edges.append(se)

    nodes_out = []
    for nid, g in geom.items():
        nodes_out.append({"id": nid, "label": g["label"], "sub": g["sub"] or "",
                          "icon": pick_icon(g["label"] + " " + (g["sub"] or ""), g["cls"]),
                          "cls": g["cls"], "x": round(g["x"]), "y": round(g["y"]),
                          "w": g["w"], "h": g["h"]})
    used = sorted({g["cls"] for g in geom.values()})
    return {"nodes": nodes_out, "edges": spec_edges, "panels": panels,
            "canvas": [int(W), int(H)], "legend": used,
            "layout_wh": (round(W), round(H))}


# ---------- 驱动 ----------

def note_meta(note_path):
    text = open(note_path, encoding="utf-8").read()
    title = ""
    for ln in text.split("\n"):
        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
            break
    sub = ""
    m = re.search(r"## 一句话总结\s*\n+([^\n]+)", text)
    if m:
        sub = m.group(1).strip().strip("*")[:88]
    am = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})", text[:3000])
    nid = os.path.basename(note_path)[:-3]
    return title or nid, sub, f"{nid}" + (f" · arXiv {am.group(1)}" if am else ""), text


def convert(note_path, write=True):
    nid = os.path.basename(note_path)[:-3]
    sub_dir = os.path.dirname(note_path)[len("notes/"):]
    figdir = f"notes/{sub_dir}/figures/{nid}"
    _, subtitle, foot, text = note_meta(note_path)
    mm = re.search(r"```mermaid\n(.*?)\n```", text, re.S)
    if not mm:
        return {"status": "no-mermaid"}
    parsed = parse_mermaid(mm.group(1))
    if parsed["error"] or not parsed["nodes"]:
        return {"status": "parse-fail", "error": parsed["error"], "note": note_path}
    spec = layout(parsed)
    W, H = spec["canvas"]
    if W > 1800 or H > 1400:
        spec = layout(parsed, tight=True)
        W, H = spec["canvas"]
        if W > 2100 or H > 1600:  # 仍超限: 接受真实画布 (SVG 在 md 中自适应缩放)
            pass
    spec["title"] = nid
    spec["subtitle"] = subtitle
    spec["foot"] = foot + " · v3 auto"
    spec["mermaid_original"] = mm.group(1)
    spec["source"] = "mermaid-auto"
    if not write:
        # dry: 图标名检查
        return {"status": "dry-ok", "n_nodes": len(spec["nodes"]), "spec": spec}
    svg = render(spec)
    os.makedirs(figdir, exist_ok=True)
    open(f"{figdir}/arch.svg", "w", encoding="utf-8").write(svg)
    json.dump(spec, open(f"{figdir}/arch.spec.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return {"status": "ok", "figdir": figdir, "spec": spec}


def main():
    args = sys.argv[1:]
    if args and args[0] == "--dry":
        notes = [n for n in sorted(glob.glob("notes/**/*.md", recursive=True))
                 if "/briefs/" not in n]
        fails, total = [], 0
        for n in notes:
            text = open(n, encoding="utf-8").read()
            if "```mermaid" not in text and "arch.svg" not in text:
                continue
            total += 1
            try:
                r = convert(n, write=False)
                if r["status"] == "parse-fail":
                    fails.append((n, r["error"]))
                    print("PARSE-FAIL", n, r["error"])
                elif r["status"] == "dry-ok":
                    for nd in r["spec"]["nodes"]:
                        pick = nd["icon"]
            except SystemExit as e:
                fails.append((n, str(e)))
                print("ICON-FAIL", n, e)
            except Exception as e:
                fails.append((n, repr(e)))
                print("ERROR", n, repr(e))
        print(f"\ndry: {total} notes with diagrams, {len(fails)} failures")
        return
    if args and args[0] == "--all":
        notes = [n for n in sorted(glob.glob("notes/**/*.md", recursive=True))
                 if "/briefs/" not in n]
        ok = fail = skip = 0
        pngs = []
        for n in notes:
            text = open(n, encoding="utf-8").read()
            if "```mermaid" not in text:
                skip += 1
                continue
            try:
                r = convert(n, write=True)
                if r["status"] == "ok":
                    ok += 1
                    pngs.append(r["figdir"] + "/arch.svg")
                else:
                    fail += 1
                    print("FAIL", n, r)
            except Exception as e:
                fail += 1
                print("ERR", n, repr(e))
        print(f"\nconverted {ok}, failed {fail}, skipped {skip}")
        return
    r = convert(args[0])
    print(r.get("status"), r.get("error", ""))


if __name__ == "__main__":
    main()
