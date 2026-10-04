#!/usr/bin/env python3
"""词边界换行修复: 从 mermaid_original 重新推导节点 label/sub/h;
panel 标签重排(<=2 行)并把标签带移出卡片区。用法:
  python3 repair_wrap.py            # 全库
  python3 repair_wrap.py <id> ...   # 指定图
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mermaid_to_spec import parse_mermaid, wrap_cjk, _units  # noqa: E402


def repair(path, report):
    spec = json.load(open(path))
    fid = path.split("/")[-2]
    changed = False
    parsed = None
    orig = spec.get("mermaid_original")
    if orig:
        try:
            parsed = parse_mermaid(orig)
        except Exception as ex:  # noqa: BLE001
            report["noparse"].append((fid, str(ex)[:60]))

    src = parsed["nodes"] if parsed and "nodes" in parsed else {}
    for n in spec["nodes"]:
        if n["id"] not in src:
            continue
        segs = re.split(r"<br\s*/?>", src[n["id"]]["label"])
        old_lines = (n.get("label", "") + "\n" + n.get("sub", "")).split("\n")
        old_text = re.sub(r"<br[^>]*>", "", "".join(old_lines)).replace(" ", "")
        src_text = "".join(segs).replace(" ", "")
        if src_text != old_text and not src_text.startswith(old_text):
            report["mismatch"].append((fid, n["id"]))  # 手改过, 不动
            continue
        # 换行宽度由卡宽决定(collide_check 文本溢出公式反推): label 15.5px / sub 12px
        mu_l = (n["w"] - 34) / (1.15 * 15.5)
        mu_s = (n["w"] - 34) / (1.15 * 12.0)
        new_lines = []
        for seg in segs:
            if len(new_lines) < 2:
                ls = wrap_cjk(seg, mu_l)
                if len(new_lines) + len(ls) > 2:  # 该段跨界: 溢出部分转 sub 预算重排
                    fit = 2 - len(new_lines)
                    new_lines += ls[:fit]
                    rest = ls[fit] if fit < len(ls) else ""
                    for extra in ls[fit + 1:]:
                        if rest and extra and rest[-1].isascii() and extra[0].isascii():
                            rest += " " + extra
                        else:
                            rest += extra
                    if rest:
                        new_lines += wrap_cjk(rest, mu_s)
                else:
                    new_lines += ls
            else:
                new_lines += wrap_cjk(seg, mu_s)
        if len(new_lines) > 5:
            report["overflow"].append((fid, n["id"], src_text))
            continue
        if len(new_lines) > 3:
            report["long"].append((fid, n["id"], " | ".join(new_lines)))
        new_lines = new_lines[:5]
        new_label = "\n".join(new_lines[:2])
        new_sub = "\n".join(new_lines[2:])
        L, S = len(new_lines[:2]), len(new_lines[2:])
        new_h = max(104, 62 + (L - 1) * 21 + (24 if S else 8) + S * 17 + 14)
        if (new_label, new_sub, new_h) != (n["label"], n.get("sub", ""), n["h"]):
            n["label"], n["sub"], n["h"] = new_label, new_sub, new_h
            changed = True

    # panel: 从 mermaid subgraph 恢复完整标题(转换期的 22 单位截断/合并残片),
    # 标签 <=2 行(词边界), 标签带让位卡片, 不进标题区(y>=92)
    mpanels = {sid: pd for sid, pd in (parsed.get("panels") or {}).items()
               if pd.get("parent") is None} if parsed else {}
    all_ids = {n["id"] for n in spec["nodes"]}
    for p in spec.get("panels", []):
        inside = {n["id"] for n in spec["nodes"]
                  if p["x"] - 1 <= n["x"] + n["w"] / 2 <= p["x"] + p["w"] + 1
                  and p["y"] - 1 <= n["y"] + n["h"] / 2 <= p["y"] + p["h"] + 1}
        matched = []
        for sid, pd in mpanels.items():
            mems = [m for m in pd["members"] if m in all_ids]
            if mems and sum(1 for m in mems if m in inside) >= max(1, (len(mems) + 1) // 2):
                matched.append(pd["label"])
        if len(matched) == 1:
            full_title = matched[0]
        elif len(matched) > 1:
            full_title = " · ".join(t.split(":")[0].strip() for t in matched)
        else:
            full_title = p["label"].replace("\n", "")
        labs = wrap_cjk(full_title, max(23.3, (p["w"] - 60) / (1.15 * 12.5)))
        if len(labs) > 2:
            labs = [labs[0], labs[1] + "…"]
        newlab = "\n".join(labs)
        if newlab != p["label"]:
            p["label"] = newlab
            changed = True
        band = 39 + (17 if "\n" in p["label"] else 0)
        if inside:
            min_top = min(n["y"] for n in spec["nodes"] if n["id"] in inside)
            if p["y"] + band > min_top - 8:
                dy = (p["y"] + band) - (min_top - 8)
                p["y"] -= dy
                p["h"] += dy
                changed = True
        if p["y"] < 92:
            dy = 92 - p["y"]
            p["y"] = 92
            p["h"] = max(60, p["h"] - dy)
            changed = True

    # 卡片增高后保证底部不压图例
    W, H = spec["canvas"]
    max_bottom = max((n["y"] + n["h"] for n in spec["nodes"]), default=0)
    max_bottom = max([max_bottom] + [p["y"] + p["h"] for p in spec.get("panels", [])])
    if max_bottom + 118 > H:
        spec["canvas"] = [W, max_bottom + 118]
        changed = True

    if changed:
        json.dump(spec, open(path, "w"), ensure_ascii=False, indent=1)
        report["changed"].append(fid)
    return changed


def main():
    ids = sys.argv[1:]
    paths = (glob.glob("notes/**/figures/*/arch.spec.json", recursive=True)
             if not ids else
             [p for p in glob.glob("notes/**/figures/*/arch.spec.json", recursive=True)
              if p.split("/")[-2] in ids])
    report = {"changed": [], "mismatch": [], "overflow": [], "long": [], "noparse": []}
    for p in sorted(paths):
        repair(p, report)
    print(f"specs: {len(paths)}  changed: {len(report['changed'])}")
    for k in ("mismatch", "overflow", "long", "noparse"):
        if report[k]:
            print(f"\n{k} ({len(report[k])}):")
            for it in report[k]:
                print("  ", it)
    json.dump(report, open("/tmp/repair_wrap_report.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
