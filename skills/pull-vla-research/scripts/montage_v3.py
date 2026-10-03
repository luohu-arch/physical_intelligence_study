#!/usr/bin/env python3
"""montage_v3: 把全部 arch.png 拼成 8 宫格 montage 供视觉抽检。

依赖: qlmanage 先批量出 PNG (render_all_png), PIL 拼图。
输出 /tmp/v3audit/montage_N.png + manifest.json
"""
import glob
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = "/tmp/v3audit"
try:
    FONT = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 30)
except Exception:
    FONT = ImageFont.load_default()


def render_pngs():
    # qlmanage 多文件会因同名互相覆盖, 必须逐个渲染
    svgs = [f for f in sorted(glob.glob("notes/**/figures/*/arch.svg", recursive=True))
            if not os.path.exists(f.replace(".svg", ".png"))]
    ok = 0
    for f in svgs:
        subprocess.run(["qlmanage", "-t", "-s", "1200", "-o", OUT, f],
                       capture_output=True, timeout=120)
        t = os.path.join(OUT, os.path.basename(f) + ".png")
        if os.path.exists(t):
            os.replace(t, f.replace(".svg", ".png"))
            ok += 1
    return ok


def montage():
    os.makedirs(OUT, exist_ok=True)
    pngs = sorted(glob.glob("notes/**/figures/*/arch.png", recursive=True))
    groups, pages = {}, []
    for p in pngs:
        nid_track = p.split("/figures/")[0].replace("notes/", "")
        groups.setdefault(nid_track, []).append(p)
    # 8 张/页, 尽量同 track 分组
    batch, pages = [], []
    for track, ps in sorted(groups.items()):
        for p in ps:
            batch.append(p)
            if len(batch) == 8:
                pages.append(batch)
                batch = []
    if batch:
        pages.append(batch)
    manifest = {}
    for pi, page in enumerate(pages, 1):
        CW, CH, LBL = 700, 520, 36
        img = Image.new("RGB", (CW * 2, (CH + LBL) * 4), "#111")
        d = ImageDraw.Draw(img)
        for k, p in enumerate(page):
            try:
                t = Image.open(p)
            except Exception:
                continue
            r = min(CW / t.width, CH / t.height)
            t = t.resize((int(t.width * r), int(t.height * r)))
            x, y = (k % 2) * CW, (k // 2) * (CH + LBL)
            img.paste(t, (x + (CW - t.width) // 2, y + LBL + (CH - t.height) // 2))
            nid = p.split("/figures/")[1].split("/")[0] + " · " + p.split("/")[1]
            d.text((x + 8, y + 2), nid, fill="#ffd54f", font=FONT)
            manifest[f"{pi}/{k}"] = nid
        out = f"{OUT}/montage_{pi}.png"
        img.save(out)
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=0)
    print(f"{len(pngs)} pngs -> {len(pages)} montages")


if __name__ == "__main__":
    if "--png" in sys.argv:
        print("rendered", render_pngs(), "new pngs")
    montage()
