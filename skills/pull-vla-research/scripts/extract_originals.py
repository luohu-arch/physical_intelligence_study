#!/usr/bin/env python3
"""extract_originals: 用 arXiv 源码包中的原始图文件替换页面截图。

流程: note -> 本地PDF -> arXiv ID -> e-print 下载 -> 解析主 .tex
-> 按序数 Figure 环境 -> 单一 includegraphics 且文件存在 -> 原图替换
-> 题注词重叠验证(>=0.5)防止错配; 多子图/TikZ/EPS/无源码保留截图。
"""
import glob
import os
import re
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(__file__))
from embed_figures import optimize  # noqa: E402

CACHE = "/tmp/arxiv_src"
UA = "physical-intelligence-study-notes/1.0 (research; contact: local)"
EXTS = [".pdf", ".eps", ".png", ".jpg", ".jpeg", ".PDF", ".PNG", ".JPG"]


def arxiv_id_of(pdf):
    m = re.search(r"(\d{4}\.\d{4,5})(?:v\d+)?\.pdf$", os.path.basename(pdf))
    return m.group(1) if m else None


def fetch(arxiv_id):
    os.makedirs(CACHE, exist_ok=True)
    tar_path = f"{CACHE}/{arxiv_id}.bin"
    if not os.path.exists(tar_path):
        for attempt in range(3):
            r = subprocess.run(
                ["curl", "-sL", "--max-time", "90", "-A", UA,
                 "-o", tar_path,
                 f"https://arxiv.org/e-print/{arxiv_id}"],
                capture_output=True)
            if r.returncode == 0 and os.path.getsize(tar_path) > 500:
                break
            time.sleep(4)
        else:
            return None
    out = f"{CACHE}/{arxiv_id}"
    if os.path.isdir(out):
        return out
    os.makedirs(out, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(out)
    except tarfile.TarError:
        import gzip
        try:
            raw = gzip.open(tar_path, "rb").read()
            open(f"{out}/main.tex", "wb").write(raw)
        except Exception:
            # PDF-only 投稿: 无源码
            return None
    return out if glob.glob(f"{out}/**/*.tex", recursive=True) else None


def read(p):
    for enc in ("utf-8", "latin-1"):
        try:
            return open(p, encoding=enc).read()
        except UnicodeDecodeError:
            continue
    return open(p, encoding="utf-8", errors="replace").read()


def inline_tex(src_dir, fname, depth=0):
    p = os.path.join(src_dir, fname)
    if not os.path.exists(p) and os.path.exists(p + ".tex"):
        p += ".tex"
    if not os.path.exists(p) or depth > 6:
        return ""
    text = read(p)
    text = re.sub(r"(?m)^\s*%.*$", "", text)

    def rep(m):
        return "\n" + inline_tex(src_dir, m.group(1).strip(), depth + 1) + "\n"

    return re.sub(r"\\(?:input|include)\{([^}]+)\}", rep, text)


def find_main_tex(src_dir):
    cands = [f for f in glob.glob(f"{src_dir}/**/*.tex", recursive=True)
             if re.search(r"\\documentclass", read(f))]
    if not cands:
        return None
    return max(cands, key=lambda f: os.path.getsize(f))


def graphics_dirs(src_dir, text):
    dirs = [""]
    for m in re.finditer(r"\\graphicspath\{((?:\{[^}]*\})+)\}", text):
        for d in re.findall(r"\{([^}]*)\}", m.group(1)):
            dirs.append(d)
    return dirs


def resolve_fig(src_dir, gdirs, name):
    name = name.strip()
    for gd in gdirs:
        base = os.path.join(src_dir, gd, name)
        if os.path.exists(base):
            return base
        for e in EXTS:
            if os.path.exists(base + e):
                return base + e
    return None


def brace_match(text, start):
    """start 指向 '{', 返回组内文本。"""
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return text[start + 1:]


def figure_envs(text):
    envs = []
    for m in re.finditer(r"\\begin\{(figure\*?)\}", text):
        end = text.find(r"\end{" + m.group(1) + "}", m.end())
        if end == -1:
            continue
        envs.append(text[m.start():end])
    return envs


def env_caption(env):
    m = re.search(r"\\caption(?:\[[^\]]*\])?\{", env)
    if not m:
        return ""
    return brace_match(env, m.end() - 1)


def env_includes(env):
    return re.findall(
        r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", env)


def norm_words(s):
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}()\[\]$~\\,%&^_]", " ", s)
    return [w for w in re.split(r"\s+", s.lower()) if len(w) > 2]


def overlap(a, b):
    wa, wb = set(norm_words(a)), set(norm_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    stats = {"orig-verified": 0, "orig-lowmatch": [], "keep-multi": 0,
             "keep-tikz": 0, "keep-eps": 0, "keep-missfile": 0,
             "keep-nosrc": 0, "keep-nonum": 0}
    done_ids = set()
    for note in sorted(glob.glob("notes/**/*.md", recursive=True)):
        nid = os.path.basename(note)[:-3]
        if "brief" in nid or (only and only != nid):
            continue
        text = open(note, encoding="utf-8").read()
        pm = re.search(r"(?:Local PDF|本地 PDF)[：:]\s*`?([^`\n]+)`?", text)
        if not pm:
            continue
        pdf = pm.group(1).strip()
        aid = arxiv_id_of(pdf)
        figs = sorted(set(int(m) for m in
                          re.findall(r"figures/%s/fig(\d+)\.png" % nid, text)))
        if not aid or not figs:
            continue
        figdir = os.path.join(os.path.dirname(note), "figures", nid)
        if aid in done_ids:
            src_dir = f"{CACHE}/{aid}" if os.path.isdir(f"{CACHE}/{aid}") else None
        else:
            src_dir = fetch(aid)
            done_ids.add(aid)
            time.sleep(3.1)
        if not src_dir:
            stats["keep-nosrc"] += len(figs)
            continue
        main_tex = find_main_tex(src_dir)
        if not main_tex:
            stats["keep-nosrc"] += len(figs)
            continue
        full = inline_tex(os.path.dirname(main_tex), os.path.basename(main_tex))
        gdirs = graphics_dirs(os.path.dirname(main_tex), full)
        envs = figure_envs(full)
        cap_note = {int(a): b for a, b in re.findall(
            r"\*论文 Figure (\d+)（p[\d-]+）：([^*]+)\*", text)}
        for n in figs:
            if n > len(envs):
                stats["keep-nonum"] += 1
                continue
            # 版本漂移: 在 ±2 邻域内选题注重叠最高的环境
            best, best_ov, best_k = None, -1.0, 0
            for k in range(-2, 3):
                i = n - 1 + k
                if 0 <= i < len(envs) and len(env_includes(envs[i])) == 1:
                    # 笔记题注仅存前 100 字, 同步截断源题注再比
                    ov = overlap(env_caption(envs[i])[:100], cap_note.get(n, ""))
                    if ov > best_ov:
                        best, best_ov, best_k = envs[i], ov, k
            if best is None:
                stats["keep-multi"] += 1
                continue
            incs = env_includes(best)
            if not incs:
                stats["keep-tikz"] += 1
                continue
            if len(incs) > 1:
                stats["keep-multi"] += 1
                continue
            path = resolve_fig(os.path.dirname(main_tex), gdirs, incs[0])
            if not path:
                stats["keep-missfile"] += 1
                continue
            if path.lower().endswith(".eps"):
                stats["keep-eps"] += 1
                continue
            if best_ov < 0.5:
                stats["orig-lowmatch"].append(
                    f"{nid} fig{n} ov={best_ov:.2f} (kept crop)")
                continue
            out = f"{figdir}/fig{n}.png"
            if path.lower().endswith(".pdf"):
                import fitz
                doc = fitz.open(path)
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72),
                                        alpha=False)
                pix.save("/tmp/_orig.png")
                doc.close()
                optimize("/tmp/_orig.png", out, max_w=1500)
            else:
                optimize(path, out, max_w=1500)
            stats["orig-verified"] += 1
    print("orig-verified:", stats["orig-verified"])
    print("keep: multi=%d tikz=%d eps=%d missfile=%d nosrc=%d nonum=%d"
          % (stats["keep-multi"], stats["keep-tikz"], stats["keep-eps"],
             stats["keep-missfile"], stats["keep-nosrc"], stats["keep-nonum"]))
    print("LOW-MATCH (需目检):")
    for x in stats["orig-lowmatch"]:
        print(" ", x)


if __name__ == "__main__":
    main()
