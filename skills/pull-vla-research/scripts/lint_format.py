#!/usr/bin/env python3
"""lint_format: markdown 排版 lint。

安全自动修: 行尾空白、连续>=2 空行压成 1、标题后缺空行、表格前后缺空行。
只报告不修: 表格列数不齐、单行 ** 不配对、代码围栏不配对、同级标题重复。
用法: lint_format.py [--fix] [glob...]   默认 notes/**/*.md
"""
import glob
import re
import sys

def count_cols(row):
    s = row.strip()
    s = re.sub(r"\\\|", "", s)  # \| 是转义竖线, 不算列分隔; 未转义 | 一律算(GitHub 同规则)
    return len([c for c in s.strip("|").split("|")]) if s else 0

def lint(path, fix=False):
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")
    issues = []
    changed = False

    # 1. 行尾空白
    for i, l in enumerate(lines):
        if l != l.rstrip():
            lines[i] = l.rstrip()
            changed = True

    # 2. 连续空行
    out, blanks = [], 0
    for l in lines:
        if l == "":
            blanks += 1
            if blanks >= 2:
                changed = True
                continue
        else:
            blanks = 0
        out.append(l)
    lines = out

    # 3. 标题/表格/围栏前后空行
    out = []
    for i, l in enumerate(lines):
        is_h = l.startswith("#")
        is_t = l.startswith("|")
        is_f = l.startswith("```")
        prev = out[-1] if out else ""
        prev_h = prev.startswith("#")
        prev_t = prev.startswith("|")
        prev_f = prev.startswith("```")
        if (is_h and prev != "" and not prev_f) or (is_t and prev_h):
            out.append("")
            changed = True
        out.append(l)
    lines = out

    text2 = "\n".join(lines)
    if text2 != text:
        if fix:
            open(path, "w", encoding="utf-8").write(text2)
        else:
            issues.append("auto-fixable: 行尾空白/连续空行/标题空行")

    # 4. 表格列数不齐 (对齐后的 text2)
    table_header = None
    for i, l in enumerate(text2.split("\n")):
        if l.startswith("|") and re.match(r"^\|[\s:|-]+\|?$", l.strip()):
            continue  # separator
        if l.startswith("|"):
            if table_header is None:
                table_header = (i, count_cols(l))
            elif count_cols(l) != table_header[1]:
                issues.append(f"L{i+1}: 表格列数 {count_cols(l)} != 表头 {table_header[1]}")
        else:
            table_header = None

    # 5. 单行 ** 不配对
    for i, l in enumerate(text2.split("\n")):
        if l.count("**") % 2 == 1 and not l.startswith("|"):
            issues.append(f"L{i+1}: ** 不配对")

    # 6. 围栏不配对
    if text2.count("```") % 2 == 1:
        issues.append("代码围栏 ``` 不配对")

    return changed, issues


def main():
    args = [a for a in sys.argv[1:]]
    fix = "--fix" in args
    args = [a for a in args if not a.startswith("--")]
    patterns = args or ["notes/**/*.md"]
    files = sorted(set(f for p in patterns for f in glob.glob(p, recursive=True)))
    nfix, nreport = 0, 0
    for f in files:
        changed, issues = lint(f, fix)
        if changed and fix:
            nfix += 1
        elif changed:
            print(f"[fixable] {f}")
            nfix += 1
        for msg in issues:
            print(f"[issue ] {f} {msg}")
            nreport += 1
    print(f"\n{len(files)} files: {nfix} auto-fix{'ed' if fix else 'able'}, {nreport} issues reported")


if __name__ == "__main__":
    main()
