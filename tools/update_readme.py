#!/usr/bin/env python3
"""
tools/update_readme.py — 自动维护 ai_workspace 各级 README 目录表

由 manage_repos.sh 调用，也可直接运行：
  python3 tools/update_readme.py rebuild                         # 从 .gitmodules 重建全部 README
  python3 tools/update_readme.py add <path> <url> [描述]        # 新增仓库后更新 README
  python3 tools/update_readme.py rm  <path>                     # 删除仓库后更新 README
  python3 tools/update_readme.py new-domain <domain> <中文名>   # 新建 research 分区
"""

import re
import sys
import json
from pathlib import Path

# ── 路径常量 ──────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent   # ai_workspace/
RESEARCH    = ROOT / "research"
GITMODULES  = ROOT / ".gitmodules"
DESCS_FILE  = Path(__file__).resolve().parent / "repo_descriptions.json"
DOMAINS_FILE= Path(__file__).resolve().parent / "domains.json"
ROOT_README = ROOT / "README.md"
RES_README  = RESEARCH / "README.md"

# ── domains 动态加载 ──────────────────────────────────────

def load_domains() -> dict:
    """从 domains.json 加载 {domain: 中文名}，文件不存在时返回内置默认值。"""
    if DOMAINS_FILE.exists():
        return json.loads(DOMAINS_FILE.read_text("utf-8"))
    return {
        "generative_vision": "生成模型与视觉",
        "embodied_ai":       "具身智能",
        "multimodal_vlm":    "多模态 VLM",
        "video_generation":  "视频生成",
    }

def save_domains(d: dict):
    DOMAINS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), "utf-8")

# ── 描述库 ────────────────────────────────────────────────

def load_descs() -> dict:
    if DESCS_FILE.exists():
        return json.loads(DESCS_FILE.read_text("utf-8"))
    return {}

def save_descs(d: dict):
    DESCS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), "utf-8")

# ── gitmodules 解析 ───────────────────────────────────────

def parse_gitmodules() -> list:
    """从 .gitmodules 解析所有 submodule，返回 [{path, url, name}]"""
    if not GITMODULES.exists():
        return []
    text = GITMODULES.read_text("utf-8")
    out = []
    for block in text.split("[submodule")[1:]:
        path_m = re.search(r'path\s*=\s*(\S+)', block)
        url_m  = re.search(r'url\s*=\s*(\S+)',  block)
        if path_m and url_m:
            p = path_m.group(1).strip()
            u = url_m.group(1).strip()
            out.append({"path": p, "url": u, "name": Path(p).name})
    return out

# ── 路径工具 ──────────────────────────────────────────────

def domain_of(path: str):
    """research/generative_vision/repos/X → 'generative_vision'，否则 None"""
    parts = Path(path).parts
    if len(parts) >= 2 and parts[0] == "research":
        return parts[1]
    return None

def display_name(path: str) -> str:
    """嵌套 submodule 显示相对路径（dino/dino），普通 repo 只显示最后一段。"""
    domain = domain_of(path)
    if domain:
        repos_prefix = f"research/{domain}/repos/"
        if path.startswith(repos_prefix):
            rel = path[len(repos_prefix):]
            if "/" in rel:
                return rel
    return Path(path).name

def gh_slug(url: str) -> str:
    u = url.rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    return "/".join(u.split("/")[-2:])

def gh_url(url: str) -> str:
    return url.rstrip("/")[:-4] if url.endswith(".git") else url.rstrip("/")

def build_structure_block(domains: dict) -> str:
    """从 domains.json 动态生成目录结构代码块。"""
    items = list(domains.items())
    lines = ["```", "ai_workspace/", "├── research/               研究参考代码（只读第三方仓库）"]
    for i, (d, zh) in enumerate(items):
        prefix = "└──" if i == len(items) - 1 else "├──"
        pad = " " * max(1, 18 - len(d))
        lines.append(f"│   {prefix} {d}/{pad}{zh}")
    lines += [
        "├── projects/               自研项目（fork 或原创）",
        "├── prompts/                提示词 / 思路文档",
        "├── dev_toolkit/            开发工具",
        "└── tools/                  工作区维护脚本",
        "```",
    ]
    return "\n".join(lines)

# ── Markdown 区块替换 ─────────────────────────────────────

def replace_between(text: str, begin: str, end: str, content: str) -> str:
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    replacement = f"{begin}\n{content}\n{end}"
    if pat.search(text):
        return pat.sub(replacement, text)
    return text

def inject_or_replace(text: str, begin: str, end: str,
                       after_heading: str, content: str) -> str:
    """
    若 begin 标记已存在 → replace_between。
    否则找 after_heading 后面的表格并包裹；找不到则在文末追加。
    """
    if begin in text:
        return replace_between(text, begin, end, content)

    pat = re.compile(
        r'(' + re.escape(after_heading) + r'[^\n]*\n\n)'
        r'(\|[^\n]*\n)(\|[-| :]+\n)((?:\|[^\n]*\n)*)',
        re.MULTILINE
    )
    new_text = pat.sub(lambda m: f"{m.group(1)}{begin}\n{content}\n{end}\n", text)
    if new_text == text:
        new_text = text.rstrip("\n") + f"\n\n{begin}\n{content}\n{end}\n"
    return new_text

# ── 表格构建 ──────────────────────────────────────────────

def build_domain_table(mods: list, domain: str, descs: dict) -> str:
    """domain README 三列表格。"""
    header = "| 仓库 | 来源 | 简介 |\n|------|------|------|"
    rows = []
    prefix = f"research/{domain}/"
    for m in sorted(mods, key=lambda x: display_name(x["path"]).lower()):
        if domain_of(m["path"]) != domain:
            continue
        rel   = m["path"][len(prefix):]
        dname = display_name(m["path"])
        slug  = gh_slug(m["url"])
        url   = gh_url(m["url"])
        desc  = descs.get(m["path"], display_name(m["path"]))
        rows.append(f"| [{dname}]({rel}) | [{slug}]({url}) | {desc} |")
    return header + "\n" + "\n".join(rows)

def build_research_section_table(mods: list, domain: str, descs: dict) -> str:
    """research/README.md 某 domain 节三列表格。"""
    header = "| 仓库 | 来源 | 简介 |\n|------|------|------|"
    rows = []
    prefix = "research/"
    for m in sorted(mods, key=lambda x: display_name(x["path"]).lower()):
        if domain_of(m["path"]) != domain:
            continue
        rel   = m["path"][len(prefix):]
        dname = display_name(m["path"])
        slug  = gh_slug(m["url"])
        url   = gh_url(m["url"])
        desc  = descs.get(m["path"], display_name(m["path"]))
        rows.append(f"| [{dname}]({rel}) | [{slug}]({url}) | {desc} |")
    return header + "\n" + "\n".join(rows)

def build_root_research_table(mods: list, domain: str, descs: dict) -> str:
    """根 README Research 节三列表格（本地路径 + GitHub 来源 + 简介）。"""
    header = "| 仓库 | 来源 | 简介 |\n|------|------|------|"
    rows = []
    for m in sorted(mods, key=lambda x: display_name(x["path"]).lower()):
        if domain_of(m["path"]) != domain:
            continue
        dname = display_name(m["path"])
        slug  = gh_slug(m["url"])
        url   = gh_url(m["url"])
        desc  = descs.get(m["path"], display_name(m["path"]))
        rows.append(f"| [{dname}]({m['path']}) | [{slug}]({url}) | {desc} |")
    return header + "\n" + "\n".join(rows)

def build_projects_table(mods: list, descs: dict) -> str:
    """projects 节三列表格。"""
    header = "| 项目 | 仓库 | 简介 |\n|------|------|------|"
    rows = []
    for m in sorted(mods, key=lambda x: x["name"].lower()):
        if not m["path"].startswith("projects/"):
            continue
        slug = gh_slug(m["url"])
        url  = gh_url(m["url"])
        desc = descs.get(m["path"], display_name(m["path"]))
        rows.append(f"| [{m['name']}]({m['path']}) | [{slug}]({url}) | {desc} |")
    return header + "\n" + "\n".join(rows)

# ── README 更新器 ─────────────────────────────────────────

def update_domain_readme(domain: str, mods: list, descs: dict):
    readme = RESEARCH / domain / "README.md"
    if not readme.exists():
        return
    text  = readme.read_text("utf-8")
    table = build_domain_table(mods, domain, descs)
    text  = inject_or_replace(text,
                               "<!-- BEGIN_REPOS -->", "<!-- END_REPOS -->",
                               "## 子仓库", table)
    readme.write_text(text, "utf-8")
    print(f"  ✔ research/{domain}/README.md")

def update_research_readme(mods: list, descs: dict):
    if not RES_README.exists():
        return
    domains = load_domains()
    text = RES_README.read_text("utf-8")
    for domain, zh in domains.items():
        table = build_research_section_table(mods, domain, descs)
        begin = f"<!-- BEGIN_DOMAIN:{domain} -->"
        end   = f"<!-- END_DOMAIN:{domain} -->"
        if begin in text:
            text = replace_between(text, begin, end, table)
        else:
            # 新分区：追加完整节
            new_section = (
                f"\n\n### {zh} → [{domain}/]({domain}/README.md)\n\n"
                f"{begin}\n{table}\n{end}\n"
            )
            text = text.rstrip("\n") + new_section
    RES_README.write_text(text, "utf-8")
    print("  ✔ research/README.md")

def update_root_readme(mods: list, descs: dict):
    if not ROOT_README.exists():
        return
    domains = load_domains()
    text = ROOT_README.read_text("utf-8")

    # Projects 节
    ptable = build_projects_table(mods, descs)
    text = inject_or_replace(text,
                              "<!-- BEGIN_PROJECTS -->", "<!-- END_PROJECTS -->",
                              "## Projects", ptable)

    # Research 汇总节（完整重建）
    parts = []
    for domain, zh in domains.items():
        t = build_root_research_table(mods, domain, descs)
        if "| [" in t:
            parts.append(
                f"### {zh}\n\n"
                f"[→ 详情](research/{domain}/README.md)\n\n{t}"
            )
    summary = "\n\n---\n\n".join(parts)
    text = inject_or_replace(text,
                              "<!-- BEGIN_RESEARCH -->", "<!-- END_RESEARCH -->",
                              "## Research 仓库", summary)

    # 目录 TOC（完整重建）
    toc_lines = [
        "- [Projects](#projects)",
        "- [Research 仓库](#research-仓库)",
    ]
    for zh in domains.values():
        # GitHub anchor: 小写，空格→-，保留中文
        anchor = zh.lower().replace(" ", "-")
        toc_lines.append(f"  - [{zh}](#{anchor})")
    toc_lines += [
        "- [目录结构](#目录结构)",
        "- [仓库管理命令](#仓库管理命令)",
    ]
    toc = "\n".join(toc_lines)
    text = inject_or_replace(text,
                              "<!-- BEGIN_TOC -->", "<!-- END_TOC -->",
                              "## 目录", toc)

    # 目录结构（从 domains.json 动态生成）
    struct = build_structure_block(domains)
    text = inject_or_replace(text,
                              "<!-- BEGIN_STRUCTURE -->", "<!-- END_STRUCTURE -->",
                              "## 目录结构", struct)

    ROOT_README.write_text(text, "utf-8")
    print("  ✔ README.md")

# ── 从现有 README 提取描述 ────────────────────────────────

def extract_descs_from_domain_readme(domain: str) -> dict:
    readme = RESEARCH / domain / "README.md"
    if not readme.exists():
        return {}
    text  = readme.read_text("utf-8")
    descs = {}
    for m in re.finditer(
        r'\|\s*\[[^\]]+\]\(([^)]+)\)\s*\|\s*\[[^\]]+\]\([^)]+\)\s*\|\s*([^|\n]+?)\s*\|',
        text
    ):
        rel_path = m.group(1).strip()
        desc     = m.group(2).strip()
        abs_path = f"research/{domain}/{rel_path}"
        if desc:
            descs[abs_path] = desc
    return descs

# ── 命令 ──────────────────────────────────────────────────

def _update_all(mods: list, descs: dict, changed_path: str):
    domains = load_domains()
    domain = domain_of(changed_path)
    if domain and domain in domains:
        update_domain_readme(domain, mods, descs)
        update_research_readme(mods, descs)
    update_root_readme(mods, descs)

def cmd_add(path: str, url: str, desc: str):
    descs = load_descs()
    if desc:
        descs[path] = desc
    save_descs(descs)
    _update_all(parse_gitmodules(), descs, path)
    print("✅ README 已更新")

def cmd_rm(path: str):
    descs = load_descs()
    descs.pop(path, None)
    save_descs(descs)
    _update_all(parse_gitmodules(), descs, path)
    print("✅ README 已更新")

def cmd_rebuild():
    """从 .gitmodules 完全重建所有 README。"""
    domains = load_domains()
    descs = load_descs()
    for domain in domains:
        for k, v in extract_descs_from_domain_readme(domain).items():
            if k not in descs:
                descs[k] = v
    save_descs(descs)

    mods = parse_gitmodules()
    print(f"找到 {len(mods)} 个 submodule，开始重建 README...")
    for domain in domains:
        update_domain_readme(domain, mods, descs)
    update_research_readme(mods, descs)
    update_root_readme(mods, descs)
    print("✅ 所有 README 已重建")

def cmd_new_domain(domain: str, zh_name: str):
    """创建新 research 分区并更新所有 README。"""
    domains = load_domains()
    if domain in domains:
        sys.exit(f"❌ 分区 '{domain}' 已存在")

    domain_dir = RESEARCH / domain
    repos_dir  = domain_dir / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    # 创建 README（带 markers）
    readme = domain_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {zh_name} 汇总仓库\n\n---\n\n"
            f"## 子仓库\n\n"
            f"<!-- BEGIN_REPOS -->\n"
            f"| 仓库 | 来源 | 简介 |\n|------|------|------|\n"
            f"<!-- END_REPOS -->\n",
            "utf-8"
        )
        print(f"  ✔ research/{domain}/README.md（新建）")

    # 更新 domains.json
    domains[domain] = zh_name
    save_domains(domains)
    print(f"  ✔ tools/domains.json")

    # 更新各 README
    mods  = parse_gitmodules()
    descs = load_descs()
    update_research_readme(mods, descs)
    update_root_readme(mods, descs)
    print(f"✅ 分区 '{domain}'（{zh_name}）已创建")
    print(f"   接下来用: ./manage_repos.sh add research/{domain}/repos/<仓库> <URL> '描述'")

# ── 入口 ──────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    if cmd == "add":
        if len(args) < 3:
            sys.exit("用法: update_readme.py add <path> <url> [描述]")
        cmd_add(args[1], args[2], args[3] if len(args) > 3 else "")
    elif cmd == "rm":
        if len(args) < 2:
            sys.exit("用法: update_readme.py rm <path>")
        cmd_rm(args[1])
    elif cmd == "rebuild":
        cmd_rebuild()
    elif cmd == "new-domain":
        if len(args) < 3:
            sys.exit("用法: update_readme.py new-domain <domain> <中文名>")
        cmd_new_domain(args[1], args[2])
    else:
        sys.exit(f"未知命令: {cmd}")

if __name__ == "__main__":
    main()
