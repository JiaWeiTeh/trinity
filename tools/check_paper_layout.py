"""List what in a paper/<project>/ folder does not follow the folder standard (paper/CLAUDE.md).

    python tools/check_paper_layout.py                 # every project folder under paper/
    python tools/check_paper_layout.py rosette pedrini # some of them
    python tools/check_paper_layout.py --root DIR      # another paper/ folder (used by the test)

It only reads. Per project it reports:
  - a reorganisation marker (REORG_IN_PROGRESS.md) in the root;
  - loose files in the root (only PLAN.md and sync.sh belong there; README.md in a light folder
    or a closed project without PLAN.md);
  - root folders whose name is not one of the standard's, synonyms named as such;
  - paths `paper/<project>/...` named in PLAN.md, README.md or prompts/PROMPT_continue*.md that
    do not exist (the stale paths a new session would follow).
Exit status 1 if anything was reported. stdlib only; the paper/ folders are local-only, so a
missing one is skipped.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT_FILES = {"PLAN.md", "sync.sh"}
FOLDERS = {
    "prompts",
    "log",
    "draft",
    "plots",
    "plotting_scripts",
    "harness",
    "data",
    "param",
    "hpc",
    "sps",
    "notes",
    "external",
    "deliveries",
    "archive",
    "to-be-deleted",
}
SYNONYMS = {
    "figures": "plotting_scripts/ (scripts), plots/ or draft/fig/ (figures)",
    "figs": "plots/ or draft/fig/",
    "figure": "draft/fig/",
    "fig": "draft/fig/ or plots/",
    "params": "param/",
    "chapter": "log/",
    "to-be-removed": "to-be-deleted/",
    "_to_delete": "to-be-deleted/",
    "runs": "data/<campaign>/big_runs/",
    "out": "plots/ or data/",
    "src": "plotting_scripts/ or harness/",
    "paper-plans": "draft/",
    "repro": "harness/",
    "progress-updates": "deliveries/",
}
LIGHT = {
    "proposal",
    "hamed",
    "pineda",
    "janice",
    "website",
    "pedrini",
}  # not papers: README.md + src/ data/ out/
SKIP = {
    "_lib",
    "methods",
    "barnes26",
    "_structure",
    "__pycache__",
}  # tracked or out of scope (ruling e)
# Recorded exceptions (each named in that project's PLAN.md)
EXCEPTIONS = {"trinity-phangs-I": {"digest_catalogue.py"}}
IGNORED = {".DS_Store", ".claude", ".ruff_cache", "__pycache__", ".pytest_cache"}
PATH_TOKEN = r"paper/{name}/[A-Za-z0-9_./+-]+"


def check(proj):
    name, out = proj.name, []
    if (proj / "REORG_IN_PROGRESS.md").exists():
        out.append(
            "REORG_IN_PROGRESS.md: a reorganisation is running or did not finish; do not write here"
        )
    light = name in LIGHT  # light folders keep their own README convention: staging names only
    allowed_files = ROOT_FILES | EXCEPTIONS.get(name, set())
    if not (proj / "PLAN.md").exists():  # a closed project may keep README.md as its entry file
        allowed_files |= {"README.md"}
    for entry in sorted(proj.iterdir()):
        n = entry.name
        if n in IGNORED or n == "REORG_IN_PROGRESS.md":
            continue
        if entry.is_dir() and not entry.is_symlink():
            if light:
                if n in {"to-be-removed", "_to_delete"}:
                    out.append(f"{n}/: a synonym; the standard name is to-be-deleted/")
            elif n in SYNONYMS:
                out.append(f"{n}/: a synonym; the standard name is {SYNONYMS[n]}")
            elif n.startswith("plots_"):
                out.append(
                    f"{n}/: one folder per campaign goes under data/ or plots/, not plots_*/"
                )
            elif n not in FOLDERS:
                out.append(f"{n}/: not a standard folder")
        elif not light and n not in allowed_files:
            out.append(f"{n}: a loose file in the project root")
    texts = [proj / "PLAN.md", proj / "README.md"]
    texts += sorted((proj / "prompts").glob("PROMPT_continue*.md"))
    pattern = re.compile(r"(>\s*)?(" + PATH_TOKEN.format(name=re.escape(name)) + ")")
    for text in texts:
        if not text.is_file() or text.name.endswith(".prev.md"):
            continue
        tokens = set()
        for m in pattern.finditer(text.read_text(errors="replace")):
            if not m.group(1):  # a `> path` redirect target is made by the command itself
                tokens.add(m.group(2).rstrip(".,;:"))
        for token in sorted(tokens):
            if not (proj.parent.parent / token).exists():
                out.append(f"{text.relative_to(proj)} names {token}, which does not exist")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("projects", nargs="*", help="project folder names (default: all)")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "paper")
    args = ap.parse_args(argv)
    names = args.projects or sorted(
        p.name
        for p in args.root.iterdir()
        if p.is_dir() and p.name not in SKIP and not p.name.startswith(".")
    )
    found = 0
    for name in names:
        proj = args.root / name
        if not proj.is_dir():
            print(f"{name}: not here (the paper/ project folders are local-only)")
            continue
        issues = check(proj)
        found += len(issues)
        print(f"{name}: {'ok' if not issues else f'{len(issues)} to look at'}")
        for issue in issues:
            print(f"  - {issue}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
