"""Mechanically extract every checkable claim in trinity/ into CSVs.

The audit's closed checklist: attention drift is the main way a hand review
misses things, so every prose assertion, numeric literal, failure-swallowing
guard, and declared-vs-consumed parameter is enumerated here and must end the
audit with a verdict.

    python docs/dev/code-audit/harness/extract_claims.py trinity docs/dev/code-audit/data
"""

import ast
import csv
import pathlib
import re
import sys
import tokenize

# A prose line is "checkable" if it asserts something the code can contradict.
CITATION = re.compile(r"\b(eq\.?|equation|et al\.?|\+\d{2}\b|arxiv|doi|thesis|paper|sect)", re.I)
UNITS = re.compile(
    r"\[[^\]]*\]|\b(pc|kpc|cm|myr|yr|s|km/s|msun|m_sun|g|erg|k|cgs|au|dyne|kelvin)\b", re.I
)
FORMULA = re.compile(r"[=*/^]|\*\*|\bproportional\b|\bscales? (as|with)\b", re.I)
GUARDS = re.compile(
    r"\bnp\.clip\b|\.clip\(|nan_to_num|np\.maximum\(|np\.minimum\(|fillna|"
    r"np\.nan\b|float\(['\"]nan|isfinite|isnan|\bmax\(0|\babs\(",
)


def rows_prose(path, src):
    """Every comment and docstring, with flags for what kind of claim it makes."""
    out = []
    lines = src.splitlines()

    readline = iter(src.splitlines(keepends=True)).__next__
    for tok in tokenize.generate_tokens(readline):
        if tok.type == tokenize.COMMENT:
            out.append(("comment", tok.start[0], tok.start[0], "", tok.string.strip()))

    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is None:
            continue
        expr = node.body[0].value
        owner = getattr(node, "name", "<module>")
        out.append(("docstring", expr.lineno, expr.end_lineno, owner, doc))

    return [
        {
            "file": path,
            "kind": kind,
            "line": lo,
            "end_line": hi,
            "owner": owner,
            "cites": bool(CITATION.search(text)),
            "units": bool(UNITS.search(text)),
            "formula": bool(FORMULA.search(text)),
            # Full text, not a preview: Lens B sees only this, so a truncated
            # docstring would hide claims (caught in the Phase 0e calibration).
            "text": " ".join(text.split()),
            "code_at_line": " ".join(lines[hi : hi + 3][:1])[:200] if hi < len(lines) else "",
            "verdict": "",
        }
        for kind, lo, hi, owner, text in sorted(out, key=lambda r: r[1])
    ]


def rows_literals(path, src):
    """Numeric literals sitting inside arithmetic — magic-number candidates."""
    lines = src.splitlines()
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        if not isinstance(node.value, (int, float)):
            continue
        line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        arithmetic = bool(re.search(r"[-+*/]|\*\*", line))
        if node.value in (0, 1, 2, -1) and not arithmetic:
            continue
        out.append(
            {
                "file": path,
                "line": node.lineno,
                "value": node.value,
                "in_arithmetic": arithmetic,
                "source": line.strip()[:200],
                "verdict": "",
            }
        )
    return out


def rows_guards(path, src):
    """try/except handlers and numeric clamps — places a failure can be swallowed."""
    lines = src.splitlines()
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ExceptHandler):
            caught = ast.unparse(node.type) if node.type else "BARE"
            body = ast.unparse(node.body[0])[:120] if node.body else ""
            out.append(
                {
                    "file": path,
                    "line": node.lineno,
                    "kind": "except",
                    "detail": f"catches {caught} -> {body}",
                    "source": lines[node.lineno - 1].strip()[:200],
                    "verdict": "",
                }
            )
    for i, line in enumerate(lines, 1):
        if GUARDS.search(line) and not line.strip().startswith("#"):
            out.append(
                {
                    "file": path,
                    "line": i,
                    "kind": "clamp",
                    "detail": GUARDS.search(line).group(0),
                    "source": line.strip()[:200],
                    "verdict": "",
                }
            )
    return out


def rows_params(root, default_param):
    """Every schema key, and whether anything in the package actually reads it."""
    keys = [
        line.split()[0]
        for line in default_param.read_text().splitlines()
        if line.strip() and not line.startswith("#") and len(line.split()) >= 1
    ]
    sources = {p: p.read_text() for p in root.rglob("*.py")}
    out = []
    for key in sorted(set(keys)):
        hits = [
            f"{p.relative_to(root.parent)}:{i}"
            for p, text in sources.items()
            for i, line in enumerate(text.splitlines(), 1)
            if key in line
        ]
        out.append(
            {
                "key": key,
                "n_refs": len(hits),
                "refs": " ".join(hits[:8]),
                "verdict": "",
            }
        )
    return out


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows):6d}  {path}")


def main(src_root, out_dir):
    root, out = pathlib.Path(src_root), pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prose, literals, guards = [], [], []
    for p in sorted(root.rglob("*.py")):
        rel, src = str(p), p.read_text()
        prose += rows_prose(rel, src)
        literals += rows_literals(rel, src)
        guards += rows_guards(rel, src)
    write(out / "claims_prose.csv", prose)
    write(out / "claims_literals.csv", literals)
    write(out / "claims_guards.csv", guards)
    write(out / "claims_params.csv", rows_params(root, root / "_input" / "default.param"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
