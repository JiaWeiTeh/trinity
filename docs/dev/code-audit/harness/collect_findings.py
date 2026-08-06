"""Normalise every finding from every audit report into one CSV.

Reports were written across several sessions and use three formats: the fenced
```json array (Phase 2 reconcilers, Phase 3 sweeps 1/2/7/9), a markdown table
(S11), and `### ID` headings with a `- **severity** —` bullet (the earlier
sweeps). A parser that understood only the first silently dropped 7 of 27
sources, which is how a phase gets called complete when it isn't.

    python docs/dev/code-audit/harness/collect_findings.py

Writes data/findings_inventory.csv and prints per-source counts. Any source that
yields zero findings is reported as a PROBLEM, not skipped quietly.
"""

import csv
import json
import pathlib
import re
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
SLICES = ROOT / "slices"
OUT = ROOT / "data" / "findings_inventory.csv"

FIELDS = [
    "id",
    "severity",
    "current_severity",
    "file",
    "line",
    "class",
    "claim",
    "confidence",
    "source",
    "revision",
]
SEV = re.compile(r"\bS([1-4])\b")
REVISIONS = ROOT / "data" / "revisions.csv"


def _from_json(text):
    """Phase 2 reconcilers + Phase 3 sweeps 1/2/7/9."""
    for block in re.findall(r"```json\s*(\[.*?\])\s*```", text, re.DOTALL):
        try:
            yield from json.loads(block)
        except json.JSONDecodeError:
            continue


def _from_table(text):
    """S11-style: | ID | S1 | axis | file:line | claim | status | confidence |"""
    for row in re.findall(r"^\|\s*([A-Z][\w-]*-R?-?\d+)\s*\|(.+)$", text, re.MULTILINE):
        fid, rest = row
        cells = [c.strip() for c in rest.split("|")]
        sev = SEV.search(cells[0]) if cells else None
        if not sev:
            continue
        yield {
            "id": fid,
            "severity": f"S{sev.group(1)}",
            "file": next((c for c in cells if "." in c and ":" in c), ""),
            "claim": max(cells, key=len) if cells else "",
        }


def _from_headings(text):
    """Earlier sweeps: `### ST-001 · claim` then `- **severity** — S1 ...`"""
    chunks = re.split(r"^### +", text, flags=re.MULTILINE)[1:]
    for chunk in chunks:
        head = chunk.split("\n", 1)[0]
        fid = head.split(None, 1)[0].strip()
        m = re.search(r"\*\*severity\*\*\s*[—-]+\s*\**S([1-4])", chunk)
        if not m:
            continue
        yield {
            "id": fid,
            "severity": f"S{m.group(1)}",
            "file": (re.search(r"`([\w/]+\.py):(\d+)`", chunk) or [""])[0].strip("`"),
            "claim": head.split("·", 1)[-1].strip() if "·" in head else head,
        }


# Phase 0e injected 8 synthetic defects into a scratch copy to measure the
# pipeline's detection rate. Its report is written in the findings format but
# describes bugs that were never in trinity/ — including them would inject
# fabricated S1s into the inventory the audit ships.
NOT_FINDINGS = {"calibration_reconciled.md"}


def main():
    sources = sorted(
        p
        for p in set(SLICES.glob("*_reconciled.md")) | set(SLICES.glob("sweep_*.md"))
        if p.name not in NOT_FINDINGS
    )
    rows, problems = [], []
    for path in sources:
        text = path.read_text()
        found = list(_from_json(text)) or list(_from_table(text)) or list(_from_headings(text))
        if not found:
            problems.append(path.name)
            continue
        for item in found:
            rows.append({k: str(item.get(k, ""))[:400] for k in FIELDS} | {"source": path.name})

    # Overlay data/revisions.csv: a finding's *current* severity is what the
    # orchestrator's lookups left it at, not what its finder first rated it.
    # Shipping birth severities would put two already-CLEARED findings into
    # FINDINGS.md as S1.
    revised = {}
    if REVISIONS.exists():
        with REVISIONS.open(encoding="utf-8") as fh:
            revised = {r["id"]: r for r in csv.DictReader(fh)}
    for row in rows:
        rev = revised.get(row["id"])
        row["current_severity"] = rev["current_severity"] if rev else row["severity"]
        row["revision"] = rev["note"][:200] if rev else ""
    # A revision id with no slice-report finding is not an error: Phase 6 raised
    # findings (P6-nn) that exist only in the register, because they came from
    # running the code rather than reading it. Carry them as first-class rows so
    # the counts include them.
    standalone = sorted(set(revised) - {r["id"] for r in rows})
    for fid in standalone:
        rev = revised[fid]
        rows.append(
            {k: "" for k in FIELDS}
            | {
                "id": fid,
                "severity": rev["current_severity"],
                "current_severity": rev["current_severity"],
                "claim": rev["note"][:400],
                "revision": rev["note"][:200],
                "source": rev.get("resolution", "revisions.csv"),
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    born = Counter(r["severity"] for r in rows)
    now = Counter(r["current_severity"] for r in rows)
    print(f"{len(rows)} findings from {len(sources) - len(problems)}/{len(sources)} sources")
    print("as first rated:", {k: born[k] for k in sorted(born)})
    print("after revision:", {k: now[k] for k in sorted(now)})
    print(f"-> {OUT.relative_to(ROOT.parents[2])}")
    if standalone:
        print(f"register-only findings carried in (Phase 6 etc.): {standalone}")
    if problems:
        print(f"\nPROBLEM - no findings parsed from {len(problems)} source(s):")
        for name in problems:
            print(f"  {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
