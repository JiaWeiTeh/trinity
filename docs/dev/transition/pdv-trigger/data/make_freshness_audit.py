#!/usr/bin/env python3
"""Freshness audit — which committed artifact is from WHEN, read off its own provenance stamp.

Maintainer ruling 2026-07-29 (the ALL-FRESH re-run): every number the f_kappa re-open conclusions
rest on must come from arms run today, not from a CSV of an earlier campaign. That ruling is only
enforceable if "when was this measured" is checkable per file, so this walks every committed CSV
under data/ and runs/data/, reads the `# generated <ISO8601> | builder <x> | code <sha>` first line
(the _stamp.py contract), and classifies it against a cutoff.

    python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py               # today's cutoff
    python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py 2026-07-29    # explicit

Deliverable: data/freshness_audit.csv + a console roll-up. Exit code is always 0 — this REPORTS,
it does not gate. Two honest limits, both visible in the output:

  * `unstamped` is not the same as `old`. Some artifacts predate _stamp.py or are hand-made (params,
    HPC harvests copied by hand); they get status `UNSTAMPED` and a git-commit date instead, which
    only UPPER-bounds their age — see the _stamp.py docstring for why that distinction matters.
  * The `code`/`tree_dirty` columns are reported but carry NO verdict here, deliberately. `+dirty`
    is unavoidable noise — `_stamp.py` reads `git status` from inside the already-open output file,
    so every in-place regeneration records it even from a spotless checkout. And an artifact always
    records the commit BEFORE the one that commits it, so "is this at HEAD" is red for correct work
    too. **The real staleness question — did the BUILDER change after its output? — is `MANIFEST.md`'s
    ⚠️ STALE-RISK flag** (`python make_manifest.py`), which compares the two properly. This script
    answers only the question the ALL-FRESH ruling asks: *when was this measured?*
"""

import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
ROOTS = [HERE, PDV / "runs" / "data"]
STAMP_RE = re.compile(r"^#\s*generated\s+(\S+)\s*\|\s*builder\s+(\S+)\s*\|\s*code\s+(\S+)")


def _git_date(path):
    r = subprocess.run(
        ["git", "-C", str(PDV), "log", "-1", "--format=%cs", "--", str(path)],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip() or ""


def audit(path, cutoff):
    """(generated_date, builder, code, status) for one artifact."""
    try:
        with path.open(errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return "", "", "", "UNREADABLE"
    m = STAMP_RE.match(first.strip())
    if not m:
        return _git_date(path), "", "", "UNSTAMPED"
    when, builder, code = m.groups()
    day = when[:10]
    return day, builder, code, ("FRESH" if day >= cutoff else "OLD")


def main(argv):
    cutoff = argv[0] if argv else dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    rows = []
    for root in ROOTS:
        for p in sorted(root.rglob("*.csv")):
            day, builder, code, status = audit(p, cutoff)
            rows.append(
                {
                    "artifact": str(p.relative_to(PDV)),
                    "generated": day,
                    "builder": builder,
                    "code": code,
                    "status": status,
                    "tree_dirty": "" if not code else ("yes" if code.endswith("+dirty") else "no"),
                }
            )

    out = HERE / "freshness_audit.csv"
    with out.open("w", newline="") as fh:
        sys.path.insert(0, str(PDV))
        from _stamp import stamp

        fh.write(stamp(__file__) + "\n")
        fh.write(
            f"# freshness audit, cutoff {cutoff}: FRESH = the artifact's own generation stamp is on "
            "or after the cutoff; OLD = before it; UNSTAMPED = no stamp line, so the date shown is "
            "the git COMMIT date, which only UPPER-bounds the artifact's age (see _stamp.py).\n"
            "# The code/tree_dirty columns are informational ONLY. +dirty is unavoidable (_stamp.py "
            "reads git status from inside the already-open output file, so every in-place "
            "regeneration records it), and an artifact always names the commit BEFORE the one that "
            "commits it. For real staleness -- did the BUILDER change after its output? -- read "
            "MANIFEST.md's STALE-RISK flag instead.\n"
            "# Regenerate: python docs/dev/transition/pdv-trigger/data/make_freshness_audit.py "
            "[YYYY-MM-DD]\n"
        )
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tally = {}
    for r in rows:
        tally[r["status"]] = tally.get(r["status"], 0) + 1
    print(f"cutoff {cutoff} — " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    fresh = [r for r in rows if r["status"] == "FRESH"]
    if fresh:
        print(f"\nFRESH (generated on/after {cutoff}):")
        for r in fresh:
            print(f"  {r['generated']}  {r['artifact']:<58s} {r['builder']}")
    print(
        "\n(For 'did the builder change after its output?' read MANIFEST.md's STALE-RISK flag — "
        "this script answers only 'when was this measured?'.)"
    )
    print(f"\nwrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
