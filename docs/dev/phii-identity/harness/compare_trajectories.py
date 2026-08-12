#!/usr/bin/env python3
"""Compare two arms' trajectories at matched simulation time — PLAN.md §5 bar.

Runs truncate at different `t` (different fates, different wall-clock budgets),
so comparing snapshot-by-snapshot silently compares different instants. This
interpolates both arms onto a shared log-spaced `t` grid over their **overlap**
and reports the deviation there, plus the stopping fate from each arm's
`metadata.json` (PLAN contamination rule C-5).

Reported per config:
  * `dR2_max_pct` / `dR2_end_pct` — the trajectory bar (attention at >5%)
  * `t_overlap_*`                 — the window actually compared; a small overlap
                                    means the arms diverged in *duration*, which
                                    the percentage alone would hide
  * `fate_base` / `fate_new`      — enumerated, never silently passed. Under a
                                    candidate fix a fate flip may be the fix
                                    working; PLAN decision D3 owns that call.

Usage (from the repo root):
    python docs/dev/phii-identity/harness/compare_trajectories.py \
        --base outputs/phii/b0__<sha> --new <worktree>/outputs/phii/b4a__<sha> \
        --label b4a --out docs/dev/phii-identity/data/b4a_ledger.csv

Exit status is 1 if any config breaches the bar or changes fate, so it can gate.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402

BAR_PCT = 5.0  # PLAN §5: the G2 bar phase1a-init adopted 2026-08-05
N_GRID = 400

COLS = [
    "config",
    "verdict",
    "dR2_max_pct",
    "dR2_end_pct",
    "t_at_max",
    "t_overlap_lo",
    "t_overlap_hi",
    "n_base",
    "n_new",
    "R2_end_base",
    "R2_end_new",
    "fate_base",
    "fate_new",
    "note",
]


def series(run_dir):
    """(t, R2) sorted and strictly increasing in t, plus the fate."""
    path = run_dir / "dictionary.jsonl"
    if not path.exists():
        return None, None, None
    pts = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t, r = d.get("t_now"), d.get("R2")
            if isinstance(t, (int, float)) and isinstance(r, (int, float)):
                if math.isfinite(t) and math.isfinite(r) and r > 0:
                    pts.append((float(t), float(r)))
    if not pts:
        return None, None, None
    pts.sort()
    ts, rs = [], []
    for t, r in pts:  # drop duplicate/non-increasing t so interpolation is well posed
        if not ts or t > ts[-1]:
            ts.append(t)
            rs.append(r)
    fate = "NA"
    meta = run_dir / "metadata.json"
    if meta.exists():
        try:
            term = json.loads(meta.read_text()).get("termination") or {}
            fate = term.get("outcome") or term.get("reason") or "NA"
        except ValueError:
            pass
    return ts, rs, fate


def interp(ts, rs, t):
    """Linear interpolation in log t / log R2 — both span decades."""
    if t <= ts[0]:
        return rs[0]
    if t >= ts[-1]:
        return rs[-1]
    lo, hi = 0, len(ts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    t0, t1, r0, r1 = ts[lo], ts[hi], rs[lo], rs[hi]
    if t1 <= t0:
        return r0
    f = (
        (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0))
        if t0 > 0
        else (t - t0) / (t1 - t0)
    )
    return math.exp(math.log(r0) + f * (math.log(r1) - math.log(r0)))


def compare(base_dir, new_dir):
    bt, br, bfate = series(base_dir)
    nt, nr, nfate = series(new_dir)
    if bt is None or nt is None:
        return {
            "config": base_dir.name,
            "verdict": "SKIP",
            "note": "missing run",
            "fate_base": bfate or "NA",
            "fate_new": nfate or "NA",
        }

    lo, hi = max(bt[0], nt[0]), min(bt[-1], nt[-1])
    if not (hi > lo > 0):
        return {
            "config": base_dir.name,
            "verdict": "NO-OVERLAP",
            "note": f"base t in [{bt[0]:.3g},{bt[-1]:.3g}], new t in [{nt[0]:.3g},{nt[-1]:.3g}]",
            "n_base": len(bt),
            "n_new": len(nt),
            "fate_base": bfate,
            "fate_new": nfate,
        }

    worst, t_worst = 0.0, lo
    step = (math.log(hi) - math.log(lo)) / (N_GRID - 1)
    for i in range(N_GRID):
        t = math.exp(math.log(lo) + i * step)
        b, n = interp(bt, br, t), interp(nt, nr, t)
        if b > 0:
            d = abs(n - b) / b * 100.0
            if d > worst:
                worst, t_worst = d, t
    b_end, n_end = interp(bt, br, hi), interp(nt, nr, hi)
    d_end = abs(n_end - b_end) / b_end * 100.0 if b_end > 0 else float("nan")

    fate_changed = bfate != nfate
    verdict = "FATE-CHANGE" if fate_changed else ("OVER-BAR" if worst > BAR_PCT else "WITHIN-BAR")
    return {
        "config": base_dir.name,
        "verdict": verdict,
        "dR2_max_pct": f"{worst:.3f}",
        "dR2_end_pct": f"{d_end:.3f}",
        "t_at_max": f"{t_worst:.6g}",
        "t_overlap_lo": f"{lo:.6g}",
        "t_overlap_hi": f"{hi:.6g}",
        "n_base": len(bt),
        "n_new": len(nt),
        "R2_end_base": f"{b_end:.6g}",
        "R2_end_new": f"{n_end:.6g}",
        "fate_base": bfate,
        "fate_new": nfate,
        "note": "fate differs" if fate_changed else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    configs = sorted(p.name for p in args.new.iterdir() if p.is_dir()) if args.new.is_dir() else []
    if not configs:
        sys.exit(f"no config dirs under {args.new}")

    rows = [compare(args.base / c, args.new / c) for c in configs]
    w = max(len(c) for c in configs)
    print(
        f"{'config':{w}}  {'verdict':>12} {'dR2_max%':>9} {'dR2_end%':>9}  "
        f"{'overlap t':>22}  fate base -> new"
    )
    for r in rows:
        print(
            f"{r['config']:{w}}  {r.get('verdict',''):>12} {r.get('dR2_max_pct','-'):>9} "
            f"{r.get('dR2_end_pct','-'):>9}  "
            f"{r.get('t_overlap_lo','-')+' .. '+r.get('t_overlap_hi','-'):>22}  "
            f"{r.get('fate_base','?')} -> {r.get('fate_new','?')}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            fh.write(f"# arm={args.label}  base={args.base}  new={args.new}\n")
            fh.write(f"# Matched-t comparison on the overlap window; bar |dR2| > {BAR_PCT}%.\n")
            fh.write("# Fate changes are enumerated, never silently passed (PLAN D3).\n")
            wr = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.out}")
    bad = [r for r in rows if r.get("verdict") in ("OVER-BAR", "FATE-CHANGE", "NO-OVERLAP")]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
