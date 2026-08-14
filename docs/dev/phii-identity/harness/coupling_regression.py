#!/usr/bin/env python3
"""Quantify how tightly `n_IF_Str` is tied to `Pb` — PLAN.md §3b.

The workstream's central question is whether `P_HII` can be decoupled from the
confining pressure. The cap (`n_IF_Str = min(n_IF_Str, shell_n0)`) is the obvious
suspect, but it is only the last link: `shell_n0` IS `Pb/(k T)·μ` and it is the
shell ODE's inner boundary condition, so it also sets where the ionisation front
lands and hence the ionised volume `ΔV` that the Strömgren balance divides by.

This measures both links on the **pre-cap** value `n_IF_Str_raw`, i.e. what an
uncapped `P_HII` would have been:

    log ΔV          vs log shell_n0   ->  slope should be strongly negative
    log n_IF_Str_raw vs log shell_n0  ->  slope ~ -0.5 x the above, since
                                          n_IF_Str ∝ ΔV^(-1/2) by construction

Per-config slopes are reported alongside the pooled fit, because a pooled
exponent across configs is a between-config average and can look tighter or
looser than any individual run — the spread is the honest error bar.

Only runs that reached a terminal state are used: an arm killed by a wall-clock
timeout or a container restart is a truncated sample of the same trajectory, and
pooling it with its own re-run double-counts those rows (PLAN rule C-7).

Usage (from the repo root):
    python docs/dev/phii-identity/harness/coupling_regression.py \
        --out docs/dev/phii-identity/data/b3b_coupling_regression.csv \
        outputs/phii/b1__<sha>/<config> [...]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stamp import stamp  # noqa: E402

COLS = [
    "scope",
    "run",
    "n_rows",
    "shell_n0_span_dex",
    "slope_dV_vs_n0",
    "r_dV_vs_n0",
    "slope_nIFraw_vs_n0",
    "r_nIFraw_vs_n0",
    "ratio_min",
    "ratio_med",
    "ratio_max",
    "frac_ratio_lt_1",
]


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None, None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None, None
    return sxy / sxx, sxy / math.sqrt(sxx * syy)


def load(run_dir):
    """(log n0, log dV, log raw, ratio) per usable row, plus terminal-state flag."""
    meta = run_dir / "metadata.json"
    fate = None
    if meta.exists():
        try:
            term = json.loads(meta.read_text()).get("termination") or {}
            fate = term.get("outcome") or term.get("reason")
        except ValueError:
            pass
    rows = []
    path = run_dir / "dictionary.jsonl"
    if not path.exists():
        return rows, fate
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        n0, raw, RIF, R2 = (d.get("shell_n0"), d.get("n_IF_Str_raw"), d.get("R_IF"), d.get("R2"))
        if not (n0 and raw and RIF and R2) or n0 <= 0 or raw <= 0 or RIF <= R2 <= 0:
            continue
        dV = RIF**3 - R2**3
        if dV <= 0:
            continue
        rows.append((math.log10(n0), math.log10(dV), math.log10(raw), raw / n0))
    return rows, fate


def summarise(scope, name, rows):
    n0s = [r[0] for r in rows]
    s_dv, r_dv = fit(n0s, [r[1] for r in rows])
    s_raw, r_raw = fit(n0s, [r[2] for r in rows])
    ratios = sorted(r[3] for r in rows)
    f = lambda v, p=".4g": "NA" if v is None else format(v, p)  # noqa: E731
    return {
        "scope": scope,
        "run": name,
        "n_rows": len(rows),
        "shell_n0_span_dex": f"{max(n0s) - min(n0s):.2f}",
        "slope_dV_vs_n0": f(s_dv),
        "r_dV_vs_n0": f(r_dv),
        "slope_nIFraw_vs_n0": f(s_raw),
        "r_nIFraw_vs_n0": f(r_raw),
        "ratio_min": f"{ratios[0]:.4g}",
        "ratio_med": f"{ratios[len(ratios) // 2]:.4g}",
        "ratio_max": f"{ratios[-1]:.4g}",
        "frac_ratio_lt_1": f"{sum(1 for x in ratios if x < 1) / len(ratios):.4f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--include-unfinished",
        action="store_true",
        help="also pool runs with no terminal state (violates C-7; off by default)",
    )
    args = ap.parse_args()

    out, pooled, skipped = [], [], []
    for run_dir in args.runs:
        rows, fate = load(run_dir)
        if not rows:
            skipped.append(f"{run_dir.name}: no usable rows")
            continue
        if not fate and not args.include_unfinished:
            skipped.append(f"{run_dir.parent.name}/{run_dir.name}: no terminal state — excluded")
            continue
        label = f"{run_dir.parent.name}/{run_dir.name}"
        out.append(summarise("per-config", label, rows))
        pooled.extend(rows)
    if not pooled:
        print("no usable runs")
        return 1
    out.insert(0, summarise("pooled", f"{len(out)} complete runs", pooled))

    for s in skipped:
        print(f"  [skip] {s}")
    w = max(len(r["run"]) for r in out)
    print(
        f"\n{'scope':11}{'run':{w}} {'N':>5} {'dex':>5} {'slope dV':>9} {'r':>7} "
        f"{'slope raw':>10} {'r':>7} {'ratio min..max':>18} {'frac<1':>7}"
    )
    for r in out:
        print(
            f"{r['scope']:11}{r['run']:{w}} {r['n_rows']:>5} {r['shell_n0_span_dex']:>5} "
            f"{r['slope_dV_vs_n0']:>9} {r['r_dV_vs_n0']:>7} {r['slope_nIFraw_vs_n0']:>10} "
            f"{r['r_nIFraw_vs_n0']:>7} "
            f"{r['ratio_min']+'..'+r['ratio_max']:>18} {r['frac_ratio_lt_1']:>7}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        fh.write(stamp(__file__) + "\n")
        fh.write("# Coupling of the PRE-CAP Stromgren density to the confining pressure.\n")
        fh.write("# ratio = n_IF_Str_raw / shell_n0 == (uncapped P_HII) / Pb, exactly.\n")
        fh.write("# Unfinished runs excluded (C-7): pooling a killed arm with its re-run\n")
        fh.write("# double-counts rows. Skipped: " + ("; ".join(skipped) or "none") + "\n")
        wr = csv.DictWriter(fh, fieldnames=COLS)
        wr.writeheader()
        wr.writerows(out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
