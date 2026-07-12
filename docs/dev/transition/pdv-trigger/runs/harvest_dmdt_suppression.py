#!/usr/bin/env python3
"""📏 theta5s fidelity reduction — the dMdt-suppression measurement (read iii of Phase 4).

The f_mix multiplier CANNOT produce this (it never touches the structure); f_A can. For each
boosted arm we interpolate its evaporative flux bubble_dMdt(t) onto the fA=1 (__none) baseline
arm's ACCEPTED-row time grid and report the suppression ratio dMdt(fA)/dMdt(1) over the shared
span (the compare_live.py pattern). El-Badry Eq 47 predicts ṁ ∝ (1−θ)^{37/35}/θ^{2/7} — a
DECREASING function of interface cooling; this is a TREND check, not a fit (state divergence at
matched t is conflated in the ratio — say so in any caption).

CAVEATS baked into the output (SOURCE_TERM_DESIGN §3 Phase 4):
  - report only where the overlap is >= MIN_OVERLAP accepted segments;
  - arms that fire before ~0.1 Myr leave a tiny overlap window -> flagged upper_limit=True;
  - theta from bubble_Lloss/Lmech_total on accepted rows (never a call-level observer).

Runs against the RAW run dirs (dictionary.jsonl), so on HPC or on downloaded arms:
    python harvest_dmdt_suppression.py "$WS"/outputs/theta5s/* \
        --csv docs/dev/transition/pdv-trigger/data/theta5s_dmdt_suppression.csv
"""
import argparse
import json
import math
import os
import sys

import numpy as np

MIN_OVERLAP = 20          # accepted-segment overlap floor to quote a ratio
EARLY_FIRE_T = 0.1        # Myr: arms firing before this contribute upper limits only


def _accepted_rows(run_dir):
    """(t, dMdt, theta) on accepted implicit rows with Pb>0 and finite dMdt, sorted by t."""
    f = os.path.join(run_dir, "dictionary.jsonl")
    ts, dm, th = [], [], []
    if not os.path.exists(f):
        return np.array([]), np.array([]), np.array([])
    for line in open(f):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        pb = d.get("Pb"); dmdt = d.get("bubble_dMdt"); ll = d.get("bubble_Lloss"); lm = d.get("Lmech_total")
        t = d.get("t_now")
        if (pb and pb > 0 and dmdt is not None and math.isfinite(dmdt) and t is not None):
            ts.append(t); dm.append(dmdt)
            th.append(ll / lm if (ll is not None and lm) else float("nan"))
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(dm)[o], np.array(th)[o]


def _config_of(run_dir):
    base = os.path.basename(run_dir.rstrip("/"))
    return base.rsplit("__", 1)[0] if "__" in base else base


def _mode_of(run_dir):
    base = os.path.basename(run_dir.rstrip("/"))
    return base.rsplit("__", 1)[1] if "__" in base else "none"


def suppression_rows(run_dirs):
    """Group run dirs by config; ratio each boosted arm's dMdt onto its __none baseline grid."""
    by_cfg = {}
    for d in run_dirs:
        by_cfg.setdefault(_config_of(d), {})[_mode_of(d)] = d
    rows = []
    for cfg, arms in sorted(by_cfg.items()):
        base_dir = arms.get("none")
        if base_dir is None:
            continue
        tb, dmb, thb = _accepted_rows(base_dir)
        if len(tb) < 2:
            continue
        for mode, d in sorted(arms.items()):
            if mode == "none":
                continue
            fa = float(mode[2:].replace("p", ".")) if mode.startswith("fa") else None
            if fa is None:
                continue
            t, dm, th = _accepted_rows(d)
            if len(t) < 2:
                continue
            lo, hi = max(tb[0], t[0]), min(tb[-1], t[-1])
            grid = tb[(tb >= lo) & (tb <= hi)]
            n = len(grid)
            if n < 2:
                rows.append(dict(config=cfg, fA=fa, n_overlap=0, ratio_med=float("nan"),
                                 upper_limit=True, note="no overlap"))
                continue
            dm_boost = np.interp(grid, t, dm)
            dm_base = np.interp(grid, tb, dmb)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = dm_boost / dm_base
            ratio = ratio[np.isfinite(ratio)]
            fired_early = bool(t[-1] < EARLY_FIRE_T)
            rows.append(dict(
                config=cfg, fA=fa, n_overlap=n,
                ratio_med=float(np.median(ratio)) if len(ratio) else float("nan"),
                ratio_lo=float(np.min(ratio)) if len(ratio) else float("nan"),
                ratio_hi=float(np.max(ratio)) if len(ratio) else float("nan"),
                theta_base_max=float(np.nanmax(thb)) if len(thb) else float("nan"),
                upper_limit=bool(n < MIN_OVERLAP or fired_early),
                note=("fired<0.1Myr" if fired_early else ("thin overlap" if n < MIN_OVERLAP else ""))))
    return rows


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args(argv)
    rows = suppression_rows(args.run_dirs)
    cols = ["config", "fA", "n_overlap", "ratio_med", "ratio_lo", "ratio_hi",
            "theta_base_max", "upper_limit", "note"]
    import csv
    out = open(args.csv, "w", newline="") if args.csv else sys.stdout
    out.write("# dMdt(fA)/dMdt(1) interpolated onto the __none baseline accepted-row grid per "
              "config. El-Badry Eq 47 => ratio<1 and falling with f_A. TREND check, not a fit "
              f"(state divergence conflated). quote only rows with n_overlap>={MIN_OVERLAP} and "
              "upper_limit=False.\n")
    w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    if args.csv:
        out.close()
        quotable = sum(1 for r in rows if not r["upper_limit"])
        print(f"wrote {args.csv}  ({len(rows)} arm ratios, {quotable} quotable)")


if __name__ == "__main__":
    main(sys.argv[1:])
