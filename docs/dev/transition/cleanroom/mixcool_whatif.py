#!/usr/bin/env python3
"""Offline what-if calibration for a mixing-layer cooling term (root-fix prototype gate).

Lancaster+2021 / El-Badry+2019: a turbulent fractal mixing layer radiates a large
fraction of the wind power, so real bubbles retain f_ret~0.01-0.1 (not Weaver ~0.45).
Simplest parameterization: L_mix = theta * L_mech added to the cooling loss.

This STATIC what-if (on the existing h0 CSVs, no re-run, ignores dynamical
back-reaction) calibrates theta by asking, per config:
  - modified cooling ratio (L_mech - L_loss - theta*L_mech)/L_mech : does it cross
    the 0.05 F0 threshold (i.e. would a cooling transition now fire)?
  - rough modified retained energy f_ret' ~= f_ret - theta : does it land in 0.01-0.1?
It is a feasibility gate, NOT the validated result (that needs the production re-run).

    python mixcool_whatif.py docs/dev/transition/cleanroom/data/c0_*_h0.csv
"""
from __future__ import annotations

import csv, glob, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
THETAS = (0.1, 0.2, 0.3, 0.4, 0.5)
BAND = (0.01, 0.10)


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if r.get("phase") != "implicit":
            continue
        try:
            t = float(r["t_now"]); Lg = float(r["Lmech_total"]); Ll = float(r["bubble_Lloss"])
            fr = float(r["f_ret"]) if r.get("f_ret") not in ("", "None") else None
        except (ValueError, TypeError, KeyError):
            continue
        if Lg > 0:
            rows.append((t, Lg, Ll, fr))
    return rows


def main():
    paths = sorted(sys.argv[1:] or glob.glob(str(HERE / "data" / "c0_*_h0.csv")))
    print(f"Mixing-layer what-if: L_loss' = L_loss + theta*L_mech  (static, ignores back-reaction)")
    print(f"{'config':22s} {'theta':>5s} {'F0 fires @':>11s} {'ratio_min':>9s} "
          f"{'f_ret_end':>9s} {'f_ret_end-theta':>15s} {'in band?':>9s}")
    for p in paths:
        name = Path(p).stem.replace("c0_", "").replace("_h0", "")
        rows = load(p)
        if not rows:
            continue
        fret_end = next((fr for t, Lg, Ll, fr in reversed(rows) if fr is not None), None)
        for th in THETAS:
            f0 = None
            ratio_min = 1e9
            for t, Lg, Ll, fr in rows:
                ratio = (Lg - Ll - th * Lg) / Lg
                ratio_min = min(ratio_min, ratio)
                if f0 is None and ratio < 0.05:
                    f0 = t
            frp = (fret_end - th) if fret_end is not None else None
            inband = "YES" if (frp is not None and BAND[0] <= frp <= BAND[1]) else \
                     ("<band" if (frp is not None and frp < BAND[0]) else "above")
            print(f"{name:22s} {th:>5.1f} {(f'{f0:.2f}' if f0 else 'never'):>11s} "
                  f"{ratio_min:>9.2f} {fret_end:>9.2f} {(frp if frp is not None else 0):>15.2f} "
                  f"{inband:>9s}")
        print()


if __name__ == "__main__":
    main()
