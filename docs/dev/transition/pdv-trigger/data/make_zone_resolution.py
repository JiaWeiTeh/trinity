#!/usr/bin/env python3
"""Zone resolution diagnostic — how many grid points, and over what radial extent, does the
bubble-structure solver actually put in L1 (hot interior), L2 (conduction front) and L3
(intermediate sliver)?  (FINDINGS §15j anatomy; phase6_brief.html §1.)

WHY: f_A scales the interface-band losses L2+L3 (bubble_luminosity.py:845). L1/L2/L3 are NOT a
partition of one array: L1 is a slice of the raw ~60k solution grid (T_array[index_CIE_switch:]),
while L2 is a fresh resample to _CONDUCTION_NPTS (=2000, masked to T<10^5.5) and L3 is a fixed
np.linspace(num=1000) off the continuous dense-output solution. So the point counts and the point
DENSITY per pc differ by orders of magnitude — this script measures the real numbers.

HOW: monkeypatch bubble_luminosity._trapezoid to record len/min/max of each zone's integration
r-array, run three committed L21b bench __none params (dense/mid/diffuse), and capture the 3rd
energy-phase evaluation (settled, past the first-step transient), then early-exit. No sims are left
running; nothing is written to the repo (cwd is a temp dir; bench path2output is git-ignored anyway).

    python docs/dev/transition/pdv-trigger/data/make_zone_resolution.py
Deliverable: data/zone_resolution.csv (committed; the durable record — do not re-run to read it).
"""
import csv
import io
import contextlib
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDV = HERE.parent
REPO = HERE.parents[3]
PARAMS = PDV / "runs" / "params" / "bench5"

# (label, n_bar_H, committed __none param) — dense / mid / diffuse span of the L21b suite.
CONFIGS = [
    ("dense", 228000.0, PARAMS / "bench5_m5e5_r2p5__none.param"),
    ("mid", 5520.0, PARAMS / "bench3_m1e5_r5__none.param"),
    ("diffuse", 43.1, PARAMS / "bench1_m5e4_r20__none.param"),
]
CAPTURE_EVAL = 3  # the settled evaluation (eval 1 is the thin-front transient)


def _probe(label, n_bar, param_path):
    import numpy as np
    import trinity.bubble_structure.bubble_luminosity as bl
    from trinity._input import read_param
    from trinity import main

    real_trap = bl._trapezoid
    calls = []

    def trap(y, x=None, **kw):
        if x is not None and hasattr(x, "__len__") and len(x) > 10:
            xa = np.asarray(x)
            calls.append((len(xa), float(xa.min()), float(xa.max())))
        return real_trap(y, x=x, **kw) if x is not None else real_trap(y, **kw)

    bl._trapezoid = trap
    real_bl = bl._bubble_luminosity
    state = {"n": 0, "row": None}

    def wrap(*a, **k):
        calls.clear()
        out = real_bl(*a, **k)
        seen = []
        for L, lo, hi in calls:  # L-integral and Tavg share the r-array -> dedupe consecutive
            if not seen or seen[-1][0] != L:
                seen.append((L, lo, hi))
        state["n"] += 1
        if state["n"] == CAPTURE_EVAL and len(seen) >= 3:
            R2 = float(a[0]["R2"].value)
            state["row"] = {"config": label, "n_bar_H": n_bar, "R2_pc": R2, "raw_grid": None}
            for lab, (L, lo, hi) in zip(["L1", "L2", "L3"], seen[:3]):
                w = hi - lo
                state["row"][f"{lab}_npts"] = L
                state["row"][f"{lab}_width_pc"] = w
                state["row"][f"{lab}_pct_R2"] = 100 * w / R2
                state["row"][f"{lab}_pts_per_pc"] = (L / w) if w > 0 else float("inf")
            raise SystemExit(0)
        return out

    bl._bubble_luminosity = wrap
    params = read_param.read_param(str(param_path))
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            main.start_expansion(params)
        except SystemExit:
            pass
    bl._trapezoid = real_trap
    bl._bubble_luminosity = real_bl
    return state["row"]


def main_():
    sys.path.insert(0, str(REPO))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        cwd = os.getcwd()
        os.chdir(td)  # bench path2output is relative + git-ignored; keep the repo clean
        try:
            for label, n_bar, p in CONFIGS:
                row = _probe(label, n_bar, p)
                if row:
                    rows.append(row)
                    print(f"{label:8s} n={n_bar:>8g} R2={row['R2_pc']:.4f}pc | "
                          f"L1 {row['L1_npts']:>6d}/{row['L1_pct_R2']:5.1f}%R2  "
                          f"L2 {row['L2_npts']:>5d}/{row['L2_pts_per_pc']:.1e}pts/pc  "
                          f"L3 {row['L3_npts']:>5d}")
        finally:
            os.chdir(cwd)
    if not rows:
        sys.exit("no rows captured (did the energy phase run? check bench params exist).")
    cols = list(rows[0].keys())
    out = HERE / "zone_resolution.csv"
    with out.open("w", newline="") as fh:
        fh.write("# Zone element counts + radial extent of L1/L2/L3 (bubble-structure solver), "
                 "3rd settled energy-phase eval, dense/mid/diffuse L21b benches (fA=1 baseline). "
                 "L1 = raw-grid slice T>10^5.5; L2 = _CONDUCTION_NPTS resample; L3 = fixed 1000-pt "
                 "linspace. See phase6_brief.html §1 / FINDINGS §15j. Regenerate: make_zone_resolution.py\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main_()
