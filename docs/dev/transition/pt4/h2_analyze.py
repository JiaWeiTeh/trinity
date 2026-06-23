#!/usr/bin/env python3
"""
H2 audit: is the Lloss collapse driven by in-cloud dilution or by the
R2 -> rCloud blowout crossing?

Reads the committed cleanroom h0 CSVs (hybr full runs, stop_t=6) and reports
per config:
  - R2/rCloud over time; whether/when R2 first reaches rCloud
  - the cooling ratio (Lgain-Lloss)/Lgain trajectory and its minimum
  - whether bubble_Lloss collapses BEFORE or AT the R2->rCloud crossing
  - Lloss at peak vs at the rCloud crossing (collapse factor)

This separates "geometry inside the cloud" (in-cloud dilution) from "blowout"
(crossing into nISM). Pure read of committed data; no sims.

Run:  python docs/dev/transition/pt4/h2_analyze.py
Writes: docs/dev/transition/pt4/h2_crossing_summary.csv
"""
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "cleanroom", "data"))
OUT_CSV = os.path.join(HERE, "h2_crossing_summary.csv")

# rCloud [pc] per config. The h0 CSVs left the rCloud column blank, so we inject
# the authoritative value computed by h2_rcloud_compute.py (production pipeline:
# read_param + get_InitCloudProp). These match harvest_h0.py's hardcoded table.
RCLOUD = {
    "simple_cluster": 1.69,
    "pl2_steep": 21.35,
    "small_dense_highsfe": 0.3255,
    "midrange_pl0": 8.53,
    "large_diffuse_lowsfe": 88.05,
    "be_sphere": 15.5,
}


def load(path):
    t, R2, rCloud, Lg, Ll, Eb, T0, v2 = ([] for _ in range(8))
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            def g(k):
                v = row.get(k, "")
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return np.nan
            t.append(g("t_now"))
            R2.append(g("R2"))
            rCloud.append(g("rCloud"))
            Lg.append(g("bubble_Lgain"))
            Ll.append(g("bubble_Lloss"))
            Eb.append(g("Eb"))
            T0.append(g("T0"))
            v2.append(g("v2"))
    return {k: np.asarray(v) for k, v in
            dict(t=t, R2=R2, rCloud=rCloud, Lg=Lg, Ll=Ll, Eb=Eb, T0=T0, v2=v2).items()}


def analyze(path):
    d = load(path)
    name = os.path.basename(path).replace("c0_", "").replace("_h0.csv", "")
    t, R2, Ll, Lg = d["t"], d["R2"], d["Ll"], d["Lg"]
    # rCloud is a run constant excluded from snapshots; the h0 CSV column is
    # blank. Inject the authoritative value (see RCLOUD table).
    rCloud = RCLOUD.get(name, np.nan)
    rc_all = d["rCloud"][np.isfinite(d["rCloud"])]
    if not np.isfinite(rCloud) and rc_all.size:
        rCloud = float(np.nanmax(rc_all))

    ratio = np.where(Lg > 0, (Lg - Ll) / Lg, np.nan)

    # First index where R2 >= rCloud (the blowout crossing)
    cross_idx = None
    if np.isfinite(rCloud):
        ge = np.where(R2 >= rCloud)[0]
        cross_idx = int(ge[0]) if ge.size else None

    # Lloss peak (only over rows with finite Lloss > 0)
    finite_ll = np.isfinite(Ll) & (Ll > 0)
    if finite_ll.any():
        ll_peak_idx = int(np.nanargmax(np.where(finite_ll, Ll, -np.inf)))
        ll_peak = Ll[ll_peak_idx]
        t_ll_peak = t[ll_peak_idx]
        R2_at_llpeak = R2[ll_peak_idx]
    else:
        ll_peak_idx = ll_peak = t_ll_peak = R2_at_llpeak = np.nan

    # Cooling-ratio minimum (closest the trigger ever gets to 0.05)
    finite_ratio = np.isfinite(ratio)
    if finite_ratio.any():
        rmin_idx = int(np.nanargmin(np.where(finite_ratio, ratio, np.inf)))
        ratio_min = ratio[rmin_idx]
        t_ratio_min = t[rmin_idx]
        R2_at_rmin = R2[rmin_idx]
    else:
        rmin_idx = ratio_min = t_ratio_min = R2_at_rmin = np.nan

    res = dict(
        config=name,
        rCloud_pc=rCloud,
        n_rows=len(t),
        t_first_finite=float(np.nanmin(t)) if t.size else np.nan,
        t_last=float(np.nanmax(t)) if t.size else np.nan,
        R2_max=float(np.nanmax(R2)) if R2.size else np.nan,
        R2max_over_rCloud=float(np.nanmax(R2) / rCloud) if np.isfinite(rCloud) else np.nan,
        reached_rCloud=cross_idx is not None,
        t_cross=float(t[cross_idx]) if cross_idx is not None else np.nan,
        R2_at_cross=float(R2[cross_idx]) if cross_idx is not None else np.nan,
        Ll_peak=float(ll_peak),
        t_Ll_peak=float(t_ll_peak),
        R2_at_Ll_peak=float(R2_at_llpeak),
        R2overRc_at_Ll_peak=float(R2_at_llpeak / rCloud) if np.isfinite(rCloud) else np.nan,
        ratio_min=float(ratio_min),
        t_ratio_min=float(t_ratio_min),
        R2overRc_at_ratio_min=float(R2_at_rmin / rCloud) if np.isfinite(rCloud) else np.nan,
    )

    # When does Lloss collapse relative to the crossing?
    # Compare the Lloss peak epoch to the crossing epoch.
    if cross_idx is not None and np.isfinite(t_ll_peak):
        res["Llpeak_vs_cross"] = (
            "peak_before_cross" if t_ll_peak < t[cross_idx]
            else "peak_at_or_after_cross"
        )
        # Lloss at crossing relative to peak (collapse factor by the crossing)
        res["Ll_at_cross_over_peak"] = (
            float(Ll[cross_idx] / ll_peak) if ll_peak > 0 else np.nan
        )
    else:
        res["Llpeak_vs_cross"] = "no_crossing"
        res["Ll_at_cross_over_peak"] = np.nan

    return res, d, ratio, cross_idx, rCloud


def main():
    files = sorted(glob.glob(os.path.join(DATA, "c0_*_h0.csv")))
    rows = []
    print(f"Reading {len(files)} h0 CSVs from {DATA}\n")
    for p in files:
        res, d, ratio, cross_idx, rCloud = analyze(p)
        rows.append(res)
        print("=" * 72)
        print(f"CONFIG: {res['config']}")
        print(f"  rCloud            = {res['rCloud_pc']:.4g} pc   ({res['n_rows']} rows, "
              f"t in [{res['t_first_finite']:.3g}, {res['t_last']:.3g}] Myr)")
        print(f"  R2 max            = {res['R2_max']:.4g} pc   "
              f"(R2max/rCloud = {res['R2max_over_rCloud']:.3g})")
        if res["reached_rCloud"]:
            print(f"  R2 reaches rCloud : YES at t={res['t_cross']:.4g} Myr "
                  f"(R2={res['R2_at_cross']:.4g} pc)")
        else:
            print("  R2 reaches rCloud : NO (stays inside cloud over recorded span)")
        print(f"  Lloss peak        = {res['Ll_peak']:.4g} at t={res['t_Ll_peak']:.4g} Myr, "
              f"R2/rCloud={res['R2overRc_at_Ll_peak']:.3g}")
        print(f"  cooling ratio min = {res['ratio_min']:.4g} at t={res['t_ratio_min']:.4g} Myr, "
              f"R2/rCloud={res['R2overRc_at_ratio_min']:.3g}  (trigger fires if < 0.05)")
        print(f"  Lloss collapse    : {res['Llpeak_vs_cross']}; "
              f"Lloss(at cross)/Lloss(peak) = {res['Ll_at_cross_over_peak']}")
    print("=" * 72)

    # Write summary CSV
    cols = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
