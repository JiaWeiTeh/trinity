#!/usr/bin/env python3
"""Sweep follow-up: the catastrophic-cooling CLIFF, the fan-out's second axis, and the metric.

Reads ONLY the committed reduced `data/summary.csv` (no sims) and quantifies three things that came
out of inspecting the 819-combo sweep:

  (A) THE CLIFF. For each (mCloud, sfe) series, θ@f_κ=1 (no boost) rises with nCore and then JUMPS
      past 0.95 — above that the cloud fires the cooling transition with NO boost. The jump (cliff)
      sits at LOWER nCore for MORE massive clouds: a 1e7 cloud is ~4.6× larger (rCloud) at fixed
      density, so it reaches the same swept COLUMN at lower nCore. The cliff lands at a roughly
      constant column N_H = nCore·rCloud ≈ 2–3×10²³ cm⁻² across cloud mass.

  (B) THE FAN-OUT'S SECOND AXIS — quantified, not hand-waved. The de-conflation answer is "f_κ does
      NOT collapse onto one nCore curve" (×2–32 spread). Testing collapse variables on the driver
      θ@f_κ=1: nCore is the best SINGLE predictor (R²≈0.73); column is slightly worse globally
      (R²≈0.71) even though it nails the cliff; adding rCloud as a 2nd axis lifts R² to ≈0.75. So the
      fan-out is genuinely multi-dimensional (nCore primary + a cloud-size term strongest at the cliff),
      and f_κ_fire is INDEPENDENT of cluster mass M★=sfe·mCloud (θ is L_mech-normalised).

  (C) THE METRIC SANITY. θ = bubble_LTotal/Lmech_total (radiative loss fraction) sampled at BLOWOUT
      (first R2>rCloud); firing uses theta_max≥0.95. Is "at blowout" a fragile choice? Empirically no:
      theta_max − theta_blowout has median ≈0.004 (>0.05 in only ~5/63 cells), so the snapshot-vs-peak
      choice barely moves the calibration. The runs split cleanly into cooled-before-escape vs
      blew-out-energy-driven — exactly the in-cloud transition question blowout is meant to answer.

REPRODUCE (from repo root; reads only the committed summary.csv, no sims):
    python docs/dev/transition/pdv-trigger/data/make_fkappa_cliff_metric.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_cliff_metric.csv
    docs/dev/transition/pdv-trigger/fkappa_cliff_metric.png
"""

import csv
import math
import os
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_SUMMARY = os.path.join(_HERE, "summary.csv")
_PC = 3.086e18      # cm per pc (rCloud is in pc)
_GRID = [1.0, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
_TRIG = 0.95


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _load():
    rows = list(csv.DictReader(open(_SUMMARY)))
    cell = defaultdict(dict)          # (mCloud,sfe,nCore) -> {f_k: row}
    for r in rows:
        cell[(_f(r["mCloud"]), _f(r["sfe"]), _f(r["nCore"]))][_f(r["cooling_boost_kappa"])] = r
    return rows, cell


def main():
    rows, cell = _load()

    # ---- (A) the cliff: per (mCloud,sfe), the nCore (and column) where θ@f_κ=1 crosses 0.95 ----
    cliffs = []
    series = defaultdict(list)        # (mCloud,sfe) -> [(nCore, theta_fk1, rCloud)]
    for (mC, sf, nc), by in cell.items():
        r1 = by[1.0]
        series[(mC, sf)].append((nc, _f(r1["theta_blowout"]), _f(r1["rCloud"])))
    for (mC, sf), pts in sorted(series.items()):
        pts.sort()
        nc_cliff = col_cliff = float("nan")
        for i in range(len(pts) - 1):
            (n0, t0, r0), (n1, t1, r1) = pts[i], pts[i + 1]
            if t0 < _TRIG <= t1:                       # log-interp the crossing in nCore
                f = (math.log(_TRIG) - math.log(t0)) / (math.log(t1) - math.log(t0))
                nc_cliff = math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0)))
                rcl = r0 + f * (r1 - r0)
                col_cliff = nc_cliff * rcl * _PC
                break
        cliffs.append({"mCloud": mC, "sfe": sf, "nCore_cliff": nc_cliff, "column_cliff": col_cliff})

    fin = [c for c in cliffs if math.isfinite(c["column_cliff"])]
    nc_sp = max(c["nCore_cliff"] for c in fin) / min(c["nCore_cliff"] for c in fin)
    col_sp = max(c["column_cliff"] for c in fin) / min(c["column_cliff"] for c in fin)
    col_med = float(np.median([c["column_cliff"] for c in fin]))
    print(f"(A) CLIFF: θ@f_κ=1 crosses 0.95 — spread across {len(fin)} series:")
    print(f"     nCore_cliff spread ×{nc_sp:.0f}   vs   column_cliff spread ×{col_sp:.1f}   "
          f"(median column ≈ {col_med:.1e} cm⁻²)")

    # ---- (B) collapse R² of the driver θ@f_κ=1 ----
    recs = []
    for (mC, sf, nc), by in cell.items():
        r1 = by[1.0]
        fire = [fk for fk in _GRID if str(by[fk]["cooling_fired"]).strip().lower() == "true"]
        recs.append(dict(mC=mC, sf=sf, nc=nc, rc=_f(r1["rCloud"]),
                         N=nc * _f(r1["rCloud"]) * _PC, th1=min(_f(r1["theta_blowout"]), 1.05),
                         fkf=(min(fire) if fire else float("nan")), Mstar=sf * mC))

    def r2(xs, ys):
        x = np.log(np.array(xs))
        y = np.array(ys, float)
        b, a = np.polyfit(x, y, 1)
        return b, 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)

    th = [r["th1"] for r in recs]
    rs = {}
    for var, lab in [("nc", "nCore"), ("N", "column"), ("rc", "rCloud")]:
        b, ss = r2([r[var] for r in recs], th)
        rs[lab] = ss
        print(f"(B) θ@f_κ=1 ~ ln({lab:7s}):  R²={ss:+.3f}  (slope {b:+.3f})")
    X = np.c_[[math.log(r["nc"]) for r in recs], [math.log(r["rc"]) for r in recs], np.ones(len(recs))]
    coef, *_ = np.linalg.lstsq(X, np.array(th), rcond=None)
    ss2 = 1 - np.sum((np.array(th) - X @ coef) ** 2) / np.sum((np.array(th) - np.mean(th)) ** 2)
    print(f"    2-var θ ~ {coef[0]:+.3f}·ln(nCore){coef[1]:+.3f}·ln(rCloud):  R²={ss2:.3f}  "
          f"(coef ratio {coef[0]/coef[1]:.1f}; ==1 ⇒ pure column)")
    good = [r for r in recs if math.isfinite(r["fkf"])]
    bM, ssM = r2([r["Mstar"] for r in good], [r["fkf"] for r in good])
    print(f"    f_κ_fire ~ ln(M★=sfe·mCloud):  R²={ssM:+.3f}  (≈0 ⇒ independent of cluster mass ✓)")

    # ---- (C) metric sanity: theta_max vs theta_blowout, and the blowout regime split ----
    gap = [_f(by[1.0]["theta_max"]) - _f(by[1.0]["theta_blowout"]) for by in cell.values()]
    n_big = sum(1 for g in gap if g > 0.05)
    blew = sum(1 for r in rows if str(r["blowout_t"]).strip() not in ("", "nan", "None"))
    print(f"(C) METRIC: theta_max − theta_blowout @f_κ=1: median={np.median(gap):.3f}  "
          f"max={max(gap):.3f}  >0.05 in {n_big}/{len(cell)} cells")
    print(f"    blowout regime: {blew}/{len(rows)} reached R2>rCloud (escaped); "
          f"{len(rows)-blew}/{len(rows)} cooled before escape")

    # ---- CSV ----
    out = os.path.join(_HERE, "fkappa_cliff_metric.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mCloud", "sfe", "nCore_cliff", "column_cliff"])
        w.writeheader()
        w.writerows([{k: c[k] for k in ("mCloud", "sfe", "nCore_cliff", "column_cliff")} for c in cliffs])
        fh.write(f"# (A) cliff: nCore spread ×{nc_sp:.0f} vs column spread ×{col_sp:.1f}; median column {col_med:.2e} cm^-2\n")
        fh.write(f"# (B) theta0 collapse R^2: nCore={rs['nCore']:.3f} column={rs['column']:.3f} rCloud={rs['rCloud']:.3f}; 2-var(nCore,rCloud)={ss2:.3f}\n")
        fh.write(f"#     f_kappa_fire ~ M_star R^2={ssM:.3f} (independent of cluster mass)\n")
        fh.write(f"# (C) metric: median(theta_max-theta_blowout)={np.median(gap):.3f}, >0.05 in {n_big}/{len(cell)}; {blew}/{len(rows)} escaped vs {len(rows)-blew} cooled-first\n")
    print(f"wrote {out}")

    # ---- figure: cliff vs nCore (fans out) vs cliff vs column (aligns) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from _trinity_style import use_trinity_style
            use_trinity_style()
        except Exception:
            pass
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    fig, (axN, axC) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    mclouds = sorted({r["mC"] for r in recs})
    cmap = plt.get_cmap("viridis")
    col = {m: cmap(i / max(1, len(mclouds) - 1)) for i, m in enumerate(mclouds)}
    for ax, key, lab in [(axN, "nc", r"$n_{\rm core}$ [cm$^{-3}$]"),
                         (axC, "N", r"column $N_H=n_{\rm core}\,r_{\rm cloud}$ [cm$^{-2}$]")]:
        for m in mclouds:
            sub = sorted([r for r in recs if r["mC"] == m and r["sf"] == 0.03], key=lambda r: r[key])
            ax.plot([r[key] for r in sub], [r["th1"] for r in sub], "o-",
                    color=col[m], lw=1.8, ms=6, label=rf"$M_{{\rm cl}}={m:.0e}$")
        ax.axhline(0.95, ls="--", color="#d62728", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
        ax.grid(True, which="both", alpha=0.2)
    axN.set_ylabel(r"baseline $\theta$ at $f_\kappa=1$")
    axN.set_title("vs density: cliff at DIFFERENT n per mass\n(massive clouds fire earlier)", fontsize=10.5, fontweight="bold")
    axC.set_title(fr"vs column: cliffs ALIGN at $N_H\!\approx\!{col_med:.0e}$" "\n(the cliff's collapse variable)",
                  fontsize=10.5, fontweight="bold")
    axC.axvline(col_med, ls=":", color="0.4", lw=1)
    axN.legend(fontsize=9, loc="upper left")
    fig.suptitle("The 1e7 'broken power law' = a catastrophic-cooling cliff at ~constant column (sfe=0.03 shown)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "fkappa_cliff_metric.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
