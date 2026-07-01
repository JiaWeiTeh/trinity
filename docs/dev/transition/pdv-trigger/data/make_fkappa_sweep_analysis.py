#!/usr/bin/env python3
"""819-combo sweep RESULTS — score the pre-registered f_κ(n_H) predictions + fit the corrected form.

The controlled sweep (7 nCore × 3 mCloud × 3 sfe = 63 cells × 13 f_κ) ran on Helix; the reduced
per-cell table is `data/fkappa_nH_sweep.csv` (output of make_fkappa_nH_sweep.py). This script reads
ONLY that committed CSV (no sims) and:
  1. scores the four predictions F_KAPPA_FUNCTIONAL_FORM.md §0 pre-registered (before the sweep);
  2. fits the corrected central form f_κ_fire ≈ A·n_core^s from the MEASURED firing f_κ;
  3. quantifies the de-conflation fan-out and the low-n/high-sfe "never fires at f_κ≤64" floor;
  4. draws prediction-vs-measured.

PRE-REGISTERED predictions (from the composed form, BEFORE this data existed):
  P1 slope:    f_κ(n_H) ∝ n^(−0.30)                          MEASURED: n^(−0.60)   ❌ too shallow ~2×
  P2 fan-out:  cells do NOT collapse to one n_H curve         MEASURED: ×2–32 spread ✅
  P3 baseline: logit θ₀ = −1.73 + 0.41·log₁₀n                 MEASURED: ~ −3.4 + 1.13·log₁₀n  ❌ ~3× steeper
  P4 leverage: p ≈ 0.31 (raw, full-range)                     MEASURED: median 0.21 ⚠ ballpark/high
  physical:    diffuse end unreachable by Spitzer f_κ → κ_mix MEASURED: 6/63 low-n high-sfe never fire ✅
Net: the qualitative conclusions held (steep decline, multi-dimensional, diffuse→κ_mix); the slope was
2× too shallow because the 6-anchor baseline θ₀(n) was undersampled (0.41 vs the real 1.13/dex).

REPRODUCE (from repo root; reads only the committed CSV, no sims):
    python docs/dev/transition/pdv-trigger/data/make_fkappa_sweep_analysis.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_sweep_scorecard.csv
    docs/dev/transition/pdv-trigger/fkappa_sweep_analysis.png
"""

import csv
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_IN = os.path.join(_HERE, "fkappa_nH_sweep.csv")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    rows = list(csv.DictReader(open(_IN)))
    for r in rows:
        for k in ("mCloud", "sfe", "nCore", "theta_fk1", "fit_p",
                  "f_kappa_fire_fit", "f_kappa_fire_measured"):
            r[k] = _f(r[k])

    x = np.array([r["nCore"] for r in rows])
    ym = np.array([r["f_kappa_fire_measured"] for r in rows])
    g = np.isfinite(x) & np.isfinite(ym) & (ym > 0)
    s, lnA = np.polyfit(np.log(x[g]), np.log(ym[g]), 1)            # corrected central form
    A = math.exp(lnA)

    # P3 baseline θ0(n): logit(theta_fk1) vs log10(nCore), finite 0<θ<1
    lx = np.array([math.log10(r["nCore"]) for r in rows])
    t0 = np.array([r["theta_fk1"] for r in rows])
    gt = np.isfinite(t0) & (t0 > 0) & (t0 < 1)
    b3, a3 = np.polyfit(lx[gt], np.log(t0[gt] / (1 - t0[gt])), 1)

    pv = np.array([r["fit_p"] for r in rows])
    pg = np.isfinite(pv)
    p_med = float(np.median(pv[pg]))

    never = [(r["mCloud"], r["sfe"], r["nCore"]) for r in rows
             if not math.isfinite(r["f_kappa_fire_measured"])]

    # fan-out: spread per nCore
    spreads = {}
    for nc in sorted(set(r["nCore"] for r in rows)):
        v = [r["f_kappa_fire_measured"] for r in rows
             if r["nCore"] == nc and math.isfinite(r["f_kappa_fire_measured"])]
        if v:
            spreads[nc] = (min(v), max(v), max(v) / min(v))

    print(f"corrected central form (measured): f_κ_fire ≈ {A:.0f}·n^({s:.2f})   [θ*=0.95]")
    print(f"  baseline θ0: logit θ0 = {a3:+.2f} {b3:+.2f}·log10(n)   (pred −1.73 +0.41)")
    print(f"  leverage p median = {p_med:.2f}   (pred 0.31)")
    print(f"  never-fire @f_κ≤64: {len(never)}/{len(rows)} cells -> {never}")

    # scorecard CSV
    score = [
        ("P1_slope", "f_kappa ~ n^-0.30", f"n^{s:.2f}", "FAIL (2x too shallow)"),
        ("P2_deconflation", "fan-out (not one curve)",
         f"x{max(v[2] for v in spreads.values()):.0f} spread + {len(never)} never-fire", "PASS"),
        ("P3_baseline_slope", "logit theta0 slope 0.41/dex", f"{b3:.2f}/dex", "FAIL (~3x steeper)"),
        ("P4_leverage", "p ~ 0.31", f"median {p_med:.2f}", "PARTIAL (ballpark; point high)"),
        ("physical_diffuse_kmix", "diffuse unreachable by f_kappa", f"{len(never)} cells never fire", "PASS"),
    ]
    sc_path = os.path.join(_HERE, "fkappa_sweep_scorecard.csv")
    with open(sc_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["prediction", "pre_registered", "measured", "grade"])
        w.writerows(score)
        fh.write(f"# corrected central form: f_kappa_fire ~= {A:.0f} * n_core^({s:.2f})  (theta*=0.95, measured-only)\n")
        fh.write(f"# baseline: logit(theta0) = {a3:+.3f} {b3:+.3f}*log10(nCore)\n")
        fh.write(f"# leverage p median = {p_med:.3f}; never-fire(f_k<=64) = {len(never)}/{len(rows)} (all low-n high-sfe)\n")
        fh.write("# NOTE fan-out (x2-32 across mCloud/sfe at fixed n) => f_kappa is multi-dimensional, not f(n_H) alone\n")
    print(f"wrote {sc_path}")

    # ---------------- figure ----------------
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

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # LEFT: de-conflation fan-out (per mCloud/sfe series) + corrected all-cell fit + never-fire markers
    cells = sorted({(r["mCloud"], r["sfe"]) for r in rows})
    cmap = plt.get_cmap("viridis")
    for i, (mC, sf) in enumerate(cells):
        sub = sorted([r for r in rows if r["mCloud"] == mC and r["sfe"] == sf], key=lambda r: r["nCore"])
        xs = np.array([r["nCore"] for r in sub])
        ys = np.array([r["f_kappa_fire_measured"] if math.isfinite(r["f_kappa_fire_measured"])
                       else r["f_kappa_fire_fit"] for r in sub])
        axL.loglog(xs, ys, "o-", color=cmap(i / max(1, len(cells) - 1)), lw=1.2, ms=4, alpha=0.8)
    xx = np.logspace(2, 5, 50)
    axL.loglog(xx, A * xx ** s, "--", color="k", lw=2,
               label=fr"measured fit: $f_\kappa\propto n^{{{s:.2f}}}$")
    axL.loglog(xx, 161 * xx ** -0.30, ":", color="#d62728", lw=2,
               label=r"my pre-reg pred: $\propto n^{-0.30}$")
    for (mC, sf, nc) in never:
        axL.plot(nc, 64, "x", color="r", ms=9, mew=2)
    axL.plot([], [], "xr", mew=2, label=f"never fires @f_κ≤64 ({len(never)} cells)")
    axL.set_xlabel(r"$n_{\rm core}$ [cm$^{-3}$]")
    axL.set_ylabel(r"$f_\kappa$ to fire ($\theta=0.95$)")
    axL.set_title("De-conflation: FAN-OUT, not collapse\n(slope steeper than predicted)", fontsize=10.5, fontweight="bold")
    axL.legend(fontsize=8.5, loc="lower left")
    axL.grid(True, which="both", alpha=0.2)

    # RIGHT: prediction vs measured (per-cell scatter, 1:1 line)
    pred = []
    meas = []
    for r in rows:
        if math.isfinite(r["f_kappa_fire_measured"]) and r["f_kappa_fire_measured"] > 0:
            # my composed prediction at this nCore (raw-power form, theta*=0.95, p=0.31, old baseline)
            th0 = 1.0 / (1.0 + math.exp(-(-1.73 + 0.41 * math.log10(r["nCore"]))))
            fk_pred = (0.95 / th0) ** (1 / 0.31) if th0 < 0.95 else 1.0
            pred.append(fk_pred)
            meas.append(r["f_kappa_fire_measured"])
    axR.loglog(meas, pred, "o", color="#1f77b4", ms=5, alpha=0.6)
    lim = [0.8, 200]
    axR.loglog(lim, lim, "-", color="0.4", lw=1, label="1:1")
    axR.set_xlim(lim)
    axR.set_ylim(lim)
    axR.set_xlabel(r"MEASURED $f_\kappa$ to fire")
    axR.set_ylabel(r"my pre-reg PREDICTED $f_\kappa$")
    axR.set_title("Prediction vs measured (per cell)\n(systematic: pred too shallow → over@dense, under@diffuse)",
                  fontsize=10.5, fontweight="bold")
    axR.legend(fontsize=9, loc="upper left")
    axR.grid(True, which="both", alpha=0.2)

    fig.suptitle("819-combo sweep: f_κ(n_H) prediction scorecard — qualitative PASS (fan-out, diffuse→κ_mix), slope 2× too shallow",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "fkappa_sweep_analysis.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
