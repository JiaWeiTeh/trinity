#!/usr/bin/env python3
"""f_kappa(n_H) calibration ESTIMATE — the merge's payoff curve (from the validated leverage).

THE GOAL (PLAN.md ⭐ synthesis): calibrate f_κ(properties) via the κ_eff mechanism
(`cooling_boost_kappa`) so the EMERGENT θ = L_cool/L_mech tracks the obs/3D target θ(n_H)
(Lancaster ≈0.9–0.99). This produces the actual calibration curve f_κ(n_H).

WHY AN ESTIMATE (not a full-run measurement). The full-run blowout-θ grid is HPC-scale: a single
sim to blowout is ~90 min (compact) to ~hours (diffuse) because the energy phase runs a fine time
grid (params `runs/params/cal_{compact,diffuse}__k{1,2,4}.param` + harvester
`make_kappa_blowout_calibration.py` are committed and ready for that HPC run). In-session we instead
COMBINE two committed, verified ingredients:
  1. the cooling LEVERAGE  L_cool(f_κ)/L_cool(1) ≈ f_κ^p  with p measured = 0.63
     (`fkappa_leverage.csv`; p read live below; L_mech is f_κ-independent so θ scales the same way), and
  2. the resolved baseline θ(n_H) at blowout per config (`fmix_table.csv` + nCore from `da_replay.csv`;
     0.25 diffuse → 0.70 dense).
giving  θ(f_κ, n_H) ≈ min(θ_max, θ_base(n_H) · f_κ^p)  and inverting for the f_κ that hits the target:
  f_κ_needed(n_H) = (θ_target / θ_base(n_H))^{1/p}.

CAVEAT (honest, must keep): the leverage p=0.63 was measured on EARLY snapshots where θ≈0.01, far
from the θ→1 ceiling. At blowout θ_base is 0.25–0.70, so the real leverage SATURATES as θ→1 — the
true f_κ_needed is therefore a LOWER bound (this estimate is optimistic near the target). The
committed full-run grid is exactly what would replace this estimate with a measurement.

REPRODUCE (from repo root):
    python docs/dev/transition/pdv-trigger/data/make_kappa_calibration_estimate.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/kappa_calibration_estimate.csv
    docs/dev/transition/pdv-trigger/kappa_calibration_estimate.png
"""

import csv
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_THETA_MAX = 0.99
_TARGETS = (0.90, 0.95, 0.99)


def _leverage_exponent():
    """Fit p in L_cool/L_cool(1) = f_κ^p from fkappa_leverage.csv (mean over states, at f_κ=64)."""
    rows = list(csv.DictReader(open(os.path.join(_HERE, "fkappa_leverage.csv"))))
    exps = []
    for r in rows:
        if r["f_kappa"] == "64" and r.get("healthy") in ("True", "true"):
            exps.append(math.log(float(r["LTotal_mult"])) / math.log(64.0))
    return sum(exps) / len(exps)


def _baselines():
    """config -> (nCore, theta_base_at_blowout). Joins fmix_table (θ) with da_replay (nCore)."""
    theta = {r["config"]: float(r["Lcool_over_Lmech_at_blowout"])
             for r in csv.DictReader(open(os.path.join(_HERE, "fmix_table.csv")))}
    ncore = {r["config"]: float(r["nCore"])
             for r in csv.DictReader(open(os.path.join(_HERE, "da_replay.csv")))}
    out = {}
    for cfg, th in theta.items():
        if cfg in ncore:
            out[cfg] = (ncore[cfg], th)
    return out


def main():
    p = _leverage_exponent()
    base = _baselines()
    print(f"leverage exponent p = {p:.3f}  (L_cool ~ f_κ^p; θ scales the same, L_mech is f_κ-independent)")

    rows = []
    for cfg, (nC, th0) in sorted(base.items(), key=lambda kv: kv[1][0]):
        rec = {"config": cfg, "nCore": nC, "theta_base": th0, "exponent_p": round(p, 4)}
        for tgt in _TARGETS:
            fk = 1.0 if th0 >= tgt else (tgt / th0) ** (1.0 / p)
            rec[f"fkappa_for_theta{int(tgt*100)}"] = round(fk, 3)
        rows.append(rec)
        print(f"  nCore={nC:.0e}  θ_base={th0:.3f}  ->  f_κ for θ=0.90/0.95/0.99 = "
              f"{rec['fkappa_for_theta90']:.2f} / {rec['fkappa_for_theta95']:.2f} / {rec['fkappa_for_theta99']:.2f}")

    csv_path = os.path.join(_HERE, "kappa_calibration_estimate.csv")
    cols = ["config", "nCore", "theta_base", "exponent_p",
            "fkappa_for_theta90", "fkappa_for_theta95", "fkappa_for_theta99"]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    nC = np.array([r["nCore"] for r in rows])
    th0 = np.array([r["theta_base"] for r in rows])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # LEFT -- the gap the calibration closes
    axL.axhspan(0.90, 0.99, color="#2ca02c", alpha=0.10, label="obs/3D target θ (Lancaster 0.9–0.99)")
    axL.plot(nC, th0, "o-", color="#1f77b4", lw=2, ms=7, label="TRINITY resolved θ at blowout")
    for x, y in zip(nC, th0):
        axL.annotate(f"{y:.2f}", (x, y), fontsize=7.5, xytext=(0, -12), textcoords="offset points", ha="center")
    axL.set_xscale("log")
    axL.set_xlabel(r"$n_{\rm Core}$ [cm$^{-3}$]")
    axL.set_ylabel(r"$\theta = L_{\rm cool}/L_{\rm mech}$ at blowout")
    axL.set_ylim(0, 1.02)
    axL.set_title("The gap: TRINITY under-cools vs obs/3D,\nworst for diffuse clouds", fontsize=10.5, fontweight="bold")
    axL.legend(fontsize=8.5, loc="center right")

    # RIGHT -- the calibration curve f_κ(n_H)
    for tgt, c in zip(_TARGETS, ["#2ca02c", "#1f77b4", "#d62728"]):
        fk = np.array([1.0 if t >= tgt else (tgt / t) ** (1.0 / p) for t in th0])
        axR.plot(nC, fk, "o-", color=c, lw=2, ms=6, label=f"reach θ={tgt}")
    axR.set_xscale("log")
    axR.set_yscale("log", base=2)
    axR.set_xlabel(r"$n_{\rm Core}$ [cm$^{-3}$]")
    axR.set_ylabel(r"$f_\kappa$ needed (cooling_boost_kappa)")
    axR.set_title("The calibration f_κ(n_H): diffuse clouds need more κ-boost\n"
                  "(ESTIMATE from the f_κ^0.63 leverage; full-run grid pending HPC)", fontsize=10.5, fontweight="bold")
    axR.legend(fontsize=8.5, loc="upper right")
    axR.text(0.02, 0.03, "caveat: leverage saturates as θ→1, so true f_κ ≥ this (estimate is optimistic)",
             transform=axR.transAxes, fontsize=7.3, color="0.35")

    fig.suptitle("f_κ(n_H) calibration estimate — the merge's payoff: a density-dependent cooling enhancement",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "kappa_calibration_estimate.png")
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
