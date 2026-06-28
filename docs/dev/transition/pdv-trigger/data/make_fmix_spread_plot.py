#!/usr/bin/env python3
"""Boost-to-trigger spread figure: the load-bearing "no constant f_mix works" result.

ZERO simulation, ZERO physics change. Pure read of committed data:
  - f_mix per config from fmix_table.csv (= 0.95 / (L_cool/L_mech at blowout); with- and no-PdV cols).
  - ambient density per config = nCore from each config's .param (recorded in NCORE below; same source
    as make_theta_density_plot.py).

Why this figure (the honest, schematic-free companion to theta_vs_density.png): the headline finding
"no single constant cooling boost works" rests on TRINITY's OWN numbers, not on any literature band.
The boost needed to push the resolved loss to the 0.95 energy->momentum trigger is
    f_mix(n) = 0.95 / (L_cool/L_mech at blowout)
and it rises ~2.8x from dense (1.36) to diffuse (3.81). No horizontal line (any constant f_mix) hits
all six configs -> a constant knob over-boosts the dense end and under-boosts the diffuse end.

DEGENERACY note carried onto the figure (verified 2026-06-25): because the trigger threshold *is* 0.95,
"calibrate the boost to a flat literature theta_lit ~= 0.95" gives f_mix = 0.95/(L_cool/L_mech) -- i.e.
this exact curve. So a flat literature band adds nothing over these internal numbers; only a target that
VARIES along the trajectory, theta_target(Da), is non-degenerate (see PLAN.md "Next deliverable").

Run from the repo root:
  python docs/dev/transition/pdv-trigger/data/make_fmix_spread_plot.py
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless; no LaTeX -- plain-text labels only
import matplotlib.pyplot as plt

HERE = "docs/dev/transition/pdv-trigger"
SRC = f"{HERE}/data/fmix_table.csv"
DST = f"{HERE}/fmix_vs_density.png"

# ambient density per config = nCore [cm^-3] (same source as make_theta_density_plot.py):
#   cleanroom .param, except simple_cluster -> schema default 1e5 (param/simple_cluster.param sets none).
NCORE = {
    "large_diffuse_lowsfe": 1e2,
    "be_sphere": 1e4,
    "midrange_pl0": 1e4,
    "pl2_steep": 1e5,
    "simple_cluster": 1e5,
    "small_dense_highsfe": 1e6,
}


def main():
    d = pd.read_csv(SRC).set_index("config")
    rows = []
    for cfg, n in NCORE.items():
        rows.append((cfg, n, float(d.loc[cfg, "fmix_no_pdv"]), float(d.loc[cfg, "fmix_with_pdv"])))
    pts = pd.DataFrame(rows, columns=["config", "nCore", "fmix_no_pdv", "fmix_with_pdv"]).sort_values("nCore")

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # the span any single constant would have to cover -> shaded to show no horizontal line fits.
    lo, hi = pts["fmix_no_pdv"].min(), pts["fmix_no_pdv"].max()
    ax.axhspan(lo, hi, color="0.85", alpha=0.6, lw=0,
               label=f"span a constant must cover ({lo:.2f}-{hi:.2f}): no single value fits")

    # an example constant boost (f=2, the value the live runs used) -> visibly over/under per density.
    ax.axhline(2.0, color="crimson", ls="--", lw=1.5,
               label="example constant f_mix = 2 (over-boosts dense, under-boosts diffuse)")

    # the two f_mix columns vs density. SCATTER ONLY -- deliberately NOT connected (matches
    # theta_vs_density.png): these are 6 independent configs (e.g. pl2_steep and simple_cluster
    # both nCore 1e5 yet far apart), not one density sweep; a line would imply a continuous f_mix(n).
    ax.scatter(pts["nCore"], pts["fmix_no_pdv"], marker="o", s=95, color="tab:blue",
               zorder=5, edgecolor="k", linewidth=0.6,
               label="f_mix to reach 0.95 trigger (cooling only) = 0.95 / (L_cool/L_mech)")
    ax.scatter(pts["nCore"], pts["fmix_with_pdv"], marker="s", s=60, color="tab:green",
               zorder=5, edgecolor="k", linewidth=0.5, alpha=0.9,
               label="f_mix with PdV folded into the loss")

    for _, r in pts.iterrows():
        ax.annotate(r["config"], (r["nCore"], r["fmix_no_pdv"]),
                    textcoords="offset points", xytext=(7, 7), fontsize=7.3, color="tab:blue")

    ax.set_xscale("log")
    ax.set_xlim(7e1, 5e6)
    ax.set_ylim(0.8, 4.2)
    ax.set_xlabel("ambient core density  nCore  [cm^-3]  (config setting)")
    ax.set_ylabel("boost f_mix needed to reach the 0.95 trigger  [x]")
    ax.set_title("Boost to reach the 0.95 trigger: 1.36 (dense) -> 3.81 (diffuse) -- no constant f_mix fits",
                 fontsize=10.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.2, framealpha=0.92)

    takeaway = (
        "Takeaway: f_mix(n) = 0.95 / (L_cool/L_mech at blowout) -- the boost needed to push TRINITY's resolved loss "
        "to the 0.95 trigger.\n"
        "It rises 1.36 (dense) -> 3.81 (diffuse); no horizontal line (any constant f_mix) crosses all six markers.\n"
        "This is TRINITY's OWN spread -- no literature band needed. DEGENERACY: since the threshold IS 0.95, a flat "
        "theta_lit~=0.95\n"
        "reproduces this very curve -> a constant target adds nothing; only a trajectory-varying theta_target(Da) is "
        "non-degenerate (PLAN.md)."
    )
    fig.subplots_adjust(bottom=0.26, top=0.93, left=0.09, right=0.97)
    fig.text(0.5, 0.015, takeaway, fontsize=7.2, va="bottom", ha="center", family="monospace",
             bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6", alpha=0.95))
    fig.savefig(DST, dpi=150)
    print(f"wrote {DST}\n")
    print("f_mix points (config, nCore, fmix_no_pdv, fmix_with_pdv):")
    for _, r in pts.iterrows():
        print(f"  {r['config']:22s} n={r['nCore']:.0e}  no_pdv={r['fmix_no_pdv']:.2f}  with_pdv={r['fmix_with_pdv']:.2f}")


if __name__ == "__main__":
    main()
