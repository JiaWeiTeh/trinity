#!/usr/bin/env python3
"""Build the "calibrate to theta_lit(n)" diagnostic figure for the pdv-trigger workstream.

ZERO simulation, ZERO physics change. Pure read of committed data:
  - TRINITY resolved loss fraction at blowout, L_cool/L_mech = 1 - cool_at_blowout, per config,
    from pdv_combined_trigger.csv (the same column make_fmix_table.py uses).
  - ambient density per config = nCore, taken from each config's .param (or the schema default
    when the .param does not set it). The nCore source for every config is recorded in NCORE below.

The point of the figure (the workstream's "first step"): TRINITY's resolved cooling fraction has the
RIGHT density trend (rises ~0.25 at diffuse n -> ~0.70 at dense n). The literature overlay is SCHEMATIC
(El-Badry+2019, Lancaster+2021 PDFs not yet digitized -- see below), so this figure does NOT quantify a
gap to theta_lit(n): recomputing band_center - TRINITY at each nCore gives a non-monotonic, even
negative-at-diffuse result, an artifact of the stand-in band shape. The case that a single constant boost
f_mix cannot bridge the shortfall rests on the fmix table (f_mix 1.36 dense -> 3.81 diffuse), motivating a
density-dependent theta_lit(n) calibration instead of tuning one constant to the 0.95 trigger threshold.

LITERATURE OVERLAY IS SCHEMATIC / TO-VERIFY.  The El-Badry/Lancaster relations are uncertain and
EXTRAPOLATED at GMC-core density; their exact equation/figure numbers (El-Badry Eq.35 / Fig.7,
Lancaster Eq.39) are the three citations flagged for a PDF check in NOTE_PATCHES.md (Patch 4).
WebFetch of the arXiv/IOP PDFs returned HTTP 403 in this session, so the overlay below is the
QUALITATIVE published trend (cooling fraction rises with ambient density, approaching ~1 at high n),
NOT digitized curve values. It is drawn as a schematic band and labelled "to-verify against PDF".
Do not cite the band's numeric values without checking the PDFs.

CAVEAT carried onto the figure: TRINITY's x-axis is the AMBIENT core density nCore, whereas the
literature theta(n) is a function of the INTERFACE / compressed density at the mixing layer, which is
higher than the ambient. Plotting both against nCore is the honest first cut but is not an
apples-to-apples density axis -- the coupled upgrade (theta_target(Da), PLAN.md) uses the interface
density n_int, not nCore.

Run from the repo root:
  python docs/dev/transition/pdv-trigger/data/make_theta_density_plot.py
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless; no LaTeX -- plain-text labels only (mathtext is fine, usetex off)
import matplotlib.pyplot as plt

HERE = "docs/dev/transition/pdv-trigger"
SRC = f"{HERE}/data/pdv_combined_trigger.csv"
DST = f"{HERE}/theta_vs_density.png"

THRESH = 0.95  # the energy->momentum trigger threshold ((Lmech - Lloss)/Lmech < 0.05  <=>  Lloss/Lmech > 0.95)
THETA_MAX = 0.95  # ceiling on theta_lit at GMC-core n: El-Badry's sqrt(rho) scaling is an extrapolation there

# ambient density per config = nCore [cm^-3], with the source of each value.
#   ".param" -> set explicitly in docs/dev/transition/cleanroom/configs/<config>.param
#   "default" -> NOT set in param/simple_cluster.param; falls back to schema default nCore=1e5
#                in trinity/_input/default.param (line 75).
# Cross-checked against the reconstructed metadata_humanreadable.txt in pt4/r1shadow/runs/<config>/.
NCORE = {
    "large_diffuse_lowsfe": (1e2, "cleanroom .param"),
    "be_sphere":            (1e4, "cleanroom .param"),
    "midrange_pl0":         (1e4, "cleanroom .param"),
    "pl2_steep":            (1e5, "cleanroom .param"),
    "simple_cluster":       (1e5, "default.param (param/simple_cluster.param sets no nCore)"),
    "small_dense_highsfe":  (1e6, "cleanroom .param"),
}


def trinity_points():
    """L_cool/L_mech = 1 - cool_at_blowout, per NORMAL config (the 6 that reach blowout)."""
    d = pd.read_csv(SRC).set_index("config")
    rows = []
    for cfg, (n, _src) in NCORE.items():
        cool = d.loc[cfg, "cool_at_blowout"]
        rows.append((cfg, n, 1.0 - float(cool)))
    return pd.DataFrame(rows, columns=["config", "nCore", "Lcool_over_Lmech"])


def theta_lit_band(n):
    """SCHEMATIC theta_lit(n) -- QUALITATIVE published trend, NOT digitized curve values.

    ponytail: schematic stand-in (known ceiling: not the real El-Badry/Lancaster curves; the exact
    relations need a PDF check -- NOTE_PATCHES.md Patch 4). Encodes only what the abstracts assert:
      * El-Badry+2019 (arXiv:1902.09547): interface cooling efficiency theta = L_int/Edot_in rises
        with ambient density (sqrt(rho)-like growth at low n); evaporation suppressed ~3-30x vs Weaver.
      * Lancaster+2021 (arXiv:2104.07722): at the dense star-forming-cloud conditions they study the
        retained fraction (1-theta) ~ 0.1-0.01, i.e. theta ~ 0.90-0.99 (near-complete cooling).
    Upgrade path: replace this with digitized El-Badry/Lancaster theta(n) once the PDFs are checked.
    Capped at THETA_MAX because the sqrt(rho) scaling is an EXTRAPOLATION at GMC-core n.
    """
    n = np.asarray(n, dtype=float)
    # monotone-rising, saturating toward ~1; anchored so it is ~0.5 near n~1e2 and ~0.9 near n~1e4-5,
    # consistent with Lancaster's theta~0.9-0.99 at dense SF-cloud density. Band = +-0.1 schematic spread.
    center = THETA_MAX * np.sqrt(n) / np.sqrt(n + 3e3)
    lo = np.clip(center - 0.10, 0.0, THETA_MAX)
    hi = np.clip(center + 0.10, 0.0, THETA_MAX)
    return np.clip(center, 0.0, THETA_MAX), lo, hi


def main():
    pts = trinity_points()

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # --- schematic literature band theta_lit(n) ---
    ngrid = np.logspace(1, 6.3, 200)
    cen, lo, hi = theta_lit_band(ngrid)
    ax.fill_between(ngrid, lo, hi, color="tab:orange", alpha=0.18, lw=0,
                    label="theta_lit(n) band (SCHEMATIC, to-verify vs PDF)")
    ax.plot(ngrid, cen, color="tab:orange", lw=2.0, alpha=0.85,
            label="theta_lit(n) El-Badry+19 / Lancaster+21 (schematic)")

    # --- 0.95 trigger threshold + theta_max ceiling ---
    ax.axhline(THRESH, color="crimson", ls="--", lw=1.6,
               label="energy->momentum trigger threshold (Lloss/Lmech = 0.95)")
    # ceiling annotation at GMC-core densities (where El-Badry sqrt(rho) is an extrapolation).
    # Upper-RIGHT (near where the ceiling applies, high n), in the headroom (ylim->1.18) so it is
    # not clipped and leaves the upper-left clear for the legend.
    ax.annotate("theta_max = 0.95 ceiling at GMC-core n\n(El-Badry sqrt(rho) extrapolated here)",
                xy=(8e5, THETA_MAX), xytext=(2.5e4, 1.05),
                fontsize=8, color="crimson", ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0))

    # --- TRINITY resolved points ---
    # SCATTER ONLY -- deliberately NOT connected. These are 6 independent configs
    # (different mCloud/sfe/profile), not one density sweep; a connecting line would
    # imply a continuous functional dependence on nCore that does not exist. The
    # density trend is read from the marker positions, not drawn as a curve.
    ax.scatter(pts["nCore"], pts["Lcool_over_Lmech"], s=95, color="tab:blue",
               zorder=5, edgecolor="k", linewidth=0.6,
               label="TRINITY resolved L_cool/L_mech at blowout (one marker per config)")
    for _, r in pts.iterrows():
        dy = 0.028 if r["config"] != "simple_cluster" else -0.05
        ax.annotate(r["config"], (r["nCore"], r["Lcool_over_Lmech"]),
                    textcoords="offset points", xytext=(6, 6 if dy > 0 else -12),
                    fontsize=7.5, color="tab:blue")

    # --- gap arrows/labels intentionally OMITTED ---
    # A schematic band cannot support a quantified gap. Recomputing band_center - TRINITY at each
    # nCore gives a NON-MONOTONIC, even NEGATIVE-at-diffuse result (the sqrt(n)/sqrt(n+3e3) stand-in
    # dips below TRINITY at n~1e2: gap=-0.08), so the old "gap ~0.45 diffuse / ~0.25 dense" arrows were
    # wrong. Quote no gap until the El-Badry/Lancaster theta(n) is digitized (NOTE_PATCHES.md Patch 4);
    # the constant-f_mix-can't-bridge-it argument lives in the fmix table, not in this band.

    ax.set_xscale("log")
    ax.set_xlim(7e1, 5e6)
    ax.set_ylim(0.0, 1.18)  # headroom above the 0.95 threshold so the ceiling note is not clipped
    ax.set_xlabel("ambient core density  nCore  [cm^-3]  (config setting)")
    ax.set_ylabel("loss fraction at blowout   L_cool / L_mech   (= 1 - cool_at_blowout)")
    ax.set_title("TRINITY resolved cooling vs theta_lit(n): right density trend (overlay SCHEMATIC, gap not quantified)")
    ax.grid(True, which="both", alpha=0.25)
    # legend upper-left (clear region between the rising band and the threshold); takeaway lives
    # BELOW the axes (fig.text) so the two never overlap.
    ax.legend(loc="upper left", fontsize=7.4, framealpha=0.92)

    # takeaway box -- placed OUTSIDE the axes (figure bottom margin) so it cannot collide with the
    # legend or any data/threshold text.
    takeaway = (
        "Takeaway: TRINITY's L_cool/L_mech rises ~0.25 (diffuse) -> ~0.70 (dense) -- the SAME density trend as "
        "theta_lit(n).\n"
        "The literature band here is SCHEMATIC (El-Badry/Lancaster PDFs not yet digitized), so NO gap is "
        "quantified on this figure.\n"
        "That a CONSTANT cooling boost cannot bridge the density-varying shortfall is shown by the fmix table "
        "(f_mix 1.36 dense -> 3.81 diffuse),\n"
        "not by this band -> motivates a density-dependent theta_lit(n) calibration.   "
        "CAVEAT: x-axis is AMBIENT nCore; theta_lit(n)\n"
        "depends on the higher INTERFACE density."
    )
    fig.subplots_adjust(bottom=0.27, top=0.93, left=0.09, right=0.97)
    fig.text(0.5, 0.015, takeaway, fontsize=7.3, va="bottom", ha="center", family="monospace",
             bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6", alpha=0.95))
    fig.savefig(DST, dpi=150)
    print(f"wrote {DST}\n")
    print("TRINITY points (config, nCore [cm^-3], L_cool/L_mech, nCore source):")
    for _, r in pts.sort_values("nCore").iterrows():
        print(f"  {r['config']:22s} n={r['nCore']:.0e}  Lcool/Lmech={r['Lcool_over_Lmech']:.3f}"
              f"   [{NCORE[r['config']][1]}]")


if __name__ == "__main__":
    main()
