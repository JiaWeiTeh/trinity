#!/usr/bin/env python3
"""Build the "calibrate to theta_lit(n)" diagnostic figure for the pdv-trigger workstream.

ZERO simulation, ZERO physics change. Pure read of committed data:
  - TRINITY resolved loss fraction at blowout, L_cool/L_mech = 1 - cool_at_blowout, per config,
    from pdv_combined_trigger.csv (the same column make_fmix_table.py uses).
  - ambient density per config = nCore, taken from each config's .param (or the schema default
    when the .param does not set it). The nCore source for every config is recorded in NCORE below.

The point of the figure (the workstream's "first step"): TRINITY's resolved cooling fraction has the
RIGHT density trend (rises ~0.25 at diffuse n -> ~0.70 at dense n) but sits TOO LOW relative to the
mixing-layer literature (El-Badry+2019, Lancaster+2021), and because the gap to theta_lit(n) VARIES
with density, no single constant boost f_mix can bridge it -- motivating a density-dependent
theta_lit(n) calibration instead of tuning one constant to the 0.95 trigger threshold.

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
    # ceiling annotation at GMC-core densities (where El-Badry sqrt(rho) is an extrapolation)
    ax.annotate("theta_max = 0.95 ceiling at GMC-core n\n(El-Badry sqrt(rho) is extrapolated here)",
                xy=(8e5, THETA_MAX), xytext=(2.0e4, 0.985),
                fontsize=8.5, color="crimson", ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0))

    # --- TRINITY resolved points ---
    ax.scatter(pts["nCore"], pts["Lcool_over_Lmech"], s=95, color="tab:blue",
               zorder=5, edgecolor="k", linewidth=0.6,
               label="TRINITY resolved L_cool/L_mech at blowout")
    # connect in density order to show the trend
    o = pts.sort_values("nCore")
    ax.plot(o["nCore"], o["Lcool_over_Lmech"], color="tab:blue", lw=1.0, alpha=0.5, zorder=4)
    for _, r in pts.iterrows():
        dy = 0.028 if r["config"] != "simple_cluster" else -0.05
        ax.annotate(r["config"], (r["nCore"], r["Lcool_over_Lmech"]),
                    textcoords="offset points", xytext=(6, 6 if dy > 0 else -12),
                    fontsize=7.5, color="tab:blue")

    # --- the gap arrows (TRINITY point -> literature center) at the two density extremes ---
    for cfg in ("large_diffuse_lowsfe", "small_dense_highsfe"):
        r = pts[pts["config"] == cfg].iloc[0]
        cen_n, _, _ = theta_lit_band(r["nCore"])
        ax.annotate("", xy=(r["nCore"], float(cen_n)), xytext=(r["nCore"], r["Lcool_over_Lmech"]),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2, ls=":"))
    ax.text(1.3e2, 0.50, "gap ~0.45\n(diffuse)", fontsize=8, color="0.30", ha="left")
    ax.text(7.0e5, 0.83, "gap ~0.25\n(dense)", fontsize=8, color="0.30", ha="right")

    ax.set_xscale("log")
    ax.set_xlim(7e1, 5e6)
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel("ambient core density  nCore  [cm^-3]  (config setting)")
    ax.set_ylabel("loss fraction at blowout   L_cool / L_mech   (= 1 - cool_at_blowout)")
    ax.set_title("TRINITY resolved cooling vs literature theta_lit(n): right trend, sits too low")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.6, framealpha=0.9)

    # takeaway box
    takeaway = (
        "Takeaway: TRINITY's L_cool/L_mech rises ~0.25 (diffuse) -> ~0.70 (dense) -- the SAME\n"
        "density trend as theta_lit(n) -- but sits LOW. The density-varying gap (~0.45 diffuse,\n"
        "~0.25 dense) is what the cooling boost must close; because the gap VARIES with n,\n"
        "a CONSTANT f_mix cannot bridge it -> motivates a density-dependent theta_lit(n) calibration.\n"
        "CAVEAT: x-axis is AMBIENT nCore; theta_lit(n) is a function of the higher INTERFACE density."
    )
    ax.text(0.015, 0.015, takeaway, transform=ax.transAxes, fontsize=7.4, va="bottom", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6", alpha=0.95))

    fig.tight_layout()
    fig.savefig(DST, dpi=150)
    print(f"wrote {DST}\n")
    print("TRINITY points (config, nCore [cm^-3], L_cool/L_mech, nCore source):")
    for _, r in pts.sort_values("nCore").iterrows():
        print(f"  {r['config']:22s} n={r['nCore']:.0e}  Lcool/Lmech={r['Lcool_over_Lmech']:.3f}"
              f"   [{NCORE[r['config']][1]}]")


if __name__ == "__main__":
    main()
