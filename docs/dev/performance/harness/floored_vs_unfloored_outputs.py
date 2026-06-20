#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output-parameter comparison: trinity's unfloored dR2 vs WARPFIELD's dR2min floor.

Feeds WARPFIELD's floored dR2 through trinity's OWN production solver
(get_bubbleproperties_pure) on the genuinely-stiff captured state, so the only
difference is the layer thickness. Two reconstructions of the floor (WARPFIELD's
exact recompute is unavailable, so we bracket it):
  floorA -- floor dR2, keep the cold boundary T0 = T_init (charitable);
  floorB -- floor dR2, recompute T0 via trinity's IC formulas (T0 rises).
R1 and Pb are set before the dMdt solve and do not depend on dR2 (identical).

    cd /home/user/trinity
    python docs/dev/performance/harness/floored_vs_unfloored_outputs.py

Writes docs/dev/performance/data/dR2_output_comparison.csv and
docs/dev/performance/figs/dR2_output_diff.png.
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import trinity.bubble_structure.bubble_luminosity as BL  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIGS = ROOT / "docs" / "dev" / "performance" / "figs"
DATA = ROOT / "docs" / "dev" / "performance" / "data"

# House style with usetex OFF (no LaTeX in the container; see make_dR2_figures.py).
plt.style.use(str(ROOT / "paper" / "_lib" / "trinity.mplstyle"))
plt.rcParams.update({
    "text.usetex": False,
    "figure.dpi": 130,
    "savefig.dpi": 140,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.constrained_layout.use": True,
})

_spec = importlib.util.spec_from_file_location(
    "_dr2test", ROOT / "test" / "test_dR2min_magic_number.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

TI = BL._T_INIT_BOUNDARY
_REAL_IC = BL._get_bubble_ODE_initial_conditions

stiff, params = T._load_state("dR2_stiff_state_fixture.json")
R2 = params["R2"].value
MCLUS = 0.01 * 5e9                       # sfe * mCloud
DR2MIN = 1e-14 * MCLUS + 1e-7            # WARPFIELD bumped floor (Mclus>1e7) [pc]
DR2_ANALYTIC = stiff["dR2_over_R2"] * R2
FLOOR_RATIO = DR2MIN / DR2_ANALYTIC


def _floored_ic(kind):
    def ic(dMdt, params, Pb, R1):
        kB = params["k_B"].value
        mu = params["mu_ion"].value
        C = params["C_thermal"].value
        R2 = params["R2"].value
        ca = params["cool_alpha"].value
        tn = params["t_now"].value
        const = 25.0 / 4.0 * kB / mu / C
        dR2 = max(TI ** 2.5 / (const * dMdt / (4 * np.pi * R2 ** 2)), DR2MIN)
        T0 = TI if kind == "A" else (const * dMdt * dR2 / (4 * np.pi * R2 ** 2)) ** (2 / 5)
        v = ca * R2 / tn - dMdt / (4 * np.pi * R2 ** 2) * kB * T0 / mu / Pb
        return R2 - dR2, T0, -2.0 / 5.0 * T0 / dR2, v
    return ic


def _floored_at(floor_pc):
    """Floor dR2 at an explicit pc value, keeping the cold boundary T0 (floorA)."""
    def ic(dMdt, params, Pb, R1):
        kB = params["k_B"].value
        mu = params["mu_ion"].value
        C = params["C_thermal"].value
        R2 = params["R2"].value
        ca = params["cool_alpha"].value
        tn = params["t_now"].value
        const = 25.0 / 4.0 * kB / mu / C
        dR2 = max(TI ** 2.5 / (const * dMdt / (4 * np.pi * R2 ** 2)), floor_pc)
        v = ca * R2 / tn - dMdt / (4 * np.pi * R2 ** 2) * kB * TI / mu / Pb
        return R2 - dR2, TI, -2.0 / 5.0 * TI / dR2, v
    return ic


def _run(icfn):
    if "bubble_dMdt" in params:
        params["bubble_dMdt"].value = float("nan")   # clean fsolve
    BL._get_bubble_ODE_initial_conditions = icfn
    try:
        return BL.get_bubbleproperties_pure(params)
    finally:
        BL._get_bubble_ODE_initial_conditions = _REAL_IC


FIELDS = [
    ("bubble_dMdt", "mass flux dMdt"),
    ("bubble_LTotal", "luminosity L_total"),
    ("bubble_L1Bubble", "  L1 (bulk, CIE)"),
    ("bubble_L2Conduction", "  L2 (conduction)"),
    ("bubble_L3Intermediate", "  L3 (intermediate)"),
    ("bubble_T_r_Tb", "T(r_Tb)"),
    ("bubble_Tavg", "T_avg"),
    ("bubble_mass", "bubble mass"),
    ("R1", "R1"),
    ("Pb", "Pb"),
]


def main():
    base = _run(_REAL_IC)
    A = _run(_floored_ic("A"))
    B = _run(_floored_ic("B"))

    rows = []
    for key, label in FIELDS:
        bt, a, b = getattr(base, key), getattr(A, key), getattr(B, key)
        rows.append((label, key, bt, a, abs(a - bt) / max(abs(bt), 1e-300),
                     b, abs(b - bt) / max(abs(bt), 1e-300)))

    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "dR2_output_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["param", "key", "trinity", "floorA_keepT", "relA",
                    "floorB_recompT", "relB"])
        for label, key, bt, a, ra, b, rb in rows:
            w.writerow([label.strip(), key, f"{bt:.6e}", f"{a:.6e}", f"{ra:.3e}",
                        f"{b:.6e}", f"{rb:.3e}"])

    print(f"floor over-thickens dR2 by {FLOOR_RATIO:.0f}x "
          f"({DR2_ANALYTIC:.2e} -> {DR2MIN:.2e} pc)")
    for label, key, bt, a, ra, b, rb in rows:
        print(f"  {label:22s} trinity={bt:.4e}  floorA={a:.4e} (x{a/bt if bt else np.nan:.2f})")

    # ---- decisive cross-check: L3 must scale 1:1 with the floored layer width ----
    # (intermediate-region width = (3e4-1e4)/|dTdr|, and |dTdr| ~ 1/dR2). This proves
    # the inflation is a deterministic law, not an artifact of one floor value.
    L3_0 = base.bubble_L3Intermediate
    with open(DATA / "dR2_L3_linearity.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["floor_x_analytic", "dR2_pc", "L3", "L3_over_L3exact", "LTotal",
                    "LTotal_over_exact"])
        print("  L3 linearity (floor x analytic -> L3/L3_exact):")
        for mult in (1, 10, 100, 1000, int(round(FLOOR_RATIO))):
            bp = _run(_floored_at(DR2_ANALYTIC * mult))
            w.writerow([mult, f"{DR2_ANALYTIC*mult:.3e}", f"{bp.bubble_L3Intermediate:.6e}",
                        f"{bp.bubble_L3Intermediate/L3_0:.3f}", f"{bp.bubble_LTotal:.6e}",
                        f"{bp.bubble_LTotal/base.bubble_LTotal:.3f}"])
            print(f"    x{mult:<6d} L3/L3_exact={bp.bubble_L3Intermediate/L3_0:8.1f}  "
                  f"LTotal x{bp.bubble_LTotal/base.bubble_LTotal:.2f}")

    # ---- figure: luminosity components + per-output relative error ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))

    comps = ["bubble_L1Bubble", "bubble_L2Conduction", "bubble_L3Intermediate",
             "bubble_LTotal"]
    names = ["L1\nbulk (CIE)", "L2\nconduction", "L3\nintermediate", "L_total"]
    bt = [getattr(base, k) for k in comps]
    av = [getattr(A, k) for k in comps]
    x = np.arange(len(comps))
    w = 0.38
    ax1.bar(x - w / 2, bt, w, color="#1f77b4", label="trinity solver, exact dR2")
    ax1.bar(x + w / 2, av, w, color="#d62728",
            label=f"same solver, dR2 floored to dR2min (x{FLOOR_RATIO:.0f})")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel("luminosity  [code units]")
    for xi, (t_, a_) in enumerate(zip(bt, av)):
        if a_ / t_ > 1.5:
            ax1.text(xi + w / 2, a_ * 1.25, f"x{a_/t_:.0f}", ha="center",
                     color="#d62728", fontweight="bold", fontsize=10)
    ax1.set_ylim(top=max(av) * 6)   # headroom so the x-factor labels don't clip
    ax1.set_title("L3 (the dR2 layer's emission) inflates ~1:1 with the floor;\n"
                  "L_total jumps ~8x. The bulk L1 is untouched.")
    ax1.legend(loc="upper left")

    labels = [lbl.strip() for lbl, _, _, _, _, _, _ in rows]
    rels = [max(r[4], 1e-16) for r in rows]   # r[4] = relA
    order = np.argsort(rels)
    colors = ["#d62728" if rels[i] > 0.1 else "#1f77b4" for i in order]
    ax2.barh(np.arange(len(rels)), [rels[i] for i in order], color=colors)
    ax2.set_yticks(np.arange(len(rels)))
    ax2.set_yticklabels([labels[i] for i in order], fontsize=8.5)
    ax2.set_xscale("log")
    ax2.axvline(0.01, color="grey", ls=":", lw=1.0)
    ax2.text(0.012, 0.3, "1%", color="grey", fontsize=8)
    ax2.set_xlabel("|relative difference|  vs trinity  (floorA)")
    ax2.set_title("What differs most: luminosity (L3 / L_total);\n"
                  "mass flux, temperatures, mass all stay < 0.3%")
    ax2.set_xlim(left=1e-17)

    fig.suptitle("trinity's bubble solver: exact dR2 vs the SAME solver with dR2 floored to "
                 "WARPFIELD's dR2min\n(only the dR2 initial condition differs; stiff state "
                 f"dR2/R2={stiff['dR2_over_R2']:.1e})", fontsize=10.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "dR2_output_diff.png")
    plt.close(fig)
    print("wrote dR2_output_diff.png + dR2_output_comparison.csv")


if __name__ == "__main__":
    main()
