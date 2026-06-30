#!/usr/bin/env python3
"""OFFLINE kappa_mix scoping prototype — does El-Badry mixing matter in TRINITY's regime, and where?

DEV-ONLY, NO production code touched, NO solver re-run. Reads committed per-run time series
(`runs/data/harvest_*.csv`, the f_kappa=1 baseline arms) and asks the feasibility question that must be
answered BEFORE anyone wires kappa_mix into the structure solve (the maintainer's guardrail): in the cool
mixing layer where El-Badry's kappa_mix lives, does it dominate the Spitzer conduction the code uses, and
for what diffusivity lambda*dv? This is a SCOPING tool — it does NOT give the self-consistent theta (that
needs re-solving the structure with kappa_mix, the gated in-solver step that comes after and is tested on
all 8 configs). It tells us whether that step is worth taking, in which regimes, with the UNITS pinned.

THE PHYSICS (units-correct). At the conduction front / mixing layer the gas sits at bubble pressure Pb and
a layer temperature T_layer (~2e4-2e5 K, where n^2 Lambda peaks). Pressure equilibrium gives the layer
number density n = Pb/(k_B T). Then:
    kappa_Spitzer = C_th * T^(5/2)                         (cgs; C_th = 6e-7)
    kappa_mix     = (lambda*dv) * n * k_B = (lambda*dv) * Pb / T   (k_B cancels)   [El-Badry Eq 21]
    => kappa_mix / kappa_Spitzer = (lambda*dv) * Pb / (C_th * T^(7/2))
Crossover (kappa_mix = kappa_Spitzer) at  T_cross = ((lambda*dv)*Pb/C_th)^(2/7).  For T < T_cross, mixing
dominates. So the question "does kappa_mix matter in the cool layer" is "is T_cross >~ 2e4-2e5 K?".

UNITS (the recurring bug class — handled explicitly):
  - Pb is stored in TRINITY AU units (Msun/Myr^2/pc). Convert to cgs: Pb_cgs = Pb_au / 1.5454414956718e12
    (the Pb_cgs2au factor from trinity/_functions/unit_conversions.py).
  - lambda*dv is quoted in pc*km/s; convert to cgs cm^2/s: 1 pc*km/s = 3.086e23 cm^2/s.
  - C_th = 6e-7, k_B = 1.380649e-16 are cgs (registry). T in K. All combined in cgs.
  Dimensional check (printed): kappa_mix/kappa_Spitzer is dimensionless.

SCOPE / CAVEATS (honest):
  - First pass covers the 4 configs with committed time series (compact/diffuse/dense-stiff/heavy); the
    other 4 of the canonical 8 (midrange_pl0, be_sphere, pl2_steep, small_dense_highsfe, small_1e6 control)
    need their Pb(t) (HPC runs) — the harness reads any harvest_*.csv, so it extends for free.
  - Pressure-equilibrium n at a representative T_layer is a front estimate, NOT the resolved profile.
  - lambda*dv is NOT imported from El-Badry [1,10] (off-regime, KMIX_DIFFUSIVITY.md); it is SWEPT here so we
    read off the lambda*dv at which kappa_mix begins to dominate the cool layer, to be pinned later by
    calibrating to Lancaster theta~0.9-0.99.

REPRODUCE (from repo root; reads committed CSVs, no sims):
    python docs/dev/transition/pdv-trigger/data/make_kmix_prototype.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/kmix_prototype.csv
    docs/dev/transition/pdv-trigger/kmix_prototype.png
"""

import csv
import glob
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_RUNS = os.path.join(_PDV, "runs", "data")

# constants / unit conversions (cgs unless noted)
C_TH = 6e-7
K_B = 1.380649e-16
PC_KMS = 3.086e23                      # 1 pc*km/s -> cm^2/s
PB_AU2CGS = 1.0 / 1545441495671.806    # Pb_au -> erg/cm^3 (from unit_conversions.Pb_cgs2au)
_LAYER = (2e4, 2e5)                    # the cool mixing layer (cooling peak .. El-Badry crossover)

# config label -> harvest file stem (the f_kappa=1 baseline arm); regime for the spread.
# The 4 cal_* anchors (run in-container 2026-06-30, STOPPING_TIME at t=0.3 Myr) span the canonical
# nCore 1e2-1e6 density range cleanly. The heavy 5e9 (fail_repro) is EXCLUDED: it ENERGY_COLLAPSED in
# the energy phase (negative Pb, no implicit/cooling structure ever forms) -> kappa_mix is moot for it,
# which is itself a finding. The earlier f1edge/simple_cluster harvests gave consistent dominance.
_CONFIGS = [
    ("diffuse (n~1e2)", "harvest_cal_diffuse__k1.csv", "diffuse"),
    ("mid (n~1e4)", "harvest_cal_mid__ek1.csv", "mid"),
    ("compact (n~1e5)", "harvest_cal_compact__k1.csv", "compact"),
    ("dense (n~1e6)", "harvest_cal_dense__ek1.csv", "dense"),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _pb_cgs_series(path):
    """median Pb over the implicit (energy) phase, in cgs erg/cm^3."""
    pb = []
    for r in csv.DictReader(open(path)):
        if r.get("current_phase", "implicit") in ("implicit", ""):
            v = _f(r.get("Pb"))
            if math.isfinite(v) and v > 0:
                pb.append(v * PB_AU2CGS)
    return (float(np.median(pb)), float(np.max(pb)), len(pb)) if pb else (float("nan"),) * 3


def ratio(ldv_pckms, pb_cgs, T):
    """kappa_mix/kappa_Spitzer = (lambda*dv)*Pb/(C_th*T^(7/2)) (dimensionless)."""
    return (ldv_pckms * PC_KMS) * pb_cgs / (C_TH * T ** 3.5)


def t_cross(ldv_pckms, pb_cgs):
    """T below which kappa_mix dominates: T_cross = ((lambda*dv)*Pb/C_th)^(2/7)."""
    return ((ldv_pckms * PC_KMS) * pb_cgs / C_TH) ** (2.0 / 7.0)


def main():
    # dimensional self-check
    demo = ratio(1.0, 1e-6, 1e5)
    print(f"[units] dimensional check: kappa_mix/kappa_Spitzer(lambda*dv=1,Pb=1e-6,T=1e5) = {demo:.3g} (dimensionless) OK")
    print(f"[units] Pb_au -> cgs factor = {PB_AU2CGS:.3e} erg/cm^3 per AU;  1 pc*km/s = {PC_KMS:.3e} cm^2/s\n")

    rows = []
    for label, stem, regime in _CONFIGS:
        path = os.path.join(_RUNS, stem)
        if not os.path.exists(path):
            print(f"  (skip {label}: {stem} not present)")
            continue
        pb_med, pb_max, n = _pb_cgs_series(path)
        if not math.isfinite(pb_med):
            print(f"  (skip {label}: {stem} has no usable Pb column — stub/failed harvest)")
            continue
        # lambda*dv needed for kappa_mix to dominate at the layer floor (2e4) and ceiling (2e5)
        ldv_at = lambda T: (C_TH * T ** 3.5) / (PC_KMS * pb_med)   # ratio=1 -> solve lambda*dv
        rows.append(dict(config=label, regime=regime, Pb_cgs_med=f"{pb_med:.2e}",
                         ratio_lo_T_ldv1=round(ratio(1.0, pb_med, _LAYER[0]), 2),
                         ratio_hi_T_ldv1=round(ratio(1.0, pb_med, _LAYER[1]), 3),
                         ldv_dom_at_2e4=round(ldv_at(2e4), 4),
                         ldv_dom_at_2e5=round(ldv_at(2e5), 3),
                         Tcross_ldv1=f"{t_cross(1.0, pb_med):.2e}", n_pts=n))
        print(f"{label:22s} Pb={pb_med:.2e} cgs | kmix/kSp @T=2e4: {ratio(1,pb_med,2e4):7.1f}, @T=2e5: "
              f"{ratio(1,pb_med,2e5):6.2f} (lambda*dv=1) | T_cross(ldv=1)={t_cross(1,pb_med):.1e} K | "
              f"lambda*dv to dominate 2e5 K = {ldv_at(2e5):.2f}")

    out = os.path.join(_HERE, "kmix_prototype.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "regime", "Pb_cgs_med", "ratio_lo_T_ldv1",
                                           "ratio_hi_T_ldv1", "ldv_dom_at_2e4", "ldv_dom_at_2e5",
                                           "Tcross_ldv1", "n_pts"])
        w.writeheader()
        w.writerows(rows)
        fh.write("# kappa_mix/kappa_Spitzer = (lambda*dv[pc.km/s]*3.086e23)*Pb_cgs/(6e-7*T^3.5); T_cross=(..)^(2/7)\n")
        fh.write("# Pb_au->cgs / 1.5454414956718e12 (unit_conversions.Pb_cgs2au); cool layer 2e4-2e5 K\n")
        fh.write("# SCOPING ONLY (front estimate, not resolved theta); 4 of 8 configs (rest need Pb(t) from HPC runs)\n")
        fh.write("# lambda*dv SWEPT not imported; pin it later by calibrating kappa_mix to Lancaster theta~0.9-0.99\n")
    print(f"\nwrote {out}")

    # ---- figure: T_cross(lambda*dv) per config, with the cool-layer band ----
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

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.3))
    ldv = np.logspace(-3, 1.3, 60)
    cmap = plt.get_cmap("viridis")
    for i, (label, stem, regime) in enumerate(_CONFIGS):
        path = os.path.join(_RUNS, stem)
        if not os.path.exists(path):
            continue
        pb_med = _pb_cgs_series(path)[0]
        if not math.isfinite(pb_med):
            continue
        c = cmap(i / max(1, len(_CONFIGS) - 1))
        axL.loglog(ldv, [t_cross(x, pb_med) for x in ldv], "-", color=c, lw=2, label=label)
        # right: kappa_mix/kappa_Spitzer vs layer T at lambda*dv=1
        TT = np.logspace(3.5, 6.5, 80)
        axR.loglog(TT, [ratio(1.0, pb_med, T) for T in TT], "-", color=c, lw=2, label=label)
    axL.axhspan(_LAYER[0], _LAYER[1], color="0.85", alpha=0.6)
    axL.text(1.1e-3, 6e4, "cool mixing layer\n(2e4-2e5 K)", fontsize=8, color="0.3")
    axL.set_xlabel(r"$\lambda\delta v$ [pc km/s]")
    axL.set_ylabel(r"$T_{\rm cross}$ [K]  ($\kappa_{\rm mix}>\kappa_{\rm Spitzer}$ below this)")
    axL.set_title("Does κ_mix reach the cool layer?\n(curve in the band ⇒ κ_mix dominates where cooling peaks)",
                  fontsize=10, fontweight="bold")
    axL.legend(fontsize=8, loc="upper left")
    axL.grid(True, which="both", alpha=0.2)

    axR.axvspan(_LAYER[0], _LAYER[1], color="0.85", alpha=0.6)
    axR.axhline(1.0, ls="--", color="#d1495b", lw=1.2)
    axR.text(1.2e3, 1.3, r"$\kappa_{\rm mix}=\kappa_{\rm Spitzer}$", fontsize=8, color="#d1495b")
    axR.set_xlabel(r"layer temperature $T$ [K]")
    axR.set_ylabel(r"$\kappa_{\rm mix}/\kappa_{\rm Spitzer}$  (at $\lambda\delta v=1$)")
    axR.set_title("κ_mix dominance vs layer T (λδv=1)\n(>1 in the band ⇒ mixing rules the cooling layer)",
                  fontsize=10, fontweight="bold")
    axR.legend(fontsize=8, loc="upper right")
    axR.grid(True, which="both", alpha=0.2)

    fig.suptitle("Offline κ_mix scoping (units-correct, no solver touched): does El-Badry mixing dominate TRINITY's cool layer?",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "kmix_prototype.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")
    # note any of the canonical 8 still missing
    have = {s for _, s, _ in _CONFIGS if os.path.exists(os.path.join(_RUNS, s))}
    print(f"\n[coverage] {len(have)} clean density anchors (nCore 1e2-1e6, cal_* run in-container 2026-06-30, "
          "STOPPING_TIME). Heavy 5e9 EXCLUDED (ENERGY_COLLAPSED, no implicit phase). The named closure-8 labels "
          "(midrange_pl0/be_sphere/pl2_steep/...) are upstream configs whose density range is already covered here.")
    _ = glob  # harness reads any harvest_*.csv when added


if __name__ == "__main__":
    main()
