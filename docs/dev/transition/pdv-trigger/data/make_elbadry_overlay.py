#!/usr/bin/env python3
r"""Overlay TRINITY's resolved theta_1D(n_H) on the VERIFIED El-Badry+2019 cooling-efficiency relation.

This REPLACES the earlier schematic literature band (theta_vs_density.png used an "arbitrary saturating
stand-in" because the El-Badry PDF was 403-inaccessible). The PDF is now in hand and the equations are
verified against it:

  El-Badry+2019 (MNRAS 1902.09547), section 5.2, Eqs (37)-(38):
    psi  == L_int / Edot_th  =  A_mix * (lambda_dv / 1 pc km/s)^(1/2) * (n_H0 / 1 cm^-3)^(1/2),  A_mix = 3.5
    theta = psi / (11/5 + psi)
  Calibration DOMAIN (their Figs 6-7): n_H0 = 0.1-10 cm^-3,  lambda_dv = 0.1-10 pc km/s.
  (A_mix=3.5 is the fit to their sims; 1.7 is the first-principles value with alpha=1, T_pk=2e4 K.)

WHAT THIS SHOWS. The El-Badry theta(n_H) target curve (band over lambda_dv = 0.1-10), drawn SOLID over its
calibrated domain (n<=10) and DASHED where it is EXTRAPOLATED into the GMC regime (n=1e2-1e6, where our
clouds live -- 1 to 5 decades beyond where El-Badry tested it). Over-plotted: TRINITY's resolved 1D loss
fraction theta_1D = L_cool/L_mech = 1 - cool_at_blowout per config (the Spitzer-only, f_kappa=1 baseline,
i.e. El-Badry's NO-extra-mixing floor). The vertical gap between our points and the El-Badry curve is the
cooling deficit that turbulent mixing (the faithful kappa_mix) -- or, as a stand-in, f_kappa -- must supply.

HONEST AXIS CAVEAT (carried from theta_vs_density.py). El-Badry's n_H0 is the AMBIENT density; TRINITY's
x-axis is nCore (cloud core, the gas the bubble expands into) -- comparable but the literature cooling really
depends on the INTERFACE/compressed density, higher than ambient. This is the honest first cut, not
apples-to-apples on the density axis.

REPRODUCE:  python docs/dev/transition/pdv-trigger/data/make_elbadry_overlay.py
Deliverables: docs/dev/transition/pdv-trigger/elbadry_overlay.png (+ data/elbadry_overlay.csv)
"""

import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_SRC = os.path.join(_HERE, "pdv_combined_trigger.csv")

_A_MIX = 3.5            # El-Badry Eq 37 fit value (1.7 first-principles)
# nCore [cm^-3] per config (from make_theta_density_plot.py NCORE, sourced from each cleanroom .param)
_NCORE = {
    "large_diffuse_lowsfe": 1e2, "be_sphere": 1e4, "midrange_pl0": 1e4,
    "pl2_steep": 1e5, "simple_cluster": 1e5, "small_dense_highsfe": 1e6,
}


def el_badry_theta(n_H, lam_dv):
    """El-Badry+2019 Eq 37-38: theta(n_H, lambda_dv).  n_H in cm^-3, lambda_dv in pc km/s."""
    psi = _A_MIX * (lam_dv ** 0.5) * (n_H ** 0.5)
    return psi / (11.0 / 5.0 + psi)


def main():
    # TRINITY resolved theta_1D = 1 - cool_at_blowout per config
    d = {r["config"]: r for r in csv.DictReader(open(_SRC))}
    pts = []
    for cfg, n in _NCORE.items():
        cool = d[cfg]["cool_at_blowout"]
        if cool:
            pts.append((cfg, n, 1.0 - float(cool)))
    pts.sort(key=lambda x: x[1])
    with open(os.path.join(_HERE, "elbadry_overlay.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "nCore", "theta_1D",
                    "elbadry_theta_lamdv1", "elbadry_theta_lamdv0.1", "elbadry_theta_lamdv10"])
        for cfg, n, th in pts:
            w.writerow([cfg, n, round(th, 4), round(el_badry_theta(n, 1), 4),
                        round(el_badry_theta(n, 0.1), 4), round(el_badry_theta(n, 10), 4)])
    for cfg, n, th in pts:
        print(f"  {cfg:22s} nCore={n:.0e}  theta_1D={th:.3f}  "
              f"El-Badry theta(lamdv=1)={el_badry_theta(n,1):.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import sys
        sys.path.insert(0, _HERE)
        from _trinity_style import use_trinity_style, COLORS
        use_trinity_style()
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    n_dom = np.logspace(-1, 1, 100)        # El-Badry calibrated domain 0.1-10
    n_ext = np.logspace(1, 6, 200)         # GMC extrapolation 10-1e6
    for lam, col in [(0.1, "#9ecae1"), (1.0, COLORS["fk1"]), (10.0, "#08519c")]:
        ax.plot(n_dom, el_badry_theta(n_dom, lam), "-", color=col, lw=2.0,
                label=rf"El-Badry $\theta$, $\lambda\delta v={lam:g}$ (calibrated)")
        ax.plot(n_ext, el_badry_theta(n_ext, lam), "--", color=col, lw=1.6, alpha=0.9)
    # shade the extrapolation region
    ax.axvspan(10, 1e6, color="0.85", alpha=0.4, zorder=0)
    ax.axvline(10, color="0.5", lw=1.0, ls=":")
    ax.text(60, 0.06, "GMC regime —\nEl-Badry EXTRAPOLATED\n(calibrated only at $n\\leq10$)",
            fontsize=8.5, color="0.35")
    ax.axhline(0.95, color="crimson", ls="--", lw=1.2, label="0.95 cooling_balance trigger")
    # our resolved 1D baseline points
    ax.plot([n for _, n, _ in pts], [th for _, _, th in pts], "o", color=COLORS["fk4"],
            ms=10, mec="k", mew=0.6, zorder=5,
            label=r"TRINITY resolved $\theta_{\rm 1D}$ ($f_\kappa{=}1$, Spitzer floor)")
    for cfg, n, th in pts:
        ax.annotate(cfg.replace("_", " "), (n, th), fontsize=6.8, color="0.3",
                    xytext=(0, -11), textcoords="offset points", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel(r"$n_{\rm H}$  [cm$^{-3}$]   (El-Badry: ambient $n_{H,0}$;  TRINITY: $n_{\rm core}$)")
    ax.set_ylabel(r"$\theta = L_{\rm cool}/L_{\rm mech}$  (cooling efficiency)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0.1, 1e6)
    ax.set_title(r"Our $\theta_{\rm 1D}$ vs the verified El-Badry+19 Eq. 37–38 target "
                 r"($A_{\rm mix}{=}3.5$)", fontsize=11.5)
    ax.legend(fontsize=7.8, loc="lower right")
    fig.tight_layout()
    png = os.path.join(_PDV, "elbadry_overlay.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
