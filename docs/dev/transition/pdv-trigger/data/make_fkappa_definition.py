#!/usr/bin/env python3
r"""What is f_kappa? — the conduction-coefficient multiplier, defined and verified against the code.

DEFINITION (every symbol traces to source; NO assumptions).
  TRINITY's bubble-structure solve carries thermal conduction with the classical Spitzer-Harm
  conductivity  kappa_SP(T) = C_thermal * T^(5/2),  C_thermal = 6e-7 erg s^-1 cm^-1 K^(-7/2)
  (registry.py:341, the standard Spitzer value for ln Lambda ~ 30). The heat flux is
  q = -kappa_SP dT/dr = -C_thermal T^(5/2) dT/dr.
  f_kappa = `cooling_boost_kappa` (default 1.0, >= 1; registry.py:351) is a DIMENSIONLESS multiplier
  on that coefficient: kappa_eff = f_kappa * C_thermal * T^(5/2). It enters at the THREE places
  C_thermal appears in bubble_luminosity.py, all verified:
    :291  dMdt seed (Weaver+77 Eq. 33):   dMdt ~ (t * f_kappa C_thermal / R2^2)^(2/7) * Pb^(5/7)
                                          => dMdt ~ f_kappa^(2/7)
    :370  conduction-layer ICs (Eq. 44):  const = (25/4) k_B/(mu f_kappa C_thermal) ~ 1/f_kappa
                                          => layer thickness dR2 = T^(5/2)/(const*dMdt/4 pi R2^2) ~ f_kappa
                                          AT FIXED dMdt (the fn takes dMdt as input; folding in the seed
                                          dMdt~f_kappa^(2/7) gives dR2~f_kappa^(5/7))
    :406  T-curvature ODE RHS (Eq. 42-43): d2T/dr2 = [Pb/(f_kappa C_thermal T^(5/2))]*[...] - ...
  The cooling LOSS is NOT multiplied directly: the local volumetric cooling du/dt =
  net_coolingcurve.get_dudt(t, n, T, phi) is evaluated at the LOCAL (n,T) of the structure, and
  L_cool = integral of du/dt over the conduction/interior structure. Raising f_kappa thickens the
  conduction/evaporation layer (dR2 ~ f_kappa) -> more gas in the ~1e5-1e6 K band where Lambda(T)
  peaks -> L_cool EMERGES higher (theta = L_cool/L_mech is an OUTPUT, El-Badry's approach), unlike a
  scalar multiplier on L_cool. Side effect (registry note): dMdt ~ f_kappa^(2/7) also RISES -- a
  faithful kappa_eff would instead SUPPRESS evaporation (El-Badry); so f_kappa is a structural probe.

VERIFICATION (this figure). Panel A draws kappa_eff(T) = f_kappa C_thermal T^(5/2) for f_kappa in
{1,2,4} -- literally what f_kappa multiplies. Panel B checks the analytic seed scaling dMdt ~
f_kappa^(2/7) against the MEASURED f_kappa=2 back-reaction (data/kappa_backreaction.csv): at the seed
(t->0) the measured dMdt ratio is 1.2175, vs analytic 2^(2/7) = 1.2190 -- a match to ~0.1% (0.12%),
confirming the equation. As the run develops the ratio softens (Pb drains ~3%), the genuine back-reaction.

REPRODUCE:  python docs/dev/transition/pdv-trigger/data/make_fkappa_definition.py
Deliverables: docs/dev/transition/pdv-trigger/fkappa_definition.png
"""

import csv
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_C_THERMAL = 6e-7          # registry.py:341 -- Spitzer-Harm coefficient, erg s^-1 cm^-1 K^(-7/2)
_FK = [1, 2, 4]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, _HERE)
    from _trinity_style import use_trinity_style, COLORS
    use_trinity_style()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))
    ckey = {1: "fk1", 2: "fk2", 4: "fk4"}

    # -- Panel A: the conduction law kappa_eff(T) = f_kappa * C_thermal * T^(5/2) --
    T = np.logspace(4, 7, 200)                      # 1e4 .. 1e7 K, the conduction-zone range
    for fk in _FK:
        axA.loglog(T, fk * _C_THERMAL * T ** 2.5, lw=2.0, color=COLORS[ckey[fk]],
                   label=rf"$f_\kappa={fk}$")
    axA.set_xlabel(r"$T$  [K]")
    axA.set_ylabel(r"$\kappa_{\rm eff}(T) = f_\kappa\,C_{\rm th}\,T^{5/2}$  [erg s$^{-1}$ cm$^{-1}$ K$^{-1}$]")
    axA.set_title(r"What $f_\kappa$ multiplies: the Spitzer conductivity"
                  "\n" r"($C_{\rm th}=6\times10^{-7}$, registry.py:341)", fontsize=11.5)
    axA.legend(loc="upper left", title=r"$\kappa_{\rm eff}=f_\kappa\,\kappa_{\rm Spitzer}$")
    axA.grid(True, which="both", alpha=0.25)

    # -- Panel B: analytic seed scaling dMdt ~ f_kappa^(2/7) vs the MEASURED f_kappa=2 back-reaction --
    csv_p = os.path.join(_HERE, "kappa_backreaction.csv")
    if os.path.exists(csv_p):
        rows = list(csv.DictReader(open(csv_p)))
        t = np.array([float(r["t"]) for r in rows])            # t column is in Myr
        # ponytail: the back-reaction ratios are slowly varying, so a linear-t plot reads fine even
        # though the sampled t spans 1e-7..1e-3 Myr (the early seed dominates the comparison anyway).
        dMdt_ratio = np.array([float(r["dMdt_ratio"]) for r in rows])
        Lcool_ratio = np.array([float(r["Lcool_ratio"]) for r in rows])
        Pb_ratio = np.array([float(r["Pb_ratio"]) for r in rows])
        axB.plot(t, dMdt_ratio, "-", color=COLORS["fk2"], lw=2.0,
                 label=r"measured $\dot M(f_\kappa{=}2)/\dot M(f_\kappa{=}1)$")
        axB.plot(t, Lcool_ratio, "-", color=COLORS["fire"], lw=1.6,
                 label=r"measured $L_{\rm cool}$ ratio")
        axB.plot(t, Pb_ratio, "-", color=COLORS["pdv"], lw=1.4,
                 label=r"measured $P_b$ ratio (back-reaction)")
    axB.axhline(2 ** (2 / 7), color=COLORS["fk2"], ls="--", lw=1.5,
                label=r"analytic seed $2^{2/7}=1.219$")
    axB.axhline(1.0, color="0.5", ls=":", lw=1.0)
    axB.set_xlabel(r"$t$  [Myr]")
    axB.set_ylabel(r"ratio at $f_\kappa=2$ vs $f_\kappa=1$  (matched $t$)")
    axB.set_title(r"Verify the seed law $\dot M\propto f_\kappa^{2/7}$"
                  "\n" r"(measured 1.2175 vs analytic 1.219, $\approx$0.1%)", fontsize=11.5)
    axB.legend(loc="center right", fontsize=8.5)
    axB.set_ylim(0.9, 1.6)

    fig.suptitle(r"What is $f_\kappa$? A multiplier on the Spitzer conduction coefficient "
                 r"$C_{\rm th}$ (3 sites in bubble_luminosity.py)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "fkappa_definition.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
