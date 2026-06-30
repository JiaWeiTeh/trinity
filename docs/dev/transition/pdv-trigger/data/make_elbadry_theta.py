#!/usr/bin/env python3
"""El-Badry+2019 analytic theta(lambda*dv, n) calculator -- the candidate TRINITY theta_target.

THE POINT (ELBADRY_REFERENCE.md sec 4 + 7). El-Badry derives a CALIBRATED closed form for the interface
cooling efficiency theta = L_int/Edot_in, which is EXACTLY TRINITY's trigger quantity theta = L_cool/L_mech
(Edot_in = ESN/dt_SNe = L_mech). So instead of porting kappa_mix into TRINITY's Weaver structure ODE (the
saturating, unstable path -- KMIX_SELFCONSISTENT.md), we can feed his theta(lambda*dv, n) DIRECTLY as the
theta_target. This script computes that prescription across TRINITY's density range, checks the Lancaster
anchor, and reports what lambda*dv reproduces theta ~ 0.9-0.99, with the honest extrapolation caveats.

THE FORMULA (El-Badry Eq 37/38):
    X        = A_mix * (lambda*dv)^(1/2) * n^(1/2)        # = L_int/Edot_th ; lambda*dv in pc.km/s, n in cm^-3
    theta    = X / (11/5 + X)                              # 11/5 = 2.2
    A_mix    = 3.5  (fit to sims) ; 1.7 (analytic, alpha=1, T_pk=2e4 K)

HONEST CAVEATS (do not assume): (a) El-Badry TESTED only n_H,0 in [0.1, 10] cm^-3; TRINITY GMC/shell densities
are far higher -- the sqrt(n) form is EXTRAPOLATED. Supports for the extrapolation: El-Badry himself notes high
theta at molecular-cloud densities, and Lancaster (3D) finds theta~0.9-0.99 at GMC density. (b) theta is a
LATE-TIME (t>~3 Myr) equilibrium. (c) n is the AMBIENT (pre-shock) density at the shell, NOT necessarily nCore.
(d) lambda*dv in [0.1,10] is uncertain; <1 is numerically unconverged in El-Badry's grid.

REPRODUCE (from repo root; pure calculator, no sims, no TRINITY imports):
    python docs/dev/transition/pdv-trigger/data/make_elbadry_theta.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/elbadry_theta.csv
    docs/dev/transition/pdv-trigger/elbadry_theta.png
"""
import csv
import math
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)

A_MIX = 3.5            # fit value (El-Badry Eq 37); 1.7 is the analytic value
FIRE = 0.95           # TRINITY cooling_balance trigger
LANCASTER = (0.90, 0.99)
_LDV = [0.1, 0.3, 1.0, 3.0, 10.0]                       # El-Badry's plausible/used range
_NCORE = [1e2, 1e4, 1e5, 1e6]                           # TRINITY GMC core densities (cleanroom span)
_N_ISM = [0.1, 0.3, 1.0, 3.0, 10.0]                    # El-Badry's TESTED ambient range (validation)


def theta(ldv, n, a_mix=A_MIX):
    """El-Badry Eq 37/38: theta(lambda*dv [pc.km/s], n [cm^-3])."""
    X = a_mix * math.sqrt(ldv * n)
    return X / (11.0 / 5.0 + X)


def ldv_for_theta(theta_target, n, a_mix=A_MIX):
    """Invert: lambda*dv needed to reach theta_target at density n. theta=X/(2.2+X) -> X=2.2 theta/(1-theta)."""
    if theta_target >= 1.0:
        return float("inf")
    X = (11.0 / 5.0) * theta_target / (1.0 - theta_target)
    return (X / (a_mix * math.sqrt(n))) ** 2


def main():
    # --- validation: reproduce El-Badry's fiducial theta(n=1, ldv=1) = 0.61 ---
    th_fid = theta(1.0, 1.0)
    print(f"[validate] theta(lambda*dv=1, n=1, A_mix=3.5) = {th_fid:.3f}  (El-Badry Fig 7 / result v: ~0.61) "
          f"-> {'OK' if abs(th_fid - 0.61) < 0.02 else 'MISMATCH'}")
    print(f"[validate] analytic A_mix=1.7: theta(1,1) = {theta(1.0, 1.0, 1.7):.3f}\n")

    rows = []
    # --- El-Badry's TESTED regime (n_H,0 = 0.1-10): in-range, trustworthy ---
    print("EL-BADRY TESTED REGIME (ambient n_H,0 = 0.1-10 cm^-3; in-range):")
    for n in _N_ISM:
        ths = {ldv: theta(ldv, n) for ldv in _LDV}
        print("  n={:5.1f}  ".format(n) + "  ".join(f"ldv={ldv:<4g}:θ={ths[ldv]:.2f}" for ldv in _LDV))
        for ldv in _LDV:
            rows.append(dict(regime="tested", n=n, ldv=ldv, theta=round(ths[ldv], 4),
                             fires=ths[ldv] >= FIRE))

    # --- TRINITY GMC regime (n = 1e2-1e6): EXTRAPOLATED ---
    print("\nTRINITY GMC REGIME (n = 1e2-1e6 cm^-3; EXTRAPOLATED beyond El-Badry's tested n<=10):")
    for n in _NCORE:
        ths = {ldv: theta(ldv, n) for ldv in _LDV}
        print("  n={:.0e}  ".format(n) + "  ".join(f"ldv={ldv:<4g}:θ={ths[ldv]:.3f}" for ldv in _LDV))
        for ldv in _LDV:
            rows.append(dict(regime="extrapolated", n=n, ldv=ldv, theta=round(ths[ldv], 4),
                             fires=ths[ldv] >= FIRE))

    # --- Lancaster calibration: what lambda*dv reproduces theta~0.9-0.99 at each density? ---
    print("\nLANCASTER CALIBRATION -- lambda*dv [pc.km/s] needed for theta in [0.90, 0.99]:")
    for n in _NCORE:
        lo = ldv_for_theta(LANCASTER[0], n)
        hi = ldv_for_theta(LANCASTER[1], n)
        print(f"  n={n:.0e}:  theta=0.90 at ldv={lo:.3g}   theta=0.99 at ldv={hi:.3g}")

    # firing-threshold density at each ldv (n where theta crosses 0.95)
    print("\nFIRING THRESHOLD -- ambient density n where theta crosses 0.95 (transition fires above this n):")
    for ldv in _LDV:
        # theta=0.95 -> X=2.2*0.95/0.05=41.8 -> n = (41.8/(A_mix sqrt(ldv)))^2
        X95 = (11.0 / 5.0) * FIRE / (1.0 - FIRE)
        n95 = (X95 / (A_MIX * math.sqrt(ldv))) ** 2
        print(f"  ldv={ldv:<5g}: n_fire = {n95:.2e} cm^-3  (clouds denser than this fire; below, fate=energy-driven)")

    out = os.path.join(_HERE, "elbadry_theta.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["regime", "n", "ldv", "theta", "fires"])
        w.writeheader(); w.writerows(rows)
        fh.write("# theta = A_mix*sqrt(ldv*n)/(11/5 + A_mix*sqrt(ldv*n)); A_mix=3.5 (El-Badry Eq 37/38)\n")
        fh.write("# regime=tested: n in El-Badry's 0.1-10; regime=extrapolated: TRINITY GMC n>10 (UNVALIDATED form)\n")
        fh.write("# theta_ElBadry == theta_TRINITY = L_cool/L_mech (trigger fires at 0.95)\n")
    print(f"\nwrote {out}")

    # --- figure ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})"); return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.3))
    nn = np.logspace(-1, 6.2, 200)
    cmap = plt.get_cmap("viridis")
    for i, ldv in enumerate(_LDV):
        c = cmap(i / (len(_LDV) - 1))
        axL.semilogx(nn, [theta(ldv, n) for n in nn], "-", color=c, lw=2, label=f"λδv={ldv}")
    axL.axhspan(*LANCASTER, color="#2ca02c", alpha=0.12)
    axL.axhline(FIRE, ls="--", color="#d1495b", lw=1.1)
    axL.axvspan(0.1, 10, color="0.85", alpha=0.5)
    axL.text(0.12, 0.06, "El-Badry\ntested\n(n≤10)", fontsize=7.5, color="0.3")
    axL.text(2e4, 0.5, "TRINITY GMC\n(extrapolated)", fontsize=8, color="#7b3f00", ha="center")
    axL.text(1.3e-1, 0.955, "trigger 0.95", fontsize=7.5, color="#d1495b")
    axL.set_xlabel(r"ambient density $n$ [cm$^{-3}$]")
    axL.set_ylabel(r"$\theta = L_{\rm cool}/L_{\rm mech}$ (El-Badry Eq 37/38)")
    axL.set_title("El-Badry θ(λδv, n) — the candidate TRINITY θ_target\n"
                  "(rises with density; Lancaster band = green)", fontsize=10, fontweight="bold")
    axL.legend(fontsize=8, loc="lower right"); axL.grid(True, which="both", alpha=0.2)
    axL.set_ylim(0, 1.02)

    # right: lambda*dv needed for theta=0.95 vs density (the calibration / firing threshold)
    n95 = [(11.0/5.0*FIRE/(1.0-FIRE) / (A_MIX * math.sqrt(ldv)))**2 for ldv in _LDV]
    axR.loglog(_LDV, n95, "o-", color="#08519c", lw=2, ms=6)
    axR.set_xlabel(r"$\lambda\delta v$ [pc km/s]")
    axR.set_ylabel(r"firing-threshold density $n_{\rm fire}$ [cm$^{-3}$] ($\theta=0.95$)")
    axR.set_title("Above n_fire the cloud transitions; below it stays energy-driven (fate)\n"
                  "(larger λδv ⇒ lower n_fire ⇒ more clouds fire)", fontsize=10, fontweight="bold")
    axR.grid(True, which="both", alpha=0.2)
    for ldv, n in zip(_LDV, n95):
        axR.annotate(f"{n:.1g}", (ldv, n), fontsize=7, ha="left", va="bottom")
    fig.suptitle("El-Badry analytic θ as TRINITY's θ_target: closed form + Lancaster calibration (no κ_mix port)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "elbadry_theta.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
