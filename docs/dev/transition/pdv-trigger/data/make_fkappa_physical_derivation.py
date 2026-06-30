#!/usr/bin/env python3
"""Deriving the PHYSICAL prescription from El-Badry (verified) + saturation — and what it implies.

The maintainer's worry: the empirical f_kappa(n) ~ n^-0.6 has a NEGATIVE power, which is not a physical
conductivity. Correct. This script derives the physics and reaches a clean (and partly self-correcting)
conclusion. Reads constants + the committed `data/summary.csv` (for the measured baseline); no sims.

THREE different "f_kappa(n)" — keep them straight:
  (i)   MECHANISM  f_kappa = kappa_mix/kappa_Spitzer.  El-Badry Eq21 kappa_mix=(lambda*dv)*n*kB/(mu*mp)
        = (lambda*dv)*n*kB, kappa_Spitzer=C_th*T^(5/2).  So f_kappa_mech ~ n / T^(5/2): RISES with n.
  (ii)  TARGET     theta_target(n; lambda*dv) = psi/(11/5+psi), psi=3.5*sqrt(lambda*dv*n)  (Eq37/38, verified):
        the cooling EFFICIENCY to aim for — flat-high (0.94-0.999) across the GMC range.
  (iii) BOOST      f_kappa to make TRINITY's emergent theta reach (ii): FALLS, because TRINITY's baseline
        theta0(n) already rises. This is the empirical n^-0.6 — a boost factor, NOT a conductivity.

WHAT THE DERIVATION SHOWS:
  - Crossover kappa_mix=kappa_Spitzer at n_crit = C_th*T^(5/2)/((lambda*dv)*kB).  At T=2e5 K, lambda*dv=1:
    n_crit=0.25 cm^-3 — matches El-Badry's stated "kappa_mix dominates n>~0.2, T<~2e5 K". VERIFIED.
  - In the COOL mixing layer (T~2e4 K, the cooling peak) kappa_mix/kappa_Spitzer ~ 1e3-1e7 and ~n, because
    Spitzer ~T^(5/2) VANISHES there. => a scalar f_kappa*Spitzer CANNOT faithfully represent cool-layer
    mixing; the faithful object is the STRUCTURAL kappa_mix term (Rung B), with lambda*dv in [1,10] pc.km/s
    the single physical parameter, capped by saturation (q_sat=5 phi rho c_s^3 ~ n; Cowie&McKee / Eq19-20).

  - The VERIFIED El-Badry target is flat-high EVEN at diffuse (theta_target(n=1e2, lambda*dv=1)=0.94). TRINITY's
    measured 1D baseline rises 0.25 -> 0.95, so the GAP theta_target-theta0 is LARGE at diffuse, small at dense.
    Reading: if El-Badry/Lancaster are right (diffuse clouds DO cool in 3D), the diffuse never-fire corner is a
    1D ARTIFACT and the faithful fix is kappa_mix (route b) -- NOT "accept non-transition" (route a, which
    trusts the 1D under-cooling as truth). The derivation therefore tilts toward kappa_mix.

BOTTOM LINE (the "derived number" the maintainer asked for): it is NOT a single f_max or a scalar power law.
It is lambda*dv in [1,10] pc.km/s (El-Badry's one mixing parameter) feeding the structural kappa_mix term,
with crossover n_crit~0.2 and a saturation cap ~n. The negative-power empirical f_kappa is a boost-factor and
must not be used as a conductivity prescription.

REPRODUCE (from repo root; reads constants + committed summary.csv, no sims):
    python docs/dev/transition/pdv-trigger/data/make_fkappa_physical_derivation.py
Deliverables:
    docs/dev/transition/pdv-trigger/data/fkappa_physical_derivation.csv
    docs/dev/transition/pdv-trigger/fkappa_physical_derivation.png
"""

import csv
import math
import os
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PDV = os.path.dirname(_HERE)
_SUMMARY = os.path.join(_HERE, "summary.csv")

# cgs constants
C_TH = 6e-7              # Spitzer: kappa = C_TH * T^(5/2)
KB = 1.381e-16
PC_KMS = 3.086e23       # 1 pc*km/s in cm^2/s


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def n_crit(T, ldv):
    """ambient density where kappa_mix = kappa_Spitzer (cm^-3)."""
    return C_TH * T ** 2.5 / (ldv * PC_KMS * KB)


def theta_target(n, ldv, A=3.5):
    """El-Badry Eq37/38 cooling efficiency (verified)."""
    psi = A * math.sqrt(ldv * n)
    return psi / (2.2 + psi)


def main():
    # measured TRINITY baseline theta0(nCore) from the sweep (theta at f_kappa=1), median over mCloud/sfe
    by_n = defaultdict(list)
    for r in csv.DictReader(open(_SUMMARY)):
        if _f(r["cooling_boost_kappa"]) == 1.0:
            by_n[_f(r["nCore"])].append(_f(r["theta_blowout"]))
    base = {n: float(np.median(v)) for n, v in by_n.items()}

    rows = []
    print(f"{'nCore':>8} | {'theta0(1D)':>10} {'theta_EB(ldv1)':>14} {'theta_EB(ldv10)':>15} {'gap(ldv1)':>10}")
    for n in sorted(base):
        t0 = base[n]
        te1, te10 = theta_target(n, 1), theta_target(n, 10)
        rows.append(dict(nCore=f"{n:.0f}", theta0_1D=round(t0, 3),
                         theta_EB_ldv1=round(te1, 3), theta_EB_ldv10=round(te10, 3),
                         gap_ldv1=round(te1 - t0, 3)))
        print(f"{n:8.0f} | {t0:10.3f} {te1:14.3f} {te10:15.3f} {te1 - t0:10.3f}")

    print("\ncrossover n_crit (kappa_mix=kappa_Spitzer):")
    for T in (2e4, 1e5, 2e5):
        print(f"  T={T:.0e}: n_crit(ldv=1)={n_crit(T,1):.2g}, n_crit(ldv=10)={n_crit(T,10):.2g} cm^-3")
    print("mechanism f_kappa=kappa_mix/kappa_Spitzer in the cool layer (T=2e4) ~ n: huge, can't be a scalar.")

    out = os.path.join(_HERE, "fkappa_physical_derivation.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["nCore", "theta0_1D", "theta_EB_ldv1", "theta_EB_ldv10", "gap_ldv1"])
        w.writeheader()
        w.writerows(rows)
        fh.write(f"# crossover n_crit(T=2e5,ldv=1)={n_crit(2e5,1):.2f} cm^-3 (matches El-Badry ~0.2)\n")
        fh.write("# kappa_mix=(lambda*dv)*n*kB (Eq21); kappa_Spitzer=C_th*T^(5/2); f_kappa_mech~n (rises)\n")
        fh.write("# physical enhancement RISES with n; empirical fire-threshold ~n^-0.6 is a boost-factor, not a conductivity\n")
        fh.write("# faithful form = structural kappa_mix (Rung B), lambda*dv in [1,10] pc.km/s; scalar f_kappa can't represent the cool layer\n")
    print(f"wrote {out}")

    # ---- figure ----
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

    # LEFT: kappa_Spitzer(T) vs kappa_mix(T) at a GMC density -> why a scalar f_kappa fails
    T = np.logspace(3.5, 7, 200)
    n_demo = 1e2
    kSp = C_TH * T ** 2.5
    for ldv, c in zip((1, 10), ("#1f77b4", "#2ca02c")):
        kmix = ldv * PC_KMS * n_demo * KB * np.ones_like(T)
        axL.loglog(T, kmix, "--", color=c, lw=1.8, label=fr"$\kappa_{{\rm mix}}$ ($\lambda\delta v={ldv}$, T-indep.)")
        Tc = (ldv * PC_KMS * n_demo * KB / C_TH) ** 0.4
        axL.plot(Tc, C_TH * Tc ** 2.5, "o", color=c, ms=7)
    axL.loglog(T, kSp, "-", color="#d1495b", lw=2.2, label=r"$\kappa_{\rm Spitzer}=C_{\rm th}T^{5/2}$")
    axL.axvspan(1e4, 2e5, color="0.85", alpha=0.5)
    axL.text(3e4, kSp.min() * 3, "cool mixing layer\n($\\kappa_{\\rm mix}$ rules,\nSpitzer vanishes)",
             fontsize=8, color="0.3", ha="center")
    axL.set_xlabel("T [K]")
    axL.set_ylabel(r"conductivity $\kappa$ [cgs]  (at $n=10^2$)")
    axL.set_title("Why a scalar f_κ can't represent mixing:\nκ_mix is T-independent, Spitzer ∝ T^5/2 → ∞ ratio at low T",
                  fontsize=10, fontweight="bold")
    axL.legend(fontsize=8.5, loc="upper left")
    axL.grid(True, which="both", alpha=0.2)

    # RIGHT: El-Badry target (flat-high) vs measured 1D baseline (rising) -> the gap kappa_mix must supply
    ns = np.logspace(2, 6, 50)
    axR.plot(ns, [theta_target(n, 1) for n in ns], "-", color="#2ca02c", lw=2, label=r"El-Badry $\theta^\star$ ($\lambda\delta v{=}1$) [verified]")
    axR.plot(ns, [theta_target(n, 10) for n in ns], "-", color="#2ca02c", lw=1.2, alpha=0.5, label=r"El-Badry $\theta^\star$ ($\lambda\delta v{=}10$)")
    axR.plot(sorted(base), [base[n] for n in sorted(base)], "o-", color="#1f77b4", lw=2, ms=6,
             label=r"TRINITY 1D baseline $\theta_0$ (measured, $f_\kappa{=}1$)")
    axR.fill_between(sorted(base), [base[n] for n in sorted(base)],
                     [theta_target(n, 1) for n in sorted(base)], color="#ff7f0e", alpha=0.15)
    axR.annotate("gap κ_mix must supply\n(large at diffuse → 1D under-cools;\nEl-Badry says diffuse SHOULD cool)",
                 xy=(3e2, 0.6), fontsize=7.6, color="0.3")
    axR.set_xscale("log")
    axR.set_xlabel(r"$n_{\rm core}$ [cm$^{-3}$]")
    axR.set_ylabel(r"$\theta = L_{\rm cool}/L_{\rm mech}$")
    axR.set_ylim(0, 1.04)
    axR.set_title("The target is flat-high even at diffuse →\nfaithful fix is κ_mix (route b), not 'accept' (route a)",
                  fontsize=10, fontweight="bold")
    axR.legend(fontsize=8, loc="lower right")
    axR.grid(True, which="both", alpha=0.2)

    fig.suptitle("Physical prescription derived: the enhancement is κ_mix(λδv) — RISES with n, can't be a scalar f_κ power law",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(_PDV, "fkappa_physical_derivation.png")
    fig.savefig(png, dpi=140)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
