#!/usr/bin/env python3
"""Emergent-θ calibration of f_κ against the El-Badry θ(n) target (corrected direction, 2026-07-01).

Re-analysis of COMMITTED data (no sims). The corrected direction (FINDINGS §8c) is: boost the cooling
MECHANISM (`cooling_boost_mode='multiplier'`, f_κ on the resolved radiative channel) and let θ EMERGE; use
El-Badry/Lancaster as the CALIBRATION TARGET for that emergent θ, not an enforced value; set f_κ at a PHYSICAL
value and accept that clouds whose achievable θ stays < 0.95 remain energy-driven (route-a).

Pieces, all measured/committed (fkappa_functional_form.csv, kappa_blowout_calibration.csv):
  emergent model:   θ(n, f_κ) = θ0(n) · f_κ^p            (measured; back-reaction makes p sub-linear)
  baseline θ0(n):   logit θ0 = -1.7269 + 0.4087·log10(n) (6-anchor fit, RMS 0.49 in logit)
  leverage p:       0.31 median (compact 0.309 / mid 0.208 / diffuse 0.421)
The NEW target (vs the old flat θ*=0.95) is El-Badry's density-dependent closed form at the calibrated λδv=3:
  θ_EB(n) = A_mix·√(λδv·n) / (11/5 + A_mix·√(λδv·n)),  A_mix=3.5,  capped at θ_max=0.99.

Prescription:  f_κ_ideal(n) = (θ_EB(n)/θ0(n))^(1/p);  f_κ(n) = min(f_max, f_κ_ideal);
               θ_achieved(n) = θ0(n)·f_κ(n)^p;  fires iff θ_achieved ≥ 0.95.
The route-a boundary n_routeA(f_max) = smallest n with θ0(n)·f_max^p ≥ 0.95 — the falsifiable energy→momentum
split at a given physical cap.

Writes fkappa_emergent_calibration.{png,csv} + a route-a-boundary figure next to this script's parent folder.
"""
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)  # workstream folder (fig/ is gitignored)

A_MIX, LDV, THETA_MAX = 3.5, 3.0, 0.99
THETA_FIRE = 0.95
B0, B1 = -1.7269, 0.4087          # logit theta0 fit
P_MED = 0.31                      # leverage (median)
FMAXES = [4.0, 8.0, 16.0]         # physical f_kappa ceilings to show

# measured baseline theta0 points (fkappa_functional_form.csv + kappa_blowout_calibration.csv)
THETA0_PTS = [(100, 0.25), (100, 0.169), (10000, 0.511), (10000, 0.61),
              (100000, 0.342), (100000, 0.667), (1000000, 0.697)]


def theta0(n):
    return 1.0 / (1.0 + math.exp(-(B0 + B1 * math.log10(n))))


def theta_EB(n):
    X = A_MIX * math.sqrt(LDV * n)
    return min(X / (11.0 / 5.0 + X), THETA_MAX)


def fkappa_ideal(n, p=P_MED):
    return (theta_EB(n) / theta0(n)) ** (1.0 / p)


def theta_achieved(n, fmax, p=P_MED):
    fk = min(fmax, fkappa_ideal(n, p))
    return theta0(n) * fk ** p


def route_a_boundary(fmax, p=P_MED):
    """smallest n (cm^-3) with theta0(n)*fmax^p >= THETA_FIRE; None if none in range."""
    ns = np.logspace(0, 6.5, 4000)
    for n in ns:
        if theta0(n) * fmax ** p >= THETA_FIRE:
            return n
    return None


# ---------------------------------------------------------------- figure 1: the calibration
def fig_calibration():
    n = np.logspace(0, 6.5, 400)
    th0 = np.array([theta0(x) for x in n])
    thEB = np.array([theta_EB(x) for x in n])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.plot(n, thEB, "-", color="#2c6fb3", lw=2.6, label="El-Badry target θ_EB(n)  (λδv=3)")
    ax.plot(n, th0, "--", color="#7b4fb0", lw=2.0, label="baseline θ₀(n)  (emergent at f_κ=1)")
    for fmax, c in zip(FMAXES, ["#f1c40f", "#e8842a", "#2f9e44"]):
        tha = np.array([theta_achieved(x, fmax) for x in n])
        ax.plot(n, tha, "-", color=c, lw=1.8, label=f"achieved θ at capped f_κ (f_max={fmax:g})")
        nb = route_a_boundary(fmax)
        if nb:
            ax.axvline(nb, color=c, lw=1.0, ls=":")
            ax.text(nb * 1.05, 0.10, f"route-a\nn≳{nb:.0f}", color=c, fontsize=8, rotation=90, va="bottom")
    ax.axhline(THETA_FIRE, color="#c0392b", lw=1.3, ls="-.")
    ax.text(1.3, THETA_FIRE + 0.008, "fire  θ≥0.95", color="#c0392b", fontsize=9.5)
    for nn, t0 in THETA0_PTS:
        ax.scatter([nn], [t0], s=34, color="#7b4fb0", edgecolor="k", lw=0.4, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"core density $n$ (cm$^{-3}$)")
    ax.set_ylabel(r"$\theta = L_{\rm cool}/L_{\rm mech}$")
    ax.set_title("Emergent-θ calibration: boost f_κ so the SOLVED θ tracks El-Badry, capped at a physical f_max\n"
                 "clouds left of the route-a line can't reach θ=0.95 at physical f_κ → stay energy-driven (by design)",
                 fontsize=10.4)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8.6, framealpha=0.95)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fkappa_emergent_calibration.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: f_kappa(n) + route-a boundary
def fig_fkappa():
    n = np.logspace(0, 6.5, 400)
    fk_ideal = np.array([fkappa_ideal(x) for x in n])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    a1.plot(n, fk_ideal, "-", color="#2c6fb3", lw=2.4, label="f_κ ideal = (θ_EB/θ₀)^(1/p)")
    for fmax, c in zip(FMAXES, ["#f1c40f", "#e8842a", "#2f9e44"]):
        a1.plot(n, np.minimum(fk_ideal, fmax), "-", color=c, lw=1.6, label=f"capped at {fmax:g}")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel(r"n (cm$^{-3}$)"); a1.set_ylabel("f_κ")
    a1.set_title("f_κ(n): ideal (to hit El-Badry) vs physical cap", fontsize=10.5)
    a1.legend(fontsize=8.3); a1.grid(alpha=0.25, which="both")
    fmx = np.linspace(1, 40, 200)
    nb = [route_a_boundary(f) or np.nan for f in fmx]
    a2.plot(fmx, nb, "-", color="#c0392b", lw=2.4)
    a2.set_yscale("log")
    a2.set_xlabel("physical ceiling f_max"); a2.set_ylabel("route-a boundary n (cm$^{-3}$)")
    a2.set_title("The falsifiable split: densest cloud that STAYS\nenergy-driven, vs the f_max you accept", fontsize=10.5)
    a2.grid(alpha=0.25, which="both")
    for f in FMAXES:
        b = route_a_boundary(f)
        if b:
            a2.scatter([f], [b], s=40, color="#c0392b", zorder=5)
            a2.annotate(f"f_max={f:g}\nn≳{b:.0f}", (f, b), fontsize=8, xytext=(6, 6),
                        textcoords="offset points")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fkappa_emergent_routea.png"), dpi=130)
    plt.close(fig)


def write_csv():
    rows = []
    for n in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
        row = {"nCore_cm3": f"{n:g}", "theta0": round(theta0(n), 3),
               "theta_EB_ldv3": round(theta_EB(n), 3), "fkappa_ideal": round(fkappa_ideal(n), 2)}
        for fmax in FMAXES:
            row[f"theta_achieved_fmax{fmax:g}"] = round(theta_achieved(n, fmax), 3)
            row[f"fires_fmax{fmax:g}"] = theta_achieved(n, fmax) >= THETA_FIRE
        rows.append(row)
    cols = list(rows[0].keys())
    with open(os.path.join(HERE, "fkappa_emergent_calibration.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    fig_calibration()
    fig_fkappa()
    rows = write_csv()
    print("route-a boundary n (cm^-3):",
          {f"f_max={f:g}": round(route_a_boundary(f) or float('nan'), 1) for f in FMAXES})
    for r in rows:
        print(f"  n={r['nCore_cm3']:>6}  θ0={r['theta0']:.3f}  θ_EB={r['theta_EB_ldv3']:.3f}  "
              f"f_κ_ideal={r['fkappa_ideal']:>7}  "
              f"θ@8={r['theta_achieved_fmax8']:.3f} fire@8={r['fires_fmax8']}")
    print("wrote fkappa_emergent_calibration.{png,csv} + fkappa_emergent_routea.png")
