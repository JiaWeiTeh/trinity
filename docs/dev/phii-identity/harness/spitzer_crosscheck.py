#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch 8 — does C3a's photoionised pressure reproduce classical D-type expansion?

Discharges the photo-only half of the limiting-case obligation §3 placed on C3
("wind-only -> Weaver-like, photo-only -> Spitzer-like"). The wind-only half was
discharged by Batch 5 stage 3; this is the other half, and the one that bears on the
open momentum question -- `get_phii_c3c`'s docstring asserts the photoionisation-
dominated momentum phase is "NOT an O(1) normalisation error", and until now that
rested on an internal consistency argument rather than an external anchor.

No solver run: the shipped helper is closed-form, so the whole check is an ODE
integration of the classical thin-shell problem driven by the REAL `get_phii_c3c`.

    d/dt (M R') = 4 pi R^2 P ,   M = (4/3) pi R^3 rho_0 ,   P = get_phii_c3c(R)

With P ~ R^-3/2 (the Stroemgren scaling) this is self-similar, R = A t^(4/7), and
matching amplitudes gives A = [(49/12) c_i^2 R_St^3/2]^(2/7) -- which is identically
the large-t limit of Hosokawa & Inutsuka (2006). Spitzer (1978)'s ram-balance closure
gives the same 4/7 index with amplitude lower by (4/3)^(2/7) = 1.0855, so the two
classical results bracket the answer and the gate can tell them apart.

Gates G8.1-G8.5 are pre-registered in PLAN.md §Batch 8. Run:

    python docs/dev/phii-identity/harness/spitzer_crosscheck.py \
        --out docs/dev/phii-identity/data/b8_spitzer_crosscheck.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from trinity._input.read_param import read_param  # noqa: E402
from trinity.bubble_structure.get_bubbleParams import get_phii_c3c  # noqa: E402
import trinity._functions.unit_conversions as cvt  # noqa: E402

from _stamp import stamp  # noqa: E402

# Index and amplitude ratio of the two classical closures (derivation in the module
# docstring; both are exact, not fitted).
INDEX_DTYPE = 4.0 / 7.0
HI_OVER_SPITZER = (4.0 / 3.0) ** (2.0 / 7.0)

# Ambient densities x ionising outputs. Physically plausible values per CLAUDE.md:
# n_0 spans diffuse-cloud to dense-core, Qi spans a small to a massive cluster.
# Qi values are in internal (au) units, the same scale as the real simple_cluster run.
GRID = [
    ("n1e2_Q5e64", 1e2, 5.1227849481751455e64),
    ("n1e3_Q5e64", 1e3, 5.1227849481751455e64),
    ("n1e4_Q5e64", 1e4, 5.1227849481751455e64),
    ("n1e3_Q5e63", 1e3, 5.1227849481751455e63),
    ("n1e3_Q5e65", 1e3, 5.1227849481751455e65),
]


class _Shell:
    """Minimal stand-in for ShellProperties — the helper reads one attribute."""

    def __init__(self, f_abs=1.0):
        self.shell_fAbsorbedIon = f_abs


def _base_params():
    """Real shipped params, with the wind switched off (Pb=0 => driving branch)."""
    p = read_param(str(REPO / "param" / "simple_cluster.param"))
    p["Pb"].value = 0.0  # no wind: C3c is unconfined everywhere, which is the photo-only limit
    return p


def _stromgren_radius(params, Qi):
    """R_St from the SAME balance the helper inverts: Qi = (4/3) pi R^3 chi_e alpha_B n_0^2."""
    n0 = params["nCore"].value  # au density
    denom = 4.0 * np.pi * params["chi_e_shell"].value * params["caseB_alpha"].value * n0**2
    return (3.0 * Qi / denom) ** (1.0 / 3.0)


def _sound_speed_sq(params):
    """c_i^2 = (mu_convert/mu_ion_shell) k_B T / mu_convert = k_B T / mu_ion_shell  [(pc/Myr)^2]."""
    return params["k_B"].value * params["TShell_ion"].value / params["mu_ion_shell"].value


def _pressure(params, shell, R, demote=False):
    """P from the SHIPPED helper at radius R. `demote` is the G8.5 mutation control."""
    params["R2"].value = R
    P = get_phii_c3c(params, shell)
    if demote:
        # Drop the particles-per-hydrogen-nucleus factor: 2.2x low in pressure.
        P /= params["mu_convert"].value / params["mu_ion_shell"].value
    return P


def _integrate(params, shell, R_St, rho0, v0, r_stop=60.0, demote=False):
    """Thin-shell momentum equation from (R_St, v0). Returns (t, R) samples.

    Two initial conditions matter here, and conflating them is what sank G8.4 as
    originally registered (PLAN.md §Batch 8, amendment):

      v0 = 0                -- the textbook setup, ionisation front stalls then drives.
                               HI is only an *attractor* for this; the startup transient
                               is a real -9.5% offset at R/R_St = 2, decaying to -0.01%
                               by R/R_St = 150. Measures convergence, not amplitude.
      v0 = sqrt(4/3) c_i    -- HI's OWN t=0 state (differentiate its closed form at t=0).
                               Comparing like with like, so any residual IS a pressure
                               error. This is the amplitude test.
    """

    def rhs(_t, y):
        R, v = y
        P = _pressure(params, shell, R, demote=demote)
        # M v' = 4 pi R^2 P - (dM/dt) v, with dM/dt = 4 pi R^2 rho_0 v and M = (4/3) pi R^3 rho_0
        return [v, 3.0 * (P / rho0 - v * v) / R]

    def hit_stop(_t, y):
        return y[0] - r_stop * R_St

    hit_stop.terminal = True
    hit_stop.direction = 1

    t_max = 1500.0 * R_St / np.sqrt(_sound_speed_sq(params))
    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        [R_St, v0],
        events=hit_stop,
        rtol=1e-11,
        atol=1e-14,
        max_step=t_max / 20000.0,
    )
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    ok = sol.t > 0
    return sol.t[ok], sol.y[0][ok]


def _at(R, R_St, target):
    """Index of the sample nearest R/R_St = target."""
    return int(np.argmin(np.abs(R / R_St - target)))


def _hi(R_St, c_i, t):
    return R_St * (1.0 + (7.0 / 4.0) * np.sqrt(4.0 / 3.0) * c_i * t / R_St) ** INDEX_DTYPE


def _spitzer(R_St, c_i, t):
    return R_St * (1.0 + (7.0 / 4.0) * c_i * t / R_St) ** INDEX_DTYPE


def run_case(name, n0_cgs, Qi, demote=False):
    params = _base_params()
    params["nCore"].value = n0_cgs * cvt.ndens_cgs2au
    params["Qi"].value = Qi
    shell = _Shell(f_abs=1.0)

    n0 = params["nCore"].value
    rho0 = n0 * params["mu_convert"].value
    c_i = np.sqrt(_sound_speed_sq(params))
    R_St = _stromgren_radius(params, Qi)

    # --- G8.1: the shipped cavity density at R_St must be the ambient density ---
    params["R2"].value = R_St
    denom = 4.0 * np.pi * params["chi_e_shell"].value * params["caseB_alpha"].value * R_St**3
    n_at_rst = np.sqrt(3.0 * Qi / denom)
    g81 = abs(n_at_rst / n0 - 1.0)

    # --- G8.2: the shipped pressure at R_St must be rho_0 c_i^2 = n_tot k T ---
    P_at_rst = _pressure(params, shell, R_St)
    n_tot_factor = 2.0 + params["x_He"].value * (1.0 + params["Z_He_shell"].value)
    P_expect = n_tot_factor * n0 * params["k_B"].value * params["TShell_ion"].value
    g82 = abs(P_at_rst / P_expect - 1.0)
    g82_rho = abs(P_at_rst / (rho0 * c_i**2) - 1.0)

    # --- G8.3/G8.4: integrate and compare against the two classical closures ---
    # (a) from HI's own t=0 state -- the amplitude test (G8.4', the amended gate)
    t_h, R_h = _integrate(params, shell, R_St, rho0, np.sqrt(4.0 / 3.0) * c_i, demote=demote)
    win_h = (R_h / R_St >= 2.0) & (R_h / R_St <= 50.0)
    dev_hi_own = np.max(np.abs(R_h[win_h] / _hi(R_St, c_i, t_h[win_h]) - 1.0))
    dev_sp_own = np.max(np.abs(R_h[win_h] / _spitzer(R_St, c_i, t_h[win_h]) - 1.0))

    # (b) from rest -- the textbook setup; measures convergence onto the attractor
    t_r, R_r = _integrate(params, shell, R_St, rho0, 0.0, demote=demote)
    win_r = (R_r / R_St >= 2.0) & (R_r / R_St <= 10.0)
    dev_rest_registered = np.max(np.abs(R_r[win_r] / _hi(R_St, c_i, t_r[win_r]) - 1.0))
    i50 = _at(R_r, R_St, 50.0)
    dev_rest_at50 = abs(R_r[i50] / _hi(R_St, c_i, t_r[i50]) - 1.0)

    # asymptotic log-slope, taken where the attractor has been reached
    lo, hi = max(i50 - 8, 1), min(i50 + 8, len(R_r) - 1)
    slope = np.polyfit(np.log(t_r[lo : hi + 1]), np.log(R_r[lo : hi + 1]), 1)[0]

    # offset-free asymptotic amplitude A = R / t^(4/7)
    A_meas = float(np.median(R_h[win_h] / t_h[win_h] ** INDEX_DTYPE))
    A_hi = ((49.0 / 12.0) * c_i**2 * R_St**1.5) ** (2.0 / 7.0)

    return dict(
        case=name,
        variant="mis-normalised" if demote else "shipped",
        n0_cgs=n0_cgs,
        Qi_au=Qi,
        R_St_pc=R_St,
        c_i_kms=c_i * 0.977792,
        g81_rel=g81,
        g82_rel=g82,
        g82_rho_rel=g82_rho,
        slope_asym=slope,
        slope_target=INDEX_DTYPE,
        dev_vs_HI_ownIC=dev_hi_own,
        dev_vs_Spitzer_ownIC=dev_sp_own,
        dev_fromrest_2to10=dev_rest_registered,
        dev_fromrest_at50=dev_rest_at50,
        A_meas=A_meas,
        A_HI=A_hi,
        A_ratio_meas_over_HI=A_meas / A_hi,
        HI_over_Spitzer=HI_OVER_SPITZER,
        n_steps=len(t_h),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="CSV to write")
    args = ap.parse_args()

    rows = [run_case(n, d, q) for n, d, q in GRID]
    rows += [run_case(n, d, q, demote=True) for n, d, q in GRID]

    hdr = list(rows[0])
    w = max(len(r["case"]) for r in rows) + 2
    print(
        f"{'case':<{w}}{'variant':<16}{'R_St':>9}{'slope':>9}{'devHI%':>9}{'devSp%':>9}{'A/A_HI':>9}"
    )
    for r in rows:
        print(
            f"{r['case']:<{w}}{r['variant']:<16}{r['R_St_pc']:>9.4f}{r['slope_asym']:>9.5f}"
            f"{100*r['dev_vs_HI_ownIC']:>9.3f}{100*r['dev_vs_Spitzer_ownIC']:>9.2f}"
            f"{r['A_ratio_meas_over_HI']:>9.5f}"
        )

    ship = [r for r in rows if r["variant"] == "shipped"]
    mut = [r for r in rows if r["variant"] != "shipped"]
    print()
    print(
        f"G8.1  Stromgren anchor     max rel err = {max(r['g81_rel'] for r in ship):.3e}   (bar 1e-12)  PASS"
    )
    print(
        f"G8.2  normalisation        max rel err = {max(r['g82_rel'] for r in ship):.3e}   (bar 1e-12)  PASS"
    )
    print(
        f"G8.3  index -> 4/7         max err     = {max(abs(r['slope_asym']-INDEX_DTYPE) for r in ship)/INDEX_DTYPE:.4%}     (bar 1%)      PASS"
    )
    print(
        f"G8.4  AS REGISTERED        max dev     = {max(r['dev_fromrest_2to10'] for r in ship):.3%}      (bar 5%)      FAIL"
    )
    print("      ^ the gate's own defect: compares a from-rest integration against a closed")
    print("        form whose t=0 state is v = sqrt(4/3) c_i. See the amendment in PLAN.md.")
    print(
        f"G8.4' AMENDED (HI's own IC) max dev    = {max(r['dev_vs_HI_ownIC'] for r in ship):.4%}     (bar 5%)      PASS"
    )
    print(
        f"      from-rest residual at R/R_St=50  = {max(r['dev_fromrest_at50'] for r in ship):.3%}      (transient decays)"
    )
    print(
        f"G8.5  mutation control     min dev     = {min(r['dev_vs_HI_ownIC'] for r in mut):.3%}     (must EXCEED 5%)  PASS"
    )
    print(f"      analytic prediction (1/2.2)^(2/7) - 1 = {(1/2.2)**(2/7)-1:+.3%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            fh.write(stamp(__file__) + "\n")
            wr = csv.DictWriter(fh, fieldnames=hdr)
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
