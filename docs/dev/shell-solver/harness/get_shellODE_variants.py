#!/usr/bin/env python3
"""Canonical shell-ODE fix VARIANTS for the de-risking matrix (production untouched).

These are monkeypatch-able drop-ins with the SAME signature as
``trinity.shell_structure.get_shellODE.get_shellODE``. The subagent test harnesses
import these so every cell measures one identical implementation (not three drifting
copies). Idea ids match OVERFLOW_FIX_PLAN.md:

  V1  get_shellODE_cgs    -- evaluate the RHS in cgs, convert derivatives back.
                             Should be an EXACT identity vs production (rtol ~1e-12)
                             in the non-overflow regime, but never overflows the
                             nShell**2 recombination pole.

Self-check (run directly):  python docs/dev/shell-solver/harness/get_shellODE_variants.py
asserts get_shellODE_cgs == production get_shellODE to rtol 1e-12 across non-overflow
inputs, for BOTH the ionised and neutral branches. Authored 2026-06-18.
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np

import trinity._functions.unit_conversions as cvt

# --- au -> cgs input factors (precomputed once; the RHS runs ~1e3-1e4x/solve) ---
_N_AU2CGS = cvt.ndens_au2cgs                      # 1/pc^3 -> 1/cm^3   (3.40e-56)
_R_AU2CGS = cvt.pc2cm                             # pc -> cm           (3.09e18)
_ALPHA_AU2CGS = 1.0 / cvt.convert2au("cm**3 * s**-1")  # caseB_alpha au -> cm^3/s (9.31e41)
_SIGMA_AU2CGS = 1.0 / cvt.convert2au("cm**2")     # dust_sigma au -> cm^2
_C_AU2CGS = cvt.v_au2cms                          # pc/Myr -> cm/s
_KB_AU2CGS = cvt.k_B_au2cgs                       # au -> erg/K
_L_AU2CGS = cvt.L_au2cgs                          # au -> erg/s
_QI_AU2CGS = cvt.s2Myr                            # 1/Myr -> 1/s   (NOT convert2au('1/Myr'))
# --- cgs -> au derivative factors ---
_DNDR_CGS2AU = cvt.convert2au("cm**-4")           # 1/cm^4 -> 1/pc^4  (9.07e73)
_DR_CGS2AU = cvt.pc2cm                            # 1/cm   -> 1/pc    (3.09e18)


def get_shellODE_cgs(y, r, f_cover, is_ionised, params):
    """V1: production RHS evaluated in cgs, derivatives converted back to code units."""
    sigma_dust = params["dust_sigma"].value * _SIGMA_AU2CGS
    mu_n = params["mu_atom"].value
    mu_p = params["mu_ion_shell"].value
    mu_H = params["mu_convert"].value
    chi_e = params["chi_e_shell"].value
    t_ion = params["TShell_ion"].value
    t_neu = params["TShell_neu"].value
    alpha_B = params["caseB_alpha"].value * _ALPHA_AU2CGS
    k_B = params["k_B"].value * _KB_AU2CGS
    c = params["c_light"].value * _C_AU2CGS
    Ln = params["Ln"].value * _L_AU2CGS
    Li = params["Li"].value * _L_AU2CGS
    Qi = params["Qi"].value * _QI_AU2CGS
    r_c = r * _R_AU2CGS

    if is_ionised:
        nShell, phi, tau = y
        n_c = nShell * _N_AU2CGS
        neg_exp_tau = 0 if tau > 500 else np.exp(-tau)
        phi = max(0.0, phi)
        dndr = (
            mu_p / mu_H / (k_B * t_ion) * (
                n_c * sigma_dust / (4 * np.pi * r_c**2 * c) * (Ln * neg_exp_tau + Li * phi)
                + chi_e * n_c**2 * alpha_B * Li / Qi / c
            )
        ) * _DNDR_CGS2AU
        dphidr = (
            -4 * np.pi * r_c**2 * chi_e * alpha_B * n_c**2 / Qi - n_c * sigma_dust * phi
        ) * _DR_CGS2AU
        dtaudr = (n_c * sigma_dust * f_cover) * _DR_CGS2AU
        return dndr, dphidr, dtaudr

    nShell, tau = y
    n_c = nShell * _N_AU2CGS
    neg_exp_tau = 0 if tau > 500 else np.exp(-tau)
    dndr = (
        mu_n / mu_H / (k_B * t_neu) * (n_c * sigma_dust / (4 * np.pi * r_c**2 * c) * (Ln * neg_exp_tau))
    ) * _DNDR_CGS2AU
    dtaudr = (n_c * sigma_dust) * _DR_CGS2AU
    return dndr, dtaudr


def get_shellODE_phiguard(y, r, f_cover, is_ionised, params):
    """V2: freeze the derivatives once the integrated phi crosses the front (phi<=0).

    Past the ionisation front the slice is discarded by shell_structure anyway, so
    freezing the state there prevents the n^2 pole from running away. Empirically
    clears the flood (ovf_idx -> -1, overflow_warns -> 0) on simple_cluster.
    """
    from trinity.shell_structure.get_shellODE import get_shellODE as _prod

    if is_ionised and y[1] <= 0.0:
        return 0.0, 0.0, 0.0
    return _prod(y, r, f_cover, is_ionised, params)


_NSHELL_CLIP = 1e120  # >> any used-region nShell (~1e65), << the 1e154 overflow threshold


def get_shellODE_clip(y, r, f_cover, is_ionised, params):
    """V4: cap nShell inside the RHS (crude). Clears the flood but is an arbitrary threshold."""
    from trinity.shell_structure.get_shellODE import get_shellODE as _prod

    if is_ionised:
        y = (min(y[0], _NSHELL_CLIP), y[1], y[2])
    else:
        y = (min(y[0], _NSHELL_CLIP), y[1])
    return _prod(y, r, f_cover, is_ionised, params)


def _self_check():
    """Assert get_shellODE_cgs == production get_shellODE to rtol 1e-12 (non-overflow)."""
    from trinity._input.read_param import read_param
    from trinity.shell_structure.get_shellODE import get_shellODE

    repo = os.getcwd()
    p = read_param(os.path.join(repo, "param", "simple_cluster.param"))
    p["Ln"].value, p["Li"].value, p["Qi"].value = 1.0e3, 2.0e3, 1.0e5

    # non-overflow inputs: nShell in cgs-ish 1e2..1e6 cm^-3 -> code units via cgs2au
    worst = 0.0
    for n_cgs in (1e2, 1e4, 1e6):
        n_au = n_cgs * cvt.ndens_cgs2au
        for r in (1.0, 5.0, 20.0):
            for phi in (1.0, 0.5, 1e-6):
                for tau in (0.0, 0.3, 3.0):
                    a = np.array(get_shellODE([n_au, phi, tau], r, 1.0, True, p), float)
                    b = np.array(get_shellODE_cgs([n_au, phi, tau], r, 1.0, True, p), float)
                    rel = np.max(np.abs((b - a) / np.where(a == 0, 1, a)))
                    worst = max(worst, rel)
                    # neutral branch
                    a2 = np.array(get_shellODE([n_au, tau], r, 1.0, False, p), float)
                    b2 = np.array(get_shellODE_cgs([n_au, tau], r, 1.0, False, p), float)
                    rel2 = np.max(np.abs((b2 - a2) / np.where(a2 == 0, 1, a2)))
                    worst = max(worst, rel2)
    print(f"max rel diff cgs-variant vs production (ion+neu) = {worst:.3e}")
    assert worst < 1e-12, f"NOT an identity: {worst:.3e}"
    print("PASS: get_shellODE_cgs is an exact identity (rtol < 1e-12) in the non-overflow regime.")


if __name__ == "__main__":
    _self_check()
