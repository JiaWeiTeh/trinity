#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated density profile module for TRINITY.

Calculates number density n(r) for power-law and Bonnor-Ebert profiles.
Supports both scalar and array inputs with consistent output types.

For power-law density profile:
    n(r) = nCore                    for r <= rCore
    n(r) = nCore * (r/rCore)^alpha  for rCore < r <= rCloud
    n(r) = nISM                     for r > rCloud

For Bonnor-Ebert sphere:
    n(r) = nCore * f_rho_rhoc(xi)   for r <= rCloud
    n(r) = nISM                     for r > rCloud

The discontinuous jump from cloud density to nISM at r=rCloud is replaced
by a tanh bridge of width SMOOTH_FRAC * rCloud (1% by default) so the ODE
solvers in the phase modules see a C^infty rhs at the cloud boundary.

@author: Jia Wei
"""

import math

import numpy as np

# Import Bonnor-Ebert sphere module for r2xi conversion
from trinity.cloud_properties.bonnorEbertSphere import r2xi as be_r2xi
from trinity._functions.unit_conversions import ndens_cgs2au

# =============================================================================
# Helper functions for scalar/array consistency
# =============================================================================

def _is_scalar(x) -> bool:
    """Check if input is scalar (not array-like)."""
    return np.ndim(x) == 0


def _to_array(x) -> np.ndarray:
    """Convert input to a 1-D float numpy array."""
    return np.atleast_1d(np.asarray(x, dtype=float))


def _to_output(result: np.ndarray, was_scalar: bool):
    """Convert result back to scalar if input was scalar."""
    if was_scalar:
        return float(result[0])
    return result


# =============================================================================
# Main function
# =============================================================================

def get_density_profile(r, params):
    """
    Calculate the number density profile n(r) at given radius/radii.

    This function computes the gas number density at position r, supporting
    both power-law (densPL) and Bonnor-Ebert sphere (densBE) density profiles.

    Parameters
    ----------
    r : float or array-like
        Radius/radii of interest [pc]. Can be scalar or array.
    params : dict
        Dictionary containing cloud parameters:
        - nISM: ISM number density [1/pc^3] (code units)
        - nCore: Core number density [1/pc^3] (code units)
        - rCloud: Cloud radius [pc]
        - rCore: Core radius [pc]
        - dens_profile: 'densPL' or 'densBE'
        - densPL_alpha: power-law exponent (for densPL)
        - densBE_f_rho_rhoc: interpolation function (for densBE)

    Returns
    -------
    n : float or np.ndarray
        Number density at radius r [1/pc^3] (code units).
        Returns scalar if input r is scalar, array if input r is array.

    Notes
    -----
    For power-law profile (densPL):
        - alpha = 0 (homogeneous): n = nCore for r <= rCloud, n = nISM for r > rCloud
        - alpha != 0: n = nCore * (r/rCore)^alpha with boundary conditions

    For Bonnor-Ebert sphere (densBE):
        - Uses tabulated density profile from bonnorEbertSphere module
        - n = nCore * f_rho_rhoc(xi) for r <= rCloud, n = nISM for r > rCloud

    Examples
    --------
    >>> # Scalar input returns scalar output
    >>> n = get_density_profile(0.5, params)
    >>> type(n)
    <class 'float'>

    >>> # Array input returns array output
    >>> n_arr = get_density_profile([0.5, 1.0, 2.0], params)
    >>> type(n_arr)
    <class 'numpy.ndarray'>
    """
    # Track if input was scalar for output conversion
    was_scalar = _is_scalar(r)
    r_arr = _to_array(r)

    # Extract parameters
    nISM = params['nISM'].value
    nCore = params['nCore'].value
    rCloud = params['rCloud'].value
    rCore = params['rCore'].value

    # Initialize output array
    n_arr = np.zeros_like(r_arr)

    # =============================================================================
    # Numerical regularization at rCloud
    # ---------------------------------
    # The cloud->ISM transition is mathematically idealized as a step. That
    # step makes mShell_dot = 4*pi*r^2*rho(r)*v in the phase ODEs jump by
    # ~10^3 across r=rCloud, which can cause LSODA to stall trying to refine
    # its step below min_step. We replace the step with a tanh bridge of
    # width SMOOTH_FRAC*rCloud so the rhs is C^infty everywhere. The width
    # is well below physical uncertainty in cloud-edge structure; mass
    # conservation holds to O(SMOOTH_FRAC^2).
    # =============================================================================
    SMOOTH_FRAC = 0.01
    delta = SMOOTH_FRAC * rCloud
    w_outside = 0.5 * (1.0 + np.tanh((r_arr - rCloud) / delta))

    # =============================================================================
    # Power-law profile
    # =============================================================================
    if params['dens_profile'].value == 'densPL':
        alpha = params['densPL_alpha'].value

        if alpha == 0:
            # Homogeneous cloud: constant nCore inside, blends to nISM at rCloud
            n_inside = np.full_like(r_arr, nCore)
        else:
            # Power-law profile: n = nCore * (r/rCore)^alpha
            n_inside = nCore * (r_arr / rCore) ** alpha
            # Inner core: constant density (rCore is far below rCloud, so the
            # smoothing band does not reach rCore in any realistic setup)
            n_inside[r_arr <= rCore] = nCore

        n_arr = n_inside * (1.0 - w_outside) + nISM * w_outside

    # =============================================================================
    # Bonnor-Ebert sphere profile
    # =============================================================================
    elif params['dens_profile'].value == 'densBE':
        f_rho_rhoc = params['densBE_f_rho_rhoc'].value

        # Convert radius to dimensionless xi coordinate
        xi_arr = be_r2xi(r_arr, params)

        # Get density ratio from interpolation function (interpolator already
        # handles xi values for r > rCloud — same as the original code path)
        rho_rhoc = f_rho_rhoc(xi_arr)

        n_inside = rho_rhoc * nCore
        n_arr = n_inside * (1.0 - w_outside) + nISM * w_outside

    else:
        raise ValueError(f"Unknown density profile: {params['dens_profile'].value}")

    # Convert back to scalar if input was scalar
    return _to_output(n_arr, was_scalar)


# B3 (docs/dev/turnaround-stall/PLAN.md, ruled 2026-11-20): the outside pressure is bridged at rCloud with the
# density's own tanh weight instead of a step: P_ISM wherever the outside pressure is built, and with alphaVirCloud > 0
# the cloud's own pressure too (get_press_cloud).
ISM_BRIDGE_FRAC = 0.01   # the tanh width in rCloud: get_density_profile's own SMOOTH_FRAC
ISM_BRIDGE_X = 20.0      # widths past which tanh is +-1 in double precision: the weight is exactly 0 or 1 there
ISM_BRIDGE_END = 1.0 + ISM_BRIDGE_X * ISM_BRIDGE_FRAC   # 1.2: r / rCloud from which the weight is 1


def ism_weight(r, rCloud):
    """The ambient medium's share at r [pc], 0.5 (1 + tanh((r - rCloud) / (ISM_BRIDGE_FRAC rCloud))): the weight
    get_density_profile gives nISM. Exactly 0 for r <= 0.8 rCloud and exactly 1 for r >= 1.2 rCloud, so
    P_ISM * ism_weight is the old step's value bit for bit there, and wherever P_ISM is 0. A NaN r gives 0, as the
    step's r >= rCloud did."""
    x = (r - rCloud) / (ISM_BRIDGE_FRAC * rCloud)
    if not x > -ISM_BRIDGE_X:          # NaN included
        return 0.0
    if x >= ISM_BRIDGE_X:
        return 1.0
    return 0.5 * (1.0 + math.tanh(x))


def get_press_cloud(r, fabs_ion, params, turbulent=True):
    """Pressure [code units] of the cloud's own gas ahead of the shell at radius r [pc]; 0 outside the cloud (PISM applies there),
    or past 1.2 rCloud when the turbulent term is on (it then fades with the density's bridge, B3).

    Two terms (docs/dev/phii-identity/weak-wind/PLAN.md section 11); TCloud defaults to 10 K (since 2026-10-29, gate
    G-CP3; 0 switches it off), alphaVirCloud to 0 (off):
      TCloud      thermal pressure of the molecular cloud gas, (mu_convert/mu_mol) n(r) k_B TCloud, while every ionising
                  photon is absorbed (if photons escape, the gas ahead is photoionised and get_press_ion gives it);
      alphaVirCloud  turbulent pressure of the cloud gas, rho sigma^2 with the virial dispersion of Krumholz & Matzner
                  (2009, eq. 17) sigma^2 = alphaVirCloud G M(<r) / (5 r), alpha_vir = 5 sigma^2 r / (G M(<r)) as in
                  Bertoldi & McKee (1992); M(<r) is the total mass inside r, stars included and spread like the gas as
                  in McKee & Tan (2003, eq. A3): the cloud gas inside r (mass_profile) / (1 - sfe). sigma is
                  the total one-dimensional dispersion, thermal included (Kauffmann et al.
                  2013, eq. 1), so the term adds rho sigma^2 minus the TCloud thermal pressure, whatever the ionisation
                  of the gas ahead; with both on and every photon absorbed the cloud gas pushes with rho sigma^2.
    Added after the photoionised and PISM terms at every place the outside pressure is built. turbulent=False gives the
    thermal part alone: swept's edge_speed uses it, because the isothermal jump there is a thermal sound-speed condition
    (with sigma in it a stalled shell would keep sweeping mass at U -> sigma).
    B3 (docs/dev/turnaround-stall/PLAN.md, ruling D3 = turb): with the turbulent term on (turbulent and
    alphaVirCloud > 0) the whole pressure is weighted by 1 - ism_weight(r), the density's own bridge, so it fades out
    to 1.2 rCloud instead of stopping at rCloud; otherwise (the defaults, and edge_speed's thermal call) the step at
    rCloud stays, bit for bit.
    """
    if params is None:                     # callers that only needed params when photons escape may pass None
        return 0.0
    T = params['TCloud'].value if 'TCloud' in params else 0.0          # hand-built params without the key: off
    a_vir = params['alphaVirCloud'].value if 'alphaVirCloud' in params else 0.0
    fade = turbulent and a_vir > 0.0   # B3 (ruling D3 = turb): the whole cloud pressure fades with the density's bridge
    if (T <= 0.0 and a_vir <= 0.0) or r >= params['rCloud'].value * (ISM_BRIDGE_END if fade else 1.0):
        return 0.0
    n = float(np.atleast_1d(get_density_profile(np.atleast_1d(r), params))[0])
    P = 0.0
    P_th = params['mu_convert'].value / params['mu_mol'].value * n * params['k_B'].value * T
    if T > 0.0 and fabs_ion >= 1.0:
        P += P_th
    if turbulent and a_vir > 0.0:
        from trinity.cloud_properties.mass_profile import get_mass_profile   # local: mass_profile imports this module
        # A1 (ruled 2026-11-20, docs/dev/turnaround-stall/b5-turbulent-pressure.md): alpha_vir counts the total mass
        # inside r as McKee & Tan (2003, eq. A3), the stars spread like the gas (their f_g = 1 - sfe), so
        # M(<r) = (cloud gas inside r) / (1 - sfe), the natal cloud's mass inside r
        M_tot = get_mass_profile(r, params) / (1.0 - params['sfe'].value)
        sigma_sq = a_vir * params['G'].value * M_tot / (5.0 * r)
        # Bertoldi & McKee's sigma is thermal + non-thermal, so only the non-thermal part rho sigma^2 - P_th is added
        P += max(params['mu_convert'].value * n * sigma_sq - P_th, 0.0)
    return P * (1.0 - ism_weight(r, params['rCloud'].value)) if fade else P


# ism_thermal_share 'fit' (docs/dev/ism-double-count/PLAN.md section 4.4, rulings D-5..D-7, 2026-11-27): the thermal
# share of the ambient's total pressure, P_th0 / P_ISM = A (P_ISM/k / 1e4 K cm^-3)^B, fitted on sigma_eff_ISM fit's own
# anchors (Ostriker & Kim 2022; Kim et al. 2024; Moon et al. 2021). These three-figure values define the share.
ISM_THERMAL_SHARE_A = 0.354
ISM_THERMAL_SHARE_B = -0.337


def get_press_ism_thermal(r, fabs_ion, params):
    """The ambient's own thermal pressure [code units] at r [pc] that the photoionised gas ahead replaces, for the
    caller to subtract from P_ISM w(r) (docs/dev/ism-double-count/PLAN.md, ruled 2026-11-27).

    P_ISM is the ambient's total pressure (sigma_eff_ISM). Once ionising photons escape (fabs_ion < 1, where
    get_press_ion adds P_i), P_i replaces the ambient's thermal pressure P_th0 but not its turbulent or magnetic part,
    so P_ISM + P_i would count P_th0 twice; this returns P_th0 w(r), with w the B3 bridge (ism_weight):
      P_th0 = min(s P_ISM, (mu_convert/mu_atom) nISM k_B TShell_ion),  s = min(1, A (P_ISM/k / 1e4 K cm^-3)^B),
    the cap (D-5) being the ambient's own gas as neutral gas at T_i (P_i/2 at nISM with Z_He_shell 1), which keeps
    P_out >= P_ISM for a hand-typed hot pair and never binds on sigma_eff_ISM fit's ambients at TShell_ion 8000 K
    (it would below ~5.5e3 K: cap / (s P_ISM) is 1.45 at its minimum, at P/k 459 K cm^-3 where s reaches 1). 0 under 'off', with every photon
    absorbed, with P_ISM 0, and inside 0.8 rCloud (w = 0), so the callers' sums are unchanged there bit for bit.
    Hand-built params without the key: off. Fresh limit only: the ionised gas is taken at the ambient's density
    (the relaxed limit, P_out -> P_ISM, is a caveat in PLAN section 2, not code)."""
    if params is None or 'ism_thermal_share' not in params or params['ism_thermal_share'].value != 'fit':
        return 0.0
    if not fabs_ion < 1.0:                 # get_press_ion's own gate (NaN included)
        return 0.0
    PISM = params['PISM'].value if 'PISM' in params else 0.0
    if not PISM > 0.0:
        return 0.0
    w = ism_weight(r, params['rCloud'].value)
    if w == 0.0:
        return 0.0
    # PISM is P/k in code units (K pc^-3); the share's P/k is in K cm^-3
    share = min(1.0, ISM_THERMAL_SHARE_A * (PISM / ndens_cgs2au / 1e4) ** ISM_THERMAL_SHARE_B)
    k_B = params['k_B'].value
    cap = params['mu_convert'].value / params['mu_atom'].value * params['nISM'].value * k_B * params['TShell_ion'].value
    return w * min(share * PISM * k_B, cap)
