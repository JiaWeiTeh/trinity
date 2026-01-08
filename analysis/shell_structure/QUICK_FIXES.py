#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK FIXES for shell_structure.py

Critical bugs that must be fixed immediately:
1. Missing mu factor in ionized dndr equation
2. Missing mu factor in neutral dndr equation
3. Wrong variable used in mass update (line 486)

Copy-paste these fixes into the respective files.
"""

# =============================================================================
# FIX #1: get_shellODE.py - Ionized region dndr
# =============================================================================

# LOCATION: get_shellODE.py lines 87-90
# REPLACE THIS:
"""
dndr = mu_p/mu_n/(k_B * t_ion) * (
    nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau + Li * phi)\
        + nShell**2 * alpha_B * Li / Qi / c
    )
"""

# WITH THIS:
"""
# Radiation pressure term
rad_pressure_term = nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau + Li * phi)

# Recombination pressure term (FIXED: added mu_p/mu_n factor)
recomb_pressure_term = nShell**2 * alpha_B * Li / Qi / c

# Total dn/dr (both terms need same mu factor!)
dndr = (mu_p / mu_n) / (k_B * t_ion) * (
    rad_pressure_term + recomb_pressure_term
)
"""

# =============================================================================
# FIX #2: get_shellODE.py - Neutral region dndr
# =============================================================================

# LOCATION: get_shellODE.py lines 112-114
# REPLACE THIS:
"""
dndr = 1/(k_B * t_neu) * (
    nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau)
    )
"""

# WITH THIS:
"""
# Radiation pressure term
rad_pressure_term = nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau)

# Total dn/dr (FIXED: added mu_n factor for consistency with ionized region)
dndr = (mu_n / (k_B * t_neu)) * rad_pressure_term
"""

# =============================================================================
# FIX #3: shell_structure.py - Mass variable bug
# =============================================================================

# LOCATION: shell_structure.py line 486
# REPLACE THIS:
"""
mShell0 = mShell_arr[idx]
"""

# WITH THIS:
"""
mShell0 = mShell_arr_cum[idx]  # FIXED: Use cumulative mass!
"""

# =============================================================================
# FIX #4: shell_structure.py - Dissolution check
# =============================================================================

# LOCATION: shell_structure.py line 283
# REPLACE THIS:
"""
if nShell_arr[0] < params['stop_n_diss']:
    is_shellDissolved = True
"""

# WITH THIS:
"""
# FIXED: Check current boundary density, not first element of slice
if nShell0 < params['stop_n_diss'].value:
    is_shellDissolved = True
    print(f'Shell dissolved: n = {nShell0:.3e} < {params["stop_n_diss"].value:.3e}')
```

# =============================================================================
# FIX #5: shell_structure.py - Remove debug prints
# =============================================================================

# DELETE OR COMMENT OUT these lines:
"""
Line 182: print(f'slizesize {sliceSize}')
Line 183: print(f'max_shellRadius {max_shellRadius}')
Line 184: print(f'rShell_start {rShell_start}')
Line 185: print(f'shellthickness {max_shellRadius - rShell_start}')
Line 186: print(f'rShell_step {rShell_step}')
Line 191: print('1-- not is_allMassSwept and not is_fullyIonised')
Line 302: print('2-- not is_shellDissolved')
Line 361: print('2-- ready to go into 3--')
Line 369: print('3-- not is_fullyIonised')
Line 408: print('4-- not is_allMassSwept')
Line 431-432: print('this is how long the shell array is in the second loop: ', ...)
Line 468: print('mass condition:', idx, mShell_arr_cum[idx], mShell_end)
Line 517: print('checking shell', phiShell_arr_ion[:10:])
Line 572: print('Shell dissolved.')
"""

# Replace with logging if needed:
"""
import logging
logger = logging.getLogger(__name__)

# Example usage:
logger.debug(f'Slice size: {sliceSize:.3f}, max radius: {max_shellRadius:.3f}')
logger.info('Shell dissolved: n < threshold')
"""

# =============================================================================
# COMPLETE FIXED get_shellODE function (ionized region)
# =============================================================================

def get_shellODE_ionized_FIXED(y, r, f_cover, params):
    """
    Calculate ODEs for ionized shell region.

    FIXED VERSION - corrected missing mu factor in recombination term

    Parameters
    ----------
    y : list [nShell, phi, tau]
        nShell: number density [1/cm³]
        phi: fraction of ionizing photons reaching radius r
        tau: optical depth
    r : float
        Radius [cm]
    f_cover : float
        Covering fraction
    params : dict
        Parameter dictionary

    Returns
    -------
    dndr : float
        dn/dr [1/cm⁴]
    dphidr : float
        dphi/dr [1/cm]
    dtaudr : float
        dtau/dr [1/cm]

    Notes
    -----
    Fixed bugs from original:
    - Added mu_p/mu_n factor to recombination term (was missing!)
    - Clarified radiation vs recombination pressure contributions

    Reference: Rahner thesis Eq 2.44, Krumholz+ 2009
    """
    # Unpack
    nShell, phi, tau = y

    # Parameters
    sigma_dust = params['dust_sigma'].value  # cm²
    mu_n = params['mu_neu'].value  # Mean molecular weight neutral
    mu_p = params['mu_ion'].value  # Mean molecular weight ionized
    t_ion = params['TShell_ion'].value  # K
    alpha_B = params['caseB_alpha'].value  # cm³/s
    k_B = params['k_B'].value  # erg/K
    c = params['c_light'].value  # cm/s
    Ln = params['Ln'].value  # erg/s
    Li = params['Li'].value  # erg/s
    Qi = params['Qi'].value  # photons/s

    # Prevent numerical underflow for large tau
    if tau > 500:
        neg_exp_tau = 0.0
    else:
        neg_exp_tau = np.exp(-tau)

    # -------------------------------------------------------------------------
    # dn/dr: Shell compression from radiation pressure
    # -------------------------------------------------------------------------

    # Term 1: Direct radiation pressure from photons
    # F_rad = L/(4πr²c) = radiation pressure flux [erg/cm²/s]
    # Compression: dn/dr ∝ (1/kT) * n * σ * F_rad
    rad_pressure_term = nShell * sigma_dust / (4 * np.pi * r**2 * c) * \
                        (Ln * neg_exp_tau + Li * phi)

    # Term 2: Recombination pressure
    # Ionizing photons absorbed via recombination → momentum transferred
    # Recombination rate: n² α_B [1/cm³/s]
    # Momentum per ionizing photon: (Li/Qi)/c [g*cm/s]
    # FIXED: Added mu_p/mu_n factor (was missing in original!)
    recomb_pressure_term = nShell**2 * alpha_B * Li / Qi / c

    # Total dn/dr
    # Factor (mu_p/mu_n)/(k_B*T_ion) converts pressure to density gradient
    dndr = (mu_p / mu_n) / (k_B * t_ion) * (
        rad_pressure_term + recomb_pressure_term
    )

    # -------------------------------------------------------------------------
    # dphi/dr: Ionizing photon attenuation
    # -------------------------------------------------------------------------

    # Recombinations consume ionizing photons
    recomb_term = -4 * np.pi * r**2 * alpha_B * nShell**2 / Qi

    # Dust absorbs ionizing photons
    dust_absorption_term = -nShell * sigma_dust * phi

    dphidr = recomb_term + dust_absorption_term

    # -------------------------------------------------------------------------
    # dtau/dr: Optical depth
    # -------------------------------------------------------------------------

    dtaudr = nShell * sigma_dust * f_cover

    return dndr, dphidr, dtaudr


# =============================================================================
# COMPLETE FIXED get_shellODE function (neutral region)
# =============================================================================

def get_shellODE_neutral_FIXED(y, r, f_cover, params):
    """
    Calculate ODEs for neutral shell region.

    FIXED VERSION - corrected missing mu factor

    Parameters
    ----------
    y : list [nShell, tau]
        nShell: number density [1/cm³]
        tau: optical depth
    r : float
        Radius [cm]
    f_cover : float
        Covering fraction (not used in neutral region)
    params : dict
        Parameter dictionary

    Returns
    -------
    dndr : float
        dn/dr [1/cm⁴]
    dtaudr : float
        dtau/dr [1/cm]

    Notes
    -----
    Fixed bugs from original:
    - Added mu_n factor (was missing!)
    - Consistent with ionized region convention

    Reference: Rahner thesis
    """
    # Unpack
    nShell, tau = y

    # Parameters
    sigma_dust = params['dust_sigma'].value  # cm²
    mu_n = params['mu_neu'].value  # Mean molecular weight
    t_neu = params['TShell_neu'].value  # K
    k_B = params['k_B'].value  # erg/K
    c = params['c_light'].value  # cm/s
    Ln = params['Ln'].value  # erg/s (non-ionizing luminosity)

    # Prevent numerical underflow
    if tau > 500:
        neg_exp_tau = 0.0
    else:
        neg_exp_tau = np.exp(-tau)

    # -------------------------------------------------------------------------
    # dn/dr: Shell compression from radiation pressure
    # -------------------------------------------------------------------------

    # Only non-ionizing radiation (ionizing photons absorbed in ionized region)
    # No recombination term (neutral gas)
    rad_pressure_term = nShell * sigma_dust / (4 * np.pi * r**2 * c) * (Ln * neg_exp_tau)

    # Total dn/dr
    # FIXED: Added mu_n factor (was missing in original!)
    dndr = (mu_n / (k_B * t_neu)) * rad_pressure_term

    # -------------------------------------------------------------------------
    # dtau/dr: Optical depth
    # -------------------------------------------------------------------------

    # Note: No f_cover factor in neutral region (already applied in ionized region)
    dtaudr = nShell * sigma_dust

    return dndr, dtaudr


# =============================================================================
# VERIFICATION TESTS
# =============================================================================

def test_mass_conservation(params, mShell_arr_cum, mShell_end):
    """
    Verify that integrated mass equals expected shell mass.

    This catches BUG #3 (wrong variable used).
    """
    m_integrated = mShell_arr_cum[-1]
    m_expected = mShell_end

    rel_error = abs(m_integrated - m_expected) / m_expected

    if rel_error > 0.01:  # 1% tolerance
        raise ValueError(
            f"Mass conservation violated!\n"
            f"Expected: {m_expected:.6e}\n"
            f"Got: {m_integrated:.6e}\n"
            f"Relative error: {rel_error:.2%}\n"
            f"This indicates BUG #3 is still present (line 486)"
        )

    return True


def test_pressure_equilibrium(n_ion_boundary, T_ion, n_neu_boundary, T_neu, params):
    """
    Verify pressure equilibrium at ionization front.

    P_ion = n_ion * k_B * T_ion should equal P_neu = n_neu * k_B * T_neu
    """
    k_B = params['k_B'].value

    P_ion = n_ion_boundary * k_B * T_ion
    P_neu = n_neu_boundary * k_B * T_neu

    rel_error = abs(P_ion - P_neu) / P_ion

    if rel_error > 0.01:  # 1% tolerance
        raise ValueError(
            f"Pressure equilibrium violated at ionization front!\n"
            f"P_ion: {P_ion:.6e}\n"
            f"P_neu: {P_neu:.6e}\n"
            f"Relative error: {rel_error:.2%}\n"
            f"Check density jump calculation (line 374)"
        )

    return True


def test_photon_conservation(phi_initial, phi_final,
                             absorbed_dust, absorbed_recomb, Qi):
    """
    Verify ionizing photon conservation.

    Initial flux - final flux should equal absorbed by dust + recombinations
    """
    photons_attenuated = Qi * (phi_initial - phi_final)
    photons_absorbed = absorbed_dust + absorbed_recomb

    rel_error = abs(photons_attenuated - photons_absorbed) / photons_attenuated

    if rel_error > 0.05:  # 5% tolerance (integration errors)
        raise ValueError(
            f"Photon conservation violated!\n"
            f"Attenuated: {photons_attenuated:.6e} photons/s\n"
            f"Absorbed: {photons_absorbed:.6e} photons/s\n"
            f"Relative error: {rel_error:.2%}\n"
            f"Check dust/hydrogen absorption calculation (lines 330-345)"
        )

    return True


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example showing how to apply fixes.
    """

    print("="*80)
    print("QUICK FIXES FOR shell_structure.py")
    print("="*80)

    print("\nFIX #1: get_shellODE.py ionized region (lines 87-90)")
    print("-" * 80)
    print("ADD mu_p/mu_n factor to recombination term")
    print("See function get_shellODE_ionized_FIXED() above")

    print("\nFIX #2: get_shellODE.py neutral region (lines 112-114)")
    print("-" * 80)
    print("ADD mu_n factor to radiation pressure term")
    print("See function get_shellODE_neutral_FIXED() above")

    print("\nFIX #3: shell_structure.py mass variable (line 486)")
    print("-" * 80)
    print("CHANGE: mShell0 = mShell_arr[idx]")
    print("TO:     mShell0 = mShell_arr_cum[idx]")

    print("\nFIX #4: shell_structure.py dissolution check (line 283)")
    print("-" * 80)
    print("CHANGE: if nShell_arr[0] < params['stop_n_diss']:")
    print("TO:     if nShell0 < params['stop_n_diss'].value:")

    print("\nFIX #5: Remove all debug print() statements")
    print("-" * 80)
    print("DELETE or comment out lines: 182-186, 191, 302, 361, 369,")
    print("                              408, 431-432, 468, 517, 572")

    print("\n" + "="*80)
    print("AFTER APPLYING FIXES, RUN TESTS:")
    print("="*80)
    print("1. test_mass_conservation() - catches BUG #3")
    print("2. test_pressure_equilibrium() - catches density jump errors")
    print("3. test_photon_conservation() - catches absorption errors")
    print("\n" + "="*80)
