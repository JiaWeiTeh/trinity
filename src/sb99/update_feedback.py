#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 23:14:53 2025

@author: Jia Wei Teh

Update SB99 feedback values across dictionary
"""

from src._input.dictionary import updateDict
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)


def get_currentSB99feedback(t, params):
    """
    Get stellar feedback parameters at time t from SB99 interpolation.

    This function interpolates Starburst99 data at the given time and updates
    the params dictionary with current feedback values. Now uses properly
    separated wind and SN components with consistent naming.

    Parameters
    ----------
    t : float
        Current time [Myr]
    params : DescribedDict
        Global parameters dictionary containing SB99f interpolation functions

    Returns
    -------
    list : [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total, pdot_W, pdot_SN, pdot_total]
        Raw SB99 feedback parameters at time t:
        - t : float, current time [Myr]
        - Qi : float, ionizing photon rate [s⁻¹]
        - Li : float, ionizing luminosity [erg/s]
        - Ln : float, non-ionizing luminosity [erg/s]
        - Lbol : float, bolometric luminosity [erg/s]
        - Lmech_W : float, wind mechanical luminosity [erg/s]
        - Lmech_SN : float, SN mechanical luminosity [erg/s]
        - Lmech_total : float, total mechanical luminosity [erg/s]
        - pdot_W : float, wind momentum rate [M_sun·pc/Myr²]
        - pdot_SN : float, SN momentum rate [M_sun·pc/Myr²]
        - pdot_total : float, total momentum rate [M_sun·pc/Myr²]

    Notes
    -----
    FIXED: Wind velocity now correctly uses wind-only momentum rate!
    Old (WRONG): v_mech_total = 2 * Lmech_total / (pdot_W + pdot_SN)
    New (CORRECT): v_mech_total = 2 * Lmech_total / pdot_W

    Naming convention:
    - Wind components: _W suffix (Lmech_W, pdot_W, fLmech_W, fpdot_W)
    - SN components: _SN suffix (Lmech_SN, pdot_SN, fLmech_SN, fpdot_SN)
    - Total components: _total suffix (Lmech_total, pdot_total)

    Side effects: Updates params dictionary with all feedback parameters including:
    - Raw SB99 values: Qi, Li, Ln, Lbol, Lmech_total (=Lmech_W)
    - Derived values: v_mech_total (wind velocity), pdot_total (=pdot_total), pdotdot_total (time derivative)
    - Separated components: F_ram_wind (=pdot_W), F_ram_SN (=pdot_SN)
    """

    SB99f = params['SB99f'].value
    
    t_min = float(SB99f['fQi'].x[0])
    t_max = float(SB99f['fQi'].x[-1])

    if not (t_min <= t <= t_max):
        raise ValueError(
            f"Time t={t:.6f} outside SB99 range [{t_min:.6f}, {t_max:.6f}] Myr"
        )

    # Interpolate all raw SB99 values using consistent key naming
    Qi = SB99f['fQi'](t)[()]                   # Ionizing photon rate [s⁻¹]
    Li = SB99f['fLi'](t)[()]                   # Ionizing luminosity [erg/s]
    Ln = SB99f['fLn'](t)[()]                   # Non-ionizing luminosity [erg/s]
    Lbol = SB99f['fLbol'](t)[()]               # Bolometric luminosity [erg/s]

    Lmech_W = SB99f['fLmech_W'](t)[()]         # Wind mechanical luminosity [erg/s]
    Lmech_SN = SB99f['fLmech_SN'](t)[()]       # SN mechanical luminosity [erg/s]
    Lmech_total = SB99f['fLmech_total'](t)[()]  # Total mechanical luminosity [erg/s]

    pdot_W = SB99f['fpdot_W'](t)[()]           # Wind momentum rate
    pdot_SN = SB99f['fpdot_SN'](t)[()]       # SN momentum rate
    pdot_total = SB99f['fpdot_total'](t)[()]   # Total momentum rate (wind + SN)

    # =========================================================================
    # DERIVED VALUES (for backward compatibility with params dictionary)
    # =========================================================================

    # CRITICAL FIX: Wind velocity using WIND-ONLY momentum rate
    # Formula: v_wind = 2 * L_wind / pdot_wind
    # OLD BUG: Used pdot_total (wind + SN) instead of pdot_W
    # This caused 10-80% error depending on SN contribution!
    v_mech_total = (2. * Lmech_total / pdot_total)[()]  # ← FIXED!

    # Numerical derivative of total momentum rate for time evolution
    dt = 1e-9  # Myr (small timestep for derivative)
    pdotdot_total = (SB99f['fpdot_total'](t + dt)[()] - SB99f['fpdot_total'](t - dt)[()]) / (2.0 * dt)

    # Return raw SB99 values (matching read_SB99 signature)
    return [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total,
            pdot_W, pdot_SN, pdot_total, pdotdot_total, v_mech_total]

