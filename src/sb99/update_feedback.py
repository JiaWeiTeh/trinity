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
    Old (WRONG): vWind = 2 * LWind / (pdot_W + pdot_SN)
    New (CORRECT): vWind = 2 * LWind / pdot_W

    Naming convention:
    - Wind components: _W suffix (Lmech_W, pdot_W, fLmech_W, fpdot_W)
    - SN components: _SN suffix (Lmech_SN, pdot_SN, fLmech_SN, fpdot_SN)
    - Total components: _total suffix (Lmech_total, pdot_total)

    Side effects: Updates params dictionary with all feedback parameters including:
    - Raw SB99 values: Qi, Li, Ln, Lbol, LWind (=Lmech_W)
    - Derived values: vWind (wind velocity), pWindDot (=pdot_total), pWindDotDot (time derivative)
    - Separated components: F_ram_wind (=pdot_W), F_ram_SN (=pdot_SN)
    """

    SB99f = params['SB99f'].value

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
    vWind = (2. * Lmech_W / pdot_W)[()]  # ← FIXED!

    # Numerical derivative of total momentum rate for time evolution
    dt = 1e-9  # Myr (small timestep for derivative)
    pTotalDotDot = (SB99f['fpdot_total'](t + dt)[()] - SB99f['fpdot_total'](t - dt)[()]) / (2.0 * dt)

    # Update params dictionary with feedback values (backward compatible names)
    # Note: LWind = Lmech_W, pWindDot = pdot_total for backward compatibility
    updateDict(
        params,
        ['Qi', 'LWind', 'Lbol', 'Ln', 'Li', 'vWind', 'pWindDot', 'pWindDotDot'],
        [Qi, Lmech_W, Lbol, Ln, Li, vWind, pdot_total, pTotalDotDot],
    )

    # Store separated wind and SN momentum rates (with correct values!)
    params['F_ram_wind'].value = pdot_W    # Wind-only momentum rate
    params['F_ram_SN'].value = pdot_SN     # SN-only momentum rate

    # Return raw SB99 values (matching read_SB99 signature)
    return [t, Qi, Li, Ln, Lbol, Lmech_W, Lmech_SN, Lmech_total,
            pdot_W, pdot_SN, pdot_total]

