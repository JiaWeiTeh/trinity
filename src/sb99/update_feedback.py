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
    separated wind and SNe components with consistent naming.

    Parameters
    ----------
    t : float
        Current time [Myr]
    params : DescribedDict
        Global parameters dictionary containing SB99f interpolation functions

    Returns
    -------
    list : [Qi, LWind, Lbol, Ln, Li, vWind, pTotalDot, pTotalDotDot]
        Stellar feedback parameters at time t

    Notes
    -----
    FIXED: Wind velocity now correctly uses wind-only momentum rate!
    Old (WRONG): vWind = 2 * LWind / (pdot_wind + pdot_SN)
    New (CORRECT): vWind = 2 * LWind / pdot_wind

    Naming convention:
    - Wind components: _wind suffix (Lmech_wind, pdot_wind, fLmech_wind, fpdot_wind)
    - SNe components: _SN suffix (Lmech_SN, pdot_SN, fLmech_SN, fpdot_SN)
    - Total components: _total suffix (Lmech_total, pdot_total)

    Side effects: Updates params dictionary with all feedback parameters
    """

    SB99f = params['SB99f'].value

    # Interpolate luminosities (consistent key naming)
    LWind = SB99f['fLmech_wind'](t)[()]  # Wind mechanical luminosity [erg/s]
    Lbol = SB99f['fLbol'](t)[()]         # Bolometric luminosity [erg/s]
    Ln = SB99f['fLn'](t)[()]             # Non-ionizing luminosity [erg/s]
    Li = SB99f['fLi'](t)[()]             # Ionizing luminosity [erg/s]
    Qi = SB99f['fQi'](t)[()]             # Ionizing photon rate [s⁻¹]

    # Interpolate momentum rates (NOW PROPERLY SEPARATED WITH CONSISTENT NAMING!)
    pdot_wind = SB99f['fpdot_wind'](t)[()]    # Wind-only momentum rate
    pdot_SN = SB99f['fpdot_SN'](t)[()]        # SNe-only momentum rate
    pTotalDot = SB99f['fpdot_total'](t)[()]   # Total momentum rate (wind + SNe)

    # =========================================================================
    # CRITICAL FIX: Wind velocity using WIND-ONLY momentum rate
    # =========================================================================
    # Formula: v_wind = 2 * L_wind / pdot_wind
    # OLD BUG: Used pTotalDot (wind + SNe) instead of pdot_wind
    # This caused 10-80% error depending on SNe contribution!
    vWind = (2. * LWind / pdot_wind)[()]  # ← FIXED!

    # Numerical derivative of total momentum rate for time evolution
    dt = 1e-9  # Myr (small timestep for derivative)
    pTotalDotDot = (SB99f['fpdot_total'](t + dt)[()] - SB99f['fpdot_total'](t - dt)[()]) / (2.0 * dt)

    # Update params dictionary with feedback values
    # Note: pWindDot variable name kept for backward compatibility, but now contains pTotalDot
    updateDict(
        params,
        ['Qi', 'LWind', 'Lbol', 'Ln', 'Li', 'vWind', 'pWindDot', 'pWindDotDot'],
        [Qi, LWind, Lbol, Ln, Li, vWind, pTotalDot, pTotalDotDot],
    )

    # Store separated wind and SNe momentum rates (with correct values!)
    params['F_ram_wind'].value = pdot_wind  # Now directly from interpolation (correct!)
    params['F_ram_SN'].value = pdot_SN

    # Return values (pWindDot actually contains pTotalDot for backward compatibility)
    return [Qi, LWind, Lbol, Ln, Li, vWind, pTotalDot, pTotalDotDot]

