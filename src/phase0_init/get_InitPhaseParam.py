#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:44:00 2022

@author: Jia Wei Teh

This script contains a function that computes the initial values for the
energy-driven phase (from a short free-streaming phase).
"""

import logging
import numpy as np

import src._functions.unit_conversions as cvt


# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# PHYSICAL CONSTANTS (with literature references)
# =============================================================================

# Energy fraction in bubble interior: E0 = (5/11) * Lw * dt
# From Weaver+77, Eq. 20 - assumes adiabatic index gamma = 5/3
WEAVER_ENERGY_FRACTION = 5.0 / 11.0

# Temperature coefficient in Weaver+77, Eq. 37
# T = 1.51e6 K * (L/10^36 erg/s)^(8/35) * (n/1 cm^-3)^(2/35) * t^(-6/35) * (1-xi)^0.4
WEAVER_TEMP_COEFFICIENT = 1.51e6  # Kelvin

# Reference luminosity for temperature scaling [erg/s]
WEAVER_L_REF = 1e36

# Minimum valid values to prevent division by zero
MIN_LUMINOSITY = 1e-100  # Prevent div by zero in Mdot calculation
MIN_MOMENTUM = 1e-100    # Prevent div by zero in velocity calculation
MIN_VELOCITY = 1e-100    # Prevent div by zero in dt_phase0 calculation



def get_y0(params):
    """
    Obtain initial values for the energy-driven (Weaver) phase by integrating
    a brief free-streaming phase from the SPS wind feedback at tSF.

    Physics references
    ------------------
    - Free-streaming phase duration: Rahner thesis Eq. 1.15
      https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf pg 17
    - Bubble energy: Weaver+77, Eq. 20
    - Bubble temperature: Weaver+77, Eq. 37

    Parameters
    ----------
    params : DescribedDict
        Must contain: tSF, sps_f, nCore, mu_convert, bubble_xi_Tb.

    Returns
    -------
    t0 : float [Myr]
        Start time for Weaver phase (= tSF + free-streaming duration).
    r0 : float [pc]
        Initial bubble outer radius R2 (= terminal velocity * free-streaming duration).
    v0 : float [pc/Myr]
        Initial expansion velocity (wind terminal velocity).
    E0 : float [au]
        Initial bubble thermal energy.
    T0 : float [K]
        Initial characteristic bubble temperature.
    """

    # Core properties - handle both DescribedItem and raw value access
    mu_convert = params['mu_convert'].value  # mass per H nucleus — for rho = n_H * mu_convert
    nCore = params['nCore'].value
    bubble_xi_Tb = params['bubble_xi_Tb'].value
        
    # =========================================================================
    # EXTRACT PARAMETERS
    # =========================================================================

    # Time of star formation [Myr]
    tSF = params['tSF'].value

    # SPS interpolation functions (sps_f naming as of PR-3 of SB99 -> SPS refactor).
    sps_f = params['sps_f'].value

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    if tSF < 0:
        raise ValueError(f"tSF must be non-negative, got {tSF}")

    if nCore <= 0:
        raise ValueError(f"nCore must be positive, got {nCore}")

    if not (0 <= bubble_xi_Tb <= 1):
        raise ValueError(f"bubble_xi_Tb must be in [0,1], got {bubble_xi_Tb}")

    # =========================================================================
    # GET SPS FEEDBACK VALUES AT tSF
    # =========================================================================

    # CRITICAL: Use WIND-ONLY quantities for wind velocity calculation.
    # The free-streaming phase describes the stellar wind expansion before
    # the Weaver phase. The wind terminal velocity v = 2L/pdot is only
    # physical when using wind-only L and pdot (not total which includes SNe).
    Lmech_W = sps_f['fLmech_W'](tSF)
    pdot_W = sps_f['fpdot_W'](tSF)

    # Validate SPS values
    if Lmech_W < MIN_LUMINOSITY:
        logger.warning(f"Lmech_W={Lmech_W} is very small at tSF={tSF} Myr")
        Lmech_W = MIN_LUMINOSITY

    if pdot_W < MIN_MOMENTUM:
        logger.warning(f"pdot_W={pdot_W} is very small at tSF={tSF} Myr")
        pdot_W = MIN_MOMENTUM

    # =========================================================================
    # COMPUTE WIND PROPERTIES (WIND-ONLY - BUG FIX)
    # =========================================================================

    # Mass loss rate from winds [AU units]
    # From: L = 0.5 * Mdot * v^2 and pdot = Mdot * v
    # => Mdot = pdot^2 / (2 * L)
    Mdot0 = pdot_W**2 / (2.0 * Lmech_W)

    # Terminal velocity from winds [pc/Myr in AU units]
    # From: v = 2 * L / pdot  (wind-only quantities)
    v0 = 2.0 * Lmech_W / pdot_W

    if v0 < MIN_VELOCITY:
        logger.warning(f"v0={v0} is very small, may cause numerical issues")
        v0 = MIN_VELOCITY

    # =========================================================================
    # COMPUTE FREE-STREAMING PHASE DURATION
    # =========================================================================

    # Ambient density [AU units: Msun/pc^3]
    # nCore is hydrogen nuclei density n_H; use mu_convert (=1.4) for mass density
    rhoa = nCore * mu_convert

    # Duration of free-streaming phase [Myr]
    # From Rahner thesis Eq. 1.15:
    # dt = sqrt(3 * Mdot / (4 * pi * rho_a * v^3))
    dt_phase0 = np.sqrt(3.0 * Mdot0 / (4.0 * np.pi * rhoa * v0**3))

    logger.debug(f"Free-streaming phase duration: dt_phase0 = {dt_phase0:.6e} Myr")

    # =========================================================================
    # COMPUTE INITIAL VALUES FOR WEAVER PHASE
    # =========================================================================

    # Start time for Weaver phase [Myr]
    t0 = tSF + dt_phase0

    # Initial separation / bubble radius [pc]
    r0 = v0 * dt_phase0

    # Initial bubble energy [AU units]
    # From Weaver+77, Eq. 20: E = (5/11) * L_w * t
    E0 = WEAVER_ENERGY_FRACTION * Lmech_W * dt_phase0

    # Initial temperature [K]
    # From Weaver+77, Eq. 37:
    # T = 1.51e6 * (L/10^36)^(8/35) * (n)^(2/35) * t^(-6/35) * (1-xi)^0.4
    T0 = WEAVER_TEMP_COEFFICIENT * \
         (Lmech_W * cvt.L_au2cgs / WEAVER_L_REF)**(8.0/35.0) * \
         (nCore * cvt.ndens_au2cgs)**(2.0/35.0) * \
         (dt_phase0)**(-6.0/35.0) * \
         (1.0 - bubble_xi_Tb)**0.4
         

    logger.info(
        f"Initial Weaver phase values: "
        f"t0={t0:.6f} Myr, r0={r0:.6e} pc, v0={v0:.6e} pc/Myr, "
        f"E0={E0:.6e}, T0={T0:.2e} K"
    )
        
    return t0, r0, v0, E0, T0











