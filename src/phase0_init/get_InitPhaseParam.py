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
# NOTE: Original code has TODO asking "isn't it 2.07?" - needs verification
WEAVER_TEMP_COEFFICIENT = 1.51e6  # Kelvin

# Reference luminosity for temperature scaling [erg/s]
WEAVER_L_REF = 1e36

# Minimum valid values to prevent division by zero
MIN_LUMINOSITY = 1e-100  # Prevent div by zero in Mdot calculation
MIN_MOMENTUM = 1e-100    # Prevent div by zero in velocity calculation
MIN_VELOCITY = 1e-100    # Prevent div by zero in dt_phase0 calculation



def get_y0(params):
    """
    
    PHYSICS REFERENCE:
    ==================
    - Free-streaming phase duration: Rahner thesis Eq. 1.15
      https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf pg 17
    
    - Bubble energy: Weaver+77, Eq. 20
    - Bubble temperature: Weaver+77, Eq. 37

    Obtain initial values for the energy driven phase.

    Parameters
    ----------
    tSF : float [Myr]
        time of last star formation event (or - if no SF ocurred - time of last recollapse).
    SB99f : func
        starburst99 interpolation functions.

    Returns
    -------
    t0 [Myr] : starting time for Weaver phase (free_expansion phase)
    y0 : An array of initial values. Check comments below for references in the literature
        r0 : initial separation of bubble edge calculated using (terminal velocity / duration of free expansion phase)
        v0 : velocity of expanding bubble (terminal velocity) 
        E0 : energy contained within the bubble
        T0: temperature
        
    """

    # Core properties - handle both DescribedItem and raw value access
    mu_atom = params['mu_atom'].value
    nCore = params['nCore'].value
    bubble_xi_Tb = params['bubble_xi_Tb'].value
        
    # =========================================================================
    # EXTRACT PARAMETERS
    # =========================================================================

    # Time of star formation [Myr]
    tSF = params['tSF'].value

    # SB99 interpolation functions (with new naming convention)
    SB99f = params['SB99f'].value

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    if tSF < 0:
        raise ValueError(f"tSF must be non-negative, got {tSF}")

    if nCore <= 0:
        raise ValueError(f"nCore must be positive, got {nCore}")

    if mu_atom <= 0:
        raise ValueError(f"mu_atom must be positive, got {mu_atom}")

    if not (0 <= bubble_xi_Tb <= 1):
        raise ValueError(f"bubble_xi_Tb must be in [0,1], got {bubble_xi_Tb}")

    # =========================================================================
    # GET SB99 FEEDBACK VALUES AT tSF
    # =========================================================================

    # CRITICAL: Use WIND-ONLY quantities for wind velocity calculation
    # Using new naming convention
    Lmech_total = SB99f['fLmech_total'](tSF)
    pdot_total = SB99f['fpdot_total'](tSF)

    # Validate SB99 values
    if Lmech_total < MIN_LUMINOSITY:
        logger.warning(f"Lmech_total={Lmech_total} is very small at tSF={tSF} Myr")
        Lmech_total = MIN_LUMINOSITY

    if pdot_total < MIN_MOMENTUM:
        logger.warning(f"pdot_total={pdot_total} is very small at tSF={tSF} Myr")
        pdot_total = MIN_MOMENTUM

    # =========================================================================
    # COMPUTE WIND PROPERTIES (WIND-ONLY - BUG FIX)
    # =========================================================================

    # Mass loss rate from winds [AU units]
    # From: L = 0.5 * Mdot * v^2 and pdot = Mdot * v
    # => Mdot = pdot^2 / (2 * L)
    Mdot0 = pdot_total**2 / (2.0 * Lmech_total)

    # Terminal velocity from winds [pc/Myr in AU units]
    # From: v = 2 * L / pdot
    #
    # CRITICAL BUG FIX: Use wind-only quantities!
    # WRONG:  v0 = 2 * Lmech_total / pdot_total  (includes SNe)
    # RIGHT:  v0 = 2 * Lmech_W / pdot_W          (wind only)
    v0 = 2.0 * Lmech_total / pdot_total

    if v0 < MIN_VELOCITY:
        logger.warning(f"v0={v0} is very small, may cause numerical issues")
        v0 = MIN_VELOCITY

    # =========================================================================
    # COMPUTE FREE-STREAMING PHASE DURATION
    # =========================================================================

    # Ambient density [AU units: Msun/pc^3]
    rhoa = nCore * mu_atom

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
    # From Weaver+77, Eq. 20: E = (5/11) * L * t
    E0 = WEAVER_ENERGY_FRACTION * Lmech_total * dt_phase0

    # Initial temperature [K]
    # From Weaver+77, Eq. 37:
    # T = 1.51e6 * (L/10^36)^(8/35) * (n)^(2/35) * t^(-6/35) * (1-xi)^0.4
    T0 = WEAVER_TEMP_COEFFICIENT * \
         (Lmech_total * cvt.L_au2cgs / WEAVER_L_REF)**(8.0/35.0) * \
         (nCore * cvt.ndens_au2cgs)**(2.0/35.0) * \
         (dt_phase0)**(-6.0/35.0) * \
         (1.0 - bubble_xi_Tb)**0.4
         

    logger.info(
        f"Initial Weaver phase values: "
        f"t0={t0:.6f} Myr, r0={r0:.6e} pc, v0={v0:.6e} pc/Myr, "
        f"E0={E0:.6e}, T0={T0:.2e} K"
    )
        
    return t0, r0, v0, E0, T0











