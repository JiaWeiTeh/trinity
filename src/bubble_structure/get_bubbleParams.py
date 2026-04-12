#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:36:10 2022

@author: Jia Wei Teh

This script contains useful functions that help compute properties and parameters
of the bubble. grep "Section" so jump between different sections.
"""
# libraries
import numpy as np
import logging
import astropy.units as u
import src._functions.unit_conversions as cvt

logger = logging.getLogger(__name__)

#--

# =============================================================================
# This section contains function which computes the ODEs that dictate the 
# structure (e.g., temperature, velocity) of the bubble. 
# =============================================================================

def delta2dTdt(t, T, delta):
    """
    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    
    Parameters
    ----------
    t : float
        time.
    T : float
        Temperature at xi = r/R2.

    Returns
    -------
    dTdt : float
    """
    dTdt = (T/t) * delta

    return dTdt


def dTdt2delta(t, T, dTdt):
    """
    See Pg 79, Eq A5, https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf.
    
    Parameters
    ----------
    t : float
        time.
    T : float
        DESCRIPTION.

    Returns
    -------
    delta : float
    """
    
    delta = (t/T) * dTdt
    
    return delta



def cool_beta_to_Ebdot(params):
    # old code: beta_to_Edot(), previously beta2Edot()
    """
    Convert Weaver cooling parameter beta to dE_b/dt.

    See pg 80, Eq A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf

    Equation implemented (bubble energy rate):

        E_b_dot = [ 2*pi * Pb_dot * d^2
                  + 3 * E_b * R_b_dot * R_b^2 * (1 - c/(E_b+c))
                  - a * R_ts^3 * E_b^2 / (E_b + c) ]
                 / [ d * (1 - c/(E_b+c)) ]

        a ≡ (3/2) * F_ram_dot / F_ram           [1/time]
        c ≡ (3/4) * F_ram     * R_ts            [energy]
        d ≡ R_b^3 - R_ts^3                      [length^3]

    Code ↔ equation mapping
    -----------------------
    Pb_dot        <- d(P_b)/dt (from beta definition: beta = -(t/Pb)(dPb/dt))
    Eb            <- E_b (bubble energy)
    R2, v2        <- R_b (outer bubble radius) and R_b_dot
    R1            <- R_ts (termination shock radius, inner)
    pdot_total    <- F_ram (total mechanical momentum injection rate)
    pdotdot_total <- F_ram_dot
    a_coeff       <- equation symbol `a`  = (3/2) * pdotdot_total / pdot_total
    c_coeff       <- equation symbol `c`  = (3/4) * pdot_total * R1
    d_coeff       <- equation symbol `d`  = R2^3 - R1^3
    c_frac        <- c/(E_b + c)

    Parameters
    ----------
    params : dict-like
        Must provide .value for: Pb, cool_beta, t_now, R1, R2, v2, Eb,
        pdot_total, pdotdot_total.

    Returns
    -------
    Eb_dot : float
        d(E_b)/dt.
    """
    # dPb/dt from the Weaver cooling parameter: beta = -(t/Pb)(dPb/dt)
    Pb_dot = -params['Pb'].value * params['cool_beta'].value / params['t_now'].value

    # Pull state
    R1 = params['R1'].value                        # R_ts
    R2 = params['R2'].value                        # R_b
    v2 = params['v2'].value                        # R_b_dot
    Eb = params['Eb'].value
    pdot_total = params['pdot_total'].value        # F_ram
    pdotdot_total = params['pdotdot_total'].value  # F_ram_dot

    # Equation coefficients (see docstring)
    a_coeff = 1.5 * pdotdot_total / pdot_total
    c_coeff = 0.75 * pdot_total * R1
    d_coeff = R2**3 - R1**3
    c_frac = c_coeff / (Eb + c_coeff)              # c/(E_b + c)

    # Main equation (Rahner thesis A12)
    numerator = (
        2 * np.pi * Pb_dot * d_coeff**2
        + 3 * Eb * v2 * R2**2 * (1 - c_frac)
        - a_coeff * R1**3 * Eb**2 / (Eb + c_coeff)
    )
    denominator = d_coeff * (1 - c_frac)

    Eb_dot = numerator / denominator
    return Eb_dot


def Ebdot_to_cool_beta(bubble_P, r1, bubble_Edot, my_params):
    # old code: Edot_to_beta(), previously Edot2beta()
    """
    Inverse of cool_beta_to_Ebdot: convert dE_b/dt to Weaver cooling parameter beta.

    See pg 80, Eq A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf

    Solves the A12 equation for Pb_dot and then returns
        cool_beta = - Pb_dot * t_now / P_b.

    See cool_beta_to_Ebdot for the equation↔code variable map.

    Parameters
    ----------
    bubble_P : float
        Bubble pressure P_b.
    r1 : float
        Termination shock radius R_ts (inner).
    bubble_Edot : float
        d(E_b)/dt.
    my_params : dict-like
        Must provide t_now, pdot_total, pdotdot_total, R2, v2, Eb
        (plain float values, not .value-wrapped).

    Returns
    -------
    cool_beta : float
        Weaver cooling parameter beta = -(t/P_b) * dP_b/dt.
    """
    t_now = my_params["t_now"]
    pdot_total = my_params["pdot_total"]           # F_ram
    pdotdot_total = my_params["pdotdot_total"]     # F_ram_dot
    R2 = my_params["R2"]                           # R_b
    v2 = my_params["v2"]                           # R_b_dot
    Eb = my_params["Eb"]

    # Equation coefficients
    a_coeff = 1.5 * pdotdot_total / pdot_total
    c_coeff = 0.75 * pdot_total * r1
    d_coeff = R2**3 - r1**3
    c_frac = c_coeff / (Eb + c_coeff)

    # Invert A12 for Pb_dot
    Pb_dot = (
        d_coeff * (1 - c_frac) * bubble_Edot
        - 3 * Eb * v2 * R2**2 * (1 - c_frac)
        + a_coeff * r1**3 * Eb**2 / (Eb + c_coeff)
    ) / (2 * np.pi * d_coeff**2)

    cool_beta = -Pb_dot * t_now / bubble_P
    return cool_beta



# =============================================================================
# Section: conversion between bubble energy and pressure. Calculation of ram pressure.
# =============================================================================

def bubble_E2P(Eb, r2, r1, gamma):
    """
    This function relates bubble energy to buble pressure

    Parameters 
    ----------
    Eb : float 
        Bubble energy.
    r1 : float 
        Inner radius of bubble (outer radius of wind cavity).
    r2 (aka rShell.rBubble) : float 
        Outer radius of bubble (inner radius of ionised shell).

    Returns
    -------
    bubble_P : float 
        Bubble pressure.

    # Note:
        # old code: PfromE()
    """
    
    # Make sure units are in cgs
    r1 *= cvt.pc2cm
    r2 *= cvt.pc2cm
    Eb *= cvt.E_au2cgs
    # avoid division by zero
    r2 += 1e-10 
    
    # pressure, see https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf
    # pg71 Eq 6.
    Pb = (gamma - 1) * Eb / (r2**3 - r1**3) / (4 * np.pi / 3)
    # return back in au
    return Pb * cvt.Pb_cgs2au
    
def bubble_P2E(Pb, r2, r1, gamma):
    """
    This function relates bubble pressure to buble energy 

    Parameters [cgs]
    ----------
    Pb : float
        Bubble pressure.
    r1 : float
        Inner radius of bubble (outer radius of wind cavity).
    r2 (aka rShell): float
        Outer radius of bubble (inner radius of ionised shell).

    Returns
    -------
    Eb : float
        Bubble energy.

    """
    # Note:
        # old code: EfromP()
    # see bubble_E2P()
    # Make sure units are in cgs
    r2 = r2.to(u.cm)
    r1 = r1.to(u.cm)
    Pb = Pb.to(u.g/u.cm/u.s**2)
    Eb = 4 * np.pi / 3 / (gamma - 1) * (r2**3 - r1**3) * Pb
    
    return Eb.to(u.erg)

def pRam(r, Lwind, v_mech_total):
    """
    This function calculates the ram pressure.

    returns in [au].

    Parameters
    ----------
    r : float
        Radius of outer edge of bubble.
    Lwind : float
        Mechanical wind luminosity.
    v_mech_total : float
        terminal velocity of wind.

    Returns
    -------
    Ram pressure.
    """
    # Note:
        # old code: Pram()

    return Lwind / (2 * np.pi * r**2 * v_mech_total)


def get_effective_bubble_pressure(current_phase, Eb, R2, R1, gamma,
                                   Lmech_total=None, v_mech_total=None,
                                   t=None, tSF=None):
    """
    Effective interior pressure felt by the shell.

    Energy phase: thermal pressure from hot bubble via bubble_E2P.
    Momentum phase: ram pressure from freely streaming wind via pRam.

    This function MUST be called in both the ODE and in compute_derived_quantities
    to guarantee consistency between the integrator and diagnostics.

    Parameters
    ----------
    current_phase : str
        Current simulation phase ('energy', 'momentum', etc.)
    Eb : float
        Bubble energy [au]
    R2 : float
        Outer bubble radius [pc]
    R1 : float
        Inner bubble radius [pc]
    gamma : float
        Adiabatic index
    Lmech_total : float, optional
        Mechanical wind luminosity (required for momentum phase)
    v_mech_total : float, optional
        Terminal wind velocity (required for momentum phase)
    t : float, optional
        Current time [Myr] (for early-phase R1 ramp-up)
    tSF : float, optional
        Star formation time [Myr] (for early-phase R1 ramp-up)

    Returns
    -------
    press_bubble : float
        Effective bubble pressure [au]
    """
    if current_phase == 'momentum':
        # Momentum phase: ram pressure from freely streaming wind
        return pRam(R2, Lmech_total, v_mech_total)
    elif current_phase == 'transition':
        # Transition phase: use max(P_thermal, P_ram) to ensure smooth
        # handoff to momentum phase.  As Eb decays on the sound-crossing
        # timescale, P_thermal drops while P_ram stays roughly constant.
        # By the time Eb hits the energy floor, P_ram already dominates,
        # so switching to momentum phase (P_ram only) is continuous.
        P_thermal = bubble_E2P(Eb, R2, R1, gamma)
        P_ram = pRam(R2, Lmech_total, v_mech_total)
        P_eff = max(P_thermal, P_ram)
        logger.debug(f"Transition pressure: P_thermal={P_thermal:.4e}, P_ram={P_ram:.4e}, "
                     f"using={'P_ram' if P_ram >= P_thermal else 'P_thermal'}, Eb={Eb:.4e}")
        return P_eff
    else:
        # Energy/implicit phases: thermal pressure from hot bubble.
        # Include the early-phase R1 ramp-up if timing info provided
        dt_switchon = 1e-3
        tmin = dt_switchon

        if t is not None and tSF is not None:
            if t <= (tmin + tSF):
                R1_tmp = (t - tSF) / tmin * R1
                return bubble_E2P(Eb, R2, R1_tmp, gamma)

        return bubble_E2P(Eb, R2, R1, gamma)


# =============================================================================
# Find inner discontinuity
# R1 = interface separating inner bubble radius and outer solar wind
# =============================================================================

def get_r1(r1, params):
    """
    Root of this equation sets r1 (see Rahners thesis, eq 1.25).
    This is derived by balancing pressure.
    
    units of au
    
    Parameters
    ----------
    r1 : variable for solving the equation 
        The inner radius of the bubble.

    Returns
    -------
    equation : equation to be solved for r1.

    """
    # Note
    # old code: R1_zero()
    Lmech_total, Ebubble, v_mech_total, r2 = params
    
    # set minimum energy to avoid zero
    if Ebubble < 1e-30:
        Ebubble = 1e-30
    # the equation to solve
    equation = np.sqrt( Lmech_total / v_mech_total / Ebubble * (r2**3 - r1**3) ) - r1
    # return
    return equation

