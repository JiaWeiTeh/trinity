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
import astropy.units as u
import src._functions.unit_conversions as cvt

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



def beta2Edot(params
              ):
    # old code: beta_to_Edot()
    """
    see pg 80, A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf 


    my_params:: contains
        
    Parameters
    ----------
    bubble_P : float
        Bubble pressure.
    bubble_E : float
        Bubble energy.
    r1 : float
        Inner bubble radius.
    r2 : float
        Outer bubble radius (aka rShell).
    beta : float
        dbubble_P/dt.
    t_now : float
        time.
    pwdot : float
        dPw/dt.
    pwdotdot : float
        dPw/dt/dt.
    r2dot : float
        Outer bubble velocity.

    Returns
    -------
    bubble_Edot : float
        dE/dt.

    """
    # dp/dt pressure 
    press_dot = - params['Pb'].value * params['cool_beta'].value / params['t_now'].value
    # define terms
    # print('pwdot', pwdot)
    # print('r1', r1)
    a = np.sqrt(params['pdot_total'].value/2)
    b = 1.5 * a**2 * params['R1'].value
    d = params['R2'].value**3 - params['R1'].value**3
    adot = 0.25 * params['pdotdot_total'].value / a
    # print('b', b)
    # print('bubble_E', bubble_E)
    e = b / ( b + params['Eb'].value )
    # main equation
    bubble_Edot = (2 * np.pi * press_dot * d**2 + 3 * params['Eb'].value * params['v2'].value * params['R2'].value**2 * (1 - e) -\
                    3 * (adot / a) * params['R1'].value**3 * params['Eb'].value**2 / (params['Eb'].value + b)) / (d * (1 - e))
    
    # return 
    return bubble_Edot


    # converts beta to dE/dt
    # :param Pb: pressure of bubble
    # :param R1: inner radius of bubble
    # :param beta: -(t/Pb)*(dPb/dt), see Weaver+77, eq. 40
    # :param my_params:
    # :return: 
    # """
    # R2 = my_params['R2']
    # v2 = my_params["v2"]
    # E = my_params['Eb']
    # Pdot = -Pb*beta/my_params["t_now"]

    # pwdot = my_params['pwdot'] # pwdot = 2.*Lw/vw

    # A = np.sqrt(pwdot/2.)
    # A2 = A**2
    # C = 1.5*A2*R1
    # D = R2**3 - R1**3
    # #Adot = (my_params['Lw_dot']*vw - Lw*my_params['vw_dot'])/(2.*A*vw**2)
    # Adot = 0.25*my_params['pwdot_dot']/A

    # F = C / (C + E)

    # #Edot = ( 3.*v2 * R2**2 * E + 2.*np.pi*Pdot*D**2 ) / D # does not take into account R1dot
    # #Edot = ( 2.*np.pi*Pdot*D**2 + 3.*E*v2*R2**2 * (1.-F) ) / (D * (1.-F)) # takes into account R1dot but not time derivative of A
    # Edot = ( 2.*np.pi*Pdot*D**2 + 3.*E*v2*R2**2 * (1.-F) - 3.*(Adot/A)*R1**3*E**2/(E+C) ) / (D * (1.-F)) # takes everything into account

    # #print "term1", "%.5e"%(2.*np.pi*Pdot*D**2), "term2", "%.5e"%(3.*E*v2*R2**2 * (1.-F)), "term3", "%.5e"%(3.*(Adot/A)*R1**3*E**2/(E+C))

    # #print "Edot", "%.5e"%Edot, "%.5e"%Edot_exact

    # return Edot


def Edot2beta(bubble_P, r1, bubble_Edot, my_params
              ):
    # old code: Edot_to_beta()
    """
    see pg 80, A12 https://www.imprs-hd.mpg.de/399417/thesis_Rahner.pdf 
    
    Parameters
    ----------
    bubble_P : float
        Bubble pressure.
    bubble_E : float
        Bubble energy.
    r1 : float
        Inner bubble radius.
    r2 : float
        Outer bubble radius (aka rShell).
    bubble_Edot : float
        dE/dt.
    t_now : float
        time.
    pwdot : float
        dPw/dt.
    pwdotdot : float
        dPw/dt/dt.
    r2dot : float
        Outer bubble velocity.

    Returns
    -------
    beta : float
        dbubble_P/dt.

    """
    t_now = my_params["t_now"]
    pwdot = my_params["pdot_total"]
    pwdotdot = my_params["pdotdot_total"]
    r2 = my_params["R2"]
    r2dot = my_params["v2"]
    bubble_E = my_params["Eb"]
    # define terms
    a = np.sqrt(pwdot/2)
    b = 1.5 * a**2 * r1
    d = r2**3 - r1**3
    adot = 0.25 * pwdotdot / a
    e = b / ( b + bubble_E ) 
    # main equation
    pdot = 1 / (2 * np.pi * d**2 ) *\
        ( d * (1 - e) * bubble_Edot - 3 * bubble_E * r2dot * r2**2 * (1 - e) + 3 * adot / a * r1**3 * bubble_E**2 / (bubble_E + b))
    beta = - pdot * t_now / bubble_P
    # return
    return beta



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
    else:
        # Energy phase: thermal pressure from hot bubble
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

