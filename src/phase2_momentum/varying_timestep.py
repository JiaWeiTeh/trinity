#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:32:47 2022

@author: Jia Wei Teh

This script includes ODE functions that help evaluate the momentum phase. 


# I dont think this is needed? Its in ODE.f_mom_grad(), which seems to 
# onlyhave use in the old momentum phase solver in phasesolver2.py. (double check!)
# TODO:
    



"""
# libraries
import src.cloud_properties.mass_profile as mass_profile
import astropy.units as u
import astropy.constants as c
import sys 

def get_momentum_ODEs(y, t, params):
    """
    This function is correct and is used for collapse of the shell when the 
    shell mass is constant (i.e., Mdot_shell = 0). Be aware that the 
    time units here are astro! (pc, Myrs, Msun)

    Parameters
    ----------
    y : [r, v]
    t : time (since the ODE is autonomous, t does not appear. The ODE solver still expects it though)
    # parameters:
        M0: Previous shell mass
        LBOL_ABS: absorbed luminosity
        tau_IR: 
        PW:
        MSTAR:
    """

    # Note:
        # old code: f_mom_grad()

    r, rd = y  # unpack current values of y (r, rdot, rdotdot)
    M0, _, LBOL_ABS, TAU_IR, PW, MSTAR, density_specific_param, A_EXP, RCLOUD, MCLOUD, SFE, warpfield_params = params  # unpack parameters

    # swept mass
    Msh, Msh_dot = mass_profile.get_mass_profile(r, density_specific_param, RCLOUD, MCLOUD,
                                                 rdot_arr = rd)
    
    # if shell collapsed a bit before but is now expanding again, it is expanding into emptiness
    if M0 > Msh:
        Msh = M0
        Msh_dot = 0.0

    rdd = 1./Msh * (LBOL_ABS/(c.c.to(u.pc/u.Myr).value)*(1.+TAU_IR) + PW - Msh_dot*rd) - ((c.G.to(u.pc**3/u.M_sun/u.Myr**2).value )/r**2) * (MSTAR + Msh/2.0)

    derivs = [rd, rdd]  # list of dy/dt=f functions
    return derivs


def update_dt(r0, v0, tInc_tmp, tStart_i, tCheck_tmp, stop_t, tmax = 1e99):
    """Update variable time step
    in the beginning and close to first SNe small stepsize
    also adapt time step during collapse

    Parameters
    ----------
    r0 : float
        shell radius
    v0 : folat
        shell velocity
    tInc_tmp : float
        last used time step inside ODE solver (tInc_tmp < tCheck_tmp)
    tStart_i : float
        start time for this time step
    tCheck_tmp: float
        last used time interval between calculation of shell structure
    (optional) tmax: float
        optional end time other than global stop time (e.g. cooling time)
    """

    # early on (< 1 Myr) and around first SNe (~3.2 Myr) small time step!
    # fabs varies a lot
    
    # Here are some variables for setting up the timesteps.
    tInc = 5e-4 # default: 5e4, unit Myr
    # time step for calculation of shell structure
    # this is the max main time step (used to calculated shell structure etc.) in Myr
    tCheck = 0.04166 #0.2 
    # time steps for first 0.1 Myr
    tCheck_small = 1e-2
    tInc_small = 5e-5 #5e-5

    if v0 > 0.0 and (tStart_i <= 1.0 or abs(tStart_i - 3.4) <= 0.4):
        tCheck_tmp = tCheck_small
        tInc_tmp = tInc_small
    elif (v0 > 0.0):
        tCheck_tmp = tCheck
        tInc_tmp = tInc
    # decrease step size if collapsing
    elif (v0 < - 0.0 and r0 < 5.0*abs(v0*tCheck_tmp)):
        dt_reduce = r0/(5.*abs(v0))
        tInc_tmp = min(dt_reduce, tInc_tmp)
        tCheck_tmp = min(dt_reduce, tCheck_tmp)
    # i.tStop is the end time of the simulation
    tStop_i = min([tStart_i + tCheck_tmp, stop_t, tmax])

    # enforce to have at least 3 small dt between tStart_i and tStop_i
    if (tStop_i - tStart_i <= 3.*tInc_tmp):
        tInc_tmp = 0.1*tInc_tmp


    return tStop_i, tInc_tmp, tCheck_tmp






